import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from langsmith import get_current_run_tree, trace, traceable

load_dotenv()

from .config import Settings
from .extractors.openai import _report_summary
from .ingestion import get_media_type
from .pipeline import MedicalReportPipeline

app = FastAPI(
    title="Medical Report Parser",
    version="0.1.0",
)


settings = Settings()
pipeline = MedicalReportPipeline(settings)

# Hard ceiling for one upload. Must exceed the client's own worst case
# (REQUEST_TIMEOUT x (MAX_RETRIES + 1) = 240s) so a normal retry is not
# cut short, while still bounding a hang anywhere else in the path.
EXTRACTION_DEADLINE = 300

# Must stay well under the shortest idle timeout in front of this service.
# nginx proxy_read_timeout defaults to 60s.
HEARTBEAT_INTERVAL = 10


def event(**payload) -> str:
    """One SSE frame."""
    return "data: " + json.dumps(payload) + "\n\n"


@traceable(
    run_type="chain",
    name="parse_report",
    # filename can identify a patient; log its shape, not the name
    process_inputs=lambda i: {
        "media_type": i.get("media_type"),
        "upload_bytes": len(i.get("file_bytes") or b""),
        "extension": (i.get("filename") or "").rsplit(".", 1)[-1].lower(),
    },
    # the extraction itself is summarised on the extract_async span
    process_outputs=lambda _: {"status": "ok"},
)
async def run_extraction(
    file_bytes: bytes,
    media_type: str,
    filename: str,  # noqa: ARG001 - read by process_inputs above, keep it
):
    """Root trace span for one upload. Children: build_content -> LLM call."""
    run = get_current_run_tree()
    if run is not None:
        run.metadata["model"] = settings.openai_model

    return await pipeline.extractor.extract_async(
        file_bytes=file_bytes,
        media_type=media_type,
    )


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/", response_class=HTMLResponse)
def monitor():
    """Dev page for watching the stream. Read per request so edits are live."""
    return (Path(__file__).resolve().parent / "static" / "index.html").read_text(
        encoding="utf-8"
    )


@app.post("/parse")
async def parse_report(
    file: UploadFile = File(...)
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    # Browsers disagree about HEIC: Safari sends image/heic, Chrome on
    # Android often sends application/octet-stream. The extension is the
    # only thing every client gets right, and it validates the upload
    # before we read the body.
    try:
        media_type = get_media_type(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    file_bytes = await file.read()

    async def generate():
        started = time.monotonic()
        stream = None
        pending = None
        result = None
        sections_sent = 0

        try:
            yield event(
                type="status",
                stage="received",
                message="Report received",
                bytes=len(file_bytes),
            )

            yield event(
                type="status",
                stage="analyzing",
                message="Analyzing medical report",
                deadline_seconds=EXTRACTION_DEADLINE,
            )

            with trace(
                name="parse_report",
                run_type="chain",
                # filename can identify a patient; log its shape, not the name
                inputs={
                    "media_type": media_type,
                    "upload_bytes": len(file_bytes),
                    "extension": file.filename.rsplit(".", 1)[-1].lower(),
                },
                metadata={"model": settings.openai_model},
            ) as run:

                stream = pipeline.extractor.extract_sections_async(
                    file_bytes=file_bytes,
                    media_type=media_type,
                )

                while True:
                    pending = asyncio.ensure_future(stream.__anext__())

                    # Categories arrive minutes apart on a slow report, so
                    # keep the keepalive running between them: an idle proxy
                    # (nginx defaults to 60s) would kill a healthy request.
                    while True:
                        done, _ = await asyncio.wait(
                            {pending},
                            timeout=HEARTBEAT_INTERVAL,
                        )
                        if done:
                            break

                        if time.monotonic() - started > EXTRACTION_DEADLINE:
                            pending.cancel()
                            raise asyncio.TimeoutError

                        yield event(
                            type="heartbeat",
                            elapsed_seconds=round(
                                time.monotonic() - started
                            ),
                        )

                    try:
                        kind, payload = pending.result()
                    except StopAsyncIteration:
                        break

                    if kind == "section":
                        sections_sent += 1
                        # Provisional: structurally complete, not yet
                        # validated. The result event below is the truth.
                        yield event(
                            type="section",
                            index=sections_sent - 1,
                            data=payload,
                        )
                        continue

                    result = payload
                    yield event(
                        type="result",
                        data=result.model_dump(),
                        elapsed_seconds=round(
                            time.monotonic() - started, 1
                        ),
                    )

                # Same shape-only summary the extractor logs, so the grouping
                # detectors live in one place instead of two.
                run.end(outputs={
                    "sections_streamed": sections_sent,
                    **(_report_summary(result) if result else {}),
                })

            yield event(type="complete")

        except asyncio.TimeoutError:
            # Bare TimeoutError stringifies to "", so say something useful.
            yield event(
                type="error",
                stage="timeout",
                message=(
                    f"Extraction exceeded {EXTRACTION_DEADLINE}s "
                    "and was cancelled."
                ),
            )

        except ValueError as exc:
            yield event(type="error", message=str(exc))

        except Exception as exc:
            yield event(
                type="error",
                message=f"Failed to process document: {exc}",
            )

        finally:
            # Client hung up mid-stream: stop paying OpenAI for a result
            # nobody will read. Order matters - the in-flight __anext__ has
            # to finish unwinding before aclose(), or it raises
            # "asynchronous generator is already running".
            if pending is not None and not pending.done():
                pending.cancel()
                try:
                    await pending
                except (Exception, asyncio.CancelledError):
                    pass

            if stream is not None:
                try:
                    await stream.aclose()
                except (Exception, asyncio.CancelledError):
                    pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
