import asyncio
import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from langsmith import trace
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import ValidationError

load_dotenv()

from .config import Settings
from .extractors.openai import report_summary
from .schemas import reject_mixed_patients, reject_unusable
from .ingestion import MAX_UPLOAD_BYTES, media_types_for
from .pipeline import MedicalReportPipeline

app = FastAPI(
    title="Medical Report Parser",
    version="0.1.0",
)


def _swagger_file_pickers(node):
    """Spell "binary" the way Swagger UI understands. /docs only.

    FastAPI emits OpenAPI 3.1, where an upload is
    {"type": "string", "contentMediaType": "application/octet-stream"}.
    The bundled Swagger UI only renders the 3.0 spelling, "format": "binary",
    as a file picker - and inside an array's `items` it gives up entirely and
    offers a text box with "Add string item". Rewriting the spelling is
    cosmetic: the endpoint itself takes ordinary multipart either way.
    """
    if isinstance(node, dict):
        if node.pop("contentMediaType", None) == "application/octet-stream":
            node["format"] = "binary"
        for value in node.values():
            _swagger_file_pickers(value)
    elif isinstance(node, list):
        for value in node:
            _swagger_file_pickers(value)
    return node


# FastAPI caches the spec, so this rewrites one dict once and is idempotent -
# a second pass finds no contentMediaType left to change.
_generated_openapi = app.openapi
app.openapi = lambda: _swagger_file_pickers(_generated_openapi())


logger = logging.getLogger(__name__)

try:
    settings = Settings()
except ValidationError as exc:
    # Field names only. The values are the secrets.
    raise SystemExit(
        "Configuration error: "
        + ", ".join(
            ".".join(str(part) for part in error["loc"]) for error in exc.errors()
        )
        + ". Copy .env.example to .env and fill it in."
    )

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


def _classify(exc: BaseException) -> tuple[str, str]:
    """(stage, message) for the error frame. The only text a client ever sees.

    Nothing here echoes an exception we did not write. An upstream message
    quotes our own request back at us - a masked key and org id on an auth
    failure, and on this path the document itself - so unknown exceptions are
    logged in full and reported as one flat sentence.

    Order matters twice: asyncio.TimeoutError IS the builtin TimeoutError,
    which is an OSError, and ValidationError IS a ValueError.
    """
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout", (
            f"Extraction exceeded {EXTRACTION_DEADLINE}s and was cancelled."
        )

    if isinstance(exc, ValidationError):
        # Unreachable today: stream_sections sanitises its own. Kept so that a
        # model_validate added anywhere later cannot leak by default.
        logger.error("unsanitised ValidationError, %d errors", exc.error_count())
        return "invalid_response", (
            "The model returned a malformed report. Please retry."
        )

    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        logger.error("OpenAI rejected our credentials: %s", exc)
        return "config", "The extraction service is not configured correctly."

    if isinstance(exc, RateLimitError):
        return "rate_limited", (
            "The extraction service is busy. Please retry in a moment."
        )

    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        logger.warning("OpenAI unreachable: %s", exc)
        return "upstream", "Could not reach the extraction service. Please retry."

    if isinstance(exc, APIError):
        logger.error("OpenAI error: %s", exc)
        return "upstream", "The extraction service could not process the document."

    if isinstance(exc, ValueError):
        # Ours, and written to be read: ingestion, downscaling and
        # stream_sections all raise plain ValueError with a safe message.
        return "invalid_document", str(exc)

    logger.error("unhandled error while parsing a report", exc_info=exc)
    return "error", "Failed to process document."


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/", response_class=HTMLResponse)
def monitor():
    """Dev page for watching the stream. Read per request so edits are live."""
    page = Path(__file__).resolve().parent / "static" / "index.html"

    try:
        return page.read_text(encoding="utf-8")
    except OSError:
        raise HTTPException(status_code=404, detail="Dev monitor page is not installed.")


async def _read_capped(upload: UploadFile, budget: int) -> bytes:
    """Read one upload, refusing to buffer more than `budget` bytes.

    `await upload.read()` with no argument is unbounded, so a single request
    can exhaust memory - and a multi-page upload multiplies it by the page
    count. Cheaper to refuse than to swap.
    """
    data = bytearray()

    while chunk := await upload.read(1 << 20):
        data += chunk
        if len(data) > budget:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Upload exceeds the "
                    f"{MAX_UPLOAD_BYTES // (1 << 20)} MB total limit."
                ),
            )

    if not data:
        raise HTTPException(
            status_code=400,
            detail=f"{upload.filename} is empty.",
        )

    return bytes(data)


@app.post("/parse")
async def parse_report(
    # Stays named `file` even though it is a list: the parameter name is the
    # multipart field name, so renaming it would break every existing client.
    # One part named `file` arrives as a 1-element list.
    file: list[UploadFile] = File(...)
):
    if any(not upload.filename for upload in file):
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    # Browsers disagree about HEIC: Safari sends image/heic, Chrome on
    # Android often sends application/octet-stream. The extension is the
    # only thing every client gets right, and it validates the whole upload
    # before we read a single body.
    try:
        media_types = media_types_for([u.filename for u in file])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Page order is upload order. The budget is shared across pages, so a
    # dozen small photos and one huge scan hit the same ceiling.
    files = []
    for upload, media_type in zip(file, media_types):
        spent = sum(len(data) for data, _ in files)
        files.append(
            (await _read_capped(upload, MAX_UPLOAD_BYTES - spent), media_type)
        )

    upload_bytes = sum(len(data) for data, _ in files)

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
                bytes=upload_bytes,
                pages=len(files),
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
                    "pages": len(files),
                    "media_types": sorted(set(media_types)),
                    "upload_bytes": upload_bytes,
                    "extensions": sorted({
                        u.filename.rsplit(".", 1)[-1].lower() for u in file
                    }),
                },
                metadata={"model": settings.openai_model},
            ) as run:

                stream = pipeline.stream_pages(files)

                while True:
                    # Checked here as well as in the heartbeat loop below: a
                    # stream that delivers an event just inside every
                    # heartbeat window never goes idle, so the inner check is
                    # never reached and the ceiling would not bind at all.
                    if time.monotonic() - started > EXTRACTION_DEADLINE:
                        raise asyncio.TimeoutError

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

                    if kind == "header":
                        # page_patients is complete here (it is written before
                        # `sections`), so a mixed upload is refused now rather
                        # than after streaming one patient's results under
                        # another's name. Measured: 4.3s instead of 13.0s.
                        reject_mixed_patients(payload.get("page_patients"))

                        # Provisional, like section below: patient identity
                        # lands ~12s before the validated result.
                        yield event(type="header", data=payload)
                        continue

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

                    reject_unusable(result)

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
                    **(report_summary(result) if result else {}),
                })

            yield event(type="complete")

        # asyncio.CancelledError is a BaseException, so a client hanging up
        # passes straight through to the finally below instead of being
        # reported as a failure to a socket that is already gone.
        except Exception as exc:
            stage, message = _classify(exc)
            yield event(type="error", stage=stage, message=message)

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
