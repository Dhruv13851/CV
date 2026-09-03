import asyncio
import base64

from langchain_openai import ChatOpenAI
from langsmith import get_current_run_tree, traceable

from app.config import PROMPTS_DIR
from app.downscaling import downscale
from app.schemas import MedicalReport
from app.streaming import stream_sections

SYSTEM_PROMPT = (PROMPTS_DIR / "extraction.md").read_text(encoding="utf-8")

# ponytail: a dense report extracts in 30-60s, so 120s is ~2x the slowest
# real case. Raise both only together - they multiply.
REQUEST_TIMEOUT = 120
MAX_RETRIES = 1

# Everything here is routed through downscale(), which returns JPEG or PNG.
IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/heic",
    "image/heif",
}


def _report_summary(report) -> dict:
    """Shape of the extraction, not its contents. Patient data stays out."""
    if not isinstance(report, MedicalReport):
        return {"result": "unavailable"}

    tests = [t for section in report.sections for t in section.tests]
    indicators = [t.indicator for t in tests]

    biggest = max((len(s.tests) for s in report.sections), default=0)

    # Measured on samples/sample2.pdf: the differential rows were written under
    # both "Complete Blood Count" and their own sub-header (1 run in 3), and CRP
    # came back as a second "Biochemistry" section (3 of 3). Neither is visible
    # in a section or test count, so count them directly. Empty sections were a
    # third defect; MedicalReport.drop_empty_sections removes those outright.
    names = [t.name for t in tests]
    categories = [s.category_name for s in report.sections]

    return {
        "sections": len(report.sections),
        "tests": len(tests),
        # A multi-page report split into one giant category means the model
        # took the page banner ("pre op major") for a category header.
        "largest_category_tests": biggest,
        "suspect_flat_grouping": len(report.sections) == 1 and biggest > 15,
        # one printed row filed under two categories
        "duplicate_test_names": len(names) - len(set(names)),
        # two sections with one name - consumers keyed by category collide
        "duplicate_category_names": len(categories) - len(set(categories)),
        "with_reference_range": sum(1 for t in tests if t.reference_ranges),
        "indicator_green": indicators.count("Green"),
        "indicator_yellow": indicators.count("Yellow"),
        "indicator_red": indicators.count("Red"),
        "indicator_null": indicators.count(None),
        "has_patient_name": report.patient.name is not None,
        "lab_name": report.lab_name,
    }


class OpenAIExtractor:

    def __init__(
        self,
        model: str,
        api_key: str,
    ):
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0,
            # LangChain discards the OpenAI SDK's default timeout and leaves
            # httpx on Timeout(None) - an unanswered request would hang the
            # task, its thread and a pool slot forever. Set it explicitly.
            # Worst case is REQUEST_TIMEOUT x (MAX_RETRIES + 1) = 240s.
            timeout=REQUEST_TIMEOUT,
            max_retries=MAX_RETRIES,
            # httpx applies the read timeout BETWEEN reads. Non-streaming,
            # the whole generation is one read, so a report that legitimately
            # takes >REQUEST_TIMEOUT to write would fail while working fine.
            # Streaming resets the clock per chunk, so only a real stall trips.
            streaming=True,
        )

        self.structured_llm = self.llm.with_structured_output(
            MedicalReport
        )

        # Same strict json_schema as with_structured_output, but without its
        # trailing RunnableLambda, which buffers and would defeat streaming.
        #
        # Deliberately NOT bind_tools: reasoning models reject function tools
        # in /v1/chat/completions ("use /v1/responses or set reasoning_effort
        # to 'none'"). response_format has no such restriction and enforces the
        # schema just as strictly, so reasoning stays on.
        self.stream_llm = self.llm.bind(response_format=MedicalReport)

    @traceable(
        run_type="tool",
        name="build_content",
        # payload carries the document itself - log shape, never content
        process_inputs=lambda i: {
            "media_type": i.get("media_type"),
            "input_bytes": len(i.get("file_bytes") or b""),
        },
        process_outputs=lambda o: {"parts": len(o) if o else 0},
    )
    def _build_content(
        self,
        file_bytes: bytes,
        media_type: str,
    ):
        if media_type in IMAGE_TYPES:
            # Also normalises: HEIC comes back as JPEG, which OpenAI accepts.
            file_bytes, media_type = downscale(file_bytes)

        encoded_file = base64.b64encode(
            file_bytes
        ).decode("utf-8")

        self._record_payload(media_type, len(encoded_file))

        if media_type == "application/pdf":

            file_data = (
                f"data:application/pdf;base64,{encoded_file}"
            )

            return [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                },
                {
                    "type": "file",
                    "file": {
                        "filename": "medical_report.pdf",
                        "file_data": file_data,
                    },
                },
            ]

        elif media_type in {
            "image/jpeg",
            "image/png",
        }:

            image_data = (
                f"data:{media_type};base64,{encoded_file}"
            )

            return [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data,
                    },
                },
            ]

        else:
            raise ValueError(
                f"Unsupported media type: {media_type}"
            )

    def _record_payload(self, media_type: str, encoded_len: int) -> None:
        run = get_current_run_tree()
        if run is not None:
            run.metadata.update({
                "sent_media_type": media_type,
                "base64_payload_mb": round(encoded_len / 1e6, 2),
            })

    @traceable(
        run_type="chain",
        name="extract",
        process_inputs=lambda i: {
            "media_type": i.get("media_type"),
            "input_bytes": len(i.get("file_bytes") or b""),
        },
        process_outputs=_report_summary,
    )
    def extract(
        self,
        file_bytes: bytes,
        media_type: str,
    ) -> MedicalReport:

        content = self._build_content(
            file_bytes,
            media_type,
        )

        return self.structured_llm.invoke(
            [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        )

    @traceable(
        run_type="chain",
        name="extract_async",
        process_inputs=lambda i: {
            "media_type": i.get("media_type"),
            "input_bytes": len(i.get("file_bytes") or b""),
        },
        process_outputs=_report_summary,
    )
    async def extract_async(
        self,
        file_bytes: bytes,
        media_type: str,
    ) -> MedicalReport:

        # Decode + LANCZOS resize + JPEG + base64 is ~600ms of CPU. On the
        # event loop that freezes every other request, /health included.
        # to_thread copies contextvars, so LangSmith spans still nest.
        content = await asyncio.to_thread(
            self._build_content,
            file_bytes,
            media_type,
        )

        return await self.structured_llm.ainvoke(
            [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        )

    async def extract_sections_async(
        self,
        file_bytes: bytes,
        media_type: str,
    ):
        """Stream ("section", dict) per category, then ("report", model)."""
        content = await asyncio.to_thread(
            self._build_content,
            file_bytes,
            media_type,
        )

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        async for item in stream_sections(self.stream_llm, messages):
            yield item
