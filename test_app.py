"""Run: python test_app.py"""
from app.schemas import MedicalReport


def test_optional_fields():
    # The common case the prompt produces: qualitative result, no usable range,
    # so no indicator. This used to blow up on indicator's Literal type.
    r = MedicalReport.model_validate({
        "patient": {},
        "sections": [{
            "category_name": "Urine",
            "tests": [{"name": "Bile Salts", "result": "Absent"}],
        }],
    })
    t = r.sections[0].tests[0]
    assert t.indicator is None
    assert t.reference_ranges == []
    assert r.patient.name is None
    assert r.report_title is None

    # Explicit null must survive too, not just an omitted key.
    assert MedicalReport.model_validate({
        "patient": {"name": "A", "age": 30, "gender": "Male"},
        "report_title": "CBC",
        "sections": [{
            "category_name": "Haematology",
            "tests": [{
                "name": "Haemoglobin",
                "result": 13.4,
                "unit": "g/dL",
                "indicator": None,
                "reference_ranges": [
                    {"label": "Male", "min_val": 13.0, "max_val": 17.0},
                    {"label": "Female", "min_val": 12.0, "max_val": 15.0},
                ],
            }],
        }],
    }).sections[0].tests[0].indicator is None


def test_indicator_still_constrained():
    MedicalReport.model_validate({
        "patient": {}, "sections": [{"category_name": "G", "tests": [
            {"name": "X", "result": 1.0, "indicator": "Green"}]}],
    })
    try:
        MedicalReport.model_validate({
            "patient": {}, "sections": [{"category_name": "G", "tests": [
                {"name": "X", "result": 1.0, "indicator": "Blue"}]}],
        })
    except Exception:
        return
    raise AssertionError("indicator accepted a value outside Green/Yellow/Red")


def test_downscale():
    import io

    from PIL import Image

    from app.downscaling import (
        HARD_PATCH_LIMIT,
        MAX_LONG_EDGE,
        downscale,
        patch_count,
    )

    def png(w, h, mode="RGB"):
        buf = io.BytesIO()
        Image.new(mode, (w, h), "white").save(buf, format="PNG")
        return buf.getvalue()

    # The real failure: a 600 DPI Letter scan is 33120 patches, over the limit.
    assert patch_count(5100, 6600) == 33120 > HARD_PATCH_LIMIT

    out, media_type = downscale(png(5100, 6600))
    w, h = Image.open(io.BytesIO(out)).size
    assert max(w, h) == MAX_LONG_EDGE, (w, h)
    assert patch_count(w, h) < HARD_PATCH_LIMIT
    assert media_type == "image/jpeg"
    assert abs((w / h) - (5100 / 6600)) < 0.01, "aspect ratio drifted"

    # Already small -> untouched, no lossy re-encode.
    small = png(800, 600)
    assert downscale(small) == (small, "image/png")

    # Alpha must not crash the JPEG encode; it gets flattened onto white.
    out, _ = downscale(png(4000, 5000, mode="RGBA"))
    assert Image.open(io.BytesIO(out)).mode == "RGB"

    # HEIC from a phone camera. OpenAI rejects HEIC, so a small one must
    # still be transcoded rather than passed through - and never upscaled.
    buf = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(buf, format="HEIF")
    heic = buf.getvalue()
    assert Image.open(io.BytesIO(heic)).format == "HEIF"

    out, media_type = downscale(heic)
    assert media_type == "image/jpeg"
    assert Image.open(io.BytesIO(out)).format == "JPEG"
    assert Image.open(io.BytesIO(out)).size == (800, 600)


def test_traces_carry_no_payload_or_phi():
    """Spans must log shape, never image bytes or patient identity."""
    from app.downscaling.image import _result_summary, _sizes_only
    from app.extractors.openai import _report_summary

    blob = b"\x89PNG" + b"\x00" * 100_000
    assert _sizes_only({"file_bytes": blob}) == {"input_bytes": 100_004}
    assert _result_summary((blob, "image/jpeg")) == {
        "output_bytes": 100_004,
        "media_type": "image/jpeg",
    }

    report = MedicalReport.model_validate({
        "patient": {"name": "SMITH, JOHN", "age": 73, "gender": "M"},
        "report_title": "CHEM PANEL",
        "sections": [{"category_name": "Chem", "tests": [
            {"name": "CALCIUM", "result": 8.9, "indicator": "Green"},
            {"name": "BILE SALTS", "result": "Absent"},
        ]}],
    })
    summary = _report_summary(report)

    assert "SMITH" not in str(summary)
    assert "73" not in str(summary.get("has_patient_name"))
    assert summary["tests"] == 2
    assert summary["indicator_null"] == 1
    assert summary["duplicate_test_names"] == 0
    assert summary["duplicate_category_names"] == 0


def test_grouping_detectors():
    """The three defects measured on sample2.pdf, none visible in a count.

    Real runs produced: the 5 differential rows written under both
    "Complete Blood Count" and their own sub-header, CRP as a second
    "Biochemistry" section, and a parent header left with no tests. The
    first two are only detectable; the third is dropped at validation.
    """
    from app.extractors.openai import _report_summary

    report = MedicalReport.model_validate({
        "patient": {},
        "sections": [
            {"category_name": "Complete Blood Count", "tests": [
                {"name": "Hemoglobin", "result": 13.1},
                {"name": "Neutrophils", "result": 60},
            ]},
            {"category_name": "Differential % WBCs count", "tests": [
                {"name": "Neutrophils", "result": 60},   # same printed row, twice
            ]},
            {"category_name": "Biochemistry", "tests": [
                {"name": "Blood Urea", "result": 24},
            ]},
            {"category_name": "Biochemistry", "tests": [   # CRP's page-9 banner
                {"name": "CRP (C-Reactive Protein)", "result": 4.1},
            ]},
            {"category_name": "URINE ROUTINE", "tests": []},  # rows all in sub-headers
        ],
    })

    # URINE ROUTINE owned no rows, so validation drops it. Five sections in,
    # four out - this fails if drop_empty_sections is ever removed.
    assert [s.category_name for s in report.sections] == [
        "Complete Blood Count",
        "Differential % WBCs count",
        "Biochemistry",
        "Biochemistry",
    ], [s.category_name for s in report.sections]

    summary = _report_summary(report)

    assert summary["duplicate_test_names"] == 1, summary
    assert summary["duplicate_category_names"] == 1, summary
    # neither defect moves the counts the original detector watched
    assert summary["sections"] == 4 and summary["tests"] == 5
    assert summary["suspect_flat_grouping"] is False


def test_streaming_never_emits_partial_values():
    """The whole point: a streamed value must equal its final value.

    Token-level parsing renders GLUCOSE 118 as 1, then 11, then 118. Only
    closed sections may be emitted.
    """
    import asyncio
    import json

    from langchain_core.messages import AIMessageChunk

    from app.streaming import stream_sections

    report = {
        "patient": {"name": "SMITH, JOHN"},
        "report_title": "PANEL",
        "sections": [
            {"category_name": "Haem", "tests": [
                {"name": "HAEMOGLOBIN", "result": 13.4, "reference_ranges": []},
                {"name": "WBC", "result": 14.2, "reference_ranges": []}]},
            {"category_name": "Biochem", "tests": [
                {"name": "GLUCOSE", "result": 118, "reference_ranges": []},
                {"name": "POTASSIUM", "result": 5.9, "reference_ranges": []}]},
            {"category_name": "Urine", "tests": [
                {"name": "BILE SALTS", "result": "Absent", "reference_ranges": []}]},
        ],
    }
    args = json.dumps(report)
    truth = {
        (s["category_name"], t["name"]): t["result"]
        for s in report["sections"] for t in s["tests"]
    }

    # The three shapes a chunk can arrive in. response_format (what we
    # actually use) puts JSON in .content; reasoning models wrap it in
    # blocks alongside reasoning that must be skipped; tool calling uses
    # .tool_call_chunks.
    SHAPES = {
        "content_str": lambda s: AIMessageChunk(content=s),
        "content_blocks": lambda s: AIMessageChunk(content=[
            {"type": "reasoning", "reasoning": "..."},
            {"type": "text", "text": s},
        ]),
        "tool_call_chunks": lambda s: AIMessageChunk(
            content="", tool_call_chunks=[{
                "name": "MedicalReport", "args": s,
                "id": "call_1", "index": 0, "type": "tool_call_chunk"}]),
    }

    class FakeLLM:
        def __init__(self, size, shape):
            self.size = size
            self.shape = SHAPES[shape]

        async def astream(self, _messages):
            for i in range(0, len(args), self.size):
                yield self.shape(args[i:i + self.size])

    async def run(size, shape):
        streamed, final = [], None
        async for kind, payload in stream_sections(FakeLLM(size, shape), []):
            if kind == "section":
                streamed.append(payload)
            else:
                final = payload
        return streamed, final

    for shape in SHAPES:
        for size in (1, 3, 7, 64, 10_000):   # incl. 1 char/chunk worst case
            where = f"{shape} @ {size}"
            streamed, final = asyncio.run(run(size, shape))

            for section in streamed:
                for test in section["tests"]:
                    key = (section["category_name"], test["name"])
                    assert truth[key] == test["result"], (
                        f"{where}: streamed {key} as {test['result']!r}, "
                        f"real value is {truth[key]!r}"
                    )

            assert isinstance(final, MedicalReport), where
            assert len(final.sections) == 3, where
            # the in-flight section is never streamed early
            assert len(streamed) == 2, (where, len(streamed))
            assert [s["category_name"] for s in streamed] == ["Haem", "Biochem"]


if __name__ == "__main__":
    test_optional_fields()
    test_indicator_still_constrained()
    test_downscale()
    test_traces_carry_no_payload_or_phi()
    test_grouping_detectors()
    test_streaming_never_emits_partial_values()
    print("ok")
