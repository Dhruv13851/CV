"""Run: python test_app.py"""
import asyncio

from PIL import Image

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


def test_service_tier_reaches_the_request():
    """Fast mode is a request field, not a client-side setting.

    The Fast mode guide names only gpt-5.6-sol, so this asserts plumbing,
    not that the model honours it - the response echoes "priority" for both
    "fast" and "priority", so only billing tells you what you actually got.
    """
    from app.extractors.openai import OpenAIExtractor

    msg = [{"role": "user", "content": "x"}]

    fast = OpenAIExtractor(model="gpt-5.6-luna", api_key="sk-test", service_tier="fast")
    assert fast.llm._get_request_payload(msg)["service_tier"] == "fast"

    # empty means "send no tier", not "send an empty one" - an unsupported
    # model should be able to fall back without a code change
    off = OpenAIExtractor(model="gpt-5.6-luna", api_key="sk-test", service_tier="")
    assert off.llm._get_request_payload(msg).get("service_tier") is None

    # the default is standard tier, matching Settings - fast mode costs a
    # premium and must never be what you get by forgetting to set it
    assert OpenAIExtractor(model="m", api_key="k").service_tier == "default"
    assert (OpenAIExtractor(model="m", api_key="k")
            .llm._get_request_payload(msg)["service_tier"] == "default")


def test_multipage_upload():
    """One report, N pages: validation, caps, and one prompt for the batch."""
    import io

    from fastapi import HTTPException

    from app.extractors.openai import OpenAIExtractor, _files_summary
    from app.ingestion import MAX_PAGES, media_types_for

    # --- whole-upload validation, before a single byte is read ---
    assert media_types_for(["r.pdf"]) == ["application/pdf"]
    # iPhone writes IMG_0001.HEIC; order is page order, never sorted
    assert media_types_for(["b.HEIC", "a.jpg"]) == ["image/heic", "image/jpeg"]

    for bad, expect in [
        ([], "No file"),
        (["a.pdf", "b.jpg"], "not both"),          # one PDF or N images
        (["x.jpg"] * (MAX_PAGES + 1), "Too many"),
        (["a.txt"], "Unsupported"),
    ]:
        try:
            media_types_for(bad)
            raise AssertionError(f"{bad} should have been rejected")
        except ValueError as exc:
            assert expect in str(exc), (bad, str(exc))

    # --- one system prompt for the batch, one part per page, order kept ---
    def png(w, h, colour):
        buf = io.BytesIO()
        Image.new("RGB", (w, h), colour).save(buf, format="PNG")
        return buf.getvalue()

    ex = OpenAIExtractor(model="m", api_key="k")
    pages = [(png(80, 100, c), "image/png") for c in ("white", "red", "blue")]
    parts = ex._build_content(pages)

    assert [p["type"] for p in parts] == [
        "text", "image_url", "image_url", "image_url"], parts
    assert sum(1 for p in parts if p["type"] == "text") == 1, "prompt sent twice"
    # small PNGs pass through untouched, so the payload order is checkable
    import base64
    for sent, (raw, _) in zip(
        [p for p in parts if p["type"] == "image_url"], pages
    ):
        assert base64.b64decode(
            sent["image_url"]["url"].split(",", 1)[1]
        ) == raw, "pages reordered"

    # a PDF still takes the file part, not image_url
    assert [p["type"] for p in ex._build_content(
        [(b"%PDF-1.4 fake", "application/pdf")]
    )] == ["text", "file"]

    # --- traces carry shape only ---
    assert _files_summary({"files": pages}) == {
        "pages": 3,
        "input_bytes": sum(len(b) for b, _ in pages),
        "media_types": ["image/png"],
    }

    # --- the read cap is a real guard, not a comment ---
    from app.main import _read_capped

    class FakeUpload:
        filename = "big.jpg"

        def __init__(self, data):
            self._buf = io.BytesIO(data)

        async def read(self, size=-1):
            return self._buf.read(size)

    assert asyncio.run(_read_capped(FakeUpload(b"x" * 100), 1000)) == b"x" * 100

    for data, budget, code in [(b"x" * 2000, 1000, 413), (b"", 1000, 400)]:
        try:
            asyncio.run(_read_capped(FakeUpload(data), budget))
            raise AssertionError(f"{len(data)}B/{budget} should have raised")
        except HTTPException as exc:
            assert exc.status_code == code, (len(data), budget, exc.status_code)


def test_parse_accepts_one_or_many_on_the_same_field():
    """The form field stays `file`, so existing single-upload clients work."""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)

    # rejected before any model call, so this needs no API key or credits
    one = client.post("/parse", files=[("file", ("a.txt", b"x", "text/plain"))])
    assert one.status_code == 400, one.text
    assert "Unsupported file type" in one.text

    many = client.post("/parse", files=[
        ("file", (f"p{i}.txt", b"x", "text/plain")) for i in range(3)
    ])
    assert many.status_code == 400, many.text

    mixed = client.post("/parse", files=[
        ("file", ("r.pdf", b"%PDF", "application/pdf")),
        ("file", ("p.jpg", b"x", "image/jpeg")),
    ])
    assert mixed.status_code == 400 and "not both" in mixed.text, mixed.text

    too_many = client.post("/parse", files=[
        ("file", (f"p{i}.jpg", b"x", "image/jpeg")) for i in range(13)
    ])
    assert too_many.status_code == 400 and "Too many" in too_many.text


def test_downscale_all_is_parallel_and_ordered():
    """Pages decode in a pool: order must hold and trace context must survive.

    Measured 9.78s -> 1.95s on 12 phone HEICs. The risk is silent: a pool
    thread that loses contextvars still returns the right bytes, it just
    orphans the @traceable downscale_image spans in LangSmith.
    """
    import contextvars
    import io
    import threading

    from app.downscaling import image as image_mod
    from app.downscaling import downscale_all

    def png(w, h, colour):
        buf = io.BytesIO()
        Image.new("RGB", (w, h), colour).save(buf, format="PNG")
        return buf.getvalue()

    # --- order survives the pool, and a PDF is forwarded untouched ---
    pdf = (b"%PDF-1.4 not really", "application/pdf")
    pages = [(png(80 + i, 100, c), "image/png")
             for i, c in enumerate(("white", "red", "blue", "green"))]

    out = downscale_all([pdf, *pages])
    assert out[0] == pdf, "PDF was modified"
    assert [b for b, _ in out[1:]] == [b for b, _ in pages], "pages reordered"

    # single page skips the pool entirely but must still work
    assert downscale_all([pages[0]]) == [pages[0]]
    assert downscale_all([]) == []

    # --- contextvars reach the workers (this is what keeps traces nested) ---
    probe = contextvars.ContextVar("probe", default="MISSING")
    seen, threads = [], set()
    real = image_mod.downscale

    def spy(file_bytes):
        seen.append(probe.get())
        threads.add(threading.current_thread().name)
        return real(file_bytes)

    image_mod.downscale = spy
    try:
        probe.set("carried")
        downscale_all(pages)
    finally:
        image_mod.downscale = real

    assert seen == ["carried"] * len(pages), seen
    assert any(n.startswith("downscale") for n in threads), threads


def test_docs_shows_a_file_picker():
    """/docs must offer file inputs, not a text box with "Add string item".

    FastAPI emits OpenAPI 3.1 (contentMediaType); Swagger UI only renders the
    3.0 spelling (format: binary) as a picker, and not at all inside `items`.
    """
    import json

    from fastapi.testclient import TestClient

    from app.main import app

    spec = TestClient(app).get("/openapi.json").json()
    body = spec["paths"]["/parse"]["post"]["requestBody"]
    ref = body["content"]["multipart/form-data"]["schema"]["$ref"]
    prop = spec["components"]["schemas"][ref.rsplit("/", 1)[-1]]["properties"]["file"]

    assert prop["type"] == "array", prop          # several pages
    assert prop["items"] == {"type": "string", "format": "binary"}, prop
    assert "contentMediaType" not in json.dumps(spec), "3.1 spelling left behind"


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
        streamed, headers, final, order = [], [], None, []
        async for kind, payload in stream_sections(FakeLLM(size, shape), []):
            order.append(kind)
            if kind == "section":
                streamed.append(payload)
            elif kind == "header":
                headers.append(payload)
            else:
                final = payload
        return streamed, headers, final, order

    for shape in SHAPES:
        for size in (1, 3, 7, 64, 10_000):   # incl. 1 char/chunk worst case
            where = f"{shape} @ {size}"
            streamed, headers, final, order = asyncio.run(run(size, shape))

            # exactly one header, whole, and before any section - a header
            # emitted early would render "SMITH, JOHN" as "S", then "SMI"
            assert len(headers) == 1, (where, headers)
            assert headers[0]["patient"] == {"name": "SMITH, JOHN"}, (where, headers[0])
            assert headers[0]["report_title"] == "PANEL", (where, headers[0])
            assert headers[0]["lab_name"] is None, (where, headers[0])
            assert order[0] == "header", (where, order[:3])

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
    test_multipage_upload()
    test_parse_accepts_one_or_many_on_the_same_field()
    test_downscale_all_is_parallel_and_ordered()
    test_docs_shows_a_file_picker()
    test_service_tier_reaches_the_request()
    test_streaming_never_emits_partial_values()
    print("ok")
