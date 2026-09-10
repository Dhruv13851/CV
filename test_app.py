"""Run: python test_app.py"""
import asyncio

from PIL import Image

from app.schemas import MedicalReport


def _png(w, h, mode="RGB"):
    import io

    buf = io.BytesIO()
    Image.new(mode, (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


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

    from vision_downscale import (
        HARD_PATCH_LIMIT,
        MAX_LONG_EDGE,
        downscale,
        patch_count,
    )

    # The real failure: a 600 DPI Letter scan is 33120 patches, over the limit.
    assert patch_count(5100, 6600) == 33120 > HARD_PATCH_LIMIT

    out, media_type = downscale(_png(5100, 6600))
    w, h = Image.open(io.BytesIO(out)).size
    assert max(w, h) == MAX_LONG_EDGE, (w, h)
    assert patch_count(w, h) < HARD_PATCH_LIMIT
    assert media_type == "image/jpeg"
    assert abs((w / h) - (5100 / 6600)) < 0.01, "aspect ratio drifted"

    # Already small -> untouched, no lossy re-encode.
    small = _png(800, 600)
    assert downscale(small) == (small, "image/png")

    # Alpha must not crash the JPEG encode; it gets flattened onto white.
    out, _ = downscale(_png(4000, 5000, mode="RGBA"))
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

    # A portrait phone photo is stored as landscape pixels plus orientation=6.
    # Small enough to skip the resize, so it takes the passthrough above -
    # which returns the ORIGINAL bytes and would send the page sideways. The
    # rotation has to be baked into the pixels instead.
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (1600, 1200), "white").save(buf, format="JPEG", exif=exif)
    sideways = buf.getvalue()
    assert Image.open(io.BytesIO(sideways)).size == (1600, 1200)

    out, media_type = downscale(sideways)
    assert out != sideways, "rotated original was passed through unchanged"
    assert media_type == "image/jpeg"
    upright = Image.open(io.BytesIO(out))
    assert upright.size == (1200, 1600), upright.size
    # tag consumed, not merely copied - two rotations would be one too many
    assert upright.getexif().get(274, 1) == 1, upright.getexif().get(274)

    # An upright JPEG of the same size still costs nothing.
    buf = io.BytesIO()
    Image.new("RGB", (1600, 1200), "white").save(buf, format="JPEG")
    plain = buf.getvalue()
    assert downscale(plain) == (plain, "image/jpeg")


def test_traces_carry_no_payload_or_phi():
    """Spans must log shape, never image bytes or patient identity."""
    from vision_downscale.downscaler import _result_summary, _sizes_only
    from app.extractors.openai import report_summary

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
    summary = report_summary(report)

    assert "SMITH" not in str(summary)
    assert "73" not in str(summary.get("has_patient_name"))
    assert summary["tests"] == 2
    assert summary["indicator_null"] == 1
    assert summary["duplicate_test_names"] == 0


def test_grouping_detectors():
    """The three defects measured on sample2.pdf, none visible in a count.

    Real runs produced: the 5 differential rows written under both
    "Complete Blood Count" and their own sub-header, CRP as a second
    "Biochemistry" section, and a parent header left with no tests. Only the
    first is still merely detectable - normalise_sections now merges the
    duplicate category and drops the empty one.
    """
    from app.extractors.openai import report_summary

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

    # URINE ROUTINE owned no rows, so validation drops it; the two
    # "Biochemistry" runs are one printed section, so validation merges them.
    # Five sections in, three out - this fails if normalise_sections goes.
    assert [s.category_name for s in report.sections] == [
        "Complete Blood Count",
        "Differential % WBCs count",
        "Biochemistry",
    ], [s.category_name for s in report.sections]

    # merged in page order, and no row was lost on the way
    assert [t.name for t in report.sections[-1].tests] == [
        "Blood Urea",
        "CRP (C-Reactive Protein)",
    ]

    summary = report_summary(report)

    # the row written under two categories survives as the one real defect,
    # and it still moves no count except this detector
    assert summary["duplicate_test_names"] == 1, summary
    assert summary["sections"] == 3 and summary["tests"] == 5
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

    # reasoning_effort is the same "" means send-nothing contract. It changes
    # what the model computes, not just how the request is routed, so the
    # default must stay the model's own - measured 12.8s -> 7.6s to first
    # event at "low", but only 1/3 runs matched the page against 2/3.
    low = OpenAIExtractor(model="m", api_key="k", reasoning_effort="low")
    assert low.llm._get_request_payload(msg)["reasoning_effort"] == "low"

    default = OpenAIExtractor(model="m", api_key="k")
    assert default.reasoning_effort is None
    assert default.llm._get_request_payload(msg).get("reasoning_effort") is None
    assert (OpenAIExtractor(model="m", api_key="k", reasoning_effort="")
            .llm._get_request_payload(msg).get("reasoning_effort") is None)


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

    from vision_downscale import OPENAI, downscale_all

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
    real = OPENAI.downscale

    def spy(file_bytes):
        seen.append(probe.get())
        threads.add(threading.current_thread().name)
        return real(file_bytes)

    # Instance attribute, shadowing the bound method - _prepare looks up
    # self.downscale per call, so the pool threads see the spy.
    OPENAI.downscale = spy
    try:
        probe.set("carried")
        downscale_all(pages)
    finally:
        del OPENAI.downscale

    assert seen == ["carried"] * len(pages), seen
    assert any(n.startswith("downscale") for n in threads), threads


def test_profiles_are_independent_but_share_one_pool():
    """The point of the class: per-service caps, but still one thread pool.

    A pool per instance would multiply 8 threads by the profile count, which
    is exactly what the module-level _POOL exists to prevent.
    """
    import io
    from concurrent.futures import ThreadPoolExecutor

    from vision_downscale import OPENAI, ImageDownscaler

    thumbs = ImageDownscaler(max_long_edge=800, profile="thumbnails")

    page = _png(4000, 3000)
    default_out, _ = OPENAI.downscale(page)
    thumb_out, _ = thumbs.downscale(page)

    assert max(Image.open(io.BytesIO(default_out)).size) == 2200
    assert max(Image.open(io.BytesIO(thumb_out)).size) == 800

    # the second profile did not mutate the first
    assert OPENAI.max_long_edge == 2200

    # ...and both went through the same pool
    assert thumbs._executor is OPENAI._executor

    # a deliberately isolated pool is still available
    own = ThreadPoolExecutor(max_workers=1)
    try:
        assert ImageDownscaler(executor=own)._executor is own
    finally:
        own.shutdown(wait=False)


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


def test_truncated_stream_carries_no_patient_data():
    """A cut-off stream must not report itself with the patient in the message.

    parse_partial_json closes the JSON into a dict with a required field
    missing, and pydantic quotes the failing input back at you - a test name
    here, the whole report when the root is not a dict. ValidationError is a
    ValueError, so main.py's handler would have put that in the SSE frame and
    the LangSmith span. Both cases must come back sanitised.
    """
    import json

    from langchain_core.messages import AIMessageChunk

    from app.streaming import stream_sections

    report = {
        "patient": {"name": "Padmaram Mali", "age": 47},
        "lab_name": "Acme Labs",
        "sections": [{"category_name": "CBC", "tests": [
            {"name": "Haemoglobin", "result": 9.1, "unit": "g/dL"},
            {"name": "Platelet Count", "result": 140}]}],
    }
    full = json.dumps(report)

    PHI = ("Padmaram", "Mali", "Haemoglobin", "Platelet", "9.1", "140", "47")

    class FakeLLM:
        def __init__(self, text):
            self.text = text

        async def astream(self, _messages):
            yield AIMessageChunk(content=self.text)

    async def run(text):
        async for kind, payload in stream_sections(FakeLLM(text), []):
            pass

    # cut mid-test, so tests.1.result is missing; and a non-dict root, whose
    # input_value is the entire payload
    for text in (full[:full.index('"result": 140')], json.dumps([report])):
        try:
            asyncio.run(run(text))
            raise AssertionError(f"{text[:40]!r} should have raised")
        except ValueError as exc:
            message = str(exc)
            for leaked in PHI:
                assert leaked not in message, (leaked, message)
            assert "input_value" not in message, message
            # the context is dropped too, or a caller could re-derive it
            assert exc.__cause__ is None and exc.__context__ is None, message


def test_empty_sections_is_an_error_not_an_empty_success():
    """Mixed patients and non-reports both come back as an error frame.

    extraction.md returns an empty sections list for pages showing more than
    one patient name (:64) and for a document that is not a report (:8). As a
    200 with sections: [] a consumer reads it as "this person had no tests".
    """
    import json

    from fastapi.testclient import TestClient

    from app import main

    report = MedicalReport.model_validate({
        "patient": {"name": "Padmaram Mali"},
        "sections": [],
    })

    async def fake_stream(_files):
        yield "header", {"patient": {"name": "Padmaram Mali"}}
        yield "report", report

    original = main.pipeline.stream_pages
    main.pipeline.stream_pages = fake_stream
    try:
        response = TestClient(main.app).post(
            "/parse", files=[("file", ("p.png", _png(64, 64), "image/png"))]
        )
    finally:
        main.pipeline.stream_pages = original

    frames = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    kinds = [f["type"] for f in frames]

    assert "result" not in kinds, frames
    assert "complete" not in kinds, frames
    assert kinds[-1] == "error", frames
    assert "more than one patient" in frames[-1]["message"], frames[-1]


def test_deadline_holds_against_a_trickling_stream():
    """The ceiling must bind even when the stream never goes idle.

    The heartbeat loop only reaches its own deadline check when a gap exceeds
    HEARTBEAT_INTERVAL, so a stream delivering an event just inside that gap
    would run forever. Scaled down 100x from the real 300s/10s.
    """
    import json

    from fastapi.testclient import TestClient

    from app import main

    async def trickle(_files):
        for i in range(200):
            await asyncio.sleep(0.09)      # just inside the heartbeat window
            yield "section", {"category_name": f"C{i}", "tests": []}

    original = main.pipeline.stream_pages
    saved = (main.EXTRACTION_DEADLINE, main.HEARTBEAT_INTERVAL)
    main.pipeline.stream_pages = trickle
    main.EXTRACTION_DEADLINE, main.HEARTBEAT_INTERVAL = 0.5, 0.1
    try:
        response = TestClient(main.app).post(
            "/parse", files=[("file", ("p.png", _png(64, 64), "image/png"))]
        )
    finally:
        main.pipeline.stream_pages = original
        main.EXTRACTION_DEADLINE, main.HEARTBEAT_INTERVAL = saved

    frames = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]

    assert frames[-1]["type"] == "error", frames[-1]
    assert frames[-1]["stage"] == "timeout", frames[-1]
    # it gave up early, rather than running all 200 events (18s of stream)
    assert sum(1 for f in frames if f["type"] == "section") < 20, len(frames)


def test_error_frames_never_echo_an_upstream_message():
    """Only exceptions we wrote ourselves reach the client as text.

    An OpenAI error quotes our own request back at us - the auth failure
    literally contains a masked API key - and the old handler interpolated
    any exception straight into the frame with an f-string.
    """
    import httpx
    from openai import (
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        RateLimitError,
    )
    from pydantic import ValidationError

    from app.main import _classify

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    def status(cls, code, message):
        return cls(message, response=httpx.Response(code, request=request), body=None)

    SECRET = "sk-abc***DEF"
    upstream = [
        ("config", status(AuthenticationError, 401,
                          f"Incorrect API key provided: {SECRET}")),
        ("rate_limited", status(RateLimitError, 429, "Rate limit reached for org-9xy")),
        ("upstream", status(BadRequestError, 400, "image exceeds 30000 patches")),
        ("upstream", APITimeoutError(request=request)),
        ("upstream", APIConnectionError(request=request)),
    ]

    for expected_stage, exc in upstream:
        stage, message = _classify(exc)
        assert stage == expected_stage, (stage, exc)
        assert SECRET not in message and "org-9xy" not in message, message
        assert str(exc) not in message, (message, str(exc))

    # an unknown exception is logged, never quoted
    stage, message = _classify(RuntimeError("connection to db-prod-7 refused"))
    assert stage == "error" and "db-prod-7" not in message, message

    # a stray ValidationError cannot leak even though it is a ValueError
    try:
        MedicalReport.model_validate({"patient": {}, "sections": "Padmaram Mali"})
    except ValidationError as exc:
        stage, message = _classify(exc)
    assert stage == "invalid_response", stage
    assert "Padmaram" not in message and "input_value" not in message, message

    # ours are written to be read, so they pass through intact
    stage, message = _classify(ValueError("Too many pages: 13. The limit is 12."))
    assert stage == "invalid_document" and "13" in message, message

    stage, message = _classify(asyncio.TimeoutError())
    assert stage == "timeout" and "cancelled" in message, message


def test_unreadable_page_names_itself():
    """A mislabelled or truncated page says WHICH page, and nothing else.

    Pillow raises UnidentifiedImageError (an OSError) for a PDF sent as .jpg
    and "image file is truncated" for a half-uploaded photo. Neither message
    is actionable; the page number is.
    """
    from vision_downscale import downscale_all

    pages = [
        (_png(64, 64), "image/png"),
        (b"%PDF-1.7 this is not an image", "image/jpeg"),   # wrong extension
        (_png(64, 64), "image/png"),
    ]

    try:
        downscale_all(pages)
        raise AssertionError("an undecodable page should have raised")
    except ValueError as exc:
        assert "Page 2" in str(exc), exc
        assert "BytesIO" not in str(exc), exc          # no Pillow internals
        assert exc.__cause__ is None, exc

    # the single-page path is guarded too, and still reports page 1
    try:
        downscale_all([(b"not an image", "image/png")])
        raise AssertionError("single page should have raised")
    except ValueError as exc:
        assert "Page 1" in str(exc), exc

    # a PDF is forwarded byte-for-byte and never goes near the decoder
    pdf = (b"%PDF-1.7 whatever", "application/pdf")
    assert downscale_all([pdf]) == [pdf]


def test_truncated_response_is_refused_not_returned():
    """Hitting the output cap must not look like a short report.

    parse_partial_json closes the JSON, so a response cut off after test 20
    of 50 validates perfectly and returns 20 tests with no indication that
    30 are missing. finish_reason is the only evidence, so it decides.
    """
    import json

    from langchain_core.messages import AIMessageChunk

    from app.streaming import stream_sections

    report = {
        "patient": {"name": "Padmaram Mali"},
        "sections": [{"category_name": "CBC", "tests": [
            {"name": "Hemoglobin", "result": 13.1}]}],
    }

    class FakeLLM:
        def __init__(self, finish_reason):
            self.finish_reason = finish_reason

        async def astream(self, _messages):
            yield AIMessageChunk(
                content=json.dumps(report),
                response_metadata={"finish_reason": self.finish_reason},
            )

    async def run(finish_reason):
        return [item async for item in stream_sections(FakeLLM(finish_reason), [])]

    # the same bytes, complete, are still a perfectly good report
    kinds = [kind for kind, _ in asyncio.run(run("stop"))]
    assert kinds[-1] == "report", kinds

    try:
        asyncio.run(run("length"))
        raise AssertionError("a cut-off response should have raised")
    except ValueError as exc:
        assert "cut off" in str(exc), exc
        assert "Padmaram" not in str(exc), exc


def test_header_fields_are_all_written_before_sections():
    """The header event only works because of schema field ORDER.

    stream_sections emits the header the moment the `sections` key appears,
    on the argument that every key above it is closed by then. A field listed
    in HEADER_FIELDS but declared after `sections` would silently arrive as
    null on every request, and nothing else would fail.
    """
    from app.streaming.sections import HEADER_FIELDS

    order = list(MedicalReport.model_fields)
    cut = order.index("sections")

    for field in HEADER_FIELDS:
        assert field in order, f"{field} is not a MedicalReport field"
        assert order.index(field) < cut, (
            f"{field} is declared after 'sections', so the header event "
            f"would always send it as null"
        )

    # and every scalar header field the schema has is actually streamed
    missing = [
        f for f in order[:cut]
        if f not in HEADER_FIELDS
    ]
    assert not missing, f"schema fields absent from HEADER_FIELDS: {missing}"


def test_mixed_patient_upload_is_refused_by_code_not_by_the_prompt():
    """Two patients in one upload must never come back as one report.

    extraction.md asks the model to notice the mismatch and return no
    sections. Measured on two pages from two patients that held in 1 run of
    3, and one failure returned both patients' 45 tests under a single name.
    page_patients turns the judgement into transcription the code compares,
    so the refusal no longer depends on the model agreeing to refuse.
    """
    from app.schemas import reject_unusable as _reject_unusable

    def report(page_patients, sections=1):
        return MedicalReport.model_validate({
            "patient": {"name": "KUSHBOO N PATEL"},
            "page_patients": page_patients,
            "sections": [
                {"category_name": "Blood Counts", "tests": [
                    {"name": "Haemoglobin", "result": 11.7}]}
            ] * sections,
        })

    # the failure the model actually produced: data extracted anyway
    try:
        _reject_unusable(report(["KUSHBOO N PATEL", "ANKITABEN B PATEL"]))
        raise AssertionError("a two-patient upload should have been refused")
    except ValueError as exc:
        assert "more than one patient" in str(exc), exc
        # the count is safe to report, the names are not
        assert "KUSHBOO" not in str(exc) and "ANKITABEN" not in str(exc), exc
        assert "2 different names" in str(exc), exc

    # same patient, spelled differently page to page, is still one patient
    for same in (
        ["KUSHBOO N PATEL", "Kushboo N Patel"],
        ["KUSHBOO N PATEL", " KUSHBOO  N  PATEL "],
        ["KUSHBOO N PATEL", "", "KUSHBOO N PATEL"],   # a page with no header
        ["KUSHBOO N PATEL"] * 9,
        [],                                            # model omitted the list
    ):
        _reject_unusable(report(same))

    # and the empty-report case still fires
    try:
        _reject_unusable(report(["KUSHBOO N PATEL"], sections=0))
        raise AssertionError("an empty report should have been refused")
    except ValueError as exc:
        assert "No test results" in str(exc), exc


def test_mixed_upload_fails_before_any_section_is_streamed():
    """No section event may escape for an upload that will be refused.

    page_patients is written before `sections`, so the header event already
    knows every name. Checking only the final report meant a consumer saw one
    patient's results rendered under another patient's header for nine
    seconds before the error frame arrived.
    """
    import json

    from fastapi.testclient import TestClient

    from app import main

    async def mixed(_files):
        yield "header", {
            "patient": {"name": "Padmaram Mali"},
            "page_patients": ["Padmaram Mali", "Padmaram Mali", "SMITH, JOHN"],
        }
        yield "section", {"category_name": "CHEMISTRY", "tests": []}
        yield "report", MedicalReport.model_validate({
            "patient": {"name": "Padmaram Mali"},
            "page_patients": ["Padmaram Mali", "SMITH, JOHN"],
            "sections": [{"category_name": "CHEMISTRY", "tests": [
                {"name": "Glucose", "result": 91}]}],
        })

    original = main.pipeline.stream_pages
    main.pipeline.stream_pages = mixed
    try:
        response = TestClient(main.app).post(
            "/parse", files=[("file", ("p.png", _png(64, 64), "image/png"))]
        )
    finally:
        main.pipeline.stream_pages = original

    frames = [json.loads(l[6:]) for l in response.text.splitlines()
              if l.startswith("data: ")]
    kinds = [f["type"] for f in frames]

    assert "section" not in kinds, f"a section leaked before the refusal: {kinds}"
    assert "header" not in kinds, f"the header leaked: {kinds}"
    assert "result" not in kinds and "complete" not in kinds, kinds
    assert kinds[-1] == "error", frames
    assert "more than one patient" in frames[-1]["message"], frames[-1]
    assert "SMITH" not in frames[-1]["message"], frames[-1]


if __name__ == "__main__":
    test_optional_fields()
    test_indicator_still_constrained()
    test_downscale()
    test_traces_carry_no_payload_or_phi()
    test_grouping_detectors()
    test_multipage_upload()
    test_parse_accepts_one_or_many_on_the_same_field()
    test_downscale_all_is_parallel_and_ordered()
    test_profiles_are_independent_but_share_one_pool()
    test_docs_shows_a_file_picker()
    test_truncated_stream_carries_no_patient_data()
    test_empty_sections_is_an_error_not_an_empty_success()
    test_deadline_holds_against_a_trickling_stream()
    test_error_frames_never_echo_an_upstream_message()
    test_unreadable_page_names_itself()
    test_truncated_response_is_refused_not_returned()
    test_header_fields_are_all_written_before_sections()
    test_mixed_patient_upload_is_refused_by_code_not_by_the_prompt()
    test_mixed_upload_fails_before_any_section_is_streamed()
    test_service_tier_reaches_the_request()
    test_streaming_never_emits_partial_values()
    print("ok")
