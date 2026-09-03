# Medical Report Parser

Extracts structured lab results from a medical report (PDF / JPG / PNG / HEIC) using an
OpenAI vision model. Output is a validated Pydantic schema: patient, lab, doctor,
and every test with its result, unit, printed reference ranges and a
normal/borderline/abnormal indicator.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env      # then fill in OPENAI_API_KEY
```

`OPENAI_SERVICE_TIER` defaults to `default` (standard tier). Set it to `fast`
for OpenAI Fast mode - up to 2.5x faster, but billed at a premium (2x standard
for gpt-5.6-sol), which is why it is not the default here. Set the variable to
empty to send no tier at all and take the project default instead.

## Run

API:

```bash
.venv/bin/uvicorn app.main:app --reload
```

- `GET /` -> stream monitor, a dev UI for watching sections arrive live
- `GET /health` -> `{"status": "ok"}`
- `POST /parse` -> multipart upload, returns a Server-Sent Events stream

Open <http://127.0.0.1:8000/> , pick a report and hit Parse. Left pane is the
timestamped event log, right pane renders each category as it lands - dashed
border while provisional, solid once the validated `result` replaces them.

```bash
curl -N -F "file=@samples/sample.pdf" http://127.0.0.1:8000/parse
```

Event sequence:

```
status    stage=received    bytes
status    stage=analyzing   deadline_seconds
heartbeat elapsed_seconds        <- every 10s while nothing else is arriving
section   index, data            <- one per category, as the model closes it
section   index, data
result    data, elapsed_seconds  <- validated MedicalReport
complete
```

### Why category-at-a-time, not token-by-token

Token-level parsing of lab values is unsafe. A partial parse of `118` renders
as `1`, then `11`, then `118` - and every intermediate is a plausible result.
A glucose of 1 instead of 118, a potassium of 5 instead of 5.9. Those look
like readings, not like errors.

`app/streaming/` emits **completed objects** instead. In a JSON array, once
element N+1 starts, element N is closed - so `sections[:-1]` are always whole,
nested tests included. Only the in-flight category is held back, and it
arrives with the final `result`.

Streamed `section` events are structurally complete but **not Pydantic
validated** - treat them as provisional render, and `result` as truth.

This binds `response_format` (the same strict json_schema
`with_structured_output` uses), so OpenAI enforces the schema exactly as
before and the JSON arrives as message content.

**Not `bind_tools`.** Reasoning models reject function tools on
`/v1/chat/completions`:

> Function tools with reasoning_effort are not supported for gpt-5.6-luna.
> To use function tools, use /v1/responses or set reasoning_effort to 'none'.

`response_format` has no such restriction, so reasoning stays on. The parser
reads `.content` (string, or text blocks on reasoning models, skipping the
reasoning) and still falls back to `.tool_call_chunks` if a future model needs
the tool path.

`with_structured_output` cannot stream:
it ends in a `RunnableLambda`, which buffers the whole response before
emitting, so `astream()` yields exactly one item. `streaming=True` on the
client does not change that - it is a transport flag and the parser step is
identical either way. It is set anyway, for the read-timeout reason under
Timeouts below.

The heartbeat remains for the gaps between categories: without traffic an idle
proxy (nginx defaults to 60s) kills a request that is working fine, and the
timeouts below never get to decide the outcome. `HEARTBEAT_INTERVAL` must stay
below the shortest idle timeout in front of the service.

If the client disconnects mid-stream, the in-flight `__anext__` is cancelled
and awaited before `aclose()`, then the upstream generator is closed - so a
dead connection stops costing OpenAI credits.

CLI (prints JSON to stdout):

```bash
.venv/bin/python -m app.cli samples/sample.pdf
```

## Tests

```bash
.venv/bin/python test_app.py
```

## Layout

```
app/
  main.py            FastAPI app: /, /health, /parse
  cli.py             command-line entrypoint
  pipeline.py        wires settings -> extractor
  config.py          env-backed settings + PROMPTS_DIR
  ingestion.py       file reading, extension -> media type//supprts image list/array 
  prompts/
    extraction.md    the system prompt - edit without touching code
  schemas/
    medical_report.py  Pydantic output schema
  downscaling/
    image.py         shrink images under OpenAI's patch limit
  streaming/
    sections.py      emit whole categories as the model closes them
  static/
    index.html       stream monitor UI served at /
  extractors/
    openai.py        the extractor - OpenAI is the only provider
samples/             test documents (gitignored - may contain patient data)
```

The prompt is plain Markdown at `app/prompts/extraction.md` and is read at
import. Edit it and restart; no code change needed.

## Supported input

PDF, JPG, JPEG, PNG, HEIC, HEIF.

**One PDF, or up to 12 images that are the pages of one report** - never a mix.
`/parse` takes repeated `file` parts, so a single-file POST still works
unchanged. Upload order is page order and is never sorted (`IMG_9` sorts after
`IMG_10`). All pages go in one request so the model can merge a table that
continues onto the next page; per-page calls would re-create the flat-grouping
bug in Python.

`MAX_PAGES = 12` and `MAX_UPLOAD_BYTES = 30 MB` in `app/ingestion.py` are spend
caps, not API limits: OpenAI allows 30000 patches **per image**, 1500 images and
512 MB per request. A photographed page costs ~4,300 image tokens against ~1,150
for a text PDF page, so photographing a report you already have as a PDF is
roughly 3x the price.

HEIC is what iPhones and recent Android phones shoot by default. Neither
Pillow nor OpenAI's vision endpoint can read it, so `pillow-heif` decodes it
locally and it is always sent as JPEG - including when it is small enough to
skip resizing. EXIF orientation is applied at the same time, otherwise a
portrait phone photo reaches the model rotated 90 degrees.

`/parse` derives the media type from the filename, not the browser's
`Content-Type`: Safari sends `image/heic` for the same file Chrome on Android
sends as `application/octet-stream`.

OpenAI rejects images above 30000 patches, where
`patches = ceil(width/32) * ceil(height/32)`. A 600 DPI Letter scan
(5100x6600) is 33120 and returns HTTP 400.

`app/downscaling/` handles this automatically: images over the cap are
flattened to RGB, resized with LANCZOS to a 2200px long edge (~200 DPI,
3726 patches) and re-encoded as 4:4:4 JPEG. Images already under the cap are
passed through untouched, unless the format is one OpenAI cannot read. Tune
`MAX_LONG_EDGE` in `app/downscaling/image.py`.

## Timeouts

| Setting | Where | Value |
|---|---|---|
| `REQUEST_TIMEOUT` | `extractors/openai.py` | 120s per attempt |
| `MAX_RETRIES` | `extractors/openai.py` | 1 (worst case 240s) |
| `EXTRACTION_DEADLINE` | `main.py` | 300s hard ceiling |

They multiply - raise `REQUEST_TIMEOUT` and `MAX_RETRIES` only together, and
keep `EXTRACTION_DEADLINE` above their product. LangChain discards the OpenAI
SDK's default timeout and leaves httpx on `Timeout(None)`, so setting these
explicitly is what stops a hung call holding a task, a thread and a pool slot
forever.

`streaming=True` is set on the client. httpx applies its read timeout *between*
reads, so without it the whole generation counts as one read and a report that
legitimately takes longer than `REQUEST_TIMEOUT` to write would fail while
working fine. Streaming resets that clock per chunk, so only a real stall trips
it.

## Tracing

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env`. With it off,
every span is a no-op — no code branching needed.

Each upload produces one trace:

```
parse_report                 in:  media_type, upload_bytes, extension, model
                             out: sections_streamed, tests
  build_content              media_type, input_bytes, base64_payload_mb
    downscale_image          width/height before+after, patches before+after,
                             patch_reduction, resized, source_format/mode
  ChatOpenAI                 auto-traced by LangChain: tokens, latency, cost
```

The non-streaming `extract` / `extract_async` methods carry their own
`@traceable` spans reporting sections, tests, `with_reference_range` and the
`indicator_*` counts. The CLI goes through `extract`.

`downscale_image` is where you can see the fix working — `patches_before:
33120` next to `patches_after: 3726` on an oversized scan.

The `indicator_null` count is the quality signal worth watching: a spike means
the model stopped being able to match results to reference ranges.

**PHI warning.** Our own spans are redacted — `process_inputs`/`process_outputs`
log sizes and counts only, never image bytes or patient identity. But LangChain
**auto-traces the ChatOpenAI call**, and that span contains the full base64
document and the complete extraction, patient name included. Turning tracing on
sends medical records to LangSmith. Decide that deliberately.

## Known gaps

Tracked, not yet done:

- Image downscaling is in; PDF page count is still unbounded.
- No auth, so a page cap bounds one request but not a thousand of them.
- No auth on `/parse`.
- No structured logging.
- Patient data is sent to OpenAI and appears in every response body; retention
  and PHI handling are undecided.
- With `LANGSMITH_TRACING=true`, LangChain's own span around the model call
  carries the base64 document and the extracted patient name. Our spans are
  redacted; that one is not.
