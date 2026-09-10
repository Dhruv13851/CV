from json import JSONDecodeError
from typing import AsyncIterator

from langchain_core.utils.json import parse_partial_json
from pydantic import ValidationError

from app.schemas import MedicalReport

# Written before `sections` in the schema, so the model emits them first and
# they are closed by the time the `sections` key appears. Reorder the schema
# and the header event degrades to nulls - `result` stays correct regardless.
HEADER_FIELDS = (
    "patient", "lab_name", "doctor_name", "referring_doctor",
    "report_date", "report_title", "page_patients",
)




def _fragment(chunk) -> str:
    """Pull JSON text out of a chunk, whatever shape the model sends.

    response_format puts it in .content - a plain string, or a list of blocks
    on reasoning models, where the reasoning blocks must be skipped. Tool
    calling puts it in .tool_call_chunks instead.
    """
    content = getattr(chunk, "content", None)

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            part.get("text") or ""
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        text = ""

    return text + "".join(
        part.get("args") or ""
        for part in (getattr(chunk, "tool_call_chunks", None) or [])
    )


async def stream_sections(tool_llm, messages) -> AsyncIterator[tuple]:
    """Yield ("header", dict), ("section", dict) per category, ("report", model).

    The final report is validated; the header and streamed sections are raw
    dicts and should be treated as provisional until the report arrives.
    """
    buffer = ""
    emitted = 0
    header_sent = False
    finish_reason = None

    async for chunk in tool_llm.astream(messages):
        finish_reason = (
            getattr(chunk, "response_metadata", None) or {}
        ).get("finish_reason") or finish_reason

        fragment = _fragment(chunk)
        if not fragment:
            continue

        buffer += fragment

        # ponytail: re-parsing the whole buffer every chunk is O(n^2). A
        # closing brace is the only thing that can complete a section, so
        # skip the parse otherwise - and that guard is what makes it cheap:
        # measured on samples/sample2.pdf, 2144 content chunks produced only
        # 104 parses, 34.5ms total, 0.14% of a 24s stream, worst single
        # event-loop stall 0.85ms.
        #
        # It is O(n^2) all the same - parse CPU 4x per doubling of the JSON:
        # 8k chars 28ms, 16k 114ms, 33k 439ms, 65k 1.8s. Swap in an
        # incremental parser past ~30k chars of report JSON (~50 sections),
        # where it turns into a half-second of blocked event loop; a 12-page
        # report today is nowhere near that.
        if "}" not in fragment:
            continue

        partial = parse_partial_json(buffer)
        if not isinstance(partial, dict):
            continue

        # Once `sections` exists, every key above it is closed - the same
        # argument that makes sections[:-1] safe. Emitting sooner would
        # render "Padmaram Mali" as "P", then "Pad".
        if not header_sent and "sections" in partial:
            header_sent = True
            yield "header", {k: partial.get(k) for k in HEADER_FIELDS}

        sections = partial.get("sections")
        if not isinstance(sections, list):
            continue

        for section in sections[:-1][emitted:]:
            emitted += 1
            yield "section", section

    if not buffer:
        raise ValueError("Model returned no tool call.")

    # A response cut off at the output-token cap must never be returned as a
    # result. parse_partial_json closes the JSON for us, so a truncated report
    # can validate cleanly while silently missing every test after the cut.
    if finish_reason == "length":
        raise ValueError(
            "The report was too large to extract in one pass and the "
            "response was cut off. Upload fewer pages at a time."
        )

    # Pydantic quotes the offending input back inside its message, and here
    # that input IS the report - patient name, test names, values. Because
    # ValidationError subclasses ValueError, main.py's handler would have put
    # all of it into the SSE error frame and the logs. Build the message from
    # field LOCATIONS only, and raise it OUTSIDE the except block: `from None`
    # clears __cause__ but leaves __context__ holding the original.
    problem = None

    try:
        report = MedicalReport.model_validate(parse_partial_json(buffer))
    except JSONDecodeError:
        problem = "the response was not valid JSON"
    except ValidationError as exc:
        where = ", ".join(
            ".".join(str(part) for part in error["loc"]) or "<root>"
            for error in exc.errors()[:3]
        )
        problem = f"{exc.error_count()} field(s) did not validate: {where}"

    if problem is not None:
        raise ValueError(
            f"The model returned a malformed report ({problem}). Please retry."
        )

    yield "report", report
