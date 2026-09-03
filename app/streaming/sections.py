"""Emit whole test categories as the model finishes writing them.

Token-by-token streaming is unsafe for lab values: a partial parse renders
GLUCOSE 118 as 1, then 11, then 118 - every intermediate a plausible result.

This streams completed objects instead. In a JSON array, once element N+1
has started, element N is closed - so `sections[:-1]` are always whole,
nested tests included. Only the in-flight category is held back, and it
arrives with the final validated report.

The schema is enforced with response_format, not bind_tools: reasoning models
reject function tools on /v1/chat/completions, and response_format is just as
strict without giving up reasoning.
"""

from typing import AsyncIterator

from langchain_core.utils.json import parse_partial_json

from app.schemas import MedicalReport




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
    """Yield ("section", dict) as each category closes, then ("report", model).

    The final report is validated; streamed sections are raw dicts and should
    be treated as provisional until the report arrives.
    """
    buffer = ""
    emitted = 0

    async for chunk in tool_llm.astream(messages):
        fragment = _fragment(chunk)
        if not fragment:
            continue

        buffer += fragment

        # ponytail: re-parsing the whole buffer every chunk is O(n^2). A
        # closing brace is the only thing that can complete a section, so
        # skip the parse otherwise. Swap in an incremental parser if
        # reports ever get big enough for this to show up.
        if "}" not in fragment:
            continue

        partial = parse_partial_json(buffer)
        if not isinstance(partial, dict):
            continue

        sections = partial.get("sections")
        if not isinstance(sections, list):
            continue

        for section in sections[:-1][emitted:]:
            emitted += 1
            yield "section", section

    if not buffer:
        raise ValueError("Model returned no tool call.")

    yield "report", MedicalReport.model_validate(parse_partial_json(buffer))
