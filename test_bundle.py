"""Does wrapping the pages in a PDF change the extraction? COSTS MONEY.

Two ways to send the SAME downscaled pixels:

  A. N image parts          - what we do today
  B. one PDF built from those images

OpenAI re-renders PDF pages at its own fixed internal size and bills per page,
so B measured 4.1x cheaper on input tokens for sample2.pdf (38,862 -> 9,580)
with identical output. The open question is whether that resolution loss - it
renders at roughly 70 DPI against our tuned 200 - silently drops rows on a
denser report. This script answers that per document.

    .venv/bin/python test_bundle.py                 # every sample
    .venv/bin/python test_bundle.py samples/sample.pdf

Tracing is left to .env on purpose: runs appear in LangSmith named
`bundle:<doc>:<images|pdf>` so they can be inspected side by side.
"""

import asyncio
import glob
import io
import os
import subprocess
import sys
import tempfile
import time
import warnings

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

from PIL import Image

from app.config import Settings
from vision_downscale import downscale_all, patch_count
from app.extractors.openai import OpenAIExtractor, report_summary
from app.ingestion import read_file
from app.schemas import MedicalReport

RENDER_DPI = 600   # a PDF page becomes an image as if photographed

DOCS = [
    "samples/sample2.pdf",
    "samples/sample.pdf",
    "samples/Paris Lab Report Printing Sample.jpg",
    "samples/sample.jpg",
]


def pages_of(path: str) -> list[tuple[bytes, str]]:
    """The document as one entry per page, before downscaling."""
    if not path.lower().endswith(".pdf"):
        return [read_file(path)]

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["pdftoppm", "-jpeg", "-jpegopt", "quality=92",
             "-r", str(RENDER_DPI), path, f"{tmp}/p"],
            check=True, capture_output=True,
        )
        return [read_file(f) for f in sorted(glob.glob(f"{tmp}/p-*.jpg"))]


def as_pdf(prepared: list[tuple[bytes, str]]) -> bytes:
    """Bundle already-downscaled pages into one PDF, quality kept high.

    Pillow re-encodes on PDF save, so pass quality explicitly - the default
    drops it noticeably (quantisation sum 1846 -> 4638 at defaults).
    """
    images = [Image.open(io.BytesIO(b)).convert("RGB") for b, _ in prepared]
    buffer = io.BytesIO()
    images[0].save(buffer, format="PDF", save_all=True,
                   append_images=images[1:], quality=95, resolution=200.0)
    return buffer.getvalue()


def flat(report: MedicalReport) -> dict:
    """Every extracted fact, keyed by test name, for diffing two runs."""
    out = {}
    for section in report.sections:
        for test in section.tests:
            key = "".join(test.name.split()).lower()
            out[key] = (
                section.category_name, repr(test.result), test.unit,
                test.indicator,
                # key=repr: labels are str|None and bounds float|None, so a
                # plain sort compares str with NoneType and raises
                tuple(sorted(((r.label, r.min_val, r.max_val)
                              for r in test.reference_ranges), key=repr)),
            )
    return out


async def run(extractor, label, files):
    content = extractor._build_content(files)
    started = time.perf_counter()
    response = await extractor.stream_llm.ainvoke(
        [{"role": "user", "content": content}],
        config={"run_name": label},
    )
    elapsed = time.perf_counter() - started

    text = response.content if isinstance(response.content, str) else "".join(
        part.get("text", "") for part in response.content
        if isinstance(part, dict) and part.get("type") == "text"
    )
    from langchain_core.utils.json import parse_partial_json
    report = MedicalReport.model_validate(parse_partial_json(text))
    usage = response.usage_metadata or {}
    return report, usage, elapsed


def stability(flats: list[dict]) -> tuple[dict, dict]:
    """Split fields into those every run agreed on, and those that wobbled."""
    keys = set().union(*(set(f) for f in flats))
    stable, unstable = {}, {}
    for key in keys:
        values = {f.get(key) for f in flats}
        if len(values) == 1:
            stable[key] = values.pop()
        else:
            unstable[key] = values
    return stable, unstable


async def compare_n(extractor, path, n):
    """N runs per mode, then separate mode effects from run-to-run noise.

    A field both modes agree on internally but disagree on across modes is
    caused by the container. A field that wobbles inside either mode is just
    this model being non-deterministic, and proves nothing about the change.
    """
    name = os.path.basename(path)
    prepared = downscale_all(pages_of(path))
    pdf = as_pdf(prepared)
    modes = {"images": prepared, "pdf": [(pdf, "application/pdf")]}

    print(f"=== {name}, {n} runs per mode ===")
    flats, tokens = {}, {}
    for mode, files in modes.items():
        results = await asyncio.gather(*(
            run(extractor, f"bundle:{name}:{mode}:{i}", files) for i in range(n)
        ))
        flats[mode] = [flat(report) for report, _, _ in results]
        tokens[mode] = [u.get("input_tokens", 0) for _, u, _ in results]
        counts = [report_summary(r)["tests"] for r, _, _ in results]
        sections = [report_summary(r)["sections"] for r, _, _ in results]
        print(f"  {mode:6} input={min(tokens[mode]):,}"
              f"{'' if len(set(tokens[mode])) == 1 else f'-{max(tokens[mode]):,}'}"
              f"   tests={counts}   sections={sections}")

    stable_i, wobbly_i = stability(flats["images"])
    stable_p, wobbly_p = stability(flats["pdf"])

    both_stable = set(stable_i) & set(stable_p)
    mode_effects = sorted(k for k in both_stable if stable_i[k] != stable_p[k])
    noise = sorted(set(wobbly_i) | set(wobbly_p))
    only_i = sorted(set(stable_i) - set(stable_p) - set(wobbly_p))
    only_p = sorted(set(stable_p) - set(stable_i) - set(wobbly_i))

    print(f"\n  fields stable in BOTH modes : {len(both_stable)}")
    print(f"  caused by the container     : {len(mode_effects)}")
    print(f"  run-to-run noise            : {len(noise)}"
          f"  (images {len(wobbly_i)}, pdf {len(wobbly_p)})")
    print(f"  present only via images     : {len(only_i)}")
    print(f"  present only via pdf        : {len(only_p)}")

    if mode_effects:
        print("\n  --- MODE EFFECTS (every run agreed within each mode) ---")
        for key in mode_effects:
            print(f"    {key}")
            print(f"        images {stable_i[key]}")
            print(f"        pdf    {stable_p[key]}")
    for key in only_i[:8]:
        print(f"    images only, all runs: {key}")
    for key in only_p[:8]:
        print(f"    pdf only, all runs   : {key}")
    if noise:
        print(f"\n  --- noise, proves nothing either way ---")
        for key in noise[:8]:
            print(f"    {key}"
                  f"{'  (images)' if key in wobbly_i else ''}"
                  f"{'  (pdf)' if key in wobbly_p else ''}")


async def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-n")]
    reps = next((int(a.split("=", 1)[1]) for a in sys.argv[1:]
                 if a.startswith("-n=")), 1)
    docs = args or DOCS
    settings = Settings()
    extractor = OpenAIExtractor(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
    )
    print(f"model={settings.openai_model}  "
          f"tracing={os.getenv('LANGSMITH_TRACING')}  "
          f"project={os.getenv('LANGSMITH_PROJECT')}\n")

    if reps > 1:
        for path in docs:
            await compare_n(extractor, path, reps)
            print()
        return

    for path in docs:
        name = os.path.basename(path)
        raw = pages_of(path)
        prepared = downscale_all(raw)
        size = Image.open(io.BytesIO(prepared[0][0])).size
        pdf = as_pdf(prepared)

        print(f"=== {name} ===")
        print(f"  {len(prepared)} page(s) at {size[0]}x{size[1]}, "
              f"{patch_count(*size)} patches each")
        print(f"  images {sum(len(b) for b, _ in prepared)/1e6:.2f} MB  |  "
              f"pdf {len(pdf)/1e6:.2f} MB")

        results = {}
        for mode, files in (("images", prepared),
                            ("pdf", [(pdf, "application/pdf")])):
            try:
                report, usage, elapsed = await run(
                    extractor, f"bundle:{name}:{mode}", files
                )
            except Exception as exc:
                print(f"  {mode:6} FAILED: {type(exc).__name__}: {str(exc)[:160]}")
                continue
            summary = report_summary(report)
            results[mode] = (report, summary)
            print(f"  {mode:6} input={usage.get('input_tokens', 0):>7,}  "
                  f"output={usage.get('output_tokens', 0):>6,}  "
                  f"{elapsed:5.1f}s   sections={summary['sections']:>2} "
                  f"tests={summary['tests']:>2} "
                  f"largest={summary['largest_category_tests']:>2}")

        if len(results) == 2:
            a, b = flat(results["images"][0]), flat(results["pdf"][0])
            only_images = sorted(set(a) - set(b))
            only_pdf = sorted(set(b) - set(a))
            differing = sorted(k for k in set(a) & set(b) if a[k] != b[k])

            # Equal counts on both sides means the same rows came back under
            # differently transcribed names (Hemoglobin/Haemoglobin,
            # Conc/Concentration). Only an imbalance is real data loss.
            lost = len(only_images) - len(only_pdf)
            verdict = ("NO DATA LOSS - names transcribed differently"
                       if lost == 0 else
                       f"PDF LOST {lost} test(s)" if lost > 0 else
                       f"PDF FOUND {-lost} extra test(s)")
            print(f"  diff: {len(only_images)} only via images, "
                  f"{len(only_pdf)} only via pdf, "
                  f"{len(differing)} differing field(s)  ->  {verdict}")
            for key in only_images[:6]:
                print(f"      images only  {key}  ({a[key][0]})")
            for key in only_pdf[:6]:
                print(f"      pdf only     {key}  ({b[key][0]})")
            for key in differing[:6]:
                print(f"      differs         {key}")
                print(f"          images {a[key]}")
                print(f"          pdf    {b[key]}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
