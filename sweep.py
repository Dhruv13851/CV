"""Latency vs accuracy across reasoning_effort / service_tier. COSTS MONEY.

Runs the real 9-page report through the real streaming path and scores the
result with test_live.py's ground truth, so a faster setting that starts
mis-grouping is visible immediately.

Runs are SEQUENTIAL on purpose - concurrent runs contend and distort the
latency numbers, which are the whole point.

    sweep.py                       # effort scan, 1 run each
    sweep.py -n 3 low medium       # 3 runs each of two settings
    sweep.py --tier fast medium    # same, at the fast service tier
"""

import argparse
import asyncio
import statistics
import time
import warnings

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

from app.config import Settings
from app.extractors.openai import OpenAIExtractor, report_summary
from app.ingestion import read_file
from test_live import TRUE_TESTS, check

DEFAULT_EFFORTS = ["default", "none", "minimal", "low", "medium"]


async def one_run(extractor, files):
    """Return (first_event_s, total_s, report) for a single extraction."""
    started = time.perf_counter()
    first = None
    report = None

    async for kind, payload in extractor.extract_sections_async(files=files):
        if first is None:
            first = time.perf_counter() - started
        if kind == "report":
            report = payload

    return first, time.perf_counter() - started, report


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("efforts", nargs="*", default=DEFAULT_EFFORTS)
    ap.add_argument("-n", type=int, default=1)
    ap.add_argument("--tier", default="default")
    ap.add_argument("--files", nargs="+", default=["samples/sample2.pdf"])
    args = ap.parse_args()

    settings = Settings()
    files = [read_file(p) for p in args.files]

    print(f"model={settings.openai_model} tier={args.tier} "
          f"n={args.n} pages={len(args.files)} truth={TRUE_TESTS} tests\n")
    print(f"{'effort':>9} {'first':>7} {'total':>7} {'secs':>6} "
          f"{'tests':>6} {'chars':>6}  accuracy")

    for effort in args.efforts:
        extractor = OpenAIExtractor(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            service_tier=args.tier,
            reasoning_effort=None if effort == "default" else effort,
        )

        firsts, totals, verdicts = [], [], []
        for _ in range(args.n):
            try:
                first, total, report = await one_run(extractor, files)
            except Exception as exc:
                print(f"{effort:>9}   REJECTED: {type(exc).__name__}: "
                      f"{str(exc)[:90]}")
                firsts = None
                break

            firsts.append(first)
            totals.append(total)
            summary = report_summary(report)
            failures = check(report)
            verdicts.append((not failures, summary, failures))

        if firsts is None:
            continue

        passed = sum(1 for ok, _, _ in verdicts if ok)
        print(f"{effort:>9} {statistics.median(firsts):>6.1f}s "
              f"{statistics.median(totals):>6.1f}s "
              f"{verdicts[0][1]['sections']:>6} "
              f"{statistics.median([s['tests'] for _, s, _ in verdicts]):>6.0f} "
              f"{len(str(verdicts[0][1])):>6}  {passed}/{args.n} match")

        for ok, _, failures in verdicts:
            if not ok:
                for failure in failures[:3]:
                    print(f"              {failure}")
                break


if __name__ == "__main__":
    asyncio.run(main())
