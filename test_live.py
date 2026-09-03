"""Live extraction check against ground truth. COSTS MONEY - run it by hand.

Deliberately NOT imported by test_app.py: every run makes real OpenAI calls.

    .venv/bin/python test_live.py                      # the PDF, 1 run
    .venv/bin/python test_live.py -n 5                  # the PDF, 5 runs
    .venv/bin/python test_live.py p1.jpg p2.jpg ...     # the same report as photos

TRUTH below is derived from `pdftotext -layout samples/sample2.pdf` - the
printed page decides the grouping, never medical knowledge. That distinction
is the whole point: the platelet rows are printed BELOW the
"Differential % WBCs count" header, so they belong to it even though a
clinician would file them under the CBC.

A name in (parens) is not printed anywhere on the page. Those rows have no
header in their own table, so the prompt's standard-panel fallback names them,
and any name from PANELS is acceptable.
"""

import argparse
import asyncio
import os
import warnings

from dotenv import load_dotenv

load_dotenv()
# The unresolved PHI decision: LangChain's own span around the model call
# carries the base64 document and the patient name. Never send that from a
# test run, whatever .env says.
os.environ["LANGSMITH_TRACING"] = "false"
warnings.filterwarnings("ignore")

from app.config import Settings
from app.extractors.openai import OpenAIExtractor, _report_summary
from app.ingestion import read_file

TRUTH = [
    ("(Complete Blood Count)", [
        "Hemoglobin", "Packed Cell Volume (HCT)", "R.B.C. Count",
        "Mean Cell Volume(MCV)", "Mean Cell Hemoglobin( MCH)",
        "Mean Cell Hb Conc(MCHC)", "RDW (CV)", "Total WBC Count"]),
    ("Differential % WBCs count", [
        "Neutrophils", "Lymphocytes", "Eosinophils", "Monocytes", "Basophils",
        "Platelet Count", "Mean Platelet Volume (MPV)",
        "Platelet Distribution Width (PDW)", "PCT (Platelet Crit)"]),
    ("Peripheral Blood Smear", [
        "RBC Morphology", "WBC Morphology", "Platelets Morphology"]),
    ("ESR (ERYTHROCYTE SEDIMENTATION RATE)", ["1 hour"]),
    ("(Biochemistry)", [
        "Random Blood Sugar", "S. Creatinine", "Blood Urea", "SGPT (ALT)"]),
    ("(Blood Group)", ["Blood Group"]),
    ("(Serology)", ["HIV 1 and 2", "HBsAg (Australia Antigen) - Rapid"]),
    ("(Coagulation)", [
        "Prothrombin Time", "Control (MNPT)", "PT (INR) Value"]),
    ("Serum Electrolytes", [
        "Sodium (Na+)", "Potassium (K+)", "Chlorides (Cl-)"]),
    ("Physical Examination", [
        "Colour", "Transparency (Appearance)", "Reaction(pH)",
        "Specific Gravity"]),
    ("Chemical Examination", [
        "Urine Protein (Albumin)", "Urine Glucose (Sugar)", "Bile Salt",
        "Bile Pigment"]),
    ("Microscopic Examination", [
        "W.B.C. (Pus Cells)", "Epithelial cells", "Red Blood Cells", "Casts",
        "Crystals", "Bacteria", "Yeast"]),
    ("C- Reactive Protein (CRP)", ["CRP (C-Reactive Protein)"]),
]

# The only names the standard-panel fallback may invent, per the prompt.
PANELS = {"Complete Blood Count", "Biochemistry", "Serology", "Coagulation",
          "Blood Group", "Urine Routine"}

def norm(name: str) -> str:
    """Compare names ignoring whitespace and case.

    Reading a label off a photo introduces spacing a text PDF does not: the
    page prints "Mean Cell Hemoglobin( MCH)" and from a rendered image the
    model returns "Mean Cell Hemoglobin( MCH )". Same row, one space apart.
    Only the comparison is normalised - the report keeps what was printed.
    """
    return "".join(name.split()).lower()


TRUE_GROUPS = [frozenset(norm(t) for t in tests) for _, tests in TRUTH]
TRUE_TESTS = sum(len(tests) for _, tests in TRUTH)


def check(report) -> list[str]:
    """Return a list of failures. Empty means it matches the printed page."""
    failures = []
    got = [(s.category_name, frozenset(norm(t.name) for t in s.tests))
           for s in report.sections]

    if [g for _, g in got] != TRUE_GROUPS:
        failures.append(
            f"partition differs: {len(got)} sections vs {len(TRUTH)}"
        )
        known = set().union(*TRUE_GROUPS)
        for name, group in got:
            if group not in TRUE_GROUPS:
                extra = sorted(group - known)
                failures.append(
                    f"  {name!r} ({len(group)} tests) is not a printed group"
                    + (f", unknown names {extra}" if extra else "")
                )
        for want, group in zip(TRUTH, TRUE_GROUPS):
            if group not in [g for _, g in got]:
                failures.append(f"  missing group {want[0]!r}")

    # names: printed headers must match the page, fallbacks must come from the list
    for (want, _), section in zip(TRUTH, report.sections):
        got_name = section.category_name
        if want.startswith("("):
            if got_name not in PANELS:
                failures.append(
                    f"fallback name {got_name!r} is not one of PANELS"
                )
        elif got_name != want:
            failures.append(f"name {got_name!r} should be the printed {want!r}")

    summary = _report_summary(report)
    if summary["tests"] != TRUE_TESTS:
        failures.append(
            f"{summary['tests']} tests, the page prints {TRUE_TESTS}"
        )
    for detector in ("duplicate_test_names", "duplicate_category_names"):
        if summary[detector]:
            failures.append(f"{detector} = {summary[detector]}")
    if summary["suspect_flat_grouping"]:
        failures.append("suspect_flat_grouping fired")

    return failures


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", default=["samples/sample2.pdf"],
                    help="one PDF, or the report's pages as images in order")
    ap.add_argument("-n", type=int, default=1, help="runs (they go concurrently)")
    args = ap.parse_args()

    settings = Settings()
    extractor = OpenAIExtractor(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
    )
    files = [read_file(path) for path in args.files]

    print(f"{len(files)} page(s), {args.n} run(s), model={settings.openai_model}")
    for path, (data, media_type) in zip(args.files, files):
        print(f"  {path}  {media_type}  {len(data)/1e6:.2f} MB")

    reports = await asyncio.gather(*(
        extractor.extract_async(files=files) for _ in range(args.n)
    ))

    passed = 0
    for i, report in enumerate(reports):
        failures = check(report)
        passed += not failures
        summary = _report_summary(report)
        print(f"\nrun {i}: sections={summary['sections']} "
              f"tests={summary['tests']} "
              f"largest={summary['largest_category_tests']} "
              f"-> {'MATCHES THE PAGE' if not failures else 'DIFFERS'}")
        for failure in failures:
            print(f"    {failure}")

    print(f"\n{passed}/{args.n} match the printed page")
    return 0 if passed == args.n else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
