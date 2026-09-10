import argparse
import json

from .config import Settings
from .schemas import reject_unusable
from .pipeline import MedicalReportPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Extract test results from a medical report."
    )

    parser.add_argument(
        "file",
        nargs="+",
        help=(
            "One PDF, or the pages of one report as images "
            "(JPG, JPEG, PNG, HEIC, HEIF) in page order."
        ),
    )

    args = parser.parse_args()

    settings = Settings()
    pipeline = MedicalReportPipeline(settings)

    # FileNotFoundError is an OSError; ingestion and downscaling raise
    # ValueError. Either way the user wants one line, not a traceback.
    try:
        result = pipeline.process_files(args.file)
        reject_unusable(result)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}")

    print(
        json.dumps(
            result.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()