import argparse
import json

from .config import Settings
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

    result = pipeline.process_files(args.file)

    print(
        json.dumps(
            result.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()