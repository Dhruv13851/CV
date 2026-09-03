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
        help="Path to PDF, JPG, JPEG, PNG, HEIC, or HEIF report.",
    )

    args = parser.parse_args()

    settings = Settings()
    pipeline = MedicalReportPipeline(settings)

    result = pipeline.process_file(args.file)

    print(
        json.dumps(
            result.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()