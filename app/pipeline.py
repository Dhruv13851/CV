from app.extractors.openai import OpenAIExtractor
from app.ingestion import read_file

class MedicalReportPipeline:

    def __init__(self, settings):
        self.extractor = OpenAIExtractor(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            # service_tier=settings.openai_service_tier,
        )

    def process_files(self, file_paths):
        """One report, one call - several paths are its pages, in order."""
        return self.extractor.extract(
            files=[read_file(path) for path in file_paths],
        )