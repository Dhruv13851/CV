from app.extractors.openai import OpenAIExtractor
from app.ingestion import read_file

class MedicalReportPipeline:

    def __init__(self, settings):
        self.extractor = OpenAIExtractor(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
        )

    def process_file(self, file_path):
        file_bytes, media_type = read_file(file_path)

        return self.extractor.extract(
            file_bytes=file_bytes,
            media_type=media_type,
        )