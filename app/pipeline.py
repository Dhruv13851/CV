from app.extractors.openai import OpenAIExtractor
from app.ingestion import read_file


class MedicalReportPipeline:

    def __init__(self, settings):
        self.extractor = OpenAIExtractor(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            service_tier=settings.openai_service_tier,
            reasoning_effort=settings.openai_reasoning_effort,
        )

    def process_files(self, file_paths):
        """One report, one call - several paths are its pages, in order."""
        return self.extractor.extract(
            files=[read_file(path) for path in file_paths],
        )

    def stream_pages(self, files):
        """Same call, streamed, for pages already read off the wire.

        Returns the extractor's async generator itself, not a coroutine
        wrapping it - callers drive it with __anext__/aclose (see main.py),
        and an `async def` here would hide both behind one await.
        """
        return self.extractor.extract_sections_async(files=files)
