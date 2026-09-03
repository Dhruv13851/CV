from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str

    # Standard tier. "fast" is up to 2.5x faster but billed at a premium
    # (2x standard for gpt-5.6-sol) - it was measurably not worth the cost
    # here. Set OPENAI_SERVICE_TIER= (empty) to send no tier at all and let
    # the project default apply instead.
   

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )