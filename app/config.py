from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str

    # The two measured latency levers, both defaulting to today's behaviour.
    # Baseline on samples/sample2.pdf (9 pages): 12.8s to first event, 26.0s
    # total. service_tier="fast" -> 7.4s / 13.0s at 2x the price.
    # reasoning_effort="low" -> 7.6s / 21.5s for free, but it changes what the
    # model computes, so re-run test_live.py before trusting it. "none" is
    # faster still (3.6s / 15.5s) and demonstrably mis-groups; "minimal" is
    # rejected by gpt-5.6-luna. Empty sends nothing and lets the default apply.
    openai_service_tier: str = "default"
    openai_reasoning_effort: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )