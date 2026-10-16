from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env")

    groq_api_key: str | None = Field(
        default=None,
        description="Groq API key for AI-powered triage (optional).",
    )
    groq_model: str | None = Field(
        default=None,
        description="Groq model to use for triage. Required when GROQ_API_KEY is set.",
    )
    groq_base_url: str | None = Field(
        default=None,
        description="Base URL for the Groq API (optional).",
    )
