from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    groq_api_key: str | None = Field(
        default=None,
        description="Groq API key for AI-powered triage (optional).",
    )
    groq_model: str = Field(
        default="mixtral-8x7b-32768",
        description="Groq model to use for triage.",
    )
    groq_base_url: str | None = Field(
        default=None,
        description="Base URL for the Groq API (optional).",
    )
