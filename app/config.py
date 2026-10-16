from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    groq_api_key: str | None = Field(
        default=None,
        description="Groq API key for AI-powered triage (optional).",
    )

