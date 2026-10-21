from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env")

    groq_api_key: str | None = Field(
        default=None,
        description="Groq API key for AI-powered analysis (optional).",
    )
    groq_model: str | None = Field(
        default=None,
        description="Groq model to use for analysis. Required when GROQ_API_KEY is set.",
    )
    groq_base_url: str | None = Field(
        default=None,
        description="Base URL for the Groq API (optional).",
    )

    database_url: str | None = Field(
        default=None,
        description="PostgreSQL connection string used for session, message, "
        "approval, and audit storage, for example "
        "postgresql://user:password@localhost:5432/ticket_triage. Read from "
        "the DATABASE_URL environment variable. When it is not configured the "
        "application still starts, and the endpoints that need storage answer "
        "with a service-unavailable response.",
    )
