from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Default number of stored messages that are replayed into a continuing run.
#:
#: A tool call and its output are always replayed together, so the number of
#: messages that are actually sent can be smaller than this limit.
DEFAULT_MAX_HISTORY_MESSAGES = 20

#: Default number of stored memories that are recalled into a run.
DEFAULT_MAX_MEMORY_ITEMS = 3

#: Default maximum number of characters of recalled memory that are injected
#: into a run, the label of the context included.
DEFAULT_MAX_MEMORY_CHARS = 1200


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

    agent_max_history_messages: int = Field(
        default=DEFAULT_MAX_HISTORY_MESSAGES,
        ge=0,
        description="Maximum number of stored messages of a session that are "
        "replayed into a continuing agent run. Read from the "
        "AGENT_MAX_HISTORY_MESSAGES environment variable. A tool call and "
        "its output are always replayed together, so the effective history "
        "can be shorter than this limit. Zero disables history replay.",
    )

    agent_max_memory_items: int = Field(
        default=DEFAULT_MAX_MEMORY_ITEMS,
        ge=0,
        description="Maximum number of stored memories that are recalled into "
        "a run. Read from the AGENT_MAX_MEMORY_ITEMS environment variable. "
        "Zero disables the recall.",
    )

    agent_max_memory_chars: int = Field(
        default=DEFAULT_MAX_MEMORY_CHARS,
        ge=0,
        description="Maximum number of characters of recalled memory that are "
        "injected into a run, the label of the context included. Read from "
        "the AGENT_MAX_MEMORY_CHARS environment variable. Zero disables the "
        "recall.",
    )
