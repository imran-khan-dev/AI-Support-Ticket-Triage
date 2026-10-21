"""Durable storage for sessions, messages, memories, approvals, and audit events.

The package is built on PostgreSQL. It holds storage concerns only: the agent
and HTTP layers live elsewhere.
"""

from app.persistence.db import (
    DATABASE_URL_ENV,
    Database,
    DatabaseConfigurationError,
    build_pool,
    close_pools,
    get_database,
    get_pool,
    resolve_database_url,
)
from app.persistence.migrations import (
    MIGRATIONS,
    apply_migrations,
    current_version,
)
from app.persistence.models import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    SESSION_STATUS_CLOSED,
    SESSION_STATUS_OPEN,
    Message,
    MessageDraft,
    Session,
)
from app.persistence.repository import (
    UnknownSessionError,
    append_messages,
    count_messages,
    create_session,
    get_session,
    list_messages,
    list_sessions,
    utc_now,
)

__all__ = [
    "DATABASE_URL_ENV",
    "Database",
    "DatabaseConfigurationError",
    "MIGRATIONS",
    "Message",
    "MessageDraft",
    "ROLE_ASSISTANT",
    "ROLE_SYSTEM",
    "ROLE_TOOL",
    "ROLE_USER",
    "SESSION_STATUS_CLOSED",
    "SESSION_STATUS_OPEN",
    "Session",
    "UnknownSessionError",
    "append_messages",
    "apply_migrations",
    "build_pool",
    "close_pools",
    "count_messages",
    "create_session",
    "current_version",
    "get_database",
    "get_pool",
    "get_session",
    "list_messages",
    "list_sessions",
    "resolve_database_url",
    "utc_now",
]
