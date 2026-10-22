"""Typed records for the rows stored by the persistence layer.

PostgreSQL returns ``UUID`` columns as ``uuid.UUID`` and ``TIMESTAMPTZ``
columns as aware ``datetime`` objects. The records below deliberately expose
plain strings so that the rest of the application never depends on the
driver's value types.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

#: Roles a stored message can have.
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"
ROLE_SYSTEM = "system"

#: Session lifecycle states.
SESSION_STATUS_OPEN = "open"
SESSION_STATUS_CLOSED = "closed"


#: Kinds a stored memory can have. The database enforces the same set.
MEMORY_KIND_SUMMARY = "summary"
MEMORY_KIND_FACT = "fact"
MEMORY_KIND_PREFERENCE = "preference"
MEMORY_KIND_OUTCOME = "outcome"

#: Every kind a memory may have.
MEMORY_KINDS = (
    MEMORY_KIND_SUMMARY,
    MEMORY_KIND_FACT,
    MEMORY_KIND_PREFERENCE,
    MEMORY_KIND_OUTCOME,
)


@dataclass(frozen=True)
class Session:
    """A conversation session."""

    id: str
    created_at: str
    updated_at: str
    status: str
    ticket_id: str | None
    title: str | None


@dataclass(frozen=True)
class Message:
    """A single stored transcript entry."""

    id: int
    session_id: str
    seq: int
    role: str
    content: str
    call_id: str | None
    tool_name: str | None
    correlation_id: str | None
    created_at: str


@dataclass(frozen=True)
class MessageDraft:
    """A message that has not been written yet."""

    role: str
    content: str
    call_id: str | None = None
    tool_name: str | None = None


@dataclass(frozen=True)
class Memory:
    """A derived, long-term memory of a session.

    A memory is not authoritative: it is a distilled fact about a session,
    kept so that it can be found again later. It belongs to exactly one
    session.
    """

    id: int
    session_id: str
    kind: str
    content: str
    source_msg_id: int | None
    created_at: str


@dataclass(frozen=True)
class MemoryDraft:
    """A memory that has not been written yet."""

    kind: str
    content: str
    source_msg_id: int | None = None


def to_iso8601(value: Any) -> str:
    """Render a timestamp as an ISO-8601 string in UTC.

    PostgreSQL returns ``TIMESTAMPTZ`` values as aware ``datetime`` objects.
    They are converted to UTC and rendered with a ``Z`` suffix, matching the
    format used for the identifiers and messages the API exposes.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, str):
        return value
    return str(value)


def session_from_row(row: Mapping[str, Any]) -> Session:
    """Build a :class:`Session` from a database row."""
    return Session(
        id=str(row["id"]),
        created_at=to_iso8601(row["created_at"]),
        updated_at=to_iso8601(row["updated_at"]),
        status=row["status"],
        ticket_id=row["ticket_id"],
        title=row["title"],
    )


def message_from_row(row: Mapping[str, Any]) -> Message:
    """Build a :class:`Message` from a database row."""
    correlation_id = row["correlation_id"]
    return Message(
        id=int(row["id"]),
        session_id=str(row["session_id"]),
        seq=int(row["seq"]),
        role=row["role"],
        content=row["content"],
        call_id=row["call_id"],
        tool_name=row["tool_name"],
        correlation_id=(
            str(correlation_id) if correlation_id is not None else None
        ),
        created_at=to_iso8601(row["created_at"]),
    )


def memory_from_row(row: Mapping[str, Any]) -> Memory:
    """Build a :class:`Memory` from a database row."""
    source_msg_id = row["source_msg_id"]
    return Memory(
        id=int(row["id"]),
        session_id=str(row["session_id"]),
        kind=row["kind"],
        content=row["content"],
        source_msg_id=(
            int(source_msg_id) if source_msg_id is not None else None
        ),
        created_at=to_iso8601(row["created_at"]),
    )
