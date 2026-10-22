"""Data access for sessions, messages, and memories.

This layer knows about storage only. It has no knowledge of the agent, the
provider, or the HTTP layer, so it can be tested on its own.

Every statement is written for PostgreSQL: ``%s`` placeholders, generated
identifiers read back with ``RETURNING``, and transcript sequence numbers
allocated from the ``sessions.next_seq`` counter so that two concurrent
appends to the same session can never reserve the same number.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Sequence

from app.persistence.db import Database
from app.persistence.models import (
    SESSION_STATUS_CLOSED,
    SESSION_STATUS_OPEN,
    Memory,
    MemoryDraft,
    Message,
    MessageDraft,
    Session,
    memory_from_row,
    message_from_row,
    session_from_row,
)


class UnknownSessionError(LookupError):
    """Raised when a write targets a session that does not exist."""


#: Maximum number of characters a stored memory may hold. A memory is a
#: distilled fact, not a transcript, so the content is capped when it is
#: stored. A memory can therefore never flood a prompt.
MEMORY_CONTENT_MAX_CHARS = 1000

#: Default number of memories a search returns.
MEMORY_SEARCH_RESULT_LIMIT = 10


def _canonical_session_id(session_id: str) -> str | None:
    """Return the canonical form of a session id, or ``None`` when invalid.

    PostgreSQL stores the session id in a ``UUID`` column, so a value that is
    not a well-formed identifier can never match a row. Returning ``None``
    lets the read helpers answer "no such session" instead of raising a
    database error for a value the caller already rejected.
    """
    try:
        return str(uuid.UUID(str(session_id)))
    except (ValueError, AttributeError, TypeError):
        return None


def utc_now() -> datetime:
    """Return the current time as an aware UTC ``datetime``."""
    return datetime.now(timezone.utc)


def create_session(
    db: Database,
    session_id: str | None = None,
    title: str | None = None,
    ticket_id: str | None = None,
) -> Session:
    """Create a session row and return it.

    The session id is generated server-side unless one is supplied by the
    caller of this function.
    """
    resolved_id = session_id or str(uuid.uuid4())
    now = utc_now()
    with db.write_transaction() as conn:
        row = conn.execute(
            "INSERT INTO sessions"
            " (id, created_at, updated_at, status, ticket_id, title)"
            " VALUES (%s, %s, %s, %s, %s, %s)"
            " RETURNING id, created_at, updated_at, status, ticket_id, title",
            (resolved_id, now, now, SESSION_STATUS_OPEN, ticket_id, title),
        ).fetchone()
    if row is None:
        raise UnknownSessionError(resolved_id)
    return session_from_row(row)


def get_session(db: Database, session_id: str) -> Session | None:
    """Return the session with this id, or ``None`` when it does not exist."""
    canonical = _canonical_session_id(session_id)
    if canonical is None:
        return None
    with db.connection() as conn:
        row = conn.execute(
            "SELECT id, created_at, updated_at, status, ticket_id, title"
            " FROM sessions WHERE id = %s",
            (canonical,),
        ).fetchone()
    if row is None:
        return None
    return session_from_row(row)


def list_sessions(db: Database) -> list[Session]:
    """Return every session, oldest first."""
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT id, created_at, updated_at, status, ticket_id, title"
            " FROM sessions ORDER BY created_at, id"
        ).fetchall()
    return [session_from_row(row) for row in rows]


def set_session_status(
    db: Database,
    session_id: str,
    status: str,
) -> Session | None:
    """Set the lifecycle status of a session and return the updated row.

    Returns ``None`` when the session does not exist.
    """
    if status not in (SESSION_STATUS_OPEN, SESSION_STATUS_CLOSED):
        raise ValueError(f"Unsupported session status: {status!r}")

    canonical = _canonical_session_id(session_id)
    if canonical is None:
        return None

    now = utc_now()
    with db.write_transaction() as conn:
        row = conn.execute(
            "UPDATE sessions SET status = %s, updated_at = %s WHERE id = %s"
            " RETURNING id, created_at, updated_at, status, ticket_id, title",
            (status, now, canonical),
        ).fetchone()
    if row is None:
        return None
    return session_from_row(row)


def close_session(db: Database, session_id: str) -> Session | None:
    """Close a session so that it no longer accepts new runs."""
    return set_session_status(db, session_id, SESSION_STATUS_CLOSED)


def reopen_session(db: Database, session_id: str) -> Session | None:
    """Reopen a closed session."""
    return set_session_status(db, session_id, SESSION_STATUS_OPEN)


def append_messages(
    db: Database,
    session_id: str,
    drafts: Sequence[MessageDraft],
    correlation_id: str | None = None,
) -> list[Message]:
    """Append messages to a session and return the stored rows.

    The sequence numbers are reserved in a single statement that increments
    the session counter and reads the new value back, so a batch is written
    contiguously and the ordering of a session stays deterministic even when
    two requests append to the same session at the same time.
    """
    if not drafts:
        return []

    canonical = _canonical_session_id(session_id)
    if canonical is None:
        raise UnknownSessionError(session_id)

    now = utc_now()
    stored: list[Message] = []

    with db.write_transaction() as conn:
        reserved = conn.execute(
            "UPDATE sessions SET next_seq = next_seq + %s WHERE id = %s"
            " RETURNING next_seq",
            (len(drafts), canonical),
        ).fetchone()
        if reserved is None:
            raise UnknownSessionError(session_id)
        first_seq = int(reserved["next_seq"]) - len(drafts)

        for offset, draft in enumerate(drafts):
            row = conn.execute(
                "INSERT INTO messages"
                " (session_id, seq, role, content, call_id, tool_name,"
                "  correlation_id, created_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                " RETURNING id, session_id, seq, role, content, call_id,"
                "  tool_name, correlation_id, created_at",
                (
                    canonical,
                    first_seq + offset,
                    draft.role,
                    draft.content,
                    draft.call_id,
                    draft.tool_name,
                    correlation_id,
                    now,
                ),
            ).fetchone()
            stored.append(message_from_row(row))

        conn.execute(
            "UPDATE sessions SET updated_at = %s WHERE id = %s",
            (now, canonical),
        )

    return stored


def list_messages(
    db: Database,
    session_id: str,
    limit: int | None = None,
) -> list[Message]:
    """Return the stored messages of a session in sequence order.

    Only messages of the given session are returned: the session id is part
    of the query, so a session never sees the messages of another one.

    When ``limit`` is given, only the most recent messages are returned. The
    limit is applied by the database, and the result is still ordered by
    sequence number, oldest first.
    """
    canonical = _canonical_session_id(session_id)
    if canonical is None:
        return []
    with db.connection() as conn:
        if limit is None:
            rows = conn.execute(
                "SELECT id, session_id, seq, role, content, call_id, tool_name,"
                " correlation_id, created_at"
                " FROM messages WHERE session_id = %s ORDER BY seq",
                (canonical,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, session_id, seq, role, content, call_id, tool_name,"
                " correlation_id, created_at"
                " FROM messages WHERE session_id = %s ORDER BY seq DESC LIMIT %s",
                (canonical, limit),
            ).fetchall()
            rows = list(reversed(rows))
    return [message_from_row(row) for row in rows]


def count_messages(db: Database, session_id: str) -> int:
    """Return how many messages a session holds."""
    canonical = _canonical_session_id(session_id)
    if canonical is None:
        return 0
    with db.connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM messages WHERE session_id = %s",
            (canonical,),
        ).fetchone()
    return int(row["count"]) if row is not None else 0


def append_memories(
    db: Database,
    session_id: str,
    drafts: Sequence[MemoryDraft],
) -> list[Memory]:
    """Store memory drafts for a session and return the stored rows.

    The memories belong to exactly one session, and the session has to
    exist. The kind of a draft is not checked here: the database constraint
    rejects a kind that is not one of the supported ones.

    The content of a draft is truncated to
    :data:`MEMORY_CONTENT_MAX_CHARS` characters, so that a memory stays a
    short, distilled fact.
    """
    if not drafts:
        return []
    canonical = _canonical_session_id(session_id)
    if canonical is None:
        raise UnknownSessionError(session_id)

    now = utc_now()
    stored: list[Memory] = []
    with db.write_transaction() as conn:
        session = conn.execute(
            "SELECT id FROM sessions WHERE id = %s",
            (canonical,),
        ).fetchone()
        if session is None:
            raise UnknownSessionError(session_id)

        for draft in drafts:
            content = (draft.content or "")[:MEMORY_CONTENT_MAX_CHARS]
            row = conn.execute(
                "INSERT INTO memories"
                " (session_id, kind, content, source_msg_id, created_at)"
                " VALUES (%s, %s, %s, %s, %s)"
                " RETURNING id, session_id, kind, content, source_msg_id,"
                " created_at",
                (
                    canonical,
                    draft.kind,
                    content,
                    draft.source_msg_id,
                    now,
                ),
            ).fetchone()
            stored.append(memory_from_row(row))

    return stored


def search_memories(
    db: Database,
    session_id: str,
    query: str,
    *,
    limit: int = MEMORY_SEARCH_RESULT_LIMIT,
    content_limit: int = MEMORY_CONTENT_MAX_CHARS,
) -> list[Memory]:
    """Return the memories of a session that match a query.

    Only the memories of the given session are searched: the session id is
    part of the query, so a session never sees the memories of another one.

    The search uses the full-text search vector that the database keeps for
    every memory. The query is matched with the English text search rules:
    the terms are stemmed, and a memory matches when it holds any of them.
    The results are ordered by relevance first and by recency second, so
    that the order is the same for the same query every time.

    The content of a result is truncated to ``content_limit`` characters,
    because the results are meant to be read by a model.
    """
    canonical = _canonical_session_id(session_id)
    if canonical is None:
        return []
    if limit <= 0:
        return []
    text = (query or "").strip()
    if not text:
        return []

    with db.connection() as conn:
        rows = conn.execute(
            "SELECT id, session_id, kind, left(content, %s) AS content,"
            " source_msg_id, created_at"
            " FROM memories"
            " WHERE session_id = %s"
            " AND search_vector @@ websearch_to_tsquery('english', %s)"
            " ORDER BY ts_rank("
            "   search_vector, websearch_to_tsquery('english', %s)"
            " ) DESC, created_at DESC, id DESC"
            " LIMIT %s",
            (content_limit, canonical, text, text, limit),
        ).fetchall()
    return [memory_from_row(row) for row in rows]
