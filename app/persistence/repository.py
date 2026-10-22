"""Data access for sessions, messages, and memories.

This layer knows about storage only. It has no knowledge of the agent, the
provider, or the HTTP layer, so it can be tested on its own.

Every statement is written for PostgreSQL: ``%s`` placeholders, generated
identifiers read back with ``RETURNING``, and transcript sequence numbers
allocated from the ``sessions.next_seq`` counter so that two concurrent
appends to the same session can never reserve the same number.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import uuid
from datetime import datetime, timezone
from typing import Sequence

from app.persistence.db import Database
from app.persistence.models import (
    ACTION_DECISION_STATES,
    ACTION_STATE_APPROVED,
    ACTION_STATE_EXECUTED,
    ACTION_STATE_PENDING,
    APPROVER_ROLES,
    EXECUTION_STATUSES,
    SESSION_STATUS_CLOSED,
    SESSION_STATUS_OPEN,
    Action,
    ActionTransition,
    Approver,
    AuditDraft,
    AuditEvent,
    Memory,
    MemoryDraft,
    Message,
    MessageDraft,
    Session,
    action_from_row,
    approver_from_row,
    audit_event_from_row,
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

#: Default number of actions a listing returns.
ACTION_LIST_LIMIT = 50

#: Default number of approvers a listing returns.
APPROVER_LIST_LIMIT = 50

#: The words of a search query. The query text is split into these before it
#: is turned into a tsquery.
_SEARCH_TERM = re.compile(r"[0-9A-Za-z]+")


def _canonical_session_id(session_id: str) -> str | None:
    """Return the canonical form of a session id, or ``None`` when invalid.

    PostgreSQL stores the session id in a ``UUID`` column, so a value that
    is not a well-formed identifier can never match a row. Returning ``None``
    lets the read helpers answer "no such session" instead of raising a
    database error for a value the caller already rejected.
    """
    return _canonical_uuid(session_id)


def _canonical_uuid(value: str) -> str | None:
    """Return the canonical form of a UUID, or ``None`` when invalid.

    The identifiers of the rows of this layer are UUIDs, so a value that is
    not a well-formed identifier can never match a row. Returning ``None``
    lets the read helpers answer "no such row" instead of raising a database
    error for a value the caller already rejected.
    """
    try:
        return str(uuid.UUID(str(value)))
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


def find_memory_by_content(
    db: Database,
    session_id: str,
    content: str,
) -> Memory | None:
    """Return the memory of a session that holds exactly this content.

    The session id and the content have to match the stored row character
    for character. A full-text search is not used for this: it matches on
    words, and would report a memory that only shares a term with the
    content.

    The oldest matching memory is returned, so that the result does not
    depend on the order the rows were written in.
    """
    canonical = _canonical_session_id(session_id)
    if canonical is None:
        return None
    text = content or ""
    if not text:
        return None

    with db.connection() as conn:
        row = conn.execute(
            "SELECT id, session_id, kind, content, source_msg_id, created_at"
            " FROM memories WHERE session_id = %s AND content = %s"
            " ORDER BY id LIMIT 1",
            (canonical, text),
        ).fetchone()

    if row is None:
        return None
    return memory_from_row(row)


def _search_query(text: str) -> str:
    """Return a tsquery that matches a memory holding any of the terms.

    The terms are quoted, so that a word of the message that PostgreSQL
    treats as an operator cannot change the meaning of the query. Stop words
    are dropped by PostgreSQL itself.
    """
    terms = _SEARCH_TERM.findall(text)
    if not terms:
        return ""
    return " | ".join(f"'{term}'" for term in terms)


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
    tsquery = _search_query(text)
    if not tsquery:
        return []

    with db.connection() as conn:
        rows = conn.execute(
            "SELECT id, session_id, kind, left(content, %s) AS content,"
            " source_msg_id, created_at"
            " FROM memories"
            " WHERE session_id = %s"
            " AND search_vector @@ to_tsquery('english', %s)"
            " ORDER BY ts_rank("
            "   search_vector, to_tsquery('english', %s)"
            " ) DESC, created_at DESC, id DESC"
            " LIMIT %s",
            (content_limit, canonical, tsquery, tsquery, limit),
        ).fetchall()
    return [memory_from_row(row) for row in rows]


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
#
# An action is a request for a consequential operation. It starts out
# pending, is moved to approved or rejected by a decision, and an approved
# action is moved to executed when it was carried out. Every move is a
# conditional update: the state of the row is part of the condition, so a
# transition that was already made is never applied a second time.
# ---------------------------------------------------------------------------


def create_action(
    db: Database,
    *,
    request_key: str,
    tool_name: str,
    effect_class: str,
    args: str,
    requested_by: str,
    correlation_id: str,
    args_hash: str | None = None,
    session_id: str | None = None,
    action_id: str | None = None,
) -> Action:
    """Create an action and return it.

    The request key identifies the request. The same request that is sent
    twice does not create a second action: the action of the first request
    is returned instead, so the caller can continue with it and a retry
    cannot queue the same operation twice.

    The hash of the arguments is computed from the arguments when it is not
    supplied, so that two requests with the same arguments can be told
    apart from two requests with the same key but different arguments.
    """
    canonical = _canonical_session_id(session_id) if session_id else None
    if session_id and canonical is None:
        raise UnknownSessionError(session_id)

    run_id = _canonical_uuid(correlation_id)
    if run_id is None:
        raise ValueError("The correlation id must be a UUID.")

    digest = args_hash or hashlib.sha256(args.encode("utf-8")).hexdigest()
    now = utc_now()

    with db.write_transaction() as conn:
        if canonical is not None:
            session = conn.execute(
                "SELECT id FROM sessions WHERE id = %s", (canonical,)
            ).fetchone()
            if session is None:
                raise UnknownSessionError(session_id)

        row = conn.execute(
            "INSERT INTO actions"
            " (id, request_key, session_id, correlation_id, tool_name,"
            "  effect_class, args, args_hash, state, requested_by, requested_at)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (request_key) DO UPDATE"
            " SET request_key = actions.request_key"
            " RETURNING id, request_key, session_id, correlation_id, tool_name,"
            " effect_class, args, args_hash, state, requested_by, requested_at,"
            " decided_by, decided_at, decision_reason, execution_status,"
            " execution_error, executed_by, executed_at",
            (
                action_id or str(uuid.uuid4()),
                request_key,
                canonical,
                run_id,
                tool_name,
                effect_class,
                args,
                digest,
                ACTION_STATE_PENDING,
                requested_by,
                now,
            ),
        ).fetchone()

    if row is None:
        raise RuntimeError("The action was not stored.")
    return action_from_row(row)


def get_action(db: Database, action_id: str) -> Action | None:
    """Return the action with this id, or ``None`` when it does not exist."""
    canonical = _canonical_uuid(action_id)
    if canonical is None:
        return None
    with db.connection() as conn:
        row = conn.execute(
            "SELECT id, request_key, session_id, correlation_id, tool_name,"
            " effect_class, args, args_hash, state, requested_by, requested_at,"
            " decided_by, decided_at, decision_reason, execution_status,"
            " execution_error, executed_by, executed_at"
            " FROM actions WHERE id = %s",
            (canonical,),
        ).fetchone()
    if row is None:
        return None
    return action_from_row(row)


def list_actions(
    db: Database,
    *,
    session_id: str | None = None,
    state: str | None = None,
    correlation_id: str | None = None,
    limit: int = ACTION_LIST_LIMIT,
) -> list[Action]:
    """Return the actions that match the filters, oldest first.

    Every filter is optional. A filter that cannot match anything, because
    its value is not a well-formed identifier, returns an empty list rather
    than every action.

    The result is bounded by ``limit``, and the limit is applied by the
    database.
    """
    if limit <= 0:
        return []

    clauses: list[str] = []
    parameters: list[object] = []

    if session_id is not None:
        canonical = _canonical_session_id(session_id)
        if canonical is None:
            return []
        clauses.append("session_id = %s")
        parameters.append(canonical)

    if state is not None:
        clauses.append("state = %s")
        parameters.append(state)

    if correlation_id is not None:
        run_id = _canonical_uuid(correlation_id)
        if run_id is None:
            return []
        clauses.append("correlation_id = %s")
        parameters.append(run_id)

    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    parameters.append(limit)

    with db.connection() as conn:
        rows = conn.execute(
            "SELECT id, request_key, session_id, correlation_id, tool_name,"
            " effect_class, args, args_hash, state, requested_by, requested_at,"
            " decided_by, decided_at, decision_reason, execution_status,"
            " execution_error, executed_by, executed_at"
            " FROM actions" + where + " ORDER BY requested_at, id LIMIT %s",
            tuple(parameters),
        ).fetchall()
    return [action_from_row(row) for row in rows]


def _decide_action_row(
    conn,
    canonical: str,
    decision: str,
    decided_by: str,
    reason: str | None,
    now: datetime,
) -> tuple[Action | None, bool]:
    """Move a pending action to a decided state on an open connection.

    Returns the action and whether the transition was applied.
    """
    row = conn.execute(
        "UPDATE actions SET state = %s, decided_by = %s, decided_at = %s,"
        " decision_reason = %s WHERE id = %s AND state = %s"
        " RETURNING id, request_key, session_id, correlation_id, tool_name,"
        " effect_class, args, args_hash, state, requested_by, requested_at,"
        " decided_by, decided_at, decision_reason, execution_status,"
        " execution_error, executed_by, executed_at",
        (
            decision,
            decided_by,
            now,
            reason,
            canonical,
            ACTION_STATE_PENDING,
        ),
    ).fetchone()
    if row is not None:
        return action_from_row(row), True

    current = conn.execute(
        "SELECT id, request_key, session_id, correlation_id, tool_name,"
        " effect_class, args, args_hash, state, requested_by, requested_at,"
        " decided_by, decided_at, decision_reason, execution_status,"
        " execution_error, executed_by, executed_at"
        " FROM actions WHERE id = %s",
        (canonical,),
    ).fetchone()
    if current is None:
        return None, False
    return action_from_row(current), False


def decide_action(
    db: Database,
    action_id: str,
    *,
    decision: str,
    decided_by: str,
    reason: str | None = None,
) -> ActionTransition:
    """Decide a pending action and return the outcome.

    Only an action that is still pending can be decided. The update is
    conditional, so a second decision cannot overwrite the first one, and a
    decision that arrives after a rejection does not revive the action.

    The result tells the two failure modes apart: an action that does not
    exist has no action in it, and an action that is in the wrong state is
    returned with ``changed`` set to ``False``.
    """
    if decision not in ACTION_DECISION_STATES:
        raise ValueError(f"Unsupported action decision: {decision!r}")

    canonical = _canonical_uuid(action_id)
    if canonical is None:
        return ActionTransition(action=None, changed=False)

    with db.write_transaction() as conn:
        action, changed = _decide_action_row(
            conn, canonical, decision, decided_by, reason, utc_now()
        )
    return ActionTransition(action=action, changed=changed)


def _mark_action_executed_row(
    conn,
    canonical: str,
    execution_status: str,
    executed_by: str,
    execution_error: str | None,
    now: datetime,
) -> tuple[Action | None, bool]:
    """Move an approved action to the executed state on an open connection.

    Returns the action and whether the transition was applied.
    """
    row = conn.execute(
        "UPDATE actions SET state = %s, execution_status = %s,"
        " execution_error = %s, executed_by = %s, executed_at = %s"
        " WHERE id = %s AND state = %s"
        " RETURNING id, request_key, session_id, correlation_id, tool_name,"
        " effect_class, args, args_hash, state, requested_by, requested_at,"
        " decided_by, decided_at, decision_reason, execution_status,"
        " execution_error, executed_by, executed_at",
        (
            ACTION_STATE_EXECUTED,
            execution_status,
            execution_error,
            executed_by,
            now,
            canonical,
            ACTION_STATE_APPROVED,
        ),
    ).fetchone()
    if row is not None:
        return action_from_row(row), True

    current = conn.execute(
        "SELECT id, request_key, session_id, correlation_id, tool_name,"
        " effect_class, args, args_hash, state, requested_by, requested_at,"
        " decided_by, decided_at, decision_reason, execution_status,"
        " execution_error, executed_by, executed_at"
        " FROM actions WHERE id = %s",
        (canonical,),
    ).fetchone()
    if current is None:
        return None, False
    return action_from_row(current), False


def mark_action_executed(
    db: Database,
    action_id: str,
    *,
    execution_status: str,
    executed_by: str,
    execution_error: str | None = None,
) -> ActionTransition:
    """Mark an approved action as executed and return the outcome.

    Only an action that was approved can be executed, and only once: the
    update is conditional, so an action that was executed already keeps its
    state and its first result.

    This records that an execution happened. It does not perform one: the
    effect itself is outside of this layer.
    """
    if execution_status not in EXECUTION_STATUSES:
        raise ValueError(
            f"Unsupported execution status: {execution_status!r}"
        )

    canonical = _canonical_uuid(action_id)
    if canonical is None:
        return ActionTransition(action=None, changed=False)

    with db.write_transaction() as conn:
        action, changed = _mark_action_executed_row(
            conn,
            canonical,
            execution_status,
            executed_by,
            execution_error,
            utc_now(),
        )
    return ActionTransition(action=action, changed=changed)


# ---------------------------------------------------------------------------
# Audit events
# ---------------------------------------------------------------------------
#
# An audit event records what happened. The table is append-only, so an
# event cannot be changed or removed afterwards, and the trigger that
# enforces this is part of the schema.
# ---------------------------------------------------------------------------


def _insert_audit_event(
    conn,
    draft: AuditDraft,
    *,
    action_id: str | None = None,
    decision: str | None = None,
    outcome: str | None = None,
    actor_id: str | None = None,
) -> AuditEvent:
    """Insert an audit event on an open connection and return it.

    The values that the caller knows better than the draft itself - the
    action the event belongs to, the decision that was made, the outcome
    that was reached, and the actor that acted - can be passed in, and take
    the place of the ones of the draft.
    """
    resolved_action = action_id if action_id is not None else draft.action_id
    resolved_decision = decision if decision is not None else draft.decision
    resolved_outcome = outcome if outcome is not None else draft.outcome
    resolved_actor = actor_id if actor_id is not None else draft.actor_id

    row = conn.execute(
        "INSERT INTO audit_events"
        " (event_id, correlation_id, session_id, ticket_id, actor_type,"
        "  actor_id, action, target, decision, outcome, payload, action_id,"
        "  created_at)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        " RETURNING id, event_id, correlation_id, session_id, ticket_id,"
        " actor_type, actor_id, action, target, decision, outcome, payload,"
        " action_id, created_at",
        (
            str(uuid.uuid4()),
            draft.correlation_id,
            draft.session_id,
            draft.ticket_id,
            draft.actor_type,
            resolved_actor,
            draft.action,
            draft.target,
            resolved_decision,
            resolved_outcome,
            draft.payload,
            resolved_action,
            utc_now(),
        ),
    ).fetchone()
    if row is None:
        raise RuntimeError("The audit event was not stored.")
    return audit_event_from_row(row)


def append_audit_event(db: Database, draft: AuditDraft) -> AuditEvent:
    """Store an audit event and return it.

    The event is written in its own transaction, and it cannot be changed or
    removed later: the table refuses to do that.
    """
    with db.write_transaction() as conn:
        return _insert_audit_event(conn, draft)


def decide_action_with_audit(
    db: Database,
    action_id: str,
    *,
    decision: str,
    decided_by: str,
    draft: AuditDraft,
    reason: str | None = None,
) -> ActionTransition:
    """Decide an action and record the decision in one transaction.

    The decision and the audit event are committed together: when the event
    cannot be written, the decision is rolled back with it, so an action
    never carries a decision that was not recorded.

    The event is only written when the decision was applied. A decision
    that was not applied leaves the action as it was and writes no event.
    """
    if decision not in ACTION_DECISION_STATES:
        raise ValueError(f"Unsupported action decision: {decision!r}")

    canonical = _canonical_uuid(action_id)
    if canonical is None:
        return ActionTransition(action=None, changed=False)

    with db.write_transaction() as conn:
        action, changed = _decide_action_row(
            conn, canonical, decision, decided_by, reason, utc_now()
        )
        if not changed:
            return ActionTransition(action=action, changed=False)
        _insert_audit_event(
            conn,
            draft,
            action_id=canonical,
            decision=decision,
            actor_id=decided_by,
        )
    return ActionTransition(action=action, changed=True)


def mark_action_executed_with_audit(
    db: Database,
    action_id: str,
    *,
    execution_status: str,
    executed_by: str,
    draft: AuditDraft,
    execution_error: str | None = None,
) -> ActionTransition:
    """Mark an action as executed and record the outcome in one transaction.

    The execution and the audit event are committed together: when the event
    cannot be written, the execution is rolled back with it, so an action
    never carries a result that was not recorded.

    The event is only written when the execution was applied.
    """
    if execution_status not in EXECUTION_STATUSES:
        raise ValueError(
            f"Unsupported execution status: {execution_status!r}"
        )

    canonical = _canonical_uuid(action_id)
    if canonical is None:
        return ActionTransition(action=None, changed=False)

    with db.write_transaction() as conn:
        action, changed = _mark_action_executed_row(
            conn,
            canonical,
            execution_status,
            executed_by,
            execution_error,
            utc_now(),
        )
        if not changed:
            return ActionTransition(action=action, changed=False)
        _insert_audit_event(
            conn,
            draft,
            action_id=canonical,
            actor_id=executed_by,
            outcome=execution_status,
        )
    return ActionTransition(action=action, changed=True)


# ---------------------------------------------------------------------------
# Approvers
# ---------------------------------------------------------------------------
#
# An approver is an identity that may decide actions. The credential of an
# approver is never stored: the column holds the digest of it, and the
# digest is the identifier of the credential. The digest is unique, so a
# credential belongs to exactly one approver.
# ---------------------------------------------------------------------------


def create_approver(
    db: Database,
    *,
    display_name: str,
    role: str,
    token_hash: str,
    approver_id: str | None = None,
) -> Approver:
    """Create an approver and return it.

    The argument is the digest of the credential, never the credential
    itself. The digest has to be the one of a credential that is not in use
    yet: the column is unique, and a second approver with the same digest
    is refused by the database.
    """
    if role not in APPROVER_ROLES:
        raise ValueError(f"Unsupported approver role: {role!r}")
    if not token_hash:
        raise ValueError("The credential digest must not be empty.")

    with db.write_transaction() as conn:
        row = conn.execute(
            "INSERT INTO approvers"
            " (id, display_name, role, token_hash, active, created_at)"
            " VALUES (%s, %s, %s, %s, %s, %s)"
            " RETURNING id, display_name, role, active, created_at",
            (
                approver_id or str(uuid.uuid4()),
                display_name,
                role,
                token_hash,
                True,
                utc_now(),
            ),
        ).fetchone()

    if row is None:
        raise RuntimeError("The approver was not stored.")
    return approver_from_row(row)


def get_approver(db: Database, approver_id: str) -> Approver | None:
    """Return the approver with this id, or ``None`` when it does not exist."""
    canonical = _canonical_uuid(approver_id)
    if canonical is None:
        return None
    with db.connection() as conn:
        row = conn.execute(
            "SELECT id, display_name, role, active, created_at"
            " FROM approvers WHERE id = %s",
            (canonical,),
        ).fetchone()
    if row is None:
        return None
    return approver_from_row(row)


def get_approver_by_token_hash(
    db: Database,
    token_hash: str,
) -> Approver | None:
    """Return the approver that holds this credential digest.

    The digest is the identifier of a credential, and the column is unique,
    so at most one approver holds it. The stored digest is compared with
    the one that was given in constant time, and it is never returned.
    """
    if not token_hash:
        return None
    if not token_hash.isascii():
        return None
    with db.connection() as conn:
        row = conn.execute(
            "SELECT id, display_name, role, active, created_at, token_hash"
            " FROM approvers WHERE token_hash = %s",
            (token_hash,),
        ).fetchone()
    if row is None:
        return None
    if not hmac.compare_digest(str(row["token_hash"]), token_hash):
        return None
    return approver_from_row(row)


def list_approvers(
    db: Database,
    *,
    role: str | None = None,
    active: bool | None = None,
    limit: int = APPROVER_LIST_LIMIT,
) -> list[Approver]:
    """Return the approvers that match the filters, oldest first.

    Every filter is optional. The result is bounded by ``limit``, and the
    limit is applied by the database.
    """
    if limit <= 0:
        return []

    clauses: list[str] = []
    parameters: list[object] = []

    if role is not None:
        clauses.append("role = %s")
        parameters.append(role)

    if active is not None:
        clauses.append("active = %s")
        parameters.append(active)

    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    parameters.append(limit)

    with db.connection() as conn:
        rows = conn.execute(
            "SELECT id, display_name, role, active, created_at"
            " FROM approvers" + where + " ORDER BY created_at, id LIMIT %s",
            tuple(parameters),
        ).fetchall()
    return [approver_from_row(row) for row in rows]


def set_approver_active(
    db: Database,
    approver_id: str,
    active: bool,
) -> Approver | None:
    """Activate or deactivate an approver and return the stored row.

    Returns ``None`` when the approver does not exist. A deactivated
    approver keeps its credential, but the credential no longer grants
    anything.
    """
    canonical = _canonical_uuid(approver_id)
    if canonical is None:
        return None

    with db.write_transaction() as conn:
        row = conn.execute(
            "UPDATE approvers SET active = %s WHERE id = %s"
            " RETURNING id, display_name, role, active, created_at",
            (active, canonical),
        ).fetchone()

    if row is None:
        return None
    return approver_from_row(row)
