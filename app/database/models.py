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

#: The state of an action that waits for a decision.
ACTION_STATE_PENDING = "pending"

#: The state of an action that was approved.
ACTION_STATE_APPROVED = "approved"

#: The state of an action that was rejected.
ACTION_STATE_REJECTED = "rejected"

#: The state of an action that was executed.
ACTION_STATE_EXECUTED = "executed"

#: Every state an action may have. The database enforces the same set.
ACTION_STATES = (
    ACTION_STATE_PENDING,
    ACTION_STATE_APPROVED,
    ACTION_STATE_REJECTED,
    ACTION_STATE_EXECUTED,
)

#: The states a decision can move an action to.
ACTION_DECISION_STATES = (ACTION_STATE_APPROVED, ACTION_STATE_REJECTED)

#: The outcome of an execution that succeeded.
EXECUTION_STATUS_SUCCESS = "success"

#: The outcome of an execution that failed.
EXECUTION_STATUS_FAILURE = "failure"

#: Every execution status the database accepts.
EXECUTION_STATUSES = (EXECUTION_STATUS_SUCCESS, EXECUTION_STATUS_FAILURE)

#: The roles an approver can have. The database enforces the same set.
APPROVER_ROLE_AGENT = "agent"
APPROVER_ROLE_SUPERVISOR = "supervisor"
APPROVER_ROLE_BILLING_ADMIN = "billing_admin"
APPROVER_ROLE_ADMIN = "admin"

APPROVER_ROLES = (
    APPROVER_ROLE_AGENT,
    APPROVER_ROLE_SUPERVISOR,
    APPROVER_ROLE_BILLING_ADMIN,
    APPROVER_ROLE_ADMIN,
)

#: The kinds of actor an audit event can name. The database enforces the
#: same set.
AUDIT_ACTOR_TYPE_AGENT = "agent"
AUDIT_ACTOR_TYPE_CUSTOMER = "customer"
AUDIT_ACTOR_TYPE_APPROVER = "approver"
AUDIT_ACTOR_TYPE_SYSTEM = "system"

AUDIT_ACTOR_TYPES = (
    AUDIT_ACTOR_TYPE_AGENT,
    AUDIT_ACTOR_TYPE_CUSTOMER,
    AUDIT_ACTOR_TYPE_APPROVER,
    AUDIT_ACTOR_TYPE_SYSTEM,
)

#: The decisions an audit event can record.
AUDIT_DECISION_APPROVED = "approved"
AUDIT_DECISION_REJECTED = "rejected"

AUDIT_DECISIONS = (AUDIT_DECISION_APPROVED, AUDIT_DECISION_REJECTED)

#: The outcomes an audit event can record.
AUDIT_OUTCOME_SUCCESS = "success"
AUDIT_OUTCOME_FAILURE = "failure"
AUDIT_OUTCOME_BLOCKED = "blocked"
AUDIT_OUTCOME_DENIED = "denied"

AUDIT_OUTCOMES = (
    AUDIT_OUTCOME_SUCCESS,
    AUDIT_OUTCOME_FAILURE,
    AUDIT_OUTCOME_BLOCKED,
    AUDIT_OUTCOME_DENIED,
)


@dataclass(frozen=True)
class Tenant:
    """A tenant."""

    id: str
    name: str
    enabled: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Integration:
    """A registered product integration."""

    id: str
    name: str
    enabled: bool
    created_at: str
    tenant_id: str | None = None


@dataclass(frozen=True)
class IntegrationCredential:
    """A credential used by an integration (hash only)."""

    id: str
    integration_id: str
    token_hash: str
    key_id: str
    enabled: bool
    expires_at: str | None
    revoked_at: str | None
    created_at: str
    last_used_at: str | None


@dataclass(frozen=True)
class Session:
    """A conversation session."""

    id: str
    created_at: str
    updated_at: str
    status: str
    ticket_id: str | None
    title: str | None
    tenant_id: str | None = None


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


@dataclass(frozen=True)
class Action:
    """A gated action that waits for a decision.

    An action records what the agent wanted to do, which tool it wanted to
    use, and the state of the request. The state moves from ``pending`` to
    ``approved`` or ``rejected``, and from ``approved`` to ``executed``.
    """

    id: str
    request_key: str
    session_id: str | None
    correlation_id: str
    tool_name: str
    effect_class: str
    args: str
    args_hash: str
    state: str
    requested_by: str
    requested_at: str
    decided_by: str | None
    decided_at: str | None
    decision_reason: str | None
    execution_status: str | None
    execution_error: str | None
    executed_by: str | None
    executed_at: str | None
    tenant_id: str | None = None


@dataclass(frozen=True)
class ActionTransition:
    """The outcome of a guarded state transition.

    ``changed`` tells whether the transition was applied. When it was not,
    ``action`` still holds the action when it exists, so that a caller can
    tell an action that is in the wrong state from an action that does not
    exist at all.
    """

    action: Action | None
    changed: bool


@dataclass(frozen=True)
class Approver:
    """An identity that may approve actions.

    The stored credential of an approver is deliberately not exposed here:
    nothing outside the persistence layer needs it.
    """

    id: str
    display_name: str
    role: str
    active: bool
    created_at: str


@dataclass(frozen=True)
class AuditDraft:
    """An audit event that has not been written yet."""

    actor_type: str
    action: str
    correlation_id: str
    action_id: str | None = None
    session_id: str | None = None
    ticket_id: str | None = None
    actor_id: str | None = None
    target: str | None = None
    decision: str | None = None
    outcome: str | None = None
    payload: str | None = None


@dataclass(frozen=True)
class AuditEvent:
    """A stored audit event.

    Audit events are append-only: the database refuses to change or remove
    one, so the history of a decision cannot be rewritten.
    """

    id: int
    event_id: str
    correlation_id: str
    session_id: str | None
    ticket_id: str | None
    actor_type: str
    actor_id: str | None
    action: str
    target: str | None
    decision: str | None
    outcome: str | None
    payload: str | None
    action_id: str | None
    created_at: str


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
    tenant_id = row.get("tenant_id")
    return Session(
        id=str(row["id"]),
        created_at=to_iso8601(row["created_at"]),
        updated_at=to_iso8601(row["updated_at"]),
        status=row["status"],
        ticket_id=row["ticket_id"],
        title=row["title"],
        tenant_id=str(tenant_id) if tenant_id is not None else None,
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


def _optional_iso8601(value: Any) -> str | None:
    """Render a nullable timestamp as an ISO-8601 string, or ``None``."""
    if value is None:
        return None
    return to_iso8601(value)


def action_from_row(row: Mapping[str, Any]) -> Action:
    """Build an :class:`Action` from a database row."""
    session_id = row["session_id"]
    tenant_id = row.get("tenant_id")
    return Action(
        id=str(row["id"]),
        request_key=row["request_key"],
        session_id=str(session_id) if session_id is not None else None,
        correlation_id=str(row["correlation_id"]),
        tool_name=row["tool_name"],
        effect_class=row["effect_class"],
        args=row["args"],
        args_hash=row["args_hash"],
        state=row["state"],
        requested_by=row["requested_by"],
        requested_at=to_iso8601(row["requested_at"]),
        decided_by=row["decided_by"],
        decided_at=_optional_iso8601(row["decided_at"]),
        decision_reason=row["decision_reason"],
        execution_status=row["execution_status"],
        execution_error=row["execution_error"],
        executed_by=row["executed_by"],
        executed_at=_optional_iso8601(row["executed_at"]),
        tenant_id=str(tenant_id) if tenant_id is not None else None,
    )


def approver_from_row(row: Mapping[str, Any]) -> Approver:
    """Build an :class:`Approver` from a database row.

    The stored credential is not read: it is not part of the record.
    """
    return Approver(
        id=str(row["id"]),
        display_name=row["display_name"],
        role=row["role"],
        active=bool(row["active"]),
        created_at=to_iso8601(row["created_at"]),
    )


def integration_from_row(row: Mapping[str, Any]) -> Integration:
    tenant_id = row.get("tenant_id")
    return Integration(
        id=str(row["id"]),
        name=row["name"],
        enabled=bool(row["enabled"]),
        created_at=to_iso8601(row["created_at"]),
        tenant_id=str(tenant_id) if tenant_id is not None else None,
    )


def integration_credential_from_row(row: Mapping[str, Any]) -> IntegrationCredential:
    expires_at = row.get("expires_at")
    revoked_at = row.get("revoked_at")
    last_used_at = row.get("last_used_at")
    return IntegrationCredential(
        id=str(row["id"]),
        integration_id=str(row["integration_id"]),
        token_hash=row["token_hash"],
        key_id=row["key_id"],
        enabled=bool(row["enabled"]),
        expires_at=to_iso8601(expires_at) if expires_at is not None else None,
        revoked_at=to_iso8601(revoked_at) if revoked_at is not None else None,
        created_at=to_iso8601(row["created_at"]),
        last_used_at=to_iso8601(last_used_at) if last_used_at is not None else None,
    )


def audit_event_from_row(row: Mapping[str, Any]) -> AuditEvent:
    """Build an :class:`AuditEvent` from a database row."""
    session_id = row["session_id"]
    action_id = row["action_id"]
    return AuditEvent(
        id=int(row["id"]),
        event_id=str(row["event_id"]),
        correlation_id=str(row["correlation_id"]),
        session_id=str(session_id) if session_id is not None else None,
        ticket_id=row["ticket_id"],
        actor_type=row["actor_type"],
        actor_id=row["actor_id"],
        action=row["action"],
        target=row["target"],
        decision=row["decision"],
        outcome=row["outcome"],
        payload=row["payload"],
        action_id=str(action_id) if action_id is not None else None,
        created_at=to_iso8601(row["created_at"]),
    )
