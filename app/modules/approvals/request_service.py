"""Application service for turning a gated tool call into a pending approval.

This module only persists the request. It never invokes the underlying tool.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from app.agent_core.tools import GATED_TOOLS, TOOL_REGISTRY
from app.database.db import Database
from app.database.models import (
    ACTION_STATE_PENDING,
    AuditDraft,
    Action,
    AuditEvent,
)
from app.database.repository import (
    ActionConflictError,
    create_action_with_audit,
    get_session,
)


class ApprovalRequestError(RuntimeError):
    """Raised when an approval request cannot be created safely."""


class ApprovalRequestConflictError(ApprovalRequestError):
    """Raised when an idempotency key is reused for a different request."""


_REQUEST_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_MAX_REQUEST_KEY_LENGTH = 256


@dataclass(frozen=True)
class ApprovalRequestResult:
    """The outcome of requesting approval for a gated tool call."""

    action: Action
    audit_event: AuditEvent | None
    created: bool


def _validate_request_key(request_key: object) -> str:
    if not isinstance(request_key, str):
        raise ApprovalRequestError("The request key must be a string.")
    if not _REQUEST_KEY_RE.fullmatch(request_key):
        raise ApprovalRequestError("The request key is missing or malformed.")
    if len(request_key) > _MAX_REQUEST_KEY_LENGTH:
        raise ApprovalRequestError("The request key is too long.")
    return request_key


def _validate_correlation_id(correlation_id: object) -> str:
    if not isinstance(correlation_id, str):
        raise ApprovalRequestError("The correlation id must be a UUID.")
    try:
        return str(uuid.UUID(correlation_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ApprovalRequestError(
            "The correlation id must be a UUID."
        ) from exc


def request_approval(
    db: Database,
    *,
    tool_name: str,
    arguments: Mapping[str, Any] | None,
    request_key: str,
    correlation_id: str,
    session_id: str | None = None,
    requested_by: str = "agent",
    effect_class: str = "gated_tool",
) -> ApprovalRequestResult:
    """Persist a pending action for a gated tool without invoking it."""
    if tool_name not in TOOL_REGISTRY:
        raise ApprovalRequestError(f"Tool '{tool_name}' is not registered.")
    if tool_name not in GATED_TOOLS:
        raise ApprovalRequestError(f"Tool '{tool_name}' is not gated.")

    args_schema, _ = TOOL_REGISTRY[tool_name]
    raw_arguments = dict(arguments or {})
    try:
        validated = args_schema(**raw_arguments)
    except Exception as exc:
        raise ApprovalRequestError(
            f"Invalid arguments for tool '{tool_name}'."
        ) from exc

    canonical_arguments = json.dumps(
        validated.model_dump(),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    args_hash = hashlib.sha256(canonical_arguments.encode("utf-8")).hexdigest()

    validated_request_key = _validate_request_key(request_key)
    validated_correlation_id = _validate_correlation_id(correlation_id)

    if session_id is not None:
        if get_session(db, session_id) is None:
            raise ApprovalRequestError("The session does not exist.")

    audit_draft = AuditDraft(
        actor_type="agent",
        action="approval.requested",
        correlation_id=validated_correlation_id,
        session_id=session_id,
        target=tool_name,
        outcome="blocked",
        payload=json.dumps(
            {"request_key": validated_request_key, "tool_name": tool_name},
            sort_keys=True,
            separators=(",", ":"),
        ),
    )

    try:
        action, audit_event, created = create_action_with_audit(
            db,
            request_key=validated_request_key,
            tool_name=tool_name,
            effect_class=effect_class,
            args=canonical_arguments,
            args_hash=args_hash,
            requested_by=requested_by,
            correlation_id=validated_correlation_id,
            session_id=session_id,
            audit_draft=audit_draft,
        )
    except ActionConflictError as exc:
        raise ApprovalRequestConflictError(str(exc)) from exc

    if not created:
        audit_event = None

    return ApprovalRequestResult(
        action=action,
        audit_event=audit_event,
        created=created,
    )
