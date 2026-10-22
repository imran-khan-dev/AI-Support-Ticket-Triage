"""Application service for executing an approved action.

The service uses the persisted action only. It never accepts replacement
arguments or caller-supplied tool implementations.

Crash recovery for abandoned claims is not implemented: an action whose
claim is left set after a process crash may require manual intervention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.agent_core.tools import GATED_TOOLS, TOOL_REGISTRY
from app.modules.approvals.policy import CAPABILITY_DECIDE_ACTION, is_authorized
from app.modules.approvals.service import authenticate_approver
from app.persistence.db import Database
from app.persistence.models import (
    ACTION_STATE_APPROVED,
    ACTION_STATE_EXECUTED,
    AuditDraft,
    Action,
    AuditEvent,
)
from app.persistence.repository import (
    ActionConflictError,
    claim_action_for_execution,
    get_action,
    mark_action_executed_with_audit,
    release_action_execution_claim,
)


class ActionExecutionError(RuntimeError):
    """Raised when an action cannot be executed safely."""


class ActionExecutionUncertainError(ActionExecutionError):
    """Raised when a side effect may have occurred but could not be recorded."""


@dataclass(frozen=True)
class ActionExecutionResult:
    """The outcome of executing an approved action."""

    action: Action
    result: Any
    executed: bool
    audit_event: AuditEvent | None


def execute_approved_action(
    db: Database,
    *,
    action_id: str,
    executed_by: str = "system",
) -> ActionExecutionResult:
    """Execute an approved action using its persisted tool and arguments."""
    action = get_action(db, action_id)
    if action is None:
        raise ActionExecutionError("The action does not exist.")
    if action.state != ACTION_STATE_APPROVED:
        if action.state == ACTION_STATE_EXECUTED:
            return ActionExecutionResult(
                action=action, result=None, executed=False, audit_event=None
            )
        raise ActionExecutionError("The action is not approved.")

    if action.tool_name not in TOOL_REGISTRY:
        raise ActionExecutionError("The action's tool is not registered.")
    if action.tool_name not in GATED_TOOLS:
        raise ActionExecutionError("The action's tool is not permitted.")

    args_schema, tool_func = TOOL_REGISTRY[action.tool_name]
    try:
        parsed_args = json.loads(action.args)
        if not isinstance(parsed_args, dict):
            raise ValueError("arguments must be an object")
        validated_args = args_schema(**parsed_args)
    except Exception as exc:
        raise ActionExecutionError(
            f"Invalid arguments for tool '{action.tool_name}'."
        ) from exc

    claimed = claim_action_for_execution(db, action.id, executed_by=executed_by)
    if claimed is None:
        current = get_action(db, action.id)
        if current is not None and current.state == ACTION_STATE_EXECUTED:
            return ActionExecutionResult(
                action=current, result=None, executed=False, audit_event=None
            )
        raise ActionExecutionError("The action could not be claimed for execution.")

    audit_draft = AuditDraft(
        actor_type="system",
        actor_id=executed_by,
        action="action.executed",
        correlation_id=claimed.correlation_id,
        session_id=claimed.session_id,
        target=claimed.tool_name,
        outcome="success",
        payload=json.dumps(
            {"action_id": claimed.id, "tool_name": claimed.tool_name},
            sort_keys=True,
            separators=(",", ":"),
        ),
    )

    try:
        tool_result = tool_func(validated_args)
    except Exception as exc:
        audit_draft = AuditDraft(
            actor_type="system",
            actor_id=executed_by,
            action="action.executed",
            correlation_id=claimed.correlation_id,
            session_id=claimed.session_id,
            target=claimed.tool_name,
            outcome="failure",
            payload=json.dumps(
                {"action_id": claimed.id, "tool_name": claimed.tool_name},
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        try:
            transition = mark_action_executed_with_audit(
                db,
                claimed.id,
                execution_status="failure",
                executed_by=executed_by,
                draft=audit_draft,
                execution_error=str(exc),
            )
        except Exception:
            release_action_execution_claim(db, claimed.id)
            raise
        if transition.action is None:
            release_action_execution_claim(db, claimed.id)
            raise ActionExecutionError("The action no longer exists.")
        return ActionExecutionResult(
            action=transition.action,
            result=None,
            executed=False,
            audit_event=None,
        )

    try:
        transition = mark_action_executed_with_audit(
            db,
            claimed.id,
            execution_status="success",
            executed_by=executed_by,
            draft=audit_draft,
        )
    except Exception as exc:
        raise ActionExecutionUncertainError(
            "The tool may have succeeded, but the execution result could not be recorded."
        ) from exc

    if not transition.changed or transition.action is None:
        raise ActionExecutionUncertainError(
            "The tool may have succeeded, but the execution result could not be recorded."
        )

    return ActionExecutionResult(
        action=transition.action,
        result=tool_result,
        executed=True,
        audit_event=None,
    )
