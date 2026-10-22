"""Application service for deciding an approval request.

This module authenticates an approver, enforces the capability policy, and
records the decision together with its audit event. It never executes the
underlying tool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from app.modules.approvals.policy import CAPABILITY_DECIDE_ACTION, is_authorized
from app.modules.approvals.service import (
    ApproverConfigurationError,
    authenticate_approver,
)
from app.persistence.db import Database
from app.persistence.models import (
    ACTION_DECISION_STATES,
    ACTION_STATE_PENDING,
    ACTION_STATE_APPROVED,
    ACTION_STATE_REJECTED,
    AuditDraft,
    Action,
    ActionTransition,
    Approver,
)
from app.persistence.repository import (
    ActionConflictError,
    decide_action_with_audit,
    get_action,
)


class ApprovalDecisionError(RuntimeError):
    """Raised when an approval decision cannot be made safely."""


class ApprovalDecisionUnauthorizedError(ApprovalDecisionError):
    """Raised when the approver is not authenticated or authorized."""


@dataclass(frozen=True)
class ApprovalDecisionResult:
    """The outcome of an approval decision."""

    action: Action | None
    approver: Approver
    changed: bool


def decide_approval(
    db: Database,
    *,
    action_id: str,
    decision: str,
    approver_token: object,
    secret: str,
    reason: str | None = None,
) -> ApprovalDecisionResult:
    """Approve or reject a pending action as an authenticated approver."""
    if decision not in ACTION_DECISION_STATES:
        raise ApprovalDecisionError(f"Unsupported action decision: {decision!r}")

    approver = authenticate_approver(db, approver_token, secret=secret)
    if approver is None:
        raise ApprovalDecisionUnauthorizedError(
            "The approver credential is not valid."
        )
    if not is_authorized(approver, CAPABILITY_DECIDE_ACTION):
        raise ApprovalDecisionUnauthorizedError(
            "The approver is not authorized to decide actions."
        )

    action = get_action(db, action_id)
    if action is None:
        raise ApprovalDecisionError("The action does not exist.")
    if action.state != ACTION_STATE_PENDING:
        same_decision = (
            (action.state == ACTION_STATE_APPROVED and decision == "approved")
            or (action.state == ACTION_STATE_REJECTED and decision == "rejected")
        )
        if not same_decision:
            raise ApprovalDecisionError("The action is not pending.")

    audit_draft = AuditDraft(
        actor_type="approver",
        actor_id=approver.id,
        action="action.decided",
        correlation_id=action.correlation_id,
        session_id=action.session_id,
        target=action.tool_name,
        decision=decision,
        outcome="success",
        payload=json.dumps(
            {"action_id": action.id, "reason": reason},
            sort_keys=True,
            separators=(",", ":"),
        ),
    )

    try:
        transition = decide_action_with_audit(
            db,
            action_id,
            decision=decision,
            decided_by=approver.id,
            draft=audit_draft,
            reason=reason,
        )
    except ActionConflictError as exc:
        raise ApprovalDecisionError(str(exc)) from exc

    if transition.action is None:
        raise ApprovalDecisionError("The action does not exist.")

    return ApprovalDecisionResult(
        action=transition.action,
        approver=approver,
        changed=transition.changed,
    )
