import uuid

import pytest

from app.core.security import generate_approver_token
from app.modules.approvals.service import register_approver
from app.modules.approvals.decision_service import (
    ApprovalDecisionError,
    ApprovalDecisionUnauthorizedError,
    decide_approval,
)
from app.database import (
    ACTION_STATE_APPROVED,
    ACTION_STATE_PENDING,
    ACTION_STATE_REJECTED,
    create_action,
    get_action,
    set_approver_active,
)

TEST_SECRET = "test-only-secret-for-approver-credentials-0123456789"
CORRELATION_ID = "11111111-1111-4111-8111-111111111111"


def _register(database, **overrides):
    values = {
        "display_name": "Sam Rivera",
        "role": "supervisor",
        "token": generate_approver_token(),
        "secret": TEST_SECRET,
    }
    values.update(overrides)
    approver = register_approver(database, **values)
    return approver, values["token"]


def _create_action(database):
    return create_action(
        database,
        request_key=f"req-{uuid.uuid4()}",
        tool_name="issue_refund",
        effect_class="billing_write",
        args='{"order_id": "A-1"}',
        requested_by="agent",
        correlation_id=CORRELATION_ID,
    )


@pytest.mark.usefixtures("clean_database")
def test_approve_pending_action(database):
    approver, token = _register(database)
    action = _create_action(database)
    result = decide_approval(
        database,
        action_id=action.id,
        decision="approved",
        approver_token=token,
        secret=TEST_SECRET,
    )
    assert result.changed is True
    assert result.action.state == ACTION_STATE_APPROVED
    assert result.action.decided_by == approver.id


@pytest.mark.usefixtures("clean_database")
def test_reject_pending_action(database):
    approver, token = _register(database)
    action = _create_action(database)
    result = decide_approval(
        database,
        action_id=action.id,
        decision="rejected",
        approver_token=token,
        secret=TEST_SECRET,
    )
    assert result.changed is True
    assert result.action.state == ACTION_STATE_REJECTED


@pytest.mark.usefixtures("clean_database")
def test_unauthenticated_approver_rejected(database):
    _register(database)
    action = _create_action(database)
    with pytest.raises(ApprovalDecisionUnauthorizedError):
        decide_approval(
            database,
            action_id=action.id,
            decision="approved",
            approver_token="not-a-token",
            secret=TEST_SECRET,
        )


@pytest.mark.usefixtures("clean_database")
def test_unauthorized_role_rejected(database):
    approver, token = _register(database, role="agent")
    action = _create_action(database)
    with pytest.raises(ApprovalDecisionUnauthorizedError):
        decide_approval(
            database,
            action_id=action.id,
            decision="approved",
            approver_token=token,
            secret=TEST_SECRET,
        )


@pytest.mark.usefixtures("clean_database")
def test_inactive_approver_rejected(database):
    approver, token = _register(database)
    set_approver_active(database, approver.id, False)
    action = _create_action(database)
    with pytest.raises(ApprovalDecisionUnauthorizedError):
        decide_approval(
            database,
            action_id=action.id,
            decision="approved",
            approver_token=token,
            secret=TEST_SECRET,
        )


@pytest.mark.usefixtures("clean_database")
def test_missing_action_rejected(database):
    approver, token = _register(database)
    with pytest.raises(ApprovalDecisionError):
        decide_approval(
            database,
            action_id=str(uuid.uuid4()),
            decision="approved",
            approver_token=token,
            secret=TEST_SECRET,
        )


@pytest.mark.usefixtures("clean_database")
def test_nonpending_action_rejected(database):
    approver, token = _register(database)
    action = _create_action(database)
    decide_approval(
        database,
        action_id=action.id,
        decision="approved",
        approver_token=token,
        secret=TEST_SECRET,
    )
    with pytest.raises(ApprovalDecisionError):
        decide_approval(
            database,
            action_id=action.id,
            decision="rejected",
            approver_token=token,
            secret=TEST_SECRET,
        )


@pytest.mark.usefixtures("clean_database")
def test_repeated_decision_returns_unchanged(database):
    approver, token = _register(database)
    action = _create_action(database)
    first = decide_approval(
        database,
        action_id=action.id,
        decision="approved",
        approver_token=token,
        secret=TEST_SECRET,
    )
    second = decide_approval(
        database,
        action_id=action.id,
        decision="approved",
        approver_token=token,
        secret=TEST_SECRET,
    )
    assert first.changed is True
    assert second.changed is False
    assert second.action.state == ACTION_STATE_APPROVED


@pytest.mark.usefixtures("clean_database")
def test_audit_failure_rolls_back_decision(database, monkeypatch):
    approver, token = _register(database)
    action = _create_action(database)

    def boom(*args, **kwargs):
        raise RuntimeError("audit failed")

    monkeypatch.setattr("app.database.repository._insert_audit_event", boom)

    with pytest.raises(RuntimeError):
        decide_approval(
            database,
            action_id=action.id,
            decision="approved",
            approver_token=token,
            secret=TEST_SECRET,
        )

    stored = get_action(database, action.id)
    assert stored.state == ACTION_STATE_PENDING
