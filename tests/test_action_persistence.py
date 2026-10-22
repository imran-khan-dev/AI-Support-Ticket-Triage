"""Tests for the storage of actions and audit events.

An action is a request for a consequential operation. It waits for a
decision, and an approved action can be marked as executed. Every move is
guarded: the state of the row is part of the condition of the update, so a
transition that was already made is never applied a second time.

The tests use the dedicated test database, which is emptied before each
test, so they never touch application data.
"""

import uuid

import psycopg
import pytest
from psycopg import errors

from app.persistence import (
    ACTION_STATE_APPROVED,
    ACTION_STATE_EXECUTED,
    ACTION_STATE_PENDING,
    ACTION_STATE_REJECTED,
    AuditDraft,
    UnknownSessionError,
    append_audit_event,
    create_action,
    create_session,
    decide_action,
    decide_action_with_audit,
    get_action,
    list_actions,
    mark_action_executed,
    mark_action_executed_with_audit,
)


#: Every test in this module writes to the database.
pytestmark = pytest.mark.usefixtures("clean_database")

#: A valid correlation id; the column is a PostgreSQL UUID.
CORRELATION_ID = "11111111-1111-4111-8111-111111111111"

#: Another valid correlation id.
OTHER_CORRELATION_ID = "22222222-2222-4222-8222-222222222222"


def _create(database, **overrides):
    """Create an action and return it."""
    values = {
        "request_key": "request-1",
        "tool_name": "issue_refund",
        "effect_class": "billing_write",
        "args": '{"order_id": "A-1"}',
        "requested_by": "agent",
        "correlation_id": CORRELATION_ID,
    }
    values.update(overrides)
    return create_action(database, **values)


def _draft(**overrides):
    """Return an audit event that has not been written yet."""
    values = {
        "actor_type": "approver",
        "action": "action.decided",
        "correlation_id": CORRELATION_ID,
    }
    values.update(overrides)
    return AuditDraft(**values)


def _audit_count(database):
    """Return how many audit events are stored."""
    with database.connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM audit_events"
        ).fetchone()
    return int(row["count"])


# ---------------------------------------------------------------------------
# Creating and reading actions
# ---------------------------------------------------------------------------


def test_an_action_is_created_and_read_back(database):
    """A new action waits for a decision."""
    action = _create(database)

    assert action.state == ACTION_STATE_PENDING
    assert action.tool_name == "issue_refund"
    assert action.effect_class == "billing_write"
    assert action.args == '{"order_id": "A-1"}'
    assert action.requested_by == "agent"
    assert action.correlation_id == CORRELATION_ID
    assert action.decided_by is None
    assert action.decided_at is None
    assert action.execution_status is None
    assert action.requested_at

    stored = get_action(database, action.id)
    assert stored is not None
    assert stored.id == action.id
    assert stored.request_key == "request-1"
    assert stored.state == ACTION_STATE_PENDING


def test_the_hash_of_the_arguments_is_computed_when_it_is_not_given(database):
    """The same arguments always get the same hash."""
    first = _create(database)
    second = _create(database, request_key="request-2")

    assert first.args_hash == second.args_hash
    assert len(first.args_hash) == 64


def test_a_duplicate_request_key_returns_the_same_action(database):
    """The same request twice does not create a second action."""
    first = _create(database)
    second = _create(database)

    assert second.id == first.id
    assert second.requested_at == first.requested_at

    with database.connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM actions").fetchone()
    assert int(row["count"]) == 1


def test_an_action_of_an_unknown_session_is_refused(database):
    """An action cannot belong to a session that does not exist."""
    with pytest.raises(UnknownSessionError):
        _create(
            database,
            session_id="33333333-3333-4333-8333-333333333333",
        )


def test_an_action_belongs_to_a_session(database):
    """An action can name the session it was asked in."""
    session = create_session(database)

    action = _create(database, session_id=session.id)

    assert action.session_id == session.id


# ---------------------------------------------------------------------------
# Listing actions
# ---------------------------------------------------------------------------


def test_the_actions_of_a_session_are_listed(database):
    """The filters narrow the result."""
    first = create_session(database)
    second = create_session(database)
    _create(database, session_id=first.id, request_key="a-1")
    _create(database, session_id=first.id, request_key="a-2")
    _create(database, session_id=second.id, request_key="b-1")

    assert len(list_actions(database)) == 3
    assert len(list_actions(database, session_id=first.id)) == 2
    assert len(list_actions(database, session_id=second.id)) == 1
    assert len(list_actions(database, state=ACTION_STATE_PENDING)) == 3
    assert len(list_actions(database, state=ACTION_STATE_APPROVED)) == 0
    assert len(list_actions(database, correlation_id=CORRELATION_ID)) == 3
    assert len(list_actions(database, correlation_id=OTHER_CORRELATION_ID)) == 0


def test_a_listing_is_bounded(database):
    """The result never grows past the limit."""
    for index in range(5):
        _create(database, request_key=f"request-{index}")

    assert len(list_actions(database, limit=2)) == 2
    assert len(list_actions(database, limit=0)) == 0
    assert len(list_actions(database, limit=-1)) == 0


def test_a_filter_that_cannot_match_returns_nothing(database):
    """A malformed identifier matches nothing instead of everything."""
    _create(database)

    assert list_actions(database, session_id="not-a-uuid") == []
    assert list_actions(database, correlation_id="not-a-uuid") == []


# ---------------------------------------------------------------------------
# Deciding an action
# ---------------------------------------------------------------------------


def test_a_pending_action_can_be_approved(database):
    """A decision moves a pending action."""
    action = _create(database)

    outcome = decide_action(
        database,
        action.id,
        decision=ACTION_STATE_APPROVED,
        decided_by="supervisor-1",
        reason="The customer is right.",
    )

    assert outcome.changed is True
    assert outcome.action is not None
    assert outcome.action.state == ACTION_STATE_APPROVED
    assert outcome.action.decided_by == "supervisor-1"
    assert outcome.action.decision_reason == "The customer is right."
    assert outcome.action.decided_at


def test_a_second_decision_does_not_overwrite_the_first(database):
    """Only a pending action can be decided."""
    action = _create(database)
    decide_action(
        database,
        action.id,
        decision=ACTION_STATE_APPROVED,
        decided_by="supervisor-1",
    )

    outcome = decide_action(
        database,
        action.id,
        decision=ACTION_STATE_REJECTED,
        decided_by="supervisor-2",
    )

    assert outcome.changed is False
    assert outcome.action is not None
    assert outcome.action.state == ACTION_STATE_APPROVED
    assert outcome.action.decided_by == "supervisor-1"


def test_a_rejected_action_cannot_be_approved_later(database):
    """A rejection is final."""
    action = _create(database)
    decide_action(
        database,
        action.id,
        decision=ACTION_STATE_REJECTED,
        decided_by="supervisor-1",
    )

    outcome = decide_action(
        database,
        action.id,
        decision=ACTION_STATE_APPROVED,
        decided_by="supervisor-2",
    )

    assert outcome.changed is False
    assert outcome.action is not None
    assert outcome.action.state == ACTION_STATE_REJECTED


def test_a_missing_action_is_reported(database):
    """A decision for an action that does not exist changes nothing."""
    missing = str(uuid.uuid4())

    outcome = decide_action(
        database,
        missing,
        decision=ACTION_STATE_APPROVED,
        decided_by="supervisor-1",
    )

    assert outcome.changed is False
    assert outcome.action is None

    invalid = decide_action(
        database,
        "not-a-uuid",
        decision=ACTION_STATE_APPROVED,
        decided_by="supervisor-1",
    )
    assert invalid.changed is False
    assert invalid.action is None


def test_an_unsupported_decision_is_refused(database):
    """Only a decision the database knows can be made."""
    action = _create(database)

    with pytest.raises(ValueError):
        decide_action(
            database,
            action.id,
            decision=ACTION_STATE_EXECUTED,
            decided_by="supervisor-1",
        )


# ---------------------------------------------------------------------------
# Executing an action
# ---------------------------------------------------------------------------


def test_only_an_approved_action_can_be_executed(database):
    """A pending action cannot be executed."""
    action = _create(database)

    outcome = mark_action_executed(
        database,
        action.id,
        execution_status="success",
        executed_by="worker-1",
    )

    assert outcome.changed is False
    assert outcome.action is not None
    assert outcome.action.state == ACTION_STATE_PENDING
    assert outcome.action.execution_status is None


def test_an_approved_action_is_marked_as_executed(database):
    """The execution of an approved action is recorded."""
    action = _create(database)
    decide_action(
        database,
        action.id,
        decision=ACTION_STATE_APPROVED,
        decided_by="supervisor-1",
    )

    outcome = mark_action_executed(
        database,
        action.id,
        execution_status="success",
        executed_by="worker-1",
    )

    assert outcome.changed is True
    assert outcome.action is not None
    assert outcome.action.state == ACTION_STATE_EXECUTED
    assert outcome.action.execution_status == "success"
    assert outcome.action.executed_by == "worker-1"
    assert outcome.action.executed_at


def test_an_executed_action_cannot_be_executed_again(database):
    """The first result of an execution is kept."""
    action = _create(database)
    decide_action(
        database,
        action.id,
        decision=ACTION_STATE_APPROVED,
        decided_by="supervisor-1",
    )
    mark_action_executed(
        database,
        action.id,
        execution_status="success",
        executed_by="worker-1",
    )

    outcome = mark_action_executed(
        database,
        action.id,
        execution_status="failure",
        executed_by="worker-2",
        execution_error="the second attempt failed",
    )

    assert outcome.changed is False
    assert outcome.action is not None
    assert outcome.action.state == ACTION_STATE_EXECUTED
    assert outcome.action.execution_status == "success"
    assert outcome.action.executed_by == "worker-1"
    assert outcome.action.execution_error is None


def test_an_unsupported_execution_status_is_refused(database):
    """Only an execution status the database knows can be recorded."""
    action = _create(database)

    with pytest.raises(ValueError):
        mark_action_executed(
            database,
            action.id,
            execution_status="maybe",
            executed_by="worker-1",
        )


# ---------------------------------------------------------------------------
# Audit events
# ---------------------------------------------------------------------------


def test_an_audit_event_is_stored(database):
    """An audit event keeps what it was told."""
    event = append_audit_event(database, _draft(actor_id="supervisor-1"))

    assert event.id
    assert event.event_id
    assert event.actor_type == "approver"
    assert event.actor_id == "supervisor-1"
    assert event.action == "action.decided"
    assert event.correlation_id == CORRELATION_ID
    assert event.action_id is None
    assert event.created_at


def test_a_decision_and_its_audit_event_are_written_together(database):
    """The decision and the event are committed as one."""
    action = _create(database)

    outcome = decide_action_with_audit(
        database,
        action.id,
        decision=ACTION_STATE_APPROVED,
        decided_by="supervisor-1",
        draft=_draft(),
    )

    assert outcome.changed is True
    assert _audit_count(database) == 1

    with database.connection() as conn:
        row = conn.execute(
            "SELECT actor_id, decision, action_id FROM audit_events"
        ).fetchone()
    assert row["actor_id"] == "supervisor-1"
    assert row["decision"] == "approved"
    assert str(row["action_id"]) == action.id


def test_a_decision_that_was_not_applied_writes_no_event(database):
    """A rejected decision leaves the action and the trail as they were."""
    action = _create(database)
    decide_action(
        database,
        action.id,
        decision=ACTION_STATE_APPROVED,
        decided_by="supervisor-1",
    )

    outcome = decide_action_with_audit(
        database,
        action.id,
        decision=ACTION_STATE_REJECTED,
        decided_by="supervisor-2",
        draft=_draft(),
    )

    assert outcome.changed is False
    assert _audit_count(database) == 0


def test_an_execution_and_its_audit_event_are_written_together(database):
    """The execution and the event are committed as one."""
    action = _create(database)
    decide_action(
        database,
        action.id,
        decision=ACTION_STATE_APPROVED,
        decided_by="supervisor-1",
    )

    outcome = mark_action_executed_with_audit(
        database,
        action.id,
        execution_status="success",
        executed_by="worker-1",
        draft=_draft(action="action.executed", actor_type="system"),
    )

    assert outcome.changed is True
    assert _audit_count(database) == 1

    with database.connection() as conn:
        row = conn.execute(
            "SELECT actor_type, outcome, action_id FROM audit_events"
        ).fetchone()
    assert row["actor_type"] == "system"
    assert row["outcome"] == "success"
    assert str(row["action_id"]) == action.id


def test_a_failed_audit_event_rolls_back_the_decision(database):
    """A decision that cannot be recorded does not happen."""
    action = _create(database)

    with pytest.raises(errors.CheckViolation):
        decide_action_with_audit(
            database,
            action.id,
            decision=ACTION_STATE_APPROVED,
            decided_by="supervisor-1",
            # The database only knows a few kinds of actor.
            draft=_draft(actor_type="wizard"),
        )

    stored = get_action(database, action.id)
    assert stored is not None
    assert stored.state == ACTION_STATE_PENDING
    assert stored.decided_by is None
    assert _audit_count(database) == 0


def test_a_failed_audit_event_rolls_back_the_execution(database):
    """An execution that cannot be recorded does not happen."""
    action = _create(database)
    decide_action(
        database,
        action.id,
        decision=ACTION_STATE_APPROVED,
        decided_by="supervisor-1",
    )

    with pytest.raises(errors.CheckViolation):
        mark_action_executed_with_audit(
            database,
            action.id,
            execution_status="success",
            executed_by="worker-1",
            draft=_draft(actor_type="wizard"),
        )

    stored = get_action(database, action.id)
    assert stored is not None
    assert stored.state == ACTION_STATE_APPROVED
    assert stored.execution_status is None
    assert _audit_count(database) == 0


def test_an_audit_event_cannot_be_changed_or_removed(database):
    """The trail of a decision cannot be rewritten."""
    event = append_audit_event(database, _draft())

    with pytest.raises(errors.RaiseException):
        with database.write_transaction() as conn:
            conn.execute(
                "UPDATE audit_events SET action = 'tampered' WHERE id = %s",
                (event.id,),
            )

    with pytest.raises(errors.RaiseException):
        with database.write_transaction() as conn:
            conn.execute(
                "DELETE FROM audit_events WHERE id = %s", (event.id,)
            )


def test_the_history_of_an_action_is_kept_when_it_cannot_be_deleted(database):
    """An action with a trail cannot be deleted, and its trail stays whole.

    The audit table is append-only, so the reference of an event cannot be
    cleared: the deletion of such an action is refused instead, and the
    history of the decision stays complete.
    """
    action = _create(database)
    decide_action_with_audit(
        database,
        action.id,
        decision=ACTION_STATE_APPROVED,
        decided_by="supervisor-1",
        draft=_draft(),
    )

    with pytest.raises(errors.ForeignKeyViolation):
        with database.write_transaction() as conn:
            conn.execute("DELETE FROM actions WHERE id = %s", (action.id,))

    assert get_action(database, action.id) is not None
    assert _audit_count(database) == 1
    with database.connection() as conn:
        row = conn.execute(
            "SELECT action_id, decision FROM audit_events"
        ).fetchone()
    assert str(row["action_id"]) == action.id
    assert row["decision"] == "approved"


def test_an_action_without_a_trail_can_be_deleted(database):
    """An action that was never decided leaves nothing behind."""
    action = _create(database)

    with database.write_transaction() as conn:
        conn.execute("DELETE FROM actions WHERE id = %s", (action.id,))

    assert get_action(database, action.id) is None
    assert _audit_count(database) == 0


def test_an_audit_event_cannot_name_an_action_that_does_not_exist(database):
    """The link to an action is checked by the database."""
    with pytest.raises(errors.ForeignKeyViolation):
        append_audit_event(
            database, _draft(action_id=str(uuid.uuid4()))
        )


def test_the_connection_survives_a_failed_audit_event(database):
    """A failed write leaves the connection usable."""
    action = _create(database)

    with pytest.raises(psycopg.Error):
        decide_action_with_audit(
            database,
            action.id,
            decision=ACTION_STATE_APPROVED,
            decided_by="supervisor-1",
            draft=_draft(actor_type="wizard"),
        )

    with database.connection() as conn:
        assert conn.execute("SELECT 1 AS one").fetchone()["one"] == 1
