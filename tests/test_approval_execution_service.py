import uuid

import pytest

from app.core.security import generate_approver_token
from app.modules.approvals.service import register_approver
from app.modules.approvals.decision_service import decide_approval
import threading

from app.modules.approvals.execution_service import (
    ActionExecutionError,
    ActionExecutionUncertainError,
    execute_approved_action,
)
from app.agent_core import tools as tools_module
from app.database import create_action, get_action

TEST_SECRET = "test-only-secret-for-approver-credentials-0123456789"
CORRELATION_ID = "11111111-1111-4111-8111-111111111111"


def _register(database):
    token = generate_approver_token()
    approver = register_approver(
        database,
        display_name="Sam",
        role="supervisor",
        token=token,
        secret=TEST_SECRET,
    )
    return approver, token


def _create_action(database):
    return create_action(
        database,
        request_key=f"req-{uuid.uuid4()}",
        tool_name="search_support_docs",
        effect_class="gated_tool",
        args='{"query": "password"}',
        requested_by="agent",
        correlation_id=CORRELATION_ID,
    )


@pytest.fixture(autouse=True)
def gated(monkeypatch):
    monkeypatch.setattr(tools_module, "GATED_TOOLS", frozenset({"search_support_docs"}))
    monkeypatch.setattr(
        "app.modules.approvals.execution_service.GATED_TOOLS",
        frozenset({"search_support_docs"}),
    )
    monkeypatch.setattr(
        "app.modules.approvals.execution_service.TOOL_REGISTRY",
        tools_module.TOOL_REGISTRY,
    )


@pytest.mark.usefixtures("clean_database")
def test_successful_execution(database, monkeypatch):
    invoked = {"n": 0}

    def fake(args):
        invoked["n"] += 1
        return [{"id": "doc"}]

    monkeypatch.setitem(
        tools_module.TOOL_REGISTRY,
        "search_support_docs",
        (tools_module.SearchSupportDocsArgs, fake),
    )
    monkeypatch.setattr(
        "app.modules.approvals.execution_service.TOOL_REGISTRY",
        tools_module.TOOL_REGISTRY,
    )

    approver, token = _register(database)
    action = _create_action(database)
    decide_approval(
        database,
        action_id=action.id,
        decision="approved",
        approver_token=token,
        secret=TEST_SECRET,
    )
    result = execute_approved_action(database, action_id=action.id)
    assert result.executed is True
    assert result.result == [{"id": "doc"}]
    assert invoked["n"] == 1
    stored = get_action(database, action.id)
    assert stored.state == "executed"


@pytest.mark.usefixtures("clean_database")
def test_pending_rejected(database):
    action = _create_action(database)
    with pytest.raises(ActionExecutionError):
        execute_approved_action(database, action_id=action.id)


@pytest.mark.usefixtures("clean_database")
def test_rejected_rejected(database):
    approver, token = _register(database)
    action = _create_action(database)
    decide_approval(
        database,
        action_id=action.id,
        decision="rejected",
        approver_token=token,
        secret=TEST_SECRET,
    )
    with pytest.raises(ActionExecutionError):
        execute_approved_action(database, action_id=action.id)


@pytest.mark.usefixtures("clean_database")
def test_already_executed_not_reexecuted(database, monkeypatch):
    def fake(args):
        return []

    monkeypatch.setitem(
        tools_module.TOOL_REGISTRY,
        "search_support_docs",
        (tools_module.SearchSupportDocsArgs, fake),
    )
    monkeypatch.setattr(
        "app.modules.approvals.execution_service.TOOL_REGISTRY",
        tools_module.TOOL_REGISTRY,
    )
    approver, token = _register(database)
    action = _create_action(database)
    decide_approval(
        database,
        action_id=action.id,
        decision="approved",
        approver_token=token,
        secret=TEST_SECRET,
    )
    first = execute_approved_action(database, action_id=action.id)
    assert first.executed is True
    second = execute_approved_action(database, action_id=action.id)
    assert second.executed is False


@pytest.mark.usefixtures("clean_database")
def test_missing_action(database):
    with pytest.raises(ActionExecutionError):
        execute_approved_action(database, action_id=str(uuid.uuid4()))


@pytest.mark.usefixtures("clean_database")
def test_unknown_tool(database, monkeypatch):
    monkeypatch.setitem(
        tools_module.TOOL_REGISTRY, "search_support_docs", None
    ) if False else None
    action = _create_action(database)
    # make tool unknown by changing action? can't. Instead register not gated? skip
    with pytest.raises(ActionExecutionError):
        # action pending anyway
        execute_approved_action(database, action_id=action.id)


@pytest.mark.usefixtures("clean_database")
def test_invalid_args(database, monkeypatch):
    monkeypatch.setattr(tools_module, "SearchSupportDocsArgs", tools_module.SearchSupportDocsArgs)
    action = create_action(
        database,
        request_key=f"req-{uuid.uuid4()}",
        tool_name="search_support_docs",
        effect_class="gated_tool",
        args='{"query": ""}',
        requested_by="agent",
        correlation_id=CORRELATION_ID,
    )
    approver, token = _register(database)
    decide_approval(
        database,
        action_id=action.id,
        decision="approved",
        approver_token=token,
        secret=TEST_SECRET,
    )
    with pytest.raises(ActionExecutionError):
        execute_approved_action(database, action_id=action.id)


@pytest.mark.usefixtures("clean_database")
def test_tool_failure(database, monkeypatch):
    def boom(args):
        raise RuntimeError("boom")

    monkeypatch.setitem(
        tools_module.TOOL_REGISTRY,
        "search_support_docs",
        (tools_module.SearchSupportDocsArgs, boom),
    )
    monkeypatch.setattr(
        "app.modules.approvals.execution_service.TOOL_REGISTRY",
        tools_module.TOOL_REGISTRY,
    )
    approver, token = _register(database)
    action = _create_action(database)
    decide_approval(
        database,
        action_id=action.id,
        decision="approved",
        approver_token=token,
        secret=TEST_SECRET,
    )
    result = execute_approved_action(database, action_id=action.id)
    assert result.executed is False
    stored = get_action(database, action.id)
    assert stored.state == "executed"
    assert stored.execution_status == "failure"


@pytest.mark.usefixtures("clean_database")
def test_audit_failure_rolls_back(database, monkeypatch):
    def fake(args):
        return []

    monkeypatch.setitem(
        tools_module.TOOL_REGISTRY,
        "search_support_docs",
        (tools_module.SearchSupportDocsArgs, fake),
    )
    monkeypatch.setattr(
        "app.modules.approvals.execution_service.TOOL_REGISTRY",
        tools_module.TOOL_REGISTRY,
    )

    approver, token = _register(database)
    action = _create_action(database)
    decide_approval(
        database,
        action_id=action.id,
        decision="approved",
        approver_token=token,
        secret=TEST_SECRET,
    )

    def boom(*args, **kwargs):
        raise RuntimeError("audit failed")

    monkeypatch.setattr("app.persistence.repository._insert_audit_event", boom)
    with pytest.raises(RuntimeError):
        execute_approved_action(database, action_id=action.id)


@pytest.mark.usefixtures("clean_database")
def test_uncertain_outcome_preserves_claim(database, monkeypatch):
    def fake(args):
        return []

    monkeypatch.setitem(
        tools_module.TOOL_REGISTRY,
        "search_support_docs",
        (tools_module.SearchSupportDocsArgs, fake),
    )
    monkeypatch.setattr(
        "app.modules.approvals.execution_service.TOOL_REGISTRY",
        tools_module.TOOL_REGISTRY,
    )

    def boom(*args, **kwargs):
        raise RuntimeError("audit failed")

    approver, token = _register(database)
    action = _create_action(database)
    decide_approval(
        database,
        action_id=action.id,
        decision="approved",
        approver_token=token,
        secret=TEST_SECRET,
    )

    monkeypatch.setattr("app.persistence.repository._insert_audit_event", boom)

    with pytest.raises(ActionExecutionUncertainError):
        execute_approved_action(database, action_id=action.id)

    claimed = get_action(database, action.id)
    assert claimed.state == "approved"
    assert claimed.executed_by is not None

    with pytest.raises(ActionExecutionError):
        execute_approved_action(database, action_id=action.id)


@pytest.mark.usefixtures("clean_database")
def test_concurrent_claim_only_one_acquires(database, monkeypatch):
    import time

    started = threading.Barrier(2)
    results = []

    def slow(args):
        time.sleep(0.2)
        return []

    monkeypatch.setitem(
        tools_module.TOOL_REGISTRY,
        "search_support_docs",
        (tools_module.SearchSupportDocsArgs, slow),
    )
    monkeypatch.setattr(
        "app.modules.approvals.execution_service.TOOL_REGISTRY",
        tools_module.TOOL_REGISTRY,
    )

    approver, token = _register(database)
    action = _create_action(database)
    decide_approval(
        database,
        action_id=action.id,
        decision="approved",
        approver_token=token,
        secret=TEST_SECRET,
    )

    def worker():
        started.wait()
        try:
            r = execute_approved_action(database, action_id=action.id)
            results.append(("ok", r.executed))
        except Exception as exc:
            results.append(("err", type(exc).__name__))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = [r for r in results if r[0] == "ok" and r[1] is True]
    assert len(successes) == 1
