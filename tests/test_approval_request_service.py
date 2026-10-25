import uuid

import pytest

from app.agent_core import tools as tools_module
from app.modules.approvals.request_service import (
    ApprovalRequestConflictError,
    ApprovalRequestError,
    request_approval,
)
from app.database import AuditDraft, create_session

CORRELATION_ID = "11111111-1111-4111-8111-111111111111"


@pytest.fixture(autouse=True)
def gated_tools(monkeypatch):
    monkeypatch.setattr(
        tools_module, "GATED_TOOLS", frozenset({"search_support_docs"})
    )
    monkeypatch.setattr(
        "app.modules.approvals.request_service.GATED_TOOLS",
        frozenset({"search_support_docs"}),
    )


@pytest.mark.usefixtures("clean_database")
def test_valid_gated_tool_creates_pending_action_and_audit(database, monkeypatch):
    invoked = {"called": False}

    def fake_tool(args):
        invoked["called"] = True
        return []

    monkeypatch.setitem(
        tools_module.TOOL_REGISTRY,
        "search_support_docs",
        (tools_module.SearchSupportDocsArgs, fake_tool),
    )
    monkeypatch.setattr(
        "app.modules.approvals.request_service.TOOL_REGISTRY",
        tools_module.TOOL_REGISTRY,
    )

    result = request_approval(
        database,
        tool_name="search_support_docs",
        arguments={"query": "password reset"},
        request_key="req-1",
        correlation_id=CORRELATION_ID,
    )

    assert result.created is True
    assert result.action.state == "pending"
    assert result.action.tool_name == "search_support_docs"
    assert result.audit_event is not None
    assert invoked["called"] is False

    with database.connection() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM audit_events").fetchone()["c"]
    assert count == 1


@pytest.mark.usefixtures("clean_database")
def test_unknown_tool_rejected(database):
    with pytest.raises(ApprovalRequestError):
        request_approval(
            database,
            tool_name="does_not_exist",
            arguments={"query": "x"},
            request_key="req-2",
            correlation_id=CORRELATION_ID,
        )


@pytest.mark.usefixtures("clean_database")
def test_nongated_tool_rejected(database):
    with pytest.raises(ApprovalRequestError):
        request_approval(
            database,
            tool_name="lookup_ticket_history",
            arguments={"query": "billing"},
            request_key="req-3",
            correlation_id=CORRELATION_ID,
        )


@pytest.mark.usefixtures("clean_database")
def test_invalid_arguments_rejected(database):
    with pytest.raises(ApprovalRequestError):
        request_approval(
            database,
            tool_name="search_support_docs",
            arguments={"query": ""},
            request_key="req-4",
            correlation_id=CORRELATION_ID,
        )


@pytest.mark.usefixtures("clean_database")
def test_missing_request_key_rejected(database):
    with pytest.raises(ApprovalRequestError):
        request_approval(
            database,
            tool_name="search_support_docs",
            arguments={"query": "password"},
            request_key="",
            correlation_id=CORRELATION_ID,
        )


@pytest.mark.usefixtures("clean_database")
def test_same_key_same_payload_returns_existing(database):
    first = request_approval(
        database,
        tool_name="search_support_docs",
        arguments={"query": "password"},
        request_key="req-5",
        correlation_id=CORRELATION_ID,
    )
    second = request_approval(
        database,
        tool_name="search_support_docs",
        arguments={"query": "password"},
        request_key="req-5",
        correlation_id=CORRELATION_ID,
    )
    assert first.action.id == second.action.id
    assert second.created is False
    assert second.audit_event is None


@pytest.mark.usefixtures("clean_database")
def test_same_key_different_payload_conflicts(database):
    request_approval(
        database,
        tool_name="search_support_docs",
        arguments={"query": "password"},
        request_key="req-6",
        correlation_id=CORRELATION_ID,
    )
    with pytest.raises(ApprovalRequestConflictError):
        request_approval(
            database,
            tool_name="search_support_docs",
            arguments={"query": "billing"},
            request_key="req-6",
            correlation_id=CORRELATION_ID,
        )


@pytest.mark.usefixtures("clean_database")
def test_invalid_session_rejected(database):
    with pytest.raises(ApprovalRequestError):
        request_approval(
            database,
            tool_name="search_support_docs",
            arguments={"query": "password"},
            request_key="req-7",
            correlation_id=CORRELATION_ID,
            session_id=str(uuid.uuid4()),
        )


@pytest.mark.usefixtures("clean_database")
def test_invalid_correlation_rejected(database):
    with pytest.raises(ApprovalRequestError):
        request_approval(
            database,
            tool_name="search_support_docs",
            arguments={"query": "password"},
            request_key="req-8",
            correlation_id="not-a-uuid",
        )


@pytest.mark.usefixtures("clean_database")
def test_audit_failure_rolls_back_action(database, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("audit failed")

    monkeypatch.setattr(
        "app.persistence.repository._insert_audit_event", boom
    )

    with pytest.raises(RuntimeError):
        request_approval(
            database,
            tool_name="search_support_docs",
            arguments={"query": "password"},
            request_key="req-9",
            correlation_id=CORRELATION_ID,
        )

    with database.connection() as conn:
        action_count = conn.execute(
            "SELECT COUNT(*) AS c FROM actions"
        ).fetchone()["c"]
        audit_count = conn.execute(
            "SELECT COUNT(*) AS c FROM audit_events"
        ).fetchone()["c"]
    assert action_count == 0
    assert audit_count == 0
