import os

os.environ.setdefault("APPROVER_TOKEN_SECRET", "test-only-secret-for-approver-credentials-0123456789")

import uuid

import pytest
from fastapi.testclient import TestClient

from main import app
from app.core.security import generate_approver_token
from app.persistence import create_action, get_database
from app.modules.approvals.service import register_approver

TEST_SECRET = os.environ["APPROVER_TOKEN_SECRET"]
CORRELATION_ID = "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.usefixtures("clean_database")
def test_list_requires_auth(client):
    assert client.get("/api/v1/tickets/approvals/actions").status_code == 401


@pytest.mark.usefixtures("clean_database")
def test_decide_unknown_action(client):
    token = generate_approver_token()
    db = get_database(os.environ["DATABASE_URL"])
    register_approver(db, display_name="Sam", role="supervisor", token=token, secret=TEST_SECRET)
    resp = client.post(
        f"/api/v1/tickets/approvals/actions/{uuid.uuid4()}/decide",
        json={"decision": "approved"},
        headers={"X-Approver-Token": token},
    )
    assert resp.status_code == 409


@pytest.mark.usefixtures("clean_database")
def test_execute_requires_auth(client):
    db = get_database(os.environ["DATABASE_URL"])
    action = create_action(
        db,
        request_key=f"req-{uuid.uuid4()}",
        tool_name="search_support_docs",
        effect_class="gated_tool",
        args='{"query": "password"}',
        requested_by="agent",
        correlation_id=CORRELATION_ID,
    )
    resp = client.post(f"/api/v1/tickets/approvals/actions/{action.id}/execute")
    assert resp.status_code == 401


@pytest.mark.usefixtures("clean_database")
def test_execute_invalid_token(client):
    db = get_database(os.environ["DATABASE_URL"])
    action = create_action(
        db,
        request_key=f"req-{uuid.uuid4()}",
        tool_name="search_support_docs",
        effect_class="gated_tool",
        args='{"query": "password"}',
        requested_by="agent",
        correlation_id=CORRELATION_ID,
    )
    resp = client.post(
        f"/api/v1/tickets/approvals/actions/{action.id}/execute",
        headers={"X-Approver-Token": "bad-token"},
    )
    assert resp.status_code == 401


@pytest.mark.usefixtures("clean_database")
def test_execute_inactive_approver(client):
    token = generate_approver_token()
    db = get_database(os.environ["DATABASE_URL"])
    approver = register_approver(db, display_name="Sam", role="supervisor", token=token, secret=TEST_SECRET)
    from app.persistence import set_approver_active

    set_approver_active(db, approver.id, False)
    action = create_action(
        db,
        request_key=f"req-{uuid.uuid4()}",
        tool_name="search_support_docs",
        effect_class="gated_tool",
        args='{"query": "password"}',
        requested_by="agent",
        correlation_id=CORRELATION_ID,
    )
    resp = client.post(
        f"/api/v1/tickets/approvals/actions/{action.id}/execute",
        headers={"X-Approver-Token": token},
    )
    assert resp.status_code == 401


@pytest.mark.usefixtures("clean_database")
def test_execute_unauthorized_role(client):
    token = generate_approver_token()
    db = get_database(os.environ["DATABASE_URL"])
    register_approver(db, display_name="Sam", role="agent", token=token, secret=TEST_SECRET)
    action = create_action(
        db,
        request_key=f"req-{uuid.uuid4()}",
        tool_name="search_support_docs",
        effect_class="gated_tool",
        args='{"query": "password"}',
        requested_by="agent",
        correlation_id=CORRELATION_ID,
    )
    resp = client.post(
        f"/api/v1/tickets/approvals/actions/{action.id}/execute",
        headers={"X-Approver-Token": token},
    )
    assert resp.status_code == 403


@pytest.mark.usefixtures("clean_database")
def test_execute_authorized_approver(client, monkeypatch):
    import app.agent_core.tools as tools_module

    monkeypatch.setattr(tools_module, "GATED_TOOLS", frozenset({"search_support_docs"}))
    monkeypatch.setattr("app.modules.approvals.execution_service.GATED_TOOLS", frozenset({"search_support_docs"}))
    monkeypatch.setattr("app.modules.approvals.execution_service.TOOL_REGISTRY", tools_module.TOOL_REGISTRY)

    token = generate_approver_token()
    db = get_database(os.environ["DATABASE_URL"])
    register_approver(db, display_name="Sam", role="supervisor", token=token, secret=TEST_SECRET)

    from app.modules.approvals.decision_service import decide_approval

    action = create_action(
        db,
        request_key=f"req-{uuid.uuid4()}",
        tool_name="search_support_docs",
        effect_class="gated_tool",
        args='{"query": "password"}',
        requested_by="agent",
        correlation_id=CORRELATION_ID,
    )
    decide_approval(db, action_id=action.id, decision="approved", approver_token=token, secret=TEST_SECRET)

    resp = client.post(
        f"/api/v1/tickets/approvals/actions/{action.id}/execute",
        headers={"X-Approver-Token": token},
    )
    assert resp.status_code == 200
    assert resp.json()["executed"] is True
