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
def test_execute_unapproved_action(client):
    action = create_action(
        get_database(os.environ["DATABASE_URL"]),
        request_key=f"req-{uuid.uuid4()}",
        tool_name="search_support_docs",
        effect_class="gated_tool",
        args='{"query": "password"}',
        requested_by="agent",
        correlation_id=CORRELATION_ID,
    )
    resp = client.post(f"/api/v1/tickets/approvals/actions/{action.id}/execute")
    assert resp.status_code == 409
