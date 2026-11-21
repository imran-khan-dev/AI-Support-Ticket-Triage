"""Tests that lock the public HTTP contracts of the API.

The field set of a response is part of the contract of an endpoint, so these
tests assert the exact set of keys rather than the presence of some of them.
An extra field in a response body is a change to the contract, so it has to
be made deliberately and documented, not slip in unnoticed.
"""

import os
import uuid
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.agent_core.agent import AgentResult
from app.database import get_database
from app.database.repository import _canonical_uuid
from app.modules.integrations.service import issue_integration_credential
from main import app

client = TestClient(app)

#: The fields the analysis endpoint answers with.
ANALYZE_FIELDS = {
    "category",
    "priority",
    "sentiment",
    "summary",
    "requires_human_review",
    "confidence",
}

#: The fields the agent endpoint answers with.
AGENT_FIELDS = {
    "status",
    "answer",
    "mock_mode",
    "tool_calls",
    "error",
    "session_id",
}


def _issue_integration_auth_header(tenant_id=None):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(
        db, integration_name=f"test-api-contract-{uuid.uuid4().hex[:8]}"
    )
    tid = tenant_id or str(uuid.uuid4())
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO tenants (id, name, enabled) VALUES (%s, %s, TRUE) ON CONFLICT (id) DO NOTHING",
            (_canonical_uuid(tid), f"api-contract-tenant-{tid[:8]}"),
        )
        conn.execute(
            "UPDATE integrations SET tenant_id = %s WHERE id = %s",
            (_canonical_uuid(tid), _canonical_uuid(integration.id)),
        )
    return {"Authorization": f"Bearer {token}"}, tid


@contextmanager
def _authenticated():
    """Yield headers for an authenticated request."""
    headers, _ = _issue_integration_auth_header()
    yield headers


@pytest.fixture()
def agent_stub():
    """Replace the provider so that no request leaves the machine."""
    with patch("app.modules.agent.service.run_agent") as mock_run:
        mock_run.return_value = AgentResult(
            status="success",
            answer="Done.",
            tool_calls=[],
            error=None,
        )
        yield mock_run


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_the_health_endpoint_answers_with_exactly_the_status():
    """The health endpoint answers with one field and nothing else."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def test_the_analysis_endpoint_answers_with_exactly_the_documented_fields():
    """The analysis endpoint answers with the six agreed fields."""
    with patch("app.modules.analyze.route.groq_client", None):
        response = client.post(
            "/api/v1/tickets/analyze",
            json={"message": "I cannot log into my account."},
        )

    assert response.status_code == 200
    assert set(response.json()) == ANALYZE_FIELDS


def test_the_analysis_endpoint_rejects_a_blank_message():
    """A blank message is refused by the request schema."""
    response = client.post(
        "/api/v1/tickets/analyze",
        json={"message": "   "},
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def test_the_agent_endpoint_answers_with_exactly_the_documented_fields(
    database, agent_stub
):
    """The agent endpoint answers with the six agreed fields."""
    with _authenticated() as headers:
        response = client.post(
            "/api/v1/tickets/agent/run",
            json={"message": "I was charged twice."},
            headers=headers,
        )

    assert response.status_code == 200
    assert set(response.json()) == AGENT_FIELDS


def test_the_agent_endpoint_rejects_a_blank_message():
    """A blank message is refused by the request schema."""
    with _authenticated() as headers:
        response = client.post(
            "/api/v1/tickets/agent/run",
            json={"message": "   "},
            headers=headers,
        )

    assert response.status_code == 422


def test_the_agent_endpoint_reports_a_missing_session():
    """An unknown session id is refused instead of creating one."""
    with _authenticated() as headers:
        response = client.post(
            "/api/v1/tickets/agent/run",
            json={
                "message": "I was charged twice.",
                "session_id": "11111111-1111-4111-8111-111111111111",
            },
            headers=headers,
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "The session does not exist."}


def test_the_agent_endpoint_reports_a_malformed_session():
    """A session id that is not a UUID is refused."""
    with _authenticated() as headers:
        response = client.post(
            "/api/v1/tickets/agent/run",
            json={"message": "I was charged twice.", "session_id": "not-a-uuid"},
            headers=headers,
        )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "The session id is not a valid session identifier."
    }
