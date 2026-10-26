"""Tests that lock the public HTTP contracts of the API.

The field set of a response is part of the contract of an endpoint, so these
tests assert the exact set of keys rather than the presence of some of them.
An extra field in a response body is a change to the contract, so it has to
be made deliberately and documented, not slip in unnoticed.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.agent_core.agent import AgentResult
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


@pytest.fixture()
def agent_stub():
    """Replace the provider so that no request leaves the machine."""
    with patch("app.modules.agent.route.run_agent") as mock_run:
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
    response = client.post(
        "/api/v1/tickets/agent/run",
        json={"message": "I was charged twice."},
    )

    assert response.status_code == 200
    assert set(response.json()) == AGENT_FIELDS


def test_the_agent_endpoint_rejects_a_blank_message():
    """A blank message is refused by the request schema."""
    response = client.post(
        "/api/v1/tickets/agent/run",
        json={"message": "   "},
    )

    assert response.status_code == 422


def test_the_agent_endpoint_reports_a_missing_session():
    """An unknown session id is refused instead of creating one."""
    response = client.post(
        "/api/v1/tickets/agent/run",
        json={
            "message": "I was charged twice.",
            "session_id": "11111111-1111-4111-8111-111111111111",
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "The session does not exist."}


def test_the_agent_endpoint_reports_a_malformed_session():
    """A session id that is not a UUID is refused."""
    response = client.post(
        "/api/v1/tickets/agent/run",
        json={"message": "I was charged twice.", "session_id": "not-a-uuid"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "The session id is not a valid session identifier."
    }
