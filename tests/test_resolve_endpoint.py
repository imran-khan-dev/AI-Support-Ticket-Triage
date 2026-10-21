import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from main import app
from app.modules.agent.schema import TicketResolveResponse

client = TestClient(app)

# The agent endpoint stores every run in the session store, so these tests
# need the PostgreSQL test database.
pytestmark = pytest.mark.usefixtures("clean_database")


def _make_agent_result(status, answer=None, tool_calls=None, error=None):
    from app.agent_core.agent import AgentResult
    return AgentResult(
        status=status,
        answer=answer,
        tool_calls=tool_calls if tool_calls is not None else [],
        error=error,
    )


# ---------------------------------------------------------------------------
# Successful agent result
# ---------------------------------------------------------------------------


def test_resolve_endpoint_successful_agent_result():
    """A successful agent run returns 200 with the answer and tool calls."""
    with patch('app.modules.agent.route.run_agent') as mock_run_agent:
        mock_run_agent.return_value = _make_agent_result(
            status="success",
            answer="To reset your password, visit account settings.",
            tool_calls=[
                {
                    "call_id": "call_1",
                    "tool": "search_support_docs",
                    "arguments": {"query": "password reset"},
                    "result": {"tool": "search_support_docs", "results": [], "count": 0},
                }
            ],
        )
        response = client.post(
            "/api/v1/tickets/agent/run",
            json={"message": "How do I reset my password?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["answer"] == "To reset your password, visit account settings."
        assert data["mock_mode"] is False
        assert len(data["tool_calls"]) == 1
        TicketResolveResponse(**data)


def test_resolve_endpoint_validates_response_schema():
    """The response conforms to TicketResolveResponse."""
    with patch('app.modules.agent.route.run_agent') as mock_run_agent:
        mock_run_agent.return_value = _make_agent_result(
            status="success",
            answer="Final answer.",
        )
        response = client.post(
            "/api/v1/tickets/agent/run",
            json={"message": "Hello."},
        )
        assert response.status_code == 200
        TicketResolveResponse(**response.json())


# ---------------------------------------------------------------------------
# Mock mode when no API key is configured
# ---------------------------------------------------------------------------


def test_resolve_endpoint_mock_mode():
    """When no API key is configured, the endpoint reports mock_mode=True."""
    with patch('app.modules.agent.route.groq_client', None):
        response = client.post(
            "/api/v1/tickets/agent/run",
            json={"message": "How do I reset my password?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "mock"
        assert data["mock_mode"] is True
        assert data["answer"] is not None
        assert "API key" in data["answer"]
        TicketResolveResponse(**data)


# ---------------------------------------------------------------------------
# Request validation failures
# ---------------------------------------------------------------------------


def test_resolve_endpoint_empty_message():
    """An empty message returns 422."""
    response = client.post(
        "/api/v1/tickets/agent/run",
        json={"message": ""},
    )
    assert response.status_code == 422


def test_resolve_endpoint_whitespace_message():
    """A whitespace-only message returns 422."""
    response = client.post(
        "/api/v1/tickets/agent/run",
        json={"message": "   "},
    )
    assert response.status_code == 422


def test_resolve_endpoint_missing_message():
    """A missing message field returns 422."""
    response = client.post(
        "/api/v1/tickets/agent/run",
        json={},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Safe handling of provider errors
# ---------------------------------------------------------------------------


def test_resolve_endpoint_provider_failure_returns_502():
    """A provider failure returns 502 with a safe message."""
    with patch('app.modules.agent.route.run_agent') as mock_run_agent:
        mock_run_agent.return_value = _make_agent_result(
            status="provider_failure",
            error="Provider service is temporarily unavailable.",
        )
        response = client.post(
            "/api/v1/tickets/agent/run",
            json={"message": "Hello."},
        )
        assert response.status_code == 502
        detail = response.json().get("detail", "")
        assert "temporarily unavailable" in detail
        assert "api_key" not in detail.lower()


# ---------------------------------------------------------------------------
# Iteration-limit outcome
# ---------------------------------------------------------------------------


def test_resolve_endpoint_max_iterations_reached():
    """Hitting the iteration limit returns a structured non-success result."""
    with patch('app.modules.agent.route.run_agent') as mock_run_agent:
        mock_run_agent.return_value = _make_agent_result(
            status="max_iterations_reached",
            answer=None,
            tool_calls=[],
            error="Agent reached the maximum of 10 iterations without producing a final answer.",
        )
        response = client.post(
            "/api/v1/tickets/agent/run",
            json={"message": "Keep searching forever."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "max_iterations_reached"
        assert data["error"] is not None
        TicketResolveResponse(**data)


# ---------------------------------------------------------------------------
# Existing /analyze endpoint continues to work
# ---------------------------------------------------------------------------


def test_analyze_endpoint_still_works():
    """The existing /analyze endpoint is unaffected by the new resolve route."""
    with patch('app.modules.analyze.route.groq_client', None):
        response = client.post(
            "/api/v1/tickets/analyze",
            json={"message": "Hello, I need help."},
        )
        assert response.status_code == 200
        json_data = response.json()
        assert "category" in json_data


def test_health_endpoint_still_works():
    """The health endpoint is unaffected."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
