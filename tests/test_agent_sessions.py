"""Tests for session handling and transcript persistence on the agent endpoint.

The provider is replaced with a fake client, so the agent loop runs without
network access. Every test writes to the dedicated PostgreSQL test database,
which is emptied before the test starts.
"""

import json
import os
import uuid
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.agent_core.agent import AgentResult
from app.core.config import Settings
from app.database import (
    MessageDraft,
    append_messages,
    create_session,
    get_session,
    list_messages,
)
from main import app
from app.database import get_database
from app.modules.integrations.service import issue_integration_credential


def _issue_integration_auth_header(tenant_id=None):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(db, integration_name=f"test-agent-integ-{uuid.uuid4().hex[:8]}")
    tid = tenant_id or str(uuid.uuid4())
    from app.database.repository import _canonical_uuid
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO tenants (id, name, enabled) VALUES (%s, %s, TRUE) ON CONFLICT (id) DO NOTHING",
            (_canonical_uuid(tid), f"agent-tenant-{tid[:8]}"),
        )
        conn.execute(
            "UPDATE integrations SET tenant_id = %s WHERE id = %s",
            (_canonical_uuid(tid), _canonical_uuid(integration.id)),
        )
    return {"Authorization": f"Bearer {token}"}, tid


# conftest points the whole application at the test database before any
# application module is imported.
TEST_DATABASE_URL = os.environ["DATABASE_URL"]

client = TestClient(app)

#: Every test in this module writes to the database.
pytestmark = pytest.mark.usefixtures("clean_database")


# ---------------------------------------------------------------------------
# Fakes and fixtures
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, output=None, output_text=""):
        self.output = output or []
        self.output_text = output_text


class _FakeResponses:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("responses.create was called too often")
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.responses = _FakeResponses(responses)


def _settings():
    return Settings(
        _env_file=None,
        groq_api_key="test_key",
        groq_model="openai/gpt-oss-20b",
        groq_base_url="https://api.groq.com/openai/v1",
        database_url=TEST_DATABASE_URL,
    )


@contextmanager
def _route(settings, groq_client=None):
    headers, tenant_id = _issue_integration_auth_header()
    with patch("app.modules.agent.service.settings", settings):
        with patch("app.modules.agent.service.groq_client", groq_client):
            yield headers, tenant_id


def _agent_result_with_tool_calls():
    return AgentResult(
        status="success",
        answer="Refunds are issued within 5 business days.",
        tool_calls=[
            {
                "call_id": "call_1",
                "tool": "search_support_docs",
                "arguments": {"query": "billing refund"},
                "result": {
                    "tool": "search_support_docs",
                    "query": "billing refund",
                    "results": [{"id": "doc-002", "title": "Billing refunds"}],
                    "count": 1,
                },
            },
            {
                "call_id": "call_2",
                "tool": "lookup_ticket_history",
                "arguments": {"query": "billing refund"},
                "result": {
                    "tool": "lookup_ticket_history",
                    "query": "billing refund",
                    "results": [{"ticket_id": "TKT-1002"}],
                    "count": 1,
                },
            },
        ],
    )


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------


def test_run_without_a_session_id_still_works(database):
    """A message-only request is unchanged and returns a session id."""
    settings = _settings()
    with _route(settings) as (headers, tenant_id):
        with patch("app.modules.agent.service.run_agent") as mock_run:
            mock_run.return_value = AgentResult(
                status="success", answer="Done.", tool_calls=[]
            )
            response = client.post(
                "/api/v1/tickets/agent/run",
                json={"message": "I need help."},
                headers=headers,
            )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["answer"] == "Done."
    assert data["session_id"]
    session = get_session(database, data["session_id"])
    assert session is not None
    assert str(session.tenant_id) == str(tenant_id)


def test_run_creates_a_session_and_stores_the_transcript(database):
    """The run creates a session and stores the whole transcript."""
    settings = _settings()
    with _route(settings) as (headers, tenant_id):
        with patch("app.modules.agent.service.run_agent") as mock_run:
            mock_run.return_value = _agent_result_with_tool_calls()
            response = client.post(
                "/api/v1/tickets/agent/run",
                json={"message": "I was charged twice."},
                headers=headers,
            )

    assert response.status_code == 200
    session_id = response.json()["session_id"]
    assert uuid.UUID(session_id)

    session = get_session(database, session_id)
    assert session is not None
    assert session.status == "open"
    assert str(session.tenant_id) == str(tenant_id)

    messages = list_messages(database, session_id)
    assert [message.seq for message in messages] == [1, 2, 3, 4, 5, 6]
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[0].content == "I was charged twice."
    assert messages[-1].content == "Refunds are issued within 5 business days."
    assert messages[1].tool_name == "search_support_docs"
    assert messages[1].call_id == "call_1"
    assert json.loads(messages[1].content) == {
        "tool": "search_support_docs",
        "arguments": {"query": "billing refund"},
    }
    assert messages[2].tool_name == "search_support_docs"
    assert json.loads(messages[2].content)["results"][0]["id"] == "doc-002"
    assert messages[3].tool_name == "lookup_ticket_history"
    assert messages[3].call_id == "call_2"
    assert json.loads(messages[4].content)["results"][0]["ticket_id"] == "TKT-1002"


def test_each_run_without_a_session_id_gets_its_own_session(database):
    """Two independent runs do not share a session."""
    settings = _settings()
    with _route(settings) as (headers, tenant_id):
        with patch("app.modules.agent.service.run_agent") as mock_run:
            mock_run.return_value = AgentResult(
                status="success", answer="Done.", tool_calls=[]
            )
            first = client.post(
                "/api/v1/tickets/agent/run", json={"message": "First."}, headers=headers
            )
            second = client.post(
                "/api/v1/tickets/agent/run", json={"message": "Second."}, headers=headers
            )
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["session_id"] != second.json()["session_id"]



# ---------------------------------------------------------------------------
# Session reuse
# ---------------------------------------------------------------------------


def test_existing_session_id_is_reused(database):
    """A known session id is associated with the new run."""
    existing = create_session(database)

    settings = _settings()
    with _route(settings) as (headers, tenant_id):
        from app.database.repository import _canonical_uuid
        with database.write_transaction() as conn:
            conn.execute("UPDATE sessions SET tenant_id = %s WHERE id = %s", (_canonical_uuid(tenant_id), _canonical_uuid(existing.id)))
        with patch("app.modules.agent.service.run_agent") as mock_run:
            mock_run.return_value = AgentResult(
                status="success", answer="Done.", tool_calls=[]
            )
            response = client.post(
                "/api/v1/tickets/agent/run",
                json={"message": "Follow-up question.", "session_id": existing.id},
                headers=headers,
            )

    assert response.status_code == 200
    assert response.json()["session_id"] == existing.id

    messages = list_messages(database, existing.id)
    assert [message.content for message in messages] == [
        "Follow-up question.",
        "Done.",
    ]


def test_transcript_accumulates_across_runs_in_one_session(database):
    """A second run continues the transcript of the same session."""
    existing = create_session(database)
    from app.database.repository import _canonical_uuid

    settings = _settings()
    with _route(settings) as (headers, tenant_id):
        with database.write_transaction() as conn:
            conn.execute("UPDATE sessions SET tenant_id = %s WHERE id = %s", (_canonical_uuid(tenant_id), _canonical_uuid(existing.id)))
        with patch("app.modules.agent.service.run_agent") as mock_run:
            mock_run.return_value = AgentResult(
                status="success", answer="First answer.", tool_calls=[]
            )
            client.post(
                "/api/v1/tickets/agent/run",
                json={"message": "First question.", "session_id": existing.id},
                headers=headers,
            )
            mock_run.return_value = AgentResult(
                status="success", answer="Second answer.", tool_calls=[]
            )
            client.post(
                "/api/v1/tickets/agent/run",
                json={"message": "Second question.", "session_id": existing.id},
                headers=headers,
            )

    messages = list_messages(database, existing.id)
    assert [message.seq for message in messages] == [1, 2, 3, 4]
    assert [message.content for message in messages] == [
        "First question.",
        "First answer.",
        "Second question.",
        "Second answer.",
    ]


def test_two_sessions_do_not_share_messages(database):
    """Messages written for one session are invisible to another."""
    first = create_session(database)
    second = create_session(database)
    from app.database.repository import _canonical_uuid

    settings = _settings()
    with _route(settings) as (headers, tenant_id):
        with database.write_transaction() as conn:
            conn.execute("UPDATE sessions SET tenant_id = %s WHERE id = %s", (_canonical_uuid(tenant_id), _canonical_uuid(first.id)))
            conn.execute("UPDATE sessions SET tenant_id = %s WHERE id = %s", (_canonical_uuid(tenant_id), _canonical_uuid(second.id)))
        with patch("app.modules.agent.service.run_agent") as mock_run:
            mock_run.return_value = AgentResult(
                status="success", answer="Answer.", tool_calls=[]
            )
            client.post(
                "/api/v1/tickets/agent/run",
                json={"message": "For the first session.", "session_id": first.id},
                headers=headers,
            )
            client.post(
                "/api/v1/tickets/agent/run",
                json={"message": "For the second session.", "session_id": second.id},
                headers=headers,
            )

    assert [m.content for m in list_messages(database, first.id)] == [
        "For the first session.",
        "Answer.",
    ]
    assert [m.content for m in list_messages(database, second.id)] == [
        "For the second session.",
        "Answer.",
    ]


# ---------------------------------------------------------------------------
# Invalid session ids
# ---------------------------------------------------------------------------


def test_unknown_session_id_returns_404(database):
    """An unknown session id is rejected instead of being trusted."""
    settings = _settings()
    with _route(settings) as (headers, tenant_id):
        with patch("app.modules.agent.service.run_agent") as mock_run:
            response = client.post(
                "/api/v1/tickets/agent/run",
                json={
                    "message": "Hello.",
                    "session_id": str(uuid.uuid4()),
                },
                headers=headers,
            )

    assert response.status_code == 404
    assert mock_run.call_count == 0
    assert "session" in response.json()["detail"].lower()
    # The agent was not run for a rejected session.
    assert mock_run.call_count == 0


def test_malformed_session_id_returns_422(database):
    """A session id that is not an identifier is rejected."""
    settings = _settings()
    with _route(settings) as (headers, tenant_id):
        with patch("app.modules.agent.service.run_agent") as mock_run:
            response = client.post(
                "/api/v1/tickets/agent/run",
                json={"message": "Hello.", "session_id": "not-a-session-id"},
                headers=headers,
            )

    assert response.status_code == 422
    assert mock_run.call_count == 0


def test_blank_session_id_returns_422(database):
    """A blank session id is rejected."""
    settings = _settings()
    with _route(settings) as (headers, tenant_id):
        response = client.post(
            "/api/v1/tickets/agent/run",
            json={"message": "Hello.", "session_id": "   "},
            headers=headers,
        )

    assert response.status_code == 422


def test_rejected_session_id_creates_no_session(database):
    """A rejected session id does not leave a session behind."""
    settings = _settings()
    with _route(settings) as (headers, tenant_id):
        with patch("app.modules.agent.service.run_agent"):
            client.post(
                "/api/v1/tickets/agent/run",
                json={"message": "Hello.", "session_id": str(uuid.uuid4())},
                headers=headers,
            )

    with database.connection() as conn:
        rows = conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()
    assert rows["n"] == 0


# ---------------------------------------------------------------------------
# Stored history is replayed into a continuing run
# ---------------------------------------------------------------------------


def test_stored_history_is_replayed_into_the_model(database):
    """A continuation run receives the earlier turns of its session."""
    existing = create_session(database)
    from app.database.repository import _canonical_uuid
    append_messages(
        database,
        existing.id,
        [
            MessageDraft(role="user", content="An older question."),
            MessageDraft(role="assistant", content="An older answer."),
        ],
    )

    fake_client = _FakeClient([_FakeResponse(output_text="The final answer.")])

    settings = _settings()
    with _route(settings, fake_client) as (headers, tenant_id):
        with database.write_transaction() as conn:
            conn.execute("UPDATE sessions SET tenant_id = %s WHERE id = %s", (_canonical_uuid(tenant_id), _canonical_uuid(existing.id)))
        response = client.post(
            "/api/v1/tickets/agent/run",
            json={"message": "A new question.", "session_id": existing.id},
            headers=headers,
        )

    assert response.status_code == 200
    assert response.json()["status"] == "success"

    request = fake_client.responses.requests[0]
    user_texts = [
        item["content"][0]["text"]
        for item in request["input"]
        if item.get("type") == "message" and item.get("role") == "user"
    ]
    assistant_texts = [
        item["content"][0]["text"]
        for item in request["input"]
        if item.get("type") == "message" and item.get("role") == "assistant"
    ]
    assert user_texts == ["An older question.", "A new question."]
    assert assistant_texts == ["An older answer."]


    # The message of the current run is sent exactly once.
    assert user_texts.count("A new question.") == 1

    # No tool output of an earlier run is replayed: the stored transcript has
    # no tool call in it.
    output_call_ids = {
        item["call_id"]
        for item in request["input"]
        if item.get("type") == "function_call_output"
    }
    assert output_call_ids == set()


# ---------------------------------------------------------------------------
# Existing behaviour is preserved
# ---------------------------------------------------------------------------


def test_mock_mode_still_works_and_records_the_session(database):
    """Mock mode is unchanged and still records a session."""
    settings = _settings()
    with _route(settings, None) as (headers, tenant_id):
        response = client.post(
            "/api/v1/tickets/agent/run", json={"message": "Help me."}, headers=headers
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "mock"
    assert data["mock_mode"] is True
    assert data["answer"]
    assert data["session_id"]
    session = get_session(database, data["session_id"])
    assert str(session.tenant_id) == str(tenant_id)

    messages = list_messages(database, data["session_id"])
    assert [m.role for m in messages] == ["user", "assistant"]


def test_provider_failure_still_returns_502(database):
    """The provider failure contract is unchanged."""
    settings = _settings()
    with _route(settings) as (headers, tenant_id):
        with patch("app.modules.agent.service.run_agent") as mock_run:
            mock_run.return_value = AgentResult(
                status="provider_failure",
                error="Provider service is temporarily unavailable.",
            )
            response = client.post(
                "/api/v1/tickets/agent/run", json={"message": "Help me."}, headers=headers
            )

    assert response.status_code == 502
    assert "temporarily unavailable" in response.json()["detail"]
    session_id = response.headers["X-Session-Id"]
    session = get_session(database, session_id)
    assert str(session.tenant_id) == str(tenant_id)


def test_max_iterations_reached_is_unchanged(database):
    """The iteration limit contract is unchanged."""
    settings = _settings()
    with _route(settings) as (headers, tenant_id):
        with patch("app.modules.agent.service.run_agent") as mock_run:
            mock_run.return_value = AgentResult(
                status="max_iterations_reached",
                answer=None,
                tool_calls=[],
                error="Agent reached the maximum of 10 iterations.",
            )
            response = client.post(
                "/api/v1/tickets/agent/run", json={"message": "Help me."}, headers=headers
            )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "max_iterations_reached"
    assert data["session_id"]
    session = get_session(database, data["session_id"])
    assert str(session.tenant_id) == str(tenant_id)


@pytest.mark.usefixtures("clean_database")
def test_agent_cross_tenant_session_denied(database):
    from app.database.repository import _canonical_uuid
    session = create_session(database)
    other_tenant = str(uuid.uuid4())
    with database.write_transaction() as conn:
        conn.execute("INSERT INTO tenants (id, name, enabled) VALUES (%s, %s, TRUE) ON CONFLICT (id) DO NOTHING", (_canonical_uuid(other_tenant), f"other-{other_tenant[:8]}"))
        conn.execute("UPDATE sessions SET tenant_id = %s WHERE id = %s", (_canonical_uuid(other_tenant), _canonical_uuid(session.id)))
    headers, _tid = _issue_integration_auth_header()
    settings = _settings()
    with patch("app.modules.agent.service.settings", settings), patch("app.modules.agent.service.groq_client", None):
        resp = client.post("/api/v1/tickets/agent/run", json={"message": "hi", "session_id": session.id}, headers=headers)
    assert resp.status_code == 404


@pytest.mark.usefixtures("clean_database")
def test_agent_null_tenant_session_denied(database):
    session = create_session(database)
    headers, _tid = _issue_integration_auth_header()
    settings = _settings()
    with patch("app.modules.agent.service.settings", settings), patch("app.modules.agent.service.groq_client", None):
        resp = client.post("/api/v1/tickets/agent/run", json={"message": "hi", "session_id": session.id}, headers=headers)
    assert resp.status_code == 404


@pytest.mark.usefixtures("clean_database")
def test_agent_forged_tenant_header_ignored(database):
    session = create_session(database)
    headers, tenant_id = _issue_integration_auth_header()
    from app.database.repository import _canonical_uuid
    with database.write_transaction() as conn:
        conn.execute("UPDATE sessions SET tenant_id = %s WHERE id = %s", (_canonical_uuid(tenant_id), _canonical_uuid(session.id)))
    settings = _settings()
    with patch("app.modules.agent.service.settings", settings), patch("app.modules.agent.service.groq_client", None):
        resp = client.post("/api/v1/tickets/agent/run", json={"message": "hi", "session_id": session.id}, headers={**headers, "X-Tenant-Id": str(uuid.uuid4())})
    assert resp.status_code == 200
