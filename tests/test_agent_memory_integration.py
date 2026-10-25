"""Tests for the memory context in an agent run.

These tests drive the agent endpoint. The provider is replaced by a fake
client, so no request leaves the machine. The database-backed tests use the
isolated PostgreSQL test database.
"""

import json
import logging
import os
from contextlib import contextmanager
from unittest.mock import patch

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.modules.agent.service import StorageError
from app.database import (
    MemoryDraft,
    append_memories,
    create_session,
    list_messages,
)
from main import app


#: Every test in this module writes to the database.
pytestmark = pytest.mark.usefixtures("clean_database")

#: The client of the agent endpoint.
api_client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, output=None, output_text=""):
        self.output = output or []
        self.output_text = output_text


class _FunctionCall:
    """A function-call output item as the provider would return it."""

    def __init__(self, call_id, name, arguments):
        self.type = "function_call"
        self.id = f"fc_{call_id}"
        self.call_id = call_id
        self.name = name
        self.arguments = arguments

    def model_dump(self):
        return {
            "id": self.id,
            "type": self.type,
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
        }


class _FakeResponses:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        # The agent reuses one mutable list for the request input. Snapshot
        # it, so each recorded request reflects the input as it was when the
        # request was made.
        recorded = dict(kwargs)
        if "input" in recorded:
            recorded["input"] = list(recorded["input"])
        self.requests.append(recorded)
        if not self._responses:
            raise AssertionError("responses.create was called too often")
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.responses = _FakeResponses(responses)


class _FailingClient:
    """A provider that fails, the way an outage would."""

    def __init__(self):
        self.responses = self

    def create(self, **kwargs):
        raise RuntimeError("the provider is down")


class _RecordingSearch:
    """A stand-in for the memory search that records its calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, db, session_id, query, **kwargs):
        self.calls.append({"session_id": session_id, "query": query, **kwargs})
        return []


class _RecordingStore:
    """A stand-in for the memory store that records its calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, db, session_id, result, source_msg_id=None, **kwargs):
        self.calls.append(
            {"session_id": session_id, "source_msg_id": source_msg_id}
        )
        return True


def _settings():
    from app.core.config import Settings

    return Settings(
        _env_file=None,
        groq_api_key="test_key",
        groq_model="openai/gpt-oss-20b",
        groq_base_url="https://api.groq.com/openai/v1",
        database_url=os.environ["DATABASE_URL"],
    )


@contextmanager
def _route(settings, groq_client=None):
    """Point the agent endpoint at a settings object and a provider."""
    with patch("app.modules.agent.route.settings", settings):
        with patch("app.modules.agent.route.groq_client", groq_client):
            yield


def _post(message, session_id=None):
    return api_client.post(
        "/api/v1/tickets/agent/run",
        json={"message": message, "session_id": session_id},
    )


def _system_items(items):
    """Return the items of the role system."""
    return [item for item in items if item.get("role") == "system"]


def _store_memory(database, session_id, content="Refunds take five days."):
    """Store a memory in a session and return the session id."""
    append_memories(
        database,
        session_id,
        [MemoryDraft(kind="fact", content=content)],
    )


def _memories(database, session_id):
    """Return the stored memories of a session, oldest first."""
    with database.connection() as conn:
        rows = conn.execute(
            "SELECT id, session_id, kind, content, source_msg_id"
            " FROM memories WHERE session_id = %s ORDER BY id",
            (session_id,),
        ).fetchall()
    return [
        {**row, "session_id": str(row["session_id"])} for row in rows
    ]


def _stored_answer(database, session_id):
    """Return the stored final answer of a session."""
    answers = [
        message
        for message in list_messages(database, session_id)
        if message.role == "assistant" and message.tool_name is None
    ]
    return answers[-1]


# ---------------------------------------------------------------------------
# The memory context in the provider input
# ---------------------------------------------------------------------------


def test_relevant_memories_are_included_exactly_once(database):
    """The recalled memory is carried by exactly one item."""
    session = create_session(database)
    _store_memory(database, session.id)

    fake_client = _FakeClient([_FakeResponse(output_text="The answer.")])

    with _route(_settings(), fake_client):
        response = _post("I need a refund", session_id=session.id)

    assert response.status_code == 200

    request_input = fake_client.responses.requests[0]["input"]
    system_items = _system_items(request_input)
    assert len(system_items) == 1
    assert system_items[0]["type"] == "message"
    assert system_items[0]["id"] == "mem_context_1"
    assert "Refunds take five days." in system_items[0]["content"][0]["text"]
    assert sum(
        1
        for item in request_input
        if item.get("type") == "message" and item.get("role") == "system"
    ) == 1


def test_the_current_user_message_appears_once_and_stays_last(database):
    """The message of the current run is sent once, at the end."""
    session = create_session(database)
    _store_memory(database, session.id)

    fake_client = _FakeClient([_FakeResponse(output_text="The answer.")])

    with _route(_settings(), fake_client):
        _post("I need a refund", session_id=session.id)

    request_input = fake_client.responses.requests[0]["input"]
    user_texts = [
        item["content"][0]["text"]
        for item in request_input
        if item.get("type") == "message" and item.get("role") == "user"
    ]

    assert user_texts == ["I need a refund"]
    assert request_input[-1]["role"] == "user"
    assert request_input[-1]["content"][0]["text"] == "I need a refund"


def test_memory_is_placed_before_the_conversation_history(database):
    """The recalled memory comes first, the conversation follows."""
    session = create_session(database)
    _store_memory(database, session.id)

    fake_client = _FakeClient(
        [
            _FakeResponse(output_text="First answer."),
            _FakeResponse(output_text="Second answer."),
        ]
    )

    with _route(_settings(), fake_client):
        first = _post("A question about a refund", session_id=session.id)
        second = _post("Another question about a refund", session_id=session.id)

    assert second.status_code == 200

    request_input = fake_client.responses.requests[1]["input"]
    assert request_input[0]["role"] == "system"
    assert request_input[0]["id"] == "mem_context_1"

    conversation = [
        item["content"][0]["text"]
        for item in request_input
        if item.get("type") == "message"
        and item.get("role") in ("user", "assistant")
    ]
    assert conversation == [
        "A question about a refund",
        "First answer.",
        "Another question about a refund",
    ]
    assert "Refunds take five days." in request_input[0]["content"][0]["text"]


def test_without_memories_the_provider_input_is_unchanged(database):
    """A run without memories sends exactly what it sent before."""
    session = create_session(database)

    fake_client = _FakeClient([_FakeResponse(output_text="The answer.")])

    with _route(_settings(), fake_client):
        response = _post("I need a refund", session_id=session.id)

    assert response.status_code == 200

    request_input = fake_client.responses.requests[0]["input"]
    assert len(request_input) == 1
    assert request_input[0]["type"] == "message"
    assert request_input[0]["role"] == "user"
    assert request_input[0]["id"] == "input_1"
    assert request_input[0]["content"][0]["text"] == "I need a refund"
    assert _system_items(request_input) == []


def test_memories_of_another_session_are_never_included(database):
    """A session never sees the memories of another session."""
    first = create_session(database)
    second = create_session(database)
    _store_memory(database, first.id)

    fake_client = _FakeClient([_FakeResponse(output_text="The answer.")])

    with _route(_settings(), fake_client):
        response = _post("I need a refund", session_id=second.id)

    assert response.status_code == 200

    request_input = fake_client.responses.requests[0]["input"]
    assert _system_items(request_input) == []
    assert "Refunds take five days." not in json.dumps(request_input)


def test_memories_are_recalled_only_after_the_session_is_resolved(
    database, monkeypatch
):
    """An unknown session is refused before any memory is searched."""
    recorder = _RecordingSearch()
    monkeypatch.setattr(
        "app.modules.agent.memory.search_memories", recorder
    )

    fake_client = _FakeClient([_FakeResponse(output_text="The answer.")])

    with _route(_settings(), fake_client):
        response = _post(
            "I need a refund",
            session_id="11111111-1111-4111-8111-111111111111",
        )

    assert response.status_code == 404
    assert recorder.calls == []


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_a_storage_failure_during_the_recall_is_logged_and_the_run_continues(
    database, caplog
):
    """A failing recall costs the customer no answer."""
    session = create_session(database)

    fake_client = _FakeClient([_FakeResponse(output_text="The answer.")])

    with caplog.at_level(logging.ERROR, logger="agent.service"):
        with patch(
            "app.modules.agent.memory.search_memories",
            side_effect=psycopg.OperationalError("the store is down"),
        ):
            with _route(_settings(), fake_client):
                response = _post("I need a refund", session_id=session.id)

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["answer"] == "The answer."

    # The run continued without a memory context.
    request_input = fake_client.responses.requests[0]["input"]
    assert _system_items(request_input) == []
    assert len(request_input) == 1

    # The failure was reported, not hidden.
    assert any(
        record.levelno == logging.ERROR
        and "Failed to recall the memories" in record.getMessage()
        for record in caplog.records
    )


def test_a_history_failure_still_answers_service_unavailable(database):
    """The established behaviour of the history read is unchanged."""
    session = create_session(database)

    fake_client = _FakeClient([_FakeResponse(output_text="The answer.")])

    with patch(
        "app.modules.agent.route.load_session_history",
        side_effect=StorageError("The session store is unavailable."),
    ):
        with _route(_settings(), fake_client):
            response = _post("I need a refund", session_id=session.id)

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Session storage is temporarily unavailable."
    )
    assert fake_client.responses.requests == []


# ---------------------------------------------------------------------------
# The rest of the endpoint
# ---------------------------------------------------------------------------


def test_mock_mode_remains_functional_and_makes_no_provider_call(database):
    """A run without a provider key answers from the mock service."""
    session = create_session(database)

    fake_client = _FakeClient([_FakeResponse(output_text="unused.")])

    with _route(_settings(), None):
        response = _post("I need a refund", session_id=session.id)

    assert response.status_code == 200
    assert response.json()["status"] == "mock"
    assert response.json()["mock_mode"] is True
    assert fake_client.responses.requests == []


def test_memory_and_tool_call_items_remain_distinct(database):
    """The memory context does not disturb the tool-call pairing."""
    session = create_session(database)
    _store_memory(database, session.id)

    fake_client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _FunctionCall(
                        "call_1",
                        "search_support_docs",
                        '{"query": "refund"}',
                    )
                ]
            ),
            _FakeResponse(output_text="The answer."),
        ]
    )

    with _route(_settings(), fake_client):
        response = _post("I need a refund", session_id=session.id)

    assert response.status_code == 200

    follow_up = fake_client.responses.requests[1]["input"]
    calls = [item for item in follow_up if item.get("type") == "function_call"]
    outputs = [
        item for item in follow_up if item.get("type") == "function_call_output"
    ]

    assert len(calls) == 1
    assert len(outputs) == 1
    assert calls[0]["call_id"] == outputs[0]["call_id"] == "call_1"
    assert calls[0]["name"] == outputs[0]["name"] == "search_support_docs"

    # The memory is a message item and nothing else.
    assert len(_system_items(follow_up)) == 1
    assert follow_up[0]["role"] == "system"


def test_the_response_contract_is_unchanged(database):
    """The body, the headers, and the stored transcript stay as they were."""
    session = create_session(database)
    _store_memory(database, session.id)

    fake_client = _FakeClient([_FakeResponse(output_text="The answer.")])

    with _route(_settings(), fake_client):
        response = _post("I need a refund", session_id=session.id)

    assert response.status_code == 200
    assert set(response.json()) == {
        "status",
        "answer",
        "mock_mode",
        "tool_calls",
        "error",
        "session_id",
    }
    assert response.headers["X-Session-Id"] == session.id
    assert response.headers["X-Transcript-Stored"] == "true"

    # The memory is not part of the stored transcript.
    messages = list_messages(database, session.id)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert [message.content for message in messages] == [
        "I need a refund",
        "The answer.",
    ]


# ---------------------------------------------------------------------------
# Writing the memory of a run
# ---------------------------------------------------------------------------


def test_a_successful_run_stores_one_summary_memory(database):
    """The answer of the agent is stored as a memory of the session."""
    session = create_session(database)

    fake_client = _FakeClient(
        [_FakeResponse(output_text="Refunds are issued within five days.")]
    )

    with _route(_settings(), fake_client):
        response = _post("I need a refund", session_id=session.id)

    assert response.status_code == 200

    memories = _memories(database, session.id)
    assert len(memories) == 1
    assert memories[0]["kind"] == "summary"
    assert memories[0]["content"] == "Refunds are issued within five days."
    assert memories[0]["session_id"] == session.id


def test_the_memory_points_at_the_stored_answer(database):
    """The memory carries the id of the stored answer of the run."""
    session = create_session(database)

    fake_client = _FakeClient(
        [_FakeResponse(output_text="Refunds are issued within five days.")]
    )

    with _route(_settings(), fake_client):
        _post("I need a refund", session_id=session.id)

    answer = _stored_answer(database, session.id)
    memories = _memories(database, session.id)

    assert answer is not None
    assert len(memories) == 1
    assert memories[0]["source_msg_id"] == answer.id


def test_the_memory_points_at_the_answer_and_not_at_a_tool_call(database):
    """A run that called a tool still points at its answer."""
    session = create_session(database)

    fake_client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _FunctionCall(
                        "call_1",
                        "search_support_docs",
                        '{"query": "refund"}',
                    )
                ]
            ),
            _FakeResponse(output_text="Refunds are issued within five days."),
        ]
    )

    with _route(_settings(), fake_client):
        _post("I need a refund", session_id=session.id)

    answer = _stored_answer(database, session.id)
    memories = _memories(database, session.id)

    assert answer.content == "Refunds are issued within five days."
    assert memories[0]["source_msg_id"] == answer.id


def test_a_repeated_answer_is_stored_once(database):
    """The same answer is not stored a second time."""
    session = create_session(database)

    fake_client = _FakeClient(
        [
            _FakeResponse(output_text="Refunds are issued within five days."),
            _FakeResponse(output_text="Refunds are issued within five days."),
        ]
    )

    with _route(_settings(), fake_client):
        _post("I need a refund", session_id=session.id)
        second = _post("I need a refund again", session_id=session.id)

    assert second.status_code == 200
    assert len(_memories(database, session.id)) == 1


def test_different_answers_are_stored_in_the_same_session(database):
    """A session can hold more than one memory."""
    session = create_session(database)

    fake_client = _FakeClient(
        [
            _FakeResponse(output_text="Refunds are issued within five days."),
            _FakeResponse(output_text="Refunds reach the payment method."),
        ]
    )

    with _route(_settings(), fake_client):
        _post("I need a refund", session_id=session.id)
        second = _post("And the other charge?", session_id=session.id)

    assert second.status_code == 200
    assert len(_memories(database, session.id)) == 2


def test_the_same_answer_in_another_session_stays_isolated(database):
    """A memory belongs to one session only."""
    first = create_session(database)
    second = create_session(database)

    fake_client = _FakeClient(
        [
            _FakeResponse(output_text="Refunds are issued within five days."),
            _FakeResponse(output_text="Refunds are issued within five days."),
        ]
    )

    with _route(_settings(), fake_client):
        _post("I need a refund", session_id=first.id)
        _post("I need a refund", session_id=second.id)

    first_memories = _memories(database, first.id)
    second_memories = _memories(database, second.id)

    assert len(first_memories) == 1
    assert len(second_memories) == 1
    assert first_memories[0]["id"] != second_memories[0]["id"]
    assert first_memories[0]["session_id"] == first.id
    assert second_memories[0]["session_id"] == second.id


def test_a_mock_run_stores_no_memory(database):
    """The mock answer is not a memory of the session."""
    session = create_session(database)

    with _route(_settings(), None):
        response = _post("I need a refund", session_id=session.id)

    assert response.status_code == 200
    assert response.json()["status"] == "mock"
    assert _memories(database, session.id) == []


def test_a_provider_failure_stores_no_memory(database):
    """A failed run keeps its behaviour and stores nothing."""
    session = create_session(database)

    with _route(_settings(), _FailingClient()):
        response = _post("I need a refund", session_id=session.id)

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Provider service is temporarily unavailable."
    )
    assert response.headers["X-Transcript-Stored"] == "true"
    assert _memories(database, session.id) == []


def test_a_failed_memory_write_leaves_the_answer_untouched(database, caplog):
    """A memory that cannot be stored costs the customer no answer."""
    session = create_session(database)

    fake_client = _FakeClient(
        [_FakeResponse(output_text="Refunds are issued within five days.")]
    )

    with caplog.at_level(logging.ERROR, logger="agent.service"):
        with patch(
            "app.modules.agent.service.append_memories",
            side_effect=psycopg.OperationalError("the store is down"),
        ):
            with _route(_settings(), fake_client):
                response = _post("I need a refund", session_id=session.id)

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["answer"] == "Refunds are issued within five days."
    assert set(response.json()) == {
        "status",
        "answer",
        "mock_mode",
        "tool_calls",
        "error",
        "session_id",
    }
    assert response.headers["X-Session-Id"] == session.id
    assert response.headers["X-Transcript-Stored"] == "true"

    # The transcript was stored, and the failure was reported.
    assert [message.role for message in list_messages(database, session.id)] == [
        "user",
        "assistant",
    ]
    assert any(
        record.levelno == logging.ERROR
        and "Failed to store the memory" in record.getMessage()
        for record in caplog.records
    )


def test_a_failed_transcript_stops_the_memory_from_being_written(database):
    """Without a stored answer there is no memory to point at."""
    session = create_session(database)

    fake_client = _FakeClient([_FakeResponse(output_text="The answer.")])
    recorder = _RecordingStore()

    with patch(
        "app.modules.agent.service.append_messages",
        side_effect=RuntimeError("storage is down"),
    ):
        with patch("app.modules.agent.route.store_run_memory", recorder):
            with _route(_settings(), fake_client):
                response = _post("I need a refund", session_id=session.id)

    assert response.status_code == 200
    assert response.json()["answer"] == "The answer."
    assert response.headers["X-Transcript-Stored"] == "false"
    assert recorder.calls == []
    assert _memories(database, session.id) == []


def test_the_written_memory_is_recalled_by_a_later_request(database):
    """A later run of the session sees what the earlier run learned."""
    session = create_session(database)

    fake_client = _FakeClient(
        [
            _FakeResponse(output_text="Refunds are issued within five days."),
            _FakeResponse(output_text="Anything else?"),
        ]
    )

    with _route(_settings(), fake_client):
        first = _post("Tell me about refunds", session_id=session.id)
        second = _post("I need a refund", session_id=session.id)

    assert first.status_code == 200
    assert second.status_code == 200

    request_input = fake_client.responses.requests[1]["input"]
    system_items = _system_items(request_input)

    assert len(system_items) == 1
    assert system_items[0]["id"] == "mem_context_1"
    assert (
        "Refunds are issued within five days."
        in system_items[0]["content"][0]["text"]
    )
