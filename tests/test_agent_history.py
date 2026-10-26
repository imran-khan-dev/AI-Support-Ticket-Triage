"""Tests for loading the prior turns of a session into the agent input.

These cover the guarantees the history loader has to keep:

* the transcript of a session is read in chronological order;
* the messages of another session are never loaded;
* an empty session yields no input items;
* the size limit keeps the most recent turns;
* a tool call and its output are always replayed together;
* the message of the current run is never sent twice.

The provider is replaced by a fake client, so the agent loop runs without
network access.
"""

import json
import os
from contextlib import contextmanager
from unittest.mock import patch

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.agent_core.agent import run_agent
from app.core.config import DEFAULT_MAX_HISTORY_MESSAGES, Settings
from app.domain.agent.history import load_history_items
from app.database import (
    MessageDraft,
    append_messages,
    create_session,
    list_messages,
)
from main import app


pytestmark = pytest.mark.usefixtures("clean_database")

#: The client of the agent endpoint.
api_client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_draft(role, content):
    return MessageDraft(role=role, content=content)


def _tool_call_drafts(call_id, tool, arguments, result):
    """Return the two drafts a tool call is stored as."""
    return [
        MessageDraft(
            role="assistant",
            content=json.dumps({"tool": tool, "arguments": arguments}),
            call_id=call_id,
            tool_name=tool,
        ),
        MessageDraft(
            role="tool",
            content=json.dumps(result),
            call_id=call_id,
            tool_name=tool,
        ),
    ]


def _texts(items):
    """Return the text of every message item of an input list."""
    return [
        item["content"][0]["text"]
        for item in items
        if item.get("type") == "message"
    ]


def _call_ids(items, item_type):
    return {
        item["call_id"] for item in items if item.get("type") == item_type
    }


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
        # the list, so each recorded request reflects the input as it was
        # when the request was made.
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


def _settings():
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


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_history_items_follow_the_transcript_order(database):
    """The stored turns are replayed oldest first."""
    session = create_session(database)
    append_messages(
        database,
        session.id,
        [
            _text_draft("user", "first question"),
            _text_draft("assistant", "first answer"),
            _text_draft("user", "second question"),
            _text_draft("assistant", "second answer"),
        ],
    )

    items = load_history_items(database, session.id)

    assert _texts(items) == [
        "first question",
        "first answer",
        "second question",
        "second answer",
    ]
    assert [item["role"] for item in items] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [item["type"] for item in items] == ["message"] * 4


def test_history_items_carry_unique_identifiers(database):
    """Every replayed item has its own identifier."""
    session = create_session(database)
    append_messages(
        database,
        session.id,
        [_text_draft("user", "one"), _text_draft("assistant", "two")],
    )

    items = load_history_items(database, session.id)

    identifiers = [item["id"] for item in items]
    assert all(identifiers)
    assert len(set(identifiers)) == len(identifiers)


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------


def test_history_contains_only_the_messages_of_its_own_session(database):
    """The messages of another session are never loaded."""
    first = create_session(database)
    second = create_session(database)
    append_messages(
        database, first.id, [_text_draft("user", "for the first session")]
    )
    append_messages(
        database, second.id, [_text_draft("user", "for the second session")]
    )

    items = load_history_items(database, first.id)

    assert _texts(items) == ["for the first session"]


def test_a_session_without_messages_has_no_history(database):
    """An empty session yields no input items."""
    session = create_session(database)

    assert load_history_items(database, session.id) == []


def test_an_unknown_session_has_no_history(database):
    """A session that does not exist yields no input items."""
    assert load_history_items(database, "does-not-exist") == []


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_the_history_is_truncated_to_the_configured_maximum(database):
    """The most recent turns are kept when the transcript is too long."""
    session = create_session(database)
    append_messages(
        database,
        session.id,
        [_text_draft("user", f"question {index}") for index in range(6)],
    )

    items = load_history_items(database, session.id, max_messages=3)

    assert _texts(items) == ["question 3", "question 4", "question 5"]


def test_a_limit_of_zero_disables_the_history(database):
    """No message is replayed when the limit is zero."""
    session = create_session(database)
    append_messages(
        database, session.id, [_text_draft("user", "an older question")]
    )

    assert load_history_items(database, session.id, max_messages=0) == []


def test_truncation_never_splits_a_tool_call_from_its_output(database):
    """A tool call and its output are replayed together or not at all."""
    session = create_session(database)
    append_messages(
        database,
        session.id,
        [
            _text_draft("user", "a question"),
            *_tool_call_drafts(
                "call_1",
                "search_support_docs",
                {"query": "billing"},
                {"tool": "search_support_docs", "results": [], "count": 0},
            ),
            *_tool_call_drafts(
                "call_2",
                "lookup_ticket_history",
                {"query": "billing"},
                {"tool": "lookup_ticket_history", "results": [], "count": 0},
            ),
            _text_draft("assistant", "the answer"),
        ],
    )

    for limit in range(1, 8):
        items = load_history_items(database, session.id, max_messages=limit)

        assert len(items) <= limit
        calls = _call_ids(items, "function_call")
        outputs = _call_ids(items, "function_call_output")
        assert calls == outputs, (
            f"a tool call was split at limit {limit}: {calls} vs {outputs}"
        )


def test_a_tool_call_is_replayed_in_the_responses_api_format(database):
    """A stored tool call is replayed as a call and an output item."""
    session = create_session(database)
    append_messages(
        database,
        session.id,
        _tool_call_drafts(
            "call_1",
            "search_support_docs",
            {"query": "password reset"},
            {"tool": "search_support_docs", "results": [], "count": 0},
        ),
    )

    items = load_history_items(database, session.id)

    call = next(i for i in items if i["type"] == "function_call")
    output = next(i for i in items if i["type"] == "function_call_output")

    # The replayed call and its output share one call id, and that id is of
    # its own so it cannot collide with a call of the current turn.
    assert call["call_id"] == output["call_id"]
    assert call["call_id"].startswith("hist_call_")

    assert call["name"] == "search_support_docs"
    assert json.loads(call["arguments"]) == {"query": "password reset"}

    assert output["name"] == "search_support_docs"
    assert json.loads(output["output"])["tool"] == "search_support_docs"


def test_the_error_record_of_a_failed_run_is_not_replayed(database):
    """The safe error message of a failed run is not part of the history."""
    session = create_session(database)
    append_messages(
        database,
        session.id,
        [
            _text_draft("user", "a question"),
            _text_draft("system", "Provider service is temporarily unavailable."),
        ],
    )

    items = load_history_items(database, session.id)

    assert _texts(items) == ["a question"]


def test_an_output_without_its_call_is_not_replayed(database):
    """A tool result whose call is missing is left out."""
    session = create_session(database)
    append_messages(
        database,
        session.id,
        [
            _text_draft("user", "a question"),
            MessageDraft(
                role="tool",
                content=json.dumps({"tool": "search_support_docs"}),
                call_id="call_orphan",
                tool_name="search_support_docs",
            ),
        ],
    )

    items = load_history_items(database, session.id)

    assert _call_ids(items, "function_call_output") == set()
    assert _texts(items) == ["a question"]


# ---------------------------------------------------------------------------
# The current message
# ---------------------------------------------------------------------------


def test_the_messages_of_the_current_run_can_be_excluded(database):
    """The rows of the current run are not part of the history."""
    session = create_session(database)
    append_messages(
        database,
        session.id,
        [_text_draft("user", "an older question")],
        correlation_id="11111111-1111-4111-8111-111111111111",
    )
    append_messages(
        database,
        session.id,
        [_text_draft("user", "the current question")],
        correlation_id="22222222-2222-4222-8222-222222222222",
    )

    items = load_history_items(
        database,
        session.id,
        exclude_correlation_ids={"22222222-2222-4222-8222-222222222222"},
    )

    assert _texts(items) == ["an older question"]


def test_the_agent_sends_the_current_message_exactly_once(database):
    """The history is placed in front of the current message."""
    session = create_session(database)
    append_messages(
        database,
        session.id,
        [
            _text_draft("user", "an older question"),
            _text_draft("assistant", "an older answer"),
        ],
    )
    history = load_history_items(database, session.id)

    client = _FakeClient(
        [_FakeResponse(output_text="the answer", output=[])]
    )

    result = run_agent(
        "the current question",
        _settings(),
        client,
        history=history,
    )

    assert result.status == "success"

    request_input = client.responses.requests[0]["input"]
    assert _texts(request_input) == [
        "an older question",
        "an older answer",
        "the current question",
    ]
    assert request_input[-1]["role"] == "user"
    assert request_input[-1]["content"][0]["text"] == "the current question"

    # The current message is sent once, the older turn stays a turn of its own.
    user_texts = [
        item["content"][0]["text"]
        for item in request_input
        if item.get("type") == "message" and item.get("role") == "user"
    ]
    assert user_texts.count("the current question") == 1


def test_the_agent_keeps_its_request_shape_without_history(database):
    """A run without history sends only the current message."""
    client = _FakeClient(
        [_FakeResponse(output_text="the answer", output=[])]
    )

    run_agent("the current question", _settings(), client)

    request_input = client.responses.requests[0]["input"]
    assert len(request_input) == 1
    assert request_input[0]["id"] == "input_1"
    assert request_input[0]["content"][0]["text"] == "the current question"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_the_maximum_history_size_is_configurable():
    """The limit is a setting with a documented default."""
    assert Settings(_env_file=None).agent_max_history_messages == (
        DEFAULT_MAX_HISTORY_MESSAGES
    )
    assert DEFAULT_MAX_HISTORY_MESSAGES > 0


def test_the_maximum_history_size_can_be_set_in_the_environment(monkeypatch):
    """The limit is read from the environment."""
    monkeypatch.setenv("AGENT_MAX_HISTORY_MESSAGES", "5")

    assert Settings(_env_file=None).agent_max_history_messages == 5


def test_the_stored_transcript_is_not_changed_by_loading_it(database):
    """Loading the history does not touch the stored transcript."""
    session = create_session(database)
    append_messages(
        database,
        session.id,
        [_text_draft("user", "a question"), _text_draft("assistant", "an answer")],
    )
    before = list_messages(database, session.id)

    load_history_items(database, session.id)

    assert list_messages(database, session.id) == before


# ---------------------------------------------------------------------------
# Endpoint integration
# ---------------------------------------------------------------------------


def _post(message, session_id=None):
    return api_client.post(
        "/api/v1/tickets/agent/run",
        json={"message": message, "session_id": session_id},
    )


def test_the_first_turn_is_persisted(database):
    """A first run stores its message and its answer."""
    fake_client = _FakeClient([_FakeResponse(output_text="The answer.")])

    with _route(_settings(), fake_client):
        response = _post("First question.")

    assert response.status_code == 200
    assert response.json()["status"] == "success"

    messages = list_messages(database, response.json()["session_id"])
    assert [message.role for message in messages] == ["user", "assistant"]
    assert [message.content for message in messages] == [
        "First question.",
        "The answer.",
    ]


def test_the_second_turn_receives_the_prior_history(database):
    """A continuing run receives the earlier turns of its session."""
    fake_client = _FakeClient(
        [
            _FakeResponse(output_text="First answer."),
            _FakeResponse(output_text="Second answer."),
        ]
    )

    with _route(_settings(), fake_client):
        first = _post("First question.")
        second = _post(
            "Second question.", session_id=first.json()["session_id"]
        )

    assert second.status_code == 200

    request_input = fake_client.responses.requests[1]["input"]
    assert _texts(request_input) == [
        "First question.",
        "First answer.",
        "Second question.",
    ]


def test_the_current_message_is_sent_exactly_once(database):
    """The message of the current run is not replayed from the history."""
    fake_client = _FakeClient(
        [
            _FakeResponse(output_text="First answer."),
            _FakeResponse(output_text="Second answer."),
        ]
    )

    with _route(_settings(), fake_client):
        first = _post("First question.")
        _post("Second question.", session_id=first.json()["session_id"])

    request_input = fake_client.responses.requests[1]["input"]
    user_texts = [
        item["content"][0]["text"]
        for item in request_input
        if item.get("type") == "message" and item.get("role") == "user"
    ]
    assert user_texts == ["First question.", "Second question."]

    # The identifiers of the request items are unique.
    identifiers = [item["id"] for item in request_input]
    assert len(set(identifiers)) == len(identifiers)


def test_a_tool_call_stays_paired_with_its_output_across_turns(database):
    """A tool call of an earlier turn is replayed as a valid pair."""
    fake_client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _FunctionCall(
                        "call_1",
                        "search_support_docs",
                        '{"query": "billing"}',
                    )
                ]
            ),
            _FakeResponse(output_text="First answer."),
            _FakeResponse(output_text="Second answer."),
        ]
    )

    with _route(_settings(), fake_client):
        first = _post("First question.")
        second = _post(
            "Second question.", session_id=first.json()["session_id"]
        )

    assert second.status_code == 200

    request_input = fake_client.responses.requests[2]["input"]
    calls = [
        item
        for item in request_input
        if item.get("type") == "function_call"
    ]
    outputs = [
        item
        for item in request_input
        if item.get("type") == "function_call_output"
    ]

    assert len(calls) == 1
    assert len(outputs) == 1

    # The pair is matched by its call id, and the call comes first.
    assert calls[0]["call_id"] == outputs[0]["call_id"]
    assert calls[0]["name"] == "search_support_docs"
    assert outputs[0]["name"] == "search_support_docs"
    assert json.loads(calls[0]["arguments"]) == {"query": "billing"}
    assert json.loads(outputs[0]["output"])["tool"] == "search_support_docs"

    call_index = request_input.index(calls[0])
    output_index = request_input.index(outputs[0])
    assert call_index < output_index

    # The replayed call id cannot be the id of a call of the current turn.
    assert calls[0]["call_id"] != "call_1"


def test_a_session_cannot_access_the_history_of_another_session(database):
    """Every run only receives the history of its own session."""
    fake_client = _FakeClient(
        [
            _FakeResponse(output_text="First answer."),
            _FakeResponse(output_text="Second answer."),
        ]
    )

    with _route(_settings(), fake_client):
        first = _post("Question for the first session.")
        second = _post("Question for the second session.")

    assert first.json()["session_id"] != second.json()["session_id"]

    first_input = fake_client.responses.requests[0]["input"]
    second_input = fake_client.responses.requests[1]["input"]

    assert _texts(first_input) == ["Question for the first session."]
    assert _texts(second_input) == ["Question for the second session."]


def test_the_history_of_a_session_is_bounded(database):
    """The endpoint replays no more than the configured number of turns."""
    settings = _settings()
    settings.agent_max_history_messages = 2

    fake_client = _FakeClient(
        [
            _FakeResponse(output_text="First answer."),
            _FakeResponse(output_text="Second answer."),
        ]
    )

    with _route(settings, fake_client):
        first = _post("First question.")
        _post("Second question.", session_id=first.json()["session_id"])

    request_input = fake_client.responses.requests[1]["input"]
    assert _texts(request_input) == [
        "First question.",
        "First answer.",
        "Second question.",
    ]

    settings.agent_max_history_messages = 1
    fake_client = _FakeClient(
        [
            _FakeResponse(output_text="First answer."),
            _FakeResponse(output_text="Second answer."),
        ]
    )

    with _route(settings, fake_client):
        first = _post("First question.")
        _post("Second question.", session_id=first.json()["session_id"])

    request_input = fake_client.responses.requests[1]["input"]
    assert _texts(request_input) == ["First answer.", "Second question."]


def test_mock_mode_still_works(database):
    """A run without a provider key answers from the mock service."""
    with _route(_settings(), None):
        response = _post("Help me.")

    assert response.status_code == 200
    assert response.json()["status"] == "mock"
    assert response.json()["mock_mode"] is True
    assert response.json()["answer"]
    assert response.json()["session_id"]

    messages = list_messages(database, response.json()["session_id"])
    assert [message.role for message in messages] == ["user", "assistant"]


def test_a_failing_history_read_answers_service_unavailable(database):
    """A storage failure while reading the history is reported as such."""
    fake_client = _FakeClient([_FakeResponse(output_text="The answer.")])

    with patch(
        "app.modules.agent.service.load_history_items",
        side_effect=psycopg.OperationalError("storage is down"),
    ):
        with _route(_settings(), fake_client):
            response = _post("A question.")

    assert response.status_code == 503
    assert "storage" in response.json()["detail"].lower()

    # The provider was never asked: the run did not start.
    assert fake_client.responses.requests == []


def test_mock_mode_keeps_storing_the_transcript_of_a_session(database):
    """A continuing mock run is stored like any other run."""
    with _route(_settings(), None):
        first = _post("First question.")
        second = _post(
            "Second question.", session_id=first.json()["session_id"]
        )

    assert second.status_code == 200

    messages = list_messages(database, first.json()["session_id"])
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [
        message.content
        for message in messages
        if message.role == "user"
    ] == ["First question.", "Second question."]
