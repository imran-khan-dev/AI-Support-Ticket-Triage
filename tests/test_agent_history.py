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

import pytest

from app.agent_core.agent import run_agent
from app.core.config import DEFAULT_MAX_HISTORY_MESSAGES, Settings
from app.modules.agent.history import load_history_items
from app.persistence import (
    MessageDraft,
    append_messages,
    create_session,
    list_messages,
)


pytestmark = pytest.mark.usefixtures("clean_database")


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
    )


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

    assert call["call_id"] == "call_1"
    assert call["name"] == "search_support_docs"
    assert json.loads(call["arguments"]) == {"query": "password reset"}

    assert output["call_id"] == "call_1"
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
