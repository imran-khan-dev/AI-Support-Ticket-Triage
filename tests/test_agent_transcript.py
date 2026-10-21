"""Focused tests for the session and transcript behaviour of the agent endpoint.

These cover the guarantees the session store has to keep:

* the lifecycle of a session, from creation to closure and back;
* the isolation between two sessions;
* the order of the records of a single run;
* the rollback of a failed transaction, and the usability of the connection
  afterwards;
* the allocation of transcript sequence numbers under concurrent writes;
* the behaviour of the endpoint when the transcript cannot be stored.
"""

import threading
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.agent_core.agent import AgentResult
from app.persistence import (
    MessageDraft,
    append_messages,
    close_session,
    count_messages,
    create_session,
    get_pool,
    get_session,
    list_messages,
    reopen_session,
)
from main import app

import os

# conftest points the whole application at the test database before any
# application module is imported, so the route's own settings already point
# at the test database and do not have to be patched per request.
TEST_DATABASE_URL = os.environ["DATABASE_URL"]

client = TestClient(app)

#: Every test in this module writes to the database.
pytestmark = pytest.mark.usefixtures("clean_database")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent_result(answer="Done.", tool_calls=None, status="success", error=None):
    return AgentResult(
        status=status,
        answer=answer,
        tool_calls=tool_calls or [],
        error=error,
    )


@pytest.fixture(autouse=True)
def agent_stub():
    """Replace the provider for the duration of the test.

    The provider is stubbed once for the whole test rather than once per
    request. ``mock.patch`` is not thread safe: when several threads patch
    the same module attribute at the same time, their save and restore steps
    interleave and a stub is left behind, which would silently change the
    behaviour of later tests.
    """
    with patch("app.modules.agent.route.run_agent") as mock_run:
        mock_run.return_value = _agent_result()
        yield mock_run


def _run(agent_stub, message, session_id=None, result=None):
    """Post to the agent endpoint with the stubbed provider."""
    if result is not None:
        agent_stub.return_value = result
    return client.post(
        "/api/v1/tickets/agent/run",
        json={"message": message, "session_id": session_id},
    )


def _tool_call(call_id, tool, result):
    return {
        "call_id": call_id,
        "tool": tool,
        "arguments": {"query": "billing"},
        "result": result,
    }


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def test_a_new_session_is_open_and_accepts_runs(database, agent_stub):
    """A run without a session id creates an open session."""
    response = _run(agent_stub, "Hello.")

    assert response.status_code == 200
    session_id = response.json()["session_id"]
    session = get_session(database, session_id)
    assert session is not None
    assert session.status == "open"


def test_closing_a_session_makes_it_reject_new_runs(database, agent_stub):
    """A closed session is refused with a conflict."""
    response = _run(agent_stub, "First.")
    session_id = response.json()["session_id"]

    closed = close_session(database, session_id)
    assert closed is not None
    assert closed.status == "closed"

    rejected = _run(agent_stub, "Second.", session_id=session_id)
    assert rejected.status_code == 409
    assert "closed" in rejected.json()["detail"].lower()

    # The rejected run did not add anything to the transcript.
    assert count_messages(database, session_id) == 2


def test_a_reopened_session_accepts_runs_again(database, agent_stub):
    """A closed session can be reopened and used again."""
    response = _run(agent_stub, "First.")
    session_id = response.json()["session_id"]

    close_session(database, session_id)
    reopened = reopen_session(database, session_id)
    assert reopened is not None
    assert reopened.status == "open"

    accepted = _run(agent_stub, "Second.", session_id=session_id)
    assert accepted.status_code == 200
    assert accepted.json()["session_id"] == session_id
    assert count_messages(database, session_id) == 4


def test_closing_an_unknown_session_reports_nothing_to_close(database):
    """Closing a session that does not exist is not an error."""
    assert close_session(database, "does-not-exist") is None
    assert reopen_session(database, "does-not-exist") is None


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------


def test_two_sessions_keep_separate_transcripts(database, agent_stub):
    """A run in one session never shows up in another."""
    first = _run(agent_stub, "For the first session.")
    second = _run(agent_stub, "For the second session.")

    first_id = first.json()["session_id"]
    second_id = second.json()["session_id"]
    assert first_id != second_id

    assert [m.content for m in list_messages(database, first_id)] == [
        "For the first session.",
        "Done.",
    ]
    assert [m.content for m in list_messages(database, second_id)] == [
        "For the second session.",
        "Done.",
    ]


def test_a_run_cannot_write_into_another_session(database, agent_stub):
    """Messages written for one session are invisible to another."""
    target = create_session(database)
    other = create_session(database)

    _run(agent_stub, "Only for the target.", session_id=target.id)

    assert count_messages(database, target.id) == 2
    assert count_messages(database, other.id) == 0


# ---------------------------------------------------------------------------
# Transcript ordering
# ---------------------------------------------------------------------------


def test_transcript_order_is_user_then_tool_calls_then_the_answer(
    database, agent_stub
):
    """The records of a run keep their documented order."""
    response = _run(
        agent_stub,
        "I was charged twice.",
        result=_agent_result(
            answer="Refunds are issued within 5 business days.",
            tool_calls=[
                _tool_call(
                    "call_1",
                    "search_support_docs",
                    {"tool": "search_support_docs", "results": [], "count": 0},
                ),
                _tool_call(
                    "call_2",
                    "lookup_ticket_history",
                    {"tool": "lookup_ticket_history", "results": [], "count": 0},
                ),
            ],
        ),
    )

    session_id = response.json()["session_id"]
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
    assert messages[1].tool_name == "search_support_docs"
    assert messages[1].call_id == "call_1"
    assert messages[2].tool_name == "search_support_docs"
    assert messages[2].call_id == "call_1"
    assert messages[3].tool_name == "lookup_ticket_history"
    assert messages[3].call_id == "call_2"
    assert messages[4].tool_name == "lookup_ticket_history"
    assert messages[4].call_id == "call_2"
    assert messages[5].content == "Refunds are issued within 5 business days."


def test_a_failed_run_is_recorded_as_a_system_message(database, agent_stub):
    """A run that produced no answer records the safe error message."""
    response = _run(
        agent_stub,
        "This will fail.",
        result=_agent_result(
            answer=None,
            status="provider_failure",
            error="Provider service is temporarily unavailable.",
        ),
    )

    assert response.status_code == 502
    session_id = response.headers["X-Session-Id"]

    messages = list_messages(database, session_id)
    assert [message.role for message in messages] == ["user", "system"]
    assert messages[1].content == "Provider service is temporarily unavailable."


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def test_a_failed_transaction_writes_nothing(database):
    """A write that fails leaves no partial data behind."""
    session = create_session(database)
    append_messages(database, session.id, [MessageDraft(role="user", content="kept")])

    with pytest.raises(Exception):
        with database.write_transaction() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, seq, role, content)"
                " VALUES (%s, %s, %s, %s)",
                (session.id, 2, "user", "written"),
            )
            # A second message with the same sequence number breaks the
            # unique constraint and aborts the whole transaction.
            conn.execute(
                "INSERT INTO messages (session_id, seq, role, content)"
                " VALUES (%s, %s, %s, %s)",
                (session.id, 2, "user", "duplicate"),
            )

    messages = list_messages(database, session.id)
    assert [message.content for message in messages] == ["kept"]


def test_a_failed_transaction_leaves_the_connection_usable(database):
    """The connection survives a failed transaction and is handed back."""
    pool = get_pool(database.url)
    created_before = pool.get_stats()["connections_num"]

    with pytest.raises(Exception):
        with database.write_transaction() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, seq, role, content)"
                " VALUES (%s, %s, %s, %s)",
                ("22222222-2222-4222-8222-222222222222", 1, "user", "orphan"),
            )

    # The connection is still usable, and it was not replaced by a new one.
    with database.connection() as conn:
        assert conn.execute("SELECT 1 AS one").fetchone()["one"] == 1
    assert pool.get_stats()["connections_num"] == created_before


def test_a_failed_transcript_write_stores_no_partial_transcript(database, agent_stub):
    """A transcript that fails halfway is not stored at all."""
    session = create_session(database)

    with patch(
        "app.modules.agent.service.append_messages",
        side_effect=RuntimeError("storage is down"),
    ):
        response = _run(
            agent_stub, "This transcript cannot be stored.", session_id=session.id
        )

    assert response.status_code == 200
    assert response.headers["X-Transcript-Stored"] == "false"
    assert count_messages(database, session.id) == 0


# ---------------------------------------------------------------------------
# Concurrent appends
# ---------------------------------------------------------------------------


def test_concurrent_appends_allocate_distinct_sequence_numbers(database):
    """Simultaneous appends never hand out the same sequence number."""
    session = create_session(database)
    writers = 6
    per_writer = 4
    failures: list[BaseException] = []

    def append_batch(worker: int) -> None:
        try:
            append_messages(
                database,
                session.id,
                [
                    MessageDraft(role="user", content=f"worker-{worker}-{index}")
                    for index in range(per_writer)
                ],
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    threads = [
        threading.Thread(target=append_batch, args=(worker,))
        for worker in range(writers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures

    messages = list_messages(database, session.id)
    sequences = [message.seq for message in messages]
    assert len(sequences) == writers * per_writer
    assert len(set(sequences)) == len(sequences)
    assert sorted(sequences) == list(range(1, writers * per_writer + 1))


def test_concurrent_appends_do_not_lose_messages(database):
    """Every message written concurrently is stored exactly once."""
    session = create_session(database)
    writers = 6
    per_writer = 4

    def append_batch(worker: int) -> None:
        append_messages(
            database,
            session.id,
            [
                MessageDraft(role="user", content=f"worker-{worker}-{index}")
                for index in range(per_writer)
            ],
        )

    threads = [
        threading.Thread(target=append_batch, args=(worker,))
        for worker in range(writers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    contents = [message.content for message in list_messages(database, session.id)]
    expected = {
        f"worker-{worker}-{index}"
        for worker in range(writers)
        for index in range(per_writer)
    }
    assert len(contents) == len(expected)
    assert set(contents) == expected


def test_concurrent_runs_on_one_session_keep_the_transcript_consistent(
    database, agent_stub
):
    """Simultaneous runs on one session leave a contiguous transcript."""
    session = create_session(database)
    statuses: list[int] = []

    def post_run(index: int) -> None:
        response = _run(
            agent_stub, f"Concurrent question {index}.", session_id=session.id
        )
        statuses.append(response.status_code)

    threads = [
        threading.Thread(target=post_run, args=(index,)) for index in range(5)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert statuses == [200] * 5

    messages = list_messages(database, session.id)
    sequences = [message.seq for message in messages]
    assert sorted(sequences) == list(range(1, len(messages) + 1))

    # Every run contributed a user message and an answer.
    assert [m.role for m in messages].count("user") == 5
    assert [m.role for m in messages].count("assistant") == 5


# ---------------------------------------------------------------------------
# Persistence failure handling
# ---------------------------------------------------------------------------


def test_a_persistence_failure_keeps_the_answer_and_reports_the_status(
    database, agent_stub
):
    """The agent's answer is returned even when the transcript is lost."""
    with patch(
        "app.modules.agent.service.append_messages",
        side_effect=RuntimeError("storage is down"),
    ):
        response = _run(agent_stub, "Please help me.")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["answer"] == "Done."
    assert response.headers["X-Transcript-Stored"] == "false"
    # The session still exists, so the client can retry the run in it.
    assert data["session_id"]
    assert get_session(database, data["session_id"]) is not None


def test_a_successful_run_reports_that_the_transcript_was_stored(
    database, agent_stub
):
    """The status flag is true when the transcript was written."""
    response = _run(agent_stub, "Please help me.")

    assert response.status_code == 200
    assert response.headers["X-Transcript-Stored"] == "true"


def test_the_response_body_has_exactly_the_documented_fields(
    database, agent_stub
):
    """The body of a successful run carries the agreed field set."""
    response = _run(agent_stub, "Please help me.")

    assert response.status_code == 200
    assert set(response.json()) == {
        "status",
        "answer",
        "mock_mode",
        "tool_calls",
        "error",
        "session_id",
    }


def test_the_session_id_is_available_in_a_header_on_success(database, agent_stub):
    """The header carries the same session id as the body."""
    response = _run(agent_stub, "Please help me.")

    assert response.status_code == 200
    assert response.headers["X-Session-Id"] == response.json()["session_id"]


def test_the_session_id_is_available_in_a_header_on_a_provider_failure(
    database, agent_stub
):
    """A failed run still lets the client recover the session id."""
    response = _run(
        agent_stub,
        "This will fail.",
        result=_agent_result(
            answer=None,
            status="provider_failure",
            error="Provider service is temporarily unavailable.",
        ),
    )

    assert response.status_code == 502
    # The error body is unchanged.
    assert response.json() == {
        "detail": "Provider service is temporarily unavailable."
    }
    # The session was created before the provider ran, so the client can
    # recover it and continue the conversation later.
    assert response.headers["X-Session-Id"]
    assert get_session(database, response.headers["X-Session-Id"]) is not None
