"""Tests for the persistence of the memory of a finished agent run.

The memory writer decides what is worth remembering; this module stores it.
The tests use the isolated PostgreSQL test database, so they never touch
application data.
"""

import logging
from unittest.mock import patch

import pytest

from app.agent_core.agent import AgentResult
from app.modules.agent.service import store_run_memory
from app.persistence import (
    MEMORY_KIND_SUMMARY,
    MessageDraft,
    append_messages,
    create_session,
    get_pool,
    search_memories,
)


#: Every test in this module writes to the database.
pytestmark = pytest.mark.usefixtures("clean_database")


#: The answer of a successful run.
ANSWER = "Refunds are issued within five business days."

#: A second answer of the same session.
OTHER_ANSWER = "Refunds reach the original payment method."


def _result(answer=None, status="success", tool_calls=None, error=None):
    """Return the outcome of a run."""
    return AgentResult(
        status=status,
        answer=answer,
        tool_calls=tool_calls or [],
        error=error,
    )


def _answer_message(database, session_id, content=ANSWER):
    """Store the assistant answer of a run and return its id."""
    stored = append_messages(
        database,
        session_id,
        [MessageDraft(role="assistant", content=content)],
    )
    return stored[0].id


def _memory_count(database, session_id):
    """Return how many memories a session holds."""
    with database.connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM memories WHERE session_id = %s",
            (session_id,),
        ).fetchone()
    return int(row["count"])


# ---------------------------------------------------------------------------
# Storing the memory of a run
# ---------------------------------------------------------------------------


def test_a_successful_run_stores_one_summary_memory(database):
    """The answer of the agent is stored with its provenance."""
    session = create_session(database)
    source_id = _answer_message(database, session.id)

    stored = store_run_memory(
        database, session.id, _result(answer=ANSWER), source_id
    )

    assert stored is True

    memories = search_memories(database, session.id, "refunds")
    assert len(memories) == 1
    assert memories[0].kind == MEMORY_KIND_SUMMARY
    assert memories[0].content == ANSWER
    assert memories[0].session_id == session.id
    assert memories[0].source_msg_id == source_id


def test_a_repeated_answer_is_not_stored_twice(database):
    """The same answer is stored once."""
    session = create_session(database)
    source_id = _answer_message(database, session.id)
    result = _result(answer=ANSWER)

    assert store_run_memory(database, session.id, result, source_id) is True
    assert store_run_memory(database, session.id, result, source_id) is False

    assert _memory_count(database, session.id) == 1


def test_a_different_answer_is_stored_in_the_same_session(database):
    """A session can hold more than one memory."""
    session = create_session(database)
    source_id = _answer_message(database, session.id)

    assert (
        store_run_memory(
            database, session.id, _result(answer=ANSWER), source_id
        )
        is True
    )
    assert (
        store_run_memory(
            database, session.id, _result(answer=OTHER_ANSWER), source_id
        )
        is True
    )

    assert _memory_count(database, session.id) == 2


def test_the_same_answer_is_stored_in_another_session(database):
    """A memory belongs to one session only."""
    first = create_session(database)
    second = create_session(database)
    first_id = _answer_message(database, first.id)
    second_id = _answer_message(database, second.id)

    assert (
        store_run_memory(
            database, first.id, _result(answer=ANSWER), first_id
        )
        is True
    )
    assert (
        store_run_memory(
            database, second.id, _result(answer=ANSWER), second_id
        )
        is True
    )

    assert _memory_count(database, first.id) == 1
    assert _memory_count(database, second.id) == 1


# ---------------------------------------------------------------------------
# Runs that hold nothing worth remembering
# ---------------------------------------------------------------------------


def test_a_mock_run_stores_no_memory(database):
    """The mock answer is not stored."""
    session = create_session(database)

    stored = store_run_memory(
        database,
        session.id,
        _result(
            answer=(
                "Agent mode requires a configured Groq API key. "
                "No tool calls were made."
            ),
            status="mock",
        ),
    )

    assert stored is False
    assert _memory_count(database, session.id) == 0


def test_a_failed_run_stores_no_memory(database):
    """A run without an answer stores nothing."""
    session = create_session(database)

    assert (
        store_run_memory(
            database,
            session.id,
            _result(
                status="provider_failure",
                error="Provider service is temporarily unavailable.",
            ),
        )
        is False
    )
    assert (
        store_run_memory(
            database,
            session.id,
            _result(
                status="max_iterations_reached",
                error="The model did not produce a final answer.",
            ),
        )
        is False
    )

    assert _memory_count(database, session.id) == 0


def test_a_too_short_answer_stores_no_memory(database):
    """A fragment that is too short to carry meaning is not stored."""
    session = create_session(database)
    source_id = _answer_message(database, session.id)

    stored = store_run_memory(
        database, session.id, _result(answer="Ok."), source_id
    )

    assert stored is False
    assert _memory_count(database, session.id) == 0


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_a_write_failure_is_logged_and_does_not_escape(database, caplog):
    """A memory that cannot be stored does not break the caller."""
    session = create_session(database)
    source_id = _answer_message(database, session.id)

    with caplog.at_level(logging.ERROR, logger="agent.service"):
        with patch(
            "app.modules.agent.service.append_memories",
            side_effect=RuntimeError("storage is down"),
        ):
            stored = store_run_memory(
                database, session.id, _result(answer=ANSWER), source_id
            )

    assert stored is False
    assert _memory_count(database, session.id) == 0
    assert any(
        record.levelno == logging.ERROR
        and "Failed to store the memory" in record.getMessage()
        for record in caplog.records
    )


def test_a_failed_duplicate_check_is_logged_and_does_not_escape(
    database, caplog
):
    """A lookup that fails does not break the caller either."""
    session = create_session(database)
    source_id = _answer_message(database, session.id)

    with caplog.at_level(logging.ERROR, logger="agent.service"):
        with patch(
            "app.modules.agent.service.find_memory_by_content",
            side_effect=RuntimeError("storage is down"),
        ):
            stored = store_run_memory(
                database, session.id, _result(answer=ANSWER), source_id
            )

    assert stored is False
    assert _memory_count(database, session.id) == 0
    assert any(
        record.levelno == logging.ERROR
        and "Failed to store the memory" in record.getMessage()
        for record in caplog.records
    )


def test_a_failed_insert_leaves_no_partial_memory_rows(database):
    """A write that fails inside the transaction stores nothing."""
    session = create_session(database)
    pool = get_pool(database.url)
    created_before = pool.get_stats()["connections_num"]

    # The source message does not exist, so the insert breaks the foreign
    # key of the memory and the transaction is rolled back.
    stored = store_run_memory(
        database, session.id, _result(answer=ANSWER), 999999
    )

    assert stored is False
    assert _memory_count(database, session.id) == 0

    # The connection is still usable, and it was not replaced by a new one.
    with database.connection() as conn:
        assert conn.execute("SELECT 1 AS one").fetchone()["one"] == 1
    assert pool.get_stats()["connections_num"] == created_before
