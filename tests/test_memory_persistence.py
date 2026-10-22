"""Tests for the PostgreSQL memory storage and retrieval layer.

Every test runs against the dedicated PostgreSQL test database, which is
emptied before the test starts, so the suite never touches application data.
"""

import pytest
from psycopg import errors

from app.persistence import (
    MEMORY_CONTENT_MAX_CHARS,
    MEMORY_KIND_FACT,
    MEMORY_KIND_OUTCOME,
    MEMORY_KIND_PREFERENCE,
    MEMORY_KIND_SUMMARY,
    MemoryDraft,
    MessageDraft,
    UnknownSessionError,
    append_memories,
    append_messages,
    create_session,
    search_memories,
)
from app.persistence.migrations import MIGRATIONS, current_version


#: Every test in this module writes to the database.
pytestmark = pytest.mark.usefixtures("clean_database")


# ---------------------------------------------------------------------------
# Migration 2
# ---------------------------------------------------------------------------


def test_migration_2_is_applied(database):
    """The schema version is the one of the latest migration."""
    with database.connection() as conn:
        assert current_version(conn) == 2
    assert MIGRATIONS[-1][0] == 2
    assert MIGRATIONS[-1][1] == "memory_full_text_search"


def test_migration_2_adds_a_generated_search_column(database):
    """The search vector is derived from the content by the database."""
    with database.connection() as conn:
        row = conn.execute(
            "SELECT data_type, is_generated, generation_expression"
            " FROM information_schema.columns"
            " WHERE table_name = 'memories' AND column_name = 'search_vector'"
        ).fetchone()

    assert row is not None
    assert row["data_type"] == "tsvector"
    assert row["is_generated"] == "ALWAYS"
    assert "to_tsvector" in row["generation_expression"]


def test_migration_2_adds_a_gin_index(database):
    """The search vector is indexed."""
    with database.connection() as conn:
        row = conn.execute(
            "SELECT indexdef FROM pg_indexes"
            " WHERE tablename = 'memories' AND indexname = 'ix_memories_search'"
        ).fetchone()

    assert row is not None
    assert "USING GIN" in row["indexdef"].upper()
    assert "search_vector" in row["indexdef"]


def test_the_migration_runner_stays_safe(database):
    """Applying the migrations again changes nothing."""
    assert database.initialize() == []
    with database.connection() as conn:
        assert current_version(conn) == 2


def test_migration_1_left_the_memory_schema_intact(database):
    """The columns and the constraints of the original table are unchanged."""
    with database.connection() as conn:
        columns = {
            row["column_name"]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns"
                " WHERE table_name = 'memories'"
            ).fetchall()
        }
        constraints = {
            (row["name"], row["def"])
            for row in conn.execute(
                "SELECT conname AS name, pg_get_constraintdef(oid) AS def"
                " FROM pg_constraint WHERE conrelid = 'memories'::regclass"
            ).fetchall()
        }

    assert {
        "id",
        "session_id",
        "kind",
        "content",
        "source_msg_id",
        "created_at",
    } <= columns

    assert any("memories_kind_check" == name for name, _ in constraints)
    assert any(
        "REFERENCES sessions(id) ON DELETE CASCADE" in definition
        for _, definition in constraints
    )
    assert any(
        "REFERENCES messages(id) ON DELETE SET NULL" in definition
        for _, definition in constraints
    )


# ---------------------------------------------------------------------------
# Storing memories
# ---------------------------------------------------------------------------


def test_memories_are_stored_and_read_back(database):
    """A stored memory keeps its kind, content, and session."""
    session = create_session(database)

    stored = append_memories(
        database,
        session.id,
        [
            MemoryDraft(
                kind=MEMORY_KIND_FACT,
                content="Refunds are issued within five business days.",
            ),
            MemoryDraft(
                kind=MEMORY_KIND_OUTCOME,
                content="The customer received a refund on 4 May.",
            ),
        ],
    )

    assert len(stored) == 2
    assert stored[0].id > 0
    assert stored[0].session_id == session.id
    assert stored[0].kind == MEMORY_KIND_FACT
    assert stored[0].content == "Refunds are issued within five business days."
    assert stored[0].source_msg_id is None
    assert stored[0].created_at.endswith("Z")
    assert stored[1].kind == MEMORY_KIND_OUTCOME


def test_every_supported_kind_can_be_stored(database):
    """The four kinds of the schema are accepted."""
    session = create_session(database)

    stored = append_memories(
        database,
        session.id,
        [
            MemoryDraft(kind=MEMORY_KIND_SUMMARY, content="A summary."),
            MemoryDraft(kind=MEMORY_KIND_FACT, content="A fact."),
            MemoryDraft(kind=MEMORY_KIND_PREFERENCE, content="A preference."),
            MemoryDraft(kind=MEMORY_KIND_OUTCOME, content="An outcome."),
        ],
    )

    assert [memory.kind for memory in stored] == [
        MEMORY_KIND_SUMMARY,
        MEMORY_KIND_FACT,
        MEMORY_KIND_PREFERENCE,
        MEMORY_KIND_OUTCOME,
    ]


def test_an_invalid_kind_is_rejected_by_the_database(database):
    """A kind outside the schema is refused by the constraint."""
    session = create_session(database)

    with pytest.raises(errors.CheckViolation):
        append_memories(
            database,
            session.id,
            [MemoryDraft(kind="bogus", content="Not a kind.")],
        )


def test_memories_of_an_unknown_session_are_rejected(database):
    """A memory cannot be stored in a session that does not exist."""
    with pytest.raises(UnknownSessionError):
        append_memories(
            database,
            "does-not-exist",
            [MemoryDraft(kind=MEMORY_KIND_FACT, content="A fact.")],
        )


def test_the_content_of_a_memory_is_capped(database):
    """A memory stays a short fact, not a transcript."""
    session = create_session(database)

    stored = append_memories(
        database,
        session.id,
        [
            MemoryDraft(
                kind=MEMORY_KIND_FACT,
                content="x" * (MEMORY_CONTENT_MAX_CHARS + 500),
            )
        ],
    )

    assert len(stored[0].content) == MEMORY_CONTENT_MAX_CHARS


def test_the_search_vector_is_filled_by_the_database(database):
    """The generated column holds the search vector of the content."""
    session = create_session(database)
    stored = append_memories(
        database,
        session.id,
        [MemoryDraft(kind=MEMORY_KIND_FACT, content="Refunds take five days.")],
    )

    with database.connection() as conn:
        row = conn.execute(
            "SELECT search_vector::text AS vector FROM memories WHERE id = %s",
            (stored[0].id,),
        ).fetchone()

    assert row is not None
    assert "refund" in row["vector"]


# ---------------------------------------------------------------------------
# Searching memories
# ---------------------------------------------------------------------------


def test_search_finds_the_relevant_memory(database):
    """A full-text search returns the memory that matches the query."""
    session = create_session(database)
    append_memories(
        database,
        session.id,
        [
            MemoryDraft(
                kind=MEMORY_KIND_PREFERENCE,
                content="The customer prefers email notifications.",
            ),
            MemoryDraft(
                kind=MEMORY_KIND_FACT,
                content="Refunds are issued within five business days.",
            ),
        ],
    )

    found = search_memories(database, session.id, "refund")

    assert [memory.content for memory in found] == [
        "Refunds are issued within five business days."
    ]

    other = search_memories(database, session.id, "notifications")
    assert [memory.content for memory in other] == [
        "The customer prefers email notifications."
    ]


def test_search_never_returns_the_memories_of_another_session(database):
    """A session only ever sees its own memories."""
    first = create_session(database)
    second = create_session(database)
    append_memories(
        database,
        first.id,
        [
            MemoryDraft(
                kind=MEMORY_KIND_FACT,
                content="The customer asked for a refund.",
            )
        ],
    )

    assert search_memories(database, second.id, "refund") == []
    assert len(search_memories(database, first.id, "refund")) == 1


def test_search_returns_at_most_the_requested_number_of_results(database):
    """The number of results is bounded."""
    session = create_session(database)
    append_memories(
        database,
        session.id,
        [
            MemoryDraft(
                kind=MEMORY_KIND_FACT,
                content=f"Refund note number {index}.",
            )
            for index in range(4)
        ],
    )

    found = search_memories(database, session.id, "refund", limit=2)

    assert len(found) == 2
    assert search_memories(database, session.id, "refund", limit=0) == []


def test_search_orders_its_results_deterministically(database):
    """The same query always returns the same order."""
    session = create_session(database)
    stored = append_memories(
        database,
        session.id,
        [
            MemoryDraft(
                kind=MEMORY_KIND_FACT,
                content="The customer asked for a refund.",
            )
            for _ in range(3)
        ],
    )

    first = search_memories(database, session.id, "refund")
    second = search_memories(database, session.id, "refund")

    identifiers = [memory.id for memory in first]
    assert identifiers == [memory.id for memory in second]
    assert identifiers == sorted(
        [memory.id for memory in stored], reverse=True
    )


def test_a_memory_matches_when_the_query_holds_one_of_its_terms(database):
    """The terms of a query are matched one by one, not all at once."""
    session = create_session(database)
    append_memories(
        database,
        session.id,
        [
            MemoryDraft(
                kind=MEMORY_KIND_FACT,
                content="Refunds are issued within five business days.",
            )
        ],
    )

    found = search_memories(database, session.id, "I need a refund")

    assert [memory.content for memory in found] == [
        "Refunds are issued within five business days."
    ]


def test_a_query_word_that_looks_like_an_operator_is_harmless(database):
    """A word of the message cannot change the meaning of the query."""
    session = create_session(database)
    append_memories(
        database,
        session.id,
        [
            MemoryDraft(
                kind=MEMORY_KIND_FACT,
                content="Refunds are issued within five business days.",
            )
        ],
    )

    found = search_memories(database, session.id, "refund or not refund")

    assert len(found) == 1


def test_an_empty_query_returns_nothing(database):
    """A search without a term matches nothing."""
    session = create_session(database)
    append_memories(
        database,
        session.id,
        [MemoryDraft(kind=MEMORY_KIND_FACT, content="A fact about refunds.")],
    )

    assert search_memories(database, session.id, "") == []
    assert search_memories(database, session.id, "   ") == []


def test_search_can_shorten_the_content_it_returns(database):
    """The caller decides how much of a memory it wants to read."""
    session = create_session(database)
    append_memories(
        database,
        session.id,
        [
            MemoryDraft(
                kind=MEMORY_KIND_FACT,
                content="Refunds are issued within five business days.",
            )
        ],
    )

    found = search_memories(database, session.id, "refund", content_limit=11)

    assert len(found) == 1
    assert found[0].content == "Refunds are"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_the_source_message_is_preserved_and_cleared_when_it_is_deleted(database):
    """A memory keeps its source, and survives the deletion of that source."""
    session = create_session(database)
    messages = append_messages(
        database,
        session.id,
        [MessageDraft(role="assistant", content="The final answer.")],
    )
    source_id = messages[0].id

    stored = append_memories(
        database,
        session.id,
        [
            MemoryDraft(
                kind=MEMORY_KIND_SUMMARY,
                content="A summary of the refund request.",
                source_msg_id=source_id,
            )
        ],
    )
    assert stored[0].source_msg_id == source_id

    found = search_memories(database, session.id, "summary")
    assert found[0].source_msg_id == source_id

    with database.write_transaction() as conn:
        conn.execute(
            "DELETE FROM messages WHERE id = %s",
            (source_id,),
        )

    found = search_memories(database, session.id, "summary")
    assert len(found) == 1
    assert found[0].source_msg_id is None
