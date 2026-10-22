"""Tests for the PostgreSQL persistence foundation.

Every test runs against the dedicated test database, which is emptied before
the test starts, so the suite never touches a developer's application data.
"""

import uuid

import psycopg
import pytest
from psycopg import errors

from app.core.config import Settings
from app.persistence import (
    Database,
    DatabaseConfigurationError,
    MessageDraft,
    append_messages,
    count_messages,
    create_session,
    get_database,
    get_pool,
    get_session,
    list_messages,
    list_sessions,
    resolve_database_url,
)
from app.persistence.migrations import MIGRATIONS, current_version

EXPECTED_TABLES = {
    "sessions",
    "messages",
    "memories",
    "actions",
    "approvers",
    "audit_events",
}

#: A valid correlation id; the column is a PostgreSQL UUID.
CORRELATION_ID = "11111111-1111-4111-8111-111111111111"


def _table_names(database):
    with database.connection() as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables"
            " WHERE table_schema = 'public'"
        ).fetchall()
    return {row["table_name"] for row in rows}


# ---------------------------------------------------------------------------
# Schema and migrations
# ---------------------------------------------------------------------------


def test_initialization_creates_every_table(database):
    """All initial tables are created by the first migration."""
    assert EXPECTED_TABLES.issubset(_table_names(database))


def test_migrations_record_the_schema_version(database):
    """The schema version is stored in the database."""
    with database.connection() as conn:
        assert current_version(conn) == 3
    assert current_version.__module__.endswith("migrations")
    assert MIGRATIONS[-1][0] == 3


def test_migrations_are_idempotent(database, clean_database):
    """Running the migrations again applies nothing and keeps the data."""
    session = create_session(database)

    assert database.initialize() == []
    assert database.ensure_initialized() is None

    assert len(list_sessions(database)) == 1
    assert get_session(database, session.id) is not None


def test_a_failed_statement_rolls_back_the_whole_transaction(database):
    """A write that fails leaves nothing behind."""
    session = create_session(database)

    with pytest.raises(errors.ForeignKeyViolation):
        with database.write_transaction() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, seq, role, content)"
                " VALUES (%s, %s, %s, %s)",
                (
                    "22222222-2222-4222-8222-222222222222",
                    1,
                    "user",
                    "orphan",
                ),
            )

    assert count_messages(database, session.id) == 0


def test_foreign_keys_are_enforced(database, clean_database):
    """A message cannot reference a session that does not exist."""
    with pytest.raises(errors.ForeignKeyViolation):
        with database.write_transaction() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, seq, role, content)"
                " VALUES (%s, %s, %s, %s)",
                (
                    "22222222-2222-4222-8222-222222222222",
                    1,
                    "user",
                    "hello",
                ),
            )


def test_session_identifiers_are_real_uuids(database, clean_database):
    """The session id column rejects a value that is not an identifier."""
    with pytest.raises(errors.InvalidTextRepresentation):
        with database.write_transaction() as conn:
            conn.execute(
                "INSERT INTO sessions (id) VALUES (%s)", ("not-a-uuid",)
            )


def test_message_roles_are_constrained(database, clean_database):
    """Only the known message roles are accepted."""
    session = create_session(database)

    with pytest.raises(errors.CheckViolation):
        with database.write_transaction() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, seq, role, content)"
                " VALUES (%s, %s, %s, %s)",
                (session.id, 1, "bogus", "hello"),
            )


def test_sequence_numbers_are_unique_per_session(database, clean_database):
    """A session cannot hold two messages with the same sequence number."""
    session = create_session(database)
    append_messages(database, session.id, [MessageDraft(role="user", content="one")])

    with pytest.raises(errors.UniqueViolation):
        with database.write_transaction() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, seq, role, content)"
                " VALUES (%s, %s, %s, %s)",
                (session.id, 1, "user", "duplicate"),
            )


def test_audit_events_are_append_only(database, clean_database):
    """Audit rows cannot be updated or deleted."""
    event_id = str(uuid.uuid4())
    with database.write_transaction() as conn:
        conn.execute(
            "INSERT INTO audit_events"
            " (event_id, correlation_id, actor_type, action)"
            " VALUES (%s, %s, %s, %s)",
            (event_id, CORRELATION_ID, "system", "test.event"),
        )

    with pytest.raises(errors.RaiseException):
        with database.write_transaction() as conn:
            conn.execute(
                "UPDATE audit_events SET action = 'tampered' WHERE event_id = %s",
                (event_id,),
            )

    with pytest.raises(errors.RaiseException):
        with database.write_transaction() as conn:
            conn.execute(
                "DELETE FROM audit_events WHERE event_id = %s", (event_id,)
            )


# ---------------------------------------------------------------------------
# Configuration and connection management
# ---------------------------------------------------------------------------


def test_database_url_is_configurable_through_the_environment(monkeypatch):
    """DATABASE_URL selects the database."""
    target = "postgresql://user:secret@db.example:5432/tickets"
    monkeypatch.setenv("DATABASE_URL", target)

    assert resolve_database_url() == target
    assert Settings(_env_file=None).database_url == target


def test_database_url_is_required(monkeypatch):
    """Without DATABASE_URL there is no silent default."""
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(DatabaseConfigurationError):
        resolve_database_url()


def test_explicit_url_wins_over_the_environment(monkeypatch):
    """An explicit connection string takes precedence over the environment."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env/db")

    assert resolve_database_url("postgresql://explicit/db") == "postgresql://explicit/db"


def test_get_pool_returns_one_instance_per_url(database):
    """The same connection string resolves to the same pool."""
    first = get_pool(database.url)
    second = get_pool(database.url)
    other = get_pool(f"{database.url}?application_name=other")

    assert first is second
    assert first is not other


def test_get_database_returns_one_instance_per_url(database):
    """The same connection string resolves to the same database handle."""
    first = get_database(database.url)
    second = get_database(database.url)
    other = get_database(f"{database.url}?application_name=other")

    assert first is second
    assert first is not other


def test_connections_are_reused_from_the_pool(database):
    """The pool reuses backend connections instead of opening one per request."""
    pool = get_pool(database.url)
    requests = 20

    backend_pids = set()
    for _ in range(requests):
        with database.connection() as conn:
            backend_pids.add(
                conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"]
            )

    # Far fewer connections than requests were used, so the connections are
    # pooled and reused rather than opened per request. The pool may already
    # hold connections from earlier tests, so the bound is its own size.
    assert len(backend_pids) < requests
    assert len(backend_pids) <= pool.get_stats()["pool_max"]


def test_a_closed_handle_refuses_to_be_used(database):
    """A handle that was closed no longer hands out connections."""
    other = Database(database.url)
    other.close()

    assert other.closed is True
    with pytest.raises(DatabaseConfigurationError):
        with other.connection():
            pass


# ---------------------------------------------------------------------------
# Sessions and messages
# ---------------------------------------------------------------------------


def test_create_session_generates_a_server_side_id(database, clean_database):
    """A new session gets a server-generated identifier."""
    session = create_session(database)

    assert session.id
    assert uuid.UUID(session.id)
    assert session.status == "open"
    assert session.created_at == session.updated_at
    assert get_session(database, session.id) is not None


def test_messages_are_ordered_deterministically(database, clean_database):
    """Sequence numbers are contiguous and increase in insertion order."""
    session = create_session(database)

    append_messages(
        database,
        session.id,
        [
            MessageDraft(role="user", content="first"),
            MessageDraft(role="assistant", content="second"),
            MessageDraft(role="tool", content="third"),
        ],
    )
    append_messages(
        database,
        session.id,
        [MessageDraft(role="assistant", content="fourth")],
    )

    messages = list_messages(database, session.id)
    assert [message.seq for message in messages] == [1, 2, 3, 4]
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert [message.content for message in messages] == [
        "first",
        "second",
        "third",
        "fourth",
    ]


def test_concurrent_appends_do_not_collide(database, clean_database):
    """Two batches written at the same time receive distinct sequence numbers."""
    import threading

    session = create_session(database)
    errors_seen: list[BaseException] = []

    def append_batch(label: str) -> None:
        try:
            append_messages(
                database,
                session.id,
                [MessageDraft(role="user", content=f"{label}-{index}") for index in range(5)],
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors_seen.append(exc)

    threads = [threading.Thread(target=append_batch, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors_seen
    sequences = sorted(message.seq for message in list_messages(database, session.id))
    assert sequences == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_sessions_do_not_share_messages(database, clean_database):
    """Messages belong to exactly one session."""
    first = create_session(database)
    second = create_session(database)

    append_messages(database, first.id, [MessageDraft(role="user", content="one")])
    append_messages(database, second.id, [MessageDraft(role="user", content="two")])

    assert count_messages(database, first.id) == 1
    assert count_messages(database, second.id) == 1
    assert [m.content for m in list_messages(database, first.id)] == ["one"]
    assert [m.content for m in list_messages(database, second.id)] == ["two"]


def test_messages_carry_the_call_id_and_tool_name(database, clean_database):
    """Tool records keep the correlation details of the call."""
    session = create_session(database)

    append_messages(
        database,
        session.id,
        [
            MessageDraft(
                role="assistant",
                content="{}",
                call_id="call_1",
                tool_name="search_support_docs",
            ),
            MessageDraft(
                role="tool",
                content="{}",
                call_id="call_1",
                tool_name="search_support_docs",
            ),
        ],
        correlation_id=CORRELATION_ID,
    )

    messages = list_messages(database, session.id)
    assert all(message.call_id == "call_1" for message in messages)
    assert all(
        message.tool_name == "search_support_docs" for message in messages
    )
    assert all(message.correlation_id == CORRELATION_ID for message in messages)


def test_appending_messages_updates_the_session(database, clean_database):
    """The session timestamp moves forward when messages are added."""
    session = create_session(database)
    before = session.updated_at

    append_messages(database, session.id, [MessageDraft(role="user", content="hi")])

    updated = get_session(database, session.id)
    assert updated.updated_at >= before


def test_timestamps_are_iso_8601_utc(database, clean_database):
    """Stored timestamps are rendered as UTC ISO-8601 strings."""
    session = create_session(database)

    assert session.created_at.endswith("Z")
    assert session.created_at.startswith("20")


def test_transcript_survives_a_new_connection(database, clean_database):
    """Stored messages are still there after the handle is replaced."""
    session = create_session(database)
    append_messages(database, session.id, [MessageDraft(role="user", content="kept")])

    other = Database(database.url)
    try:
        reopened = get_session(other, session.id)
        assert reopened is not None
        assert [m.content for m in list_messages(other, session.id)] == ["kept"]
    finally:
        other.close()


def test_deleting_a_session_deletes_its_messages(database, clean_database):
    """The foreign key cascade keeps orphaned messages out of the store."""
    session = create_session(database)
    append_messages(database, session.id, [MessageDraft(role="user", content="hi")])

    with database.write_transaction() as conn:
        conn.execute("DELETE FROM sessions WHERE id = %s", (session.id,))

    assert count_messages(database, session.id) == 0


def test_unknown_session_returns_none(database, clean_database):
    """Looking up an unknown session is not an error."""
    assert get_session(database, "does-not-exist") is None
