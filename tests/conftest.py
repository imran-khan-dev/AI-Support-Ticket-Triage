"""Shared pytest configuration.

The test suite runs against a dedicated PostgreSQL database so that it never
touches a developer's application data. The connection string is resolved
without storing a credential in version control:

1. ``TEST_DATABASE_URL`` when it is set;
2. otherwise ``DATABASE_URL`` from the environment or from the local ``.env``
   file, with the database name replaced by the test database;
3. otherwise a password-less default, which works with trust authentication
   or a ``.pgpass`` file.

The environment variable is set here because ``conftest.py`` is imported by
pytest before the test modules (and therefore before the application reads
its configuration).
"""

import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest

#: Name of the database the suite is allowed to write to.
TEST_DATABASE_NAME = "ticket_triage_test"

#: Environment variable that overrides the test connection string.
TEST_DATABASE_URL_ENV = "TEST_DATABASE_URL"

#: Fallback used when no connection string can be discovered locally.
DEFAULT_TEST_DATABASE_URL = (
    f"postgresql://postgres@127.0.0.1:5432/{TEST_DATABASE_NAME}"
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _database_url_from_dotenv() -> str | None:
    """Return the DATABASE_URL entry of the local .env file, if there is one."""
    env_file = _PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return None
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "DATABASE_URL":
            return value.strip().strip("'\"") or None
    return None


def _with_database_name(url: str, name: str) -> str:
    """Return the connection string with the database name replaced."""
    parts = urlsplit(url)
    return urlunsplit(
        (parts.scheme, parts.netloc, f"/{name}", parts.query, parts.fragment)
    )


def _resolve_test_database_url() -> str:
    """Return the connection string the suite must use."""
    override = os.environ.get(TEST_DATABASE_URL_ENV)
    if override:
        return override.strip()
    base = os.environ.get("DATABASE_URL") or _database_url_from_dotenv()
    if base:
        return _with_database_name(base.strip(), TEST_DATABASE_NAME)
    return DEFAULT_TEST_DATABASE_URL


TEST_DATABASE_URL = _resolve_test_database_url()

# The application settings read DATABASE_URL, so the whole application is
# pointed at the test database before any application module is imported.
os.environ["DATABASE_URL"] = TEST_DATABASE_URL


@pytest.fixture(scope="session")
def database():
    """The shared PostgreSQL handle, with the schema applied.

    The session scope keeps one pool open for the whole run; the schema is
    applied once because the migration runner is idempotent.
    """
    from app.database.db import close_pools, get_database

    db = get_database(TEST_DATABASE_URL)
    db.initialize()
    yield db
    close_pools()


@pytest.fixture()
def clean_database(database):
    """Empty every table so that a test starts from a known state."""
    with database.write_transaction() as conn:
        conn.execute(
            "TRUNCATE TABLE messages, memories, actions, sessions, approvers,"
            " audit_events RESTART IDENTITY CASCADE"
        )
    return database
