"""PostgreSQL connection management for durable session and message storage.

Driver choice
-------------
``psycopg`` (psycopg 3) is used together with ``psycopg_pool``:

* psycopg 3 is the maintained PostgreSQL driver for Python and the successor
  to psycopg2. It supports both a synchronous and an asynchronous API, so it
  fits the existing synchronous service layer without an async rewrite.
* ``psycopg_pool`` provides a thread-safe connection pool, which the service
  needs because a FastAPI deployment handles requests on a thread pool and
  PostgreSQL connections are expensive to establish.
* No ORM is used. The schema is six tables with straightforward relations and
  a small amount of hand-written SQL. An ORM would add a large dependency and
  a second layer of abstractions without changing the architecture, and the
  transactional guarantees this milestone depends on are clearer when the SQL
  is explicit.

Connection management
---------------------
* The pool is opened lazily so that importing the application never requires a
  reachable database, and it is closed on application shutdown.
* Pooled connections run with ``autocommit`` off, so a statement starts a
  transaction. :meth:`Database.write_transaction` commits it when the block
  succeeds and rolls it back when it raises.
* Rows are returned as dictionaries so that the repository addresses columns
  by name instead of by position.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.persistence.migrations import apply_migrations

logger = logging.getLogger("persistence.db")

#: Environment variable holding the PostgreSQL connection string.
DATABASE_URL_ENV = "DATABASE_URL"

DEFAULT_POOL_MIN_SIZE = 1
DEFAULT_POOL_MAX_SIZE = 10

#: How long a request waits for a pooled connection before giving up.
DEFAULT_POOL_TIMEOUT = 10.0


class DatabaseConfigurationError(RuntimeError):
    """Raised when no PostgreSQL connection string is configured."""


def resolve_database_url(url: str | None = None) -> str:
    """Resolve the PostgreSQL connection string.

    Precedence: the explicit argument, then the ``DATABASE_URL`` environment
    variable. An empty value is an error rather than a silent default, so a
    missing configuration is reported instead of guessing a server.
    """
    if url is None:
        url = os.environ.get(DATABASE_URL_ENV) or ""
    url = url.strip()
    if not url:
        raise DatabaseConfigurationError(
            "DATABASE_URL is not configured. Set it to a PostgreSQL connection "
            "string, for example postgresql://user:password@localhost:5432/db."
        )
    return url


def build_pool(
    url: str,
    min_size: int = DEFAULT_POOL_MIN_SIZE,
    max_size: int = DEFAULT_POOL_MAX_SIZE,
    timeout: float = DEFAULT_POOL_TIMEOUT,
) -> ConnectionPool:
    """Create a connection pool for a connection string.

    The pool is opened in the background (``wait=False``) so that creating it
    never blocks; the first request that needs a connection waits for one.
    """
    pool = ConnectionPool(
        conninfo=url,
        min_size=min_size,
        max_size=max_size,
        timeout=timeout,
        open=False,
        # Connections are handed back with any open transaction rolled back,
        # so a failed statement can never leak into the next user of the
        # connection.
        kwargs={"autocommit": False, "row_factory": dict_row},
    )
    pool.open(wait=False)
    return pool


_pools: dict[str, ConnectionPool] = {}
_pool_references: dict[str, int] = {}
# Reentrant: get_database() holds the lock while it builds a Database, which
# acquires a pool reference and takes the lock again.
_pools_lock = threading.RLock()

#: Shared database handles, one per connection string.
_databases: dict[str, Database] = {}

#: Connection strings whose schema is known to be up to date in this process.
_initialized_urls: set[str] = set()
_initialized_lock = threading.Lock()


def get_pool(
    url: str | None = None,
    min_size: int = DEFAULT_POOL_MIN_SIZE,
    max_size: int = DEFAULT_POOL_MAX_SIZE,
    timeout: float = DEFAULT_POOL_TIMEOUT,
) -> ConnectionPool:
    """Return the shared connection pool for a connection string.

    The same connection string always yields the same pool, so connections are
    reused across requests instead of being opened per request.
    """
    resolved = resolve_database_url(url)
    with _pools_lock:
        pool = _pools.get(resolved)
        if pool is None:
            pool = build_pool(resolved, min_size, max_size, timeout)
            _pools[resolved] = pool
        return pool


def _acquire_pool(
    url: str,
    min_size: int,
    max_size: int,
    timeout: float,
) -> ConnectionPool:
    """Return the shared pool for a connection string and take a reference."""
    resolved = resolve_database_url(url)
    with _pools_lock:
        pool = _pools.get(resolved)
        if pool is None:
            pool = build_pool(resolved, min_size, max_size, timeout)
            _pools[resolved] = pool
        _pool_references[resolved] = _pool_references.get(resolved, 0) + 1
        return pool


def _release_pool(pool: ConnectionPool) -> None:
    """Release a reference to a pool, closing it when the last one goes."""
    with _pools_lock:
        for url, candidate in list(_pools.items()):
            if candidate is not pool:
                continue
            remaining = _pool_references.get(url, 1) - 1
            if remaining <= 0:
                candidate.close()
                _pools.pop(url, None)
                _pool_references.pop(url, None)
            else:
                _pool_references[url] = remaining
            return


def close_pools() -> None:
    """Close every open pool. Used on application shutdown."""
    with _pools_lock:
        for database in _databases.values():
            database._closed = True
        for pool in _pools.values():
            pool.close()
        _pools.clear()
        _pool_references.clear()
        _databases.clear()
        _initialized_urls.clear()


@contextmanager
def write_transaction(conn: psycopg.Connection) -> Iterator[psycopg.Connection]:
    """Run a block inside a transaction.

    The transaction is committed when the block succeeds and rolled back when
    it raises, so a failed write can never leave a half-applied change behind
    and can never leave the connection in a broken transaction state.
    """
    with conn.transaction():
        yield conn


class Database:
    """A PostgreSQL-backed handle around a connection pool.

    The handle owns the pool lifetime and the schema-initialization state. It
    is the only database object the repository layer needs, and it knows
    nothing about the agent, the provider, or the HTTP layer.
    """

    def __init__(
        self,
        url: str | None = None,
        min_size: int = DEFAULT_POOL_MIN_SIZE,
        max_size: int = DEFAULT_POOL_MAX_SIZE,
        timeout: float = DEFAULT_POOL_TIMEOUT,
    ) -> None:
        self._url = resolve_database_url(url)
        self._pool = _acquire_pool(self._url, min_size, max_size, timeout)
        self._closed = False

    @property
    def url(self) -> str:
        """The connection string this handle is bound to."""
        return self._url

    @property
    def closed(self) -> bool:
        """True once the handle has been closed."""
        return self._closed

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        """Yield a pooled connection for reads."""
        self._require_open()
        with self._pool.connection() as conn:
            yield conn

    @contextmanager
    def write_transaction(self) -> Iterator[psycopg.Connection]:
        """Yield a pooled connection inside a committed transaction."""
        self._require_open()
        with self._pool.connection() as conn:
            with write_transaction(conn):
                yield conn

    def initialize(self) -> list[int]:
        """Apply every pending migration and return the applied versions.

        Applying migrations to an up-to-date database is a no-op and returns
        an empty list, so initialization is idempotent.
        """
        self._require_open()
        with self._pool.connection() as conn:
            return apply_migrations(conn)

    def ensure_initialized(self) -> None:
        """Make sure the schema is present, at most once per process.

        The migration runner takes a PostgreSQL advisory lock, so several
        application instances can call this at the same time without
        migrating the database concurrently.
        """
        self._require_open()
        with _initialized_lock:
            if self._url in _initialized_urls:
                return
            applied = self.initialize()
            _initialized_urls.add(self._url)
        if applied:
            logger.info("Applied database migrations: %s", applied)

    def close(self) -> None:
        """Release this handle's reference to the connection pool."""
        if self._closed:
            return
        self._closed = True
        _release_pool(self._pool)

    def _require_open(self) -> None:
        if self._closed:
            raise DatabaseConfigurationError(
                "The database handle has been closed."
            )


def get_database(url: str | None = None) -> Database:
    """Return the shared database handle for a connection string.

    The same connection string always yields the same handle, so the pool and
    the schema-initialization state are shared across requests.
    """
    resolved = resolve_database_url(url)
    with _pools_lock:
        database = _databases.get(resolved)
        if database is None:
            database = Database(resolved)
            _databases[resolved] = database
        return database
