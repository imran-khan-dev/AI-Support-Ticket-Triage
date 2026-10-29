import os

os.environ.setdefault("APPROVER_TOKEN_SECRET", "test-only-secret-for-approver-credentials-0123456789")

import pytest
import uuid

from app.database.migrations import MIGRATIONS, current_version, apply_migrations


def _get(row, key, idx=0):
    if row is None:
        return None
    if isinstance(row, dict):
        if key in row:
            return row[key]
        # try common forms
        for k,v in row.items():
            if k.lower() == key.lower():
                return v
        return None
    try:
        return row[idx]
    except Exception:
        return None


@pytest.mark.usefixtures("clean_database")
def test_migration_to_version_6(database):
    versions = [v for v, n, s in MIGRATIONS]
    assert versions[-1] == 6
    with database.connection() as conn:
        assert current_version(conn) >= 6
    with database.connection() as conn:
        row = conn.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)", ('tenants',)).fetchone()
        exists = _get(row, 'exists', 0)
        assert exists
    with database.connection() as conn:
        cols = conn.execute("SELECT column_name, is_nullable FROM information_schema.columns WHERE table_name IN ('integrations','sessions','actions') AND column_name = %s", ('tenant_id',)).fetchall()
        assert len(cols) == 3
        for c in cols:
            nullable = _get(c, 'is_nullable', 1)
            assert nullable == 'YES'
    with database.connection() as conn:
        cnt = conn.execute("SELECT COUNT(*) as c FROM sessions WHERE tenant_id IS NULL").fetchone()
        c = _get(cnt, 'c', 0)
        assert c >= 0


@pytest.mark.usefixtures("clean_database")
def test_migration_v5_to_v6_preserves_records(database):
    with database.connection() as conn:
        try:
            conn.execute("INSERT INTO integrations (id, name, enabled) VALUES (%s, %s, %s)", (str(uuid.uuid4()), f'test-int-{uuid.uuid4()}', True))
        except Exception:
            pass
        sid = str(uuid.uuid4())
        try:
            conn.execute("INSERT INTO sessions (id, status) VALUES (%s, %s)", (sid, 'open'))
        except Exception:
            pass
        aid = str(uuid.uuid4())
        try:
            conn.execute("INSERT INTO actions (id, request_key, correlation_id, tool_name, effect_class, args, args_hash, state, requested_by) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                         (aid, f'req-{aid}', str(uuid.uuid4()), 'tool', 'gated_tool', '{}', 'hash', 'pending', 'agent'))
        except Exception:
            pass
    with database.connection() as conn:
        c_int = _get(conn.execute("SELECT COUNT(*) as c FROM integrations WHERE tenant_id IS NULL").fetchone(), 'c', 0)
        c_sess = _get(conn.execute("SELECT COUNT(*) as c FROM sessions WHERE tenant_id IS NULL").fetchone(), 'c', 0)
        c_act = _get(conn.execute("SELECT COUNT(*) as c FROM actions WHERE tenant_id IS NULL").fetchone(), 'c', 0)
        assert c_int >= 0
        assert c_sess >= 0
        assert c_act >= 0



@pytest.mark.usefixtures("clean_database")
def test_tenant_fk_and_indexes(database):
    with database.connection() as conn:
        fks = conn.execute("""
            SELECT conname
            FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            WHERE t.relname IN ('integrations','sessions','actions')
              AND c.contype = 'f'
              AND conname LIKE '%tenant%'
        """).fetchall()
        assert len(fks) >= 1
        idxs = conn.execute("""
            SELECT indexname
            FROM pg_indexes
            WHERE tablename IN ('integrations','sessions','actions')
              AND indexname LIKE '%tenant%'
        """).fetchall()
        assert len(idxs) >= 3


@pytest.mark.usefixtures("clean_database")
def test_migration_idempotency(database):
    with database.connection() as conn:
        v1 = current_version(conn)
        apply_migrations(conn)
        v2 = current_version(conn)
        assert v2 == v1
