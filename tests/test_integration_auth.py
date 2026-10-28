import os

os.environ.setdefault("APPROVER_TOKEN_SECRET", "test-only-secret-for-approver-credentials-0123456789")

import uuid

import pytest
from fastapi.testclient import TestClient

from main import app
from app.core.security import generate_integration_token, hash_integration_token
from app.database import get_database
from app.modules.integrations.service import ensure_integration, issue_integration_credential


def _assign_tenant_to_integration(db, integration_id):
    from app.database.repository import _canonical_uuid
    tenant_id = str(uuid.uuid4())
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO tenants (id, name, enabled) VALUES (%s, %s, TRUE) ON CONFLICT (id) DO NOTHING",
            (_canonical_uuid(tenant_id), f"test-tenant-{tenant_id[:8]}"),
        )
        conn.execute(
            "UPDATE integrations SET tenant_id = %s WHERE id = %s",
            (_canonical_uuid(tenant_id), _canonical_uuid(integration_id)),
        )
    return tenant_id


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.usefixtures("clean_database")
def test_integration_ping_requires_auth(client):
    resp = client.get("/api/v1/tickets/integrations/ping")
    assert resp.status_code == 401


@pytest.mark.usefixtures("clean_database")
def test_integration_auth_valid_token(client):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(db, integration_name="test-integ")
    _assign_tenant_to_integration(db, integration.id)
    resp = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["integration"] == "test-integ"


@pytest.mark.usefixtures("clean_database")
def test_integration_auth_invalid_token(client):
    resp = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": "Bearer invalidtokeninvalidtokeninvalidtoken"})
    assert resp.status_code == 401


@pytest.mark.usefixtures("clean_database")
def test_integration_auth_revoked_token(client):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(db, integration_name="test-integ2")
    _assign_tenant_to_integration(db, integration.id)
    from app.database.repository import revoke_integration_credential
    revoke_integration_credential(db, credential.id)
    resp = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


@pytest.mark.usefixtures("clean_database")
def test_integration_auth_rotation_both_valid(client):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential1, token1 = issue_integration_credential(db, integration_name="test-integ3")
    integration, credential2, token2 = issue_integration_credential(db, integration_name="test-integ3")
    _assign_tenant_to_integration(db, integration.id)
    resp1 = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token1}"})
    resp2 = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token2}"})
    assert resp1.status_code == 200
    assert resp2.status_code == 200


@pytest.mark.usefixtures("clean_database")
def test_integration_auth_caller_supplied_identity_ignored(client):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(db, integration_name="test-integ4")
    _assign_tenant_to_integration(db, integration.id)
    resp = client.get(
        "/api/v1/tickets/integrations/ping",
        headers={"Authorization": f"Bearer {token}", "X-Integration-Id": "spoof"},
    )
    assert resp.status_code == 200
    assert resp.json()["integration"] == "test-integ4"


@pytest.mark.usefixtures("clean_database")
def test_integration_auth_expired_credential(client):
    from datetime import datetime, timedelta, timezone
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(
        db, integration_name="test-integ-expired", expires_at=datetime.now(timezone.utc) - timedelta(hours=1)
    )
    _assign_tenant_to_integration(db, integration.id)
    resp = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


@pytest.mark.usefixtures("clean_database")
def test_integration_auth_disabled_integration(client):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(db, integration_name="test-integ-disabled")
    _assign_tenant_to_integration(db, integration.id)
    from app.database.repository import _canonical_uuid
    with db.write_transaction() as conn:
        conn.execute("UPDATE integrations SET enabled = FALSE WHERE id = %s", (_canonical_uuid(integration.id),))
    resp = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


@pytest.mark.usefixtures("clean_database")
def test_integration_auth_disabled_credential(client):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(db, integration_name="test-integ-cred-disabled")
    _assign_tenant_to_integration(db, integration.id)
    from app.database.repository import _canonical_uuid
    with db.write_transaction() as conn:
        conn.execute("UPDATE integration_credentials SET enabled = FALSE WHERE id = %s", (_canonical_uuid(credential.id),))
    resp = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


@pytest.mark.usefixtures("clean_database")
def test_integration_tenant_none_rejected(client):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(db, integration_name="test-integ-no-tenant")
    resp = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


@pytest.mark.usefixtures("clean_database")
def test_integration_tenant_missing_rejected(client):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(db, integration_name="test-integ-missing-tenant")
    from app.database.repository import _canonical_uuid
    tenant_id = str(uuid.uuid4())
    other_tenant_id = str(uuid.uuid4())
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO tenants (id, name, enabled) VALUES (%s, %s, TRUE), (%s, %s, TRUE)",
            (_canonical_uuid(tenant_id), f"temp-tenant-{tenant_id[:8]}", _canonical_uuid(other_tenant_id), f"other-tenant-{other_tenant_id[:8]}"),
        )
        conn.execute(
            "UPDATE integrations SET tenant_id = %s WHERE id = %s",
            (_canonical_uuid(tenant_id), _canonical_uuid(integration.id)),
        )
        conn.execute("ALTER TABLE integrations DROP CONSTRAINT integrations_tenant_id_fkey")
        conn.execute(
            "UPDATE integrations SET tenant_id = %s WHERE id = %s",
            (_canonical_uuid(other_tenant_id), _canonical_uuid(integration.id)),
        )
        conn.execute("DELETE FROM tenants WHERE id = %s", (_canonical_uuid(other_tenant_id),))
        conn.execute("ALTER TABLE integrations ADD CONSTRAINT integrations_tenant_id_fkey FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE RESTRICT")
    resp = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


@pytest.mark.usefixtures("clean_database")
def test_integration_tenant_missing_rejected(client, monkeypatch):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(db, integration_name="test-integ-missing-tenant")
    _assign_tenant_to_integration(db, integration.id)
    import app.modules.integrations.auth as auth_mod
    def fake_get_tenant(*args, **kwargs):
        return None
    monkeypatch.setattr(auth_mod, "get_tenant", fake_get_tenant)
    resp = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401



@pytest.mark.usefixtures("clean_database")
def test_integration_tenant_override_ignored(client):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(db, integration_name="test-integ-override")
    _assign_tenant_to_integration(db, integration.id)
    resp = client.get(
        "/api/v1/tickets/integrations/ping",
        headers={"Authorization": f"Bearer {token}", "X-Tenant-Id": str(uuid.uuid4())},
    )
    assert resp.status_code == 200
    assert resp.json()["integration"] == "test-integ-override"


@pytest.mark.usefixtures("clean_database")
def test_integration_auth_db_error_during_verification(client, monkeypatch):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(db, integration_name="test-integ-db-err")
    _assign_tenant_to_integration(db, integration.id)
    import app.modules.integrations.service as svc

    def boom(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(svc, "get_integration_credential_by_hash", boom)
    resp = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Unauthorized"


@pytest.mark.usefixtures("clean_database")
def test_integration_auth_db_error_during_tenant_lookup(client, monkeypatch):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(db, integration_name="test-integ-db-err-tenant")
    _assign_tenant_to_integration(db, integration.id)
    import app.modules.integrations.auth as auth_mod

    def boom(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(auth_mod, "get_tenant", boom)
    resp = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Unauthorized"
