import os

os.environ.setdefault("APPROVER_TOKEN_SECRET", "test-only-secret-for-approver-credentials-0123456789")

import uuid

import pytest
from fastapi.testclient import TestClient

from main import app
from app.core.security import generate_integration_token, hash_integration_token
from app.database import get_database
from app.modules.integrations.service import ensure_integration, issue_integration_credential


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
    from app.database.repository import revoke_integration_credential
    revoke_integration_credential(db, credential.id)
    resp = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


@pytest.mark.usefixtures("clean_database")
def test_integration_auth_rotation_both_valid(client):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential1, token1 = issue_integration_credential(db, integration_name="test-integ3")
    integration, credential2, token2 = issue_integration_credential(db, integration_name="test-integ3")
    resp1 = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token1}"})
    resp2 = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token2}"})
    assert resp1.status_code == 200
    assert resp2.status_code == 200


@pytest.mark.usefixtures("clean_database")
def test_integration_auth_caller_supplied_identity_ignored(client):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(db, integration_name="test-integ4")
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
    resp = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


@pytest.mark.usefixtures("clean_database")
def test_integration_auth_disabled_integration(client):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(db, integration_name="test-integ-disabled")
    from app.database.repository import _canonical_uuid
    with db.write_transaction() as conn:
        conn.execute("UPDATE integrations SET enabled = FALSE WHERE id = %s", (_canonical_uuid(integration.id),))
    resp = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


@pytest.mark.usefixtures("clean_database")
def test_integration_auth_disabled_credential(client):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(db, integration_name="test-integ-cred-disabled")
    from app.database.repository import _canonical_uuid
    with db.write_transaction() as conn:
        conn.execute("UPDATE integration_credentials SET enabled = FALSE WHERE id = %s", (_canonical_uuid(credential.id),))
    resp = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


@pytest.mark.usefixtures("clean_database")
def test_integration_auth_db_error_during_verification(client, monkeypatch):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(db, integration_name="test-integ-db-err")
    import app.modules.integrations.service as svc
    def boom(*args, **kwargs):
        raise RuntimeError("db down")
    monkeypatch.setattr(svc, "get_integration_credential_by_hash", boom)
    resp = client.get("/api/v1/tickets/integrations/ping", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
