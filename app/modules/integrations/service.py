"""Integration authentication service."""
from __future__ import annotations

from datetime import datetime, timezone

from app.core.security import (
    CredentialError,
    generate_integration_token,
    hash_integration_token,
    validate_token,
)
from app.database.db import Database
from app.database.models import Integration, IntegrationCredential
from app.database.repository import (
    create_integration,
    create_integration_credential,
    get_integration,
    get_integration_by_name,
    get_integration_credential_by_hash,
    revoke_integration_credential,
    touch_integration_credential,
)


class IntegrationAuthError(RuntimeError):
    """Base error for integration auth."""


class IntegrationNotFoundError(IntegrationAuthError):
    pass


class IntegrationCredentialError(IntegrationAuthError):
    pass


def ensure_integration(db: Database, name: str) -> Integration:
    integration = get_integration_by_name(db, name)
    if integration is None:
        integration = create_integration(db, name=name)
    return integration


def issue_integration_credential(
    db: Database,
    *,
    integration_name: str,
    expires_at: datetime | None = None,
) -> tuple[Integration, IntegrationCredential, str]:
    """Issue a new credential. Returns (integration, credential_record, plaintext_token)."""
    integration = ensure_integration(db, name=integration_name)
    token, key_id = generate_integration_token()
    token_hash = hash_integration_token(token)
    credential = create_integration_credential(
        db,
        integration_id=integration.id,
        token_hash=token_hash,
        key_id=key_id,
        expires_at=expires_at,
    )
    return integration, credential, token


def verify_integration_token(
    db: Database,
    token: object,
) -> tuple[Integration, IntegrationCredential] | None:
    """Verify bearer token. Returns (integration, credential) or None."""
    try:
        validated = validate_token(token)
    except CredentialError:
        return None
    try:
        token_hash = hash_integration_token(validated)
    except Exception:
        return None
    credential = get_integration_credential_by_hash(db, token_hash)
    if credential is None:
        return None
    if not credential.enabled or credential.revoked_at is not None:
        return None
    if credential.expires_at is not None:
        try:
            exp_str = credential.expires_at
            exp_dt = None
            try:
                # Try fromisoformat first (handles most ISO formats)
                exp_dt = datetime.fromisoformat(exp_str.replace('Z', '+00:00'))
            except Exception:
                try:
                    from dateutil import parser

                    exp_dt = parser.parse(exp_str)
                except Exception:
                    return None  # fail closed on unparsable expiry
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=timezone.utc)
            else:
                exp_dt = exp_dt.astimezone(timezone.utc)
            if exp_dt < datetime.now(timezone.utc):
                return None
        except Exception:
            return None  # fail closed
    integration = get_integration(db, credential.integration_id)
    if integration is None or not integration.enabled:
        return None
    try:
        touch_integration_credential(db, credential.id)
    except Exception:
        pass
    return integration, credential


def revoke_credential(db: Database, credential_id: str) -> bool:
    return revoke_integration_credential(db, credential_id)
