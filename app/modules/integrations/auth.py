from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings
from app.database import get_database, DatabaseConfigurationError
from app.database.repository import get_tenant
from app.modules.integrations.service import verify_integration_token

security = HTTPBearer(auto_error=False)


def _database():
    try:
        return get_database(Settings().database_url)
    except DatabaseConfigurationError:
        raise HTTPException(status_code=503, detail="Storage unavailable.")


def require_integration(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    database=Depends(_database),
):
    """Require a valid integration credential via Authorization: Bearer.

    Identity is derived from verified credentials only.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        result = verify_integration_token(database, credentials.credentials)
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if result is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    integration, credential = result
    try:
        if integration.tenant_id is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        tenant = get_tenant(database, integration.tenant_id)
        if tenant is None or not tenant.enabled:
            raise HTTPException(status_code=401, detail="Unauthorized")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized")
    request.state.integration = integration
    request.state.integration_credential = credential
    request.state.tenant = tenant
    return integration
