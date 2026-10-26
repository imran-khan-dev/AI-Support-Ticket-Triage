from fastapi import APIRouter, Depends

from app.modules.integrations.auth import require_integration

router = APIRouter(tags=["integrations"])


@router.get("/ping")
def integration_ping(integration=Depends(require_integration)):
    """Opt-in integration-protected endpoint for auth verification."""
    return {"status": "ok", "integration": integration.name}
