from fastapi import APIRouter, Depends, Request, Response
from app.modules.agent.controller import resolve_ticket_controller
from app.modules.agent.schema import (
    TicketResolveRequest,
    TicketResolveResponse,
)
from app.modules.integrations.auth import require_integration

router = APIRouter(tags=["agent"])

@router.post("/run", response_model=TicketResolveResponse)

async def resolve_ticket_endpoint(
    request: TicketResolveRequest,
    response: Response,
    http_request: Request,
    integration=Depends(require_integration),
) -> TicketResolveResponse:
    
    """Resolve a support ticket using the agent controller."""
    return await resolve_ticket_controller(request, response, http_request)
