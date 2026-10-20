
from fastapi import APIRouter,  HTTPException
from app.agent_core.agent import run_agent

from app.core.config import Settings
from app.ai.provider import get_groq_client
from app.modules.agent.schema import TicketResolveRequest, TicketResolveResponse


settings = Settings()
groq_client = get_groq_client(settings)

router = APIRouter(tags=["agent"])


@router.post("/run", response_model=TicketResolveResponse)
async def resolve_ticket_endpoint(request: TicketResolveRequest):
    """Resolves a support ticket using the tool-using agent.

    Invokes the agent loop, which may call registered read-only tools
    (search_support_docs, lookup_ticket_history) before producing a final
    answer. Returns a structured response whose ``status`` field indicates
    the outcome and whose ``mock_mode`` field is true when no Groq API key
    is configured.
    """
    result = run_agent(request.message, settings, groq_client)

    if result.status == "mock":
        return TicketResolveResponse(
            status="mock",
            answer=result.answer,
            mock_mode=True,
            tool_calls=[],
        )

    if result.status == "success":
        return TicketResolveResponse(
            status="success",
            answer=result.answer,
            mock_mode=False,
            tool_calls=result.tool_calls,
        )

    if result.status == "provider_failure":
        raise HTTPException(
            status_code=502,
            detail="Provider service is temporarily unavailable.",
        )

    if result.status == "max_iterations_reached":
        return TicketResolveResponse(
            status="max_iterations_reached",
            answer=None,
            mock_mode=False,
            tool_calls=result.tool_calls,
            error=result.error,
        )

    raise HTTPException(
        status_code=500,
        detail="Agent service error.",
    )



