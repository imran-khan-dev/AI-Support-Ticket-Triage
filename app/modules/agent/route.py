
from fastapi import APIRouter,  HTTPException
from app.agent_core.agent import run_agent
from app.persistence import DatabaseConfigurationError, get_database

from app.core.config import Settings
from app.ai.provider import get_groq_client
from app.modules.agent.schema import TicketResolveRequest, TicketResolveResponse
from app.modules.agent.service import (
    InvalidSessionIdError,
    SessionClosedError,
    SessionNotFoundError,
    StorageError,
    persist_agent_run,
    prepare_session_store,
    resolve_session_id,
)


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

    The run is associated with a session: a new session is created when the
    request carries no ``session_id``, and the transcript of the run is stored
    under that session. A supplied ``session_id`` is used only when it
    identifies an existing, open session.
    """
    try:
        database = get_database(settings.database_url)
        prepare_session_store(database)
        session_id = resolve_session_id(database, request.session_id)
    except InvalidSessionIdError:
        raise HTTPException(
            status_code=422,
            detail="The session id is not a valid session identifier.",
        )
    except SessionNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="The session does not exist.",
        )
    except SessionClosedError:
        raise HTTPException(
            status_code=409,
            detail="The session is closed.",
        )
    except DatabaseConfigurationError:
        raise HTTPException(
            status_code=503,
            detail="Session storage is temporarily unavailable.",
        )
    except StorageError:
        raise HTTPException(
            status_code=503,
            detail="Session storage is temporarily unavailable.",
        )

    result = run_agent(request.message, settings, groq_client)

    persist_agent_run(database, session_id, request.message, result)

    if result.status == "mock":
        return TicketResolveResponse(
            status="mock",
            answer=result.answer,
            mock_mode=True,
            tool_calls=[],
            session_id=session_id,
        )

    if result.status == "success":
        return TicketResolveResponse(
            status="success",
            answer=result.answer,
            mock_mode=False,
            tool_calls=result.tool_calls,
            session_id=session_id,
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
            session_id=session_id,
        )

    raise HTTPException(
        status_code=500,
        detail="Agent service error.",
    )


