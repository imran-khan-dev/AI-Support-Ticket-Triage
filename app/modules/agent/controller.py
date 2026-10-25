import logging
from fastapi import HTTPException, Response

from app.database import DatabaseConfigurationError
from app.modules.agent.schema import (
    TicketResolveRequest,
    TicketResolveResponse,
)
from app.modules.agent.service import (
    run_ticket_service,
)

from app.domain.agent.services import (
    InvalidSessionIdError,
    SessionClosedError,
    SessionNotFoundError,
    StorageError,
)

logger = logging.getLogger("agent.controller")

SESSION_ID_HEADER = "X-Session-Id"
TRANSCRIPT_STORED_HEADER = "X-Transcript-Stored"


async def resolve_ticket_controller(
    request: TicketResolveRequest,
    response: Response,
) -> TicketResolveResponse:
    """Coordinate the service call and translate outcomes into HTTP responses."""
    try:
        outcome = run_ticket_service(request)

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

    except (DatabaseConfigurationError, StorageError):
        raise HTTPException(
            status_code=503,
            detail="Session storage is temporarily unavailable.",
        )

    response.headers[SESSION_ID_HEADER] = outcome.session_id
    response.headers[TRANSCRIPT_STORED_HEADER] = (
        "true" if outcome.transcript_stored else "false"
    )

    result = outcome.result

    if result.status == "mock":
        return TicketResolveResponse(
            status="mock",
            answer=result.answer,
            mock_mode=True,
            tool_calls=[],
            session_id=outcome.session_id,
        )

    if result.status == "success":
        return TicketResolveResponse(
            status="success",
            answer=result.answer,
            mock_mode=False,
            tool_calls=result.tool_calls,
            session_id=outcome.session_id,
        )

    if result.status == "provider_failure":
        raise HTTPException(
            status_code=502,
            detail="Provider service is temporarily unavailable.",
            headers={
                SESSION_ID_HEADER: outcome.session_id,
                TRANSCRIPT_STORED_HEADER: (
                    "true" if outcome.transcript_stored else "false"
                ),
            },
        )

    if result.status == "max_iterations_reached":
        return TicketResolveResponse(
            status="max_iterations_reached",
            answer=None,
            mock_mode=False,
            tool_calls=result.tool_calls,
            error=result.error,
            session_id=outcome.session_id,
        )

    logger.error("Unexpected agent status: %s", result.status)

    raise HTTPException(
        status_code=500,
        detail="Agent service error.",
    )
