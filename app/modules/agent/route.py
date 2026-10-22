
import uuid

from fastapi import APIRouter,  HTTPException, Response
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
    load_session_history,
    persist_agent_run,
    prepare_session_store,
    resolve_session_id,
)


settings = Settings()
groq_client = get_groq_client(settings)

router = APIRouter(tags=["agent"])

#: Response header carrying the session id of the run.
#:
#: The session id is also part of the successful response body. The header
#: additionally makes it recoverable when a run ends in an error response,
#: where the body is kept unchanged.
SESSION_ID_HEADER = "X-Session-Id"

#: Response header reporting whether the transcript of the run was stored.
#:
#: The outcome is reported in a header rather than in the body, because the
#: body of this endpoint has a fixed field set. The answer is returned either
#: way, because the answer is the primary result of the request.
TRANSCRIPT_STORED_HEADER = "X-Transcript-Stored"


@router.post("/run", response_model=TicketResolveResponse)
async def resolve_ticket_endpoint(
    request: TicketResolveRequest,
    response: Response,
):
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

    A continuing run also receives the earlier turns of its session. The
    history is bounded by ``AGENT_MAX_HISTORY_MESSAGES``, and a tool call is
    never separated from its output. The message of the current run is not
    part of the history and is sent exactly once.

    The session id is returned in the ``X-Session-Id`` header as well as in
    the response body, so that a client can recover it even when the run ends
    in an error response. The ``X-Transcript-Stored`` header reports whether
    the transcript of the run was written.
    """
    # One identifier per run: it ties the transcript of this run to the
    # history of the session, so the messages of this run are never replayed
    # into it.
    run_id = str(uuid.uuid4())

    try:
        database = get_database(settings.database_url)
        prepare_session_store(database)
        session_id = resolve_session_id(database, request.session_id)
        history = load_session_history(
            database,
            session_id,
            max_messages=settings.agent_max_history_messages,
            exclude_correlation_ids={run_id},
        )
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

    # The session id is published as a header before the run starts, so that
    # it survives an error response whose body must stay unchanged.
    response.headers[SESSION_ID_HEADER] = session_id

    result = run_agent(request.message, settings, groq_client, history=history)

    # A transcript that cannot be stored must not discard the answer, so the
    # outcome is reported instead of raised.
    persisted = persist_agent_run(
        database,
        session_id,
        request.message,
        result,
        correlation_id=run_id,
    )
    response.headers[TRANSCRIPT_STORED_HEADER] = "true" if persisted else "false"

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
            headers={
                SESSION_ID_HEADER: session_id,
                TRANSCRIPT_STORED_HEADER: "true" if persisted else "false",
            },
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


