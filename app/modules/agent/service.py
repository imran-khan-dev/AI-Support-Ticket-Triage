import uuid
from dataclasses import dataclass

from app.agent_core.agent import run_agent, AgentResult
from app.ai.provider import get_groq_client
from app.core.config import Settings
from app.database import get_database
from app.modules.agent.schema import TicketResolveRequest
from app.domain.agent.services import (
    prepare_session_store,
    resolve_session_id,
    load_session_history,
    load_session_memory,
    persist_agent_run,
    final_answer_message_id,
    store_run_memory,
)

settings = Settings()
groq_client = get_groq_client(settings)


@dataclass(frozen=True)
class AgentRunOutcome:
    """Result of a completed agent workflow."""
    result: AgentResult
    session_id: str
    transcript_stored: bool


def run_ticket_service(
    request: TicketResolveRequest,
) -> AgentRunOutcome:
    """Coordinate session setup, agent execution, and persistence."""
    run_id = str(uuid.uuid4())

    # 1. Prepare the database and resolve the session.
    database = get_database(settings.database_url)
    prepare_session_store(database)
    session_id = resolve_session_id(database, request.session_id)

    # 2. Load prior history and relevant memories.
    history = load_session_history(
        database,
        session_id,
        max_messages=settings.agent_max_history_messages,
        exclude_correlation_ids={run_id},
    )

    memory_items = load_session_memory(
        database,
        session_id,
        request.message,
        max_items=settings.agent_max_memory_items,
        max_chars=settings.agent_max_memory_chars,
    )

    agent_history = [*memory_items, *history]

    # 3. Run the agent.
    result = run_agent(
        request.message,
        settings,
        groq_client,
        history=agent_history,
    )

    # 4. Persist the transcript before attempting to store its memory.
    stored_messages = persist_agent_run(
        database,
        session_id,
        request.message,
        result,
        correlation_id=run_id,
    )

    transcript_stored = bool(stored_messages)

    # 5. Store memory only when a persisted final answer exists.
    answer_message_id = final_answer_message_id(stored_messages)

    if answer_message_id is not None:
        store_run_memory(
            database,
            session_id,
            result,
            answer_message_id,
        )

    # 6. Return the outcome; the controller handles HTTP responses.
    return AgentRunOutcome(
        result=result,
        session_id=session_id,
        transcript_stored=transcript_stored,
    )
