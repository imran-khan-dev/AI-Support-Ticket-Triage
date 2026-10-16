import json
from typing import Optional

from openai import OpenAI

from app.config import Settings
from app.schemas import TicketRequest, TriageResponse


def get_groq_client(settings: Settings) -> Optional[OpenAI]:
    """Initialize and return an OpenAI client configured for Groq, or None for mock mode.

    Raises ValueError when GROQ_API_KEY is configured without GROQ_MODEL so the
    misconfiguration is surfaced clearly instead of failing at request time.
    """
    if not settings.groq_api_key:
        return None
    if not settings.groq_model:
        raise ValueError(
            "GROQ_MODEL must be set when GROQ_API_KEY is provided"
        )
    return OpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
    )


def mock_triage() -> TriageResponse:
    """Return a deterministic mock triage response."""
    return TriageResponse(
        category="technical",
        priority="medium",
        sentiment="neutral",
        summary="This is a mock summary.",
        requires_human_review=False,
        confidence=0.85,
    )


def get_system_prompt() -> str:
    """Return the system prompt for the Groq model."""
    return (
        "You are a support ticket triage assistant. Your task is to classify the user's message "
        "into a category, priority, sentiment, and provide a summary. "
        "You must respond with a valid JSON object that matches the following schema: "
        "{"
        "\"category\": \"billing | technical | account_access | bug_report | feature_request | other\", "
        "\"priority\": \"low | medium | high | urgent\", "
        "\"sentiment\": \"positive | neutral | frustrated | angry\", "
        "\"summary\": \"a brief summary of the ticket (max 200 characters)\", "
        "\"requires_human_review\": \"true | false\", "
        "\"confidence\": \"a number between 0.0 and 1.0\""
        "}"
        "Only output the JSON object, no additional text."
    )


def triage_ticket(
    request: TicketRequest,
    settings: Settings,
    client: Optional[OpenAI] = None,
) -> TriageResponse:
    """Triages a support ticket request.

    Uses the Groq Responses API (via the OpenAI SDK) when a client is provided;
    otherwise returns a deterministic mock response. Provider and validation
    errors are propagated to the caller.
    """
    if client is None:
        return mock_triage()

    response = client.responses.create(
        model=settings.groq_model,
        instructions=get_system_prompt(),
        input=[
            {
                "role": "user",
                "content": request.message,
            },
        ],
        temperature=0.0,
        text={"format": {"type": "json_object"}},
    )
    response_text = response.output_text
    response_data = json.loads(response_text)
    return TriageResponse(**response_data)
