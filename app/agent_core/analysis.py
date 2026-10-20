import json
from typing import Optional

from openai import OpenAI

from app.core.config import Settings
from app.modules.analyze.schema import TicketRequest, AnalysisResponse
from app.ai.provider import get_system_prompt


def mock_analysis() -> AnalysisResponse:
    """Return a deterministic mock analysis response."""
    return AnalysisResponse(
        category="technical",
        priority="medium",
        sentiment="neutral",
        summary="This is a mock summary.",
        requires_human_review=False,
        confidence=0.85,
    )


def analyze_ticket(
    request: TicketRequest,
    settings: Settings,
    client: Optional[OpenAI] = None,
) -> AnalysisResponse:
    """Analyzes a support ticket request.

    Uses the Groq Responses API (via the OpenAI SDK) when a client is provided;
    otherwise returns a deterministic mock response. Provider and validation
    errors are propagated to the caller.
    """
    if client is None:
        return mock_analysis()

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
    return AnalysisResponse(**response_data)
