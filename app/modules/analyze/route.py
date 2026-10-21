from fastapi import APIRouter,  HTTPException
from pydantic import ValidationError
from app.core.config import Settings
from app.ai.provider import get_groq_client
from app.agent_core.analysis import analyze_ticket
from app.modules.analyze.schema import TicketRequest, AnalysisResponse


settings = Settings()
groq_client = get_groq_client(settings)

router = APIRouter(tags=["analyze"])


@router.post("", response_model=AnalysisResponse)
async def analyze_ticket_endpoint(request: TicketRequest):
    """Analyzes a support ticket and returns the category, priority, sentiment, summary, and whether human review is required. Note: This endpoint uses a deterministic mock analysis service when no Groq API key is configured, and uses the Groq API when a key is provided."""
    try:
        return analyze_ticket(request, settings, groq_client)
    except HTTPException:
        raise
    except ValidationError:
        raise HTTPException(
            status_code=502,
            detail="Provider response validation failed.",
        )
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Provider service is temporarily unavailable.",
        )


