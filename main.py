from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from app.config import Settings
from app.schemas import TicketRequest, AnalysisResponse
from app.services.analysis import get_groq_client, analyze_ticket

app = FastAPI(
    title="AI Support Ticket Analysis",
    version="0.1.0",
    description="A service to analyze support tickets using AI. "
                "Groq API key is optional and can be set via GROQ_API_KEY environment variable.",
)

# Explicit, predictable dependencies: load settings once and resolve the Groq
# client up front. A missing GROQ_MODEL with a configured GROQ_API_KEY raises
# here with a clear configuration error rather than mid-request.
settings = Settings()
groq_client = get_groq_client(settings)


@app.post("/api/v1/tickets/triage", response_model=AnalysisResponse)
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# uvicorn main:app --reload
