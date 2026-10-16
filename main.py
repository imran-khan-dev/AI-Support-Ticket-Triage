from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from app.config import Settings
from app.schemas import TicketRequest, TriageResponse
from app.services.triage import get_groq_client, triage_ticket

app = FastAPI(
    title="AI Support Ticket Triage",
    version="0.1.0",
    description="A service to triage support tickets using AI. "
                "Groq API key is optional and can be set via GROQ_API_KEY environment variable.",
)

# Explicit, predictable dependencies: load settings once and resolve the Groq
# client up front. A missing GROQ_MODEL with a configured GROQ_API_KEY raises
# here with a clear configuration error rather than mid-request.
settings = Settings()
groq_client = get_groq_client(settings)


@app.post("/api/v1/tickets/triage", response_model=TriageResponse)
async def triage_ticket_endpoint(request: TicketRequest):
    """Triages a support ticket and returns the category, priority, sentiment, summary, and whether human review is required. Note: This endpoint uses a deterministic mock triage service when no Groq API key is configured, and uses the Groq API when a key is provided."""
    try:
        return triage_ticket(request, settings, groq_client)
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
