from fastapi import FastAPI
from app.schemas import TicketRequest, TriageResponse
from app.services.triage import TriageService
from app.config import Settings

app = FastAPI(
    title="AI Support Ticket Triage",
    version="0.1.0",
    description="A service to triage support tickets using AI. "
                "Groq API key is optional and can be set via GROQ_API_KEY environment variable.",
)

# Instantiate the triage service
triage_service = TriageService()

# Instantiate settings (for potential use in dependencies or future features)
settings = Settings()

@app.post("/api/v1/tickets/triage", response_model=TriageResponse)
async def triage_ticket(request: TicketRequest):
    """
    Triages a support ticket and returns the category, priority, sentiment, summary,
    and whether human review is required.
    Note: This endpoint uses a deterministic mock triage service and does not
    invoke any LLM or external API.
    """
    return triage_service.triage(request)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

