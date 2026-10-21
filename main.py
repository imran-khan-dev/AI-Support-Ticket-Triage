from fastapi import FastAPI
from app.router import router
import os
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="AI Support Ticket Analysis",
    version="0.1.0",
    description="A service to analyze support tickets using AI. "
                "Groq API key is optional and can be set via GROQ_API_KEY environment variable.",
)

app.include_router(router, prefix="/api/v1/tickets")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

_frontend_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
)
app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")

# python -m uvicorn main:app --reload