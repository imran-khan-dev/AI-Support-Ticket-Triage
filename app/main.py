from fastapi import FastAPI

app = FastAPI(
    title="AI Support Ticket Triage",
    version="0.1.0",
    description="A service to triage support tickets using AI.",
)