from pydantic import BaseModel, Field, field_validator
from typing import Literal

class TicketRequest(BaseModel):
    """Request model for ticket triage."""
    message: str = Field(
        ...,
        description="The customer message to be triaged.",
        min_length=1,
        max_length=10000,
        examples=["I cannot log into my account."],
    )

    @field_validator('message', mode='before')
    @classmethod
    def strip_message(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value

class TriageResponse(BaseModel):
    """Response model for ticket triage."""
    category: Literal[
        "billing",
        "technical",
        "account_access",
        "bug_report",
        "feature_request",
        "other",
    ] = Field(..., description="The category of the ticket.")
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        ..., description="The priority of the ticket."
    )
    sentiment: Literal["positive", "neutral", "frustrated", "angry"] = Field(
        ..., description="The sentiment of the customer message."
    )
    summary: str = Field(
        ...,
        description="A brief summary of the ticket.",
        min_length=1,
        max_length=200,
        examples=["Customer unable to log in."],
    )
    requires_human_review: bool = Field(
        ..., description="Whether the ticket requires human review."
    )

    @field_validator('summary', mode='before')
    @classmethod
    def strip_summary(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value
