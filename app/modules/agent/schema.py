from pydantic import BaseModel, Field, field_validator
from typing import Literal

class TicketRequest(BaseModel):
    """Request model for ticket analysis."""
    message: str = Field(
        ...,
        description="The customer message to be analyzed.",
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

class AnalysisResponse(BaseModel):
    """Response model for ticket analysis."""
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

    confidence: float = Field(
        ...,
        description="Model-estimated confidence score for the analysis output "
        "(0.0 to 1.0, not a calibrated probability of correctness).",
        ge=0.0,
        le=1.0,
    )

    @field_validator('summary', mode='before')
    @classmethod
    def strip_summary(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value


class TicketResolveRequest(BaseModel):
    """Request model for ticket resolution via the tool-using agent."""

    message: str = Field(
        ...,
        description="The customer message to resolve.",
        min_length=1,
        max_length=10000,
        examples=["I cannot log into my account."],
    )

    session_id: str | None = Field(
        default=None,
        description="Optional identifier of an existing session to continue. "
        "When omitted, the server creates a new session. A supplied id must "
        "identify an existing, open session; it is never trusted as-is.",
    )

    @field_validator('message', mode='before')
    @classmethod
    def strip_message(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value


class TicketResolveResponse(BaseModel):
    """Response model for ticket resolution via the tool-using agent."""

    status: str = Field(
        ...,
        description="Outcome of the agent run: success, mock, provider_failure, or max_iterations_reached.",
    )
    answer: str | None = Field(
        default=None,
        description="The agent's final answer, when available.",
    )
    mock_mode: bool = Field(
        default=False,
        description="True when no Groq API key is configured and no LLM call was made.",
    )
    tool_calls: list = Field(
        default_factory=list,
        description="Records of each tool call made during the agent run.",
    )
    error: str | None = Field(
        default=None,
        description="A safe, user-facing error message, when applicable.",
    )
    session_id: str | None = Field(
        default=None,
        description="Identifier of the session the run belongs to. The session "
        "holds the stored transcript of the run.",
    )