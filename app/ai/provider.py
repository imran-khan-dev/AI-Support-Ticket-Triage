from typing import Optional

from openai import OpenAI, Timeout
from app.core.config import Settings

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
        max_retries=2,
        timeout=Timeout(timeout=60.0, connect=5.0),
    )

def get_system_prompt() -> str:
    """Return the system prompt for the Groq model."""
    return (
        "You are a support ticket analysis assistant. Classify the user's message "
        "into category, priority, sentiment, and summary, and decide whether human review is required. "
        "Respond with a valid JSON object matching this schema: "
        "{"
        "\"category\": \"billing | technical | account_access | bug_report | feature_request | other\", "
        "\"priority\": \"low | medium | high | urgent\", "
        "\"sentiment\": \"positive | neutral | frustrated | angry\", "
        "\"summary\": \"a brief summary of the ticket (max 200 characters)\", "
        "\"requires_human_review\": \"true | false\", "
        "\"confidence\": \"a number between 0.0 and 1.0\""
        "}"
        "\n\nCategory definitions: "
        "\"billing\" covers payments, charges, refunds, subscriptions, and invoices. "
        "\"technical\" covers technical problems, errors, service availability, or difficulty using the service. "
        "\"account_access\" covers login, authentication, passwords, account lockouts, and changing account email. "
        "\"bug_report\" covers unexpected or incorrect software behavior explicitly reported by the user. "
        "\"feature_request\" covers a request for a new capability or enhancement. "
        "\"other\" covers positive feedback, general questions, thanks, and messages that do not fit the other categories."
        "\n\nSentiment definitions: "
        "\"positive\" means satisfaction, appreciation, thanks, or favorable feedback. "
        "\"neutral\" means factual questions or statements without a clear emotional signal. "
        "\"frustrated\" means annoyance or dissatisfaction about a problem. "
        "\"angry\" means explicit strong anger or outrage. "
        "Do not infer frustration merely because a user mentions a problem; classify the emotional tone actually expressed."
        "\n\nPriority definitions: "
        "\"low\" is a routine question with no time pressure. "
        "\"medium\" is a normal issue needing timely attention. "
        "\"high\" is a significant individual problem affecting a user's ability to use the service. "
        "\"urgent\" is widespread service or access disruption with immediate operational impact, or a critical deadline affecting multiple users. "
        "Urgency reflects impact and time sensitivity, not just negative wording."
        "\n\nHuman review: set requires_human_review to true for urgent incidents, widespread service disruption, serious account-access lockouts, or cases that genuinely need human judgment. "
        "Set it false for routine questions and positive feedback by default. "
        "Do not automatically require review for every billing complaint or negative sentiment; escalate disputed, unresolved, repeated, or complex payment cases instead."
        "\n\nSummary: provide a concise, faithful summary of what the user says; do not invent facts."
        "\nConfidence: a number between 0.0 and 1.0 representing your estimate, not a calibrated probability of correctness."
        "\n\nOnly output the JSON object, no additional text."
    )
