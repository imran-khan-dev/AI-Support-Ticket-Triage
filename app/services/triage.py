import json
from typing import Optional

from groq import Groq
from pydantic import ValidationError

from app.schemas import TicketRequest, TriageResponse
from app.config import Settings


class TriageService:
    """Triage service that uses Groq when API key is available, otherwise returns a mock response.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Optional[Groq] = None
        if self.settings.groq_api_key:
            self._client = Groq(
                api_key=self.settings.groq_api_key,
                base_url=self.settings.groq_base_url
            )

    def triage(self, request: TicketRequest) -> TriageResponse:
        """Triages a support ticket request.
        If Groq API key is configured, uses Groq to classify the ticket.
        Otherwise, returns a deterministic mock response.
        Args:
            request: A validated TicketRequest.
        Returns:
            A TriageResponse.
        """
        if not self._client:
            return self._mock_triage()

        try:
            completion = self._client.chat.completions.create(
                model=self.settings.groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": request.message,
                    },
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            response_text = completion.choices[0].message.content
            response_data = json.loads(response_text)
            return TriageResponse(**response_data)
        except Exception as e:
            # In a production setting, we might want to log the error and return a fallback.
            # However, the requirement is not to silently hide Groq failures behind mock responses.
            # We will re-raise the exception so that the endpoint returns a 500 error.
            raise e

    def _mock_triage(self) -> TriageResponse:
        """Return a deterministic mock triage response."""
        return TriageResponse(
            category="technical",
            priority="medium",
            sentiment="neutral",
            summary="This is a mock summary.",
            requires_human_review=False,
        )

    def _get_system_prompt(self) -> str:
        """Return the system prompt for the Groq model."""
        return (
            "You are a support ticket triage assistant. Your task is to classify the user's message "
            "into a category, priority, sentiment, and provide a summary. "
            "You must respond with a valid JSON object that matches the following schema: "
            "{"
            "\"category\": \"billing | technical | account_access | bug_report | feature_request | other\", "
            "\"priority\": \"low | medium | high | urgent\", "
            "\"sentiment\": \"positive | neutral | frustrated | angry\", "
            "\"summary\": \"a brief summary of the ticket (max 200 characters)\", "
            "\"requires_human_review\": \"true | false\""
            "}"
            "Only output the JSON object, no additional text."
        )