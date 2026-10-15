from app.schemas import TicketRequest, TriageResponse

class TriageService:
    """Deterministic mock triage service for testing.
    This service returns a fixed response regardless of input.
    It does not use an LLM or external APIs.
    """
    def triage(self, request: TicketRequest) -> TriageResponse:
        """Triages a support ticket request with a mock response.
        Args:
            request: A validated TicketRequest.
        Returns:
            A TriageResponse with fixed values.
        """
        return TriageResponse(
            category="technical",
            priority="medium",
            sentiment="neutral",
            summary="This is a mock summary.",
            requires_human_review=False,
        )
