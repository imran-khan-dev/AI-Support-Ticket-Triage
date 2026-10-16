import json
import pytest
from unittest.mock import Mock, patch
from pydantic import ValidationError

from app.schemas import TicketRequest, TriageResponse
from app.config import Settings
from app.services.triage import get_groq_client, mock_triage, triage_ticket, get_system_prompt


def test_triage_uses_mock_when_no_api_key():
    """Test that triage returns a mock response when no API key is set."""
    settings = Settings(_env_file=None)  # Do not load from .env file
    client = get_groq_client(settings)
    assert client is None
    request = TicketRequest(message="Hello")
    response = triage_ticket(request, settings, client)
    assert response == mock_triage()
    assert 0.0 <= response.confidence <= 1.0


def test_triage_handles_missing_confidence_from_groq():
    """Test that triage raises ValidationError when Groq omits confidence."""
    with patch('app.services.triage.OpenAI') as MockOpenAI:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = '{"category": "billing", "priority": "high", "sentiment": "frustrated", "summary": "Test summary", "requires_human_review": true}'
        mock_client.responses.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        settings = Settings(_env_file=None, groq_api_key="test_key", groq_model="test_model")
        client = get_groq_client(settings)
        request = TicketRequest(message="Hello")
        with pytest.raises(ValidationError):
            triage_ticket(request, settings, client)


def test_triage_calls_groq_when_api_key_set():
    """Test that triage calls Groq Responses API when API key is set."""
    with patch('app.services.triage.OpenAI') as MockOpenAI:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = '{"category": "billing", "priority": "high", "sentiment": "frustrated", "summary": "Test summary", "requires_human_review": true, "confidence": 0.88}'
        mock_client.responses.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        settings = Settings(_env_file=None, groq_api_key="test_key", groq_model="test_model")
        client = get_groq_client(settings)
        request = TicketRequest(message="Hello")
        response = triage_ticket(request, settings, client)

        # Check that the Responses API was called with the expected arguments
        mock_client.responses.create.assert_called_once()
        args, kwargs = mock_client.responses.create.call_args
        assert kwargs['model'] == 'test_model'
        assert kwargs['instructions'] == get_system_prompt()
        assert kwargs['input'][0]['role'] == 'user'
        assert kwargs['input'][0]['content'] == 'Hello'
        assert kwargs['temperature'] == 0.0
        assert kwargs['text'] == {'format': {'type': 'json_object'}}
        # Check that the response is the validated one
        assert response.category == "billing"
        assert response.priority == "high"
        assert response.sentiment == "frustrated"
        assert response.summary == "Test summary"
        assert response.requires_human_review is True
        assert response.confidence == 0.88


def test_triage_uses_custom_base_url():
    """Test that get_groq_client uses the configured base URL."""
    with patch('app.services.triage.OpenAI') as MockOpenAI:
        mock_client = Mock()
        MockOpenAI.return_value = mock_client

        settings = Settings(_env_file=None, groq_api_key="test_key", groq_model="test_model", groq_base_url="https://custom.example.com")
        client = get_groq_client(settings)
        MockOpenAI.assert_called_once_with(api_key="test_key", base_url="https://custom.example.com")


def test_triage_handles_invalid_json_from_groq():
    """Test that triage raises when Groq returns invalid JSON."""
    with patch('app.services.triage.OpenAI') as MockOpenAI:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = 'invalid json'
        mock_client.responses.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        settings = Settings(_env_file=None, groq_api_key="test_key", groq_model="test_model")
        client = get_groq_client(settings)
        request = TicketRequest(message="Hello")
        with pytest.raises(Exception):  # json.loads will raise JSONDecodeError
            triage_ticket(request, settings, client)


def test_triage_handles_missing_fields_from_groq():
    """Test that triage raises ValidationError when Groq returns missing fields."""
    with patch('app.services.triage.OpenAI') as MockOpenAI:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = '{"category": "billing", "priority": "high"}'  # missing sentiment, summary, requires_human_review
        mock_client.responses.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        settings = Settings(_env_file=None, groq_api_key="test_key", groq_model="test_model")
        client = get_groq_client(settings)
        request = TicketRequest(message="Hello")
        with pytest.raises(ValidationError):
            triage_ticket(request, settings, client)


def test_triage_propagates_groq_errors():
    """Test that triage propagates errors from the Groq client."""
    with patch('app.services.triage.OpenAI') as MockOpenAI:
        mock_client = Mock()
        mock_client.responses.create.side_effect = Exception("Groq error")
        MockOpenAI.return_value = mock_client

        settings = Settings(_env_file=None, groq_api_key="test_key", groq_model="test_model")
        client = get_groq_client(settings)
        request = TicketRequest(message="Hello")
        with pytest.raises(Exception, match="Groq error"):
            triage_ticket(request, settings, client)


def test_get_groq_client_raises_when_api_key_set_but_model_missing():
    """Test that a clear error is raised when API key is set but model is missing."""
    settings = Settings(_env_file=None, groq_api_key="test_key")
    with pytest.raises(ValueError, match="GROQ_MODEL must be set"):
        get_groq_client(settings)


def test_get_system_prompt_defines_categories():
    """The system prompt defines every supported category."""
    prompt = get_system_prompt()
    for category in (
        "billing",
        "technical",
        "account_access",
        "bug_report",
        "feature_request",
        "other",
    ):
        assert category in prompt
    assert "payments" in prompt
    assert "changing account email" in prompt


def test_get_system_prompt_defines_sentiment():
    """The system prompt defines every supported sentiment, including positive feedback."""
    prompt = get_system_prompt()
    for sentiment in ("positive", "neutral", "frustrated", "angry"):
        assert sentiment in prompt
    assert "appreciation" in prompt or "thanks" in prompt
    assert "Do not infer frustration merely because a user mentions a problem" in prompt


def test_get_system_prompt_defines_priority():
    """The system prompt defines every supported priority level."""
    prompt = get_system_prompt()
    for priority in ("low", "medium", "high", "urgent"):
        assert priority in prompt


def test_get_system_prompt_defines_human_review_rules():
    """The system prompt guides when human review is required."""
    prompt = get_system_prompt()
    assert "requires_human_review" in prompt
    assert "positive feedback" in prompt
    assert "urgent" in prompt


def test_triage_passes_refined_prompt_as_instructions():
    """The service passes get_system_prompt() as the Responses API instructions argument."""
    with patch('app.services.triage.OpenAI') as MockOpenAI:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = '{"category": "other", "priority": "low", "sentiment": "positive", "summary": "Thanks for the help.", "requires_human_review": false, "confidence": 0.9}'
        mock_client.responses.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        settings = Settings(_env_file=None, groq_api_key="test_key", groq_model="test_model")
        client = get_groq_client(settings)
        request = TicketRequest(message="Your support team solved my problem quickly. Thank you!")
        triage_ticket(request, settings, client)

        mock_client.responses.create.assert_called_once()
        args, kwargs = mock_client.responses.create.call_args
        assert kwargs['instructions'] == get_system_prompt()
