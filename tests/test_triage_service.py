import pytest
from unittest.mock import Mock, patch

from app.schemas import TicketRequest, TriageResponse
from app.services.triage import TriageService
from app.config import Settings


def test_triage_service_uses_mock_when_no_api_key():
    """Test that TriageService returns mock response when no API key is set."""
    settings = Settings(_env_file=None)  # Do not load from .env file
    service = TriageService(settings)
    request = TicketRequest(message="Hello")
    response = service.triage(request)
    assert response == service._mock_triage()
    assert 0.0 <= response.confidence <= 1.0


def test_triage_service_handles_missing_confidence_from_groq():
    """Test that TriageService raises ValidationError when Groq omits confidence."""
    with patch('app.services.triage.Groq') as MockGroq:
        mock_client = Mock()
        mock_completion = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = '{"category": "billing", "priority": "high", "sentiment": "frustrated", "summary": "Test summary", "requires_human_review": true}'
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        MockGroq.return_value = mock_client

        settings = Settings(_env_file=None, groq_api_key="test_key")
        service = TriageService(settings)
        request = TicketRequest(message="Hello")
        with pytest.raises(Exception):  # Pydantic will raise a ValidationError
            service.triage(request)


def test_triage_service_calls_groq_when_api_key_set():
    """Test that TriageService calls Groq client when API key is set."""
    with patch('app.services.triage.Groq') as MockGroq:
        mock_client = Mock()
        mock_completion = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = '{"category": "billing", "priority": "high", "sentiment": "frustrated", "summary": "Test summary", "requires_human_review": true, "confidence": 0.88}'
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        MockGroq.return_value = mock_client

        settings = Settings(_env_file=None, groq_api_key="test_key", groq_model="test_model")
        service = TriageService(settings)
        request = TicketRequest(message="Hello")
        response = service.triage(request)

        # Check that the client was called with the expected arguments
        mock_client.chat.completions.create.assert_called_once()
        args, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs['model'] == 'test_model'
        assert kwargs['messages'][0]['role'] == 'system'
        assert kwargs['messages'][1]['role'] == 'user'
        assert kwargs['messages'][1]['content'] == 'Hello'
        assert kwargs['temperature'] == 0.0
        assert kwargs['response_format'] == {'type': 'json_object'}
        # Check that the response is the validated one
        assert response.category == "billing"
        assert response.priority == "high"
        assert response.sentiment == "frustrated"
        assert response.summary == "Test summary"
        assert response.requires_human_review is True
        assert response.confidence == 0.88


def test_triage_service_uses_custom_base_url():
    """Test that TriageService uses the configured base URL."""
    with patch('app.services.triage.Groq') as MockGroq:
        mock_client = Mock()
        MockGroq.return_value = mock_client

        settings = Settings(_env_file=None, groq_api_key="test_key", groq_base_url="https://custom.example.com")
        service = TriageService(settings)
        # The client is initialized in __init__, so we can check the mock was called with the base_url
        MockGroq.assert_called_once_with(api_key="test_key", base_url="https://custom.example.com")


def test_triage_service_handles_invalid_json_from_groq():
    """Test that TriageService raises ValidationError when Groq returns invalid JSON."""
    with patch('app.services.triage.Groq') as MockGroq:
        mock_client = Mock()
        mock_completion = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = 'invalid json'
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        MockGroq.return_value = mock_client

        settings = Settings(_env_file=None, groq_api_key="test_key")
        service = TriageService(settings)
        request = TicketRequest(message="Hello")
        with pytest.raises(Exception):  # json.loads will raise JSONDecodeError
            service.triage(request)


def test_triage_service_handles_missing_fields_from_groq():
    """Test that TriageService raises ValidationError when Groq returns missing fields."""
    with patch('app.services.triage.Groq') as MockGroq:
        mock_client = Mock()
        mock_completion = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = '{"category": "billing", "priority": "high"}'  # missing sentiment, summary, requires_human_review
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        MockGroq.return_value = mock_client

        settings = Settings(_env_file=None, groq_api_key="test_key")
        service = TriageService(settings)
        request = TicketRequest(message="Hello")
        with pytest.raises(Exception):  # Pydantic will raise a ValidationError
            service.triage(request)


def test_triage_service_propagates_groq_errors():
    """Test that TriageService propagates errors from the Groq client."""
    with patch('app.services.triage.Groq') as MockGroq:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Groq error")
        MockGroq.return_value = mock_client

        settings = Settings(_env_file=None, groq_api_key="test_key")
        service = TriageService(settings)
        request = TicketRequest(message="Hello")
        with pytest.raises(Exception, match="Groq error"):
            service.triage(request)