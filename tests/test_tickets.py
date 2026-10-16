
import pytest
from unittest.mock import patch
from pydantic import ValidationError
from fastapi.testclient import TestClient
from app.schemas import TicketRequest, TriageResponse
from app.services.triage import TriageService
from app.config import Settings
from main import app

client = TestClient(app)

# Existing schema tests
def test_valid_ticket_request():
    """Test that a valid ticket request is accepted."""
    req = TicketRequest(message="Hello, I need help.")
    assert req.message == "Hello, I need help."

def test_ticket_request_empty_message():
    """Test that an empty message is rejected."""
    with pytest.raises(ValidationError):
        TicketRequest(message="")

def test_ticket_request_whitespace_message():
    """Test that a whitespace-only message is rejected."""
    with pytest.raises(ValidationError):
        TicketRequest(message="   ")

def test_ticket_request_max_length():
    """Test that a message exceeding the maximum length is rejected."""
    long_message = "x" * 10001
    with pytest.raises(ValidationError):
        TicketRequest(message=long_message)

def test_valid_triage_response():
    """Test that a valid triage output is accepted."""
    resp = TriageResponse(
        category="billing",
        priority="high",
        sentiment="frustrated",
        summary="Customer has billing issue.",
        requires_human_review=True,
        confidence=0.9,
    )
    assert resp.category == "billing"
    assert resp.priority == "high"
    assert resp.sentiment == "frustrated"
    assert resp.summary == "Customer has billing issue."
    assert resp.requires_human_review is True
    assert resp.confidence == 0.9


def test_triage_response_confidence_below_range():
    """Test that a confidence below 0.0 is rejected."""
    with pytest.raises(ValidationError):
        TriageResponse(
            category="billing",
            priority="high",
            sentiment="frustrated",
            summary="Summary",
            requires_human_review=False,
            confidence=-0.1,
        )


def test_triage_response_confidence_above_range():
    """Test that a confidence above 1.0 is rejected."""
    with pytest.raises(ValidationError):
        TriageResponse(
            category="billing",
            priority="high",
            sentiment="frustrated",
            summary="Summary",
            requires_human_review=False,
            confidence=1.1,
        )


def test_triage_response_missing_confidence():
    """Test that omitting confidence is rejected."""
    with pytest.raises(ValidationError):
        TriageResponse(
            category="billing",
            priority="high",
            sentiment="frustrated",
            summary="Summary",
            requires_human_review=False,
        )


def test_triage_response_boundary_confidence_values():
    """Test that confidence of 0.0 and 1.0 are accepted."""
    for value in (0.0, 1.0):
        resp = TriageResponse(
            category="billing",
            priority="high",
            sentiment="frustrated",
            summary="Summary",
            requires_human_review=False,
            confidence=value,
        )
        assert resp.confidence == value


def test_triage_response_invalid_category():
    """Test that an invalid category is rejected."""
    with pytest.raises(ValidationError):
        TriageResponse(
            category="invalid",
            priority="high",
            sentiment="frustrated",
            summary="Summary",
            requires_human_review=False,
            confidence=0.5,
        )

def test_triage_response_invalid_priority():
    """Test that an invalid priority is rejected."""
    with pytest.raises(ValidationError):
        TriageResponse(
            category="billing",
            priority="invalid",
            sentiment="frustrated",
            summary="Summary",
            requires_human_review=False,
            confidence=0.5,
        )

def test_triage_response_invalid_sentiment():
    """Test that an invalid sentiment is rejected."""
    with pytest.raises(ValidationError):
        TriageResponse(
            category="billing",
            priority="high",
            sentiment="invalid",
            summary="Summary",
            requires_human_review=False,
            confidence=0.5,
        )

def test_triage_response_empty_summary():
    """Test that an empty summary is rejected."""
    with pytest.raises(ValidationError):
        TriageResponse(
            category="billing",
            priority="high",
            sentiment="frustrated",
            summary="",
            requires_human_review=False,
            confidence=0.5,
        )

def test_triage_response_whitespace_summary():
    """Test that a whitespace-only summary is rejected."""
    with pytest.raises(ValidationError):
        TriageResponse(
            category="billing",
            priority="high",
            sentiment="frustrated",
            summary="   ",
            requires_human_review=False,
            confidence=0.5,
        )

# API tests
def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_triage_endpoint_valid_request():
    """Test the triage endpoint with a valid request."""
    mock_svc = TriageService(Settings(_env_file=None))
    with patch('main.triage_service', mock_svc):
        response = client.post(
            "/api/v1/tickets/triage",
            json={"message": "Hello, I need help."},
        )
        assert response.status_code == 200
        json_data = response.json()
        # Check that the response has the expected fields and values from the mock service
        assert json_data["category"] == "technical"
        assert json_data["priority"] == "medium"
        assert json_data["sentiment"] == "neutral"
        assert json_data["summary"] == "This is a mock summary."
        assert json_data["requires_human_review"] is False
        assert 0.0 <= json_data["confidence"] <= 1.0
        # Additionally, validate that the response conforms to TriageResponse schema
        # by attempting to create a TriageResponse instance (should not raise)
        TriageResponse(**json_data)

def test_triage_endpoint_empty_message():
    """Test the triage endpoint with an empty message."""
    response = client.post(
        "/api/v1/tickets/triage",
        json={"message": ""},
    )
    assert response.status_code == 422  # Unprocessable Entity

def test_triage_endpoint_whitespace_message():
    """Test the triage endpoint with a whitespace-only message."""
    response = client.post(
        "/api/v1/tickets/triage",
        json={"message": "   "},
    )
    assert response.status_code == 422  # Unprocessable Entity

def test_triage_endpoint_message_too_long():
    """Test the triage endpoint with a message exceeding the maximum length."""
    long_message = "x" * 10001
    response = client.post(
        "/api/v1/tickets/triage",
        json={"message": long_message},
    )
    assert response.status_code == 422  # Unprocessable Entity

# Configuration tests
def test_settings_without_groq_api_key(monkeypatch):
    """Test that settings loads without GROQ_API_KEY environment variable."""
    # Ensure the environment variable is not set
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # Reload the Settings class to pick up the environment change
    # Since Settings is instantiated at import time in app/main, we need to create a new instance
    # for each test to reflect the current environment.
    settings = Settings(_env_file=None)  # Do not load from .env file
    assert settings.groq_api_key is None

def test_settings_with_groq_api_key(monkeypatch):
    """Test that settings reads GROQ_API_KEY from environment variable."""
    # Set the environment variable
    monkeypatch.setenv("GROQ_API_KEY", "test_groq_key")
    # Create a new Settings instance, not loading from .env file
    settings = Settings(_env_file=None)
    assert settings.groq_api_key == "test_groq_key"


def test_settings_reads_groq_model_from_env(monkeypatch):
    """Test that settings reads GROQ_MODEL from environment variable."""
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    settings = Settings(_env_file=None)
    assert settings.groq_model == "llama-3.1-8b-instant"


def test_settings_reads_groq_base_url_from_env(monkeypatch):
    """Test that settings reads GROQ_BASE_URL from environment variable."""
    monkeypatch.setenv("GROQ_BASE_URL", "https://custom.groq.example.com")
    settings = Settings(_env_file=None)
    assert settings.groq_base_url == "https://custom.groq.example.com"


def test_settings_groq_model_none_when_not_set(monkeypatch):
    """Test that GROQ_MODEL is None when not set in environment."""
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    settings = Settings(_env_file=None)
    assert settings.groq_model is None


def test_settings_groq_base_url_none_when_not_set(monkeypatch):
    """Test that GROQ_BASE_URL is None when not set in environment."""
    monkeypatch.delenv("GROQ_BASE_URL", raising=False)
    settings = Settings(_env_file=None)
    assert settings.groq_base_url is None


def test_settings_loads_dotenv_file(tmp_path, monkeypatch):
    """Test that Settings loads configuration from .env file by default."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GROQ_API_KEY=dummy_key_12345\n"
        "GROQ_MODEL=dummy_model\n"
        "GROQ_BASE_URL=https://dummy.example.com\n"
    )
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert settings.groq_api_key == "dummy_key_12345"
    assert settings.groq_model == "dummy_model"
    assert settings.groq_base_url == "https://dummy.example.com"
