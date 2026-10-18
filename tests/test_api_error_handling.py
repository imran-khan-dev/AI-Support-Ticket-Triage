import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from main import app
from app.schemas import AnalysisResponse

client = TestClient(app)


def test_successful_analysis_request():
    """A successful analysis request returns the expected schema and status."""
    with patch('main.groq_client', None):
        response = client.post(
            "/api/v1/tickets/analyze",
            json={"message": "Hello, I need help."},
        )
        assert response.status_code == 200
        json_data = response.json()
        # Validate against schema
        AnalysisResponse(**json_data)
        assert "category" in json_data
        assert "priority" in json_data
        assert "sentiment" in json_data
        assert "summary" in json_data
        assert "requires_human_review" in json_data


def test_provider_failure_returns_502():
    """A simulated provider failure returns 502 and a safe error message."""
    with patch('main.analyze_ticket') as mock_analysis_fn:
        mock_analysis_fn.side_effect = Exception("Simulated provider failure")
        response = client.post(
            "/api/v1/tickets/analyze",
            json={"message": "Hello, I need help."},
        )
        assert response.status_code == 502
        assert "detail" in response.json()
        detail = response.json()["detail"]
        assert "provider" in detail.lower()
        # Ensure no secrets or raw details leaked
        assert "Simulated provider failure" not in detail
        assert "api_key" not in detail.lower()


def test_provider_failure_does_not_leak_secrets():
    """Response does not contain simulated secret or raw exception details."""
    with patch('main.analyze_ticket') as mock_analysis_fn:
        mock_analysis_fn.side_effect = Exception("Connection failed with api_key=supersecret123")
        response = client.post(
            "/api/v1/tickets/analyze",
            json={"message": "Hello, I need help."},
        )
        assert response.status_code == 502
        json_data = response.json()
        detail = json_data.get("detail", "")
        assert "supersecret123" not in detail
        assert "api_key" not in detail.lower()
        assert "Connection failed" not in detail


def test_malformed_provider_output_returns_502():
    """Malformed provider output (validation error) returns safe 502."""
    with patch('main.analyze_ticket') as mock_analysis_fn:
        from pydantic import ValidationError
        mock_analysis_fn.side_effect = ValidationError.from_exception_data(
            "AnalysisResponse",
            [{"type": "missing", "loc": ("category",), "input": {}}],
        )
        response = client.post(
            "/api/v1/tickets/analyze",
            json={"message": "Hello, I need help."},
        )
        assert response.status_code == 502
        assert "detail" in response.json()
        detail = response.json()["detail"]
        assert "validation" in detail.lower()


def test_provider_invalid_confidence_returns_502():
    """Provider output missing confidence returns safe 502."""
    with patch('main.analyze_ticket') as mock_analysis_fn:
        from pydantic import ValidationError
        mock_analysis_fn.side_effect = ValidationError.from_exception_data(
            "AnalysisResponse",
            [{"type": "missing", "loc": ("confidence",), "input": {}}],
        )
        response = client.post(
            "/api/v1/tickets/analyze",
            json={"message": "Hello, I need help."},
        )
        assert response.status_code == 502
        json_data = response.json()
        detail = json_data.get("detail", "")
        # No raw provider details leaked
        assert "missing" not in detail.lower()
        assert "api_key" not in detail.lower()


def test_invalid_json_from_provider_returns_502():
    """Invalid JSON from the provider returns safe 502 without leaking details."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.output_text = 'invalid json'
    mock_client.responses.create.return_value = mock_response

    with patch('main.groq_client', mock_client):
        response = client.post(
            "/api/v1/tickets/analyze",
            json={"message": "Hello, I need help."},
        )
        assert response.status_code == 502
        detail = response.json().get("detail", "")
        # Safe generic message, raw JSON error text not leaked
        assert "invalid json" not in detail
        assert "api_key" not in detail.lower()


def test_empty_output_text_from_provider_returns_502():
    """Empty output_text from the provider returns safe 502 without leaking details."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.output_text = ''
    mock_client.responses.create.return_value = mock_response

    with patch('main.groq_client', mock_client):
        response = client.post(
            "/api/v1/tickets/analyze",
            json={"message": "Hello, I need help."},
        )
        assert response.status_code == 502
        detail = response.json().get("detail", "")
        assert "api_key" not in detail.lower()


def test_missing_output_text_attribute_returns_502():
    """Provider response missing the output_text attribute returns safe 502."""
    mock_client = Mock()
    mock_response = Mock(spec_set=[])  # no attributes are accessible
    mock_client.responses.create.return_value = mock_response

    with patch('main.groq_client', mock_client):
        response = client.post(
            "/api/v1/tickets/analyze",
            json={"message": "Hello, I need help."},
        )
        assert response.status_code == 502
        detail = response.json().get("detail", "")
        assert "api_key" not in detail.lower()


def test_invalid_request_still_returns_422():
    """Invalid request input still returns 422."""
    response = client.post(
        "/api/v1/tickets/analyze",
        json={"message": ""},
    )
    assert response.status_code == 422


def test_invalid_request_whitespace_returns_422():
    """Whitespace-only message returns 422."""
    response = client.post(
        "/api/v1/tickets/analyze",
        json={"message": "   "},
    )
    assert response.status_code == 422


def test_invalid_request_too_long_returns_422():
    """Message exceeding max length returns 422."""
    long_message = "x" * 10001
    response = client.post(
        "/api/v1/tickets/analyze",
        json={"message": long_message},
    )
    assert response.status_code == 422


def test_mock_mode_works_without_api_key():
    """Mock mode works without a configured API key."""
    with patch('main.groq_client', None):
        response = client.post(
            "/api/v1/tickets/analyze",
            json={"message": "Test mock mode"},
        )
        assert response.status_code == 200
        AnalysisResponse(**response.json())


def test_health_endpoint_unchanged():
    """The health endpoint remains unchanged."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}