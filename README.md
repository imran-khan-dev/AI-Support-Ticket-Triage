# AI Support Ticket Triage

A service to triage support tickets using AI.

## Current Scope

This milestone implements configuration and environment variables:
- Project structure with FastAPI
- Pydantic schemas for ticket requests and triage responses
- Deterministic mock triage service
- Configuration using Pydantic Settings
- Tests for schemas, API, and configuration

## Setup

1. Clone the repository.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in the required values:
   ```bash
   copy .env.example .env  # on Windows
   # or
   cp .env.example .env  # on Unix
   ```

## Configuration

The application uses [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for AI-powered triage (optional) | `None` |

The application will start without a Groq API key, using the mock triage service.

### Usage

To configure the application, create a `.env` file in the project root with the desired variables, or set them in the environment.

Example `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Running Tests

To run the test suite:

```bash
pytest
```

## Next Steps

- Implement the ticket triage endpoint with Groq API integration.
- Add database persistence.
- Implement authentication and authorization.

