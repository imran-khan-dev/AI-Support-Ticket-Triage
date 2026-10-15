# AI Support Ticket Triage

A service to triage support tickets using AI.

## Current Scope

This milestone implements the project foundation, data contract, API endpoints, configuration, and Groq integration:
- Project structure with FastAPI
- Pydantic schemas for ticket requests and triage responses
- Triage service with optional Groq API integration (falls back to mock)
- Configuration using Pydantic Settings
- Tests for schemas, API, and service

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
| `GROQ_MODEL` | Groq model to use for triage | `mixtral-8x7b-32768` |
| `GROQ_BASE_URL` | Base URL for the Groq API (optional) | `None` (uses Groq's default) |

The application will start without a Groq API key, using the mock triage service.

### Usage

To configure the application, create a `.env` file in the project root with the desired variables, or set them in the environment.

Example `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=mixtral-8x7b-32768
GROQ_BASE_URL=https://api.groq.com
```

## Running Tests

To run the test suite:

```bash
pytest
```

## Next Steps

- Add database persistence.
- Implement authentication and authorization.
- Enhance error handling and logging.