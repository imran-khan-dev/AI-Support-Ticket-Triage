# AI Support Ticket Triage

A service to triage support tickets using AI.

## Current Scope

This milestone implements the project foundation and data contract only:
- Project structure with FastAPI
- Pydantic schemas for ticket requests and triage responses
- Placeholder for configuration and triage service
- Tests for the data contract

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

## Running Tests

To run the test suite:

```bash
pytest
```

## Next Steps

- Implement the ticket triage endpoint.
- Integrate with Groq API for AI-powered triage.
- Add database persistence.
- Implement authentication and authorization.
