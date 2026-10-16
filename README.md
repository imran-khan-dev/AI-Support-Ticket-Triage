# AI Support Ticket Analysis

A FastAPI backend that analyzes customer support tickets into category, priority, sentiment, and summary, and decides whether human review is required.

## Current capabilities

- `POST /api/v1/tickets/triage` — analyze a support ticket and return a validated response.
- `GET /health` — health check.
- Interactive API docs at `/docs` (FastAPI/Swagger UI).
- Deterministic mock analysis when no Groq API key is configured.
- Live AI-powered analysis via the OpenAI SDK calling the Groq Responses API when `GROQ_API_KEY` and `GROQ_MODEL` are configured.
- Environment-driven configuration using Pydantic Settings with `.env` support.
- Request/response validation with Pydantic; invalid requests return HTTP 422, provider failures return HTTP 502 without leaking raw errors or secrets.

## Project structure

```
.
├── main.py                # FastAPI application entry point (repository root)
├── requirements.txt       # Python dependencies
├── .env.example           # Example environment variables (copy to .env)
├── .env                   # Local environment variables (ignored by git, do not commit)
├── app/
│   ├── __init__.py
│   ├── config.py          # Pydantic Settings (GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL)
│   ├── schemas.py         # Pydantic request/response models
│   └── services/
│       ├── __init__.py
│       └── analysis.py      # Analysis logic: client, mock fallback, prompt, Responses API call
└── tests/
    ├── __init__.py
    ├── test_analysis_service.py
    ├── test_tickets.py
    └── test_api_error_handling.py
```

## Request lifecycle

1. Client sends `POST /api/v1/tickets/triage` with a JSON body such as `{"message": "I cannot log into my account."}`.
2. FastAPI validates the body against `TicketRequest` (message: non-empty string, max 10,000 characters, whitespace stripped). Invalid input returns HTTP 422.
3. `analyze_ticket()` in `app/services/analysis.py` is called with the validated request, the loaded `Settings`, and the resolved Groq client.
4. If no Groq API key is configured, the client is `None` and `mock_analysis()` returns a deterministic response:
   - category: `technical`, priority: `medium`, sentiment: `neutral`, summary: `This is a mock summary.`, requires_human_review: `false`, confidence: `0.85`.
5. If a Groq API key is configured, the OpenAI SDK calls the Groq Responses API (`responses.create`) using the configured model, base URL, and system prompt, with `temperature=0.0` and JSON-object output format. The model's `output_text` is parsed as JSON and validated against `AnalysisResponse`.
6. The validated `AnalysisResponse` is returned to the client. Provider or validation errors map to HTTP 502 with a safe message; raw exceptions and secrets are never exposed.

## Setup on Windows PowerShell

1. Clone the repository and open PowerShell in the repository root (`E:\AI\AI-Support-Ticket-Triage-clean`).

2. Create and activate a virtual environment (recommended):

```powershell
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

Dependencies: `fastapi`, `uvicorn`, `pydantic-settings`, `pytest`, `httpx`, `openai`.

## Configuration

The application reads configuration from environment variables, with `.env` loaded automatically by Pydantic Settings from the repository root.

1. Copy the example file:

```powershell
copy .env.example .env
```

2. Edit `.env` and fill in values. Example:

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

| Variable | Required? | Description | Default |
|---|---|---|---|
| `GROQ_API_KEY` | Optional | Groq API key for AI-powered analysis (optional). | `None` (mock mode) |
| `GROQ_MODEL` | Required when `GROQ_API_KEY` is set | Groq model to use for analysis. | `None` |
| `GROQ_BASE_URL` | Optional | Base URL for the Groq API. | `None` |

### Behavior without live AI configuration

If `GROQ_API_KEY` is not set, the Groq client is `None` and the endpoint uses the deterministic mock analysis response described above. This lets you develop and test without a Groq account.

If `GROQ_API_KEY` is set but `GROQ_MODEL` is not, the application raises a clear configuration error at startup rather than failing mid-request.

## Running the server

From the repository root (where `main.py` lives), with the virtual environment activated:

```powershell
uvicorn main:app --reload
```

This starts the FastAPI server on `http://127.0.0.1:8000` by default.

## Using the API

Open the interactive docs in a browser:

```
http://127.0.0.1:8000/docs
```

### Health check

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```

Returns `{"status": "healthy"}`.

### Analyze a ticket

```powershell
$body = @{ message = "I cannot log into my account." } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/tickets/triage" -Method Post -Body $body -ContentType "application/json"
```

Example response:

```json
{
  "category": "account_access",
  "priority": "medium",
  "sentiment": "neutral",
  "summary": "Customer cannot log into their account.",
  "requires_human_review": false,
  "confidence": 0.85
}
```

### Response fields

| Field | Type | Description |
|---|---|---|
| `category` | `billing \| technical \| account_access \| bug_report \| feature_request \| other` | Ticket category. |
| `priority` | `low \| medium \| high \| urgent` | Ticket priority. |
| `sentiment` | `positive \| neutral \| frustrated \| angry` | Customer sentiment. |
| `summary` | string (1–200 chars) | Brief summary of the ticket. |
| `requires_human_review` | boolean | Whether the ticket needs human review. |
| `confidence` | number (0.0–1.0) | Model-estimated confidence. |

## Running tests

From the repository root, with the virtual environment activated:

```powershell
python -m pytest -v
```

Tests are mocked and do not require a real Groq API key or make paid API requests. The suite currently has 52 passing tests covering schema validation, mock analysis, mocked Groq analysis, environment-driven configuration, missing model configuration, custom base URL, invalid provider output, provider error handling, confidence validation, health endpoint behavior, and independence from the developer's real `.env` file.

## Limitations

- `confidence` is a model-estimated score between 0.0 and 1.0, not a calibrated probability of correctness or measured accuracy.
- Classification quality depends on the model and system prompt and should be evaluated against real tickets before production use.
- When no `GROQ_API_KEY` is configured, responses come from the deterministic mock service, not a live model.
- There is no database persistence, authentication, or authorization in this milestone.
