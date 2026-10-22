# AI Support Ticket Analysis

A FastAPI backend that analyzes customer support tickets into category, priority, sentiment, and summary, and decides whether human review is required.

## Current capabilities

- `POST /api/v1/tickets/analyze` — analyze a support ticket and return a validated response.
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
│   ├── core/
│   │   └── config.py      # Pydantic Settings (GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL, DATABASE_URL)
│   ├── ai/
│   │   └── provider.py    # Groq client construction
│   ├── agent_core/        # Agent loop, tool registry, argument schemas
│   ├── modules/
│   │   ├── analyze/       # POST /api/v1/tickets/analyze
│   │   └── agent/         # POST /api/v1/tickets/agent/run
│   └── persistence/       # PostgreSQL access: pool, migrations, models, repository
└── tests/
    ├── conftest.py        # Points the suite at the PostgreSQL test database
    ├── test_analysis_service.py
    ├── test_tickets.py
    ├── test_api_error_handling.py
    ├── test_persistence.py
    └── test_agent_sessions.py
```

## Request lifecycle

1. Client sends `POST /api/v1/tickets/analyze` with a JSON body such as `{"message": "I cannot log into my account."}`.
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

Dependencies: `fastapi`, `uvicorn`, `pydantic-settings`, `pytest`, `httpx`, `openai`, `psycopg[binary]`, `psycopg-pool`.

### PostgreSQL setup

The application stores sessions, transcripts, memories, approval actions,
approvers, and audit events in PostgreSQL. The schema is created by a
versioned migration runner, so the database only has to exist.

1. Make sure a PostgreSQL server is running and reachable.

2. Create the application database and a separate database for the tests:

```powershell
createdb ticket_triage
createdb ticket_triage_test
```

3. Point the application at it through `DATABASE_URL` (see
   [Configuration](#configuration)):

```
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/ticket_triage
```

4. Start the server. The schema is applied automatically the first time a
   request needs storage, and applying it again is a no-op.

The migration runner takes a PostgreSQL advisory lock, so several instances
may start at the same time without migrating the database concurrently. Each
migration runs inside its own transaction: because PostgreSQL has
transactional DDL, a failed migration leaves neither the schema nor the
recorded version changed.

The six tables are `sessions`, `messages`, `memories`, `actions`,
`approvers`, and `audit_events`. `audit_events` is append-only: a trigger
rejects any `UPDATE` or `DELETE` on it.

#### What the role needs

The user in `DATABASE_URL` owns the schema it migrates, so it needs to be
able to create tables, indexes, and triggers in that database. The
administrator role of a local server satisfies this out of the box. The
`plpgsql` extension, which PostgreSQL installs by default, has to be
available, because the append-only trigger is written in it.

```powershell
psql -U postgres -c "CREATE DATABASE ticket_triage;"
psql -U postgres -c "CREATE DATABASE ticket_triage_test;"
psql -U postgres -d ticket_triage -c "CREATE EXTENSION IF NOT EXISTS plpgsql;"
```

The application was verified against PostgreSQL 17.5 on `127.0.0.1:5432`,
with the schema created by the migration runner from an empty database.

#### What was not verified

The migration runner was exercised against a single local server, on an empty
database, with one application instance. The following were not verified and
should be checked before running this against a shared or managed server:

- migrating a database that already holds data from an older version of the
  schema;
- running the server on several instances against the same database at the
  same time, beyond the advisory lock that the runner takes;
- behaviour when the database is read-only, when the role lacks the
  permissions above, or when the server is unreachable mid-request;

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
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/ticket_triage
```

| Variable | Required? | Description | Default |
|---|---|---|---|
| `GROQ_API_KEY` | Optional | Groq API key for AI-powered analysis (optional). | `None` (mock mode) |
| `GROQ_MODEL` | Required when `GROQ_API_KEY` is set | Groq model to use for analysis. | `None` |
| `GROQ_BASE_URL` | Optional | Base URL for the Groq API. | `None` |
| `DATABASE_URL` | Required for the agent endpoint | PostgreSQL connection string for session, message, approval, and audit storage. | `None` |
| `AGENT_MAX_HISTORY_MESSAGES` | Optional | Maximum number of stored messages of a session that are replayed into a continuing agent run. A tool call and its output are always replayed together, so the effective history can be shorter. Zero disables the replay. | `20` |
| `AGENT_MAX_MEMORY_ITEMS` | Optional | Maximum number of stored memories of a session that are recalled into an agent run. Zero disables the recall. | `3` |
| `AGENT_MAX_MEMORY_CHARS` | Optional | Maximum number of characters of recalled memory that are injected into an agent run, the label of the context included. Zero disables the recall. | `1200` |

### Behavior without live AI configuration

If `GROQ_API_KEY` is not set, the Groq client is `None` and the endpoint uses the deterministic mock analysis response described above. This lets you develop and test without a Groq account.

If `GROQ_API_KEY` is set but `GROQ_MODEL` is not, the application raises a clear configuration error at startup rather than failing mid-request.

### Behavior without a database

The application starts even when `DATABASE_URL` is not configured, and
`GET /health` keeps answering. The endpoints that need storage - currently
`POST /api/v1/tickets/agent/run` - answer HTTP 503 instead of failing with an
unhandled error. `POST /api/v1/tickets/analyze` does not use the database and
is unaffected.

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
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/tickets/analyze" -Method Post -Body $body -ContentType "application/json"
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

### Resolve a ticket with the agent

```powershell
$body = @{ message = "I was charged twice." } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/tickets/agent/run" -Method Post -Body $body -ContentType "application/json"
```

The agent stores the run in a session. Omit `session_id` to start a new
session, or pass a `session_id` to continue an existing one.

Example response:

```json
{
  "status": "success",
  "answer": "Refunds are issued within 5 business days.",
  "mock_mode": false,
  "tool_calls": [],
  "error": null,
  "session_id": "6d5996cb-43f7-422c-a50f-b8ad57044617"
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | `success`, `mock`, or `max_iterations_reached`. |
| `answer` | string | The agent's answer, when there is one. |
| `mock_mode` | boolean | True when no Groq API key is configured. |
| `tool_calls` | array | One record per tool call the agent made. |
| `error` | string | A safe error message, when applicable. |
| `session_id` | string | The session the run belongs to. |

Two response headers carry the details that do not belong in the body:

| Header | Description |
|---|---|
| `X-Session-Id` | The session the run belongs to. |
| `X-Transcript-Stored` | `true` when the transcript of the run was written, `false` when it was lost. |

The session id matters when the run ends in an error: a provider failure
answers HTTP 502 with an unchanged error body, and the header still carries
the session id, so the client can recover it and continue the conversation
later. The answer is returned even when the transcript could not be stored,
because the answer is the primary result of the request.

### Continuing a conversation

A run that passes a `session_id` receives the earlier turns of that session.
The transcript is read from PostgreSQL in sequence order, only for that
session, and is bounded by `AGENT_MAX_HISTORY_MESSAGES`:

* the message of the current run is not part of the history, and is sent to
  the provider exactly once;
* a tool call and its output are always replayed together, so a call is never
  separated from its output;
* a replayed tool call gets a call id of its own, so it cannot be confused
  with a call the model makes in the current turn;
* the history of one session is never visible to another session.

### Memories

A session keeps a memory of what the agent told the customer, so that a later
run of the same session can take it into account. A memory belongs to exactly
one session and is never visible to another one.

**Writing a memory.** A run stores at most one memory, and only after its
transcript was stored:

* only a successful run writes a memory. A mock answer, a provider failure,
  and a run that ran out of iterations write nothing;
* the content is the answer of the agent and nothing else. The message of the
  customer, the tool calls, the tool results, and the error text are never
  part of a memory;
* the answer is normalized first: control characters are removed, and every
  run of whitespace becomes a single space;
* an answer that is empty, or shorter than 24 characters, is not stored;
* a longer answer is shortened to 400 characters, at a word boundary where
  possible;
* the kind of the memory is `summary`: it records what the agent said, not a
  verified fact about the customer;
* the memory points at the stored answer of the run through `source_msg_id`,
  so that the two can be traced back to each other. When that message is
  deleted, the reference is cleared and the memory is kept.

The same content is not stored twice in a session: an exact match is skipped.
The check is a read that is followed by a write, and the two are not one
atomic step, so two concurrent runs of the same session can still store the
same content twice. The table has no unique constraint that could rule this
out.

Writing a memory is best-effort. A storage failure is logged, and the answer
is returned unchanged: the body, the status, and the headers of the response
are the same whether the memory was stored or not.

**Recalling a memory.** Before the run, the memories of the session that match
the message of the customer are searched with the full-text index of the
database, bounded by `AGENT_MAX_MEMORY_ITEMS` and `AGENT_MAX_MEMORY_CHARS`.
They are placed in front of the conversation as one labelled item of the role
`system`, and the text inside it is treated as untrusted data, never as
instructions. Recalling them is best-effort as well: a storage failure is
logged, and the run continues without them.

No model is used to extract a memory: the stored text is the answer of the
agent, unchanged apart from the normalization and the cap described above.

## Running tests

From the repository root, with the virtual environment activated:

```powershell
python -m pytest -v
```

Tests are mocked and do not require a real Groq API key or make paid API requests.

The persistence tests need a running PostgreSQL server and the
`ticket_triage_test` database. The connection string is resolved in this
order, so no credential has to be stored in the repository:

1. `TEST_DATABASE_URL`, when it is set;
2. otherwise `DATABASE_URL` from the environment or from the local `.env`
   file, with the database name replaced by `ticket_triage_test`;
3. otherwise `postgresql://postgres@127.0.0.1:5432/ticket_triage_test`, which
   works with trust authentication or a `pgpass` file.

The suite empties the test database before each test, so it never touches
application data.

## Limitations

- `confidence` is a model-estimated score between 0.0 and 1.0, not a calibrated probability of correctness or measured accuracy.
- Classification quality depends on the model and system prompt and should be evaluated against real tickets before production use.
- When no `GROQ_API_KEY` is configured, responses come from the deterministic mock service, not a live model.
- Stored conversation history is replayed into a continuing run, bounded by `AGENT_MAX_HISTORY_MESSAGES`. The memories of a session are written after a successful run and recalled into a later run, bounded by `AGENT_MAX_MEMORY_ITEMS` and `AGENT_MAX_MEMORY_CHARS`, and placed in front of the conversation as one labelled item of the role `system`: recalled content is untrusted data, not instructions. A memory holds what the agent said, is scoped to one session, and is not extracted by a model. The duplicate check is an exact-content comparison that is not atomic, so concurrent runs of one session can store the same content twice. The approval and audit tables have a persistence layer with guarded state transitions, but nothing in the request path uses it yet: the schema and the storage exist so that those features can be built on them.
