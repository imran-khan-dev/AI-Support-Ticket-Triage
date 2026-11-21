# AI Support Ticket Triage Backend

FastAPI service that provides AI-assisted support ticket analysis and a tool-using agent with approval-gated consequential actions. Designed to be integrated by existing host applications.

## Key capabilities

- `POST /api/v1/tickets/analyze` for ticket analysis.
- `POST /api/v1/tickets/agent/run` for an authenticated, tenant-scoped, tool-using agent run.
- DB-backed integration authentication with tenant context.
- Persisted session history and memory with full-text recall.
- Approval workflow for gated tools (`create_support_case` does not execute before approval).
- Atomic action claiming, idempotency by request key, and audit persistence.
- `SupportCaseAdapter` interface with `FakeSupportCaseAdapter` default and explicit adapter injection.

## Architecture and flow

1. Host app authenticates with an integration bearer token; tenant context is derived from the verified credential.
2. Agent run creates/resolves a tenant-scoped session, replays history/memory, runs the tool loop.
3. Gated tool calls return `approval_required` instead of executing.
4. An approver (bearer token + capability) approves/rejects the action.
5. Approved actions are claimed atomically and executed via an explicitly injected adapter.
6. Outcomes and audits are persisted; uncertain outcomes preserve the claim.

## Stack

FastAPI, Uvicorn, Pydantic Settings, OpenAI SDK (Groq-compatible), psycopg/psycopg-pool, pytest, httpx.

## Setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Requires PostgreSQL. Tests use `ticket_triage_test`.

## Environment variables

- `DATABASE_URL` — PostgreSQL connection string (optional; storage endpoints return 503 if unset).
- `APPROVER_TOKEN_SECRET` — secret for approver credentials (optional; required for approvals).
- `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_BASE_URL` — optional; mock mode if unset.
- `AGENT_MAX_HISTORY_MESSAGES` (default 20), `AGENT_MAX_MEMORY_ITEMS` (default 3), `AGENT_MAX_MEMORY_CHARS` (default 1200).
- `TEST_DATABASE_URL` — override test database connection.

## Run

```powershell
uvicorn main:app --reload
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_approval_execution_service.py tests/test_approval_api.py tests/test_approval_request_service.py tests/test_approval_decision_service.py tests/test_approver_policy.py tests/test_external_api_adapter.py tests/test_tools.py tests/test_integration_auth.py -v
```

## API

Base: `POST /api/v1/tickets/agent/run` with `Authorization: Bearer <token>`.

Request: `{"message": "...", "session_id": null}`.

See `docs/API_INTEGRATION.md`.

## Limitations

- Mock mode when no Groq key is configured.
- `FakeSupportCaseAdapter` is in-memory only.
- No verified complete public-route end-to-end test for the approval execution flow.
- Abandoned claims after crash may require manual intervention.
