# API Integration Guide (AI Support Ticket Triage)

## Overview
This service provides AI-assisted support ticket analysis and resolution with optional approval-gated execution of consequential actions. Host applications integrate via REST endpoints under `/api/v1/tickets`. Authentication and tenant scoping depend on the endpoint (integration bearer tokens for agent/integrations endpoints; approver tokens for approvals).

## Configuration (required)
Set environment variables (see `app/core/config.py`):
- `DATABASE_URL` (required for sessions/actions/audit/integrations; if missing, storage-dependent endpoints return 503)
- `APPROVER_TOKEN_SECRET` (required to create/verify approver credentials for approvals API)
- `GROQ_API_KEY` / `GROQ_MODEL` (optional; if missing, agent runs in mock mode)
- Agent limits: `AGENT_MAX_HISTORY_MESSAGES`, `AGENT_MAX_MEMORY_ITEMS`, `AGENT_MAX_MEMORY_CHARS` (optional)

## Integration authentication and tenant scoping
**Endpoints requiring integration auth** (via `require_integration`):
- `POST /api/v1/tickets/agent/run`
- `GET /api/v1/tickets/integrations/ping`

Flow:
1. Issue an integration credential via `scripts/issue_integration_credential.py <name>` (prints plaintext token once). Only token hash is stored.
2. Send `Authorization: Bearer <token>`. Identity is derived only from verified credentials (headers like `X-Integration-Id`/`X-Tenant-Id` are ignored).
3. Integration must have `tenant_id` set and the tenant must exist and be enabled; otherwise requests are rejected with 401.
4. On success, `request.state.integration` and `request.state.tenant` are set.

**Approvals API** uses approver bearer tokens via `X-Approver-Token` header with role-based capabilities.

## Agent endpoint
**Route:** `POST /api/v1/tickets/agent/run`  
**Auth:** Integration bearer token required. Tenant required.  
**Headers (response):** `X-Session-Id`, `X-Transcript-Stored`

**Request schema**:
```json
{
  "message": "string (1-10000 chars, trimmed)",
  "session_id": "string|null (optional; must identify existing open session if provided)"
}
```

**Response schema**:
```json
{
  "status": "success|mock|provider_failure|max_iterations_reached",
  "answer": "string|null",
  "mock_mode": true|false,
  "tool_calls": [],
  "error": "string|null",
  "session_id": "string|null"
}
```

**Behavior:**
- Creates/resolves tenant-scoped session. History and memories are loaded and replayed.
- Runs agent loop with registered tools (`search_support_docs`, `create_support_case`). 
- `create_support_case` is gated - `execute_tool()` returns `{"error":"approval_required", ...}` when called by the agent.
- Persists transcript; stores memory only if a persisted final answer exists.
- Errors mapped to HTTP codes (422 invalid session id, 404 session not found, 409 session closed, 503 storage unavailable, 502 provider failure).

## Approval workflow
1. **Request approval:** `request_approval()` persists a pending action idempotently by `request_key`. Different args with same key ? conflict. Creates action + audit atomically.
2. **Approve/reject:** Approver authenticates with `X-Approver-Token`, requires `action.decide`. `POST /api/v1/tickets/approvals/actions/{action_id}/decide` with `{"decision":"approved"|"rejected","reason":...}` moves pending ? approved/rejected atomically with audit.
3. **Execute approved action:** Approver requires `action.execute`. `POST /api/v1/tickets/approvals/actions/{action_id}/execute` claims approved action atomically, re-validates args/schema, executes via injected adapter (for `create_support_case`). If audit persistence fails after success, returns uncertain outcome and claim is preserved.

**Approvals API (base `/api/v1/tickets/approvals`):**
- `GET /actions` - requires `X-Approver-Token`, capability `action.list`. Query: `session_id`, `state`, `correlation_id`, `limit`.
- `POST /actions/{action_id}/decide` - requires `X-Approver-Token`, capability `action.decide`. Body: `DecisionRequest`. Returns `ActionResponse`.
- `POST /actions/{action_id}/execute` - requires `X-Approver-Token`, capability `action.execute`. Tenant enforcement: tenant-owned actions require matching request tenant context or capability `action.execute_any_tenant` (admin only). Passes approver id as `executed_by` and tenant context to service. Returns `ExecutionResponse`.

**Error codes:** 401 invalid/inactive/unauthorized approver or missing secret, 403 unauthorized capability/tenant, 409 invalid state/conflict/uncertain execution/missing action, 503 storage unavailable.

## Support case adapter
- Interface: `app/integrations/external_api.py::SupportCaseAdapter` with `create_support_case(args: CreateSupportCaseArgs) -> dict`.
- Default: `FakeSupportCaseAdapter` (in-memory, deterministic). 
- Injection: `execute_approved_action(adapter=...)` accepts an explicit adapter (server-controlled). No adapter derived from model args. No global swapping.

## Limitations
- Mock mode: If no Groq API key configured, agent returns mock response with no provider calls.
- Fake adapter: Default is in-memory only (not production). Host apps must inject a production adapter.
- Storage required: Many features require PostgreSQL; endpoints return 503 if storage unavailable.
- Manual intervention: Abandoned execution claims after crash may require manual intervention.
- Tenant scoping: Integration endpoints require tenant; approvals enforce tenant scope with explicit `action.execute_any_tenant` for cross-tenant/admin cases.
