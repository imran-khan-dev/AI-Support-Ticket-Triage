# AI Support Ticket Triage Backend

## Session history and memory

Conversation history is stored per session in PostgreSQL and replayed for later requests. Memories are stored per session, deduplicated by exact content, and recalled with full-text search. Recall is best-effort; storage failures do not abort an agent run.

Limits are configured through `AGENT_MAX_HISTORY_MESSAGES`, `AGENT_MAX_MEMORY_ITEMS`, and `AGENT_MAX_MEMORY_CHARS`.

## Approval workflow

Gated tools are configured server-side in `GATED_TOOLS`. `execute_tool()` returns `approval_required` for those tools without invoking them.

`request_approval()` creates a pending action idempotently by `request_key`. Reusing a key with different tool or arguments raises a conflict. Approvers authenticate with a bearer token; only active approvers with the required capability may decide actions. Actions move from pending to approved/rejected, then to executed. Decisions and audit events are persisted atomically.

## Execution safety

Approved actions are claimed before execution using a conditional update. The claim prevents competing attempts from executing the same action. If the tool succeeds but final state/audit persistence fails, the claim is preserved and an uncertain outcome is reported. External tool side effects cannot be rolled back by database transactions. Abandoned claims after a process crash may require manual intervention.

## Approval API

Base path: `/api/v1/tickets/approvals`

- `GET /actions` requires `X-Approver-Token`; capability `action.list`.
- `POST /actions/{action_id}/decide` requires `X-Approver-Token`; capability `action.decide`; body `{"decision":"approved"|"rejected","reason":...}`.
- `POST /actions/{action_id}/execute` requires `X-Approver-Token`; capability `action.execute`.

Errors: 401 invalid/inactive token, 403 unauthorized, 409 invalid/conflicting/non-pending/uncertain execution, 503 storage unavailable.

## Testing

Run focused tests:

```powershell
python -m pytest tests/test_approval_request_service.py tests/test_approval_decision_service.py tests/test_approval_execution_service.py tests/test_approval_api.py -q
```

Run the full suite:

```powershell
python -m pytest -q
```

The suite uses the isolated PostgreSQL test database `ticket_triage_test`. Never run tests against production. Set `TEST_DATABASE_URL` to override the connection string.
