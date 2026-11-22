# AI Support Ticket Triage Backend

## Architecture — Three Reusable Components

The backend integrates three standalone, reusable AI components:

| Component | Responsibility | Location |
|-----------|---------------|----------|
| **Retrieval Stack** | Document ingestion, chunking, BM25/dense/hybrid retrieval, reranking, offline evaluation | `Retrieval-Stack/` |
| **Agent Orchestrator** | Provider-independent AgentLoop, tool execution, retries, tracing, hooks, execution limits | `Agent Orchestrator/` |
| **Eval Harness** | Datasets, evaluators, experiment runner, metrics, regression checks, reports | `eval_harness/` |

The application layer owns session lifecycle, tenant-scoped session resolution,
conversation history, agent memory, transcript persistence, approval gating,
trusted tenant/database context injection, and HTTP response contracts. The
orchestrator owns the agent execution loop and tool dispatch only.

## Groq Provider Adapter

`app/agent_core/groq_provider.py` wraps the OpenAI-compatible Groq client as an
`LLMProvider` for the Agent Orchestrator. It translates orchestrator-style
messages to Groq's Responses API input items, maps system messages to Groq's
`instructions` parameter, and normalizes `function_call` / `function_call_output`
output items back into provider-agnostic `ToolCall`s.

Groq does not support `previous_response_id`; the full conversation history is
passed on every request.

## Application Tool Registration

`app/agent_core/orchestrator_tools.py` builds a `ToolRegistry` from the
application's existing `TOOL_REGISTRY`. Two name sets are kept separate:

- **`REGISTERED_TOOL_NAMES`** — every registered tool is executable through the
  orchestrator, so legacy/read-only tools still yield a safe result.
- **`ADVERTISED_TOOL_NAMES`** — only the subset sent to the model as available
  function definitions (`search_knowledge`, `create_support_case`).

Handlers are bound to trusted, server-side dependencies (`tenant_id`, `db`,
`is_approved`) at registry-construction time. Any `tenant_id` the model supplies
in its arguments is stripped before execution. Approval state comes exclusively
from trusted application context; the model cannot approve its own actions.

## Tenant Context and Approval Handling

- `tenant_id` and `db` are injected by the application, never accepted from
  model-supplied arguments.
- `is_approved` defaults to `False`; gated tools (such as
  `create_support_case`) are refused before execution when not approved.
- Approval decisions are made through the existing approval API
  (`/api/v1/tickets/approvals`) and the trusted context is bound into the tool
  handlers by `run_ticket_service()`.

## Session / History / Memory Responsibilities (Application Layer)

- **Sessions**: created or resolved server-side; a client-supplied session id is
  accepted only when it identifies an existing, open session within the trusted
  tenant. Legacy records without tenant IDs are never auto-assigned.
- **History**: prior turns are loaded from PostgreSQL, grouped so tool calls
  stay paired with their outputs, and replayed bounded by
  `AGENT_MAX_HISTORY_MESSAGES`. The current message is added exactly once.
- **Memory**: stored per session, deduplicated by exact content, recalled with
  full-text search bounded by `AGENT_MAX_MEMORY_ITEMS` /
  `AGENT_MAX_MEMORY_CHARS`. Recall is best-effort; storage failures do not
  abort an agent run. Memory is delivered to the provider through Groq's
  `instructions` parameter, not as conversation input items.
- **Transcript**: each run's user message, tool calls, tool results, and final
  answer are persisted as an ordered transcript; a later run continues from it.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (none) | Groq API key; when unset, agent runs in mock mode. |
| `GROQ_MODEL` | (none) | Groq model name; required when `GROQ_API_KEY` is set. |
| `GROQ_BASE_URL` | (none) | Base URL for the Groq API. |
| `DATABASE_URL` | (none) | PostgreSQL connection string for session/storage. |
| `AGENT_MAX_HISTORY_MESSAGES` | 20 | Max stored messages replayed into a continuing run. |
| `AGENT_MAX_MEMORY_ITEMS` | 3 | Max memories recalled into a run. |
| `AGENT_MAX_MEMORY_CHARS` | 1200 | Max characters of recalled memory injected into a run. |
| `APPROVER_TOKEN_SECRET` | (none) | Secret keying approver credential digests. |

`MAX_ITERATIONS` is 10 by default and can be overridden per `run_agent()` call.

## Testing

Run focused tests:

```powershell
python -m pytest tests/test_agent_sessions.py tests/test_agent_history.py tests/test_agent_memory_integration.py tests/test_agent_orchestration.py tests/test_agent.py tests/test_groq_provider.py tests/test_orchestrator_tools.py -q
```

Run the full suite:

```powershell
python -m pytest -q
```

The suite uses the isolated PostgreSQL test database `ticket_triage_test`. Never run tests against production. Set `TEST_DATABASE_URL` to override the connection string.

## Retrieval Evaluation

The offline retrieval evaluation workflow (`retrieval_eval/`) runs four
retrieval configurations (BM25, Dense, Hybrid RRF, Hybrid + cross-encoder
rerank) against a labeled support dataset and produces JSON and Markdown
reports:

```powershell
python -m retrieval_eval --methods bm25,dense,hybrid,hybrid_rerank --output-dir eval_output
```
