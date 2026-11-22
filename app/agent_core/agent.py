"""Agent loop for tool-using support ticket resolution.

This module implements a support agent that:

1. Receives a support ticket message and relevant context.
2. Asks the configured LLM what to do next via the Responses API.
3. Interprets the model's tool-call requests from response output items.
4. Validates tool arguments and executes only registered tools through
   the existing ``execute_tool()`` dispatcher.
5. Sends tool results back to the model as function-call output items.
6. Continues until the model produces a final answer or the maximum
   iteration limit is reached.

The agent is independent of the existing ``POST /api/v1/tickets/analyze``
endpoint and does not modify it.

Provider compatibility notes (verified from Groq documentation):
- Groq's Responses API is fully compatible with OpenAI's Responses API.
- The model ``openai/gpt-oss-20b`` supports the Responses API.
- Groq does NOT support ``previous_response_id``; conversation history is
  managed by passing the full input on every request.
- Local tool calling (function calling) is supported; tool execution happens
  in application code.
"""

import json
from typing import Any

from openai import OpenAI

from agent_orchestrator.config import OrchestratorConfig
from agent_orchestrator.core.loop import AgentLoop
from agent_orchestrator.core.models import ExecutionStatus
from app.core.config import Settings
from app.agent_core.tool_schemas import strip_titles

from app.agent_core.groq_provider import GroqLLMProviderAdapter
from app.agent_core.orchestrator_tools import (
    ADVERTISED_TOOL_NAMES,
    build_advertised_tool_definitions,
    build_tool_executor,
    build_tool_registry,
)
from app.agent_core.tools import execute_tool


class AgentResult:
    """Result of an agent run.

    Attributes:
        status: One of "success", "mock", "provider_failure",
            "max_iterations_reached".
        answer: The final answer text from the model, if any.
        tool_calls: A list of records describing each tool call made.
        error: A safe, user-facing error message, if any.
    """

    def __init__(
        self,
        status: str,
        answer: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> None:
        self.status = status
        self.answer = answer
        self.tool_calls = tool_calls if tool_calls is not None else []
        self.error = error

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AgentResult):
            return NotImplemented
        return (
            self.status == other.status
            and self.answer == other.answer
            and self.tool_calls == other.tool_calls
            and self.error == other.error
        )

    def __repr__(self) -> str:
        return (
            f"AgentResult(status={self.status!r}, answer={self.answer!r}, "
            f"tool_calls={self.tool_calls!r}, error={self.error!r})"
        )


MAX_ITERATIONS = 10


def _get_system_prompt() -> str:
    """Return the system instruction for the support agent."""
    return (
        "You are a support ticket triage agent. Your job is to help resolve "
        "customer support tickets by gathering evidence using the available "
        "tools, then providing a final answer.\n\n"
        "Available tools:\n"
        "- search_knowledge: search the authenticated tenant's support "
        "documentation for answers to common questions about password resets, "
        "billing, login, and more.\n"
        "- create_support_case: create a support case for consequential actions "
        "(requires approval).\n\n"
        "Rules:\n"
        "1. Use tools to gather evidence before answering when the question "
        "relates to support documentation.\n"
        "2. Do not invent ticket history, support policies, or factual claims. "
        "If the tools do not return relevant information, say so honestly.\n"
        "3. When you have enough evidence, provide a clear, concise final "
        "answer in plain text.\n"
        "4. Do not ask the user for more information unless absolutely "
        "necessary.\n"
        "5. Only call tools that are registered; do not attempt to call "
        "unknown tools.\n"
    )


def _safe_tool_error(tool_name: str, error_message: str) -> dict[str, Any]:
    """Build a safe ``execute_tool``-style error result for orchestrator errors.

    The format mirrors :func:`app.agent_core.tools.execute_tool` so that
    callers can read ``result["tool"]``, ``result["error"]`` and
    ``result["results"]`` regardless of whether a tool was registered or not.
    """
    lowered = (error_message or "").lower()
    if "notfounderror" in lowered or "not registered" in lowered:
        return {
            "tool": tool_name,
            "error": "unknown_tool",
            "message": f"Tool '{tool_name}' is not registered.",
            "results": [],
        }
    if "validationerror" in lowered or "invalid" in lowered:
        return {
            "tool": tool_name,
            "error": "invalid_arguments",
            "message": f"Invalid arguments for tool '{tool_name}'.",
            "results": [],
        }
    return {
        "tool": tool_name,
        "error": "tool_failure",
        "message": f"Tool '{tool_name}' failed to execute.",
        "results": [],
    }


def run_agent(
    message: str,
    settings: Settings,
    client: OpenAI | None = None,
    max_iterations: int = MAX_ITERATIONS,
    history: list[dict[str, Any]] | None = None,
    tenant_id: str | None = None,
    db: Any = None,
    is_approved: bool = False,
) -> AgentResult:
    """Run the support agent loop using the standalone Agent Orchestrator.

    The custom per-iteration loop previously implemented here has been replaced
    by :class:`agent_orchestrator.core.loop.AgentLoop` (driven through a custom
    step function). The application-facing interface, the ``AgentResult``
    contract, mock mode, approval gating, tenant isolation, and safe error
    semantics are all preserved.

    Args:
        message: The customer's support ticket message.
        settings: Application settings (provides model, base_url).
        client: Optional OpenAI client configured for Groq. When ``None``
            (mock mode, no API key), returns a deterministic mock result.
        max_iterations: Maximum number of model-tool rounds before stopping.
        history: Optional input items of the earlier conversation, oldest
            first, as produced by the session store. They are placed in front
            of the current message, which is added exactly once.
        tenant_id: Trusted server-side tenant id injected into tool calls.
        db: Trusted database handle injected into tool calls.
        is_approved: Trusted approval state. When False, gated tools (such as
            ``create_support_case``) are refused before execution.

    Returns:
        AgentResult with status, answer, tool_calls, and optional error.
    """
    # Mock mode: no API key configured.
    if client is None:
        return AgentResult(
            status="mock",
            answer=(
                "Agent mode requires a configured Groq API key. "
                "No tool calls were made."
            ),
            tool_calls=[],
        )

    # Wire the Groq provider adapter and the application tool registry. The
    # tenant id, database, and approval state are trusted application context
    # and are bound into the tool handlers; the model cannot influence them.
    adapter = GroqLLMProviderAdapter(client, settings.groq_model)
    registry = build_tool_registry(
        tenant_id=tenant_id, db=db, is_approved=is_approved
    )
    executor = build_tool_executor(registry)
    loop = AgentLoop(
        config=OrchestratorConfig(max_iterations=max_iterations),
        tool_executor=executor,
    )

    system_prompt = _get_system_prompt()
    advertised = build_advertised_tool_definitions(registry)

    # Build the initial conversation. Groq does not support previous_response_id,
    # so the full conversation history is passed on every request. The session
    # history is placed in front of the current message, added exactly once.
    initial_messages: list[dict[str, Any]] = []
    if history:
        initial_messages.extend(history)
    initial_messages.append({"role": "system", "content": system_prompt})
    initial_messages.append({"role": "user", "content": message})

    tool_calls_record: list[dict[str, Any]] = []

    def _finish(status: str, answer: str | None = None, error: str | None = None):
        return AgentResult(
            status=status, answer=answer, tool_calls=tool_calls_record, error=error
        )

    def step_fn(state):
        nonlocal tool_calls_record
        from agent_orchestrator.core.lifecycle_helpers import (
            IterationDecision,
            IterationOutcome,
        )
        from agent_orchestrator.providers.models import ProviderRequest

        request = ProviderRequest(messages=list(state.messages), tools=advertised)

        try:
            response = adapter.complete(request)
        except Exception:
            # Provider failure: return a safe, secret-free error.
            return IterationOutcome(
                decision=IterationDecision.FAIL,
                error="provider_failure|Provider service is temporarily unavailable.",
            )

        content = response.content or ""
        tool_calls = list(response.tool_calls or [])

        if tool_calls:
            # Record the assistant's tool-call messages so the model can match
            # each tool result to the call that produced it.
            assistant_tool_calls = [
                {
                    "id": tc.id,
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in tool_calls
            ]
            state.messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": assistant_tool_calls,
                }
            )

            for tool_call in tool_calls:
                tool_result = executor.execute(tool_call, context=None)

                if tool_result.success:
                    result_dict = tool_result.output
                    if not isinstance(result_dict, dict):
                        result_dict = {
                            "tool": tool_call.name,
                            "error": "tool_failure",
                            "message": "Tool returned a non-object result.",
                            "results": [],
                        }
                else:
                    result_dict = _safe_tool_error(
                        tool_call.name, tool_result.error or ""
                    )

                tool_calls_record.append(
                    {
                        "call_id": tool_call.id,
                        "tool": tool_call.name,
                        "arguments": tool_call.arguments,
                        "result": result_dict,
                    }
                )

                # Send the tool result back to the model. The payload mirrors
                # the execute_tool result dict so ``result["tool"]`` is readable.
                state.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": json.dumps(result_dict),
                    }
                )

            # Let the model process the tool results on the next iteration.
            return IterationOutcome(decision=IterationDecision.CONTINUE)

        # No tool calls: produce a final answer.
        if content:
            return IterationOutcome(
                decision=IterationDecision.COMPLETE, final_response=content
            )

        # No tool call and no text: incomplete response.
        return IterationOutcome(
            decision=IterationDecision.FAIL,
            error="The model did not produce a final answer or a tool call.",
        )

    try:
        state, result = loop.run(messages=initial_messages, step=step_fn)
    except Exception as exc:  # noqa: BLE001 - convert to a safe agent error
        import logging

        logging.getLogger("agent").exception(
            "Agent execution failed: %s", type(exc).__name__
        )
        return _finish(
            status="provider_failure",
            error="Provider service is temporarily unavailable.",
        )

    if result.status == ExecutionStatus.COMPLETED:
        return _finish(status="success", answer=result.final_response)

    if result.status == ExecutionStatus.FAILED:
        err = result.error or ""
        if err.startswith("provider_failure|"):
            return _finish(
                status="provider_failure",
                error="Provider service is temporarily unavailable.",
            )
        return _finish(status="max_iterations_reached", error=err)

    # MAX_ITERATIONS_REACHED
    return _finish(
        status="max_iterations_reached",
        error=(
            f"Agent reached the maximum of {max_iterations} iterations "
            "without producing a final answer."
        ),
    )

