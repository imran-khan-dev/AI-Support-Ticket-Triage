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

from app.core.config import Settings
from app.agent_core.tools import TOOL_REGISTRY, execute_tool


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


def _strip_titles(schema_obj: Any) -> Any:
    """Recursively remove ``title`` keys from a JSON schema dict.

    Pydantic's ``model_json_schema()`` adds ``title`` fields at the root and
    on every property. OpenAI's function tool parameters do not need them and
    some providers reject unknown schema fields.
    """
    if isinstance(schema_obj, dict):
        return {
            key: _strip_titles(value)
            for key, value in schema_obj.items()
            if key != "title"
        }
    if isinstance(schema_obj, list):
        return [_strip_titles(item) for item in schema_obj]
    return schema_obj

# Prepare the tools for Groq
def _build_tool_definitions() -> list[dict[str, Any]]:
    """Build OpenAI function tool definitions from the registered tools.

    Derives JSON schemas from the Pydantic argument schemas in
    ``app.tool_schemas`` so tool argument validation is defined in one place.
    """
    definitions: list[dict[str, Any]] = []
    for tool_name, (args_schema, _func) in TOOL_REGISTRY.items():
        json_schema = args_schema.model_json_schema()
        parameters = _strip_titles(json_schema)
        parameters["additionalProperties"] = False
        description = ""
        if "query" in args_schema.model_fields:
            field_info = args_schema.model_fields["query"]
            if field_info.description:
                description = field_info.description
        definitions.append(
            {
                "type": "function",
                "name": tool_name,
                "description": description,
                "parameters": parameters,
            }
        )
    return definitions


def _get_system_prompt() -> str:
    """Return the system instruction for the support agent."""
    return (
        "You are a support ticket triage agent. Your job is to help resolve "
        "customer support tickets by gathering evidence using the available "
        "tools, then providing a final answer.\n\n"
        "Available tools:\n"
        "- search_support_docs: search support documentation for answers to "
        "common questions about password resets, billing, login, and more.\n"
        "- lookup_ticket_history: look up previous ticket records for similar "
        "issues and their resolutions.\n\n"
        "Rules:\n"
        "1. Use tools to gather evidence before answering when the question "
        "relates to support documentation or previous tickets.\n"
        "2. Do not invent ticket history, support policies, or factual claims. "
        "If the tools do not return relevant information, say so honestly.\n"
        "3. When you have enough evidence, provide a clear, concise final "
        "answer in plain text.\n"
        "4. Do not ask the user for more information unless absolutely "
        "necessary.\n"
        "5. Only call tools that are registered; do not attempt to call "
        "unknown tools.\n"
    )


def _extract_text_from_output(output_items: list[Any]) -> str:
    """Extract plain text from message output items."""
    parts: list[str] = []
    for item in output_items:
        content = getattr(item, "content", None)
        if not content:
            continue
        for part in content:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
    return "".join(parts)


def _extract_function_calls(output_items: list[Any]) -> list[Any]:
    """Return output items that are function-call tool requests."""
    calls: list[Any] = []
    for item in output_items:
        if getattr(item, "type", None) == "function_call":
            calls.append(item)
    return calls

def run_agent(
    message: str,
    settings: Settings,
    client: OpenAI | None = None,
    max_iterations: int = MAX_ITERATIONS,
    history: list[dict[str, Any]] | None = None,
) -> AgentResult:
    """Run the support agent loop.

    Args:
        message: The customer's support ticket message.
        settings: Application settings (provides model, base_url).
        client: Optional OpenAI client configured for Groq. When ``None``
            (mock mode, no API key), returns a deterministic mock result.
        max_iterations: Maximum number of model-tool rounds before stopping.
        history: Optional input items of the earlier conversation, oldest
            first, as produced by the session store. They are placed in front
            of the current message, which is added exactly once.

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

    tool_definitions = _build_tool_definitions()
    system_prompt = _get_system_prompt()

    # Build the initial input. Groq does not support previous_response_id, so
    # the full conversation history is passed on every request. The history
    # of the session, when there is one, is placed in front of the current
    # message; the current message itself is added exactly once, here.
    input_items: list[dict[str, Any]] = []
    if history:
        input_items.extend(history)
    input_items.append(
        {
            "id": f"input_{len(input_items) + 1}",
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": message}],
        }
    )

    tool_calls_record: list[dict[str, Any]] = []

    for iteration in range(max_iterations):
        try:
            response = client.responses.create(
                model=settings.groq_model,
                instructions=system_prompt,
                input=input_items,
                tools=tool_definitions,
                temperature=0.0,
            )
        except Exception as exc:
            import logging

            logging.getLogger("agent").exception(
                "Agent loop iteration %d failed: %s",
                iteration,
                type(exc).__name__,
            )
            return AgentResult(
                status="provider_failure",
                error="Provider service is temporarily unavailable.",
                tool_calls=tool_calls_record,
            )

        output_items = list(getattr(response, "output", None) or [])
        output_text = getattr(response, "output_text", None) or ""

        function_calls = _extract_function_calls(output_items)

        # If the model requests tool calls, execute each one and feed results
        # back to the model on the next iteration.
        if function_calls:
            # Add the assistant's tool-call messages to the input so that
            # Groq's Harmony renderer can associate each tool output with the
            # tool call that produced it. Without these preceding messages,
            # the renderer reports "Tools should have a name!" because it
            # cannot find the tool call that a function_call_output refers to.
            for item in output_items:
                if getattr(item, "type", None) == "function_call":
                    model_dump = getattr(item, "model_dump", None)
                    if callable(model_dump):
                        try:
                            dumped = model_dump()
                            if isinstance(dumped, dict):
                                input_items.append(dumped)
                                continue
                        except Exception:
                            pass
                    input_items.append(
                        {
                            "id": getattr(item, "id", None),
                            "type": "function_call",
                            "call_id": getattr(item, "call_id", None),
                            "name": getattr(item, "name", None),
                            "arguments": getattr(item, "arguments", None),
                        }
                    )

            for tool_call in function_calls:
                call_id = getattr(tool_call, "call_id", None)
                tool_name = getattr(tool_call, "name", None)
                raw_arguments = getattr(tool_call, "arguments", None)

                # Parse arguments safely; malformed JSON is caught below.
                arguments: dict[str, Any] = {}
                if raw_arguments:
                    try:
                        arguments = json.loads(raw_arguments)
                        if not isinstance(arguments, dict):
                            arguments = {}
                    except Exception:
                        arguments = {}

                # Execute through the registry dispatcher. Unknown tool names
                # and invalid arguments are handled safely inside execute_tool.
                result = execute_tool(tool_name or "", arguments)

                tool_calls_record.append(
                    {
                        "call_id": call_id,
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": result,
                    }
                )

                # Add the tool result to the input so the model can use it.
                # The Responses API expects ``output`` to be a string (or list of
                # content items), not a raw dict. Serialize the tool result as
                # JSON so the SDK can send it to the provider. Include the tool
                # name so Groq's Harmony renderer can associate the output with
                # the tool that was called.
                input_items.append(
                    {
                        "id": f"output_{iteration}_{len(input_items)}",
                        "type": "function_call_output",
                        "status": "completed",
                        "call_id": call_id,
                        "name": tool_name,
                        "output": json.dumps(result),
                    }
                )

            # Continue the loop to let the model process tool results.
            continue

        # No tool calls: look for a final text answer.
        final_text = output_text.strip() or _extract_text_from_output(
            output_items
        )
        if final_text:
            return AgentResult(
                status="success",
                answer=final_text,
                tool_calls=tool_calls_record,
            )

        # No tool call and no text: incomplete response.
        return AgentResult(
            status="max_iterations_reached",
            error=(
                "The model did not produce a final answer or a tool call."
            ),
            tool_calls=tool_calls_record,
        )

    # Maximum iterations reached without a final answer.
    return AgentResult(
        status="max_iterations_reached",
        error=(
            f"Agent reached the maximum of {max_iterations} iterations "
            "without producing a final answer."
        ),
        tool_calls=tool_calls_record,
    )
