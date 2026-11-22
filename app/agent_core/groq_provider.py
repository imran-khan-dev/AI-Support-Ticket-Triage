"""Groq provider adapter for the Agent Orchestrator.

Wraps the existing OpenAI-compatible Groq client and translates between the
orchestrator's provider-agnostic :class:`ProviderRequest`/:class:`ProviderResponse`
and Groq's Responses API (``client.responses.create(...)``).

The adapter is intentionally independent of application tenant, session,
approval, and database logic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Make the standalone Agent Orchestrator package importable without requiring it
# to be installed as a dependency. This mirrors how the Retrieval Stack and Eval
# Harness are wired into the backend.
_ORCHESTRATOR_ROOT = Path(__file__).resolve().parents[2] / "Agent Orchestrator"
if _ORCHESTRATOR_ROOT.is_dir() and str(_ORCHESTRATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATOR_ROOT))

from agent_orchestrator.exceptions import ProviderError
from agent_orchestrator.providers.base import LLMProvider
from agent_orchestrator.providers.models import ProviderRequest, ProviderResponse
from agent_orchestrator.tools.models import ToolCall

from openai import OpenAI


class GroqLLMProviderAdapter:
    """Adapter wrapping a Groq (OpenAI-compatible) client as an LLMProvider."""

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Send a request to Groq and return a normalized response."""
        try:
            input_items = self._convert_messages(request.messages)
            tools = self._convert_tools(request.tools)
            instructions = self._extract_instructions(request.messages)

            response = self._client.responses.create(
                model=self._model,
                instructions=instructions,
                input=input_items,
                tools=tools or None,
                tool_choice=request.tool_choice or None,
                temperature=0.0,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Groq provider request failed: {type(exc).__name__}: {exc}"
            ) from exc

        return self._normalize_response(response)

    # -- request translation -------------------------------------------------

    @staticmethod
    def _extract_instructions(messages: list[dict[str, Any]]) -> str | None:
        """Combine system messages into the `instructions` parameter."""
        parts: list[str] = []
        for message in messages:
            if message.get("role") == "system":
                text = GroqLLMProviderAdapter._message_text(message)
                if text:
                    parts.append(text)
        return "\n".join(parts) if parts else None

    @staticmethod
    def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert orchestrator-style messages to Groq Responses API input items."""
        items: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                # System messages are carried by the `instructions` parameter.
                continue
            items.extend(GroqLLMProviderAdapter._message_to_items(message))
        return items

    @staticmethod
    def _message_to_items(message: dict[str, Any]) -> list[dict[str, Any]]:
        role = message.get("role")
        if role == "user":
            return [GroqLLMProviderAdapter._user_item(message)]
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if tool_calls:
                return [
                    GroqLLMProviderAdapter._function_call_item(tc)
                    for tc in tool_calls
                ]
            return [GroqLLMProviderAdapter._assistant_item(message)]
        if role == "tool":
            return [GroqLLMProviderAdapter._function_call_output_item(message)]
        # Unknown roles are skipped so malformed input never reaches Groq.
        return []

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        """Return the plain text of a message's content."""
        content = message.get("content")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if text:
                        parts.append(text)
                elif isinstance(part, str):
                    parts.append(part)
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _user_item(message: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": GroqLLMProviderAdapter._message_text(message)}
            ],
        }

    @staticmethod
    def _assistant_item(message: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": GroqLLMProviderAdapter._message_text(message)}
            ],
        }

    @staticmethod
    def _function_call_item(tool_call: dict[str, Any]) -> dict[str, Any]:
        arguments = tool_call.get("arguments")
        if isinstance(arguments, str):
            arguments_json = arguments
        elif arguments is None:
            arguments_json = ""
        else:
            arguments_json = json.dumps(arguments)
        call_id = tool_call.get("call_id") or tool_call.get("id")
        return {
            "type": "function_call",
            "id": call_id,
            "call_id": call_id,
            "name": tool_call.get("name", ""),
            "arguments": arguments_json,
        }

    @staticmethod
    def _function_call_output_item(message: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function_call_output",
            "call_id": message.get("tool_call_id") or message.get("call_id"),
            "name": message.get("name", ""),
            "output": GroqLLMProviderAdapter._message_text(message),
        }

    @staticmethod
    def _convert_tools(
        tools: list[Any] | None,
    ) -> list[dict[str, Any]] | None:
        """Convert ToolDefinition objects to Grox function tool definitions."""
        if not tools:
            return None
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, dict):
                name = tool.get("name", "")
                description = tool.get("description", "")
                parameters = tool.get(
                    "parameters_schema", tool.get("parameters", {})
                )
            else:
                name = getattr(tool, "name", "")
                description = getattr(tool, "description", "")
                parameters = getattr(tool, "parameters_schema", {}) or {}
            converted.append(
                {
                    "type": "function",
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }
            )
        return converted

    # -- response normalization ----------------------------------------------

    @staticmethod
    def _normalize_response(response: Any) -> ProviderResponse:
        output_items = list(getattr(response, "output", None) or [])
        output_text = getattr(response, "output_text", None) or ""

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for item in output_items:
            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                tool_calls.append(GroqLLMProviderAdapter._coerce_function_call(item))
            elif item_type == "message":
                content_parts.extend(
                    GroqLLMProviderAdapter._message_text_parts(item)
                )

        content = output_text.strip() or "".join(content_parts).strip() or None

        finish_reason = getattr(response, "finish_reason", None)
        if finish_reason is not None and not isinstance(finish_reason, str):
            finish_reason = str(finish_reason)

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _message_text_parts(message_item: Any) -> list[str]:
        parts: list[str] = []
        content = getattr(message_item, "content", None)
        if not content:
            return parts
        for part in content:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
        return parts

    @staticmethod
    def _coerce_function_call(item: Any) -> ToolCall:
        call_id = getattr(item, "call_id", None) or getattr(item, "id", None) or ""
        name = getattr(item, "name", None) or ""
        arguments = GroqLLMProviderAdapter._parse_arguments(
            getattr(item, "arguments", None)
        )
        return ToolCall(
            id=call_id or "unknown",
            name=name or "unknown",
            arguments=arguments,
        )

    @staticmethod
    def _parse_arguments(raw_arguments: Any) -> dict[str, Any]:
        """Parse Groq's JSON-string arguments into a dict, safely."""
        if raw_arguments is None:
            return {}
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return {}
