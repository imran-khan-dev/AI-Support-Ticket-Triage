"""Tests for the Groq provider adapter.

All tests use mocked OpenAI clients and never make real API calls.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from app.agent_core.groq_provider import GroqLLMProviderAdapter

from agent_orchestrator.exceptions import ProviderError
from agent_orchestrator.providers.models import ProviderRequest
from agent_orchestrator.tools.models import ToolCall, ToolDefinition


def _make_response(output_items=None, output_text=""):
    mock_response = Mock()
    mock_response.output = output_items if output_items is not None else []
    mock_response.output_text = output_text
    mock_response.finish_reason = None
    return mock_response


def _make_function_call(call_id, name, arguments):
    item = Mock()
    item.type = "function_call"
    item.call_id = call_id
    item.name = name
    item.arguments = arguments
    return item


def _make_text_message(text):
    part = Mock()
    part.text = text
    item = Mock()
    item.type = "message"
    item.content = [part]
    return item


def _make_system_message(text):
    part = Mock()
    part.text = text
    item = Mock()
    item.type = "message"
    item.role = "system"
    item.content = [part]
    return item


def _tool(name="search_knowledge", description="Search support docs"):
    return ToolDefinition(
        name=name,
        description=description,
        parameters_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )


def _adapter(client=None, model="openai/gpt-oss-20b"):
    return GroqLLMProviderAdapter(client or Mock(), model)


# ---------------------------------------------------------------------------
# Normal assistant text response
# ---------------------------------------------------------------------------


def test_normal_text_response():
    client = Mock()
    client.responses.create.return_value = _make_response(
        output_items=[_make_text_message("Your password can be reset here.")],
        output_text="Your password can be reset here.",
    )
    adapter = _adapter(client)
    response = adapter.complete(
        ProviderRequest(messages=[{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}])
    )
    assert response.content == "Your password can be reset here."
    assert response.tool_calls == []
    assert client.responses.create.called


# ---------------------------------------------------------------------------
# One function call
# ---------------------------------------------------------------------------


def test_one_function_call():
    client = Mock()
    client.responses.create.return_value = _make_response(
        output_items=[_make_function_call("call_1", "search_knowledge", '{"query": "password reset"}')],
        output_text="",
    )
    adapter = _adapter(client)
    response = adapter.complete(
        ProviderRequest(messages=[{"role": "user", "content": "How do I reset my password?"}])
    )
    assert response.content is None
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "search_knowledge"
    assert call.arguments == {"query": "password reset"}


# ---------------------------------------------------------------------------
# Multiple function calls
# ---------------------------------------------------------------------------


def test_multiple_function_calls():
    client = Mock()
    client.responses.create.return_value = _make_response(
        output_items=[
            _make_function_call("call_1", "search_knowledge", '{"query": "password reset"}'),
            _make_function_call("call_2", "lookup_ticket_history", '{"query": "password reset"}'),
        ],
        output_text="",
    )
    adapter = _adapter(client)
    response = adapter.complete(
        ProviderRequest(messages=[{"role": "user", "content": "Help me with password reset."}])
    )
    assert len(response.tool_calls) == 2
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "search_knowledge"
    assert response.tool_calls[1].id == "call_2"
    assert response.tool_calls[1].name == "lookup_ticket_history"


# ---------------------------------------------------------------------------
# JSON-string arguments
# ---------------------------------------------------------------------------


def test_arguments_returned_as_json_string_are_parsed():
    client = Mock()
    client.responses.create.return_value = _make_response(
        output_items=[
            _make_function_call("call_1", "search_knowledge", '{"query": "billing refund", "top_k": 5}')
        ],
        output_text="",
    )
    adapter = _adapter(client)
    response = adapter.complete(
        ProviderRequest(messages=[{"role": "user", "content": "I need a refund."}])
    )
    assert response.tool_calls[0].arguments == {"query": "billing refund", "top_k": 5}


# ---------------------------------------------------------------------------
# Empty response
# ---------------------------------------------------------------------------


def test_empty_response():
    client = Mock()
    client.responses.create.return_value = _make_response(output_items=[], output_text="")
    adapter = _adapter(client)
    response = adapter.complete(
        ProviderRequest(messages=[{"role": "user", "content": "Hello"}])
    )
    assert response.content is None
    assert response.tool_calls == []


def test_none_response_is_handled():
    client = Mock()
    client.responses.create.return_value = None
    adapter = _adapter(client)
    response = adapter.complete(
        ProviderRequest(messages=[{"role": "user", "content": "Hello"}])
    )
    assert response.content is None
    assert response.tool_calls == []


# ---------------------------------------------------------------------------
# Malformed function-call data
# ---------------------------------------------------------------------------


def test_malformed_arguments_yield_empty_dict():
    client = Mock()
    item = Mock()
    item.type = "function_call"
    item.call_id = "call_1"
    item.name = "search_knowledge"
    item.arguments = "not-json{{{"
    client.responses.create.return_value = _make_response(
        output_items=[item], output_text=""
    )
    adapter = _adapter(client)
    response = adapter.complete(
        ProviderRequest(messages=[{"role": "user", "content": "Hello"}])
    )
    assert response.tool_calls[0].arguments == {}


def test_function_call_missing_ids_falls_back():
    client = Mock()
    item = Mock()
    item.type = "function_call"
    item.call_id = None
    item.id = None
    item.name = "search_knowledge"
    item.arguments = "{}"
    client.responses.create.return_value = _make_response(
        output_items=[item], output_text=""
    )
    adapter = _adapter(client)
    response = adapter.complete(
        ProviderRequest(messages=[{"role": "user", "content": "Hello"}])
    )
    assert response.tool_calls[0].id == "unknown"


# ---------------------------------------------------------------------------
# Provider exception handling
# ---------------------------------------------------------------------------


def test_provider_exception_mapped_to_provider_error():
    client = Mock()
    client.responses.create.side_effect = RuntimeError("connection reset")
    adapter = _adapter(client)
    with pytest.raises(ProviderError) as exc_info:
        adapter.complete(
            ProviderRequest(messages=[{"role": "user", "content": "Hello"}])
        )
    assert "connection reset" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Correct request construction
# ---------------------------------------------------------------------------


def test_request_construction_passes_model_instructions_and_tools():
    client = Mock()
    client.responses.create.return_value = _make_response(
        output_items=[_make_text_message("ok")], output_text="ok"
    )
    adapter = _adapter(client, model="openai/gpt-oss-20b")
    request = ProviderRequest(
        messages=[
            {"role": "system", "content": [{"type": "input_text", "text": "You are a helper."}]},
            {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
        ],
        tools=[_tool()],
    )
    adapter.complete(request)

    call_kwargs = client.responses.create.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-oss-20b"
    assert call_kwargs["instructions"] == "You are a helper."
    assert call_kwargs["temperature"] == 0.0
    assert isinstance(call_kwargs["input"], list)
    assert any(item.get("role") == "user" for item in call_kwargs["input"])
    assert not any(item.get("role") == "system" for item in call_kwargs["input"])
    assert call_kwargs["tools"] == [
        {
            "type": "function",
            "name": "search_knowledge",
            "description": "Search support docs",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]


def test_tool_calls_from_messages_are_sent_as_function_call_items():
    client = Mock()
    client.responses.create.return_value = _make_response(
        output_items=[_make_text_message("done")], output_text="done"
    )
    adapter = _adapter(client)
    adapter.complete(
        ProviderRequest(
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "call_id": "tc_1",
                            "name": "search_knowledge",
                            "arguments": {"query": "x"},
                        }
                    ],
                }
            ]
        )
    )
    input_items = client.responses.create.call_args.kwargs["input"]
    assert any(
        item.get("type") == "function_call"
        and item.get("call_id") == "tc_1"
        and item.get("name") == "search_knowledge"
        for item in input_items
    )


def test_implements_llm_provider_protocol():
    from agent_orchestrator.providers.base import LLMProvider

    adapter = _adapter()
    assert isinstance(adapter, LLMProvider)
