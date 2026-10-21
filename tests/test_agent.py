import json
from unittest.mock import Mock, patch

import pytest

from app.core.config import Settings
from app.agent_core.agent import AgentResult, MAX_ITERATIONS, run_agent


def _make_response(output_items=None, output_text=""):
    """Build a mock OpenAI Responses API response object."""
    mock_response = Mock()
    mock_response.output = output_items if output_items is not None else []
    mock_response.output_text = output_text
    return mock_response


def _make_function_call(call_id, name, arguments):
    """Build a mock function-call output item."""
    item = Mock()
    item.type = "function_call"
    item.call_id = call_id
    item.name = name
    item.arguments = arguments
    return item


def _make_text_message(text):
    """Build a mock message output item with text content."""
    part = Mock()
    part.text = text
    item = Mock()
    item.type = "message"
    item.content = [part]
    return item


# ---------------------------------------------------------------------------
# Fixtures and settings
# ---------------------------------------------------------------------------


def _settings():
    return Settings(
        _env_file=None,
        groq_api_key="test_key",
        groq_model="openai/gpt-oss-20b",
        groq_base_url="https://api.groq.com/openai/v1",
    )


# ---------------------------------------------------------------------------
# 1. Model returns a final answer without requesting a tool.
# ---------------------------------------------------------------------------


def test_agent_returns_final_answer_without_tool():
    settings = _settings()
    with patch('app.agent_core.agent.OpenAI') as MockOpenAI:
        mock_client = Mock()
        mock_response = _make_response(
            output_items=[_make_text_message("Your password can be reset in account settings.")],
            output_text="Your password can be reset in account settings.",
        )
        mock_client.responses.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        result = run_agent("How do I reset my password?", settings, mock_client)

        assert result.status == "success"
        assert result.answer == "Your password can be reset in account settings."
        assert result.tool_calls == []
        assert result.error is None


# ---------------------------------------------------------------------------
# 2. Model requests one registered tool and then returns a final answer.
# ---------------------------------------------------------------------------


def test_agent_executes_one_tool_then_answers():
    settings = _settings()
    with patch('app.agent_core.agent.OpenAI') as MockOpenAI:
        mock_client = Mock()
        # First response: requests search_support_docs tool
        first_response = _make_response(
            output_items=[
                _make_function_call(
                    "call_1", "search_support_docs", '{"query": "password reset"}'
                )
            ],
            output_text="",
        )
        # Second response: final answer after tool result
        second_response = _make_response(
            output_items=[
                _make_text_message(
                    "To reset your password, visit account settings and use "
                    "'Forgot password'."
                )
            ],
            output_text=(
                "To reset your password, visit account settings and use "
                "'Forgot password'."
            ),
        )
        mock_client.responses.create.side_effect = [
            first_response,
            second_response,
        ]
        MockOpenAI.return_value = mock_client

        result = run_agent("How do I reset my password?", settings, mock_client)

        assert result.status == "success"
        assert "Forgot password" in result.answer
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "search_support_docs"
        assert result.tool_calls[0]["call_id"] == "call_1"
        assert result.tool_calls[0]["arguments"] == {"query": "password reset"}
        assert result.tool_calls[0]["result"]["tool"] == "search_support_docs"
        assert len(result.tool_calls[0]["result"]["results"]) >= 1
        assert result.error is None

        # Verify the model was called twice (once with tool request, once
        # with tool result fed back).
        assert mock_client.responses.create.call_count == 2


# ---------------------------------------------------------------------------
# 3. Model requests multiple tools in successive rounds.
# ---------------------------------------------------------------------------


def test_agent_executes_multiple_tools_in_successive_rounds():
    settings = _settings()
    with patch('app.agent_core.agent.OpenAI') as MockOpenAI:
        mock_client = Mock()
        # Round 1: search_support_docs
        round1 = _make_response(
            output_items=[
                _make_function_call(
                    "call_1", "search_support_docs", '{"query": "billing refund"}'
                )
            ],
            output_text="",
        )
        # Round 2: lookup_ticket_history
        round2 = _make_response(
            output_items=[
                _make_function_call(
                    "call_2", "lookup_ticket_history", '{"query": "billing refund"}'
                )
            ],
            output_text="",
        )
        # Round 3: final answer
        round3 = _make_response(
            output_items=[
                _make_text_message(
                    "Refund requests are reviewed within 5 business days."
                )
            ],
            output_text="Refund requests are reviewed within 5 business days.",
        )
        mock_client.responses.create.side_effect = [round1, round2, round3]
        MockOpenAI.return_value = mock_client

        result = run_agent("I need a refund for my subscription.", settings, mock_client)

        assert result.status == "success"
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["tool"] == "search_support_docs"
        assert result.tool_calls[1]["tool"] == "lookup_ticket_history"
        assert mock_client.responses.create.call_count == 3


# ---------------------------------------------------------------------------
# 4. Tool results are passed back to the model correctly.
# ---------------------------------------------------------------------------


def test_agent_passes_tool_results_back_to_model():
    settings = _settings()
    with patch('app.agent_core.agent.OpenAI') as MockOpenAI:
        mock_client = Mock()
        first_response = _make_response(
            output_items=[
                _make_function_call(
                    "call_1", "lookup_ticket_history", '{"query": "password reset"}'
                )
            ],
            output_text="",
        )
        second_response = _make_response(
            output_items=[
                _make_text_message(
                    "Ticket TKT-1001 was resolved by verifying identity."
                )
            ],
            output_text="Ticket TKT-1001 was resolved by verifying identity.",
        )
        mock_client.responses.create.side_effect = [first_response, second_response]
        MockOpenAI.return_value = mock_client

        result = run_agent("What happened with my password reset ticket?", settings, mock_client)

        assert result.status == "success"
        # Verify the second call received the tool result in its input.
        second_call_args = mock_client.responses.create.call_args_list[1]
        input_items = second_call_args.kwargs["input"]
        # Should contain the original user message plus the tool result.
        assert any(
            item.get("type") == "function_call_output"
            and item.get("call_id") == "call_1"
            for item in input_items
        )
        tool_result_item = next(
            item for item in input_items
            if item.get("type") == "function_call_output"
        )
        assert tool_result_item["call_id"] == "call_1"
        # The Responses API requires `output` to be a JSON string, not a dict.
        assert isinstance(tool_result_item["output"], str)
        tool_output = json.loads(tool_result_item["output"])
        assert tool_output["tool"] == "lookup_ticket_history"


# ---------------------------------------------------------------------------
# 5. Multiple tool calls in one model response are handled correctly.
# ---------------------------------------------------------------------------


def test_agent_handles_multiple_tool_calls_in_one_response():
    settings = _settings()
    with patch('app.agent_core.agent.OpenAI') as MockOpenAI:
        mock_client = Mock()
        # One response with two tool calls
        first_response = _make_response(
            output_items=[
                _make_function_call(
                    "call_1", "search_support_docs", '{"query": "password reset"}'
                ),
                _make_function_call(
                    "call_2", "lookup_ticket_history", '{"query": "password reset"}'
                ),
            ],
            output_text="",
        )
        second_response = _make_response(
            output_items=[
                _make_text_message(
                    "Here is what I found about password resets."
                )
            ],
            output_text="Here is what I found about password resets.",
        )
        mock_client.responses.create.side_effect = [first_response, second_response]
        MockOpenAI.return_value = mock_client

        result = run_agent("Help me with password reset.", settings, mock_client)

        assert result.status == "success"
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["call_id"] == "call_1"
        assert result.tool_calls[1]["call_id"] == "call_2"
        assert mock_client.responses.create.call_count == 2


# ---------------------------------------------------------------------------
# 6. Unknown tool names cannot execute arbitrary functions.
# ---------------------------------------------------------------------------


def test_agent_unknown_tool_cannot_execute():
    settings = _settings()
    with patch('app.agent_core.agent.OpenAI') as MockOpenAI:
        mock_client = Mock()
        # Model requests a tool that is not registered
        first_response = _make_response(
            output_items=[
                _make_function_call(
                    "call_1", "delete_customer_data", '{"query": "test"}'
                )
            ],
            output_text="",
        )
        second_response = _make_response(
            output_items=[
                _make_text_message(
                    "I cannot help with that request."
                )
            ],
            output_text="I cannot help with that request.",
        )
        mock_client.responses.create.side_effect = [first_response, second_response]
        MockOpenAI.return_value = mock_client

        result = run_agent("Delete all my data.", settings, mock_client)

        assert result.status == "success"
        # The unknown tool was rejected by execute_tool, not executed.
        assert result.tool_calls[0]["result"]["error"] == "unknown_tool"
        assert result.tool_calls[0]["result"]["results"] == []


# ---------------------------------------------------------------------------
# 7. Malformed tool arguments are handled safely.
# ---------------------------------------------------------------------------


def test_agent_malformed_arguments_handled_safely():
    settings = _settings()
    with patch('app.agent_core.agent.OpenAI') as MockOpenAI:
        mock_client = Mock()
        # Model sends invalid JSON as arguments
        first_response = _make_response(
            output_items=[
                _make_function_call(
                    "call_1", "search_support_docs", "not valid json{{{"
                )
            ],
            output_text="",
        )
        second_response = _make_response(
            output_items=[
                _make_text_message(
                    "I had trouble understanding that request."
                )
            ],
            output_text="I had trouble understanding that request.",
        )
        mock_client.responses.create.side_effect = [first_response, second_response]
        MockOpenAI.return_value = mock_client

        result = run_agent("Help me.", settings, mock_client)

        assert result.status == "success"
        # Malformed args should produce an invalid_arguments error.
        assert result.tool_calls[0]["result"]["error"] == "invalid_arguments"


# ---------------------------------------------------------------------------
# 8. Tool validation failures do not crash the application.
# ---------------------------------------------------------------------------


def test_agent_validation_failure_does_not_crash():
    settings = _settings()
    with patch('app.agent_core.agent.OpenAI') as MockOpenAI:
        mock_client = Mock()
        # Model sends empty query (fails Pydantic validation)
        first_response = _make_response(
            output_items=[
                _make_function_call(
                    "call_1", "search_support_docs", '{"query": ""}'
                )
            ],
            output_text="",
        )
        second_response = _make_response(
            output_items=[
                _make_text_message("Please provide a search term.")
            ],
            output_text="Please provide a search term.",
        )
        mock_client.responses.create.side_effect = [first_response, second_response]
        MockOpenAI.return_value = mock_client

        result = run_agent("Help me.", settings, mock_client)

        assert result.status == "success"
        assert result.tool_calls[0]["result"]["error"] == "invalid_arguments"


# ---------------------------------------------------------------------------
# 9. Maximum iteration limit stops repeated tool calls.
# ---------------------------------------------------------------------------


def test_agent_max_iteration_limit_stops_loop():
    settings = _settings()
    with patch('app.agent_core.agent.OpenAI') as MockOpenAI:
        mock_client = Mock()
        # Every response requests another tool call, never a final answer.
        tool_call_response = _make_response(
            output_items=[
                _make_function_call(
                    "call_loop", "search_support_docs", '{"query": "test"}'
                )
            ],
            output_text="",
        )
        mock_client.responses.create.return_value = tool_call_response
        MockOpenAI.return_value = mock_client

        result = run_agent("Keep searching.", settings, mock_client)

        assert result.status == "max_iterations_reached"
        assert result.error is not None
        assert "maximum" in result.error.lower()
        assert result.tool_calls is not None
        # Should have stopped at MAX_ITERATIONS, not looped forever.
        assert len(result.tool_calls) <= MAX_ITERATIONS
        assert mock_client.responses.create.call_count == MAX_ITERATIONS


# ---------------------------------------------------------------------------
# 10. Provider exceptions produce safe errors.
# ---------------------------------------------------------------------------


def test_agent_provider_exception_produces_safe_error():
    settings = _settings()
    with patch('app.agent_core.agent.OpenAI') as MockOpenAI:
        mock_client = Mock()
        mock_client.responses.create.side_effect = Exception(
            "Connection reset by peer with api_key=supersecret"
        )
        MockOpenAI.return_value = mock_client

        result = run_agent("Help me.", settings, mock_client)

        assert result.status == "provider_failure"
        assert result.error is not None
        assert "temporarily unavailable" in result.error
        # No secret leakage.
        assert "supersecret" not in result.error
        assert "api_key" not in result.error.lower()
        assert result.tool_calls == []


# ---------------------------------------------------------------------------
# 11. Missing API key / mock-mode behavior.
# ---------------------------------------------------------------------------


def test_agent_mock_mode_without_api_key():
    settings = Settings(_env_file=None)  # no API key
    client = None  # simulate no client configured

    result = run_agent("Help me.", settings, client)

    assert result.status == "mock"
    assert result.answer is not None
    assert "API key" in result.answer
    assert result.tool_calls == []
    assert result.error is None


# ---------------------------------------------------------------------------
# 12. Existing /analyze endpoint and tests remain unaffected.
# ---------------------------------------------------------------------------


def test_agent_does_not_affect_existing_analyze_service():
    """Sanity check: existing analysis service still works independently."""
    from app.agent_core.analysis import mock_analysis
    from app.ai.provider import get_groq_client

    settings = Settings(_env_file=None)
    client = get_groq_client(settings)
    assert client is None
    response = mock_analysis()
    assert response.category == "technical"
