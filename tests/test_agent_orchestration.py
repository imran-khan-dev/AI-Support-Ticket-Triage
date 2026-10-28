"""Milestone 2 orchestration verification for the tool-using agent.

Every test here is deterministic: the provider is replaced with a fake
``client.responses.create`` so the full agent loop runs without network
access. These tests do not prove live-provider compatibility; they verify the
orchestration contract against the Responses API request shape the agent
builds:

1. Multiple tool calls in one response all execute and return with the
   correct call IDs.
2. Repeated tool-call rounds complete with a final answer.
3. Every ``function_call_output`` correlates to its ``function_call`` by
   ``call_id``.
4. The agent never exceeds ``max_iterations`` and reports the limit.
5. Invalid arguments, unknown tools, tool execution failures and provider
   failures are handled correctly.
6. Mock mode, the ticket-analysis path, read-only tools and the registered
   tool permission boundary remain intact.
"""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agent_core.agent import AgentResult, MAX_ITERATIONS, run_agent
from app.agent_core.tool_schemas import SearchSupportDocsArgs
from app.agent_core.tools import TOOL_REGISTRY, execute_tool
from app.core.config import Settings


# ---------------------------------------------------------------------------
# Deterministic fakes for the Responses API
# ---------------------------------------------------------------------------


class _SdkFunctionCall:
    """Stand-in for an OpenAI SDK ``function_call`` output item."""

    def __init__(self, call_id, name, arguments, item_id=None):
        self.type = "function_call"
        self.id = item_id or f"fc_{call_id}"
        self.call_id = call_id
        self.name = name
        self.arguments = arguments

    def model_dump(self):
        return {
            "id": self.id,
            "type": self.type,
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
        }


class _BareFunctionCall:
    """Function-call item without ``model_dump`` (fallback serialization path)."""

    def __init__(self, call_id, name, arguments):
        self.type = "function_call"
        self.id = f"fc_{call_id}"
        self.call_id = call_id
        self.name = name
        self.arguments = arguments


class _BrokenDumpFunctionCall:
    """Function-call item whose ``model_dump`` raises."""

    def __init__(self, call_id, name, arguments):
        self.type = "function_call"
        self.id = f"fc_{call_id}"
        self.call_id = call_id
        self.name = name
        self.arguments = arguments

    def model_dump(self):
        raise RuntimeError("cannot serialize this item")


class _ContentlessItem:
    """Output item that carries no ``content`` attribute."""

    def __init__(self, item_type="reasoning"):
        self.type = item_type
        self.id = "rs_1"


class _SdkMessage:
    """Stand-in for an OpenAI SDK assistant message output item."""

    def __init__(self, text):
        self.type = "message"
        self.id = "msg_1"
        self.content = [SimpleNamespace(type="output_text", text=text)]


class _FakeResponse:
    def __init__(self, output=None, output_text=""):
        self.output = output or []
        self.output_text = output_text


class _FakeResponses:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        # The agent reuses one mutable list for the request input, and the
        # real SDK serializes it at request time. Snapshot the list so each
        # recorded call reflects the input as it was when it was made.
        recorded = dict(kwargs)
        if "input" in recorded:
            recorded["input"] = list(recorded["input"])
        self.calls.append(recorded)
        if not self._responses:
            raise AssertionError(
                "responses.create was called more times than the test expected"
            )
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.responses = _FakeResponses(responses)


def _settings():
    return Settings(
        _env_file=None,
        groq_api_key="test_key",
        groq_model="openai/gpt-oss-20b",
        groq_base_url="https://api.groq.com/openai/v1",
    )


def _output_items(request_input, item_type):
    return [item for item in request_input if item.get("type") == item_type]


def _outputs_by_call_id(request_input):
    outputs = {}
    for item in _output_items(request_input, "function_call_output"):
        outputs[item["call_id"]] = item
    return outputs


# ---------------------------------------------------------------------------
# 1. Multiple tool calls in one response all execute and return correctly.
# ---------------------------------------------------------------------------


def test_multiple_tool_calls_all_results_returned_to_model():
    """Two tool calls in one response both execute and both results return."""
    client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_1",
                        "search_support_docs",
                        '{"query": "password reset"}',
                    ),
                    _SdkFunctionCall(
                        "call_2",
                        "lookup_ticket_history",
                        '{"query": "billing refund"}',
                    ),
                ]
            ),
            _FakeResponse(
                output=[_SdkMessage("Here is what I found.")],
                output_text="Here is what I found.",
            ),
        ]
    )

    result = run_agent("Help me with my account.", _settings(), client)

    assert result.status == "success"
    assert [call["call_id"] for call in result.tool_calls] == ["call_1", "call_2"]
    assert [call["tool"] for call in result.tool_calls] == [
        "search_support_docs",
        "lookup_ticket_history",
    ]

    # Both tools executed; restricted tool is rejected as unknown/unavailable.
    docs_result = result.tool_calls[0]["result"]
    history_result = result.tool_calls[1]["result"]
    assert "error" not in docs_result
    assert any(doc["id"] == "doc-001" for doc in docs_result["results"])
    assert history_result.get("error") == "unknown_tool"
    assert history_result.get("results") == []

    # Both results were returned to the model, keyed by their own call_id.
    assert len(client.responses.calls) == 2
    follow_up_input = client.responses.calls[1]["input"]
    outputs = _output_items(follow_up_input, "function_call_output")
    assert len(outputs) == 2

    by_call_id = _outputs_by_call_id(follow_up_input)
    assert set(by_call_id) == {"call_1", "call_2"}

    docs_payload = json.loads(by_call_id["call_1"]["output"])
    assert docs_payload["tool"] == "search_support_docs"
    assert docs_payload["count"] >= 1
    assert by_call_id["call_1"]["name"] == "search_support_docs"

    history_payload = json.loads(by_call_id["call_2"]["output"])
    assert history_payload["tool"] == "lookup_ticket_history"
    assert history_payload.get("error") == "unknown_tool"
    assert by_call_id["call_2"]["name"] == "lookup_ticket_history"


def test_multiple_tool_call_results_keep_model_request_order():
    """The returned outputs stay in the order the model requested the calls."""
    client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_b", "lookup_ticket_history", '{"query": "login fails"}'
                    ),
                    _SdkFunctionCall(
                        "call_a", "search_support_docs", '{"query": "login"}'
                    ),
                ]
            ),
            _FakeResponse(
                output=[_SdkMessage("Done.")],
                output_text="Done.",
            ),
        ]
    )

    result = run_agent("Login problem.", _settings(), client)

    assert result.status == "success"
    follow_up_input = client.responses.calls[1]["input"]
    outputs = _output_items(follow_up_input, "function_call_output")
    assert [item["call_id"] for item in outputs] == ["call_b", "call_a"]
    assert [call["call_id"] for call in result.tool_calls] == ["call_b", "call_a"]


def test_tool_outputs_correlate_one_to_one_with_tool_calls():
    """Each output maps to exactly one tool call, with a matching tool name."""
    client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_x", "search_support_docs", '{"query": "password reset"}'
                    ),
                    _SdkFunctionCall(
                        "call_y", "lookup_ticket_history", '{"query": "password reset"}'
                    ),
                ]
            ),
            _FakeResponse(
                output=[_SdkMessage("Summary.")],
                output_text="Summary.",
            ),
        ]
    )

    run_agent("Password question.", _settings(), client)

    follow_up_input = client.responses.calls[1]["input"]
    call_items = _output_items(follow_up_input, "function_call")
    output_items = _output_items(follow_up_input, "function_call_output")

    assert len(call_items) == 2
    assert len(output_items) == 2

    calls_by_id = {item["call_id"]: item for item in call_items}
    outputs_by_id = {item["call_id"]: item for item in output_items}

    assert len(calls_by_id) == 2, "tool call ids must be unique"
    assert len(outputs_by_id) == 2, "tool output ids must be unique"
    assert set(calls_by_id) == set(outputs_by_id)

    for call_id, call_item in calls_by_id.items():
        assert outputs_by_id[call_id]["name"] == call_item["name"]
        assert outputs_by_id[call_id]["call_id"] == call_id


def test_assistant_tool_calls_precede_their_outputs_in_input_order():
    """Each assistant tool call is forwarded before its matching output."""
    client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_1", "search_support_docs", '{"query": "billing"}'
                    ),
                    _SdkFunctionCall(
                        "call_2", "lookup_ticket_history", '{"query": "billing"}'
                    ),
                ]
            ),
            _FakeResponse(
                output=[_SdkMessage("Done.")],
                output_text="Done.",
            ),
        ]
    )

    run_agent("Billing question.", _settings(), client)

    follow_up_input = client.responses.calls[1]["input"]
    for call_id in ("call_1", "call_2"):
        call_index = next(
            index
            for index, item in enumerate(follow_up_input)
            if item.get("type") == "function_call" and item.get("call_id") == call_id
        )
        output_index = next(
            index
            for index, item in enumerate(follow_up_input)
            if item.get("type") == "function_call_output"
            and item.get("call_id") == call_id
        )
        assert call_index < output_index, (
            f"tool call {call_id} must precede its output in the request input"
        )


def test_assistant_tool_call_is_forwarded_without_model_dump():
    """A tool-call item lacking ``model_dump`` is still forwarded correctly."""
    client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _BareFunctionCall(
                        "call_bare", "search_support_docs", '{"query": "billing"}'
                    )
                ]
            ),
            _FakeResponse(
                output=[_SdkMessage("Done.")],
                output_text="Done.",
            ),
        ]
    )

    result = run_agent("Billing question.", _settings(), client)

    assert result.status == "success"
    follow_up_input = client.responses.calls[1]["input"]
    call_items = _output_items(follow_up_input, "function_call")
    assert len(call_items) == 1
    assert call_items[0]["call_id"] == "call_bare"
    assert call_items[0]["name"] == "search_support_docs"
    assert call_items[0]["arguments"] == '{"query": "billing"}'


def test_assistant_tool_call_is_forwarded_when_model_dump_fails():
    """A failing ``model_dump`` falls back to explicit field forwarding."""
    client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _BrokenDumpFunctionCall(
                        "call_broken", "search_support_docs", '{"query": "billing"}'
                    )
                ]
            ),
            _FakeResponse(
                output=[_SdkMessage("Done.")],
                output_text="Done.",
            ),
        ]
    )

    result = run_agent("Billing question.", _settings(), client)

    assert result.status == "success"
    follow_up_input = client.responses.calls[1]["input"]
    call_items = _output_items(follow_up_input, "function_call")
    assert len(call_items) == 1
    assert call_items[0]["call_id"] == "call_broken"
    assert call_items[0]["name"] == "search_support_docs"
    assert call_items[0]["arguments"] == '{"query": "billing"}'

    outputs = _outputs_by_call_id(follow_up_input)
    assert "call_broken" in outputs


def test_output_items_without_content_are_skipped_when_extracting_text():
    """A contentless output item does not break final-answer extraction."""
    client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _ContentlessItem(),
                    _SdkMessage("The answer is 42."),
                ]
            )
        ]
    )

    result = run_agent("Help me.", _settings(), client)

    assert result.status == "success"
    assert result.answer == "The answer is 42."


def test_non_object_json_arguments_are_treated_as_empty_arguments():
    """Valid JSON that is not an object yields empty arguments, not a crash."""
    client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_1", "search_support_docs", '["password", "reset"]'
                    )
                ]
            ),
            _FakeResponse(
                output=[_SdkMessage("I could not read the request.")],
                output_text="I could not read the request.",
            ),
        ]
    )

    result = run_agent("Password help.", _settings(), client)

    assert result.status == "success"
    assert result.tool_calls[0]["arguments"] == {}
    assert result.tool_calls[0]["result"]["error"] == "invalid_arguments"


# ---------------------------------------------------------------------------
# 2. Repeated tool-call rounds complete with a final answer.
# ---------------------------------------------------------------------------


def test_repeated_rounds_feed_each_result_back_to_model():
    """Every round's tool result is fed back on the following request."""
    client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_1", "search_support_docs", '{"query": "billing refund"}'
                    )
                ]
            ),
            _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_2", "lookup_ticket_history", '{"query": "billing refund"}'
                    )
                ]
            ),
            _FakeResponse(
                output=[_SdkMessage("Refunds take 5 business days.")],
                output_text="Refunds take 5 business days.",
            ),
        ]
    )

    result = run_agent("I need a refund.", _settings(), client)

    assert result.status == "success"
    assert result.answer == "Refunds take 5 business days."
    assert len(result.tool_calls) == 2
    assert len(client.responses.calls) == 3

    second_request = client.responses.calls[1]["input"]
    third_request = client.responses.calls[2]["input"]

    # Round 1's result is available for round 2, and both results for round 3.
    assert set(_outputs_by_call_id(second_request)) == {"call_1"}
    assert set(_outputs_by_call_id(third_request)) == {"call_1", "call_2"}

    # The original user message survives in every request.
    for request_input in (second_request, third_request):
        first_item = request_input[0]
        assert first_item["type"] == "message"
        assert first_item["role"] == "user"
        assert first_item["content"][0]["text"] == "I need a refund."

    # Each round's output carries its own payload.
    third_outputs = _outputs_by_call_id(third_request)
    assert json.loads(third_outputs["call_1"]["output"])["tool"] == "search_support_docs"
    assert (
        json.loads(third_outputs["call_2"]["output"])["tool"] == "lookup_ticket_history"
    )


def test_final_answer_is_read_from_message_content_when_output_text_empty():
    """A final answer is extracted from the message content as a fallback."""
    client = _FakeClient(
        [_FakeResponse(output=[_SdkMessage("Answer from message content.")])]
    )

    result = run_agent("Help me.", _settings(), client)

    assert result.status == "success"
    assert result.answer == "Answer from message content."


# ---------------------------------------------------------------------------
# 3. Iteration limits.
# ---------------------------------------------------------------------------


def test_custom_max_iterations_is_enforced():
    """A custom ``max_iterations`` caps the number of provider requests."""
    client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_loop", "search_support_docs", '{"query": "test"}'
                    )
                ]
            )
            for _ in range(5)
        ]
    )

    result = run_agent("Keep searching.", _settings(), client, max_iterations=2)

    assert result.status == "max_iterations_reached"
    assert result.answer is None
    assert result.error is not None
    assert "maximum of 2" in result.error
    assert len(client.responses.calls) == 2
    assert len(result.tool_calls) == 2


def test_max_iterations_of_one_stops_after_a_single_round():
    """A limit of one performs a single request and then reports the limit."""
    client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_1", "search_support_docs", '{"query": "test"}'
                    )
                ]
            )
        ]
    )

    result = run_agent("Keep searching.", _settings(), client, max_iterations=1)

    assert result.status == "max_iterations_reached"
    assert "maximum of 1" in result.error
    assert len(client.responses.calls) == 1
    assert len(result.tool_calls) == 1


def test_default_max_iterations_is_the_documented_limit():
    """The default limit is MAX_ITERATIONS and is never exceeded."""
    client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_loop", "search_support_docs", '{"query": "test"}'
                    )
                ]
            )
            for _ in range(MAX_ITERATIONS + 5)
        ]
    )

    result = run_agent("Keep searching.", _settings(), client)

    assert result.status == "max_iterations_reached"
    assert len(client.responses.calls) == MAX_ITERATIONS
    assert len(result.tool_calls) == MAX_ITERATIONS


def test_incomplete_model_response_reports_limit_status():
    """A response with neither a tool call nor text reports the limit status."""
    client = _FakeClient([_FakeResponse(output=[])])

    result = run_agent("Help me.", _settings(), client)

    assert result.status == "max_iterations_reached"
    assert result.answer is None
    assert result.error is not None
    assert "did not produce" in result.error
    assert len(client.responses.calls) == 1


# ---------------------------------------------------------------------------
# 4. Failure handling.
# ---------------------------------------------------------------------------


def test_execute_tool_reports_tool_failure_when_the_tool_raises(monkeypatch):
    """A raising tool yields a structured tool_failure result."""

    def _explode(args):
        raise RuntimeError("internal detail: /secret/path")

    monkeypatch.setitem(
        TOOL_REGISTRY, "search_support_docs", (SearchSupportDocsArgs, _explode)
    )

    result = execute_tool("search_support_docs", {"query": "password reset"})

    assert result["error"] == "tool_failure"
    assert result["results"] == []
    assert result["tool"] == "search_support_docs"
    assert "internal detail" not in json.dumps(result)
    assert "secret" not in json.dumps(result)


def test_agent_continues_after_a_tool_execution_failure(monkeypatch):
    """A failing tool does not abort the loop; the failure is reported to the model."""

    def _explode(args):
        raise RuntimeError("internal detail: /secret/path")

    monkeypatch.setitem(
        TOOL_REGISTRY, "search_support_docs", (SearchSupportDocsArgs, _explode)
    )

    client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_1", "search_support_docs", '{"query": "password"}'
                    )
                ]
            ),
            _FakeResponse(
                output=[_SdkMessage("I could not reach the documentation.")],
                output_text="I could not reach the documentation.",
            ),
        ]
    )

    result = run_agent("Password help.", _settings(), client)

    assert result.status == "success"
    assert result.answer == "I could not reach the documentation."
    assert result.tool_calls[0]["result"]["error"] == "tool_failure"

    # The failure payload was handed back to the model rather than swallowed.
    follow_up_input = client.responses.calls[1]["input"]
    output_item = _outputs_by_call_id(follow_up_input)["call_1"]
    payload = json.loads(output_item["output"])
    assert payload["error"] == "tool_failure"
    assert "internal detail" not in output_item["output"]


def test_provider_failure_after_a_tool_round_is_reported_safely():
    """A provider error on a later round is mapped to provider_failure."""
    client = _FakeClient([])

    def _create(**kwargs):
        client.responses.calls.append(kwargs)
        if len(client.responses.calls) == 1:
            return _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_1", "search_support_docs", '{"query": "password"}'
                    )
                ]
            )
        raise RuntimeError("connection reset api_key=supersecret")

    client.responses.create = _create

    result = run_agent("Password help.", _settings(), client)

    assert result.status == "provider_failure"
    assert "temporarily unavailable" in result.error
    assert "supersecret" not in result.error
    assert len(result.tool_calls) == 1
    assert len(client.responses.calls) == 2


def test_unregistered_tool_is_rejected_and_registry_is_unchanged():
    """The model cannot invoke anything outside the registered tool set."""
    client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_1", "delete_customer_data", '{"query": "all records"}'
                    )
                ]
            ),
            _FakeResponse(
                output=[_SdkMessage("I cannot do that.")],
                output_text="I cannot do that.",
            ),
        ]
    )

    registry_before = dict(TOOL_REGISTRY)
    result = run_agent("Delete my data.", _settings(), client)

    assert result.status == "success"
    assert result.tool_calls[0]["result"]["error"] == "unknown_tool"
    assert result.tool_calls[0]["result"]["results"] == []
    assert dict(TOOL_REGISTRY) == registry_before


# ---------------------------------------------------------------------------
# 5. Permission boundary and advertised tool definitions.
# ---------------------------------------------------------------------------


def test_agent_advertises_only_registered_tools():
    """The tools sent to the model match the registry exactly."""
    client = _FakeClient(
        [_FakeResponse(output=[_SdkMessage("Done.")], output_text="Done.")]
    )

    run_agent("Hello.", _settings(), client)

    tools = client.responses.calls[0]["tools"]
    advertised = [tool["name"] for tool in tools]

    assert advertised == list(TOOL_REGISTRY)
    assert set(advertised) == {"search_support_docs"}
    for tool in tools:
        assert tool["type"] == "function"
        assert tool["name"]
        assert tool["description"]
        assert tool["parameters"]["type"] == "object"
        assert "query" in tool["parameters"]["properties"]
        assert tool["parameters"]["additionalProperties"] is False


def test_tool_definitions_are_stable_across_rounds():
    """The same tool definitions are sent on every request of a run."""
    client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_1", "search_support_docs", '{"query": "billing"}'
                    )
                ]
            ),
            _FakeResponse(
                output=[_SdkMessage("Done.")],
                output_text="Done.",
            ),
        ]
    )

    run_agent("Billing help.", _settings(), client)

    first_tools = client.responses.calls[0]["tools"]
    second_tools = client.responses.calls[1]["tools"]
    assert first_tools == second_tools


# ---------------------------------------------------------------------------
# 6. Mock mode and existing behaviour remain intact.
# ---------------------------------------------------------------------------


def test_mock_mode_makes_no_provider_request():
    """Mock mode short-circuits before any provider call is made."""
    settings = Settings(_env_file=None)

    result = run_agent("Help me.", settings, None)

    assert result.status == "mock"
    assert result.tool_calls == []
    assert result.error is None
    assert result.answer is not None


def test_agent_result_equality_and_representation():
    """AgentResult keeps its comparison and debugging behaviour."""
    first = AgentResult(status="success", answer="ok", tool_calls=[])
    second = AgentResult(status="success", answer="ok", tool_calls=[])
    different = AgentResult(status="mock", answer="ok", tool_calls=[])

    assert first == second
    assert first != different
    assert "AgentResult" in repr(first)


def test_agent_result_comparison_with_an_unrelated_type_is_not_equal():
    """Comparing against a non-AgentResult yields inequality, not an error."""
    result = AgentResult(status="success", answer="ok", tool_calls=[])

    assert (result == "not an AgentResult") is False
    assert (result != "not an AgentResult") is True


def test_tool_schemas_reject_a_non_string_query():
    """A non-string query is rejected by the argument schemas."""
    from app.agent_core.tool_schemas import LookupTicketHistoryArgs

    with pytest.raises(ValidationError):
        SearchSupportDocsArgs(query=123)

    with pytest.raises(ValidationError):
        LookupTicketHistoryArgs(query=123)
