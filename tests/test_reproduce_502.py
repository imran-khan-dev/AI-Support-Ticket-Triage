import json
from unittest.mock import Mock, patch
from app.core.config import Settings
from app.agent_core.agent import run_agent


def test_function_call_definitions_have_names_on_all_calls():
    """Regression test: every tool definition passed to responses.create()
    must have a valid non-empty `name` field, on both the first call and any
    follow-up call after a function call.

    Groq rejects requests with "Tools should have a name!" when a tool
    definition is missing its name.
    """
    settings = Settings(
        _env_file=None,
        groq_api_key="test_key",
        groq_model="openai/gpt-oss-20b",
        groq_base_url="https://api.groq.com/openai/v1",
    )

    captured_calls = []

    with patch("app.agent_core.agent.OpenAI") as MockOpenAI:
        mock_client = Mock()

        call_item = Mock()
        call_item.type = "function_call"
        call_item.call_id = "fc_7d1a2b6a-148a-4af2-a639-91dccf9110d2"
        call_item.name = "search_support_docs"
        call_item.arguments = '{"query":"login troubleshooting"}'

        first_response = Mock()
        first_response.output = [call_item]
        first_response.output_text = ""

        second_response = Mock()
        second_response.output = []
        second_response.output_text = "Here is how to reset your password."

        def capture_and_respond(*args, **kwargs):
            captured_calls.append(kwargs)
            if len(captured_calls) == 1:
                return first_response
            return second_response

        mock_client.responses.create.side_effect = capture_and_respond
        MockOpenAI.return_value = mock_client

        result = run_agent(
            "How do I reset my password?", settings, mock_client
        )

        assert len(captured_calls) == 2
        assert result.status == "success"

        valid_tool_names = {"search_support_docs", "lookup_ticket_history"}
        for i, call_kwargs in enumerate(captured_calls):
            tools = call_kwargs.get("tools", [])
            assert len(tools) == 2, f"Call {i+1}: expected 2 tools, got {len(tools)}"
            for j, tool in enumerate(tools):
                assert "name" in tool, (
                    f"Call {i+1}, tool {j} missing 'name' field: {tool}"
                )
                assert tool["name"], (
                    f"Call {i+1}, tool {j} has empty 'name' field"
                )
                assert tool["name"] in valid_tool_names, (
                    f"Call {i+1}, tool {j} has unexpected name: {tool['name']}"
                )
                assert tool["type"] == "function", (
                    f"Call {i+1}, tool {j} has unexpected type: {tool.get('type')}"
                )
                assert "parameters" in tool, (
                    f"Call {i+1}, tool {j} missing 'parameters' field"
                )


def test_function_call_output_has_name_and_json_string_output():
    """Regression test: the ``function_call_output`` input item must include
    a ``name`` field matching the called tool, and ``output`` must be a JSON
    string (not a dict) with the matching ``call_id``.

    Groq's Harmony renderer needs the tool name on the output item to
    associate it with the tool that was called. Without it, Groq returns
    "Tools should have a name!".
    """
    settings = Settings(
        _env_file=None,
        groq_api_key="test_key",
        groq_model="openai/gpt-oss-20b",
        groq_base_url="https://api.groq.com/openai/v1",
    )

    with patch("app.agent_core.agent.OpenAI") as MockOpenAI:
        mock_client = Mock()

        call_item = Mock()
        call_item.type = "function_call"
        call_item.call_id = "fc_7d1a2b6a-148a-4af2-a639-91dccf9110d2"
        call_item.name = "search_support_docs"
        call_item.arguments = '{"query":"login troubleshooting"}'

        first_response = Mock()
        first_response.output = [call_item]
        first_response.output_text = ""

        second_response = Mock()
        second_response.output = []
        second_response.output_text = "Here is how to reset your password."

        mock_client.responses.create.side_effect = [
            first_response,
            second_response,
        ]
        MockOpenAI.return_value = mock_client

        result = run_agent(
            "How do I reset my password?", settings, mock_client
        )

        assert result.status == "success"
        assert result.answer == "Here is how to reset your password."
        assert mock_client.responses.create.call_count == 2

        # Inspect the second call's input_items for the function_call_output.
        second_call_kwargs = mock_client.responses.create.call_args_list[1]
        input_items = second_call_kwargs.kwargs["input"]
        output_items = [
            item
            for item in input_items
            if item.get("type") == "function_call_output"
        ]
        assert len(output_items) == 1
        output_item = output_items[0]

        # The output item must reference the tool by name.
        assert output_item["name"] == "search_support_docs", (
            f"Expected name='search_support_docs', got {output_item.get('name')!r}"
        )
        assert output_item["call_id"] == "fc_7d1a2b6a-148a-4af2-a639-91dccf9110d2"

        # The output must be a JSON string, not a dict.
        assert isinstance(output_item["output"], str), (
            f"Expected output to be str, got {type(output_item['output'])}"
        )
        parsed = json.loads(output_item["output"])
        assert parsed["tool"] == "search_support_docs"
        assert len(parsed["results"]) >= 1


def test_second_call_includes_assistant_tool_call_before_output():
    """Regression test: the second responses.create() input must include the
    assistant's tool-call message before the function_call_output item.

    Groq's Harmony renderer needs to see the assistant's tool call (with its
    tool name) before the tool output, so it can associate the output with the
    tool that was called. Without this preceding message, Groq returns
    "Tools should have a name!" even when the function_call_output itself has
    a name field.
    """
    settings = Settings(
        _env_file=None,
        groq_api_key="test_key",
        groq_model="openai/gpt-oss-20b",
        groq_base_url="https://api.groq.com/openai/v1",
    )

    with patch("app.agent_core.agent.OpenAI") as MockOpenAI:
        mock_client = Mock()

        call_item = Mock()
        call_item.type = "function_call"
        call_item.id = "call_abc"
        call_item.call_id = "fc_7d1a2b6a-148a-4af2-a639-91dccf9110d2"
        call_item.name = "search_support_docs"
        call_item.arguments = '{"query":"login troubleshooting"}'

        first_response = Mock()
        first_response.output = [call_item]
        first_response.output_text = ""

        second_response = Mock()
        second_response.output = []
        second_response.output_text = "Here is how to reset your password."

        mock_client.responses.create.side_effect = [
            first_response,
            second_response,
        ]
        MockOpenAI.return_value = mock_client

        result = run_agent(
            "How do I reset my password?", settings, mock_client
        )

        assert result.status == "success"
        assert mock_client.responses.create.call_count == 2

        second_call_kwargs = mock_client.responses.create.call_args_list[1]
        input_items = second_call_kwargs.kwargs["input"]

        # Find the assistant's tool-call item and the function_call_output.
        tool_call_items = [
            item
            for item in input_items
            if item.get("type") == "function_call"
        ]
        output_items = [
            item
            for item in input_items
            if item.get("type") == "function_call_output"
        ]

        # The assistant's tool call must be present.
        assert len(tool_call_items) == 1, (
            f"Expected 1 assistant tool-call item, got {len(tool_call_items)}"
        )
        tool_call_item = tool_call_items[0]
        assert tool_call_item["call_id"] == "fc_7d1a2b6a-148a-4af2-a639-91dccf9110d2"
        assert tool_call_item["name"] == "search_support_docs"
        assert tool_call_item["arguments"] == '{"query":"login troubleshooting"}'

        # The tool output must follow the tool call.
        assert len(output_items) == 1
        output_item = output_items[0]
        assert output_item["call_id"] == tool_call_item["call_id"]
        assert output_item["name"] == tool_call_item["name"]

        # The tool call must appear before the tool output in the input list.
        tool_call_index = input_items.index(tool_call_item)
        output_index = input_items.index(output_item)
        assert tool_call_index < output_index, (
            "Assistant tool call must precede function_call_output in input"
        )
