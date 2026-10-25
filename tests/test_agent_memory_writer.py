"""Tests for the memory writer of a finished agent run.

The writer is pure: it turns the outcome of a run into memory drafts and
writes nothing. These tests therefore need no database and no provider.
"""

from app.agent_core.agent import AgentResult
from app.domain.agent.memory_writer import (
    MEMORY_WRITE_MAX_CHARS,
    build_memory_drafts,
)
from app.database.models import MEMORY_KIND_FACT, MEMORY_KIND_SUMMARY


#: The id of the stored assistant message a memory points back to.
SOURCE_MSG_ID = 42


def _result(
    answer=None,
    status="success",
    tool_calls=None,
    error=None,
):
    """Return the outcome of a run."""
    return AgentResult(
        status=status,
        answer=answer,
        tool_calls=tool_calls or [],
        error=error,
    )


# ---------------------------------------------------------------------------
# A successful run
# ---------------------------------------------------------------------------


def test_a_successful_run_produces_one_summary_draft():
    """The answer of the agent becomes a memory."""
    result = _result(answer="Refunds are issued within five business days.")

    drafts = build_memory_drafts(result, SOURCE_MSG_ID)

    assert len(drafts) == 1
    assert drafts[0].kind == MEMORY_KIND_SUMMARY
    assert drafts[0].content == "Refunds are issued within five business days."
    assert drafts[0].source_msg_id == SOURCE_MSG_ID


def test_the_memory_is_a_summary_and_not_a_verified_fact():
    """The memory records what the agent said."""
    drafts = build_memory_drafts(
        _result(answer="The customer was refunded on 4 May."),
        SOURCE_MSG_ID,
    )

    assert drafts[0].kind != MEMORY_KIND_FACT
    assert drafts[0].kind == MEMORY_KIND_SUMMARY


def test_the_source_message_is_optional():
    """A memory without a source keeps the source empty."""
    drafts = build_memory_drafts(_result(answer="A memory without a source."))

    assert len(drafts) == 1
    assert drafts[0].source_msg_id is None


# ---------------------------------------------------------------------------
# Runs that produced no answer
# ---------------------------------------------------------------------------


def test_a_mock_run_produces_no_draft():
    """The mock answer is not a fact about the customer."""
    result = _result(
        answer=(
            "Agent mode requires a configured Groq API key. "
            "No tool calls were made."
        ),
        status="mock",
    )

    assert build_memory_drafts(result, SOURCE_MSG_ID) == []


def test_a_provider_failure_produces_no_draft():
    """A failed run holds no answer."""
    result = _result(
        status="provider_failure",
        error="Provider service is temporarily unavailable.",
    )

    assert build_memory_drafts(result, SOURCE_MSG_ID) == []


def test_a_run_without_an_answer_produces_no_draft():
    """A run that ran out of iterations holds no answer."""
    result = _result(
        status="max_iterations_reached",
        error="The model did not produce a final answer or a tool call.",
    )

    assert build_memory_drafts(result, SOURCE_MSG_ID) == []


def test_an_unknown_status_produces_no_draft():
    """Only a successful run produces a memory."""
    result = _result(answer="An answer of a run that failed.", status="other")

    assert build_memory_drafts(result, SOURCE_MSG_ID) == []


# ---------------------------------------------------------------------------
# Answers that are not worth remembering
# ---------------------------------------------------------------------------


def test_an_empty_answer_produces_no_draft():
    """There is nothing to store."""
    assert build_memory_drafts(_result(answer=""), SOURCE_MSG_ID) == []
    assert build_memory_drafts(_result(answer=None), SOURCE_MSG_ID) == []


def test_a_whitespace_only_answer_produces_no_draft():
    """Whitespace is not content."""
    assert build_memory_drafts(_result(answer="   \n\t "), SOURCE_MSG_ID) == []


def test_an_answer_below_the_minimum_length_produces_no_draft():
    """A fragment that is too short to carry meaning is dropped."""
    assert build_memory_drafts(_result(answer="a" * 23), SOURCE_MSG_ID) == []


def test_an_answer_at_the_minimum_length_produces_a_draft():
    """The minimum length is inclusive."""
    drafts = build_memory_drafts(_result(answer="a" * 24), SOURCE_MSG_ID)

    assert len(drafts) == 1
    assert drafts[0].content == "a" * 24


# ---------------------------------------------------------------------------
# What a memory never holds
# ---------------------------------------------------------------------------


def test_a_tool_result_is_never_part_of_a_memory():
    """A tool result is documentation, not something learned."""
    result = _result(
        answer="Refunds are issued within five business days.",
        tool_calls=[
            {
                "call_id": "call_1",
                "tool": "search_support_docs",
                "arguments": {"query": "refund"},
                "result": {
                    "tool": "search_support_docs",
                    "count": 1,
                    "results": [
                        {
                            "id": "doc-002",
                            "title": "Billing refunds and charges",
                            "content": "Refund requests are reviewed.",
                        }
                    ],
                },
            }
        ],
    )

    drafts = build_memory_drafts(result, SOURCE_MSG_ID)

    assert len(drafts) == 1
    assert "doc-002" not in drafts[0].content
    assert "Refund requests are reviewed" not in drafts[0].content
    assert drafts[0].content == "Refunds are issued within five business days."


def test_an_error_message_is_never_part_of_a_memory():
    """The error of a run is not remembered."""
    result = _result(
        answer="Refunds are issued within five business days.",
        error="Provider service is temporarily unavailable.",
    )

    drafts = build_memory_drafts(result, SOURCE_MSG_ID)

    assert len(drafts) == 1
    assert "Provider service" not in drafts[0].content


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_control_characters_and_extra_whitespace_are_removed():
    """The answer is written the way the recall side reads it."""
    result = _result(
        answer="Refunds   take\x00 five\n\ndays,  and  the\tlink  expires."
    )

    drafts = build_memory_drafts(result, SOURCE_MSG_ID)

    assert drafts[0].content == (
        "Refunds take five days, and the link expires."
    )


def test_a_long_answer_is_capped_at_the_limit():
    """A memory stays a short note."""
    result = _result(answer="word " * 600)

    drafts = build_memory_drafts(result, SOURCE_MSG_ID)

    assert len(drafts[0].content) <= MEMORY_WRITE_MAX_CHARS


def test_a_long_answer_is_cut_at_a_word_boundary():
    """The note does not end in the middle of a word."""
    result = _result(answer="word " * 600)

    drafts = build_memory_drafts(result, SOURCE_MSG_ID)

    content = drafts[0].content
    assert content == " ".join(["word"] * 80)
    assert not content.endswith(" ")
    assert content.split()[-1] == "word"


def test_a_long_answer_without_a_word_boundary_is_cut_at_the_limit():
    """A single long word is cut at the limit."""
    result = _result(answer="x" * 500)

    drafts = build_memory_drafts(result, SOURCE_MSG_ID)

    assert drafts[0].content == "x" * MEMORY_WRITE_MAX_CHARS
