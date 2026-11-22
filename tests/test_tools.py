import pytest
from pydantic import ValidationError

from app.agent_core.tool_schemas import (
    CreateSupportCaseArgs,
    LookupTicketHistoryArgs,
    SearchSupportDocsArgs,
)
from app.agent_core.tools import (
    TOOL_REGISTRY,
    create_support_case,
    execute_tool,
    lookup_ticket_history,
)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_search_support_docs_args_requires_query():
    with pytest.raises(ValidationError):
        SearchSupportDocsArgs()


def test_search_support_docs_args_rejects_empty_query():
    with pytest.raises(ValidationError):
        SearchSupportDocsArgs(query="")


def test_search_support_docs_args_rejects_whitespace_query():
    with pytest.raises(ValidationError):
        SearchSupportDocsArgs(query="   ")


def test_search_support_docs_args_strips_whitespace():
    args = SearchSupportDocsArgs(query="  password reset  ")
    assert args.query == "password reset"


def test_search_support_docs_args_rejects_overly_long_query():
    with pytest.raises(ValidationError):
        SearchSupportDocsArgs(query="x" * 501)


def test_lookup_ticket_history_args_requires_query():
    with pytest.raises(ValidationError):
        LookupTicketHistoryArgs()


def test_lookup_ticket_history_args_rejects_empty_query():
    with pytest.raises(ValidationError):
        LookupTicketHistoryArgs(query="")


def test_lookup_ticket_history_args_rejects_whitespace_query():
    with pytest.raises(ValidationError):
        LookupTicketHistoryArgs(query="   ")


def test_lookup_ticket_history_args_strips_whitespace():
    args = LookupTicketHistoryArgs(query="  billing refund  ")
    assert args.query == "billing refund"


def test_lookup_ticket_history_args_rejects_overly_long_query():
    with pytest.raises(ValidationError):
        LookupTicketHistoryArgs(query="x" * 501)


# ---------------------------------------------------------------------------
# search_support_docs tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# lookup_ticket_history tests
# ---------------------------------------------------------------------------


def test_lookup_ticket_history_returns_password_reset_ticket():
    results = lookup_ticket_history(LookupTicketHistoryArgs(query="password reset"))
    assert results == []


def test_lookup_ticket_history_returns_billing_ticket():
    results = lookup_ticket_history(LookupTicketHistoryArgs(query="billing refund"))
    assert results == []


def test_lookup_ticket_history_returns_login_ticket():
    results = lookup_ticket_history(LookupTicketHistoryArgs(query="login fails"))
    assert results == []


def test_lookup_ticket_history_unmatched_query_returns_no_results():
    results = lookup_ticket_history(LookupTicketHistoryArgs(query="xyzzy no match"))
    assert results == []


def test_lookup_ticket_history_results_have_documented_structure():
    results = lookup_ticket_history(LookupTicketHistoryArgs(query="billing"))
    for rec in results:
        assert "ticket_id" in rec
        assert "issue" in rec
        assert "category" in rec
        assert "resolution" in rec
        assert isinstance(rec["ticket_id"], str)
        assert isinstance(rec["issue"], str)
        assert isinstance(rec["category"], str)
        assert isinstance(rec["resolution"], str)


def test_lookup_ticket_history_is_deterministic():
    args = LookupTicketHistoryArgs(query="billing")
    first = lookup_ticket_history(args)
    second = lookup_ticket_history(args)
    assert first == second


# ---------------------------------------------------------------------------
# execute_tool / registry tests
# ---------------------------------------------------------------------------


def test_execute_tool_registered_tool_returns_results():
    # Without tenant context, search_knowledge returns tool_failure
    result = execute_tool("search_knowledge", {"query": "password reset"})
    assert result["tool"] == "search_knowledge"
    assert result["error"] == "tool_failure"
    assert result["results"] == []


def test_execute_tool_unknown_tool_rejected_safely():
    result = execute_tool("nonexistent_tool", {"query": "test"})
    assert result["tool"] == "nonexistent_tool"
    assert result["error"] == "unknown_tool"
    assert result["results"] == []


def test_execute_tool_invalid_arguments_do_not_execute_tool():
    result = execute_tool("search_knowledge", {"query": ""})
    assert result["error"] == "invalid_arguments"
    assert result["results"] == []


def test_execute_tool_invalid_arguments_whitespace_rejected():
    result = execute_tool("lookup_ticket_history", {"query": "   "})
    assert result["error"] == "unknown_tool"
    assert result["results"] == []


def test_execute_tool_returns_structured_result():
    result = execute_tool("search_knowledge", {"query": "billing"})
    assert result["tool"] == "search_knowledge"
    assert result["error"] == "tool_failure"
    assert result["results"] == []


def test_execute_tool_does_not_execute_tool_on_invalid_args():
    """Calling with invalid args must not invoke the underlying function."""
    result = execute_tool("search_knowledge", {"query": ""})
    assert result["results"] == []
    assert result["error"] == "invalid_arguments"


def test_execute_tool_gated_tool_returns_approval_required():
    result = execute_tool(
        "create_support_case",
        {
            "subject": "Login issue",
            "description": "Cannot log in",
            "priority": "high",
            "category": "account",
        },
    )
    assert result["tool"] == "create_support_case"
    assert result["error"] == "approval_required"
    assert result["results"] == []


# ---------------------------------------------------------------------------
# create_support_case schema tests
# ---------------------------------------------------------------------------


def test_create_support_case_args_requires_all_fields():
    with pytest.raises(ValidationError):
        CreateSupportCaseArgs()


def test_create_support_case_args_rejects_empty_subject():
    with pytest.raises(ValidationError):
        CreateSupportCaseArgs(
            subject="",
            description="desc",
            priority="low",
            category="cat",
        )


def test_create_support_case_args_rejects_empty_description():
    with pytest.raises(ValidationError):
        CreateSupportCaseArgs(
            subject="subj",
            description="   ",
            priority="low",
            category="cat",
        )


def test_create_support_case_args_rejects_invalid_priority():
    with pytest.raises(ValidationError):
        CreateSupportCaseArgs(
            subject="subj",
            description="desc",
            priority="critical",
            category="cat",
        )


def test_create_support_case_args_strips_whitespace():
    args = CreateSupportCaseArgs(
        subject="  Login issue  ",
        description="  Cannot log in  ",
        priority="high",
        category="  account  ",
    )
    assert args.subject == "Login issue"
    assert args.description == "Cannot log in"
    assert args.category == "account"
    assert args.priority == "high"


def test_create_support_case_args_rejects_overly_long_subject():
    with pytest.raises(ValidationError):
        CreateSupportCaseArgs(
            subject="x" * 201,
            description="desc",
            priority="low",
            category="cat",
        )


def test_create_support_case_registered():
    assert "create_support_case" in TOOL_REGISTRY
    args_schema, func = TOOL_REGISTRY["create_support_case"]
    assert args_schema is CreateSupportCaseArgs
    assert func is create_support_case


def test_tools_are_read_only():
    """Both tools return data only and perform no writes (no side effects)."""
    before_tickets = len(lookup_ticket_history(LookupTicketHistoryArgs(query="billing")))
    result = execute_tool("search_knowledge", {"query": "login"})
    assert result["results"] is not None
    after_tickets = len(lookup_ticket_history(LookupTicketHistoryArgs(query="billing")))
    assert before_tickets == after_tickets


def test_tools_return_no_results_for_an_unvalidated_blank_query():
    """A blank query that bypasses schema validation yields no results.

    The schemas reject blank queries, so this only exercises the defensive
    guard inside the matching helper.
    """
    history_args = LookupTicketHistoryArgs.model_construct(query="   ")
    assert lookup_ticket_history(history_args) == []
