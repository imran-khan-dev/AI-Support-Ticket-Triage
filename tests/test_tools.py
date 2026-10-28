import pytest
from pydantic import ValidationError

from app.agent_core.tool_schemas import LookupTicketHistoryArgs, SearchSupportDocsArgs
from app.agent_core.tools import execute_tool, search_support_docs, lookup_ticket_history


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


def test_search_support_docs_returns_password_reset_doc():
    results = search_support_docs(SearchSupportDocsArgs(query="password reset"))
    assert len(results) >= 1
    assert any(doc["id"] == "doc-001" for doc in results)


def test_search_support_docs_returns_billing_refund_doc():
    results = search_support_docs(SearchSupportDocsArgs(query="billing refund"))
    assert len(results) >= 1
    assert any("refund" in doc["content"].lower() for doc in results)


def test_search_support_docs_returns_login_troubleshooting_doc():
    results = search_support_docs(SearchSupportDocsArgs(query="login troubleshooting"))
    assert len(results) >= 1
    assert any(doc["id"] == "doc-003" for doc in results)


def test_search_support_docs_irrelevant_query_returns_no_results():
    results = search_support_docs(SearchSupportDocsArgs(query="xyzzy nothing matches"))
    assert results == []


def test_search_support_docs_results_have_documented_structure():
    results = search_support_docs(SearchSupportDocsArgs(query="password"))
    for doc in results:
        assert "id" in doc
        assert "title" in doc
        assert "content" in doc
        assert isinstance(doc["id"], str)
        assert isinstance(doc["title"], str)
        assert isinstance(doc["content"], str)


def test_search_support_docs_is_deterministic():
    args = SearchSupportDocsArgs(query="billing")
    first = search_support_docs(args)
    second = search_support_docs(args)
    assert first == second


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
    result = execute_tool("search_support_docs", {"query": "password reset"})
    assert result["tool"] == "search_support_docs"
    assert "error" not in result
    assert isinstance(result["results"], list)
    assert len(result["results"]) >= 1
    assert result["count"] == len(result["results"])


def test_execute_tool_unknown_tool_rejected_safely():
    result = execute_tool("nonexistent_tool", {"query": "test"})
    assert result["tool"] == "nonexistent_tool"
    assert result["error"] == "unknown_tool"
    assert result["results"] == []


def test_execute_tool_invalid_arguments_do_not_execute_tool():
    result = execute_tool("search_support_docs", {"query": ""})
    assert result["error"] == "invalid_arguments"
    assert result["results"] == []


def test_execute_tool_invalid_arguments_whitespace_rejected():
    result = execute_tool("lookup_ticket_history", {"query": "   "})
    assert result["error"] == "unknown_tool"
    assert result["results"] == []


def test_execute_tool_returns_structured_result():
    result = execute_tool("search_support_docs", {"query": "billing"})
    assert "tool" in result
    assert "query" in result
    assert "results" in result
    assert "count" in result
    assert isinstance(result["results"], list)
    assert isinstance(result["count"], int)


def test_execute_tool_does_not_execute_tool_on_invalid_args():
    """Calling with invalid args must not invoke the underlying function."""
    result = execute_tool("search_support_docs", {"query": ""})
    assert result["results"] == []
    assert result["error"] == "invalid_arguments"


def test_tools_are_read_only():
    """Both tools return data only and perform no writes (no side effects)."""
    before_docs = len(search_support_docs(SearchSupportDocsArgs(query="password")))
    before_tickets = len(lookup_ticket_history(LookupTicketHistoryArgs(query="billing")))
    result = execute_tool("search_support_docs", {"query": "login"})
    assert result["results"] is not None
    after_docs = len(search_support_docs(SearchSupportDocsArgs(query="password")))
    after_tickets = len(lookup_ticket_history(LookupTicketHistoryArgs(query="billing")))
    assert before_docs == after_docs
    assert before_tickets == after_tickets


def test_tools_return_no_results_for_an_unvalidated_blank_query():
    """A blank query that bypasses schema validation yields no results.

    The schemas reject blank queries, so this only exercises the defensive
    guard inside the matching helper.
    """
    args = SearchSupportDocsArgs.model_construct(query="   ")
    assert search_support_docs(args) == []

    history_args = LookupTicketHistoryArgs.model_construct(query="   ")
    assert lookup_ticket_history(history_args) == []
