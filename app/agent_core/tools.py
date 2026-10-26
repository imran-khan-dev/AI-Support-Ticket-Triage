"""Deterministic, read-only support tools for the tool-using agent layer.

This module implements two small, hardcoded, read-only tools:

- search_support_docs: searches a fixed collection of fictional support articles.
- lookup_ticket_history: searches a fixed collection of fictional ticket records.

The tools are deterministic, require no external services, and perform no writes.
An explicit registry maps tool names to their argument schemas and callables, and
a central execute_tool function validates and dispatches calls safely.
"""

from typing import Any

from app.agent_core.tool_schemas import LookupTicketHistoryArgs, SearchSupportDocsArgs

# ---------------------------------------------------------------------------
# Fixture data: fictional support documentation
# ---------------------------------------------------------------------------

_SUPPORT_DOCS = [
    {
        "id": "doc-001",
        "title": "How to reset your password",
        "content": (
            "To reset your password, go to the account settings page and select "
            "'Forgot password'. You will receive an email with a secure reset link. "
            "The link expires after 24 hours. If you do not receive the email, "
            "check your spam folder or contact support."
        ),
    },
    {
        "id": "doc-002",
        "title": "Billing refunds and charges",
        "content": (
            "Refund requests are reviewed within 5 business days. Eligible charges "
            "must be within the last 30 days. To request a refund, open a billing "
            "ticket with your order ID and the reason for the request. Approved "
            "refunds are issued to the original payment method."
        ),
    },
    {
        "id": "doc-003",
        "title": "Login troubleshooting",
        "content": (
            "If you cannot log in, verify that your email address and password are "
            "correct and that Caps Lock is off. Try clearing your browser cache or "
            "using an incognito window. If your account is locked after multiple "
            "failed attempts, wait 15 minutes before trying again or use the "
            "account recovery flow."
        ),
    },
    {
        "id": "doc-004",
        "title": "Subscription management",
        "content": (
            "You can upgrade, downgrade, or cancel your subscription from the "
            "billing page at any time. Cancellations take effect at the end of the "
            "current billing period. Prorated credits apply to downgrades and are "
            "applied to your next invoice."
        ),
    },
]

# ---------------------------------------------------------------------------
# Fixture data: fictional ticket history records
# ---------------------------------------------------------------------------

_TICKET_HISTORY = [
    {
        "ticket_id": "TKT-1001",
        "issue": "Customer could not reset password after email change.",
        "category": "account_access",
        "resolution": "Verified identity and sent a manual password reset link.",
    },
    {
        "ticket_id": "TKT-1002",
        "issue": "Double charge on monthly subscription invoice.",
        "category": "billing",
        "resolution": "Issued a prorated refund for the duplicate charge.",
    },
    {
        "ticket_id": "TKT-1003",
        "issue": "Login fails with correct credentials on mobile app.",
        "category": "technical",
        "resolution": "Guided customer through app cache clear and reinstall.",
    },
    {
        "ticket_id": "TKT-1004",
        "issue": "Refund request for annual plan cancelled early.",
        "category": "billing",
        "resolution": "Approved partial refund based on unused months.",
    },
    {
        "ticket_id": "TKT-1005",
        "issue": "Account locked after repeated failed login attempts.",
        "category": "account_access",
        "resolution": "Unlocked account after 15-minute cooldown period.",
    },
]


def _match_query(query: str, text_fields: list[str]) -> bool:
    """Return True if any whole word in the query appears in any text field.

    Simple, deterministic, case-insensitive whole-word matching. No external
    dependencies, no embeddings, no vector search.
    """
    query_lower = query.lower()
    query_words = [word for word in query_lower.split() if word]
    if not query_words:
        return False
    combined_text = " ".join(text_fields).lower()
    text_words = set(combined_text.split())
    return any(word in text_words for word in query_words)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def search_support_docs(args: SearchSupportDocsArgs) -> list[dict[str, Any]]:
    """Search fictional support documentation for documents matching the query.

    Returns a list of documents with id, title, and content. Returns an empty
    list when no documents match.
    """
    query = args.query
    results: list[dict[str, Any]] = []
    for doc in _SUPPORT_DOCS:
        if _match_query(
            query,
            [doc["title"], doc["content"]],
        ):
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "content": doc["content"],
                }
            )
    return results


def lookup_ticket_history(args: LookupTicketHistoryArgs) -> list[dict[str, Any]]:
    """Look up fictional ticket history records matching the query.

    Returns a list of records with ticket_id, issue, category, and resolution.
    Returns an empty list when no records match.
    """
    query = args.query
    results: list[dict[str, Any]] = []
    for record in _TICKET_HISTORY:
        if _match_query(
            query,
            [record["issue"], record["category"], record["resolution"]],
        ):
            results.append(
                {
                    "ticket_id": record["ticket_id"],
                    "issue": record["issue"],
                    "category": record["category"],
                    "resolution": record["resolution"],
                }
            )
    return results


# ---------------------------------------------------------------------------
# Tool registry and central execution
# ---------------------------------------------------------------------------

# Server-controlled set of tools that require approval before execution.
# Trusted configuration only; never derived from model arguments or client claims.
GATED_TOOLS: frozenset[str] = frozenset()

# Explicit registry: tool name -> (argument schema, callable).
# This is the single place where tools are registered. Adding a new tool
# requires adding an entry here plus its implementation function above.
TOOL_REGISTRY: dict[str, tuple[type[Any], Any]] = {
    "search_support_docs": (SearchSupportDocsArgs, search_support_docs),
    "lookup_ticket_history": (LookupTicketHistoryArgs, lookup_ticket_history),
}


def execute_tool(tool_name: str, raw_arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a registered tool by name with validated arguments.

    Steps:
    1. Reject unknown tool names.
    2. Validate arguments with the tool's Pydantic schema.
    3. Execute only the registered function.
    4. Return a predictable structured result.

    Errors are handled safely: invalid arguments and tool failures do not
    expose internal details to the caller.
    """
    if tool_name not in TOOL_REGISTRY:
        return {
            "tool": tool_name,
            "error": "unknown_tool",
            "message": f"Tool '{tool_name}' is not registered.",
            "results": [],
        }

    if tool_name in GATED_TOOLS:
        return {
            "tool": tool_name,
            "error": "approval_required",
            "message": f"Tool '{tool_name}' requires approval and is not executed.",
            "results": [],
        }

    args_schema, func = TOOL_REGISTRY[tool_name]

    # Validate arguments; invalid arguments do not execute the tool.
    try:
        validated_args = args_schema(**raw_arguments)
    except Exception:
        return {
            "tool": tool_name,
            "error": "invalid_arguments",
            "message": f"Invalid arguments for tool '{tool_name}'.",
            "results": [],
        }

    # Execute only the registered function.
    try:
        results = func(validated_args)
    except Exception:
        return {
            "tool": tool_name,
            "error": "tool_failure",
            "message": f"Tool '{tool_name}' failed to execute.",
            "results": [],
        }

    return {
        "tool": tool_name,
        "query": getattr(validated_args, "query", ""),
        "results": results,
        "count": len(results),
    }
