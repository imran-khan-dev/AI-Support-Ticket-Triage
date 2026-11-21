"""Deterministic, read-only support tools for the tool-using agent layer.

This module implements small support tools.

The tools are deterministic, require no external services, and perform no writes
by default. An explicit registry maps tool names to their argument schemas and
callables, and a central execute_tool function validates and dispatches calls safely.
"""

import inspect
from typing import Any

from app.agent_core.tool_schemas import (
    CreateSupportCaseArgs,
    LookupTicketHistoryArgs,
    SearchKnowledgeArgs,
    SearchSupportDocsArgs,
)
from app.integrations.external_api import FakeSupportCaseAdapter, SupportCaseAdapter


# Server-controlled default adapter (trusted configuration only)
_DEFAULT_SUPPORT_CASE_ADAPTER: SupportCaseAdapter = FakeSupportCaseAdapter()

# Knowledge service factory - can be overridden for testing or custom configuration
_knowledge_service_factory = None


def set_knowledge_service_factory(factory):
    """Set a custom factory function for creating KnowledgeService instances.
    
    The factory should accept a Database and return a KnowledgeService.
    Pass None to reset to default behavior.
    """
    global _knowledge_service_factory
    _knowledge_service_factory = factory


def get_knowledge_service(db):
    """Get a KnowledgeService instance, using custom factory if set."""
    if _knowledge_service_factory is not None:
        return _knowledge_service_factory(db)
    from app.knowledge.service import KnowledgeService
    return KnowledgeService(db)


def get_default_support_case_adapter() -> SupportCaseAdapter:
    """Return the default server-controlled adapter."""
    return _DEFAULT_SUPPORT_CASE_ADAPTER


def get_support_case_adapter() -> SupportCaseAdapter:
    """Return the default server-controlled adapter used for create_support_case execution."""
    return _DEFAULT_SUPPORT_CASE_ADAPTER


def set_support_case_adapter(adapter: SupportCaseAdapter) -> None:
    """Deprecated: injection is now explicit via execute_approved_action(adapter=...)."""
    raise RuntimeError("Adapter injection via set_support_case_adapter is disabled; pass adapter explicitly.")

# ---------------------------------------------------------------------------
# Fixture data: fictional support documentation
# These documents are intentionally global and non-sensitive support knowledge.
# They must not contain tenant-private data. No tenant-specific filtering is applied.
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
    """Legacy support docs search (kept for approval test compatibility).

    This tool is NOT in the active TOOL_REGISTRY. The LLM only sees
    search_knowledge for support document retrieval.
    """
    return []


def search_knowledge(
    args: SearchKnowledgeArgs,
    *,
    tenant_id: str | None = None,
    db: Any = None,
) -> list[dict[str, Any]]:
    """Search the tenant's knowledge base using BM25 or dense retrieval.

    Returns a list of relevant chunks with document name, content, and score.
    Returns an empty list when no documents are found.

    Raises an exception on retrieval infrastructure or database failure,
    which is caught by execute_tool and returned as a tool_failure error.
    """
    if tenant_id is None or db is None:
        raise RuntimeError("Missing tenant context for knowledge search.")

    service = get_knowledge_service(db)
    if args.mode == "dense":
        results = service.search_dense(tenant_id, args.query, args.top_k, rerank=args.rerank)
    elif args.mode == "hybrid":
        results = service.search_hybrid(tenant_id, args.query, args.top_k, rerank=args.rerank)
    else:
        results = service.search(tenant_id, args.query, args.top_k, rerank=args.rerank)

    return [
        {
            "chunk_id": r.chunk.id,
            "document_name": r.chunk.metadata.get("filename", "unknown"),
            "content": r.chunk.content,
            "relevance_score": r.score,
            "score_type": r.score_type,
            "chunk_index": r.chunk.chunk_index,
        }
        for r in results
    ]


def create_support_case(args: CreateSupportCaseArgs) -> list[dict[str, Any]]:
    result = get_support_case_adapter().create_support_case(args)
    return [result]


def lookup_ticket_history(args: LookupTicketHistoryArgs) -> list[dict[str, Any]]:
    return []


# ---------------------------------------------------------------------------
# Tool registry and central execution
# ---------------------------------------------------------------------------

GATED_TOOLS: frozenset[str] = frozenset({"create_support_case"})

TOOL_REGISTRY: dict[str, tuple[type[Any], Any]] = {
    "search_support_docs": (SearchSupportDocsArgs, search_support_docs),
    "search_knowledge": (SearchKnowledgeArgs, search_knowledge),
    "create_support_case": (CreateSupportCaseArgs, create_support_case),
}


def execute_tool(
    tool_name: str,
    raw_arguments: dict[str, Any],
    *,
    tenant_id: str | None = None,
    db: Any = None,
) -> dict[str, Any]:
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

    try:
        validated_args = args_schema(**raw_arguments)
    except Exception:
        return {
            "tool": tool_name,
            "error": "invalid_arguments",
            "message": f"Invalid arguments for tool '{tool_name}'.",
            "results": [],
        }

    try:
        sig = inspect.signature(func)
        kwargs = {}
        if "tenant_id" in sig.parameters:
            kwargs["tenant_id"] = tenant_id
        if "db" in sig.parameters:
            kwargs["db"] = db
        results = func(validated_args, **kwargs)
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