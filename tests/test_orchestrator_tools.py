"""Focused tests for application tool registration with the Agent Orchestrator.

These tests use mocks for external services and database dependencies and never
make real API calls or modify production code.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from app.agent_core import orchestrator_tools
from app.agent_core.tool_schemas import (
    CreateSupportCaseArgs,
    SearchKnowledgeArgs,
)
from app.agent_core.tools import TOOL_REGISTRY

from agent_orchestrator.tools.executor import ToolExecutor
from agent_orchestrator.tools.models import ToolCall
from agent_orchestrator.tools.registry import ToolRegistry


def _make_tool_call(name, arguments):
    return ToolCall(id="call_1", name=name, arguments=arguments)


# ---------------------------------------------------------------------------
# Registry construction
# ---------------------------------------------------------------------------


def test_all_active_tools_are_registered():
    registry = orchestrator_tools.build_tool_registry(
        tenant_id="tenant-1", db=object()
    )
    names = {d.name for d in registry.list_tools()}
    assert names == set(orchestrator_tools.ACTIVE_TOOL_NAMES)
    assert "search_knowledge" in names
    assert "create_support_case" in names


def test_registry_definitions_match_application_schemas():
    registry = orchestrator_tools.build_tool_registry(
        tenant_id="tenant-1", db=object()
    )
    for tool_name in orchestrator_tools.ACTIVE_TOOL_NAMES:
        args_schema, _ = TOOL_REGISTRY[tool_name]
        definition = registry.get_definition(tool_name)
        assert definition.name == tool_name
        assert isinstance(definition.parameters_schema, dict)
        assert definition.parameters_schema.get("additionalProperties") is False
        # Schema must require the same fields as the application schema.
        assert set(definition.parameters_schema.get("required", [])) == set(
            args_schema.model_fields.keys()
        ) - {k for k, v in args_schema.model_fields.items() if not v.is_required()}


def test_build_tool_executor_returns_tool_executor():
    registry = orchestrator_tools.build_tool_registry(
        tenant_id="tenant-1", db=object()
    )
    executor = orchestrator_tools.build_tool_executor(registry)
    assert isinstance(executor, ToolExecutor)


# ---------------------------------------------------------------------------
# Handler dispatch
# ---------------------------------------------------------------------------


def test_handler_dispatch_delegates_to_execute_tool():
    captured = {}

    def fake_execute_tool(name, raw_arguments, *, tenant_id=None, db=None):
        captured.update(
            name=name, arguments=raw_arguments, tenant_id=tenant_id, db=db
        )
        return {"tool": name, "results": [], "count": 0}

    with patch.object(orchestrator_tools, "execute_tool", side_effect=fake_execute_tool):
        registry = orchestrator_tools.build_tool_registry(
            tenant_id="trusted-tenant", db="trusted-db"
        )
        executor = orchestrator_tools.build_tool_executor(registry)
        result = executor.execute(
            _make_tool_call("search_knowledge", {"query": "password reset"})
        )

    assert result.success is True
    assert result.output["tool"] == "search_knowledge"
    assert captured["name"] == "search_knowledge"
    assert captured["arguments"] == {"query": "password reset"}
    assert captured["tenant_id"] == "trusted-tenant"
    assert captured["db"] == "trusted-db"


def test_tenant_context_injected_from_trusted_dependency():
    captured = {}

    def fake_execute_tool(name, raw_arguments, *, tenant_id=None, db=None):
        captured["tenant_id"] = tenant_id
        return {"tool": name, "results": [], "count": 0}

    with patch.object(orchestrator_tools, "execute_tool", side_effect=fake_execute_tool):
        registry = orchestrator_tools.build_tool_registry(
            tenant_id="trusted-tenant", db=object()
        )
        executor = orchestrator_tools.build_tool_executor(registry)
        executor.execute(
            _make_tool_call("search_knowledge", {"query": "x"})
        )

    assert captured["tenant_id"] == "trusted-tenant"


# ---------------------------------------------------------------------------
# Tenant cannot be overridden by LLM arguments
# ---------------------------------------------------------------------------


def test_llm_supplied_tenant_id_is_rejected_by_schema():
    # The tool schema has additionalProperties=false, so the model cannot
    # supply a tenant_id; the validator rejects it before the handler runs.
    registry = orchestrator_tools.build_tool_registry(
        tenant_id="trusted-tenant", db=object()
    )
    executor = orchestrator_tools.build_tool_executor(registry)
    result = executor.execute(
        _make_tool_call(
            "search_knowledge",
            {"query": "x", "tenant_id": "llm-tenant"},
        )
    )

    assert result.success is False
    assert result.output is None
    assert result.error is not None


def test_handler_strips_tenant_id_defense_in_depth():
    # Even when validation is bypassed, the handler never forwards a
    # model-supplied tenant_id to execute_tool.
    captured = {}

    def fake_execute_tool(name, raw_arguments, *, tenant_id=None, db=None):
        captured["tenant_id"] = tenant_id
        captured["arguments"] = raw_arguments
        return {"tool": name, "results": [], "count": 0}

    with patch.object(orchestrator_tools, "execute_tool", side_effect=fake_execute_tool):
        registry = orchestrator_tools.build_tool_registry(
            tenant_id="trusted-tenant", db=object()
        )
        # Call the registered handler directly, bypassing the validator.
        registered = registry.get("search_knowledge")
        registered.handler(query="x", tenant_id="llm-tenant")

    assert captured["tenant_id"] == "trusted-tenant"
    assert "tenant_id" not in captured["arguments"]


# ---------------------------------------------------------------------------
# Approval-gated support case behavior
# ---------------------------------------------------------------------------


def test_create_support_case_unapproved_returns_refusal_without_execution():
    executed = []

    def fake_execute_tool(name, raw_arguments, *, tenant_id=None, db=None):
        executed.append(name)
        return {"tool": name, "results": [], "count": 0}

    with patch.object(orchestrator_tools, "execute_tool", side_effect=fake_execute_tool):
        registry = orchestrator_tools.build_tool_registry(
            tenant_id="t", db=object(), is_approved=False
        )
        executor = orchestrator_tools.build_tool_executor(registry)
        result = executor.execute(
            _make_tool_call(
                "create_support_case",
                {
                    "subject": "s",
                    "description": "d",
                    "priority": "low",
                    "category": "billing",
                },
            )
        )

    assert executed == []  # execute_tool was never called
    assert result.success is True
    assert result.output["error"] == "approval_required"
    assert result.output["results"] == []


def test_create_support_case_approved_delegates_to_execute_tool():
    captured = {}

    def fake_execute_tool(name, raw_arguments, *, tenant_id=None, db=None):
        captured["name"] = name
        return {"tool": name, "results": [], "count": 0}

    with patch.object(orchestrator_tools, "execute_tool", side_effect=fake_execute_tool):
        registry = orchestrator_tools.build_tool_registry(
            tenant_id="t", db=object(), is_approved=True
        )
        executor = orchestrator_tools.build_tool_executor(registry)
        executor.execute(
            _make_tool_call(
                "create_support_case",
                {
                    "subject": "s",
                    "description": "d",
                    "priority": "low",
                    "category": "billing",
                },
            )
        )

    assert captured["name"] == "create_support_case"


# ---------------------------------------------------------------------------
# Knowledge search continues using KnowledgeService
# ---------------------------------------------------------------------------


def test_knowledge_search_uses_knowledge_service():
    fake_chunk = SimpleNamespace(
        id="chunk-1",
        content="To reset your password, visit account settings.",
        metadata={"filename": "password_reset.txt"},
        chunk_index=0,
    )
    fake_result = SimpleNamespace(chunk=fake_chunk, score=0.9, score_type="bm25")

    fake_service = Mock()
    fake_service.search.return_value = [fake_result]

    with patch(
        "app.agent_core.tools.get_knowledge_service", return_value=fake_service
    ):
        registry = orchestrator_tools.build_tool_registry(
            tenant_id="trusted-tenant", db=object()
        )
        executor = orchestrator_tools.build_tool_executor(registry)
        result = executor.execute(
            _make_tool_call("search_knowledge", {"query": "password reset"})
        )

    assert result.success is True
    outputs = result.output["results"]
    assert len(outputs) == 1
    assert outputs[0]["chunk_id"] == "chunk-1"
    assert (
        outputs[0]["content"] == "To reset your password, visit account settings."
    )
    assert outputs[0]["document_name"] == "password_reset.txt"
    fake_service.search.assert_called_once()
    call_args = fake_service.search.call_args
    assert call_args.args[0] == "trusted-tenant"


# ---------------------------------------------------------------------------
# Tool failures preserve safe result semantics
# ---------------------------------------------------------------------------


def test_unknown_tool_returns_safe_failure():
    registry = orchestrator_tools.build_tool_registry(
        tenant_id="t", db=object()
    )
    executor = orchestrator_tools.build_tool_executor(registry)
    result = executor.execute(_make_tool_call("unknown_tool", {}))

    assert result.success is False
    assert result.output is None
    assert "ToolNotFoundError" in result.error or "unknown_tool" in result.error


def test_invalid_arguments_return_safe_failure():
    registry = orchestrator_tools.build_tool_registry(
        tenant_id="t", db=object()
    )
    executor = orchestrator_tools.build_tool_executor(registry)
    # search_knowledge requires a non-empty "query".
    result = executor.execute(_make_tool_call("search_knowledge", {"query": ""}))

    assert result.success is False
    assert result.output is None
    assert result.error is not None


def test_handler_exception_is_captured_as_failed_tool_result():
    def raising_execute_tool(name, raw_arguments, *, tenant_id=None, db=None):
        raise RuntimeError("knowledge base unavailable")

    with patch.object(
        orchestrator_tools, "execute_tool", side_effect=raising_execute_tool
    ):
        registry = orchestrator_tools.build_tool_registry(
            tenant_id="t", db=object()
        )
        executor = orchestrator_tools.build_tool_executor(registry)
        result = executor.execute(
            _make_tool_call("search_knowledge", {"query": "x"})
        )

    assert result.success is False
    assert result.output is None
    assert "RuntimeError" in result.error or "knowledge base unavailable" in result.error
