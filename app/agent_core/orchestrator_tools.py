"""Application tool registration for the Agent Orchestrator.

Builds a :class:`agent_orchestrator.tools.registry.ToolRegistry` containing the
application's existing support tools, with each handler bound to trusted,
server-side dependencies (tenant id, database, approval state).

Two distinct concerns are kept separate:

* :data:`REGISTERED_TOOL_NAMES` -- every tool in the application's
  ``TOOL_REGISTRY``. These are *executable* by the orchestrator so that tools
  the model is not shown (for example legacy read-only tools) still produce a
  safe, deterministic result instead of an unhandled error.
* :data:`ADVERTISED_TOOL_NAMES` -- the subset sent to the model as available
  function definitions. This matches the historical ``run_agent`` behaviour
  (``search_support_docs`` is registered but not advertised).

The module does not duplicate business logic: every handler delegates to the
existing :func:`app.agent_core.tools.execute_tool` dispatcher, which owns schema
validation, approval gating, execution, and safe result semantics.

Handlers are intentionally independent of the LLM: ``tenant_id`` and ``db`` are
bound from trusted application context at registry-construction time, and any
``tenant_id`` the LLM supplies in its arguments is stripped before execution, so
the model can never override tenant context. Approval state likewise comes
exclusively from trusted application context; the model cannot approve its own
actions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Make the standalone Agent Orchestrator package importable without requiring it
# to be installed as a dependency. Mirrors how Retrieval Stack / Eval Harness are
# wired into the backend.
_ORCHESTRATOR_ROOT = Path(__file__).resolve().parents[2] / "Agent Orchestrator"
if _ORCHESTRATOR_ROOT.is_dir() and str(_ORCHESTRATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATOR_ROOT))

from agent_orchestrator.config import OrchestratorConfig
from agent_orchestrator.tools.executor import ToolExecutor
from agent_orchestrator.tools.models import ToolCall, ToolDefinition
from agent_orchestrator.tools.registry import ToolRegistry

from app.agent_core.tool_schemas import strip_titles
from app.agent_core.tools import GATED_TOOLS, TOOL_REGISTRY, execute_tool


#: Tool names the model may see and call (advertised function definitions).
ADVERTISED_TOOL_NAMES = ("search_knowledge", "create_support_case")

#: Backward-compatible alias used by existing tests.
ACTIVE_TOOL_NAMES = ADVERTISED_TOOL_NAMES

#: Every tool in the application registry. These are executable even when not
#: advertised, so legacy/read-only tools still yield a safe result.
REGISTERED_TOOL_NAMES = tuple(TOOL_REGISTRY.keys())


def _tool_description(tool_name: str, args_schema: type) -> str:
    """Return the description for a tool, mirroring run_agent's definitions."""
    if tool_name == "create_support_case":
        return "Create a support case"
    field_info = args_schema.model_fields.get("query")
    if field_info is not None and field_info.description:
        return field_info.description
    return ""


def _build_parameters_schema(args_schema: type) -> dict[str, Any]:
    """Build the JSON schema for a tool's parameters, mirroring run_agent."""
    parameters = strip_titles(args_schema.model_json_schema())
    parameters["additionalProperties"] = False
    return parameters


def _strip_tenant_argument(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return arguments with any LLM-supplied ``tenant_id`` removed.

    Tenant context is always injected from trusted server-side dependencies and
    must never be controllable by the model. This is defense-in-depth: the
    schema itself also rejects unknown properties, but stripping here guarantees
    the model can never override the trusted tenant id.
    """
    return {key: value for key, value in arguments.items() if key != "tenant_id"}


def _make_handler(
    tool_name: str,
    args_schema: type,
    *,
    tenant_id: str | None,
    db: Any,
    is_approved: bool,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a handler for a single tool bound to trusted dependencies."""

    def handler(**kwargs: Any) -> dict[str, Any]:
        # Tenant id and db come from trusted dependencies, never from the
        # model's arguments. Strip any tenant_id the model supplied.
        safe_arguments = _strip_tenant_argument(kwargs)

        # Delegate all validation, approval gating, execution, and safe error
        # semantics to the existing application dispatcher. Approval state is
        # trusted application context; the model cannot influence it.
        return execute_tool(
            tool_name,
            safe_arguments,
            tenant_id=tenant_id,
            db=db,
            is_approved=is_approved,
        )

    return handler


def build_tool_registry(
    *,
    tenant_id: str | None = None,
    db: Any = None,
    is_approved: bool = False,
    config: OrchestratorConfig | None = None,
) -> ToolRegistry:
    """Build a ``ToolRegistry`` for every registered application tool.

    Args:
        tenant_id: Trusted server-side tenant id injected into every tool call.
        db: Trusted database handle injected into every tool call.
        is_approved: Trusted approval state. When False, gated tools (such as
            ``create_support_case``) are refused before execution.
        config: Optional orchestrator configuration.

    Returns:
        A registry with one ``ToolDefinition`` and handler per registered tool.
    """
    registry = ToolRegistry()
    for tool_name in REGISTERED_TOOL_NAMES:
        args_schema, _handler_func = TOOL_REGISTRY[tool_name]

        definition = ToolDefinition(
            name=tool_name,
            description=_tool_description(tool_name, args_schema),
            parameters_schema=_build_parameters_schema(args_schema),
        )

        registry.register(
            definition,
            _make_handler(
                tool_name,
                args_schema,
                tenant_id=tenant_id,
                db=db,
                is_approved=is_approved,
            ),
        )

    return registry


def build_advertised_tool_definitions(registry: ToolRegistry) -> list[ToolDefinition]:
    """Return the tool definitions advertised to the model.

    Only the LLM-visible subset (``ADVERTISED_TOOL_NAMES``) is returned, in a
    stable order. Registered-but-unadvertised tools remain executable but are
    never offered to the model.
    """
    definitions: list[ToolDefinition] = []
    for tool_name in ADVERTISED_TOOL_NAMES:
        definition = registry.get_definition(tool_name)
        if definition is not None:
            definitions.append(definition)
    return definitions


def build_tool_executor(
    registry: ToolRegistry,
    config: OrchestratorConfig | None = None,
) -> ToolExecutor:
    """Build a :class:`ToolExecutor` for a registry.

    Args:
        registry: The registry of application tools.
        config: Optional orchestrator configuration (defaults to
            :class:`OrchestratorConfig`).

    Returns:
        A tool executor wired to the registry.
    """
    return ToolExecutor(registry=registry, config=config)


__all__ = [
    "ACTIVE_TOOL_NAMES",
    "ADVERTISED_TOOL_NAMES",
    "REGISTERED_TOOL_NAMES",
    "build_tool_registry",
    "build_advertised_tool_definitions",
    "build_tool_executor",
]
