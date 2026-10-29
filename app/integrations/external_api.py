"""Minimal adapter interface for creating support cases."""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.agent_core.tool_schemas import CreateSupportCaseArgs


@runtime_checkable
class SupportCaseAdapter(Protocol):
    """Contract for a support-case creation adapter."""

    def create_support_case(self, args: CreateSupportCaseArgs) -> dict:
        """Create a support case and return its representation."""
        ...


@dataclass
class FakeSupportCaseAdapter:
    """In-memory adapter for deterministic tests."""

    _next_id: int = 1
    created: list[dict] = field(default_factory=list)

    def create_support_case(self, args: CreateSupportCaseArgs) -> dict:
        case = {
            "id": f"case-{self._next_id}",
            "subject": args.subject,
            "description": args.description,
            "priority": args.priority,
            "category": args.category,
            "status": "created",
        }
        self._next_id += 1
        self.created.append(case)
        return case.copy()
