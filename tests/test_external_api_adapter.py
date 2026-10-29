
from app.agent_core.tool_schemas import CreateSupportCaseArgs
from app.integrations.external_api import (
    FakeSupportCaseAdapter,
    SupportCaseAdapter,
)


def make_args() -> CreateSupportCaseArgs:
    return CreateSupportCaseArgs(
        subject="Cannot log in",
        description="The password reset link fails.",
        priority="high",
        category="login",
    )


def test_fake_adapter_implements_support_case_contract():
    adapter = FakeSupportCaseAdapter()

    assert isinstance(adapter, SupportCaseAdapter)

    result = adapter.create_support_case(make_args())

    assert result["id"] == "case-1"
    assert result["subject"] == "Cannot log in"
    assert result["priority"] == "high"
    assert result["status"] == "created"


def test_fake_adapter_records_created_cases():
    adapter = FakeSupportCaseAdapter()

    first = adapter.create_support_case(make_args())
    second = adapter.create_support_case(make_args())

    assert first["id"] == "case-1"
    assert second["id"] == "case-2"
    assert len(adapter.created) == 2
    assert adapter.created[0]["description"] == (
        "The password reset link fails."
    )
