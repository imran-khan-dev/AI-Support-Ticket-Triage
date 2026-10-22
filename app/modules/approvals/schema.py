from pydantic import BaseModel, Field


class ActionResponse(BaseModel):
    id: str
    request_key: str
    session_id: str | None = None
    correlation_id: str
    tool_name: str
    effect_class: str
    state: str
    requested_by: str
    requested_at: str
    decided_by: str | None = None
    decided_at: str | None = None
    decision_reason: str | None = None
    execution_status: str | None = None
    execution_error: str | None = None
    executed_by: str | None = None
    executed_at: str | None = None


class DecisionRequest(BaseModel):
    decision: str = Field(..., pattern="^(approved|rejected)$")
    reason: str | None = None


class ExecutionResponse(BaseModel):
    action: ActionResponse
    executed: bool
    result: object | None = None