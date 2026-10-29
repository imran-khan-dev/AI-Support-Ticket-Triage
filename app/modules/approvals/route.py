from fastapi import APIRouter, Depends, HTTPException, Header, Request
from app.core.config import Settings
from app.database import get_database, DatabaseConfigurationError
from app.modules.approvals.service import authenticate_approver
from app.modules.approvals.policy import (
    is_authorized,
    CAPABILITY_LIST_ACTIONS,
    CAPABILITY_DECIDE_ACTION,
    CAPABILITY_EXECUTE_ACTION,
    CAPABILITY_EXECUTE_ANY_TENANT,
)
from app.modules.approvals.decision_service import (
    decide_approval,
    ApprovalDecisionError,
    ApprovalDecisionUnauthorizedError,
)
from app.modules.approvals.execution_service import execute_approved_action, ActionExecutionError, ActionExecutionUncertainError
from app.database.repository import list_actions, get_action, get_action_for_tenant
from app.modules.approvals.schema import ActionResponse, DecisionRequest, ExecutionResponse

from app.core.config import Settings
router = APIRouter(tags=["approvals"])


def _database():
    try:
        return get_database(Settings().database_url)
    except DatabaseConfigurationError:
        raise HTTPException(status_code=503, detail="Storage unavailable.")


def _approver(database, token):
    if not Settings().approver_token_secret:
        raise HTTPException(status_code=503, detail="Approver authentication unavailable.")
    approver = authenticate_approver(database, token, secret=Settings().approver_token_secret)
    if approver is None:
        raise HTTPException(status_code=401, detail="Invalid approver credential.")
    return approver


@router.get("/actions", response_model=list[ActionResponse])
def list_actions_endpoint(
    session_id: str | None = None,
    state: str | None = None,
    correlation_id: str | None = None,
    limit: int = 50,
    x_approver_token: str | None = Header(default=None),
    database=Depends(_database),
):
    approver = _approver(database, x_approver_token)
    if not is_authorized(approver, CAPABILITY_LIST_ACTIONS):
        raise HTTPException(status_code=403, detail="Not authorized.")
    actions = list_actions(database, session_id=session_id, state=state, correlation_id=correlation_id, limit=limit)
    return [ActionResponse(**a.__dict__) for a in actions]


@router.post("/actions/{action_id}/decide", response_model=ActionResponse)
def decide_action_endpoint(
    action_id: str,
    request: DecisionRequest,
    x_approver_token: str | None = Header(default=None),
    database=Depends(_database),
):
    approver_token = x_approver_token
    database_handle = database
    if not Settings().approver_token_secret:
        raise HTTPException(status_code=503, detail="Approver authentication unavailable.")
    try:
        result = decide_approval(
            database_handle,
            action_id=action_id,
            decision=request.decision,
            approver_token=approver_token,
            secret=Settings().approver_token_secret,
            reason=request.reason,
        )
    except ApprovalDecisionUnauthorizedError:
        raise HTTPException(status_code=401, detail="Invalid or unauthorized approver.")
    except ApprovalDecisionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return ActionResponse(**result.action.__dict__)


@router.post("/actions/{action_id}/execute", response_model=ExecutionResponse)
def execute_action_endpoint(
    action_id: str,
    request: Request,
    x_approver_token: str | None = Header(default=None),
    database=Depends(_database),
):
    approver = _approver(database, x_approver_token)
    if not is_authorized(approver, CAPABILITY_EXECUTE_ACTION):
        raise HTTPException(status_code=403, detail="Not authorized.")

    # Load action first to determine tenant ownership
    action = get_action(database, action_id)
    if action is None:
        raise HTTPException(status_code=409, detail="The action does not exist.")

    # Determine request tenant context
    req_tenant = getattr(request.state, "tenant", None)
    req_tenant_id = getattr(req_tenant, "id", None) if req_tenant is not None else None

    can_execute_any_tenant = is_authorized(approver, CAPABILITY_EXECUTE_ANY_TENANT)

    # Tenant-owned actions require explicit tenant context and must match, unless explicitly authorized
    if action.tenant_id is not None:
        if req_tenant_id is None:
            if not can_execute_any_tenant:
                raise HTTPException(status_code=403, detail="Not authorized.")
        else:
            if str(req_tenant_id) != str(action.tenant_id):
                if not can_execute_any_tenant:
                    raise HTTPException(status_code=403, detail="Not authorized.")
            else:
                action = get_action_for_tenant(database, action_id, req_tenant_id) or action
    else:
        # No tenant on action - if request provides tenant context, do not allow cross-contamination
        if req_tenant_id is not None:
            if not can_execute_any_tenant:
                raise HTTPException(status_code=403, detail="Not authorized.")

    try:
        result = execute_approved_action(
            database,
            action_id=action_id,
            executed_by=str(approver.id),
            tenant_id=req_tenant_id if action.tenant_id is not None and req_tenant_id is not None and str(req_tenant_id) == str(action.tenant_id) else (req_tenant_id if (action.tenant_id is None and req_tenant_id is not None and can_execute_any_tenant) else (None if not can_execute_any_tenant else req_tenant_id)),
        )
    except ActionExecutionUncertainError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ActionExecutionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return ExecutionResponse(
        action=ActionResponse(**result.action.__dict__),
        executed=result.executed,
        result=result.result,
    )
