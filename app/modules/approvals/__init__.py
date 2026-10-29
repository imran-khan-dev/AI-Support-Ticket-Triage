"""The approver registry, the authentication of approvers, and the policy.

This package holds the parts of the approval flow that do not need an HTTP
layer: who may decide an action, and how an approver proves who they are.

The approval endpoints are not part of it yet.
"""

from app.modules.approvals.policy import (
    CAPABILITY_DECIDE_ACTION,
    CAPABILITY_EXECUTE_ACTION,
    CAPABILITY_EXECUTE_ANY_TENANT,
    CAPABILITY_LIST_ACTIONS,
    CAPABILITY_MANAGE_APPROVERS,
    KNOWN_CAPABILITIES,
    ROLE_CAPABILITIES,
    capabilities_of,
    is_authorized,
)
from app.modules.approvals.service import (
    ApproverConfigurationError,
    ApproverCredentialInUseError,
    ApproverError,
    authenticate_approver,
    register_approver,
    set_approver_active,
)

__all__ = [
    "ApproverConfigurationError",
    "ApproverCredentialInUseError",
    "ApproverError",
    "CAPABILITY_DECIDE_ACTION",
    "CAPABILITY_EXECUTE_ACTION",
    "CAPABILITY_EXECUTE_ANY_TENANT",
    "CAPABILITY_LIST_ACTIONS",
    "CAPABILITY_MANAGE_APPROVERS",
    "KNOWN_CAPABILITIES",
    "ROLE_CAPABILITIES",
    "authenticate_approver",
    "capabilities_of",
    "is_authorized",
    "register_approver",
    "set_approver_active",
]
