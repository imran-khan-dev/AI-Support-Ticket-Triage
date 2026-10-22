"""Which role may do what.

The approval flow needs a small, explicit policy: a capability is something
that can be done to an action or to the registry, and a role is a set of
capabilities. The policy is written down here instead of being spread over
the code that uses it, so that it can be read, reviewed, and tested on its
own.

The rules of the policy are:

* the least privilege. A role only holds the capabilities it needs, and the
  capabilities are the ones this project actually implements.
* an agent is never a human approver. The agent role is the identity of the
  party that asks for an action, so it holds no capability at all: an agent
  can never decide its own request.
* an inactive approver holds no capability either, whatever its role is.
* everything else is denied. A role that is not known, and a capability that
  is not known, are refused.
* the decision is made on the role that is stored with the approver, never
  on a role that a client sent along.
"""

from __future__ import annotations

from app.persistence.models import (
    APPROVER_ROLE_ADMIN,
    APPROVER_ROLE_AGENT,
    APPROVER_ROLE_BILLING_ADMIN,
    APPROVER_ROLE_SUPERVISOR,
    Approver,
)


#: See the actions of a session, so that they can be decided.
CAPABILITY_LIST_ACTIONS = "action.list"

#: Approve or reject an action that waits for a decision.
CAPABILITY_DECIDE_ACTION = "action.decide"

#: Execute an approved action.
CAPABILITY_EXECUTE_ACTION = "action.execute"

#: Create approvers, and deactivate or reactivate them.
CAPABILITY_MANAGE_APPROVERS = "approver.manage"

#: Every capability the policy knows. A capability that is not in this list
#: is denied.
KNOWN_CAPABILITIES = (
    CAPABILITY_LIST_ACTIONS,
    CAPABILITY_DECIDE_ACTION,
    CAPABILITY_EXECUTE_ACTION,
    CAPABILITY_MANAGE_APPROVERS,
)

#: The capabilities of every role.
#:
#: The agent holds none: it asks for actions, it does not decide them.
#:
#: A supervisor and a billing administrator decide actions. The schema
#: names a billing administrator, but there is no list of effect classes
#: yet, so the billing scope of the role cannot be expressed here. It will
#: be narrowed when the effect classes are defined.
#:
#: An administrator decides actions and manages the registry.
ROLE_CAPABILITIES: dict[str, frozenset[str]] = {
    APPROVER_ROLE_AGENT: frozenset(),
    APPROVER_ROLE_SUPERVISOR: frozenset(
        {CAPABILITY_LIST_ACTIONS, CAPABILITY_DECIDE_ACTION, CAPABILITY_EXECUTE_ACTION}
    ),
    APPROVER_ROLE_BILLING_ADMIN: frozenset(
        {CAPABILITY_LIST_ACTIONS, CAPABILITY_DECIDE_ACTION, CAPABILITY_EXECUTE_ACTION}
    ),
    APPROVER_ROLE_ADMIN: frozenset(
        {
            CAPABILITY_LIST_ACTIONS,
            CAPABILITY_DECIDE_ACTION,
            CAPABILITY_EXECUTE_ACTION,
            CAPABILITY_MANAGE_APPROVERS,
        }
    ),
}


def capabilities_of(role: str) -> frozenset[str]:
    """Return the capabilities of a role.

    A role that is not known holds no capability at all.
    """
    return ROLE_CAPABILITIES.get(role, frozenset())


def is_authorized(approver: Approver | None, capability: str) -> bool:
    """Return True when the approver may do the thing.

    The approver is the stored record of the identity, so the role that is
    used is the one that is stored with it, and never one that a client
    supplied. An approver that does not exist, an approver that was
    deactivated, a role that is not known, and a capability that is not
    known are all denied.
    """
    if approver is None:
        return False
    if not approver.active:
        return False
    if capability not in KNOWN_CAPABILITIES:
        return False
    return capability in capabilities_of(approver.role)
