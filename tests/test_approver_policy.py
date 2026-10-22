"""Tests for the role policy of the approval flow.

The policy says which role may do what. It is checked on the role that is
stored with the approver, never on one that a client supplied, and it
refuses everything it does not know.
"""

import pytest

from app.modules.approvals.policy import (
    CAPABILITY_DECIDE_ACTION,
    CAPABILITY_LIST_ACTIONS,
    CAPABILITY_MANAGE_APPROVERS,
    KNOWN_CAPABILITIES,
    ROLE_CAPABILITIES,
    capabilities_of,
    is_authorized,
)
from app.persistence.models import (
    APPROVER_ROLE_ADMIN,
    APPROVER_ROLE_AGENT,
    APPROVER_ROLE_BILLING_ADMIN,
    APPROVER_ROLE_SUPERVISOR,
    Approver,
)


def _approver(role, active=True, approver_id="approver-1"):
    """Return an approver record with the given role."""
    return Approver(
        id=approver_id,
        display_name="Sam Rivera",
        role=role,
        active=active,
        created_at="2026-01-01T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Who may decide an action
# ---------------------------------------------------------------------------


def test_a_supervisor_may_decide_an_action():
    """A supervisor is a human approver."""
    approver = _approver(APPROVER_ROLE_SUPERVISOR)

    assert is_authorized(approver, CAPABILITY_DECIDE_ACTION) is True
    assert is_authorized(approver, CAPABILITY_LIST_ACTIONS) is True


def test_a_billing_administrator_may_decide_an_action():
    """A billing administrator is a human approver."""
    approver = _approver(APPROVER_ROLE_BILLING_ADMIN)

    assert is_authorized(approver, CAPABILITY_DECIDE_ACTION) is True


def test_an_administrator_may_decide_an_action_and_manage_the_registry():
    """An administrator holds every capability."""
    approver = _approver(APPROVER_ROLE_ADMIN)

    for capability in KNOWN_CAPABILITIES:
        assert is_authorized(approver, capability) is True


def test_an_agent_may_never_decide_an_action():
    """An agent asks for actions, it does not decide them."""
    approver = _approver(APPROVER_ROLE_AGENT)

    assert is_authorized(approver, CAPABILITY_DECIDE_ACTION) is False
    assert is_authorized(approver, CAPABILITY_LIST_ACTIONS) is False
    assert is_authorized(approver, CAPABILITY_MANAGE_APPROVERS) is False
    assert capabilities_of(APPROVER_ROLE_AGENT) == frozenset()


def test_a_supervisor_may_not_manage_the_registry():
    """The registry is managed by administrators only."""
    approver = _approver(APPROVER_ROLE_SUPERVISOR)

    assert is_authorized(approver, CAPABILITY_MANAGE_APPROVERS) is False


# ---------------------------------------------------------------------------
# Everything that is not known is refused
# ---------------------------------------------------------------------------


def test_an_unknown_role_is_denied():
    """A role that is not in the policy holds nothing."""
    approver = _approver("wizard")

    for capability in KNOWN_CAPABILITIES:
        assert is_authorized(approver, capability) is False
    assert capabilities_of("wizard") == frozenset()


def test_an_unknown_capability_is_denied():
    """A capability that is not in the policy is refused, even for an admin."""
    approver = _approver(APPROVER_ROLE_ADMIN)

    assert is_authorized(approver, "action.execute") is False
    assert is_authorized(approver, "") is False


def test_no_approver_is_denied():
    """Nobody is not somebody."""
    assert is_authorized(None, CAPABILITY_DECIDE_ACTION) is False


def test_a_deactivated_approver_is_denied():
    """An approver that was deactivated holds nothing."""
    approver = _approver(APPROVER_ROLE_ADMIN, active=False)

    for capability in KNOWN_CAPABILITIES:
        assert is_authorized(approver, capability) is False


def test_the_role_is_read_from_the_stored_record():
    """The role of the stored record decides, not a role that is passed in."""
    approver = _approver(APPROVER_ROLE_SUPERVISOR)

    # The record says supervisor, so it may decide, whatever the caller
    # believes the role to be.
    assert is_authorized(approver, CAPABILITY_DECIDE_ACTION) is True
    assert capabilities_of(approver.role) == capabilities_of(
        APPROVER_ROLE_SUPERVISOR
    )


# ---------------------------------------------------------------------------
# The shape of the policy
# ---------------------------------------------------------------------------


def test_every_role_of_the_schema_is_in_the_policy():
    """A role of the database is never left without an entry."""
    from app.persistence.models import APPROVER_ROLES

    for role in APPROVER_ROLES:
        assert role in ROLE_CAPABILITIES


def test_every_capability_of_the_policy_is_known():
    """A capability that is granted is a capability that is known."""
    for capabilities in ROLE_CAPABILITIES.values():
        for capability in capabilities:
            assert capability in KNOWN_CAPABILITIES
