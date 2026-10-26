"""Tests for the approver registry and the authentication of approvers.

The credentials of the tests are generated for the tests and are used
against the dedicated test database, which is emptied before each test.
"""

import pytest

from app.core.security import (
    CredentialError,
    generate_approver_token,
    hash_approver_token,
    validate_secret,
    validate_token,
)
from app.modules.approvals.service import (
    ApproverConfigurationError,
    ApproverCredentialInUseError,
    ApproverError,
    authenticate_approver,
    register_approver,
    set_approver_active,
)
from app.database import (
    APPROVER_ROLE_ADMIN,
    APPROVER_ROLE_SUPERVISOR,
    get_approver,
    get_approver_by_token_hash,
    list_approvers,
)


#: Every test in this module writes to the database.
pytestmark = pytest.mark.usefixtures("clean_database")

#: A secret that is only used by these tests. It is not a production
#: secret and it protects nothing but a test database.
TEST_SECRET = "test-only-secret-for-approver-credentials-0123456789"


def _register(database, **overrides):
    """Register an approver and return it and its credential."""
    values = {
        "display_name": "Sam Rivera",
        "role": APPROVER_ROLE_SUPERVISOR,
        "token": generate_approver_token(),
        "secret": TEST_SECRET,
    }
    values.update(overrides)
    approver = register_approver(database, **values)
    return approver, values["token"]


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_an_approver_is_created_and_read_back(database):
    """A new approver is active and holds its role."""
    approver, _ = _register(database)

    assert approver.id
    assert approver.display_name == "Sam Rivera"
    assert approver.role == APPROVER_ROLE_SUPERVISOR
    assert approver.active is True
    assert approver.created_at

    stored = get_approver(database, approver.id)
    assert stored is not None
    assert stored.id == approver.id
    assert stored.display_name == "Sam Rivera"
    assert stored.role == APPROVER_ROLE_SUPERVISOR
    assert stored.active is True


def test_a_missing_approver_is_reported(database):
    """An approver that does not exist is not invented."""
    import uuid

    assert get_approver(database, str(uuid.uuid4())) is None
    assert get_approver(database, "not-a-uuid") is None


def test_the_credential_itself_is_never_stored(database):
    """The database holds the digest of a credential, not the credential."""
    approver, token = _register(database)

    with database.connection() as conn:
        row = conn.execute(
            "SELECT token_hash FROM approvers WHERE id = %s", (approver.id,)
        ).fetchone()

    stored = row["token_hash"]
    assert stored != token
    assert token not in stored
    assert stored == hash_approver_token(token, TEST_SECRET)


def test_the_digest_of_a_credential_never_reaches_the_model(database):
    """The stored digest is not part of the record that is handed out."""
    approver, token = _register(database)
    digest = hash_approver_token(token, TEST_SECRET)

    assert not hasattr(approver, "token_hash")
    assert digest not in repr(approver)
    assert token not in repr(approver)

    found = get_approver_by_token_hash(database, digest)
    assert found is not None
    assert not hasattr(found, "token_hash")
    assert digest not in repr(found)


def test_an_approver_can_be_deactivated_and_reactivated(database):
    """A deactivated approver keeps its place in the registry."""
    approver, _ = _register(database)

    deactivated = set_approver_active(database, approver.id, False)
    assert deactivated is not None
    assert deactivated.active is False
    assert deactivated.id == approver.id

    stored = get_approver(database, approver.id)
    assert stored is not None
    assert stored.active is False

    reactivated = set_approver_active(database, approver.id, True)
    assert reactivated is not None
    assert reactivated.active is True


def test_deactivating_an_approver_that_does_not_exist_reports_nothing(database):
    """There is nothing to deactivate."""
    import uuid

    assert set_approver_active(database, str(uuid.uuid4()), False) is None
    assert set_approver_active(database, "not-a-uuid", False) is None


def test_approvers_can_be_listed(database):
    """The listing is filtered and bounded."""
    _register(database, display_name="One", role=APPROVER_ROLE_SUPERVISOR)
    second, _ = _register(
        database, display_name="Two", role=APPROVER_ROLE_ADMIN
    )
    set_approver_active(database, second.id, False)

    assert len(list_approvers(database)) == 2
    assert len(list_approvers(database, role=APPROVER_ROLE_ADMIN)) == 1
    assert len(list_approvers(database, active=True)) == 1
    assert len(list_approvers(database, active=False)) == 1
    assert len(list_approvers(database, limit=1)) == 1
    assert len(list_approvers(database, limit=0)) == 0


def test_a_credential_belongs_to_one_approver(database):
    """A credential cannot be given to a second approver."""
    approver, token = _register(database)

    with pytest.raises(ApproverCredentialInUseError):
        _register(database, display_name="Someone else", token=token)

    assert len(list_approvers(database)) == 1
    assert get_approver(database, approver.id) is not None


def test_the_credential_of_an_approver_is_unique_in_the_database(database):
    """The database refuses a second approver with the same digest."""
    digest = hash_approver_token(generate_approver_token(), TEST_SECRET)

    with database.write_transaction() as conn:
        conn.execute(
            "INSERT INTO approvers (id, display_name, role, token_hash)"
            " VALUES (gen_random_uuid(), 'One', 'supervisor', %s)",
            (digest,),
        )

    with pytest.raises(Exception):
        with database.write_transaction() as conn:
            conn.execute(
                "INSERT INTO approvers (id, display_name, role, token_hash)"
                " VALUES (gen_random_uuid(), 'Two', 'supervisor', %s)",
                (digest,),
            )


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def test_a_presented_credential_authenticates_its_approver(database):
    """The approver of a credential is found by it."""
    approver, token = _register(database)

    found = authenticate_approver(database, token, secret=TEST_SECRET)

    assert found is not None
    assert found.id == approver.id
    assert found.role == APPROVER_ROLE_SUPERVISOR
    assert found.active is True


def test_an_unknown_credential_is_refused(database):
    """A credential that was never given is refused."""
    _register(database)

    found = authenticate_approver(
        database, generate_approver_token(), secret=TEST_SECRET
    )

    assert found is None


def test_an_empty_or_malformed_credential_is_refused(database):
    """A credential that cannot be one is refused, and nothing is raised."""
    _register(database)

    for token in ["", "   ", "short", "x" * 512, None, 42, b"bytes"]:
        found = authenticate_approver(
            database, token, secret=TEST_SECRET
        )
        assert found is None


def test_a_credential_of_another_secret_is_refused(database):
    """A credential is only good for the secret it was made with."""
    approver, token = _register(database)

    found = authenticate_approver(
        database,
        token,
        secret="another-test-secret-for-approver-credentials-987654321",
    )

    assert found is None
    assert approver.active is True


def test_a_deactivated_approver_is_refused(database):
    """A credential of an approver that was deactivated grants nothing."""
    approver, token = _register(database)
    set_approver_active(database, approver.id, False)

    found = authenticate_approver(database, token, secret=TEST_SECRET)

    assert found is None


def test_a_reactivated_approver_is_accepted_again(database):
    """Deactivating an approver does not destroy its credential."""
    approver, token = _register(database)
    set_approver_active(database, approver.id, False)
    set_approver_active(database, approver.id, True)

    found = authenticate_approver(database, token, secret=TEST_SECRET)

    assert found is not None
    assert found.id == approver.id


def test_the_registry_refuses_to_work_without_a_secret(database):
    """A missing or short secret stops the registry, it does not weaken it."""
    for secret in [None, "", "short"]:
        with pytest.raises(ApproverConfigurationError):
            register_approver(
                database,
                display_name="Sam Rivera",
                role=APPROVER_ROLE_SUPERVISOR,
                token=generate_approver_token(),
                secret=secret,
            )

        with pytest.raises(ApproverConfigurationError):
            authenticate_approver(
                database, generate_approver_token(), secret=secret
            )


def test_a_weak_credential_is_refused(database):
    """A credential that is too short to be safe is not stored."""
    with pytest.raises(ApproverError):
        register_approver(
            database,
            display_name="Sam Rivera",
            role=APPROVER_ROLE_SUPERVISOR,
            token="too-short",
            secret=TEST_SECRET,
        )

    assert list_approvers(database) == []


# ---------------------------------------------------------------------------
# The credential primitives
# ---------------------------------------------------------------------------


def test_a_generated_credential_is_long_and_different_every_time():
    """A generated credential carries enough entropy to be unguessable."""
    first = generate_approver_token()
    second = generate_approver_token()

    assert first != second
    assert len(first) >= 32
    assert first.isprintable()
    assert not any(character.isspace() for character in first)


def test_the_digest_of_a_credential_is_the_same_every_time():
    """A credential can be looked up by its digest."""
    token = generate_approver_token()

    first = hash_approver_token(token, TEST_SECRET)
    second = hash_approver_token(token, TEST_SECRET)

    assert first == second
    assert first != token


def test_the_digest_depends_on_the_secret():
    """The secret is part of the digest, so the database alone is not enough."""
    token = generate_approver_token()

    with_secret = hash_approver_token(token, TEST_SECRET)
    with_other = hash_approver_token(
        token, "another-test-secret-for-approver-credentials-987654321"
    )

    assert with_secret != with_other


def test_a_credential_that_cannot_be_used_is_refused():
    """Empty, short, long, and unreadable credentials are refused."""
    for token in [None, 42, b"bytes", "", "short", "x" * 300, "with space"]:
        with pytest.raises(CredentialError):
            validate_token(token)

    assert validate_token(generate_approver_token())


def test_a_secret_that_cannot_be_used_is_refused():
    """A short secret would weaken every digest that is computed with it."""
    for secret in [None, 42, "", "short"]:
        with pytest.raises(CredentialError):
            validate_secret(secret)

    assert validate_secret(TEST_SECRET)
