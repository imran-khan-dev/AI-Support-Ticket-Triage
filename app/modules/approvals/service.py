"""The approver registry and the authentication of approvers.

An approver is an identity that may decide actions. The service owns two
things: the registry of approvers, and the check of a credential that is
presented by one of them.

The credential of an approver is a bearer credential: a long random string
that the approver keeps. The service never stores it. What it stores is the
digest of it, computed with the server side secret of the configuration, and
the digest is what a presented credential is looked up by.

Nothing here logs a credential or a digest, and nothing here returns the
digest of a credential.
"""

from __future__ import annotations

import psycopg

from app.core.security import (
    CredentialError,
    hash_approver_token,
    validate_secret,
    validate_token,
)
from app.persistence.db import Database
from app.persistence.models import Approver
from app.persistence.repository import create_approver, get_approver_by_token_hash
from app.persistence.repository import set_approver_active as update_approver_active


class ApproverError(RuntimeError):
    """Raised when an approver cannot be registered or used."""


class ApproverConfigurationError(ApproverError):
    """Raised when the server side secret is not configured.

    Without the secret a credential cannot be digested, so no approver can
    be registered and no credential can be checked. The service refuses to
    work instead of falling back to something weaker.
    """


class ApproverCredentialInUseError(ApproverError):
    """Raised when a credential belongs to an approver already.

    A credential identifies exactly one approver, so a second one cannot
    hold it.
    """


def _require_secret(secret: object) -> str:
    """Return the server side secret, or refuse to work without it."""
    try:
        return validate_secret(secret)
    except CredentialError as exc:
        raise ApproverConfigurationError(
            "The approver token secret is not configured."
        ) from exc


def register_approver(
    db: Database,
    *,
    display_name: str,
    role: str,
    token: str,
    secret: str,
) -> Approver:
    """Store an approver and return it.

    The credential is given by the caller, which keeps it and hands it to
    the approver once. Only the digest of the credential is stored, and the
    digest is never returned.

    Raises :class:`ApproverCredentialInUseError` when the credential
    belongs to an approver already, and :class:`ApproverConfigurationError`
    when the server side secret is missing or too short.
    """
    key = _require_secret(secret)
    try:
        validate_token(token)
    except CredentialError as exc:
        raise ApproverError("The credential cannot be used.") from exc

    digest = hash_approver_token(token, key)
    try:
        return create_approver(
            db,
            display_name=display_name,
            role=role,
            token_hash=digest,
        )
    except psycopg.errors.UniqueViolation as exc:
        raise ApproverCredentialInUseError(
            "The credential is already in use."
        ) from exc


def authenticate_approver(
    db: Database,
    token: object,
    *,
    secret: str,
) -> Approver | None:
    """Return the approver that presented this credential.

    ``None`` is returned when the credential is not known, when it is not
    usable, or when the approver was deactivated. The reason is not
    returned, so that a caller cannot tell one refusal from another.
    """
    key = _require_secret(secret)
    if not isinstance(token, str):
        return None

    try:
        digest = hash_approver_token(token, key)
    except CredentialError:
        return None

    approver = get_approver_by_token_hash(db, digest)
    if approver is None:
        return None
    if not approver.active:
        return None
    return approver


def set_approver_active(
    db: Database,
    approver_id: str,
    active: bool,
) -> Approver | None:
    """Activate or deactivate an approver and return the stored row.

    Returns ``None`` when the approver does not exist. The credential of a
    deactivated approver is kept, so that the approver can be reactivated,
    but it no longer grants anything.
    """
    return update_approver_active(db, approver_id, active)
