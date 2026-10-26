"""Credential primitives for approvers.

An approver holds a bearer credential: a long, random string that the
approver presents instead of a password. The credential itself is never
stored. What is stored is a digest of it, computed with a keyed hash, so
that a stolen database alone cannot be used to test a guess.

The digest is keyed with a server side secret, the pepper, that lives in
the configuration and not in the database. Two consequences follow:

* a credential cannot be recovered from the database, because the pepper is
  needed to compute a digest of a guess;
* the digest of a credential is the same every time, so the approver of a
  presented credential can be found with a lookup on the digest.

The credentials are generated with 256 bits of entropy, so a slow key
derivation is not needed for them: a slow function protects a guessable
password, and a credential of this strength is not guessable. The
mechanism used here is HMAC, which is a standard construction, and a
digest is compared in constant time where two digests are compared.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets


#: Number of random bytes in a generated credential.
TOKEN_BYTES = 32

#: Shortest credential that is accepted. A credential with less entropy
#: than this is refused instead of being stored.
TOKEN_MIN_LENGTH = 32

#: Longest credential that is accepted. The bound keeps a hostile input
#: from being hashed without limit.
TOKEN_MAX_LENGTH = 256

#: Shortest server side secret that is accepted as the pepper.
SECRET_MIN_LENGTH = 32


class CredentialError(ValueError):
    """Raised when a credential or a secret cannot be used."""


def generate_approver_token() -> str:
    """Return a new, random credential for an approver.

    The credential is built from the operating system's random source and
    is safe to use in a header or a query string. It is shown to the
    approver once, and only its digest is stored.
    """
    return secrets.token_urlsafe(TOKEN_BYTES)


def validate_token(token: object) -> str:
    """Return the credential when it is usable, and raise otherwise.

    A credential is refused when it is not text, when it is empty, when it
    is too short or too long, or when it holds whitespace or a control
    character. The reason is not part of the message, so that a caller
    cannot tell one kind of refusal from another.
    """
    if not isinstance(token, str):
        raise CredentialError("The credential is not usable.")
    if not TOKEN_MIN_LENGTH <= len(token) <= TOKEN_MAX_LENGTH:
        raise CredentialError("The credential is not usable.")
    for character in token:
        if character.isspace() or not character.isprintable():
            raise CredentialError("The credential is not usable.")
    return token


def validate_secret(secret: object) -> str:
    """Return the server side secret when it is usable, and raise otherwise.

    A secret that is too short is refused: a short pepper weakens every
    digest that is computed with it.
    """
    if not isinstance(secret, str) or len(secret) < SECRET_MIN_LENGTH:
        raise CredentialError("The server side secret is not usable.")
    return secret


def hash_approver_token(token: object, secret: object) -> str:
    """Return the digest that is stored for a credential.

    The digest is a keyed hash of the credential. The same credential
    always gives the same digest, which is what makes a lookup by digest
    possible, and the key is what makes the digest useless on its own.
    """
    text = validate_token(token)
    key = validate_secret(secret)
    return hmac.new(
        key.encode("utf-8"),
        text.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
