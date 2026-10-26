class StorageError(RuntimeError):
    """Raised when the session store cannot be read or written."""


class InvalidSessionIdError(ValueError):
    """Raised when a client-supplied session id is not a well-formed id."""


class SessionNotFoundError(LookupError):
    """Raised when a client-supplied session id does not identify a session."""


class SessionClosedError(RuntimeError):
    """Raised when a run is submitted against a closed session."""

