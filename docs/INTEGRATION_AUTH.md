# Integration Authentication (Milestone 4)

## Overview
The AI backend authenticates incoming requests from registered product integrations using bearer tokens. Identity is derived only from verified credentials stored in the database.

## Authentication Flow
1. Integration presents `Authorization: Bearer <token>` to a protected endpoint.
2. FastAPI dependency `require_integration` extracts the token and calls `verify_integration_token`.
3. Token is hashed with SHA-256 and looked up by hash; integration is loaded.
4. Checks: credential enabled, not revoked, not expired; integration enabled.
5. On success, integration identity is attached to `request.state.integration`. On failure, returns 401 (consistent for missing/invalid/revoked/expired/disabled).
6. Outbound calls to product APIs use separate configuration and are not implemented in this milestone.

## Credential Lifecycle
- **Issuance**: Use CLI script `scripts/issue_integration_credential.py <name>` to create integration (if needed) and issue a credential. The plaintext token is printed once.
- **Storage**: Only `token_hash`, `key_id`, metadata stored. No plaintext tokens logged.
- **Rotation**: Issue a new credential while old remains valid; revoke old with `revoke_credential`.
- **Revocation**: Set `revoked_at`; credential becomes invalid immediately.
- **Expiry**: Optional `expires_at`; expired credentials are rejected.
- **Multiple credentials**: Supported per integration for safe rotation.

## Security Notes
- Never trust `X-Integration-Id`, `tenant_id`, or body identity fields.
- No tenant authorization implemented (Milestone 5).
- No consequential product actions implemented.
- Database errors fail closed (return 503/401 as appropriate).
