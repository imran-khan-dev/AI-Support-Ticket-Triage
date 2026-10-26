#!/usr/bin/env python
"""CLI/admin script to issue integration credentials.

Usage:
  python scripts/issue_integration_credential.py <integration_name>
"""
import sys
from app.database import get_database
from app.core.config import Settings
from app.modules.integrations.service import issue_integration_credential


def main():
    if len(sys.argv) < 2:
        print("Usage: issue_integration_credential.py <integration_name>", file=sys.stderr)
        sys.exit(1)
    name = sys.argv[1]
    db = get_database(Settings().database_url)
    db.initialize()
    integration, credential, token = issue_integration_credential(db, integration_name=name)
    print(f"INTEGRATION_ID={integration.id}")
    print(f"INTEGRATION_NAME={integration.name}")
    print(f"CREDENTIAL_ID={credential.id}")
    print(f"KEY_ID={credential.key_id}")
    print(f"TOKEN={token}")
    print("WARNING: Store this token securely. It will not be shown again.")


if __name__ == "__main__":
    main()
