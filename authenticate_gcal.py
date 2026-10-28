"""One-time Google Calendar OAuth2 authorization script.

Run this script in your terminal to log into your Google Account and generate
`token.json` for live calendar synchronization:

    python authenticate_gcal.py
"""

from __future__ import annotations

import os
from google_auth_oauthlib.flow import InstalledAppFlow

from app import config

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCOPES = ["https://www.googleapis.com/auth/calendar"]


def get_credentials_path() -> str:
    # 1. Path from .env / config
    configured = os.path.join(_ROOT_DIR, getattr(config, "GOOGLE_CREDENTIALS_PATH", "credentials.json"))
    if os.path.isfile(configured):
        return configured
    # 2. credentials.json in root
    default_path = os.path.join(_ROOT_DIR, "credentials.json")
    if os.path.isfile(default_path):
        return default_path
    # 3. Any client_secret*.json file
    for fname in os.listdir(_ROOT_DIR):
        if fname.startswith("client_secret") and fname.endswith(".json"):
            return os.path.join(_ROOT_DIR, fname)
    return default_path


def authenticate():
    cred_path = get_credentials_path()
    token_path = os.path.join(_ROOT_DIR, getattr(config, "GOOGLE_TOKEN_PATH", "token.json"))

    if not os.path.isfile(cred_path):
        print(f"\n[ERROR] Credentials file not found at: {cred_path}")
        print("Please place your credentials.json or client_secret_*.json file in the project root.\n")
        return

    print("\n" + "=" * 70)
    print("  GOOGLE CALENDAR OAUTH2 AUTHORIZATION")
    print("=" * 70)
    print(f"Using client secrets from: {os.path.basename(cred_path)}")
    print("A browser window will open. Sign in to your Google Account and grant Calendar access.\n")

    try:
        flow = InstalledAppFlow.from_client_secrets_file(cred_path, _SCOPES)
        creds = flow.run_local_server(port=0)

        with open(token_path, "w") as token_file:
            token_file.write(creds.to_json())

        print(f"\n[SUCCESS] Authentication complete! Token saved to: {os.path.basename(token_path)}")
        print("Live Google Calendar synchronization is now active across all agent tools!\n")
    except Exception as e:
        print(f"\n[ERROR] OAuth flow failed: {e}\n")


if __name__ == "__main__":
    authenticate()
