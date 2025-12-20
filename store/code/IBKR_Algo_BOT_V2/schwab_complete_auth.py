"""
Complete Schwab Authentication
==============================
Run this after getting the redirect URL from Schwab login.

Usage:
    python schwab_complete_auth.py "https://127.0.0.1:6969/?code=XXXXX&session=YYYY"
"""

import sys
import os
import json
import base64
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from dotenv import load_dotenv

load_dotenv()

TOKEN_FILE = Path(__file__).parent / "schwab_token.json"
SCHWAB_API_BASE = "https://api.schwabapi.com"


def extract_code(redirect_url: str) -> str:
    """Extract authorization code from redirect URL"""
    parsed = urlparse(redirect_url)
    params = parse_qs(parsed.query)

    if 'code' not in params:
        raise ValueError("No authorization code found in URL")

    return params['code'][0]


def exchange_code_for_token(code: str) -> dict:
    """Exchange authorization code for access token"""
    app_key = os.getenv('SCHWAB_APP_KEY')
    app_secret = os.getenv('SCHWAB_APP_SECRET')
    callback_url = "HTTPS://127.0.0.1:6969"

    if not app_key or not app_secret:
        raise ValueError("SCHWAB_APP_KEY and SCHWAB_APP_SECRET must be set in .env")

    # Create Basic auth header
    credentials = f"{app_key}:{app_secret}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": callback_url
    }

    response = httpx.post(
        f"{SCHWAB_API_BASE}/v1/oauth/token",
        headers=headers,
        data=data,
        timeout=30.0
    )

    if response.status_code != 200:
        raise Exception(f"Token exchange failed: {response.status_code} - {response.text}")

    return response.json()


def save_token(token_data: dict):
    """Save token to file"""
    token_data['saved_at'] = datetime.now().isoformat()

    with open(TOKEN_FILE, 'w') as f:
        json.dump(token_data, f, indent=2)

    print(f"Token saved to: {TOKEN_FILE}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python schwab_complete_auth.py \"REDIRECT_URL\"")
        print()
        print("Example:")
        print('  python schwab_complete_auth.py "https://127.0.0.1:6969/?code=XXXXX&session=YYYY"')
        sys.exit(1)

    redirect_url = sys.argv[1]

    print("="*60)
    print("COMPLETING SCHWAB AUTHENTICATION")
    print("="*60)
    print()

    try:
        # Extract code
        print("Extracting authorization code...")
        code = extract_code(redirect_url)
        print(f"Code: {code[:20]}...")

        # Exchange for token
        print("\nExchanging code for access token...")
        token_data = exchange_code_for_token(code)

        # Save token
        print("\nSaving token...")
        save_token(token_data)

        print()
        print("="*60)
        print("SUCCESS! Token saved.")
        print("="*60)
        print()
        print("Token expires in:", token_data.get('expires_in', 0), "seconds")
        print("Refresh token valid for: 7 days")
        print()
        print("You can now use Schwab market data and trading!")

    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
