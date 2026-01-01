"""
Schwab API Authentication Script
Run this script ONCE to authenticate with Schwab and save your token.
"""

import base64
import json
import os
import sys
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TOKEN_FILE = Path(__file__).parent / "schwab_token.json"


def main():
    print("=" * 60)
    print("  SCHWAB/THINKORSWIM API AUTHENTICATION")
    print("=" * 60)
    print()

    # Check for credentials
    app_key = os.getenv("SCHWAB_APP_KEY")
    app_secret = os.getenv("SCHWAB_APP_SECRET")
    callback_url = os.getenv("SCHWAB_CALLBACK_URL", "https://127.0.0.1:6969")

    if not app_key or app_key == "YOUR_APP_KEY_HERE":
        print("ERROR: SCHWAB_APP_KEY not configured in .env!")
        input("\nPress Enter to exit...")
        return False

    if not app_secret or app_secret == "YOUR_APP_SECRET_HERE":
        print("ERROR: SCHWAB_APP_SECRET not configured in .env!")
        input("\nPress Enter to exit...")
        return False

    print(f"App Key: {app_key[:10]}...{app_key[-4:]}")
    print(f"Callback URL: {callback_url}")
    print()

    # Build authorization URL
    auth_url = "https://api.schwabapi.com/v1/oauth/authorize"
    params = {
        "client_id": app_key,
        "redirect_uri": callback_url,
        "response_type": "code",
    }
    full_auth_url = f"{auth_url}?{urlencode(params)}"

    print("=" * 60)
    print("  STEP 1: BROWSER LOGIN")
    print("=" * 60)
    print()
    print("Opening browser for Schwab login...")
    print()
    print("IMPORTANT:")
    print("1. Log in to your Schwab account")
    print("2. Accept the permissions")
    print("3. You'll be redirected to a page that won't load")
    print("4. COPY THE ENTIRE URL from your browser's address bar")
    print()

    webbrowser.open(full_auth_url)

    print("=" * 60)
    print("  STEP 2: PASTE THE REDIRECT URL")
    print("=" * 60)
    print()
    print("After logging in, paste the ENTIRE URL from your browser here.")
    print("It will look like: https://127.0.0.1:6969/?code=XXXXX&session=YYYY")
    print()

    redirect_url = input("Paste URL here: ").strip()

    if not redirect_url:
        print("\nERROR: No URL provided!")
        input("\nPress Enter to exit...")
        return False

    # Parse the authorization code from the URL
    try:
        parsed = urlparse(redirect_url)
        query_params = parse_qs(parsed.query)

        if "code" not in query_params:
            print(f"\nERROR: No authorization code found in URL!")
            print(f"URL received: {redirect_url}")
            print(f"Query params: {query_params}")
            input("\nPress Enter to exit...")
            return False

        auth_code = query_params["code"][0]
        print(f"\nAuthorization code received: {auth_code[:20]}...")

    except Exception as e:
        print(f"\nERROR parsing URL: {e}")
        input("\nPress Enter to exit...")
        return False

    # Exchange code for token
    print()
    print("=" * 60)
    print("  STEP 3: EXCHANGING CODE FOR TOKEN")
    print("=" * 60)
    print()

    token_url = "https://api.schwabapi.com/v1/oauth/token"

    # Create Basic auth header
    credentials = f"{app_key}:{app_secret}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": callback_url,
    }

    print("Requesting access token...")

    try:
        response = httpx.post(token_url, headers=headers, data=data)

        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            token_data = response.json()

            # Save token to file
            with open(TOKEN_FILE, "w") as f:
                json.dump(token_data, f, indent=2)

            print()
            print("=" * 60)
            print("  SUCCESS!")
            print("=" * 60)
            print()
            print(f"Token saved to: {TOKEN_FILE}")
            print()
            print(
                "Access token expires in:",
                token_data.get("expires_in", "unknown"),
                "seconds",
            )
            print(
                "Refresh token received:",
                "Yes" if "refresh_token" in token_data else "No",
            )
            print()
            print("Your Schwab integration is now ready!")
            print("Restart the trading platform to use real-time TOS data.")
            print()
            input("Press Enter to exit...")
            return True

        else:
            print()
            print("ERROR: Token exchange failed!")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            print()

            if response.status_code == 401:
                print("This usually means:")
                print("- App Key or Secret is incorrect")
                print("- The authorization code expired (try again quickly)")

            elif response.status_code == 400:
                print("This usually means:")
                print("- The callback URL doesn't match exactly")
                print("- The authorization code was already used")

            input("\nPress Enter to exit...")
            return False

    except Exception as e:
        print(f"\nERROR during token exchange: {e}")
        input("\nPress Enter to exit...")
        return False


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        input("\nPress Enter to exit...")
