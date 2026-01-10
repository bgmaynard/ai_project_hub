"""Quick Schwab Auth - paste URL immediately after redirect"""
import json, base64, os, webbrowser
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import httpx

load_dotenv()

app_key = os.getenv('SCHWAB_APP_KEY')
app_secret = os.getenv('SCHWAB_APP_SECRET')
callback_url = 'HTTPS://127.0.0.1:6969'  # Must be uppercase HTTPS to match Schwab config

print("Opening Schwab login...")
webbrowser.open(f"https://api.schwabapi.com/v1/oauth/authorize?client_id={app_key}&redirect_uri=HTTPS%3A//127.0.0.1%3A6969&response_type=code")

print("\n1. Login to Schwab")
print("2. Accept permissions")
print("3. Copy the redirect URL immediately")
print("4. Paste here and press Enter FAST!\n")

url = input("Paste URL: ").strip()

auth_code = parse_qs(urlparse(url).query)['code'][0]
print(f"Got code, exchanging...")

response = httpx.post(
    'https://api.schwabapi.com/v1/oauth/token',
    headers={
        'Authorization': f'Basic {base64.b64encode(f"{app_key}:{app_secret}".encode()).decode()}',
        'Content-Type': 'application/x-www-form-urlencoded'
    },
    data={'grant_type': 'authorization_code', 'code': auth_code, 'redirect_uri': callback_url},
    timeout=30.0
)

if response.status_code == 200:
    with open('schwab_token.json', 'w') as f:
        json.dump(response.json(), f, indent=2)
    print("\n*** SUCCESS! Token saved! ***")
    print("Restart the server to use Schwab.")
else:
    print(f"\nERROR: {response.text}")

input("\nPress Enter to exit...")
