# Manual Setup Instructions for IBKR Trading Bot

If the PowerShell bootstrap script doesn't work, follow these manual steps:

## Step 1: Setup Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install ib-insync fastapi uvicorn[standard] python-dotenv pandas numpy
```

## Step 2: Configure Environment
1. Copy `.env.example` to `.env`
2. Edit `.env` file:
   - Change `LOCAL_API_KEY` to something secure
   - Set `TWS_PORT=7497` for paper trading
   - Set `TWS_PORT=7496` for live trading (be careful!)

## Step 3: Setup TWS/IB Gateway
1. Start TWS or IB Gateway
2. Go to: Global Configuration → API → Settings
3. Check "Enable ActiveX and Socket Clients"
4. Set Socket Port to 7497 (paper) or 7496 (live)
5. Add 127.0.0.1 to Trusted IPs
6. Restart TWS

## Step 4: Test Connection
```bash
# Test if TWS is listening
python -c "import socket; socket.create_connection(('127.0.0.1', 7497), timeout=3); print('TWS is listening')"
```

## Step 5: Start the API
```bash
# Make sure you're in the venv
venv\Scripts\activate

# Start the server
python dashboard_api.py
```

## Step 6: Access the Interface
- Status Dashboard: http://127.0.0.1:9101/ui/status.html
- Trading Interface: http://127.0.0.1:9101/ui/trading.html
- API Docs: http://127.0.0.1:9101

## Troubleshooting
1. Make sure TWS/IB Gateway is running first
2. Check that API is enabled in TWS settings
3. Verify ports match between .env and TWS settings
4. Check Windows Firewall isn't blocking connections
5. Try different CLIENT_ID values in .env if you get "client id in use" errors

## Quick Test Commands
```bash
# Test API health
curl http://127.0.0.1:9101/health

# Test connection status
curl http://127.0.0.1:9101/api/status

# Test debug info
curl http://127.0.0.1:9101/api/debug/ibkr
```
