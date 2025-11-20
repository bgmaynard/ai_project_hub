# IBKR Auto-Connect Feature

**Created:** 2025-11-17
**Status:** Active

---

## Overview

The IBKR Auto-Connect feature automatically connects to Interactive Brokers TWS/Gateway when the dashboard server starts. This eliminates the need to manually click "Connect IBKR" in the UI every time you restart the server.

## Configuration

Auto-connect is configured via the `.env` file in the project root:

```env
# IBKR Auto-Connect Settings
IBKR_AUTO_CONNECT=true          # Enable/disable auto-connect
IBKR_HOST=127.0.0.1             # IBKR host (usually localhost)
IBKR_PORT=7497                  # Port number
IBKR_CLIENT_ID=1                # Client ID
```

### Port Numbers

| Service | Environment | Port |
|---------|-------------|------|
| TWS | Paper Trading | 7497 |
| TWS | Live Trading | 7496 |
| Gateway | Paper Trading | 4002 |
| Gateway | Live Trading | 4001 |

## How It Works

1. **Server Startup**: When `dashboard_api.py` starts, the `startup_event()` function runs
2. **Check Enabled**: Reads `IBKR_AUTO_CONNECT` from `.env`
3. **If Enabled**: Attempts to connect to IBKR using configured settings
4. **Success**: Logs "✓ Auto-connected to IBKR successfully"
5. **Failure**: Logs warning but doesn't crash server - you can manually connect via UI

## Enable/Disable

### Enable Auto-Connect
```env
IBKR_AUTO_CONNECT=true
```

### Disable Auto-Connect
```env
IBKR_AUTO_CONNECT=false
```

Or simply remove the `IBKR_AUTO_CONNECT` line (defaults to disabled).

## Requirements

Before auto-connect will work:

1. **TWS/Gateway Running**: IBKR must be running and listening on the configured port
2. **API Enabled**: TWS must have API access enabled
   - Edit → Global Configuration → API → Settings
   - ✓ Enable ActiveX and Socket Clients
3. **Correct Port**: `.env` port must match TWS settings
4. **Firewall**: Windows Firewall must allow Python and TWS to communicate

## Testing

### Test Auto-Connect
```powershell
# 1. Make sure TWS is running on port 7497
# 2. Check .env has IBKR_AUTO_CONNECT=true
# 3. Restart server
.\RESTART_SERVER.ps1

# 4. Check logs for:
# "Attempting to auto-connect to IBKR at 127.0.0.1:7497..."
# "✓ Auto-connected to IBKR successfully on port 7497"
```

### Check Connection Status
Open browser to: http://127.0.0.1:9101/api/ibkr/status

Expected response when connected:
```json
{
  "connected": true,
  "available": true
}
```

## Troubleshooting

### Auto-Connect Fails

**Error:** `✗ Auto-connect to IBKR failed: [Errno 10061] No connection could be made`

**Causes:**
1. TWS/Gateway not running
2. Wrong port number in `.env`
3. API not enabled in TWS
4. Firewall blocking connection

**Solutions:**
1. Start TWS/Gateway first, then restart server
2. Verify port: TWS Settings → API → Socket Port
3. Enable API in TWS (see Requirements above)
4. Add Python to Windows Firewall exceptions

### Auto-Connect Disabled

**Message:** `IBKR auto-connect disabled (set IBKR_AUTO_CONNECT=true in .env to enable)`

**Solution:** Set `IBKR_AUTO_CONNECT=true` in `.env` and restart server

### Manual Connection Override

Even with auto-connect enabled, you can still manually connect via:
- UI: Click "🔌 Connect IBKR" button
- API: POST to `/api/ibkr/connect` with custom parameters

## Benefits

✅ **Faster Development**: No need to click "Connect" every server restart
✅ **Production Ready**: Server automatically reconnects after crashes
✅ **Configurable**: Easy to enable/disable via `.env`
✅ **Safe**: Failure doesn't crash server - manual connection still available
✅ **Flexible**: Can override with manual connection if needed

## Server Logs

### Successful Auto-Connect
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Attempting to auto-connect to IBKR at 127.0.0.1:7497...
INFO:     ✓ Connected to IBKR on port 7497
INFO:     ✓ Auto-connected to IBKR successfully on port 7497
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9101
```

### Failed Auto-Connect (TWS not running)
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Attempting to auto-connect to IBKR at 127.0.0.1:7497...
WARNING:  ✗ Auto-connect to IBKR failed: [Errno 10061] No connection could be made
INFO:     You can manually connect via the UI or /api/ibkr/connect endpoint
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9101
```

### Auto-Connect Disabled
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     IBKR auto-connect disabled (set IBKR_AUTO_CONNECT=true in .env to enable)
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9101
```

## Files Modified

| File | Purpose |
|------|---------|
| `.env` | Added IBKR connection settings |
| `dashboard_api.py` | Added `startup_event()` handler (lines 185-210) |
| `IBKR_AUTO_CONNECT_GUIDE.md` | This documentation |

## Quick Start

1. **Enable auto-connect:**
   ```bash
   # Edit .env file
   IBKR_AUTO_CONNECT=true
   IBKR_PORT=7497
   ```

2. **Start TWS:**
   - Launch TWS/Gateway
   - Login to account
   - Verify API enabled

3. **Restart server:**
   ```powershell
   .\RESTART_SERVER.ps1
   ```

4. **Verify connection:**
   - Open http://127.0.0.1:9101/ui/complete_platform.html
   - Status should show "IBKR: Connected" (green)
   - No more 503 errors in logs

---

**Status:** ✅ Auto-connect feature implemented and ready to use
**Next Steps:** Restart server with TWS running to test auto-connect
