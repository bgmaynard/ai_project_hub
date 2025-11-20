# Server Restart Guide
## How to Restart the Trading Bot

---

## Problem Solved

**Issue:** Port 9101 stays in use after stopping the bot, preventing restart.

**Solution:** Added force restart script and dashboard restart button to properly free port 9101.

**Update (Nov 19, 2025):** Fixed PID parsing bug in FORCE_RESTART_BOT.ps1 that was showing "PID 0" error.

---

## Method 1: Dashboard Restart Button (Easiest) ✅

### How to Use:

1. **Open Dashboard:**
   ```
   http://127.0.0.1:9101/ui/complete_platform.html
   ```

2. **Click "🔄 Restart Server" button** in the top-right corner (orange button)

3. **Confirm restart** when prompted

4. **Wait 10 seconds** for server to restart

5. **Page will auto-reload** and reconnect

### When to Use:
- Quick restarts during trading
- After changing settings
- To clear any connection issues
- When port 9101 is in use

### Requirements:
- Bot must be running with watchdog: `.\START_WITH_WATCHDOG.ps1`
- If not using watchdog, manually restart after button click

---

## Method 2: Force Restart Script (Most Reliable) ✅

### How to Use:

```powershell
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
.\FORCE_RESTART_BOT.ps1
```

### What It Does:

1. **Finds processes using port 9101** using `netstat`
2. **Kills those processes** forcefully
3. **Kills all Python processes** as backup
4. **Waits 3 seconds** for cleanup
5. **Verifies port 9101 is free**
6. **Retries if needed** (second attempt)
7. **Starts the bot** cleanly

### Output Example:

```
========================================
  FORCE RESTART - Freeing Port 9101
========================================

[1] Finding processes using port 9101...
    Found 1 process(es) using port 9101
    Killing PID 12345 (python)

[2] Killing all Python processes...
    Killing Python PID: 12345

[3] Waiting for processes to terminate...

[4] Verifying port 9101 is free...
    [OK] Port 9101 is free!

[5] Verifying no Python processes remain...
    [OK] All Python processes stopped

[6] Starting IBKR Trading Bot...
    Server will start on: http://127.0.0.1:9101
    Dashboard: http://127.0.0.1:9101/ui/complete_platform.html

    Press Ctrl+C to stop the bot
```

### When to Use:
- Port 9101 is stuck in use
- Dashboard restart button doesn't work
- After system crash or freeze
- Clean slate restart needed

---

## Method 3: Regular Restart Script

### How to Use:

```powershell
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
.\RESTART_BOT.ps1
```

### What It Does:
- Kills all Python processes
- Starts the bot
- **Does NOT** check port 9101 specifically

### When to Use:
- Normal restarts when port isn't stuck
- Quick restart during development

---

## Comparison

| Feature | Dashboard Button | Force Restart | Regular Restart |
|---------|-----------------|---------------|-----------------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Frees Port 9101** | ✅ Yes | ✅✅ Yes (verified) | ❌ Not guaranteed |
| **Auto-reload Page** | ✅ Yes | ❌ Manual | ❌ Manual |
| **Requires Watchdog** | ✅ Yes | ❌ No | ❌ No |
| **Reliability** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## Troubleshooting

### Error: "Port 9101 is still in use!"

**Try these steps in order:**

1. **Run Force Restart Script:**
   ```powershell
   .\FORCE_RESTART_BOT.ps1
   ```

2. **If that fails, find and kill manually:**
   ```powershell
   netstat -ano | findstr :9101
   # Note the PID (last column)
   taskkill /F /PID <PID>
   ```

3. **If still failing, reboot computer:**
   - Port may be held by Windows
   - Reboot will free all ports

### Error: "Dashboard button doesn't work"

**Possible causes:**

1. **Bot not running with watchdog:**
   - Solution: Use `.\START_WITH_WATCHDOG.ps1` instead of `.\RESTART_BOT.ps1`
   - Watchdog automatically restarts bot after shutdown

2. **API endpoint not loaded:**
   - Solution: Use Force Restart Script instead

3. **Network timeout:**
   - Solution: Wait 30 seconds and try again

### Error: "Cannot kill PID"

**If you see "Access Denied":**

1. **Run PowerShell as Administrator:**
   - Right-click PowerShell
   - Select "Run as Administrator"
   - Try force restart again

2. **Use Task Manager:**
   - Press `Ctrl+Shift+Esc`
   - Find `python.exe` processes
   - Right-click → End Task
   - Run restart script

---

## How the Restart Button Works

### Technical Details:

1. **User clicks button** in dashboard
2. **JavaScript calls** `/api/server/restart` endpoint
3. **Server receives request**, responds with success
4. **Background thread waits 1 second** (allows response to be sent)
5. **Server sends SIGTERM** to itself (graceful shutdown)
6. **Watchdog detects shutdown**, waits 5 seconds
7. **Watchdog restarts bot** automatically
8. **Dashboard auto-reloads** after 10 seconds

### API Endpoint:

```http
POST /api/server/restart
```

**Response:**
```json
{
  "success": true,
  "message": "Server restart initiated. Please refresh page in 10 seconds.",
  "restart_delay_seconds": 10
}
```

---

## Best Practices

### For Development:
- Use **Dashboard Button** for quick restarts
- Keep watchdog running in background
- Auto-reload makes testing faster

### For Production:
- Use **START_WITH_WATCHDOG.ps1** to start bot
- Watchdog handles crashes automatically
- Use **Dashboard Button** for planned restarts
- Use **Force Restart** if bot becomes unresponsive

### For Troubleshooting:
- Always try **Force Restart** first
- Check logs: `bot_output.log` and `bot_error.log`
- Verify port 9101 is free before starting
- Reboot computer if port remains stuck

---

## Quick Reference

### Start Bot (Recommended):
```powershell
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
.\START_WITH_WATCHDOG.ps1
```

### Restart from Dashboard:
1. Click "🔄 Restart Server" button (top-right)
2. Confirm
3. Wait 10 seconds

### Force Restart:
```powershell
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
.\FORCE_RESTART_BOT.ps1
```

### Check Port 9101:
```powershell
netstat -ano | findstr :9101
```

### Kill Port 9101:
```powershell
# Get PID from netstat output
taskkill /F /PID <PID>
```

---

## Summary

**Problem:** Port 9101 stays in use, preventing restart.

**Solutions Added:**
1. ✅ **Dashboard restart button** - Easy one-click restart
2. ✅ **Force restart script** - Aggressively frees port 9101
3. ✅ **API endpoint** - `/api/server/restart` for programmatic restart

**Recommendation:**
- Start with: `.\START_WITH_WATCHDOG.ps1`
- Restart with: **Dashboard button** (or `.\FORCE_RESTART_BOT.ps1` if stuck)

Your port 9101 issues are now solved! 🎉
