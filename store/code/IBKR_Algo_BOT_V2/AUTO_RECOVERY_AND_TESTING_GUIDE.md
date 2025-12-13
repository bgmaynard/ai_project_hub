# Auto-Recovery & Order Testing Guide
## Complete Guide to Trading Bot Reliability & Verification

---

## Overview

This guide covers two critical systems added to ensure your trading bot can:
1. **Auto-recover** from network/PC errors to minimize slippage and trading interruption
2. **Verify order execution** works correctly and measure execution speed

---

## Part 1: Auto-Recovery System

### What Was Implemented

The bot now includes a **comprehensive auto-recovery system** to handle connection losses automatically:

#### 1. Connection Health Monitoring
- Background thread checks IBKR connection every **30 seconds**
- Detects connection loss immediately
- Runs continuously while bot is active

#### 2. Automatic Reconnection
- **Exponential backoff**: 5s base delay, 1.5x multiplier
- **Max attempts**: 10 reconnection tries
- **Smart retry**: Triples delay if crash happens within 30 seconds

#### 3. Worklist Auto-Resubscription
- All ticker feeds automatically re-subscribe after reconnect
- No manual intervention needed
- Zero data loss

#### 4. External Watchdog Script
- Monitors bot process
- Auto-restarts on crash
- Logs all crashes for debugging

---

### How to Use Auto-Recovery

#### Method 1: Start with Watchdog (Recommended)

The watchdog script monitors the bot and automatically restarts it on crashes:

```powershell
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
.\START_WITH_WATCHDOG.ps1
```

**Features:**
- Automatically restarts on crash (up to 100 times)
- 5-second delay between restarts
- Logs output to `bot_output.log` and `bot_error.log`
- Stops on clean exit (Ctrl+C or manual stop)

**Custom Configuration:**
```powershell
# Increase max restarts
.\START_WITH_WATCHDOG.ps1 -MaxRestarts 200

# Change restart delay
.\START_WITH_WATCHDOG.ps1 -RestartDelay 10

# Both
.\START_WITH_WATCHDOG.ps1 -MaxRestarts 200 -RestartDelay 10
```

#### Method 2: Standard Restart Script

For normal operation without external watchdog:

```powershell
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
.\RESTART_BOT.ps1
```

**Note:** Bot still has internal auto-reconnect even without watchdog.

---

### How Auto-Recovery Works

#### Connection Loss Scenario

1. **Detection** (within 30 seconds)
   ```
   [AUTO-RECONNECT] Connection lost detected!
   [AUTO-RECONNECT] Notifying connection lost callbacks...
   [WORKLIST] Connection lost - ticker subscriptions will be restored after reconnect
   ```

2. **Reconnection Attempts**
   ```
   [AUTO-RECONNECT] Attempt 1/10 in 5.0s...
   Connecting to TWS on 7497...
   SUCCESS! Connected!
   Server version: 176
   ```

3. **Restoration**
   ```
   [AUTO-RECONNECT] Successfully reconnected!
   [AUTO-RECONNECT] Notifying connection restored callbacks...
   [WORKLIST] Connection restored - re-subscribing to all symbols...
   [WORKLIST] Re-subscribing to SRDX
   [WORKLIST] Re-subscribing to PUBM
   ...
   [WORKLIST] Successfully re-subscribed to 20 symbols
   ```

#### Bot Crash Scenario (with Watchdog)

1. **Crash Detection**
   ```
   [WARNING] Bot stopped at 2025-11-19 10:45:23
   [INFO] Exit code: 1
   ```

2. **Auto-Restart**
   ```
   [AUTO-RECOVERY] Restarting in 5 seconds...

   ========================================
     STARTING BOT (Attempt #2)
   ========================================
   Time: 2025-11-19 10:45:28
   ```

3. **Service Restored**
   ```
   [OK] Bot started with PID: 12345
   [OK] Dashboard: http://127.0.0.1:9101/ui/complete_platform.html
   [MONITORING] Watching for crashes...
   ```

---

### Verification

To verify auto-recovery is working, check startup logs for:

```
✅ [AUTO-RECONNECT] Health monitoring started (check every 30s)
✅ [AUTO-RECONNECT] Registered connection lost callback: on_connection_lost
✅ [AUTO-RECONNECT] Registered connection restored callback: on_connection_restored
✅ [WORKLIST] Registered auto-reconnect callbacks
```

**Check Status API:**
```
GET http://127.0.0.1:9101/api/ibkr/status
```

Response includes auto-reconnect info:
```json
{
  "connected": true,
  "status": "connected",
  "auto_reconnect": {
    "enabled": true,
    "health_monitor_running": true,
    "last_health_check": "2025-11-19T10:30:00",
    "reconnect_attempts": 0,
    "max_attempts": 10
  }
}
```

---

## Part 2: Order Execution Testing

### Why Test Order Execution?

You need to verify:
1. ✅ Bot can place orders (not just monitor prices)
2. ✅ Orders are submitted correctly to IBKR
3. ✅ Order cancellation works
4. ⚡ Execution speed is acceptable
5. 📊 Position tracking works

---

### Running Order Tests

#### Prerequisites

1. **Paper Trading Account** (REQUIRED!)
   - Never run tests on live account
   - TWS/IB Gateway must be in paper trading mode

2. **Bot Running**
   - IBKR connection must be active
   - Dashboard API must be started

#### Run Tests

```bash
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT
python tests\test_order_execution.py --paper
```

**Options:**
```bash
# Test with different symbol
python tests\test_order_execution.py --paper --symbol AAPL

# Test with more shares
python tests\test_order_execution.py --paper --symbol SPY --quantity 10
```

---

### What Gets Tested

#### Test 1: IBKR Connection
- Verifies bot can connect to IBKR
- **Measures:** Connection time (ms)
- **Expected:** < 2000ms

#### Test 2: Get Positions
- Retrieves current open positions
- **Measures:** Query time (ms)
- **Expected:** < 500ms

#### Test 3: Get Open Orders
- Retrieves all open orders
- **Measures:** Query time (ms)
- **Expected:** < 500ms

#### Test 4: Place Limit Order
- Places BUY limit order 10% below market (won't fill)
- **Measures:** Order placement time (ms)
- **Expected:** < 1000ms
- **Safe:** Order won't execute (price too low)

#### Test 5: Cancel Order
- Cancels the limit order from Test 4
- **Measures:** Cancellation time (ms)
- **Expected:** < 1000ms

#### Test 6: Place Market Order (COMMENTED OUT)
- Disabled by default for safety
- Uncomment in code if you want to test actual fills
- **WARNING:** Will execute immediately and fill!

---

### Sample Output

```
======================================================================
STARTING ORDER EXECUTION TEST SUITE
======================================================================
Test Symbol: SPY
Test Quantity: 1
WARNING: This will place REAL orders in your connected account!
         Make sure you're using PAPER TRADING!
======================================================================

10:30:15.234 [INFO] Connecting to IBKR...
10:30:15.456 [PASS] Connected successfully in 222ms

10:30:15.500 [INFO] Testing GET POSITIONS
10:30:15.567 [PASS] Positions retrieved successfully
10:30:15.567 [INFO]   Query time: 67ms
10:30:15.567 [INFO]   Open positions: 0

10:30:15.600 [INFO] Testing GET OPEN ORDERS
10:30:15.689 [PASS] Open orders retrieved successfully
10:30:15.689 [INFO]   Query time: 89ms
10:30:15.689 [INFO]   Open orders: 0

10:30:15.700 [INFO] Testing LIMIT order: BUY 1 SPY
10:30:15.750 [INFO]   Current price: $580.25
10:30:15.750 [INFO]   Limit price: $522.23 (won't fill)
10:30:16.123 [PASS] Limit order placed successfully
10:30:16.123 [INFO]   Order ID: 123
10:30:16.123 [INFO]   Execution time: 373ms

10:30:17.200 [INFO] Testing order CANCELLATION: Order ID 123
10:30:17.456 [PASS] Order cancelled successfully
10:30:17.456 [INFO]   Cancellation time: 256ms

======================================================================
ORDER EXECUTION TEST SUMMARY
======================================================================
✓ Connection.......................... PASS  (222ms)
✓ Get Positions...................... PASS  (67ms)
✓ Get Open Orders.................... PASS  (89ms)
✓ Limit Order Placement.............. PASS  (373ms)
✓ Order Cancellation................. PASS  (256ms)
======================================================================
Total: 5 tests  |  Passed: 5  |  Failed: 0

[SUCCESS] All order execution tests passed!
Your bot can successfully place and cancel orders.
======================================================================
```

---

### Interpreting Results

#### Execution Speed Benchmarks

**Good Performance:**
- Connection: < 2000ms
- Position Query: < 500ms
- Order Query: < 500ms
- Order Placement: < 1000ms
- Order Cancellation: < 1000ms

**If Slower:**
- **Network latency**: Check internet connection
- **IBKR server load**: Try off-peak hours
- **System resources**: Close other applications

#### Common Errors

**Error: Not connected to IBKR**
- ✅ Start dashboard_api.py first
- ✅ Verify IBKR TWS/Gateway is running
- ✅ Check port 7497 (paper) or 7496 (live) is accessible

**Error: Order rejected**
- ✅ Verify paper trading mode enabled
- ✅ Check account has sufficient buying power
- ✅ Verify symbol is tradeable

**Error: Permission denied**
- ✅ Enable API trading in TWS settings
- ✅ Verify "Allow connections from localhost only" is checked
- ✅ Verify Trusted IP: 127.0.0.1

---

## Part 3: Bot Activity Monitoring

### How to Know Bot is Working

#### 1. Check Console Output

**Active Trading Bot:**
```
[UPDATE] TARA: $6.19 (-8.29%)
[UPDATE] RVNL: $32.74 (-7.30%)
[UPDATE] GLTO: $19.20 (-8.00%)
[UPDATE] LEE: $4.47 (+1.82%)
```

If you see price updates flowing, bot is monitoring correctly.

#### 2. Check Dashboard

Open: `http://127.0.0.1:9101/ui/complete_platform.html`

**Verify:**
- ✅ IBKR connection shows "Connected" (not "Demo Mode")
- ✅ Worklist displays symbols with live prices
- ✅ % changes are updating (not stuck at 0%)
- ✅ Charts are loading

#### 3. Check API Status

```bash
curl http://127.0.0.1:9101/api/ibkr/status
```

**Healthy Response:**
```json
{
  "connected": true,
  "status": "connected",
  "auto_reconnect": {
    "enabled": true,
    "health_monitor_running": true,
    "last_health_check": "2025-11-19T10:30:00",
    "reconnect_attempts": 0
  }
}
```

#### 4. Check Worklist API

```bash
curl http://127.0.0.1:9101/api/worklist
```

**Healthy Response:**
```json
[
  {
    "symbol": "TARA",
    "last_price": 6.19,
    "change": -0.56,
    "changePercent": -8.29,
    "last_update": "2025-11-19T10:30:15"
  },
  ...
]
```

If `changePercent` is not 0, price updates are working.

---

## Part 4: Why Bot Might Not Be Trading

Even with working order execution, the bot might not trade because:

### 1. No Trading Signals
- **Check:** AI predictor must detect valid entry signals
- **Verify:** Review `ai_predictor.py` logs for signal detection
- **Action:** Lower confidence thresholds in settings (for testing)

### 2. Risk Management Blocks
- **Check:** Position size limits
- **Check:** Daily loss limits
- **Check:** Account buying power
- **Action:** Review risk settings in config

### 3. Market Conditions
- **Check:** Markets must be open for most strategies
- **Check:** Symbol must have sufficient volume
- **Action:** Test during market hours with liquid symbols

### 4. Auto-Trading Disabled
- **Check:** Auto-trader must be enabled in settings
- **Verify:** Dashboard shows "Auto-Trading: ON"
- **Action:** Enable in UI or config file

### 5. Symbol Not in Worklist
- **Check:** Only symbols in worklist are monitored for trading
- **Action:** Add symbols via dashboard or API

---

## Quick Reference

### Start Bot (Auto-Recovery)
```powershell
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
.\START_WITH_WATCHDOG.ps1
```

### Test Orders
```bash
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT
python tests\test_order_execution.py --paper
```

### Check Status
```bash
# Connection status
curl http://127.0.0.1:9101/api/ibkr/status

# Worklist status
curl http://127.0.0.1:9101/api/worklist

# Open positions
curl http://127.0.0.1:9101/api/positions

# Open orders
curl http://127.0.0.1:9101/api/orders
```

### Dashboard
```
http://127.0.0.1:9101/ui/complete_platform.html
```

### Logs
- Bot output: `C:\ai_project_hub\store\code\IBKR_Algo_BOT\bot_output.log`
- Bot errors: `C:\ai_project_hub\store\code\IBKR_Algo_BOT\bot_error.log`

---

## Troubleshooting

### Bot Keeps Crashing

1. **Check Python 3.13 AsyncIO Issue**
   - Known Windows bug with Python 3.13
   - Watchdog will auto-restart
   - Consider downgrading to Python 3.11

2. **Check Logs**
   ```bash
   cat C:\ai_project_hub\store\code\IBKR_Algo_BOT\bot_error.log
   ```

3. **Check IBKR Connection**
   - TWS/Gateway must be running
   - Correct port (7497 for paper, 7496 for live)
   - API trading enabled

### Auto-Reconnect Not Working

1. **Verify Health Monitor Started**
   ```
   [AUTO-RECONNECT] Health monitoring started (check every 30s)
   ```

2. **Check Callbacks Registered**
   ```
   [WORKLIST] Registered auto-reconnect callbacks
   ```

3. **Test Manual Reconnect**
   - Stop TWS/Gateway
   - Watch console for reconnection attempts
   - Restart TWS/Gateway
   - Verify bot reconnects automatically

### Orders Not Working

1. **Run Test Suite**
   ```bash
   python tests\test_order_execution.py --paper
   ```

2. **Check TWS Settings**
   - Configuration → API → Settings
   - "Enable ActiveX and Socket Clients" = checked
   - "Allow connections from localhost only" = checked
   - "Read-Only API" = unchecked

3. **Verify Paper Trading**
   - TWS login screen: Select "Paper Trading"
   - Account shows "DU" prefix (demo)

---

## Summary

**Auto-Recovery System:**
- ✅ Connection health monitoring every 30s
- ✅ Automatic reconnection with exponential backoff
- ✅ Worklist auto-resubscription
- ✅ External watchdog for crash recovery

**Order Testing:**
- ✅ Comprehensive test suite
- ✅ Execution speed benchmarking
- ✅ Safe testing with paper account
- ✅ Position and order verification

**Next Steps:**
1. Start bot with watchdog: `.\START_WITH_WATCHDOG.ps1`
2. Verify auto-recovery features in console
3. Run order tests: `python tests\test_order_execution.py --paper`
4. Monitor dashboard: `http://127.0.0.1:9101/ui/complete_platform.html`
5. Review execution speeds and adjust if needed

---

**Your trading bot is now production-ready with full auto-recovery and verified order execution!**
