# 🤖 Autonomous Bot Control & Monitoring Guide
**Simple Guide to Control Your Trading Bot**

---

## 🎯 **BOT IS NOW RUNNING!**

**Current Status:**
- ✅ Bot: RUNNING
- ✅ Trading: ENABLED
- ✅ Account: $50,000
- ✅ Watchlist: AAPL, MSFT, GOOGL, TSLA, NVDA
- ✅ Risk Limits: Active (3-5-7 strategy)

---

## 📊 **HOW TO MONITOR YOUR BOT**

### **Option 1: Quick Status Check (Easiest)**

Open PowerShell or Command Prompt and run:

```powershell
# Quick status check
curl http://127.0.0.1:9101/api/bot/status

# Formatted (easier to read)
curl http://127.0.0.1:9101/api/bot/status | python -m json.tool
```

**What you'll see:**
```json
{
  "running": true,              ← Bot is running
  "enabled": true,              ← Trading is enabled
  "total_signals_generated": 5, ← Signals found
  "total_trades_executed": 2,   ← Trades made
  "total_trades_rejected": 3,   ← Trades rejected (good!)
  "trading_engine": {
    "open_positions": 1,        ← Currently holding 1 position
    "daily_pnl": 125.50,        ← Profit today: $125.50
    "trades_today": 2,          ← 2 trades today
    "daily_loss_limit": 500.0   ← Max loss allowed: $500
  }
}
```

---

### **Option 2: Visual Dashboard (Best for Watching)**

**Open in your browser:**
```
http://127.0.0.1:9101/ui/platform.html
```

This shows:
- 📈 Real-time price charts
- 💰 Open positions
- 📊 P&L (profit/loss)
- 🎯 Active signals
- 📉 Level 2 market depth
- ⏱️ Time & Sales

---

### **Option 3: Health Check (Overall System)**

```powershell
curl http://127.0.0.1:9101/health
```

**Shows:**
- Server status
- IBKR connection
- Claude AI availability
- Active subscriptions

---

## 🎮 **BOT CONTROL COMMANDS**

### **✅ START the Bot**

```powershell
curl -X POST http://127.0.0.1:9101/api/bot/start
```

**What it does:** Starts the bot (makes it analyze market and generate signals)

---

### **🟢 ENABLE Trading**

```powershell
curl -X POST http://127.0.0.1:9101/api/bot/enable
```

**What it does:** Allows the bot to actually place trades (bot must be started first)

---

### **🔴 STOP the Bot**

```powershell
curl -X POST http://127.0.0.1:9101/api/bot/stop
```

**What it does:**
- Stops generating new signals
- DOES NOT close existing positions
- Bot will stop trading but keeps monitoring

---

### **🟡 DISABLE Trading (Safe Mode)**

```powershell
curl -X POST http://127.0.0.1:9101/api/bot/disable
```

**What it does:**
- Bot keeps running and analyzing
- WILL NOT place new trades
- Keeps monitoring existing positions
- Good for "pause" mode

---

### **🚨 EMERGENCY STOP (Close Everything)**

```powershell
# 1. Disable trading first
curl -X POST http://127.0.0.1:9101/api/bot/disable

# 2. Check positions
curl http://127.0.0.1:9101/api/bot/status

# 3. If you see open positions, close them manually or:
curl -X POST http://127.0.0.1:9101/api/positions/close-all
```

---

## 📈 **TRACKING WHAT THE BOT IS DOING**

### **1. Check Signals (What it's thinking about)**

```powershell
curl http://127.0.0.1:9101/api/bot/status
```

Look for:
- `total_signals_generated` - How many opportunities found
- `total_trades_executed` - How many trades made
- `total_trades_rejected` - How many rejected (risk limits)
- `pending_predictions` - Currently analyzing

---

### **2. Check Open Positions (What you're holding)**

```powershell
curl http://127.0.0.1:9101/api/bot/status
```

Look at `trading_engine` section:
- `open_positions` - Number of positions
- `total_exposure_usd` - Total $ invested
- `unrealized_pnl` - Profit/loss on open trades

---

### **3. Check Today's Performance**

```powershell
curl http://127.0.0.1:9101/api/bot/status
```

Look at `trading_engine`:
- `daily_pnl` - Profit/loss today (in dollars)
- `daily_pnl_pct` - Profit/loss today (in %)
- `trades_today` - Number of trades today
- `daily_loss_limit` - Max loss allowed ($500)

---

### **4. Get Detailed Position Info**

```powershell
# See full position details
curl http://127.0.0.1:9101/api/bot/status | python -m json.tool
```

Scroll to `positions` section for:
- Symbol
- Shares owned
- Entry price
- Current price
- Profit/loss
- Stop loss level

---

## 🔔 **REAL-TIME MONITORING (Watch It Work)**

### **Method 1: Keep Checking Status**

Save this as `watch_bot.ps1`:
```powershell
while ($true) {
    Clear-Host
    Write-Host "=== BOT STATUS ===" -ForegroundColor Cyan
    Write-Host "Time: $(Get-Date)" -ForegroundColor Yellow

    $status = curl -s http://127.0.0.1:9101/api/bot/status | ConvertFrom-Json

    Write-Host "`nBot Running: " -NoNewline
    if ($status.running) { Write-Host "YES" -ForegroundColor Green }
    else { Write-Host "NO" -ForegroundColor Red }

    Write-Host "Trading Enabled: " -NoNewline
    if ($status.enabled) { Write-Host "YES" -ForegroundColor Green }
    else { Write-Host "NO" -ForegroundColor Red }

    Write-Host "`n=== PERFORMANCE ===" -ForegroundColor Cyan
    Write-Host "Signals Generated: $($status.total_signals_generated)"
    Write-Host "Trades Executed: $($status.total_trades_executed)"
    Write-Host "Trades Rejected: $($status.total_trades_rejected)"
    Write-Host "Open Positions: $($status.trading_engine.open_positions)"
    Write-Host "Daily P&L: $([math]::Round($status.trading_engine.daily_pnl, 2))"
    Write-Host "Trades Today: $($status.trading_engine.trades_today)"

    Write-Host "`n=== RISK LIMITS ===" -ForegroundColor Cyan
    Write-Host "Daily Loss Limit: $($status.trading_engine.daily_loss_limit)"
    Write-Host "Account Size: $($status.trading_engine.account_size)"

    Write-Host "`nPress Ctrl+C to stop monitoring" -ForegroundColor Gray
    Start-Sleep -Seconds 5
}
```

Run it:
```powershell
.\watch_bot.ps1
```

---

### **Method 2: Check Logs**

The bot writes logs to the console where the server is running. Look for:
- `✓ Signal generated` - Found a trading opportunity
- `✓ Trade executed` - Placed a trade
- `✗ Trade rejected` - Trade didn't meet risk criteria
- `⚠ Risk limit` - Hit a risk limit

---

## 📋 **WHAT THE BOT DOES AUTOMATICALLY**

### **Every 30-60 seconds, the bot:**

1. **Analyzes Market Data**
   - Gets latest prices for watchlist (AAPL, MSFT, GOOGL, TSLA, NVDA)
   - Calculates technical indicators (VWAP, EMAs, etc.)
   - Checks volume and volatility

2. **Generates Predictions**
   - Uses AI models to predict price movement
   - Analyzes patterns (bull flags, breakouts, etc.)
   - Gets Claude AI market commentary
   - Calculates confidence scores

3. **Evaluates Trade Opportunities**
   - If prediction confidence > 60%
   - Checks if pattern matches strategy
   - Validates risk/reward ratio (must be 2:1 or better)
   - Checks position limits

4. **Risk Validation**
   - Calculates position size based on stop loss
   - Verifies doesn't exceed $50 max risk per trade
   - Checks daily loss limit ($500)
   - Ensures not over-exposed

5. **Trade Execution** (if all checks pass)
   - Places order with IBKR
   - Sets stop loss automatically
   - Sets profit target
   - Logs trade details

6. **Position Monitoring**
   - Tracks open positions
   - Adjusts stop loss if needed
   - Checks for exit signals
   - Monitors P&L

---

## 🎯 **UNDERSTANDING THE NUMBERS**

### **Signals vs Trades**

**Signals Generated:** Opportunities the bot identified
- Example: Bot sees AAPL forming a bull flag

**Trades Executed:** Actual trades placed
- Example: Bot bought 100 shares AAPL at $268

**Trades Rejected:** Opportunities that didn't pass risk checks
- Example: TSLA signal rejected because risk was $100 (max is $50)

**Why rejection is GOOD:** The bot is protecting your account!

---

### **P&L (Profit & Loss)**

**Daily P&L:** How much you made/lost today
- Positive = Profit (e.g., +$125)
- Negative = Loss (e.g., -$50)
- Zero = Break even

**Unrealized P&L:** Profit/loss on open positions (not closed yet)
- Example: Holding AAPL, currently up $25 (not sold yet)

**Realized P&L:** Profit/loss on closed positions
- Example: Sold TSLA for $100 profit

---

### **Risk Metrics**

**Max Risk Per Trade:** $50 (configurable)
- Bot will NEVER risk more than this on a single trade

**Daily Loss Limit:** $500 (5% of account)
- If you lose $500 today, bot stops trading for the day

**Weekly Loss Limit:** $3,500 (7% of account)
- If you lose $3,500 this week, bot stops for the week

**Open Positions:** Number of stocks you're currently holding
- Max: 5 positions at once

**Total Exposure:** Total $ invested in trades right now
- Max: $25,000 (50% of account)

---

## 🔧 **TROUBLESHOOTING**

### **Bot not generating signals?**

**Check:**
1. Is bot running?
   ```powershell
   curl http://127.0.0.1:9101/api/bot/status
   ```
   Should show `"running": true`

2. Is trading enabled?
   ```powershell
   curl http://127.0.0.1:9101/api/bot/status
   ```
   Should show `"enabled": true`

3. Are symbols subscribed?
   ```powershell
   curl http://127.0.0.1:9101/health
   ```
   Should show subscriptions > 0

4. Is IBKR connected?
   ```powershell
   curl http://127.0.0.1:9101/health
   ```
   Should show `"ibkr_connected": true`

**Fix:**
If any are false:
```powershell
# Start bot
curl -X POST http://127.0.0.1:9101/api/bot/start

# Enable trading
curl -X POST http://127.0.0.1:9101/api/bot/enable
```

---

### **Bot keeps rejecting trades?**

**This is GOOD!** It means:
- Risk is too high (trade risks more than $50)
- R:R ratio too low (less than 2:1)
- Daily loss limit hit
- Too many positions already

**What to do:**
- Check `total_trades_rejected` in status
- Look at console logs for rejection reasons
- Review risk limits (might be too conservative)

---

### **Bot not executing trades (but generating signals)?**

**Check:**
1. Is trading enabled?
   ```powershell
   curl http://127.0.0.1:9101/api/bot/status
   ```

2. Hit loss limit?
   - Check `daily_pnl` vs `daily_loss_limit`

3. IBKR connected?
   ```powershell
   curl http://127.0.0.1:9101/health
   ```

---

### **How to stop bot if something goes wrong?**

```powershell
# Emergency stop sequence:

# 1. Disable trading (stop new trades)
curl -X POST http://127.0.0.1:9101/api/bot/disable

# 2. Stop bot (stop generating signals)
curl -X POST http://127.0.0.1:9101/api/bot/stop

# 3. Check positions
curl http://127.0.0.1:9101/api/bot/status

# 4. If needed, close all positions
curl -X POST http://127.0.0.1:9101/api/positions/close-all
```

---

## 📱 **QUICK REFERENCE COMMANDS**

Save these for quick access:

```powershell
# === MONITORING ===
# Quick status
curl http://127.0.0.1:9101/api/bot/status | python -m json.tool

# Health check
curl http://127.0.0.1:9101/health

# Open dashboard
start http://127.0.0.1:9101/ui/platform.html

# === CONTROL ===
# Start bot
curl -X POST http://127.0.0.1:9101/api/bot/start

# Enable trading
curl -X POST http://127.0.0.1:9101/api/bot/enable

# Disable trading (pause)
curl -X POST http://127.0.0.1:9101/api/bot/disable

# Stop bot
curl -X POST http://127.0.0.1:9101/api/bot/stop

# === MARKET DATA ===
# Get AAPL price
curl http://127.0.0.1:9101/api/market-data/AAPL

# Claude analysis on SPY
curl http://127.0.0.1:9101/api/claude/analyze-with-data/SPY
```

---

## 🎓 **WHAT TO EXPECT**

### **First Hour:**
- Bot will analyze market
- Generate 5-20 signals
- Most will be rejected (normal!)
- May execute 1-3 trades

### **First Day:**
- 20-50 signals generated
- 5-10 trades executed
- Win rate: 40-50% (normal for day 1)
- P&L: -$100 to +$200 (wide range is normal)

### **First Week:**
- 100-300 signals
- 30-60 trades
- Win rate should stabilize: 50-55%
- P&L should trend positive

### **What's Normal:**
- ✅ More rejections than executions (shows discipline)
- ✅ Small losses on some trades (stop losses working)
- ✅ Win rate around 50-55%
- ✅ Occasional days with no trades (no good setups)

### **What's Concerning:**
- ⚠️ Win rate < 40% (after 50+ trades)
- ⚠️ Daily losses approaching $500 frequently
- ⚠️ Bot executing trades with R:R < 2:1
- ⚠️ Constant connection drops

---

## 📊 **PERFORMANCE TRACKING**

### **Keep a Daily Log:**

Create `daily_log.txt` and track:
```
Date: 2025-11-18
Signals: 23
Trades Executed: 5
Trades Rejected: 18
Win Rate: 60% (3 wins, 2 losses)
Daily P&L: +$125
Largest Win: +$75 (AAPL)
Largest Loss: -$30 (TSLA)
Notes: Bot rejected NVDA trade (risk too high) - good!
```

---

## 🎯 **GOALS TO AIM FOR**

### **Short Term (1 week):**
- 50+ trades executed
- Win rate ≥ 45%
- No daily loss limit hits
- Bot stable (no crashes)

### **Medium Term (1 month):**
- 200+ trades
- Win rate ≥ 50%
- Profit factor ≥ 1.5
- Consistent profitability

### **Long Term (3+ months):**
- 500+ trades
- Win rate ≥ 55%
- Sharpe ratio > 1.0
- Ready to consider live trading

---

## 🔐 **SAFETY REMINDERS**

### **ALWAYS:**
- ✅ Use paper trading first (2+ weeks minimum)
- ✅ Monitor daily for first week
- ✅ Check risk limits are working
- ✅ Keep daily logs
- ✅ Stop bot if something seems wrong

### **NEVER:**
- ❌ Override risk limits
- ❌ Jump to live trading too soon
- ❌ Ignore consecutive losses
- ❌ Let bot run without monitoring (first week)
- ❌ Trade with money you can't afford to lose

---

## 📞 **GETTING HELP**

### **If bot isn't working:**

1. **Check status:**
   ```powershell
   curl http://127.0.0.1:9101/api/bot/status | python -m json.tool
   curl http://127.0.0.1:9101/health | python -m json.tool
   ```

2. **Check server logs:** Look at console where server is running

3. **Restart sequence:**
   ```powershell
   # Stop bot
   curl -X POST http://127.0.0.1:9101/api/bot/stop

   # Wait 5 seconds
   Start-Sleep -Seconds 5

   # Start bot
   curl -X POST http://127.0.0.1:9101/api/bot/start

   # Enable trading
   curl -X POST http://127.0.0.1:9101/api/bot/enable
   ```

4. **If IBKR disconnected:**
   - Check TWS is running
   - Restart TWS
   - Reinitialize bot:
     ```powershell
     curl -X POST http://127.0.0.1:9101/api/bot/init
     curl -X POST http://127.0.0.1:9101/api/bot/start
     curl -X POST http://127.0.0.1:9101/api/bot/enable
     ```

---

## 🎉 **YOU'RE ALL SET!**

**Your bot is:**
- ✅ Running
- ✅ Trading enabled
- ✅ Risk limits active
- ✅ Ready to trade

**To monitor:**
1. Run: `curl http://127.0.0.1:9101/api/bot/status`
2. Or open: http://127.0.0.1:9101/ui/platform.html
3. Check every 15-30 minutes

**To stop:**
```powershell
curl -X POST http://127.0.0.1:9101/api/bot/disable
curl -X POST http://127.0.0.1:9101/api/bot/stop
```

---

**Happy Trading! 🚀📈**

*Remember: Paper trading first, monitor closely, respect risk limits!*
