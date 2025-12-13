# AI Auto-Trading Guide
## How to Enable and Test Autonomous Trading

---

## What You Discovered

You said: **"I can trade manually but have never had the ai fire to test an order from it"**

**ROOT CAUSE**: The AI auto-trader exists but is **DISABLED by default** for safety.

---

## How AI Auto-Trading Works

### System Flow:
```
1. AI Predictor generates signal →
2. Auto-Trader checks thresholds →
3. Risk management validates →
4. Order placed automatically →
5. Trade logged to file
```

### Decision Process (ai/auto_trader.py):
```python
✓ Is auto-trading enabled?
✓ Under daily trade limit? (default: 5/day)
✓ Under daily loss limit? (default: $500)
✓ AI confidence ≥ threshold? (default: 65%)
✓ Probability ≥ threshold? (default: 60%)
✓ Is signal BULLISH? (only buys, no shorts)
→ If ALL true: Place BUY order
```

---

## Configuration Added to .env

I've added these settings to your `.env` file:

```bash
# Auto-Trading Settings (TEST MODE - CONSERVATIVE)
AUTO_TRADE_ENABLED=false           # Set to "true" to enable
AUTO_TRADE_MIN_CONFIDENCE=0.40     # Lower for testing (default: 0.65)
AUTO_TRADE_MIN_PROB=0.40           # Lower for testing (default: 0.60)
AUTO_TRADE_MAX_SHARES=1            # MINIMAL for testing (default: 10)
AUTO_TRADE_MAX_DAILY=2             # Max 2 test trades/day (default: 5)
AUTO_TRADE_MAX_LOSS=50.0           # Stop at $50 loss (default: 500.0)
AUTO_TRADE_STOP_LOSS=0.05          # 5% stop loss
```

**These are CONSERVATIVE test settings!**
- Only 1 share per trade
- Max 2 trades per day
- Stops at $50 total loss
- Lower confidence thresholds (40% instead of 65%)

---

## How to Test AI Auto-Trading

### Option 1: Controlled Test (RECOMMENDED)

Run the test script I created:

```powershell
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT
python tests\test_ai_auto_trading.py
```

**What it does:**
1. Temporarily enables auto-trading
2. Tests SPY and QQQ for trading signals
3. Shows you each trade decision
4. **ASKS FOR CONFIRMATION before placing real orders**
5. Logs all trades to `logs/trades.json`

**You'll see:**
```
[TESTING SPY]
--------------------------------------------------
Step 1: Evaluating SPY...
  Should Trade: True
  Reason: BULLISH: 67.00% confidence, STRONG_BUY
  Prediction: 1 (UP)
  Confidence: 67.00%
  Prob Up: 72.00%
  Signal: STRONG_BUY

  SPY PASSED ALL CHECKS!
  Action: BUY
  Quantity: 1 shares

  [!] This will place a REAL order!
  [!] BUY 1 shares of SPY

  Execute this trade? (type 'YES' to proceed):
```

**Safety Features:**
- Asks for confirmation before each trade
- Only trades 1 share
- Logs everything
- You can cancel anytime

---

### Option 2: Enable Continuous Auto-Trading

**WARNING: This enables REAL autonomous trading!**

1. **Edit .env:**
   ```bash
   AUTO_TRADE_ENABLED=true    # Change from false to true
   ```

2. **Restart bot:**
   ```powershell
   .\FORCE_RESTART_BOT.ps1
   ```

3. **Bot will now:**
   - Check signals automatically (every minute/interval)
   - Place orders when conditions met
   - Log all trades to `logs/trades.json`
   - Stop after 2 trades or $50 loss (per test settings)

4. **Monitor activity:**
   ```powershell
   Get-Content logs\trades.json | ConvertFrom-Json
   ```

---

## Why AI Never Traded Before

Looking at your earlier test, the AI generated:
- **Signal**: STRONG_BEARISH
- **Confidence**: 44.95%

**This wouldn't trigger a trade because:**
1. ✗ AUTO_TRADE_ENABLED was false (disabled)
2. ✗ Signal was BEARISH (only BULLISH signals buy)
3. ✗ Confidence 44.95% < 65% threshold

**For AI to trade, you need:**
- ✓ Enabled in .env
- ✓ BULLISH signal (predicts price UP)
- ✓ Confidence ≥ 40% (test) or 65% (production)
- ✓ Probability ≥ 40% (test) or 60% (production)

---

## Trade Logging

All AI trades are logged to: `logs/trades.json`

**Example log entry:**
```json
{
  "timestamp": "2025-11-19T18:30:00.000Z",
  "symbol": "SPY",
  "action": "BUY",
  "quantity": 1,
  "limit_price": 450.25,
  "current_price": 450.00,
  "order_id": "12345",
  "prediction": {
    "confidence": 0.67,
    "prob_up": 0.72,
    "signal_detail": "STRONG_BUY"
  },
  "reason": "BULLISH: 67.00% confidence, STRONG_BUY",
  "pnl": 0
}
```

---

## Safety Features

### Built-in Protections:
1. **Daily Trade Limit**: Max 2 trades/day (test setting)
2. **Daily Loss Limit**: Stops at $50 loss
3. **Position Size Limit**: Only 1 share per trade
4. **Stop Loss**: 5% automatic stop loss
5. **Confidence Threshold**: Won't trade low-confidence signals
6. **Disabled by Default**: Must explicitly enable

### Manual Controls:
- **Emergency stop**: Set `AUTO_TRADE_ENABLED=false` in .env
- **Cancel orders**: Use TWS/IB Gateway or dashboard UI
- **Review trades**: Check `logs/trades.json`
- **Restart bot**: Reloads configuration from .env

---

## Testing Checklist

Before enabling continuous auto-trading:

- [ ] Run controlled test: `python tests\test_ai_auto_trading.py`
- [ ] Verify AI generates signals correctly
- [ ] Test with 1 share position size
- [ ] Monitor trades in `logs/trades.json`
- [ ] Check orders appear in TWS/IB Gateway
- [ ] Verify you can cancel test orders
- [ ] Understand all configuration settings
- [ ] Set appropriate thresholds for your risk tolerance

---

## Production Settings (After Testing)

Once comfortable, adjust thresholds in `.env`:

```bash
AUTO_TRADE_ENABLED=true            # Enable
AUTO_TRADE_MIN_CONFIDENCE=0.70     # Higher confidence (70%)
AUTO_TRADE_MIN_PROB=0.65           # Higher probability (65%)
AUTO_TRADE_MAX_SHARES=10           # More shares per trade
AUTO_TRADE_MAX_DAILY=5             # More trades per day
AUTO_TRADE_MAX_LOSS=500.0          # Higher loss tolerance
AUTO_TRADE_STOP_LOSS=0.05          # Keep 5% stop loss
```

**Restart bot after changes:**
```powershell
.\FORCE_RESTART_BOT.ps1
```

---

## Troubleshooting

### "AI never places orders"
**Check:**
1. `AUTO_TRADE_ENABLED=true` in .env
2. Bot restarted after changing .env
3. AI generating BULLISH signals (not BEARISH)
4. Confidence/probability above thresholds
5. Not over daily trade/loss limits
6. Check `logs/trades.json` for "reason" messages

### "How do I know if AI is checking?"
Add logging to see AI checks:
```bash
# In dashboard_api.py or auto_trader.py
print(f"[AUTO-TRADER] Checking {symbol}...")
```

### "AI placed too many trades"
Adjust limits in .env:
```bash
AUTO_TRADE_MAX_DAILY=1             # Only 1 trade per day
AUTO_TRADE_MAX_LOSS=25.0           # Lower loss limit
```

---

## Quick Start Commands

### Test AI Auto-Trading (Safe):
```powershell
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT
python tests\test_ai_auto_trading.py
```

### Enable Continuous Auto-Trading:
```powershell
# 1. Edit .env
notepad C:\ai_project_hub\store\code\IBKR_Algo_BOT\.env
# Change: AUTO_TRADE_ENABLED=true

# 2. Restart bot
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
.\FORCE_RESTART_BOT.ps1
```

### Monitor Trades:
```powershell
# View trade log
type C:\ai_project_hub\store\code\IBKR_Algo_BOT\logs\trades.json

# Watch log in real-time (if using PowerShell)
Get-Content -Path logs\trades.json -Wait
```

### Disable Auto-Trading:
```powershell
# Edit .env
notepad C:\ai_project_hub\store\code\IBKR_Algo_BOT\.env
# Change: AUTO_TRADE_ENABLED=false

# Restart bot
.\FORCE_RESTART_BOT.ps1
```

---

## Summary

**Problem**: AI never fired orders because it was disabled
**Solution**: Configure and enable auto-trading in .env
**Test**: Run `test_ai_auto_trading.py` for controlled testing
**Enable**: Set `AUTO_TRADE_ENABLED=true` for continuous trading

**You now have:**
- ✅ AI signal generation (working)
- ✅ Auto-trader configuration (added to .env)
- ✅ Comprehensive test script (tests/test_ai_auto_trading.py)
- ✅ Trade logging (logs/trades.json)
- ✅ Safety controls (limits, stops, thresholds)

Ready to test! 🚀
