# AI Trading Bot - Quick Commands

## 🚀 Add Stock to Watchlist (INSTANT)

When a momentum stock pops up in your scanner:

### Option 1: PowerShell Script (Recommended - Shows Everything)
```powershell
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
.\ADD_STOCK.ps1 SYMBOL
```

**Example:**
```powershell
.\ADD_STOCK.ps1 TSLA
```

**What it does:**
1. ✅ Adds stock to watchlist
2. ✅ Gets current price & % change
3. ✅ Runs immediate AI evaluation
4. ✅ Shows if bot will trade it
5. ✅ Shows updated watchlist

### Option 2: Quick One-Liner (Fast)
```powershell
curl -X POST http://127.0.0.1:9101/api/worklist/add/SYMBOL
```

**Example:**
```powershell
curl -X POST http://127.0.0.1:9101/api/worklist/add/TSLA
```

---

## 📊 Check What's Happening

### See Current Watchlist
```powershell
curl -s http://127.0.0.1:9101/api/worklist | python -m json.tool
```

### Check AI Status
```powershell
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
.\CHECK_AI_STATUS.ps1
```

### Watch Scanner Live (Real-Time Activity)
```powershell
# Watch scanner output in real-time
Get-Content C:\ai_project_hub\store\code\IBKR_Algo_BOT\logs\dashboard.log -Tail 50 -Wait | Select-String "AUTO-SCANNER"
```

### Check Specific Stock AI Evaluation
```powershell
curl http://127.0.0.1:9101/api/ai/evaluate/SYMBOL
```

**Example:**
```powershell
curl http://127.0.0.1:9101/api/ai/evaluate/MNDR
```

---

## 🗑️ Remove Stock from Watchlist

```powershell
curl -X DELETE http://127.0.0.1:9101/api/worklist/remove/SYMBOL
```

**Example:**
```powershell
curl -X DELETE http://127.0.0.1:9101/api/worklist/remove/AAPL
```

---

## 🎯 Quick Trading Workflow

### Morning Routine:
```powershell
# 1. Start the bot
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
.\START_TRADING_BOT.ps1

# 2. Check status
.\CHECK_AI_STATUS.ps1

# 3. Open dashboard
start http://127.0.0.1:9101/ui/complete_platform.html
```

### When Stock Pops Up in Scanner:
```powershell
# Quick add
.\ADD_STOCK.ps1 SYMBOL

# Examples:
.\ADD_STOCK.ps1 MNDR
.\ADD_STOCK.ps1 INM
.\ADD_STOCK.ps1 PBM
```

### Monitor Activity:
```powershell
# Watch what AI is doing
Get-Content C:\ai_project_hub\store\code\IBKR_Algo_BOT\logs\dashboard.log -Tail 20 -Wait | Select-String "AUTO-SCANNER"
```

---

## 🛠️ Bot Control

### Start Bot
```powershell
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
.\START_TRADING_BOT.ps1
```

### Restart Bot
```powershell
.\FORCE_RESTART_BOT.ps1
```

### Check if Bot is Running
```powershell
Get-Process python
```

### View Dashboard
```
http://127.0.0.1:9101/ui/complete_platform.html
```

---

## ⚙️ Configuration

### Check Current Settings
```powershell
type C:\ai_project_hub\store\code\IBKR_Algo_BOT\.env
```

### Edit Settings
```powershell
notepad C:\ai_project_hub\store\code\IBKR_Algo_BOT\.env
```

**Current Settings:**
- Auto-Trading: **ENABLED**
- Min Confidence: **30%** (was 40%)
- Min Probability: **30%** (was 40%)
- Max Shares per Trade: **1 share**
- Max Trades per Day: **2 trades**
- Max Daily Loss: **$50**
- Scanner Interval: **60 seconds**

---

## 📈 Understanding AI Decisions

When you add a stock, the AI evaluates:

1. **Prediction**: UP (bullish) or DOWN (bearish)
2. **Confidence**: How sure the AI is (needs > 30%)
3. **Probability**: Likelihood of move (needs > 30%)
4. **Trade Decision**: Will trade if ALL conditions met:
   - ✅ Prediction is UP (bullish)
   - ✅ Confidence ≥ 30%
   - ✅ Probability ≥ 30%
   - ✅ Under daily trade limit (2)
   - ✅ Under daily loss limit ($50)

**Example Outputs:**

```
✓ WILL TRADE
  - Prediction: UP
  - Confidence: 42%
  - Probability: 38%
  - Decision: TRADE APPROVED

✗ NO TRADE - BEARISH signal
  - Prediction: DOWN
  - AI predicts price will drop

✗ NO TRADE - Confidence too low (25% < 30%)
  - Confidence not high enough
```

---

## 💡 Pro Tips

1. **Ross Cameron Scanner Integration:**
   - When momentum stocks pop up in your HOD/momentum scanner
   - Immediately run: `.\ADD_STOCK.ps1 SYMBOL`
   - AI evaluates within 2 seconds
   - If approved, bot trades within 60 seconds

2. **Breaking News Stocks:**
   - Add immediately when you see breaking news
   - AI evaluates momentum and trend
   - Fast execution (< 60 seconds)

3. **Multiple Stocks:**
   - You can add 3-5 stocks quickly:
     ```powershell
     .\ADD_STOCK.ps1 MNDR
     .\ADD_STOCK.ps1 INM
     .\ADD_STOCK.ps1 PBM
     ```

4. **Monitor Scanner:**
   - Leave PowerShell window open with:
     ```powershell
     Get-Content C:\ai_project_hub\store\code\IBKR_Algo_BOT\logs\dashboard.log -Tail 20 -Wait | Select-String "AUTO-SCANNER"
     ```
   - You'll see every evaluation in real-time

---

## 🆘 Troubleshooting

### Bot Not Responding
```powershell
# Check if running
Get-Process python

# Restart
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
.\FORCE_RESTART_BOT.ps1
```

### Stock Not Added
```powershell
# Check if symbol is valid
curl http://127.0.0.1:9101/api/price/SYMBOL

# Check API status
curl http://127.0.0.1:9101/health
```

### AI Not Trading
```powershell
# Check specific stock evaluation
curl http://127.0.0.1:9101/api/ai/evaluate/SYMBOL

# Check configuration
type C:\ai_project_hub\store\code\IBKR_Algo_BOT\.env
```

---

**All scripts located in:**
`C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2\`

**Dashboard:**
http://127.0.0.1:9101/ui/complete_platform.html
