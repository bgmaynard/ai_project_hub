# 🎯 QUICK START GUIDE FOR INVESTORS
## Claude AI Trading Assistant - Non-Programmer Edition

---

## 📋 What You're Building

A smart trading assistant that:
- ✅ Analyzes market conditions before you trade
- ✅ Validates every trade to prevent costly mistakes
- ✅ Monitors your portfolio risk automatically
- ✅ Gives you AI-powered insights in plain English

**No coding experience needed!** Just follow the steps.

---

## 🛠️ Part 1: One-Time Setup (30 minutes)

### Step 1: Install Python

1. Go to https://www.python.org/downloads/
2. Download Python 3.11 (big yellow button)
3. **IMPORTANT**: Check "Add Python to PATH" during installation
4. Click "Install Now"
5. Verify: Open Command Prompt and type:
   ```
   python --version
   ```
   Should show: `Python 3.11.x`

### Step 2: Download Your Files

1. Create a folder: `C:\Trading\IBKR_AI`
2. Save these files in that folder:
   - `market_analyst.py`
   - `trade_validator.py`
   - `claude_api_integration.py`

### Step 3: Install Required Libraries

1. Open Command Prompt
2. Navigate to your folder:
   ```
   cd C:\Trading\IBKR_AI
   ```
3. Create virtual environment:
   ```
   python -m venv venv
   ```
4. Activate it:
   ```
   venv\Scripts\activate
   ```
   You'll see `(venv)` at the start of your command line
5. Install libraries:
   ```
   pip install fastapi uvicorn ib_insync python-dotenv
   ```

### Step 4: Set Up Interactive Brokers

1. Open TWS (Trader Workstation)
2. Go to: **File → Global Configuration → API → Settings**
3. Enable these checkboxes:
   - ☑️ Enable ActiveX and Socket Clients
   - ☑️ Allow connections from localhost only
   - ☑️ Read-Only API (for safety during testing)
4. Set Socket port: `3333`
5. Add trusted IP: `127.0.0.1`
6. Click OK and restart TWS

---

## 🚀 Part 2: Daily Usage (5 minutes)

### Starting the System

1. Open TWS and log in
2. Open Command Prompt
3. Go to your folder:
   ```
   cd C:\Trading\IBKR_AI
   venv\Scripts\activate
   ```
4. Start the AI assistant:
   ```
   python claude_api_integration.py
   ```
5. You'll see:
   ```
   🤖 CLAUDE AI TRADING ASSISTANT - STARTING UP
   ✅ AI Modules Loaded
   📍 Server available at: http://localhost:8000
   ```

### Using the Web Interface

Open your browser to: `http://localhost:8000/docs`

You'll see an interactive API page. Here's what each button does:

---

## 🎮 Features Explained (In Plain English)

### 1. 📊 Analyze Market
**What it does:** Gives you AI analysis of any stocks you're watching

**How to use:**
1. Click on `POST /api/claude/analyze-market`
2. Click "Try it out"
3. Enter:
   ```json
   {
     "symbols": ["AAPL", "MSFT", "GOOGL"],
     "include_news": false
   }
   ```
4. Click "Execute"
5. Read the analysis below

**When to use:** Every morning before trading, or before entering a position

---

### 2. ✅ Validate Trade
**What it does:** Checks if your trade idea is safe BEFORE you place it

**How to use:**
1. Click on `POST /api/claude/validate-trade`
2. Click "Try it out"
3. Enter your trade details:
   ```json
   {
     "symbol": "AAPL",
     "action": "BUY",
     "quantity": 100,
     "entry_price": 150.00,
     "stop_loss": 145.00,
     "take_profit": 160.00,
     "reason": "Breaking above resistance level"
   }
   ```
4. Click "Execute"
5. Read the validation result

**Response meanings:**
- ✅ **APPROVED**: Safe to trade
- ⚠️ **REVIEW**: Check the warnings
- ❌ **REJECTED**: Don't make this trade

**When to use:** EVERY TIME before placing a trade

---

### 3. 🛡️ Portfolio Risk Check
**What it does:** Shows if you're taking too much risk

**How to use:**
1. Click on `POST /api/claude/portfolio-risk`
2. Click "Try it out"
3. Enter your current positions:
   ```json
   {
     "positions": [
       {"symbol": "AAPL", "quantity": 100, "avg_price": 150},
       {"symbol": "MSFT", "quantity": 50, "avg_price": 300}
     ],
     "portfolio_value": 50000
   }
   ```
4. Click "Execute"
5. Review risk warnings and recommendations

**When to use:** Once daily, or before adding new positions

---

### 4. 📰 Daily Summary
**What it does:** Morning briefing of what's important today

**How to use:**
1. Click on `GET /api/claude/daily-summary`
2. Click "Try it out"
3. Click "Execute"
4. Read your personalized summary

**When to use:** First thing each trading day

---

### 5. 🔍 Quick Stock Check
**What it does:** Fast lookup on any stock

**How to use:**
1. Click on `GET /api/claude/quick-check/{symbol}`
2. Replace `{symbol}` with your stock (e.g., `AAPL`)
3. Click "Execute"
4. Get instant analysis

**When to use:** When you hear about a stock and want quick info

---

## 📝 Typical Daily Workflow

### Morning (Before Market Open)
```
1. Start TWS
2. Start AI Assistant
3. Get Daily Summary
4. Check Portfolio Risk
5. Analyze key stocks
```

### During Trading
```
1. See a trading opportunity
2. Validate the trade FIRST
3. If APPROVED → Place order in TWS
4. If REJECTED → Look for better entry
5. If REVIEW → Check warnings carefully
```

### End of Day
```
1. Run Portfolio Risk check
2. Review any new validation stats
3. Close AI Assistant (Ctrl+C in Command Prompt)
4. Log out of TWS
```

---

## ⚠️ Important Safety Rules

### DO:
- ✅ Always validate trades before placing them
- ✅ Start with paper trading (TWS demo account)
- ✅ Use stop losses on every trade
- ✅ Keep position sizes small (max 10% per position)
- ✅ Review AI suggestions, don't blindly follow

### DON'T:
- ❌ Override REJECTED trades without very good reason
- ❌ Trade when market is closed (system may not work correctly)
- ❌ Ignore risk warnings
- ❌ Let any single position exceed 20% of portfolio
- ❌ Trade without stop losses

---

## 🔧 Troubleshooting

### "Connection Refused" Error
**Problem:** Can't connect to IBKR

**Solutions:**
1. Is TWS running?
2. Is API enabled in TWS settings?
3. Did you set port to 3333?
4. Try restarting TWS

Test connection:
```
Test-NetConnection 127.0.0.1 -Port 3333
```

---

### "Module Not Found" Error
**Problem:** Python can't find a library

**Solution:**
```
venv\Scripts\activate
pip install fastapi uvicorn ib_insync python-dotenv
```

---

### API Shows Error 500
**Problem:** Something broke in the code

**Solution:**
1. Check the Command Prompt for error details
2. Restart the AI Assistant
3. Verify TWS is connected
4. Make sure you entered valid data in the API

---

### Trade Validation Always Rejects
**Problem:** Settings may be too strict

**Solution:**
Edit the risk settings (ask for help with this)

---

## 📊 Understanding Risk Settings

Default safety limits (can be customized):
- Max 10% of portfolio per position
- Max 2% risk per trade
- Stop loss required on all trades
- Max 5 trades per day
- Minimum 1.5:1 risk/reward ratio

These are CONSERVATIVE settings designed to protect you.

---

## 💡 Pro Tips

1. **Start Small**: Test with 1 share trades first
2. **Paper Trade First**: Use TWS demo account for 2 weeks
3. **Journal Everything**: Save your trade validations
4. **Learn From Rejections**: If AI rejects your trade, understand why
5. **Don't Fight the System**: If validation fails, there's usually a good reason
6. **Review Weekly**: Check your validation statistics every week
7. **Update Gradually**: As you get comfortable, adjust risk settings

---

## 📈 Next Steps After Mastering Basics

1. Connect to real market data feed
2. Add news sentiment analysis
3. Implement automated alerts
4. Create custom trading strategies
5. Add performance tracking dashboard
6. Integrate with tax reporting

---

## 🆘 Getting Help

### Documentation Locations
- API docs: `http://localhost:8000/docs` (when running)
- IBKR API: https://interactivebrokers.github.io/tws-api/
- FastAPI: https://fastapi.tiangolo.com/

### Common Questions

**Q: Can this place trades automatically?**
A: No, by design. It validates and suggests, but YOU decide and execute.

**Q: Does it work with paper trading?**
A: Yes! Recommended for learning.

**Q: What if I close the Command Prompt?**
A: The system stops. You need to keep it running during trading hours.

**Q: Can I use this on mobile?**
A: Not directly, but you can access the API from any browser on your network.

**Q: Is my data safe?**
A: Yes, everything runs locally on your computer. Nothing is sent to external servers.

---

## 🎓 Learning Path

### Week 1: Setup & Learn Interface
- [ ] Install everything
- [ ] Connect to paper trading
- [ ] Test all API endpoints
- [ ] Place 5 paper trades with validation

### Week 2: Build Habits
- [ ] Use daily summary every morning
- [ ] Validate ALL trades before placing
- [ ] Check portfolio risk daily
- [ ] Review validation statistics

### Week 3: Refine Strategy
- [ ] Analyze which validations were correct
- [ ] Adjust risk settings if needed
- [ ] Start tracking trade outcomes
- [ ] Identify patterns in AI suggestions

### Week 4: Go Live (If Comfortable)
- [ ] Switch to live account
- [ ] Start with smallest position sizes
- [ ] Continue using all safety features
- [ ] Gradually increase confidence

---

## 📌 Quick Reference Card

Print this and keep it by your trading desk:

```
===========================================
DAILY TRADING CHECKLIST
===========================================

BEFORE MARKET OPENS:
□ Start TWS
□ Start AI Assistant (python claude_api_integration.py)
□ Check Daily Summary
□ Review Portfolio Risk
□ Analyze watchlist stocks

DURING TRADING:
□ Validate EVERY trade before placing
□ Only execute APPROVED trades
□ Always use stop losses
□ Review risk after each trade

END OF DAY:
□ Final portfolio risk check
□ Save trade log/screenshots
□ Close AI Assistant (Ctrl+C)
□ Close TWS

COMMANDS TO REMEMBER:
Start: cd C:\Trading\IBKR_AI && venv\Scripts\activate && python claude_api_integration.py
Stop: Ctrl+C
API: http://localhost:8000/docs
TWS Port: 3333
```

---

## ✅ You're Ready!

You now have a professional AI trading assistant. Remember:
- **Be patient** - Learning takes time
- **Stay disciplined** - Follow the validation system
- **Start small** - Paper trade first
- **Keep learning** - Markets always teach new lessons

Good luck with your trading! 🚀📈

---

**Version:** 2.0  
**Last Updated:** 2025-11-08  
**Support:** See troubleshooting section above
