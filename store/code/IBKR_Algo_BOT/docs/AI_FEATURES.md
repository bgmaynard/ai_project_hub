# 🧠 IBKR_Algo_BOT - Claude AI Enhanced Edition

A modular, AI-powered trading assistant for Interactive Brokers with Claude integration for advanced analysis and decision support.

## 🎯 Core Features

### Current Implementation ✅
- 🔮 AI prediction & strategy routing
- 🤖 Auto-trading with IBKR via ib_insync
- 🧪 Backtesting and market data analysis
- 📊 Frontend UI dashboard (via TradingView)
- ⚙️ FastAPI-based backend service

### Claude AI Enhancements 🚀
- 📈 **Real-time Market Analysis** - Natural language market commentary
- 🎯 **Strategy Validation** - AI review of trading signals before execution
- 📊 **Risk Assessment** - Automated portfolio risk analysis with explanations
- 📝 **Trade Journaling** - AI-powered trade performance analysis
- 🔍 **News Sentiment Analysis** - Web search + Claude analysis for market events
- 💡 **Strategy Suggestions** - AI-generated trading ideas based on market conditions

---

## 📁 Enhanced Project Structure

```
IBKR_Algo_BOT/
├── ui/                          # Trading UI (HTML)
│   └── platform.html
├── server/                      # FastAPI routers
│   ├── claude_integration/      # NEW: Claude AI modules
│   │   ├── market_analyst.py
│   │   ├── risk_manager.py
│   │   ├── strategy_validator.py
│   │   └── trade_journal.py
│   └── existing routers...
├── bridge/                      # IBKR adapters
├── dashboard_api.py            # Main API entry point
├── .env                        # Environment variables
└── _install_backups/           # Auto-backups
```

---

## ⚙️ Quick Start Guide (For Non-Programmers)

### Step 1: Install Python
1. Download Python 3.11 from python.org
2. During installation, check "Add Python to PATH"
3. Verify: Open Command Prompt and type `python --version`

### Step 2: Set Up Project
```bash
# Open Command Prompt in the project folder
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Configure IBKR Connection
Create a file named `.env` with:
```
TWS_HOST=127.0.0.1
TWS_PORT=3333
TWS_CLIENT_ID=1
```

### Step 4: Start Trading System
```bash
# Activate virtual environment (if not already active)
venv\Scripts\activate

# Start the server
python dashboard_api.py
```

### Step 5: Access Dashboard
Open browser to: `http://127.0.0.1:8000/ui/platform.html`

---

## 🤖 Claude AI Integration Examples

### 1. Market Analysis Endpoint
**What it does:** Get AI-powered market commentary and analysis

```python
# Example: GET /api/claude/analyze-market?symbols=AAPL,MSFT,GOOGL
# Returns: Natural language analysis of current market conditions
```

**Use Case:** Before opening positions, ask Claude to analyze market sentiment and technical conditions.

### 2. Trade Validation
**What it does:** Validate trading signals before execution

```python
# Example: POST /api/claude/validate-trade
# Body: {"symbol": "AAPL", "action": "BUY", "quantity": 100, "reason": "breakout"}
# Returns: AI assessment of trade quality and risk factors
```

**Use Case:** Prevent emotional trading - get objective AI review of each trade.

### 3. Portfolio Risk Check
**What it does:** Analyze current portfolio exposure and suggest adjustments

```python
# Example: GET /api/claude/portfolio-risk
# Returns: Risk analysis, concentration warnings, diversification suggestions
```

**Use Case:** Daily portfolio health check with actionable recommendations.

### 4. News Impact Analysis
**What it does:** Search news and analyze impact on your positions

```python
# Example: GET /api/claude/news-impact?symbol=AAPL
# Returns: Recent news + AI analysis of potential market impact
```

**Use Case:** Stay informed on breaking news affecting your holdings.

---

## 📊 Claude Integration Architecture

```
User Request
    ↓
FastAPI Endpoint
    ↓
Claude AI Module
    ↓
┌─────────────────┬──────────────────┬─────────────────┐
│                 │                  │                 │
│  Web Search     │  Market Data     │  Portfolio Data │
│  (Real-time)    │  (IBKR API)      │  (Database)     │
│                 │                  │                 │
└─────────────────┴──────────────────┴─────────────────┘
    ↓
AI Analysis & Response
    ↓
User Interface / API Response
```

---

## 🛠️ Dependencies

### Core Trading Stack
```
ib_insync>=0.9.86
fastapi>=0.104.0
uvicorn>=0.24.0
python-dotenv>=1.0.0
pandas>=2.0.0
numpy>=1.24.0
```

### Claude Integration (New)
```
anthropic>=0.7.0
requests>=2.31.0
pydantic>=2.0.0
```

---

## 🚀 Implementation Roadmap

### Phase 1: Foundation ✅ DONE
- [x] IBKR API connection
- [x] Basic trading functionality
- [x] Dashboard UI
- [x] Port auto-detection

### Phase 2: Claude Integration 🔄 IN PROGRESS
- [ ] Market analysis module
- [ ] Trade validation endpoint
- [ ] Risk management system
- [ ] News sentiment analysis

### Phase 3: Advanced Features 📋 PLANNED
- [ ] Automated trade journaling
- [ ] Strategy backtesting with AI commentary
- [ ] Real-time alert system
- [ ] Performance attribution analysis

### Phase 4: Production Ready 🎯 FUTURE
- [ ] Multi-account support
- [ ] Advanced risk controls
- [ ] Strategy optimization
- [ ] Mobile app integration

---

## 💡 Example Use Cases for Investors

### Morning Routine
```bash
1. Start system: python dashboard_api.py
2. Check overnight news: /api/claude/news-summary
3. Portfolio health: /api/claude/portfolio-risk
4. Market outlook: /api/claude/analyze-market
```

### During Trading Hours
```bash
1. New signal appears → validate with: /api/claude/validate-trade
2. Breaking news alert → analyze with: /api/claude/news-impact
3. Profit target hit → journal trade: /api/claude/journal-trade
```

### End of Day
```bash
1. Performance review: /api/claude/daily-summary
2. Tomorrow's watchlist: /api/claude/suggest-opportunities
3. Risk check: /api/claude/portfolio-risk
```

---

## ⚠️ Important Notes for Non-Programmers

### IBKR TWS Setup
1. Open TWS (Trader Workstation)
2. Go to: File → Global Configuration → API → Settings
3. Enable these options:
   - ✅ Enable ActiveX and Socket Clients
   - ✅ Read-Only API (safer for testing)
   - ✅ Socket Port: 3333
4. Add trusted IP: 127.0.0.1

### Testing Before Real Money
```python
# Always test with paper trading account first
# In TWS: File → Switch to Paper Trading
```

### Safety Features
- All Claude suggestions are advisory only
- System requires manual confirmation for trades
- Position size limits enforced
- Daily loss limits configurable

---

## 🔒 Security Best Practices

1. **Never share your `.env` file** - Contains sensitive credentials
2. **Use paper trading first** - Test all features before live trading
3. **Review AI suggestions** - Claude assists, you decide
4. **Set risk limits** - Configure maximum position sizes
5. **Keep logs** - All trades are journaled for review

---

## 📚 Learning Resources

### For Beginners
- **Python Basics**: python.org/about/gettingstarted
- **IBKR API Guide**: interactivebrokers.com/api
- **Trading Concepts**: investopedia.com

### For Intermediate
- **FastAPI Docs**: fastapi.tiangolo.com
- **Algorithmic Trading**: quantstart.com
- **Risk Management**: tradingsim.com

---

## 🆘 Troubleshooting

### "Connection Refused" Error
**Problem:** Can't connect to IBKR TWS
**Solution:**
1. Verify TWS is running
2. Check API settings (see IBKR TWS Setup above)
3. Test connection: `Test-NetConnection 127.0.0.1 -Port 3333`

### "Module Not Found" Error
**Problem:** Missing Python packages
**Solution:**
```bash
pip install -r requirements.txt
# or for specific package:
pip install ib_insync
```

### "Claude API Error"
**Problem:** AI features not working
**Solution:**
1. Check you're using Claude.ai interface correctly
2. Verify internet connection for news features
3. Review error logs in console

---

## 📞 Getting Help

### Documentation Structure
1. **Quick Start** - Get up and running (you are here!)
2. **API Reference** - Endpoint documentation
3. **Strategy Guide** - Trading strategy examples
4. **Troubleshooting** - Common issues and solutions

### Support Channels
- Check `docs/` folder for detailed guides
- Review error logs in `logs/` folder
- Test individual components with test scripts

---

## 🎓 Next Steps

### For New Users
1. ✅ Complete Quick Start guide above
2. ✅ Run system in paper trading mode
3. ✅ Test each Claude AI feature
4. ✅ Review one week of AI suggestions
5. ✅ Gradually enable live trading features

### For Developers
1. Review `server/claude_integration/` modules
2. Implement custom analysis functions
3. Add new trading strategies
4. Extend API with custom endpoints

---

## 📈 Success Metrics

Track these KPIs to measure system effectiveness:

- **AI Accuracy**: % of profitable AI-suggested trades
- **Risk Management**: Maximum drawdown reduction
- **Decision Quality**: Trades validated vs. rejected
- **Time Saved**: Hours saved on market analysis
- **Learning Curve**: Confidence improvement over time

---

## 🔗 Useful Links

- **IBKR API Docs**: https://interactivebrokers.github.io/tws-api/
- **FastAPI Tutorial**: https://fastapi.tiangolo.com/tutorial/
- **Python for Finance**: https://python-for-finance.com/
- **Algorithmic Trading**: https://www.quantstart.com/

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🙏 Credits

Built with:
- Interactive Brokers API (ib_insync)
- Claude AI by Anthropic
- FastAPI framework
- TradingView charts

**Maintained by:** AI-Assisted Development Team
**Version:** 2.0 (Claude Enhanced)
**Last Updated:** 2025-11-08
