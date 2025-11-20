# 🚀 IBKR Algorithmic Trading Bot V2

**Professional Trading Platform with AI-Powered Analysis**

> Full-featured algorithmic trading system integrating Interactive Brokers (IBKR), Claude AI, and machine learning predictions for sophisticated market analysis and automated trading.

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Troubleshooting](#-troubleshooting)
- [Safety & Risk Management](#-safety--risk-management)

---

## ✨ Features

### 🎯 Core Capabilities

- **Real-Time IBKR Integration**
  - Live market data streaming (Level 1 quotes)
  - Level 2 market depth (order book)
  - Time & Sales (tape reading)
  - Order execution (paper & live trading)
  - Portfolio management

- **Claude AI Analysis**
  - Real-time market commentary with live IBKR data
  - Technical analysis and pattern recognition
  - News sentiment integration
  - Trade idea generation
  - Risk assessment

- **Machine Learning Predictions**
  - LightGBM-powered momentum prediction
  - Custom feature engineering (50+ technical indicators)
  - Multi-timeframe analysis
  - Backtesting and validation
  - Prediction logging for performance tracking

- **Pre-Market Scanner**
  - Gap & Go strategy detection
  - News catalyst filtering
  - Volume and liquidity screening
  - Breaking news integration
  - Custom criteria configuration

### 🛠️ Technical Features

- FastAPI backend with WebSocket streaming
- Multi-strategy support (Gap & Go, Warrior Momentum, Bull Flag, Flat Top)
- Professional risk management (3-5-7 strategy: 3% per trade, 5% daily limit, 7% max drawdown)
- TradingView webhook integration
- Modular AI architecture
- Real-time dashboard with multiple windows
- Comprehensive logging and analytics

---

## 🚀 Quick Start

### Prerequisites

- Windows 10/11 (PowerShell 5.1+)
- Python 3.8 or higher
- Interactive Brokers TWS or IB Gateway
- Anthropic API key (Claude AI)
- Active IBKR account (paper trading recommended for testing)

### 1. Deploy the Complete System

**Option A: Automated (Recommended)**
```powershell
# Navigate to project directory
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2

# Run deployment script
.\DEPLOY_COMPLETE_API.ps1

# Follow the prompts
```

**Option B: Manual**
```powershell
# 1. Copy the complete API file
Copy-Item "C:\Users\bgmay\Downloads\dashboard_api_COMPLETE.py" "dashboard_api.py" -Force

# 2. Set API key
$env:ANTHROPIC_API_KEY = "your-api-key-here"

# 3. Start server
python dashboard_api.py
```

### 2. Connect to IBKR

```powershell
# Make sure TWS is running first!

# Connect to paper trading (port 7497)
$connectBody = @{
    host = "127.0.0.1"
    port = 7497
    client_id = 1
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/ibkr/connect" `
    -Method POST `
    -ContentType "application/json" `
    -Body $connectBody
```

### 3. Subscribe to Symbols

```powershell
# Subscribe to AAPL
$subscribeBody = @{
    symbol = "AAPL"
    exchange = "SMART"
    data_types = @("quote")
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/subscribe" `
    -Method POST `
    -ContentType "application/json" `
    -Body $subscribeBody
```

### 4. Open Platform

Navigate to: http://127.0.0.1:9101/ui/platform.html

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FRONTEND LAYER                         │
│  platform.html → Multi-window dashboard with real-time UI   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  API LAYER (Port 9101)                      │
│  FastAPI → REST endpoints + WebSocket streaming             │
└────────────────────────┬────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            │                         │
┌───────────▼──────────┐  ┌──────────▼─────────────┐
│   MARKET DATA BUS    │  │     AI MODULES         │
│  • IBKR streaming    │  │  • Claude API          │
│  • Level 2 depth     │  │  • LightGBM ML         │
│  • Time & Sales      │  │  • Market Analyst      │
│  • Scanner results   │  │  • Trade Validator     │
└──────────────────────┘  │  • Alpha Fusion        │
                          │  • Backtester          │
                          └────────────────────────┘
```

### Directory Structure

```
IBKR_Algo_BOT_V2/
├── dashboard_api.py           # Main FastAPI server
├── requirements.txt           # Python dependencies
├── .env                       # Configuration (API keys)
├── START_TRADING_BOT.ps1     # Quick start script
│
├── ai/                        # AI modules
│   ├── __init__.py
│   ├── claude_api.py         # Claude AI integration
│   ├── ai_predictor.py       # LightGBM predictions
│   ├── market_analyst.py     # Technical analysis
│   ├── trade_validator.py    # Risk validation
│   ├── alpha_fusion.py       # Multi-model ensemble
│   ├── backtester.py         # Strategy backtesting
│   └── prediction_logger.py  # Performance tracking
│
├── ui/                        # Frontend files
│   ├── platform.html         # Main trading dashboard
│   ├── worklist_analytics.html
│   ├── level2_module.html
│   └── time_sales_module.html
│
├── models/                    # Trained ML models
├── logs/                      # Prediction logs
├── strategies/                # Pine Script strategies
└── backups/                   # Auto-generated backups
```

---

## 📦 Installation

### 1. Install Dependencies

```powershell
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
pip install -r requirements.txt
```

**Required packages:**
- fastapi
- uvicorn[standard]
- anthropic
- ib-insync
- pandas
- numpy
- scikit-learn
- lightgbm
- yfinance
- ta (technical analysis)
- python-dotenv
- pydantic

### 2. Configure IBKR TWS

**Enable API Access:**
1. Open TWS or IB Gateway
2. Go to **File → Global Configuration → API → Settings**
3. Enable "ActiveX and Socket Clients"
4. Set Socket Port: **7497** (paper trading) or **7496** (live)
5. Add to Trusted IPs: **127.0.0.1**
6. Click **OK** and restart TWS

**Important:** Always use paper trading for testing!

### 3. Set Up API Key

**Option A: Environment Variable (Session)**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-api03-your-key-here"
```

**Option B: .env File (Permanent)**
```powershell
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
notepad .env
```

Add this line:
```
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

---

## ⚙️ Configuration

### Server Configuration

Edit `dashboard_api.py` to customize:

```python
# Server settings
PORT = 9101
HOST = "0.0.0.0"

# IBKR settings
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497  # Paper trading
IBKR_CLIENT_ID = 1

# Scanner settings
SCANNER_MIN_GAP = 3.0  # Minimum gap percentage
SCANNER_MIN_VOLUME = 500000  # Minimum pre-market volume
SCANNER_PRICE_MIN = 5.0  # Minimum price
SCANNER_PRICE_MAX = 50.0  # Maximum price

# Risk management (3-5-7 strategy)
MAX_RISK_PER_TRADE = 0.03  # 3% per trade
MAX_DAILY_LOSS = 0.05  # 5% daily limit
MAX_DRAWDOWN = 0.07  # 7% maximum drawdown
```

### Strategy Configuration

The system supports multiple TradingView strategies:

1. **Gap and Go** - Pre-market gap momentum
2. **Warrior Momentum** - Intraday trend following
3. **Bull Flag Micro Pullback** - Continuation patterns
4. **Flat Top Breakout** - Resistance breakouts

Configure strategy parameters in the respective modules.

---

## 💻 Usage

### Starting the Server

```powershell
# Method 1: Direct start
python dashboard_api.py

# Method 2: Using start script
.\START_TRADING_BOT.ps1

# Method 3: With specific port
uvicorn dashboard_api:app --host 0.0.0.0 --port 9101 --reload
```

### Checking System Status

```powershell
# Run comprehensive status check
.\CHECK_STATUS.ps1

# Or manually check health
Invoke-RestMethod -Uri "http://127.0.0.1:9101/health"
```

**Expected Health Response:**
```json
{
  "status": "healthy",
  "ibkr_connected": true,
  "ibkr_available": true,
  "ai_predictor_loaded": true,
  "claude_available": true,
  "active_subscriptions": 3,
  "symbols_tracked": ["AAPL", "TSLA", "SPY"]
}
```

### Common Operations

**Subscribe to a Symbol:**
```powershell
$body = @{symbol="TSLA"; exchange="SMART"; data_types=@("quote")} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/subscribe" -Method POST -ContentType "application/json" -Body $body
```

**Get Real-Time Price:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/price/AAPL"
```

**Get Claude Analysis:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/claude/analyze-with-data/AAPL"
```

**Get Scanner Results:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/scanner/results"
```

### Training the AI Predictor

```python
# Open Python console
python

# Train on SPY (2 years of data)
from ai.ai_predictor import get_predictor
predictor = get_predictor()
predictor.train("SPY", period="2y")

# Train on multiple symbols
for symbol in ["AAPL", "TSLA", "QQQ", "IWM"]:
    predictor.train(symbol, period="1y")
```

**Training time:** 5-10 minutes per symbol (2 years of data)

---

## 📡 API Reference

### Core Endpoints

#### Health Check
```http
GET /health
```
Returns system status including IBKR connection, Claude availability, and subscriptions.

#### IBKR Connection
```http
POST /api/ibkr/connect
Content-Type: application/json

{
  "host": "127.0.0.1",
  "port": 7497,
  "client_id": 1
}
```

#### Subscribe to Symbol
```http
POST /api/subscribe
Content-Type: application/json

{
  "symbol": "AAPL",
  "exchange": "SMART",
  "data_types": ["quote"]
}
```

### Market Data Endpoints

#### Get Price
```http
GET /api/price/{symbol}
```
Returns real-time quote data.

#### Get Level 2
```http
GET /api/level2/{symbol}
```
Returns order book depth (bid/ask levels).

#### Get Time & Sales
```http
GET /api/timesales/{symbol}
```
Returns recent trades.

### AI Endpoints

#### Claude Analysis (with market data)
```http
GET /api/claude/analyze-with-data/{symbol}
```
Returns Claude's analysis using live IBKR market data.

#### ML Prediction
```http
GET /api/predict/{symbol}
```
Returns LightGBM momentum prediction.

### Scanner Endpoints

#### Get Scanner Results
```http
GET /api/scanner/results
```
Returns pre-market gap setups.

#### Configure Scanner
```http
POST /api/scanner/configure
Content-Type: application/json

{
  "min_gap": 3.0,
  "min_volume": 500000,
  "price_min": 5.0,
  "price_max": 50.0
}
```

### WebSocket Streaming

#### Market Data Stream
```javascript
ws://127.0.0.1:9101/ws/market-data

// Send subscription
{
  "action": "subscribe",
  "symbol": "AAPL"
}

// Receive updates
{
  "symbol": "AAPL",
  "last": 226.50,
  "bid": 226.48,
  "ask": 226.52,
  "volume": 45234567,
  "timestamp": "2025-11-13T14:30:00"
}
```

---

## 🔧 Troubleshooting

### Server Won't Start

**Issue:** Port 9101 already in use

**Solution:**
```powershell
# Find process using port 9101
$pid = (Get-NetTCPConnection -LocalPort 9101).OwningProcess
Stop-Process -Id $pid -Force

# Restart server
python dashboard_api.py
```

### IBKR Won't Connect

**Issue:** Connection refused or timeout

**Solutions:**
1. Verify TWS is running and logged in
2. Check API settings are enabled in TWS
3. Confirm correct port (7497 for paper, 7496 for live)
4. Add 127.0.0.1 to TWS trusted IPs
5. Try different client_id (use 999 instead of 1)

**Check IBKR connection:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/ibkr/status"
```

### Claude Says "No Market Data"

**Issue:** Claude can't access market data

**Solutions:**
1. Verify you deployed `dashboard_api_COMPLETE.py`
2. Subscribe to symbol first: `/api/subscribe`
3. Use correct endpoint: `/api/claude/analyze-with-data/{symbol}`
4. Wait 2-3 seconds after subscribing for data to arrive

### Platform Shows 404 Errors

**Issue:** Old endpoints not working

**Solution:** The COMPLETE API has backward compatibility. If still seeing 404s:
```powershell
# Verify correct file deployed
Get-Content dashboard_api.py | Select-String "COMPLETE UNIFIED API"

# Should return: "IBKR Algorithmic Trading Bot V2 - COMPLETE UNIFIED API"
```

### AI Predictor Not Loading

**Issue:** `ai_predictor_loaded: false`

**Solution:** Train the model first:
```python
from ai.ai_predictor import get_predictor
predictor = get_predictor()
predictor.train("SPY", period="2y")
```

### WebSocket Connection Fails

**Issue:** WebSocket won't connect

**Solutions:**
1. Check firewall isn't blocking port 9101
2. Verify server is running: check `/health` endpoint
3. Try different browser
4. Check browser console for errors

---

## 🛡️ Safety & Risk Management

### ⚠️ Critical Safety Rules

1. **ALWAYS START WITH PAPER TRADING**
   - Never test with real money
   - Use TWS Paper Trading account (port 7497)
   - Verify all strategies in simulation first

2. **3-5-7 Risk Management**
   - **3%** maximum risk per trade
   - **5%** daily loss limit (stop trading)
   - **7%** maximum drawdown (shut down system)

3. **Position Sizing**
   - Never risk more than 3% of account on single trade
   - Use proper stop losses on every trade
   - Limit total position size to 20% of account

4. **Pre-Market Trading Hours**
   - System focuses on 4:00 AM - 10:00 AM ET window
   - Gap & Go strategy for pre-market momentum
   - Avoid low-liquidity periods

### Auto-Trading Checklist

Before enabling auto-trading:

- [ ] Thoroughly backtested strategy (min 6 months)
- [ ] Tested in paper trading for at least 2 weeks
- [ ] Risk management rules configured and tested
- [ ] Stop losses working correctly
- [ ] Position sizing validated
- [ ] Emergency kill switch ready
- [ ] Monitoring and alerts set up
- [ ] Small account size for initial live testing

### Emergency Stop

**To immediately stop all trading:**

```powershell
# Stop server
$pythonProcess = Get-Process -Name "python" | Where-Object {$_.CommandLine -like "*dashboard_api*"}
Stop-Process -Id $pythonProcess.Id -Force

# Or use Ctrl+C in server terminal
```

### Monitoring

**Key metrics to monitor:**
- Win rate (target: >50%)
- Average win vs average loss (target: >1.5:1)
- Maximum consecutive losses (limit: 3)
- Daily P&L vs limit
- Current drawdown vs maximum

---

## 📊 Performance Tracking

### Prediction Logging

All AI predictions are logged automatically:

```python
# View prediction log
import pandas as pd
df = pd.read_csv("logs/predictions.csv")

# Analyze performance
accuracy = (df['prediction'] == df['actual']).mean()
print(f"Model Accuracy: {accuracy:.2%}")
```

### Backtesting

Run backtests on strategies:

```python
from ai.backtester import Backtester

backtester = Backtester()
results = backtester.run_backtest(
    symbol="SPY",
    strategy="gap_and_go",
    start_date="2024-01-01",
    end_date="2025-01-01"
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

---

## 🔗 Useful Resources

### Documentation
- **API Docs:** http://127.0.0.1:9101/docs (when server running)
- **IBKR API Docs:** https://interactivebrokers.github.io/tws-api/
- **ib-insync Docs:** https://ib-insync.readthedocs.io/
- **Claude API Docs:** https://docs.anthropic.com/

### Support
- **IBKR Support:** https://www.interactivebrokers.com/support
- **Anthropic Support:** https://support.anthropic.com

### Trading Education
- **Gap & Go Strategy:** https://bullishbears.com/gap-and-go/
- **Pre-Market Trading:** https://www.warriortrading.com/pre-market-trading/
- **Risk Management:** https://www.investopedia.com/terms/r/risk-management.asp

---

## 📝 License & Disclaimer

**DISCLAIMER:** This software is provided for educational and research purposes only. 

- Trading involves substantial risk of loss
- Past performance is not indicative of future results
- The developers are not responsible for any financial losses
- Always consult with financial professionals before trading
- Use at your own risk

**For Paper Trading and Educational Use Only**

---

## 🎯 What's Next?

### After Getting System Operational

1. **Train AI Models** - Get LightGBM predictions working
2. **Customize Scanner** - Fine-tune gap detection criteria
3. **Add More Strategies** - Implement additional Pine Script strategies
4. **Monitor Performance** - Track predictions and adjust
5. **Paper Trade** - Test extensively before considering live trading

### Advanced Features to Explore

- **Multi-timeframe analysis** - Analyze multiple timeframes simultaneously
- **Options integration** - Add options data and strategies
- **Portfolio optimization** - Implement portfolio allocation algorithms
- **Custom indicators** - Create your own technical indicators
- **News sentiment** - Integrate real-time news analysis

---

## 📞 Quick Command Reference

```powershell
# Navigate to project
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2

# Deploy system
.\DEPLOY_COMPLETE_API.ps1

# Check status
.\CHECK_STATUS.ps1

# Start server manually
python dashboard_api.py

# Set API key
$env:ANTHROPIC_API_KEY = "your-key"

# Test health
Invoke-RestMethod "http://127.0.0.1:9101/health"

# Open platform
start http://127.0.0.1:9101/ui/platform.html
```

---

**Version:** 2.0  
**Last Updated:** November 13, 2025  
**Status:** Production Ready (Paper Trading)

🚀 **Happy Trading!** (Responsibly!)
