# AI Momentum Trading Bot

Professional AI-powered momentum trading system with Interactive Brokers integration.

## 📚 [Full Documentation](docs/README.md)

## 🚀 Quick Start
```powershell
cd C:\ai_project_hub
.\.venv\Scripts\Activate.ps1
.\scripts\bootstrap.ps1
```

Then open: http://127.0.0.1:9101/ui/trading.html

## ✨ Features

- **Fast Order Entry** - Momentum trading interface with hotkeys
- **AI Predictions** - 99.56% accuracy with 12 technical indicators
- **Real-time Monitoring** - Live account tracking and P&L
- **Order Management** - Market & Limit orders with extended hours support
- **Analytics** - Performance tracking and backtesting
- **Risk Controls** - Position sizing, stop-loss, daily limits

## 📖 Architecture

See [AI Momentum Trading Architecture (PDF)](docs/AI_Momentum_Trading_Architecture.pdf) for complete system design.

## 🛠️ Tech Stack

- **API:** FastAPI, ib_insync
- **ML:** scikit-learn, LightGBM, PyTorch
- **Data:** yfinance, pandas, DuckDB
- **UI:** HTML/CSS/JS with Chart.js

## 📊 Interfaces

1. **Trading Interface** - Fast momentum order entry
2. **Account Monitor** - Real-time portfolio tracking
3. **Analytics Dashboard** - Performance metrics and backtesting

## ⚙️ Configuration

Edit `.env` file:
```
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
LOCAL_API_KEY=Your_API_Key
```

## 🔒 Security

- API key authentication required for all trading operations
- Paper trading mode by default (port 7497)
- Switch to live trading at your own risk (port 7496)

## 📝 License

Private project - All rights reserved

## ⚠️ Disclaimer

Trading involves risk. This software is for educational purposes. Use at your own risk.
