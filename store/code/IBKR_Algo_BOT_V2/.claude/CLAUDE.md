# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Morpheus Trading Bot - automated trading platform using **Schwab** as the primary broker (Alpaca removed). Dashboard runs at `http://localhost:9100/dashboard`.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     morpheus_trading_api.py (port 9100)              │
│                         FastAPI + WebSocket                          │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
┌──────▼──────┐      ┌────────▼────────┐    ┌───────▼───────┐
│unified_broker│      │unified_market_data│    │compatibility_  │
│   (Schwab)   │      │    (Schwab)       │    │   routes       │
└──────┬──────┘      └────────┬────────┘    └───────────────┘
       │                      │
┌──────▼──────┐      ┌────────▼────────┐
│schwab_trading│      │schwab_market_data│
│              │      │schwab_streaming  │
└──────────────┘      └──────────────────┘
```

The codebase uses:
- **unified_broker.py** - Broker abstraction (Schwab orders/positions)
- **unified_market_data.py** - Market data abstraction (Schwab quotes)
- **compatibility_routes.py** - Maps legacy IBKR endpoints to Schwab

## Commands

```bash
# Start server
python morpheus_trading_api.py

# Test API
curl http://localhost:9100/api/status
curl http://localhost:9100/api/news/info
curl http://localhost:9100/api/scanner/results

# Open dashboard
start http://localhost:9100/dashboard
```

## Key Files

| File | Purpose |
|------|---------|
| `morpheus_trading_api.py` | Main FastAPI server (port 9100) |
| `unified_broker.py` | Broker abstraction layer (Schwab) |
| `unified_market_data.py` | Market data from Schwab |
| `schwab_trading.py` | Schwab API integration |
| `schwab_market_data.py` | Schwab streaming data |
| `compatibility_routes.py` | API route compatibility layer |
| `ui/complete_platform.html` | Main trading dashboard |

## AI Components

| File | Purpose |
|------|---------|
| `ai/news_feed_monitor.py` | News monitoring (Benzinga API) |
| `ai/benzinga_fast_news.py` | Fast breaking news detection |
| `ai/fundamental_analysis.py` | Stock fundamental analysis |
| `ai/news_api_routes.py` | News REST API endpoints |
| `ai/claude_stock_scanner.py` | AI-powered stock scanner |
| `ai/claude_api.py` | Claude API integration |
| `ai/ai_predictor.py` | LightGBM predictions |
| `ai/alpha_fusion.py` | Multi-model ensemble |
| `ai/trading_engine.py` | Trade execution logic |
| `ai/trade_validator.py` | Risk validation |
| `ai/circuit_breaker.py` | Loss protection |

The `ai/` directory also contains 50+ "warrior_" prefixed modules for momentum scanning, pattern detection, risk management, and sentiment analysis.

## Batch Files

| File | Purpose |
|------|---------|
| `RESTART_CLEAN.bat` | Full stop + clean restart |
| `START_FULL_TRADING_SYSTEM.bat` | Start complete system |
| `STOP_TRADING_PLATFORM.bat` | Stop everything |
| `START_TRADING_PLATFORM.bat` | Basic platform start |

## API Endpoints

- `/api/status` - System status
- `/api/account` - Account info
- `/api/positions` - Current positions
- `/api/worklist` - Watchlist symbols
- `/api/news/info` - News system status
- `/api/news/fetch` - Fetch news from Benzinga
- `/api/scanner/results` - AI scanner results
- `/api/price/{symbol}` - Get price quote
- `/api/level2/{symbol}` - Level 2 data
- `/dashboard` - Main UI

## Configuration

Environment variables (`.env`):
- `SCHWAB_APP_KEY` - Schwab API key
- `SCHWAB_SECRET` - Schwab secret
- `BENZINGA_API_KEY` - Benzinga news API key
- `ANTHROPIC_API_KEY` - Claude AI key
- Server port: 9100

## Recent Changes (Dec 2024)

- Removed Alpaca completely - Schwab only
- Fixed news API (was 404) - created fundamental_analysis.py
- Updated news_feed_monitor.py to use Benzinga instead of Alpaca
- Fixed worklist display (async issue)
- Reduced polling intervals to prevent browser resource exhaustion

## Known Issues / Notes

- Market data only works during market hours
- News feed shows mock data when Benzinga API unavailable
- Dashboard polling intervals set conservatively to prevent ERR_INSUFFICIENT_RESOURCES

## Dependencies

Core: `fastapi`, `uvicorn`, `anthropic`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `yfinance`, `ta`, `httpx`, `python-dotenv`
