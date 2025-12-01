# Claude Delta Sync - AI Project Hub
## Last Updated: November 30, 2025

This file provides context for AI assistants (Claude, ChatGPT, etc.) when continuing work on this project.

---

## Project Overview

**Alpaca Trading Platform** - AI-powered trading platform with:
- Alpaca broker integration (paper trading)
- Multi-channel parallel data architecture
- TradingView Desktop integration
- AI predictions using XGBoost/LightGBM
- Real-time market data and order execution

---

## Current Architecture

### Backend (Python/FastAPI)
```
alpaca_dashboard_api.py    - Main API server (port 9100)
alpaca_integration.py      - Alpaca trading client wrapper
alpaca_market_data.py      - Market data provider
multi_channel_data.py      - NEW: Multi-channel parallel data provider
alpaca_api_routes.py       - Trading API routes
watchlist_routes.py        - Watchlist management
tradingview_integration.py - TradingView webhook integration
```

### Frontend
```
ui/complete_platform.html  - Main trading dashboard (single-file app)
```

### AI Components
```
ai/alpaca_ai_predictor.py      - XGBoost prediction model
ai/claude_bot_intelligence.py  - Claude AI integration
src/pipeline/                  - Advanced training pipeline
src/mesh/task_queue.py         - Multi-agent task queue
```

---

## Latest Changes (Nov 30, 2025)

### 1. Multi-Channel Data Architecture
Created `multi_channel_data.py` with 5 dedicated parallel data channels:

| Channel | Purpose |
|---------|---------|
| ORDERS | Order execution, positions |
| CHARTS | TradingView/charting data |
| AI | AI predictions, training |
| SCANNER | Market scanning |
| REALTIME | Live quotes |

**New API Endpoints:**
- `GET /api/data/channels/status` - Channel stats
- `GET /api/data/channels/multi-quote?symbols=` - Parallel quotes
- `GET /api/data/channels/snapshots?symbols=` - Parallel snapshots
- `GET /api/data/channels/bars/{symbol}` - Historical bars

### 2. TradingView Desktop Integration
Direct integration with TV Desktop app:
- Uses `tradingview://` protocol
- Open charts in specific timeframes
- Bypasses embedded widget's 2-indicator limit
- Auto-detects exchange for symbols

### 3. Chart Template System
- 8 built-in templates (scalping, daytrading, swing, etc.)
- Custom template creation
- Timeframe default mapping
- localStorage persistence

---

## Key Configuration

### Environment Variables (.env)
```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### Starting the Platform
```bash
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
python alpaca_dashboard_api.py
# Server runs on http://localhost:9100
# Dashboard: http://localhost:9100/dashboard
```

---

## Important Files to Know

| File | Purpose |
|------|---------|
| `alpaca_dashboard_api.py` | Main server - all API routes |
| `ui/complete_platform.html` | Full dashboard UI |
| `multi_channel_data.py` | Parallel data provider |
| `config/broker_config.py` | Broker configuration |
| `ai/alpaca_ai_predictor.py` | AI prediction model |

---

## Current Branch
`feat/unified-claude-chatgpt-2025-10-31`

## GitHub Repository
`github.com:bgmaynard/ai_project_hub.git`

---

## Session Handoff Notes

### What's Working
- Alpaca paper trading connected
- Multi-channel data fetching (5 channels active)
- TradingView Desktop integration
- Chart templates with custom creation
- AI predictions (XGBoost model)
- Watchlist management
- Order placement/cancellation
- PDT rule monitoring

### Known Issues
- MCP server tools show "Not connected" (requires session restart)
- Embedded TradingView widget limited to 2 indicators (use TV Desktop instead)

### Next Steps Suggested
1. Real-time WebSocket streaming per channel
2. Channel health monitoring widget
3. TV Desktop watchlist auto-sync
4. Advanced order types (bracket, OCO)

---

## Quick Commands

```bash
# Start platform
python alpaca_dashboard_api.py

# Test health
curl http://localhost:9100/api/health

# Test multi-channel
curl "http://localhost:9100/api/data/channels/status"
curl "http://localhost:9100/api/data/channels/multi-quote?symbols=AAPL,TSLA,NVDA"

# Git status
git status
git log --oneline -5
```

---

## For AI Assistants

When continuing this project:
1. Read this file first for context
2. Check `docs/SESSION_*.md` for recent session details
3. The platform runs on port 9100
4. Use multi-channel endpoints for faster data access
5. TradingView Desktop is preferred over embedded charts
