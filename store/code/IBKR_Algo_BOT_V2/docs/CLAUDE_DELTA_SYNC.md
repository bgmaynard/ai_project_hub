# Claude Delta Sync - AI Project Hub
## Last Updated: December 2, 2025

This file provides context for AI assistants (Claude, ChatGPT, etc.) when continuing work on this project.

---

## Project Overview

**Alpaca Trading Platform** - AI-powered trading platform with:
- Alpaca broker integration (paper trading)
- Multi-channel parallel data architecture
- Real-time WebSocket streaming
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
multi_channel_data.py      - Multi-channel parallel data provider
realtime_streaming.py      - NEW: WebSocket streaming module
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

## Latest Changes (Dec 2, 2025)

### 1. Real-Time WebSocket Streaming
Created `realtime_streaming.py` for live market data:

**Features:**
- Quote streaming (Level 1 bid/ask)
- Trade streaming (executed trades)
- Bar streaming (minute bars)
- Multi-client broadcasting
- Auto-reconnection

**New API Endpoints:**
- `GET /api/streaming/status` - Streaming service status
- `POST /api/streaming/subscribe` - Subscribe to symbols
- `POST /api/streaming/unsubscribe` - Unsubscribe from symbols
- `POST /api/streaming/start` - Start streaming
- `POST /api/streaming/stop` - Stop streaming
- `WS /ws/market` - WebSocket endpoint

### 2. Channel Health Monitoring Widget
Added to dashboard top bar:
- **Data status indicator** - Shows 5ch when all channels active
- **Streaming status indicator** - Shows subscription count
- **Channel Status Modal** - Detailed per-channel metrics
- **Streaming Control Modal** - Subscribe/unsubscribe controls

### 3. Health Check Updates
Health endpoint now includes streaming status:
```json
{
  "services": {
    "multi_channel_data": "active (5 channels)",
    "realtime_streaming": "available"
  }
}
```

---

## Previous Session (Nov 30, 2025)

### Multi-Channel Data Architecture
5 dedicated parallel data channels:
| Channel | Purpose |
|---------|---------|
| ORDERS | Order execution, positions |
| CHARTS | TradingView/charting data |
| AI | AI predictions, training |
| SCANNER | Market scanning |
| REALTIME | Live quotes |

### TradingView Desktop Integration
- Uses `tradingview://` protocol
- Open charts in specific timeframes
- Bypasses embedded widget's 2-indicator limit

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
| `realtime_streaming.py` | WebSocket streaming |
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
- Real-time WebSocket streaming infrastructure
- Channel health monitoring in dashboard
- Streaming control modal in dashboard
- TradingView Desktop integration
- TV Desktop watchlist sync (export & copy)
- Chart templates with custom creation
- AI predictions (XGBoost model)
- Watchlist management
- Order placement/cancellation
- **Advanced Order Types**:
  - Market, Limit, Stop, Stop-Limit
  - Bracket (entry + TP + SL)
  - OCO (One-Cancels-Other)
  - Trailing Stop (% or $)
- PDT rule monitoring

### Known Issues
- MCP server tools show "Not connected" (requires session restart)
- Embedded TradingView widget limited to 2 indicators (use TV Desktop instead)
- Streaming requires market hours for data (pre-market shows no data)

### Next Steps Suggested
1. Connect dashboard watchlist to WebSocket stream for live prices
2. Add bracket/OCO order UI in the order panel
3. Implement auto-trading with bracket orders
4. Real-time P&L tracking via streaming

---

## Quick Commands

```bash
# Start platform
python alpaca_dashboard_api.py

# Test health
curl http://localhost:9100/api/health

# Test multi-channel
curl "http://localhost:9100/api/data/channels/status"

# Test streaming
curl http://localhost:9100/api/streaming/status

# Subscribe to streaming
curl -X POST http://localhost:9100/api/streaming/subscribe \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "TSLA"], "types": ["quotes"]}'

# Git status
git status
git log --oneline -5
```

---

## WebSocket Usage

Connect to `ws://localhost:9100/ws/market`:

```javascript
const ws = new WebSocket('ws://localhost:9100/ws/market');

ws.onopen = () => {
    // Subscribe to quotes
    ws.send(JSON.stringify({
        action: 'subscribe',
        symbols: ['AAPL', 'TSLA'],
        types: ['quotes']
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // Handle quote/trade/bar data
    console.log(data);
};
```

---

## For AI Assistants

When continuing this project:
1. Read this file first for context
2. Check `docs/SESSION_*.md` for recent session details
3. The platform runs on port 9100
4. Use multi-channel endpoints for faster data access
5. TradingView Desktop is preferred over embedded charts
6. Streaming requires market hours for real data
7. WebSocket endpoint ready for dashboard integration
