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
| `ai/fundamental_analysis.py` | Stock fundamental analysis (float, metrics) |
| `ai/chart_patterns.py` | Chart pattern recognition (S/R, triangles, flags) |
| `ai/news_api_routes.py` | News REST API endpoints |
| `ai/claude_stock_scanner.py` | AI-powered stock scanner |
| `ai/claude_api.py` | Claude API integration |
| `ai/ai_predictor.py` | LightGBM predictions (78% accuracy) |
| `ai/alpha_fusion.py` | Multi-model ensemble |
| `ai/trading_engine.py` | Trade execution logic |
| `ai/trade_validator.py` | Risk validation |
| `ai/circuit_breaker.py` | Loss protection |

## HFT Scalper System (Dec 17, 2024)

| File | Purpose |
|------|---------|
| `ai/hft_scalper.py` | Automated HFT momentum scalper |
| `ai/halt_detector.py` | Trading halt detection (LULD, news) |
| `ai/halt_analytics.py` | Halt pattern analytics |
| `ai/split_tracker.py` | Stock split tracking & SMI score |
| `ai/pattern_correlator.py` | Mover pattern correlation analysis |
| `ai/scalper_config.json` | Scalper configuration (persisted) |
| `ai/scalper_trades.json` | Scalper trade history |

### HFT Scalper API Endpoints

```
GET  /api/scanner/scalper/status     - Status & config
POST /api/scanner/scalper/start      - Start monitoring
POST /api/scanner/scalper/stop       - Stop monitoring
POST /api/scanner/scalper/enable     - Enable trading (paper_mode=true/false)
POST /api/scanner/scalper/disable    - Disable trading
GET  /api/scanner/scalper/config     - Get configuration
POST /api/scanner/scalper/config     - Update configuration
POST /api/scanner/scalper/reset      - Reset daily stats
GET  /api/scanner/scalper/positions  - Open positions
GET  /api/scanner/scalper/trades     - Trade history
GET  /api/scanner/scalper/stats      - Performance statistics
GET  /api/scanner/scalper/watchlist  - Scalper watchlist
POST /api/scanner/scalper/watchlist/add/{symbol}    - Add symbol
DELETE /api/scanner/scalper/watchlist/{symbol}      - Remove symbol
```

### Current Scalper Parameters (Optimized Dec 17)

```json
{
  "account_size": 1000.0,
  "risk_percent": 1.0,
  "use_risk_based_sizing": true,
  "min_spike_percent": 5.0,
  "min_volume_surge": 3.0,
  "profit_target_percent": 3.0,
  "stop_loss_percent": 3.0,
  "trailing_stop_percent": 1.5,
  "max_hold_seconds": 180,
  "max_spread_percent": 1.0,
  "min_price": 1.0,
  "max_price": 20.0
}
```

### Scalper Exit Logic
1. **Stop Loss (-3%)** - Always protect capital
2. **Hit +3% target** - DON'T exit, START trailing from high
3. **Trail drops 1.5% from high** - Exit with profit locked
4. **Reversal detection (2% drop)** - Exit if was up
5. **Max hold time (3 min)** - Only exit if flat/down, let winners run

## Market Data Providers

| File | Purpose |
|------|---------|
| `schwab_market_data.py` | Real-time quotes & price history |
| `polygon_data.py` | Polygon.io historical bars & reference data |

The `ai/` directory also contains 50+ "warrior_" prefixed modules for momentum scanning, pattern detection, risk management, and sentiment analysis.

## Batch Files

| File | Purpose |
|------|---------|
| `RESTART_CLEAN.bat` | Full stop + clean restart |
| `START_FULL_TRADING_SYSTEM.bat` | Start complete system |
| `STOP_TRADING_PLATFORM.bat` | Stop everything |
| `START_TRADING_PLATFORM.bat` | Basic platform start |

## API Endpoints

**Core:**
- `/api/status` - System status
- `/api/account` - Account info
- `/api/positions` - Current positions
- `/api/worklist` - Watchlist with quotes, float, news, AI prediction
- `/api/price/{symbol}` - Get price quote
- `/api/level2/{symbol}` - Level 2 bid/ask data
- `/dashboard` - Main UI

**AI Analysis:**
- `/api/stock/ai-prediction/{symbol}` - AI price prediction
- `/api/stock/float/{symbol}` - Float shares data
- `/api/stock/news-check/{symbol}` - Breaking news indicator
- `/api/patterns/{symbol}` - Chart pattern recognition
- `/api/patterns/scan` - Scan watchlist for patterns

**News:**
- `/api/news/info` - News system status
- `/api/news/fetch` - Fetch news from Benzinga

**Market Data (Polygon.io):**
- `/api/polygon/status` - Polygon connection status
- `/api/polygon/ticker/{symbol}` - Reference data
- `/api/polygon/bars/{symbol}` - Historical minute bars
- `/api/polygon/daily/{symbol}` - Historical daily bars
- `/api/polygon/trades/{symbol}` - Trades (requires paid)

## Configuration

Environment variables (`.env`):
- `SCHWAB_APP_KEY` - Schwab API key
- `SCHWAB_APP_SECRET` - Schwab secret
- `BENZINGA_API_KEY` - Benzinga news API key
- `ANTHROPIC_API_KEY` - Claude AI key
- `FINNHUB_API_KEY` - Finnhub API key
- `POLYGON_API_KEY` - Polygon.io API key
- Server port: 9100

## Trading Context & User Preferences

### Trading Schedule
- **Pre-market (4:00 AM - 9:30 AM ET)**: Primary trading window - different momentum dynamics
- **Market Open (9:30 - 10:00 AM)**: Watch only, too chaotic for manual trading
- **Mid-day (10:00 AM - 4:00 PM)**: Trends emerge, safer for manual entries

### Trading Style
- HFT scalping approach (speed matters)
- Focus on low-float momentum stocks ($1-$20 price range)
- No shorting (learned from $1800 halt reversal loss)
- 1% risk per trade = 100 attempts before account depleted
- "Green is green" - take profits, don't get greedy
- "Be a better loser" - cut losses fast, preserve capital
- Wait for "front runner" - the stock with market attention

### Key Indicators Watched
- MACD crossover/divergence
- VWAP (entry above, stop below)
- 200 EMA support
- Float rotation
- Relative volume (RVol)
- Bid/ask spread (wide = red flag)

### Pre-Market vs Market Hours
Pre-market has different characteristics:
- Lower volume, wider spreads
- Gaps can be extreme
- Momentum moves are faster/more volatile
- Need different parameters than regular hours
- Halts are more dangerous (can't exit)

## Recent Changes (Dec 2024)

- Removed Alpaca completely - Schwab only
- Fixed news API (was 404) - created fundamental_analysis.py
- Updated news_feed_monitor.py to use Benzinga instead of Alpaca
- Fixed worklist display (async issue)
- Reduced polling intervals to prevent browser resource exhaustion
- Added Float data to watchlist (via yfinance)
- Added Breaking News indicator to watchlist
- Added AI Prediction column to watchlist (LightGBM 78% accuracy)
- Created Chart Pattern Recognition system (S/R, triangles, flags, breakouts)
- Integrated Polygon.io for historical market data
- Fixed % change calculation (now shows proper daily change)

## Dec 17, 2024 Session

### New Components Built
1. **HFT Scalper** (`ai/hft_scalper.py`) - Automated momentum scalper
2. **Split Tracker** (`ai/split_tracker.py`) - Tracks splits, calculates SMI score
3. **Pattern Correlator** (`ai/pattern_correlator.py`) - Records movers for pattern analysis
4. **Halt Detector** (`ai/halt_detector.py`) - Detects LULD/news halts

### Paper Trading Results
- Round 1 (morning, 3% spike threshold): 48 trades, -$101 (overtrading)
- Round 2 (afternoon, 5% spike threshold): 4 trades, +$0.54 (selective)
- **Lesson**: Selectivity matters - tighter entry filters reduce bad trades

### Next Steps
- Test tighter parameters on pre-market session
- Pre-market momentum detection may need different thresholds
- Need to handle reversal detection better for faster moves
- Consider time-of-day adjustments to parameters

## Dec 18, 2024 Session

### Manual Trading (WeBull) - GREEN DAY
- ATHA: -$21.57 (stopped out, no momentum)
- EVTV: +$3.50 (grinder, took profit on pullback)
- LHAI: +$20.45 (breakout over $14, nailed it)
- **Net: +$2.38** - recovered from red to green

### Scalper Paper Trading
- 57 trades, -$170.66 (overtrading in holiday chop)
- Adjusted params: 7% spike, 5x volume surge, 2% trailing

### New Components Added
- `ai/trade_signals.py` - Secondary trigger capture for correlation analysis
- `ai/pybroker_walkforward.py` - Walkforward analysis framework (needs minute data)

### Research: Financial Libraries
- **PyBroker** - Installed, walkforward framework ready
- **Microsoft Qlib** - For future model research (40+ quant models)

### TODO: Future Work
1. Connect Polygon minute data to PyBroker for proper walkforward testing
2. Wire trade_signals.py into scalper for correlation data capture
3. Add ETB/HTB borrow status to watchlist (Schwab has `shortable` field)
4. Investigate Qlib for advanced alpha factor discovery
5. Build correlation report: which secondary triggers predict winners?

## Known Issues / Notes

- Market data only works during market hours
- News feed shows mock data when Benzinga API unavailable
- Dashboard polling intervals set conservatively to prevent ERR_INSUFFICIENT_RESOURCES
- Time & Sales requires Polygon.io paid subscription ($199/mo) for real tick data
- Polygon.io free tier provides delayed historical data only

## Dependencies

Core: `fastapi`, `uvicorn`, `anthropic`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `yfinance`, `ta`, `httpx`, `python-dotenv`
