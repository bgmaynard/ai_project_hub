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

## HFT Scalp Assistant (Dec 20, 2024)

AI-assisted exit manager for manual trades. Human enters via ThinkOrSwim, AI monitors and auto-exits.

| File | Purpose |
|------|---------|
| `ai/scalp_assistant.py` | Core monitoring logic, exit triggers |
| `ai/scalp_assistant_routes.py` | REST API endpoints |
| `ai/scalp_assistant_config.json` | Configuration (persisted) |
| `ai/scalp_assistant_history.json` | Exit history |

### Scalp Assistant API Endpoints

```
GET  /api/scalp/status             - Status & monitored positions
POST /api/scalp/start              - Start monitoring
POST /api/scalp/stop               - Stop monitoring
POST /api/scalp/takeover/{symbol}  - Enable AI control for position
POST /api/scalp/release/{symbol}   - Disable AI, return to manual
GET  /api/scalp/config             - Get configuration
POST /api/scalp/config             - Update configuration
GET  /api/scalp/positions          - Get all positions
GET  /api/scalp/history            - Get exit history
POST /api/scalp/sync               - Force sync from broker
POST /api/scalp/paper/{mode}       - Enable/disable paper mode
GET  /api/scalp/stats              - Performance statistics
POST /api/scalp/reset-stats        - Reset statistics
```

### Dashboard Integration

- **Positions Table**: New "AI" column with checkbox to enable/disable AI takeover per position
- **Scalp Monitor Window**: Tools > Scalp Assistant - shows real-time monitoring status
- AI status indicator in positions header

### Exit Triggers

1. **Stop Loss (-3%)** - Always protect capital
2. **Profit Target (+3%)** - Activates trailing mode
3. **Trailing Stop (1.5%)** - Locks in gains from high
4. **Momentum Reversal** - 3 consecutive red candles
5. **Velocity Death** - Price stalls while down
6. **Max Hold Time (180s)** - Only exits if flat/down

### Key Insight (from training analysis)

- 63% of momentum spikes fade within 3 minutes
- Prediction models don't work for scalping - execution/exit strategy matters
- Human edge: Entry identification (intuition, pattern recognition)
- AI edge: Exit discipline (no emotion, no FOMO)

## Market Data Providers

| File | Purpose |
|------|---------|
| `schwab_market_data.py` | Real-time quotes & price history |
| `polygon_data.py` | Polygon.io historical bars & reference data |
| `polygon_streaming.py` | Real-time WebSocket trades/quotes (paid tier) |
| `polygon_streaming_routes.py` | Streaming API endpoints |

### Polygon Streaming API (Advanced Plan - $199/mo)

```
GET  /api/polygon/stream/status          - Stream status
POST /api/polygon/stream/start           - Start WebSocket
POST /api/polygon/stream/subscribe       - Subscribe to symbol
GET  /api/polygon/stream/trades/{sym}    - Get real-time trades (tape)
GET  /api/polygon/stream/quote/{sym}     - Get real-time quote
GET  /api/polygon/stream/tape            - Combined tape all symbols
POST /api/polygon/stream/subscribe-watchlist - Subscribe multiple symbols
```

**Features:**
- Real-time trades (time & sales) via WebSocket (~50ms latency)
- Real-time NBBO quotes
- Level 2 order book depth
- 20 years historical data
- Dashboard Time & Sales auto-uses Polygon when streaming

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

### AI/ML Libraries Integrated
- **Amazon Chronos** - Zero-shot time series forecasting (active)
- **Microsoft Qlib Alpha158** - 158 quant factors integrated into ensemble (active)
- **PyBroker** - Walkforward backtesting framework (ready)
- **LightGBM** - Gradient boosting classifier (active)

### TODO: Future Work
1. ~~Wire trade_signals.py into scalper for correlation data capture~~ ✅ DONE Dec 19
2. ~~Build correlation report: which secondary triggers predict winners?~~ ✅ DONE Dec 19
3. ~~ATR-based dynamic stops~~ ✅ DONE Dec 19
4. ~~ETB/HTB borrow status~~ ✅ DONE Dec 19
5. ~~PyBroker walkforward validation~~ ✅ DONE Dec 19
6. ~~Train Qlib model on historical data~~ ✅ DONE Dec 19
7. ~~Connect news → watchlist → auto-trade pipeline~~ ✅ DONE Dec 19

### Future Upgrades (Post-Testing)
1. **Real-Time AI Trade Monitor** - Live chart showing trades as they happen
   - Auto-refresh chart every 5-10 seconds for new trades
   - Flash/highlight markers when entry or exit occurs
   - Toast notifications: "ENTERED AAPL @ $185.25" / "EXITED +$12.50"
   - Dedicated monitor window for watching AI trade in real-time
2. **Signal-based exit triggers** - Use MACD bearish crossover to exit (100% win rate in backtest)
3. **Multi-timeframe confirmation** - Require 1M + 5M alignment before entry

### News Auto-Trader System (Dec 19)

Complete pipeline connecting news detection to automated trade execution.

| File | Purpose |
|------|---------|
| `ai/news_auto_trader.py` | News-triggered trade coordinator |
| `ai/benzinga_fast_news.py` | Ultra-fast news detection (1-2s polling) |
| `ai/news_feed_monitor.py` | News monitoring with sentiment analysis |
| `ai/intelligent_watchlist.py` | Watchlist with jacknife/whipsaw protection |

**Pipeline Flow:**
1. BenzingaFastNews detects breaking news (FDA, earnings, M&A, upgrades)
2. News with confidence >= 70% and urgency >= "high" triggers evaluation
3. Symbol auto-added to watchlist via `intelligent_watchlist.add_from_news()`
4. AI filters evaluate: Chronos, Qlib (59.9% accuracy), Order Flow
5. If ALL filters pass -> HFT Scalper executes via `priority_symbols` queue
6. Cooldown prevents re-entry on same symbol for 5 minutes

**News Auto-Trader API Endpoints:**
```
GET  /api/scanner/news-trader/status      - Status & config
POST /api/scanner/news-trader/start       - Start (paper_mode=true/false)
POST /api/scanner/news-trader/stop        - Stop
GET  /api/scanner/news-trader/config      - Get configuration
POST /api/scanner/news-trader/config      - Update configuration
GET  /api/scanner/news-trader/candidates  - Evaluated trade candidates
GET  /api/scanner/news-trader/trades      - Executed trades
```

**Configuration (news_auto_trader_config.json):**
```json
{
  "enabled": false,
  "paper_mode": true,
  "min_news_confidence": 0.7,
  "min_news_urgency": "high",
  "require_chronos": true,
  "min_chronos_score": 0.6,
  "require_qlib": true,
  "min_qlib_score": 0.55,
  "require_order_flow": true,
  "min_buy_pressure": 0.55,
  "max_concurrent_trades": 2,
  "symbol_cooldown_minutes": 5,
  "max_daily_trades": 10,
  "auto_add_to_watchlist": true
}
```

**Scalper Priority Queue:**
News-triggered entries bypass momentum detection (already passed news filters).
Added to `scalper.priority_symbols[]` for immediate execution with relaxed spread limits.

### Qlib Trained Model (Dec 19)
Trained LightGBM on Alpha158 features with 14,910 samples from 30 symbols.

| Metric | Value |
|--------|-------|
| Directional Accuracy | 59.9% |
| RMSE | 0.054 |
| Training Samples | 11,928 |
| Test Samples | 2,982 |

**Top Features:**
- TURN_MA10, WVMA_60, QTLD_30, ROC_30, BETA_60

**Files:**
- `ai/qlib_trainer.py` - Training pipeline
- `ai/qlib_model.pkl` - Trained model
- `ai/qlib_model_meta.json` - Training metadata

### Correlation Report API (Dec 19)
```
GET /api/scanner/scalper/correlation-report       - Full JSON report
GET /api/scanner/scalper/correlation-report/text  - Formatted text report
GET /api/scanner/scalper/correlation-report/recommendations - Summary + recommendations
```

### Order Flow Analyzer (Dec 19)
New entry filter analyzing bid/ask imbalance from Schwab Level 2 data.

| File | Purpose |
|------|---------|
| `ai/order_flow_analyzer.py` | Bid/ask imbalance analysis |

**Entry Criteria:**
- Buy pressure > 55% = ENTER (buyers dominate)
- Buy pressure < 45% = SKIP (sellers dominate)
- Spread > 3% = SKIP (too wide)

**API Endpoints:**
```
GET /api/scanner/order-flow/{symbol}           - Single symbol analysis
GET /api/scanner/order-flow/batch/{symbols}    - Multiple symbols (comma-separated)
GET /api/scanner/order-flow/summary            - Summary of all tracked symbols
```

**Scalper Config:**
```json
{
  "use_order_flow_filter": true,
  "min_buy_pressure": 0.55
}
```

### ATR Dynamic Stops (Dec 19)
Volatility-adjusted stop losses using Average True Range.

| File | Purpose |
|------|---------|
| `ai/atr_stops.py` | ATR calculation and dynamic stop levels |

**Volatility Regimes:**
- LOW (ATR < 1%): Tighter stops (1.0x ATR)
- NORMAL (1-2.5%): Standard stops (1.5x ATR)
- HIGH (2.5-5%): Wider stops (2.0x ATR)
- EXTREME (> 5%): Very wide stops (2.5x ATR)

**API Endpoints:**
```
GET /api/scanner/atr-stops/{symbol}?entry_price=X  - Get dynamic stops
```

**Scalper Config:**
```json
{
  "use_atr_stops": true,
  "atr_stop_multiplier": 1.5,
  "atr_target_multiplier": 2.0,
  "atr_trail_multiplier": 1.0
}
```

### ETB/HTB Borrow Status (Dec 19)
Tracks short interest to identify easy/hard to borrow stocks.

| File | Purpose |
|------|---------|
| `ai/borrow_status.py` | Short interest analysis |

**Classification:**
- ETB: Short % < 10%, Days to Cover < 3
- HTB: Short % > 20% OR Days to Cover > 5
- CAUTION: In between

**API Endpoints:**
```
GET /api/scanner/borrow-status/{symbol}        - Single symbol
GET /api/scanner/borrow-status/batch/{symbols} - Multiple symbols
GET /api/scanner/borrow-status/htb-list        - List of HTB stocks
```

### PyBroker Walkforward Validation (Dec 19)
Tests scalper parameters on historical data to detect overfitting.

| File | Purpose |
|------|---------|
| `ai/pybroker_walkforward.py` | Walkforward backtesting |

**API Endpoints:**
```
GET /api/scanner/backtest/validate?symbols=X&days=30  - Validate current params
GET /api/scanner/backtest/walkforward?symbols=X&days=60 - Run walkforward test
```

## Known Issues / Notes

- Market data only works during market hours
- News feed shows mock data when Benzinga API unavailable
- Dashboard polling intervals set conservatively to prevent ERR_INSUFFICIENT_RESOURCES
- Time & Sales requires Polygon.io paid subscription ($199/mo) for real tick data
- Polygon.io free tier provides delayed historical data only

## Dependencies

Core: `fastapi`, `uvicorn`, `anthropic`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `yfinance`, `ta`, `httpx`, `python-dotenv`

## Dec 19, 2024 Evening Session

### Completed This Session
1. **News Auto-Trader Pipeline** - Full implementation connecting news → watchlist → AI filters → HFT scalper
2. **Retrained AI Models**:
   - Qlib: 54.4% accuracy on 41 momentum stocks (20,459 samples)
   - AI Predictor: 64.7% accuracy (76.7% AUC) on 23 stocks
3. **Dashboard Panels Added**:
   - 📰 News Auto-Trader panel (AI Menu) - Shows live status, candidates, AI filter results
   - 📊 AI Signals panel (AI Menu) - Shows Chronos/Qlib/OrderFlow scores for watchlist
4. **Optimized Scalper Config**: 7% spike, 5x volume, 2% stop (tighter stops)
5. **Backtest Validated**: Chronos enhancement +$5.47 improvement

### Current State
- News Auto-Trader: **RUNNING** (paper mode)
- Server: Running on port 9100
- All TODO items from Dec 18-19: **COMPLETED**

### Next Steps (When Resuming)
1. Monitor pre-market (4AM-9:30AM) with news auto-trader
2. Review correlation report after more trades
3. Consider adding VWAP/EMA confirmation filters
4. Target: Improve win rate from 27.7% → 40%+
5. Eventually switch to live trading when paper results improve

### Quick Start Commands
```bash
# Start server
python morpheus_trading_api.py

# Start news auto-trader (API)
curl -X POST "http://localhost:9100/api/scanner/news-trader/start?paper_mode=true"

# Check status
curl http://localhost:9100/api/scanner/news-trader/status

# Open dashboard
start http://localhost:9100/dashboard
```

## Dec 20, 2024 Session - Signal Contract & Gating Architecture

### Architectural Refactor
Implemented proper separation of concerns between Qlib, Chronos, and execution based on ChatGPT's feedback.

**Key Principles:**
1. **Chronos MUST NOT place trades** - Only provides regime/context classification
2. **Qlib MUST NOT do live inference** - Offline research only, exports SignalContracts
3. **Execution MUST validate** - ALL trades go through Signal Gating Engine
4. **Every veto is logged** - Full audit trail of gating decisions

### New Architecture Files

| File | Purpose |
|------|---------|
| `ai/signal_contract.py` | Immutable signal definition schema |
| `ai/signal_gating_engine.py` | Enforces sequencing, veto logging |
| `ai/chronos_adapter.py` | Chronos wrapper - context only, no trades |
| `ai/qlib_exporter.py` | Exports SignalContracts from offline research |
| `ai/gated_trading.py` | Integration layer with HFT Scalper |
| `ai/signal_contracts.json` | Pre-approved signal contracts |

### Signal Flow Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                        OFFLINE PHASE                            │
├─────────────────────────────────────────────────────────────────┤
│  Historical Data → Qlib Research → SignalContracts (exported)  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        LIVE PHASE                               │
├─────────────────────────────────────────────────────────────────┤
│  Trigger → Find Contract → Chronos Context → Gating Engine     │
│                                                    ↓            │
│                                        [APPROVED | VETOED]      │
│                                                    ↓            │
│                                          Risk Engine → Execute  │
└─────────────────────────────────────────────────────────────────┘
```

### SignalContract Schema
```python
@dataclass
class SignalContract:
    signal_id: str
    symbol: str
    direction: str       # LONG | SHORT
    horizon: str         # SCALP | INTRADAY | SWING | POSITION
    features: List[str]  # Features that triggered in research
    confidence_required: float  # Min Chronos confidence to execute
    valid_regimes: List[str]    # TRENDING_UP, RANGING, etc.
    invalid_regimes: List[str]  # Regimes that veto the signal
    max_drawdown_allowed: float
    expected_return: float
    historical_win_rate: float
    profit_factor: float
    source: str          # "QLIB"
    expires_at: str
```

### Gating Engine Checks
1. **Contract expired?** → VETO (CONTRACT_EXPIRED)
2. **Regime mismatch?** → VETO (REGIME_MISMATCH)
3. **Invalid regime?** → VETO (INVALID_REGIME)
4. **Low confidence?** → VETO (CONFIDENCE_LOW)
5. **Symbol cooldown?** → VETO (COOLDOWN_ACTIVE)
6. **Risk limits?** → VETO (RISK_LIMIT_EXCEEDED)
7. **All pass** → APPROVED

### ChronosAdapter Output
```python
@dataclass
class ChronosContext:
    market_regime: str      # TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE
    regime_confidence: float
    prob_up: float          # Informational only
    expected_return_5d: float
    current_volatility: float
    trend_strength: float
    trend_direction: int    # -1, 0, 1
```

### Initial Contracts Generated
Exported 7 validated contracts from Qlib research:
- AAPL: 56.7% WR, 1.66 PF, Sharpe 2.75
- TSLA: 56.7% WR, 1.22 PF, Sharpe 1.28
- NVDA: 52.2% WR, 1.09 PF, Sharpe 0.53
- AMD: 52.6% WR, 1.72 PF, Sharpe 2.76
- SPY: 59.7% WR, 1.21 PF, Sharpe 1.18
- QQQ: 56.7% WR, 1.19 PF, Sharpe 1.03
- GOOGL: 53.7% WR, 1.55 PF, Sharpe 2.73

### Usage
```python
# Get gated trading manager
from ai.gated_trading import get_gated_trading_manager
manager = get_gated_trading_manager()

# Gate a trade attempt
approved, exec_request, reason = manager.gate_trade_attempt(
    symbol="AAPL",
    trigger_type="momentum_spike",
    quote=current_quote
)

if approved:
    # Proceed with execution
    execute_trade(exec_request)
else:
    # Log veto reason
    log.warning(f"Trade vetoed: {reason}")

# Export new contracts from Qlib research
from ai.qlib_exporter import run_research_export
results = run_research_export(["AAPL", "TSLA", "NVDA"])
```

### Integration with Scalper
```python
# Wrap scalper with gating
from ai.gated_trading import integrate_gating_with_scalper
from ai.hft_scalper import get_hft_scalper

scalper = get_hft_scalper()
wrapper = integrate_gating_with_scalper(scalper)

# All trades now go through gating automatically
```

### Benefits
1. **Auditability**: Every trade attempt logged with approval/veto reason
2. **Separation of concerns**: Chronos=context, Qlib=research, Execution=execution
3. **Pre-approved signals**: No live inference, only validated contracts
4. **Regime-aware**: Trades blocked in unfavorable market conditions
5. **Risk protection**: Multiple layers of validation before execution

## Dec 20, 2024 Afternoon - Pre-Market Automation System

### New Automation Components

| File | Purpose |
|------|---------|
| `auto_scan_trade.py` | Auto-scan + trade with watchlist sync |
| `premarket_scanner.py` | Multi-source scanner (Yahoo + Schwab + News) |
| `monitor.py` | Quick system status overview |
| `SCHEDULE_4AM_PREMARKET.bat` | Windows Task Scheduler setup |

### Auto Scan & Trade Pipeline

**Flow:**
1. Scan Yahoo Finance (day_gainers, most_actives) - works 24/7
2. Scan Schwab movers (market hours only)
3. Filter for scalp criteria: $1-$20 price, 5%+ gap, 500K+ volume
4. Score = (Change% × 2) + (Volume in millions)
5. Add top 5 picks to BOTH watchlists (scalper + dashboard)
6. Start HFT Scalper in paper mode

**Usage:**
```bash
# One-time scan
python auto_scan_trade.py

# Continuous mode (every 5 min)
python auto_scan_trade.py --continuous

# Scheduled 4AM start
START_4AM_PREMARKET.bat
```

### Watchlist Sync Fix

**Problem:** Scanner added stocks to scalper's internal watchlist but not the dashboard worklist user sees.

**Solution:** `auto_scan_trade.py` now adds to BOTH:
- `POST /api/scanner/scalper/watchlist/add/{symbol}` - for trading
- `POST /api/worklist/add` with `{"symbol": "XXX"}` - for visibility

### Market Movers API Endpoints

Added to `schwab_market_data.py` and `morpheus_trading_api.py`:

```
GET /api/market/movers?direction=up      - Schwab gainers
GET /api/market/movers?direction=down    - Schwab losers
GET /api/market/movers?direction=all     - Both gainers & losers
GET /api/market/movers/scalp             - Pre-filtered for scalping
```

### Monitor Tool

Quick status check for all system components:
```bash
python monitor.py
```

Shows:
- Server status
- Scalper status (monitoring, trading, paper mode)
- Current watchlist with prices
- AI filters status (Chronos, Qlib, Order Flow, Regime, Scalp Fade)
- Recent trades
- Lifetime statistics

### Batch Files Updated

| File | Purpose |
|------|---------|
| `START_4AM_PREMARKET.bat` | Now uses auto_scan_trade.py |
| `SCHEDULE_4AM_PREMARKET.bat` | Creates Windows Task Scheduler job for weekdays 4AM |

**To schedule 4AM auto-start:**
1. Right-click `SCHEDULE_4AM_PREMARKET.bat` → Run as Administrator
2. Task "Morpheus_4AM_PreMarket" created for weekdays at 4:00 AM

### Quick Start for Monday

```bash
# Option 1: Manual start
python morpheus_trading_api.py
python auto_scan_trade.py

# Option 2: Batch file
START_4AM_PREMARKET.bat

# Monitor
python monitor.py
start http://localhost:9100/dashboard
```

### Current Model Accuracies

| Model | Accuracy | Samples |
|-------|----------|---------|
| Qlib | 56.1% | - |
| Momentum | 54.1% | - |
| Scalp | 62.4% | - |
| Polygon Daily | 53.7% | - |
| Polygon Scalp | 57.5% | - |

### Win Rate Status
- Backtest: 28.9% win rate, -$304.45 total PnL
- Target: Improve to 40%+ through better entry filtering

## Dec 23, 2024 Session

### 4AM Pre-Market Scanner System

New pre-market scanner that builds fresh watchlist each morning at 4 AM.

| File | Purpose |
|------|---------|
| `ai/premarket_scanner.py` | PreMarketScanner + NewsLogger classes |
| `ai/premarket_routes.py` | API endpoints for scanner & news log |

**Features:**
- Scans for pre-market movers (gaps, volume, news catalysts)
- Checks after-hours continuations from previous day
- Logs ALL breaking news with timestamps
- Builds fresh daily watchlist filtered $1-$20 price range
- Syncs to common worklist (data bus pattern)

**API Endpoints:**
```
GET  /api/scanner/premarket/status           - Scanner status
POST /api/scanner/premarket/scan             - Run scan manually
GET  /api/scanner/premarket/watchlist        - Current watchlist
POST /api/scanner/premarket/news-monitor/start - Start news monitor
POST /api/scanner/premarket/news-monitor/stop  - Stop news monitor
GET  /api/news-log                           - Timestamped news log (JSON)
GET  /api/news-log/today                     - Today's news only
GET  /api/news-log/formatted                 - Text format (like screenshot)
GET  /api/news-log/symbol/{symbol}           - News for specific symbol
POST /api/news-log/add                       - Manually add news entry
```

**News Log Format:**
```
$AXP - Truist Securities Maintains Buy on American Express...  12:26 PM
$JNJ - Johnson & Johnson Hit With Record $1.5 Billion...       12:26 PM
```

### Dashboard Updates

- Added **NEWS LOG (Live)** panel in News & Fundamentals tab
- Auto-refreshes every 30 seconds
- Shows timestamped breaking news in real-time

### Common Data Bus

HFT Scalper now syncs with common worklist on each monitor loop:
- All modules share same watchlist via `/api/worklist`
- Prevents conflicting processes and disconnected data

### Trading Analysis

**Webull Manual Trading Analysis:**
- Win Rate: 66.7% (10/15 trades)
- Total P&L: -$43.63
- Problem: Letting losers run (avg loss 2.5x avg win)

**Bot Paper Trading:**
- Win Rate: 29.1% (37/127 trades)
- Total P&L: -$361.02
- Problem: Entry quality, too many small losses

**Key Insight:** User's entry instincts are strong (66% win rate), bot has better exit discipline. Hybrid approach (manual entry + AI exit) is the winning formula.

### Holiday Schedule
- Dec 24: Market closes 1 PM ET (half day)
- Dec 25: Market closed

### Quick Commands
```bash
# Start all systems
python morpheus_trading_api.py
curl -X POST "http://localhost:9100/api/scanner/scalper/start"
curl -X POST "http://localhost:9100/api/scanner/news-trader/start?paper_mode=true"
curl -X POST "http://localhost:9100/api/scanner/premarket/news-monitor/start"

# Check news log
curl http://localhost:9100/api/news-log/formatted
```

### Chronos Exit Manager (Smart Stop Loss Prevention)

**Problem Solved:** Historical data showed 28 stop losses = -$460 (0% win rate), while trailing stops = 85% WR, +$96. Fixed stop losses don't work for momentum plays.

**Solution:** `ai/chronos_exit_manager.py` - Uses Chronos regime detection + Ross Cameron's "failed momentum" rule to exit BEFORE hitting stop losses.

**Key Exit Triggers:**
1. **FAILED_MOMENTUM** (Ross Cameron Rule): If stock doesn't gain 0.5% within 30s, exit
2. **MOMENTUM_STALLED**: Price not moving up for 5 consecutive checks
3. **MOMENTUM_FADING**: Price dropping 0.5% from high while flat/losing
4. **REGIME_SHIFT**: Market regime changed from TRENDING_UP to VOLATILE/TRENDING_DOWN
5. **CONFIDENCE_LOW**: Chronos confidence dropped below 40%
6. **TREND_WEAK**: ADX trend strength below 30%
7. **VOLATILITY_SPIKE**: Annualized volatility > 50%

**Configuration:**
```python
{
    # Ross Cameron Rule - Most Important
    "use_failed_momentum_exit": True,
    "momentum_check_seconds": 30,      # Check after 30s
    "expected_gain_30s": 0.5,          # Expect 0.5% gain
    "momentum_stall_threshold": 0.2,   # < 0.2% = stalled
    "fade_from_high_percent": 0.5,     # Exit if down 0.5% from high
    "consecutive_stall_checks": 3,     # 3 stalls = exit

    # Regime-based exits
    "exit_on_regime_change": True,
    "favorable_regimes": ["TRENDING_UP", "RANGING"],
    "danger_regimes": ["TRENDING_DOWN", "VOLATILE"],

    # Technical thresholds
    "min_confidence": 0.4,
    "min_trend_strength": 0.3,
    "max_volatility": 0.5,
    "min_prob_up": 0.45
}
```

**API Endpoints:**
```
GET  /api/scanner/chronos-exit/status           - Status & monitored positions
GET  /api/scanner/chronos-exit/config           - Get configuration
POST /api/scanner/chronos-exit/config           - Update configuration
GET  /api/scanner/chronos-exit/check/{symbol}?current_price=X&entry_price=Y - Test exit signal
POST /api/scanner/chronos-exit/register/{symbol}?entry_price=X - Register position
DELETE /api/scanner/chronos-exit/{symbol}       - Unregister position
```

**Integration with Scalper:**
- Automatically integrated via `check_exit_signal()` in HFT Scalper
- Config option: `use_chronos_exit: true` (enabled by default)
- Exit signals prefixed with `CHRONOS_` in trade history

**Research Sources:**
- [IEEE Breakout & Reversal Strategies](https://ieeexplore.ieee.org/document/10488993/) - RSI divergences signal momentum changes
- [Wiley Trend Reversal Paper](https://onlinelibrary.wiley.com/doi/10.1002/int.22601) - Directional changes predict reversals
- [TradingView Ross Cameron Strategy](https://www.tradingview.com/script/TcCkGnoS-Ross-Cameron-Inspired-Day-Trading-Strategy/) - Warrior Trading methodology

## Dec 26, 2024 Session - Warrior Trading Setup Detection System

### Overview

Implemented complete Ross Cameron trading methodology from Warrior Trading PDF analysis. System provides pattern detection, tape reading, setup classification, and LULD halt tracking.

### New Modules

| File | Purpose |
|------|---------|
| `ai/tape_analyzer.py` | Tape reading (green/red flow, seller thinning, flush detection, first green print) |
| `ai/pattern_detector.py` | Pattern detection (Bull Flag, ABCD, Micro Pullback, HOD Break) |
| `ai/setup_classifier.py` | A/B/C grading with position sizing rules |
| `ai/warrior_setup_detector.py` | Integration module combining all components |
| `ai/warrior_routes.py` | REST API endpoints for all features |
| `ai/hod_momentum_scanner.py` | High of Day momentum scanner with Polygon streaming |
| `ai/top_gappers_scanner.py` | Top percentage gainers with 5-criteria grading |
| `docs/WARRIOR_TRADING_STRATEGY_GUIDE.md` | Comprehensive strategy reference document |

### Enhanced Modules

| File | Enhancement |
|------|-------------|
| `ai/halt_detector.py` | Added LULD band calculations, 15-second countdown timer, false halt detection (10-12s bounce), resume direction prediction |

### Ross Cameron's 5 Criteria (Stock Grading)

| Criteria | Threshold | Grade Impact |
|----------|-----------|--------------|
| Float | < 10 million shares | Required for A/B |
| Relative Volume | > 5x average | Required for A/B |
| Change % | > 10% on the day | Required for A/B |
| Price | $1.00 - $20.00 | Required for A/B |
| News Catalyst | Breaking news present | Required for A |

**Grading:**
- **A Grade (5/5)**: Full position (75%)
- **B Grade (4/5)**: Half position (50%)
- **C Grade (3/5 or less)**: Scalp only (25%)

### Pattern Detection

| Pattern | Description | Entry |
|---------|-------------|-------|
| **Bull Flag** | 10-20% pole + 2-8 candle consolidation < 50% retrace | Break above flag high |
| **ABCD** | A→B move, C pullback 50-61.8% fib, D = equal move | Break above B after C |
| **Micro Pullback** | Strong momentum + 1-3 candle dip < 30% retrace | First green candle |
| **HOD Break** | Consolidation near high of day | Break above HOD |

### Tape Analysis Signals

| Signal | Description | Action |
|--------|-------------|--------|
| `FIRST_GREEN_PRINT` | First green after red streak (5+ red) | BUY_NOW |
| `SELLER_THINNING` | Large seller absorbed (25k→19k→15k→11k→5k) | PREPARE_TO_BUY |
| `IRRATIONAL_FLUSH` | 3%+ drop in 30 seconds | DIP_BUY |
| `STRONG_BUY_PRESSURE` | Green ratio > 65% | BUY_SUPPORTED |

### LULD Halt Detection

**Band Calculations by Price Tier:**
- Tier 1 (S&P 500, Russell 1000): ±5%
- Tier 2 (Other NMS stocks): ±10%
- Under $3.00: ±20%
- Under $0.75: Lesser of $0.15 or 75%

**Halt Rules:**
- **15 seconds** at band = halt triggered
- **10-12 seconds** = false halt (price bounces back)
- **Resume Prediction**:
  - First up halt → usually continues up
  - 3+ up halts → buyer exhaustion, fade if flat
  - Down halt opens flat → potential long

### Warrior Trading API Endpoints

```
GET  /api/warrior/status              - System status (all components)
GET  /api/warrior/signal/{symbol}     - Complete trading signal
POST /api/warrior/analyze             - Analyze multiple symbols
GET  /api/warrior/patterns/{symbol}   - Pattern detection results
GET  /api/warrior/tape/{symbol}       - Tape analysis
GET  /api/warrior/luld/{symbol}       - LULD band status
POST /api/warrior/luld                - Calculate LULD bands
GET  /api/warrior/halts               - All current halts and warnings
GET  /api/warrior/grade/{symbol}      - Ross Cameron A/B/C grade
GET  /api/warrior/setup-rules/{type}  - Entry rules for setup type
POST /api/warrior/feed/tape           - Feed trade data to analyzer
POST /api/warrior/feed/candles        - Feed candle data to detector
```

### Usage Examples

```bash
# Get trading signal
curl http://localhost:9100/api/warrior/signal/AAPL

# Calculate LULD bands
curl -X POST http://localhost:9100/api/warrior/luld \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","reference_price":250.00,"tier":1}'

# Get stock grade
curl http://localhost:9100/api/warrior/grade/NVDA

# Get setup entry rules
curl http://localhost:9100/api/warrior/setup-rules/bull_flag

# Analyze multiple symbols
curl -X POST http://localhost:9100/api/warrior/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbols":["AAPL","TSLA","NVDA","AMD"]}'
```

### Integration with HFT Scalper

```python
from ai.warrior_setup_detector import get_warrior_detector

detector = get_warrior_detector()

# Get trading signal
signal = await detector.analyze(symbol, current_price, quote)

if signal.action == "BUY":
    # Execute with signal parameters
    entry = signal.entry_price
    stop = signal.stop_loss
    target = signal.target_price
    size_pct = signal.position_size_pct  # Based on grade
```

### Strategy Guide Reference

Full documentation in `docs/WARRIOR_TRADING_STRATEGY_GUIDE.md` includes:
- Core trading philosophy
- All 7 technical setups with if/then logic
- Position sizing rules
- Exit strategies (breakout or bailout, failed momentum)
- Halt trading mechanics
- Implementation priorities

### Warrior Filter Integration (HFT Scalper)

The Warrior Trading methodology is now integrated as an entry filter in the HFT Scalper.

**New Config Options:**
```json
{
  "use_warrior_filter": true,      // Enable Warrior setup grading
  "warrior_min_grade": "B",        // Minimum grade to enter (A, B, or C)
  "warrior_require_pattern": false, // Require Bull Flag, ABCD, etc.
  "warrior_require_tape_signal": false, // Require tape confirmation
  "warrior_max_float": 20.0,       // Max float in millions
  "warrior_min_rvol": 2.0          // Minimum relative volume
}
```

**Grade-Based Position Sizing:**
- Grade A (5/5 criteria): 75% of calculated position
- Grade B (4/5 criteria): 50% of calculated position
- Grade C (3/5 or less): 25% of calculated position

**Warrior Levels:**
When Warrior setup provides entry/stop/target levels, they override ATR-calculated stops.

**Trade History Fields:**
- `warrior_grade`: A, B, or C
- `warrior_score`: 0-100 setup quality
- `warrior_patterns`: Detected patterns (Bull Flag, ABCD, etc.)
- `warrior_tape_signals`: Tape reading signals (green flow, seller thinning)

### Live Data Wiring (Polygon Streaming)

The Polygon WebSocket stream is now wired to all Warrior components:

```python
# In polygon_streaming.py
wire_warrior_trading()  # Wires all components:
  # - wire_hod_scanner() - HOD momentum tracking
  # - wire_tape_analyzer() - Tape reading from trades
  # - wire_pattern_detector() - Pattern detection from candles
```

When Polygon streaming starts, all Warrior components receive live trade data automatically.

### Multi-Timeframe Confirmation (MTF)

Ross Cameron rule: **Only enter when 1-minute and 5-minute charts agree.**

| File | Purpose |
|------|---------|
| `ai/mtf_confirmation.py` | Multi-timeframe analysis engine |

**What It Checks:**
- Trend alignment (both bullish or both bearish)
- MACD alignment (both above signal line)
- EMA alignment (both EMA9 > EMA20)
- VWAP alignment (both above VWAP)
- RSI confirmation
- Price structure (higher highs/lows)

**Signals:**
- `CONFIRMED_LONG` - Both timeframes bullish with full alignment
- `WEAK_LONG` - Trend aligned but missing some confirmations
- `NO_CONFIRMATION` - Timeframes disagree
- `CONFIRMED_SHORT` - Both bearish (avoid longs)

**API Endpoints:**
```
GET  /api/warrior/mtf/{symbol}    - Get MTF confirmation
POST /api/warrior/mtf/analyze     - Batch analyze symbols
GET  /api/warrior/mtf/confirmed   - Get confirmed symbols from watchlist
GET  /api/warrior/mtf/status      - Engine status
```

**Scalper Config:**
```json
{
  "use_mtf_filter": true,          // Enable MTF confirmation
  "mtf_min_confidence": 60.0,      // Minimum confidence (0-100)
  "mtf_require_vwap_aligned": true, // Both timeframes above VWAP
  "mtf_require_macd_aligned": true  // Both timeframes MACD bullish
}
```

**Example Response:**
```json
{
  "symbol": "AAPL",
  "signal": "CONFIRMED_LONG",
  "confidence": 85.0,
  "recommendation": "ENTER",
  "alignment": {
    "trend": true,
    "macd": true,
    "ema": true,
    "vwap": true
  },
  "reasons": ["Both timeframes bullish with MACD and VWAP confirmation"]
}
```

### VWAP Manager (Ross Cameron - Line in the Sand)

Ross Cameron rule: **VWAP is the institutional support line. Only enter above VWAP.**

| File | Purpose |
|------|---------|
| `ai/vwap_manager.py` | VWAP calculation and trailing stop management |

**Key Concepts:**
- VWAP = Sum(Price × Volume) / Sum(Volume) for the day
- Resets at market open each day
- Institutions use VWAP for large order execution
- Price above VWAP = bullish (buyers in control)
- Price below VWAP = bearish (sellers in control)
- Extended above VWAP (>3%) = risky to chase

**Entry Rules:**
- `ABOVE` (0-3%) - Ideal entry zone
- `AT_VWAP` - Waiting for direction
- `FAR_ABOVE` (>3%) - Avoid, wait for pullback
- `BELOW` - No entry on longs

**Exit Rules:**
- Trail stop 0.3% below VWAP
- Exit if price breaks below VWAP

**API Endpoints:**
```
GET  /api/warrior/vwap/{symbol}      - Get VWAP data
GET  /api/warrior/vwap/check/{symbol} - Check entry validity
POST /api/warrior/vwap/feed          - Feed trade data
GET  /api/warrior/vwap/status        - Manager status
POST /api/warrior/vwap/reset         - Reset daily VWAP
```

**Scalper Config:**
```json
{
  "use_vwap_filter": true,           // Require price above VWAP
  "vwap_max_extension_pct": 3.0,     // Max % above VWAP (avoid chasing)
  "use_vwap_trailing_stop": true,    // Use VWAP as trailing stop
  "vwap_stop_offset_pct": 0.3        // Trail 0.3% below VWAP
}
```

**Example Response:**
```json
{
  "symbol": "AAPL",
  "vwap": 185.25,
  "current_price": 186.50,
  "position": "ABOVE",
  "distance_pct": 0.67,
  "signal": "BUY_SUPPORTED",
  "entry_valid": true,
  "stop_price": 184.69,
  "bands": {
    "upper_1": 186.75,
    "upper_2": 188.25,
    "lower_1": 183.75,
    "lower_2": 182.25
  }
}
```

**Polygon Integration:**
VWAP manager is wired to Polygon streaming via `wire_vwap_manager()`. Every trade tick updates the VWAP calculation in real-time.
