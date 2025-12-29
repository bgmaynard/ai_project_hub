# Morpheus Trading Bot - System Manifest

## Overview
Automated trading platform using Schwab as the primary broker, with AI-powered signal generation, risk management, and execution.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │     morpheus_trading_api.py (9100)      │
                    │          FastAPI + WebSocket            │
                    └──────────────────┬──────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
   ┌────▼────┐                  ┌──────▼──────┐              ┌────────▼────────┐
   │ Broker  │                  │ Market Data │              │   AI Pipeline   │
   │ Layer   │                  │   Layer     │              │                 │
   └────┬────┘                  └──────┬──────┘              └────────┬────────┘
        │                              │                              │
   unified_broker.py            unified_market_data.py         Signal Gating
   schwab_trading.py            schwab_market_data.py          Chronos Adapter
                                polygon_streaming.py           Qlib Exporter
```

## Core Components

### 1. Broker Integration
| File | Purpose |
|------|---------|
| `unified_broker.py` | Broker abstraction layer |
| `schwab_trading.py` | Schwab API orders/positions |
| `schwab_market_data.py` | Schwab quotes & streaming |

### 2. Market Data
| File | Purpose |
|------|---------|
| `unified_market_data.py` | Market data abstraction |
| `polygon_streaming.py` | Real-time WebSocket trades/quotes |
| `polygon_data.py` | Historical bars & reference data |

### 3. AI Signal Pipeline
| File | Purpose |
|------|---------|
| `ai/signal_contract.py` | Immutable signal schema |
| `ai/signal_gating_engine.py` | Validates signals, enforces sequencing |
| `ai/chronos_adapter.py` | Market regime classification (context only) |
| `ai/qlib_exporter.py` | Exports SignalContracts from research |
| `ai/gated_trading.py` | Integration with HFT Scalper |

### 4. Execution
| File | Purpose |
|------|---------|
| `ai/hft_scalper.py` | HFT momentum scalper |
| `ai/scalp_assistant.py` | AI exit manager for manual trades |
| `ai/chronos_exit_manager.py` | Smart stop loss prevention |
| `ai/trading_engine.py` | Trade execution logic |

### 5. News & Scanning
| File | Purpose |
|------|---------|
| `ai/benzinga_fast_news.py` | Fast breaking news detection |
| `ai/news_auto_trader.py` | News-triggered trade coordinator |
| `ai/premarket_scanner.py` | Pre-market gapper scanner |

## Data Flow

```
News/Scanner Trigger
        │
        ▼
┌───────────────────┐
│ Find Signal       │ ← Lookup in signal_contracts.json
│ Contract          │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Chronos Context   │ ← Market regime, confidence, volatility
│ (NO TRADES)       │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Signal Gating     │ ← Validates contract, regime, confidence
│ Engine            │
└────────┬──────────┘
         │
    ┌────┴────┐
    │         │
 APPROVED   VETOED
    │         │
    ▼         ▼
 Execute    Log reason
```

## Key Principles

1. **Chronos MUST NOT place trades** - Only provides regime/context
2. **Qlib MUST NOT do live inference** - Offline research only
3. **All trades go through Signal Gating Engine** - Every entry validated
4. **Every veto is logged** - Full audit trail

## API Endpoints

### Core
- `GET /api/status` - System status
- `GET /api/account` - Account info
- `GET /api/positions` - Current positions
- `GET /api/worklist` - Watchlist with AI scores

### AI/Signals
- `GET /api/scanner/scalper/status` - Scalper status
- `POST /api/scanner/scalper/start` - Start scalper
- `GET /api/scanner/news-trader/status` - News trader status

### Market Data
- `GET /api/price/{symbol}` - Price quote
- `GET /api/level2/{symbol}` - Level 2 depth
- `GET /api/polygon/bars/{symbol}` - Historical bars

## Configuration Files

| File | Purpose |
|------|---------|
| `ai/scalper_config.json` | HFT Scalper parameters |
| `ai/signal_contracts.json` | Pre-approved trading signals |
| `ai/chronos_exit_config.json` | Exit manager settings |
| `.env` | API keys (Schwab, Polygon, Benzinga) |

## Server

- **Port**: 9100
- **Dashboard**: http://localhost:9100/dashboard
- **Trading UI**: http://localhost:9100/trading-new/
- **API Docs**: http://localhost:9100/docs
