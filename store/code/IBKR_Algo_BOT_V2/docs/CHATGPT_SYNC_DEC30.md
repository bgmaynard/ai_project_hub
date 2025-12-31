# ChatGPT Sync Document - Dec 30, 2025

## Current State Summary

### System Architecture
- **Server**: `morpheus_trading_api.py` on port 9100
- **Broker**: Schwab only (Alpaca removed)
- **UI**: React app at `/trading-new` (Golden Layout)
- **Dashboard**: `/dashboard` now redirects to `/trading-new` (consolidated)

---

## Recent Changes (Dec 30 Session)

### 1. Momentum Watchlist - Dynamic Rel Vol Floor

**File**: `ai/momentum_watchlist.py`

```python
# Uses ZoneInfo instead of pytz
from zoneinfo import ZoneInfo
ET = ZoneInfo('America/New_York')

# Rel vol schedule tuple in WatchlistConfig
rel_vol_schedule: Tuple[Tuple[int, int, float], ...] = (
    (4, 9, 5.0),    # 04:00-09:00 ET → rel_vol >= 5.0
    (9, 10, 3.0),   # 09:00-10:00 ET → rel_vol >= 3.0
    (10, 12, 2.0),  # 10:00-12:00 ET → rel_vol >= 2.0
    (12, 20, 1.5),  # 12:00-20:00 ET → rel_vol >= 1.5
)

# Session date uses ET timezone
def session_date(self) -> date:
    et_now = datetime.now(timezone.utc).astimezone(ET)
    return et_now.date()
```

### 2. Connection Manager - HTTP Resilience

**File**: `ai/connection_manager.py` (NEW)

- Circuit breaker pattern
- Retry with exponential backoff
- `_reset_http_client()` on TimeoutException/ConnectError
- Data freshness tracking

### 3. UI Consolidation

**Single Dashboard**: `/trading-new` is now the ONLY active UI

| Route | Behavior |
|-------|----------|
| `/trading-new` | Primary React trading UI |
| `/dashboard` | 302 redirect to `/trading-new` |

**Updated files**:
- `morpheus_trading_api.py` - Redirect logic
- All `.bat` and `.ps1` startup scripts
- `ui/ai_control_dashboard.html`
- `ui/complete_platform.html`

### 4. UI Wiring to Backend APIs

**File**: `ui/trading/src/components/Worklist.tsx`

Now makes these HTTP calls:

| Button | API Call |
|--------|----------|
| Discovery | `POST /api/task-queue/run` |
| Refresh | `POST /api/watchlist/refresh` |
| Purge | `POST /api/watchlist/purge` |
| Row Delete (✕) | `DELETE /api/worklist/{symbol}` |
| Row Delete (⊘) | `DELETE /api/watchlist/{symbol}` |

**On load + every 5s**:
- `GET /api/watchlist/status`
- `GET /api/worklist`

**Status bar shows**: Session date, rel_vol floor, active count, cycle count

---

## Key API Endpoints

### Momentum Watchlist
```
GET  /api/watchlist/status     - Session info, config, active symbols
GET  /api/watchlist/active     - Active symbols with metrics
GET  /api/watchlist/config     - Current configuration
POST /api/watchlist/refresh    - Reset for fresh session
POST /api/watchlist/purge      - Remove all active symbols
DELETE /api/watchlist/{symbol} - Delete one symbol
```

### Task Queue (Discovery Pipeline)
```
POST /api/task-queue/run       - Run full discovery pipeline
GET  /api/task-queue/status    - Pipeline status
```

### Legacy Worklist
```
GET  /api/worklist             - All symbols with live quotes
POST /api/worklist/add         - Add symbol
DELETE /api/worklist/{symbol}  - Remove symbol
```

---

## File Structure

```
ai/
├── momentum_watchlist.py       # Session-scoped ranked watchlist
├── momentum_watchlist_routes.py # Operator control endpoints
├── connection_manager.py       # HTTP resilience (NEW)
├── task_group_1_discovery.py   # R1 discovery task
├── task_queue_manager.py       # Pipeline orchestration
└── task_queue_routes.py        # Pipeline API endpoints

ui/trading/src/
├── services/api.ts             # API service layer
├── components/Worklist.tsx     # Watchlist UI with controls
└── stores/watchlistStore.ts    # Zustand state
```

---

## Git Status

```
Latest commit: 4fd6066
Branch: main
Message: feat: UI consolidation + momentum watchlist API wiring
```

---

## Quick Start

```bash
# Start server
python morpheus_trading_api.py

# Open dashboard
start http://localhost:9100/trading-new

# Test APIs
curl http://localhost:9100/api/watchlist/status
curl -X POST http://localhost:9100/api/watchlist/refresh
curl -X POST http://localhost:9100/api/task-queue/run
```

---

## Key Principles (Established Earlier)

1. **Full Recompute Model**: Every cycle re-evaluates ALL symbols, no grandfathering
2. **Hard Rel Vol Floor**: Symbols below threshold are REJECTED (dynamic by time)
3. **Session Boundary**: New day (ET) = fresh watchlist
4. **Separation of Concerns**:
   - Chronos = Context/Regime only (no trades)
   - Qlib = Offline research only (exports SignalContracts)
   - Gating Engine = Final authority on all trades
5. **Audit Trail**: All operator actions logged to `reports/R0_operator_actions.json`

---

## TODO / Next Steps

1. Test discovery pipeline with live market data
2. Verify rel_vol floor changes by time window during trading hours
3. Monitor Network tab to confirm all API calls are firing
4. Consider adding WebSocket for real-time updates (currently polling)
