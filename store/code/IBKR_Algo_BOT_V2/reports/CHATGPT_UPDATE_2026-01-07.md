# MORPHEUS Trading Bot - Status Update
**Date:** January 7, 2026
**Session Time:** 12:00 PM - 11:45 PM ET
**Report For:** ChatGPT Architecture Review

---

## Executive Summary

**ALL 20 TASKS FROM CHATGPT DIRECTIVES COMPLETED (A-T)**

### Directive 1: "No Trades in 2 Weeks" Recovery Patch (Tasks A-F)
- Task A: Trading Window Watchdog (auto-enable scalper)
- Task B: Candidate Funnel Metrics (pipeline observability)
- Task C: RelVol Robustness (avgVolume fallback chain)
- Task D: Micro-Momentum Exception (bypass macro regime veto)
- Task E: UI Setup vs Execution split (new scanner columns)
- Task F: Probe Entry Execution Layer (controlled initiation for scalper)

### Directive 2: Adaptive Baseline Configuration (Tasks G-J)
- Task G: Baseline Profile System (CONSERVATIVE/NEUTRAL/AGGRESSIVE)
- Task H: Market Condition Evaluation (breadth, participation, gap rate)
- Task I: Profile Selection Logic (checkpoints, locking)
- Task J: Observability (API + UI visibility)

### Directive 3: Time-of-Day Strategy Orchestration (Tasks K-O)
- Task K: Market Phase Definition (OPEN_IGNITION, STRUCTURED_MOMENTUM, MIDDAY_COMPRESSION, POWER_HOUR)
- Task L: Strategy Compatibility Matrix (Phase -> Allowed Strategies)
- Task M: Phase Evaluation Engine (time-based + metric overrides)
- Task N: Strategy Enable/Disable Hooks (gate layer, decorators)
- Task O: Observability & UI (14 API endpoints for phase visibility)

### Directive 4: Momentum Discovery & Strategy Orchestration Calibration (Tasks P-T)
- Task P: Momentum Scout Mode (CRITICAL) - early discovery, small size, no Chronos blocking
- Task Q: Scout → Strategy Handoff (escalate to ATS/Scalper when confirmed)
- Task R: Strategy-Specific Chronos Weighting (IGNORED for scouts, HARD_REQUIREMENT for swings)
- Task S: Phase-Controlled Exploration Policy (scouts enabled/disabled by market phase)
- Task T: Observability & Diagnostics (25+ API endpoints + funnel metrics integration)

---

## ChatGPT Recovery Patch Implementation

### Task A: Trading Window Watchdog

**Purpose:** Auto-enable scalper during trading windows to prevent missed trades.

**Files Created:**
- `ai/trading_window_watchdog.py` - Core watchdog logic
- `ai/watchdog_routes.py` - REST API endpoints

**Trading Windows:**
- Pre-market: 4:00 AM - 9:30 AM ET
- Market hours: 9:30 AM - 4:00 PM ET
- After hours: 4:00 PM - 8:00 PM ET

**API Endpoints:**
```
GET  /api/watchdog/status    - Watchdog status
POST /api/watchdog/start     - Start watchdog
POST /api/watchdog/stop      - Stop watchdog
POST /api/watchdog/check     - Force check
```

**Integration:** Watchdog starts automatically 15 seconds after server startup.

---

### Task B: Candidate Funnel Metrics

**Purpose:** Track where candidates die in the pipeline (scanner → gating → execution).

**Files Created:**
- `ai/funnel_metrics.py` - Funnel tracking module
- `ai/funnel_routes.py` - REST API endpoints

**Metrics Tracked:**
| Stage | Function | Event |
|-------|----------|-------|
| Scanner | `record_scanner_find()` | Symbol found by scanner |
| Injection | `record_symbol_injection()` | Symbol added to watchlist |
| Rate Limit | `record_rate_limit_defer()` | Deferred due to rate limit |
| Quality | `record_quality_reject()` | Rejected by quality check |
| Chronos | `record_chronos_signal()` | Chronos signal received |
| Gating | `record_gating_attempt()` | Gating check attempted |
| Gating | `record_gating_approval()` | Gating approved |
| Gating | `record_gating_veto()` | Gating vetoed (with reason) |
| Execution | `record_trade_execution()` | Trade executed |

**API Endpoints:**
```
GET /api/ops/funnel/status      - Full funnel status
GET /api/ops/funnel/vetoes      - Recent vetoes with reasons
GET /api/ops/funnel/diagnostic  - Bottleneck identification
```

**Diagnostic Output:**
```json
{
  "health": "DEGRADED",
  "bottleneck": "gating",
  "recommendation": "Check regime/gating rules - high veto rate",
  "funnel_efficiency": {
    "scanner_to_inject": "50%",
    "inject_to_gate": "80%",
    "gate_to_exec": "10%"
  }
}
```

---

### Task C: RelVol Robustness

**Purpose:** Prevent over-exclusion when avgVolume is missing from Schwab.

**Files Created:**
- `ai/relvol_resolver.py` - Fallback chain resolver
- `ai/relvol_routes.py` - REST API endpoints

**Fallback Chain:**
1. **Schwab** (primary) - Real-time avgVolume
2. **yfinance** (fallback) - Cached 24 hours
3. **UNKNOWN** (degraded) - Routes as degraded candidate, NOT excluded

**Key Design Decision:** Symbols with UNKNOWN avgVolume are NOT excluded at R1. They are routed as "degraded candidates" with provenance metadata so downstream stages can decide.

**API Endpoints:**
```
GET  /api/ops/relvol/status           - Resolver status
GET  /api/ops/relvol/resolve/{symbol} - Resolve avgVolume
POST /api/ops/relvol/resolve/batch    - Batch resolve
GET  /api/ops/relvol/degraded         - List degraded symbols
GET  /api/ops/relvol/cache            - View cache contents
POST /api/ops/relvol/clear-cache      - Clear cache
```

---

### Task D: Micro-Momentum Exception

**Purpose:** Allow trades in bad MACRO regime if MICRO momentum is strong.

**Problem Solved:** REGIME_MISMATCH was vetoing ALL entries in TRENDING_DOWN macro regime, even when individual symbols had strong bullish micro momentum.

**Files Created:**
- `ai/micro_momentum_override.py` - Override logic
- `ai/micro_override_routes.py` - REST API endpoints

**Files Modified:**
- `ai/signal_gating_engine.py` - Integration at REGIME_MISMATCH veto point
- `ai/signal_contract.py` - Added override fields to GateResult

**Override Conditions (ALL must be true):**
1. ATS state is ACTIVE, CONFIRMED, or IGNITING
2. Micro confidence >= 65%
3. RelVol >= 2.0x (if provided)
4. Volume >= 500K (if provided)
5. Float <= 50M (if provided)
6. Rate limit not exceeded (1 per 10 minutes)

**Override Config:**
```python
{
    "enabled": True,
    "max_per_10min": 1,
    "size_multiplier": 0.5,  # Reduced position size
    "min_micro_confidence": 0.65,
    "min_rel_vol": 2.0,
    "qualifying_ats_states": ["ACTIVE", "CONFIRMED", "IGNITING"]
}
```

**API Endpoints:**
```
GET  /api/ops/micro-override/status   - Override status
POST /api/ops/micro-override/check    - Test override eligibility
GET  /api/ops/micro-override/config   - Get config
POST /api/ops/micro-override/config   - Update config
POST /api/ops/micro-override/enable   - Enable override
POST /api/ops/micro-override/disable  - Disable override
```

**Integration:** When gating would veto with REGIME_MISMATCH, it now checks micro-override first. If override granted:
- Trade proceeds with 50% position size
- Event logged: `GATING_MICRO_OVERRIDE_APPLIED`
- GateResult.override_applied = True

---

### Task E: UI Setup vs Execution Split

**Purpose:** Show Setup quality (A/B/C) separately from Execution permission (YES/NO) in scanner.

**Files Modified:**
- `ui/trading/src/stores/scannerStore.ts` - Added `execStatus`, `execReason` to ScannerResult
- `ui/trading/src/components/ScannerPanel.tsx` - Added Setup and Exec columns
- `ui/trading/src/services/api.ts` - Added `getSetupExecStatus()` and `getBatchSetupExecStatus()`
- `ai/scanner_api_routes.py` - Added `/setup-exec-status/{symbol}` endpoint

**New Scanner Columns:**
| Column | Description | Values |
|--------|-------------|--------|
| Setup | Warrior Trading grade | A (green), B (yellow), C (gray) |
| Exec | Gating permission | YES (green), NO (red) |

**UI Display:**
```
┌──────────┬───────────────────┬───────────────────┐
│ Symbol   │ Setup (A/B/C)     │ Execution (Yes/No)│
├──────────┼───────────────────┼───────────────────┤
│ NVVE     │ A (98)            │ YES               │
│ ATHA     │ B (72)            │ NO                │
│ GPUS     │ C (45)            │ NO                │
└──────────┴───────────────────┴───────────────────┘
```

**API Endpoints:**
```
GET  /api/scanner/setup-exec-status/{symbol}  - Single symbol
POST /api/scanner/setup-exec-status/batch     - Multiple symbols
```

---

## Earlier Session Work

### Chronos-Bolt Upgrade (250x Faster Inference)

**Upgraded from:** `amazon/chronos-t5-small`
**Upgraded to:** `amazon/chronos-bolt-small`

**API Change:**
```python
# Bolt uses predict_quantiles() instead of predict()
quantiles, mean = pipeline.predict_quantiles(
    context,
    prediction_length=horizon,
    quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9]
)
```

### Auto-Scan Feature

Added periodic Finviz scanning with new config options:
```json
{
  "auto_scan_enabled": true,
  "auto_scan_interval": 180,
  "auto_scan_types": ["momentum", "new_highs", "gainers"]
}
```

---

## Files Created This Session

| File | Purpose |
|------|---------|
| `ai/trading_window_watchdog.py` | Auto-enable scalper during trading hours |
| `ai/watchdog_routes.py` | Watchdog REST API |
| `ai/funnel_metrics.py` | Pipeline funnel tracking |
| `ai/funnel_routes.py` | Funnel REST API |
| `ai/relvol_resolver.py` | avgVolume fallback chain |
| `ai/relvol_routes.py` | RelVol REST API |
| `ai/micro_momentum_override.py` | Macro regime bypass for strong micro |
| `ai/micro_override_routes.py` | Override REST API |

## Files Modified This Session

| File | Change |
|------|--------|
| `ai/chronos_predictor.py` | Added Bolt API support |
| `ai/signal_gating_engine.py` | Micro-override integration, funnel metrics |
| `ai/signal_contract.py` | Added override fields to GateResult |
| `ai/finviz_scanner.py` | Added funnel metrics recording |
| `ai/scanner_api_routes.py` | Added setup-exec-status endpoint |
| `morpheus_trading_api.py` | Added all new routes + watchdog startup |
| `ui/trading/src/stores/scannerStore.ts` | Added execStatus, execReason fields |
| `ui/trading/src/components/ScannerPanel.tsx` | Added Setup/Exec columns |
| `ui/trading/src/services/api.ts` | Added setup-exec API calls |

---

## Verification Commands

```bash
# Test all new components
python -c "
from ai.trading_window_watchdog import get_watchdog
from ai.funnel_metrics import get_funnel_metrics
from ai.relvol_resolver import get_relvol_resolver
from ai.micro_momentum_override import check_micro_override
from ai.signal_gating_engine import get_gating_engine
print('All Task A-E components OK!')
"

# Test micro-override
curl -X POST "http://localhost:9100/api/ops/micro-override/check?symbol=NVVE&macro_regime=TRENDING_DOWN&micro_regime=TRENDING_UP&micro_confidence=0.75&ats_state=ACTIVE"

# Check funnel diagnostic
curl http://localhost:9100/api/ops/funnel/diagnostic

# Check watchdog status
curl http://localhost:9100/api/watchdog/status
```

---

## Quick Start (Tomorrow)

```bash
# Start server (watchdog auto-starts after 15s)
python morpheus_trading_api.py

# Verify watchdog is running
curl http://localhost:9100/api/watchdog/status

# Manual scalper start if needed
curl -X POST "http://localhost:9100/api/scanner/scalper/start?paper_mode=true"
curl -X POST "http://localhost:9100/api/scanner/scalper/enable"

# Check funnel for bottlenecks after trades
curl http://localhost:9100/api/ops/funnel/diagnostic
```

---

### Task F: Probe Entry Execution Layer

**Purpose:** Resolve "zero trades despite valid permission" by adding controlled probe execution when normal triggers don't fire.

**Files Created:**
- `ai/probe_entry_manager.py` - Core probe entry logic with ProbeConfig, ProbeEntry, ProbeEntryManager
- `ai/probe_entry_routes.py` - REST API endpoints

**Files Modified:**
- `ai/hft_scalper.py` - Added check_probe_entry(), probe sizing in execute_entry(), probe fallback in monitor loop
- `morpheus_trading_api.py` - Added probe entry routes

**Probe Entry Rules:**
| Rule | Value |
|------|-------|
| Position Size | 25-33% of normal (configurable, default 30%) |
| Stop Loss | 1.5% (tighter than normal) |
| Max per Symbol | 1 per session |
| Max per Hour | 3 across all symbols |
| Cooldown | 15 minutes after stopped probe |

**Probe Entry Conditions (ALL required):**
1. ATS state in {IGNITING, ACTIVE, CONFIRMED}
2. Chronos micro confidence >= 60%
3. Volume acceleration detected (>1.5x avg)
4. Price crosses micro level (one of):
   - VWAP_RECLAIM: Price crosses above VWAP
   - PREMARKET_HIGH_BREAK: Price breaks premarket high
   - RANGE_HIGH_1M_BREAK: Price breaks 1-minute range high
   - RANGE_HIGH_3M_BREAK: Price breaks 3-minute range high
   - HOD_BREAK: Price breaks high of day

**Probe States:**
- `ACTIVE`: Probe entry in progress, monitoring for confirmation
- `STOPPED`: Hit stop loss or failed momentum (triggers cooldown)
- `CONFIRMED`: Momentum continued, normal scalper logic takes over

**Blocked Regimes (no probes allowed):**
- CRASH
- HALT_RISK
- TRENDING_DOWN

**API Endpoints:**
```
GET  /api/probe/status           - Probe manager status
POST /api/probe/check            - Test eligibility (does NOT execute)
GET  /api/probe/active           - Active probe positions
GET  /api/probe/active/{symbol}  - Active probe for symbol
GET  /api/probe/history          - Probe history
GET  /api/probe/cooldowns        - Symbols in cooldown
GET  /api/probe/config           - Get configuration
POST /api/probe/config           - Update configuration
POST /api/probe/enable           - Enable probes
POST /api/probe/disable          - Disable probes
POST /api/probe/close/{symbol}   - Manually close probe
GET  /api/probe/stats            - Probe statistics
POST /api/probe/reset-stats      - Reset statistics
GET  /api/probe/funnel-impact    - Probe impact on funnel metrics
```

**Scalper Config Options:**
```json
{
  "use_probe_entries": true,
  "probe_size_multiplier": 0.30,
  "probe_stop_loss_percent": 1.5,
  "probe_min_micro_confidence": 0.60,
  "probe_max_per_symbol": 1,
  "probe_max_per_hour": 3,
  "probe_cooldown_minutes": 15
}
```

**Required Events (logged):**
- `PROBE_ENTRY_ATTEMPT`: Probe entry initiated
- `PROBE_ENTRY_STOPPED`: Probe hit stop or failed
- `PROBE_ENTRY_CONFIRMED`: Probe confirmed, continuation detected

**Integration with HFT Scalper:**
When normal `check_entry_signal()` returns None but conditions are favorable:
1. Falls back to `check_probe_entry()`
2. If eligible, creates probe signal with reduced size/tighter stops
3. Probe state tracked in trade record
4. Monitor loop updates probe state (STOPPED or CONFIRMED)
5. If stopped, cooldown applied; if confirmed, normal exit logic takes over

---

## Adaptive Baseline Configuration (Tasks G-J)

### Task G: Baseline Profile System

**Purpose:** Three profiles controlling all entry/risk parameters.

**Files Created:**
- `ai/baseline_profiles.py` - Core profile definitions and management

**Profiles:**
| Profile | Rel Vol Floor | Chronos Conf | Probe Size | Scalper Aggr | Loss Cooldown |
|---------|--------------|--------------|------------|--------------|---------------|
| CONSERVATIVE | 3.0x | 70% | 25% | 1 | 120s |
| NEUTRAL | 2.0x | 60% | 30% | 2 | 60s |
| AGGRESSIVE | 1.5x | 50% | 40% | 3 | 30s |

**ProfileParameters Dataclass:**
```python
@dataclass
class ProfileParameters:
    name: str
    rel_vol_floor: float = 2.0
    chronos_micro_confidence_min: float = 0.60
    probe_enabled: bool = True
    probe_size_multiplier: float = 0.30
    scalper_aggressiveness: int = 2
    cooldown_after_loss: int = 60
    max_concurrent_positions: int = 3
    max_drawdown_percent: float = 2.0
    stop_loss_multiplier: float = 1.0
```

---

### Task H: Market Condition Evaluation

**Purpose:** Evaluate market health to determine appropriate profile.

**Files Created:**
- `ai/market_condition_evaluator.py` - Market condition assessment

**Metrics Evaluated:**
| Metric | Source | Weight |
|--------|--------|--------|
| Market Breadth | SPY vs QQQ advance/decline | 25% |
| Small-Cap Participation | IWM performance vs SPY | 20% |
| Gap Continuation Rate | % of gaps that hold | 20% |
| Chronos Regime Distribution | % bullish/bearish/neutral | 20% |
| Volume Profile | Relative volume levels | 15% |

**Conditions:**
- **WEAK** (score < 40): Conservative profile recommended
- **MIXED** (40-60): Neutral profile recommended
- **STRONG** (score > 60): Aggressive profile recommended

---

### Task I: Profile Selection Logic

**Purpose:** Auto-select profile at trading checkpoints.

**Files Created:**
- `ai/profile_selector.py` - Checkpoint-based profile selection

**Checkpoints:**
| Checkpoint | Time (ET) | Priority |
|------------|-----------|----------|
| PRE_MARKET | 4:00-9:30 AM | 1 |
| POST_OPEN | 9:45-10:00 AM | 2 |
| MIDDAY | 11:30-12:00 PM | 3 |
| AFTERNOON | 14:00-14:30 PM | 4 |
| CLOSE | 15:30-16:00 PM | 5 |

**Selection Rules:**
- Profile locked for 30 minutes after selection
- BASELINE_PROFILE_SELECTED event emitted on change
- Background loop evaluates every 60 seconds

---

### Task J: Baseline Observability

**Files Created:**
- `ai/baseline_routes.py` - REST API endpoints

**API Endpoints:**
```
GET  /api/baseline/status           - Full system status
GET  /api/baseline/profile          - Current profile & parameters
GET  /api/baseline/profiles         - All available profiles
GET  /api/baseline/market-condition - Current market evaluation
GET  /api/baseline/selector/status  - Profile selector status
POST /api/baseline/selector/evaluate - Force evaluation
POST /api/baseline/profile/{name}   - Set profile manually
GET  /api/baseline/dashboard        - UI summary
```

---

## Time-of-Day Strategy Orchestration (Tasks K-O)

### Task K: Market Phase Definition

**Purpose:** Define discrete trading phases with different characteristics.

**Files Created:**
- `ai/market_phases.py` - Phase definitions and configuration

**Phases:**
| Phase | Time (ET) | Volatility | Strategies Allowed |
|-------|-----------|------------|-------------------|
| PRE_MARKET | 4:00-9:30 | HIGH | WARRIOR, FAST_SCALPER, NEWS_TRADER, PROBE_ENTRY |
| OPEN_IGNITION | 9:30-9:45 | EXTREME | WARRIOR, FAST_SCALPER only |
| STRUCTURED_MOMENTUM | 9:45-11:30 | MEDIUM | ATS, PULLBACK_SCALPER, PROBE_ENTRY, NEWS_TRADER |
| MIDDAY_COMPRESSION | 11:30-14:00 | LOW | DEFENSIVE_SCALPER only |
| POWER_HOUR | 14:00-16:00 | MEDIUM | ATS, SWING_ENTRY, PULLBACK_SCALPER |
| AFTER_HOURS | 16:00-20:00 | MEDIUM | NEWS_TRADER only |
| CLOSED | 20:00-4:00 | LOW | None |

**Phase Configuration:**
```python
@dataclass
class PhaseConfig:
    name: str
    description: str
    start_time: str  # HH:MM (ET)
    end_time: str
    expected_volatility: str  # LOW, MEDIUM, HIGH, EXTREME
    expected_volume: str
    trend_reliability: str
    allowed_strategies: List[str]
    default_aggressiveness: int
    position_size_multiplier: float = 1.0
    stop_loss_multiplier: float = 1.0
    max_trades_per_phase: int = 10
```

---

### Task L: Strategy Compatibility Matrix

**Purpose:** Map phases to allowed strategies with risk adjustments.

**Matrix (in market_phases.py):**
| Phase | Max Trades | Position Size | Stop Multiplier |
|-------|------------|---------------|-----------------|
| OPEN_IGNITION | 5 | 50% | 2.0x (wider) |
| STRUCTURED_MOMENTUM | 15 | 100% | 1.0x |
| MIDDAY_COMPRESSION | 3 | 50% | 0.75x (tighter) |
| POWER_HOUR | 10 | 100% | 1.0x |
| AFTER_HOURS | 3 | 50% | 1.5x |

---

### Task M: Phase Evaluation Engine

**Purpose:** Determine current phase with optional metric-based overrides.

**Files Created:**
- `ai/phase_evaluator.py` - Phase evaluation logic

**Evaluation Logic:**
1. **Primary:** Time-based phase determination
2. **Secondary:** Metric-based overrides for extreme conditions

**Override Examples:**
- MIDDAY_COMPRESSION upgraded to STRUCTURED_MOMENTUM if:
  - Market breadth > 70%
  - Small-cap follow-through > 60%
  - Momentum count >= 5
- POWER_HOUR downgraded to MIDDAY_COMPRESSION if:
  - No momentum (count = 0)
  - Weak breadth < 40%

**Phase Lock:** 15 minutes minimum after evaluation.

---

### Task N: Strategy Enable/Disable Hooks

**Purpose:** Gate layer preventing unauthorized strategy signals.

**Files Created:**
- `ai/strategy_gate.py` - Strategy gating enforcement

**Gate Rules:**
1. All strategies MUST call `is_strategy_enabled()` before generating signals
2. Disabled strategies return `None` instead of signals
3. Suppressed signals are logged with reasons
4. Admin override available for testing

**Decorator Pattern:**
```python
@phase_gated("WARRIOR")
async def check_warrior_signal(symbol, quote):
    # Only executes if WARRIOR enabled in current phase
    return signal

@phase_gated_sync("DEFENSIVE_SCALPER")
def check_defensive_signal(symbol, quote):
    # Synchronous version
    return signal
```

**Gate Status:**
```python
{
    "enabled_strategies": ["WARRIOR", "FAST_SCALPER"],
    "suppressed_count": {"ATS": 15, "PULLBACK_SCALPER": 8},
    "override_enabled": False,
    "can_trade": True,
    "trades_this_phase": 3,
    "max_trades_this_phase": 5
}
```

---

### Task O: Phase Observability & UI

**Files Created:**
- `ai/phase_routes.py` - REST API endpoints

**API Endpoints:**
```
GET  /api/phase/status              - Full phase manager status
GET  /api/phase/current             - Current phase only
GET  /api/phase/strategies          - Enabled/disabled strategies
GET  /api/phase/gate                - Strategy gate status
GET  /api/phase/history             - Phase change history
GET  /api/phase/suppressed          - Suppressed signals
GET  /api/phase/evaluation          - Last evaluation result
POST /api/phase/evaluate            - Force phase evaluation
POST /api/phase/set                 - Manually set phase (admin)
POST /api/phase/gate/override       - Enable/disable strategy override
GET  /api/phase/dashboard           - Dashboard summary
GET  /api/phase/config              - All phase configurations
GET  /api/phase/config/{phase}      - Specific phase config
GET  /api/phase/check/{strategy}    - Check if strategy enabled
```

**Dashboard Response Example:**
```json
{
  "current_time_et": "10:15:32 ET",
  "current_phase": {
    "name": "STRUCTURED_MOMENTUM",
    "description": "Post-open momentum phase",
    "time_window": "09:45 - 11:30 ET",
    "volatility": "MEDIUM"
  },
  "phase_lock": {
    "locked": true,
    "remaining_seconds": 542,
    "reason": "Evaluation: Time-based (conf=90%)"
  },
  "trading_limits": {
    "trades_this_phase": 5,
    "max_trades": 15,
    "position_size_mult": 1.0,
    "can_trade": true
  },
  "strategies": [
    {"name": "ATS", "enabled": true, "overridden": false},
    {"name": "WARRIOR", "enabled": false, "overridden": false}
  ]
}
```

---

## Files Created This Session (Tasks G-O)

| File | Task | Purpose |
|------|------|---------|
| `ai/baseline_profiles.py` | G | Profile definitions (CONSERVATIVE/NEUTRAL/AGGRESSIVE) |
| `ai/market_condition_evaluator.py` | H | Market condition assessment |
| `ai/profile_selector.py` | I | Checkpoint-based profile selection |
| `ai/baseline_routes.py` | J | Baseline API endpoints |
| `ai/market_phases.py` | K, L | Phase definitions + strategy matrix |
| `ai/phase_evaluator.py` | M | Phase evaluation engine |
| `ai/strategy_gate.py` | N | Strategy gating enforcement |
| `ai/phase_routes.py` | O | Phase API endpoints |

## Files Modified This Session

| File | Change |
|------|--------|
| `morpheus_trading_api.py` | Added baseline routes, phase routes, startup events |

---

## Verification Commands

```bash
# Test Task G-J imports
python -c "
from ai.baseline_profiles import get_baseline_manager
from ai.market_condition_evaluator import get_condition_evaluator
from ai.profile_selector import get_profile_selector
from ai.baseline_routes import router as baseline_router
print('Tasks G-J: All imports OK!')
"

# Test Task K-O imports
python -c "
from ai.market_phases import get_phase_manager, MarketPhase, TradingStrategy
from ai.phase_evaluator import get_phase_evaluator
from ai.strategy_gate import get_strategy_gate, phase_gated
from ai.phase_routes import router as phase_router
print('Tasks K-O: All imports OK!')
"

# Check current phase (API)
curl http://localhost:9100/api/phase/current

# Check strategy status
curl http://localhost:9100/api/phase/strategies

# Check baseline profile
curl http://localhost:9100/api/baseline/profile

# Force phase evaluation
curl -X POST http://localhost:9100/api/phase/evaluate
```

---

## Non-Negotiables (Per ChatGPT Directive)

- Paper mode: **ON** (maintained)
- Gating remains final authority: **YES** (micro-override goes through gating)
- No uncontrolled loosening: **YES** (probes have rate limits + caps)
- Instrumentation first: **YES** (funnel metrics + all endpoints)
- Strategy gate enforcement: **YES** (all strategies must check gate)
- Phase-based risk adjustment: **YES** (position size + stop multipliers by phase)

---

## Momentum Discovery & Strategy Orchestration Calibration (Tasks P-T)

### Task P: Momentum Scout Mode (CRITICAL)

**Purpose:** Early momentum discovery through lightweight probe entries that do NOT require Chronos confirmation.

**Philosophy:** Probe → Validate → Expand
- Enter small (15-25% of normal size)
- If momentum continues (hold X bars + gain threshold), escalate to full strategy
- If stopped, cooldown applied, minimal capital lost

**Files Created:**
- `ai/momentum_scout.py` - Core scout mode implementation

**Scout Configuration:**
```python
@dataclass
class ScoutConfig:
    enabled: bool = True
    size_multiplier: float = 0.20      # 20% of normal position
    stop_loss_percent: float = 1.0     # Tight 1% stop
    max_per_symbol_per_session: int = 1
    max_per_hour: int = 5
    cooldown_minutes: int = 30
    confirm_hold_bars: int = 3         # Hold 3 bars minimum
    confirm_gain_percent: float = 0.5  # Need +0.5% gain
    allowed_triggers: List[str]        # PMH_BREAK, VWAP_RECLAIM, etc.
```

**Scout Triggers:**
| Trigger | Description |
|---------|-------------|
| PMH_BREAK | Pre-market high break |
| VWAP_RECLAIM | Price crosses above VWAP |
| HOD_BREAK | High of day break |
| VOLUME_SPIKE | 3x+ volume surge |
| RANGE_HIGH_1M | 1-minute range break |
| RANGE_HIGH_3M | 3-minute range break |
| MOMENTUM_CROSSOVER | MACD/RSI crossover |
| PULLBACK_BOUNCE | Bounce from VWAP/EMA support |

**Scout States:**
- `IDLE` - No scout activity
- `TRIGGERED` - Entry triggered, awaiting fill
- `ACTIVE` - Position active, monitoring
- `CONFIRMED` - Momentum held, ready for handoff
- `STOPPED` - Hit stop loss (triggers cooldown)
- `TIMED_OUT` - Max hold exceeded
- `HANDED_OFF` - Escalated to strategy

**Key Principle:** Chronos is IGNORED for scout entries. Only validates after confirmation.

---

### Task Q: Scout → Strategy Handoff

**Purpose:** Escalate successful scouts to full strategies (ATS or Scalper).

**Files Created:**
- `ai/scout_handoff.py` - Handoff management

**Handoff Conditions (ALL required):**
1. Scout held minimum bars (3 bars)
2. Scout gained minimum % (+0.5%)
3. ATS SmartZone rating is favorable (ENTERING or IN_SMARTZONE)
4. Volume continuation detected

**Handoff Targets:**
| Scout Status | Target Strategy | Chronos Required |
|--------------|-----------------|------------------|
| CONFIRMED + ATS favorable | ATS | Yes (now applies) |
| CONFIRMED + No ATS | FAST_SCALPER | Directional bias only |

**Handoff Process:**
```python
HandoffRequest:
    symbol: str
    scout_entry: ScoutEntry
    target_strategy: str
    chronos_required: bool
    position_to_scale: float  # Amount to add (25% → 75%)
```

**Size Scaling:**
- Scout entry: 20% of normal
- Handoff adds: 55% (to reach 75% total)
- Final position: 75% of normal (safer than 100%)

---

### Task R: Strategy-Specific Chronos Weighting

**Purpose:** Chronos applies differently per strategy type.

**Files Created:**
- `ai/chronos_strategy_weights.py` - Strategy-specific Chronos roles

**Chronos Roles:**
| Role | Behavior |
|------|----------|
| IGNORED | No influence at all (scouts) |
| DIRECTIONAL_BIAS | Affects direction, doesn't block |
| STRUCTURAL_CONFIRM | Required for structure confirmation |
| HARD_REQUIREMENT | Must pass or entry blocked |

**Strategy Configuration:**
| Strategy | Chronos Role | Entry Gate | Size Scaling | Exit Accel |
|----------|-------------|------------|--------------|------------|
| MOMENTUM_SCOUT | IGNORED | No | No | No |
| FAST_SCALPER | DIRECTIONAL_BIAS | No | Yes | Yes |
| WARRIOR | DIRECTIONAL_BIAS | No | Yes | Yes |
| ATS | STRUCTURAL_CONFIRM | 50% min | Yes | Yes |
| PULLBACK_SCALPER | STRUCTURAL_CONFIRM | 45% min | Yes | Yes |
| SWING_ENTRY | HARD_REQUIREMENT | 60% min | Yes | Yes |
| NEWS_TRADER | DIRECTIONAL_BIAS | No | Yes | Yes |

**Position Size Multipliers (based on confidence):**
- High (≥70%): 1.2x
- Medium (50-70%): 1.0x
- Low (<50%): 0.7x

**Hold Time Multipliers (based on regime):**
- Favorable (TRENDING_UP, RANGING): 1.5x (hold longer)
- Unfavorable (TRENDING_DOWN, VOLATILE): 0.5x (exit faster)

---

### Task S: Phase-Controlled Exploration Policy

**Purpose:** Market phase controls how exploratory the system is.

**Files Created:**
- `ai/exploration_policy.py` - Phase-controlled exploration

**Exploration Levels:**
| Level | Description |
|-------|-------------|
| DISABLED | No exploration (scouts off) |
| MINIMAL | Very limited exploration |
| NORMAL | Standard exploration |
| AGGRESSIVE | High exploration mode |

**Policy Matrix (Phase × Baseline):**
| Phase | CONSERVATIVE | NEUTRAL | AGGRESSIVE |
|-------|-------------|---------|------------|
| PRE_MARKET | MINIMAL (1/hr) | MINIMAL (2/hr) | NORMAL (3/hr) |
| OPEN_IGNITION | MINIMAL (2/hr) | NORMAL (4/hr) | AGGRESSIVE (6/hr) |
| STRUCTURED_MOMENTUM | MINIMAL (2/hr) | NORMAL (4/hr) | AGGRESSIVE (5/hr) |
| MIDDAY_COMPRESSION | DISABLED | DISABLED | MINIMAL (1/hr) |
| POWER_HOUR | MINIMAL (1/hr) | NORMAL (3/hr) | NORMAL (4/hr) |
| AFTER_HOURS | DISABLED | DISABLED | MINIMAL (1/hr) |

**Size Multipliers by Phase:**
- OPEN_IGNITION AGGRESSIVE: 25% of normal
- STRUCTURED_MOMENTUM NEUTRAL: 20% of normal
- MIDDAY_COMPRESSION: 10% of normal (if enabled)

**Phase Changes:**
- Log `MARKET_PHASE_EXPLORATION_POLICY` event
- Immediately enable/disable scout logic
- Update scout config (max per hour, size multiplier)

---

### Task T: Observability & Diagnostics

**Purpose:** Full visibility into scout operations.

**Files Created:**
- `ai/scout_routes.py` - 25+ API endpoints

**API Endpoints:**
```
# Scout Status & Control
GET  /api/scout/status                - Full scout system status
GET  /api/scout/stats                 - Scout statistics
GET  /api/scout/active                - Active scouts
GET  /api/scout/history               - Scout history
POST /api/scout/enable                - Enable scouts
POST /api/scout/disable               - Disable scouts
POST /api/scout/reset-session         - Reset session counters
GET  /api/scout/config                - Get configuration
POST /api/scout/config                - Update configuration

# Handoff Management
GET  /api/scout/handoffs/pending      - Pending handoffs
GET  /api/scout/handoffs/completed    - Completed handoffs
POST /api/scout/handoffs/process      - Process pending handoffs
GET  /api/scout/handoffs/config       - Handoff configuration
POST /api/scout/handoffs/config       - Update handoff config

# Exploration Policy
GET  /api/scout/exploration/status    - Exploration policy status
GET  /api/scout/exploration/policy    - Current policy
GET  /api/scout/exploration/matrix    - Full policy matrix
POST /api/scout/exploration/apply     - Apply policy to scout

# Chronos Strategy Weights
GET  /api/scout/chronos/weights       - All strategy weights
GET  /api/scout/chronos/weights/{strategy} - Single strategy weight
POST /api/scout/chronos/check         - Check Chronos decision

# Funnel & Diagnostics
GET  /api/scout/funnel                - Scout funnel metrics
GET  /api/scout/diagnostic            - Diagnostic info
GET  /api/scout/dashboard             - Dashboard summary
```

**Funnel Metrics Added:**
```python
# Scout-specific metrics
scout_attempts: int = 0
scout_confirmed: int = 0
scout_failed: int = 0
scout_to_trade: int = 0
scout_block_reasons: Dict[str, int]

# Recording functions
record_scout_attempt(symbol, trigger)
record_scout_confirmed(symbol, gain_pct)
record_scout_failed(symbol, reason)
record_scout_to_trade(symbol, target_strategy)
record_scout_blocked(symbol, reason)
```

**Dashboard Response:**
```json
{
  "scout_status": {
    "enabled": true,
    "active_count": 2,
    "session_attempts": 12,
    "session_confirmed": 4,
    "confirmation_rate": 33.3
  },
  "exploration_policy": {
    "phase": "OPEN_IGNITION",
    "level": "NORMAL",
    "max_scouts_per_hour": 4,
    "scouts_this_hour": 2
  },
  "handoffs": {
    "pending": 1,
    "completed_today": 3
  },
  "funnel_impact": {
    "scouts_to_trades_pct": 25.0,
    "top_block_reason": "RATE_LIMIT_EXCEEDED"
  }
}
```

---

## Files Created (Tasks P-T)

| File | Task | Purpose |
|------|------|---------|
| `ai/momentum_scout.py` | P | Scout mode implementation |
| `ai/scout_handoff.py` | Q | Scout → Strategy handoff |
| `ai/chronos_strategy_weights.py` | R | Strategy-specific Chronos weights |
| `ai/exploration_policy.py` | S | Phase-controlled exploration |
| `ai/scout_routes.py` | T | 25+ Scout API endpoints |

## Files Modified (Tasks P-T)

| File | Change |
|------|--------|
| `ai/funnel_metrics.py` | Added scout metrics + recording methods |
| `morpheus_trading_api.py` | Added scout routes |

---

## Verification Commands (Tasks P-T)

```bash
# Test Task P-T imports
python -c "
from ai.momentum_scout import get_momentum_scout, ScoutState, ScoutTrigger
from ai.scout_handoff import get_handoff_manager, HandoffCondition
from ai.chronos_strategy_weights import get_chronos_strategy_manager, ChronosRole
from ai.exploration_policy import get_exploration_manager, ExplorationLevel
from ai.funnel_metrics import record_scout_attempt, record_scout_confirmed
from ai.scout_routes import router as scout_router
print('Tasks P-T: All imports OK!')
"

# Check scout status (API)
curl http://localhost:9100/api/scout/status

# Check exploration policy
curl http://localhost:9100/api/scout/exploration/policy

# Check Chronos weights
curl http://localhost:9100/api/scout/chronos/weights

# Check scout funnel metrics
curl http://localhost:9100/api/scout/funnel

# Check dashboard summary
curl http://localhost:9100/api/scout/dashboard
```

---

## Non-Negotiables (Tasks P-T - Per ChatGPT Directive)

- Paper mode: **ON** (maintained)
- Max risk per trade: **UNCHANGED** (scouts use 20% of normal size)
- Kill-switches: **UNCHANGED** (circuit breaker still active)
- All behavior observable: **YES** (25+ endpoints + funnel metrics)
- Chronos NOT blocking scouts: **YES** (IGNORED role for MOMENTUM_SCOUT)
- Phase controls exploration: **YES** (disabled in MIDDAY_COMPRESSION)
- Scout size capped: **YES** (15-25% of normal, never more)
- Rate limits enforced: **YES** (max per hour, max per symbol)

---

**End of Report**
