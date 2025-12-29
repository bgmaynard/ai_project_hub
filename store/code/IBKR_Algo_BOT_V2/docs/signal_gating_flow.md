# Signal Gating Flow

## Overview

The Signal Gating Engine enforces a strict validation pipeline for all trade signals. No trade can execute without passing through this gate.

## Flow Diagram

```
                           TRADE TRIGGER
                          (News, Scanner, Manual)
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │  1. FIND SIGNAL CONTRACT    │
                    │     signal_contracts.json   │
                    └──────────────┬──────────────┘
                                   │
                          ┌────────┴────────┐
                          │                 │
                       FOUND            NOT FOUND
                          │                 │
                          │                 ▼
                          │         ┌───────────────┐
                          │         │ VETO:         │
                          │         │ NO_CONTRACT   │
                          │         └───────────────┘
                          │
                          ▼
                    ┌─────────────────────────────┐
                    │  2. CHECK CONTRACT EXPIRY   │
                    │     expires_at > now?       │
                    └──────────────┬──────────────┘
                                   │
                          ┌────────┴────────┐
                          │                 │
                        VALID           EXPIRED
                          │                 │
                          │                 ▼
                          │         ┌───────────────┐
                          │         │ VETO:         │
                          │         │ CONTRACT_     │
                          │         │ EXPIRED       │
                          │         └───────────────┘
                          │
                          ▼
                    ┌─────────────────────────────┐
                    │  3. GET CHRONOS CONTEXT     │
                    │     (Regime, Confidence)    │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │  4. CHECK INVALID REGIMES   │
                    │     regime in invalid_list? │
                    └──────────────┬──────────────┘
                                   │
                          ┌────────┴────────┐
                          │                 │
                         NO               YES
                          │                 │
                          │                 ▼
                          │         ┌───────────────┐
                          │         │ VETO:         │
                          │         │ INVALID_      │
                          │         │ REGIME        │
                          │         └───────────────┘
                          │
                          ▼
                    ┌─────────────────────────────┐
                    │  5. CHECK VALID REGIMES     │
                    │     regime in valid_list?   │
                    └──────────────┬──────────────┘
                                   │
                          ┌────────┴────────┐
                          │                 │
                        YES               NO
                          │                 │
                          │                 ▼
                          │         ┌───────────────┐
                          │         │ VETO:         │
                          │         │ REGIME_       │
                          │         │ MISMATCH      │
                          │         └───────────────┘
                          │
                          ▼
                    ┌─────────────────────────────┐
                    │  6. CHECK CONFIDENCE        │
                    │     conf >= required?       │
                    └──────────────┬──────────────┘
                                   │
                          ┌────────┴────────┐
                          │                 │
                        YES               NO
                          │                 │
                          │                 ▼
                          │         ┌───────────────┐
                          │         │ VETO:         │
                          │         │ CONFIDENCE_   │
                          │         │ LOW           │
                          │         └───────────────┘
                          │
                          ▼
                    ┌─────────────────────────────┐
                    │  7. CHECK SYMBOL COOLDOWN   │
                    │     traded recently?        │
                    └──────────────┬──────────────┘
                                   │
                          ┌────────┴────────┐
                          │                 │
                         NO               YES
                          │                 │
                          │                 ▼
                          │         ┌───────────────┐
                          │         │ VETO:         │
                          │         │ COOLDOWN_     │
                          │         │ ACTIVE        │
                          │         └───────────────┘
                          │
                          ▼
                    ┌─────────────────────────────┐
                    │  8. CHECK RISK LIMITS       │
                    │     within daily limits?    │
                    └──────────────┬──────────────┘
                                   │
                          ┌────────┴────────┐
                          │                 │
                        YES               NO
                          │                 │
                          │                 ▼
                          │         ┌───────────────┐
                          │         │ VETO:         │
                          │         │ RISK_LIMIT_   │
                          │         │ EXCEEDED      │
                          │         └───────────────┘
                          │
                          ▼
              ┌───────────────────────────────────┐
              │           APPROVED                │
              │   Generate ExecutionRequest       │
              │   Send to Risk Engine → Execute   │
              └───────────────────────────────────┘
```

## Validation Steps

### Step 1: Find Signal Contract
- Lookup `signal_id` in `signal_contracts.json`
- If not found: `VETO (NO_CONTRACT)`

### Step 2: Check Contract Expiry
- Compare `expires_at` to current timestamp
- If expired: `VETO (CONTRACT_EXPIRED)`

### Step 3: Get Chronos Context
- Call `ChronosAdapter.get_context(symbol)`
- Returns: regime, confidence, volatility, trend

### Step 4: Check Invalid Regimes
- If current regime in `invalid_regimes`: `VETO (INVALID_REGIME)`
- Example: LONG signal vetoed in TRENDING_DOWN

### Step 5: Check Valid Regimes
- If current regime NOT in `valid_regimes`: `VETO (REGIME_MISMATCH)`
- Example: Signal only valid in TRENDING_UP, but market is RANGING

### Step 6: Check Confidence
- If `regime_confidence < confidence_required`: `VETO (CONFIDENCE_LOW)`
- Ensures high-conviction regime classification

### Step 7: Check Symbol Cooldown
- If symbol traded within cooldown period: `VETO (COOLDOWN_ACTIVE)`
- Default cooldown: 5 minutes

### Step 8: Check Risk Limits
- Daily loss limit exceeded: `VETO (RISK_LIMIT_EXCEEDED)`
- Max positions exceeded: `VETO (RISK_LIMIT_EXCEEDED)`
- Max daily trades exceeded: `VETO (RISK_LIMIT_EXCEEDED)`

## Veto Codes

| Code | Meaning | Resolution |
|------|---------|------------|
| `NO_CONTRACT` | No signal contract exists | Export from Qlib research |
| `CONTRACT_EXPIRED` | Contract past expiry date | Re-run research, export new contract |
| `INVALID_REGIME` | Market in dangerous regime | Wait for regime change |
| `REGIME_MISMATCH` | Market not in favorable regime | Wait for regime change |
| `CONFIDENCE_LOW` | Regime detection uncertain | Wait for clearer signal |
| `COOLDOWN_ACTIVE` | Recently traded this symbol | Wait for cooldown |
| `RISK_LIMIT_EXCEEDED` | Daily limits hit | Stop trading for day |

## Execution Request

When APPROVED, generates:

```python
@dataclass
class ExecutionRequest:
    signal_id: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: int
    max_slippage: float
    time_in_force: str  # DAY, IOC, GTC
    timestamp: str
```

## Audit Log

Every gate decision is logged:

```json
{
  "timestamp": "2024-12-28T10:30:00Z",
  "signal_id": "AAPL_MOMENTUM_LONG_2024",
  "symbol": "AAPL",
  "trigger_type": "news",
  "decision": "VETOED",
  "reason": "REGIME_MISMATCH",
  "context": {
    "current_regime": "VOLATILE",
    "valid_regimes": ["TRENDING_UP", "RANGING"],
    "confidence": 0.72
  }
}
```

## API Endpoints

```
POST /api/gating/evaluate
Body: { "symbol": "AAPL", "trigger_type": "news" }
Response: {
  "approved": false,
  "reason": "REGIME_MISMATCH",
  "signal_id": "AAPL_MOMENTUM_LONG_2024",
  "context": { ... }
}

GET /api/gating/audit?symbol=AAPL&limit=50
Response: [ ... audit log entries ... ]
```

## Key Principles

1. **Fail Closed** - If any check fails, trade is vetoed
2. **Full Audit** - Every decision logged with context
3. **No Bypass** - All trades must pass through gate
4. **Separation** - Chronos = context only, Qlib = research only
