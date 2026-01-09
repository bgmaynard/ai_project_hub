# Dynamic Risk Throttle Design Document

**Date:** 2026-01-08
**Status:** DESIGN ONLY (No implementation tonight)
**Author:** Claude Code (Post-Momentum Day Analysis)

---

## Problem Statement

The current `max_daily_trades: 10` hard limit blocked profitable opportunities when the system was winning (FLYX 30-cent spike missed due to trade count).

**User Quote:** "Limit losses, not wins - if you are still winning trades then we should be making all of the money we can."

---

## Current Behavior

```
max_daily_trades = 10  (hard limit)
max_daily_loss = $200  (hard limit)
```

When either limit is hit, ALL trading stops for the day - regardless of P&L trajectory.

---

## Proposed Behavior: Dynamic Risk Throttle

### Core Principle

**Throttle based on P&L, not trade count.**

```
if daily_pnl > 0:     # WINNING
    → Full freedom (no trade count limit)

if daily_pnl < -$50:  # LOSING
    → Reduce max trades to 5

if daily_pnl < -$100: # DANGER
    → Halt trading for the day
```

---

## Proposed Configuration

```json
{
  "use_dynamic_risk_throttle": true,

  "throttle_tiers": [
    {
      "name": "WINNING",
      "pnl_threshold": 0,
      "max_trades": 100,
      "position_size_pct": 100,
      "message": "Green day - full freedom"
    },
    {
      "name": "BREAK_EVEN",
      "pnl_threshold": -25,
      "max_trades": 20,
      "position_size_pct": 100,
      "message": "Near breakeven - normal operation"
    },
    {
      "name": "CAUTION",
      "pnl_threshold": -50,
      "max_trades": 10,
      "position_size_pct": 75,
      "message": "Red zone - reduced activity"
    },
    {
      "name": "DANGER",
      "pnl_threshold": -100,
      "max_trades": 5,
      "position_size_pct": 50,
      "message": "Danger zone - minimal trades"
    },
    {
      "name": "HALT",
      "pnl_threshold": -150,
      "max_trades": 0,
      "position_size_pct": 0,
      "message": "Trading halted - max loss reached"
    }
  ],

  "cooldown_on_tier_change": 60,
  "log_tier_transitions": true
}
```

---

## Logic Flow

```
check_entry_signal():
    1. Calculate current daily_pnl
    2. Determine current tier based on pnl_threshold
    3. Check if daily_trades >= tier.max_trades
       → If yes: Block entry (log tier name)
    4. Adjust position size by tier.position_size_pct
    5. Log tier state for diagnostics
```

---

## Transition Behavior

| From | To | Trigger | Action |
|------|-----|---------|--------|
| WINNING | BREAK_EVEN | P&L drops below $0 | Reduce max trades to 20 |
| BREAK_EVEN | CAUTION | P&L drops below -$50 | Reduce to 10 trades, 75% size |
| CAUTION | DANGER | P&L drops below -$100 | Reduce to 5 trades, 50% size |
| DANGER | HALT | P&L drops below -$150 | Stop all trading |
| HALT | DANGER | P&L recovers above -$150 | Resume with 5 trade max |

---

## Why This Replaces Hard Trade Counts

### Old System
- `max_daily_trades: 10` means 11th trade is blocked even if up +$500
- No differentiation between winning and losing days

### New System
- Winning day: Can take 100 trades (capture all opportunities)
- Losing day: Automatically throttled to protect capital
- Self-adjusting based on actual performance

---

## Edge Cases

### What if we win 10 trades then hit a losing streak?

Tier transitions are based on **cumulative P&L**, not streak.

Example:
```
Trade 1-5: +$50 total → WINNING tier (100 max)
Trade 6-10: -$80 → Now at -$30 → BREAK_EVEN tier (20 max)
Trade 11-15: -$50 → Now at -$80 → CAUTION tier (10 max)
```

Throttling engages **as losses accumulate**, not based on streak length.

### What about tier bouncing?

To prevent thrashing near tier boundaries, add hysteresis:

```python
# Only transition down if below threshold by 10%
# Only transition up if above threshold by 20%
DOWN_HYSTERESIS = 0.10
UP_HYSTERESIS = 0.20
```

---

## Integration Points

### Where to add (NO CODE TONIGHT)

1. `HFTScalper.check_entry_signal()` - Check tier before entry
2. `HFTScalper.get_current_tier()` - New method to determine tier
3. `ScalperConfig` - Add throttle_tiers configuration
4. `get_status()` - Include current tier in diagnostics

### API Endpoints (Future)

```
GET /api/scanner/scalper/risk-tier  - Get current tier
GET /api/scanner/scalper/risk-history - Tier transitions today
```

---

## Alternatives Considered

### Option A: Fixed Trade Count (Current)
- Simple but inflexible
- Penalizes winning days equally as losing days
- **Rejected:** Does not align with user goals

### Option B: Trailing Stop on P&L
- Halt if P&L drops X% from high-water mark
- Complex to implement, harder to reason about
- **Rejected:** Overengineered for current needs

### Option C: Tiered Throttle (Proposed)
- Clear tiers with defined behavior
- Easy to understand and configure
- Aligns with "limit losses, not wins"
- **Selected**

---

## Implementation Priority

| Priority | Item | Notes |
|----------|------|-------|
| P1 | Define tier thresholds | $0 / -$50 / -$100 / -$150 |
| P2 | Add tier check in entry flow | Before position sizing |
| P3 | Log tier transitions | For diagnostics |
| P4 | UI indicator | Show current tier in dashboard |
| P5 | Hysteresis logic | Prevent tier bouncing |

---

## Testing Plan (When Implemented)

1. **Unit tests:** Verify tier calculation at boundary values
2. **Backtest:** Apply to historical trade data, compare outcomes
3. **Paper trade:** Run one full session with tiered throttle
4. **Review:** Analyze if throttle engaged appropriately

---

## Decision

**DO NOT IMPLEMENT TONIGHT.**

This document is for ChatGPT review and future implementation.

System is in observation mode. No strategy changes until multi-day validation.

---

*Design Document - v1.0*
*Created during post-momentum stabilization phase*
