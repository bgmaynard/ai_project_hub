# Morning Validation Report - January 9, 2026

## Session Overview
- **Date:** Friday, January 9, 2026
- **Window:** Pre-Market (4:00 AM - 9:30 AM ET)
- **Mode:** Paper Trading
- **Starting Balance:** $1,000.00

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Trades | 50 |
| Winners | 23 |
| Losers | 27 |
| Win Rate | 46.0% |
| Total P&L | -$80.22 |
| Avg Win | +$4.93 |
| Avg Loss | -$7.17 |
| Best Trade | +$22.61 |
| Worst Trade | -$40.00 |
| Profit Factor | 0.59 |
| Avg Hold Time | 163.3 seconds |

## Exit Reason Analysis (CRITICAL)

### Winners (100% Win Rate)
| Exit Type | Count | P&L | Avg P&L | Avg MFE | Symbols |
|-----------|-------|-----|---------|---------|---------|
| **TRAILING_STOP** | 2 | +$32.93 | +$16.46 | 3.85% | OPAD, CETX |
| **MAX_HOLD_TIME** | 8 | +$34.18 | +$4.27 | 1.60% | SXTC, OPAD, LRHC, CETX |
| **REVERSAL_DETECTED** | 1 | +$7.04 | +$7.04 | 1.06% | GP |
| **CHRONOS_MOMENTUM_FADING** | 2 | +$3.35 | +$1.68 | 1.14% | OPAD, CETX |

### Losers (Problem Exits)
| Exit Type | Count | P&L | Win Rate | Avg MAE | Issue |
|-----------|-------|-----|----------|---------|-------|
| **VWAP_STOP** | 10 | -$86.28 | 20% | 1.63% | BIGGEST LOSER - Exits too early |
| **FAILED_MOMENTUM** | 20 | -$39.20 | 35% | 0.42% | Too aggressive 30s check |
| **CHRONOS_TREND_WEAK** | 7 | -$32.24 | 14% | 0.76% | Exiting on minor weakness |

## Symbol Performance

### Profitable Symbols
| Symbol | Trades | Win Rate | P&L | Notes |
|--------|--------|----------|-----|-------|
| CETX | 9 | 66.7% | +$28.41 | Best performer |
| OPAD | 5 | 60.0% | +$22.42 | Good continuation plays |
| WHLR | 2 | 100% | +$6.37 | Perfect 2/2 |
| TNGX | 1 | 100% | +$1.48 | |
| MNTS | 1 | 100% | +$1.10 | |

### Losing Symbols
| Symbol | Trades | Win Rate | P&L | Issue |
|--------|--------|----------|-----|-------|
| LRHC | 4 | 25% | -$52.40 | VWAP_STOP killed it |
| KUST | 3 | 0% | -$21.84 | No winners |
| GP | 3 | 66.7% | -$21.54 | One big VWAP_STOP loss (-$36) |
| IBIO | 3 | 66.7% | -$20.01 | One big VWAP_STOP loss |
| FEED | 5 | 0% | -$11.27 | Never worked |

## Time of Day Analysis

| Session | Trades | Win Rate | P&L |
|---------|--------|----------|-----|
| Pre-Market Early (4-7 AM) | 10 | 70.0% | -$24.91 |
| Pre-Market Late (7-9:30 AM) | 30 | 40.0% | -$57.00 |
| Morning (9:30-10 AM) | 3 | 33.3% | +$1.96 |
| Midday | 7 | 42.9% | -$0.28 |

**Observation:** Pre-market EARLY had 70% win rate but still lost money due to big losses. Pre-market LATE is where most losses occurred.

## MFE/MAE Analysis

| Metric | Value |
|--------|-------|
| Avg MFE (all trades) | 0.61% |
| Avg MAE (all trades) | 0.61% |
| Winners Avg MFE | 1.27% |
| Losers Avg MAE | 1.11% |
| Trades reaching 1%+ | 11 |
| Trades reaching 2%+ | 4 |

**Insight:** Winners only capture 1.27% MFE on average - money being left on table. TRAILING_STOP trades captured 3.85% MFE average.

## Hold Time Analysis

| Metric | Value |
|--------|-------|
| Avg Hold (all) | 163.3s |
| Avg Hold (winners) | 202.5s |
| Avg Hold (losers) | 129.9s |
| Shortest Winner | 14s |
| Longest Loser | 520s |

**Insight:** Winners hold longer (202s) than losers (130s). Losers are cut faster but still losing. The 520s loser suggests MAX_HOLD_TIME (180s) is being bypassed sometimes.

## Critical Issues Identified

### 1. VWAP_STOP is Destroying Performance
- **10 trades, -$86.28 total, 20% win rate**
- Single GP trade lost -$36.07
- Single LRHC trade lost -$40.00
- VWAP is being used as a stop but it's triggering on normal volatility

### 2. FAILED_MOMENTUM Too Aggressive
- **20 trades (40% of all trades), 35% win rate, -$39.20**
- 30-second momentum check is cutting winners
- MFE only 0.11% before exit - not giving trades room to work

### 3. CHRONOS_TREND_WEAK Premature
- **7 trades, 14% win rate, -$32.24**
- Exiting on minor trend weakness
- Winners avg 1.27% MFE but CHRONOS cuts at 0.22% MFE

### 4. Avg Loss > Avg Win
- Avg loss: $7.17
- Avg win: $4.93
- Need 59% win rate just to break even
- Current win rate: 46%

## Tuning Recommendations

### Priority 1: Fix VWAP_STOP (Highest Impact)
```
Current: VWAP_STOP triggers on any break below VWAP
Recommendation:
- Add buffer (0.5% below VWAP before triggering)
- Or disable VWAP_STOP entirely, use ATR stops instead
- VWAP_STOP alone cost -$86.28 today
```

### Priority 2: Relax FAILED_MOMENTUM Check
```
Current: Exit if no +0.5% gain in 30 seconds
Recommendation:
- Extend to 60 seconds
- Or reduce threshold to +0.3%
- Or require 3 consecutive flat checks before exit
```

### Priority 3: Tune CHRONOS_TREND_WEAK
```
Current: 14% win rate suggests it's too sensitive
Recommendation:
- Require trend_strength < 0.2 (currently probably 0.3)
- Or require 2 consecutive weak readings
- Or disable and rely on other exit signals
```

### Priority 4: Let Winners Run
```
Current: Winners avg 1.27% MFE, TRAILING_STOP winners avg 3.85%
Recommendation:
- Increase trailing trigger from 1.5% to 2.5%
- Widen trailing distance from 0.75% to 1.0%
- Let MAX_HOLD_TIME do its job (it's 100% WR)
```

## What's Working

1. **TRAILING_STOP** - 100% win rate, +$16.46 avg, captures 3.85% MFE
2. **MAX_HOLD_TIME** - 100% win rate, +$4.27 avg, lets trades mature
3. **REVERSAL_DETECTED** - 100% win rate, catches actual tops
4. **CETX trades** - 66.7% WR, +$28.41 total, good momentum stock
5. **OPAD trades** - 60% WR, +$22.42 total, consistent performer

## Config Changes to Test

```json
{
  "use_vwap_trailing_stop": false,  // Disable VWAP stops
  "momentum_check_seconds": 60,      // Was 30
  "expected_gain_30s": 0.3,          // Was 0.5
  "chronos_trend_threshold": 0.2,    // More lenient
  "profit_target_percent": 2.5,      // Was 1.5
  "trailing_stop_percent": 1.0       // Was 0.75
}
```

## Questions for Next Session

1. Should VWAP_STOP be disabled entirely given 20% win rate?
2. Is 30-second FAILED_MOMENTUM check too aggressive for pre-market volatility?
3. Should different parameters be used for pre-market vs market hours?
4. Why did pre-market EARLY (70% WR) still lose money while pre-market LATE (40% WR) lost more?

## Raw Data Available

- Trade log: `/api/scanner/scalper/trades`
- Analytics: `/api/paper/analytics`
- CSV Export: `/api/paper/export/csv`

---
Generated: 2026-01-09 08:51 AM ET
Session: Pre-Market Paper Trading
Bot Version: Morpheus Trading Bot v2.1.0
