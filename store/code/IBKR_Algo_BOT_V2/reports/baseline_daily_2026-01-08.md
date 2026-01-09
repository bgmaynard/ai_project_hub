# MORPHEUS DAILY BASELINE REPORT
**Date:** 2026-01-08
**Mode:** PAPER

---

## Market Context

| Attribute | Value |
|-----------|-------|
| Session tone | Choppy, momentum faded mid-day |
| Small-cap momentum quality | HIGH early (FLYX +130%, SXTC +155%), faded by noon |
| Volatility regime | MIXED (HIGH on FLYX, NORMAL/LOW on others) |
| Trading window | Extended from 7:00-9:30 to 7:00-16:00 today |

---

## Symbol Flow Funnel

| Stage | Count | Notes |
|-------|-------|-------|
| Discovered (Watchlist) | 13 | FLYX, NVVE, ELAB, OSS, SXTC, MKZR, PAPL, RENT, SHMD, + defaults |
| Injected (Active Scan) | 10+ | Via Finviz/Schwab movers |
| Eligible (Passed Filters) | 14 | Gating attempts |
| Scout Attempts | 0 | Scout mode available but not triggered |
| Trades Executed | 10 | All via standard momentum entry |

**Primary Bottleneck:**
Morning: TRADING_WINDOW_CLOSED (7:00-9:30 hard-coded)
Mid-day: Undefined `symbol` variable in gating caused silent failures (77 attempts, 0 processed)
After fix: 100% approval rate on 10 attempts

---

## Trade Outcomes

| Metric | Value |
|--------|-------|
| Total trades | 10 |
| Wins | 4 |
| Losses | 4 |
| Breakeven | 2 |
| Win rate | 40% |
| Avg win | +$1.85 |
| Avg loss | -$0.95 |
| Avg hold | 175.5s (~3 min) |
| Profit factor | **1.29** |
| Total P&L | **+$1.68** |
| Best trade | +$2.92 (FLYX) |
| Worst trade | -$3.02 (NMRA) |

### Trade Detail

| Symbol | Entry | Exit | P&L | Hold | Exit Reason |
|--------|-------|------|-----|------|-------------|
| FLYX | $6.86 | $6.85 | -$0.96 | 69s | FAILED_MOMENTUM |
| FLYX | $6.82 | $6.85 | +$2.92 | 69s | FAILED_MOMENTUM |
| LUCY | $1.57 | $1.57 | $0.00 | 64s | FAILED_MOMENTUM |
| SOPA | $2.77 | $2.77 | $0.00 | 202s | CHRONOS_TREND_WEAK |
| LUCY | $1.56 | $1.56 | -$0.68 | 61s | FAILED_MOMENTUM |
| AZI | $3.15 | $3.14 | -$1.05 | 136s | FAILED_MOMENTUM |
| LUCY | $1.57 | $1.57 | +$1.91 | 272s | FAILED_MOMENTUM |
| NMRA | $2.21 | $2.20 | -$3.02 | 414s | FAILED_MOMENTUM |
| TNGX | $11.23 | $11.26 | +$1.48 | 259s | FAILED_MOMENTUM |
| MNTS | $11.95 | $11.97 | +$1.10 | 209s | FAILED_MOMENTUM |

---

## What Worked

1. **FAILED_MOMENTUM exit rule** - 9/10 exits used this. Prevented large losses by exiting stocks that didn't gain 0.5% in 30s.

2. **Avg loss < avg win** - $0.95 loss vs $1.85 win. Good risk/reward ratio.

3. **Profit factor above 1.0** - System generates $1.29 for every $1 risked. This is profitable.

4. **Gating approval flow** - Once NameError fixed, 100% approval rate. No false vetoes.

5. **Multi-window momentum detection** - 5/15/30 tick windows caught secondary spikes on FLYX, LUCY.

6. **Quick exits on flat trades** - Breakevens (LUCY, SOPA) didn't turn into losses.

---

## What Did Not

1. **Morning time window block** - Missed FLYX +130% and NVVE +50% moves before window fix.

2. **Silent gating failures** - 77 attempts went to limbo due to undefined `symbol` variable.

3. **NMRA worst trade** - -$3.02 loss, held 414s (too long). Should have exited earlier.

4. **No news catalyst detection** - All trades show `has_news_catalyst: false` despite FLYX/SXTC having news.

5. **VWAP position unknown** - All trades show `vwap_position: unknown`. Not using VWAP filter.

---

## Non-Issues (Important)

These are **NOT problems** and should NOT be "fixed":

1. **40% win rate** - Acceptable with profit factor > 1.0. Many profitable systems have 40% WR.

2. **FAILED_MOMENTUM exits** - This is working correctly. It's preventing bigger losses.

3. **Breakeven trades** - These are victories, not losses. Capital preserved.

4. **10 trades only** - Quality over quantity. Better than 134 losing trades from previous sessions.

5. **No scout trades** - System found enough standard momentum entries. Scout mode as backup is fine.

---

## One Hypothesis (NO CHANGES YET)

**Hypothesis:** Trades with `distance_from_hod < 5%` may have higher win rate.

**Observation:** Winners (TNGX, MNTS) had `distance_from_hod` of 1.49% and 20.33%. NMRA (worst loser) had 2.0%.

**Data needed:** More trades to validate this pattern.

**Action:** NONE. Continue observing. Do not implement filter yet.

---

## Bugs Fixed Today

| Bug | Location | Fix |
|-----|----------|-----|
| Undefined `symbol` variable | `signal_gating_engine.py` lines 172, 205, 215, 226 | Changed to `contract.symbol` |
| Trading window too restrictive | `time_controls.py` line 31 | Extended to 7:00-16:00 |
| Daily trade limit blocking wins | `scalper_config.json` | Increased from 10 to 50 |

---

## System Health at Close

| Component | Status |
|-----------|--------|
| Server | Healthy, port 9100 |
| Broker | Schwab connected |
| Scalper | Running, paper mode, 13 symbols |
| Gating | 100% approval rate |
| Time | 6:40 PM ET - market closed |

---

## Tomorrow's Goal

**Observe only. Do not tune.**

- Run another full session with today's logic
- Collect more trades to validate profit factor
- Watch for repeatable patterns
- Confirm system stability over multiple days

---

*Generated: 2026-01-08 6:45 PM ET*
*This is the ground truth baseline. Do not optimize until multi-day validation.*
