# JAN 2 RUNBOOK - Warrior Discovery Validation
**Status:** LOCKED
**Mode:** Schwab-Only, Paper Trading
**Session Type:** Clean Morning Discovery Validation

---

## PURPOSE

Validate:
- Scanner → Watchlist flow
- Momentum stability
- Pullback timing accuracy
- ENTRY_WINDOW placement

## EXPLICIT NON-GOALS

- P&L evaluation
- Strategy tuning
- Threshold changes
- Auto-execution

---

## SCHEDULE

### 04:00 ET - System Reset
```bash
# Reset scanners
curl -X POST http://localhost:9100/api/scanners/reset-daily

# Clear candidates
curl -X POST http://localhost:9100/api/scanners/clear

# Purge MomentumWatchlist
curl -X POST http://localhost:9100/api/watchlist/purge

# Confirm paper account
curl http://localhost:9100/api/scanner/scalper/status | jq '.paper_mode'

# Schwab health gate - MUST PASS
curl http://localhost:9100/api/scanners/status | jq '.scanner_status'
```

### 04:00-07:00 ET - GAPPER Only
- Log gap candidates
- No feed to watchlist
- Observe only

### 07:00-09:15 ET - GAPPER + GAINER
- Manual feed to watchlist:
  ```bash
  curl -X POST http://localhost:9100/api/scanners/feed-watchlist
  ```
- Observe dominance stability
- Log candidate quality

### 09:15-09:30 ET - HOD Monitoring
- HOD scanner active
- EXPANSION events only
- No pullback entries

### After 09:30 ET - Pullback Tracking
- First pullback tracking begins
- ENTRY_WINDOW logging only
- NO execution

### Hard Stop
- Preserve all logs
- No reruns
- Session complete

---

## CHANGE CONTROL

### DO NOT:
- Adjust thresholds
- Expand price range
- Loosen rel-vol
- Blend data sources
- Use Finviz for logic
- Evaluate success by P&L

**Any change requires post-session review.**

---

## VALIDATION CHECKS

| Time | Check | Expected |
|------|-------|----------|
| 04:00 | Schwab healthy | `true` |
| 04:00 | Paper mode | `true` |
| 04:00 | Scanners reset | 0 candidates |
| 07:00 | GAPPER has candidates | > 0 |
| 07:00 | GAINER activates | `active: true` |
| 09:15 | HOD activates | `active: true` |
| 09:30 | Pullback FSM running | States logged |

---

## RESUME INSTRUCTION

If context is lost, start with:

> "Resume from Warrior Discovery Validation — Schwab-only, Option 1, Jan 2 pending."

This restores full project context.

---

## FILES TO PRESERVE

After session, save:
- `/api/scanners/status` snapshot
- `/api/watchlist/status` snapshot
- ENTRY_WINDOW logs
- Pullback state transitions
- Any errors or anomalies

---

**LOCKED - DO NOT MODIFY UNTIL POST-SESSION REVIEW**
