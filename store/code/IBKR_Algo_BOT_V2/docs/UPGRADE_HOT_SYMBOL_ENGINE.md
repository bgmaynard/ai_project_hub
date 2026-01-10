# UPGRADE SHEET - FREE HOT SYMBOL ENGINE

**Project:** AI_Project_Hub / Morpheus Trading Bot
**Target:** Replace slow breaking-news detection with instant market-reaction hot symbol detection
**Created:** 2026-01-10

## Constraints

- No IBKR
- No paid news feeds (Benzinga, Reuters, etc.)
- Free public data only
- Event-driven, low-latency architecture

---

## 1. Problem Statement

The current breaking-news detector is headline-first and poll-based, causing unacceptable latency.
By the time a symbol is identified, the first tradable momentum window is gone.

We must switch to a **market-reaction-first architecture** that:
- Detects abnormal price/volume behavior
- Promotes symbols instantly into monitoring
- Uses news only as enrichment, never as a blocker

---

## 2. Design Goal

A symbol must enter active monitoring **within seconds of abnormal market behavior**, even if no headline has been processed yet.

**Success criteria:**
- Hot symbol injection <= 1-2 minutes from move start
- Zero dependency on paid APIs
- Modular, event-driven, FSM-compatible
- Deterministic, debuggable, backtestable

---

## 3. High-Level Architecture

```
+------------------------------+
| PRICE / VOLUME SHOCK SCANNER |  <- PRIMARY TRIGGER
| (Yahoo / Polygon / Schwab)   |
+-------------+----------------+
              |
              v
+------------------------------+
| CROWD SIGNAL SCANNER         |  <- SECONDARY CONFIRM
| (StockTwits / Reddit)        |
+-------------+----------------+
              |
              v
       HOT SYMBOL QUEUE
       (TTL 120-300 seconds)
              |
              v
+------------------------------+
| MOMENTUM FSM / AI MONITOR    |
| (confirm -> arm -> trade)    |
+-------------+----------------+
              |
              v
+------------------------------+
| ASYNC NEWS ENRICHMENT        |  <- LABEL ONLY
| (SEC / PR / Halts)           |
+------------------------------+
```

---

## 4. Core Module: HotSymbolQueue

### Purpose
Acts as a temporary promotion layer for symbols experiencing abnormal behavior.

### Requirements
- TTL-based (default: 180s)
- Deduplicated
- Priority-aware
- Reason-tagged

### Data Model
```python
HotSymbol = {
    "symbol": "AMD",
    "first_seen": datetime,
    "last_update": datetime,
    "reasons": ["PRICE_SPIKE", "VOLUME_SHOCK"],
    "confidence": 0.0-1.0,
    "ttl": 180
}
```

### Behavior
- Re-triggering the same symbol:
  - Refreshes TTL
  - Increases confidence
- Expired symbols auto-evict

---

## 5. Price / Volume Shock Detector (PRIMARY)

### Purpose
Detect abnormal market behavior without news.

### Inputs (free data)
- Price candles (1-5 minute)
- Volume
- VWAP proxy
- Range expansion

### Baseline Metrics (rolling)
- Avg volume (EWMA)
- Avg candle range
- Avg % change

### Trigger Conditions

```
PRICE_SPIKE:
  % change >= +3% within 2-5 minutes

VOLUME_SHOCK:
  volume >= 3x rolling average

RANGE_EXPANSION:
  candle range >= 2x average

MOMENTUM_CHAIN:
  3 consecutive green candles
```

### Output Event
```json
{
  "event": "MARKET_SHOCK",
  "symbol": "AMD",
  "metrics": {
    "pct_change": 4.2,
    "volume_ratio": 3.6
  },
  "ts": "UTC"
}
```

---

## 6. Crowd Signal Scanner (SECONDARY)

### Purpose
Detect human reaction velocity (often beats free news).

### Sources
- StockTwits trending symbols
- Reddit ticker frequency (WSB, stocks)
- X/Twitter ticker count (optional)

### Trigger
```
MENTION_SPIKE:
  ticker mentions >= 3x rolling average in <5 min
```

### Output Event
```json
{
  "event": "CROWD_SURGE",
  "symbol": "AMD",
  "mentions": 128,
  "baseline": 34
}
```

---

## 7. FSM Integration

### New Entry Path
Symbols from HotSymbolQueue bypass discovery scans.

### FSM Priority Order
1. HOT SYMBOLS
2. Existing watchlist
3. Passive scanner

### FSM State Injection
```
IDLE
  -> HOT_DETECTED
  -> MOMENTUM_CONFIRMING
  -> ARMED
  -> EXITED
```

---

## 8. Async News Enrichment (NON-BLOCKING)

### Purpose
Label the move after it starts.

### Sources
- SEC 8-K
- NASDAQ halts
- Company PR pages
- Yahoo Finance headlines

### Rules
- Never delays symbol injection
- Never vetoes a hot symbol
- Only annotates

---

## 9. Dashboard: Hot Symbols Panel

### Columns
- Symbol
- Time since trigger
- Trigger reason(s)
- Confidence %
- % change
- Volume ratio

### Visual Behavior
- Flash highlight on first appearance
- Countdown timer (TTL)
- Auto-fade on expiration

---

## 10. Implementation Order

1. HotSymbolQueue
2. Price / Volume Shock Detector
3. Crowd Signal Scanner
4. FSM priority injection
5. Dashboard hot symbols panel
6. Async news enrichment

---

## 11. Definition of Success

You will know this works when:
- Symbols appear before headlines
- You see moves forming, not finishing
- The FSM is active earlier
- Missed opportunities drop sharply

---

## Core Directive

**Stop waiting for news.**
**Treat market reaction as the news.**
**Everything else is context.**
