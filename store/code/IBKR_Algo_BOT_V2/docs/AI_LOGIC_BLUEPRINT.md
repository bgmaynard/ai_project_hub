# Next-Gen AI Logic Blueprint for Momentum Trading Bot

**Project:** AI Trading Bot (Schwab TOS + Alpaca Testing)
**Purpose:** Expand the AI bot's predictive intelligence, market awareness, and reasoning depth with state-of-the-art modular logics. This document is designed to be ingested by Claude Code, Copilot, or other AI collaborators and embedded into the bot's architecture.

---

## Summary
- **Platform:** TradingView GUI + Schwab ThinkOrSwim (live) + Alpaca (test/reporting)
- **Primary Strategy:** Short-horizon momentum detection, VWAP and sentiment overlays, barrier trigger zones
- **Current Strengths:** Logistic + LLN model, modular routing, safe execution logic, dashboard-aware
- **Next Goals:** Add situational awareness, narrative logic, smart exit and size logic, causal learning, and auto-healing

---

## 1. Temporal & Regime-Aware Reasoning

### Logic Modules to Build
- `regime_classifier.py`: Classify sessions as Trending / Ranging / Volatile / Quiet using rolling std of returns and price slope
- `market_clock.py`: Create time-aware state machine (market open, midday lull, power hour)
- `day_of_week_bias.py`: Adjust signal weightings based on historical performance by weekday

### AI Action
- Tune aggressiveness of order logic based on time (e.g., allow higher risk during open)
- Suppress false breakouts during chop
- Output regime tag in each trade log

---

## 2. Causal Feature Learning & Trade Attribution

### Modules
- `shap_explainer.py`: Attach SHAP/feature attribution scores to each prediction
- `granger_test_util.py`: Run lightweight Granger causality tests on tick features vs price moves

### AI Action
- Filter out weak/non-contributing features over time
- Print ranked feature influence in `ai_signal_log.json`
- Highlight why signal fired in dashboard (e.g., "80% weight on VWAP distance")

---

## 3. Macro, Sector, and News Context Integration

### Modules
- `finbert_headline_embedder.py`: Convert headlines/news into sentiment vectors using FinBERT
- `sector_map_booster.py`: Dynamically boost correlated tickers when SPY, XLF, etc. break out
- `event_guardrail.py`: Temporarily pause AI during Fed/CPI/earnings if high volatility risk detected

### AI Action
- Prioritize trades in sectors with positive sentiment
- Pause trading during risk events
- Annotate macro triggers in trade logs

---

## 4. Trade Reasoning & Narrative Generation

### Modules
- `trade_narrative.py`: Generate plain-English reason for each trade (e.g., "Breakout at VWAP with RSI confirmation + drift")
- `counter_signal_filter.py`: Cancel or reduce size if RSI says overbought + sentiment turns negative

### AI Action
- Improve human trust & debugging
- Enable conditional logic for exit override
- Allow trader override when narrative contradicts confidence

---

## 5. Fault Detection & Self-Healing AI

### Modules
- `brier_tracker.py`: Tracks rolling prediction loss (Brier score) per symbol and model
- `slippage_anomaly_watcher.py`: Detects unexpected fill slippage -> flags spoofing, latency
- `self_heal_controller.py`: Disables modules that degrade past threshold

### AI Action
- Auto-pause failing models
- Reduce position size in symbols with degraded fill behavior
- Trigger alert in GUI for manual review

---

## 6. Meta-Learning & Controller Layer

### Modules
- `strategy_selector.py`: Chooses best-performing module per sector, time, or market type
- `meta_weight_adjuster.py`: Dynamically reweights modules based on last N trades' success

### AI Action
- Prefer breakout model on high volatility days
- Use sentiment model more in slow, steady climbs
- Learn from each trading session and adjust behavior

---

## 7. Smart Exits, Risk Engine & Sizing

### Modules
- `kelly_sizer.py`: Risk-adjusted sizing logic based on current edge
- `trailing_exit_strategy.py`: Adapts trailing stop to volatility and signal confidence
- `account_guardrail.py`: Prevent trade entry when cash/margin/PDT risk exceeded (supports Schwab + Alpaca)

### AI Action
- Tailor position size to trade confidence
- Exit more aggressively when confidence drops
- Keep bot compliant with FINRA, SEC, broker risk rules

---

## Real-Time Feedback Layer (Claude/Copilot Tasks)

- Update `mesh_heartbeat.json` with:
  ```json
  { "id": "chatgpt", "ver": "controller-1.0", "status": "phase-6-live", "modules": ["regime_classifier", "kelly_sizer"] }
  ```
- Output trade logs as:
  ```json
  {
    "symbol": "TSLA",
    "p_up": 0.78,
    "confidence": 0.91,
    "decision": "buy",
    "narrative": "Breakout from VWAP with strong sentiment & low spread",
    "exit": "trailing",
    "regime": "trending",
    "risk": "within PDT limit"
  }
  ```

---

## Benchmarking & Metrics (for Claude to track)

| Metric                | Purpose                                | Tool/API |
|-----------------------|-----------------------------------------|----------|
| Win rate              | Detect actual trade accuracy            | Alpaca trades |
| Average P&L per trade | Profitability of logic combinations     | Alpaca reports |
| Calibration error     | Quality of probability predictions      | Brier score |
| Decision latency      | Real-time AI execution responsiveness   | Log timestamps |
| Regime accuracy       | % of sessions correctly tagged          | Compare with VIX or trend index |

---

## Final Direction: Where to Aim Next

1. **Chain-of-Reasoning AI**:
   - Future AI should reason step-by-step: "If VWAP confirms + RSI strong -> trend continuation likely -> act."
   - Use LangChain-style thought processing per tick

2. **Multi-Agent AI Execution**:
   - Allow breakout bot + mean-revert bot to argue
   - Let a referee logic score the debate

3. **Memory-Augmented AI**:
   - Store trade outcome history per ticker
   - Reuse "lessons" when trading same symbol again (LLM + memory index)

---

## Next Steps
- Claude: begin by drafting `regime_classifier.py` and `trade_narrative.py`
- ChatGPT: package smart exit and sizing logic into a single `.zip`
- Copilot: integrate GUI logs and new signal metadata into `dashboard_api.py`

---

**Prepared by:** ChatGPT (controller-1.0)
**Date:** 2025-12-03
