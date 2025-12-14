# Claude Delta Sync — Reconciliation Plan

This patch aligns the repo with Claude's Oct 29 updates while adding guardrails:

## What we standardized
- `.env.example` provided; `.env` kept local only.
- API key gating for sensitive endpoints (`X-API-Key`).
- Health and status endpoints stabilized.
- Default to PAPER trading (7497) in configs; live (7496) available via `.env`.

## What you should verify on your machine
- UI loads at `http://127.0.0.1:9101/ui/platform.html`.
- `/api/status` shows correct `tws_port` (7496 live or 7497 paper).
- `/api/ai/predict` returns real model output after wiring to your `ai_predictor`.
- Orders place successfully from the UI with `outsideRth` respected when checked.

## Suggested follow-ups
- Convert current LightGBM predictor to ONNX for faster inference.
- Add prediction history & slippage feedback table.
- Add tests for price endpoint and order flow using `ib_insync` mock.
