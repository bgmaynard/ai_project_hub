# Contributing

## Quick start
1. Clone the repo and create a virtual environment.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set your local values (never commit `.env`).
4. Run: `uvicorn store.code.IBKR_Algo_BOT.dashboard_api:app --host 127.0.0.1 --port 9101`

## Coding guidelines
- Python 3.11+, black/ruff preferred (format on save).
- Keep public endpoints resilient: never crash if IBKR is down.
- Don’t log secrets.

## PRs
- Include scope in title, e.g. `fix(api): ..`, `feat(adapter): ..`.
- Add a short test plan in the PR body.
