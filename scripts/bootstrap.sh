#!/usr/bin/env bash
# scripts/bootstrap.sh
set -euo pipefail
REF="${1:-refs/tags/working-2025-10-18}"

echo "==> Syncing repo to $REF"
git fetch --all --prune
git checkout --detach "$REF"
git submodule update --init --recursive || true

echo "==> Python 3.11 venv + deps"
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt

echo "==> .env (if missing)"
[ -f .env ] || cat > .env <<'ENV'
API_HOST=127.0.0.1
API_PORT=9101
OFFLINE_MODE=0
TWS_HOST=127.0.0.1
TWS_PORT=7497
TWS_CLIENT_ID=777
ENV

echo "==> Start API"
python -m uvicorn dashboard_api:app --app-dir store/code/IBKR_Algo_BOT --host 127.0.0.1 --port 9101
