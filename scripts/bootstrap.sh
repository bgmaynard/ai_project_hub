#!/usr/bin/env bash
set -euo pipefail

# Move to repo root (assumes this script lives in scripts/)
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
cd "$root"

echo "== AI_Project_Hub bootstrap (bash) =="

# Pick python
if command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else
  echo "Python not found. Install Python 3.11+."; exit 1
fi

# venv
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment at .venv ..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m ensurepip --upgrade >/dev/null
python -m pip install --upgrade pip wheel setuptools >/dev/null

if [ -f requirements.txt ]; then
  echo "Installing from requirements.txt ..."
  python -m pip install -r requirements.txt
else
  echo "Installing core dependencies ..."
  python -m pip install 'fastapi==0.110.*' 'uvicorn[standard]==0.27.*' 'ib-insync==0.9.*' 'python-dotenv==1.*'
fi

# Load .env if present (export KEY=VALUE per line)
if [ -f .env ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs -0 -I{} bash -c 'echo {}' 2>/dev/null || true)
else
  : "${LOCAL_API_KEY:=dev_local_key_change_me}"
  : "${API_HOST:=127.0.0.1}"
  : "${API_PORT:=9101}"
  : "${TWS_HOST:=127.0.0.1}"
  : "${TWS_PORT:=7497}"
  : "${TWS_CLIENT_ID:=1101}"
fi

mod="store.code.IBKR_Algo_BOT.dashboard_api:app"
echo "Starting Uvicorn: $mod on http://${API_HOST}:${API_PORT} ..."
python -m uvicorn "$mod" --host "${API_HOST}" --port "${API_PORT}" --reload
