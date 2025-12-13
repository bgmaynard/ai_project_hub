from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings

# Optional CSV logger
try:
    from ai.prediction_logger import log_prediction  # writes logs/predictions.csv
    _CSV_LOGGER = "module"
except Exception:
    _CSV_LOGGER = "inline"

app = FastAPI(title="AI Project Hub – Patched API")

# Ensure logs dir exists
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = LOG_DIR / "predictions.csv"

def _inline_log_prediction(symbol: str, payload: Dict[str, Any]) -> None:
    """Fallback simple CSV logger if ai.prediction_logger is unavailable."""
    is_new = not CSV_PATH.exists()
    line = (
        f'{datetime.now(timezone.utc).isoformat()},'
        f'{symbol},'
        f'{payload.get("signal","")},'
        f'{payload.get("prob_up","")},'
        f'{payload.get("prob_down","")},'
        f'{payload.get("confidence","")}'
        '\n'
    )
    if is_new:
        CSV_PATH.write_text("ts,symbol,signal,prob_up,prob_down,confidence\n", encoding="utf-8")
    with CSV_PATH.open("a", encoding="utf-8") as f:
        f.write(line)

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/api/status")
def status():
    s = get_settings()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ib_connection": True,
        "ai_connection": True,
        "auto_trader": True,
        "analytics": True,
        "backtest": True,
        "current_client_id": s.tws_client_id,
        "state": "CONNECTED",
        "api_key_configured": bool(s.api_key),
        "mode": "live" if str(s.tws_port) == "7496" else "paper",
        "host": s.tws_host,
        "port": s.tws_port,
        "asof": datetime.now(timezone.utc).isoformat(),
    }

@app.post("/api/ai/predict")
def predict(body: Dict[str, Any] = Body(...)):
    # Minimal stub prediction (replace with your real model call if desired)
    symbol = (body or {}).get("symbol", "SPY")
    # Example dummy numbers—just to exercise the pipe while you develop
    out = {
        "signal": "BULLISH",
        "prob_up": 0.62,
        "prob_down": 0.38,
        "confidence": 0.24,
        "prediction": 1,
        "symbol": symbol,
        "asof": datetime.now(timezone.utc).isoformat(),
    }
    try:
        if _CSV_LOGGER == "module":
            log_prediction(symbol=symbol, payload=out)
        else:
            _inline_log_prediction(symbol=symbol, payload=out)
    except Exception as e:
        # Log failure but continue responding
        print(f"[WARN] Failed to log prediction: {e}")
    return out

@app.get("/api/ai/predict/last")
def last_prediction():
    if not CSV_PATH.exists():
        return {"status": "empty"}
    rows = CSV_PATH.read_text(encoding="utf-8").strip().splitlines()
    if len(rows) <= 1:
        return {"status": "empty"}
    cols = rows[-1].split(",")
    return {
        "ts": cols[0],
        "symbol": cols[1],
        "signal": cols[2],
        "prob_up": float(cols[3]) if cols[3] else None,
        "prob_down": float(cols[4]) if cols[4] else None,
        "confidence": float(cols[5]) if cols[5] else None,
        "raw": cols,
    }

# Redirect / -> /docs (Swagger)
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

# Serve existing UI if present
_UI_DIR = Path(r"C:\ai_project_hub\ui")
if _UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")

def run():
    import uvicorn
    s = get_settings()
    # Respect env or .env bind address/port if you want; here we use defaults
    uvicorn.run("server.dashboard_api_patch:app", host="127.0.0.1", port=9101, reload=False)

if __name__ == "__main__":
    run()
