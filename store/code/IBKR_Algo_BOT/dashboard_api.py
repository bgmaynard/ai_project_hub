from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import json
import pathlib

app = FastAPI(title="IBKR Algo BOT Dashboard API")

# --- CORS (safe defaults) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount the UI folder ---
ui_dir = pathlib.Path("store/code/IBKR_Algo_BOT/ui")
ui_dir.mkdir(parents=True, exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

# --- Simple state + signals log ---
STATE = {
    "connected": True,      # TODO: replace with real TWS/IBKR probe
    "mtf_running": False,
    "warrior_running": False,
    "mtf_symbol": None,
    "warrior_symbol": None,
}

DATA_DIR = pathlib.Path("store/code/IBKR_Algo_BOT/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_FILE = DATA_DIR / "signals_tail.log"

def log_signal(kind: str, symbol: str, note: str):
    payload = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "kind": kind,
        "symbol": symbol,
        "note": note,
    }
    with SIGNALS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

class StartPayload(BaseModel):
    symbol: str

@app.get("/api/status")
def api_status():
    return {"connected": STATE["connected"], "ts": datetime.utcnow().isoformat() + "Z"}

@app.post("/api/bots/mtf/start")
def mtf_start(body: StartPayload, background_tasks: BackgroundTasks):
    STATE["mtf_running"] = True
    STATE["mtf_symbol"] = body.symbol.upper()
    log_signal("mtf_start", STATE["mtf_symbol"], "MTF bot started")
    # Hook for real loop: background_tasks.add_task(run_mtf_loop, STATE["mtf_symbol"])
    return {"ok": True, "running": STATE["mtf_running"], "symbol": STATE["mtf_symbol"]}

@app.post("/api/bots/mtf/stop")
def mtf_stop():
    was = STATE["mtf_running"]
    STATE["mtf_running"] = False
    log_signal("mtf_stop", STATE["mtf_symbol"] or "", "MTF bot stopped")
    return {"ok": True, "was_running": was, "running": STATE["mtf_running"]}

@app.post("/api/bots/warrior/start")
def warrior_start(body: StartPayload, background_tasks: BackgroundTasks):
    STATE["warrior_running"] = True
    STATE["warrior_symbol"] = body.symbol.upper()
    log_signal("warrior_start", STATE["warrior_symbol"], "Warrior bot started")
    # Hook for real loop: background_tasks.add_task(run_warrior_loop, STATE["warrior_symbol"])
    return {"ok": True, "running": STATE["warrior_running"], "symbol": STATE["warrior_symbol"]}

@app.post("/api/bots/warrior/stop")
def warrior_stop():
    was = STATE["warrior_running"]
    STATE["warrior_running"] = False
    log_signal("warrior_stop", STATE["warrior_symbol"] or "", "Warrior bot stopped")
    return {"ok": True, "was_running": was, "running": STATE["warrior_running"]}

@app.get("/api/signals/tail")
def signals_tail(limit: Optional[int] = 200):
    if not SIGNALS_FILE.exists():
        return "(no data)"
    lines: List[str] = SIGNALS_FILE.read_text(encoding="utf-8").splitlines()[-limit:]
    return "\n".join(lines) if lines else "(no data)"

# --- Local runner (optional):  uvicorn dashboard_api:app --reload --port 9101 ---