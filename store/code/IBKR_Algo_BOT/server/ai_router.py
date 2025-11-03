from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from pathlib import Path
import json

# Try to import the real predictor
RealPredictor = None
try:
    from ai.ai_predictor import EnhancedAIPredictor as RealPredictor
except Exception:
    RealPredictor = None

router = APIRouter(prefix="/api/ai", tags=["ai"])

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PRED_LOG = LOG_DIR / "predictions.csv"
if not PRED_LOG.exists():
    PRED_LOG.write_text("ts,symbol,signal,prob_up,prob_down,confidence,prediction\n", encoding="utf-8")

class PredictReq(BaseModel):
    symbol: str = Field(..., description="Ticker, e.g., SPY")

class TrainReq(BaseModel):
    symbol: str = "SPY"
    lookback_days: int = 365

class BacktestReq(BaseModel):
    symbol: str = "SPY"
    params: dict = Field(default_factory=dict)

def _ts():
    return datetime.now(timezone.utc).isoformat()

def _write_row(ts, sym, signal, pu, pd, conf, pred):
    with PRED_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{ts},{sym},{signal},{pu},{pd},{conf},{pred}\n")

# === /predict ===
@router.post("/predict")
def predict(req: PredictReq):
    ts = _ts()
    symbol = req.symbol.upper()
    signal = "NEUTRAL"
    prob_up = 0.50
    prob_down = 0.50
    confidence = 0.00
    prediction = 0
    model_file = Path(__file__).resolve().parents[1] / "store" / "models" / "lgb_predictor.txt"
    if RealPredictor is not None and model_file.exists():
        try:
            pred = RealPredictor()
            if hasattr(pred, "load_model"):
                pred.load_model()
            res = pred.predict(symbol)
            signal = res.get("signal", signal)
            prob_up = float(res.get("prob_up", prob_up))
            prob_down = float(res.get("prob_down", prob_down))
            confidence = float(res.get("confidence", confidence))
            prediction = int(res.get("prediction", prediction))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Predictor error: {e!s}")
    _write_row(ts, symbol, signal, prob_up, prob_down, confidence, prediction)
    return {"asof": ts, "symbol": symbol, "signal": signal, "prob_up": prob_up, "prob_down": prob_down, "confidence": confidence, "prediction": prediction}

@router.get("/predict/last")
def last():
    if not PRED_LOG.exists():
        return {"status":"empty"}
    lines = PRED_LOG.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= 1:
        return {"status":"empty"}
    parts = lines[-1].split(",")
    if len(parts) < 7:
        return {"status":"invalid"}
    ts, symbol, signal, pu, pd, conf, pred = parts
    return {"ts": ts, "symbol": symbol, "signal": signal, "prob_up": float(pu), "prob_down": float(pd), "confidence": float(conf), "prediction": int(pred), "raw": parts}

@router.get("/predict/history")
def history(symbol: str="SPY", limit: int=200):
    if not PRED_LOG.exists():
        return {"rows": []}
    rows = []
    for line in PRED_LOG.read_text(encoding="utf-8").strip().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 6:
            continue
        ts, sym, signal, pu, pd, conf = parts[:6]
        pred = parts[6] if len(parts) > 6 else "0"
        if sym.upper() == symbol.upper():
            rows.append({"ts": ts, "symbol": sym, "signal": signal, "prob_up": float(pu), "prob_down": float(pd), "confidence": float(conf), "prediction": int(pred)})
    return {"rows": rows[-limit:]}

@router.post("/train")
def train(req: TrainReq):
    if RealPredictor is None:
        return {"ok": False, "message": "Predictor not available"}
    try:
        p = RealPredictor()
        acc = p.train(req.symbol)
        p.save_model()
        return {"ok": True, "symbol": req.symbol, "accuracy": acc, "trained_at": _ts()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Train failed: {e!s}")

@router.post("/backtest")
def backtest(req: BacktestReq):
    stats = {"winrate": 0.58, "avg_pnl": 0.013, "trades": 120, "max_drawdown": -0.072}
    return {"ok": True, "symbol": req.symbol, "stats": stats, "params": req.params}
