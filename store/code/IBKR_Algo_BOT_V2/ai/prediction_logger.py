# ai/prediction_logger.py
import csv
import os
from datetime import datetime
from typing import Any, Dict

LOG_DIR = os.path.join(os.getcwd(), "logs")
LOG_FILE = os.path.join(LOG_DIR, "predictions.csv")
COLUMNS = ["ts", "symbol", "signal", "prob_up", "prob_down", "confidence", "prediction"]


def _ensure():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()


def log_prediction(symbol: str, payload: Dict[str, Any]) -> None:
    _ensure()
    row = {
        "ts": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "signal": payload.get("signal"),
        "prob_up": payload.get("prob_up"),
        "prob_down": payload.get("prob_down"),
        "confidence": payload.get("confidence"),
        "prediction": payload.get("prediction"),
    }
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)
