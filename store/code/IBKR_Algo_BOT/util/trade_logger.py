from __future__ import annotations
import os
import json
import asyncio
import datetime
from typing import Any, Dict

LOG_PATH = os.getenv("TRADE_LOG_PATH", os.path.join("store", "logs", "trades.jsonl"))
_LOCK = asyncio.Lock()

def _ts() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

async def log_event(kind: str, payload: Dict[str, Any]) -> None:
    line = json.dumps({"ts": _ts(), "type": kind, **payload}, ensure_ascii=False)
    async with _LOCK:
        # Minimal async append
        await asyncio.to_thread(_append_line, LOG_PATH, line)

def _append_line(path: str, line: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

async def tail(n: int = 100) -> str:
    """Return last n lines joined; small file approach."""
    try:
        text = await asyncio.to_thread(lambda: open(LOG_PATH, "r", encoding="utf-8").read())
    except FileNotFoundError:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-max(1, n):])
