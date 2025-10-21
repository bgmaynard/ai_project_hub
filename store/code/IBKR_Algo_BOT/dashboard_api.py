
import os
from datetime import datetime
from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(filename='.env', usecwd=True))

from store.code.IBKR_Algo_BOT.bridge.ib_adapter import IBAdapter, IBConfig

app = FastAPI(title="IBKR Dashboard API (Real)")

def require_api_key(x_api_key: str = Header(None)):
    expected = os.getenv("LOCAL_API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="Server misconfigured: LOCAL_API_KEY not set")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

def make_ib():
    cfg = IBConfig(
        host=os.getenv("TWS_HOST", "127.0.0.1"),
        port=int(os.getenv("TWS_PORT", "7497")),
        client_id=int(os.getenv("TWS_CLIENT_ID", "6001")),
        read_only=os.getenv("IB_READ_ONLY", "1") not in ("0","false","False")
    )
    ib = IBAdapter(cfg)
    try:
        ib.connect()
    except Exception:
        pass
    return ib

ib = make_ib()

class OrderIn(BaseModel):
    symbol: str
    side: str
    qty: int
    limitPrice: float | None = None

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ibkr_available": bool(ib and ib.connected),
        "adapter_created": bool(ib is not None),
        "read_only": True if getattr(ib, 'cfg', None) and getattr(ib.cfg, 'read_only', False) else False
    }

@app.get("/api/status")
def status():
    ro = True if getattr(ib, 'cfg', None) and getattr(ib.cfg, 'read_only', False) else False
    return {
        "ib_connection": bool(ib and ib.connected),
        "state": "READY" if ib and ib.connected else "NOT_AVAILABLE",
        "error": None if ib and ib.connected else "IB Adapter not available",
        "read_only": ro,
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.post("/api/order/preview")
def order_preview(order: OrderIn, _=Depends(require_api_key)):
    if not ib:
        raise HTTPException(status_code=503, detail="IB adapter missing")
    if not ib.connected:
        raise HTTPException(status_code=503, detail="IB not connected")
    return ib.preview_order(order.symbol, order.side, order.qty, order.limitPrice)

@app.post("/api/order/place")
def order_place(order: OrderIn, _=Depends(require_api_key)):
    if not ib:
        raise HTTPException(status_code=503, detail="IB adapter missing")
    if not ib.connected:
        raise HTTPException(status_code=503, detail="IB not connected")
    try:
        res = ib.place_order(order.symbol, order.side, order.qty, order.limitPrice)
        return res
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
