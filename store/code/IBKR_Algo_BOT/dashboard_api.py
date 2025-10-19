import os, importlib.util, traceback
from typing import Optional
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Root is three levels up from this file: ...\ai_project_hub\store\code\IBKR_Algo_BOT\dashboard_api.py
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BRIDGE_FILE = os.path.join(ROOT, "src", "bridge", "ibkr_bridge_check.py")

def _load_bridge():
    if not os.path.isfile(BRIDGE_FILE):
        raise FileNotFoundError(f"Missing bridge module at: {BRIDGE_FILE}")
    spec = importlib.util.spec_from_file_location("ibkr_bridge_check", BRIDGE_FILE)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)                 # type: ignore
    return mod

_bridge = _load_bridge()
app = FastAPI(title="AI Bot Project Connection API", version="0.3.0")

def _try(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}

# Optional API key guard (only enforced if API_KEY is set in .env)
def _require_key(x_api_key: Optional[str]):
    required = os.getenv("API_KEY")
    if not required:
        return
    if x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/health")
def health():
    offline = os.getenv("OFFLINE_MODE","0").lower() in ("1","true","yes","on")
    return {"ok": True, "offline": offline, "service": "ai-bot-connection",
            "env": {"host": os.getenv("API_HOST"), "port": os.getenv("API_PORT")}}

@app.get("/api/tws/status")
def tws_status():
    return _try(_bridge.ib_status)

@app.get("/api/quote/{symbol}")
def quote(symbol: str):
    return _try(_bridge.get_quote, symbol.upper())

# ----- Orders (incl. TRAIL) -----
class OrderReq(BaseModel):
    symbol: str = Field(..., description="Ticker, e.g., AAPL")
    qty: float = Field(..., gt=0)
    type: str = Field(..., description="MKT | LMT | STP | STP_LMT | TRAIL")
    limitPrice: Optional[float] = None
    stopPrice: Optional[float]  = None
    tif: str = "DAY"
    outsideRth: bool = False
    trailType: Optional[str] = None
    trailValue: Optional[float] = None
    activationPrice: Optional[float] = None

class PreviewReq(OrderReq):
    side: str = Field(..., description="BUY or SELL")

@app.post("/api/order/buy")
def order_buy(body: OrderReq, x_api_key: Optional[str] = Header(default=None)):
    _require_key(x_api_key)
    return _try(_bridge.place_order, symbol=body.symbol, side="BUY", qty=body.qty, order_type=body.type,
                limit_price=body.limitPrice, stop_price=body.stopPrice, tif=body.tif, outside_rth=body.outsideRth,
                trail_type=body.trailType, trail_value=body.trailValue, activation_price=body.activationPrice)

@app.post("/api/order/sell")
def order_sell(body: OrderReq, x_api_key: Optional[str] = Header(default=None)):
    _require_key(x_api_key)
    return _try(_bridge.place_order, symbol=body.symbol, side="SELL", qty=body.qty, order_type=body.type,
                limit_price=body.limitPrice, stop_price=body.stopPrice, tif=body.tif, outside_rth=body.outsideRth,
                trail_type=body.trailType, trail_value=body.trailValue, activation_price=body.activationPrice)

@app.post("/api/order/cancel_all")
def order_cancel_all(x_api_key: Optional[str] = Header(default=None)):
    _require_key(x_api_key)
    return _try(_bridge.cancel_all_orders)

@app.get("/api/orders/open")
def orders_open():
    return _try(_bridge.list_open_orders)

@app.get("/api/positions")
def positions():
    return _try(_bridge.list_positions)

@app.get("/api/account")
def account():
    return _try(_bridge.account_snapshot)

@app.post("/api/order/preview")
def order_preview(body: PreviewReq, x_api_key: Optional[str] = Header(default=None)):
    _require_key(x_api_key)
    return _try(_bridge.preview_order, symbol=body.symbol, side=body.side, qty=body.qty, order_type=body.type,
                limit_price=body.limitPrice, stop_price=body.stopPrice, tif=body.tif, outside_rth=body.outsideRth,
                trail_type=body.trailType, trail_value=body.trailValue, activation_price=body.activationPrice)

# ----- Bracket orders -----
class BracketReq(BaseModel):
    symbol: str
    side: str
    qty: float
    entryType: str = "MKT"
    entryLimitPrice: Optional[float] = None
    takeProfitPrice: Optional[float] = None
    stopType: str = "STP"
    stopPrice: Optional[float] = None
    stopLimitPrice: Optional[float] = None
    tif: str = "DAY"
    outsideRth: bool = False
    trailType: Optional[str] = None
    trailValue: Optional[float] = None
    activationPrice: Optional[float] = None

@app.post("/api/order/bracket")
def order_bracket(body: BracketReq, x_api_key: Optional[str] = Header(default=None)):
    _require_key(x_api_key)
    return _try(_bridge.place_bracket_order,
                symbol=body.symbol, side=body.side, qty=body.qty,
                entryType=body.entryType, entryLimitPrice=body.entryLimitPrice,
                takeProfitPrice=body.takeProfitPrice,
                stopType=body.stopType, stopPrice=body.stopPrice, stopLimitPrice=body.stopLimitPrice,
                tif=body.tif, outsideRth=body.outsideRth,
                trailType=body.trailType, trailValue=body.trailValue, activationPrice=body.activationPrice)

@app.post("/api/order/preview_bracket")
def order_preview_bracket(body: BracketReq, x_api_key: Optional[str] = Header(default=None)):
    _require_key(x_api_key)
    return _try(_bridge.preview_bracket_order,
                symbol=body.symbol, side=body.side, qty=body.qty,
                entryType=body.entryType, entryLimitPrice=body.entryLimitPrice,
                takeProfitPrice=body.takeProfitPrice,
                stopType=body.stopType, stopPrice=body.stopPrice, stopLimitPrice=body.stopLimitPrice,
                tif=body.tif, outsideRth=body.outsideRth,
                trailType=body.trailType, trailValue=body.trailValue, activationPrice=body.activationPrice)
# --- orders helpers (appended) ---
try:
    _bridge  # noqa: F821
except NameError:
    # fallback import if file was opened standalone (normally _bridge is set above)
    from src.bridge import ibkr_bridge_check as _bridge  # type: ignore

@app.get("/api/orders/open")
def orders_open():
    """List open orders (live or mock)."""
    return _bridge.list_open_orders()

@app.post("/api/orders/cancel_all")
@app.post("/api/orders/cancel-all")
def orders_cancel_all():
    """Cancel all open orders (live or mock)."""
    return _bridge.cancel_all_orders()
# === appended: orders routes ===
try:
    _bridge  # noqa: F821
except NameError:
    from src.bridge import ibkr_bridge_check as _bridge  # type: ignore

@app.get("/api/orders/open")
def orders_open():
    """List open orders (live or mock)."""
    return _bridge.list_open_orders()

@app.post("/api/orders/cancel_all")
@app.post("/api/orders/cancel-all")
@app.post("/api/order/cancel_all")   # alias
def orders_cancel_all():
    """Cancel all open orders (live or mock)."""
    return _bridge.cancel_all_orders()
# === end appended ===
# === appended: single-order cancel routes ===
from fastapi import Path

@app.post("/api/orders/cancel/{order_id}")
@app.post("/api/order/{order_id}/cancel")
def order_cancel(order_id: int = Path(..., ge=1)):
    """Cancel a single order by orderId."""
    return _bridge.cancel_order(order_id)
# === end appended ===
# === appended: JSON body cancel endpoints ===
from pydantic import BaseModel
from typing import List, Optional

class CancelRequest(BaseModel):
    orderId: int

class CancelManyRequest(BaseModel):
    orderIds: List[int]

@app.post("/api/orders/cancel")  # body: {"orderId": 12345}
def orders_cancel_body(payload: CancelRequest):
    return _bridge.cancel_order(int(payload.orderId))

@app.post("/api/orders/cancel_many")  # body: {"orderIds": [12345, 23456]}
def orders_cancel_many(payload: CancelManyRequest):
    results = []
    for oid in payload.orderIds:
        try:
            results.append(_bridge.cancel_order(int(oid)))
        except Exception as e:
            results.append({"ok": False, "orderId": int(oid), "error": str(e)})
    # overall ok if all succeeded
    overall_ok = all(r.get("ok") for r in results)
    return {"ok": overall_ok, "results": results}
# === end appended ===
# === appended: tws ping + system info ===
import os, time
from fastapi import Header
from typing import Optional

@app.get("/api/tws/ping")
def tws_ping():
    """Lightweight health for IBKR connectivity."""
    try:
        # Use existing bridge status; fast and safe
        s = _bridge.ib_status()
        return {"connected": bool(s.get("connected")), "ts": int(time.time())}
    except Exception as e:
        return {"connected": False, "error": str(e), "ts": int(time.time())}

@app.get("/api/info")
def api_info(x_api_key: Optional[str] = Header(None, convert_underscores=False)):
    """Feature flags + env snapshot (non-secret)."""
    from platform import python_version
    offline = os.getenv("OFFLINE_MODE","0").lower() in ("1","true","yes","on")
    require_key = bool(os.getenv("LOCAL_API_KEY",""))
    host = os.getenv("API_HOST")
    port = os.getenv("API_PORT","9101")
    client_id = os.getenv("TWS_CLIENT_ID")
    # hide api key value; only indicate presence and whether header matched (for quick diag)
    key_ok = (x_api_key == os.getenv("LOCAL_API_KEY")) if require_key else True

    # Basic route inventory (top-level only)
    routes = []
    try:
        for r in app.routes:
            p = getattr(r, "path", "")
            if p.startswith("/api/"):
                routes.append({"path": p, "methods": list(getattr(r, "methods", []))})
    except Exception:
        pass

    return {
        "service": "ai-bot-connection",
        "version": (open("VERSION").read().strip() if os.path.exists("VERSION") else None),
        "python": python_version(),
        "features": {
            "twsPing": True,
            "ordersRequireApiKey": require_key,
            "offlineMode": offline,
        },
        "env": {
            "host": host, "port": port,
            "tws": {"host": os.getenv("TWS_HOST"), "port": os.getenv("TWS_PORT"), "clientId": client_id}
        },
        "auth": {"apiKeyConfigured": require_key, "apiKeyHeaderValid": key_ok},
        "routes": routes[:60],
        "ts": int(time.time())
    }
# === end appended ===
# === appended: UI static mount + root redirect ===
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import pathlib

_ui_dir = pathlib.Path(__file__).parent / "ui"
if _ui_dir.exists():
    try:
        app.mount("/ui", StaticFiles(directory=str(_ui_dir), html=True), name="ui")
    except Exception:
        pass

@app.get("/")
def _root():
    # land on status page
    return RedirectResponse(url="/ui/status.html")
# === end appended ===
