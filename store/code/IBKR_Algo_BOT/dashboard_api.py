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
# === appended: simple X-API-Key guard for order routes ===
import os
from fastapi import Request
from fastapi.responses import JSONResponse

_API_KEY = os.getenv("LOCAL_API_KEY", "")

@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    # Protect only trading routes (mutations + order views)
    protected = request.url.path.startswith(("/api/order", "/api/orders"))
    if protected:
        # If a key is set in env, require it.
        if _API_KEY:
            key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
            if key != _API_KEY:
                return JSONResponse({"ok": False, "error": "Unauthorized"}, status_code=401)
        # If no key configured, allow (useful for dev), but you can flip this to block.
    return await call_next(request)

# (Optional) reflect whether auth is required in /health
try:
    # Find and patch the existing /health handler result on the fly
    for r in app.router.routes:
        if getattr(r, "path", None) == "/health":
            old = r.endpoint
            async def _patched_health():
                res = await old() if callable(old) else {"ok": True}
                # ensure dict
                if hasattr(res, "dict"): res = res.dict()
                if isinstance(res, dict):
                    res.setdefault("security", {})["ordersRequireApiKey"] = bool(_API_KEY)
                return res
            r.endpoint = _patched_health
            break
except Exception:
    pass
# === end appended ===
