import os
import datetime
import asyncio
from typing import Optional

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder, StopLimitOrder, Order, util as ib_util
except Exception:
    # Allow running in OFFLINE_MODE without ib_insync installed
    IB = object  # type: ignore
    Stock = MarketOrder = LimitOrder = StopOrder = StopLimitOrder = Order = object  # type: ignore
    ib_util = None  # type: ignore

# -------------------- helpers --------------------

def _offline() -> bool:
    return os.getenv("OFFLINE_MODE", "0").lower() in ("1", "true", "yes", "on")

def _stamp() -> str:
    try:
        return datetime.datetime.now().astimezone().isoformat()
    except Exception:
        return datetime.datetime.utcnow().isoformat() + "Z"

def _ensure_loop() -> None:
    """Ensure an asyncio event loop exists in this thread (FastAPI / AnyIO worker)."""
    try:
        asyncio.get_running_loop()
        return
    except RuntimeError:
        pass
    try:
        if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# -------------------- module state --------------------

_last_error: Optional[str] = None
_last_attempt: Optional[str] = None
_last_ok: Optional[str] = None

# initialize loop at import time too
_ensure_loop()

# -------------------- mock payloads (offline/weekend) --------------------

def _mock_status():
    return {
        "connected": False,
        "account": None,
        "serverTime": None,
        "diag": {
            "lastAttempt": _stamp(),
            "lastOk": None,
            "lastError": "OFFLINE_MODE",
            "host": os.getenv("TWS_HOST", "127.0.0.1"),
            "port": int(os.getenv("TWS_PORT", "7497")),
            "clientId": int(os.getenv("TWS_CLIENT_ID", "19")),
            "readonly": True,
            "mock": True,
        },
    }

def _mock_quote(symbol: str):
    return {"ok": True, "symbol": symbol.upper(), "last": 123.45, "bid": 123.40, "ask": 123.50, "mock": True}

def _mock_place(symbol, side, order_type, qty):
    return {
        "ok": True, "mock": True, "symbol": symbol.upper(), "side": side.upper(),
        "type": order_type.upper(), "orderId": 900000, "status": "Submitted (mock)",
        "filled": 0, "remaining": qty, "avgFillPrice": None
    }

def _mock_preview(symbol, side, order_type, limit_price, stop_price, trail_type, trail_value, activation_price, tif, outside_rth, note="OFFLINE_MODE mock"):
    return {
        "ok": True, "mock": True, "symbol": symbol.upper(), "side": side.upper(), "type": order_type.upper(),
        "tif": tif.upper(), "outsideRth": bool(outside_rth),
        "prices": {
            "limitPrice": limit_price, "stopPrice": stop_price, "trailType": trail_type,
            "trailValue": trail_value, "activationPrice": activation_price
        },
        "preview": {
            "initMarginChange": "100.00", "maintMarginChange": "100.00", "equityWithLoanChange": "-100.00",
            "commission": "1.00", "minCommission": "1.00", "maxCommission": "1.00", "warningText": note
        }
    }

# -------------------- connection --------------------

def _connect(readonly: bool = True):
    """Connect to TWS/Gateway. Uses async connect when available; records diag timestamps."""
    global _last_error, _last_attempt, _last_ok
    _ensure_loop()

    host = os.getenv("TWS_HOST", "127.0.0.1")
    port = int(os.getenv("TWS_PORT", "7497"))
    client_id = int(os.getenv("TWS_CLIENT_ID", "19"))

    _last_attempt = _stamp()
    _last_error = None

    ib = IB() if hasattr(IB, "__name__") else None
    if _offline() or ib is None:
        return ib
    try:
        if ib_util is not None and hasattr(ib, "connectAsync"):
            ib_util.run(ib.connectAsync(host, port, clientId=client_id, readonly=readonly, timeout=3))
        else:
            ib.connect(host, port, clientId=client_id, readonly=readonly, timeout=3)
        if ib.isConnected():
            _last_ok = _stamp()
    except Exception as e:
        _last_error = f"{type(e).__name__}: {e}"
    return ib

def _contract_for(symbol: str):
    return Stock(symbol, "SMART", "USD") if hasattr(Stock, "__name__") else None

# -------------------- status & quotes --------------------

def ib_status():
    if _offline():
        return _mock_status()
    ib = _connect(readonly=True)
    ok = bool(ib and ib.isConnected())
    acct = None
    server_time = None
    if ok:
        try:
            summary = ib.reqAccountSummary()
            acct = summary[0].account if summary else None
        except Exception:
            pass
        try:
            server_time = ib.reqCurrentTime()
        except Exception:
            pass
        try:
            ib.disconnect()
        except Exception:
            pass
    return {
        "connected": ok,
        "account": acct,
        "serverTime": str(server_time) if server_time else None,
        "diag": {
            "lastAttempt": _last_attempt, "lastOk": _last_ok, "lastError": _last_error,
            "host": os.getenv("TWS_HOST", "127.0.0.1"),
            "port": int(os.getenv("TWS_PORT", "7497")),
            "clientId": int(os.getenv("TWS_CLIENT_ID", "19")),
            "readonly": True
        }
    }

def get_quote(symbol: str):
    if _offline():
        return _mock_quote(symbol)
    ib = _connect(readonly=True)
    if not ib or not ib.isConnected():
        return {"ok": False, "error": "Not connected to TWS"}
    c = _contract_for(symbol.upper())
    try:
        t = ib.reqMktData(c, "", False, False)
        ib.sleep(1.0)
        last = float(t.last) if t.last is not None else None
        bid = float(t.bid) if t.bid is not None else None
        ask = float(t.ask) if t.ask is not None else None
        ib.disconnect()
        return {"ok": True, "symbol": symbol.upper(), "last": last, "bid": bid, "ask": ask}
    except Exception as e:
        try:
            ib.disconnect()
        except Exception:
            pass
        return {"ok": False, "error": str(e)}

# -------------------- order building --------------------

def _build_order(side, order_type, qty, limit_price, stop_price, tif, outside_rth, *, trail_type=None, trail_value=None, activation_price=None):
    side = side.upper()
    order_type = order_type.upper()
    tif = tif.upper()

    if order_type == "MKT":
        o = MarketOrder(side, qty)
    elif order_type == "LMT":
        if limit_price is None:
            raise ValueError("limitPrice required for LMT")
        o = LimitOrder(side, qty, float(limit_price))
    elif order_type in ("STP", "STOP"):
        if stop_price is None:
            raise ValueError("stopPrice required for STP")
        o = StopOrder(side, qty, float(stop_price))
    elif order_type in ("STP_LMT", "STOP_LIMIT", "STOP-LIMIT"):
        if stop_price is None or limit_price is None:
            raise ValueError("stopPrice and limitPrice required for STOP_LIMIT")
        o = StopLimitOrder(side, qty, float(limit_price), float(stop_price))
    elif order_type in ("TRAIL", "TRAILING", "TRAILING_STOP"):
        if trail_value is None:
            raise ValueError("trailValue required for TRAIL")
        o = Order()
        o.orderType = "TRAIL"
        o.action = side
        o.totalQuantity = qty
        mode = (trail_type or "amt").lower()
        if mode in ("amt", "amount", "dollar", "usd"):
            o.auxPrice = float(trail_value)
        elif mode in ("pct", "percent", "percentage"):
            o.trailingPercent = float(trail_value)
        else:
            raise ValueError("trailType must be 'amt' or 'pct'")
        if activation_price is not None:
            o.trailStopPrice = float(activation_price)
    else:
        raise ValueError(f"Unsupported order type: {order_type}")

    o.tif = tif
    o.outsideRth = bool(outside_rth)
    return o

# -------------------- orders / account --------------------

def place_order(symbol, side, qty, order_type, limit_price=None, stop_price=None, tif="DAY", outside_rth=False, *, trail_type=None, trail_value=None, activation_price=None):
    if _offline():
        return _mock_place(symbol, side, order_type, qty)
    ib = _connect(readonly=False)
    if not ib or not ib.isConnected():
        return {"ok": False, "error": "Not connected to TWS for orders"}
    c = _contract_for(symbol.upper())
    try:
        ib.qualifyContracts(c)
    except Exception:
        pass
    try:
        o = _build_order(
            side, order_type, qty, limit_price, stop_price, tif, outside_rth,
            trail_type=trail_type, trail_value=trail_value, activation_price=activation_price
        )
    except Exception as e:
        try:
            ib.disconnect()
        except Exception:
            pass
        return {"ok": False, "error": str(e)}
    try:
        tr = ib.placeOrder(c, o)
        ib.sleep(0.2)
        s = tr.orderStatus
        res = {
            "ok": True,
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "orderId": getattr(s, "orderId", getattr(tr, "order", None).orderId if getattr(tr, "order", None) else None),
            "status": getattr(s, "status", None),
            "filled": getattr(s, "filled", None),
            "remaining": getattr(s, "remaining", None),
            "avgFillPrice": getattr(s, "avgFillPrice", None),
        }
        ib.disconnect()
        return res
    except Exception as e:
        err = str(e)
        try:
            ib.disconnect()
        except Exception:
            pass
        return {"ok": False, "error": err}

def cancel_all_orders():
    if _offline():
        return {"ok": True, "mock": True}
    ib = _connect(readonly=False)
    if not ib or not ib.isConnected():
        return {"ok": False, "error": "Not connected to TWS for orders"}
    try:
        ib.reqGlobalCancel()
        ib.disconnect()
        return {"ok": True}
    except Exception as e:
        try:
            ib.disconnect()
        except Exception:
            pass
        return {"ok": False, "error": str(e)}

def list_open_orders():
    if _offline():
        return {"ok": True, "mock": True, "openOrders": []}
    ib = _connect(readonly=False)
    if not ib or not ib.isConnected():
        return {"ok": False, "error": "Not connected to TWS"}
    try:
        out = []
        for o in ib.openOrders():
            s = getattr(o, "order", o)
            c = getattr(o, "contract", None)
            out.append({
                "orderId": getattr(s, "orderId", None),
                "action": getattr(s, "action", None),
                "orderType": getattr(s, "orderType", None),
                "totalQuantity": getattr(s, "totalQuantity", None),
                "lmtPrice": getattr(s, "lmtPrice", None),
                "auxPrice": getattr(s, "auxPrice", None),
                "trailingPercent": getattr(s, "trailingPercent", None),
                "trailStopPrice": getattr(s, "trailStopPrice", None),
                "tif": getattr(s, "tif", None),
                "symbol": getattr(c, "symbol", None) if c else None,
            })
        ib.disconnect()
        return {"ok": True, "openOrders": out}
    except Exception as e:
        try:
            ib.disconnect()
        except Exception:
            pass
        return {"ok": False, "error": str(e)}

def list_positions():
    if _offline():
        return {"ok": True, "mock": True, "positions": []}
    ib = _connect(readonly=False)
    if not ib or not ib.isConnected():
        return {"ok": False, "error": "Not connected to TWS"}
    try:
        out = []
        for p in ib.positions():
            c = p.contract
            out.append({
                "symbol": c.symbol,
                "exchange": c.exchange,
                "position": p.position,
                "avgCost": p.avgCost,
            })
        ib.disconnect()
        return {"ok": True, "positions": out}
    except Exception as e:
        try:
            ib.disconnect()
        except Exception:
            pass
        return {"ok": False, "error": str(e)}

def account_snapshot():
    if _offline():
        return {"ok": True, "mock": True, "account": None, "summary": {}}
    ib = _connect(readonly=False)
    if not ib or not ib.isConnected():
        return {"ok": False, "error": "Not connected to TWS"}
    try:
        summary = ib.reqAccountSummary()
        out = {item.tag: item.value for item in summary}
        acct = summary[0].account if summary else None
        ib.disconnect()
        return {"ok": True, "account": acct, "summary": out}
    except Exception as e:
        try:
            ib.disconnect()
        except Exception:
            pass
        return {"ok": False, "error": str(e)}

# -------------------- What-If / Preview --------------------

def preview_order(symbol, side, qty, order_type, limit_price=None, stop_price=None, tif="DAY", outside_rth=False, *, trail_type=None, trail_value=None, activation_price=None):
    if _offline():
        return _mock_preview(symbol, side, order_type, limit_price, stop_price, trail_type, trail_value, activation_price, tif, outside_rth)
    ib = _connect(readonly=False)
    if not ib or not ib.isConnected():
        return {"ok": False, "error": "Not connected to TWS for preview"}
    c = _contract_for(symbol.upper())
    try:
        ib.qualifyContracts(c)
    except Exception:
        pass
    try:
        o = _build_order(
            side, order_type, qty, limit_price, stop_price, tif, outside_rth,
            trail_type=trail_type, trail_value=trail_value, activation_price=activation_price
        )
        o.whatIf = True
    except Exception as e:
        try:
            ib.disconnect()
        except Exception:
            pass
        return {"ok": False, "error": str(e)}
    try:
        try:
            st = ib.whatIfOrder(c, o)
        except Exception:
            tr = ib.placeOrder(c, o)
            ib.sleep(0.2)
            st = tr.orderState
        ib.disconnect()
        return {
            "ok": True,
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "tif": tif.upper(),
            "outsideRth": bool(outside_rth),
            "preview": {
                "initMarginChange": getattr(st, "initMarginChange", None),
                "maintMarginChange": getattr(st, "maintMarginChange", None),
                "equityWithLoanChange": getattr(st, "equityWithLoanChange", None),
                "commission": getattr(st, "commission", None),
                "minCommission": getattr(st, "minCommission", None),
                "maxCommission": getattr(st, "maxCommission", None),
                "warningText": getattr(st, "warningText", None),
            },
            "prices": {
                "limitPrice": limit_price, "stopPrice": stop_price,
                "trailType": trail_type, "trailValue": trail_value,
                "activationPrice": activation_price
            },
        }
    except Exception as e:
        err = str(e)
        try:
            ib.disconnect()
        except Exception:
            pass
        return {"ok": False, "error": err}
def cancel_order(order_id: int):
    """Cancel a specific order by orderId."""
    if _offline():
        return {"ok": True, "mock": True, "orderId": int(order_id), "status": "Canceled (mock)"}
    ib = _connect(readonly=False)
    if not ib or not ib.isConnected():
        return {"ok": False, "error": "Not connected to TWS for orders"}
    try:
        # refresh & collect open orders
        try:
            ib.reqOpenOrders()
            ib.sleep(0.2)
        except Exception:
            pass
        opens = ib.openOrders() or []
        # find matches (ib_insync returns Order objects in openOrders)
        matches = [o for o in opens if getattr(o, "orderId", None) == int(order_id)]
        if not matches:
            # fall back: scan trades for any open order with matching id
            try:
                for tr in ib.trades():
                    o = getattr(tr, "order", None)
                    if getattr(o, "orderId", None) == int(order_id):
                        matches.append(o)
                        break
            except Exception:
                pass
        if not matches:
            ib.disconnect()
            return {"ok": False, "error": f"OrderId {order_id} not found among open orders"}

        # cancel
        for o in matches:
            try:
                ib.cancelOrder(o)
            except Exception as e:
                try:
                    ib.disconnect()
                except Exception:
                    pass
                return {"ok": False, "orderId": int(order_id), "error": str(e)}
        ib.sleep(0.2)
        ib.disconnect()
        return {"ok": True, "orderId": int(order_id), "status": "CancelRequested"}
    except Exception as e:
        try:
            ib.disconnect()
        except Exception:
            pass
        return {"ok": False, "orderId": int(order_id), "error": str(e)}
