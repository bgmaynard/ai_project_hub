"""
Schwab Trading API Routes
FastAPI router for Schwab/ThinkOrSwim trading endpoints
"""
import logging
from typing import Optional, List
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/schwab", tags=["schwab"])

# Import Schwab trading module
try:
    from schwab_trading import (
        get_schwab_trading,
        is_schwab_trading_available,
        SchwabTrading
    )
    HAS_SCHWAB_TRADING = True
except ImportError as e:
    logger.warning(f"Schwab trading not available: {e}")
    HAS_SCHWAB_TRADING = False


# ============================================================================
# REQUEST MODELS
# ============================================================================

class MarketOrderRequest(BaseModel):
    symbol: str
    quantity: int
    side: str  # BUY, SELL


class LimitOrderRequest(BaseModel):
    symbol: str
    quantity: int
    side: str
    limit_price: float


class StopOrderRequest(BaseModel):
    symbol: str
    quantity: int
    side: str
    stop_price: float


class StopLimitOrderRequest(BaseModel):
    symbol: str
    quantity: int
    side: str
    stop_price: float
    limit_price: float


class BracketOrderRequest(BaseModel):
    symbol: str
    quantity: int
    side: str
    take_profit_price: float
    stop_loss_price: float
    limit_price: Optional[float] = None


class AccountSelectRequest(BaseModel):
    account_number: str


class UnifiedOrderRequest(BaseModel):
    """Unified order request matching UI format"""
    symbol: str
    action: str  # BUY, SELL (UI sends action, convert to side)
    quantity: int
    order_type: str  # MARKET, LIMIT
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tif: Optional[str] = "DAY"
    extended_hours: Optional[bool] = False
    exchange: Optional[str] = "SMART"


# ============================================================================
# STATUS ENDPOINTS
# ============================================================================

@router.get("/status")
async def get_schwab_status():
    """Get Schwab trading status"""
    if not HAS_SCHWAB_TRADING:
        return {
            "available": False,
            "message": "Schwab trading module not available"
        }

    try:
        trading = get_schwab_trading()
        if not trading:
            return {
                "available": False,
                "message": "Schwab not authenticated. Run schwab_authenticate.py"
            }

        accounts = trading.get_accounts()
        selected = trading.get_selected_account()

        return {
            "available": True,
            "authenticated": True,
            "accounts_count": len(accounts),
            "selected_account": selected,
            "accounts": accounts
        }
    except Exception as e:
        logger.error(f"Error getting Schwab status: {e}")
        return {"available": False, "error": str(e)}


@router.post("/connect")
async def connect_schwab():
    """Connect to Schwab (validates authentication)"""
    if not HAS_SCHWAB_TRADING:
        return {
            "connected": False,
            "status": "unavailable",
            "message": "Schwab trading module not available"
        }

    try:
        trading = get_schwab_trading()
        if not trading:
            return {
                "connected": False,
                "status": "not_authenticated",
                "message": "Schwab not authenticated. Run schwab_authenticate.py"
            }

        # Verify connection by getting accounts
        accounts = trading.get_accounts()
        selected = trading.get_selected_account()

        return {
            "connected": True,
            "status": "connected",
            "broker": "Schwab",
            "accounts_count": len(accounts),
            "selected_account": selected,
            "message": "Connected to Schwab successfully"
        }
    except Exception as e:
        logger.error(f"Error connecting to Schwab: {e}")
        return {"connected": False, "status": "error", "error": str(e)}


# ============================================================================
# ACCOUNT ENDPOINTS
# ============================================================================

@router.get("/accounts")
async def get_accounts():
    """Get list of available Schwab accounts"""
    if not HAS_SCHWAB_TRADING:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    trading = get_schwab_trading()
    if not trading:
        raise HTTPException(status_code=401, detail="Schwab not authenticated")

    return {
        "accounts": trading.get_accounts(),
        "selected": trading.get_selected_account()
    }


@router.post("/accounts/select")
async def select_account(request: AccountSelectRequest):
    """Select a Schwab account for trading"""
    if not HAS_SCHWAB_TRADING:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    trading = get_schwab_trading()
    if not trading:
        raise HTTPException(status_code=401, detail="Schwab not authenticated")

    success = trading.select_account(request.account_number)
    if not success:
        raise HTTPException(status_code=404, detail=f"Account not found: {request.account_number}")

    return {
        "success": True,
        "selected_account": request.account_number
    }


@router.get("/account")
async def get_account_info():
    """Get account information for selected account"""
    if not HAS_SCHWAB_TRADING:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    trading = get_schwab_trading()
    if not trading:
        raise HTTPException(status_code=401, detail="Schwab not authenticated")

    info = trading.get_account_info()
    if not info:
        raise HTTPException(status_code=500, detail="Failed to get account info")

    return info


@router.get("/positions")
async def get_positions():
    """Get all positions for selected account"""
    if not HAS_SCHWAB_TRADING:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    trading = get_schwab_trading()
    if not trading:
        raise HTTPException(status_code=401, detail="Schwab not authenticated")

    positions = trading.get_positions()
    return {
        "positions": positions,
        "count": len(positions),
        "source": "schwab"
    }


# ============================================================================
# ORDER ENDPOINTS
# ============================================================================

@router.get("/orders")
async def get_orders(status: Optional[str] = None):
    """Get orders for selected account"""
    if not HAS_SCHWAB_TRADING:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    trading = get_schwab_trading()
    if not trading:
        raise HTTPException(status_code=401, detail="Schwab not authenticated")

    orders = trading.get_orders(status=status)
    return {
        "orders": orders,
        "count": len(orders),
        "source": "schwab"
    }


@router.post("/place-order")
async def place_unified_order(request: UnifiedOrderRequest):
    """
    Place order - unified endpoint matching UI format.
    Handles both MARKET and LIMIT orders based on order_type.
    Accepts 'action' (BUY/SELL) and converts to 'side'.
    """
    if not HAS_SCHWAB_TRADING:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    trading = get_schwab_trading()
    if not trading:
        raise HTTPException(status_code=401, detail="Schwab not authenticated")

    # Convert action to side (UI sends 'action', Schwab expects 'side')
    side = request.action.upper()

    extended = request.extended_hours or False
    session_type = "SEAMLESS (extended)" if extended else "NORMAL"
    logger.info(f"Schwab order: {side} {request.quantity} {request.symbol} @ {request.order_type} | Session: {session_type}")

    try:
        if request.order_type.upper() == "MARKET":
            result = trading.place_market_order(
                symbol=request.symbol,
                quantity=request.quantity,
                side=side,
                extended_hours=extended
            )
        elif request.order_type.upper() == "LIMIT":
            if not request.limit_price:
                raise HTTPException(status_code=400, detail="Limit price required for LIMIT orders")
            result = trading.place_limit_order(
                symbol=request.symbol,
                quantity=request.quantity,
                side=side,
                limit_price=request.limit_price,
                extended_hours=extended
            )
        elif request.order_type.upper() == "STOP":
            if not request.stop_price:
                raise HTTPException(status_code=400, detail="Stop price required for STOP orders")
            result = trading.place_stop_order(
                symbol=request.symbol,
                quantity=request.quantity,
                side=side,
                stop_price=request.stop_price,
                extended_hours=extended
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported order type: {request.order_type}")

        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])

        # Get selected account for response
        selected_account = trading.get_selected_account() if hasattr(trading, 'get_selected_account') else "unknown"

        # Format response to match UI expectations
        return {
            "success": True,
            "order_id": result.get("order_id", result.get("id", "unknown")),
            "status": result.get("status", "submitted"),
            "message": f"Order placed: {side} {request.quantity} {request.symbol}",
            "broker": "schwab",
            "account": selected_account,
            "session": "SEAMLESS" if extended else "NORMAL",
            "details": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Schwab order error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/place-limit-order")
async def place_limit_order(request: LimitOrderRequest):
    """Place a limit order"""
    if not HAS_SCHWAB_TRADING:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    trading = get_schwab_trading()
    if not trading:
        raise HTTPException(status_code=401, detail="Schwab not authenticated")

    result = trading.place_limit_order(
        symbol=request.symbol,
        quantity=request.quantity,
        side=request.side,
        limit_price=request.limit_price
    )

    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/place-stop-order")
async def place_stop_order(request: StopOrderRequest):
    """Place a stop order"""
    if not HAS_SCHWAB_TRADING:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    trading = get_schwab_trading()
    if not trading:
        raise HTTPException(status_code=401, detail="Schwab not authenticated")

    result = trading.place_stop_order(
        symbol=request.symbol,
        quantity=request.quantity,
        side=request.side,
        stop_price=request.stop_price
    )

    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/place-stop-limit-order")
async def place_stop_limit_order(request: StopLimitOrderRequest):
    """Place a stop-limit order"""
    if not HAS_SCHWAB_TRADING:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    trading = get_schwab_trading()
    if not trading:
        raise HTTPException(status_code=401, detail="Schwab not authenticated")

    result = trading.place_stop_limit_order(
        symbol=request.symbol,
        quantity=request.quantity,
        side=request.side,
        stop_price=request.stop_price,
        limit_price=request.limit_price
    )

    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/place-bracket-order")
async def place_bracket_order(request: BracketOrderRequest):
    """Place a bracket order (entry + take profit + stop loss)"""
    if not HAS_SCHWAB_TRADING:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    trading = get_schwab_trading()
    if not trading:
        raise HTTPException(status_code=401, detail="Schwab not authenticated")

    result = trading.place_bracket_order(
        symbol=request.symbol,
        quantity=request.quantity,
        side=request.side,
        take_profit_price=request.take_profit_price,
        stop_loss_price=request.stop_loss_price,
        limit_price=request.limit_price
    )

    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel a specific order"""
    if not HAS_SCHWAB_TRADING:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    trading = get_schwab_trading()
    if not trading:
        raise HTTPException(status_code=401, detail="Schwab not authenticated")

    result = trading.cancel_order(order_id)

    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])

    return result


class CancelOrderRequest(BaseModel):
    order_id: str


@router.post("/cancel-order")
async def cancel_order_post(request: CancelOrderRequest):
    """Cancel a specific order (POST method for UI compatibility)"""
    if not HAS_SCHWAB_TRADING:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    trading = get_schwab_trading()
    if not trading:
        raise HTTPException(status_code=401, detail="Schwab not authenticated")

    result = trading.cancel_order(request.order_id)

    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.delete("/orders")
async def cancel_all_orders():
    """Cancel all open orders"""
    if not HAS_SCHWAB_TRADING:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    trading = get_schwab_trading()
    if not trading:
        raise HTTPException(status_code=401, detail="Schwab not authenticated")

    result = trading.cancel_all_orders()
    return result


# ============================================================================
# ORDER TYPES INFO
# ============================================================================

@router.get("/order-types")
async def get_order_types():
    """Get supported order types for Schwab"""
    return {
        "broker": "schwab",
        "order_types": [
            {
                "type": "MARKET",
                "description": "Execute immediately at best available price",
                "endpoint": "/api/schwab/place-order"
            },
            {
                "type": "LIMIT",
                "description": "Execute at specified price or better",
                "endpoint": "/api/schwab/place-limit-order"
            },
            {
                "type": "STOP",
                "description": "Trigger market order when stop price is reached",
                "endpoint": "/api/schwab/place-stop-order"
            },
            {
                "type": "STOP_LIMIT",
                "description": "Trigger limit order when stop price is reached",
                "endpoint": "/api/schwab/place-stop-limit-order"
            },
            {
                "type": "BRACKET",
                "description": "Entry with automatic take profit and stop loss",
                "endpoint": "/api/schwab/place-bracket-order"
            }
        ]
    }


# ============================================================================
# MARKET DATA ENDPOINTS
# ============================================================================

@router.get("/quote/{symbol}")
async def get_schwab_quote(symbol: str):
    """Get real-time quote from Schwab"""
    try:
        from schwab_market_data import get_schwab_market_data
        market_data = get_schwab_market_data()

        if not market_data:
            raise HTTPException(status_code=503, detail="Schwab market data not available")

        quote = market_data.get_quote(symbol.upper())

        if not quote:
            raise HTTPException(status_code=404, detail=f"Quote not found for {symbol}")

        return quote
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Schwab quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quotes")
async def get_schwab_quotes(symbols: str = Query(..., description="Comma-separated symbols")):
    """Get real-time quotes for multiple symbols from Schwab"""
    try:
        from schwab_market_data import get_schwab_market_data
        market_data = get_schwab_market_data()

        if not market_data:
            raise HTTPException(status_code=503, detail="Schwab market data not available")

        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        quotes = market_data.get_quotes(symbol_list)

        return {
            "quotes": quotes,
            "count": len(quotes),
            "source": "schwab"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Schwab quotes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# REAL-TIME STREAMING ENDPOINTS
# ============================================================================

# Import streaming module
try:
    from schwab_streaming import (
        start_streaming,
        stop_streaming,
        subscribe,
        unsubscribe,
        get_status as get_stream_status,
        get_cached_quote,
        get_quote_cache
    )
    HAS_SCHWAB_STREAMING = True
except ImportError as e:
    logger.warning(f"Schwab streaming not available: {e}")
    HAS_SCHWAB_STREAMING = False


class StreamSubscribeRequest(BaseModel):
    symbols: List[str]


@router.get("/streaming/status")
async def get_streaming_status():
    """Get Schwab real-time streaming status"""
    if not HAS_SCHWAB_STREAMING:
        return {
            "available": False,
            "message": "Schwab streaming module not available"
        }

    try:
        status = get_stream_status()
        return {
            "available": True,
            **status
        }
    except Exception as e:
        logger.error(f"Error getting streaming status: {e}")
        return {"available": False, "error": str(e)}


@router.post("/streaming/start")
async def start_streaming_endpoint(request: Optional[StreamSubscribeRequest] = None):
    """Start Schwab WebSocket streaming"""
    if not HAS_SCHWAB_STREAMING:
        raise HTTPException(status_code=503, detail="Schwab streaming not available")

    try:
        symbols = request.symbols if request else None
        start_streaming(symbols)
        return {
            "success": True,
            "message": "Streaming started",
            "subscribed": symbols or []
        }
    except Exception as e:
        logger.error(f"Error starting streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/streaming/stop")
async def stop_streaming_endpoint():
    """Stop Schwab WebSocket streaming"""
    if not HAS_SCHWAB_STREAMING:
        raise HTTPException(status_code=503, detail="Schwab streaming not available")

    try:
        stop_streaming()
        return {"success": True, "message": "Streaming stopped"}
    except Exception as e:
        logger.error(f"Error stopping streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/streaming/subscribe")
async def subscribe_symbols(request: StreamSubscribeRequest):
    """Subscribe to additional symbols for streaming"""
    if not HAS_SCHWAB_STREAMING:
        raise HTTPException(status_code=503, detail="Schwab streaming not available")

    try:
        subscribe(request.symbols)
        return {
            "success": True,
            "subscribed": request.symbols
        }
    except Exception as e:
        logger.error(f"Error subscribing to symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/streaming/unsubscribe")
async def unsubscribe_symbols(request: StreamSubscribeRequest):
    """Unsubscribe from symbols"""
    if not HAS_SCHWAB_STREAMING:
        raise HTTPException(status_code=503, detail="Schwab streaming not available")

    try:
        unsubscribe(request.symbols)
        return {
            "success": True,
            "unsubscribed": request.symbols
        }
    except Exception as e:
        logger.error(f"Error unsubscribing from symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/streaming/quote/{symbol}")
async def get_streaming_quote(symbol: str):
    """Get real-time quote from streaming cache (instant, no API call)"""
    if not HAS_SCHWAB_STREAMING:
        raise HTTPException(status_code=503, detail="Schwab streaming not available")

    try:
        quote = get_cached_quote(symbol.upper())
        if not quote:
            return {
                "symbol": symbol.upper(),
                "available": False,
                "message": "Symbol not in streaming cache. Subscribe first."
            }
        return {
            "symbol": symbol.upper(),
            "available": True,
            **quote
        }
    except Exception as e:
        logger.error(f"Error getting streaming quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/streaming/quotes")
async def get_all_streaming_quotes():
    """Get all quotes from streaming cache"""
    if not HAS_SCHWAB_STREAMING:
        raise HTTPException(status_code=503, detail="Schwab streaming not available")

    try:
        quotes = get_quote_cache()
        return {
            "quotes": quotes,
            "count": len(quotes),
            "source": "schwab_stream"
        }
    except Exception as e:
        logger.error(f"Error getting streaming quotes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# FAST POLLING ENDPOINTS (Fallback for WebSocket streaming)
# ============================================================================

# Import fast polling module
try:
    from schwab_fast_polling import (
        start_polling,
        stop_polling,
        subscribe as fast_subscribe,
        unsubscribe as fast_unsubscribe,
        get_status as get_fast_poll_status,
        get_cached_quote as get_fast_cached_quote,
        get_quote_cache as get_fast_quote_cache,
        set_poll_interval
    )
    HAS_FAST_POLLING = True
except ImportError as e:
    logger.warning(f"Schwab fast polling not available: {e}")
    HAS_FAST_POLLING = False


@router.get("/fast-polling/status")
async def get_fast_polling_status():
    """Get Schwab fast polling status"""
    if not HAS_FAST_POLLING:
        return {
            "available": False,
            "message": "Schwab fast polling module not available"
        }

    try:
        status = get_fast_poll_status()
        return {
            "available": True,
            **status
        }
    except Exception as e:
        logger.error(f"Error getting fast polling status: {e}")
        return {"available": False, "error": str(e)}


@router.post("/fast-polling/start")
async def start_fast_polling_endpoint(request: Optional[StreamSubscribeRequest] = None):
    """Start Schwab fast polling service"""
    if not HAS_FAST_POLLING:
        raise HTTPException(status_code=503, detail="Schwab fast polling not available")

    try:
        symbols = request.symbols if request else None
        start_polling(symbols)
        return {
            "success": True,
            "message": "Fast polling started",
            "subscribed": symbols or []
        }
    except Exception as e:
        logger.error(f"Error starting fast polling: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fast-polling/stop")
async def stop_fast_polling_endpoint():
    """Stop Schwab fast polling service"""
    if not HAS_FAST_POLLING:
        raise HTTPException(status_code=503, detail="Schwab fast polling not available")

    try:
        stop_polling()
        return {"success": True, "message": "Fast polling stopped"}
    except Exception as e:
        logger.error(f"Error stopping fast polling: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fast-polling/interval/{interval_ms}")
async def set_fast_polling_interval(interval_ms: int):
    """Set fast polling interval in milliseconds (100-5000)"""
    if not HAS_FAST_POLLING:
        raise HTTPException(status_code=503, detail="Schwab fast polling not available")

    try:
        set_poll_interval(interval_ms)
        return {"success": True, "interval_ms": max(100, min(5000, interval_ms))}
    except Exception as e:
        logger.error(f"Error setting poll interval: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fast-polling/quote/{symbol}")
async def get_fast_polling_quote(symbol: str):
    """Get quote from fast polling cache (instant, no API call)"""
    if not HAS_FAST_POLLING:
        raise HTTPException(status_code=503, detail="Schwab fast polling not available")

    try:
        quote = get_fast_cached_quote(symbol.upper())
        if not quote:
            return {
                "symbol": symbol.upper(),
                "available": False,
                "message": "Symbol not in fast polling cache. Subscribe first."
            }
        return {
            "symbol": symbol.upper(),
            "available": True,
            **quote
        }
    except Exception as e:
        logger.error(f"Error getting fast polling quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fast-polling/quotes")
async def get_all_fast_polling_quotes():
    """Get all quotes from fast polling cache"""
    if not HAS_FAST_POLLING:
        raise HTTPException(status_code=503, detail="Schwab fast polling not available")

    try:
        quotes = get_fast_quote_cache()
        return {
            "quotes": quotes,
            "count": len(quotes),
            "source": "schwab_fast_poll"
        }
    except Exception as e:
        logger.error(f"Error getting fast polling quotes: {e}")
        raise HTTPException(status_code=500, detail=str(e))
