"""
Alpaca API Routes for Dashboard
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, Tuple
from alpaca_integration import get_alpaca_connector

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/alpaca", tags=["alpaca"])


# ============================================================================
# AI SAFETY HELPER FUNCTIONS
# ============================================================================

def check_circuit_breaker(connector) -> Tuple[bool, str, Dict]:
    """
    Check circuit breaker before placing an order.

    Returns:
        Tuple of (allowed, level, response_dict)
        - allowed: True if trading is allowed
        - level: Current circuit breaker level
        - response_dict: If blocked, contains the error response to return
    """
    try:
        from ai.circuit_breaker import get_circuit_breaker
        breaker = get_circuit_breaker()

        # Get current equity
        account = connector.get_account()
        current_equity = float(account.get('equity', 0))

        # Check if trading is allowed
        level = breaker.check_trade_allowed(current_equity)

        if level == "HALT":
            return False, level, {
                "success": False,
                "error": "CIRCUIT BREAKER HALT: Daily loss limit exceeded. Trading suspended.",
                "circuit_breaker_level": level,
                "order_blocked": True
            }
        elif level in ["WARNING", "CAUTION"]:
            logger.warning(f"Circuit breaker {level}: Proceed with caution")

        return True, level, {}
    except Exception as e:
        logger.warning(f"Circuit breaker check failed: {e}")
        return True, "UNKNOWN", {}  # Allow trade if check fails


def record_to_brain(symbol: str, action: str, price: float = 0, confidence: float = 0.5):
    """Record trade prediction to Background Brain for learning."""
    try:
        from ai.background_brain import get_background_brain
        brain = get_background_brain()
        if brain.is_running:
            brain.record_prediction(
                symbol=symbol,
                prediction=action,
                confidence=confidence,
                price=price
            )
    except Exception as e:
        logger.debug(f"Brain recording skipped: {e}")


def update_circuit_breaker_post_trade():
    """Update circuit breaker after a trade is placed."""
    try:
        from ai.circuit_breaker import get_circuit_breaker
        breaker = get_circuit_breaker()
        breaker.record_trade(won=True)  # Will be updated when position closes
    except Exception:
        pass


def record_symbol_trade(symbol: str, action: str, price: float, quantity: int):
    """Record trade to Symbol Memory for learning."""
    try:
        from ai.symbol_memory import get_symbol_memory
        memory = get_symbol_memory()
        memory.record_trade(
            symbol=symbol,
            entry_price=price,
            exit_price=price,  # Will be updated on close
            pnl=0,
            won=True  # Unknown until closed
        )
    except Exception as e:
        logger.debug(f"Symbol memory recording skipped: {e}")

class OrderRequest(BaseModel):
    symbol: str
    action: str  # BUY or SELL
    quantity: int
    order_type: str = "MKT"  # MKT or LMT
    limit_price: Optional[float] = None

@router.get("/status")
async def get_status():
    """Get Alpaca connection status"""
    connector = get_alpaca_connector()

    return {
        "connected": connector.is_connected(),
        "broker": "Alpaca",
        "paper_trading": True
    }

@router.post("/connect")
async def connect_alpaca():
    """Connect/reconnect to Alpaca - verifies connection is active"""
    connector = get_alpaca_connector()

    try:
        # Test the connection by fetching account info
        if connector.is_connected():
            account = connector.get_account()
            return {
                "status": "connected",
                "connected": True,
                "broker": "Alpaca",
                "account_id": account.get("account_id", "unknown"),
                "buying_power": account.get("buying_power", 0)
            }
        else:
            return {
                "status": "disconnected",
                "connected": False,
                "message": "Unable to connect to Alpaca. Check API credentials."
            }
    except Exception as e:
        return {
            "status": "error",
            "connected": False,
            "message": str(e)
        }

@router.get("/account")
async def get_account():
    """Get account information"""
    connector = get_alpaca_connector()

    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")

    return connector.get_account()


# ============================================================================
# PDT (PATTERN DAY TRADER) RULE MONITORING
# ============================================================================

@router.get("/pdt-status")
async def get_pdt_status():
    """
    Get PDT (Pattern Day Trader) rule status.

    SEC Rule: Accounts under $25,000 are limited to 3 day trades in a rolling 5-day period.
    A day trade = buying AND selling the same security on the same day.
    """
    connector = get_alpaca_connector()

    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")

    try:
        account = connector.get_account()
        equity = float(account.get("equity", 0))
        is_pdt = account.get("pattern_day_trader", False)

        # Get recent orders to count day trades
        orders = connector.get_orders("closed")

        # Track day trades in rolling 5-day window
        from datetime import datetime, timedelta
        five_days_ago = datetime.now() - timedelta(days=5)

        # Group trades by symbol and date to detect day trades
        trades_by_symbol_date = {}
        day_trades = []

        for order in orders:
            if order.get("status") != "filled":
                continue

            filled_at = order.get("filled_at", "")
            if not filled_at:
                continue

            # Parse date
            try:
                if isinstance(filled_at, str):
                    trade_date = datetime.fromisoformat(filled_at.replace("Z", "+00:00")).date()
                else:
                    trade_date = filled_at.date()
            except:
                continue

            # Check if within 5-day window
            if trade_date < five_days_ago.date():
                continue

            symbol = order.get("symbol", "")
            side = order.get("side", "")
            key = f"{symbol}_{trade_date}"

            if key not in trades_by_symbol_date:
                trades_by_symbol_date[key] = {"buys": 0, "sells": 0, "symbol": symbol, "date": str(trade_date)}

            if side == "buy":
                trades_by_symbol_date[key]["buys"] += 1
            elif side == "sell":
                trades_by_symbol_date[key]["sells"] += 1

        # Count day trades (buy + sell same day)
        for key, data in trades_by_symbol_date.items():
            if data["buys"] > 0 and data["sells"] > 0:
                count = min(data["buys"], data["sells"])
                for _ in range(count):
                    day_trades.append({
                        "symbol": data["symbol"],
                        "date": data["date"]
                    })

        day_trade_count = len(day_trades)
        day_trades_remaining = max(0, 3 - day_trade_count)
        pdt_threshold = 25000.0

        # Determine status
        if equity >= pdt_threshold:
            status = "EXEMPT"
            status_message = "Account equity >= $25,000. No day trade limits."
            warning_level = "none"
        elif is_pdt:
            status = "PDT_FLAGGED"
            status_message = "Account flagged as Pattern Day Trader. Trading restricted until equity >= $25,000."
            warning_level = "critical"
        elif day_trades_remaining == 0:
            status = "LIMIT_REACHED"
            status_message = "Day trade limit reached (3/3). Avoid day trading to prevent PDT flag."
            warning_level = "critical"
        elif day_trades_remaining == 1:
            status = "WARNING"
            status_message = f"Only {day_trades_remaining} day trade remaining. Use with caution."
            warning_level = "high"
        elif day_trades_remaining == 2:
            status = "CAUTION"
            status_message = f"{day_trades_remaining} day trades remaining in 5-day window."
            warning_level = "medium"
        else:
            status = "OK"
            status_message = f"{day_trades_remaining} day trades available."
            warning_level = "low"

        return {
            "status": status,
            "message": status_message,
            "warning_level": warning_level,
            "day_trades_used": day_trade_count,
            "day_trades_remaining": day_trades_remaining,
            "day_trades_limit": 3,
            "rolling_window_days": 5,
            "equity": equity,
            "pdt_threshold": pdt_threshold,
            "is_pdt_flagged": is_pdt,
            "is_exempt": equity >= pdt_threshold,
            "recent_day_trades": day_trades[-10:],  # Last 10 day trades
            "rules": {
                "description": "SEC Pattern Day Trader Rule",
                "limit": "3 day trades per 5 rolling business days for accounts under $25,000",
                "day_trade_definition": "Buying and selling (or short selling and buying) the same security on the same day",
                "consequence": "If you make 4+ day trades in 5 days with <$25k equity, your account gets flagged as PDT and restricted"
            }
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "message": f"Error checking PDT status: {str(e)}",
            "warning_level": "unknown",
            "day_trades_used": 0,
            "day_trades_remaining": 3,
            "equity": 0,
            "is_pdt_flagged": False
        }

@router.get("/positions")
async def get_positions():
    """Get all positions"""
    connector = get_alpaca_connector()
    
    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")
    
    return connector.get_positions()

# Order cache for fast responses - longer TTL since Alpaca API is slow (~4s)
_orders_cache = {"open": {"data": None, "timestamp": 0}, "closed": {"data": None, "timestamp": 0}}
_ORDERS_CACHE_TTL = 30  # 30 second cache - UI uses optimistic updates anyway

@router.get("/orders")
async def get_orders(status: str = "open", limit: int = 100, refresh: bool = False):
    """Get orders - defaults to open orders for speed"""
    import time
    connector = get_alpaca_connector()

    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")

    # Normalize status for cache key
    cache_key = "open" if status in ["open", "all"] else "closed"
    cache = _orders_cache[cache_key]

    # Check cache (unless refresh forced)
    now = time.time()
    if (not refresh and
        cache["data"] is not None and
        now - cache["timestamp"] < _ORDERS_CACHE_TTL):
        return {"orders": cache["data"], "count": len(cache["data"]), "cached": True}

    orders = connector.get_orders(status, limit)

    # Update cache
    cache["data"] = orders
    cache["timestamp"] = now

    return {"orders": orders, "count": len(orders)}

@router.post("/place-order")
async def place_order(order: OrderRequest):
    """Place an order with AI safety checks"""
    connector = get_alpaca_connector()

    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")

    try:
        # AI SAFETY CHECK: Circuit Breaker
        allowed, level, block_response = check_circuit_breaker(connector)
        if not allowed:
            return block_response

        # AI TRACKING: Record to Background Brain
        record_to_brain(order.symbol, order.action, order.limit_price or 0)

        # Normalize order type - accept both formats (MARKET/MKT, LIMIT/LMT)
        order_type = order.order_type.upper()
        is_market = order_type in ("MKT", "MARKET")
        is_limit = order_type in ("LMT", "LIMIT")

        if is_market:
            result = connector.place_market_order(
                symbol=order.symbol,
                quantity=order.quantity,
                side=order.action
            )
        elif is_limit:
            if order.limit_price is None:
                raise HTTPException(status_code=400, detail="Limit price required for limit orders")

            result = connector.place_limit_order(
                symbol=order.symbol,
                quantity=order.quantity,
                side=order.action,
                limit_price=order.limit_price
            )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid order type: {order.order_type}. Use MARKET or LIMIT.")

        # AI POST-TRADE: Update Circuit Breaker
        update_circuit_breaker_post_trade()

        # Add success field for UI compatibility
        result["success"] = True
        result["message"] = f"Order placed: {order.action} {order.quantity} {order.symbol}"
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an order"""
    connector = get_alpaca_connector()
    
    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")
    
    success = connector.cancel_order(order_id)
    
    return {"success": success}

@router.get("/quote/{symbol}")
async def get_quote(symbol: str):
    """Get quote for symbol"""
    connector = get_alpaca_connector()
    
    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")
    
    return connector.get_quote(symbol)

@router.delete("/positions/{symbol}")
async def close_position(symbol: str):
    """Close a position"""
    connector = get_alpaca_connector()
    
    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")
    
    success = connector.close_position(symbol)
    
    return {"success": success}

@router.delete("/positions")
async def close_all_positions():
    """Close all positions"""
    connector = get_alpaca_connector()

    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")

    success = connector.close_all_positions()

    return {"success": success}


# ============================================================================
# ADVANCED ORDER TYPES (Bracket, OCO, Trailing Stop)
# ============================================================================

class BracketOrderRequest(BaseModel):
    """Bracket order request with entry, take profit, and stop loss"""
    symbol: str
    quantity: int
    action: str  # BUY or SELL
    take_profit_price: float
    stop_loss_price: float
    limit_price: Optional[float] = None  # If None, market entry


class OCOOrderRequest(BaseModel):
    """OCO (One-Cancels-Other) order request"""
    symbol: str
    quantity: int
    action: str  # BUY or SELL
    take_profit_price: float
    stop_loss_price: float


class TrailingStopRequest(BaseModel):
    """Trailing stop order request"""
    symbol: str
    quantity: int
    action: str  # BUY or SELL
    trail_percent: Optional[float] = None  # e.g., 5.0 for 5%
    trail_price: Optional[float] = None  # Dollar amount


class StopOrderRequest(BaseModel):
    """Stop order request"""
    symbol: str
    quantity: int
    action: str  # BUY or SELL
    stop_price: float


class StopLimitOrderRequest(BaseModel):
    """Stop-limit order request"""
    symbol: str
    quantity: int
    action: str  # BUY or SELL
    stop_price: float
    limit_price: float


@router.post("/place-bracket-order")
async def place_bracket_order(order: BracketOrderRequest):
    """
    Place a bracket order with automatic take profit and stop loss.

    A bracket order creates three linked orders:
    1. Entry (market or limit)
    2. Take profit (limit order)
    3. Stop loss (stop order)

    Example:
    {
        "symbol": "AAPL",
        "quantity": 100,
        "action": "BUY",
        "take_profit_price": 160.00,
        "stop_loss_price": 145.00,
        "limit_price": 150.00  // Optional, market if not specified
    }
    """
    connector = get_alpaca_connector()

    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")

    try:
        # AI SAFETY CHECK: Circuit Breaker
        allowed, level, block_response = check_circuit_breaker(connector)
        if not allowed:
            return block_response

        # AI TRACKING: Record to Background Brain
        record_to_brain(order.symbol, order.action, order.limit_price or 0)

        result = connector.place_bracket_order(
            symbol=order.symbol,
            quantity=order.quantity,
            side=order.action,
            take_profit_price=order.take_profit_price,
            stop_loss_price=order.stop_loss_price,
            limit_price=order.limit_price
        )

        # AI POST-TRADE: Update Circuit Breaker
        update_circuit_breaker_post_trade()

        result["success"] = True
        result["message"] = f"Bracket order placed: {order.action} {order.quantity} {order.symbol}"
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/place-oco-order")
async def place_oco_order(order: OCOOrderRequest):
    """
    Place an OCO (One-Cancels-Other) order.

    Creates two linked orders where one filling cancels the other.
    Typically used to close a position with either profit target or stop loss.

    Example:
    {
        "symbol": "AAPL",
        "quantity": 100,
        "action": "SELL",
        "take_profit_price": 160.00,
        "stop_loss_price": 145.00
    }
    """
    connector = get_alpaca_connector()

    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")

    try:
        # AI SAFETY CHECK: Circuit Breaker
        allowed, level, block_response = check_circuit_breaker(connector)
        if not allowed:
            return block_response

        # AI TRACKING: Record to Background Brain
        record_to_brain(order.symbol, order.action, order.take_profit_price)

        result = connector.place_oco_order(
            symbol=order.symbol,
            quantity=order.quantity,
            side=order.action,
            take_profit_price=order.take_profit_price,
            stop_loss_price=order.stop_loss_price
        )

        # AI POST-TRADE: Update Circuit Breaker
        update_circuit_breaker_post_trade()

        result["success"] = True
        result["message"] = f"OCO order placed: {order.action} {order.quantity} {order.symbol}"
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/place-trailing-stop")
async def place_trailing_stop_order(order: TrailingStopRequest):
    """
    Place a trailing stop order.

    A trailing stop follows the price by a specified amount or percentage.
    Useful for locking in profits while letting winners run.

    Example (percentage):
    {
        "symbol": "AAPL",
        "quantity": 100,
        "action": "SELL",
        "trail_percent": 5.0
    }

    Example (dollar amount):
    {
        "symbol": "AAPL",
        "quantity": 100,
        "action": "SELL",
        "trail_price": 5.00
    }
    """
    connector = get_alpaca_connector()

    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")

    if not order.trail_percent and not order.trail_price:
        raise HTTPException(status_code=400, detail="Either trail_percent or trail_price must be specified")

    try:
        # AI SAFETY CHECK: Circuit Breaker
        allowed, level, block_response = check_circuit_breaker(connector)
        if not allowed:
            return block_response

        # AI TRACKING: Record to Background Brain
        record_to_brain(order.symbol, order.action)

        result = connector.place_trailing_stop_order(
            symbol=order.symbol,
            quantity=order.quantity,
            side=order.action,
            trail_percent=order.trail_percent,
            trail_price=order.trail_price
        )

        # AI POST-TRADE: Update Circuit Breaker
        update_circuit_breaker_post_trade()

        result["success"] = True
        trail_info = f"{order.trail_percent}%" if order.trail_percent else f"${order.trail_price}"
        result["message"] = f"Trailing stop placed: {order.action} {order.quantity} {order.symbol} ({trail_info})"
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/place-stop-order")
async def place_stop_order(order: StopOrderRequest):
    """
    Place a stop order.

    A stop order becomes a market order when the stop price is reached.

    Example:
    {
        "symbol": "AAPL",
        "quantity": 100,
        "action": "SELL",
        "stop_price": 145.00
    }
    """
    connector = get_alpaca_connector()

    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")

    try:
        # AI SAFETY CHECK: Circuit Breaker
        allowed, level, block_response = check_circuit_breaker(connector)
        if not allowed:
            return block_response

        # AI TRACKING: Record to Background Brain
        record_to_brain(order.symbol, order.action, order.stop_price)

        result = connector.place_stop_order(
            symbol=order.symbol,
            quantity=order.quantity,
            side=order.action,
            stop_price=order.stop_price
        )

        # AI POST-TRADE: Update Circuit Breaker
        update_circuit_breaker_post_trade()

        result["success"] = True
        result["message"] = f"Stop order placed: {order.action} {order.quantity} {order.symbol} @ ${order.stop_price}"
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/place-stop-limit-order")
async def place_stop_limit_order(order: StopLimitOrderRequest):
    """
    Place a stop-limit order.

    A stop-limit order becomes a limit order when the stop price is reached.

    Example:
    {
        "symbol": "AAPL",
        "quantity": 100,
        "action": "SELL",
        "stop_price": 145.00,
        "limit_price": 144.50
    }
    """
    connector = get_alpaca_connector()

    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")

    try:
        # AI SAFETY CHECK: Circuit Breaker
        allowed, level, block_response = check_circuit_breaker(connector)
        if not allowed:
            return block_response

        # AI TRACKING: Record to Background Brain
        record_to_brain(order.symbol, order.action, order.limit_price)

        result = connector.place_stop_limit_order(
            symbol=order.symbol,
            quantity=order.quantity,
            side=order.action,
            stop_price=order.stop_price,
            limit_price=order.limit_price
        )

        # AI POST-TRADE: Update Circuit Breaker
        update_circuit_breaker_post_trade()

        result["success"] = True
        result["message"] = f"Stop-limit order placed: {order.action} {order.quantity} {order.symbol} (stop: ${order.stop_price}, limit: ${order.limit_price})"
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/order-types")
async def get_available_order_types():
    """Get list of available order types with descriptions"""
    return {
        "order_types": [
            {
                "type": "market",
                "endpoint": "/place-order",
                "description": "Executes immediately at best available price"
            },
            {
                "type": "limit",
                "endpoint": "/place-order",
                "description": "Executes only at specified price or better"
            },
            {
                "type": "stop",
                "endpoint": "/place-stop-order",
                "description": "Becomes market order when stop price is reached"
            },
            {
                "type": "stop_limit",
                "endpoint": "/place-stop-limit-order",
                "description": "Becomes limit order when stop price is reached"
            },
            {
                "type": "bracket",
                "endpoint": "/place-bracket-order",
                "description": "Entry with automatic take profit and stop loss"
            },
            {
                "type": "oco",
                "endpoint": "/place-oco-order",
                "description": "One-Cancels-Other: two orders where one cancels the other"
            },
            {
                "type": "trailing_stop",
                "endpoint": "/place-trailing-stop",
                "description": "Stop that follows price by percentage or amount"
            },
            {
                "type": "smart",
                "endpoint": "/place-smart-order",
                "description": "Auto-detects session and uses appropriate order type for extended hours"
            }
        ]
    }


# ============================================================================
# EXTENDED HOURS TRADING ENDPOINTS
# ============================================================================

@router.get("/market-session")
async def get_market_session():
    """
    Get current market session status.

    Returns whether we're in pre-market, regular hours, after-hours, or closed.
    Critical for determining which order types can be used.

    Pre-market: 4:00 AM - 9:30 AM ET (limit orders only)
    Regular: 9:30 AM - 4:00 PM ET (all order types)
    After-hours: 4:00 PM - 8:00 PM ET (limit orders only)
    """
    connector = get_alpaca_connector()

    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")

    return connector.get_market_session()


class SmartOrderRequest(BaseModel):
    symbol: str
    action: str  # BUY or SELL
    quantity: int
    limit_price: Optional[float] = None
    force_extended: bool = False


@router.post("/place-smart-order")
async def place_smart_order(order: SmartOrderRequest):
    """
    Place a SMART order that automatically handles extended hours.

    This is the PRIMARY order method for early morning trading.

    Behavior:
    - Pre-market/After-hours: Converts to limit order with extended_hours=True
    - Regular hours: Uses market order (or limit if price specified)

    The limit price is auto-calculated from current bid/ask if not provided.
    """
    connector = get_alpaca_connector()

    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")

    try:
        # AI SAFETY CHECK: Circuit Breaker
        allowed, level, block_response = check_circuit_breaker(connector)
        if not allowed:
            return block_response

        # AI TRACKING: Record to Background Brain
        record_to_brain(order.symbol, order.action, order.limit_price or 0)

        result = connector.place_smart_order(
            symbol=order.symbol,
            quantity=order.quantity,
            side=order.action,
            limit_price=order.limit_price,
            force_extended=order.force_extended
        )

        # AI POST-TRADE: Update Circuit Breaker
        update_circuit_breaker_post_trade()

        result["success"] = True
        result["message"] = f"Smart order placed: {order.action} {order.quantity} {order.symbol}"
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ClosePositionRequest(BaseModel):
    symbol: str


@router.post("/close-position-smart")
async def close_position_smart(request: ClosePositionRequest):
    """
    Smart position close that handles extended hours.

    During extended hours, uses limit order at current bid.
    During regular hours, uses market order for immediate execution.
    """
    connector = get_alpaca_connector()

    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")

    try:
        result = connector.close_position_smart(request.symbol)

        if result.get("success", True):
            result["message"] = f"Position close initiated for {request.symbol}"
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel-and-replace-extended/{symbol}")
async def cancel_and_replace_extended(symbol: str):
    """
    Cancel existing orders for a symbol and resubmit as extended hours orders.

    Use this to convert pending market orders (that can't fill pre-market)
    into extended hours limit orders.
    """
    connector = get_alpaca_connector()

    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")

    try:
        result = connector.cancel_and_replace_extended(symbol)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# POSITION SYNC & SLIPPAGE COMBAT ENDPOINTS
# ============================================================================

@router.get("/sync/status")
async def get_sync_status():
    """Get position synchronizer status"""
    try:
        from ai.position_sync import get_position_sync
        sync = get_position_sync()
        return sync.get_sync_status()
    except Exception as e:
        return {"error": str(e), "sync_available": False}


@router.post("/sync/now")
async def force_sync():
    """Force immediate sync with Alpaca"""
    try:
        from ai.position_sync import get_position_sync
        sync = get_position_sync()
        result = sync.force_sync()
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/sync/start")
async def start_background_sync():
    """Start background position sync (every 5 seconds)"""
    try:
        from ai.position_sync import start_sync
        sync = start_sync()
        return {"success": True, "message": "Background sync started", "status": sync.get_sync_status()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/sync/stop")
async def stop_background_sync():
    """Stop background position sync"""
    try:
        from ai.position_sync import stop_sync
        stop_sync()
        return {"success": True, "message": "Background sync stopped"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/sync/positions")
async def get_synced_positions():
    """Get positions from sync cache (faster than direct API call)"""
    try:
        from ai.position_sync import get_position_sync
        sync = get_position_sync()
        positions = sync.get_positions()
        return {"positions": positions, "count": len(positions), "from_cache": True}
    except Exception as e:
        # Fallback to direct API call
        connector = get_alpaca_connector()
        positions = connector.get_positions()
        return {"positions": positions, "count": len(positions), "from_cache": False}


@router.get("/sync/orders")
async def get_synced_orders():
    """Get orders from sync cache (faster than direct API call)"""
    try:
        from ai.position_sync import get_position_sync
        sync = get_position_sync()
        orders = sync.get_orders()
        return {"orders": orders, "count": len(orders), "from_cache": True}
    except Exception as e:
        # Fallback to direct API call
        connector = get_alpaca_connector()
        orders = connector.get_orders(status="open")
        return {"orders": orders, "count": len(orders), "from_cache": False}


# ============================================================================
# MARKET REGIME & LOSS CUTTER ENDPOINTS
# ============================================================================

@router.get("/regime/current")
async def get_current_regime():
    """Get current market regime and trading phase"""
    try:
        from ai.market_regime import get_regime_detector
        detector = get_regime_detector()
        return detector.get_summary()
    except Exception as e:
        return {"error": str(e)}


@router.get("/regime/phase")
async def get_current_phase():
    """Get detailed info about current trading phase"""
    try:
        from ai.market_regime import get_regime_detector
        detector = get_regime_detector()
        return detector.get_phase_info()
    except Exception as e:
        return {"error": str(e)}


@router.get("/losscutter/evaluate")
async def evaluate_positions_for_cutting():
    """Evaluate all positions against loss cutting rules"""
    try:
        from ai.loss_cutter import get_loss_cutter
        connector = get_alpaca_connector()

        cutter = get_loss_cutter()
        positions = connector.get_positions()
        account = connector.get_account()

        evaluation = cutter.evaluate_all_positions(positions)
        capital_status = cutter.get_capital_preservation_status(positions, account)

        return {
            "evaluation": evaluation,
            "capital_status": capital_status
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/losscutter/cut-all")
async def cut_all_losing_positions():
    """IMMEDIATELY cut all positions exceeding loss thresholds"""
    try:
        from ai.loss_cutter import get_loss_cutter
        connector = get_alpaca_connector()

        cutter = get_loss_cutter()
        positions = connector.get_positions()

        to_cut = cutter.get_positions_to_cut(positions)
        results = []

        for symbol in to_cut:
            try:
                close_result = connector.close_position_smart(symbol)
                results.append({"symbol": symbol, "status": "CUT", "result": close_result})
                pos = next((p for p in positions if p.get("symbol") == symbol), {})
                cutter.record_cut(symbol, pos.get("unrealized_pl", 0), "Manual cut-all")
            except Exception as e:
                results.append({"symbol": symbol, "status": "FAILED", "error": str(e)})

        return {
            "success": True,
            "positions_cut": len([r for r in results if r["status"] == "CUT"]),
            "results": results
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# CLAUDE AI MORPHIC INTELLIGENCE ENDPOINTS
# ============================================================================

@router.get("/ai/status")
async def get_ai_status():
    """Get Claude AI intelligence status"""
    try:
        from ai.claude_bot_intelligence import get_bot_intelligence
        ai = get_bot_intelligence()
        return ai.get_status()
    except Exception as e:
        return {"error": str(e), "ai_available": False}


@router.post("/ai/optimize")
async def trigger_ai_optimization():
    """Trigger AI auto-optimization cycle"""
    try:
        from ai.claude_bot_intelligence import get_bot_intelligence
        ai = get_bot_intelligence()
        result = ai.auto_optimize()
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/adapt")
async def trigger_morphic_adaptation():
    """Trigger morphic adaptation based on current conditions"""
    try:
        from ai.claude_bot_intelligence import get_bot_intelligence
        from portfolio_analytics import get_portfolio_analytics
        from alpaca_integration import get_alpaca_connector

        ai = get_bot_intelligence()
        analytics = get_portfolio_analytics()
        connector = get_alpaca_connector()

        # Gather data
        account = connector.get_account()
        performance = analytics.get_portfolio_metrics()

        market_data = {
            "account": account,
            "equity": float(account.get("equity", 0))
        }

        result = ai.morphic_adapt(market_data, performance)
        return {"success": True, "adaptations": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/insights")
async def get_ai_insights(limit: int = 10):
    """Get recent AI insights"""
    try:
        from ai.claude_bot_intelligence import get_bot_intelligence
        ai = get_bot_intelligence()
        return {"insights": ai.get_recent_insights(limit)}
    except Exception as e:
        return {"error": str(e), "insights": []}


@router.post("/ai/chat")
async def ai_chat(data: dict):
    """
    Full conversational AI chat with Claude as the bot's brain.
    Claude IS the bot - not an external assistant.
    Uses the ClaudeBrain morphic controller for deep integration.
    """
    try:
        # Use the new ClaudeBrain for deep embedded experience
        from ai.claude_brain import get_claude_brain
        brain = get_claude_brain()

        message = data.get("message", "")
        session_id = data.get("session_id", "default")

        result = brain.think(message, session_id=session_id)

        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"AI chat error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/ai/chat/assistant")
async def ai_chat_assistant(data: dict):
    """
    Conversational AI chat as an assistant (original mode).
    Provides unrestricted conversation with trading context awareness.
    """
    try:
        from ai.claude_conversational_ai import get_conversational_ai
        ai = get_conversational_ai()

        message = data.get("message", "")
        session_id = data.get("session_id", "default")
        use_tools = data.get("use_tools", True)

        result = ai.chat(message, session_id=session_id, use_tools=use_tools)

        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"AI chat assistant error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/ai/chat/legacy")
async def ai_chat_legacy(data: dict):
    """Legacy chat endpoint using the original bot intelligence (for backwards compatibility)"""
    try:
        from ai.claude_bot_intelligence import get_bot_intelligence
        ai = get_bot_intelligence()
        message = data.get("message", "")
        use_tools = data.get("use_tools", True)

        if use_tools:
            result = ai.chat_with_tools(message, use_tools=True)
        else:
            result = ai.chat(message)

        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/chat/history")
async def get_chat_history(session_id: str = "default"):
    """Get conversation history for a session"""
    try:
        from ai.claude_conversational_ai import get_conversational_ai
        ai = get_conversational_ai()
        history = ai.get_conversation_history(session_id)
        return {"success": True, "history": history, "session_id": session_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/chat/clear")
async def clear_chat_history(data: dict):
    """Clear conversation history for a session"""
    try:
        from ai.claude_conversational_ai import get_conversational_ai
        ai = get_conversational_ai()
        session_id = data.get("session_id", "default")
        result = ai.clear_conversation(session_id)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/chat/sessions")
async def list_chat_sessions():
    """List all conversation sessions"""
    try:
        from ai.claude_conversational_ai import get_conversational_ai
        ai = get_conversational_ai()
        sessions = ai.list_sessions()
        return {"success": True, "sessions": sessions}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ==================== CLAUDE BRAIN / MORPHIC CONTROLLER ENDPOINTS ====================

@router.get("/ai/brain/state")
async def get_brain_state():
    """Get the bot's complete internal state as seen by Claude Brain"""
    try:
        from ai.claude_brain import get_claude_brain
        brain = get_claude_brain()
        state = brain.refresh_state()
        return {
            "success": True,
            "state": {
                "equity": state.equity,
                "buying_power": state.buying_power,
                "cash": state.cash,
                "positions_value": state.positions_value,
                "daily_pnl": state.daily_pnl,
                "position_count": state.position_count,
                "market_open": state.market_open,
                "market_regime": state.market_regime,
                "bot_mode": state.bot_mode,
                "auto_trading": state.auto_trading_enabled,
                "win_rate": state.win_rate,
                "profit_factor": state.profit_factor,
                "models_loaded": state.models_loaded
            },
            "ai_available": brain.ai_available,
            "tools_count": len(brain.tools)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/brain/modules")
async def list_brain_modules():
    """List all modules the brain can inspect and modify"""
    try:
        from ai.claude_brain import get_claude_brain
        brain = get_claude_brain()
        result = brain._execute_tool("list_my_modules", {})
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/brain/diagnose")
async def diagnose_module(data: dict):
    """Run diagnostics on a specific module"""
    try:
        from ai.claude_brain import get_claude_brain
        brain = get_claude_brain()
        module = data.get("module", "")
        deep = data.get("deep", False)
        result = brain._execute_tool("diagnose_module", {"module": module, "deep": deep})
        return {"success": True, "diagnostics": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/brain/enhancements")
async def get_enhancements():
    """Get list of proposed enhancements"""
    try:
        from ai.claude_brain import get_claude_brain
        brain = get_claude_brain()
        result = brain._execute_tool("get_enhancement_log", {})
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/brain/modifications")
async def get_modifications():
    """Get history of self-modifications"""
    try:
        from ai.claude_brain import get_claude_brain
        brain = get_claude_brain()
        result = brain._execute_tool("get_modification_history", {"limit": 50})
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/brain/analyze")
async def brain_analyze_position(data: dict):
    """Have the brain do a deep analysis of a position"""
    try:
        from ai.claude_brain import get_claude_brain
        brain = get_claude_brain()
        symbol = data.get("symbol", "")
        result = brain.analyze_position(symbol)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/brain/opportunities")
async def brain_find_opportunities():
    """Have the brain scan for trading opportunities"""
    try:
        from ai.claude_brain import get_claude_brain
        brain = get_claude_brain()
        result = brain.find_opportunities()
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/brain/daily-review")
async def brain_daily_review():
    """Have the brain do a daily portfolio review"""
    try:
        from ai.claude_brain import get_claude_brain
        brain = get_claude_brain()
        result = brain.daily_review()
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/heal")
async def trigger_self_healing(data: dict):
    """Manually trigger self-healing for an error"""
    try:
        from ai.claude_bot_intelligence import get_bot_intelligence
        ai = get_bot_intelligence()

        error_message = data.get("error", "Manual healing trigger")
        context = data.get("context", {})

        # Create a mock exception
        class ManualError(Exception):
            pass

        error = ManualError(error_message)
        result = ai.self_heal(error, context)

        return {"success": True, "healing_result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/comprehensive-analysis")
async def get_comprehensive_analysis():
    """Get comprehensive AI analysis of current trading situation"""
    try:
        from ai.claude_bot_intelligence import get_bot_intelligence
        ai = get_bot_intelligence()
        result = ai.comprehensive_analysis()
        return {"success": True, "analysis": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# MOMENTUM SPIKE DETECTOR - HYPER-FAST MONITORING
# ============================================================================

@router.post("/spikes/start")
async def start_spike_monitoring(data: dict = None):
    """
    Start the momentum spike detector.

    Pass a watchlist to monitor specific stocks:
    {"watchlist": ["PLRZ", "HTOO", "ARTL", ...]}
    """
    try:
        from ai.momentum_spike_detector import get_spike_detector

        detector = get_spike_detector()

        if data and data.get("watchlist"):
            detector.set_watchlist(data.get("watchlist"))

        detector.start_monitoring()

        return {
            "success": True,
            "message": "Spike monitoring STARTED - scanning every 15 seconds",
            "watchlist": detector.watchlist,
            "thresholds": detector.spike_thresholds
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/spikes/stop")
async def stop_spike_monitoring():
    """Stop the momentum spike detector"""
    try:
        from ai.momentum_spike_detector import get_spike_detector

        detector = get_spike_detector()
        detector.stop_monitoring()

        return {"success": True, "message": "Spike monitoring STOPPED"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/spikes/watchlist")
async def set_spike_watchlist(data: dict):
    """
    Set the watchlist for spike monitoring.

    {"watchlist": ["PLRZ", "HTOO", "ARTL"]}
    """
    try:
        from ai.momentum_spike_detector import get_spike_detector

        detector = get_spike_detector()
        watchlist = data.get("watchlist", [])
        detector.set_watchlist(watchlist)

        return {
            "success": True,
            "watchlist": detector.watchlist,
            "count": len(detector.watchlist)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/spikes/add/{symbol}")
async def add_to_spike_watchlist(symbol: str):
    """Add a single symbol to spike watchlist"""
    try:
        from ai.momentum_spike_detector import get_spike_detector

        detector = get_spike_detector()
        detector.add_to_watchlist(symbol)

        return {
            "success": True,
            "added": symbol,
            "watchlist": detector.watchlist
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.delete("/spikes/remove/{symbol}")
async def remove_from_spike_watchlist(symbol: str):
    """Remove a symbol from spike watchlist"""
    try:
        from ai.momentum_spike_detector import get_spike_detector

        detector = get_spike_detector()
        detector.remove_from_watchlist(symbol)

        return {
            "success": True,
            "removed": symbol,
            "watchlist": detector.watchlist
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/spikes/scan")
async def force_spike_scan():
    """Force immediate scan for momentum spikes"""
    try:
        from ai.momentum_spike_detector import get_spike_detector

        detector = get_spike_detector()
        result = detector.force_scan()

        return result
    except Exception as e:
        return {"error": str(e)}


@router.get("/spikes/alerts")
async def get_spike_alerts():
    """Get current active spike alerts"""
    try:
        from ai.momentum_spike_detector import get_spike_detector

        detector = get_spike_detector()

        return {
            "active_alerts": detector.get_active_alerts(),
            "recent_alerts": detector.get_recent_alerts(30),
            "monitoring": detector._running,
            "watchlist": detector.watchlist
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/spikes/status")
async def get_spike_status():
    """Get spike detector status"""
    try:
        from ai.momentum_spike_detector import get_spike_detector

        detector = get_spike_detector()

        return {
            "monitoring": detector._running,
            "scan_interval_seconds": detector.scan_interval_seconds,
            "watchlist_count": len(detector.watchlist),
            "watchlist": detector.watchlist,
            "thresholds": detector.spike_thresholds,
            "active_alerts": len(detector.active_alerts),
            "watchlist_status": detector.get_watchlist_status()
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/spikes/auto-watchlist")
async def auto_populate_spike_watchlist():
    """
    Auto-populate watchlist from scanner results and current positions.
    Gets pre-market gappers + current holdings.
    """
    try:
        from ai.momentum_spike_detector import get_spike_detector
        connector = get_alpaca_connector()

        detector = get_spike_detector()

        # Get current positions
        positions = connector.get_positions()
        position_symbols = [p.get("symbol") for p in positions]

        # Get scanner results if available
        scanner_symbols = []
        try:
            from alpaca_scanner import get_scanner
            scanner = get_scanner()
            results = scanner.scan_premarket_gappers(min_gap_pct=3.0, max_results=10)
            scanner_symbols = [r.get("symbol") for r in results if r.get("symbol")]
        except:
            pass

        # Combine and dedupe
        all_symbols = list(set(position_symbols + scanner_symbols))

        if all_symbols:
            detector.set_watchlist(all_symbols)

        return {
            "success": True,
            "watchlist": detector.watchlist,
            "from_positions": position_symbols,
            "from_scanner": scanner_symbols,
            "total": len(detector.watchlist)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# WARRIOR NEWS DETECTOR - Breaking news scalp alerts
# ============================================================================

@router.post("/news/start")
async def start_news_detector(symbols: list = None):
    """
    Start Warrior News Detector - hyper-fast news monitoring.
    Polls every 10 seconds for breaking news.
    When catalyst detected, generates scalp signals.
    """
    try:
        from ai.warrior_news_detector import get_news_detector, setup_news_to_spike_integration

        detector = get_news_detector()

        # Setup integration with spike detector
        setup_news_to_spike_integration()

        # Start monitoring
        detector.start_monitoring(symbols)

        return {
            "success": True,
            "status": "started",
            "poll_interval": detector.poll_interval,
            "watched_symbols": detector.watched_symbols or "ALL"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/news/stop")
async def stop_news_detector():
    """Stop news monitoring"""
    try:
        from ai.warrior_news_detector import get_news_detector

        detector = get_news_detector()
        detector.stop_monitoring()

        return {"success": True, "status": "stopped"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/news/status")
async def get_news_status():
    """Get news detector status"""
    try:
        from ai.warrior_news_detector import get_news_detector

        detector = get_news_detector()
        status = detector.get_status()

        return {
            "success": True,
            **status
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/news/alerts")
async def get_news_alerts():
    """Get active news alerts within scalp window"""
    try:
        from ai.warrior_news_detector import get_news_detector, NewsUrgency

        detector = get_news_detector()

        # Get all active alerts
        all_alerts = detector.get_active_alerts()

        # Get actionable scalp signals
        scalp_signals = detector.get_scalp_signals()

        return {
            "success": True,
            "active_alerts": [a.to_dict() for a in all_alerts],
            "scalp_signals": [a.to_dict() for a in scalp_signals],
            "total_alerts": len(all_alerts),
            "actionable_signals": len(scalp_signals)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/news/signals")
async def get_scalp_signals():
    """Get actionable scalp signals from breaking news"""
    try:
        from ai.warrior_news_detector import get_news_detector

        detector = get_news_detector()
        signals = detector.get_scalp_signals()

        return {
            "success": True,
            "signals": [s.to_dict() for s in signals],
            "count": len(signals)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/news/poll")
async def force_news_poll():
    """Force immediate news poll"""
    try:
        from ai.warrior_news_detector import get_news_detector

        detector = get_news_detector()
        result = detector.force_poll()

        return {
            "success": True,
            **result
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/news/history")
async def get_news_history(limit: int = 50):
    """Get news alert history"""
    try:
        from ai.warrior_news_detector import get_news_detector

        detector = get_news_detector()
        history = detector.alert_history[:limit]

        return {
            "success": True,
            "history": [a.to_dict() for a in history],
            "count": len(history)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/news/symbols")
async def set_news_symbols(symbols: list):
    """Set symbols to watch for news"""
    try:
        from ai.warrior_news_detector import get_news_detector

        detector = get_news_detector()
        detector.watched_symbols = symbols

        return {
            "success": True,
            "watched_symbols": symbols
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/news/auto-start")
async def auto_start_news_detector():
    """
    Auto-start news detector with symbols from:
    1. Current positions
    2. Spike detector watchlist
    3. Scanner results
    """
    try:
        from ai.warrior_news_detector import get_news_detector, setup_news_to_spike_integration
        from ai.momentum_spike_detector import get_spike_detector

        connector = get_alpaca_connector()
        news_detector = get_news_detector()

        # Gather symbols from various sources
        all_symbols = []

        # From positions
        try:
            positions = connector.get_positions()
            all_symbols.extend([p.get("symbol") for p in positions])
        except:
            pass

        # From spike detector
        try:
            spike_detector = get_spike_detector()
            all_symbols.extend(spike_detector.watchlist)
        except:
            pass

        # From scanner
        try:
            from alpaca_scanner import get_scanner
            scanner = get_scanner()
            results = scanner.scan_premarket_gappers(min_gap_pct=3.0, max_results=10)
            all_symbols.extend([r.get("symbol") for r in results if r.get("symbol")])
        except:
            pass

        # Dedupe
        all_symbols = list(set(all_symbols))

        # Setup integration
        setup_news_to_spike_integration()

        # Start monitoring
        news_detector.start_monitoring(all_symbols if all_symbols else None)

        return {
            "success": True,
            "status": "started",
            "watched_symbols": all_symbols or "ALL",
            "poll_interval": news_detector.poll_interval
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# REAL-TIME NEWS STREAM - Dedicated WebSocket (INSTANT!)
# ============================================================================

@router.post("/news-stream/start")
async def start_news_stream(symbols: list = None):
    """
    Start REAL-TIME news stream via dedicated WebSocket.

    This is INSTANT - no polling delay!
    Uses Alpaca's dedicated news WebSocket: wss://stream.data.alpaca.markets/v1beta1/news

    News arrives the moment it's published.
    """
    try:
        from ai.warrior_news_stream import get_news_stream

        stream = get_news_stream()
        stream.start(symbols)

        return {
            "success": True,
            "status": "started",
            "stream_type": "REAL-TIME WebSocket",
            "latency": "INSTANT (no polling)",
            "watched_symbols": symbols or "ALL"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/news-stream/stop")
async def stop_news_stream():
    """Stop the real-time news stream"""
    try:
        from ai.warrior_news_stream import get_news_stream

        stream = get_news_stream()
        stream.stop()

        return {"success": True, "status": "stopped"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/news-stream/status")
async def get_news_stream_status():
    """Get real-time news stream status"""
    try:
        from ai.warrior_news_stream import get_news_stream

        stream = get_news_stream()
        status = stream.get_status()

        return {
            "success": True,
            **status
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/news-stream/alerts")
async def get_news_stream_alerts():
    """Get active alerts from real-time stream"""
    try:
        from ai.warrior_news_stream import get_news_stream

        stream = get_news_stream()
        all_alerts = stream.get_active_alerts()
        scalp_signals = stream.get_scalp_signals()

        return {
            "success": True,
            "active_alerts": [a.to_dict() for a in all_alerts],
            "scalp_signals": [a.to_dict() for a in scalp_signals],
            "total_alerts": len(all_alerts),
            "actionable_signals": len(scalp_signals)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/news-stream/signals")
async def get_news_stream_signals():
    """Get actionable scalp signals from real-time news"""
    try:
        from ai.warrior_news_stream import get_news_stream

        stream = get_news_stream()
        signals = stream.get_scalp_signals()

        return {
            "success": True,
            "signals": [s.to_dict() for s in signals],
            "count": len(signals)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/news-stream/auto-start")
async def auto_start_news_stream():
    """
    Auto-start real-time news stream with symbols from positions + scanner.
    """
    try:
        from ai.warrior_news_stream import get_news_stream
        from ai.momentum_spike_detector import get_spike_detector

        connector = get_alpaca_connector()
        stream = get_news_stream()

        # Gather symbols
        all_symbols = []

        # From positions
        try:
            positions = connector.get_positions()
            all_symbols.extend([p.get("symbol") for p in positions])
        except:
            pass

        # From spike detector
        try:
            spike_detector = get_spike_detector()
            all_symbols.extend(spike_detector.watchlist)
        except:
            pass

        # Dedupe
        all_symbols = list(set(all_symbols))

        # Start stream (empty = ALL news which is actually better for catching catalysts)
        stream.start(all_symbols if all_symbols else None)

        return {
            "success": True,
            "status": "started",
            "stream_type": "REAL-TIME WebSocket",
            "latency": "INSTANT",
            "watched_symbols": all_symbols or "ALL"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# MACD INDICATOR - Only trade WITH the MACD!
# ============================================================================

@router.get("/macd/{symbol}")
async def get_macd(symbol: str):
    """
    Get MACD indicator for a symbol.

    WARRIOR TRADING RULE:
    - MACD positive (above signal) = BUY ONLY
    - MACD negative (below signal) = SELL ONLY / NO NEW LONGS
    """
    try:
        from ai.macd_indicator import get_macd_indicator
        from alpaca_market_data import get_alpaca_market_data

        indicator = get_macd_indicator()
        market_data = get_alpaca_market_data()

        # Get recent bars to build MACD (DataFrame)
        df = market_data.get_historical_bars(symbol.upper(), timeframe="5Min", limit=50)

        if df.empty:
            return {"success": False, "error": "No price data available"}

        # Feed prices to MACD indicator (DataFrame has Close column)
        macd_data = None
        for idx, row in df.iterrows():
            price = row.get("Close", 0)
            if price > 0:
                macd_data = indicator.add_price(symbol.upper(), price)

        if not macd_data:
            return {
                "success": False,
                "error": "Not enough data to calculate MACD (need 26+ bars)"
            }

        # Get buy/sell recommendations
        ok_to_buy, buy_reason = indicator.is_ok_to_buy(symbol.upper())
        should_sell, sell_reason = indicator.should_sell(symbol.upper())
        signal_strength = indicator.get_signal_strength(symbol.upper())

        return {
            "success": True,
            "symbol": symbol.upper(),
            "macd": macd_data.to_dict(),
            "ok_to_buy": ok_to_buy,
            "buy_reason": buy_reason,
            "should_sell": should_sell,
            "sell_reason": sell_reason,
            "signal_strength": signal_strength,
            "signal_strength_label": ["", "STRONG SELL", "SELL", "NEUTRAL", "BUY", "STRONG BUY"][signal_strength]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/macd/{symbol}/ok-to-buy")
async def macd_ok_to_buy(symbol: str):
    """
    WARRIOR CHECK: Is it OK to buy this stock based on MACD?

    Returns True only if MACD is positive and above signal line.
    """
    try:
        from ai.macd_indicator import get_macd_indicator
        from alpaca_market_data import get_alpaca_market_data

        indicator = get_macd_indicator()
        market_data = get_alpaca_market_data()

        # Get recent bars (DataFrame)
        df = market_data.get_historical_bars(symbol.upper(), timeframe="5Min", limit=50)

        if df.empty:
            return {
                "success": True,
                "symbol": symbol.upper(),
                "ok_to_buy": False,
                "reason": "No price data available"
            }

        # Feed to indicator
        for idx, row in df.iterrows():
            price = row.get("Close", 0)
            if price > 0:
                indicator.add_price(symbol.upper(), price)

        ok_to_buy, reason = indicator.is_ok_to_buy(symbol.upper())

        return {
            "success": True,
            "symbol": symbol.upper(),
            "ok_to_buy": ok_to_buy,
            "reason": reason
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/macd/{symbol}/should-sell")
async def macd_should_sell(symbol: str):
    """
    WARRIOR CHECK: Should we sell/exit this position based on MACD?
    """
    try:
        from ai.macd_indicator import get_macd_indicator
        from alpaca_market_data import get_alpaca_market_data

        indicator = get_macd_indicator()
        market_data = get_alpaca_market_data()

        # Get recent bars (DataFrame)
        df = market_data.get_historical_bars(symbol.upper(), timeframe="5Min", limit=50)

        if df.empty:
            return {
                "success": True,
                "symbol": symbol.upper(),
                "should_sell": False,
                "reason": "No price data available"
            }

        # Feed to indicator
        for idx, row in df.iterrows():
            price = row.get("Close", 0)
            if price > 0:
                indicator.add_price(symbol.upper(), price)

        should_sell, reason = indicator.should_sell(symbol.upper())

        return {
            "success": True,
            "symbol": symbol.upper(),
            "should_sell": should_sell,
            "reason": reason
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/macd/signals/all")
async def get_all_macd_signals():
    """Get MACD signals for all positions"""
    try:
        from ai.macd_indicator import get_macd_indicator
        from alpaca_market_data import get_alpaca_market_data

        connector = get_alpaca_connector()
        indicator = get_macd_indicator()
        market_data = get_alpaca_market_data()

        # Get current positions
        positions = connector.get_positions()

        results = {}
        for pos in positions:
            symbol = pos.get("symbol")

            # Get bars and calculate MACD (DataFrame)
            df = market_data.get_historical_bars(symbol, timeframe="5Min", limit=50)

            if not df.empty:
                for idx, row in df.iterrows():
                    price = row.get("Close", 0)
                    if price > 0:
                        indicator.add_price(symbol, price)

                macd = indicator.get_macd(symbol)
                ok_to_buy, buy_reason = indicator.is_ok_to_buy(symbol)
                should_sell, sell_reason = indicator.should_sell(symbol)

                results[symbol] = {
                    "macd": macd.to_dict() if macd else None,
                    "ok_to_buy": ok_to_buy,
                    "buy_reason": buy_reason,
                    "should_sell": should_sell,
                    "sell_reason": sell_reason,
                    "signal_strength": indicator.get_signal_strength(symbol),
                    "position_qty": pos.get("quantity"),
                    "unrealized_pl": pos.get("unrealized_pl")
                }

        return {
            "success": True,
            "signals": results,
            "position_count": len(positions)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# ASSET STATUS - Halt, ETB, HTB, SSR, Shortable
# ============================================================================

@router.get("/asset/{symbol}/status")
async def get_asset_status(symbol: str):
    """
    Get complete asset status including:
    - Tradeable (is it halted?)
    - Shortable (can you short it?)
    - Easy to Borrow (ETB) vs Hard to Borrow (HTB)
    - SSR (Short Sale Restriction) status
    - Marginable

    CRITICAL FOR DAY TRADING:
    - If tradable=False, stock is HALTED - can't trade!
    - If easy_to_borrow=False, it's HTB (higher borrowing cost)
    - SSR means can only short on upticks
    """
    try:
        connector = get_alpaca_connector()
        asset = connector.trading_client.get_asset(symbol.upper())

        # Extract status
        status_str = str(asset.status).split('.')[-1] if asset.status else "UNKNOWN"
        is_halted = not asset.tradable or status_str not in ["ACTIVE", "active"]

        # Determine borrow status
        if not asset.shortable:
            borrow_status = "NOT_SHORTABLE"
        elif asset.easy_to_borrow:
            borrow_status = "ETB"  # Easy To Borrow
        else:
            borrow_status = "HTB"  # Hard To Borrow

        return {
            "success": True,
            "symbol": symbol.upper(),
            "name": asset.name,
            "exchange": str(asset.exchange).split('.')[-1] if asset.exchange else None,

            # Trading Status
            "tradable": asset.tradable,
            "is_halted": is_halted,
            "status": status_str,

            # Shorting Status
            "shortable": asset.shortable,
            "easy_to_borrow": asset.easy_to_borrow,
            "borrow_status": borrow_status,

            # Margin
            "marginable": asset.marginable,
            "maintenance_margin_requirement": asset.maintenance_margin_requirement,

            # Other
            "fractionable": asset.fractionable,

            # Quick indicators for UI
            "indicators": {
                "halt": is_halted,
                "etb": asset.easy_to_borrow and asset.shortable,
                "htb": not asset.easy_to_borrow and asset.shortable,
                "no_short": not asset.shortable
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/asset/{symbol}/halt-check")
async def check_halt_status(symbol: str):
    """
    Quick halt check for a symbol.

    Returns simple is_halted status and reason.
    """
    try:
        connector = get_alpaca_connector()
        asset = connector.trading_client.get_asset(symbol.upper())

        status_str = str(asset.status).split('.')[-1] if asset.status else "UNKNOWN"
        is_halted = not asset.tradable or status_str not in ["ACTIVE", "active"]

        if is_halted:
            if not asset.tradable:
                reason = "Asset not tradeable"
            else:
                reason = f"Status: {status_str}"
        else:
            reason = "Trading normally"

        return {
            "success": True,
            "symbol": symbol.upper(),
            "is_halted": is_halted,
            "tradable": asset.tradable,
            "status": status_str,
            "reason": reason
        }
    except Exception as e:
        return {"success": False, "error": str(e), "is_halted": False}


@router.get("/positions/status")
async def get_positions_status():
    """
    Get halt/borrow status for all positions.

    CRITICAL: Know if any of your positions got halted!
    """
    try:
        connector = get_alpaca_connector()
        positions = connector.get_positions()

        results = {}
        halted_positions = []
        htb_positions = []

        for pos in positions:
            symbol = pos.get("symbol")
            try:
                asset = connector.trading_client.get_asset(symbol)

                status_str = str(asset.status).split('.')[-1] if asset.status else "UNKNOWN"
                is_halted = not asset.tradable or status_str not in ["ACTIVE", "active"]

                if is_halted:
                    halted_positions.append(symbol)

                if asset.shortable and not asset.easy_to_borrow:
                    htb_positions.append(symbol)

                # Determine borrow status
                if not asset.shortable:
                    borrow_status = "NOT_SHORTABLE"
                elif asset.easy_to_borrow:
                    borrow_status = "ETB"
                else:
                    borrow_status = "HTB"

                results[symbol] = {
                    "is_halted": is_halted,
                    "tradable": asset.tradable,
                    "status": status_str,
                    "shortable": asset.shortable,
                    "easy_to_borrow": asset.easy_to_borrow,
                    "borrow_status": borrow_status,
                    "quantity": pos.get("quantity"),
                    "unrealized_pl": pos.get("unrealized_pl")
                }
            except Exception as e:
                results[symbol] = {
                    "error": str(e),
                    "is_halted": False,
                    "tradable": True
                }

        return {
            "success": True,
            "positions": results,
            "position_count": len(positions),
            "halted_count": len(halted_positions),
            "halted_symbols": halted_positions,
            "htb_count": len(htb_positions),
            "htb_symbols": htb_positions,
            "alert": f"WARNING: {len(halted_positions)} position(s) HALTED!" if halted_positions else None
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# AI INTELLIGENCE ENDPOINTS - Symbol Memory, Signal Explainer, Trade Reasoner
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/ai/memory/summary")
async def get_memory_summary():
    """Get overall symbol memory summary"""
    try:
        from ai.symbol_memory import get_symbol_memory
        memory = get_symbol_memory()
        return {"success": True, **memory.get_summary()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/memory/{symbol}")
async def get_symbol_memory_stats(symbol: str):
    """Get memory stats for a specific symbol"""
    try:
        from ai.symbol_memory import get_symbol_memory
        memory = get_symbol_memory()
        stats = memory.get_stats(symbol.upper())

        if stats:
            should_trade, reason = memory.should_trade(symbol)
            size_mult = memory.get_position_size_multiplier(symbol)

            return {
                "success": True,
                "symbol": symbol.upper(),
                "stats": stats.to_dict(),
                "should_trade": should_trade,
                "trade_reason": reason,
                "size_multiplier": size_mult
            }
        else:
            return {
                "success": True,
                "symbol": symbol.upper(),
                "stats": None,
                "message": "No trade history for this symbol"
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/memory/best")
async def get_best_symbols():
    """Get top performing symbols"""
    try:
        from ai.symbol_memory import get_symbol_memory
        memory = get_symbol_memory()
        best = memory.get_best_symbols(min_trades=3, top_n=10)

        return {
            "success": True,
            "best_symbols": [
                {"symbol": sym, "stats": stats.to_dict()}
                for sym, stats in best
            ]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/memory/worst")
async def get_worst_symbols():
    """Get worst performing symbols (avoid these!)"""
    try:
        from ai.symbol_memory import get_symbol_memory
        memory = get_symbol_memory()
        worst = memory.get_worst_symbols(min_trades=3, top_n=10)

        return {
            "success": True,
            "avoid_symbols": [
                {"symbol": sym, "stats": stats.to_dict()}
                for sym, stats in worst
            ]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/memory/time-analysis")
async def get_time_of_day_analysis(symbol: str = None):
    """Get performance by time of day"""
    try:
        from ai.symbol_memory import get_symbol_memory
        memory = get_symbol_memory()
        analysis = memory.get_time_of_day_analysis(symbol)

        return {
            "success": True,
            "symbol": symbol,
            "hourly_performance": analysis
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/memory/record-trade")
async def record_trade(trade_data: dict):
    """Record a completed trade to memory"""
    try:
        from ai.symbol_memory import get_symbol_memory
        from datetime import datetime

        memory = get_symbol_memory()

        # Parse dates
        entry_time = datetime.fromisoformat(trade_data.get('entry_time', datetime.now().isoformat()))
        exit_time = datetime.fromisoformat(trade_data.get('exit_time', datetime.now().isoformat()))

        trade = memory.record_trade(
            symbol=trade_data['symbol'],
            side=trade_data['side'],
            entry_price=trade_data['entry_price'],
            exit_price=trade_data['exit_price'],
            quantity=trade_data['quantity'],
            entry_time=entry_time,
            exit_time=exit_time,
            prediction_confidence=trade_data.get('confidence', 0),
            indicators=trade_data.get('indicators', {})
        )

        return {"success": True, "trade": trade.to_dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/explain/{symbol}")
async def explain_prediction(symbol: str):
    """Get explanation for current prediction on a symbol"""
    try:
        from ai.signal_explainer import get_signal_explainer
        from ai.alpaca_ai_predictor import get_alpaca_predictor

        explainer = get_signal_explainer()
        predictor = get_alpaca_predictor()

        # Get current prediction
        result = predictor.predict(symbol.upper())

        if result and result.get('prediction'):
            # Build feature dict from prediction data
            features = {
                "macd": result.get('macd', 0),
                "macd_signal": result.get('macd_signal', 0),
                "macd_histogram": result.get('macd_histogram', 0),
                "rsi": result.get('rsi', 50),
                "volume_ratio": result.get('volume_ratio', 1.0),
                "trend": result.get('trend', 0),
                "price_change_pct": result.get('change_percent', 0),
            }

            explanation = explainer.explain_prediction(
                symbol=symbol.upper(),
                features=features,
                prediction=result['prediction'],
                confidence=result.get('confidence', 50)
            )

            return {
                "success": True,
                "explanation": explanation.to_dict()
            }
        else:
            return {"success": False, "error": "No prediction available"}

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/reason-trade")
async def generate_trade_reasoning(data: dict):
    """Generate LLM-powered trade reasoning"""
    try:
        from ai.llm_trade_reasoner import get_trade_reasoner

        reasoner = get_trade_reasoner()

        reasoning = reasoner.generate_trade_reasoning(
            symbol=data['symbol'],
            action=data['action'],
            price=data['price'],
            indicators=data.get('indicators', {}),
            history=data.get('history'),
            news=data.get('news')
        )

        return {
            "success": True,
            "reasoning": reasoning.to_dict(),
            "journal_entry": reasoning.to_journal_entry()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/journal")
async def get_trade_journal(symbol: str = None, limit: int = 20):
    """Get trade journal entries"""
    try:
        from ai.llm_trade_reasoner import get_trade_reasoner

        reasoner = get_trade_reasoner()
        entries = reasoner.get_journal_entries(symbol, limit)

        return {
            "success": True,
            "entries": entries,
            "count": len(entries)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/sectors")
async def get_sector_performance():
    """Get performance aggregated by sector"""
    try:
        from ai.symbol_memory import get_symbol_memory
        memory = get_symbol_memory()
        sectors = memory.get_sector_performance()

        return {
            "success": True,
            "sectors": sectors
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER ENDPOINTS - Drawdown Protection
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/ai/circuit-breaker/status")
async def get_circuit_breaker_status():
    """Get current circuit breaker status"""
    try:
        from ai.circuit_breaker import get_circuit_breaker

        breaker = get_circuit_breaker()

        # Initialize with current account equity if not already
        connector = get_alpaca_connector()
        account = connector.trading_client.get_account()
        equity = float(account.equity)

        breaker.initialize_day(equity)
        breaker.update_equity(equity)

        state = breaker.check_breaker()

        return {
            "success": True,
            "status": state.to_dict(),
            "account_equity": equity
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/circuit-breaker/can-trade")
async def can_open_trade():
    """Quick check if trading is allowed"""
    try:
        from ai.circuit_breaker import get_circuit_breaker

        breaker = get_circuit_breaker()
        can_trade, reason, size_mult = breaker.can_open_trade()

        return {
            "success": True,
            "can_trade": can_trade,
            "reason": reason,
            "size_multiplier": size_mult,
            "level": breaker.current_level.value
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/circuit-breaker/record-trade")
async def record_trade_for_breaker(data: dict):
    """Record a trade to the circuit breaker"""
    try:
        from ai.circuit_breaker import get_circuit_breaker

        breaker = get_circuit_breaker()
        breaker.record_trade(
            pnl=data['pnl'],
            symbol=data.get('symbol', '')
        )

        state = breaker.check_breaker()

        return {
            "success": True,
            "status": state.to_dict()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/circuit-breaker/daily-summary")
async def get_breaker_daily_summary():
    """Get daily trading summary"""
    try:
        from ai.circuit_breaker import get_circuit_breaker

        breaker = get_circuit_breaker()
        summary = breaker.get_daily_summary()

        return {
            "success": True,
            **summary
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/circuit-breaker/history")
async def get_breaker_history(days: int = 7):
    """Get circuit breaker history"""
    try:
        from ai.circuit_breaker import get_circuit_breaker

        breaker = get_circuit_breaker()
        history = breaker.get_history(days)

        return {
            "success": True,
            "history": history,
            "days": len(history)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/circuit-breaker/reset")
async def reset_circuit_breaker():
    """Manually reset the circuit breaker (use with caution!)"""
    try:
        from ai.circuit_breaker import get_circuit_breaker

        breaker = get_circuit_breaker()
        breaker.reset_breaker()

        return {
            "success": True,
            "message": "Circuit breaker reset",
            "warning": "Use this with caution - it overrides safety limits"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/circuit-breaker/settings")
async def update_breaker_settings(data: dict):
    """Update circuit breaker thresholds"""
    try:
        from ai.circuit_breaker import get_circuit_breaker

        breaker = get_circuit_breaker()
        breaker.update_thresholds(**data)

        return {
            "success": True,
            "message": "Settings updated",
            "current_settings": {
                "daily_loss_warning": breaker.daily_loss_warning,
                "daily_loss_caution": breaker.daily_loss_caution,
                "daily_loss_halt": breaker.daily_loss_halt,
                "consecutive_loss_halt": breaker.consecutive_loss_halt,
                "max_trades_halt": breaker.max_trades_halt
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND BRAIN ENDPOINTS - Continuous Learning & Adaptation
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/ai/brain/status")
async def get_brain_status():
    """Get background brain status and metrics"""
    try:
        from ai.background_brain import get_background_brain

        brain = get_background_brain()
        metrics = brain.get_metrics()

        return {
            "success": True,
            "running": brain.is_running,
            "metrics": metrics.to_dict(),
            "market_regime": brain.get_market_regime()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/brain/start")
async def start_brain(data: dict = None):
    """
    Start the background AI brain.

    Optional params:
    - cpu_target: Float 0.1-0.9, percentage of CPU to use (default 0.5 = 50%)
    """
    try:
        from ai.background_brain import start_brain as start_brain_fn, get_background_brain

        cpu_target = 0.5
        if data and 'cpu_target' in data:
            cpu_target = float(data['cpu_target'])

        brain = start_brain_fn(cpu_target)

        return {
            "success": True,
            "message": f"Background brain started ({cpu_target*100:.0f}% CPU target)",
            "worker_threads": brain.worker_threads,
            "num_cores": brain.num_cores
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/brain/stop")
async def stop_brain():
    """Stop the background AI brain"""
    try:
        from ai.background_brain import get_background_brain

        brain = get_background_brain()
        brain.stop()

        return {
            "success": True,
            "message": "Background brain stopped"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/brain/cpu-target")
async def set_brain_cpu_target(data: dict):
    """
    Adjust CPU target for background brain.

    {"cpu_target": 0.7}  # 70% CPU usage
    """
    try:
        from ai.background_brain import get_background_brain

        brain = get_background_brain()
        cpu_target = float(data.get('cpu_target', 0.5))
        brain.set_cpu_target(cpu_target)

        return {
            "success": True,
            "cpu_target": brain.cpu_target,
            "worker_threads": brain.worker_threads
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/brain/regime")
async def get_market_regime():
    """Get current detected market regime"""
    try:
        from ai.background_brain import get_background_brain

        brain = get_background_brain()
        regime = brain.get_market_regime()

        return {
            "success": True,
            "regime": regime
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/brain/record-prediction")
async def record_brain_prediction(data: dict):
    """Record a prediction for later evaluation"""
    try:
        from ai.background_brain import get_background_brain

        brain = get_background_brain()
        brain.record_prediction(
            symbol=data['symbol'],
            prediction=data['prediction'],
            confidence=data.get('confidence', 0),
            price=data.get('price', 0)
        )

        return {
            "success": True,
            "message": "Prediction recorded"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/brain/record-outcome")
async def record_brain_outcome(data: dict):
    """Record actual outcome for prediction evaluation"""
    try:
        from ai.background_brain import get_background_brain

        brain = get_background_brain()
        brain.record_outcome(
            symbol=data['symbol'],
            timestamp=data['timestamp'],
            actual_outcome=data['outcome']
        )

        return {
            "success": True,
            "message": "Outcome recorded"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/brain/metrics")
async def get_brain_metrics():
    """Get detailed brain performance metrics"""
    try:
        from ai.background_brain import get_background_brain

        brain = get_background_brain()
        metrics = brain.get_metrics()

        return {
            "success": True,
            "metrics": metrics.to_dict()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#                          CIRCUIT BREAKER ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/ai/circuit-breaker/status")
async def get_circuit_breaker_status():
    """Get circuit breaker status and drawdown info"""
    try:
        from ai.circuit_breaker import get_circuit_breaker

        breaker = get_circuit_breaker()

        # Get current equity from Alpaca
        connector = get_alpaca_connector()
        current_equity = 0
        if connector.is_connected():
            account = connector.get_account()
            current_equity = float(account.get('equity', 0))

        # Check current level
        level = breaker.check_trade_allowed(current_equity)

        # Calculate P&L and drawdown
        pnl = current_equity - breaker.starting_equity if breaker.starting_equity > 0 else 0
        drawdown_pct = (breaker.starting_equity - current_equity) / breaker.starting_equity if breaker.starting_equity > 0 else 0

        return {
            "success": True,
            "level": level,
            "starting_equity": breaker.starting_equity,
            "current_equity": current_equity,
            "pnl": pnl,
            "drawdown_pct": max(0, drawdown_pct),
            "warning_threshold": breaker.daily_loss_warning,
            "halt_threshold": breaker.daily_loss_halt,
            "trades_today": breaker.trades_today,
            "trading_allowed": level != "HALT"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/circuit-breaker/reset")
async def reset_circuit_breaker():
    """Reset circuit breaker for new trading day"""
    try:
        from ai.circuit_breaker import get_circuit_breaker

        breaker = get_circuit_breaker()

        # Get current equity from Alpaca
        connector = get_alpaca_connector()
        current_equity = 0
        if connector.is_connected():
            account = connector.get_account()
            current_equity = float(account.get('equity', 0))

        breaker.initialize_day(current_equity)

        return {
            "success": True,
            "message": "Circuit breaker reset",
            "starting_equity": current_equity
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#                          TRADE JOURNAL ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/ai/trade-journal")
async def get_trade_journal(symbol: str = None, limit: int = 20):
    """Get trade journal entries"""
    try:
        from ai.llm_trade_reasoner import get_trade_reasoner

        reasoner = get_trade_reasoner()
        entries = reasoner.get_journal_entries(symbol=symbol, limit=limit)

        return {
            "success": True,
            "entries": entries,
            "count": len(entries)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/trade-reasoning")
async def generate_trade_reasoning(data: dict):
    """Generate AI trade reasoning for a trade"""
    try:
        from ai.llm_trade_reasoner import get_trade_reasoner

        reasoner = get_trade_reasoner()

        symbol = data.get('symbol', '')
        action = data.get('action', 'BUY')
        price = float(data.get('price', 0))
        indicators = data.get('indicators', {})

        reasoning = reasoner.generate_trade_reasoning(
            symbol=symbol,
            action=action,
            price=price,
            indicators=indicators
        )

        return {
            "success": True,
            "reasoning": reasoning.to_dict(),
            "journal_entry": reasoning.to_journal_entry()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#                          SYMBOL MEMORY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/ai/symbol-memory/{symbol}")
async def get_symbol_memory(symbol: str):
    """Get trading memory for a specific symbol"""
    try:
        from ai.symbol_memory import get_symbol_memory

        memory = get_symbol_memory()
        symbol_data = memory.get_symbol_data(symbol.upper())

        if symbol_data:
            return {
                "success": True,
                "symbol": symbol.upper(),
                "memory": symbol_data.to_dict()
            }
        else:
            return {
                "success": True,
                "symbol": symbol.upper(),
                "memory": None,
                "message": f"No trading history for {symbol}"
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/symbol-memory/{symbol}/record")
async def record_symbol_trade(symbol: str, data: dict):
    """Record a trade for symbol memory"""
    try:
        from ai.symbol_memory import get_symbol_memory

        memory = get_symbol_memory()

        pnl = float(data.get('pnl', 0))
        entry_price = float(data.get('entry_price', 0))
        exit_price = float(data.get('exit_price', 0))
        volume = int(data.get('volume', 0))
        ai_confidence = float(data.get('ai_confidence', 0))

        memory.record_trade(
            symbol=symbol.upper(),
            pnl=pnl,
            entry_price=entry_price,
            exit_price=exit_price,
            volume=volume,
            ai_confidence=ai_confidence
        )

        return {
            "success": True,
            "message": f"Trade recorded for {symbol.upper()}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/symbol-memory/{symbol}/position-size")
async def get_recommended_position_size(symbol: str, account_value: float = 100000):
    """Get AI-recommended position size based on symbol history"""
    try:
        from ai.symbol_memory import get_symbol_memory

        memory = get_symbol_memory()
        recommended_size = memory.get_recommended_size(symbol.upper(), account_value)

        return {
            "success": True,
            "symbol": symbol.upper(),
            "account_value": account_value,
            "recommended_size": recommended_size,
            "recommended_allocation": recommended_size / account_value if account_value > 0 else 0
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
