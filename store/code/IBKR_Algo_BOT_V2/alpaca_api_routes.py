"""
Alpaca API Routes for Dashboard
"""
import logging
import time
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, Tuple
from alpaca_integration import get_alpaca_connector

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/alpaca", tags=["alpaca"])

# ============================================================================
# DUPLICATE ORDER PREVENTION
# ============================================================================
# Track recent orders to prevent double-clicks/duplicate submissions
_recent_orders: Dict[str, float] = {}  # key: "symbol_side_qty" -> timestamp
DUPLICATE_PREVENTION_WINDOW = 2.0  # seconds - reject same order within this window


def check_duplicate_order(symbol: str, side: str, quantity: int) -> Tuple[bool, str]:
    """
    Check if this order was recently submitted (prevents double-clicks).

    Returns:
        Tuple of (is_duplicate, message)
    """
    global _recent_orders

    # Clean up old entries (older than 10 seconds)
    current_time = time.time()
    _recent_orders = {k: v for k, v in _recent_orders.items() if current_time - v < 10}

    # Create order key
    order_key = f"{symbol.upper()}_{side.upper()}_{quantity}"

    # Check if duplicate
    if order_key in _recent_orders:
        time_diff = current_time - _recent_orders[order_key]
        if time_diff < DUPLICATE_PREVENTION_WINDOW:
            return True, f"Duplicate order blocked. Same order submitted {time_diff:.1f}s ago. Wait {DUPLICATE_PREVENTION_WINDOW - time_diff:.1f}s."

    # Record this order
    _recent_orders[order_key] = current_time
    return False, ""


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
        # DUPLICATE ORDER CHECK: Prevent double-clicks
        is_duplicate, dup_message = check_duplicate_order(order.symbol, order.action, order.quantity)
        if is_duplicate:
            logger.warning(f"Duplicate order blocked: {order.symbol} {order.action} {order.quantity}")
            return {
                "success": False,
                "error": dup_message,
                "duplicate_blocked": True
            }

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
        # DUPLICATE ORDER CHECK: Prevent double-clicks
        is_duplicate, dup_message = check_duplicate_order(order.symbol, order.action, order.quantity)
        if is_duplicate:
            logger.warning(f"Duplicate bracket order blocked: {order.symbol} {order.action} {order.quantity}")
            return {
                "success": False,
                "error": dup_message,
                "duplicate_blocked": True
            }

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
        # DUPLICATE ORDER CHECK: Prevent double-clicks
        is_duplicate, dup_message = check_duplicate_order(order.symbol, order.action, order.quantity)
        if is_duplicate:
            logger.warning(f"Duplicate smart order blocked: {order.symbol} {order.action} {order.quantity}")
            return {
                "success": False,
                "error": dup_message,
                "duplicate_blocked": True
            }

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
#              ENHANCED BRAIN ENDPOINTS (December 2025)
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/ai/brain/volatility")
async def get_brain_volatility():
    """
    Get enhanced volatility analysis:
    - ATR (Average True Range) with baseline comparison
    - VIX proxy calculation
    - Bollinger Band width/position
    - Intraday range analysis
    - Spike detection with direction
    """
    try:
        from ai.background_brain import get_background_brain

        brain = get_background_brain()
        volatility = brain.get_volatility_state()

        return {
            "success": True,
            "volatility": volatility,
            "running": brain.is_running,
            "interpretation": {
                "level_meaning": {
                    "EXTREME": "ATR 3x+ normal - very high risk",
                    "HIGH": "ATR 2x normal - elevated risk",
                    "ELEVATED": "ATR 1.5x normal - above average",
                    "NORMAL": "Baseline volatility",
                    "LOW": "ATR below 0.5x - unusually calm"
                },
                "vix_regime_meaning": {
                    "PANIC": "VIX > 40 - market panic",
                    "FEAR": "VIX > 30 - high fear",
                    "ELEVATED": "VIX > 20 - elevated concern",
                    "NORMAL": "VIX 12-20 - normal trading",
                    "COMPLACENT": "VIX < 12 - complacency"
                }
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/brain/sectors")
async def get_brain_sectors():
    """
    Get enhanced sector rotation analysis:
    - 1-hour returns per sector
    - 5-day momentum scores
    - RSI per sector
    - Risk-on/Risk-off signal with strength
    - Market breadth
    - Hot/Cold sectors
    """
    try:
        from ai.background_brain import get_background_brain

        brain = get_background_brain()
        sectors = brain.get_sector_rotation()

        return {
            "success": True,
            "sector_rotation": sectors,
            "running": brain.is_running,
            "interpretation": {
                "RISK_ON": "Growth sectors leading - bullish environment",
                "RISK_OFF": "Defensive sectors leading - bearish/cautious",
                "NEUTRAL": "Mixed sector leadership"
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/brain/drift")
async def get_brain_drift():
    """
    Get prediction drift analysis with statistical tests:
    - Current vs baseline accuracy
    - P-value and Z-score for significance
    - 95% confidence interval
    - Trend direction and slope
    - Per-symbol accuracy breakdown
    - Time-of-day bias
    - Retraining recommendation
    """
    try:
        from ai.background_brain import get_background_brain

        brain = get_background_brain()
        drift = brain.get_prediction_drift()

        return {
            "success": True,
            "prediction_drift": drift,
            "running": brain.is_running,
            "interpretation": {
                "recommendations": {
                    "RETRAIN_URGENT": "Critical accuracy drop - immediate retraining needed",
                    "RETRAIN_SCHEDULED": "Significant drift - schedule retraining",
                    "WATCH": "Minor drift detected - monitoring",
                    "OK": "Performance within acceptable range"
                },
                "p_value": "P-value < 0.05 indicates statistically significant drift",
                "z_score": "Z-score < -2 indicates 2+ std devs below baseline"
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/brain/full-state")
async def get_brain_full_state():
    """
    Get complete brain state including all enhanced features:
    - Basic metrics (CPU, tasks, accuracy)
    - Market regime detection
    - Volatility analysis
    - Sector rotation
    - Prediction drift
    - Retraining status
    """
    try:
        from ai.background_brain import get_background_brain

        brain = get_background_brain()
        full_state = brain.get_full_brain_state()

        return {
            "success": True,
            "brain_state": full_state
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/brain/update-baseline")
async def update_brain_baseline(data: dict):
    """
    Update the baseline accuracy after model retraining.
    This sets the new benchmark for drift detection.

    Body:
        accuracy: float (0.0-1.0) - New baseline accuracy
    """
    try:
        from ai.background_brain import get_background_brain

        accuracy = data.get("accuracy", 0)
        if not 0 < accuracy <= 1:
            return {"success": False, "error": "Accuracy must be between 0 and 1"}

        brain = get_background_brain()
        brain.update_baseline_accuracy(accuracy)

        return {
            "success": True,
            "message": f"Baseline accuracy updated to {accuracy:.2%}",
            "new_baseline": accuracy
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/brain/retrain-status")
async def get_retrain_status():
    """
    Check if model retraining is needed and why.
    Returns all factors influencing the retraining decision.
    """
    try:
        from ai.background_brain import get_background_brain

        brain = get_background_brain()

        # Gather all retrain factors
        factors = []

        # Time factor
        if brain.last_retrain_time:
            from datetime import datetime
            import pytz
            hours_since = (datetime.now(pytz.timezone('US/Eastern')) - brain.last_retrain_time).total_seconds() / 3600
            factors.append({
                "factor": "time_since_retrain",
                "value": f"{hours_since:.1f} hours",
                "threshold": "24 hours",
                "triggered": hours_since > 24
            })
        else:
            factors.append({
                "factor": "never_trained",
                "value": "No previous training",
                "threshold": "N/A",
                "triggered": True
            })

        # Drift factor
        drift = brain.prediction_drift
        factors.append({
            "factor": "prediction_drift",
            "value": f"{drift.drift_percentage:.1f}% drop",
            "threshold": "10% (scheduled) / 20% (urgent)",
            "triggered": drift.recommendation in ["RETRAIN_SCHEDULED", "RETRAIN_URGENT"],
            "recommendation": drift.recommendation
        })

        # Statistical significance
        factors.append({
            "factor": "statistical_significance",
            "value": f"p={drift.p_value:.4f}, z={drift.z_score:.2f}",
            "threshold": "p < 0.05 and z < -2",
            "triggered": drift.p_value < 0.05 and drift.z_score < -2
        })

        # Volatility factor
        vol = brain.volatility_state
        factors.append({
            "factor": "volatility_regime",
            "value": f"{vol.vix_regime} (ATR ratio: {vol.atr_ratio}x)",
            "threshold": "PANIC or FEAR regime",
            "triggered": vol.vix_regime in ["PANIC", "FEAR"]
        })

        # Overall decision
        should_retrain = brain._should_retrain()
        urgent = brain._urgent_retrain_needed()

        return {
            "success": True,
            "retrain_needed": should_retrain,
            "urgent": urgent,
            "factors": factors,
            "retrain_triggered": brain.retrain_triggered,
            "last_retrain": brain.last_retrain_time.isoformat() if brain.last_retrain_time else None
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/brain/force-retrain")
async def force_brain_retrain(data: dict = None):
    """
    Force immediate model retraining.
    Use with caution - this may take a few minutes.

    Body (optional):
        symbols: list - Symbols to retrain on (default: recent + core symbols)
    """
    try:
        from ai.background_brain import get_background_brain

        brain = get_background_brain()

        if brain.retrain_triggered:
            return {
                "success": False,
                "error": "Retraining already in progress"
            }

        # Start retraining (runs async in background)
        brain.thread_pool.submit(brain._run_incremental_training)

        return {
            "success": True,
            "message": "Retraining started in background",
            "note": "Check /ai/brain/retrain-status for progress"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#                          AI SIGNALS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/ai/signals")
async def get_ai_signals():
    """
    Get live AI signals for watchlist symbols.
    Returns predictions for monitored symbols.
    """
    try:
        from ai.alpaca_ai_predictor import get_alpaca_predictor

        predictor = get_alpaca_predictor()

        # Get watchlist symbols
        watchlist_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD", "QQQ", "SPY"]

        # Try to get custom watchlist
        try:
            from watchlist_manager import get_watchlist_manager
            wm = get_watchlist_manager()
            primary = wm.get_primary_watchlist()
            if primary and primary.get("symbols"):
                watchlist_symbols = primary["symbols"][:15]
        except:
            pass

        signals = []
        from alpaca_market_data import get_alpaca_market_data
        market_data = get_alpaca_market_data()

        for symbol in watchlist_symbols:
            try:
                # Get prediction
                prediction = predictor.predict(symbol)
                if prediction:
                    # Get current price
                    price = None
                    try:
                        quote = market_data.get_quote(symbol)
                        price = quote.get("last_price") or quote.get("price")
                    except:
                        pass

                    signals.append({
                        "symbol": symbol,
                        "signal": prediction.get("action", "HOLD"),
                        "confidence": prediction.get("confidence", 0.5),
                        "prob_up": prediction.get("prob_up", 0.5),
                        "prob_down": prediction.get("prob_down", 0.5),
                        "price": price,
                        "timestamp": prediction.get("timestamp")
                    })
            except Exception as e:
                logger.debug(f"Signal error for {symbol}: {e}")
                continue

        # Sort by confidence
        signals.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "success": True,
            "signals": signals,
            "count": len(signals)
        }
    except Exception as e:
        return {"success": False, "error": str(e), "signals": []}


@router.get("/ai/model/status")
async def get_model_status():
    """
    Get AI model status including accuracy, training info, and loaded state.
    """
    try:
        from ai.alpaca_ai_predictor import get_alpaca_predictor
        import os
        from pathlib import Path

        predictor = get_alpaca_predictor()

        # Check if model is loaded
        model_loaded = predictor.model is not None

        # Get accuracy from predictor
        accuracy = predictor.accuracy if hasattr(predictor, 'accuracy') else 0.5

        # Get feature count
        features = predictor.features if hasattr(predictor, 'features') else 0

        # Check model file
        model_path = Path("store/models/lgb_predictor.txt")
        last_trained = None
        if model_path.exists():
            import datetime
            last_trained = datetime.datetime.fromtimestamp(
                model_path.stat().st_mtime
            ).isoformat()

        # Get symbols we've trained on
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD"]

        # Sample count estimate
        samples = features * 100 if features else 0

        return {
            "success": True,
            "loaded": model_loaded,
            "accuracy": accuracy,
            "features": features,
            "last_trained": last_trained,
            "samples": samples,
            "symbols": symbols,
            "model_path": str(model_path) if model_path.exists() else None
        }
    except Exception as e:
        return {"success": False, "error": str(e), "loaded": False}


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


# ═══════════════════════════════════════════════════════════════════════════════
#                          AI BACKTESTER ENDPOINTS (December 2025)
# ═══════════════════════════════════════════════════════════════════════════════

class BacktestRequest(BaseModel):
    """Request model for running a backtest"""
    symbols: list  # List of symbols to backtest
    start_date: str  # YYYY-MM-DD format
    end_date: str  # YYYY-MM-DD format
    initial_capital: float = 100000
    position_size_pct: float = 0.10
    max_positions: int = 5
    stop_loss_pct: float = 0.04
    take_profit_pct: float = 0.06
    trailing_stop_pct: float = 0.025
    run_monte_carlo: bool = False
    monte_carlo_simulations: int = 1000


@router.post("/ai/backtest/run")
async def run_backtest(request: BacktestRequest):
    """
    Run a comprehensive backtest with slippage modeling.

    Warrior Trading Rules applied:
    - Small cap momentum focus
    - No overnight holds (EOD liquidation at 3:45 PM ET)
    - Quick profit targets
    - Tight stop losses

    Features:
    - Realistic slippage (bid/ask spread, volume impact)
    - Commission modeling
    - Stop-loss, take-profit, trailing stops
    - Monte Carlo simulation (optional)
    """
    try:
        from ai.ai_backtester import AIBacktester, BacktestConfig, SlippageConfig

        # Create config
        config = BacktestConfig(
            initial_capital=request.initial_capital,
            position_size_pct=request.position_size_pct,
            max_positions=request.max_positions,
            stop_loss_pct=request.stop_loss_pct,
            take_profit_pct=request.take_profit_pct,
            trailing_stop_pct=request.trailing_stop_pct,
            slippage=SlippageConfig()
        )

        backtester = AIBacktester(config)

        # Run backtest
        result = backtester.run(
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date
        )

        # Run Monte Carlo if requested
        if request.run_monte_carlo:
            monte_carlo = backtester.run_monte_carlo(
                result,
                num_simulations=request.monte_carlo_simulations
            )
            result.monte_carlo = monte_carlo

        # Convert to dict for JSON response
        from dataclasses import asdict
        result_dict = asdict(result)

        return {
            "success": True,
            "result": result_dict,
            "summary": {
                "total_return_pct": result.total_return_pct,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown_pct": result.max_drawdown_pct,
                "profit_factor": result.profit_factor
            }
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@router.post("/ai/backtest/quick")
async def run_quick_backtest(data: dict = None):
    """
    Run a quick 30-day backtest with default settings.

    Body (optional):
        symbols: list - Symbols to backtest (default: ["AAPL", "MSFT", "TSLA"])
        days: int - Number of days to backtest (default: 30)
        capital: float - Starting capital (default: 100000)
    """
    try:
        from ai.ai_backtester import run_quick_backtest
        from dataclasses import asdict

        symbols = data.get("symbols", ["AAPL", "MSFT", "TSLA"]) if data else ["AAPL", "MSFT", "TSLA"]
        days = data.get("days", 30) if data else 30
        capital = data.get("capital", 100000) if data else 100000

        result = run_quick_backtest(
            symbols=symbols,
            days=days,
            initial_capital=capital
        )

        result_dict = asdict(result)

        return {
            "success": True,
            "result": result_dict,
            "summary": {
                "symbols": symbols,
                "days": days,
                "total_return_pct": result.total_return_pct,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown_pct": result.max_drawdown_pct
            }
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@router.post("/ai/backtest/monte-carlo")
async def run_monte_carlo_on_trades(data: dict):
    """
    Run Monte Carlo simulation on a set of trade P&Ls.

    Body:
        trade_pnls: list - List of trade P&L values
        initial_capital: float - Starting capital (default: 100000)
        num_simulations: int - Number of simulations (default: 1000)
        confidence_level: float - Confidence level (default: 0.95)
    """
    try:
        import numpy as np

        trade_pnls = data.get("trade_pnls", [])
        initial_capital = float(data.get("initial_capital", 100000))
        num_simulations = int(data.get("num_simulations", 1000))
        confidence_level = float(data.get("confidence_level", 0.95))

        if not trade_pnls:
            return {
                "success": False,
                "error": "trade_pnls list is required"
            }

        final_equities = []
        max_drawdowns = []

        for _ in range(num_simulations):
            # Random permutation of trades
            shuffled = np.random.permutation(trade_pnls)

            # Simulate equity curve
            equity = initial_capital
            peak = equity
            max_dd = 0

            for pnl in shuffled:
                equity += pnl
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

            final_equities.append(equity)
            max_drawdowns.append(max_dd)

        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)

        # Calculate percentiles
        ci_lower = (1 - confidence_level) / 2 * 100
        ci_upper = (1 + confidence_level) / 2 * 100

        return {
            "success": True,
            "num_simulations": num_simulations,
            "confidence_level": confidence_level,
            "num_trades": len(trade_pnls),
            "final_equity": {
                "mean": round(float(np.mean(final_equities)), 2),
                "median": round(float(np.median(final_equities)), 2),
                "std": round(float(np.std(final_equities)), 2),
                "min": round(float(np.min(final_equities)), 2),
                "max": round(float(np.max(final_equities)), 2),
                "ci_lower": round(float(np.percentile(final_equities, ci_lower)), 2),
                "ci_upper": round(float(np.percentile(final_equities, ci_upper)), 2)
            },
            "max_drawdown": {
                "mean": round(float(np.mean(max_drawdowns)) * 100, 2),
                "median": round(float(np.median(max_drawdowns)) * 100, 2),
                "worst_case_95": round(float(np.percentile(max_drawdowns, 95)) * 100, 2)
            },
            "probability_of_profit": round(float((final_equities > initial_capital).mean()) * 100, 2),
            "probability_of_ruin": round(float((final_equities < initial_capital * 0.5).mean()) * 100, 2)
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@router.get("/ai/backtest/config")
async def get_backtest_config():
    """Get default backtest configuration and slippage settings"""
    try:
        from ai.ai_backtester import BacktestConfig, SlippageConfig

        config = BacktestConfig()
        slippage = SlippageConfig()

        return {
            "success": True,
            "config": {
                "initial_capital": config.initial_capital,
                "position_size_pct": config.position_size_pct,
                "max_positions": config.max_positions,
                "max_position_value": config.max_position_value,
                "confidence_threshold": config.confidence_threshold,
                "stop_loss_pct": config.stop_loss_pct,
                "take_profit_pct": config.take_profit_pct,
                "trailing_stop_pct": config.trailing_stop_pct,
                "trailing_activation_pct": config.trailing_activation_pct,
                "no_entry_after": config.no_entry_after,
                "eod_liquidation_time": config.eod_liquidation_time,
                "friday_early_close": config.friday_early_close
            },
            "slippage": {
                "base_spread_pct": slippage.base_spread_pct,
                "volume_impact_factor": slippage.volume_impact_factor,
                "volatility_multiplier": slippage.volatility_multiplier,
                "small_cap_threshold": slippage.small_cap_threshold,
                "small_cap_penalty": slippage.small_cap_penalty,
                "commission_per_share": slippage.commission_per_share
            },
            "explanation": {
                "slippage": "Slippage = Base Spread + (Volume Impact * Trade Size) + (Volatility * Multiplier) + Small Cap Penalty",
                "warrior_rules": [
                    "EOD liquidation at 3:45 PM ET (3:30 PM Fridays)",
                    "No new entries after 3:30 PM ET",
                    "Small cap momentum focus (quick entries/exits)",
                    "Tight stops (4%) and quick profits (6%)"
                ]
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/backtest/slippage-estimate")
async def estimate_slippage(
    price: float = 5.0,
    quantity: int = 100,
    avg_volume: float = 100000,
    volatility: float = 1.0,
    side: str = "BUY"
):
    """
    Estimate slippage for a hypothetical trade.

    Args:
        price: Stock price
        quantity: Number of shares
        avg_volume: Average daily volume
        volatility: Volatility multiplier (1.0 = normal)
        side: BUY or SELL
    """
    try:
        from ai.ai_backtester import SlippageConfig

        slippage_config = SlippageConfig()
        slippage_pct = slippage_config.calculate_slippage(
            price=price,
            quantity=quantity,
            avg_volume=avg_volume,
            volatility=volatility,
            side=side
        )

        # Calculate execution price
        if side.upper() == "BUY":
            exec_price = price * (1 + slippage_pct / 100)
        else:
            exec_price = price * (1 - slippage_pct / 100)

        slippage_cost = abs(exec_price - price) * quantity

        return {
            "success": True,
            "input": {
                "price": price,
                "quantity": quantity,
                "avg_volume": avg_volume,
                "volatility": volatility,
                "side": side
            },
            "slippage_pct": round(slippage_pct, 4),
            "execution_price": round(exec_price, 4),
            "slippage_cost": round(slippage_cost, 2),
            "breakdown": {
                "base_spread": f"{slippage_config.base_spread_pct}%",
                "is_small_cap": price < slippage_config.small_cap_threshold,
                "small_cap_penalty": f"{slippage_config.small_cap_penalty}%" if price < slippage_config.small_cap_threshold else "N/A"
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#                    PORTFOLIO GUARD ENDPOINTS (December 2025)
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/ai/portfolio-guard/status")
async def get_portfolio_guard_status():
    """
    Get complete portfolio risk assessment.

    Returns:
    - VAR (Value at Risk) at 95% and 99% confidence
    - Drawdown state (current, max, days in DD)
    - Position-level risks
    - Concentration metrics
    - Sector exposure
    - Risk flags and overall risk level
    """
    try:
        from ai.portfolio_guard import get_portfolio_guard

        guard = get_portfolio_guard()
        connector = get_alpaca_connector()

        # Get account and positions
        account = connector.get_account()
        positions = connector.get_positions()

        total_equity = float(account.get('equity', 0))
        cash = float(account.get('cash', 0))

        # Format positions for guard
        formatted_positions = []
        for pos in positions:
            formatted_positions.append({
                'symbol': pos.get('symbol', ''),
                'qty': int(pos.get('qty', 0)),
                'market_value': float(pos.get('market_value', 0)),
                'current_price': float(pos.get('current_price', 0))
            })

        # Get risk assessment
        risk_state = guard.assess_portfolio_risk(
            positions=formatted_positions,
            total_equity=total_equity,
            cash=cash
        )

        return {
            "success": True,
            "risk_state": risk_state.to_dict(),
            "risk_level": risk_state.risk_level,
            "risk_flags": risk_state.risk_flags,
            "summary": {
                "total_equity": total_equity,
                "num_positions": len(positions),
                "var_95_pct": risk_state.var.var_pct_95 if risk_state.var else 0,
                "var_95_dollars": risk_state.var.var_1day_95 if risk_state.var else 0,
                "drawdown_pct": risk_state.drawdown.drawdown_pct,
                "max_drawdown_pct": risk_state.drawdown.max_drawdown_pct
            }
        }
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/ai/portfolio-guard/var")
async def get_portfolio_var(method: str = "historical"):
    """
    Calculate portfolio Value at Risk.

    Args:
        method: "historical", "parametric", or "monte_carlo"

    Returns:
    - 1-day VAR at 95% and 99% confidence
    - 5-day VAR at 95%
    - Expected Shortfall (CVaR)
    - VAR as % of portfolio
    """
    try:
        from ai.portfolio_guard import get_portfolio_guard

        guard = get_portfolio_guard()
        connector = get_alpaca_connector()

        account = connector.get_account()
        positions = connector.get_positions()

        total_equity = float(account.get('equity', 0))

        formatted_positions = [{
            'symbol': pos.get('symbol', ''),
            'market_value': float(pos.get('market_value', 0))
        } for pos in positions]

        var_result = guard.calculate_var(
            positions=formatted_positions,
            total_equity=total_equity,
            method=method
        )

        return {
            "success": True,
            "var": var_result.to_dict(),
            "interpretation": {
                "var_1day_95": f"95% chance daily loss won't exceed ${var_result.var_1day_95:,.2f}",
                "var_1day_99": f"99% chance daily loss won't exceed ${var_result.var_1day_99:,.2f}",
                "expected_shortfall": f"If VAR is breached, expected loss is ${var_result.expected_shortfall:,.2f}"
            }
        }
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/ai/portfolio-guard/drawdown")
async def get_portfolio_drawdown():
    """
    Get current drawdown state.

    Returns:
    - Current vs peak equity
    - Drawdown in dollars and %
    - Max drawdown (all-time)
    - Days in drawdown
    - Recovery needed %
    """
    try:
        from ai.portfolio_guard import get_portfolio_guard

        guard = get_portfolio_guard()
        connector = get_alpaca_connector()

        account = connector.get_account()
        total_equity = float(account.get('equity', 0))

        drawdown = guard.calculate_drawdown(total_equity)

        return {
            "success": True,
            "drawdown": drawdown.to_dict(),
            "interpretation": {
                "status": "At all-time high!" if drawdown.is_at_peak else f"Down {drawdown.drawdown_pct:.1f}% from peak",
                "recovery": f"Need {drawdown.recovery_needed_pct:.1f}% gain to recover" if not drawdown.is_at_peak else "No recovery needed"
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/portfolio-guard/concentration")
async def get_portfolio_concentration():
    """
    Get portfolio concentration analysis.

    Returns:
    - Largest position %
    - Top 3 concentration
    - Sector exposure breakdown
    - Small cap exposure
    """
    try:
        from ai.portfolio_guard import get_portfolio_guard
        from collections import defaultdict

        guard = get_portfolio_guard()
        connector = get_alpaca_connector()

        account = connector.get_account()
        positions = connector.get_positions()

        total_equity = float(account.get('equity', 0))

        # Calculate weights
        position_weights = []
        sector_exposure = defaultdict(float)
        small_cap_exposure = 0

        for pos in positions:
            market_value = float(pos.get('market_value', 0))
            weight = (market_value / total_equity * 100) if total_equity > 0 else 0
            current_price = float(pos.get('current_price', 0))

            position_weights.append({
                'symbol': pos.get('symbol', ''),
                'weight_pct': round(weight, 2),
                'market_value': round(market_value, 2)
            })

            # Sector
            sector = guard._get_sector(pos.get('symbol', ''))
            sector_exposure[sector] += weight

            # Small cap
            if current_price > 0 and current_price < 10:
                small_cap_exposure += weight

        # Sort by weight
        position_weights.sort(key=lambda x: x['weight_pct'], reverse=True)

        largest = position_weights[0]['weight_pct'] if position_weights else 0
        top_3 = sum(p['weight_pct'] for p in position_weights[:3])

        limits = guard.config

        return {
            "success": True,
            "concentration": {
                "largest_position_pct": round(largest, 2),
                "top_3_concentration_pct": round(top_3, 2),
                "small_cap_exposure_pct": round(small_cap_exposure, 2),
                "num_positions": len(positions)
            },
            "limits": {
                "max_single_position": limits.max_single_position_pct,
                "max_top3_concentration": limits.max_top3_concentration_pct,
                "max_small_cap": limits.max_small_cap_pct
            },
            "sector_exposure": dict(sector_exposure),
            "position_weights": position_weights[:10]  # Top 10
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/portfolio-guard/check-position")
async def check_position_allowed(data: dict):
    """
    Check if a new position would violate risk limits.

    Body:
        symbol: str - Symbol to add
        shares: int - Number of shares
        price: float - Current price

    Returns:
    - allowed: bool
    - reason: str
    - warnings: list
    """
    try:
        from ai.portfolio_guard import get_portfolio_guard

        guard = get_portfolio_guard()
        connector = get_alpaca_connector()

        account = connector.get_account()
        positions = connector.get_positions()

        total_equity = float(account.get('equity', 0))

        # Format current positions
        current_positions = [{
            'symbol': pos.get('symbol', ''),
            'market_value': float(pos.get('market_value', 0)),
            'current_price': float(pos.get('current_price', 0))
        } for pos in positions]

        # Format new position
        symbol = data.get('symbol', '')
        shares = int(data.get('shares', 0))
        price = float(data.get('price', 0))
        market_value = shares * price

        new_position = {
            'symbol': symbol,
            'shares': shares,
            'market_value': market_value,
            'current_price': price
        }

        allowed, reason, warnings = guard.can_add_position(
            current_positions=current_positions,
            new_position=new_position,
            total_equity=total_equity
        )

        return {
            "success": True,
            "allowed": allowed,
            "reason": reason,
            "warnings": warnings,
            "position_weight_pct": round(market_value / total_equity * 100, 2) if total_equity > 0 else 0
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/portfolio-guard/limits")
async def get_risk_limits():
    """Get current risk limit configuration"""
    try:
        from ai.portfolio_guard import get_portfolio_guard

        guard = get_portfolio_guard()

        return {
            "success": True,
            "limits": guard.get_risk_limits()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/portfolio-guard/limits")
async def update_risk_limits(data: dict):
    """
    Update risk limit configuration.

    Body: Key-value pairs of limits to update
        Example: {"max_var_pct": 4.0, "max_drawdown_pct": 12.0}
    """
    try:
        from ai.portfolio_guard import get_portfolio_guard

        guard = get_portfolio_guard()
        guard.update_limits(**data)

        return {
            "success": True,
            "message": "Limits updated",
            "new_limits": guard.get_risk_limits()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/portfolio-guard/position-risks")
async def get_position_risks():
    """
    Get risk metrics for each position.

    Returns per position:
    - Weight %
    - Daily VAR
    - Beta
    - Volatility
    - Correlation to SPY
    - Sector
    - Contribution to portfolio VAR
    """
    try:
        from ai.portfolio_guard import get_portfolio_guard

        guard = get_portfolio_guard()
        connector = get_alpaca_connector()

        account = connector.get_account()
        positions = connector.get_positions()

        total_equity = float(account.get('equity', 0))

        position_risks = []
        for pos in positions:
            formatted = {
                'symbol': pos.get('symbol', ''),
                'qty': int(pos.get('qty', 0)),
                'market_value': float(pos.get('market_value', 0)),
                'current_price': float(pos.get('current_price', 0))
            }
            risk = guard.analyze_position_risk(formatted, total_equity)
            position_risks.append(risk.to_dict())

        # Sort by VAR contribution
        position_risks.sort(key=lambda x: x.get('daily_var_95', 0), reverse=True)

        return {
            "success": True,
            "position_risks": position_risks,
            "total_equity": total_equity,
            "num_positions": len(positions)
        }
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# ============================================================================
# ENSEMBLE PREDICTOR ENDPOINTS
# ============================================================================

@router.get("/ai/ensemble/predict/{symbol}")
async def ensemble_predict(symbol: str):
    """
    Generate ensemble prediction for a symbol.

    Combines:
    - LightGBM ML model
    - Technical analysis heuristics
    - Momentum scoring

    Weights are adjusted based on market regime.
    """
    try:
        from ai.ensemble_predictor import get_ensemble_predictor
        import yfinance as yf
        import pandas as pd

        predictor = get_ensemble_predictor()

        # Get price data
        df = yf.download(symbol.upper(), period="6mo", progress=False)
        if df.empty:
            return {"success": False, "error": f"No data for {symbol}"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Generate prediction
        result = predictor.predict(symbol.upper(), df)

        return {
            "success": True,
            "symbol": symbol.upper(),
            "prediction": "BULLISH" if result.prediction == 1 else "BEARISH",
            "confidence": round(result.confidence, 4),
            "components": {
                "lgb": {
                    "score": round(result.lgb_score, 4),
                    "weight": round(result.lgb_weight, 4)
                },
                "heuristic": {
                    "score": round(result.heuristic_score, 4),
                    "weight": round(result.heuristic_weight, 4)
                },
                "momentum": {
                    "score": round(result.momentum_score, 4),
                    "weight": round(result.momentum_weight, 4)
                }
            },
            "regime": {
                "type": result.market_regime,
                "confidence": round(result.regime_confidence, 4)
            },
            "signals": result.signals,
            "timestamp": result.timestamp
        }
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


@router.post("/ai/ensemble/predict-batch")
async def ensemble_predict_batch(data: dict):
    """
    Generate ensemble predictions for multiple symbols.

    Request body: {"symbols": ["AAPL", "TSLA", "NVDA"]}
    """
    try:
        from ai.ensemble_predictor import get_ensemble_predictor
        import yfinance as yf
        import pandas as pd

        symbols = data.get("symbols", [])
        if not symbols:
            return {"success": False, "error": "No symbols provided"}

        predictor = get_ensemble_predictor()
        results = []

        for symbol in symbols[:10]:  # Limit to 10
            try:
                df = yf.download(symbol.upper(), period="6mo", progress=False)
                if df.empty:
                    results.append({"symbol": symbol, "error": "No data"})
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)

                result = predictor.predict(symbol.upper(), df)
                results.append({
                    "symbol": symbol.upper(),
                    "prediction": "BULLISH" if result.prediction == 1 else "BEARISH",
                    "confidence": round(result.confidence, 4),
                    "lgb_score": round(result.lgb_score, 4),
                    "heuristic_score": round(result.heuristic_score, 4),
                    "momentum_score": round(result.momentum_score, 4),
                    "regime": result.market_regime
                })
            except Exception as e:
                results.append({"symbol": symbol, "error": str(e)})

        return {
            "success": True,
            "predictions": results,
            "count": len(results)
        }
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/ai/ensemble/performance")
async def ensemble_performance():
    """
    Get recent ensemble prediction performance statistics.
    """
    try:
        from ai.ensemble_predictor import get_ensemble_predictor

        predictor = get_ensemble_predictor()
        stats = predictor.get_performance_stats()

        return {
            "success": True,
            "performance": stats,
            "interpretation": {
                "accuracy": f"{stats['accuracy']:.1%}" if stats['total'] > 0 else "N/A",
                "sample_size": stats['total'],
                "recent_trend": "improving" if stats.get('recent_10', 0) > stats.get('accuracy', 0) else "stable"
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/ensemble/record-outcome")
async def ensemble_record_outcome(data: dict):
    """
    Record prediction outcome for adaptive learning.

    Request body: {"symbol": "AAPL", "prediction": 1, "actual": 1}
    """
    try:
        from ai.ensemble_predictor import get_ensemble_predictor

        symbol = data.get("symbol", "")
        prediction = data.get("prediction", 0)
        actual = data.get("actual", 0)

        if not symbol:
            return {"success": False, "error": "Symbol required"}

        predictor = get_ensemble_predictor()
        predictor.record_outcome(symbol, prediction, actual)

        return {
            "success": True,
            "message": f"Recorded outcome for {symbol}",
            "correct": prediction == actual
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#                    RL AGENT ENDPOINTS (December 2025)
# ═══════════════════════════════════════════════════════════════════════════════

class RLTrainRequest(BaseModel):
    """Request to train RL agent"""
    symbols: list = ["AAPL", "MSFT", "NVDA"]
    episodes: int = 10
    period: str = "1y"


class RLPredictRequest(BaseModel):
    """Request for RL prediction"""
    symbol: str


@router.get("/ai/rl/status")
async def get_rl_status():
    """
    Get RL agent status and training info.

    Returns current model state, training metrics, and whether agent is loaded.
    """
    try:
        from ai.rl_trading_agent import get_rl_agent, HAS_TORCH

        agent = get_rl_agent()

        return {
            "success": True,
            "status": {
                "pytorch_available": HAS_TORCH,
                "model_loaded": agent.policy_net is not None if HAS_TORCH else len(agent.q_table) > 0,
                "episode": agent.episode,
                "training_step": agent.training_step,
                "epsilon": agent.epsilon,
                "device": str(agent.device) if HAS_TORCH else "cpu",
                "replay_buffer_size": len(agent.memory),
                "state_size": agent.state_size,
                "action_size": agent.action_size,
                "learning_rate": agent.learning_rate,
                "gamma": agent.gamma
            }
        }
    except Exception as e:
        logger.error(f"Error getting RL status: {e}")
        return {"success": False, "error": str(e)}


@router.get("/ai/rl/predict/{symbol}")
async def get_rl_prediction(symbol: str):
    """
    Get RL agent's action recommendation for a symbol.

    Returns action probabilities and recommended action.
    """
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        import ta
        from ai.rl_trading_agent import get_rl_agent, TradingState

        # Get market data
        df = yf.download(symbol.upper(), period="6mo", progress=False)
        if df.empty:
            return {"success": False, "error": f"No data for {symbol}"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Calculate state features
        close = df['Close']
        returns = close.pct_change().dropna()

        rsi = ta.momentum.rsi(close, window=14).iloc[-1] if len(close) > 14 else 50

        macd_obj = ta.trend.MACD(close)
        macd_line = macd_obj.macd().iloc[-1] if len(close) > 26 else 0

        bb = ta.volatility.BollingerBands(close)
        bb_high = bb.bollinger_hband().iloc[-1]
        bb_low = bb.bollinger_lband().iloc[-1]
        bb_position = (close.iloc[-1] - bb_low) / (bb_high - bb_low) if bb_high != bb_low else 0.5

        vol_ratio = df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1] if len(df) > 20 else 1.0

        sma_20 = close.rolling(20).mean().iloc[-1] if len(close) > 20 else close.iloc[-1]
        trend = (close.iloc[-1] - sma_20) / sma_20 if sma_20 > 0 else 0
        volatility = returns.std() * np.sqrt(252) if len(returns) > 5 else 0.25

        # Create state
        state = TradingState(
            price_change_1d=returns.iloc[-1] if len(returns) > 0 else 0,
            price_change_5d=(close.iloc[-1] / close.iloc[-5] - 1) if len(close) > 5 else 0,
            price_change_20d=(close.iloc[-1] / close.iloc[-20] - 1) if len(close) > 20 else 0,
            rsi_normalized=(rsi - 50) / 50,
            macd_normalized=np.tanh(macd_line),
            bb_position=bb_position,
            volume_ratio=min(vol_ratio, 5) / 5,
            trend_strength=abs(trend),
            trend_direction=np.sign(trend),
            volatility=min(volatility, 1),
            ensemble_score=0.5,
            lgb_score=0.5,
            momentum_score=0.5,
            has_position=0,
            position_pnl=0,
            holding_time=0,
            regime_trending=1.0 if abs(trend) > 0.02 else 0.0,
            regime_volatile=1.0 if volatility > 0.4 else 0.0
        )

        # Get prediction
        agent = get_rl_agent()
        result = agent.get_action_probabilities(state)

        return {
            "success": True,
            "symbol": symbol.upper(),
            "action": result['recommended'],
            "probabilities": {
                "HOLD": result['HOLD'],
                "BUY": result['BUY'],
                "SELL": result['SELL']
            },
            "q_values": result['q_values'],
            "current_price": float(close.iloc[-1]),
            "state_features": {
                "rsi": rsi,
                "macd": macd_line,
                "bb_position": bb_position,
                "trend": trend,
                "volatility": volatility
            }
        }
    except Exception as e:
        logger.error(f"Error getting RL prediction: {e}")
        return {"success": False, "error": str(e)}


@router.post("/ai/rl/train")
async def train_rl_agent(request: RLTrainRequest):
    """
    Train the RL agent on historical data.

    This runs training episodes using historical price data.
    Warning: Training can take several minutes depending on episodes count.
    """
    try:
        import yfinance as yf
        import pandas as pd
        from ai.rl_trading_agent import get_rl_agent, get_rl_environment

        agent = get_rl_agent()
        env = get_rl_environment()

        all_metrics = []

        for symbol in request.symbols:
            logger.info(f"Training RL agent on {symbol}...")

            # Get data
            df = yf.download(symbol, period=request.period, progress=False)
            if df.empty:
                logger.warning(f"No data for {symbol}, skipping")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            # Train episodes
            for episode in range(request.episodes):
                metrics = agent.train_episode(env, df)
                all_metrics.append({
                    "symbol": symbol,
                    "episode": metrics.episode,
                    "total_reward": metrics.total_reward,
                    "win_rate": metrics.win_rate,
                    "trades": metrics.trades,
                    "sharpe_ratio": metrics.sharpe_ratio
                })

        # Save model
        agent.save_model()

        # Calculate summary
        if all_metrics:
            avg_reward = sum(m['total_reward'] for m in all_metrics) / len(all_metrics)
            avg_win_rate = sum(m['win_rate'] for m in all_metrics) / len(all_metrics)
            total_trades = sum(m['trades'] for m in all_metrics)
        else:
            avg_reward = 0
            avg_win_rate = 0
            total_trades = 0

        return {
            "success": True,
            "training_summary": {
                "symbols_trained": request.symbols,
                "episodes_per_symbol": request.episodes,
                "total_episodes": len(all_metrics),
                "avg_reward": avg_reward,
                "avg_win_rate": avg_win_rate,
                "total_trades": total_trades,
                "epsilon_final": agent.epsilon
            },
            "episode_metrics": all_metrics[-10:] if len(all_metrics) > 10 else all_metrics
        }
    except Exception as e:
        logger.error(f"Error training RL agent: {e}")
        return {"success": False, "error": str(e)}


@router.get("/ai/rl/metrics")
async def get_rl_metrics():
    """
    Get RL agent training metrics and performance history.
    """
    try:
        from ai.rl_trading_agent import get_rl_agent
        from pathlib import Path
        import json

        agent = get_rl_agent()

        # Try to load training history
        metrics_path = Path("store/models/rl_training_history.json")
        history = []
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                history = json.load(f)

        return {
            "success": True,
            "current_state": {
                "episode": agent.episode,
                "training_step": agent.training_step,
                "epsilon": agent.epsilon,
                "replay_buffer_size": len(agent.memory)
            },
            "training_history": history[-50:] if history else []
        }
    except Exception as e:
        logger.error(f"Error getting RL metrics: {e}")
        return {"success": False, "error": str(e)}


@router.post("/ai/rl/reset")
async def reset_rl_agent():
    """
    Reset the RL agent to untrained state.
    Warning: This will clear all learned weights!
    """
    try:
        from ai.rl_trading_agent import RLTradingAgent
        import ai.rl_trading_agent as rl_module

        # Create fresh agent
        rl_module._rl_agent = RLTradingAgent()

        return {
            "success": True,
            "message": "RL agent reset to initial state",
            "epsilon": rl_module._rl_agent.epsilon
        }
    except Exception as e:
        logger.error(f"Error resetting RL agent: {e}")
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#                    PARALLEL ENSEMBLE ENDPOINTS (December 2025)
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/ai/parallel/predict/{symbol}")
async def get_parallel_prediction(symbol: str):
    """
    Get multi-chain parallel ensemble prediction.

    Runs 5 independent prediction chains in parallel:
    1. Technical Momentum (RSI, MACD, Stochastic)
    2. Statistical ML (LightGBM)
    3. RL Timing (DQN agent)
    4. Mean Reversion (Bollinger, Z-score)
    5. Trend Following (MA crossovers, ADX)

    Returns aggregated prediction with multiple averaging methods.
    """
    try:
        import yfinance as yf
        import pandas as pd
        from ai.parallel_ensemble import get_parallel_ensemble

        # Get market data
        df = yf.download(symbol.upper(), period="6mo", progress=False)
        if df.empty:
            return {"success": False, "error": f"No data for {symbol}"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Get prediction
        ensemble = get_parallel_ensemble()
        result = ensemble.predict(symbol.upper(), df)

        return {
            "success": True,
            "symbol": result.symbol,
            "prediction": {
                "direction": result.direction,
                "confidence": float(result.confidence),
                "signal_strength": float(result.signal_strength)
            },
            "consensus": {
                "chains_total": int(result.num_chains),
                "chains_agree": int(result.chains_agree),
                "disagreement_level": float(result.disagreement_level),
                "majority_vote": result.majority_vote
            },
            "aggregation_methods": {
                "simple_average": float(result.simple_avg),
                "confidence_weighted": float(result.confidence_weighted_avg),
                "accuracy_weighted": float(result.accuracy_weighted_avg)
            },
            "risk_flags": {
                "high_disagreement": bool(result.high_disagreement),
                "low_confidence": bool(result.low_confidence),
                "conflicting_signals": bool(result.conflicting_signals)
            },
            "chain_details": [
                {
                    "chain": cp.chain_name,
                    "direction": cp.direction,
                    "confidence": float(cp.confidence),
                    "signal_strength": float(cp.signal_strength),
                    "reasoning": cp.reasoning
                }
                for cp in result.chain_predictions
            ],
            "timestamp": result.timestamp
        }
    except Exception as e:
        logger.error(f"Error in parallel prediction: {e}")
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/ai/parallel/chain-performance")
async def get_chain_performance():
    """
    Get performance statistics for each prediction chain.

    Shows accuracy history and identifies best-performing chains.
    """
    try:
        from ai.parallel_ensemble import get_parallel_ensemble

        ensemble = get_parallel_ensemble()
        stats = ensemble.get_chain_performance()
        best_chains = ensemble.get_best_chains(3)

        return {
            "success": True,
            "chain_stats": stats,
            "best_chains": best_chains,
            "total_predictions": len(ensemble.prediction_history)
        }
    except Exception as e:
        logger.error(f"Error getting chain performance: {e}")
        return {"success": False, "error": str(e)}


@router.post("/ai/parallel/record-outcome")
async def record_parallel_outcome(data: dict):
    """
    Record trade outcome to update chain accuracy weights.

    Body: {
        "symbol": "AAPL",
        "predicted_direction": "BUY",
        "was_profitable": true
    }
    """
    try:
        from ai.parallel_ensemble import get_parallel_ensemble

        symbol = data.get("symbol", "")
        predicted = data.get("predicted_direction", "HOLD")
        profitable = data.get("was_profitable", False)

        if not symbol:
            return {"success": False, "error": "Symbol required"}

        ensemble = get_parallel_ensemble()

        # Note: In real usage, you'd store chain_predictions from the original prediction
        # For now, just log the outcome
        ensemble.prediction_history.append({
            'symbol': symbol,
            'predicted': predicted,
            'profitable': profitable,
            'timestamp': datetime.now().isoformat()
        })

        return {
            "success": True,
            "message": f"Recorded outcome for {symbol}: {predicted} was {'profitable' if profitable else 'not profitable'}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/parallel/batch-predict")
async def parallel_batch_predict(data: dict = None):
    """
    Run parallel ensemble on multiple symbols.

    Body: {"symbols": ["AAPL", "MSFT", "NVDA"]}
    """
    try:
        import yfinance as yf
        import pandas as pd
        from ai.parallel_ensemble import get_parallel_ensemble

        symbols = data.get("symbols", ["AAPL", "MSFT", "NVDA"]) if data else ["AAPL", "MSFT", "NVDA"]

        ensemble = get_parallel_ensemble()
        results = []

        for symbol in symbols[:10]:  # Limit to 10
            try:
                df = yf.download(symbol, period="6mo", progress=False)
                if df.empty:
                    results.append({"symbol": symbol, "error": "No data"})
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)

                result = ensemble.predict(symbol, df)

                results.append({
                    "symbol": result.symbol,
                    "direction": result.direction,
                    "confidence": result.confidence,
                    "signal_strength": result.signal_strength,
                    "chains_agree": f"{result.chains_agree}/{result.num_chains}",
                    "majority_vote": result.majority_vote,
                    "flags": {
                        "disagreement": result.high_disagreement,
                        "low_conf": result.low_confidence,
                        "conflict": result.conflicting_signals
                    }
                })
            except Exception as e:
                results.append({"symbol": symbol, "error": str(e)})

        # Sort by confidence
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        return {
            "success": True,
            "predictions": results,
            "summary": {
                "total": len(results),
                "buy_signals": sum(1 for r in results if r.get('direction') == 'BUY'),
                "sell_signals": sum(1 for r in results if r.get('direction') == 'SELL'),
                "hold_signals": sum(1 for r in results if r.get('direction') == 'HOLD')
            }
        }
    except Exception as e:
        logger.error(f"Error in batch parallel predict: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# ENSEMBLE BACKTEST COMPARISON ENDPOINTS
# ============================================================================

@router.post("/ai/backtest/compare")
async def compare_ensemble_backtest(data: dict = None):
    """
    Run backtest comparison between single and parallel ensemble.

    Body: {
        "symbols": ["AAPL", "MSFT", "NVDA"],  # Optional, defaults to tech stocks
        "period": "6mo"  # Optional, default 6 months
    }

    Returns comprehensive comparison metrics showing which approach performs better.
    """
    try:
        from ai.ensemble_backtester import EnsembleBacktester

        symbols = data.get("symbols", ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]) if data else ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]
        period = data.get("period", "6mo") if data else "6mo"

        backtester = EnsembleBacktester(
            confidence_threshold=0.25,
            holding_period=3
        )

        result = backtester.backtest_portfolio(symbols, period)

        return result
    except Exception as e:
        logger.error(f"Error running backtest comparison: {e}")
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/ai/backtest/compare/{symbol}")
async def compare_single_symbol_backtest(symbol: str, period: str = "6mo"):
    """
    Run backtest comparison on a single symbol.

    Returns detailed comparison for one stock.
    """
    try:
        from ai.ensemble_backtester import EnsembleBacktester

        backtester = EnsembleBacktester(
            confidence_threshold=0.25,
            holding_period=3
        )

        result = backtester.backtest_symbol(symbol.upper(), period)

        return {
            "success": True,
            "comparison": result.to_dict()
        }
    except Exception as e:
        logger.error(f"Error running single symbol backtest: {e}")
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/ai/backtest/quick")
async def quick_backtest_comparison():
    """
    Quick comparison using default symbols and 3-month period.
    Good for fast testing.
    """
    try:
        from ai.ensemble_backtester import run_quick_comparison

        result = run_quick_comparison(
            symbols=["AAPL", "NVDA", "MSFT"],
            period="3mo"
        )

        return result
    except Exception as e:
        logger.error(f"Error running quick backtest: {e}")
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# ============================================================================
# HYBRID ENSEMBLE ENDPOINTS
# ============================================================================

@router.get("/ai/hybrid/predict/{symbol}")
async def hybrid_predict(symbol: str):
    """
    Get hybrid ensemble prediction combining single and parallel approaches.

    Returns direction, confidence, position size multiplier, and agreement analysis.
    """
    try:
        import yfinance as yf
        import pandas as pd
        from ai.hybrid_ensemble import get_hybrid_ensemble

        # Get data
        df = yf.download(symbol.upper(), period="6mo", progress=False)
        if df.empty:
            return {"success": False, "error": f"No data for {symbol}"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        hybrid = get_hybrid_ensemble()
        result = hybrid.predict(symbol.upper(), df)

        return {
            "success": True,
            "prediction": {
                "symbol": result.symbol,
                "direction": result.direction,
                "confidence": float(result.confidence),
                "signal_strength": float(result.signal_strength),
                "recommended_action": result.recommended_action,
                "position_size_multiplier": float(result.position_size_multiplier)
            },
            "agreement": {
                "ensembles_agree": bool(result.ensembles_agree),
                "agreement_level": result.agreement_level,
                "high_conviction": bool(result.high_conviction),
                "conflicting_signals": bool(result.conflicting_signals)
            },
            "components": {
                "single": {
                    "direction": result.single_direction,
                    "confidence": float(result.single_confidence)
                },
                "parallel": {
                    "direction": result.parallel_direction,
                    "confidence": float(result.parallel_confidence),
                    "chains_agree": f"{result.parallel_chains_agree}/{result.parallel_num_chains}"
                }
            },
            "context": {
                "market_regime": result.market_regime,
                "volatility_state": result.volatility_state
            },
            "timestamp": result.timestamp
        }
    except Exception as e:
        logger.error(f"Error in hybrid prediction: {e}")
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


@router.post("/ai/hybrid/batch-predict")
async def hybrid_batch_predict(data: dict = None):
    """
    Run hybrid ensemble on multiple symbols.

    Body: {"symbols": ["AAPL", "MSFT", "NVDA"]}
    """
    try:
        import yfinance as yf
        import pandas as pd
        from ai.hybrid_ensemble import get_hybrid_ensemble

        symbols = data.get("symbols", ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]) if data else ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]

        hybrid = get_hybrid_ensemble()
        results = []

        for symbol in symbols[:10]:  # Limit to 10
            try:
                df = yf.download(symbol, period="6mo", progress=False)
                if df.empty:
                    results.append({"symbol": symbol, "error": "No data"})
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)

                result = hybrid.predict(symbol, df)

                results.append({
                    "symbol": result.symbol,
                    "direction": result.direction,
                    "action": result.recommended_action,
                    "confidence": float(result.confidence),
                    "position_size": float(result.position_size_multiplier),
                    "agreement": result.agreement_level,
                    "single": result.single_direction,
                    "parallel": result.parallel_direction,
                    "high_conviction": bool(result.high_conviction)
                })
            except Exception as e:
                results.append({"symbol": symbol, "error": str(e)})

        # Sort by confidence
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        # Summarize
        actionable = [r for r in results if r.get('direction') in ['BUY', 'SELL']]
        strong_signals = [r for r in results if r.get('high_conviction')]

        return {
            "success": True,
            "predictions": results,
            "summary": {
                "total": len(results),
                "actionable": len(actionable),
                "strong_signals": len(strong_signals),
                "buy_signals": sum(1 for r in results if r.get('direction') == 'BUY'),
                "sell_signals": sum(1 for r in results if r.get('direction') == 'SELL'),
                "hold_signals": sum(1 for r in results if r.get('direction') == 'HOLD')
            }
        }
    except Exception as e:
        logger.error(f"Error in hybrid batch predict: {e}")
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/ai/hybrid/stats")
async def hybrid_stats():
    """Get hybrid ensemble agreement statistics"""
    try:
        from ai.hybrid_ensemble import get_hybrid_ensemble

        hybrid = get_hybrid_ensemble()
        stats = hybrid.get_stats()

        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# MULTI-ARMED BANDIT ENDPOINTS
# ============================================================================

@router.get("/ai/bandit/rankings/{arm_type}")
async def bandit_rankings(arm_type: str, regime: Optional[str] = None):
    """
    Get rankings for bandit arms (symbols, models, or strategies).

    Args:
        arm_type: 'symbol', 'model', or 'strategy'
        regime: Optional market regime for contextual rankings
    """
    try:
        from ai.bandit_selector import get_bandit, ArmType

        bandit = get_bandit()

        # Map string to enum
        type_map = {
            'symbol': ArmType.SYMBOL,
            'model': ArmType.MODEL,
            'strategy': ArmType.STRATEGY,
            'timeframe': ArmType.TIMEFRAME
        }

        if arm_type.lower() not in type_map:
            return {"success": False, "error": f"Invalid arm type: {arm_type}"}

        rankings = bandit.get_rankings(type_map[arm_type.lower()], regime)

        return {
            "success": True,
            "arm_type": arm_type,
            "regime": regime,
            "rankings": rankings
        }
    except Exception as e:
        logger.error(f"Error getting bandit rankings: {e}")
        return {"success": False, "error": str(e)}


@router.post("/ai/bandit/select/{arm_type}")
async def bandit_select(arm_type: str, n: int = 1, regime: Optional[str] = None):
    """
    Select best arm(s) using Thompson Sampling.

    Args:
        arm_type: 'symbol', 'model', or 'strategy'
        n: Number of arms to select
        regime: Optional market regime
    """
    try:
        from ai.bandit_selector import get_bandit, ArmType

        bandit = get_bandit()
        type_map = {
            'symbol': ArmType.SYMBOL,
            'model': ArmType.MODEL,
            'strategy': ArmType.STRATEGY
        }

        if arm_type.lower() not in type_map:
            return {"success": False, "error": f"Invalid arm type: {arm_type}"}

        if n == 1:
            result = bandit.select(type_map[arm_type.lower()], regime=regime)
            return {
                "success": True,
                "selected": result.selected,
                "sample_value": result.sample_value,
                "expected_value": result.expected_value,
                "confidence": result.confidence,
                "alternatives": result.alternatives
            }
        else:
            selected = bandit.select_multiple(type_map[arm_type.lower()], n, regime=regime)
            return {
                "success": True,
                "selected": selected,
                "count": len(selected)
            }
    except Exception as e:
        logger.error(f"Error in bandit selection: {e}")
        return {"success": False, "error": str(e)}


@router.post("/ai/bandit/record-trade")
async def bandit_record_trade(
    symbol: str,
    model: str,
    strategy: str,
    pnl_pct: float,
    regime: Optional[str] = None
):
    """Record trade outcome to update bandit arms."""
    try:
        from ai.bandit_selector import get_bandit

        bandit = get_bandit()
        bandit.record_trade_outcome(symbol, model, strategy, pnl_pct, regime)
        bandit.save()

        return {
            "success": True,
            "message": f"Recorded trade: {symbol} ({model}/{strategy}) -> {pnl_pct:+.2f}%"
        }
    except Exception as e:
        logger.error(f"Error recording trade to bandit: {e}")
        return {"success": False, "error": str(e)}


@router.get("/ai/bandit/stats")
async def bandit_stats():
    """Get overall bandit statistics."""
    try:
        from ai.bandit_selector import get_bandit

        bandit = get_bandit()
        return {
            "success": True,
            "stats": bandit.get_stats()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/bandit/best-symbols")
async def bandit_best_symbols(n: int = 5, regime: Optional[str] = None):
    """Get top N symbols according to bandit."""
    try:
        from ai.bandit_selector import get_symbol_selector

        selector = get_symbol_selector()
        symbols = selector.select_symbols(n=n, regime=regime)
        scores = selector.get_symbol_scores()

        return {
            "success": True,
            "symbols": symbols,
            "scores": {s: scores.get(s, {}) for s in symbols}
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# ADAPTIVE WEIGHTS ENDPOINTS
# ============================================================================

@router.get("/ai/adaptive-weights/status")
async def adaptive_weights_status():
    """Get current adaptive model weights."""
    try:
        from ai.adaptive_weights import get_weight_manager

        manager = get_weight_manager()
        return {
            "success": True,
            "weights": manager.get_weights(),
            "model_stats": manager.get_model_stats()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/adaptive-weights/ranking")
async def adaptive_weights_ranking(regime: Optional[str] = None):
    """Get models ranked by performance."""
    try:
        from ai.adaptive_weights import get_weight_manager

        manager = get_weight_manager()
        ranking = manager.get_ranking(regime)

        return {
            "success": True,
            "regime": regime,
            "ranking": ranking,
            "best_model": manager.get_best_model(regime)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/adaptive-weights/record")
async def adaptive_weights_record(
    model: str,
    prediction: int,
    actual: int,
    regime: Optional[str] = None
):
    """Record a prediction outcome for adaptive weight learning."""
    try:
        from ai.adaptive_weights import get_weight_manager

        manager = get_weight_manager()
        manager.record_prediction(model, prediction, actual, regime)

        return {
            "success": True,
            "message": f"Recorded {model}: pred={prediction}, actual={actual}",
            "new_weight": manager.models[model].current_weight if model in manager.models else None
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/adaptive-weights/rebalance")
async def adaptive_weights_rebalance(regime: Optional[str] = None):
    """Force rebalance of model weights."""
    try:
        from ai.adaptive_weights import get_weight_manager

        manager = get_weight_manager()
        manager.force_rebalance(regime)

        return {
            "success": True,
            "new_weights": manager.get_weights(regime)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ai/adaptive-ensemble/predict/{symbol}")
async def adaptive_ensemble_predict(symbol: str, regime: Optional[str] = None):
    """Get prediction from adaptive-weighted ensemble."""
    try:
        import yfinance as yf
        from ai.adaptive_weights import get_adaptive_ensemble

        # Get data
        df = yf.download(symbol, period="6mo", progress=False)
        if df.empty:
            return {"success": False, "error": f"No data for {symbol}"}

        # Handle multi-index columns
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.droplevel(1)

        ensemble = get_adaptive_ensemble()
        result = ensemble.predict(symbol, df, regime)

        return {
            "success": True,
            **result
        }
    except Exception as e:
        logger.error(f"Error in adaptive ensemble predict: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# SELF-PLAY OPTIMIZER ENDPOINTS
# ============================================================================

@router.get("/ai/selfplay/status")
async def selfplay_status():
    """Get self-play trainer status."""
    try:
        from ai.self_play_optimizer import get_selfplay_trainer

        trainer = get_selfplay_trainer()
        return {
            "success": True,
            "stats": trainer.get_stats()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/ai/selfplay/train")
async def selfplay_train(symbol: str = "SPY", episodes: int = 5):
    """Train self-play agents on historical data."""
    try:
        import yfinance as yf
        import ta
        from ai.self_play_optimizer import get_selfplay_trainer

        # Get data
        df = yf.download(symbol, period="1y", progress=False)
        if df.empty:
            return {"success": False, "error": f"No data for {symbol}"}

        # Handle multi-index columns
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.droplevel(1)

        # Add indicators
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        df['macd'] = ta.trend.MACD(df['Close']).macd()
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df = df.dropna()

        prices = df['Close'].values

        # Prepare indicators
        indicators = {
            'rsi': df['rsi'].iloc[-1],
            'macd': df['macd'].iloc[-1],
            'bb_position': (df['Close'].iloc[-1] - df['bb_low'].iloc[-1]) /
                          (df['bb_high'].iloc[-1] - df['bb_low'].iloc[-1]) if df['bb_high'].iloc[-1] != df['bb_low'].iloc[-1] else 0.5
        }

        trainer = get_selfplay_trainer()
        results = []

        for ep in range(episodes):
            result = trainer.train_episode(prices, indicators)
            results.append(result)

        return {
            "success": True,
            "symbol": symbol,
            "episodes_trained": episodes,
            "results": results,
            "final_stats": trainer.get_stats()
        }
    except Exception as e:
        logger.error(f"Error in self-play training: {e}")
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/ai/selfplay/signals/{symbol}")
async def selfplay_signals(symbol: str):
    """Get entry/exit signals from self-play agents."""
    try:
        import yfinance as yf
        import ta
        import numpy as np
        from ai.self_play_optimizer import get_selfplay_trainer, GameState

        # Get data
        df = yf.download(symbol, period="3mo", progress=False)
        if df.empty:
            return {"success": False, "error": f"No data for {symbol}"}

        # Handle multi-index columns
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.droplevel(1)

        # Add indicators
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        df['macd'] = ta.trend.MACD(df['Close']).macd()
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['trend'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()
        df['volatility'] = df['Close'].pct_change().rolling(20).std()
        df = df.dropna()

        prices = df['Close'].values[-50:]

        # Build game state
        state = GameState(
            prices=prices,
            current_price=df['Close'].iloc[-1],
            rsi=df['rsi'].iloc[-1],
            macd=df['macd'].iloc[-1],
            bb_position=(df['Close'].iloc[-1] - df['bb_low'].iloc[-1]) /
                       (df['bb_high'].iloc[-1] - df['bb_low'].iloc[-1]) if df['bb_high'].iloc[-1] != df['bb_low'].iloc[-1] else 0.5,
            volume_ratio=df['volume_ratio'].iloc[-1],
            trend=df['trend'].iloc[-1],
            volatility=df['volatility'].iloc[-1]
        )

        trainer = get_selfplay_trainer()
        entry_signal = trainer.get_entry_signal(state)
        exit_signal = trainer.get_exit_signal(state)

        return {
            "success": True,
            "symbol": symbol,
            "current_price": float(df['Close'].iloc[-1]),
            "entry_signal": entry_signal,
            "exit_signal": exit_signal,
            "recommendation": "ENTER" if entry_signal['should_enter'] else ("EXIT" if exit_signal['should_exit'] else "HOLD"),
            "indicators": {
                "rsi": float(df['rsi'].iloc[-1]),
                "macd": float(df['macd'].iloc[-1]),
                "bb_position": float(state.bb_position),
                "volume_ratio": float(df['volume_ratio'].iloc[-1]),
                "trend": float(df['trend'].iloc[-1])
            }
        }
    except Exception as e:
        logger.error(f"Error getting self-play signals: {e}")
        return {"success": False, "error": str(e)}


@router.get("/ai/selfplay/evaluate/{symbol}")
async def selfplay_evaluate(symbol: str):
    """Evaluate self-play agents on recent data."""
    try:
        import yfinance as yf
        import ta
        from ai.self_play_optimizer import get_selfplay_trainer

        # Get data
        df = yf.download(symbol, period="6mo", progress=False)
        if df.empty:
            return {"success": False, "error": f"No data for {symbol}"}

        # Handle multi-index columns
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.droplevel(1)

        # Add indicators
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        df['macd'] = ta.trend.MACD(df['Close']).macd()
        df = df.dropna()

        prices = df['Close'].values
        indicators = {
            'rsi': df['rsi'].iloc[-1],
            'macd': df['macd'].iloc[-1]
        }

        trainer = get_selfplay_trainer()
        result = trainer.evaluate(prices, indicators)

        return {
            "success": True,
            "symbol": symbol,
            **result
        }
    except Exception as e:
        logger.error(f"Error in self-play evaluation: {e}")
        return {"success": False, "error": str(e)}
