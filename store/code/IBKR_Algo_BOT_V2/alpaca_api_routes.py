"""
Alpaca API Routes for Dashboard
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from alpaca_integration import get_alpaca_connector

router = APIRouter(prefix="/api/alpaca", tags=["alpaca"])

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
    """Place an order"""
    connector = get_alpaca_connector()

    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")

    try:
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
        result = connector.place_bracket_order(
            symbol=order.symbol,
            quantity=order.quantity,
            side=order.action,
            take_profit_price=order.take_profit_price,
            stop_loss_price=order.stop_loss_price,
            limit_price=order.limit_price
        )

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
        result = connector.place_oco_order(
            symbol=order.symbol,
            quantity=order.quantity,
            side=order.action,
            take_profit_price=order.take_profit_price,
            stop_loss_price=order.stop_loss_price
        )

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
        result = connector.place_trailing_stop_order(
            symbol=order.symbol,
            quantity=order.quantity,
            side=order.action,
            trail_percent=order.trail_percent,
            trail_price=order.trail_price
        )

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
        result = connector.place_stop_order(
            symbol=order.symbol,
            quantity=order.quantity,
            side=order.action,
            stop_price=order.stop_price
        )

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
        result = connector.place_stop_limit_order(
            symbol=order.symbol,
            quantity=order.quantity,
            side=order.action,
            stop_price=order.stop_price,
            limit_price=order.limit_price
        )

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
            }
        ]
    }


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
    Full conversational AI chat with Claude-like reasoning.
    Provides unrestricted conversation with trading context awareness.
    """
    try:
        # Use the new full conversational AI for Claude-like experience
        from ai.claude_conversational_ai import get_conversational_ai
        ai = get_conversational_ai()

        message = data.get("message", "")
        session_id = data.get("session_id", "default")
        use_tools = data.get("use_tools", True)

        result = ai.chat(message, session_id=session_id, use_tools=use_tools)

        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"AI chat error: {e}")
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
