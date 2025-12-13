"""
Analytics API Routes
Portfolio analytics, trade tracking, and execution metrics
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
import logging

from portfolio_analytics import get_portfolio_analytics
from trade_execution import get_execution_tracker
from alpaca_integration import get_alpaca_connector
from alpaca_market_data import get_alpaca_market_data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


# ============================================================================
# REQUEST MODELS
# ============================================================================

class TradeEntryRequest(BaseModel):
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    price: float
    order_id: Optional[str] = None
    ai_signal: Optional[str] = None
    ai_confidence: Optional[float] = None


class TradeExitRequest(BaseModel):
    symbol: str
    price: float
    order_id: Optional[str] = None


class ExecutionStartRequest(BaseModel):
    order_id: str
    symbol: str
    side: str
    quantity: int
    order_type: str  # MKT or LMT
    requested_price: Optional[float] = None
    market_price: Optional[float] = None


class ExecutionCompleteRequest(BaseModel):
    order_id: str
    executed_price: float
    status: str = "FILLED"


# ============================================================================
# PORTFOLIO ANALYTICS ENDPOINTS
# ============================================================================

@router.get("/portfolio/metrics")
async def get_portfolio_metrics():
    """Get comprehensive portfolio performance metrics"""
    try:
        analytics = get_portfolio_analytics()
        connector = get_alpaca_connector()

        # Get current account value from Alpaca
        account_value = 100000  # Default
        if connector.is_connected():
            account = connector.get_account()
            account_value = float(account.get("portfolio_value", 100000))

        metrics = analytics.calculate_metrics(account_value)
        return metrics

    except Exception as e:
        logger.error(f"Error getting portfolio metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/positions")
async def get_positions_with_pnl():
    """Get current positions with real-time P&L"""
    try:
        connector = get_alpaca_connector()
        market_data = get_alpaca_market_data()
        analytics = get_portfolio_analytics()

        if not connector.is_connected():
            raise HTTPException(status_code=503, detail="Not connected to broker")

        positions = connector.get_positions()

        # Enhance positions with P&L data
        enhanced_positions = []
        for pos in positions:
            symbol = pos.get("symbol")
            current_price = pos.get("current_price", 0)

            # Get trade record P&L if available
            trade_pnl = analytics.calculate_position_pnl(symbol, current_price)

            enhanced_positions.append({
                **pos,
                "trade_record": trade_pnl if "error" not in trade_pnl else None
            })

        return {
            "positions": enhanced_positions,
            "count": len(enhanced_positions)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting positions with P&L: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/portfolio/trade/entry")
async def record_trade_entry(request: TradeEntryRequest):
    """Record a new trade entry"""
    try:
        analytics = get_portfolio_analytics()
        trade = analytics.record_trade_entry(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            price=request.price,
            order_id=request.order_id,
            ai_signal=request.ai_signal,
            ai_confidence=request.ai_confidence
        )

        return {
            "success": True,
            "trade": {
                "symbol": trade.symbol,
                "side": trade.side,
                "quantity": trade.quantity,
                "entry_price": trade.entry_price,
                "entry_time": trade.entry_time
            }
        }

    except Exception as e:
        logger.error(f"Error recording trade entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/portfolio/trade/exit")
async def record_trade_exit(request: TradeExitRequest):
    """Record a trade exit and calculate P&L"""
    try:
        analytics = get_portfolio_analytics()
        trade = analytics.record_trade_exit(
            symbol=request.symbol,
            price=request.price,
            order_id=request.order_id
        )

        if trade is None:
            raise HTTPException(status_code=404, detail=f"No open trade found for {request.symbol}")

        return {
            "success": True,
            "trade": {
                "symbol": trade.symbol,
                "side": trade.side,
                "quantity": trade.quantity,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "pnl": trade.pnl,
                "pnl_percent": trade.pnl_percent
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording trade exit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/trades/open")
async def get_open_trades():
    """Get all open trades"""
    try:
        analytics = get_portfolio_analytics()
        trades = analytics.get_open_trades()
        return {"trades": trades, "count": len(trades)}

    except Exception as e:
        logger.error(f"Error getting open trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/trades/closed")
async def get_closed_trades(days: int = Query(default=30, ge=1, le=365)):
    """Get closed trades from last N days"""
    try:
        analytics = get_portfolio_analytics()
        trades = analytics.get_closed_trades(days)
        return {"trades": trades, "count": len(trades), "period_days": days}

    except Exception as e:
        logger.error(f"Error getting closed trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/trades/{symbol}")
async def get_symbol_trades(symbol: str):
    """Get all trades for a specific symbol"""
    try:
        analytics = get_portfolio_analytics()
        trades = analytics.get_trade_by_symbol(symbol)
        return {"symbol": symbol, "trades": trades, "count": len(trades)}

    except Exception as e:
        logger.error(f"Error getting trades for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/ai-performance")
async def get_ai_performance():
    """Analyze AI signal performance"""
    try:
        analytics = get_portfolio_analytics()
        return analytics.get_ai_performance()

    except Exception as e:
        logger.error(f"Error getting AI performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EXECUTION TRACKING ENDPOINTS
# ============================================================================

@router.post("/execution/start")
async def start_execution_tracking(request: ExecutionStartRequest):
    """Start tracking an order for execution quality analysis"""
    try:
        tracker = get_execution_tracker()
        tracker.start_order_tracking(
            order_id=request.order_id,
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            order_type=request.order_type,
            requested_price=request.requested_price,
            market_price=request.market_price
        )

        return {"success": True, "order_id": request.order_id, "status": "tracking"}

    except Exception as e:
        logger.error(f"Error starting execution tracking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execution/complete")
async def complete_execution_tracking(request: ExecutionCompleteRequest):
    """Record an order execution"""
    try:
        tracker = get_execution_tracker()
        execution = tracker.record_execution(
            order_id=request.order_id,
            executed_price=request.executed_price,
            status=request.status
        )

        if execution is None:
            raise HTTPException(status_code=404, detail=f"Order {request.order_id} not being tracked")

        return {
            "success": True,
            "execution": {
                "order_id": execution.order_id,
                "symbol": execution.symbol,
                "executed_price": execution.executed_price,
                "slippage": execution.slippage,
                "slippage_percent": execution.slippage_percent,
                "execution_time_ms": execution.execution_time_ms
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing execution tracking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/execution/stats")
async def get_execution_stats(days: int = Query(default=30, ge=1, le=365)):
    """Get execution statistics for the last N days"""
    try:
        tracker = get_execution_tracker()
        return tracker.get_execution_stats(days)

    except Exception as e:
        logger.error(f"Error getting execution stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/execution/recent")
async def get_recent_executions(limit: int = Query(default=20, ge=1, le=100)):
    """Get recent execution records"""
    try:
        tracker = get_execution_tracker()
        executions = tracker.get_recent_executions(limit)
        return {"executions": executions, "count": len(executions)}

    except Exception as e:
        logger.error(f"Error getting recent executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/execution/symbol/{symbol}")
async def get_symbol_execution_stats(symbol: str):
    """Get execution stats for a specific symbol"""
    try:
        tracker = get_execution_tracker()
        return tracker.get_symbol_stats(symbol)

    except Exception as e:
        logger.error(f"Error getting execution stats for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/execution/slippage-alerts")
async def get_slippage_alerts(threshold: float = Query(default=0.5, ge=0.01, le=5.0)):
    """Get executions with high slippage"""
    try:
        tracker = get_execution_tracker()
        alerts = tracker.check_slippage_alert(threshold)
        return {
            "threshold_percent": threshold,
            "alerts": alerts,
            "count": len(alerts)
        }

    except Exception as e:
        logger.error(f"Error getting slippage alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# COMBINED DASHBOARD ENDPOINT
# ============================================================================

@router.get("/dashboard")
async def get_analytics_dashboard():
    """Get combined analytics dashboard data"""
    try:
        analytics = get_portfolio_analytics()
        tracker = get_execution_tracker()
        connector = get_alpaca_connector()

        # Get account info
        account_value = 100000
        account_info = {}
        if connector.is_connected():
            account = connector.get_account()
            account_value = float(account.get("portfolio_value", 100000))
            account_info = {
                "portfolio_value": account_value,
                "buying_power": float(account.get("buying_power", 0)),
                "cash": float(account.get("cash", 0)),
                "equity": float(account.get("equity", 0))
            }

        return {
            "account": account_info,
            "portfolio_metrics": analytics.calculate_metrics(account_value),
            "open_trades": len(analytics.get_open_trades()),
            "execution_stats": tracker.get_execution_stats(30),
            "ai_performance": analytics.get_ai_performance(),
            "timestamp": __import__("datetime").datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting analytics dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))
