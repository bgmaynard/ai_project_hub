"""
Analytics API Routes
Portfolio analytics, trade tracking, and execution metrics
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Optional imports - these modules may not exist
try:
    from portfolio_analytics import get_portfolio_analytics

    HAS_PORTFOLIO_ANALYTICS = True
except ImportError:
    HAS_PORTFOLIO_ANALYTICS = False
    get_portfolio_analytics = None

try:
    from trade_execution import get_execution_tracker

    HAS_EXECUTION_TRACKER = True
except ImportError:
    HAS_EXECUTION_TRACKER = False
    get_execution_tracker = None

try:
    from unified_broker import get_unified_broker

    HAS_UNIFIED_BROKER = True
except ImportError:
    HAS_UNIFIED_BROKER = False
    get_unified_broker = None

try:
    from unified_market_data import get_unified_market_data

    HAS_UNIFIED_DATA = True
except ImportError:
    HAS_UNIFIED_DATA = False
    get_unified_market_data = None

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
        broker = get_unified_broker()

        # Get current account value from broker
        account_value = 100000  # Default
        if broker.is_connected:
            account = broker.get_account()
            if account:
                account_value = float(
                    account.get("market_value", account.get("portfolio_value", 100000))
                )

        metrics = analytics.calculate_metrics(account_value)
        return metrics

    except Exception as e:
        logger.error(f"Error getting portfolio metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/positions")
async def get_positions_with_pnl():
    """Get current positions with real-time P&L"""
    try:
        broker = get_unified_broker()
        analytics = get_portfolio_analytics()

        if not broker.is_connected:
            raise HTTPException(status_code=503, detail="Not connected to broker")

        positions = broker.get_positions()

        # Enhance positions with P&L data
        enhanced_positions = []
        for pos in positions:
            symbol = pos.get("symbol")
            current_price = pos.get("current_price", 0)

            # Get trade record P&L if available
            trade_pnl = analytics.calculate_position_pnl(symbol, current_price)

            enhanced_positions.append(
                {**pos, "trade_record": trade_pnl if "error" not in trade_pnl else None}
            )

        return {"positions": enhanced_positions, "count": len(enhanced_positions)}

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
            ai_confidence=request.ai_confidence,
        )

        return {
            "success": True,
            "trade": {
                "symbol": trade.symbol,
                "side": trade.side,
                "quantity": trade.quantity,
                "entry_price": trade.entry_price,
                "entry_time": trade.entry_time,
            },
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
            symbol=request.symbol, price=request.price, order_id=request.order_id
        )

        if trade is None:
            raise HTTPException(
                status_code=404, detail=f"No open trade found for {request.symbol}"
            )

        return {
            "success": True,
            "trade": {
                "symbol": trade.symbol,
                "side": trade.side,
                "quantity": trade.quantity,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "pnl": trade.pnl,
                "pnl_percent": trade.pnl_percent,
            },
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
            market_price=request.market_price,
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
            status=request.status,
        )

        if execution is None:
            raise HTTPException(
                status_code=404, detail=f"Order {request.order_id} not being tracked"
            )

        return {
            "success": True,
            "execution": {
                "order_id": execution.order_id,
                "symbol": execution.symbol,
                "executed_price": execution.executed_price,
                "slippage": execution.slippage,
                "slippage_percent": execution.slippage_percent,
                "execution_time_ms": execution.execution_time_ms,
            },
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
        return {"threshold_percent": threshold, "alerts": alerts, "count": len(alerts)}

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
        broker = get_unified_broker()

        # Get account info
        account_value = 100000
        account_info = {}
        if broker.is_connected:
            account = broker.get_account()
            if account:
                account_value = float(
                    account.get("market_value", account.get("portfolio_value", 100000))
                )
                account_info = {
                    "portfolio_value": account_value,
                    "buying_power": float(account.get("buying_power", 0)),
                    "cash": float(account.get("cash", 0)),
                    "equity": float(account.get("market_value", 0)),
                }

        return {
            "account": account_info,
            "portfolio_metrics": analytics.calculate_metrics(account_value),
            "open_trades": len(analytics.get_open_trades()),
            "execution_stats": tracker.get_execution_stats(30),
            "ai_performance": analytics.get_ai_performance(),
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting analytics dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TRADE JOURNAL & ANALYTICS (TraderVue-style)
# ============================================================================

# Trade Analytics Module
try:
    from trade_analytics import get_trade_analytics

    HAS_TRADE_ANALYTICS = True
except ImportError as e:
    logger.warning(f"Trade analytics not available: {e}")
    HAS_TRADE_ANALYTICS = False

# Schwab Trading (for account switching)
try:
    from schwab_trading import get_schwab_trading

    HAS_SCHWAB_DIRECT = True
except ImportError:
    HAS_SCHWAB_DIRECT = False


class ManualTradeRecord(BaseModel):
    """Trade record for manual entry"""

    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    strategy: Optional[str] = ""
    setup: Optional[str] = ""
    notes: Optional[str] = ""


@router.get("/accounts/all")
async def get_all_accounts():
    """Get all available trading accounts"""
    if not HAS_SCHWAB_DIRECT:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    schwab = get_schwab_trading()
    accounts = schwab.get_accounts()
    selected = schwab.get_selected_account()

    return {"accounts": accounts, "selected": selected, "count": len(accounts)}


@router.post("/accounts/switch/{account_number}")
async def switch_account(account_number: str):
    """Switch to a different trading account"""
    if not HAS_SCHWAB_DIRECT:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    schwab = get_schwab_trading()
    accounts = schwab.get_accounts()
    valid_accounts = [a["account_number"] for a in accounts]

    if account_number not in valid_accounts:
        raise HTTPException(
            status_code=404, detail=f"Account {account_number} not found"
        )

    success = schwab.select_account(account_number)

    if success:
        # Get the account info for the newly selected account
        info = schwab.get_account_info()
        return {
            "success": True,
            "selected_account": account_number,
            "account_info": info,
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to switch account")


@router.get("/accounts/summary/all")
async def get_all_accounts_summary():
    """Get summary of all accounts with P&L"""
    if not HAS_SCHWAB_DIRECT:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    schwab = get_schwab_trading()
    accounts = schwab.get_accounts()
    original_account = schwab.get_selected_account()

    summaries = []
    total_value = 0
    total_pnl = 0

    for acc in accounts:
        acc_num = acc["account_number"]
        schwab.select_account(acc_num)
        info = schwab.get_account_info()
        positions = schwab.get_positions()

        if info:
            summaries.append(
                {
                    "account": acc_num,
                    "type": info.get("type", "N/A"),
                    "cash": round(info.get("cash", 0), 2),
                    "market_value": round(info.get("market_value", 0), 2),
                    "daily_pnl": round(info.get("daily_pl", 0), 2),
                    "daily_pnl_pct": round(info.get("daily_pl_pct", 0), 2),
                    "positions_count": len(positions) if positions else 0,
                }
            )
            total_value += info.get("market_value", 0)
            total_pnl += info.get("daily_pl", 0)

    # Restore original account
    if original_account:
        schwab.select_account(original_account)

    return {
        "accounts": summaries,
        "totals": {
            "total_value": round(total_value, 2),
            "total_daily_pnl": round(total_pnl, 2),
            "account_count": len(summaries),
        },
    }


@router.post("/journal/sync")
async def sync_trades_from_broker():
    """Sync filled orders from Schwab into trade journal"""
    if not HAS_SCHWAB_DIRECT or not HAS_TRADE_ANALYTICS:
        raise HTTPException(status_code=503, detail="Required services not available")

    schwab = get_schwab_trading()
    analytics = get_trade_analytics()

    accounts = schwab.get_accounts()
    original_account = schwab.get_selected_account()

    total_imported = 0
    results = []

    for acc in accounts:
        acc_num = acc["account_number"]
        schwab.select_account(acc_num)

        # Get filled orders
        orders = schwab.get_orders(status="FILLED")
        imported = analytics.sync_from_schwab(orders, acc_num)
        total_imported += imported

        results.append(
            {"account": acc_num, "orders_found": len(orders), "imported": imported}
        )

    if original_account:
        schwab.select_account(original_account)

    return {"success": True, "total_imported": total_imported, "by_account": results}


@router.post("/journal/record")
async def record_manual_trade(trade: ManualTradeRecord):
    """Manually record a trade to the journal"""
    if not HAS_TRADE_ANALYTICS:
        raise HTTPException(status_code=503, detail="Trade analytics not available")

    from datetime import date, datetime

    analytics = get_trade_analytics()

    trade_id = f"MANUAL-{datetime.now().strftime('%Y%m%d%H%M%S')}-{trade.symbol}"

    trade_data = {
        "trade_id": trade_id,
        "symbol": trade.symbol.upper(),
        "side": trade.side.upper(),
        "quantity": trade.quantity,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "pnl": trade.pnl or 0,
        "strategy": trade.strategy,
        "setup": trade.setup,
        "notes": trade.notes,
        "entry_time": datetime.now().isoformat(),
        "trade_date": date.today().isoformat(),
    }

    success = analytics.record_trade(trade_data)

    if success:
        return {"success": True, "trade_id": trade_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to record trade")


@router.get("/journal/trades")
async def get_journal_trades(limit: int = Query(default=50, le=200)):
    """Get recent trades from journal"""
    if not HAS_TRADE_ANALYTICS:
        raise HTTPException(status_code=503, detail="Trade analytics not available")

    analytics = get_trade_analytics()
    trades = analytics.get_recent_trades(limit)

    return {"trades": trades, "count": len(trades)}


@router.get("/journal/daily")
async def get_daily_journal_summary(target_date: Optional[str] = None):
    """Get daily trading summary"""
    if not HAS_TRADE_ANALYTICS:
        raise HTTPException(status_code=503, detail="Trade analytics not available")

    from datetime import date, datetime

    analytics = get_trade_analytics()

    if target_date:
        try:
            dt = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid date format. Use YYYY-MM-DD"
            )
    else:
        dt = date.today()

    return analytics.get_daily_summary(dt)


@router.get("/journal/stats")
async def get_journal_overall_stats(days: int = Query(default=30, ge=1, le=365)):
    """Get overall trading statistics"""
    if not HAS_TRADE_ANALYTICS:
        raise HTTPException(status_code=503, detail="Trade analytics not available")

    analytics = get_trade_analytics()
    return analytics.get_overall_stats(days)


@router.get("/journal/symbols")
async def get_symbol_stats(limit: int = Query(default=20, le=100)):
    """Get performance breakdown by symbol"""
    if not HAS_TRADE_ANALYTICS:
        raise HTTPException(status_code=503, detail="Trade analytics not available")

    analytics = get_trade_analytics()
    return {"symbols": analytics.get_symbol_performance(limit)}


@router.get("/journal/strategies")
async def get_strategy_stats():
    """Get performance breakdown by strategy/setup"""
    if not HAS_TRADE_ANALYTICS:
        raise HTTPException(status_code=503, detail="Trade analytics not available")

    analytics = get_trade_analytics()
    return {"strategies": analytics.get_strategy_performance()}


@router.get("/journal/time-analysis")
async def get_time_stats():
    """Get performance by time of day and day of week"""
    if not HAS_TRADE_ANALYTICS:
        raise HTTPException(status_code=503, detail="Trade analytics not available")

    analytics = get_trade_analytics()
    return analytics.get_time_analysis()


@router.get("/journal/insights")
async def get_trading_insights():
    """Get actionable trading insights - what's working, what's not"""
    if not HAS_TRADE_ANALYTICS:
        raise HTTPException(status_code=503, detail="Trade analytics not available")

    analytics = get_trade_analytics()
    return analytics.get_insights()


@router.get("/journal/report")
async def get_full_trading_report(days: int = Query(default=30, ge=1, le=365)):
    """Get comprehensive trading report (TraderVue-style)"""
    if not HAS_TRADE_ANALYTICS:
        raise HTTPException(status_code=503, detail="Trade analytics not available")

    from datetime import datetime

    analytics = get_trade_analytics()

    return {
        "report_type": "comprehensive",
        "period_days": days,
        "generated_at": datetime.now().isoformat(),
        "overall_stats": analytics.get_overall_stats(days),
        "today_summary": analytics.get_daily_summary(),
        "top_symbols": analytics.get_symbol_performance(10),
        "strategy_performance": analytics.get_strategy_performance(),
        "time_analysis": analytics.get_time_analysis(),
        "insights": analytics.get_insights(),
    }


@router.post("/journal/calculate-pnl")
async def calculate_pnl_from_fills():
    """Calculate P&L by matching buy/sell fills (FIFO method)"""
    if not HAS_TRADE_ANALYTICS:
        raise HTTPException(status_code=503, detail="Trade analytics not available")

    analytics = get_trade_analytics()
    result = analytics.calculate_pnl_from_fills()

    return result
