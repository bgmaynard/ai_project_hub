"""
Analytics API Routes
Trade journaling, analytics, and account management endpoints
"""
import logging
from datetime import date, datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# Trade Analytics
try:
    from trade_analytics import get_trade_analytics
    HAS_ANALYTICS = True
except ImportError as e:
    logger.warning(f"Trade analytics not available: {e}")
    HAS_ANALYTICS = False

# Schwab Trading
try:
    from schwab_trading import get_schwab_trading
    HAS_SCHWAB = True
except ImportError as e:
    logger.warning(f"Schwab trading not available: {e}")
    HAS_SCHWAB = False


class TradeRecord(BaseModel):
    """Trade record for manual entry"""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    pnl: Optional[float] = None
    strategy: Optional[str] = ""
    setup: Optional[str] = ""
    notes: Optional[str] = ""
    tags: Optional[List[str]] = []


# ============================================================================
# ACCOUNT MANAGEMENT
# ============================================================================

@router.get("/accounts")
async def get_accounts():
    """Get all available trading accounts"""
    if not HAS_SCHWAB:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    schwab = get_schwab_trading()
    accounts = schwab.get_accounts()
    selected = schwab.get_selected_account()

    return {
        "accounts": accounts,
        "selected": selected,
        "count": len(accounts)
    }


@router.post("/accounts/select/{account_number}")
async def select_account(account_number: str):
    """Switch to a different trading account"""
    if not HAS_SCHWAB:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    schwab = get_schwab_trading()

    # Verify account exists
    accounts = schwab.get_accounts()
    valid_accounts = [a['account_number'] for a in accounts]

    if account_number not in valid_accounts:
        raise HTTPException(status_code=404, detail=f"Account {account_number} not found")

    success = schwab.select_account(account_number)

    if success:
        return {
            "success": True,
            "selected_account": account_number,
            "message": f"Switched to account {account_number}"
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to switch account")


@router.get("/accounts/summary")
async def get_all_accounts_summary():
    """Get summary of all accounts"""
    if not HAS_SCHWAB:
        raise HTTPException(status_code=503, detail="Schwab trading not available")

    schwab = get_schwab_trading()
    accounts = schwab.get_accounts()
    original_account = schwab.get_selected_account()

    summaries = []
    total_value = 0
    total_pnl = 0

    for acc in accounts:
        acc_num = acc['account_number']
        schwab.select_account(acc_num)
        info = schwab.get_account_info()
        positions = schwab.get_positions()

        if info:
            summaries.append({
                "account": acc_num,
                "type": info.get('type', 'N/A'),
                "cash": info.get('cash', 0),
                "market_value": info.get('market_value', 0),
                "daily_pnl": info.get('daily_pl', 0),
                "daily_pnl_pct": info.get('daily_pl_pct', 0),
                "positions_count": len(positions) if positions else 0,
                "buying_power": info.get('buying_power', 0)
            })
            total_value += info.get('market_value', 0)
            total_pnl += info.get('daily_pl', 0)

    # Restore original account selection
    if original_account:
        schwab.select_account(original_account)

    return {
        "accounts": summaries,
        "totals": {
            "total_value": round(total_value, 2),
            "total_daily_pnl": round(total_pnl, 2),
            "account_count": len(summaries)
        }
    }


# ============================================================================
# TRADE SYNC & RECORDING
# ============================================================================

@router.post("/trades/sync")
async def sync_trades_from_schwab():
    """Sync filled orders from Schwab into trade journal"""
    if not HAS_SCHWAB or not HAS_ANALYTICS:
        raise HTTPException(status_code=503, detail="Required services not available")

    schwab = get_schwab_trading()
    analytics = get_trade_analytics()

    accounts = schwab.get_accounts()
    original_account = schwab.get_selected_account()

    total_imported = 0
    results = []

    for acc in accounts:
        acc_num = acc['account_number']
        schwab.select_account(acc_num)

        # Get filled orders from last 7 days
        orders = schwab.get_orders(status="FILLED")

        imported = analytics.sync_from_schwab(orders, acc_num)
        total_imported += imported

        results.append({
            "account": acc_num,
            "orders_found": len(orders),
            "imported": imported
        })

    # Restore original account
    if original_account:
        schwab.select_account(original_account)

    return {
        "success": True,
        "total_imported": total_imported,
        "by_account": results
    }


@router.post("/trades/record")
async def record_trade(trade: TradeRecord):
    """Manually record a trade"""
    if not HAS_ANALYTICS:
        raise HTTPException(status_code=503, detail="Analytics not available")

    analytics = get_trade_analytics()

    trade_id = f"MANUAL-{datetime.now().strftime('%Y%m%d%H%M%S')}-{trade.symbol}"

    trade_data = {
        "trade_id": trade_id,
        "symbol": trade.symbol.upper(),
        "side": trade.side.upper(),
        "quantity": trade.quantity,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "entry_time": trade.entry_time or datetime.now().isoformat(),
        "exit_time": trade.exit_time,
        "pnl": trade.pnl or 0,
        "strategy": trade.strategy,
        "setup": trade.setup,
        "notes": trade.notes,
        "tags": trade.tags,
        "trade_date": date.today().isoformat()
    }

    success = analytics.record_trade(trade_data)

    if success:
        return {"success": True, "trade_id": trade_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to record trade")


@router.get("/trades/recent")
async def get_recent_trades(limit: int = Query(default=50, le=200)):
    """Get recent trades"""
    if not HAS_ANALYTICS:
        raise HTTPException(status_code=503, detail="Analytics not available")

    analytics = get_trade_analytics()
    trades = analytics.get_recent_trades(limit)

    return {
        "trades": trades,
        "count": len(trades)
    }


# ============================================================================
# ANALYTICS & STATISTICS
# ============================================================================

@router.get("/daily")
async def get_daily_summary(target_date: Optional[str] = None):
    """Get daily trading summary"""
    if not HAS_ANALYTICS:
        raise HTTPException(status_code=503, detail="Analytics not available")

    analytics = get_trade_analytics()

    if target_date:
        try:
            dt = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        dt = date.today()

    return analytics.get_daily_summary(dt)


@router.get("/overall")
async def get_overall_stats(days: int = Query(default=30, ge=1, le=365)):
    """Get overall trading statistics"""
    if not HAS_ANALYTICS:
        raise HTTPException(status_code=503, detail="Analytics not available")

    analytics = get_trade_analytics()
    return analytics.get_overall_stats(days)


@router.get("/symbols")
async def get_symbol_performance(limit: int = Query(default=20, le=100)):
    """Get performance breakdown by symbol"""
    if not HAS_ANALYTICS:
        raise HTTPException(status_code=503, detail="Analytics not available")

    analytics = get_trade_analytics()
    return {
        "symbols": analytics.get_symbol_performance(limit)
    }


@router.get("/strategies")
async def get_strategy_performance():
    """Get performance breakdown by strategy/setup"""
    if not HAS_ANALYTICS:
        raise HTTPException(status_code=503, detail="Analytics not available")

    analytics = get_trade_analytics()
    return {
        "strategies": analytics.get_strategy_performance()
    }


@router.get("/time-analysis")
async def get_time_analysis():
    """Get performance analysis by time of day and day of week"""
    if not HAS_ANALYTICS:
        raise HTTPException(status_code=503, detail="Analytics not available")

    analytics = get_trade_analytics()
    return analytics.get_time_analysis()


@router.get("/insights")
async def get_trading_insights():
    """Get AI-generated trading insights and recommendations"""
    if not HAS_ANALYTICS:
        raise HTTPException(status_code=503, detail="Analytics not available")

    analytics = get_trade_analytics()
    return analytics.get_insights()


@router.get("/report")
async def get_comprehensive_report(days: int = Query(default=30, ge=1, le=365)):
    """Get comprehensive trading report"""
    if not HAS_ANALYTICS:
        raise HTTPException(status_code=503, detail="Analytics not available")

    analytics = get_trade_analytics()

    return {
        "period_days": days,
        "generated_at": datetime.now().isoformat(),
        "overall_stats": analytics.get_overall_stats(days),
        "daily_summary": analytics.get_daily_summary(),
        "top_symbols": analytics.get_symbol_performance(10),
        "strategy_performance": analytics.get_strategy_performance(),
        "time_analysis": analytics.get_time_analysis(),
        "insights": analytics.get_insights()
    }
