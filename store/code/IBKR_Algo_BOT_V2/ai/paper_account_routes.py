"""
Paper Account API Routes
========================
Endpoints for paper trading account visibility.
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import Dict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/paper", tags=["paper-account"])


@router.get("/account")
async def get_paper_account():
    """Get paper account summary with positions and equity"""
    try:
        from .paper_account_metrics import get_paper_metrics
        from .hft_scalper import get_hft_scalper

        metrics = get_paper_metrics()
        scalper = get_hft_scalper()

        # Get open positions
        open_positions = []
        current_prices = {}

        if scalper:
            for trade_id, pos in scalper.open_positions.items():
                open_positions.append({
                    "symbol": pos.symbol,
                    "shares": pos.shares,
                    "entry_price": pos.entry_price,
                    "entry_time": pos.entry_time.isoformat() if pos.entry_time else "",
                    "hold_time_seconds": pos.hold_time_seconds
                })

            # Get current prices for positions
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                for pos in open_positions:
                    try:
                        resp = await client.get(f"http://localhost:9100/api/price/{pos['symbol']}")
                        if resp.status_code == 200:
                            data = resp.json()
                            current_prices[pos['symbol']] = data.get('price', pos['entry_price'])
                    except:
                        current_prices[pos['symbol']] = pos['entry_price']

            # Sync from scalper stats
            stats = scalper.get_stats()
            metrics.sync_from_scalper(stats, open_positions)

        summary = metrics.get_account_summary(open_positions, current_prices)
        return {"success": True, **summary}

    except Exception as e:
        logger.error(f"Error getting paper account: {e}")
        return {"success": False, "error": str(e)}


@router.get("/balance")
async def get_paper_balance():
    """Get quick balance overview"""
    try:
        from .paper_account_metrics import get_paper_metrics
        from .hft_scalper import get_hft_scalper

        metrics = get_paper_metrics()
        scalper = get_hft_scalper()

        # Sync realized P&L from scalper
        if scalper:
            stats = scalper.get_stats()
            metrics.realized_pnl = stats.get("total_pnl", 0.0)

        equity = metrics.get_current_equity()

        return {
            "success": True,
            "starting_balance": metrics.starting_balance,
            "realized_pnl": round(metrics.realized_pnl, 2),
            "current_equity": round(equity, 2),
            "roi_percent": round((equity - metrics.starting_balance) / metrics.starting_balance * 100, 2)
        }

    except Exception as e:
        logger.error(f"Error getting balance: {e}")
        return {"success": False, "error": str(e)}


@router.get("/trades")
async def get_paper_trades():
    """Get list of all paper trades"""
    try:
        from .hft_scalper import get_hft_scalper

        scalper = get_hft_scalper()
        if not scalper:
            return {"success": False, "error": "Scalper not initialized"}

        trades = []
        for trade in scalper.trades:
            trades.append({
                "trade_id": trade.trade_id,
                "symbol": trade.symbol,
                "entry_time": trade.entry_time.isoformat() if trade.entry_time else "",
                "exit_time": trade.exit_time.isoformat() if trade.exit_time else "",
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "shares": trade.shares,
                "pnl": round(trade.pnl, 2),
                "pnl_percent": round(trade.pnl_percent, 2),
                "exit_reason": trade.exit_reason,
                "hold_time_seconds": trade.hold_time_seconds,
                "max_gain_percent": round(trade.max_gain_percent, 2),
                "max_drawdown_percent": round(trade.max_drawdown_percent, 2),
                "status": trade.status
            })

        # Sort by exit time descending (most recent first)
        trades.sort(key=lambda x: x['exit_time'] if x['exit_time'] else x['entry_time'], reverse=True)

        # Calculate summary
        total_pnl = sum(t['pnl'] for t in trades if t['status'] == 'closed')
        winners = [t for t in trades if t['pnl'] > 0 and t['status'] == 'closed']
        losers = [t for t in trades if t['pnl'] <= 0 and t['status'] == 'closed']

        return {
            "success": True,
            "total_trades": len(trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(len(winners) / len(trades) * 100, 1) if trades else 0,
            "total_pnl": round(total_pnl, 2),
            "trades": trades
        }

    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


@router.get("/trades/winners")
async def get_winning_trades():
    """Get only winning trades"""
    result = await get_paper_trades()
    if result.get("success"):
        result["trades"] = [t for t in result["trades"] if t["pnl"] > 0]
        result["total_trades"] = len(result["trades"])
    return result


@router.get("/trades/losers")
async def get_losing_trades():
    """Get only losing trades"""
    result = await get_paper_trades()
    if result.get("success"):
        result["trades"] = [t for t in result["trades"] if t["pnl"] <= 0]
        result["total_trades"] = len(result["trades"])
    return result


@router.get("/equity-curve")
async def get_equity_curve():
    """Get equity curve data for charting"""
    try:
        from .paper_account_metrics import get_paper_metrics

        metrics = get_paper_metrics()
        return {
            "success": True,
            "starting_balance": metrics.starting_balance,
            "high_water_mark": metrics.high_water_mark,
            "equity_history": metrics.equity_history
        }

    except Exception as e:
        logger.error(f"Error getting equity curve: {e}")
        return {"success": False, "error": str(e)}


@router.get("/stats")
async def get_paper_stats():
    """Get detailed paper trading statistics"""
    try:
        from .hft_scalper import get_hft_scalper
        from .paper_account_metrics import get_paper_metrics

        scalper = get_hft_scalper()
        metrics = get_paper_metrics()

        if not scalper:
            return {"success": False, "error": "Scalper not initialized"}

        stats = scalper.get_stats()

        # Analyze by exit reason
        exit_reasons = {}
        for trade in scalper.trades:
            reason = trade.exit_reason or "unknown"
            if reason not in exit_reasons:
                exit_reasons[reason] = {"count": 0, "pnl": 0.0, "wins": 0}
            exit_reasons[reason]["count"] += 1
            exit_reasons[reason]["pnl"] += trade.pnl
            if trade.pnl > 0:
                exit_reasons[reason]["wins"] += 1

        # Analyze by symbol
        symbols = {}
        for trade in scalper.trades:
            sym = trade.symbol
            if sym not in symbols:
                symbols[sym] = {"count": 0, "pnl": 0.0, "wins": 0}
            symbols[sym]["count"] += 1
            symbols[sym]["pnl"] += trade.pnl
            if trade.pnl > 0:
                symbols[sym]["wins"] += 1

        return {
            "success": True,
            "account": {
                "starting_balance": metrics.starting_balance,
                "current_equity": round(metrics.starting_balance + stats.get("total_pnl", 0), 2),
                "roi_percent": round(stats.get("total_pnl", 0) / metrics.starting_balance * 100, 2)
            },
            "performance": stats,
            "by_exit_reason": {k: {
                "count": v["count"],
                "pnl": round(v["pnl"], 2),
                "win_rate": round(v["wins"] / v["count"] * 100, 1) if v["count"] > 0 else 0
            } for k, v in exit_reasons.items()},
            "by_symbol": {k: {
                "count": v["count"],
                "pnl": round(v["pnl"], 2),
                "win_rate": round(v["wins"] / v["count"] * 100, 1) if v["count"] > 0 else 0
            } for k, v in sorted(symbols.items(), key=lambda x: x[1]["pnl"], reverse=True)}
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"success": False, "error": str(e)}


@router.get("/analytics")
async def get_trade_analytics():
    """Get comprehensive trade analytics for tuning"""
    try:
        from .hft_scalper import get_hft_scalper
        from datetime import datetime

        scalper = get_hft_scalper()
        if not scalper:
            return {"success": False, "error": "Scalper not initialized"}

        trades = [t for t in scalper.trades if t.status == "closed"]

        if not trades:
            return {"success": True, "message": "No closed trades yet", "analytics": {}}

        # By Exit Reason Analysis
        by_exit = {}
        for t in trades:
            reason = t.exit_reason or "unknown"
            if reason not in by_exit:
                by_exit[reason] = {"count": 0, "wins": 0, "pnl": 0.0, "avg_hold": 0.0,
                                   "avg_mfe": 0.0, "avg_mae": 0.0, "trades": []}
            by_exit[reason]["count"] += 1
            by_exit[reason]["pnl"] += t.pnl
            by_exit[reason]["avg_hold"] += t.hold_time_seconds
            by_exit[reason]["avg_mfe"] += t.max_gain_percent
            by_exit[reason]["avg_mae"] += abs(t.max_drawdown_percent)
            if t.pnl > 0:
                by_exit[reason]["wins"] += 1
            by_exit[reason]["trades"].append(t.symbol)

        for reason in by_exit:
            cnt = by_exit[reason]["count"]
            by_exit[reason]["win_rate"] = round(by_exit[reason]["wins"] / cnt * 100, 1)
            by_exit[reason]["avg_pnl"] = round(by_exit[reason]["pnl"] / cnt, 2)
            by_exit[reason]["avg_hold"] = round(by_exit[reason]["avg_hold"] / cnt, 1)
            by_exit[reason]["avg_mfe"] = round(by_exit[reason]["avg_mfe"] / cnt, 2)
            by_exit[reason]["avg_mae"] = round(by_exit[reason]["avg_mae"] / cnt, 2)
            by_exit[reason]["pnl"] = round(by_exit[reason]["pnl"], 2)
            by_exit[reason]["symbols"] = list(set(by_exit[reason]["trades"]))
            del by_exit[reason]["trades"]

        # By Symbol Analysis
        by_symbol = {}
        for t in trades:
            sym = t.symbol
            if sym not in by_symbol:
                by_symbol[sym] = {"count": 0, "wins": 0, "pnl": 0.0, "avg_hold": 0.0}
            by_symbol[sym]["count"] += 1
            by_symbol[sym]["pnl"] += t.pnl
            by_symbol[sym]["avg_hold"] += t.hold_time_seconds
            if t.pnl > 0:
                by_symbol[sym]["wins"] += 1

        for sym in by_symbol:
            cnt = by_symbol[sym]["count"]
            by_symbol[sym]["win_rate"] = round(by_symbol[sym]["wins"] / cnt * 100, 1)
            by_symbol[sym]["avg_pnl"] = round(by_symbol[sym]["pnl"] / cnt, 2)
            by_symbol[sym]["avg_hold"] = round(by_symbol[sym]["avg_hold"] / cnt, 1)
            by_symbol[sym]["pnl"] = round(by_symbol[sym]["pnl"], 2)

        # By Time of Day Analysis
        by_time = {}
        for t in trades:
            tod = t.secondary_triggers.get("time_of_day", "unknown") if t.secondary_triggers else "unknown"
            if tod not in by_time:
                by_time[tod] = {"count": 0, "wins": 0, "pnl": 0.0}
            by_time[tod]["count"] += 1
            by_time[tod]["pnl"] += t.pnl
            if t.pnl > 0:
                by_time[tod]["wins"] += 1

        for tod in by_time:
            cnt = by_time[tod]["count"]
            by_time[tod]["win_rate"] = round(by_time[tod]["wins"] / cnt * 100, 1)
            by_time[tod]["pnl"] = round(by_time[tod]["pnl"], 2)

        # MFE/MAE Analysis (money left on table vs drawdown taken)
        mfe_analysis = {
            "avg_mfe": round(sum(t.max_gain_percent for t in trades) / len(trades), 2),
            "avg_mae": round(sum(abs(t.max_drawdown_percent) for t in trades) / len(trades), 2),
            "trades_with_mfe_gt_1pct": len([t for t in trades if t.max_gain_percent > 1]),
            "trades_with_mfe_gt_2pct": len([t for t in trades if t.max_gain_percent > 2]),
            "winners_avg_mfe": round(sum(t.max_gain_percent for t in trades if t.pnl > 0) / max(1, len([t for t in trades if t.pnl > 0])), 2),
            "losers_avg_mae": round(sum(abs(t.max_drawdown_percent) for t in trades if t.pnl <= 0) / max(1, len([t for t in trades if t.pnl <= 0])), 2),
        }

        # Hold Time Analysis
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        hold_analysis = {
            "avg_hold_all": round(sum(t.hold_time_seconds for t in trades) / len(trades), 1),
            "avg_hold_winners": round(sum(t.hold_time_seconds for t in winners) / max(1, len(winners)), 1),
            "avg_hold_losers": round(sum(t.hold_time_seconds for t in losers) / max(1, len(losers)), 1),
            "shortest_winner": min([t.hold_time_seconds for t in winners]) if winners else 0,
            "longest_loser": max([t.hold_time_seconds for t in losers]) if losers else 0,
        }

        # Tuning Recommendations
        recommendations = []

        # Check if TRAILING_STOP is best performer
        if "TRAILING_STOP" in by_exit and by_exit["TRAILING_STOP"]["win_rate"] > 70:
            recommendations.append("TRAILING_STOP exits have {:.0f}% win rate - let more trades reach trailing".format(
                by_exit["TRAILING_STOP"]["win_rate"]))

        # Check if avg loss > avg win
        avg_win = sum(t.pnl for t in winners) / max(1, len(winners))
        avg_loss = abs(sum(t.pnl for t in losers) / max(1, len(losers)))
        if avg_loss > avg_win:
            recommendations.append("Avg loss (${:.2f}) > Avg win (${:.2f}) - tighten stops or widen targets".format(
                avg_loss, avg_win))

        # Check MFE left on table
        if mfe_analysis["winners_avg_mfe"] > 1.5:
            recommendations.append("Winners avg {:.1f}% MFE - consider wider profit targets".format(
                mfe_analysis["winners_avg_mfe"]))

        # Check FAILED_MOMENTUM
        if "FAILED_MOMENTUM" in by_exit and by_exit["FAILED_MOMENTUM"]["win_rate"] < 50:
            recommendations.append("FAILED_MOMENTUM has {:.0f}% win rate - tune 30s momentum check".format(
                by_exit["FAILED_MOMENTUM"]["win_rate"]))

        return {
            "success": True,
            "total_trades": len(trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(len(winners) / len(trades) * 100, 1),
            "total_pnl": round(sum(t.pnl for t in trades), 2),
            "by_exit_reason": dict(sorted(by_exit.items(), key=lambda x: x[1]["pnl"], reverse=True)),
            "by_symbol": dict(sorted(by_symbol.items(), key=lambda x: x[1]["pnl"], reverse=True)),
            "by_time_of_day": by_time,
            "mfe_mae_analysis": mfe_analysis,
            "hold_time_analysis": hold_analysis,
            "tuning_recommendations": recommendations,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


@router.get("/export/csv")
async def export_trades_csv():
    """Export all trades to CSV format"""
    from fastapi.responses import PlainTextResponse

    try:
        from .hft_scalper import get_hft_scalper

        scalper = get_hft_scalper()
        if not scalper:
            return PlainTextResponse("Error: Scalper not initialized")

        trades = [t for t in scalper.trades if t.status == "closed"]

        # CSV Header
        lines = ["symbol,entry_time,exit_time,entry_price,exit_price,shares,pnl,pnl_pct,hold_secs,exit_reason,max_gain_pct,max_dd_pct,spread_at_entry,day_change,volume,float_rotation,time_of_day"]

        for t in trades:
            st = t.secondary_triggers or {}
            line = ",".join([
                t.symbol,
                t.entry_time.isoformat() if t.entry_time else "",
                t.exit_time.isoformat() if t.exit_time else "",
                str(t.entry_price),
                str(t.exit_price),
                str(t.shares),
                str(round(t.pnl, 2)),
                str(round(t.pnl_percent, 2)),
                str(int(t.hold_time_seconds)),
                t.exit_reason or "",
                str(round(t.max_gain_percent, 2)),
                str(round(t.max_drawdown_percent, 2)),
                str(st.get("spread_at_entry", "")),
                str(st.get("day_change_at_entry", "")),
                str(st.get("volume_at_entry", "")),
                str(st.get("float_rotation", "")),
                st.get("time_of_day", "")
            ])
            lines.append(line)

        return PlainTextResponse("\n".join(lines), media_type="text/csv")

    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        return PlainTextResponse(f"Error: {str(e)}")


@router.post("/reset")
async def reset_paper_account(starting_balance: float = 1000.0):
    """Reset paper account to starting balance"""
    try:
        from .paper_account_metrics import get_paper_metrics
        from .hft_scalper import get_hft_scalper

        metrics = get_paper_metrics()
        metrics.reset(starting_balance)

        # Also reset scalper stats
        scalper = get_hft_scalper()
        if scalper:
            scalper.reset_daily_stats()

        return {
            "success": True,
            "message": f"Paper account reset to ${starting_balance:.2f}",
            "starting_balance": starting_balance
        }

    except Exception as e:
        logger.error(f"Error resetting account: {e}")
        return {"success": False, "error": str(e)}
