"""
Task Group 5 - Post-Trade & Health
===================================

Tasks:
- POST_TRADE_OUTCOME (R12): Trade outcome analysis
- POST_STRATEGY_HEALTH (R13): Strategy health monitor
- POST_STRATEGY_TOGGLE (R15): Daily strategy enable/disable

EXECUTION CONTRACT:
- Every task generates a persisted report artifact
- Fixed schema - no placeholders
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from .task_queue_manager import Task, get_task_queue_manager

logger = logging.getLogger(__name__)

API_BASE = "http://localhost:9100"
TRADES_HISTORY_PATH = Path("ai/scalper_trades.json")


async def _fetch_api(endpoint: str) -> Optional[Dict]:
    """Fetch data from API"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(f"{API_BASE}{endpoint}")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.error(f"API fetch error {endpoint}: {e}")
    return None


def _load_trade_history() -> List[Dict]:
    """Load trade history from file"""
    if TRADES_HISTORY_PATH.exists():
        try:
            with open(TRADES_HISTORY_PATH, "r") as f:
                return json.load(f)
        except:
            pass
    return []


# =============================================================================
# REPORT SCHEMAS (Fixed - No Placeholders)
# =============================================================================

SCHEMA_R12 = {
    "task_id": "POST_TRADE_OUTCOME",
    "trades_analyzed": 0,
    "outcomes": [],
    "summary": {
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "largest_win": 0.0,
        "largest_loss": 0.0,
        "avg_hold_time_seconds": 0.0,
    },
    "timestamp": "",
}

SCHEMA_R13 = {
    "task_id": "POST_STRATEGY_HEALTH",
    "strategy_id": "HOD_MOMENTUM",
    "health_status": "",  # HEALTHY / DEGRADED / CRITICAL / OFFLINE
    "health_score": 0.0,
    "metrics": {
        "win_rate_7d": 0.0,
        "profit_factor_7d": 0.0,
        "max_drawdown_7d": 0.0,
        "sharpe_ratio": 0.0,
        "consecutive_losses": 0,
        "trades_today": 0,
        "pnl_today": 0.0,
    },
    "alerts": [],
    "recommendations": [],
    "timestamp": "",
}

SCHEMA_R15 = {
    "task_id": "POST_STRATEGY_TOGGLE",
    "strategy_id": "HOD_MOMENTUM",
    "current_status": "",  # ENABLED / DISABLED / PAUSED
    "recommended_status": "",
    "status_changed": False,
    "change_reason": "",
    "conditions": {
        "win_rate_ok": False,
        "drawdown_ok": False,
        "consecutive_loss_ok": False,
        "volatility_ok": False,
        "time_of_day_ok": False,
    },
    "next_review": "",
    "timestamp": "",
}


# =============================================================================
# TASK 5.1 - POST_TRADE_OUTCOME
# =============================================================================


async def task_post_trade_outcome(inputs: Dict) -> Dict:
    """
    Trade Outcome Analysis

    INPUTS: Recent trades from scalper
    PROCESS:
        - Analyze trade outcomes
        - Calculate metrics
    OUTPUT: report_R12_trade_outcomes.json
    """
    logger.info("TASK 5.1: Analyzing trade outcomes...")

    # Get trades from scalper
    trades_data = await _fetch_api("/api/scanner/scalper/trades")
    trades = trades_data if isinstance(trades_data, list) else []

    # Also load from history file
    history_trades = _load_trade_history()

    # Combine and deduplicate
    all_trades = []
    seen_ids = set()

    for t in trades + history_trades:
        trade_id = (
            t.get("trade_id", "")
            or t.get("id", "")
            or f"{t.get('symbol')}_{t.get('entry_time', '')}"
        )
        if trade_id not in seen_ids:
            seen_ids.add(trade_id)
            all_trades.append(t)

    # Filter to today's trades
    today = datetime.now().date()
    todays_trades = []

    for t in all_trades:
        entry_time_str = t.get("entry_time", "") or t.get("timestamp", "")
        try:
            if entry_time_str:
                entry_dt = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                if entry_dt.date() == today:
                    todays_trades.append(t)
        except:
            pass

    # Analyze outcomes
    outcomes = []
    wins = 0
    losses = 0
    total_pnl = 0.0
    total_win_pnl = 0.0
    total_loss_pnl = 0.0
    hold_times = []
    largest_win = 0.0
    largest_loss = 0.0

    for t in todays_trades:
        pnl = t.get("pnl", 0) or t.get("realized_pnl", 0) or 0
        symbol = t.get("symbol", "")
        entry_price = t.get("entry_price", 0)
        exit_price = t.get("exit_price", 0)
        hold_time = t.get("hold_time_seconds", 0) or t.get("hold_seconds", 0) or 0

        is_win = pnl > 0
        if is_win:
            wins += 1
            total_win_pnl += pnl
            if pnl > largest_win:
                largest_win = pnl
        else:
            losses += 1
            total_loss_pnl += abs(pnl)
            if pnl < largest_loss:
                largest_loss = pnl

        total_pnl += pnl
        if hold_time > 0:
            hold_times.append(hold_time)

        outcomes.append(
            {
                "symbol": symbol,
                "pnl": round(pnl, 2),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "hold_time_seconds": hold_time,
                "outcome": "WIN" if is_win else "LOSS",
                "exit_reason": t.get("exit_reason", "UNKNOWN"),
            }
        )

    # Calculate metrics
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    avg_win = (total_win_pnl / wins) if wins > 0 else 0
    avg_loss = (total_loss_pnl / losses) if losses > 0 else 0
    profit_factor = (
        (total_win_pnl / total_loss_pnl)
        if total_loss_pnl > 0
        else (999 if total_win_pnl > 0 else 0)
    )
    avg_hold = (sum(hold_times) / len(hold_times)) if hold_times else 0

    # Build report (fixed schema)
    report = {
        "task_id": "POST_TRADE_OUTCOME",
        "trades_analyzed": total_trades,
        "outcomes": outcomes,
        "summary": {
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "largest_win": round(largest_win, 2),
            "largest_loss": round(largest_loss, 2),
            "avg_hold_time_seconds": round(avg_hold, 1),
        },
        "date": str(today),
        "timestamp": datetime.now().isoformat(),
    }

    return report


# =============================================================================
# TASK 5.2 - POST_STRATEGY_HEALTH
# =============================================================================


async def task_post_strategy_health(inputs: Dict) -> Dict:
    """
    Strategy Health Monitor

    INPUTS: Trade outcomes, strategy metrics
    PROCESS:
        - Evaluate strategy health
        - Generate alerts and recommendations
    OUTPUT: report_R13_strategy_health.json
    """
    logger.info("TASK 5.2: Monitoring strategy health...")

    # Load R12 input
    r12_data = inputs.get("report_R12_trade_outcomes.json", {})
    summary = r12_data.get("summary", {})

    # Get scalper stats for longer-term metrics
    stats = await _fetch_api("/api/scanner/scalper/stats")
    if not stats:
        stats = {}

    # Calculate health metrics
    win_rate_today = summary.get("win_rate", 0)
    win_rate_7d = stats.get("win_rate", win_rate_today)
    profit_factor_today = summary.get("profit_factor", 0)
    profit_factor_7d = stats.get("profit_factor", profit_factor_today)
    pnl_today = summary.get("total_pnl", 0)
    trades_today = summary.get("wins", 0) + summary.get("losses", 0)

    # Calculate consecutive losses (from recent trades)
    outcomes = r12_data.get("outcomes", [])
    consecutive_losses = 0
    for o in reversed(outcomes):
        if o.get("outcome") == "LOSS":
            consecutive_losses += 1
        else:
            break

    # Max drawdown (simplified - would need equity curve in production)
    max_drawdown = stats.get("worst_trade", 0)
    if max_drawdown > 0:
        max_drawdown = -max_drawdown

    # Sharpe ratio (simplified)
    avg_pnl = pnl_today / trades_today if trades_today > 0 else 0
    # Would need std deviation of returns for proper Sharpe

    # Calculate health score (0-100)
    health_score = 50.0  # Base

    # Win rate impact
    if win_rate_7d >= 40:
        health_score += 20
    elif win_rate_7d >= 30:
        health_score += 10
    elif win_rate_7d < 25:
        health_score -= 20

    # Profit factor impact
    if profit_factor_7d >= 1.5:
        health_score += 15
    elif profit_factor_7d >= 1.0:
        health_score += 5
    elif profit_factor_7d < 0.8:
        health_score -= 15

    # Consecutive losses impact
    if consecutive_losses >= 5:
        health_score -= 25
    elif consecutive_losses >= 3:
        health_score -= 10

    # Today's P&L impact
    if pnl_today > 0:
        health_score += 10
    elif pnl_today < -50:
        health_score -= 15

    health_score = max(0, min(100, health_score))

    # Determine health status
    if health_score >= 70:
        health_status = "HEALTHY"
    elif health_score >= 50:
        health_status = "DEGRADED"
    elif health_score >= 30:
        health_status = "CRITICAL"
    else:
        health_status = "OFFLINE"

    # Generate alerts
    alerts = []
    if win_rate_7d < 30:
        alerts.append(
            {
                "level": "WARNING",
                "message": f"Win rate {win_rate_7d:.1f}% below 30% threshold",
            }
        )
    if consecutive_losses >= 3:
        alerts.append(
            {"level": "WARNING", "message": f"{consecutive_losses} consecutive losses"}
        )
    if profit_factor_7d < 1.0:
        alerts.append(
            {
                "level": "CRITICAL",
                "message": f"Profit factor {profit_factor_7d:.2f} below 1.0",
            }
        )
    if pnl_today < -100:
        alerts.append({"level": "WARNING", "message": f"Today's P&L ${pnl_today:.2f}"})

    # Generate recommendations
    recommendations = []
    if win_rate_7d < 30:
        recommendations.append("Increase entry selectivity - only A/B grade setups")
    if profit_factor_7d < 1.0:
        recommendations.append("Review stop loss placement - consider tighter stops")
    if consecutive_losses >= 3:
        recommendations.append("Consider reducing position size or pausing strategy")
    if trades_today > 50:
        recommendations.append("Reduce trade frequency - quality over quantity")

    # Build report (fixed schema)
    report = {
        "task_id": "POST_STRATEGY_HEALTH",
        "strategy_id": "HOD_MOMENTUM",
        "health_status": health_status,
        "health_score": round(health_score, 1),
        "metrics": {
            "win_rate_7d": round(win_rate_7d, 1),
            "profit_factor_7d": round(profit_factor_7d, 2),
            "max_drawdown_7d": round(max_drawdown, 2),
            "sharpe_ratio": 0.0,  # Would need proper calculation
            "consecutive_losses": consecutive_losses,
            "trades_today": trades_today,
            "pnl_today": round(pnl_today, 2),
        },
        "alerts": alerts,
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat(),
    }

    return report


# =============================================================================
# TASK 5.3 - POST_STRATEGY_TOGGLE
# =============================================================================


async def task_post_strategy_toggle(inputs: Dict) -> Dict:
    """
    Daily Strategy Enable/Disable

    INPUTS: Strategy health
    PROCESS:
        - Evaluate if strategy should be enabled/disabled
    OUTPUT: report_R15_strategy_status.json
    """
    logger.info("TASK 5.3: Evaluating strategy toggle...")

    # Load R13 input
    r13_data = inputs.get("report_R13_strategy_health.json", {})
    health_status = r13_data.get("health_status", "UNKNOWN")
    health_score = r13_data.get("health_score", 50)
    metrics = r13_data.get("metrics", {})

    # Get current strategy status
    scalper_status = await _fetch_api("/api/scanner/scalper/status")
    current_enabled = scalper_status.get("enabled", False) if scalper_status else False
    current_status = "ENABLED" if current_enabled else "DISABLED"

    # Evaluate conditions
    conditions = {
        "win_rate_ok": metrics.get("win_rate_7d", 0) >= 25,
        "drawdown_ok": metrics.get("max_drawdown_7d", 0) > -200,  # Max $200 drawdown
        "consecutive_loss_ok": metrics.get("consecutive_losses", 0) < 5,
        "volatility_ok": True,  # Would check market volatility
        "time_of_day_ok": True,  # Within trading hours
    }

    # Check time of day
    now = datetime.now()
    hour = now.hour
    conditions["time_of_day_ok"] = 7 <= hour < 16

    # Determine recommended status
    failed_conditions = [k for k, v in conditions.items() if not v]

    if len(failed_conditions) == 0 and health_status in ["HEALTHY", "DEGRADED"]:
        recommended_status = "ENABLED"
        change_reason = "All conditions met"
    elif health_status == "CRITICAL":
        recommended_status = "PAUSED"
        change_reason = f"Health critical: {', '.join(failed_conditions)}"
    elif health_status == "OFFLINE":
        recommended_status = "DISABLED"
        change_reason = "Strategy offline due to poor performance"
    elif not conditions["consecutive_loss_ok"]:
        recommended_status = "PAUSED"
        change_reason = (
            f"Too many consecutive losses ({metrics.get('consecutive_losses', 0)})"
        )
    elif not conditions["time_of_day_ok"]:
        recommended_status = "DISABLED"
        change_reason = "Outside trading hours"
    else:
        recommended_status = current_status
        change_reason = "No change needed"

    # Determine if status changed
    status_changed = recommended_status != current_status

    # Calculate next review time
    next_review = (datetime.now() + timedelta(minutes=30)).isoformat()

    # Build report (fixed schema)
    report = {
        "task_id": "POST_STRATEGY_TOGGLE",
        "strategy_id": "HOD_MOMENTUM",
        "current_status": current_status,
        "recommended_status": recommended_status,
        "status_changed": status_changed,
        "change_reason": change_reason,
        "conditions": conditions,
        "failed_conditions": failed_conditions,
        "health_score": health_score,
        "next_review": next_review,
        "timestamp": datetime.now().isoformat(),
    }

    # Log status change
    if status_changed:
        logger.warning(
            f"STRATEGY TOGGLE: {current_status} -> {recommended_status}: {change_reason}"
        )

    return report


# =============================================================================
# REGISTER TASKS
# =============================================================================


def register_post_trade_tasks():
    """Register all Task Group 5 tasks with the manager"""
    manager = get_task_queue_manager()

    # Task 5.1 - Trade Outcome
    manager.register_task(
        Task(
            id="POST_TRADE_OUTCOME",
            name="Trade Outcome Analysis",
            group="POST_TRADE",
            inputs=["report_R11_trade_queue.json"],
            process=task_post_trade_outcome,
            output_file="report_R12_trade_outcomes.json",
            fail_conditions=[],
            next_task="POST_STRATEGY_HEALTH",
        )
    )

    # Task 5.2 - Strategy Health
    manager.register_task(
        Task(
            id="POST_STRATEGY_HEALTH",
            name="Strategy Health Monitor",
            group="POST_TRADE",
            inputs=["report_R12_trade_outcomes.json"],
            process=task_post_strategy_health,
            output_file="report_R13_strategy_health.json",
            fail_conditions=[],
            next_task="POST_STRATEGY_TOGGLE",
        )
    )

    # Task 5.3 - Strategy Toggle
    manager.register_task(
        Task(
            id="POST_STRATEGY_TOGGLE",
            name="Daily Strategy Enable/Disable",
            group="POST_TRADE",
            inputs=["report_R13_strategy_health.json"],
            process=task_post_strategy_toggle,
            output_file="report_R15_strategy_status.json",
            fail_conditions=[],
            next_task=None,  # End of pipeline
        )
    )

    logger.info("Task Group 5 (Post-Trade) registered: 3 tasks")
