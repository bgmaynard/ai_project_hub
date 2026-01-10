"""
Orchestrator API Routes
========================
Read-only endpoints for the Orchestration & Observer Console.
Aggregates data from existing modules - no new logic.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import pytz

from fastapi import APIRouter, Query, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/orchestrator", tags=["Orchestrator"])


# =========================================================================
# GATING DECISIONS TRACKING
# =========================================================================

# In-memory store for gating decisions (for observability)
_gating_decisions: List[Dict[str, Any]] = []
MAX_DECISIONS = 500


def record_gating_decision(
    symbol: str,
    strategy: str,
    decision: str,
    primary_reason: str,
    secondary_factors: List[str] = None,
    chronos_regime: str = "UNKNOWN",
    chronos_confidence: float = 0.0,
    ats_state: str = "UNKNOWN",
    micro_override_applied: bool = False
):
    """Record a gating decision for observability"""
    global _gating_decisions

    record = {
        "timestamp": datetime.now(pytz.timezone('US/Eastern')).isoformat(),
        "symbol": symbol,
        "strategy": strategy,
        "decision": decision,
        "primary_reason": primary_reason,
        "secondary_factors": secondary_factors or [],
        "chronos_regime": chronos_regime,
        "chronos_confidence": chronos_confidence,
        "ats_state": ats_state,
        "micro_override_applied": micro_override_applied
    }

    _gating_decisions.insert(0, record)

    # Trim to max size
    if len(_gating_decisions) > MAX_DECISIONS:
        _gating_decisions = _gating_decisions[:MAX_DECISIONS]

    logger.debug(f"Orchestrator: Recorded gating decision for {symbol}: {decision}")


@router.get("/gating/decisions")
async def get_gating_decisions(
    limit: int = Query(default=50, le=500),
    symbol: str = Query(default=None),
    decision: str = Query(default=None),
    strategy: str = Query(default=None)
):
    """Get gating decisions with optional filters"""
    decisions = _gating_decisions

    # Apply filters
    if symbol:
        decisions = [d for d in decisions if d["symbol"] == symbol]
    if decision:
        decisions = [d for d in decisions if d["decision"] == decision]
    if strategy:
        decisions = [d for d in decisions if d["strategy"] == strategy]

    return {
        "decisions": decisions[:limit],
        "total": len(decisions),
        "filtered": len(_gating_decisions) != len(decisions)
    }


@router.get("/gating/stats")
async def get_gating_stats():
    """Get gating statistics"""
    if not _gating_decisions:
        return {
            "total": 0,
            "approved": 0,
            "vetoed": 0,
            "approval_rate": 0,
            "top_veto_reasons": []
        }

    approved = sum(1 for d in _gating_decisions if d["decision"] == "APPROVED")
    vetoed = sum(1 for d in _gating_decisions if d["decision"] == "VETOED")

    # Count veto reasons
    reason_counts: Dict[str, int] = {}
    for d in _gating_decisions:
        if d["decision"] == "VETOED":
            reason = d["primary_reason"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total": len(_gating_decisions),
        "approved": approved,
        "vetoed": vetoed,
        "approval_rate": round(approved / len(_gating_decisions) * 100, 1) if _gating_decisions else 0,
        "top_veto_reasons": [{"reason": r, "count": c} for r, c in top_reasons]
    }


# =========================================================================
# SYMBOL LIFECYCLE TRACKING
# =========================================================================

# In-memory store for symbol lifecycles
_symbol_lifecycles: Dict[str, Dict[str, Any]] = {}


def record_lifecycle_event(
    symbol: str,
    event_type: str,
    reason: str = None,
    metrics: Dict[str, Any] = None
):
    """Record a lifecycle event for a symbol"""
    global _symbol_lifecycles

    now = datetime.now(pytz.timezone('US/Eastern'))

    if symbol not in _symbol_lifecycles:
        _symbol_lifecycles[symbol] = {
            "symbol": symbol,
            "events": [],
            "current_state": event_type,
            "first_seen": now.isoformat(),
            "last_event": now.isoformat()
        }

    event = {
        "timestamp": now.isoformat(),
        "event_type": event_type,
        "reason": reason,
        "metrics": metrics or {}
    }

    _symbol_lifecycles[symbol]["events"].append(event)
    _symbol_lifecycles[symbol]["current_state"] = event_type
    _symbol_lifecycles[symbol]["last_event"] = now.isoformat()

    # Trim to last 100 events per symbol
    if len(_symbol_lifecycles[symbol]["events"]) > 100:
        _symbol_lifecycles[symbol]["events"] = _symbol_lifecycles[symbol]["events"][-100:]

    logger.debug(f"Orchestrator: Lifecycle event for {symbol}: {event_type}")


@router.get("/symbol/{symbol}/lifecycle")
async def get_symbol_lifecycle(symbol: str):
    """Get complete lifecycle for a symbol"""
    if symbol not in _symbol_lifecycles:
        raise HTTPException(status_code=404, detail=f"No lifecycle data for {symbol}")

    return _symbol_lifecycles[symbol]


@router.get("/symbols/active")
async def get_active_symbols():
    """Get list of symbols with lifecycle data"""
    return {
        "symbols": list(_symbol_lifecycles.keys()),
        "count": len(_symbol_lifecycles)
    }


def reset_lifecycles_daily():
    """Reset lifecycles at start of new trading day"""
    global _symbol_lifecycles
    _symbol_lifecycles.clear()
    logger.info("Orchestrator: Symbol lifecycles reset for new day")


# =========================================================================
# DAILY REPORT GENERATION
# =========================================================================

def generate_daily_report(date: str = None) -> Dict[str, Any]:
    """Generate a daily report from current data"""
    et_tz = pytz.timezone('US/Eastern')
    now = datetime.now(et_tz)
    report_date = date or now.strftime('%Y-%m-%d')

    # Try to load from file first
    report_path = Path(f"reports/daily/{report_date}_baseline_report.json")
    if report_path.exists():
        try:
            with open(report_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load daily report: {e}")

    # Generate from current data
    try:
        from ai.funnel_metrics import get_funnel_metrics
        funnel = get_funnel_metrics()
        funnel_status = funnel.get_funnel_status()
    except Exception:
        funnel_status = {
            "stages": {
                "1_found_by_scanners": 0,
                "2_injected_symbols": 0,
                "6_gating_attempts": 0,
                "7_gating_approvals": 0,
                "9_trade_executions": 0
            },
            "conversion_rates": {"overall_funnel_pct": 0}
        }

    # Get market condition
    try:
        from ai.market_condition_evaluator import get_condition_evaluator
        evaluator = get_condition_evaluator()
        condition = evaluator.get_current_condition()
        market_context = {
            "market_breadth": condition.get("score", 50),
            "small_cap_participation": condition.get("small_cap_score", 50),
            "gap_continuation_rate": condition.get("gap_rate", 50),
            "dominant_regime": condition.get("regime", "NEUTRAL")
        }
    except Exception:
        market_context = {
            "market_breadth": 50,
            "small_cap_participation": 50,
            "gap_continuation_rate": 50,
            "dominant_regime": "NEUTRAL"
        }

    # Get phase transitions
    try:
        from ai.phase_evaluator import get_phase_evaluator
        evaluator = get_phase_evaluator()
        transitions = evaluator.get_transition_history()
    except Exception:
        transitions = []

    # Get trade summary
    try:
        from ai.hft_scalper import get_hft_scalper
        scalper = get_hft_scalper()
        stats = scalper.get_statistics()
        trade_summary = {
            "total_trades": stats.get("total_trades", 0),
            "wins": stats.get("wins", 0),
            "losses": stats.get("losses", 0),
            "win_rate": stats.get("win_rate", 0),
            "total_pnl": stats.get("total_pnl", 0),
            "avg_win": stats.get("avg_win", 0),
            "avg_loss": stats.get("avg_loss", 0)
        }
    except Exception:
        trade_summary = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0
        }

    # Build veto reasons from gating decisions
    veto_counts: Dict[str, int] = {}
    total_vetoes = 0
    for d in _gating_decisions:
        if d["decision"] == "VETOED":
            reason = d["primary_reason"]
            veto_counts[reason] = veto_counts.get(reason, 0) + 1
            total_vetoes += 1

    top_veto_reasons = [
        {
            "reason": reason,
            "count": count,
            "percentage": round(count / total_vetoes * 100, 1) if total_vetoes > 0 else 0
        }
        for reason, count in sorted(veto_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    return {
        "date": report_date,
        "generated_at": now.isoformat(),
        "market_context": market_context,
        "funnel_summary": {
            "total_discovered": funnel_status["stages"].get("1_found_by_scanners", 0),
            "total_injected": funnel_status["stages"].get("2_injected_symbols", 0),
            "total_gated": funnel_status["stages"].get("6_gating_attempts", 0),
            "total_approved": funnel_status["stages"].get("7_gating_approvals", 0),
            "total_executed": funnel_status["stages"].get("9_trade_executions", 0),
            "overall_conversion": funnel_status["conversion_rates"].get("overall_funnel_pct", 0)
        },
        "strategy_activity": [],  # Populated from strategy tracking if available
        "phase_transitions": [
            {
                "from_phase": t.get("from_phase", "UNKNOWN"),
                "to_phase": t.get("to_phase", "UNKNOWN"),
                "timestamp": t.get("timestamp", ""),
                "reason": t.get("reason", "Time-based")
            }
            for t in transitions
        ] if transitions else [],
        "top_veto_reasons": top_veto_reasons,
        "trade_summary": trade_summary
    }


@router.get("/reports/daily/{date}")
async def get_daily_report(date: str):
    """Get daily baseline report"""
    try:
        return generate_daily_report(date)
    except Exception as e:
        logger.error(f"Failed to generate daily report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/dates")
async def get_available_report_dates():
    """Get list of available report dates"""
    dates = []

    # Check reports directory
    reports_dir = Path("reports/daily")
    if reports_dir.exists():
        for f in reports_dir.glob("*_baseline_report.json"):
            date = f.stem.replace("_baseline_report", "")
            dates.append(date)

    # Always include today
    today = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
    if today not in dates:
        dates.append(today)

    return {"dates": sorted(dates, reverse=True)}


# =========================================================================
# SYSTEM STATUS AGGREGATION
# =========================================================================

@router.get("/status")
async def get_orchestrator_status():
    """Get aggregated system status for orchestrator"""
    try:
        # Get phase info
        from ai.market_phases import get_phase_manager
        phase_manager = get_phase_manager()
        current_phase = phase_manager.current_phase.value if phase_manager.current_phase else "CLOSED"
    except Exception:
        current_phase = "CLOSED"

    try:
        # Get baseline info
        from ai.baseline_profiles import get_baseline_manager
        baseline = get_baseline_manager()
        current_profile = baseline.current_profile.value
    except Exception:
        current_profile = "NEUTRAL"

    try:
        # Get scout info
        from ai.momentum_scout import get_momentum_scout
        scout = get_momentum_scout()
        scout_enabled = scout.is_enabled()
    except Exception:
        scout_enabled = False

    return {
        "timestamp": datetime.now(pytz.timezone('US/Eastern')).isoformat(),
        "current_phase": current_phase,
        "baseline_profile": current_profile,
        "scout_enabled": scout_enabled,
        "gating_decisions_count": len(_gating_decisions),
        "symbols_tracked": len(_symbol_lifecycles)
    }


# =========================================================================
# CLEANUP FUNCTIONS
# =========================================================================

def daily_reset():
    """Reset orchestrator data for new trading day"""
    global _gating_decisions
    _gating_decisions.clear()
    reset_lifecycles_daily()
    logger.info("Orchestrator: Daily reset completed")
