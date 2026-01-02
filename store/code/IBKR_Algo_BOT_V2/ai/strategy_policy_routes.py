"""
Strategy Policy Engine API Routes

Endpoints:
- GET  /api/strategy/policy - Get all policies
- POST /api/strategy/policy/evaluate - Evaluate and adjust policy
- POST /api/strategy/policy/enable/{strategy_id} - Enable strategy
- POST /api/strategy/policy/disable/{strategy_id} - Disable strategy
- POST /api/strategy/policy/reset/{strategy_id} - Reset to default
- GET  /api/strategy/policy/audit - Get audit log
- POST /api/strategy/policy/config - Update config
- POST /api/strategy/policy/record-trade - Record trade result
- POST /api/strategy/policy/record-veto - Record veto
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from datetime import datetime
from .strategy_policy_engine import (
    get_strategy_policy_engine,
    PerformanceMetrics
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategy/policy", tags=["Strategy Policy"])


class EvaluateRequest(BaseModel):
    strategy_id: str
    chronos_context: Optional[Dict[str, Any]] = None
    force_reevaluate: bool = False


class TradeResultRequest(BaseModel):
    strategy_id: str
    is_win: bool
    pnl: float


class VetoRecordRequest(BaseModel):
    strategy_id: str
    veto_reason: str


class ConfigUpdateRequest(BaseModel):
    config: Dict[str, Any]


@router.get("")
async def get_all_policies():
    """
    Get all current strategy policies.

    Returns:
    - policies: Dict of strategy_id -> policy state
    - config: Current policy engine configuration
    - veto_counts: Accumulated veto counts per strategy
    """
    engine = get_strategy_policy_engine()
    return engine.get_all_policies()


# IMPORTANT: Specific routes MUST come BEFORE parameterized routes
@router.get("/audit")
async def get_audit_log(
    strategy_id: Optional[str] = Query(None, description="Filter by strategy"),
    limit: int = Query(100, description="Max entries to return")
):
    """
    Get gating decisions audit log for frontend display.

    Returns decisions in format expected by AI Control Center:
    - timestamp: ISO timestamp
    - approved: boolean
    - symbol: stock symbol
    - reasons: array of reason strings
    """
    from ai.signal_gating_engine import get_gating_engine

    try:
        gating = get_gating_engine()
        history = gating.veto_logger.veto_history

        # Convert GateResult objects to frontend-expected format
        decisions = []
        for result in history[-limit:]:
            decisions.append({
                "timestamp": result.gated_at if hasattr(result, 'gated_at') else datetime.utcnow().isoformat(),
                "approved": result.approved,
                "symbol": result.symbol,
                "type": "entry",
                "reasons": [result.veto_details] if result.veto_details else [result.veto_reason or "No reason"],
                "veto_reason": result.veto_reason,
                "regime": result.current_regime,
                "confidence": result.chronos_confidence
            })

        # Return newest first
        decisions.reverse()
        return decisions

    except Exception as e:
        logger.error(f"Error fetching gating decisions: {e}")
        # Return empty array - frontend handles this
        return []


@router.get("/audit-log")
async def get_policy_audit_log(
    strategy_id: Optional[str] = Query(None, description="Filter by strategy"),
    limit: int = Query(100, description="Max entries to return")
):
    """Get policy change audit log (enable/disable/adjust events)"""
    engine = get_strategy_policy_engine()
    return {
        "entries": engine.get_audit_log(strategy_id=strategy_id, limit=limit)
    }


@router.get("/{strategy_id}")
async def get_strategy_policy(strategy_id: str):
    """Get policy for a specific strategy"""
    engine = get_strategy_policy_engine()
    policy = engine.get_policy(strategy_id)
    return {
        "strategy_id": strategy_id,
        "enabled": policy.enabled,
        "confidence_threshold_override": policy.confidence_threshold_override,
        "position_size_multiplier": policy.position_size_multiplier,
        "reason": policy.reason,
        "last_updated": policy.last_updated
    }


@router.post("/evaluate")
async def evaluate_policy(request: EvaluateRequest):
    """
    Evaluate and potentially adjust policy for a strategy.

    This is the main entry point for policy evaluation.
    Called by Signal Gating Engine before trade execution.

    Inputs:
    - strategy_id: The strategy to evaluate
    - chronos_context: Current market regime context (optional)
    - force_reevaluate: Force re-evaluation even if recently evaluated

    Returns:
    - enabled: Whether strategy is enabled
    - confidence_override: Confidence threshold override (or null for default)
    - position_multiplier: Position size multiplier
    - actions_taken: List of policy actions taken
    - reasons: List of reasons for actions
    """
    engine = get_strategy_policy_engine()

    result = engine.evaluate(
        strategy_id=request.strategy_id,
        chronos_context=request.chronos_context,
        force_reevaluate=request.force_reevaluate
    )

    return {
        "strategy_id": result.strategy_id,
        "enabled": result.enabled,
        "confidence_override": result.confidence_override,
        "position_multiplier": result.position_multiplier,
        "actions_taken": result.actions_taken,
        "reasons": result.reasons
    }


@router.post("/enable/{strategy_id}")
async def enable_strategy(strategy_id: str, reason: str = "manual"):
    """Manually enable a strategy"""
    engine = get_strategy_policy_engine()
    engine.enable_strategy(strategy_id, reason)
    return {
        "status": "enabled",
        "strategy_id": strategy_id,
        "reason": reason
    }


@router.post("/disable/{strategy_id}")
async def disable_strategy(strategy_id: str, reason: str = "manual"):
    """Manually disable a strategy"""
    engine = get_strategy_policy_engine()
    engine.disable_strategy(strategy_id, reason)
    return {
        "status": "disabled",
        "strategy_id": strategy_id,
        "reason": reason
    }


@router.post("/reset/{strategy_id}")
async def reset_strategy(strategy_id: str, reason: str = "manual reset"):
    """Reset a strategy to default policy"""
    engine = get_strategy_policy_engine()
    engine.reset_strategy(strategy_id, reason)
    return {
        "status": "reset",
        "strategy_id": strategy_id,
        "reason": reason
    }


@router.post("/config")
async def update_config(request: ConfigUpdateRequest):
    """Update policy engine configuration"""
    engine = get_strategy_policy_engine()
    new_config = engine.update_config(request.config)
    return {
        "status": "updated",
        "config": new_config
    }


@router.post("/record-trade")
async def record_trade_result(request: TradeResultRequest):
    """
    Record a trade result for performance tracking.

    This updates rolling metrics that influence policy decisions.
    """
    engine = get_strategy_policy_engine()
    engine.record_trade_result(
        strategy_id=request.strategy_id,
        is_win=request.is_win,
        pnl=request.pnl
    )
    return {
        "status": "recorded",
        "strategy_id": request.strategy_id,
        "is_win": request.is_win,
        "pnl": request.pnl
    }


@router.post("/record-veto")
async def record_veto(request: VetoRecordRequest):
    """
    Record a veto from Signal Gating Engine.

    Used for veto pattern detection.
    """
    engine = get_strategy_policy_engine()
    engine.record_veto(
        strategy_id=request.strategy_id,
        veto_reason=request.veto_reason
    )
    return {
        "status": "recorded",
        "strategy_id": request.strategy_id,
        "veto_reason": request.veto_reason
    }


@router.get("/metrics/{strategy_id}")
async def get_performance_metrics(strategy_id: str):
    """Get rolling performance metrics for a strategy"""
    engine = get_strategy_policy_engine()
    metrics = engine.performance.get(strategy_id)

    if not metrics:
        return {
            "strategy_id": strategy_id,
            "message": "No performance data recorded yet"
        }

    return {
        "strategy_id": strategy_id,
        "win_rate": metrics.win_rate,
        "total_trades": metrics.total_trades,
        "winning_trades": metrics.winning_trades,
        "losing_trades": metrics.losing_trades,
        "total_pnl": metrics.total_pnl,
        "consecutive_losses": metrics.consecutive_losses,
        "last_5_trades": metrics.last_5_trades
    }
