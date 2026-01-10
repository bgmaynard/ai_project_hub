"""
Scout Observability & Diagnostics API Routes (Task T)
======================================================
Exposes scout system via REST API for visibility and control.

Endpoints for:
- Scout attempts (count)
- Scout success rate
- Scout → ATS conversion rate
- Reason scouts were blocked
- Current exploration policy
- Funnel metrics integration
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scout", tags=["Momentum Scout"])


class ScoutConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    size_multiplier: Optional[float] = None
    stop_loss_percent: Optional[float] = None
    max_per_symbol_per_session: Optional[int] = None
    max_per_hour: Optional[int] = None
    cooldown_minutes: Optional[int] = None
    min_volume_acceleration: Optional[float] = None


class HandoffConfigUpdate(BaseModel):
    min_bars_for_handoff: Optional[int] = None
    min_gain_for_handoff_pct: Optional[float] = None
    continuation_volume_ratio: Optional[float] = None
    ats_smartzone_enabled: Optional[bool] = None
    scalper_handoff_enabled: Optional[bool] = None


# =============================================================================
# SCOUT STATUS ENDPOINTS
# =============================================================================

@router.get("/status")
async def get_scout_status() -> Dict:
    """Get full scout system status."""
    try:
        from ai.momentum_scout import get_momentum_scout
        from ai.scout_handoff import get_handoff_manager
        from ai.exploration_policy import get_exploration_manager

        scout = get_momentum_scout()
        handoff = get_handoff_manager()
        exploration = get_exploration_manager()

        return {
            "scout": scout.get_status(),
            "handoff": handoff.get_status(),
            "exploration": exploration.get_status(),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting scout status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_scout_stats() -> Dict:
    """Get scout statistics and conversion rates."""
    try:
        from ai.momentum_scout import get_momentum_scout
        from ai.scout_handoff import get_handoff_manager

        scout = get_momentum_scout()
        handoff = get_handoff_manager()

        scout_stats = scout.get_stats()
        handoff_stats = handoff.get_stats()

        return {
            "scout": scout_stats,
            "handoff": handoff_stats,
            "summary": {
                "total_attempts": scout_stats["scout_attempts"],
                "confirmation_rate": scout_stats["confirmation_rate"],
                "conversion_rate": scout_stats["conversion_rate"],
                "handoff_success_rate": handoff_stats["success_rate"],
                "total_pnl": scout_stats["total_pnl"]
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting scout stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active")
async def get_active_scouts() -> Dict:
    """Get currently active scout positions."""
    try:
        from ai.momentum_scout import get_momentum_scout

        scout = get_momentum_scout()
        active = {
            s: e.to_dict() for s, e in scout.active_scouts.items()
        }

        return {
            "count": len(active),
            "scouts": active,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting active scouts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_scout_history(limit: int = Query(default=50, ge=1, le=200)) -> Dict:
    """Get scout history."""
    try:
        from ai.momentum_scout import get_momentum_scout

        scout = get_momentum_scout()

        return {
            "count": len(scout.scout_history),
            "history": scout.get_history(limit),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting scout history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SCOUT CONTROL ENDPOINTS
# =============================================================================

@router.post("/enable")
async def enable_scouts() -> Dict:
    """Enable scout mode."""
    try:
        from ai.momentum_scout import get_momentum_scout

        scout = get_momentum_scout()
        scout.enable()

        return {
            "success": True,
            "enabled": True,
            "message": "Scout mode enabled",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error enabling scouts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disable")
async def disable_scouts() -> Dict:
    """Disable scout mode."""
    try:
        from ai.momentum_scout import get_momentum_scout

        scout = get_momentum_scout()
        scout.disable()

        return {
            "success": True,
            "enabled": False,
            "message": "Scout mode disabled",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error disabling scouts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-session")
async def reset_scout_session() -> Dict:
    """Reset scout session counters."""
    try:
        from ai.momentum_scout import get_momentum_scout

        scout = get_momentum_scout()
        scout.reset_session()

        return {
            "success": True,
            "message": "Scout session reset",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error resetting scout session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config")
async def update_scout_config(updates: ScoutConfigUpdate) -> Dict:
    """Update scout configuration."""
    try:
        from ai.momentum_scout import get_momentum_scout

        scout = get_momentum_scout()
        update_dict = {k: v for k, v in updates.dict().items() if v is not None}
        config = scout.update_config(update_dict)

        return {
            "success": True,
            "config": config,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error updating scout config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_scout_config() -> Dict:
    """Get scout configuration."""
    try:
        from ai.momentum_scout import get_momentum_scout

        scout = get_momentum_scout()

        return {
            "config": scout.config.to_dict(),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting scout config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HANDOFF ENDPOINTS
# =============================================================================

@router.get("/handoffs/pending")
async def get_pending_handoffs() -> Dict:
    """Get pending handoff requests."""
    try:
        from ai.scout_handoff import get_handoff_manager

        manager = get_handoff_manager()

        return {
            "pending": manager.get_pending_handoffs(),
            "count": len(manager.pending_handoffs),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting pending handoffs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/handoffs/completed")
async def get_completed_handoffs(limit: int = Query(default=20, ge=1, le=100)) -> Dict:
    """Get completed handoffs."""
    try:
        from ai.scout_handoff import get_handoff_manager

        manager = get_handoff_manager()

        return {
            "completed": manager.get_completed_handoffs(limit),
            "total_count": len(manager.completed_handoffs),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting completed handoffs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/handoffs/process")
async def process_pending_handoffs() -> Dict:
    """Process all pending handoffs."""
    try:
        from ai.scout_handoff import get_handoff_manager

        manager = get_handoff_manager()
        results = await manager.process_pending_handoffs()

        return {
            "processed": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error processing handoffs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/handoffs/config")
async def update_handoff_config(updates: HandoffConfigUpdate) -> Dict:
    """Update handoff configuration."""
    try:
        from ai.scout_handoff import get_handoff_manager

        manager = get_handoff_manager()
        update_dict = {k: v for k, v in updates.dict().items() if v is not None}
        config = manager.update_config(update_dict)

        return {
            "success": True,
            "config": config,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error updating handoff config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# EXPLORATION POLICY ENDPOINTS
# =============================================================================

@router.get("/exploration/status")
async def get_exploration_status() -> Dict:
    """Get current exploration policy status."""
    try:
        from ai.exploration_policy import get_exploration_manager

        manager = get_exploration_manager()

        return manager.get_status()

    except Exception as e:
        logger.error(f"Error getting exploration status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exploration/policy")
async def get_current_exploration_policy() -> Dict:
    """Get current exploration policy."""
    try:
        from ai.exploration_policy import get_exploration_manager

        manager = get_exploration_manager()
        policy = manager.get_current_policy()

        return {
            "policy": policy.to_dict(),
            "allowed": manager.is_exploration_allowed()[0],
            "reason": manager.is_exploration_allowed()[1],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting exploration policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exploration/matrix")
async def get_exploration_matrix() -> Dict:
    """Get full exploration policy matrix."""
    try:
        from ai.exploration_policy import get_exploration_manager

        manager = get_exploration_manager()

        return {
            "matrix": manager.get_policy_matrix(),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting exploration matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/exploration/apply")
async def apply_exploration_policy() -> Dict:
    """Apply current exploration policy to scout mode."""
    try:
        from ai.exploration_policy import get_exploration_manager

        manager = get_exploration_manager()
        success = manager.apply_policy_to_scout()

        return {
            "success": success,
            "policy": manager.get_current_policy().to_dict(),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error applying exploration policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CHRONOS STRATEGY WEIGHTS ENDPOINTS
# =============================================================================

@router.get("/chronos/weights")
async def get_chronos_weights() -> Dict:
    """Get Chronos strategy weights."""
    try:
        from ai.chronos_strategy_weights import get_chronos_strategy_manager

        manager = get_chronos_strategy_manager()

        return manager.get_status()

    except Exception as e:
        logger.error(f"Error getting Chronos weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chronos/weights/{strategy}")
async def get_strategy_chronos_weight(strategy: str) -> Dict:
    """Get Chronos weight for a specific strategy."""
    try:
        from ai.chronos_strategy_weights import get_chronos_strategy_manager

        manager = get_chronos_strategy_manager()
        config = manager.get_config(strategy)

        if config is None:
            raise HTTPException(status_code=404, detail=f"No config for {strategy}")

        return {
            "strategy": strategy,
            "config": config.to_dict(),
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy weight: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# FUNNEL METRICS INTEGRATION
# =============================================================================

@router.get("/funnel")
async def get_scout_funnel_metrics() -> Dict:
    """Get scout-specific funnel metrics."""
    try:
        from ai.momentum_scout import get_momentum_scout
        from ai.scout_handoff import get_handoff_manager

        scout = get_momentum_scout()
        handoff = get_handoff_manager()

        scout_stats = scout.get_stats()
        handoff_stats = handoff.get_stats()

        # Build funnel
        funnel = {
            "stages": [
                {
                    "stage": "scout_attempts",
                    "count": scout_stats["scout_attempts"],
                    "description": "Scout entry attempts"
                },
                {
                    "stage": "scout_confirmed",
                    "count": scout_stats["scout_confirmed"],
                    "description": "Scouts that confirmed (held + gained)"
                },
                {
                    "stage": "scout_to_trade",
                    "count": scout_stats["scout_to_trade"],
                    "description": "Scouts handed to ATS/Scalper"
                },
                {
                    "stage": "handoff_completed",
                    "count": handoff_stats["handoffs_completed"],
                    "description": "Successful strategy handoffs"
                }
            ],
            "conversion_rates": {
                "attempt_to_confirm": scout_stats["confirmation_rate"],
                "confirm_to_handoff": scout_stats["conversion_rate"],
                "handoff_success": handoff_stats["success_rate"]
            },
            "blocked": {
                "scout_stopped": scout_stats["scout_stopped"],
                "scout_failed": scout_stats["scout_failed"],
                "handoff_rejected": handoff_stats["handoffs_rejected"]
            }
        }

        return {
            "funnel": funnel,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting scout funnel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/diagnostic")
async def get_scout_diagnostic() -> Dict:
    """Get diagnostic information about scout system."""
    try:
        from ai.momentum_scout import get_momentum_scout
        from ai.scout_handoff import get_handoff_manager
        from ai.exploration_policy import get_exploration_manager

        scout = get_momentum_scout()
        handoff = get_handoff_manager()
        exploration = get_exploration_manager()

        # Determine health
        stats = scout.get_stats()
        attempts = stats["scout_attempts"]
        confirmed = stats["scout_confirmed"]
        to_trade = stats["scout_to_trade"]

        health = "HEALTHY"
        issues = []
        recommendations = []

        # Check if scouts are happening
        if attempts == 0:
            if not scout.is_enabled():
                health = "DISABLED"
                issues.append("Scout mode is disabled")
                recommendations.append("Enable scouts: POST /api/scout/enable")
            else:
                policy = exploration.get_current_policy()
                if not policy.scouts_enabled:
                    health = "BLOCKED"
                    issues.append(f"Scouts blocked by exploration policy ({policy.phase})")
                    recommendations.append("Wait for favorable phase or adjust policy")
                else:
                    health = "WAITING"
                    issues.append("No scout triggers detected yet")
                    recommendations.append("Ensure watchlist has qualifying symbols")

        # Check confirmation rate
        elif confirmed == 0 and attempts > 5:
            health = "DEGRADED"
            issues.append(f"0 confirmations from {attempts} attempts")
            recommendations.append("Review scout stop loss settings")
            recommendations.append("Check if market conditions are too volatile")

        # Check handoff rate
        elif to_trade == 0 and confirmed > 3:
            health = "DEGRADED"
            issues.append(f"No handoffs from {confirmed} confirmed scouts")
            recommendations.append("Check handoff conditions")
            recommendations.append("Verify ATS/Scalper are running")

        return {
            "health": health,
            "issues": issues,
            "recommendations": recommendations,
            "stats": stats,
            "exploration_allowed": exploration.is_exploration_allowed()[0],
            "exploration_level": exploration.get_exploration_level().value,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting scout diagnostic: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DASHBOARD ENDPOINT
# =============================================================================

@router.get("/dashboard")
async def get_scout_dashboard() -> Dict:
    """Get dashboard summary for UI display."""
    try:
        from ai.momentum_scout import get_momentum_scout
        from ai.scout_handoff import get_handoff_manager
        from ai.exploration_policy import get_exploration_manager
        from ai.chronos_strategy_weights import get_chronos_strategy_manager

        scout = get_momentum_scout()
        handoff = get_handoff_manager()
        exploration = get_exploration_manager()
        chronos = get_chronos_strategy_manager()

        stats = scout.get_stats()
        policy = exploration.get_current_policy()

        return {
            "enabled": scout.is_enabled(),
            "exploration_level": policy.level.value,
            "exploration_allowed": exploration.is_exploration_allowed()[0],
            "current_phase": policy.phase,
            "active_scouts": len(scout.active_scouts),
            "active_symbols": list(scout.active_scouts.keys()),
            "hourly_count": len([t for t in scout.hourly_counts if (datetime.now() - t).total_seconds() < 3600]),
            "hourly_limit": policy.max_scouts_per_hour,
            "stats": {
                "attempts": stats["scout_attempts"],
                "confirmed": stats["scout_confirmed"],
                "stopped": stats["scout_stopped"],
                "to_trade": stats["scout_to_trade"],
                "confirmation_rate": f"{stats['confirmation_rate']:.1f}%",
                "conversion_rate": f"{stats['conversion_rate']:.1f}%",
                "total_pnl": f"${stats['total_pnl']:.2f}"
            },
            "pending_handoffs": len(handoff.pending_handoffs),
            "chronos_roles": chronos.get_status()["roles_summary"],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting scout dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))
