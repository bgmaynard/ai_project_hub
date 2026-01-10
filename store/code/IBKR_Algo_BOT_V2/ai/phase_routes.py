"""
Market Phase API Routes (Task O - Observability)
=================================================
Exposes market phase system via REST API for visibility and control.

Endpoints:
- GET /api/phase/status - Full phase manager status
- GET /api/phase/current - Current phase only
- GET /api/phase/strategies - Enabled/disabled strategies
- GET /api/phase/gate - Strategy gate status
- GET /api/phase/history - Phase change history
- GET /api/phase/suppressed - Suppressed signals
- POST /api/phase/evaluate - Force phase evaluation
- POST /api/phase/set - Manually set phase (admin)
- GET /api/phase/dashboard - Dashboard summary
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/phase", tags=["Market Phase"])


class PhaseSetRequest(BaseModel):
    phase: str
    reason: str = "Manual override"
    lock_minutes: int = 15


class StrategyOverrideRequest(BaseModel):
    strategies: List[str] = []
    enable: bool = True


# =============================================================================
# STATUS ENDPOINTS
# =============================================================================

@router.get("/status")
async def get_phase_status() -> Dict:
    """
    Get full market phase manager status.

    Returns current phase, enabled strategies, trade limits, and more.
    """
    try:
        from ai.market_phases import get_phase_manager
        from ai.phase_evaluator import get_phase_evaluator
        from ai.strategy_gate import get_strategy_gate

        manager = get_phase_manager()
        evaluator = get_phase_evaluator()
        gate = get_strategy_gate()

        manager_status = manager.get_status()
        evaluator_status = evaluator.get_status()
        gate_status = gate.get_status()

        return {
            "phase_manager": manager_status,
            "phase_evaluator": evaluator_status,
            "strategy_gate": gate_status,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting phase status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current")
async def get_current_phase() -> Dict:
    """Get current market phase with description."""
    try:
        from ai.market_phases import get_phase_manager

        manager = get_phase_manager()
        config = manager.get_phase_config()

        lock_remaining = 0
        if manager.phase_locked_until and datetime.now() < manager.phase_locked_until:
            lock_remaining = (manager.phase_locked_until - datetime.now()).total_seconds()

        return {
            "phase": manager.current_phase.value if manager.current_phase else None,
            "description": config.description,
            "expected_volatility": config.expected_volatility,
            "expected_volume": config.expected_volume,
            "trend_reliability": config.trend_reliability,
            "time_window": f"{config.start_time} - {config.end_time} ET",
            "phase_locked": lock_remaining > 0,
            "lock_remaining_seconds": lock_remaining,
            "last_change_reason": manager.phase_change_reason,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting current phase: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def get_strategy_status() -> Dict:
    """Get enabled and disabled strategies for current phase."""
    try:
        from ai.market_phases import get_phase_manager, TradingStrategy
        from ai.strategy_gate import get_strategy_gate

        manager = get_phase_manager()
        gate = get_strategy_gate()
        config = manager.get_phase_config()

        all_strategies = [s.value for s in TradingStrategy]
        enabled = config.allowed_strategies
        disabled = [s for s in all_strategies if s not in enabled]

        return {
            "current_phase": manager.current_phase.value if manager.current_phase else None,
            "enabled_strategies": enabled,
            "disabled_strategies": disabled,
            "override_enabled": gate.override_enabled,
            "override_strategies": list(gate.override_strategies),
            "position_size_multiplier": config.position_size_multiplier,
            "stop_loss_multiplier": config.stop_loss_multiplier,
            "default_aggressiveness": config.default_aggressiveness,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting strategy status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gate")
async def get_gate_status() -> Dict:
    """Get strategy gate status and suppression counts."""
    try:
        from ai.strategy_gate import get_strategy_gate

        gate = get_strategy_gate()
        return gate.get_status()

    except Exception as e:
        logger.error(f"Error getting gate status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_phase_history(
    limit: int = Query(default=20, ge=1, le=100)
) -> Dict:
    """Get phase change history."""
    try:
        from ai.market_phases import get_phase_manager

        manager = get_phase_manager()
        history = manager.phase_history[-limit:] if manager.phase_history else []

        return {
            "count": len(history),
            "history": list(reversed(history)),  # Most recent first
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting phase history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suppressed")
async def get_suppressed_signals(
    limit: int = Query(default=50, ge=1, le=200)
) -> Dict:
    """Get signals that were suppressed due to phase/strategy mismatch."""
    try:
        from ai.market_phases import get_phase_manager
        from ai.strategy_gate import get_strategy_gate

        manager = get_phase_manager()
        gate = get_strategy_gate()

        signals = manager.suppressed_signals[-limit:] if manager.suppressed_signals else []

        return {
            "count": len(signals),
            "suppressed_by_strategy": gate.suppressed_count,
            "signals": list(reversed(signals)),  # Most recent first
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting suppressed signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# EVALUATION ENDPOINTS
# =============================================================================

@router.get("/evaluation")
async def get_last_evaluation() -> Dict:
    """Get the last phase evaluation result."""
    try:
        from ai.phase_evaluator import get_phase_evaluator

        evaluator = get_phase_evaluator()

        if evaluator.last_evaluation:
            return {
                "has_evaluation": True,
                "evaluation": evaluator.last_evaluation.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "has_evaluation": False,
                "message": "No evaluation performed yet",
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error getting evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
async def force_evaluation() -> Dict:
    """Force a phase evaluation and apply if appropriate."""
    try:
        from ai.phase_evaluator import evaluate_and_apply_phase

        result = await evaluate_and_apply_phase(force=True)

        logger.info(f"Phase evaluation forced: {result.get('new_phase')}")

        return {
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error forcing evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ADMIN ENDPOINTS
# =============================================================================

@router.post("/set")
async def set_phase(request: PhaseSetRequest) -> Dict:
    """
    Manually set the market phase (admin override).

    Use with caution - bypasses automatic phase evaluation.
    """
    try:
        from ai.market_phases import get_phase_manager, MarketPhase

        manager = get_phase_manager()

        # Validate phase
        try:
            phase = MarketPhase(request.phase.upper())
        except ValueError:
            valid_phases = [p.value for p in MarketPhase]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid phase: {request.phase}. Valid phases: {valid_phases}"
            )

        old_phase = manager.current_phase
        success = manager.set_phase(phase, f"ADMIN: {request.reason}", request.lock_minutes)

        if success:
            logger.warning(f"Phase manually set: {old_phase.value if old_phase else None} -> {phase.value}")
            return {
                "success": True,
                "old_phase": old_phase.value if old_phase else None,
                "new_phase": phase.value,
                "reason": request.reason,
                "lock_minutes": request.lock_minutes,
                "timestamp": datetime.now().isoformat()
            }
        else:
            lock_remaining = 0
            if manager.phase_locked_until:
                lock_remaining = (manager.phase_locked_until - datetime.now()).total_seconds()
            return {
                "success": False,
                "reason": "Phase is locked",
                "lock_remaining_seconds": lock_remaining,
                "timestamp": datetime.now().isoformat()
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting phase: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gate/override")
async def set_strategy_override(request: StrategyOverrideRequest) -> Dict:
    """Enable or disable strategy override (bypass gate)."""
    try:
        from ai.strategy_gate import get_strategy_gate

        gate = get_strategy_gate()

        if request.enable:
            gate.enable_override(request.strategies if request.strategies else None)
            action = "enabled"
        else:
            gate.disable_override(request.strategies if request.strategies else None)
            action = "disabled"

        return {
            "success": True,
            "action": f"Override {action}",
            "strategies": request.strategies if request.strategies else "ALL",
            "current_override_enabled": gate.override_enabled,
            "current_override_strategies": list(gate.override_strategies),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error setting override: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DASHBOARD ENDPOINT
# =============================================================================

@router.get("/dashboard")
async def get_phase_dashboard() -> Dict:
    """
    Get dashboard summary for UI display.

    Single endpoint with all phase/strategy info needed for UI.
    """
    try:
        from ai.market_phases import get_phase_manager, TradingStrategy
        from ai.phase_evaluator import get_phase_evaluator
        from ai.strategy_gate import get_strategy_gate
        import pytz

        manager = get_phase_manager()
        evaluator = get_phase_evaluator()
        gate = get_strategy_gate()
        config = manager.get_phase_config()

        # Current time in ET
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)

        # Lock remaining
        lock_remaining = 0
        if manager.phase_locked_until and datetime.now() < manager.phase_locked_until:
            lock_remaining = (manager.phase_locked_until - datetime.now()).total_seconds()

        # Build strategy list with enabled status
        all_strategies = []
        for s in TradingStrategy:
            all_strategies.append({
                "name": s.value,
                "enabled": s.value in config.allowed_strategies,
                "overridden": s.value in gate.override_strategies
            })

        # Last evaluation info
        last_eval = None
        if evaluator.last_evaluation:
            last_eval = {
                "phase": evaluator.last_evaluation.phase,
                "confidence": evaluator.last_evaluation.confidence,
                "reasons": evaluator.last_evaluation.reasons[:3],
                "age_seconds": (datetime.now() - evaluator.last_evaluation.timestamp).total_seconds()
            }

        return {
            "current_time_et": now_et.strftime("%H:%M:%S ET"),
            "current_phase": {
                "name": manager.current_phase.value if manager.current_phase else "UNKNOWN",
                "description": config.description,
                "time_window": f"{config.start_time} - {config.end_time} ET",
                "volatility": config.expected_volatility,
                "volume": config.expected_volume
            },
            "phase_lock": {
                "locked": lock_remaining > 0,
                "remaining_seconds": lock_remaining,
                "reason": manager.phase_change_reason
            },
            "trading_limits": {
                "trades_this_phase": manager.trades_this_phase,
                "max_trades": config.max_trades_per_phase,
                "position_size_mult": config.position_size_multiplier,
                "stop_loss_mult": config.stop_loss_multiplier,
                "can_trade": gate.can_trade()
            },
            "strategies": all_strategies,
            "gate_override": {
                "enabled": gate.override_enabled,
                "strategies": list(gate.override_strategies)
            },
            "suppressed_count": len(manager.suppressed_signals),
            "last_evaluation": last_eval,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting phase dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PHASE CONFIG ENDPOINTS
# =============================================================================

@router.get("/config")
async def get_phase_configs() -> Dict:
    """Get all phase configurations."""
    try:
        from ai.market_phases import get_phase_manager

        manager = get_phase_manager()

        configs = {}
        for name, config in manager.phases.items():
            configs[name] = config.to_dict()

        return {
            "phases": configs,
            "current_phase": manager.current_phase.value if manager.current_phase else None,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting phase configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config/{phase_name}")
async def get_phase_config(phase_name: str) -> Dict:
    """Get configuration for a specific phase."""
    try:
        from ai.market_phases import get_phase_manager, MarketPhase

        manager = get_phase_manager()

        # Validate phase
        try:
            phase = MarketPhase(phase_name.upper())
        except ValueError:
            valid_phases = [p.value for p in MarketPhase]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid phase: {phase_name}. Valid phases: {valid_phases}"
            )

        config = manager.get_phase_config(phase)

        return {
            "phase": phase.value,
            "config": config.to_dict(),
            "is_current": manager.current_phase == phase,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting phase config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# STRATEGY CHECK ENDPOINT
# =============================================================================

@router.get("/check/{strategy_name}")
async def check_strategy_enabled(strategy_name: str) -> Dict:
    """Check if a specific strategy is enabled for the current phase."""
    try:
        from ai.strategy_gate import get_strategy_gate
        from ai.market_phases import get_phase_manager

        gate = get_strategy_gate()
        manager = get_phase_manager()
        config = manager.get_phase_config()

        enabled = gate.is_strategy_enabled(strategy_name)
        in_allowed = strategy_name.upper() in [s.upper() for s in config.allowed_strategies]
        overridden = strategy_name.upper() in [s.upper() for s in gate.override_strategies]

        return {
            "strategy": strategy_name,
            "enabled": enabled,
            "in_allowed_list": in_allowed,
            "overridden": overridden,
            "global_override": gate.override_enabled,
            "current_phase": manager.current_phase.value if manager.current_phase else None,
            "allowed_strategies": config.allowed_strategies,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error checking strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))
