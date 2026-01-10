"""
Baseline Profile API Routes (Task J - Observability)
====================================================
REST endpoints for baseline profile system visibility and control.

Exposes:
- Current baseline profile
- Last evaluation reason
- Time until next allowed change
- Market condition metrics
"""

import logging
from fastapi import APIRouter, Query
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/baseline", tags=["Baseline Profiles"])


# ========== Profile Management ==========

@router.get("/status")
async def get_baseline_status():
    """
    Get complete baseline profile status.

    Returns current profile, lock status, and recent changes.
    """
    try:
        from ai.baseline_profiles import get_baseline_manager
        manager = get_baseline_manager()
        return manager.get_status()
    except Exception as e:
        logger.error(f"Error getting baseline status: {e}")
        return {"error": str(e)}


@router.get("/profile")
async def get_current_profile():
    """Get the currently active profile and its parameters"""
    try:
        from ai.baseline_profiles import get_baseline_manager
        manager = get_baseline_manager()
        params = manager.get_current_params()
        return {
            "profile": manager.current_profile.value,
            "description": params.description,
            "parameters": params.to_dict(),
            "change_reason": manager.change_reason,
            "last_change": manager.last_change_time.isoformat() if manager.last_change_time else None
        }
    except Exception as e:
        logger.error(f"Error getting profile: {e}")
        return {"error": str(e)}


@router.get("/profiles")
async def list_all_profiles():
    """List all available profiles and their configurations"""
    try:
        from ai.baseline_profiles import get_baseline_manager
        manager = get_baseline_manager()
        return {
            "current": manager.current_profile.value,
            "profiles": {k: v.to_dict() for k, v in manager.profiles.items()}
        }
    except Exception as e:
        logger.error(f"Error listing profiles: {e}")
        return {"error": str(e)}


@router.post("/profile/{profile_name}")
async def set_profile(
    profile_name: str,
    reason: str = Query("Manual override", description="Reason for profile change"),
    lock_minutes: int = Query(30, description="Lock duration in minutes")
):
    """
    Manually set the active profile.

    This is an admin override - use with caution.
    """
    try:
        from ai.baseline_profiles import get_baseline_manager, BaselineProfile

        # Validate profile name
        try:
            profile = BaselineProfile(profile_name.upper())
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid profile: {profile_name}. Valid: CONSERVATIVE, NEUTRAL, AGGRESSIVE"
            }

        manager = get_baseline_manager()
        success = manager.set_profile(profile, f"MANUAL: {reason}", lock_minutes)

        return {
            "success": success,
            "profile": manager.current_profile.value,
            "locked_until": manager.lock_until.isoformat() if manager.lock_until else None,
            "message": "Profile set" if success else "Profile locked, change blocked"
        }
    except Exception as e:
        logger.error(f"Error setting profile: {e}")
        return {"success": False, "error": str(e)}


@router.post("/unlock")
async def force_unlock():
    """Force unlock the profile (admin override)"""
    try:
        from ai.baseline_profiles import get_baseline_manager
        manager = get_baseline_manager()
        success = manager.force_unlock()
        return {
            "success": success,
            "message": "Profile unlocked" if success else "No lock to remove"
        }
    except Exception as e:
        logger.error(f"Error unlocking profile: {e}")
        return {"success": False, "error": str(e)}


@router.get("/param/{param_name}")
async def get_profile_param(param_name: str):
    """Get a specific parameter from the current profile"""
    try:
        from ai.baseline_profiles import get_profile_param
        value = get_profile_param(param_name)
        if value is None:
            return {"error": f"Parameter {param_name} not found"}
        return {"param": param_name, "value": value}
    except Exception as e:
        logger.error(f"Error getting param: {e}")
        return {"error": str(e)}


# ========== Market Condition ==========

@router.get("/market-condition")
async def get_market_condition():
    """
    Get current market condition evaluation.

    Returns: WEAK | MIXED | STRONG with confidence and metrics.
    """
    try:
        from ai.market_condition_evaluator import get_market_evaluator
        evaluator = get_market_evaluator()
        result = await evaluator.evaluate()
        return result.to_dict()
    except Exception as e:
        logger.error(f"Error getting market condition: {e}")
        return {"error": str(e)}


@router.post("/market-condition/evaluate")
async def force_evaluate_market():
    """Force a fresh market condition evaluation (bypasses cache)"""
    try:
        from ai.market_condition_evaluator import get_market_evaluator
        evaluator = get_market_evaluator()
        result = await evaluator.evaluate(force=True)
        return result.to_dict()
    except Exception as e:
        logger.error(f"Error evaluating market: {e}")
        return {"error": str(e)}


@router.get("/market-condition/status")
async def get_evaluator_status():
    """Get market condition evaluator status"""
    try:
        from ai.market_condition_evaluator import get_market_evaluator
        evaluator = get_market_evaluator()
        return evaluator.get_status()
    except Exception as e:
        logger.error(f"Error getting evaluator status: {e}")
        return {"error": str(e)}


# ========== Profile Selector ==========

@router.get("/selector/status")
async def get_selector_status():
    """
    Get profile selector status.

    Shows current checkpoint, next checkpoint, and recent selections.
    """
    try:
        from ai.profile_selector import get_profile_selector
        selector = get_profile_selector()
        return selector.get_status()
    except Exception as e:
        logger.error(f"Error getting selector status: {e}")
        return {"error": str(e)}


@router.post("/selector/start")
async def start_selector():
    """Start the automatic profile selector"""
    try:
        from ai.profile_selector import get_profile_selector
        selector = get_profile_selector()
        selector.start()
        return {"success": True, "message": "Profile selector started"}
    except Exception as e:
        logger.error(f"Error starting selector: {e}")
        return {"success": False, "error": str(e)}


@router.post("/selector/stop")
async def stop_selector():
    """Stop the automatic profile selector"""
    try:
        from ai.profile_selector import get_profile_selector
        selector = get_profile_selector()
        selector.stop()
        return {"success": True, "message": "Profile selector stopped"}
    except Exception as e:
        logger.error(f"Error stopping selector: {e}")
        return {"success": False, "error": str(e)}


@router.post("/selector/evaluate")
async def force_evaluate_and_select():
    """Force an immediate profile evaluation and selection"""
    try:
        from ai.profile_selector import get_profile_selector
        selector = get_profile_selector()
        result = await selector.evaluate_and_select(force=True)
        return result
    except Exception as e:
        logger.error(f"Error evaluating: {e}")
        return {"error": str(e)}


# ========== Combined Dashboard View ==========

@router.get("/dashboard")
async def get_dashboard_data():
    """
    Get all baseline data for dashboard display.

    Single endpoint for UI to get everything needed.
    """
    try:
        from ai.baseline_profiles import get_baseline_manager
        from ai.market_condition_evaluator import get_market_evaluator
        from ai.profile_selector import get_profile_selector

        manager = get_baseline_manager()
        evaluator = get_market_evaluator()
        selector = get_profile_selector()

        params = manager.get_current_params()

        # Calculate lock remaining
        lock_remaining = 0
        if manager.lock_until:
            from datetime import datetime
            remaining = (manager.lock_until - datetime.now()).total_seconds()
            lock_remaining = max(0, remaining)

        return {
            # Current profile
            "current_profile": manager.current_profile.value,
            "profile_description": params.description,

            # Lock status
            "is_locked": lock_remaining > 0,
            "lock_remaining_seconds": lock_remaining,
            "next_change_allowed": manager.lock_until.isoformat() if manager.lock_until else None,

            # Last change
            "last_change_reason": manager.change_reason,
            "last_change_time": manager.last_change_time.isoformat() if manager.last_change_time else None,

            # Market condition (from cache)
            "market_condition": evaluator.last_evaluation.market_condition.value if evaluator.last_evaluation else "UNKNOWN",
            "market_confidence": evaluator.last_evaluation.confidence if evaluator.last_evaluation else 0,
            "market_reasons": evaluator.last_evaluation.reasons[:3] if evaluator.last_evaluation else [],

            # Selector status
            "selector_running": selector.running,
            "current_checkpoint": selector.last_checkpoint.value if selector.last_checkpoint else None,
            "selector_status": selector.get_status(),

            # Key parameters for display
            "key_params": {
                "rel_vol_floor": params.rel_vol_floor,
                "chronos_confidence": params.chronos_micro_confidence_min,
                "probe_enabled": params.probe_enabled,
                "scalper_aggressiveness": params.scalper_aggressiveness,
                "min_warrior_grade": params.min_warrior_grade
            }
        }
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        return {"error": str(e)}


# ========== History ==========

@router.get("/history")
async def get_profile_history(limit: int = Query(20, description="Max results")):
    """Get profile change history"""
    try:
        from ai.baseline_profiles import get_baseline_manager
        manager = get_baseline_manager()
        history = manager.profile_history[-limit:] if manager.profile_history else []
        return {
            "count": len(history),
            "history": history
        }
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return {"error": str(e)}


@router.get("/evaluation-history")
async def get_evaluation_history(limit: int = Query(20, description="Max results")):
    """Get market condition evaluation history"""
    try:
        from ai.market_condition_evaluator import get_market_evaluator
        evaluator = get_market_evaluator()
        history = [r.to_dict() for r in evaluator.evaluation_history[-limit:]]
        return {
            "count": len(history),
            "history": history
        }
    except Exception as e:
        logger.error(f"Error getting evaluation history: {e}")
        return {"error": str(e)}
