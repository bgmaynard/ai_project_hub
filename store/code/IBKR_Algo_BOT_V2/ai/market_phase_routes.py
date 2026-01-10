"""
Market Phase Router API Routes

REST endpoints for market phase strategy routing.
"""

import logging
from fastapi import APIRouter, Query
from typing import Optional

from .market_phase_router import (
    get_market_phase_router,
    MarketPhase,
    PHASE_CONFIGS,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/phase", tags=["Market Phase"])


@router.get("/status")
async def get_phase_status():
    """Get current market phase and trading parameters"""
    try:
        phase_router = get_market_phase_router()
        return phase_router.get_status()
    except Exception as e:
        logger.error(f"Error getting phase status: {e}")
        return {"error": str(e)}


@router.get("/current")
async def get_current_phase():
    """Get just the current phase name"""
    try:
        phase_router = get_market_phase_router()
        phase = phase_router.get_current_phase()
        config = phase_router.get_phase_config(phase)
        return {
            "phase": phase.value,
            "strategy_bias": config.strategy_bias.value,
            "trading_allowed": config.trading_allowed,
        }
    except Exception as e:
        logger.error(f"Error getting current phase: {e}")
        return {"error": str(e)}


@router.get("/can-trade")
async def check_can_trade(
    setup_grade: str = Query("C", description="Setup grade (A, B, or C)")
):
    """Check if trading is allowed in current phase"""
    try:
        phase_router = get_market_phase_router()
        allowed, reason = phase_router.can_trade(setup_grade)
        status = phase_router.get_status()
        return {
            "allowed": allowed,
            "reason": reason,
            "phase": status["current_phase"],
            "strategy_bias": status["strategy_bias"],
            "min_grade_required": status["min_setup_grade"],
            "trades_remaining": status["max_trades_per_hour"] - status["trades_this_hour"],
        }
    except Exception as e:
        logger.error(f"Error checking can_trade: {e}")
        return {"error": str(e)}


@router.get("/params")
async def get_trade_params():
    """Get trading parameters for current phase"""
    try:
        phase_router = get_market_phase_router()
        params = phase_router.get_trade_params()
        status = phase_router.get_status()
        return {
            "phase": status["current_phase"],
            "strategy_bias": status["strategy_bias"],
            **params,
        }
    except Exception as e:
        logger.error(f"Error getting trade params: {e}")
        return {"error": str(e)}


@router.get("/all")
async def get_all_phases():
    """Get configuration for all market phases"""
    try:
        phase_router = get_market_phase_router()
        return {
            "current_phase": phase_router.get_current_phase().value,
            "phases": phase_router.get_all_phases(),
        }
    except Exception as e:
        logger.error(f"Error getting all phases: {e}")
        return {"error": str(e)}


@router.get("/schedule")
async def get_phase_schedule():
    """Get the daily phase schedule"""
    return {
        "timezone": "US/Eastern",
        "schedule": [
            {"phase": "PREMARKET_EARLY", "start": "04:00", "end": "07:00", "bias": "Warrior Momentum"},
            {"phase": "PREMARKET_LATE", "start": "07:00", "end": "09:30", "bias": "ATS / Pullbacks"},
            {"phase": "OPEN", "start": "09:30", "end": "09:45", "bias": "Breakout / Halt"},
            {"phase": "POST_OPEN", "start": "09:45", "end": "11:00", "bias": "Scalper"},
            {"phase": "MIDDAY", "start": "11:00", "end": "14:30", "bias": "Light Only"},
            {"phase": "POWER_HOUR", "start": "15:00", "end": "16:00", "bias": "Continuation"},
            {"phase": "AFTER_HOURS", "start": "16:00", "end": "20:00", "bias": "Watch Only"},
            {"phase": "CLOSED", "start": "20:00", "end": "04:00", "bias": "No Trade"},
        ],
    }


@router.post("/record-trade")
async def record_trade():
    """Record a trade for hourly limit tracking"""
    try:
        phase_router = get_market_phase_router()
        phase_router.record_trade()
        status = phase_router.get_status()
        return {
            "success": True,
            "trades_this_hour": status["trades_this_hour"],
            "max_trades_per_hour": status["max_trades_per_hour"],
            "remaining": status["max_trades_per_hour"] - status["trades_this_hour"],
        }
    except Exception as e:
        logger.error(f"Error recording trade: {e}")
        return {"success": False, "error": str(e)}
