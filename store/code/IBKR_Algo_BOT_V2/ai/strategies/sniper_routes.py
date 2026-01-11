"""
ATS + 9 EMA Sniper Strategy API Routes
======================================
REST API endpoints for strategy control and monitoring.
"""

import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategy/sniper", tags=["ATS 9EMA Sniper"])


class ProcessRequest(BaseModel):
    """Request to process a symbol"""
    symbol: str
    price: float
    volume: float = 0
    high: Optional[float] = None
    low: Optional[float] = None
    vwap: Optional[float] = None
    rvol: Optional[float] = None
    change_percent: Optional[float] = None
    is_green: Optional[bool] = None


# ========================================
# STATUS & CONTROL ENDPOINTS
# ========================================

@router.get("/status")
async def get_strategy_status():
    """
    Get ATS + 9 EMA Sniper strategy status.

    Returns:
    - enabled: Whether strategy is enabled
    - active_window: Whether currently in active time window (9:40-11:00 AM ET)
    - market_session: Current market session
    - config: Strategy configuration
    - active_symbols: Symbols currently being tracked
    - today_stats: Today's performance statistics
    """
    try:
        from ai.strategies.ats_9ema_sniper import get_sniper_strategy
        strategy = get_sniper_strategy()
        return {"success": True, **strategy.get_status()}
    except Exception as e:
        logger.error(f"Strategy status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enable")
async def enable_strategy():
    """Enable the ATS + 9 EMA Sniper strategy"""
    try:
        from ai.strategies.ats_9ema_sniper import get_sniper_strategy
        strategy = get_sniper_strategy()
        strategy.enable()
        return {"success": True, "enabled": True}
    except Exception as e:
        logger.error(f"Enable strategy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disable")
async def disable_strategy():
    """Disable the ATS + 9 EMA Sniper strategy"""
    try:
        from ai.strategies.ats_9ema_sniper import get_sniper_strategy
        strategy = get_sniper_strategy()
        strategy.disable()
        return {"success": True, "enabled": False}
    except Exception as e:
        logger.error(f"Disable strategy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-daily")
async def reset_daily():
    """Reset daily counters and states"""
    try:
        from ai.strategies.ats_9ema_sniper import get_sniper_strategy
        strategy = get_sniper_strategy()
        strategy.reset_daily()
        return {"success": True, "message": "Daily reset complete"}
    except Exception as e:
        logger.error(f"Reset daily error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# SYMBOL PROCESSING ENDPOINTS
# ========================================

@router.post("/process")
async def process_symbol(request: ProcessRequest):
    """
    Process a symbol through the strategy.

    This is the main entry point for the strategy.
    Returns the action to take (QUALIFY, WAIT_PULLBACK, ENTER, EXIT, NO_ACTION).
    """
    try:
        from ai.strategies.ats_9ema_sniper import get_sniper_strategy
        strategy = get_sniper_strategy()

        if not strategy.is_enabled():
            return {
                "success": True,
                "action": "STRATEGY_DISABLED",
                "details": {"message": "Strategy is disabled"}
            }

        quote = {
            "price": request.price,
            "last": request.price,
            "volume": request.volume,
            "high": request.high or request.price,
            "low": request.low or request.price,
            "vwap": request.vwap or 0,
            "rvol": request.rvol or 1.0,
            "relative_volume": request.rvol or 1.0,
            "change_percent": request.change_percent or 0,
            "is_green": request.is_green if request.is_green is not None else True
        }

        result = strategy.process(request.symbol, quote)
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Process symbol error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/qualify/{symbol}")
async def check_qualification(symbol: str, price: float, vwap: float = 0,
                               rvol: float = 1.0, change_percent: float = 0):
    """
    Check ATS qualification for a symbol without advancing FSM state.

    Returns qualification result with reasons.
    """
    try:
        from ai.strategies.ats_9ema_sniper import get_sniper_strategy
        strategy = get_sniper_strategy()

        quote = {
            "price": price,
            "vwap": vwap,
            "rvol": rvol,
            "relative_volume": rvol,
            "change_percent": change_percent
        }

        result = strategy.check_ats_qualification(symbol, quote)
        return {"success": True, **result.to_dict()}
    except Exception as e:
        logger.error(f"Qualification check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pullback/{symbol}")
async def check_pullback(symbol: str, price: float):
    """
    Check pullback setup status for a symbol.

    Returns whether pullback requirements are met.
    """
    try:
        from ai.strategies.ats_9ema_sniper import get_sniper_strategy
        strategy = get_sniper_strategy()

        valid, reason = strategy.check_pullback_setup(symbol, price)
        return {
            "success": True,
            "symbol": symbol.upper(),
            "pullback_valid": valid,
            "reason": reason
        }
    except Exception as e:
        logger.error(f"Pullback check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# FSM STATE ENDPOINTS
# ========================================

@router.get("/fsm/states")
async def get_all_fsm_states():
    """Get FSM states for all symbols"""
    try:
        from ai.fsm.strategy_states import get_sniper_fsm
        fsm = get_sniper_fsm()
        return {
            "success": True,
            "states": fsm.get_all_states(),
            "active_symbols": fsm.get_active_symbols()
        }
    except Exception as e:
        logger.error(f"FSM states error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fsm/state/{symbol}")
async def get_symbol_fsm_state(symbol: str):
    """Get FSM state for a specific symbol"""
    try:
        from ai.fsm.strategy_states import get_sniper_fsm
        fsm = get_sniper_fsm()
        state = fsm.get_state(symbol)
        return {"success": True, **state.to_dict()}
    except Exception as e:
        logger.error(f"FSM state error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fsm/cancel/{symbol}")
async def cancel_symbol(symbol: str, reason: str = "Manual cancel"):
    """Cancel tracking for a symbol, return to IDLE"""
    try:
        from ai.fsm.strategy_states import get_sniper_fsm
        fsm = get_sniper_fsm()
        success = fsm.cancel(symbol, reason)
        return {"success": success, "symbol": symbol.upper(), "reason": reason}
    except Exception as e:
        logger.error(f"FSM cancel error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# EMA TRACKING ENDPOINTS
# ========================================

@router.get("/ema/states")
async def get_all_ema_states():
    """Get EMA states for all tracked symbols"""
    try:
        from ai.indicators.ema import get_ema_tracker
        tracker = get_ema_tracker()
        return {
            "success": True,
            "states": tracker.get_all_states()
        }
    except Exception as e:
        logger.error(f"EMA states error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ema/{symbol}")
async def get_symbol_ema(symbol: str):
    """Get EMA state for a specific symbol"""
    try:
        from ai.indicators.ema import get_ema_tracker
        tracker = get_ema_tracker()
        state = tracker.get_state(symbol)
        if state:
            return {"success": True, **state.to_dict()}
        else:
            return {"success": True, "symbol": symbol.upper(), "message": "No EMA data"}
    except Exception as e:
        logger.error(f"EMA state error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ema/update/{symbol}")
async def update_ema(symbol: str, price: float):
    """Update EMA with new price"""
    try:
        from ai.indicators.ema import get_ema_tracker
        tracker = get_ema_tracker()
        state = tracker.update(symbol, price)
        return {"success": True, **state.to_dict()}
    except Exception as e:
        logger.error(f"EMA update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# LOGGING & STATS ENDPOINTS
# ========================================

@router.get("/events/today")
async def get_today_events():
    """Get all trading events from today"""
    try:
        from ai.logging.events import get_event_logger
        logger_instance = get_event_logger()
        return {
            "success": True,
            "events": logger_instance.get_today_events(),
            "stats": logger_instance.get_today_stats()
        }
    except Exception as e:
        logger.error(f"Events error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/{symbol}")
async def get_symbol_events(symbol: str):
    """Get trading events for a specific symbol"""
    try:
        from ai.logging.events import get_event_logger
        logger_instance = get_event_logger()
        return {
            "success": True,
            "symbol": symbol.upper(),
            "events": logger_instance.get_events_by_symbol(symbol)
        }
    except Exception as e:
        logger.error(f"Symbol events error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_strategy_stats():
    """Get strategy performance statistics"""
    try:
        from ai.logging.events import get_event_logger
        logger_instance = get_event_logger()
        return {"success": True, **logger_instance.get_today_stats()}
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# TIME WINDOW ENDPOINTS
# ========================================

@router.get("/time-window")
async def get_time_window_status():
    """
    Check if strategy is within active time window.

    Active window: 9:40 AM - 11:00 AM ET
    """
    try:
        from ai.strategies.ats_9ema_sniper import get_sniper_strategy
        import pytz
        from datetime import datetime

        strategy = get_sniper_strategy()
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)

        return {
            "success": True,
            "active_window": strategy.is_active_window(),
            "market_session": strategy.get_market_session(),
            "current_time_et": now_et.strftime("%H:%M:%S"),
            "window_start": strategy.config.start_time_et.strftime("%H:%M"),
            "window_end": strategy.config.end_time_et.strftime("%H:%M")
        }
    except Exception as e:
        logger.error(f"Time window error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
