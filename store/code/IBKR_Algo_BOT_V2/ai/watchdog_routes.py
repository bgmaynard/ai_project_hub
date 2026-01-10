"""
Trading Window Watchdog API Routes
==================================
REST endpoints for watchdog monitoring and control.
"""

import logging
from fastapi import APIRouter, Query
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/watchdog", tags=["Trading Window Watchdog"])

try:
    from ai.trading_window_watchdog import (
        get_trading_watchdog,
        start_watchdog,
        stop_watchdog
    )
    HAS_WATCHDOG = True
except ImportError as e:
    logger.warning(f"Watchdog not available: {e}")
    HAS_WATCHDOG = False


@router.get("/status")
async def get_watchdog_status():
    """Get watchdog status and recent events"""
    if not HAS_WATCHDOG:
        return {"error": "Watchdog not available", "available": False}

    watchdog = get_trading_watchdog()
    return watchdog.get_status()


@router.get("/window")
async def get_current_window():
    """Get current trading window information"""
    if not HAS_WATCHDOG:
        return {"error": "Watchdog not available"}

    watchdog = get_trading_watchdog()
    return watchdog.get_current_window()


@router.post("/start")
async def start_watchdog_service():
    """Start the watchdog service"""
    if not HAS_WATCHDOG:
        return {"error": "Watchdog not available", "success": False}

    watchdog = get_trading_watchdog()

    if watchdog.running:
        return {"success": True, "message": "Watchdog already running"}

    await watchdog.start()
    return {
        "success": True,
        "message": "Watchdog started",
        "status": watchdog.get_status()
    }


@router.post("/stop")
async def stop_watchdog_service():
    """Stop the watchdog service"""
    if not HAS_WATCHDOG:
        return {"error": "Watchdog not available", "success": False}

    watchdog = get_trading_watchdog()

    if not watchdog.running:
        return {"success": True, "message": "Watchdog not running"}

    await watchdog.stop()
    return {
        "success": True,
        "message": "Watchdog stopped"
    }


@router.post("/check")
async def manual_check():
    """Manually trigger a watchdog check"""
    if not HAS_WATCHDOG:
        return {"error": "Watchdog not available"}

    watchdog = get_trading_watchdog()
    result = await watchdog.check_and_intervene()
    return result


@router.get("/events")
async def get_events(limit: int = Query(20, description="Max events to return")):
    """Get recent watchdog events"""
    if not HAS_WATCHDOG:
        return {"error": "Watchdog not available", "events": []}

    watchdog = get_trading_watchdog()
    events = watchdog.events[-limit:]

    return {
        "count": len(events),
        "events": [
            {
                "event_type": e.event_type,
                "timestamp": e.timestamp,
                "details": e.details,
                "action_taken": e.action_taken
            }
            for e in events
        ]
    }


@router.post("/config")
async def update_config(
    auto_start: Optional[bool] = None,
    auto_enable: Optional[bool] = None,
    check_interval: Optional[int] = None,
    enable_premarket: Optional[bool] = None,
    enable_market: Optional[bool] = None,
    enable_afterhours: Optional[bool] = None
):
    """Update watchdog configuration"""
    if not HAS_WATCHDOG:
        return {"error": "Watchdog not available", "success": False}

    watchdog = get_trading_watchdog()

    if auto_start is not None:
        watchdog.config.auto_start_scalper = auto_start
    if auto_enable is not None:
        watchdog.config.auto_enable_scalper = auto_enable
    if check_interval is not None:
        watchdog.config.check_interval_seconds = max(10, min(300, check_interval))
    if enable_premarket is not None:
        watchdog.config.enable_in_premarket = enable_premarket
    if enable_market is not None:
        watchdog.config.enable_in_market = enable_market
    if enable_afterhours is not None:
        watchdog.config.enable_in_afterhours = enable_afterhours

    return {
        "success": True,
        "config": {
            "auto_start_scalper": watchdog.config.auto_start_scalper,
            "auto_enable_scalper": watchdog.config.auto_enable_scalper,
            "check_interval_seconds": watchdog.config.check_interval_seconds,
            "enable_in_premarket": watchdog.config.enable_in_premarket,
            "enable_in_market": watchdog.config.enable_in_market,
            "enable_in_afterhours": watchdog.config.enable_in_afterhours
        }
    }
