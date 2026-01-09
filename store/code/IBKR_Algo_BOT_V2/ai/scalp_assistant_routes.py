"""
Scalp Assistant API Routes
==========================
REST API endpoints for HFT Scalp Assistant.

Endpoints:
- GET  /api/scalp/status           - Status and monitored positions
- POST /api/scalp/start            - Start monitoring
- POST /api/scalp/stop             - Stop monitoring
- POST /api/scalp/takeover/{sym}   - Enable AI takeover for position
- POST /api/scalp/release/{sym}    - Disable AI takeover (manual control)
- GET  /api/scalp/config           - Get configuration
- POST /api/scalp/config           - Update configuration
- GET  /api/scalp/positions        - Get all positions
- GET  /api/scalp/history          - Get exit history
- POST /api/scalp/sync             - Force sync positions from broker
- POST /api/scalp/paper/{mode}     - Enable/disable paper mode
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scalp", tags=["Scalp Assistant"])


# Import scalp assistant
try:
    from ai.scalp_assistant import ScalpAssistant, get_scalp_assistant

    HAS_SCALP_ASSISTANT = True
except ImportError as e:
    logger.warning(f"Scalp Assistant not available: {e}")
    HAS_SCALP_ASSISTANT = False


class ScalpConfigUpdate(BaseModel):
    """Config update request"""

    stop_loss_pct: Optional[float] = None
    profit_target_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    max_hold_seconds: Optional[int] = None
    reversal_red_candles: Optional[int] = None
    velocity_death_pct: Optional[float] = None
    min_gain_for_reversal_exit: Optional[float] = None
    max_spread_pct: Optional[float] = None
    enabled: Optional[bool] = None
    paper_mode: Optional[bool] = None
    check_interval_ms: Optional[int] = None


@router.get("/status")
async def get_scalp_status():
    """Get scalp assistant status and monitored positions"""
    if not HAS_SCALP_ASSISTANT:
        raise HTTPException(status_code=503, detail="Scalp Assistant not available")

    assistant = get_scalp_assistant()
    return assistant.get_status()


@router.post("/start")
async def start_scalp_assistant():
    """Start the scalp assistant monitoring"""
    if not HAS_SCALP_ASSISTANT:
        raise HTTPException(status_code=503, detail="Scalp Assistant not available")

    assistant = get_scalp_assistant()
    assistant.start()
    return {
        "success": True,
        "message": "Scalp Assistant started",
        "running": assistant.running,
    }


@router.post("/stop")
async def stop_scalp_assistant():
    """Stop the scalp assistant monitoring"""
    if not HAS_SCALP_ASSISTANT:
        raise HTTPException(status_code=503, detail="Scalp Assistant not available")

    assistant = get_scalp_assistant()
    assistant.stop()
    return {
        "success": True,
        "message": "Scalp Assistant stopped",
        "running": assistant.running,
    }


@router.post("/takeover/{symbol}")
async def enable_ai_takeover(symbol: str):
    """Enable AI takeover for a position"""
    if not HAS_SCALP_ASSISTANT:
        raise HTTPException(status_code=503, detail="Scalp Assistant not available")

    assistant = get_scalp_assistant()
    symbol = symbol.upper()

    # Sync positions first
    assistant.sync_positions_from_broker()

    if symbol not in assistant.positions:
        raise HTTPException(status_code=404, detail=f"No position found for {symbol}")

    success = assistant.enable_ai_takeover(symbol)
    if success:
        return {
            "success": True,
            "message": f"AI takeover enabled for {symbol}",
            "position": assistant.positions[symbol].to_dict(),
        }
    else:
        raise HTTPException(
            status_code=400, detail=f"Failed to enable AI takeover for {symbol}"
        )


@router.post("/release/{symbol}")
async def disable_ai_takeover(symbol: str):
    """Disable AI takeover - return to manual control"""
    if not HAS_SCALP_ASSISTANT:
        raise HTTPException(status_code=503, detail="Scalp Assistant not available")

    assistant = get_scalp_assistant()
    symbol = symbol.upper()

    success = assistant.disable_ai_takeover(symbol)
    if success:
        return {
            "success": True,
            "message": f"Manual control restored for {symbol}",
            "position": assistant.positions.get(symbol, {}),
        }
    else:
        return {"success": False, "message": f"Position {symbol} not found"}


@router.get("/config")
async def get_scalp_config():
    """Get scalp assistant configuration"""
    if not HAS_SCALP_ASSISTANT:
        raise HTTPException(status_code=503, detail="Scalp Assistant not available")

    assistant = get_scalp_assistant()
    return assistant.config.to_dict()


@router.post("/config")
async def update_scalp_config(config: ScalpConfigUpdate):
    """Update scalp assistant configuration"""
    if not HAS_SCALP_ASSISTANT:
        raise HTTPException(status_code=503, detail="Scalp Assistant not available")

    assistant = get_scalp_assistant()

    # Only update non-None values
    updates = {k: v for k, v in config.dict().items() if v is not None}

    if updates:
        new_config = assistant.update_config(**updates)
        return {"success": True, "config": new_config}
    else:
        return {"success": False, "message": "No config values provided"}


@router.get("/positions")
async def get_scalp_positions():
    """Get all positions (both AI-monitored and manual)"""
    if not HAS_SCALP_ASSISTANT:
        raise HTTPException(status_code=503, detail="Scalp Assistant not available")

    assistant = get_scalp_assistant()
    assistant.sync_positions_from_broker()

    return {
        "total": len(assistant.positions),
        "positions": [p.to_dict() for p in assistant.positions.values()],
    }


@router.get("/history")
async def get_scalp_history(limit: int = 50):
    """Get exit history"""
    if not HAS_SCALP_ASSISTANT:
        raise HTTPException(status_code=503, detail="Scalp Assistant not available")

    assistant = get_scalp_assistant()

    # Load from file
    history = []
    try:
        import json

        if assistant._history_path().exists():
            with open(assistant._history_path(), "r") as f:
                history = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load history: {e}")

    return {"total": len(history), "exits": history[-limit:], "stats": assistant.stats}


@router.post("/sync")
async def sync_positions():
    """Force sync positions from broker"""
    if not HAS_SCALP_ASSISTANT:
        raise HTTPException(status_code=503, detail="Scalp Assistant not available")

    assistant = get_scalp_assistant()
    assistant.sync_positions_from_broker()

    return {
        "success": True,
        "positions": len(assistant.positions),
        "position_list": [p.to_dict() for p in assistant.positions.values()],
    }


@router.post("/paper/{mode}")
async def set_paper_mode(mode: str):
    """Enable/disable paper mode (true/false)"""
    if not HAS_SCALP_ASSISTANT:
        raise HTTPException(status_code=503, detail="Scalp Assistant not available")

    assistant = get_scalp_assistant()

    paper = mode.lower() in ("true", "1", "yes", "on")
    assistant.config.paper_mode = paper
    assistant._save_config()

    return {
        "success": True,
        "paper_mode": assistant.config.paper_mode,
        "message": f"Paper mode {'enabled' if paper else 'DISABLED - LIVE TRADING'}",
    }


@router.get("/stats")
async def get_scalp_stats():
    """Get performance statistics"""
    if not HAS_SCALP_ASSISTANT:
        raise HTTPException(status_code=503, detail="Scalp Assistant not available")

    assistant = get_scalp_assistant()

    stats = assistant.stats.copy()
    stats["win_rate"] = (
        (stats["winning_exits"] / stats["total_exits"] * 100)
        if stats["total_exits"] > 0
        else 0
    )
    stats["avg_pnl"] = (
        (stats["total_pnl"] / stats["total_exits"]) if stats["total_exits"] > 0 else 0
    )

    return stats


@router.post("/reset-stats")
async def reset_stats():
    """Reset performance statistics"""
    if not HAS_SCALP_ASSISTANT:
        raise HTTPException(status_code=503, detail="Scalp Assistant not available")

    assistant = get_scalp_assistant()
    assistant.stats = {
        "total_exits": 0,
        "stop_loss_exits": 0,
        "trailing_stop_exits": 0,
        "reversal_exits": 0,
        "timeout_exits": 0,
        "manual_exits": 0,
        "total_pnl": 0.0,
        "winning_exits": 0,
        "losing_exits": 0,
    }
    assistant._save_config()

    return {"success": True, "message": "Stats reset", "stats": assistant.stats}
