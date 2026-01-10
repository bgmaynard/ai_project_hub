"""
Probe Entry API Routes
======================
REST endpoints for probe entry execution layer.
"""

import logging
from fastapi import APIRouter, Query
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/probe", tags=["Probe Entry Layer"])

try:
    from ai.probe_entry_manager import (
        get_probe_manager,
        check_probe_eligibility,
        ProbeTriggerType,
        ProbeState
    )
    HAS_PROBE_MANAGER = True
except ImportError as e:
    logger.warning(f"Probe entry manager not available: {e}")
    HAS_PROBE_MANAGER = False


@router.get("/status")
async def get_probe_status():
    """Get probe entry manager status and stats"""
    if not HAS_PROBE_MANAGER:
        return {"error": "Probe entry manager not available", "available": False}

    manager = get_probe_manager()
    return {
        "available": True,
        **manager.get_status()
    }


@router.post("/check")
async def check_eligibility(
    symbol: str = Query(..., description="Stock symbol"),
    ats_state: str = Query("ACTIVE", description="Current ATS state"),
    micro_confidence: float = Query(0.7, description="Chronos micro confidence"),
    macro_regime: str = Query("TRENDING_UP", description="Current macro regime"),
    current_price: float = Query(..., description="Current price"),
    current_volume: int = Query(0, description="Today's volume"),
    avg_volume: int = Query(0, description="Average daily volume"),
    vwap: float = Query(0, description="Current VWAP"),
    premarket_high: float = Query(0, description="Premarket high"),
    range_high_1m: float = Query(0, description="1-minute range high"),
    range_high_3m: float = Query(0, description="3-minute range high"),
    day_high: float = Query(0, description="Day's high")
):
    """
    Check if a probe entry is eligible.

    This is a TEST endpoint - does NOT execute, just checks eligibility.
    """
    if not HAS_PROBE_MANAGER:
        return {"error": "Probe entry manager not available"}

    eligible, reason, trigger_type = check_probe_eligibility(
        symbol=symbol.upper(),
        ats_state=ats_state,
        micro_confidence=micro_confidence,
        macro_regime=macro_regime,
        current_price=current_price,
        current_volume=current_volume,
        avg_volume=avg_volume,
        vwap=vwap,
        premarket_high=premarket_high,
        range_high_1m=range_high_1m,
        range_high_3m=range_high_3m,
        day_high=day_high
    )

    return {
        "symbol": symbol.upper(),
        "eligible": eligible,
        "reason": reason,
        "trigger_type": trigger_type.value if trigger_type else None,
        "inputs": {
            "ats_state": ats_state,
            "micro_confidence": micro_confidence,
            "macro_regime": macro_regime,
            "current_price": current_price,
            "current_volume": current_volume,
            "avg_volume": avg_volume,
            "vwap": vwap,
            "premarket_high": premarket_high,
            "range_high_1m": range_high_1m,
            "range_high_3m": range_high_3m,
            "day_high": day_high
        }
    }


@router.get("/active")
async def get_active_probes():
    """Get all active probe positions"""
    if not HAS_PROBE_MANAGER:
        return {"error": "Probe entry manager not available", "probes": []}

    manager = get_probe_manager()
    status = manager.get_status()

    return {
        "count": len(status.get("active_probes", [])),
        "probes": status.get("active_probes", [])
    }


@router.get("/active/{symbol}")
async def get_active_probe(symbol: str):
    """Get active probe for a specific symbol"""
    if not HAS_PROBE_MANAGER:
        return {"error": "Probe entry manager not available"}

    manager = get_probe_manager()
    probe = manager.get_active_probe(symbol.upper())

    if probe:
        return {
            "found": True,
            "probe": probe.to_dict()
        }
    else:
        return {
            "found": False,
            "message": f"No active probe for {symbol.upper()}"
        }


@router.get("/history")
async def get_probe_history(limit: int = Query(20, description="Max results")):
    """Get probe entry history"""
    if not HAS_PROBE_MANAGER:
        return {"error": "Probe entry manager not available", "history": []}

    manager = get_probe_manager()
    history = manager.get_history(limit)

    return {
        "count": len(history),
        "history": history
    }


@router.get("/cooldowns")
async def get_cooldowns():
    """Get symbols currently in cooldown"""
    if not HAS_PROBE_MANAGER:
        return {"error": "Probe entry manager not available", "cooldowns": {}}

    manager = get_probe_manager()
    status = manager.get_status()

    return {
        "count": len(status.get("cooldowns", {})),
        "cooldowns": status.get("cooldowns", {})
    }


@router.get("/config")
async def get_config():
    """Get probe configuration"""
    if not HAS_PROBE_MANAGER:
        return {"error": "Probe entry manager not available"}

    manager = get_probe_manager()
    status = manager.get_status()

    return status.get("config", {})


@router.post("/config")
async def update_config(
    enabled: Optional[bool] = None,
    size_multiplier: Optional[float] = None,
    stop_loss_percent: Optional[float] = None,
    min_micro_confidence: Optional[float] = None,
    max_probes_per_hour: Optional[int] = None,
    cooldown_minutes: Optional[int] = None
):
    """Update probe configuration"""
    if not HAS_PROBE_MANAGER:
        return {"error": "Probe entry manager not available", "success": False}

    manager = get_probe_manager()
    manager.update_config(
        enabled=enabled,
        size_multiplier=size_multiplier,
        stop_loss_percent=stop_loss_percent,
        min_micro_confidence=min_micro_confidence,
        max_probes_per_hour=max_probes_per_hour,
        cooldown_minutes=cooldown_minutes
    )

    return {
        "success": True,
        "message": "Configuration updated",
        "config": manager.get_status().get("config", {})
    }


@router.post("/enable")
async def enable_probes():
    """Enable probe entries"""
    if not HAS_PROBE_MANAGER:
        return {"error": "Probe entry manager not available", "success": False}

    manager = get_probe_manager()
    manager.update_config(enabled=True)

    return {
        "success": True,
        "message": "Probe entries enabled",
        "enabled": True
    }


@router.post("/disable")
async def disable_probes():
    """Disable probe entries"""
    if not HAS_PROBE_MANAGER:
        return {"error": "Probe entry manager not available", "success": False}

    manager = get_probe_manager()
    manager.update_config(enabled=False)

    return {
        "success": True,
        "message": "Probe entries disabled",
        "enabled": False
    }


@router.post("/close/{symbol}")
async def close_probe(
    symbol: str,
    exit_price: float = Query(..., description="Exit price"),
    reason: str = Query("MANUAL", description="Exit reason")
):
    """Manually close an active probe"""
    if not HAS_PROBE_MANAGER:
        return {"error": "Probe entry manager not available", "success": False}

    manager = get_probe_manager()
    probe = manager.close_probe(symbol.upper(), exit_price, reason)

    if probe:
        return {
            "success": True,
            "message": f"Probe for {symbol.upper()} closed",
            "probe": probe.to_dict()
        }
    else:
        return {
            "success": False,
            "message": f"No active probe for {symbol.upper()}"
        }


@router.get("/stats")
async def get_stats():
    """Get probe entry statistics"""
    if not HAS_PROBE_MANAGER:
        return {"error": "Probe entry manager not available"}

    manager = get_probe_manager()
    status = manager.get_status()

    return {
        "stats": status.get("stats", {}),
        "session_probes_per_symbol": status.get("session_probes_per_symbol", {})
    }


@router.post("/reset-stats")
async def reset_stats():
    """Reset probe statistics (keeps config)"""
    if not HAS_PROBE_MANAGER:
        return {"error": "Probe entry manager not available", "success": False}

    manager = get_probe_manager()

    # Reset stats
    manager.probe_attempts = 0
    manager.probe_entries = 0
    manager.probe_confirmations = 0
    manager.probe_stops = 0
    manager.total_pnl = 0.0

    return {
        "success": True,
        "message": "Statistics reset"
    }


@router.get("/funnel-impact")
async def get_funnel_impact():
    """
    Get probe entry impact on funnel metrics.

    Shows gate_to_exec improvement from probe entries.
    """
    if not HAS_PROBE_MANAGER:
        return {"error": "Probe entry manager not available"}

    manager = get_probe_manager()
    stats = manager.get_status().get("stats", {})

    # Calculate impact
    attempts = stats.get("probe_attempts", 0)
    entries = stats.get("probe_entries", 0)
    confirmations = stats.get("probe_confirmations", 0)

    return {
        "probe_attempts": attempts,
        "probe_entries": entries,
        "probe_confirmations": confirmations,
        "entry_rate": f"{(entries / attempts * 100):.1f}%" if attempts > 0 else "0%",
        "confirmation_rate": f"{(confirmations / entries * 100):.1f}%" if entries > 0 else "0%",
        "total_pnl": stats.get("total_pnl", "$0.00"),
        "funnel_contribution": {
            "gate_to_exec_without_probe": "0%",  # Before probe
            "gate_to_exec_with_probe": f"{(entries / max(1, attempts) * 100):.1f}%"  # After probe
        }
    }
