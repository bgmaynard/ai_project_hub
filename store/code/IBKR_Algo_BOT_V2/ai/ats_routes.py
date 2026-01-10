"""
ATS (Advanced Trading Signal) API Routes

REST endpoints for ATS feed, detector, and hooks.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
from datetime import datetime
import logging

from .ats import (
    get_ats_feed,
    get_ats_detector,
    get_ats_registry,
    get_ats_gating_hook,
    get_ats_scalper_hook,
    Bar,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ats", tags=["ATS"])


# ============================================================================
# Status Endpoints
# ============================================================================

@router.get("/status")
async def get_ats_status():
    """Get complete ATS system status"""
    feed = get_ats_feed()
    return feed.get_status()


@router.get("/detector/status")
async def get_detector_status():
    """Get ATS detector status"""
    detector = get_ats_detector()
    return detector.get_status()


@router.get("/registry/status")
async def get_registry_status():
    """Get ATS registry status"""
    registry = get_ats_registry()
    return registry.get_status()


@router.get("/gating/stats")
async def get_gating_stats():
    """Get ATS gating statistics"""
    gating = get_ats_gating_hook()
    return gating.get_stats()


@router.get("/scalper/status")
async def get_scalper_hook_status():
    """Get ATS scalper hook status"""
    scalper = get_ats_scalper_hook()
    return scalper.get_status()


# ============================================================================
# Symbol State Endpoints
# ============================================================================

@router.get("/state/{symbol}")
async def get_symbol_state(symbol: str):
    """Get ATS state for symbol"""
    registry = get_ats_registry()
    state = registry.get_state(symbol.upper())

    if not state:
        return {
            "symbol": symbol.upper(),
            "tracked": False,
            "state": "UNKNOWN",
        }

    return {
        "symbol": symbol.upper(),
        "tracked": True,
        "state": state.state.value,
        "score": state.score,
        "score_trend": state.score_trend,
        "avg_score": state.avg_score,
        "consecutive_greens": state.consecutive_greens,
        "consecutive_reds": state.consecutive_reds,
        "bars_in_state": state.bars_in_state,
        "last_update": state.last_update.isoformat(),
    }


@router.get("/states")
async def get_all_states():
    """Get all symbol states"""
    registry = get_ats_registry()
    states = registry.get_all_states()

    return {
        "count": len(states),
        "states": {
            sym: {
                "state": s.state.value,
                "score": s.score,
                "score_trend": s.score_trend,
            }
            for sym, s in states.items()
        }
    }


@router.get("/active")
async def get_active_symbols():
    """Get symbols in ACTIVE state"""
    registry = get_ats_registry()
    return {
        "active_symbols": registry.get_active_symbols(),
        "count": len(registry.get_active_symbols()),
    }


@router.get("/forming")
async def get_forming_symbols():
    """Get symbols in FORMING state"""
    registry = get_ats_registry()
    return {
        "forming_symbols": registry.get_forming_symbols(),
        "count": len(registry.get_forming_symbols()),
    }


# ============================================================================
# Trigger Endpoints
# ============================================================================

@router.get("/triggers")
async def get_recent_triggers(minutes: int = 30):
    """Get recent trigger events"""
    feed = get_ats_feed()
    triggers = feed.get_recent_triggers(minutes)

    return {
        "count": len(triggers),
        "triggers": [
            {
                "symbol": t.symbol,
                "trigger_type": t.trigger_type,
                "score": t.score,
                "entry_price": t.entry_price,
                "stop_loss": t.stop_loss,
                "target_1": t.target_1,
                "risk_reward": t.risk_reward,
                "permission": t.permission,
                "timestamp": t.timestamp.isoformat(),
            }
            for t in triggers
        ]
    }


@router.get("/alerts")
async def get_recent_alerts(minutes: int = 30):
    """Get recent alert events"""
    registry = get_ats_registry()
    alerts = registry.get_recent_alerts(minutes)

    return {
        "count": len(alerts),
        "alerts": [
            {
                "symbol": a.symbol,
                "alert_type": a.alert_type,
                "message": a.message,
                "score": a.score,
                "timestamp": a.timestamp.isoformat(),
            }
            for a in alerts
        ]
    }


# ============================================================================
# Trade Permission Endpoints
# ============================================================================

@router.get("/check/{symbol}")
async def check_symbol_tradeable(symbol: str):
    """Check if symbol is tradeable via ATS"""
    feed = get_ats_feed()
    allowed, reason = feed.is_symbol_tradeable(symbol.upper())

    return {
        "symbol": symbol.upper(),
        "tradeable": allowed,
        "reason": reason,
    }


@router.post("/check-entry")
async def check_entry_permission(
    symbol: str,
    entry_price: float,
    size: float = 100
):
    """Check if entry should be approved"""
    scalper = get_ats_scalper_hook()
    allowed, reason, rec = scalper.check_entry_permission(
        symbol.upper(),
        entry_price,
        size
    )

    return {
        "symbol": symbol.upper(),
        "approved": allowed,
        "reason": reason,
        "recommendation": rec,
    }


@router.get("/check-exit/{symbol}")
async def check_exit_signal(symbol: str, current_price: float):
    """Check if position should be exited"""
    scalper = get_ats_scalper_hook()
    should_exit, reason = scalper.check_exit_signal(symbol.upper(), current_price)

    return {
        "symbol": symbol.upper(),
        "should_exit": should_exit,
        "reason": reason,
    }


@router.get("/gating/{symbol}")
async def check_gating_approval(symbol: str, entry_price: Optional[float] = None):
    """Check gating approval for symbol"""
    gating = get_ats_gating_hook()
    approved, reason, trigger = gating.check_approval(symbol.upper(), entry_price)

    result = {
        "symbol": symbol.upper(),
        "approved": approved,
        "reason": reason,
    }

    if trigger:
        result["trigger"] = {
            "entry_price": trigger.entry_price,
            "stop_loss": trigger.stop_loss,
            "target_1": trigger.target_1,
            "target_2": trigger.target_2,
            "score": trigger.score,
            "risk_reward": trigger.risk_reward,
        }

    return result


# ============================================================================
# Data Feed Endpoints
# ============================================================================

@router.post("/feed/bar")
async def feed_bar(
    symbol: str,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    vwap: Optional[float] = None
):
    """Feed a bar to ATS detector"""
    feed = get_ats_feed()

    bar = Bar(
        timestamp=datetime.now(),
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=vwap,
    )

    feed.on_bar(symbol.upper(), bar)

    # Get resulting state
    registry = get_ats_registry()
    state = registry.get_state(symbol.upper())

    return {
        "symbol": symbol.upper(),
        "bar_processed": True,
        "state": state.state.value if state else "UNKNOWN",
        "score": state.score if state else 0,
    }


@router.post("/feed/trade")
async def feed_trade(
    symbol: str,
    price: float,
    volume: float
):
    """Feed a trade tick to ATS detector"""
    feed = get_ats_feed()
    feed.on_trade(symbol.upper(), price, volume, datetime.now())

    return {
        "symbol": symbol.upper(),
        "trade_processed": True,
    }


# ============================================================================
# Control Endpoints
# ============================================================================

@router.post("/reset")
async def reset_ats():
    """Reset all ATS state"""
    feed = get_ats_feed()
    feed.reset()

    return {"reset": True}


@router.post("/reset/{symbol}")
async def reset_symbol(symbol: str):
    """Reset ATS state for symbol"""
    detector = get_ats_detector()
    registry = get_ats_registry()

    detector.reset_symbol(symbol.upper())
    registry.clear_symbol(symbol.upper())

    return {"symbol": symbol.upper(), "reset": True}


@router.post("/gating/reset-stats")
async def reset_gating_stats():
    """Reset gating statistics"""
    gating = get_ats_gating_hook()
    gating.reset_stats()

    return {"reset": True}


# ============================================================================
# SmartZone Endpoints
# ============================================================================

@router.get("/zones")
async def get_active_zones():
    """Get all active SmartZones"""
    from .ats.smartzone import get_smartzone_detector

    detector = get_smartzone_detector()
    zones = detector.get_all_active_zones()

    return {
        "count": len(zones),
        "zones": {
            sym: {
                "zone_type": z.zone_type.value,
                "zone_high": z.zone_high,
                "zone_low": z.zone_low,
                "zone_mid": z.zone_mid,
                "break_level": z.break_level,
                "formation_bars": z.formation_bars,
                "confidence": z.confidence,
            }
            for sym, z in zones.items()
        }
    }


@router.get("/zone/{symbol}")
async def get_symbol_zone(symbol: str):
    """Get SmartZone for symbol"""
    from .ats.smartzone import get_smartzone_detector

    detector = get_smartzone_detector()
    zone = detector.get_active_zone(symbol.upper())

    if not zone:
        return {
            "symbol": symbol.upper(),
            "has_zone": False,
        }

    return {
        "symbol": symbol.upper(),
        "has_zone": True,
        "zone": {
            "zone_type": zone.zone_type.value,
            "zone_high": zone.zone_high,
            "zone_low": zone.zone_low,
            "zone_mid": zone.zone_mid,
            "zone_width_pct": zone.zone_width_pct,
            "break_level": zone.break_level,
            "formation_bars": zone.formation_bars,
            "compression_ratio": zone.compression_ratio,
            "confidence": zone.confidence,
            "is_resolved": zone.is_resolved,
        }
    }


# ============================================================================
# Schwab Streaming Integration
# ============================================================================

@router.get("/schwab/status")
async def get_schwab_wiring_status():
    """Get Schwab-ATS wiring status"""
    from .ats.schwab_adapter import is_wired, get_status
    
    return {
        "wired": is_wired(),
        **get_status()
    }


@router.post("/schwab/wire")
async def wire_schwab():
    """Wire Schwab streaming to ATS feed"""
    from .ats.schwab_adapter import wire_schwab_to_ats
    
    success = wire_schwab_to_ats()
    
    return {
        "success": success,
        "message": "Schwab streaming wired to ATS" if success else "Failed to wire Schwab to ATS"
    }


@router.post("/schwab/unwire")
async def unwire_schwab():
    """Unwire Schwab streaming from ATS feed"""
    from .ats.schwab_adapter import unwire_schwab_from_ats
    
    success = unwire_schwab_from_ats()
    
    return {
        "success": success,
        "message": "Schwab streaming unwired from ATS" if success else "Failed to unwire"
    }
