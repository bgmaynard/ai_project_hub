"""
Micro-Momentum Override API Routes
==================================
REST endpoints for micro-momentum override management.
"""

import logging
from fastapi import APIRouter, Query
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ops/micro-override", tags=["Ops - Micro Override"])

try:
    from ai.micro_momentum_override import (
        get_micro_override,
        check_micro_override,
        MicroMomentumOverride
    )
    HAS_OVERRIDE = True
except ImportError as e:
    logger.warning(f"Micro-momentum override not available: {e}")
    HAS_OVERRIDE = False


@router.get("/status")
async def get_override_status():
    """Get micro-momentum override status and stats"""
    if not HAS_OVERRIDE:
        return {"error": "Micro-momentum override not available", "available": False}

    override = get_micro_override()
    return {
        "available": True,
        **override.get_status()
    }


@router.post("/check")
async def check_override_eligibility(
    symbol: str = Query(..., description="Stock symbol"),
    macro_regime: str = Query("TRENDING_DOWN", description="Current macro regime"),
    micro_regime: str = Query("TRENDING_UP", description="Current micro regime"),
    micro_confidence: float = Query(0.7, description="Micro regime confidence"),
    ats_state: str = Query("ACTIVE", description="Current ATS state"),
    current_volume: int = Query(0, description="Today's volume"),
    float_millions: float = Query(0, description="Float in millions"),
    rel_vol: float = Query(0, description="Relative volume")
):
    """
    Check if a symbol qualifies for micro-momentum override.

    This is a TEST endpoint - does NOT grant override, just checks eligibility.
    """
    if not HAS_OVERRIDE:
        return {"error": "Micro-momentum override not available"}

    allowed, reason, size_mult = check_micro_override(
        symbol=symbol.upper(),
        macro_regime=macro_regime,
        micro_regime=micro_regime,
        micro_confidence=micro_confidence,
        ats_state=ats_state,
        current_volume=current_volume,
        float_millions=float_millions,
        rel_vol=rel_vol
    )

    return {
        "symbol": symbol.upper(),
        "override_allowed": allowed,
        "reason": reason,
        "position_size_multiplier": size_mult,
        "inputs": {
            "macro_regime": macro_regime,
            "micro_regime": micro_regime,
            "micro_confidence": micro_confidence,
            "ats_state": ats_state,
            "current_volume": current_volume,
            "float_millions": float_millions,
            "rel_vol": rel_vol
        }
    }


@router.get("/config")
async def get_override_config():
    """Get override configuration"""
    if not HAS_OVERRIDE:
        return {"error": "Micro-momentum override not available"}

    override = get_micro_override()
    return {
        "enabled": override.config.enabled,
        "max_per_10min": override.config.max_per_10min,
        "size_multiplier": override.config.size_multiplier,
        "min_micro_confidence": override.config.min_micro_confidence,
        "min_rel_vol": override.config.min_rel_vol,
        "min_volume": override.config.min_volume,
        "max_float_millions": override.config.max_float_millions,
        "qualifying_ats_states": override.config.qualifying_ats_states
    }


@router.post("/config")
async def update_override_config(
    enabled: Optional[bool] = None,
    max_per_10min: Optional[int] = None,
    size_multiplier: Optional[float] = None,
    min_micro_confidence: Optional[float] = None,
    min_rel_vol: Optional[float] = None
):
    """Update override configuration"""
    if not HAS_OVERRIDE:
        return {"error": "Micro-momentum override not available", "success": False}

    override = get_micro_override()
    override.update_config(
        enabled=enabled,
        max_per_10min=max_per_10min,
        size_multiplier=size_multiplier,
        min_micro_confidence=min_micro_confidence,
        min_rel_vol=min_rel_vol
    )

    return {
        "success": True,
        "message": "Configuration updated",
        "config": {
            "enabled": override.config.enabled,
            "max_per_10min": override.config.max_per_10min,
            "size_multiplier": override.config.size_multiplier,
            "min_micro_confidence": override.config.min_micro_confidence,
            "min_rel_vol": override.config.min_rel_vol
        }
    }


@router.post("/enable")
async def enable_override():
    """Enable micro-momentum override"""
    if not HAS_OVERRIDE:
        return {"error": "Micro-momentum override not available", "success": False}

    override = get_micro_override()
    override.update_config(enabled=True)

    return {
        "success": True,
        "message": "Micro-momentum override enabled",
        "enabled": True
    }


@router.post("/disable")
async def disable_override():
    """Disable micro-momentum override"""
    if not HAS_OVERRIDE:
        return {"error": "Micro-momentum override not available", "success": False}

    override = get_micro_override()
    override.update_config(enabled=False)

    return {
        "success": True,
        "message": "Micro-momentum override disabled",
        "enabled": False
    }


@router.get("/recent")
async def get_recent_overrides(limit: int = Query(10, description="Max recent overrides")):
    """Get recent override events"""
    if not HAS_OVERRIDE:
        return {"error": "Micro-momentum override not available", "overrides": []}

    override = get_micro_override()
    status = override.get_status()

    return {
        "count": len(status.get("recent_overrides", [])),
        "overrides": status.get("recent_overrides", [])[:limit],
        "stats": status.get("stats", {})
    }


@router.get("/denial-reasons")
async def get_denial_reasons():
    """Get breakdown of why overrides were denied"""
    if not HAS_OVERRIDE:
        return {"error": "Micro-momentum override not available"}

    override = get_micro_override()
    status = override.get_status()
    stats = status.get("stats", {})

    return {
        "total_requests": stats.get("override_requests", 0),
        "granted": stats.get("overrides_granted", 0),
        "denied": stats.get("overrides_denied", 0),
        "grant_rate": f"{stats.get('grant_rate', 0):.1f}%",
        "denial_breakdown": stats.get("denial_reasons", {})
    }
