"""
RelVol Resolver API Routes
==========================
REST endpoints for average volume resolution with fallback.
"""

import logging
from fastapi import APIRouter, Query
from typing import List

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ops/relvol", tags=["Ops - RelVol Resolver"])

try:
    from ai.relvol_resolver import get_relvol_resolver, resolve_avg_volume
    HAS_RESOLVER = True
except ImportError as e:
    logger.warning(f"RelVol resolver not available: {e}")
    HAS_RESOLVER = False


@router.get("/status")
async def get_relvol_status():
    """Get RelVol resolver status and stats"""
    if not HAS_RESOLVER:
        return {"error": "RelVol resolver not available", "available": False}

    resolver = get_relvol_resolver()
    return {
        "available": True,
        "stats": resolver.get_stats()
    }


@router.get("/resolve/{symbol}")
async def resolve_symbol(
    symbol: str,
    current_volume: int = Query(0, description="Today's volume for RelVol calc")
):
    """
    Resolve avgVolume for a symbol.

    Uses fallback chain: Schwab → yfinance → UNKNOWN

    UNKNOWN symbols are NOT excluded - they route as degraded candidates.
    """
    if not HAS_RESOLVER:
        return {"error": "RelVol resolver not available"}

    data = resolve_avg_volume(symbol.upper(), current_volume)

    return {
        "symbol": data.symbol,
        "avg_volume": data.avg_volume,
        "current_volume": data.current_volume,
        "rel_vol": round(data.rel_vol, 2) if data.rel_vol else None,
        "source": data.source,
        "is_degraded": data.is_degraded,
        "cached_at": data.cached_at,
        "error": data.error
    }


@router.post("/resolve/batch")
async def resolve_batch(symbols: List[str]):
    """Resolve avgVolume for multiple symbols"""
    if not HAS_RESOLVER:
        return {"error": "RelVol resolver not available", "results": {}}

    resolver = get_relvol_resolver()
    results = resolver.batch_resolve([s.upper() for s in symbols])

    return {
        "count": len(results),
        "results": {
            symbol: {
                "avg_volume": data.avg_volume,
                "source": data.source,
                "is_degraded": data.is_degraded
            }
            for symbol, data in results.items()
        },
        "degraded_count": sum(1 for d in results.values() if d.is_degraded)
    }


@router.get("/degraded")
async def get_degraded_symbols():
    """Get list of symbols with UNKNOWN avgVolume (degraded candidates)"""
    if not HAS_RESOLVER:
        return {"error": "RelVol resolver not available", "symbols": []}

    resolver = get_relvol_resolver()

    # Find symbols marked as unknown in recent resolves
    degraded = []
    for symbol, cached in resolver._cache.items():
        if cached.get("source") == "UNKNOWN":
            degraded.append(symbol)

    return {
        "count": len(degraded),
        "symbols": degraded,
        "message": "These symbols have UNKNOWN avgVolume but are NOT excluded from pipeline"
    }


@router.post("/clear-cache")
async def clear_cache():
    """Clear the avgVolume cache"""
    if not HAS_RESOLVER:
        return {"error": "RelVol resolver not available", "success": False}

    resolver = get_relvol_resolver()
    resolver.clear_cache()

    return {
        "success": True,
        "message": "avgVolume cache cleared"
    }


@router.get("/cache")
async def get_cache():
    """Get current cache contents"""
    if not HAS_RESOLVER:
        return {"error": "RelVol resolver not available"}

    resolver = get_relvol_resolver()

    return {
        "size": len(resolver._cache),
        "ttl_hours": resolver._cache_ttl_hours,
        "entries": {
            symbol: {
                "avg_volume": data.get("avg_volume"),
                "source": data.get("source"),
                "cached_at": data.get("cached_at")
            }
            for symbol, data in list(resolver._cache.items())[:50]  # Limit to 50
        }
    }
