"""
Finviz Scanner API Routes
=========================
REST endpoints for Finviz stock screening.

PERFORMANCE OPTIMIZED:
- Uses async wrappers to avoid blocking FastAPI event loop
- 5-minute cache with rate limiting
- Scan lock prevents concurrent scans
"""
import logging
from typing import Optional
from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scanner/finviz", tags=["Finviz Scanner"])

try:
    from ai.finviz_scanner import (
        get_finviz_scanner,
        async_get_top_gainers,
        async_get_low_float_movers,
        async_get_high_volume_breakouts,
        async_scan_all
    )
    HAS_FINVIZ = True
except ImportError as e:
    logger.warning(f"Finviz scanner not available: {e}")
    HAS_FINVIZ = False


@router.get("/status")
async def get_finviz_status():
    """Get Finviz scanner status"""
    scanner_status = {}
    if HAS_FINVIZ:
        scanner = get_finviz_scanner()
        scanner_status = scanner.get_status()

    return {
        "available": HAS_FINVIZ,
        "source": "finviz",
        "cost": "free",
        "rate_limit": "30 seconds between scans",
        "cache_ttl": "5 minutes",
        "data_delay": "15-20 minutes (free tier)",
        **scanner_status
    }


@router.get("/gainers")
async def get_gainers(
    min_change: float = Query(5.0, description="Minimum % change"),
    max_price: float = Query(20.0, description="Maximum price"),
    min_volume: int = Query(500000, description="Minimum average volume")
):
    """Get top percentage gainers matching scalping criteria (async, non-blocking)"""
    if not HAS_FINVIZ:
        return {"error": "Finviz not available", "results": []}

    try:
        # Use async wrapper to avoid blocking event loop
        results = await async_get_top_gainers(min_change, max_price, min_volume)
        return {
            "count": len(results),
            "cached": get_finviz_scanner()._is_cache_valid(f"gainers_{min_change}_{max_price}_{min_volume}"),
            "filters": {
                "min_change": min_change,
                "max_price": max_price,
                "min_volume": min_volume
            },
            "results": [
                {
                    "symbol": r.symbol,
                    "price": r.price,
                    "change_pct": r.change_pct,
                    "volume": r.volume,
                    "market_cap": r.market_cap,
                    "sector": r.sector,
                    "source": "finviz"
                }
                for r in results
            ]
        }
    except Exception as e:
        logger.error(f"Finviz gainers error: {e}")
        return {"error": str(e), "results": []}


@router.get("/low-float")
async def get_low_float(max_float: float = Query(20.0, description="Max float in millions")):
    """Get low float stocks with momentum (async, non-blocking)"""
    if not HAS_FINVIZ:
        return {"error": "Finviz not available", "results": []}

    try:
        results = await async_get_low_float_movers(max_float)
        return {
            "count": len(results),
            "results": [
                {
                    "symbol": r.symbol,
                    "price": r.price,
                    "change_pct": r.change_pct,
                    "volume": r.volume,
                    "source": "finviz"
                }
                for r in results
            ]
        }
    except Exception as e:
        logger.error(f"Finviz low float error: {e}")
        return {"error": str(e), "results": []}


@router.get("/breakouts")
async def get_breakouts():
    """Get high relative volume breakout candidates (async, non-blocking)"""
    if not HAS_FINVIZ:
        return {"error": "Finviz not available", "results": []}

    try:
        results = await async_get_high_volume_breakouts()
        return {
            "count": len(results),
            "description": "Stocks with 3x+ normal volume (potential breakouts)",
            "results": [
                {
                    "symbol": r.symbol,
                    "price": r.price,
                    "change_pct": r.change_pct,
                    "volume": r.volume,
                    "source": "finviz"
                }
                for r in results
            ]
        }
    except Exception as e:
        logger.error(f"Finviz breakouts error: {e}")
        return {"error": str(e), "results": []}


@router.get("/scan-all")
async def scan_all():
    """Run all Finviz scans and return combined results (async, non-blocking)"""
    if not HAS_FINVIZ:
        return {"error": "Finviz not available"}

    try:
        results = await async_scan_all()
        return {
            "top_gainers": [
                {"symbol": r.symbol, "price": r.price, "change_pct": r.change_pct}
                for r in results.get("top_gainers", [])[:15]
            ],
            "low_float": [
                {"symbol": r.symbol, "price": r.price, "change_pct": r.change_pct}
                for r in results.get("low_float", [])[:15]
            ],
            "high_volume": [
                {"symbol": r.symbol, "price": r.price, "change_pct": r.change_pct}
                for r in results.get("high_volume", [])[:15]
            ],
            "timestamp": results.get("timestamp")
        }
    except Exception as e:
        logger.error(f"Finviz scan-all error: {e}")
        return {"error": str(e)}


@router.post("/sync-to-watchlist")
async def sync_to_watchlist(
    min_change: float = Query(5.0),
    max_count: int = Query(10, description="Max symbols to add")
):
    """Sync top Finviz gainers to the trading watchlist (async, non-blocking)"""
    if not HAS_FINVIZ:
        return {"error": "Finviz not available", "added": []}

    try:
        results = await async_get_top_gainers(min_change=min_change)
        results = results[:max_count]

        # Add to worklist using async httpx
        added = []
        import httpx
        async with httpx.AsyncClient() as client:
            for r in results:
                try:
                    resp = await client.post(
                        "http://localhost:9100/api/worklist/add",
                        json={"symbol": r.symbol},
                        timeout=5
                    )
                    if resp.status_code == 200:
                        added.append(r.symbol)
                except:
                    pass

        return {
            "success": True,
            "added": added,
            "count": len(added)
        }

    except Exception as e:
        logger.error(f"Finviz sync error: {e}")
        return {"error": str(e), "added": []}


@router.post("/stop")
async def stop_finviz():
    """Stop any running Finviz scan (clears cache to force fresh scan next time)"""
    if not HAS_FINVIZ:
        return {"success": False, "error": "Finviz not available"}

    scanner = get_finviz_scanner()
    # Clear cache to allow fresh scan
    scanner._cache.clear()
    scanner._cache_time.clear()

    return {
        "success": True,
        "message": "Finviz scanner cache cleared",
        "running": scanner._is_scanning
    }


@router.post("/clear-cache")
async def clear_cache():
    """Clear Finviz scanner cache"""
    if not HAS_FINVIZ:
        return {"success": False, "error": "Finviz not available"}

    scanner = get_finviz_scanner()
    scanner._cache.clear()
    scanner._cache_time.clear()

    return {
        "success": True,
        "message": "Cache cleared"
    }


@router.post("/momentum")
async def get_momentum_breakouts(min_change: float = 5.0):
    """Get momentum breakouts - high relative volume + up movement"""
    if not HAS_FINVIZ:
        return {"success": False, "error": "Finviz not available", "results": []}

    try:
        scanner = get_finviz_scanner()
        results = scanner.get_momentum_breakouts(min_change=min_change)

        return {
            "success": True,
            "count": len(results),
            "scan_type": "momentum",
            "results": [
                {
                    "symbol": r.symbol,
                    "price": r.price,
                    "change_pct": r.change_pct,
                    "volume": r.volume,
                    "market_cap": r.market_cap
                }
                for r in results[:25]
            ]
        }
    except Exception as e:
        logger.error(f"Finviz momentum scan error: {e}")
        return {"success": False, "error": str(e), "results": []}


@router.post("/new-highs")
async def get_new_highs(max_price: float = 20.0):
    """Get stocks hitting new highs (HOD/52-week high) - momentum trigger"""
    if not HAS_FINVIZ:
        return {"success": False, "error": "Finviz not available", "results": []}

    try:
        scanner = get_finviz_scanner()
        results = scanner.get_new_highs(max_price=max_price)

        return {
            "success": True,
            "count": len(results),
            "scan_type": "new_highs",
            "results": [
                {
                    "symbol": r.symbol,
                    "price": r.price,
                    "change_pct": r.change_pct,
                    "volume": r.volume,
                    "market_cap": r.market_cap
                }
                for r in results[:25]
            ]
        }
    except Exception as e:
        logger.error(f"Finviz new highs scan error: {e}")
        return {"success": False, "error": str(e), "results": []}
