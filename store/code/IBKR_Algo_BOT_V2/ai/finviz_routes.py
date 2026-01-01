"""
Finviz Scanner API Routes
=========================
REST endpoints for Finviz stock screening.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scanner/finviz", tags=["Finviz Scanner"])

try:
    from ai.finviz_scanner import get_finviz_scanner, scan_top_gainers

    HAS_FINVIZ = True
except ImportError as e:
    logger.warning(f"Finviz scanner not available: {e}")
    HAS_FINVIZ = False


@router.get("/status")
async def get_finviz_status():
    """Get Finviz scanner status"""
    return {
        "available": HAS_FINVIZ,
        "source": "finviz",
        "cost": "free",
        "rate_limit": "~1 request/second recommended",
        "data_delay": "15-20 minutes (free tier)",
    }


@router.get("/gainers")
async def get_gainers(
    min_change: float = Query(5.0, description="Minimum % change"),
    max_price: float = Query(20.0, description="Maximum price"),
    min_volume: int = Query(500000, description="Minimum average volume"),
):
    """Get top percentage gainers matching scalping criteria"""
    if not HAS_FINVIZ:
        return {"error": "Finviz not available", "results": []}

    try:
        scanner = get_finviz_scanner()
        results = scanner.get_top_gainers(
            min_change=min_change, max_price=max_price, min_volume=min_volume
        )
        return {
            "count": len(results),
            "filters": {
                "min_change": min_change,
                "max_price": max_price,
                "min_volume": min_volume,
            },
            "results": [
                {
                    "symbol": r.symbol,
                    "price": r.price,
                    "change_pct": r.change_pct,
                    "volume": r.volume,
                    "market_cap": r.market_cap,
                    "sector": r.sector,
                    "source": "finviz",
                }
                for r in results
            ],
        }
    except Exception as e:
        logger.error(f"Finviz gainers error: {e}")
        return {"error": str(e), "results": []}


@router.get("/low-float")
async def get_low_float(
    max_float: float = Query(20.0, description="Max float in millions")
):
    """Get low float stocks with momentum"""
    if not HAS_FINVIZ:
        return {"error": "Finviz not available", "results": []}

    try:
        scanner = get_finviz_scanner()
        results = scanner.get_low_float_movers(max_float=max_float)
        return {
            "count": len(results),
            "results": [
                {
                    "symbol": r.symbol,
                    "price": r.price,
                    "change_pct": r.change_pct,
                    "volume": r.volume,
                    "source": "finviz",
                }
                for r in results
            ],
        }
    except Exception as e:
        logger.error(f"Finviz low float error: {e}")
        return {"error": str(e), "results": []}


@router.get("/breakouts")
async def get_breakouts():
    """Get high relative volume breakout candidates"""
    if not HAS_FINVIZ:
        return {"error": "Finviz not available", "results": []}

    try:
        scanner = get_finviz_scanner()
        results = scanner.get_high_volume_breakouts()
        return {
            "count": len(results),
            "description": "Stocks with 3x+ normal volume (potential breakouts)",
            "results": [
                {
                    "symbol": r.symbol,
                    "price": r.price,
                    "change_pct": r.change_pct,
                    "volume": r.volume,
                    "source": "finviz",
                }
                for r in results
            ],
        }
    except Exception as e:
        logger.error(f"Finviz breakouts error: {e}")
        return {"error": str(e), "results": []}


@router.get("/scan-all")
async def scan_all():
    """Run all Finviz scans and return combined results"""
    if not HAS_FINVIZ:
        return {"error": "Finviz not available"}

    try:
        scanner = get_finviz_scanner()
        results = scanner.scan_all()
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
            "timestamp": results.get("timestamp"),
        }
    except Exception as e:
        logger.error(f"Finviz scan-all error: {e}")
        return {"error": str(e)}


@router.post("/sync-to-watchlist")
async def sync_to_watchlist(
    min_change: float = Query(5.0),
    max_count: int = Query(10, description="Max symbols to add"),
):
    """Sync top Finviz gainers to the trading watchlist"""
    if not HAS_FINVIZ:
        return {"error": "Finviz not available", "added": []}

    try:
        from ai.finviz_scanner import get_finviz_scanner

        scanner = get_finviz_scanner()
        results = scanner.get_top_gainers(min_change=min_change)[:max_count]

        # Add to worklist
        added = []
        import httpx

        for r in results:
            try:
                resp = httpx.post(
                    "http://localhost:9100/api/worklist/add",
                    json={"symbol": r.symbol},
                    timeout=5,
                )
                if resp.status_code == 200:
                    added.append(r.symbol)
            except:
                pass

        return {"success": True, "added": added, "count": len(added)}

    except Exception as e:
        logger.error(f"Finviz sync error: {e}")
        return {"error": str(e), "added": []}
