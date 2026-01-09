"""
Hybrid Data API Routes
======================
Routes for the hybrid data architecture that separates fast and background channels.

FAST CHANNEL ENDPOINTS (Low latency, real-time):
- /api/hybrid/fast/quote/{symbol}
- /api/hybrid/fast/level2/{symbol}
- /api/hybrid/fast/timesales/{symbol}
- /api/hybrid/fast/positions

BACKGROUND CHANNEL ENDPOINTS (Cached, non-blocking):
- /api/hybrid/bg/bars/{symbol}
- /api/hybrid/bg/account
- /api/hybrid/bg/orders
- /api/hybrid/bg/ai-data/{symbol}

STATUS:
- /api/hybrid/status
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/hybrid", tags=["Hybrid Data"])

# Import hybrid provider
try:
    from schwab_hybrid_data import (ChannelPriority, DataType, bg_account,
                                    bg_ai_data, bg_bars, bg_orders,
                                    fast_level2, fast_positions, fast_quote,
                                    fast_time_sales, get_hybrid_provider)

    HAS_HYBRID = True
except ImportError as e:
    logger.warning(f"Hybrid data not available: {e}")
    HAS_HYBRID = False


# ============================================================================
# FAST CHANNEL ENDPOINTS - Real-time trading data
# ============================================================================


@router.get("/fast/quote/{symbol}")
async def get_fast_quote(symbol: str):
    """
    Get live quote via FAST channel.
    Prioritized for minimal latency.
    """
    if not HAS_HYBRID:
        return {"error": "Hybrid data not available"}

    result = fast_quote(symbol)
    result["channel"] = "fast"
    result["priority"] = "high"
    return result


@router.get("/fast/level2/{symbol}")
async def get_fast_level2(symbol: str, depth: int = Query(10, ge=1, le=20)):
    """
    Get Level 2 order book via FAST channel.
    Real top-of-book with simulated depth.
    """
    if not HAS_HYBRID:
        return {"error": "Hybrid data not available"}

    result = fast_level2(symbol, depth)
    result["channel"] = "fast"
    result["priority"] = "high"
    return result


@router.get("/fast/timesales/{symbol}")
async def get_fast_time_sales(symbol: str, limit: int = Query(50, ge=10, le=200)):
    """
    Get Time & Sales via FAST channel.
    Real prices with simulated tick flow.
    """
    if not HAS_HYBRID:
        return {"error": "Hybrid data not available"}

    result = fast_time_sales(symbol, limit)
    result["channel"] = "fast"
    result["priority"] = "high"
    return result


@router.get("/fast/positions")
async def get_fast_positions():
    """
    Get live positions via FAST channel.
    Real-time position updates.
    """
    if not HAS_HYBRID:
        return {"error": "Hybrid data not available"}

    result = fast_positions()
    result["channel"] = "fast"
    result["priority"] = "high"
    return result


# ============================================================================
# BACKGROUND CHANNEL ENDPOINTS - Cached, non-blocking
# ============================================================================


@router.get("/bg/bars/{symbol}")
async def get_bg_bars(
    symbol: str,
    timeframe: str = Query("1D", description="Timeframe: 1m, 5m, 15m, 1H, 1D"),
    limit: int = Query(100, ge=10, le=500),
):
    """
    Get historical bars via BACKGROUND channel.
    Cached for 60 seconds to reduce API load.
    """
    if not HAS_HYBRID:
        return {"error": "Hybrid data not available"}

    result = bg_bars(symbol, timeframe, limit)
    result["channel"] = "background"
    result["priority"] = "low"
    return result


@router.get("/bg/account")
async def get_bg_account():
    """
    Get account balance via BACKGROUND channel.
    Cached for 30 seconds.
    """
    if not HAS_HYBRID:
        return {"error": "Hybrid data not available"}

    result = bg_account()
    result["channel"] = "background"
    result["priority"] = "low"
    return result


@router.get("/bg/orders")
async def get_bg_orders(days: int = Query(7, ge=1, le=30)):
    """
    Get order history via BACKGROUND channel.
    Cached for 2 minutes.
    """
    if not HAS_HYBRID:
        return {"error": "Hybrid data not available"}

    result = bg_orders(days)
    result["channel"] = "background"
    result["priority"] = "low"
    return result


@router.get("/bg/ai-data/{symbol}")
async def get_bg_ai_data(
    symbol: str, period: str = Query("3mo", description="Period: 1mo, 3mo, 6mo, 1y")
):
    """
    Get AI training data via BACKGROUND channel.
    Cached for 5 minutes - uses Yahoo Finance.
    """
    if not HAS_HYBRID:
        return {"error": "Hybrid data not available"}

    result = bg_ai_data(symbol, period)
    result["channel"] = "background"
    result["priority"] = "low"
    return result


# ============================================================================
# STATUS ENDPOINT
# ============================================================================


@router.get("/status")
async def get_hybrid_status():
    """
    Get status of both fast and background channels.
    Shows request counts, latencies, and cache stats.
    """
    if not HAS_HYBRID:
        return {"error": "Hybrid data not available", "available": False}

    provider = get_hybrid_provider()
    status = provider.get_status()
    status["available"] = True
    return status


@router.get("/info")
async def get_hybrid_info():
    """
    Get information about the hybrid data architecture.
    """
    return {
        "name": "Schwab Hybrid Data Architecture",
        "description": "Optimized data flow separating real-time and background operations",
        "channels": {
            "fast": {
                "description": "Real-time trading data with minimal latency",
                "use_cases": [
                    "Level 2",
                    "Time & Sales",
                    "Live quotes",
                    "Order execution",
                    "Position updates",
                ],
                "latency_target": "< 50ms",
                "caching": "None - always fresh",
            },
            "background": {
                "description": "Non-critical data with caching to reduce load",
                "use_cases": [
                    "Historical charts",
                    "Account balance",
                    "Order history",
                    "AI training",
                    "Backtesting",
                ],
                "latency_target": "< 500ms",
                "caching": "TTL-based (30s - 10min)",
            },
        },
        "benefits": [
            "Trading operations never blocked by heavy data requests",
            "Background caching reduces API rate limit usage",
            "Dedicated thread pools prevent resource contention",
            "Clear separation of concerns for debugging",
        ],
        "endpoints": {
            "fast": [
                "GET /api/hybrid/fast/quote/{symbol}",
                "GET /api/hybrid/fast/level2/{symbol}",
                "GET /api/hybrid/fast/timesales/{symbol}",
                "GET /api/hybrid/fast/positions",
            ],
            "background": [
                "GET /api/hybrid/bg/bars/{symbol}",
                "GET /api/hybrid/bg/account",
                "GET /api/hybrid/bg/orders",
                "GET /api/hybrid/bg/ai-data/{symbol}",
            ],
            "status": ["GET /api/hybrid/status", "GET /api/hybrid/info"],
        },
    }
