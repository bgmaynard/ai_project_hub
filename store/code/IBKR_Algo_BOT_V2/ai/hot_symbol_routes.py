"""
Hot Symbol Engine API Routes
============================
REST API endpoints for the Hot Symbol Engine.

Endpoints:
- GET /api/hot-symbols/status - Full system status
- GET /api/hot-symbols/queue - Current hot symbols
- GET /api/hot-symbols/priority - Priority-ordered symbols for trading
- POST /api/hot-symbols/add - Manually add a hot symbol
- DELETE /api/hot-symbols/{symbol} - Remove a symbol from queue
- POST /api/hot-symbols/scan - Trigger a manual scan
- GET /api/hot-symbols/shock/status - Shock detector status
- GET /api/hot-symbols/crowd/status - Crowd scanner status

Author: AI Trading Bot Team
Version: 1.0
Created: 2026-01-10
"""

import logging
import asyncio
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/hot-symbols", tags=["Hot Symbols"])


# Request/Response models
class AddHotSymbolRequest(BaseModel):
    symbol: str
    reason: str = "MANUAL"
    confidence: float = 0.7
    ttl: int = 180


class ScanRequest(BaseModel):
    symbols: Optional[List[str]] = None
    include_crowd: bool = True


# Lazy imports to avoid circular dependencies
def get_queue():
    from ai.hot_symbol_queue import get_hot_symbol_queue
    return get_hot_symbol_queue()


def get_shock_detector():
    from ai.price_volume_shock_detector import get_shock_detector
    return get_shock_detector()


def get_crowd_scanner():
    from ai.crowd_signal_scanner import get_crowd_scanner
    return get_crowd_scanner()


@router.get("/status")
async def get_hot_symbol_status():
    """Get full Hot Symbol Engine status"""
    queue = get_queue()
    shock = get_shock_detector()
    crowd = get_crowd_scanner()

    return {
        "timestamp": datetime.now().isoformat(),
        "queue": queue.get_status(),
        "shock_detector": shock.get_status(),
        "crowd_scanner": crowd.get_status()
    }


@router.get("/queue")
async def get_hot_queue():
    """Get current hot symbols with details"""
    queue = get_queue()
    return {
        "count": len(queue.get_all()),
        "symbols": [h.to_dict() for h in queue.get_all()],
        "by_reason": {
            reason: len(queue.get_by_reason(reason))
            for reason in ["PRICE_SPIKE", "VOLUME_SHOCK", "CROWD_SURGE", "HOD_BREAK"]
        }
    }


@router.get("/priority")
async def get_priority_symbols(limit: int = Query(default=10, ge=1, le=50)):
    """Get priority-ordered hot symbols for trading"""
    queue = get_queue()
    symbols = queue.get_priority_symbols(limit)

    # Include details
    result = []
    for symbol in symbols:
        hot = queue.get(symbol)
        if hot:
            result.append({
                "symbol": symbol,
                "confidence": hot.confidence,
                "reasons": hot.reasons,
                "age_seconds": hot.age_seconds,
                "time_remaining": hot.time_remaining
            })

    return {
        "count": len(result),
        "priority_symbols": result
    }


@router.get("/is-hot/{symbol}")
async def check_if_hot(symbol: str):
    """Check if a specific symbol is hot"""
    queue = get_queue()
    hot = queue.get(symbol.upper())

    if hot:
        return {
            "is_hot": True,
            "symbol": symbol.upper(),
            "details": hot.to_dict()
        }
    else:
        return {
            "is_hot": False,
            "symbol": symbol.upper(),
            "details": None
        }


@router.post("/add")
async def add_hot_symbol(request: AddHotSymbolRequest):
    """Manually add a symbol to the hot queue"""
    queue = get_queue()

    hot = queue.add(
        symbol=request.symbol,
        reason=request.reason,
        confidence=request.confidence,
        ttl=request.ttl
    )

    logger.info(f"Manual hot symbol added: {request.symbol}")

    return {
        "success": True,
        "symbol": hot.symbol,
        "details": hot.to_dict()
    }


@router.delete("/{symbol}")
async def remove_hot_symbol(symbol: str):
    """Remove a symbol from the hot queue"""
    queue = get_queue()
    removed = queue.remove(symbol.upper())

    return {
        "success": removed,
        "symbol": symbol.upper(),
        "message": "Removed" if removed else "Not found"
    }


@router.post("/clear")
async def clear_hot_queue():
    """Clear all hot symbols"""
    queue = get_queue()
    count = queue.clear()

    return {
        "success": True,
        "cleared": count
    }


@router.get("/history")
async def get_hot_symbol_history(limit: int = Query(default=50, ge=1, le=200)):
    """Get recent hot symbol events"""
    queue = get_queue()
    return {
        "events": queue.get_history(limit)
    }


# Shock Detector endpoints
@router.get("/shock/status")
async def get_shock_status():
    """Get shock detector status"""
    detector = get_shock_detector()
    return detector.get_status()


@router.get("/shock/tracked")
async def get_tracked_symbols():
    """Get symbols being tracked by shock detector"""
    detector = get_shock_detector()
    return {
        "symbols": detector.get_tracked_symbols()
    }


@router.post("/shock/thresholds")
async def update_shock_thresholds(thresholds: dict):
    """Update shock detection thresholds"""
    detector = get_shock_detector()
    detector.update_thresholds(thresholds)

    return {
        "success": True,
        "thresholds": detector.thresholds
    }


# Crowd Scanner endpoints
@router.get("/crowd/status")
async def get_crowd_status():
    """Get crowd scanner status"""
    scanner = get_crowd_scanner()
    return scanner.get_status()


@router.get("/crowd/top-mentioned")
async def get_top_mentioned(limit: int = Query(default=20, ge=1, le=100)):
    """Get top mentioned symbols from crowd sources"""
    scanner = get_crowd_scanner()
    return {
        "top_mentioned": scanner.get_top_mentioned(limit)
    }


@router.post("/crowd/set-tickers")
async def set_valid_tickers(tickers: List[str]):
    """Set valid tickers for crowd scanning"""
    scanner = get_crowd_scanner()
    scanner.set_valid_tickers(tickers)

    return {
        "success": True,
        "count": len(tickers)
    }


@router.post("/scan")
async def trigger_scan(request: ScanRequest):
    """
    Trigger a manual scan of all sources.
    Optionally provide specific symbols to scan.
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "shock_signals": [],
        "crowd_signals": []
    }

    # Scan crowd sources
    if request.include_crowd:
        try:
            scanner = get_crowd_scanner()
            crowd_signals = await scanner.scan_all_sources()
            results["crowd_signals"] = [s.to_dict() for s in crowd_signals]
        except Exception as e:
            logger.error(f"Crowd scan error: {e}")
            results["crowd_error"] = str(e)

    return results


# Wire everything together on startup
def setup_hot_symbol_engine():
    """
    Initialize and wire all Hot Symbol Engine components.
    Call this during app startup.
    """
    from ai.price_volume_shock_detector import wire_shock_detector_to_hot_queue
    from ai.crowd_signal_scanner import wire_crowd_scanner_to_hot_queue

    try:
        wire_shock_detector_to_hot_queue()
        wire_crowd_scanner_to_hot_queue()
        logger.info("Hot Symbol Engine fully wired and ready")
        return True
    except Exception as e:
        logger.error(f"Failed to setup Hot Symbol Engine: {e}")
        return False


# Module info
if __name__ == "__main__":
    print("Hot Symbol Routes - API endpoints for Hot Symbol Engine")
    print("\nEndpoints:")
    for route in router.routes:
        print(f"  {route.methods} {route.path}")
