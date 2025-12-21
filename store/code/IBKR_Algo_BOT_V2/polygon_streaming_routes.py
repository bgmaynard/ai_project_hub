"""
Polygon Streaming API Routes
=============================
REST endpoints for real-time Polygon.io data streaming.

Endpoints:
- GET  /api/polygon/stream/status     - Stream status
- POST /api/polygon/stream/start      - Start streaming
- POST /api/polygon/stream/stop       - Stop streaming
- POST /api/polygon/stream/subscribe  - Subscribe to symbol
- POST /api/polygon/stream/unsubscribe - Unsubscribe from symbol
- GET  /api/polygon/stream/trades/{symbol} - Get recent trades
- GET  /api/polygon/stream/quote/{symbol}  - Get latest quote
- GET  /api/polygon/stream/tape       - Get combined tape
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/polygon/stream", tags=["Polygon Streaming"])


# Import polygon streaming
try:
    from polygon_streaming import get_polygon_stream, is_polygon_streaming_available
    HAS_POLYGON_STREAM = True
except ImportError as e:
    logger.warning(f"Polygon streaming not available: {e}")
    HAS_POLYGON_STREAM = False


class SubscribeRequest(BaseModel):
    """Subscribe request"""
    symbol: str
    trades: bool = True
    quotes: bool = True


@router.get("/status")
async def get_stream_status():
    """Get Polygon stream status"""
    if not HAS_POLYGON_STREAM:
        return {"available": False, "message": "Polygon streaming not available"}

    stream = get_polygon_stream()
    status = stream.get_status()
    status["available"] = True
    return status


@router.post("/start")
async def start_stream():
    """Start the Polygon stream"""
    if not HAS_POLYGON_STREAM:
        raise HTTPException(status_code=503, detail="Polygon streaming not available")

    stream = get_polygon_stream()

    if not stream.api_key:
        raise HTTPException(status_code=400, detail="Polygon API key not configured")

    stream.start()
    return {
        "success": True,
        "message": "Polygon stream started",
        "status": stream.get_status()
    }


@router.post("/stop")
async def stop_stream():
    """Stop the Polygon stream"""
    if not HAS_POLYGON_STREAM:
        raise HTTPException(status_code=503, detail="Polygon streaming not available")

    stream = get_polygon_stream()
    stream.stop()
    return {
        "success": True,
        "message": "Polygon stream stopped"
    }


@router.post("/subscribe")
async def subscribe_symbol(request: SubscribeRequest):
    """Subscribe to trades/quotes for a symbol"""
    if not HAS_POLYGON_STREAM:
        raise HTTPException(status_code=503, detail="Polygon streaming not available")

    stream = get_polygon_stream()
    symbol = request.symbol.upper()

    if request.trades:
        stream.subscribe_trades(symbol)
    if request.quotes:
        stream.subscribe_quotes(symbol)

    return {
        "success": True,
        "symbol": symbol,
        "trades": request.trades,
        "quotes": request.quotes,
        "status": stream.get_status()
    }


@router.post("/unsubscribe/{symbol}")
async def unsubscribe_symbol(symbol: str):
    """Unsubscribe from a symbol"""
    if not HAS_POLYGON_STREAM:
        raise HTTPException(status_code=503, detail="Polygon streaming not available")

    stream = get_polygon_stream()
    stream.unsubscribe(symbol.upper())

    return {
        "success": True,
        "symbol": symbol.upper(),
        "message": f"Unsubscribed from {symbol.upper()}"
    }


@router.get("/trades/{symbol}")
async def get_trades(symbol: str, limit: int = 50):
    """Get recent trades for a symbol"""
    if not HAS_POLYGON_STREAM:
        raise HTTPException(status_code=503, detail="Polygon streaming not available")

    stream = get_polygon_stream()
    trades = stream.get_trades(symbol.upper(), limit)

    return {
        "symbol": symbol.upper(),
        "count": len(trades),
        "trades": trades
    }


@router.get("/quote/{symbol}")
async def get_quote(symbol: str):
    """Get latest quote for a symbol"""
    if not HAS_POLYGON_STREAM:
        raise HTTPException(status_code=503, detail="Polygon streaming not available")

    stream = get_polygon_stream()
    quote = stream.get_quote(symbol.upper())

    if quote:
        return quote
    else:
        return {
            "symbol": symbol.upper(),
            "message": "No quote available - symbol may not be subscribed"
        }


@router.get("/tape")
async def get_tape(limit: int = 100):
    """Get combined tape (all symbols)"""
    if not HAS_POLYGON_STREAM:
        raise HTTPException(status_code=503, detail="Polygon streaming not available")

    stream = get_polygon_stream()
    tape = stream.get_tape(limit)

    return {
        "count": len(tape),
        "trades": tape
    }


@router.post("/subscribe-watchlist")
async def subscribe_watchlist(symbols: List[str]):
    """Subscribe to multiple symbols at once"""
    if not HAS_POLYGON_STREAM:
        raise HTTPException(status_code=503, detail="Polygon streaming not available")

    stream = get_polygon_stream()

    subscribed = []
    for symbol in symbols:
        symbol = symbol.upper().strip()
        if symbol:
            stream.subscribe_trades(symbol)
            stream.subscribe_quotes(symbol)
            subscribed.append(symbol)

    return {
        "success": True,
        "subscribed": subscribed,
        "count": len(subscribed),
        "status": stream.get_status()
    }
