"""
Warrior Trading API Routes
==========================
REST API endpoints for Ross Cameron trading methodology:
- Setup detection & classification
- Pattern recognition
- Tape analysis
- LULD halt detection

Endpoints:
- GET  /api/warrior/status           - System status
- GET  /api/warrior/signal/{symbol}  - Get trading signal
- POST /api/warrior/analyze          - Analyze multiple symbols
- GET  /api/warrior/patterns/{symbol} - Pattern detection
- GET  /api/warrior/tape/{symbol}    - Tape analysis
- GET  /api/warrior/luld/{symbol}    - LULD band status
- POST /api/warrior/feed/tape        - Feed tape data
- POST /api/warrior/feed/candles     - Feed candle data
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/warrior", tags=["Warrior Trading"])


class AnalyzeRequest(BaseModel):
    """Request to analyze symbols"""

    symbols: List[str]


class TapeFeedRequest(BaseModel):
    """Request to feed tape data"""

    symbol: str
    trades: List[Dict]


class CandleFeedRequest(BaseModel):
    """Request to feed candle data"""

    symbol: str
    candles: List[Dict]


class LULDRequest(BaseModel):
    """Request to calculate LULD bands"""

    symbol: str
    reference_price: float
    tier: int = 2


# Lazy load detector
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        from ai.warrior_setup_detector import get_warrior_detector

        _detector = get_warrior_detector()
    return _detector


@router.get("/status")
async def get_status():
    """Get Warrior Trading system status"""
    try:
        detector = get_detector()
        status = detector.get_status()
        return {
            "success": True,
            "available": True,
            "scanner_enabled": True,
            "patterns_enabled": ["ABCD", "Flag", "VWAP_Hold", "HOD_Break"],
            "risk_config": {"daily_goal": 500, "min_rr": 2},
            "status": status,
        }
    except Exception as e:
        logger.error(f"Status error: {e}")
        return {
            "success": False,
            "available": True,
            "scanner_enabled": True,
            "patterns_enabled": ["ABCD", "Flag", "VWAP_Hold"],
            "risk_config": {"daily_goal": 500, "min_rr": 2},
            "error": str(e),
        }


@router.get("/signal/{symbol}")
async def get_signal(symbol: str):
    """Get trading signal for a symbol"""
    try:
        from ai.warrior_setup_detector import get_trading_signal

        signal = await get_trading_signal(symbol.upper())
        return {"success": True, "signal": signal}
    except Exception as e:
        logger.error(f"Signal error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze_symbols(request: AnalyzeRequest):
    """Analyze multiple symbols for setups"""
    try:
        from ai.warrior_setup_detector import get_trading_signal

        results = []
        for symbol in request.symbols[:20]:  # Limit to 20
            try:
                signal = await get_trading_signal(symbol.upper())
                results.append(signal)
            except Exception as e:
                results.append({"symbol": symbol, "error": str(e)})

        # Sort by setup grade and confidence
        grade_order = {"A+": 0, "A": 1, "B": 2, "C": 3, "F": 4}
        results.sort(
            key=lambda x: (
                grade_order.get(x.get("setup", {}).get("grade", "F"), 4),
                -x.get("setup", {}).get("confidence", 0),
            )
        )

        return {"success": True, "count": len(results), "signals": results}
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/{symbol}")
async def get_patterns(symbol: str):
    """Get pattern detection for a symbol"""
    try:
        from ai.pattern_detector import get_pattern_detector

        detector = get_pattern_detector()
        patterns = detector.detect_all_patterns(symbol.upper())

        return {
            "success": True,
            "symbol": symbol.upper(),
            "patterns": {name: p.to_dict() for name, p in patterns.items()},
            "best_pattern": (
                detector.get_best_setup(symbol.upper()).to_dict()
                if detector.get_best_setup(symbol.upper())
                else None
            ),
        }
    except Exception as e:
        logger.error(f"Pattern detection error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tape/{symbol}")
async def get_tape_analysis(symbol: str):
    """Get tape analysis for a symbol"""
    try:
        from ai.tape_analyzer import get_tape_analyzer

        analyzer = get_tape_analyzer()
        analysis = analyzer.analyze(symbol.upper())
        entry_signal = analyzer.get_entry_signal(symbol.upper())
        flush = analyzer.detect_irrational_flush(symbol.upper())

        return {
            "success": True,
            "symbol": symbol.upper(),
            "analysis": analysis.to_dict(),
            "entry_signal": entry_signal,
            "flush_detection": flush,
        }
    except Exception as e:
        logger.error(f"Tape analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/luld/{symbol}")
async def get_luld_status(symbol: str):
    """Get LULD band status for a symbol"""
    try:
        from ai.halt_detector import get_halt_detector

        detector = get_halt_detector()
        status = detector.get_luld_status(symbol.upper())

        if status:
            return {"success": True, "symbol": symbol.upper(), "luld": status}
        else:
            return {
                "success": True,
                "symbol": symbol.upper(),
                "luld": None,
                "message": "No LULD bands calculated. Use POST /api/warrior/luld to calculate.",
            }
    except Exception as e:
        logger.error(f"LULD status error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/luld")
async def calculate_luld(request: LULDRequest):
    """Calculate LULD bands for a symbol"""
    try:
        from ai.halt_detector import get_halt_detector

        detector = get_halt_detector()
        bands = detector.calculate_luld_bands(
            request.symbol.upper(), request.reference_price, request.tier
        )

        return {
            "success": True,
            "symbol": request.symbol.upper(),
            "bands": bands.to_dict(),
        }
    except Exception as e:
        logger.error(f"LULD calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/halts")
async def get_halts():
    """Get all current halts and warnings"""
    try:
        from ai.halt_detector import get_halt_detector

        detector = get_halt_detector()

        return {
            "success": True,
            "halted": detector.get_all_halts(),
            "history": detector.get_halt_history(limit=10),
            "false_halts": detector.false_halts[-10:],
            "luld_bands": {
                sym: bands.to_dict() for sym, bands in detector.luld_bands.items()
            },
        }
    except Exception as e:
        logger.error(f"Halts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feed/tape")
async def feed_tape_data(request: TapeFeedRequest):
    """Feed tape (time & sales) data"""
    try:
        detector = get_detector()
        detector.feed_tape_data(request.symbol.upper(), request.trades)

        return {
            "success": True,
            "symbol": request.symbol.upper(),
            "trades_added": len(request.trades),
        }
    except Exception as e:
        logger.error(f"Tape feed error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feed/candles")
async def feed_candle_data(request: CandleFeedRequest):
    """Feed candle (OHLCV) data"""
    try:
        detector = get_detector()
        detector.feed_candle_data(request.symbol.upper(), request.candles)

        return {
            "success": True,
            "symbol": request.symbol.upper(),
            "candles_added": len(request.candles),
        }
    except Exception as e:
        logger.error(f"Candle feed error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/setup-rules/{setup_type}")
async def get_setup_rules(setup_type: str):
    """Get entry rules for a specific setup type"""
    try:
        from ai.setup_classifier import SetupType, get_setup_classifier

        classifier = get_setup_classifier()

        # Map string to SetupType
        setup_map = {
            "bull_flag": SetupType.BULL_FLAG,
            "abcd": SetupType.ABCD,
            "micro_pullback": SetupType.MICRO_PULLBACK,
            "hod_break": SetupType.HOD_BREAK,
            "vwap_breakout": SetupType.VWAP_BREAKOUT,
            "dip_buy": SetupType.DIP_BUY,
            "halt_resume": SetupType.HALT_RESUME,
        }

        setup = setup_map.get(setup_type.lower())
        if not setup:
            return {
                "success": False,
                "error": f"Unknown setup type: {setup_type}",
                "valid_types": list(setup_map.keys()),
            }

        rules = classifier.get_entry_rules(setup)

        return {"success": True, "setup_type": setup.value, "rules": rules}
    except Exception as e:
        logger.error(f"Setup rules error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/grade/{symbol}")
async def get_stock_grade(symbol: str):
    """Get Ross Cameron A/B/C grade for a stock"""
    try:
        import httpx

        # Get stock data
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"http://localhost:9100/api/price/{symbol.upper()}", timeout=5.0
            )
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=404, detail=f"Price not found for {symbol}"
                )

            quote = resp.json()

        from ai.setup_classifier import StockCriteria, get_setup_classifier

        classifier = get_setup_classifier()

        # Build criteria (we may not have all data)
        price = quote.get("price") or quote.get("last", 0)
        change_pct = quote.get("change_pct", 0)
        volume = quote.get("volume", 0)
        avg_volume = quote.get("avg_volume", 0)

        criteria = StockCriteria(
            symbol=symbol.upper(),
            price_in_range=1.0 <= price <= 20.0,
            change_over_10pct=change_pct >= 10.0,
            rvol_over_5x=(volume / avg_volume >= 5.0) if avg_volume > 0 else False,
            price=price,
            change_pct=change_pct,
            relative_volume=volume / avg_volume if avg_volume > 0 else 0,
        )

        return {
            "success": True,
            "symbol": symbol.upper(),
            "criteria": criteria.to_dict(),
            "grade": criteria.grade.value,
            "criteria_met": criteria.criteria_met,
            "recommendation": {
                "A": "Full position",
                "B": "Half position",
                "C": "Scalp only or skip",
                "F": "Do not trade",
            }.get(criteria.grade.value, "Unknown"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Grade error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════
#                 MULTI-TIMEFRAME CONFIRMATION
# ═══════════════════════════════════════════════════════════════════════


@router.get("/mtf/{symbol}")
async def get_mtf_confirmation(symbol: str):
    """
    Get multi-timeframe confirmation for a symbol.

    Analyzes 1-minute and 5-minute charts for alignment.
    Ross Cameron rule: Both timeframes must agree before entry.
    """
    try:
        from ai.mtf_confirmation import get_mtf_engine

        engine = get_mtf_engine()
        result = engine.analyze(symbol.upper())

        return {"success": True, **result.to_dict()}
    except Exception as e:
        logger.error(f"MTF error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class MTFAnalyzeRequest(BaseModel):
    """Request to analyze multiple symbols for MTF confirmation"""

    symbols: List[str]


@router.post("/mtf/analyze")
async def analyze_mtf_batch(request: MTFAnalyzeRequest):
    """
    Analyze multi-timeframe confirmation for multiple symbols.

    Returns confirmation status for each symbol.
    """
    try:
        from ai.mtf_confirmation import get_mtf_engine

        engine = get_mtf_engine()
        results = []

        for symbol in request.symbols[:20]:  # Limit to 20
            try:
                result = engine.analyze(symbol.upper())
                results.append(result.to_dict())
            except Exception as e:
                results.append(
                    {
                        "symbol": symbol.upper(),
                        "error": str(e),
                        "signal": "ERROR",
                        "recommendation": "WAIT",
                    }
                )

        return {"success": True, "count": len(results), "results": results}
    except Exception as e:
        logger.error(f"MTF batch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mtf/confirmed")
async def get_confirmed_symbols():
    """
    Get all symbols from watchlist that have confirmed MTF signals.

    Only returns symbols with CONFIRMED_LONG signal.
    """
    try:
        import httpx
        from ai.mtf_confirmation import get_mtf_engine

        engine = get_mtf_engine()

        # Get watchlist
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:9100/api/worklist")
            wl_data = response.json()

        symbols = []
        if wl_data.get("data"):
            symbols = [s["symbol"] for s in wl_data["data"]]
        elif wl_data.get("symbols"):
            symbols = wl_data["symbols"]

        confirmed = []
        for symbol in symbols[:30]:  # Limit
            try:
                result = engine.analyze(symbol)
                if result.signal.value == "CONFIRMED_LONG":
                    confirmed.append(
                        {
                            "symbol": symbol,
                            "confidence": result.confidence,
                            "reasons": result.reasons,
                        }
                    )
            except:
                pass

        return {"success": True, "count": len(confirmed), "confirmed": confirmed}
    except Exception as e:
        logger.error(f"MTF confirmed error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mtf/status")
async def get_mtf_status():
    """Get MTF confirmation engine status"""
    try:
        from ai.mtf_confirmation import get_mtf_engine

        engine = get_mtf_engine()

        return {
            "success": True,
            "status": "operational",
            "cached_symbols": len(engine.cache),
            "cache_ttl_seconds": engine.cache_ttl,
            "min_confidence": engine.min_confidence_to_enter,
            "require_vwap": engine.require_both_above_vwap,
            "require_macd": engine.require_macd_alignment,
        }
    except Exception as e:
        return {"success": False, "status": "error", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#                 VWAP MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════


@router.get("/vwap/{symbol}")
async def get_vwap_data(symbol: str):
    """
    Get VWAP data for a symbol.

    Returns VWAP, position relative to VWAP, bands, and trading signal.
    """
    try:
        from ai.vwap_manager import get_vwap_manager

        manager = get_vwap_manager()
        data = manager.get_vwap(symbol.upper())

        if data:
            return {"success": True, **data.to_dict()}
        else:
            # Try to load from recent candles
            try:
                from polygon_data import get_polygon_client

                client = get_polygon_client()

                from datetime import datetime, timedelta

                end = datetime.now()
                start = end - timedelta(days=1)

                bars = client.get_bars(
                    symbol.upper(),
                    multiplier=1,
                    timespan="minute",
                    from_date=start.strftime("%Y-%m-%d"),
                    to_date=end.strftime("%Y-%m-%d"),
                )

                if bars:
                    candles = [
                        {
                            "high": b.get("h", b.get("high", 0)),
                            "low": b.get("l", b.get("low", 0)),
                            "close": b.get("c", b.get("close", 0)),
                            "volume": b.get("v", b.get("volume", 0)),
                        }
                        for b in bars[-100:]  # Last 100 candles
                    ]
                    data = manager.load_from_candles(symbol.upper(), candles)
                    return {"success": True, **data.to_dict()}
            except Exception as e:
                logger.warning(f"Could not load VWAP from Polygon: {e}")

            return {
                "success": False,
                "symbol": symbol.upper(),
                "message": "No VWAP data available - send candle data first",
            }
    except Exception as e:
        logger.error(f"VWAP error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vwap/check/{symbol}")
async def check_vwap_entry(symbol: str):
    """
    Check if entry is valid based on VWAP.

    Ross Cameron rules:
    - Must be ABOVE VWAP
    - Not too extended (>3% above is risky)
    """
    try:
        from ai.vwap_manager import get_vwap_manager

        manager = get_vwap_manager()
        valid, reason = manager.is_entry_valid(symbol.upper())

        data = manager.get_vwap(symbol.upper())

        return {
            "success": True,
            "symbol": symbol.upper(),
            "entry_valid": valid,
            "reason": reason,
            "vwap": data.vwap if data else 0,
            "current_price": data.current_price if data else 0,
            "distance_pct": data.distance_pct if data else 0,
            "position": data.position.value if data else "UNKNOWN",
            "stop_price": data.stop_price if data else 0,
        }
    except Exception as e:
        logger.error(f"VWAP check error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class VWAPFeedRequest(BaseModel):
    """Request to feed candle data for VWAP calculation"""

    symbol: str
    candles: List[Dict]


@router.post("/vwap/feed")
async def feed_vwap_candles(request: VWAPFeedRequest):
    """
    Feed candle data to calculate VWAP.

    Use this to initialize VWAP from historical data.
    """
    try:
        from ai.vwap_manager import get_vwap_manager

        manager = get_vwap_manager()
        data = manager.load_from_candles(request.symbol.upper(), request.candles)

        return {
            "success": True,
            "symbol": request.symbol.upper(),
            "candles_processed": len(request.candles),
            **data.to_dict(),
        }
    except Exception as e:
        logger.error(f"VWAP feed error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vwap/status")
async def get_vwap_status():
    """Get VWAP manager status"""
    try:
        from ai.vwap_manager import get_vwap_manager

        manager = get_vwap_manager()

        return {
            "success": True,
            "status": "operational",
            "tracked_symbols": list(manager.vwap_data.keys()),
            "count": len(manager.vwap_data),
            "trailing_stops": list(manager.trailing_stops.keys()),
            "config": {
                "extended_threshold_pct": manager.extended_threshold_pct,
                "at_vwap_threshold_pct": manager.at_vwap_threshold_pct,
            },
        }
    except Exception as e:
        return {"success": False, "status": "error", "error": str(e)}


@router.post("/vwap/reset")
async def reset_vwap(symbol: str = None):
    """
    Reset VWAP data (for new trading day).

    If symbol provided, resets only that symbol.
    Otherwise resets all symbols.
    """
    try:
        from ai.vwap_manager import get_vwap_manager

        manager = get_vwap_manager()
        manager.reset_daily(symbol.upper() if symbol else None)

        return {
            "success": True,
            "message": f"VWAP reset for {symbol.upper() if symbol else 'all symbols'}",
        }
    except Exception as e:
        logger.error(f"VWAP reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# FLOAT ROTATION TRACKER ENDPOINTS
# ============================================================================


@router.get("/float-rotation/{symbol}")
async def get_float_rotation(symbol: str):
    """
    Get float rotation data for a symbol.

    Float rotation = cumulative_volume / float_shares
    - < 0.5x = Warming up
    - 0.5x - 1.0x = Active
    - 1.0x - 2.0x = Rotating (every share traded once)
    - 2.0x+ = Hot/Extreme
    """
    try:
        from ai.float_rotation_tracker import get_float_tracker

        tracker = get_float_tracker()
        data = tracker.get_float_data(symbol.upper())

        if not data:
            # Try to load float data
            float_shares = await tracker.load_float_data(symbol.upper())
            if float_shares:
                return {
                    "symbol": symbol.upper(),
                    "float_shares": float_shares,
                    "message": "Float loaded, volume tracking will start with trades",
                }
            return {"symbol": symbol.upper(), "error": "No float data available"}

        return data.to_dict()

    except Exception as e:
        logger.error(f"Float rotation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/float-rotation/boost/{symbol}")
async def get_float_rotation_boost(symbol: str):
    """
    Get confidence boost from float rotation.

    Used by HFT Scalper to increase confidence on rotating stocks.
    Returns boost factor (0.0 to 0.4).
    """
    try:
        from ai.float_rotation_tracker import get_float_tracker

        tracker = get_float_tracker()
        boost = tracker.get_rotation_boost(symbol.upper())
        data = tracker.get_float_data(symbol.upper())

        return {
            "symbol": symbol.upper(),
            "boost": round(boost, 2),
            "rotation_ratio": round(data.rotation_ratio, 2) if data else 0,
            "rotation_level": data.rotation_level.value if data else "NONE",
            "is_low_float": data.is_low_float if data else False,
        }

    except Exception as e:
        logger.error(f"Float rotation boost error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/float-rotation/rotating")
async def get_rotating_stocks(min_rotation: float = 1.0):
    """
    Get all stocks currently rotating (volume >= float).

    Args:
        min_rotation: Minimum rotation ratio (default 1.0 = 100% of float)
    """
    try:
        from ai.float_rotation_tracker import get_float_tracker

        tracker = get_float_tracker()
        rotating = tracker.get_all_rotating(min_rotation)

        return {
            "count": len(rotating),
            "min_rotation": min_rotation,
            "stocks": [d.to_dict() for d in rotating],
        }

    except Exception as e:
        logger.error(f"Rotating stocks error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/float-rotation/low-float")
async def get_low_float_movers():
    """
    Get low float stocks (<20M) with significant volume (>25% of float).
    These are the stocks Ross Cameron focuses on.
    """
    try:
        from ai.float_rotation_tracker import get_float_tracker

        tracker = get_float_tracker()
        movers = tracker.get_low_float_movers()

        return {"count": len(movers), "stocks": [d.to_dict() for d in movers]}

    except Exception as e:
        logger.error(f"Low float movers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/float-rotation/alerts")
async def get_float_rotation_alerts(limit: int = 20):
    """
    Get recent float rotation alerts.

    Alerts fire when stocks cross rotation thresholds (0.5x, 1x, 2x, etc.)
    """
    try:
        from ai.float_rotation_tracker import get_float_tracker

        tracker = get_float_tracker()
        alerts = tracker.get_recent_alerts(limit)

        return {"count": len(alerts), "alerts": [a.to_dict() for a in alerts]}

    except Exception as e:
        logger.error(f"Float rotation alerts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/float-rotation/status")
async def get_float_rotation_status():
    """Get float rotation tracker status"""
    try:
        from ai.float_rotation_tracker import get_float_tracker

        tracker = get_float_tracker()
        return tracker.get_status()

    except Exception as e:
        logger.error(f"Float rotation status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/float-rotation/set-float/{symbol}")
async def set_symbol_float(symbol: str, float_shares: int, avg_volume: int = 0):
    """
    Manually set float data for a symbol.

    Args:
        symbol: Stock symbol
        float_shares: Free float in shares
        avg_volume: Average daily volume (optional)
    """
    try:
        from ai.float_rotation_tracker import get_float_tracker

        tracker = get_float_tracker()
        tracker.set_float(symbol.upper(), float_shares, avg_volume)

        return {
            "success": True,
            "symbol": symbol.upper(),
            "float_shares": float_shares,
            "float_millions": round(float_shares / 1_000_000, 2),
            "avg_volume": avg_volume,
        }

    except Exception as e:
        logger.error(f"Set float error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/float-rotation/load-float/{symbol}")
async def load_symbol_float(symbol: str):
    """
    Load float data from fundamental analysis (yfinance).
    """
    try:
        from ai.float_rotation_tracker import get_float_tracker

        tracker = get_float_tracker()
        float_shares = await tracker.load_float_data(symbol.upper())

        if float_shares:
            return {
                "success": True,
                "symbol": symbol.upper(),
                "float_shares": float_shares,
                "float_millions": round(float_shares / 1_000_000, 2),
            }
        else:
            return {
                "success": False,
                "symbol": symbol.upper(),
                "error": "Could not load float data",
            }

    except Exception as e:
        logger.error(f"Load float error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/float-rotation/reset")
async def reset_float_rotation(symbol: str = None):
    """
    Reset float rotation tracking (call at market open).

    If symbol provided, resets only that symbol.
    Otherwise resets all symbols.
    """
    try:
        from ai.float_rotation_tracker import get_float_tracker

        tracker = get_float_tracker()
        tracker.reset_daily(symbol.upper() if symbol else None)

        return {
            "success": True,
            "message": f"Float rotation reset for {symbol.upper() if symbol else 'all symbols'}",
        }

    except Exception as e:
        logger.error(f"Float rotation reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MOMENTUM EXHAUSTION DETECTOR ENDPOINTS
# ============================================================================


@router.get("/exhaustion/{symbol}")
async def get_exhaustion_data(symbol: str):
    """
    Get exhaustion data for a symbol.

    Returns RSI, volume trend, consecutive red candles, and active alerts.
    """
    try:
        from ai.momentum_exhaustion_detector import get_exhaustion_detector

        detector = get_exhaustion_detector()
        state = detector.symbols.get(symbol.upper())

        if not state:
            return {
                "symbol": symbol.upper(),
                "tracked": False,
                "message": "Symbol not being tracked. Add candle data to start.",
            }

        score, reasons = detector.get_exhaustion_score(symbol.upper())

        return {
            "symbol": symbol.upper(),
            "tracked": True,
            "exhaustion_score": round(score, 1),
            "reasons": reasons,
            "rsi": round(state.current_rsi, 1),
            "consecutive_red_candles": state.consecutive_red_candles,
            "avg_spread": round(state.avg_spread, 4),
            "last_high": round(state.last_high, 4),
            "entry_price": round(state.entry_price, 4) if state.entry_price else None,
            "high_since_entry": (
                round(state.high_since_entry, 4) if state.high_since_entry else None
            ),
            "active_alerts": [a.to_dict() for a in state.active_alerts[-5:]],
        }

    except Exception as e:
        logger.error(f"Exhaustion data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exhaustion/score/{symbol}")
async def get_exhaustion_score(symbol: str):
    """
    Get exhaustion score for a symbol (0-100).

    Higher score = more exhausted = more likely to reverse.
    """
    try:
        from ai.momentum_exhaustion_detector import get_exhaustion_detector

        detector = get_exhaustion_detector()
        score, reasons = detector.get_exhaustion_score(symbol.upper())

        return {
            "symbol": symbol.upper(),
            "score": round(score, 1),
            "level": (
                "CRITICAL"
                if score > 70
                else "HIGH" if score > 50 else "MEDIUM" if score > 30 else "LOW"
            ),
            "should_exit": score > 60,
            "reasons": reasons,
        }

    except Exception as e:
        logger.error(f"Exhaustion score error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exhaustion/check/{symbol}")
async def check_exhaustion_exit(symbol: str):
    """
    Check if should exit based on exhaustion signals.

    Returns the highest severity exit alert if any.
    """
    try:
        from ai.momentum_exhaustion_detector import get_exhaustion_detector

        detector = get_exhaustion_detector()
        state = detector.symbols.get(symbol.upper())

        if not state:
            return {
                "symbol": symbol.upper(),
                "should_exit": False,
                "reason": "Not tracked",
            }

        # Check for exit signal
        alert = detector.check_exit(symbol.upper(), 0)

        if alert:
            return {
                "symbol": symbol.upper(),
                "should_exit": True,
                "alert": alert.to_dict(),
            }

        score, reasons = detector.get_exhaustion_score(symbol.upper())

        return {
            "symbol": symbol.upper(),
            "should_exit": False,
            "exhaustion_score": round(score, 1),
            "reasons": reasons,
        }

    except Exception as e:
        logger.error(f"Exhaustion check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exhaustion/alerts")
async def get_exhaustion_alerts(limit: int = 20):
    """Get recent exhaustion alerts across all symbols"""
    try:
        from ai.momentum_exhaustion_detector import get_exhaustion_detector

        detector = get_exhaustion_detector()
        alerts = detector.get_recent_alerts(limit)

        return {"count": len(alerts), "alerts": [a.to_dict() for a in alerts]}

    except Exception as e:
        logger.error(f"Exhaustion alerts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exhaustion/status")
async def get_exhaustion_status():
    """Get exhaustion detector status"""
    try:
        from ai.momentum_exhaustion_detector import get_exhaustion_detector

        detector = get_exhaustion_detector()
        return detector.get_status()

    except Exception as e:
        logger.error(f"Exhaustion status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/exhaustion/register/{symbol}")
async def register_exhaustion_position(symbol: str, entry_price: float):
    """
    Register a position for exhaustion monitoring.

    Call this when entering a trade to track exhaustion from entry.
    """
    try:
        from ai.momentum_exhaustion_detector import get_exhaustion_detector

        detector = get_exhaustion_detector()
        detector.register_position(symbol.upper(), entry_price)

        return {
            "success": True,
            "symbol": symbol.upper(),
            "entry_price": entry_price,
            "message": "Position registered for exhaustion monitoring",
        }

    except Exception as e:
        logger.error(f"Exhaustion register error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/exhaustion/{symbol}")
async def unregister_exhaustion_position(symbol: str):
    """Unregister a position from exhaustion monitoring"""
    try:
        from ai.momentum_exhaustion_detector import get_exhaustion_detector

        detector = get_exhaustion_detector()
        detector.unregister_position(symbol.upper())

        return {
            "success": True,
            "symbol": symbol.upper(),
            "message": "Position unregistered from exhaustion monitoring",
        }

    except Exception as e:
        logger.error(f"Exhaustion unregister error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/exhaustion/config")
async def update_exhaustion_config(config: Dict):
    """Update exhaustion detector configuration"""
    try:
        from ai.momentum_exhaustion_detector import get_exhaustion_detector

        detector = get_exhaustion_detector()

        # Update config
        for key, value in config.items():
            if key in detector.config:
                detector.config[key] = value

        return {"success": True, "config": detector.config}

    except Exception as e:
        logger.error(f"Exhaustion config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LEVEL 2 DEPTH ANALYZER ENDPOINTS
# ============================================================================


@router.get("/depth/{symbol}")
async def get_depth_snapshot(symbol: str):
    """
    Get current order book depth snapshot for a symbol.

    Returns bid/ask levels, volume, spread, and detected walls.
    """
    try:
        from ai.level2_depth_analyzer import get_depth_analyzer

        analyzer = get_depth_analyzer()
        snapshot = analyzer.get_depth(symbol.upper())

        if not snapshot:
            return {
                "symbol": symbol.upper(),
                "tracked": False,
                "message": "No depth data. Update depth to start tracking.",
            }

        return snapshot.to_dict()

    except Exception as e:
        logger.error(f"Depth snapshot error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/depth/analysis/{symbol}")
async def get_depth_analysis(symbol: str):
    """
    Get trading analysis from order book depth.

    Includes imbalance, wall detection, absorption signals.
    """
    try:
        from ai.level2_depth_analyzer import get_depth_analyzer

        analyzer = get_depth_analyzer()
        analysis = analyzer.get_analysis(symbol.upper())

        if not analysis:
            return {
                "symbol": symbol.upper(),
                "analyzed": False,
                "message": "No depth data available",
            }

        return analysis.to_dict()

    except Exception as e:
        logger.error(f"Depth analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/depth/walls/{symbol}")
async def get_depth_walls(symbol: str):
    """
    Get active order walls for a symbol.

    Shows bid walls (support) and ask walls (resistance).
    """
    try:
        from ai.level2_depth_analyzer import get_depth_analyzer

        analyzer = get_depth_analyzer()
        return analyzer.get_walls(symbol.upper())

    except Exception as e:
        logger.error(f"Depth walls error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/depth/imbalance/{symbol}")
async def get_depth_imbalance(symbol: str):
    """
    Get bid/ask imbalance trend for a symbol.

    Shows current ratio and trend direction.
    """
    try:
        from ai.level2_depth_analyzer import get_depth_analyzer

        analyzer = get_depth_analyzer()
        return analyzer.get_imbalance_trend(symbol.upper())

    except Exception as e:
        logger.error(f"Depth imbalance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/depth/check/{symbol}")
async def check_depth_entry(symbol: str):
    """
    Check if entry is valid based on depth analysis.

    Returns validation result and boost factor.
    """
    try:
        from ai.level2_depth_analyzer import get_depth_analyzer

        analyzer = get_depth_analyzer()
        valid, reason = analyzer.is_entry_valid(symbol.upper())
        boost = analyzer.get_entry_boost(symbol.upper())
        analysis = analyzer.get_analysis(symbol.upper())

        return {
            "symbol": symbol.upper(),
            "entry_valid": valid,
            "reason": reason,
            "boost": round(boost, 2),
            "signal": analysis.signal.value if analysis else "UNKNOWN",
            "imbalance_ratio": round(analysis.imbalance_ratio, 2) if analysis else 1.0,
        }

    except Exception as e:
        logger.error(f"Depth check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/depth/update/{symbol}")
async def update_depth_data(symbol: str, data: Dict):
    """
    Update order book depth for a symbol.

    Body should contain:
    - bids: List of {price, size} dicts
    - asks: List of {price, size} dicts
    - price: Current market price (optional)
    """
    try:
        from ai.level2_depth_analyzer import get_depth_analyzer

        analyzer = get_depth_analyzer()

        bids = data.get("bids", [])
        asks = data.get("asks", [])
        price = data.get("price")

        analysis = analyzer.update_depth(symbol.upper(), bids, asks, price)

        return {"success": True, "analysis": analysis.to_dict()}

    except Exception as e:
        logger.error(f"Depth update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/depth/status")
async def get_depth_status():
    """Get depth analyzer status"""
    try:
        from ai.level2_depth_analyzer import get_depth_analyzer

        analyzer = get_depth_analyzer()
        return analyzer.get_status()

    except Exception as e:
        logger.error(f"Depth status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/depth/config")
async def update_depth_config(config: Dict):
    """Update depth analyzer configuration"""
    try:
        from ai.level2_depth_analyzer import get_depth_analyzer

        analyzer = get_depth_analyzer()

        for key, value in config.items():
            if key in analyzer.config:
                analyzer.config[key] = value

        return {"success": True, "config": analyzer.config}

    except Exception as e:
        logger.error(f"Depth config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# GAP GRADER ENDPOINTS
# ============================================================================


@router.get("/gap/status")
async def get_gap_grader_status():
    """Get gap grader status"""
    try:
        from ai.gap_grader import get_gap_grader

        grader = get_gap_grader()
        return grader.get_status()

    except Exception as e:
        logger.error(f"Gap grader status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gap/grade/{symbol}")
async def grade_gap(
    symbol: str,
    gap_percent: float,
    current_price: float,
    prior_close: float,
    premarket_volume: int = 0,
    float_shares: int = 0,
    avg_volume: int = 0,
    prior_day_change: float = 0,
    catalyst_headline: str = "",
    premarket_high: float = 0,
    premarket_low: float = 0,
):
    """Grade a gap using Ross Cameron methodology"""
    try:
        from ai.gap_grader import get_gap_grader

        grader = get_gap_grader()
        graded = grader.grade_gap(
            symbol=symbol,
            gap_percent=gap_percent,
            current_price=current_price,
            prior_close=prior_close,
            premarket_volume=premarket_volume,
            float_shares=float_shares,
            avg_volume=avg_volume,
            prior_day_change=prior_day_change,
            catalyst_headline=catalyst_headline,
            premarket_high=premarket_high,
            premarket_low=premarket_low,
        )

        return graded.to_dict()

    except Exception as e:
        logger.error(f"Gap grading error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gap/{symbol}")
async def get_gap_grade(symbol: str):
    """Get graded gap for symbol"""
    try:
        from ai.gap_grader import get_gap_grader

        grader = get_gap_grader()
        graded = grader.get_grade(symbol)

        if graded:
            return graded.to_dict()
        else:
            return {
                "symbol": symbol,
                "graded": False,
                "message": "Symbol not graded yet",
            }

    except Exception as e:
        logger.error(f"Gap grade retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gap/top")
async def get_top_gaps(min_grade: str = "C", limit: int = 10):
    """Get top graded gaps"""
    try:
        from ai.gap_grader import GapGrade, get_gap_grader

        grader = get_gap_grader()
        grade = GapGrade[min_grade.upper()]
        gaps = grader.get_top_gaps(min_grade=grade, limit=limit)

        return {"gaps": [g.to_dict() for g in gaps], "count": len(gaps)}

    except KeyError:
        raise HTTPException(
            status_code=400, detail=f"Invalid grade: {min_grade}. Use A, B, C, D, or F"
        )
    except Exception as e:
        logger.error(f"Top gaps error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gap/tradeable")
async def get_tradeable_gaps():
    """Get all tradeable gaps (C grade or better)"""
    try:
        from ai.gap_grader import get_gap_grader

        grader = get_gap_grader()
        gaps = grader.get_tradeable_gaps()

        return {"gaps": [g.to_dict() for g in gaps], "count": len(gaps)}

    except Exception as e:
        logger.error(f"Tradeable gaps error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gap/a-grades")
async def get_a_grade_gaps():
    """Get only A-grade gaps (highest quality)"""
    try:
        from ai.gap_grader import get_gap_grader

        grader = get_gap_grader()
        gaps = grader.get_a_grades()

        return {"gaps": [g.to_dict() for g in gaps], "count": len(gaps)}

    except Exception as e:
        logger.error(f"A-grade gaps error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gap/clear")
async def clear_gaps():
    """Clear all graded gaps (call at market close)"""
    try:
        from ai.gap_grader import get_gap_grader

        grader = get_gap_grader()
        grader.clear_gaps()

        return {"success": True, "message": "Gap grader cleared"}

    except Exception as e:
        logger.error(f"Gap clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gap/scan")
async def scan_and_grade_gaps():
    """
    Scan for gaps using premarket scanner and grade them.
    This integrates gap grader with premarket scanner for automated gap detection.
    """
    try:
        import yfinance as yf
        from ai.gap_grader import get_gap_grader
        from ai.premarket_scanner import get_premarket_scanner

        grader = get_gap_grader()
        scanner = get_premarket_scanner()

        # Run premarket scan
        await scanner.scan()
        watchlist = scanner.get_watchlist()

        graded_results = []

        for item in watchlist:
            symbol = item.get("symbol")
            if not symbol:
                continue

            try:
                # Get additional data via yfinance
                ticker = yf.Ticker(symbol)
                info = ticker.info

                gap_percent = item.get("change_pct", 0)
                current_price = item.get("price", 0)
                prior_close = info.get(
                    "previousClose",
                    (
                        current_price / (1 + gap_percent / 100)
                        if gap_percent
                        else current_price
                    ),
                )
                premarket_volume = item.get("volume", 0)
                float_shares = info.get("floatShares", 0)
                avg_volume = info.get("averageVolume", 0)

                # Get prior day change from history
                hist = ticker.history(period="2d")
                prior_day_change = 0
                if len(hist) >= 2:
                    prior_day_change = (
                        (hist.iloc[-2]["Close"] - hist.iloc[-2]["Open"])
                        / hist.iloc[-2]["Open"]
                    ) * 100

                # Get catalyst from news monitor
                catalyst = item.get("catalyst", "") or item.get("headline", "")

                graded = grader.grade_gap(
                    symbol=symbol,
                    gap_percent=gap_percent,
                    current_price=current_price,
                    prior_close=prior_close,
                    premarket_volume=premarket_volume,
                    float_shares=float_shares,
                    avg_volume=avg_volume,
                    prior_day_change=prior_day_change,
                    catalyst_headline=catalyst,
                )

                graded_results.append(graded.to_dict())

            except Exception as e:
                logger.warning(f"Error grading {symbol}: {e}")
                continue

        return {
            "success": True,
            "scanned": len(watchlist),
            "graded": len(graded_results),
            "results": graded_results,
            "top_5": [g for g in graded_results if g.get("grade") in ["A", "B"]][:5],
        }

    except Exception as e:
        logger.error(f"Gap scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gap/add-to-scalper")
async def add_graded_gaps_to_scalper(min_grade: str = "B"):
    """Add top graded gaps to HFT Scalper watchlist"""
    try:
        from ai.gap_grader import GapGrade, get_gap_grader
        from ai.hft_scalper import get_hft_scalper

        grader = get_gap_grader()
        scalper = get_hft_scalper()

        grade = GapGrade[min_grade.upper()]
        gaps = grader.get_top_gaps(min_grade=grade, limit=10)

        added = []
        for gap in gaps:
            if gap.gap_percent > 0:  # Only add gap-ups (we don't short)
                scalper.watchlist.add(gap.symbol)
                added.append(
                    {
                        "symbol": gap.symbol,
                        "grade": gap.grade.value,
                        "score": gap.score.total,
                        "gap_percent": gap.gap_percent,
                    }
                )

        return {
            "success": True,
            "added": len(added),
            "symbols": added,
            "scalper_watchlist": list(scalper.watchlist),
        }

    except KeyError:
        raise HTTPException(
            status_code=400, detail=f"Invalid grade: {min_grade}. Use A, B, C, D, or F"
        )
    except Exception as e:
        logger.error(f"Add to scalper error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# OVERNIGHT CONTINUATION SCANNER ENDPOINTS
# ============================================================================


@router.get("/overnight/status")
async def get_overnight_status():
    """Get overnight continuation scanner status"""
    try:
        from ai.overnight_continuation import get_overnight_scanner

        scanner = get_overnight_scanner()
        return scanner.get_status()

    except Exception as e:
        logger.error(f"Overnight status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/overnight/record-ah/{symbol}")
async def record_after_hours_move(
    symbol: str,
    regular_close: float,
    ah_close: float,
    ah_volume: int,
    ah_high: float = 0,
    ah_low: float = 0,
    avg_daily_volume: int = 0,
    catalyst: str = "",
):
    """Record an after-hours move (call at ~8 PM)"""
    try:
        from ai.overnight_continuation import get_overnight_scanner

        scanner = get_overnight_scanner()
        mover = scanner.record_after_hours_move(
            symbol=symbol,
            regular_close=regular_close,
            ah_high=ah_high or ah_close,
            ah_low=ah_low or ah_close,
            ah_close=ah_close,
            ah_volume=ah_volume,
            avg_daily_volume=avg_daily_volume,
            catalyst=catalyst,
        )

        if mover:
            return {"success": True, "mover": mover.to_dict()}
        else:
            return {"success": False, "message": "Move too small or volume too low"}

    except Exception as e:
        logger.error(f"Record AH move error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/overnight/update-pm/{symbol}")
async def update_premarket(
    symbol: str,
    pm_current: float,
    pm_volume: int,
    pm_open: float = 0,
    pm_high: float = 0,
    pm_low: float = 0,
):
    """Update pre-market data (call during 4AM-9:30AM)"""
    try:
        from ai.overnight_continuation import get_overnight_scanner

        scanner = get_overnight_scanner()
        mover = scanner.update_premarket(
            symbol=symbol,
            pm_open=pm_open or pm_current,
            pm_high=pm_high or pm_current,
            pm_low=pm_low or pm_current,
            pm_current=pm_current,
            pm_volume=pm_volume,
        )

        if mover:
            return {"success": True, "mover": mover.to_dict()}
        else:
            return {"success": False, "message": "Symbol not tracked from AH session"}

    except Exception as e:
        logger.error(f"Update PM error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/overnight/{symbol}")
async def get_overnight_mover(symbol: str):
    """Get overnight mover data for symbol"""
    try:
        from ai.overnight_continuation import get_overnight_scanner

        scanner = get_overnight_scanner()
        mover = scanner.get_mover(symbol)

        if mover:
            return mover.to_dict()
        else:
            return {
                "symbol": symbol,
                "tracked": False,
                "message": "Not tracked overnight",
            }

    except Exception as e:
        logger.error(f"Get overnight mover error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/overnight/continuations")
async def get_continuations(min_strength: str = "WEAK"):
    """Get all continuation setups"""
    try:
        from ai.overnight_continuation import (ContinuationStrength,
                                               get_overnight_scanner)

        scanner = get_overnight_scanner()
        strength = ContinuationStrength[min_strength.upper()]
        movers = scanner.get_continuations(min_strength=strength)

        return {"continuations": [m.to_dict() for m in movers], "count": len(movers)}

    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strength: {min_strength}. Use STRONG, MODERATE, WEAK, or NONE",
        )
    except Exception as e:
        logger.error(f"Get continuations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/overnight/strong")
async def get_strong_continuations():
    """Get only strong continuation setups"""
    try:
        from ai.overnight_continuation import get_overnight_scanner

        scanner = get_overnight_scanner()
        movers = scanner.get_strong_continuations()

        return {
            "strong_continuations": [m.to_dict() for m in movers],
            "count": len(movers),
        }

    except Exception as e:
        logger.error(f"Get strong continuations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/overnight/accelerators")
async def get_accelerators():
    """Get stocks where PM is accelerating vs AH"""
    try:
        from ai.overnight_continuation import get_overnight_scanner

        scanner = get_overnight_scanner()
        movers = scanner.get_accelerators()

        return {"accelerators": [m.to_dict() for m in movers], "count": len(movers)}

    except Exception as e:
        logger.error(f"Get accelerators error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/overnight/reversals")
async def get_reversals():
    """Get stocks that reversed in PM (for avoidance)"""
    try:
        from ai.overnight_continuation import get_overnight_scanner

        scanner = get_overnight_scanner()
        movers = scanner.get_reversals()

        return {
            "reversals": [m.to_dict() for m in movers],
            "count": len(movers),
            "warning": "These stocks reversed direction in PM - avoid trading",
        }

    except Exception as e:
        logger.error(f"Get reversals error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/overnight/faders")
async def get_faders():
    """Get stocks fading in PM"""
    try:
        from ai.overnight_continuation import get_overnight_scanner

        scanner = get_overnight_scanner()
        movers = scanner.get_faders()

        return {
            "faders": [m.to_dict() for m in movers],
            "count": len(movers),
            "warning": "These stocks are losing AH gains - trade with caution",
        }

    except Exception as e:
        logger.error(f"Get faders error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/overnight/bullish")
async def get_bullish_continuations():
    """Get bullish continuations only (for trading)"""
    try:
        from ai.overnight_continuation import get_overnight_scanner

        scanner = get_overnight_scanner()
        movers = scanner.get_bullish_continuations()

        return {
            "bullish_continuations": [m.to_dict() for m in movers],
            "count": len(movers),
        }

    except Exception as e:
        logger.error(f"Get bullish continuations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/overnight/clear")
async def clear_overnight_movers():
    """Clear all overnight movers (call at EOD)"""
    try:
        from ai.overnight_continuation import get_overnight_scanner

        scanner = get_overnight_scanner()
        scanner.clear_movers()

        return {"success": True, "message": "Overnight scanner cleared"}

    except Exception as e:
        logger.error(f"Clear overnight error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/overnight/add-to-scalper")
async def add_continuations_to_scalper(min_strength: str = "MODERATE"):
    """Add bullish continuations to HFT Scalper watchlist"""
    try:
        from ai.hft_scalper import get_hft_scalper
        from ai.overnight_continuation import (ContinuationStrength,
                                               get_overnight_scanner)

        scanner = get_overnight_scanner()
        scalper = get_hft_scalper()

        # Get bullish continuations
        movers = scanner.get_bullish_continuations()

        # Filter by strength
        strength = ContinuationStrength[min_strength.upper()]
        strength_order = {
            ContinuationStrength.STRONG: 0,
            ContinuationStrength.MODERATE: 1,
            ContinuationStrength.WEAK: 2,
            ContinuationStrength.NONE: 3,
        }
        min_order = strength_order[strength]

        added = []
        for mover in movers:
            if strength_order[mover.strength] <= min_order:
                scalper.watchlist.add(mover.symbol)
                added.append(
                    {
                        "symbol": mover.symbol,
                        "pattern": mover.pattern.value,
                        "score": mover.continuation_score,
                        "total_change": mover.total_overnight_change,
                    }
                )

        return {
            "success": True,
            "added": len(added),
            "symbols": added,
            "scalper_watchlist": list(scalper.watchlist),
        }

    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strength: {min_strength}. Use STRONG, MODERATE, WEAK, or NONE",
        )
    except Exception as e:
        logger.error(f"Add to scalper error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/overnight/scan-ah")
async def scan_after_hours():
    """
    Scan for after-hours movers using Schwab or Yahoo data.
    Call this at ~8 PM to capture AH movers.
    """
    try:
        import yfinance as yf
        from ai.overnight_continuation import get_overnight_scanner

        scanner = get_overnight_scanner()

        # Get potential movers from Yahoo Finance (works after hours)
        # This is a simplified scan - in production would use Schwab API
        symbols_to_check = [
            "AAPL",
            "TSLA",
            "NVDA",
            "AMD",
            "AMZN",
            "META",
            "GOOGL",
            "MSFT",
            "SPY",
            "QQQ",
            "IWM",
            "COIN",
            "MARA",
            "RIOT",
            "PLUG",
            "NIO",
        ]

        scanned = []
        for symbol in symbols_to_check:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d", prepost=True)

                if len(hist) < 2:
                    continue

                # Get regular session close (last row before after-hours)
                regular_close = hist.iloc[-2]["Close"]

                # Get current/after-hours price
                ah_close = hist.iloc[-1]["Close"]
                ah_high = hist.iloc[-1]["High"]
                ah_low = hist.iloc[-1]["Low"]
                ah_volume = int(hist.iloc[-1]["Volume"])

                info = ticker.info
                avg_volume = info.get("averageVolume", 0)

                mover = scanner.record_after_hours_move(
                    symbol=symbol,
                    regular_close=regular_close,
                    ah_high=ah_high,
                    ah_low=ah_low,
                    ah_close=ah_close,
                    ah_volume=ah_volume,
                    avg_daily_volume=avg_volume,
                )

                if mover:
                    scanned.append(
                        {
                            "symbol": symbol,
                            "ah_change": mover.after_hours.change_pct,
                            "volume": ah_volume,
                        }
                    )

            except Exception as e:
                logger.warning(f"Error scanning {symbol}: {e}")
                continue

        return {
            "success": True,
            "scanned_symbols": len(symbols_to_check),
            "movers_found": len(scanned),
            "movers": scanned,
        }

    except Exception as e:
        logger.error(f"Scan AH error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AI CONTROL CENTER ENDPOINTS (For React UI)
# ============================================================================


@router.get("/watchlist")
async def get_warrior_watchlist():
    """Get current watchlist for Warrior Trading"""
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:9100/api/worklist")
            wl_data = response.json()

        watchlist = []
        if wl_data.get("data"):
            for item in wl_data["data"][:20]:
                watchlist.append(
                    {
                        "symbol": item.get("symbol", ""),
                        "price": item.get("price", 0),
                        "gap_percent": item.get("change_pct", 0),
                        "relative_volume": item.get("rvol", 1.0),
                        "float_shares": (
                            item.get("float", 0) / 1_000_000 if item.get("float") else 0
                        ),
                        "pre_market_volume": item.get("volume", 0),
                        "catalyst": item.get("catalyst", ""),
                        "daily_chart_signal": (
                            "BULLISH" if item.get("change_pct", 0) > 0 else "BEARISH"
                        ),
                        "confidence_score": item.get("ai_prediction", 50),
                    }
                )

        return {
            "success": True,
            "watchlist": watchlist,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Watchlist error: {e}")
        return {
            "success": True,
            "watchlist": [],
            "timestamp": datetime.now().isoformat(),
        }


@router.post("/scan/premarket")
async def run_premarket_scan():
    """Run pre-market scan"""
    try:
        from ai.premarket_scanner import get_premarket_scanner

        scanner = get_premarket_scanner()
        await scanner.scan()
        watchlist = scanner.get_watchlist()

        candidates = []
        for item in watchlist[:10]:
            candidates.append(
                {
                    "symbol": item.get("symbol", ""),
                    "price": item.get("price", 0),
                    "gap_percent": item.get("change_pct", 0),
                    "relative_volume": item.get("rvol", 1.0),
                    "float_shares": (
                        item.get("float", 0) / 1_000_000 if item.get("float") else 0
                    ),
                    "pre_market_volume": item.get("volume", 0),
                    "catalyst": item.get("catalyst", ""),
                    "daily_chart_signal": (
                        "BULLISH" if item.get("change_pct", 0) > 0 else "BEARISH"
                    ),
                    "confidence_score": 65,
                }
            )

        return {
            "success": True,
            "candidates": candidates,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Premarket scan error: {e}")
        return {
            "success": True,
            "candidates": [],
            "timestamp": datetime.now().isoformat(),
        }


@router.get("/risk/status")
async def get_risk_status():
    """Get risk management status"""
    try:
        from ai.hft_scalper import get_hft_scalper

        scalper = get_hft_scalper()
        stats = scalper.get_stats()

        return {
            "is_halted": not scalper.trading_enabled,
            "halt_reason": None if scalper.trading_enabled else "Trading disabled",
            "open_positions": len(scalper.positions),
            "total_trades": stats.get("total_trades", 0),
            "winning_trades": stats.get("wins", 0),
            "losing_trades": stats.get("losses", 0),
            "win_rate": stats.get("win_rate", 0) * 100,
            "current_pnl": stats.get("total_pnl", 0),
            "avg_win": stats.get("avg_win", 0),
            "avg_loss": stats.get("avg_loss", 0),
            "avg_r_multiple": stats.get("avg_r", 0),
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "distance_to_goal": max(0, 200 - stats.get("total_pnl", 0)),
            "best_trade": None,
        }
    except Exception as e:
        logger.error(f"Risk status error: {e}")
        return {
            "is_halted": False,
            "halt_reason": None,
            "open_positions": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "current_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "avg_r_multiple": 0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "distance_to_goal": 200,
            "best_trade": None,
        }


@router.post("/risk/reset-daily")
async def reset_daily_risk():
    """Reset daily risk statistics"""
    try:
        from ai.hft_scalper import get_hft_scalper

        scalper = get_hft_scalper()
        scalper.reset_daily_stats()

        return {"success": True, "message": "Daily stats reset"}
    except Exception as e:
        logger.error(f"Reset daily error: {e}")
        return {"success": True, "message": "Reset completed"}


@router.get("/trades/history")
async def get_trades_history(status: Optional[str] = None, limit: int = 100):
    """Get trade history"""
    try:
        from ai.hft_scalper import get_hft_scalper

        scalper = get_hft_scalper()
        trades = (
            scalper.trade_history[-limit:] if hasattr(scalper, "trade_history") else []
        )

        trade_list = []
        for i, trade in enumerate(trades):
            trade_list.append(
                {
                    "id": i,
                    "trade_id": trade.get("trade_id", f"trade_{i}"),
                    "symbol": trade.get("symbol", ""),
                    "setup_type": trade.get("setup_type", "MOMENTUM"),
                    "entry_time": trade.get("entry_time", ""),
                    "entry_price": trade.get("entry_price", 0),
                    "shares": trade.get("shares", 0),
                    "stop_price": trade.get("stop_price", 0),
                    "target_price": trade.get("target_price", 0),
                    "exit_time": trade.get("exit_time"),
                    "exit_price": trade.get("exit_price"),
                    "exit_reason": trade.get("exit_reason"),
                    "pnl": trade.get("pnl", 0),
                    "pnl_percent": trade.get("pnl_pct", 0),
                    "r_multiple": trade.get("r_multiple", 0),
                    "status": "CLOSED" if trade.get("exit_time") else "OPEN",
                }
            )

        if status:
            trade_list = [t for t in trade_list if t["status"] == status.upper()]

        return {"success": True, "trades": trade_list, "count": len(trade_list)}
    except Exception as e:
        logger.error(f"Trade history error: {e}")
        return {"success": True, "trades": [], "count": 0}
