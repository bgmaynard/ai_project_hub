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
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

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
        return {
            "success": True,
            "status": detector.get_status()
        }
    except Exception as e:
        logger.error(f"Status error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/signal/{symbol}")
async def get_signal(symbol: str):
    """Get trading signal for a symbol"""
    try:
        from ai.warrior_setup_detector import get_trading_signal
        signal = await get_trading_signal(symbol.upper())
        return {
            "success": True,
            "signal": signal
        }
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
                results.append({
                    "symbol": symbol,
                    "error": str(e)
                })

        # Sort by setup grade and confidence
        grade_order = {"A+": 0, "A": 1, "B": 2, "C": 3, "F": 4}
        results.sort(key=lambda x: (
            grade_order.get(x.get('setup', {}).get('grade', 'F'), 4),
            -x.get('setup', {}).get('confidence', 0)
        ))

        return {
            "success": True,
            "count": len(results),
            "signals": results
        }
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
            "patterns": {
                name: p.to_dict() for name, p in patterns.items()
            },
            "best_pattern": detector.get_best_setup(symbol.upper()).to_dict()
                           if detector.get_best_setup(symbol.upper()) else None
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
            "flush_detection": flush
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
            return {
                "success": True,
                "symbol": symbol.upper(),
                "luld": status
            }
        else:
            return {
                "success": True,
                "symbol": symbol.upper(),
                "luld": None,
                "message": "No LULD bands calculated. Use POST /api/warrior/luld to calculate."
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
            request.symbol.upper(),
            request.reference_price,
            request.tier
        )

        return {
            "success": True,
            "symbol": request.symbol.upper(),
            "bands": bands.to_dict()
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
                sym: bands.to_dict()
                for sym, bands in detector.luld_bands.items()
            }
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
            "trades_added": len(request.trades)
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
            "candles_added": len(request.candles)
        }
    except Exception as e:
        logger.error(f"Candle feed error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/setup-rules/{setup_type}")
async def get_setup_rules(setup_type: str):
    """Get entry rules for a specific setup type"""
    try:
        from ai.setup_classifier import get_setup_classifier, SetupType

        classifier = get_setup_classifier()

        # Map string to SetupType
        setup_map = {
            'bull_flag': SetupType.BULL_FLAG,
            'abcd': SetupType.ABCD,
            'micro_pullback': SetupType.MICRO_PULLBACK,
            'hod_break': SetupType.HOD_BREAK,
            'vwap_breakout': SetupType.VWAP_BREAKOUT,
            'dip_buy': SetupType.DIP_BUY,
            'halt_resume': SetupType.HALT_RESUME
        }

        setup = setup_map.get(setup_type.lower())
        if not setup:
            return {
                "success": False,
                "error": f"Unknown setup type: {setup_type}",
                "valid_types": list(setup_map.keys())
            }

        rules = classifier.get_entry_rules(setup)

        return {
            "success": True,
            "setup_type": setup.value,
            "rules": rules
        }
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
                f"http://localhost:9100/api/price/{symbol.upper()}",
                timeout=5.0
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=404, detail=f"Price not found for {symbol}")

            quote = resp.json()

        from ai.setup_classifier import get_setup_classifier, StockCriteria

        classifier = get_setup_classifier()

        # Build criteria (we may not have all data)
        price = quote.get('price') or quote.get('last', 0)
        change_pct = quote.get('change_pct', 0)
        volume = quote.get('volume', 0)
        avg_volume = quote.get('avg_volume', 0)

        criteria = StockCriteria(
            symbol=symbol.upper(),
            price_in_range=1.0 <= price <= 20.0,
            change_over_10pct=change_pct >= 10.0,
            rvol_over_5x=(volume / avg_volume >= 5.0) if avg_volume > 0 else False,
            price=price,
            change_pct=change_pct,
            relative_volume=volume / avg_volume if avg_volume > 0 else 0
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
                "F": "Do not trade"
            }.get(criteria.grade.value, "Unknown")
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

        return {
            "success": True,
            **result.to_dict()
        }
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
                results.append({
                    "symbol": symbol.upper(),
                    "error": str(e),
                    "signal": "ERROR",
                    "recommendation": "WAIT"
                })

        return {
            "success": True,
            "count": len(results),
            "results": results
        }
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
        from ai.mtf_confirmation import get_mtf_engine
        import httpx

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
                    confirmed.append({
                        "symbol": symbol,
                        "confidence": result.confidence,
                        "reasons": result.reasons
                    })
            except:
                pass

        return {
            "success": True,
            "count": len(confirmed),
            "confirmed": confirmed
        }
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
            "require_macd": engine.require_macd_alignment
        }
    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "error": str(e)
        }


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
            return {
                "success": True,
                **data.to_dict()
            }
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
                    to_date=end.strftime("%Y-%m-%d")
                )

                if bars:
                    candles = [
                        {
                            "high": b.get("h", b.get("high", 0)),
                            "low": b.get("l", b.get("low", 0)),
                            "close": b.get("c", b.get("close", 0)),
                            "volume": b.get("v", b.get("volume", 0))
                        }
                        for b in bars[-100:]  # Last 100 candles
                    ]
                    data = manager.load_from_candles(symbol.upper(), candles)
                    return {
                        "success": True,
                        **data.to_dict()
                    }
            except Exception as e:
                logger.warning(f"Could not load VWAP from Polygon: {e}")

            return {
                "success": False,
                "symbol": symbol.upper(),
                "message": "No VWAP data available - send candle data first"
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
            "stop_price": data.stop_price if data else 0
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
            **data.to_dict()
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
                "at_vwap_threshold_pct": manager.at_vwap_threshold_pct
            }
        }
    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "error": str(e)
        }


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
            "message": f"VWAP reset for {symbol.upper() if symbol else 'all symbols'}"
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
                    "message": "Float loaded, volume tracking will start with trades"
                }
            return {
                "symbol": symbol.upper(),
                "error": "No float data available"
            }

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
            "is_low_float": data.is_low_float if data else False
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
            "stocks": [d.to_dict() for d in rotating]
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

        return {
            "count": len(movers),
            "stocks": [d.to_dict() for d in movers]
        }

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

        return {
            "count": len(alerts),
            "alerts": [a.to_dict() for a in alerts]
        }

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
            "avg_volume": avg_volume
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
                "float_millions": round(float_shares / 1_000_000, 2)
            }
        else:
            return {
                "success": False,
                "symbol": symbol.upper(),
                "error": "Could not load float data"
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
            "message": f"Float rotation reset for {symbol.upper() if symbol else 'all symbols'}"
        }

    except Exception as e:
        logger.error(f"Float rotation reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
