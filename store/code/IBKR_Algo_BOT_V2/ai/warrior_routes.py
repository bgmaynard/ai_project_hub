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
