"""
Data Collection API Routes
==========================
API endpoints for unified data collection and history.
"""

import logging
from typing import Optional
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# SCHWAB HISTORY / MINUTE BARS
# ============================================================================

@router.get("/api/schwab/history/{symbol}")
async def get_schwab_history(
    symbol: str,
    days: int = 1,
    frequency: str = "minute"
):
    """
    Get historical price bars from Schwab.

    Args:
        symbol: Stock ticker
        days: Number of days (1, 2, 3, 4, 5, 10 for minute data)
        frequency: 'minute' or 'daily'

    Returns:
        List of OHLCV bars
    """
    symbol = symbol.upper()
    try:
        # Import here to avoid circular imports
        from schwab_market_data import get_schwab_market_data

        schwab = get_schwab_market_data()
        if not schwab:
            return {"success": False, "error": "Schwab market data not connected", "bars": []}

        # Map to Schwab API parameters
        if frequency == "minute":
            period_type = "day"
            frequency_type = "minute"
            freq = 1
        else:
            period_type = "month"
            frequency_type = "daily"
            freq = 1
            days = min(days, 30)  # Cap at 30 days for daily

        data = schwab.get_price_history(
            symbol=symbol,
            period_type=period_type,
            period=min(days, 10),  # Schwab limits minute data
            frequency_type=frequency_type,
            frequency=freq
        )

        if not data:
            return {"success": False, "error": "No data returned", "bars": []}

        # Extract candles
        candles = data.get("candles", [])

        # Format for our system
        bars = []
        for c in candles:
            bars.append({
                "timestamp": c.get("datetime"),
                "open": c.get("open"),
                "high": c.get("high"),
                "low": c.get("low"),
                "close": c.get("close"),
                "volume": c.get("volume")
            })

        return {
            "success": True,
            "symbol": symbol,
            "frequency": frequency,
            "bars": bars,
            "count": len(bars)
        }

    except Exception as e:
        logger.error(f"Error getting Schwab history for {symbol}: {e}")
        return {"success": False, "error": str(e), "bars": []}


# ============================================================================
# UNIFIED DATA COLLECTOR ENDPOINTS
# ============================================================================

@router.get("/api/data/summary")
async def get_data_summary():
    """Get summary of collected market data"""
    try:
        from ai.unified_data_collector import get_data_collector
        collector = get_data_collector()
        return {"success": True, "data": collector.get_data_summary()}
    except Exception as e:
        logger.error(f"Error getting data summary: {e}")
        return {"success": False, "error": str(e)}


@router.post("/api/data/collect/{symbol}")
async def collect_history(symbol: str, days: int = 5):
    """
    Fetch and store historical minute bars for a symbol.

    Args:
        symbol: Stock ticker
        days: Number of days to fetch (1-10 for minute data)
    """
    symbol = symbol.upper()
    try:
        from ai.unified_data_collector import get_data_collector
        collector = get_data_collector()

        # Fetch from Schwab
        bars = await collector.fetch_schwab_history(symbol, days)

        if bars:
            # Store in database
            stored = collector.store_minute_bars(symbol, bars, "schwab")
            return {
                "success": True,
                "symbol": symbol,
                "fetched": len(bars),
                "stored": stored
            }
        else:
            return {
                "success": False,
                "symbol": symbol,
                "error": "No bars fetched from Schwab"
            }

    except Exception as e:
        logger.error(f"Error collecting data for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.get("/api/data/bars/{symbol}")
async def get_stored_bars(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get stored minute bars from database.

    Args:
        symbol: Stock ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    symbol = symbol.upper()
    try:
        from ai.unified_data_collector import get_data_collector
        collector = get_data_collector()

        bars = collector.get_minute_bars(symbol, start_date, end_date)

        return {
            "success": True,
            "symbol": symbol,
            "bars": bars,
            "count": len(bars)
        }

    except Exception as e:
        logger.error(f"Error getting bars for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.get("/api/data/signals")
async def get_trade_signals(symbol: Optional[str] = None, limit: int = 100):
    """Get stored trade signals for analysis"""
    try:
        from ai.unified_data_collector import get_data_collector
        collector = get_data_collector()

        signals = collector.get_trade_signals(symbol, limit)

        return {
            "success": True,
            "signals": signals,
            "count": len(signals)
        }

    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        return {"success": False, "error": str(e)}


@router.get("/api/data/correlations")
async def get_signal_correlations():
    """Analyze correlations between signal characteristics and outcomes"""
    try:
        from ai.unified_data_collector import get_data_collector
        collector = get_data_collector()

        analysis = collector.analyze_signal_correlations()

        return {
            "success": True,
            "analysis": analysis
        }

    except Exception as e:
        logger.error(f"Error analyzing correlations: {e}")
        return {"success": False, "error": str(e)}


@router.post("/api/data/signal")
async def store_trade_signal(signal: dict):
    """Store a trade signal with all indicators"""
    try:
        from ai.unified_data_collector import get_data_collector
        collector = get_data_collector()

        collector.store_trade_signal(signal)

        return {"success": True, "message": "Signal stored"}

    except Exception as e:
        logger.error(f"Error storing signal: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# BACKTEST ENDPOINTS
# ============================================================================

@router.post("/api/backtest/minute")
async def run_minute_backtest(
    symbols: str = "SPY",
    days: int = 5,
    initial_cash: float = 1000.0
):
    """
    Run minute-bar backtest using Schwab data.

    Args:
        symbols: Comma-separated list of symbols (e.g., "SPY,AAPL")
        days: Days of data to use (max 10)
        initial_cash: Starting capital

    Returns:
        Backtest results including trades and metrics
    """
    try:
        from ai.pybroker_walkforward import run_minute_backtest as do_backtest

        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        result = do_backtest(
            symbols=symbol_list,
            days=days,
            initial_cash=initial_cash
        )

        return result

    except Exception as e:
        logger.error(f"Minute backtest error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/api/backtest/walkforward")
async def run_walkforward(
    symbols: str = "SPY",
    initial_cash: float = 1000.0
):
    """
    Run walkforward analysis using YFinance daily data.

    Tests strategy on out-of-sample data to detect overfitting.

    Args:
        symbols: Comma-separated list of symbols
        initial_cash: Starting capital

    Returns:
        Walkforward results with performance metrics
    """
    try:
        from ai.pybroker_walkforward import run_walkforward_test

        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        result = run_walkforward_test(
            symbols=symbol_list,
            initial_cash=initial_cash
        )

        return result

    except Exception as e:
        logger.error(f"Walkforward error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/api/backtest/full")
async def run_full_backtest(
    symbols: str = "SPY",
    initial_cash: float = 1000.0
):
    """
    Run comprehensive analysis: minute backtest + daily walkforward.

    Args:
        symbols: Comma-separated list of symbols
        initial_cash: Starting capital

    Returns:
        Combined analysis results
    """
    try:
        from ai.pybroker_walkforward import run_full_analysis

        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        result = run_full_analysis(
            symbols=symbol_list,
            initial_cash=initial_cash
        )

        return {"success": True, "results": result}

    except Exception as e:
        logger.error(f"Full analysis error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/api/backtest/status")
async def get_backtest_status():
    """Get status of backtest capabilities"""
    status = {
        "schwab_connected": False,
        "pybroker_available": False,
        "data_collector_ready": False
    }

    try:
        from schwab_market_data import is_schwab_available
        status["schwab_connected"] = is_schwab_available()
    except:
        pass

    try:
        from pybroker import Strategy
        status["pybroker_available"] = True
    except:
        pass

    try:
        from ai.unified_data_collector import get_data_collector
        collector = get_data_collector()
        summary = collector.get_data_summary()
        status["data_collector_ready"] = True
        status["stored_minute_bars"] = summary.get("minute_bars", 0)
        status["stored_signals"] = summary.get("trade_signals", 0)
    except:
        pass

    return {"success": True, "status": status}


# Quick function to include this router in the main app
def include_data_routes(app):
    """Include data routes in the main FastAPI app"""
    app.include_router(router, tags=["data"])
    logger.info("Data collection routes registered")
