"""
Charts API Routes for Lightweight Charts Integration
=====================================================
Provides OHLC data, trade markers, indicators, and equity curves
for TradingView Lightweight Charts visualization.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from fastapi import APIRouter, Query
import pandas as pd

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/charts", tags=["Charts"])

# Import Polygon data provider
try:
    from polygon_data import get_polygon_data
    HAS_POLYGON = True
except ImportError:
    HAS_POLYGON = False
    logger.warning("Polygon data not available for charts")

# Trade data path
SCALPER_TRADES_PATH = "ai/scalper_trades.json"
BACKTEST_RESULTS_PATH = "ai/backtest_results.json"


def _load_trades() -> List[Dict]:
    """Load trades from scalper_trades.json"""
    try:
        with open(SCALPER_TRADES_PATH, 'r') as f:
            data = json.load(f)
            return data.get('trades', [])
    except Exception as e:
        logger.error(f"Error loading trades: {e}")
        return []


def _load_backtest() -> Dict:
    """Load backtest results"""
    try:
        with open(BACKTEST_RESULTS_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading backtest: {e}")
        return {}


def _iso_to_unix(iso_str: str) -> int:
    """Convert ISO 8601 timestamp to Unix seconds"""
    try:
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return int(dt.timestamp())
    except:
        return 0


def _timeframe_to_multiplier(timeframe: str) -> tuple:
    """Convert timeframe string to Polygon multiplier and timespan"""
    mapping = {
        '1m': (1, 'minute'),
        '5m': (5, 'minute'),
        '15m': (15, 'minute'),
        '1h': (60, 'minute'),
        '1d': (1, 'day')
    }
    return mapping.get(timeframe, (5, 'minute'))


# ============================================================================
# OHLC DATA ENDPOINT
# ============================================================================

@router.get("/ohlc/{symbol}")
async def get_chart_ohlc(
    symbol: str,
    timeframe: str = Query("5m", description="Timeframe: 1m, 5m, 15m, 1h, 1d"),
    bars: int = Query(500, description="Number of bars", ge=10, le=10000),
    days: int = Query(None, description="Days of history (overrides bars)")
):
    """
    Get OHLC candlestick data for Lightweight Charts.

    Returns data in format required by Lightweight Charts:
    - time: Unix timestamp in seconds
    - open, high, low, close: prices
    - volume: trading volume
    """
    symbol = symbol.upper()

    if not HAS_POLYGON:
        return {"success": False, "error": "Polygon data not available"}

    try:
        polygon = get_polygon_data()
        multiplier, timespan = _timeframe_to_multiplier(timeframe)

        # Calculate date range - use days param if provided
        to_date = datetime.now().strftime("%Y-%m-%d")
        if days:
            # User specified days directly
            fetch_days = days + 2  # buffer for weekends
        elif timespan == 'minute':
            # Assume ~390 trading minutes per day, add buffer
            fetch_days = max(7, (bars * multiplier) // 390 + 3)
        else:
            fetch_days = bars + 10  # Daily bars

        from_date = (datetime.now() - timedelta(days=fetch_days)).strftime("%Y-%m-%d")

        # Fetch data from Polygon
        if timespan == 'day':
            raw_bars = polygon.get_daily_bars(symbol, from_date, to_date)
        else:
            raw_bars = polygon.get_minute_bars(symbol, from_date, to_date, multiplier=multiplier)

        if not raw_bars:
            return {"success": False, "error": f"No data available for {symbol}"}

        # Convert to Lightweight Charts format
        chart_data = []
        volume_data = []

        # If days specified, use all data; otherwise limit to bars
        max_bars = len(raw_bars) if days else bars
        for bar in raw_bars[-max_bars:]:  # Take last N bars
            # Handle timestamp (could be in ms or as datetime)
            ts = bar.get('timestamp') or bar.get('t')
            if isinstance(ts, (int, float)):
                # Assume milliseconds if large number
                unix_time = int(ts / 1000) if ts > 1e12 else int(ts)
            else:
                unix_time = int(datetime.now().timestamp())

            chart_data.append({
                "time": unix_time,
                "open": bar.get('open') or bar.get('o', 0),
                "high": bar.get('high') or bar.get('h', 0),
                "low": bar.get('low') or bar.get('l', 0),
                "close": bar.get('close') or bar.get('c', 0)
            })

            vol = bar.get('volume') or bar.get('v', 0)
            close = bar.get('close') or bar.get('c', 0)
            open_price = bar.get('open') or bar.get('o', 0)
            volume_data.append({
                "time": unix_time,
                "value": vol,
                "color": "#22c55e" if close >= open_price else "#ef4444"
            })

        # Sort by time ascending
        chart_data.sort(key=lambda x: x['time'])
        volume_data.sort(key=lambda x: x['time'])

        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "data": chart_data,
            "volume": volume_data,
            "count": len(chart_data),
            "source": "polygon"
        }

    except Exception as e:
        logger.error(f"Error fetching OHLC for {symbol}: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# TRADE MARKERS ENDPOINT
# ============================================================================

def _round_to_bar(timestamp: int, timeframe: str) -> int:
    """Round timestamp to nearest bar start based on timeframe"""
    intervals = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '1h': 3600,
        '1d': 86400
    }
    interval = intervals.get(timeframe, 300)  # Default 5m
    return (timestamp // interval) * interval


@router.get("/trades/{symbol}")
async def get_trade_markers(
    symbol: str,
    timeframe: str = Query("5m", description="Timeframe to align markers: 1m, 5m, 15m, 1h, 1d"),
    from_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    to_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    limit: int = Query(100, description="Max trades to return")
):
    """
    Get trade entry/exit markers for chart overlay.

    Returns markers formatted for Lightweight Charts setMarkers():
    - time: Unix timestamp (rounded to bar interval)
    - position: 'aboveBar' or 'belowBar'
    - color: green for wins, red for losses
    - shape: 'arrowUp' for entry, 'arrowDown' for exit
    - text: trade info
    """
    symbol = symbol.upper()
    trades = _load_trades()

    # Filter by symbol
    if symbol != "ALL":
        trades = [t for t in trades if t.get('symbol', '').upper() == symbol]

    # Filter by date range
    if from_date:
        from_dt = datetime.strptime(from_date, "%Y-%m-%d")
        trades = [t for t in trades if datetime.fromisoformat(t.get('entry_time', '2000-01-01')) >= from_dt]

    if to_date:
        to_dt = datetime.strptime(to_date, "%Y-%m-%d") + timedelta(days=1)
        trades = [t for t in trades if datetime.fromisoformat(t.get('entry_time', '2099-12-31')) < to_dt]

    # Sort by entry time and limit
    trades.sort(key=lambda x: x.get('entry_time', ''), reverse=True)
    trades = trades[:limit]

    markers = []
    summary = {"total_trades": 0, "winners": 0, "losers": 0, "total_pnl": 0}

    for trade in trades:
        pnl = trade.get('pnl', 0)
        pnl_pct = trade.get('pnl_percent', 0)
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)

        # Determine color based on P&L
        if pnl > 0:
            color = "#22c55e"  # Green
            summary["winners"] += 1
        elif pnl < 0:
            color = "#ef4444"  # Red
            summary["losers"] += 1
        else:
            color = "#eab308"  # Yellow

        summary["total_trades"] += 1
        summary["total_pnl"] += pnl

        # Entry marker - round to bar interval
        entry_time_raw = _iso_to_unix(trade.get('entry_time', ''))
        if entry_time_raw:
            entry_time = _round_to_bar(entry_time_raw, timeframe)
            markers.append({
                "time": entry_time,
                "position": "belowBar",
                "color": color,
                "shape": "arrowUp",
                "text": "Buy",
                "id": f"entry_{trade.get('trade_id', '')}",
                "size": 2
            })

        # Exit marker - round to bar interval
        exit_time_raw = _iso_to_unix(trade.get('exit_time', ''))
        if exit_time_raw:
            exit_time = _round_to_bar(exit_time_raw, timeframe)
            # Simple P&L indicator for exit
            if pnl > 0:
                exit_text = f"Sell +${pnl:.0f}"
            elif pnl < 0:
                exit_text = f"Sell -${abs(pnl):.0f}"
            else:
                exit_text = "Sell"  # Break-even
            markers.append({
                "time": exit_time,
                "position": "aboveBar",
                "color": color,
                "shape": "arrowDown",
                "text": exit_text,
                "id": f"exit_{trade.get('trade_id', '')}",
                "size": 2
            })

    # Sort markers by time
    markers.sort(key=lambda x: x['time'])

    # Calculate win rate
    if summary["total_trades"] > 0:
        summary["win_rate"] = round(summary["winners"] / summary["total_trades"] * 100, 1)
    else:
        summary["win_rate"] = 0

    summary["total_pnl"] = round(summary["total_pnl"], 2)

    return {
        "success": True,
        "symbol": symbol,
        "markers": markers,
        "summary": summary
    }


# ============================================================================
# INDICATORS ENDPOINT
# ============================================================================

@router.get("/indicators/{symbol}")
async def get_indicators(
    symbol: str,
    indicators: str = Query("sma20,ema9,vwap", description="Comma-separated: sma20,ema9,ema20,vwap,rsi14,macd"),
    timeframe: str = Query("5m", description="Timeframe"),
    bars: int = Query(500, description="Number of bars"),
    days: int = Query(None, description="Days of history (overrides bars)")
):
    """
    Calculate technical indicators server-side.

    Supported indicators:
    - sma{period}: Simple Moving Average
    - ema{period}: Exponential Moving Average
    - vwap: Volume Weighted Average Price
    - rsi{period}: Relative Strength Index
    - macd: MACD (12/26/9)
    """
    symbol = symbol.upper()

    if not HAS_POLYGON:
        return {"success": False, "error": "Polygon data not available"}

    try:
        # First get OHLC data
        polygon = get_polygon_data()
        multiplier, timespan = _timeframe_to_multiplier(timeframe)

        # Calculate date range - use days param if provided
        to_date = datetime.now().strftime("%Y-%m-%d")
        if days:
            fetch_days = days + 2  # buffer for weekends
        elif timespan == 'day':
            fetch_days = bars + 50
        else:
            fetch_days = max(7, (bars * multiplier) // 390 + 5)

        from_date = (datetime.now() - timedelta(days=fetch_days)).strftime("%Y-%m-%d")

        if timespan == 'day':
            raw_bars = polygon.get_daily_bars(symbol, from_date, to_date)
        else:
            raw_bars = polygon.get_minute_bars(symbol, from_date, to_date, multiplier=multiplier)

        if not raw_bars or len(raw_bars) < 20:
            return {"success": False, "error": f"Insufficient data for {symbol}"}

        # Convert to DataFrame
        df = pd.DataFrame(raw_bars)

        # Normalize column names
        col_map = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'}
        df.rename(columns=col_map, inplace=True)

        # Convert timestamp
        if 'timestamp' in df.columns:
            df['time'] = df['timestamp'].apply(lambda x: int(x / 1000) if x > 1e12 else int(x))
        else:
            df['time'] = range(len(df))

        # Calculate requested indicators
        result = {"success": True, "symbol": symbol, "timeframe": timeframe}
        indicator_list = [i.strip().lower() for i in indicators.split(',')]

        # If days specified, return all data; otherwise limit to bars
        max_results = len(df) if days else bars

        for ind in indicator_list:
            try:
                if ind.startswith('sma'):
                    period = int(ind[3:]) if len(ind) > 3 else 20
                    df[f'sma{period}'] = df['close'].rolling(window=period).mean()
                    result[f'sma{period}'] = [
                        {"time": row['time'], "value": round(row[f'sma{period}'], 4)}
                        for _, row in df.dropna(subset=[f'sma{period}']).iterrows()
                    ][-max_results:]

                elif ind.startswith('ema'):
                    period = int(ind[3:]) if len(ind) > 3 else 9
                    df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                    result[f'ema{period}'] = [
                        {"time": row['time'], "value": round(row[f'ema{period}'], 4)}
                        for _, row in df.dropna(subset=[f'ema{period}']).iterrows()
                    ][-max_results:]

                elif ind == 'vwap':
                    # VWAP calculation
                    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
                    df['tp_volume'] = df['typical_price'] * df['volume']
                    df['cum_tp_vol'] = df['tp_volume'].cumsum()
                    df['cum_vol'] = df['volume'].cumsum()
                    df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
                    result['vwap'] = [
                        {"time": row['time'], "value": round(row['vwap'], 4)}
                        for _, row in df.dropna(subset=['vwap']).iterrows()
                    ][-max_results:]

                elif ind.startswith('rsi'):
                    period = int(ind[3:]) if len(ind) > 3 else 14
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    df[f'rsi{period}'] = 100 - (100 / (1 + rs))
                    result[f'rsi{period}'] = [
                        {"time": row['time'], "value": round(row[f'rsi{period}'], 2)}
                        for _, row in df.dropna(subset=[f'rsi{period}']).iterrows()
                    ][-max_results:]

                elif ind == 'macd':
                    ema12 = df['close'].ewm(span=12, adjust=False).mean()
                    ema26 = df['close'].ewm(span=26, adjust=False).mean()
                    df['macd_line'] = ema12 - ema26
                    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
                    df['macd_hist'] = df['macd_line'] - df['macd_signal']
                    result['macd'] = [
                        {
                            "time": row['time'],
                            "macd": round(row['macd_line'], 4),
                            "signal": round(row['macd_signal'], 4),
                            "histogram": round(row['macd_hist'], 4)
                        }
                        for _, row in df.dropna(subset=['macd_line']).iterrows()
                    ][-max_results:]

            except Exception as e:
                logger.warning(f"Error calculating {ind}: {e}")
                result[ind] = []

        return result

    except Exception as e:
        logger.error(f"Error calculating indicators for {symbol}: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# EQUITY CURVE ENDPOINT
# ============================================================================

@router.get("/equity-curve")
async def get_equity_curve(
    from_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    to_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    initial_capital: float = Query(1000.0, description="Starting capital")
):
    """
    Generate equity curve from trade history.

    Returns:
    - curve: Array of {time, equity} points
    - total_pnl: Total profit/loss
    - max_drawdown: Maximum drawdown
    """
    trades = _load_trades()

    # Filter by date
    if from_date:
        from_dt = datetime.strptime(from_date, "%Y-%m-%d")
        trades = [t for t in trades if datetime.fromisoformat(t.get('entry_time', '2000-01-01')) >= from_dt]

    if to_date:
        to_dt = datetime.strptime(to_date, "%Y-%m-%d") + timedelta(days=1)
        trades = [t for t in trades if datetime.fromisoformat(t.get('entry_time', '2099-12-31')) < to_dt]

    # Sort by exit time (handle None values)
    trades.sort(key=lambda x: x.get('exit_time') or x.get('entry_time') or '')

    # Build equity curve
    equity = initial_capital
    curve = [{"time": int(datetime.now().timestamp()) - 86400 * 30, "equity": initial_capital}]
    peak = initial_capital
    max_drawdown = 0

    for trade in trades:
        if trade.get('status') == 'closed' and trade.get('exit_time'):
            pnl = trade.get('pnl', 0)
            equity += pnl

            exit_time = _iso_to_unix(trade['exit_time'])
            curve.append({
                "time": exit_time,
                "equity": round(equity, 2)
            })

            # Track drawdown
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

    # Sort curve by time
    curve.sort(key=lambda x: x['time'])

    # Summary stats
    total_pnl = equity - initial_capital
    win_count = len([t for t in trades if t.get('pnl', 0) > 0])
    total_trades = len([t for t in trades if t.get('status') == 'closed'])

    return {
        "success": True,
        "initial_capital": initial_capital,
        "final_equity": round(equity, 2),
        "total_pnl": round(total_pnl, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "trades_count": total_trades,
        "win_rate": round(win_count / total_trades * 100, 1) if total_trades > 0 else 0,
        "curve": curve
    }


# ============================================================================
# BACKTEST RESULTS ENDPOINT
# ============================================================================

@router.get("/backtest")
async def get_backtest_results(
    symbol: Optional[str] = Query(None, description="Filter by symbol")
):
    """
    Get backtest results with trade details for visualization.
    """
    backtest = _load_backtest()

    if not backtest:
        # Fall back to scalper trades
        trades = _load_trades()
        backtest = {"trades": trades}

    trades = backtest.get('trades', backtest.get('detailed_results', []))

    # Filter by symbol if specified
    if symbol:
        symbol = symbol.upper()
        trades = [t for t in trades if t.get('symbol', '').upper() == symbol]

    # Calculate summary
    closed_trades = [t for t in trades if t.get('status') == 'closed' or t.get('exit_time')]
    winners = [t for t in closed_trades if t.get('pnl', 0) > 0]
    losers = [t for t in closed_trades if t.get('pnl', 0) < 0]

    total_pnl = sum(t.get('pnl', 0) for t in closed_trades)

    # Group by exit reason
    by_exit_reason = {}
    for trade in closed_trades:
        reason = trade.get('exit_reason', 'UNKNOWN')
        if reason not in by_exit_reason:
            by_exit_reason[reason] = {"count": 0, "pnl": 0}
        by_exit_reason[reason]["count"] += 1
        by_exit_reason[reason]["pnl"] += trade.get('pnl', 0)

    # Round PnL values
    for reason in by_exit_reason:
        by_exit_reason[reason]["pnl"] = round(by_exit_reason[reason]["pnl"], 2)

    return {
        "success": True,
        "report": {
            "total_trades": len(closed_trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(len(winners) / len(closed_trades) * 100, 1) if closed_trades else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(sum(t.get('pnl', 0) for t in winners) / len(winners), 2) if winners else 0,
            "avg_loss": round(sum(t.get('pnl', 0) for t in losers) / len(losers), 2) if losers else 0,
            "by_exit_reason": by_exit_reason
        },
        "trades": [
            {
                "trade_id": t.get('trade_id'),
                "symbol": t.get('symbol'),
                "entry_time": _iso_to_unix(t.get('entry_time', '')),
                "entry_price": t.get('entry_price'),
                "exit_time": _iso_to_unix(t.get('exit_time', '')),
                "exit_price": t.get('exit_price'),
                "exit_reason": t.get('exit_reason'),
                "pnl": t.get('pnl'),
                "pnl_percent": t.get('pnl_percent'),
                "shares": t.get('shares')
            }
            for t in closed_trades[:500]  # Limit to 500 trades
        ]
    }


# ============================================================================
# TECHNICAL SIGNALS ENDPOINTS
# ============================================================================

# Import signal analyzer
try:
    from ai.technical_signals import get_signal_analyzer, SignalEvent
    HAS_SIGNALS = True
except ImportError:
    HAS_SIGNALS = False
    logger.warning("Technical signals module not available")


@router.get("/signals/{symbol}")
async def get_technical_signals(
    symbol: str,
    timeframe: str = Query("5m", description="Timeframe: 1m, 5m, 15m"),
    days: int = Query(5, description="Days of history")
):
    """
    Get current technical signal state for a symbol.

    Returns confluence score, EMA/MACD/VWAP states, and entry/exit recommendations.
    """
    symbol = symbol.upper()

    if not HAS_SIGNALS:
        return {"success": False, "error": "Technical signals module not available"}

    if not HAS_POLYGON:
        return {"success": False, "error": "Polygon data not available"}

    try:
        # Fetch OHLC data
        polygon = get_polygon_data()
        multiplier, timespan = _timeframe_to_multiplier(timeframe)

        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days + 2)).strftime("%Y-%m-%d")

        if timespan == 'day':
            raw_bars = polygon.get_daily_bars(symbol, from_date, to_date)
        else:
            raw_bars = polygon.get_minute_bars(symbol, from_date, to_date, multiplier=multiplier)

        if not raw_bars or len(raw_bars) < 50:
            return {"success": False, "error": f"Insufficient data for {symbol}"}

        # Convert to OHLC format
        ohlc_data = []
        for bar in raw_bars:
            ts = bar.get('timestamp') or bar.get('t')
            unix_time = int(ts / 1000) if ts > 1e12 else int(ts)
            ohlc_data.append({
                "time": unix_time,
                "open": bar.get('open') or bar.get('o', 0),
                "high": bar.get('high') or bar.get('h', 0),
                "low": bar.get('low') or bar.get('l', 0),
                "close": bar.get('close') or bar.get('c', 0),
                "volume": bar.get('volume') or bar.get('v', 0)
            })

        # Analyze signals
        analyzer = get_signal_analyzer()
        state = analyzer.analyze(symbol, ohlc_data, timeframe)

        if not state:
            return {"success": False, "error": "Failed to analyze signals"}

        # Get entry/exit recommendations
        should_enter, enter_reason = analyzer.should_enter(state)

        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "signals": state.to_dict(),
            "recommendation": {
                "should_enter": should_enter,
                "reason": enter_reason
            }
        }

    except Exception as e:
        logger.error(f"Error getting signals for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.get("/signals/{symbol}/events")
async def get_signal_events(
    symbol: str,
    timeframe: str = Query("5m", description="Timeframe: 1m, 5m, 15m"),
    days: int = Query(5, description="Days of history")
):
    """
    Get signal events (crossovers) for chart plotting.

    Returns list of events with time, type, direction for marker display.
    """
    symbol = symbol.upper()

    if not HAS_SIGNALS:
        return {"success": False, "error": "Technical signals module not available"}

    if not HAS_POLYGON:
        return {"success": False, "error": "Polygon data not available"}

    try:
        # Fetch OHLC data
        polygon = get_polygon_data()
        multiplier, timespan = _timeframe_to_multiplier(timeframe)

        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days + 2)).strftime("%Y-%m-%d")

        if timespan == 'day':
            raw_bars = polygon.get_daily_bars(symbol, from_date, to_date)
        else:
            raw_bars = polygon.get_minute_bars(symbol, from_date, to_date, multiplier=multiplier)

        if not raw_bars or len(raw_bars) < 50:
            return {"success": False, "error": f"Insufficient data for {symbol}"}

        # Convert to OHLC format
        ohlc_data = []
        for bar in raw_bars:
            ts = bar.get('timestamp') or bar.get('t')
            unix_time = int(ts / 1000) if ts > 1e12 else int(ts)
            ohlc_data.append({
                "time": unix_time,
                "open": bar.get('open') or bar.get('o', 0),
                "high": bar.get('high') or bar.get('h', 0),
                "low": bar.get('low') or bar.get('l', 0),
                "close": bar.get('close') or bar.get('c', 0),
                "volume": bar.get('volume') or bar.get('v', 0)
            })

        # Get signal events
        analyzer = get_signal_analyzer()
        events = analyzer.get_signal_events(symbol, ohlc_data, timeframe)

        # Convert to chart markers format
        markers = []
        for event in events:
            if event.direction == "BULLISH":
                color = "#22c55e"  # Green
                position = "belowBar"
                shape = "arrowUp"
            else:
                color = "#ef4444"  # Red
                position = "aboveBar"
                shape = "arrowDown"

            # Different shapes for different event types
            if event.event_type == "EMA_CROSS":
                text = "EMA"
            elif event.event_type == "MACD_CROSS":
                text = "MACD"
            elif event.event_type == "VWAP_CROSS":
                text = "VWAP"
            else:
                text = event.event_type

            markers.append({
                "time": event.time,
                "position": position,
                "color": color,
                "shape": shape,
                "text": text,
                "size": 1
            })

        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "events": [
                {
                    "time": e.time,
                    "type": e.event_type,
                    "direction": e.direction,
                    "description": e.description,
                    "strength": e.strength
                }
                for e in events
            ],
            "markers": markers,
            "count": len(events)
        }

    except Exception as e:
        logger.error(f"Error getting signal events for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.get("/signals/{symbol}/alignment")
async def get_multi_timeframe_alignment(
    symbol: str,
    days: int = Query(5, description="Days of history")
):
    """
    Check multi-timeframe alignment between 1M and 5M charts.

    Returns alignment status and combined recommendation.
    """
    symbol = symbol.upper()

    if not HAS_SIGNALS:
        return {"success": False, "error": "Technical signals module not available"}

    if not HAS_POLYGON:
        return {"success": False, "error": "Polygon data not available"}

    try:
        polygon = get_polygon_data()
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days + 2)).strftime("%Y-%m-%d")

        # Fetch both timeframes
        raw_1m = polygon.get_minute_bars(symbol, from_date, to_date, multiplier=1)
        raw_5m = polygon.get_minute_bars(symbol, from_date, to_date, multiplier=5)

        def convert_bars(raw_bars):
            ohlc = []
            for bar in raw_bars:
                ts = bar.get('timestamp') or bar.get('t')
                unix_time = int(ts / 1000) if ts > 1e12 else int(ts)
                ohlc.append({
                    "time": unix_time,
                    "open": bar.get('open') or bar.get('o', 0),
                    "high": bar.get('high') or bar.get('h', 0),
                    "low": bar.get('low') or bar.get('l', 0),
                    "close": bar.get('close') or bar.get('c', 0),
                    "volume": bar.get('volume') or bar.get('v', 0)
                })
            return ohlc

        ohlc_1m = convert_bars(raw_1m) if raw_1m else []
        ohlc_5m = convert_bars(raw_5m) if raw_5m else []

        if len(ohlc_1m) < 50 or len(ohlc_5m) < 50:
            return {"success": False, "error": "Insufficient data for alignment check"}

        analyzer = get_signal_analyzer()
        alignment = analyzer.get_multi_timeframe_alignment(symbol, ohlc_1m, ohlc_5m)

        return {
            "success": True,
            "symbol": symbol,
            "alignment": alignment
        }

    except Exception as e:
        logger.error(f"Error checking alignment for {symbol}: {e}")
        return {"success": False, "error": str(e)}
