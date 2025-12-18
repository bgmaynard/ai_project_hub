"""
Polygon.io (Massive.com) Market Data Integration
================================================
Provides historical market data for pattern analysis and backtesting.
Free tier includes: reference data, previous day bars, historical minute bars.
Paid tier adds: real-time trades, snapshots, tick data.
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)

# API Configuration
POLYGON_BASE_URL = "https://api.polygon.io"
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

# Rate limiting
_last_request_time = 0
_MIN_REQUEST_INTERVAL = 0.12  # Free tier: 5 calls/minute = 12 seconds between calls


def _rate_limit():
    """Ensure we don't exceed rate limits"""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def _make_request(endpoint: str, params: dict = None) -> Optional[dict]:
    """Make API request to Polygon"""
    if not POLYGON_API_KEY:
        logger.warning("Polygon API key not configured")
        return None

    _rate_limit()

    url = f"{POLYGON_BASE_URL}{endpoint}"
    params = params or {}
    params["apiKey"] = POLYGON_API_KEY

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, params=params)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                logger.error("Polygon API: Not authorized - check API key")
            elif response.status_code == 403:
                logger.warning("Polygon API: Feature requires paid subscription")
            else:
                logger.error(f"Polygon API error: {response.status_code} - {response.text}")

    except Exception as e:
        logger.error(f"Polygon API request failed: {e}")

    return None


class PolygonMarketData:
    """
    Polygon.io market data provider.
    Provides historical data for analysis and backtesting.
    """

    def __init__(self):
        self.api_key = POLYGON_API_KEY
        self.is_available = bool(self.api_key)

        if self.is_available:
            logger.info("Polygon.io market data initialized")
        else:
            logger.warning("Polygon API key not set - features disabled")

    def get_ticker_details(self, symbol: str) -> Optional[Dict]:
        """
        Get reference data for a ticker (company info, market cap, etc.)
        Available on free tier.
        """
        data = _make_request(f"/v3/reference/tickers/{symbol.upper()}")
        if data and data.get("status") == "OK":
            return data.get("results", {})
        return None

    def get_previous_close(self, symbol: str) -> Optional[Dict]:
        """
        Get previous day's OHLCV bar.
        Available on free tier.
        """
        data = _make_request(f"/v2/aggs/ticker/{symbol.upper()}/prev", {
            "adjusted": "true"
        })
        if data and data.get("status") == "OK":
            results = data.get("results", [])
            if results:
                bar = results[0]
                return {
                    "symbol": symbol.upper(),
                    "open": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "close": bar.get("c"),
                    "volume": bar.get("v"),
                    "vwap": bar.get("vw"),
                    "trades": bar.get("n"),
                    "timestamp": bar.get("t")
                }
        return None

    def get_minute_bars(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        multiplier: int = 1,
        limit: int = 5000
    ) -> List[Dict]:
        """
        Get historical minute bars.
        Available on free tier (delayed data).

        Args:
            symbol: Stock ticker
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            multiplier: Bar size in minutes (1, 5, 15, etc.)
            limit: Max number of bars

        Returns:
            List of OHLCV bars
        """
        data = _make_request(
            f"/v2/aggs/ticker/{symbol.upper()}/range/{multiplier}/minute/{from_date}/{to_date}",
            {
                "adjusted": "true",
                "sort": "asc",
                "limit": str(limit)
            }
        )

        if data and data.get("status") == "OK":
            results = data.get("results", [])
            bars = []
            for bar in results:
                bars.append({
                    "open": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "close": bar.get("c"),
                    "volume": bar.get("v"),
                    "vwap": bar.get("vw"),
                    "trades": bar.get("n"),
                    "timestamp": bar.get("t")
                })
            return bars

        return []

    def get_daily_bars(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        limit: int = 365
    ) -> List[Dict]:
        """
        Get historical daily bars.
        Available on free tier.

        Args:
            symbol: Stock ticker
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Max number of bars

        Returns:
            List of daily OHLCV bars
        """
        data = _make_request(
            f"/v2/aggs/ticker/{symbol.upper()}/range/1/day/{from_date}/{to_date}",
            {
                "adjusted": "true",
                "sort": "asc",
                "limit": str(limit)
            }
        )

        if data and data.get("status") == "OK":
            results = data.get("results", [])
            bars = []
            for bar in results:
                bars.append({
                    "date": datetime.fromtimestamp(bar.get("t", 0) / 1000).strftime("%Y-%m-%d"),
                    "open": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "close": bar.get("c"),
                    "volume": bar.get("v"),
                    "vwap": bar.get("vw"),
                    "trades": bar.get("n"),
                    "timestamp": bar.get("t")
                })
            return bars

        return []

    def get_last_trade(self, symbol: str) -> Optional[Dict]:
        """
        Get last trade for a ticker.
        REQUIRES PAID SUBSCRIPTION.
        """
        data = _make_request(f"/v2/last/trade/{symbol.upper()}")
        if data and data.get("status") == "OK":
            result = data.get("results", {})
            return {
                "price": result.get("p"),
                "size": result.get("s"),
                "exchange": result.get("x"),
                "timestamp": result.get("t"),
                "conditions": result.get("c", [])
            }
        return None

    def get_trades(
        self,
        symbol: str,
        date: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get historical trades for Time & Sales.
        REQUIRES PAID SUBSCRIPTION.

        Args:
            symbol: Stock ticker
            date: Date (YYYY-MM-DD)
            limit: Max trades to return

        Returns:
            List of individual trades
        """
        # Convert date to timestamp
        dt = datetime.strptime(date, "%Y-%m-%d")
        timestamp_gte = int(dt.timestamp() * 1000000000)

        data = _make_request(f"/v3/trades/{symbol.upper()}", {
            "timestamp.gte": str(timestamp_gte),
            "limit": str(limit),
            "sort": "timestamp",
            "order": "desc"
        })

        if data and data.get("status") == "OK":
            results = data.get("results", [])
            trades = []
            for trade in results:
                ts = trade.get("sip_timestamp", 0)
                trade_time = datetime.fromtimestamp(ts / 1000000000) if ts else None

                trades.append({
                    "price": trade.get("price"),
                    "size": trade.get("size"),
                    "exchange": trade.get("exchange"),
                    "time": trade_time.strftime("%H:%M:%S.%f")[:-3] if trade_time else None,
                    "timestamp": ts,
                    "conditions": trade.get("conditions", [])
                })
            return trades

        return []

    def get_snapshot(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time snapshot (quote + last trade + day bar).
        REQUIRES PAID SUBSCRIPTION.
        """
        data = _make_request(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol.upper()}")
        if data and data.get("status") == "OK":
            return data.get("ticker", {})
        return None


# Global instance
_polygon_data: Optional[PolygonMarketData] = None


def get_polygon_data() -> Optional[PolygonMarketData]:
    """Get or create the Polygon data instance"""
    global _polygon_data
    if _polygon_data is None:
        _polygon_data = PolygonMarketData()
    return _polygon_data


def is_polygon_available() -> bool:
    """Check if Polygon API is configured"""
    return bool(POLYGON_API_KEY)


logger.info("Polygon.io market data module loaded")
