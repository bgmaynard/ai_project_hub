"""
Schwab/ThinkOrSwim Market Data Integration
Real-time market data from Schwab API for AI Trading Platform
Uses direct HTTP API calls with automatic token refresh
"""

import base64
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Token file location
TOKEN_FILE = Path(__file__).parent / "schwab_token.json"

# Schwab API base URL
SCHWAB_API_BASE = "https://api.schwabapi.com"

# Token singleton with thread-safe lock
_token_lock = threading.RLock()  # Reentrant lock for nested calls
_token_data: Optional[Dict] = None
_token_expiry: Optional[datetime] = None
_schwab_available = False
_last_refresh_attempt: Optional[datetime] = None
_refresh_cooldown_seconds = 30  # Prevent refresh spam


def _load_token() -> Optional[Dict]:
    """Load token from file (thread-safe)"""
    global _token_data, _token_expiry

    with _token_lock:
        if not TOKEN_FILE.exists():
            logger.warning("Schwab token file not found")
            return None

        try:
            with open(TOKEN_FILE, "r") as f:
                _token_data = json.load(f)

            # Calculate expiry (access token expires in 30 mins, but we refresh early)
            _token_expiry = datetime.now() + timedelta(
                seconds=_token_data.get("expires_in", 1800) - 300
            )
            logger.info("Schwab token loaded successfully")
            return _token_data
        except Exception as e:
            logger.error(f"Failed to load Schwab token: {e}")
            return None


def _save_token(token_data: Dict):
    """Save token to file (thread-safe with atomic write)"""
    global _token_data, _token_expiry

    with _token_lock:
        try:
            # Atomic write: write to temp file then rename
            temp_file = TOKEN_FILE.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(token_data, f, indent=2)
            temp_file.replace(TOKEN_FILE)  # Atomic on most systems

            _token_data = token_data
            _token_expiry = datetime.now() + timedelta(
                seconds=token_data.get("expires_in", 1800) - 300
            )
            logger.info("Schwab token saved successfully")
        except Exception as e:
            logger.error(f"Failed to save Schwab token: {e}")


def _refresh_token() -> bool:
    """Refresh the access token using refresh token (thread-safe with cooldown)"""
    global _token_data, _token_expiry, _schwab_available, _last_refresh_attempt

    with _token_lock:
        # Check cooldown to prevent refresh spam
        if _last_refresh_attempt:
            elapsed = (datetime.now() - _last_refresh_attempt).total_seconds()
            if elapsed < _refresh_cooldown_seconds:
                logger.debug(
                    f"Token refresh on cooldown ({elapsed:.0f}s < {_refresh_cooldown_seconds}s)"
                )
                return _schwab_available  # Return current state

        _last_refresh_attempt = datetime.now()

        if not _token_data or "refresh_token" not in _token_data:
            logger.error("No refresh token available")
            _schwab_available = False
            return False

        app_key = os.getenv("SCHWAB_APP_KEY")
        app_secret = os.getenv("SCHWAB_APP_SECRET")

        if not app_key or not app_secret:
            logger.error("Schwab credentials not configured")
            _schwab_available = False
            return False

        try:
            # Create Basic auth header
            credentials = f"{app_key}:{app_secret}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()

            headers = {
                "Authorization": f"Basic {encoded_credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            }

            data = {
                "grant_type": "refresh_token",
                "refresh_token": _token_data["refresh_token"],
            }

            response = httpx.post(
                f"{SCHWAB_API_BASE}/v1/oauth/token",
                headers=headers,
                data=data,
                timeout=30.0,
            )

            if response.status_code == 200:
                new_token = response.json()
                # Keep refresh token if not returned
                if "refresh_token" not in new_token:
                    new_token["refresh_token"] = _token_data["refresh_token"]
                _save_token(new_token)
                _schwab_available = True
                logger.info("Schwab token refreshed successfully")
                return True
            else:
                logger.error(
                    f"Token refresh failed: {response.status_code} - {response.text}"
                )
                _schwab_available = False
                return False

        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            _schwab_available = False
            return False


def _ensure_token() -> Optional[str]:
    """Ensure we have a valid access token (thread-safe)"""
    global _token_data, _token_expiry, _schwab_available

    with _token_lock:
        # Load token if not loaded
        if _token_data is None:
            _load_token()

        if _token_data is None:
            _schwab_available = False
            return None

        # Check if token needs refresh
        if _token_expiry is None or datetime.now() >= _token_expiry:
            logger.info("Token expired or expiring soon, refreshing...")
            if not _refresh_token():
                return None

        _schwab_available = True
        return _token_data.get("access_token")


def _make_request(endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
    """Make authenticated request to Schwab API"""
    access_token = _ensure_token()
    if not access_token:
        return None

    try:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

        response = httpx.get(
            f"{SCHWAB_API_BASE}{endpoint}", headers=headers, params=params, timeout=30.0
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            # Token might be invalid, try refresh
            logger.warning("Got 401, attempting token refresh...")
            if _refresh_token():
                # Retry request with new token
                headers["Authorization"] = f"Bearer {_token_data.get('access_token')}"
                response = httpx.get(
                    f"{SCHWAB_API_BASE}{endpoint}",
                    headers=headers,
                    params=params,
                    timeout=30.0,
                )
                if response.status_code == 200:
                    return response.json()
            logger.error(f"Request failed after refresh: {response.status_code}")
            return None
        else:
            logger.error(
                f"Schwab API error: {response.status_code} - {response.text[:200]}"
            )
            return None

    except Exception as e:
        logger.error(f"Schwab API request error: {e}")
        return None


def is_schwab_available() -> bool:
    """Check if Schwab API is configured and available"""
    global _schwab_available

    # Check credentials
    api_key = os.getenv("SCHWAB_APP_KEY")
    app_secret = os.getenv("SCHWAB_APP_SECRET")

    if not api_key or not app_secret:
        return False

    # Check if token exists
    if not TOKEN_FILE.exists():
        return False

    # Try to ensure we have a valid token
    if _ensure_token():
        return True

    return _schwab_available


def get_token_status() -> Dict:
    """Get Schwab token status for monitoring (thread-safe)"""
    global _token_data, _token_expiry, _schwab_available

    with _token_lock:
        # Ensure token is loaded
        if _token_data is None:
            _load_token()

        now = datetime.now()

        if _token_data is None:
            return {
                "valid": False,
                "status": "no_token",
                "message": "No Schwab token found. Run schwab_authenticate.py",
                "expires_in_seconds": 0,
                "needs_refresh": True,
            }

        if _token_expiry is None:
            return {
                "valid": False,
                "status": "unknown_expiry",
                "message": "Token expiry unknown",
                "expires_in_seconds": 0,
                "needs_refresh": True,
            }

        expires_in = (_token_expiry - now).total_seconds()
        is_expired = expires_in <= 0
        needs_refresh = expires_in < 300  # Less than 5 minutes

        if is_expired:
            status = "expired"
            message = "Token has expired"
        elif needs_refresh:
            status = "expiring_soon"
            message = f"Token expires in {int(expires_in)}s - will auto-refresh"
        else:
            status = "valid"
            message = f"Token valid for {int(expires_in/60)}m {int(expires_in%60)}s"

        return {
            "valid": not is_expired,
            "status": status,
            "message": message,
            "expires_in_seconds": max(0, int(expires_in)),
            "expires_at": _token_expiry.isoformat() if _token_expiry else None,
            "needs_refresh": needs_refresh,
            "has_refresh_token": (
                _token_data.get("refresh_token") is not None if _token_data else False
            ),
            "schwab_available": _schwab_available,
        }


class SchwabMarketData:
    """Real-time market data from Schwab/ThinkOrSwim"""

    def __init__(self):
        """Initialize Schwab market data provider"""
        self._cache: Dict[str, Dict] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl = 3.0  # Cache TTL in seconds (3s to reduce API calls)

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cache is still valid"""
        if symbol not in self._cache_time:
            return False
        elapsed = (datetime.now() - self._cache_time[symbol]).total_seconds()
        return elapsed < self._cache_ttl

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time quote for a symbol

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with quote data or None
        """
        symbol = symbol.upper()

        # Check cache
        if self._is_cache_valid(symbol):
            return self._cache.get(symbol)

        try:
            data = _make_request(f"/marketdata/v1/quotes", params={"symbols": symbol})

            if not data or symbol not in data:
                return None

            quote_data = data[symbol].get("quote", {})

            result = {
                "symbol": symbol,
                "bid": float(quote_data.get("bidPrice", 0) or 0),
                "ask": float(quote_data.get("askPrice", 0) or 0),
                "last": float(quote_data.get("lastPrice", 0) or 0),
                "bid_size": int(quote_data.get("bidSize", 0) or 0),
                "ask_size": int(quote_data.get("askSize", 0) or 0),
                "volume": int(quote_data.get("totalVolume", 0) or 0),
                "high": float(quote_data.get("highPrice", 0) or 0),
                "low": float(quote_data.get("lowPrice", 0) or 0),
                "open": float(quote_data.get("openPrice", 0) or 0),
                "close": float(quote_data.get("closePrice", 0) or 0),
                "change": float(quote_data.get("netChange", 0) or 0),
                "change_percent": float(
                    quote_data.get("netPercentChangeInDouble", 0) or 0
                ),
                "timestamp": datetime.now().isoformat(),
                "source": "schwab",
            }

            # Calculate change_percent if API returns 0 (common in pre-market)
            if result["change_percent"] == 0 and result["close"] > 0:
                result["change_percent"] = round(
                    (result["change"] / result["close"]) * 100, 2
                )

            # Update cache
            self._cache[symbol] = result
            self._cache_time[symbol] = datetime.now()

            return result

        except Exception as e:
            logger.error(f"Error getting Schwab quote for {symbol}: {e}")
            return None

    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get real-time quotes for multiple symbols

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Dictionary mapping symbols to quote data
        """
        import time as _time

        start = _time.time()

        results = {}
        symbols_to_fetch = []

        # Check cache first
        for symbol in symbols:
            symbol = symbol.upper()
            if self._is_cache_valid(symbol):
                results[symbol] = self._cache[symbol]
            else:
                symbols_to_fetch.append(symbol)

        if not symbols_to_fetch:
            logger.info(f"[SCHWAB] get_quotes: All {len(results)} symbols from cache")
            return results

        try:
            # Schwab API accepts comma-separated symbols
            symbol_str = ",".join(symbols_to_fetch)
            logger.info(
                f"[SCHWAB] Making batch API call for {len(symbols_to_fetch)} symbols"
            )
            t1 = _time.time()
            data = _make_request(
                f"/marketdata/v1/quotes", params={"symbols": symbol_str}
            )
            t2 = _time.time()
            logger.info(f"[SCHWAB] HTTP request took {(t2-t1)*1000:.0f}ms")

            if not data:
                logger.warning(
                    f"[SCHWAB] API returned no data for {len(symbols_to_fetch)} symbols"
                )
                return results

            for symbol in symbols_to_fetch:
                if symbol in data:
                    quote_data = data[symbol].get("quote", {})
                    result = {
                        "symbol": symbol,
                        "bid": float(quote_data.get("bidPrice", 0) or 0),
                        "ask": float(quote_data.get("askPrice", 0) or 0),
                        "last": float(quote_data.get("lastPrice", 0) or 0),
                        "bid_size": int(quote_data.get("bidSize", 0) or 0),
                        "ask_size": int(quote_data.get("askSize", 0) or 0),
                        "volume": int(quote_data.get("totalVolume", 0) or 0),
                        "high": float(quote_data.get("highPrice", 0) or 0),
                        "low": float(quote_data.get("lowPrice", 0) or 0),
                        "open": float(quote_data.get("openPrice", 0) or 0),
                        "close": float(quote_data.get("closePrice", 0) or 0),
                        "change": float(quote_data.get("netChange", 0) or 0),
                        "change_percent": float(
                            quote_data.get("netPercentChangeInDouble", 0) or 0
                        ),
                        "timestamp": datetime.now().isoformat(),
                        "source": "schwab",
                    }

                    # Calculate change_percent if API returns 0 (common in pre-market)
                    if result["change_percent"] == 0 and result["close"] > 0:
                        result["change_percent"] = round(
                            (result["change"] / result["close"]) * 100, 2
                        )

                    results[symbol] = result

                    # Update cache
                    self._cache[symbol] = result
                    self._cache_time[symbol] = datetime.now()

            logger.info(
                f"[SCHWAB] get_quotes TOTAL: {(_time.time()-start)*1000:.0f}ms, returned {len(results)} quotes"
            )

        except Exception as e:
            logger.error(f"Error getting Schwab quotes: {e}")

        return results

    def get_snapshot(self, symbol: str) -> Optional[Dict]:
        """
        Get comprehensive market snapshot for a symbol

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with snapshot data or None
        """
        # For Schwab, snapshot is same as quote (real-time)
        return self.get_quote(symbol)

    def get_price_history(
        self,
        symbol: str,
        period_type: str = "day",
        period: int = 1,
        frequency_type: str = "minute",
        frequency: int = 1,
    ) -> Optional[Dict]:
        """
        Get historical price data

        Args:
            symbol: Stock ticker symbol
            period_type: Type of period (day, month, year, ytd)
            period: Number of periods
            frequency_type: Type of frequency (minute, daily, weekly, monthly)
            frequency: Frequency value

        Returns:
            Dictionary with historical data or None
        """
        try:
            params = {
                "periodType": period_type,
                "period": period,
                "frequencyType": frequency_type,
                "frequency": frequency,
            }

            data = _make_request(
                f"/marketdata/v1/pricehistory",
                params={"symbol": symbol.upper(), **params},
            )

            return data

        except Exception as e:
            logger.error(f"Error getting Schwab price history for {symbol}: {e}")
            return None


# Global instance
_schwab_market_data: Optional[SchwabMarketData] = None


def get_schwab_market_data() -> Optional[SchwabMarketData]:
    """
    Get or create the global Schwab market data instance

    Returns:
        SchwabMarketData instance or None if not available
    """
    global _schwab_market_data

    if not is_schwab_available():
        return None

    if _schwab_market_data is None:
        _schwab_market_data = SchwabMarketData()

    return _schwab_market_data


def get_schwab_quote(symbol: str) -> Optional[Dict]:
    """
    Convenience function to get a real-time quote from Schwab

    Args:
        symbol: Stock ticker symbol

    Returns:
        Quote dictionary or None
    """
    schwab = get_schwab_market_data()
    if schwab:
        return schwab.get_quote(symbol)
    return None


def get_schwab_movers(
    index: str = "$SPX", direction: str = "up", change_type: str = "percent"
) -> List[Dict]:
    """
    Get top movers from Schwab (gainers or losers)

    Args:
        index: Index to get movers for ($SPX, $COMPX, $DJI)
        direction: "up" for gainers, "down" for losers
        change_type: "percent" or "value"

    Returns:
        List of mover dictionaries
    """
    try:
        # Schwab movers endpoint
        endpoint = f"/marketdata/v1/movers/{index}"
        params = {
            "sort": "PERCENT_CHANGE_UP" if direction == "up" else "PERCENT_CHANGE_DOWN",
            "frequency": "0",  # 0 = all, 1 = 5 min, etc.
        }

        data = _make_request(endpoint, params)

        if not data:
            logger.warning(f"No movers data returned for {index}")
            return []

        movers = []
        screeners = data.get("screeners", [])

        # Schwab API returns stocks directly in screeners array (not nested in instruments)
        for item in screeners:
            # Handle both formats: direct stock data or nested instruments
            if "instruments" in item:
                # Old format with nested instruments
                for inst in item.get("instruments", []):
                    movers.append(
                        {
                            "symbol": inst.get("symbol", ""),
                            "description": inst.get("description", ""),
                            "price": inst.get("lastPrice", inst.get("last", 0)),
                            "change": inst.get("netChange", 0),
                            "change_pct": inst.get("netPercentChange", 0)
                            * 100,  # Convert to percentage
                            "volume": inst.get("totalVolume", inst.get("volume", 0)),
                            "direction": direction,
                        }
                    )
            else:
                # Direct stock data format
                movers.append(
                    {
                        "symbol": item.get("symbol", ""),
                        "description": item.get("description", ""),
                        "price": item.get("lastPrice", item.get("last", 0)),
                        "change": item.get("netChange", 0),
                        "change_pct": item.get("netPercentChange", 0)
                        * 100,  # Convert to percentage
                        "volume": item.get("totalVolume", item.get("volume", 0)),
                        "direction": direction,
                    }
                )

        return movers

    except Exception as e:
        logger.error(f"Error getting Schwab movers: {e}")
        return []


def get_all_movers() -> Dict[str, List[Dict]]:
    """
    Get both gainers and losers from multiple indices

    Returns:
        Dictionary with 'gainers' and 'losers' lists
    """
    results = {"gainers": [], "losers": [], "timestamp": datetime.now().isoformat()}

    # Get SPX movers (most common)
    results["gainers"] = get_schwab_movers("$SPX", "up")
    results["losers"] = get_schwab_movers("$SPX", "down")

    # Also try NASDAQ
    nasdaq_gainers = get_schwab_movers("$COMPX", "up")
    for m in nasdaq_gainers:
        if m["symbol"] not in [g["symbol"] for g in results["gainers"]]:
            results["gainers"].append(m)

    # Sort by change percent
    results["gainers"] = sorted(
        results["gainers"], key=lambda x: x.get("change_pct", 0), reverse=True
    )
    results["losers"] = sorted(results["losers"], key=lambda x: x.get("change_pct", 0))

    return results


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Schwab Market Data...")
    print("=" * 50)

    # Check availability
    print(f"\nSchwab Available: {is_schwab_available()}")

    if is_schwab_available():
        schwab = get_schwab_market_data()

        # Test single quote
        print("\n--- Single Quote (AAPL) ---")
        quote = schwab.get_quote("AAPL")
        if quote:
            print(f"Symbol: {quote['symbol']}")
            print(f"Last: ${quote['last']:.2f}")
            print(f"Bid: ${quote['bid']:.2f}")
            print(f"Ask: ${quote['ask']:.2f}")
            print(f"Change: {quote['change']:+.2f} ({quote['change_percent']:+.2f}%)")
        else:
            print("No quote available")

        # Test multiple quotes
        print("\n--- Multiple Quotes ---")
        symbols = ["AAPL", "MSFT", "TSLA", "NVDA", "SPY"]
        quotes = schwab.get_quotes(symbols)
        for sym, q in quotes.items():
            print(f"{sym}: ${q['last']:.2f} ({q['change_percent']:+.2f}%)")

    print("\n" + "=" * 50)
    print("Test complete!")
