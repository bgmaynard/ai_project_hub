"""
Unified Market Data Provider
============================
Single source of truth for all market data across the platform.
Prioritizes Schwab real-time data, falls back to Alpaca if unavailable.

This ensures ALL modules show consistent prices.

Author: AI Trading Bot Team
Version: 1.0
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import threading
import time

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class DataSource(str, Enum):
    """Data source priority"""
    SCHWAB = "schwab"      # Primary - Real-time Schwab/TOS data
    ALPACA = "alpaca"      # Fallback - Alpaca market data
    CACHE = "cache"        # Cached data (stale)


@dataclass
class QuoteData:
    """Standardized quote data structure"""
    symbol: str
    bid: float
    ask: float
    last: float
    bid_size: int = 0
    ask_size: int = 0
    volume: int = 0
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    close: float = 0.0
    change: float = 0.0
    change_percent: float = 0.0
    timestamp: str = ""
    source: str = ""

    @property
    def mid(self) -> float:
        """Calculate mid price"""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread(self) -> float:
        """Calculate spread"""
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "mid": self.mid,
            "spread": self.spread,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "volume": self.volume,
            "high": self.high,
            "low": self.low,
            "open": self.open,
            "close": self.close,
            "change": self.change,
            "change_percent": self.change_percent,
            "timestamp": self.timestamp,
            "source": self.source
        }


class UnifiedMarketData:
    """
    Unified market data provider that ensures consistent prices across all modules.

    Priority order:
    1. Schwab (real-time, most accurate)
    2. Alpaca (reliable fallback)
    3. Cache (last known prices)
    """

    def __init__(self):
        """Initialize unified market data provider"""
        self._schwab_client = None
        self._alpaca_client = None
        self._cache: Dict[str, QuoteData] = {}
        self._cache_ttl = 0.5  # Cache TTL in seconds (500ms for near real-time)
        self._cache_timestamps: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        self._source_stats = {
            DataSource.SCHWAB: {"requests": 0, "success": 0, "failures": 0},
            DataSource.ALPACA: {"requests": 0, "success": 0, "failures": 0}
        }

        # Initialize data sources
        self._init_schwab()
        self._init_alpaca()

        logger.info(f"[UNIFIED] Market data initialized - Schwab: {self._schwab_available}, Alpaca: {self._alpaca_available}")

    def _init_schwab(self):
        """Initialize Schwab client"""
        self._schwab_available = False
        try:
            from schwab_market_data import get_schwab_market_data, is_schwab_available
            if is_schwab_available():
                self._schwab_client = get_schwab_market_data()
                if self._schwab_client:
                    self._schwab_available = True
                    logger.info("[UNIFIED] Schwab data source available")
        except ImportError:
            logger.warning("[UNIFIED] Schwab module not available")
        except Exception as e:
            logger.warning(f"[UNIFIED] Schwab init error: {e}")

    def _init_alpaca(self):
        """Initialize Alpaca client"""
        self._alpaca_available = False
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest, StockSnapshotRequest

            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')

            if api_key and secret_key:
                self._alpaca_client = StockHistoricalDataClient(api_key, secret_key)
                self._alpaca_available = True
                logger.info("[UNIFIED] Alpaca data source available")
        except ImportError:
            logger.warning("[UNIFIED] Alpaca module not available")
        except Exception as e:
            logger.warning(f"[UNIFIED] Alpaca init error: {e}")

    def _get_from_schwab(self, symbol: str) -> Optional[QuoteData]:
        """Get quote from Schwab"""
        if not self._schwab_available or not self._schwab_client:
            return None

        self._source_stats[DataSource.SCHWAB]["requests"] += 1

        try:
            quote = self._schwab_client.get_quote(symbol)
            if quote:
                self._source_stats[DataSource.SCHWAB]["success"] += 1
                return QuoteData(
                    symbol=symbol.upper(),
                    bid=float(quote.get("bid", 0)),
                    ask=float(quote.get("ask", 0)),
                    last=float(quote.get("last", 0)),
                    bid_size=int(quote.get("bid_size", 0)),
                    ask_size=int(quote.get("ask_size", 0)),
                    volume=int(quote.get("volume", 0)),
                    high=float(quote.get("high", 0)),
                    low=float(quote.get("low", 0)),
                    open=float(quote.get("open", 0)),
                    close=float(quote.get("close", 0)),
                    change=float(quote.get("change", 0)),
                    change_percent=float(quote.get("change_percent", 0)),
                    timestamp=quote.get("timestamp", datetime.now().isoformat()),
                    source=DataSource.SCHWAB.value
                )
        except Exception as e:
            self._source_stats[DataSource.SCHWAB]["failures"] += 1
            logger.debug(f"[UNIFIED] Schwab quote failed for {symbol}: {e}")

        return None

    def _get_from_alpaca(self, symbol: str) -> Optional[QuoteData]:
        """Get quote from Alpaca"""
        if not self._alpaca_available or not self._alpaca_client:
            return None

        self._source_stats[DataSource.ALPACA]["requests"] += 1

        try:
            from alpaca.data.requests import StockLatestQuoteRequest, StockSnapshotRequest

            # Try snapshot first (more data)
            try:
                request = StockSnapshotRequest(symbol_or_symbols=symbol.upper())
                snapshots = self._alpaca_client.get_stock_snapshot(request)

                if symbol.upper() in snapshots:
                    s = snapshots[symbol.upper()]

                    bid = ask = last = 0.0
                    if s.latest_quote:
                        bid = float(s.latest_quote.bid_price)
                        ask = float(s.latest_quote.ask_price)
                    if s.latest_trade:
                        last = float(s.latest_trade.price)

                    high = low = open_price = close = volume = 0
                    prev_close = 0.0
                    if s.daily_bar:
                        high = float(s.daily_bar.high)
                        low = float(s.daily_bar.low)
                        open_price = float(s.daily_bar.open)
                        close = float(s.daily_bar.close)
                        volume = int(s.daily_bar.volume)
                    if s.previous_daily_bar:
                        prev_close = float(s.previous_daily_bar.close)

                    change = close - prev_close if prev_close > 0 else 0
                    change_pct = (change / prev_close * 100) if prev_close > 0 else 0

                    self._source_stats[DataSource.ALPACA]["success"] += 1
                    return QuoteData(
                        symbol=symbol.upper(),
                        bid=bid,
                        ask=ask,
                        last=last if last > 0 else close,
                        volume=volume,
                        high=high,
                        low=low,
                        open=open_price,
                        close=close,
                        change=change,
                        change_percent=change_pct,
                        timestamp=datetime.now().isoformat(),
                        source=DataSource.ALPACA.value
                    )
            except Exception:
                pass

            # Fallback to quote only
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol.upper())
            quotes = self._alpaca_client.get_stock_latest_quote(request)

            if symbol.upper() in quotes:
                q = quotes[symbol.upper()]
                bid = float(q.bid_price)
                ask = float(q.ask_price)

                self._source_stats[DataSource.ALPACA]["success"] += 1
                return QuoteData(
                    symbol=symbol.upper(),
                    bid=bid,
                    ask=ask,
                    last=(bid + ask) / 2,
                    bid_size=int(q.bid_size),
                    ask_size=int(q.ask_size),
                    timestamp=q.timestamp.isoformat() if q.timestamp else datetime.now().isoformat(),
                    source=DataSource.ALPACA.value
                )

        except Exception as e:
            self._source_stats[DataSource.ALPACA]["failures"] += 1
            logger.debug(f"[UNIFIED] Alpaca quote failed for {symbol}: {e}")

        return None

    def _update_cache(self, symbol: str, quote: QuoteData):
        """Update the cache"""
        with self._lock:
            self._cache[symbol.upper()] = quote
            self._cache_timestamps[symbol.upper()] = datetime.now()

    def _get_from_cache(self, symbol: str) -> Optional[QuoteData]:
        """Get from cache if not stale"""
        with self._lock:
            if symbol.upper() in self._cache:
                cached_time = self._cache_timestamps.get(symbol.upper())
                if cached_time and (datetime.now() - cached_time).total_seconds() < self._cache_ttl:
                    quote = self._cache[symbol.upper()]
                    quote.source = DataSource.CACHE.value
                    return quote
        return None

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def get_quote(self, symbol: str, use_cache: bool = True) -> Optional[Dict]:
        """
        Get real-time quote for a symbol.

        Priority: Schwab -> Alpaca -> Cache

        Args:
            symbol: Stock ticker symbol
            use_cache: Whether to use cached data if fresh

        Returns:
            Quote dictionary with standardized fields
        """
        symbol = symbol.upper()

        # Check cache first if enabled
        if use_cache:
            cached = self._get_from_cache(symbol)
            if cached:
                return cached.to_dict()

        # Try Schwab first (primary source)
        quote = self._get_from_schwab(symbol)

        # Fallback to Alpaca
        if quote is None:
            quote = self._get_from_alpaca(symbol)

        # Update cache and return
        if quote:
            self._update_cache(symbol, quote)
            return quote.to_dict()

        # Last resort - return stale cache
        if symbol in self._cache:
            stale = self._cache[symbol]
            stale.source = f"{DataSource.CACHE.value}_stale"
            return stale.to_dict()

        return None

    def get_quotes(self, symbols: List[str], use_cache: bool = True) -> Dict[str, Dict]:
        """
        Get quotes for multiple symbols.

        Args:
            symbols: List of stock ticker symbols
            use_cache: Whether to use cached data if fresh

        Returns:
            Dictionary mapping symbols to quote data
        """
        results = {}
        symbols_to_fetch = []

        # Check cache first
        if use_cache:
            for symbol in symbols:
                cached = self._get_from_cache(symbol.upper())
                if cached:
                    results[symbol.upper()] = cached.to_dict()
                else:
                    symbols_to_fetch.append(symbol)
        else:
            symbols_to_fetch = symbols

        # Batch fetch from Schwab
        if symbols_to_fetch and self._schwab_available and self._schwab_client:
            try:
                schwab_quotes = self._schwab_client.get_quotes(symbols_to_fetch)
                for symbol, quote in schwab_quotes.items():
                    quote_data = QuoteData(
                        symbol=symbol,
                        bid=float(quote.get("bid", 0)),
                        ask=float(quote.get("ask", 0)),
                        last=float(quote.get("last", 0)),
                        bid_size=int(quote.get("bid_size", 0)),
                        ask_size=int(quote.get("ask_size", 0)),
                        volume=int(quote.get("volume", 0)),
                        high=float(quote.get("high", 0)),
                        low=float(quote.get("low", 0)),
                        open=float(quote.get("open", 0)),
                        close=float(quote.get("close", 0)),
                        change=float(quote.get("change", 0)),
                        change_percent=float(quote.get("change_percent", 0)),
                        timestamp=quote.get("timestamp", datetime.now().isoformat()),
                        source=DataSource.SCHWAB.value
                    )
                    self._update_cache(symbol, quote_data)
                    results[symbol] = quote_data.to_dict()
                    symbols_to_fetch.remove(symbol) if symbol in symbols_to_fetch else None
            except Exception as e:
                logger.debug(f"[UNIFIED] Schwab batch quote failed: {e}")

        # Fallback remaining to Alpaca
        for symbol in symbols_to_fetch:
            quote = self._get_from_alpaca(symbol)
            if quote:
                self._update_cache(symbol, quote)
                results[symbol.upper()] = quote.to_dict()

        return results

    def get_snapshot(self, symbol: str) -> Optional[Dict]:
        """
        Get comprehensive market snapshot for a symbol.
        Same as get_quote but semantic clarity for full data.
        """
        return self.get_quote(symbol, use_cache=False)

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol (convenience method).

        Returns the last traded price, or mid price if last unavailable.
        """
        quote = self.get_quote(symbol)
        if quote:
            if quote.get("last", 0) > 0:
                return quote["last"]
            return quote.get("mid", 0)
        return None

    def get_bid_ask(self, symbol: str) -> Optional[tuple]:
        """
        Get bid/ask prices for a symbol.

        Returns:
            Tuple of (bid, ask) or None
        """
        quote = self.get_quote(symbol)
        if quote:
            return (quote.get("bid", 0), quote.get("ask", 0))
        return None

    def get_status(self) -> Dict:
        """Get status of all data sources"""
        return {
            "schwab": {
                "available": self._schwab_available,
                "stats": self._source_stats[DataSource.SCHWAB]
            },
            "alpaca": {
                "available": self._alpaca_available,
                "stats": self._source_stats[DataSource.ALPACA]
            },
            "cache_size": len(self._cache),
            "cache_ttl_seconds": self._cache_ttl,
            "primary_source": DataSource.SCHWAB.value if self._schwab_available else DataSource.ALPACA.value,
            "timestamp": datetime.now().isoformat()
        }

    def clear_cache(self):
        """Clear the quote cache"""
        with self._lock:
            self._cache.clear()
            self._cache_timestamps.clear()
        logger.info("[UNIFIED] Cache cleared")

    def set_cache_ttl(self, seconds: int):
        """Set cache TTL in seconds"""
        self._cache_ttl = max(1, seconds)
        logger.info(f"[UNIFIED] Cache TTL set to {self._cache_ttl} seconds")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_unified_provider: Optional[UnifiedMarketData] = None
_provider_lock = threading.Lock()


def get_unified_market_data() -> UnifiedMarketData:
    """Get or create the unified market data singleton"""
    global _unified_provider

    if _unified_provider is None:
        with _provider_lock:
            if _unified_provider is None:
                _unified_provider = UnifiedMarketData()

    return _unified_provider


# ============================================================================
# CONVENIENCE FUNCTIONS (Drop-in replacements for existing code)
# ============================================================================

def get_quote(symbol: str) -> Optional[Dict]:
    """Get quote using unified provider"""
    return get_unified_market_data().get_quote(symbol)


def get_quotes(symbols: List[str]) -> Dict[str, Dict]:
    """Get multiple quotes using unified provider"""
    return get_unified_market_data().get_quotes(symbols)


def get_snapshot(symbol: str) -> Optional[Dict]:
    """Get snapshot using unified provider"""
    return get_unified_market_data().get_snapshot(symbol)


def get_price(symbol: str) -> Optional[float]:
    """Get current price using unified provider"""
    return get_unified_market_data().get_price(symbol)


def get_bid_ask(symbol: str) -> Optional[tuple]:
    """Get bid/ask using unified provider"""
    return get_unified_market_data().get_bid_ask(symbol)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    print("Testing Unified Market Data Provider...")
    print("=" * 60)

    provider = get_unified_market_data()

    # Test status
    print("\n=== Data Source Status ===")
    status = provider.get_status()
    print(json.dumps(status, indent=2))

    # Test single quote
    print("\n=== Single Quote Test (AAPL) ===")
    quote = provider.get_quote("AAPL")
    if quote:
        print(f"Symbol: {quote['symbol']}")
        print(f"Last: ${quote['last']:.2f}")
        print(f"Bid: ${quote['bid']:.2f} x {quote.get('bid_size', 0)}")
        print(f"Ask: ${quote['ask']:.2f} x {quote.get('ask_size', 0)}")
        print(f"Source: {quote['source']}")
    else:
        print("No quote available")

    # Test multiple quotes
    print("\n=== Multiple Quotes Test ===")
    symbols = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT"]
    quotes = provider.get_quotes(symbols)
    for sym, q in quotes.items():
        print(f"{sym}: ${q['last']:.2f} (source: {q['source']})")

    # Test convenience function
    print("\n=== Convenience Function Test ===")
    price = get_price("SPY")
    print(f"SPY price: ${price:.2f}" if price else "SPY price unavailable")

    print("\n" + "=" * 60)
    print("Test complete!")
