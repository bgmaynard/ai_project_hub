"""
CENTRALIZED MARKET DATA BUS
============================
Single source of truth for all market data in the trading platform.
ALL MODULES MUST USE THIS BUS FOR MARKET DATA - ORDER INTEGRITY DEPENDS ON IT!

Data Source:
- SCHWAB ONLY - Real-time, accurate quotes for ALL trading decisions

IMPORTANT: Alpaca data is ONLY for simulator/display purposes, NOT for trading!

CRITICAL RULE: NEVER use stale "last" price - only use live bid/ask from Schwab!
"""

import logging
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Global singleton
_market_data_bus = None
_bus_lock = Lock()


class MarketDataBus:
    """
    Centralized Market Data Bus - ALL market data requests go through here.

    SCHWAB ONLY for trading decisions!
    Alpaca is only for simulator/display purposes.

    This ensures:
    1. All trading decisions use Schwab data ONLY
    2. Never using stale data
    3. Proper validation of quotes
    4. Centralized logging for debugging
    """

    def __init__(self):
        self._schwab = None
        self._schwab_available = False
        self._stats = {
            "schwab_hits": 0,
            "schwab_errors": 0,
            "failed_quotes": 0,
            "stale_data_blocked": 0,
        }
        self._init_providers()
        logger.info(
            "Market Data Bus initialized - Schwab: %s (ONLY SOURCE FOR TRADING)",
            "AVAILABLE" if self._schwab_available else "UNAVAILABLE",
        )

    def _init_providers(self):
        """Initialize market data providers - SCHWAB ONLY for trading"""
        # Initialize Schwab (ONLY SOURCE FOR TRADING)
        try:
            from schwab_market_data import (SchwabMarketData,
                                            is_schwab_available)

            if is_schwab_available():
                self._schwab = SchwabMarketData()
                self._schwab_available = True
                logger.info(
                    "Schwab market data provider initialized - ONLY SOURCE FOR TRADING"
                )
            else:
                logger.error(
                    "CRITICAL: Schwab market data NOT available - trading decisions WILL FAIL!"
                )
                logger.error(
                    "Please ensure Schwab credentials are configured and token is valid."
                )
        except Exception as e:
            logger.error(f"CRITICAL: Failed to initialize Schwab provider: {e}")
            logger.error("Trading decisions will fail without Schwab data!")

    def _validate_quote(self, quote: Dict, symbol: str) -> bool:
        """
        Validate that a quote has usable bid/ask data.
        CRITICAL: Reject quotes with only stale "last" price!
        """
        if not quote:
            return False

        bid = quote.get("bid", 0) or 0
        ask = quote.get("ask", 0) or 0

        # Must have at least one valid bid or ask
        if bid <= 0 and ask <= 0:
            self._stats["stale_data_blocked"] += 1
            logger.warning(
                f"STALE DATA BLOCKED for {symbol}: bid={bid}, ask={ask}, last={quote.get('last')}"
            )
            return False

        return True

    def get_quote(self, symbol: str, require_bid_ask: bool = True) -> Optional[Dict]:
        """
        Get a market quote for a symbol from SCHWAB ONLY.

        CRITICAL: This is the ONLY method that should be used for getting quotes!
        ALL trading decisions MUST use this method!

        Args:
            symbol: Stock ticker symbol
            require_bid_ask: If True, reject quotes without valid bid/ask (default: True)

        Returns:
            Quote dictionary with bid, ask, last, etc. or None if unavailable
        """
        symbol = symbol.upper()
        quote = None

        # SCHWAB ONLY - No fallback to Alpaca for trading decisions!
        if not self._schwab_available or not self._schwab:
            self._stats["failed_quotes"] += 1
            logger.error(
                f"CANNOT GET QUOTE for {symbol} - Schwab not available! Trading decisions require Schwab data!"
            )
            return None

        try:
            quote = self._schwab.get_quote(symbol)
            if quote and (not require_bid_ask or self._validate_quote(quote, symbol)):
                self._stats["schwab_hits"] += 1
                logger.debug(
                    f"Quote from Schwab for {symbol}: bid=${quote.get('bid'):.2f}, ask=${quote.get('ask'):.2f}"
                )
                return quote
            else:
                self._stats["failed_quotes"] += 1
                logger.error(
                    f"Invalid quote data from Schwab for {symbol} - bid/ask may be missing"
                )
                return None
        except Exception as e:
            self._stats["schwab_errors"] += 1
            self._stats["failed_quotes"] += 1
            logger.error(f"Schwab quote FAILED for {symbol}: {e}")
            return None

    def get_quotes(
        self, symbols: List[str], require_bid_ask: bool = True
    ) -> Dict[str, Dict]:
        """
        Get quotes for multiple symbols from SCHWAB ONLY.

        Args:
            symbols: List of stock ticker symbols
            require_bid_ask: If True, reject quotes without valid bid/ask

        Returns:
            Dictionary mapping symbols to quote data
        """
        results = {}

        # SCHWAB ONLY - No Alpaca fallback!
        if not self._schwab_available or not self._schwab:
            logger.error(
                "CANNOT GET BATCH QUOTES - Schwab not available! Trading decisions require Schwab data!"
            )
            return results

        try:
            schwab_quotes = self._schwab.get_quotes(symbols)
            for symbol, quote in schwab_quotes.items():
                if not require_bid_ask or self._validate_quote(quote, symbol):
                    results[symbol] = quote
                    self._stats["schwab_hits"] += 1
                else:
                    self._stats["failed_quotes"] += 1
        except Exception as e:
            self._stats["schwab_errors"] += 1
            logger.error(f"Schwab batch quote FAILED: {e}")

        return results

    def get_price_for_order(self, symbol: str, side: str) -> Optional[float]:
        """
        Get the appropriate price for placing an order.

        CRITICAL: This ensures orders are placed at correct prices!

        Args:
            symbol: Stock ticker symbol
            side: "BUY" or "SELL"

        Returns:
            Appropriate limit price or None if no valid data
        """
        quote = self.get_quote(symbol, require_bid_ask=True)
        if not quote:
            return None

        side = side.upper()

        if side == "BUY":
            # For buys: use ask price (what sellers are offering)
            price = quote.get("ask", 0)
            if not price or price <= 0:
                # Fallback to bid + 2% if no ask
                bid = quote.get("bid", 0)
                if bid and bid > 0:
                    price = bid * 1.02
        else:
            # For sells: use bid price (what buyers are offering)
            price = quote.get("bid", 0)
            if not price or price <= 0:
                # Fallback to ask - 2% if no bid
                ask = quote.get("ask", 0)
                if ask and ask > 0:
                    price = ask * 0.98

        # Final validation
        if not price or price <= 0:
            logger.error(
                f"Cannot determine order price for {symbol} {side} - quote: {quote}"
            )
            return None

        return round(price, 2)

    def get_stats(self) -> Dict:
        """Get usage statistics for the data bus"""
        return {
            **self._stats,
            "schwab_available": self._schwab_available,
            "data_source": "SCHWAB_ONLY",
            "status": (
                "READY" if self._schwab_available else "ERROR: Schwab not available!"
            ),
        }

    def refresh_providers(self):
        """Re-initialize providers (useful if tokens were refreshed)"""
        self._init_providers()


def get_market_data_bus() -> MarketDataBus:
    """
    Get the global market data bus singleton.

    ALL modules should use this to get market data!
    """
    global _market_data_bus

    with _bus_lock:
        if _market_data_bus is None:
            _market_data_bus = MarketDataBus()

    return _market_data_bus


def get_quote(symbol: str) -> Optional[Dict]:
    """
    Convenience function to get a quote via the bus.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Quote dictionary or None
    """
    return get_market_data_bus().get_quote(symbol)


def get_order_price(symbol: str, side: str) -> Optional[float]:
    """
    Convenience function to get order price via the bus.

    Args:
        symbol: Stock ticker symbol
        side: "BUY" or "SELL"

    Returns:
        Appropriate limit price or None
    """
    return get_market_data_bus().get_price_for_order(symbol, side)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Centralized Market Data Bus...")
    print("=" * 60)

    bus = get_market_data_bus()

    print(f"\nStats: {bus.get_stats()}")

    # Test single quote
    print("\n--- Single Quote (AAPL) ---")
    quote = bus.get_quote("AAPL")
    if quote:
        print(f"Symbol: {quote['symbol']}")
        print(f"Bid: ${quote.get('bid', 0):.2f}")
        print(f"Ask: ${quote.get('ask', 0):.2f}")
        print(f"Source: {quote.get('source', 'unknown')}")
    else:
        print("No quote available")

    # Test order pricing
    print("\n--- Order Pricing ---")
    buy_price = bus.get_price_for_order("AAPL", "BUY")
    sell_price = bus.get_price_for_order("AAPL", "SELL")
    print(f"BUY price for AAPL: ${buy_price:.2f}" if buy_price else "No buy price")
    print(f"SELL price for AAPL: ${sell_price:.2f}" if sell_price else "No sell price")

    # Test batch quotes
    print("\n--- Batch Quotes ---")
    symbols = ["AAPL", "MSFT", "TSLA", "SPY"]
    quotes = bus.get_quotes(symbols)
    for sym, q in quotes.items():
        print(
            f"{sym}: bid=${q.get('bid', 0):.2f}, ask=${q.get('ask', 0):.2f}, source={q.get('source', 'unknown')}"
        )

    print(f"\nFinal Stats: {bus.get_stats()}")
    print("\n" + "=" * 60)
    print("Test complete!")
