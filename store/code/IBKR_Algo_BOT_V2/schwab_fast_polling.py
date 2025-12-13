"""
Schwab Fast Polling for Near Real-Time Market Data
Provides fast market data updates by polling the Schwab HTTP API in a background thread
This is a fallback when WebSocket streaming is not available
"""
import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Set
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Global state
_quote_cache: Dict[str, Dict] = {}
_subscribed_symbols: Set[str] = set()
_polling_thread: Optional[threading.Thread] = None
_is_running = False
_poll_interval = 0.3  # 300ms polling interval for near real-time
_quote_callbacks: List[Callable] = []
_last_update: Dict[str, datetime] = {}

# Thread pool for parallel quote fetching
_executor = ThreadPoolExecutor(max_workers=4)


def get_quote_cache() -> Dict[str, Dict]:
    """Get the current quote cache"""
    return _quote_cache.copy()


def get_cached_quote(symbol: str) -> Optional[Dict]:
    """Get cached quote for a symbol"""
    return _quote_cache.get(symbol.upper())


def register_callback(callback: Callable):
    """Register a callback for quote updates"""
    if callback not in _quote_callbacks:
        _quote_callbacks.append(callback)


def unregister_callback(callback: Callable):
    """Unregister a callback"""
    if callback in _quote_callbacks:
        _quote_callbacks.remove(callback)


def _notify_callbacks(symbol: str, quote: Dict):
    """Notify all registered callbacks"""
    for callback in _quote_callbacks:
        try:
            callback(symbol, quote)
        except Exception as e:
            logger.error(f"Callback error: {e}")


def _fetch_quotes_batch(symbols: List[str]) -> Dict[str, Dict]:
    """Fetch quotes for a batch of symbols"""
    try:
        from schwab_market_data import get_schwab_market_data
        market_data = get_schwab_market_data()
        if not market_data:
            return {}

        quotes = market_data.get_quotes(symbols)
        return quotes
    except Exception as e:
        logger.error(f"Error fetching quotes: {e}")
        return {}


def _poll_loop():
    """Main polling loop"""
    global _quote_cache, _is_running

    logger.info("Schwab fast polling started")

    while _is_running:
        try:
            if not _subscribed_symbols:
                time.sleep(_poll_interval)
                continue

            symbols = list(_subscribed_symbols)

            # Split into batches of 50 for efficiency
            batch_size = 50
            all_quotes = {}

            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                quotes = _fetch_quotes_batch(batch)
                all_quotes.update(quotes)

            # Update cache and notify callbacks
            now = datetime.now()
            for symbol, quote in all_quotes.items():
                old_quote = _quote_cache.get(symbol)
                _quote_cache[symbol] = quote
                _last_update[symbol] = now

                # Only notify if price changed
                if old_quote is None or old_quote.get('last') != quote.get('last'):
                    _notify_callbacks(symbol, quote)

            # Sleep for poll interval
            time.sleep(_poll_interval)

        except Exception as e:
            logger.error(f"Polling error: {e}")
            time.sleep(1)  # Sleep longer on error

    logger.info("Schwab fast polling stopped")


def start_polling(symbols: Optional[List[str]] = None):
    """Start the fast polling service"""
    global _polling_thread, _subscribed_symbols, _is_running

    if _is_running:
        logger.info("Fast polling already running")
        if symbols:
            _subscribed_symbols.update(s.upper() for s in symbols)
        return True

    if symbols:
        _subscribed_symbols.update(s.upper() for s in symbols)

    _is_running = True
    _polling_thread = threading.Thread(target=_poll_loop, daemon=True)
    _polling_thread.start()
    logger.info(f"Schwab fast polling started with {len(_subscribed_symbols)} symbols")
    return True


def stop_polling():
    """Stop the fast polling service"""
    global _is_running
    _is_running = False
    logger.info("Schwab fast polling stop requested")


def subscribe(symbols: List[str]):
    """Subscribe to symbols for polling"""
    global _subscribed_symbols
    symbols = [s.upper() for s in symbols]
    _subscribed_symbols.update(symbols)
    logger.info(f"Added {len(symbols)} symbols to fast polling")


def unsubscribe(symbols: List[str]):
    """Unsubscribe from symbols"""
    global _subscribed_symbols
    for s in symbols:
        _subscribed_symbols.discard(s.upper())


def get_status() -> Dict:
    """Get polling status"""
    return {
        "running": _is_running,
        "mode": "fast_polling",
        "poll_interval_ms": int(_poll_interval * 1000),
        "subscribed_symbols": list(_subscribed_symbols),
        "cached_quotes": len(_quote_cache),
        "callbacks_registered": len(_quote_callbacks)
    }


def set_poll_interval(interval_ms: int):
    """Set polling interval in milliseconds (min 100ms, max 5000ms)"""
    global _poll_interval
    interval_ms = max(100, min(5000, interval_ms))
    _poll_interval = interval_ms / 1000.0
    logger.info(f"Poll interval set to {interval_ms}ms")


# Singleton access
_fast_poller = None


def get_fast_poller():
    """Get the fast poller singleton"""
    global _fast_poller
    if _fast_poller is None:
        _fast_poller = {
            "start": start_polling,
            "stop": stop_polling,
            "subscribe": subscribe,
            "unsubscribe": unsubscribe,
            "get_quote": get_cached_quote,
            "get_all_quotes": get_quote_cache,
            "get_status": get_status,
            "register_callback": register_callback,
            "set_interval": set_poll_interval
        }
    return _fast_poller
