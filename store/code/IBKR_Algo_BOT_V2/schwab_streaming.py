"""
Schwab Real-Time WebSocket Streaming
Provides sub-second market data updates via Schwab's streaming API
"""
import asyncio
import json
import logging
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable, Set
from pathlib import Path

logger = logging.getLogger(__name__)

# Global state
_stream_client = None
_quote_cache: Dict[str, Dict] = {}
_book_cache: Dict[str, Dict] = {}  # Level 2 order book cache
_quote_callbacks: List[Callable] = []
_book_callbacks: List[Callable] = []  # L2 book callbacks
_subscribed_symbols: Set[str] = set()
_book_symbols: Set[str] = set()  # Symbols with L2 book subscription
_stream_thread: Optional[threading.Thread] = None
_stream_loop: Optional[asyncio.AbstractEventLoop] = None
_is_running = False

TOKEN_FILE = Path(__file__).parent / "schwab_token.json"


def get_quote_cache() -> Dict[str, Dict]:
    """Get the current quote cache (updated in real-time by streaming)"""
    return _quote_cache.copy()


def get_cached_quote(symbol: str) -> Optional[Dict]:
    """Get cached quote for a symbol"""
    return _quote_cache.get(symbol.upper())


def get_book_cache() -> Dict[str, Dict]:
    """Get the current L2 book cache"""
    return _book_cache.copy()


def get_cached_book(symbol: str) -> Optional[Dict]:
    """Get cached L2 book for a symbol"""
    return _book_cache.get(symbol.upper())


def register_book_callback(callback: Callable):
    """Register a callback for L2 book updates"""
    if callback not in _book_callbacks:
        _book_callbacks.append(callback)


def _notify_book_callbacks(symbol: str, book: Dict):
    """Notify all registered callbacks of a book update"""
    for callback in _book_callbacks:
        try:
            callback(symbol, book)
        except Exception as e:
            logger.error(f"Book callback error: {e}")


def register_callback(callback: Callable):
    """Register a callback to be called when quotes update"""
    if callback not in _quote_callbacks:
        _quote_callbacks.append(callback)


def unregister_callback(callback: Callable):
    """Unregister a callback"""
    if callback in _quote_callbacks:
        _quote_callbacks.remove(callback)


def _notify_callbacks(symbol: str, quote: Dict):
    """Notify all registered callbacks of a quote update"""
    for callback in _quote_callbacks:
        try:
            callback(symbol, quote)
        except Exception as e:
            logger.error(f"Callback error: {e}")


async def _run_stream():
    """Main streaming coroutine"""
    global _stream_client, _is_running, _quote_cache

    try:
        import schwab
        from schwab.streaming import StreamClient

        # Load token
        if not TOKEN_FILE.exists():
            logger.error("Schwab token file not found")
            return

        app_key = os.getenv('SCHWAB_APP_KEY')
        app_secret = os.getenv('SCHWAB_APP_SECRET')
        if not app_key or not app_secret:
            logger.error("SCHWAB_APP_KEY or SCHWAB_APP_SECRET not set")
            return

        # Create client from token file (requires both app_key and app_secret)
        client = schwab.auth.client_from_token_file(str(TOKEN_FILE), app_key, app_secret)
        _stream_client = StreamClient(client)

        # Login to stream
        await _stream_client.login()
        logger.info("Schwab streaming: logged in successfully")

        # Handler for quote updates
        def quote_handler(msg):
            """Process incoming quote messages"""
            try:
                if 'content' in msg:
                    for item in msg['content']:
                        symbol = item.get('key', '').upper()
                        if symbol:
                            quote = {
                                'symbol': symbol,
                                'bid': item.get('BID_PRICE', item.get('1', 0)),
                                'ask': item.get('ASK_PRICE', item.get('2', 0)),
                                'last': item.get('LAST_PRICE', item.get('3', 0)),
                                'bid_size': item.get('BID_SIZE', item.get('4', 0)),
                                'ask_size': item.get('ASK_SIZE', item.get('5', 0)),
                                'volume': item.get('TOTAL_VOLUME', item.get('8', 0)),
                                'high': item.get('HIGH_PRICE', item.get('12', 0)),
                                'low': item.get('LOW_PRICE', item.get('13', 0)),
                                'close': item.get('CLOSE_PRICE', item.get('15', 0)),
                                'open': item.get('OPEN_PRICE', item.get('28', 0)),
                                'change': item.get('NET_CHANGE', item.get('29', 0)),
                                'change_percent': item.get('NET_CHANGE_PERCENT', item.get('30', 0)),
                                'timestamp': datetime.now().isoformat(),
                                'source': 'schwab_stream'
                            }
                            _quote_cache[symbol] = quote
                            _notify_callbacks(symbol, quote)
            except Exception as e:
                logger.error(f"Quote handler error: {e}")

        # Add handler
        _stream_client.add_level_one_equity_handler(quote_handler)

        # Handler for NYSE Level 2 book updates
        def nyse_book_handler(msg):
            """Process NYSE L2 book messages"""
            try:
                if 'content' in msg:
                    for item in msg['content']:
                        symbol = item.get('key', '').upper()
                        if symbol:
                            # Parse bids and asks from book data
                            bids = []
                            asks = []
                            # Book data comes as arrays of price/size/exchange
                            for i in range(10):  # Up to 10 levels
                                bid_price = item.get(f'BID_PRICE_{i}', item.get(str(i*3), 0))
                                bid_size = item.get(f'BID_SIZE_{i}', item.get(str(i*3+1), 0))
                                ask_price = item.get(f'ASK_PRICE_{i}', item.get(str(i*3+10), 0))
                                ask_size = item.get(f'ASK_SIZE_{i}', item.get(str(i*3+11), 0))
                                if bid_price and bid_size:
                                    bids.append({'price': bid_price, 'size': bid_size, 'exchange': 'NYSE'})
                                if ask_price and ask_size:
                                    asks.append({'price': ask_price, 'size': ask_size, 'exchange': 'NYSE'})

                            book = {
                                'symbol': symbol,
                                'bids': sorted(bids, key=lambda x: x['price'], reverse=True),
                                'asks': sorted(asks, key=lambda x: x['price']),
                                'timestamp': datetime.now().isoformat(),
                                'source': 'schwab_nyse_book'
                            }
                            _book_cache[symbol] = book
                            _notify_book_callbacks(symbol, book)
                            logger.debug(f"NYSE book update: {symbol} - {len(bids)} bids, {len(asks)} asks")
            except Exception as e:
                logger.error(f"NYSE book handler error: {e}")

        # Handler for NASDAQ Level 2 book updates
        def nasdaq_book_handler(msg):
            """Process NASDAQ L2 book messages"""
            try:
                if 'content' in msg:
                    for item in msg['content']:
                        symbol = item.get('key', '').upper()
                        if symbol:
                            bids = []
                            asks = []
                            for i in range(10):
                                bid_price = item.get(f'BID_PRICE_{i}', item.get(str(i*3), 0))
                                bid_size = item.get(f'BID_SIZE_{i}', item.get(str(i*3+1), 0))
                                ask_price = item.get(f'ASK_PRICE_{i}', item.get(str(i*3+10), 0))
                                ask_size = item.get(f'ASK_SIZE_{i}', item.get(str(i*3+11), 0))
                                mm = item.get(f'MARKET_MAKER_{i}', '')
                                if bid_price and bid_size:
                                    bids.append({'price': bid_price, 'size': bid_size, 'market_maker': mm})
                                if ask_price and ask_size:
                                    asks.append({'price': ask_price, 'size': ask_size, 'market_maker': mm})

                            book = {
                                'symbol': symbol,
                                'bids': sorted(bids, key=lambda x: x['price'], reverse=True),
                                'asks': sorted(asks, key=lambda x: x['price']),
                                'timestamp': datetime.now().isoformat(),
                                'source': 'schwab_nasdaq_book'
                            }
                            _book_cache[symbol] = book
                            _notify_book_callbacks(symbol, book)
                            logger.debug(f"NASDAQ book update: {symbol} - {len(bids)} bids, {len(asks)} asks")
            except Exception as e:
                logger.error(f"NASDAQ book handler error: {e}")

        # Add book handlers
        try:
            _stream_client.add_nyse_book_handler(nyse_book_handler)
            _stream_client.add_nasdaq_book_handler(nasdaq_book_handler)
            logger.info("Schwab streaming: L2 book handlers registered")
        except Exception as e:
            logger.warning(f"Could not add book handlers: {e}")

        # Subscribe to symbols if any are pending
        if _subscribed_symbols:
            await _stream_client.level_one_equity_subs(list(_subscribed_symbols))
            logger.info(f"Schwab streaming: subscribed to {len(_subscribed_symbols)} symbols")

        # Subscribe to L2 books if any symbols pending
        if _book_symbols:
            try:
                await _stream_client.nasdaq_book_subs(list(_book_symbols))
                await _stream_client.nyse_book_subs(list(_book_symbols))
                logger.info(f"Schwab streaming: L2 book subscribed to {len(_book_symbols)} symbols")
            except Exception as e:
                logger.warning(f"Could not subscribe to L2 books: {e}")

        _is_running = True
        logger.info("Schwab streaming: started")

        # Keep stream alive
        while _is_running:
            await _stream_client.handle_message()

    except Exception as e:
        logger.error(f"Schwab streaming error: {e}")
        _is_running = False


def _stream_thread_func():
    """Thread function to run the async stream"""
    global _stream_loop
    _stream_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_stream_loop)
    try:
        _stream_loop.run_until_complete(_run_stream())
    except Exception as e:
        logger.error(f"Stream thread error: {e}")
    finally:
        _stream_loop.close()


def start_streaming(symbols: Optional[List[str]] = None):
    """Start the streaming service"""
    global _stream_thread, _subscribed_symbols, _is_running

    if _is_running:
        logger.info("Streaming already running")
        return True

    if symbols:
        _subscribed_symbols.update(s.upper() for s in symbols)

    _stream_thread = threading.Thread(target=_stream_thread_func, daemon=True)
    _stream_thread.start()
    logger.info("Schwab streaming thread started")
    return True


def stop_streaming():
    """Stop the streaming service"""
    global _is_running, _stream_client
    _is_running = False
    if _stream_client:
        try:
            # Close the stream
            asyncio.run_coroutine_threadsafe(
                _stream_client.logout(),
                _stream_loop
            )
        except:
            pass
    logger.info("Schwab streaming stopped")


async def _subscribe_symbols(symbols: List[str]):
    """Subscribe to symbols (async)"""
    global _stream_client, _subscribed_symbols
    if _stream_client and _is_running:
        try:
            await _stream_client.level_one_equity_subs(symbols)
            _subscribed_symbols.update(s.upper() for s in symbols)
            logger.info(f"Subscribed to: {symbols}")
        except Exception as e:
            logger.error(f"Subscribe error: {e}")


def subscribe(symbols: List[str]):
    """Subscribe to symbols (sync wrapper)"""
    global _subscribed_symbols, _stream_loop

    symbols = [s.upper() for s in symbols]
    _subscribed_symbols.update(symbols)

    if _is_running and _stream_loop:
        try:
            future = asyncio.run_coroutine_threadsafe(
                _subscribe_symbols(symbols),
                _stream_loop
            )
            future.result(timeout=5)
        except Exception as e:
            logger.error(f"Subscribe error: {e}")


def unsubscribe(symbols: List[str]):
    """Unsubscribe from symbols"""
    global _subscribed_symbols
    for s in symbols:
        _subscribed_symbols.discard(s.upper())


async def _subscribe_book(symbols: List[str]):
    """Subscribe to L2 book (async)"""
    global _stream_client, _book_symbols
    if _stream_client and _is_running:
        try:
            await _stream_client.nasdaq_book_subs(symbols)
            await _stream_client.nyse_book_subs(symbols)
            _book_symbols.update(s.upper() for s in symbols)
            logger.info(f"L2 book subscribed: {symbols}")
        except Exception as e:
            logger.error(f"L2 book subscribe error: {e}")


def subscribe_book(symbols: List[str]):
    """Subscribe to L2 order book for symbols"""
    global _book_symbols, _stream_loop

    symbols = [s.upper() for s in symbols]
    _book_symbols.update(symbols)

    if _is_running and _stream_loop:
        try:
            future = asyncio.run_coroutine_threadsafe(
                _subscribe_book(symbols),
                _stream_loop
            )
            future.result(timeout=5)
            return True
        except Exception as e:
            logger.error(f"L2 book subscribe error: {e}")
            return False
    return False


def get_status() -> Dict:
    """Get streaming status"""
    return {
        "running": _is_running,
        "subscribed_symbols": list(_subscribed_symbols),
        "book_symbols": list(_book_symbols),
        "cached_quotes": len(_quote_cache),
        "cached_books": len(_book_cache),
        "callbacks_registered": len(_quote_callbacks)
    }


# Singleton access
_schwab_streamer = None


def get_schwab_streamer():
    """Get the Schwab streamer singleton"""
    global _schwab_streamer
    if _schwab_streamer is None:
        _schwab_streamer = {
            "start": start_streaming,
            "stop": stop_streaming,
            "subscribe": subscribe,
            "unsubscribe": unsubscribe,
            "get_quote": get_cached_quote,
            "get_all_quotes": get_quote_cache,
            "get_status": get_status,
            "register_callback": register_callback
        }
    return _schwab_streamer
