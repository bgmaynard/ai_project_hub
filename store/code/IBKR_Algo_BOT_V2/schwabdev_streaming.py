"""
Schwabdev Real-Time Streaming Integration
=========================================
Fast WebSocket streaming using schwabdev package.
Provides sub-second market data updates with auto-restart capability.

Features:
- Auto-restart if connection drops
- Real-time quote cache for instant access
- Callback system for UI updates
- Integrates with unified market data system
"""
import os
import json
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable, Set
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Global state
_client = None
_stream = None
_quote_cache: Dict[str, Dict] = {}
_quote_callbacks: List[Callable] = []
_subscribed_symbols: Set[str] = set()
_is_running = False
_stats = {
    "messages_received": 0,
    "quotes_received": 0,
    "errors": 0,
    "reconnects": 0,
    "started_at": None,
    "last_quote": None
}

TOKEN_FILE = Path(__file__).parent / "schwab_token.json"
SCHWABDEV_TOKEN_FILE = Path(__file__).parent / "schwabdev_tokens.json"


def _convert_token_format():
    """Convert existing schwab_token.json to schwabdev format"""
    try:
        if not TOKEN_FILE.exists():
            logger.warning("[SCHWABDEV] No source token file found")
            return False

        with open(TOKEN_FILE, 'r') as f:
            old_token = json.load(f)

        # Check if already in schwabdev format
        if "token_dictionary" in old_token:
            return True

        # Convert to schwabdev format
        now = datetime.now(tz=__import__('datetime').timezone.utc).isoformat()
        schwabdev_token = {
            "token_dictionary": {
                "access_token": old_token.get("access_token"),
                "refresh_token": old_token.get("refresh_token"),
                "id_token": old_token.get("id_token")
            },
            "access_token_issued": now,
            "refresh_token_issued": now
        }

        with open(SCHWABDEV_TOKEN_FILE, 'w') as f:
            json.dump(schwabdev_token, f, indent=2)

        logger.info(f"[SCHWABDEV] Token converted to schwabdev format: {SCHWABDEV_TOKEN_FILE}")
        return True

    except Exception as e:
        logger.error(f"[SCHWABDEV] Token conversion failed: {e}")
        return False


def _get_client():
    """Get or create schwabdev client"""
    global _client

    if _client is not None:
        return _client

    try:
        import schwabdev

        app_key = os.getenv('SCHWAB_APP_KEY')
        app_secret = os.getenv('SCHWAB_APP_SECRET')
        callback_url = os.getenv('SCHWAB_CALLBACK_URL', 'https://127.0.0.1:6969')

        # Ensure callback_url is lowercase https (schwabdev requirement)
        if callback_url:
            callback_url = callback_url.replace('HTTPS://', 'https://').replace('HTTP://', 'http://')

        if not app_key or not app_secret:
            logger.error("SCHWAB_APP_KEY or SCHWAB_APP_SECRET not configured")
            return None

        # Convert existing token to schwabdev format if needed
        if not SCHWABDEV_TOKEN_FILE.exists():
            if not _convert_token_format():
                logger.error("[SCHWABDEV] Failed to convert token format")
                return None

        # Use schwabdev-specific token file
        _client = schwabdev.Client(
            app_key=app_key,
            app_secret=app_secret,
            callback_url=callback_url,
            tokens_file=str(SCHWABDEV_TOKEN_FILE),
            timeout=30
        )

        logger.info("[SCHWABDEV] Client initialized successfully")
        return _client

    except Exception as e:
        logger.error(f"[SCHWABDEV] Client initialization failed: {e}")
        return None


def _stream_receiver(message):
    """Process incoming stream messages"""
    global _quote_cache, _stats

    try:
        _stats["messages_received"] += 1

        # Parse the message if it's a string
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except json.JSONDecodeError:
                logger.debug(f"[SCHWABDEV] Non-JSON message: {message[:100]}")
                return

        # Handle data array (main quote data format)
        if "data" in message:
            for item in message.get("data", []):
                service = item.get("service", "")

                if service in ["LEVELONE_EQUITIES", "QUOTE", "LEVELONE_FUTURES"]:
                    content = item.get("content", [])
                    if content:
                        _process_quotes(content)
                elif service == "CHART_EQUITY":
                    _process_charts(item.get("content", []))

        # Handle response messages (login, subscription confirmations)
        elif "response" in message:
            for resp in message.get("response", []):
                service = resp.get("service", "")
                content = resp.get("content", {})
                code = content.get("code", -1)
                msg = content.get("msg", "")
                logger.debug(f"[SCHWABDEV] {service} response: code={code}, msg={msg[:50]}")

        # Handle direct content format
        elif "content" in message:
            _process_quotes(message.get("content", []))

    except Exception as e:
        _stats["errors"] += 1
        logger.error(f"[SCHWABDEV] Stream receiver error: {e}")


def _process_quotes(content: List[Dict]):
    """Process quote updates from stream"""
    global _quote_cache, _stats

    for item in content:
        try:
            symbol = item.get("key", item.get("symbol", "")).upper()
            if not symbol:
                continue

            # Helper to get value from numeric or named key
            def get_val(numeric_key, named_key, default=0):
                # Try numeric key first (schwabdev format)
                val = item.get(numeric_key)
                if val is not None:
                    return val
                # Then try named key
                val = item.get(named_key)
                if val is not None:
                    return val
                return default

            # Extract quote data - schwabdev uses numeric keys
            # Schwab Level 1 field mapping:
            # 1=bid, 2=ask, 3=last, 4=bid_size, 5=ask_size, 8=volume, 12=high
            # 13=low (when numeric), 14=close, 29=open, 18=change, 19=change%
            # NOTE: Some fields return non-numeric (exchange code, description)

            # Safe float conversion
            def safe_float(val, default=0.0):
                try:
                    if val is None:
                        return default
                    return float(val)
                except (ValueError, TypeError):
                    return default

            # Safe int conversion
            def safe_int(val, default=0):
                try:
                    if val is None:
                        return default
                    return int(float(val))
                except (ValueError, TypeError):
                    return default

            bid = safe_float(get_val("1", "bid", 0))
            ask = safe_float(get_val("2", "ask", 0))
            last = safe_float(get_val("3", "last", 0))

            quote = {
                "symbol": symbol,
                "bid": bid,
                "ask": ask,
                "last": last,
                "mid": (bid + ask) / 2 if bid > 0 and ask > 0 else last,
                "bid_size": safe_int(get_val("4", "bid_size", 0)),
                "ask_size": safe_int(get_val("5", "ask_size", 0)),
                "volume": safe_int(get_val("8", "volume", 0)),
                "high": safe_float(get_val("12", "high", 0)),
                "low": safe_float(get_val("14", "low", 0)),  # Field 14 for low
                "close": safe_float(get_val("29", "close", 0)),  # Field 29 often has close
                "open": safe_float(get_val("17", "open", 0)),  # Field 17 for open
                "change": safe_float(get_val("18", "change", 0)),  # Field 18 for net change
                "change_percent": safe_float(get_val("19", "change_percent", 0)),  # Field 19 for change %
                "timestamp": datetime.now().isoformat(),
                "source": "schwabdev_stream"
            }

            # Update cache
            _quote_cache[symbol] = quote
            _stats["quotes_received"] += 1
            _stats["last_quote"] = datetime.now().isoformat()

            # Notify callbacks
            _notify_callbacks(symbol, quote)

            logger.debug(f"[SCHWABDEV] Quote: {symbol} ${quote['last']:.2f}")

        except Exception as e:
            logger.debug(f"[SCHWABDEV] Quote parse error for {item}: {e}")


def _process_charts(content: List[Dict]):
    """Process chart/bar updates from stream"""
    # Chart data processing (for future use)
    pass


def _notify_callbacks(symbol: str, quote: Dict):
    """Notify all registered callbacks"""
    for callback in _quote_callbacks:
        try:
            callback(symbol, quote)
        except Exception as e:
            logger.error(f"[SCHWABDEV] Callback error: {e}")


# ============================================================================
# PUBLIC API
# ============================================================================

def start_streaming(symbols: Optional[List[str]] = None) -> bool:
    """
    Start the schwabdev streaming service.

    Args:
        symbols: Optional list of symbols to subscribe to

    Returns:
        True if started successfully
    """
    global _stream, _subscribed_symbols, _is_running, _stats

    if _is_running:
        logger.info("[SCHWABDEV] Streaming already running")
        return True

    try:
        client = _get_client()
        if not client:
            logger.error("[SCHWABDEV] No client available")
            return False

        # Create stream
        _stream = client.stream

        # Start the stream with our receiver
        _stream.start(receiver=_stream_receiver, daemon=True)

        _is_running = True
        _stats["started_at"] = datetime.now().isoformat()
        logger.info("[SCHWABDEV] Streaming started")

        # Wait for stream to connect before subscribing
        import time
        time.sleep(2)

        # Subscribe to symbols if provided
        if symbols:
            subscribe(symbols)
        elif _subscribed_symbols:
            subscribe(list(_subscribed_symbols))

        return True

    except Exception as e:
        logger.error(f"[SCHWABDEV] Failed to start streaming: {e}")
        _is_running = False
        return False


def stop_streaming():
    """Stop the streaming service"""
    global _stream, _is_running

    if _stream:
        try:
            _stream.stop()
        except Exception as e:
            logger.warning(f"[SCHWABDEV] Stop error: {e}")

    _is_running = False
    logger.info("[SCHWABDEV] Streaming stopped")


def subscribe(symbols: List[str]) -> bool:
    """
    Subscribe to real-time quotes for symbols.

    Args:
        symbols: List of stock symbols

    Returns:
        True if subscription request sent
    """
    global _stream, _subscribed_symbols

    if not symbols:
        return False

    symbols = [s.upper() for s in symbols]
    _subscribed_symbols.update(symbols)

    if _stream and _is_running:
        try:
            # Subscribe to level 1 equity quotes using SUBS command
            # Fields: 0=symbol, 1=bid, 2=ask, 3=last, 4=bid_size, 5=ask_size,
            #         8=volume, 12=high, 13=low, 15=close, 28=open, 29=change, 30=change%
            sub_request = _stream.level_one_equities(
                symbols,
                "0,1,2,3,4,5,8,12,13,15,28,29,30",
                command="SUBS"  # Use SUBS for initial subscription
            )

            # Handle async context (FastAPI runs an event loop)
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # Already in async context - schedule send_async as a task
                asyncio.ensure_future(_stream.send_async(sub_request))
                logger.info(f"[SCHWABDEV] Subscribed (async) to {len(symbols)} symbols: {symbols[:5]}...")
            except RuntimeError:
                # No running loop - safe to use sync send
                _stream.send(sub_request)
                logger.info(f"[SCHWABDEV] Subscribed to {len(symbols)} symbols: {symbols[:5]}...")

            return True
        except Exception as e:
            logger.error(f"[SCHWABDEV] Subscribe error: {e}")
            import traceback
            traceback.print_exc()
            return False

    logger.warning("[SCHWABDEV] Cannot subscribe - stream not running")
    return False


def unsubscribe(symbols: List[str]) -> bool:
    """Unsubscribe from symbols"""
    global _stream, _subscribed_symbols

    symbols = [s.upper() for s in symbols]
    _subscribed_symbols.difference_update(symbols)

    if _stream and _is_running:
        try:
            unsub_request = _stream.level_one_equities(symbols, command="UNSUBS")

            # Handle async context (FastAPI runs an event loop)
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                asyncio.ensure_future(_stream.send_async(unsub_request))
            except RuntimeError:
                _stream.send(unsub_request)

            logger.info(f"[SCHWABDEV] Unsubscribed from {len(symbols)} symbols")
            return True
        except Exception as e:
            logger.error(f"[SCHWABDEV] Unsubscribe error: {e}")

    return False


def get_cached_quote(symbol: str) -> Optional[Dict]:
    """Get cached quote from stream (fastest access)"""
    return _quote_cache.get(symbol.upper())


def get_all_cached_quotes() -> Dict[str, Dict]:
    """Get all cached quotes"""
    return _quote_cache.copy()


def register_callback(callback: Callable):
    """Register a callback for quote updates"""
    if callback not in _quote_callbacks:
        _quote_callbacks.append(callback)


def unregister_callback(callback: Callable):
    """Unregister a callback"""
    if callback in _quote_callbacks:
        _quote_callbacks.remove(callback)


def get_status() -> Dict:
    """Get streaming status"""
    return {
        "running": _is_running,
        "subscribed_symbols": list(_subscribed_symbols),
        "subscription_count": len(_subscribed_symbols),
        "cached_quotes": len(_quote_cache),
        "stats": _stats.copy(),
        "source": "schwabdev"
    }


def is_streaming() -> bool:
    """Check if streaming is active"""
    return _is_running


# ============================================================================
# AUTO-START HELPER
# ============================================================================

def auto_start_with_watchlist():
    """Auto-start streaming with watchlist symbols"""
    try:
        # Default watchlist symbols
        default_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN"]

        # Try to load from worklist
        try:
            worklist_file = Path(__file__).parent / "store" / "worklist.json"
            if worklist_file.exists():
                with open(worklist_file) as f:
                    data = json.load(f)
                    if data.get("symbols"):
                        default_symbols = list(set(default_symbols + data["symbols"]))
        except Exception:
            pass

        return start_streaming(default_symbols)

    except Exception as e:
        logger.error(f"[SCHWABDEV] Auto-start failed: {e}")
        return False


# ============================================================================
# MODULE TEST
# ============================================================================

if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)
    print("Testing schwabdev streaming...")

    # Define a simple callback
    def print_quote(symbol, quote):
        print(f"  {symbol}: ${quote['last']:.2f} (bid: ${quote['bid']:.2f}, ask: ${quote['ask']:.2f})")

    register_callback(print_quote)

    # Start streaming
    if start_streaming(["AAPL", "MSFT", "NVDA"]):
        print("Streaming started! Waiting for quotes...")

        # Wait for some quotes
        for i in range(30):
            time.sleep(1)
            status = get_status()
            print(f"[{i+1}s] Quotes received: {status['stats']['quotes_received']}")

            if status['stats']['quotes_received'] > 0:
                print("\nCached quotes:")
                for sym, q in get_all_cached_quotes().items():
                    print(f"  {sym}: ${q['last']:.2f}")

        stop_streaming()
        print("Streaming stopped.")
    else:
        print("Failed to start streaming")
