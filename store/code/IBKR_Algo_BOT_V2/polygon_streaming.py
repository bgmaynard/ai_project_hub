"""
Polygon.io Real-Time WebSocket Streaming
=========================================
Provides real-time trades (time & sales) and quotes (bid/ask) via WebSocket.
Requires Polygon.io paid subscription.

Features:
- Real-time trade stream (tape)
- Real-time NBBO quotes
- LULD (Limit Up/Limit Down) bands
- Minute bar aggregation from trades
- Auto-reconnect on disconnect
- Multi-symbol support

Usage:
    stream = PolygonStream()
    stream.subscribe_trades("AAPL")
    stream.subscribe_quotes("AAPL")
    stream.subscribe_luld("AAPL")
    stream.start()
"""

import os
import json
import logging
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable, Set
from collections import deque
import websockets
from websockets.exceptions import ConnectionClosed

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logger = logging.getLogger(__name__)

POLYGON_WS_URL = "wss://socket.polygon.io/stocks"

def get_polygon_api_key():
    """Get Polygon API key at runtime (after dotenv loads)"""
    return os.getenv("POLYGON_API_KEY", "")


class PolygonStream:
    """
    Real-time WebSocket streaming from Polygon.io

    Streams:
    - T.{symbol} - Trades (time & sales)
    - Q.{symbol} - Quotes (NBBO bid/ask)
    - AM.{symbol} - Minute aggregates
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or get_polygon_api_key()
        self.ws = None
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Subscriptions
        self.trade_symbols: Set[str] = set()
        self.quote_symbols: Set[str] = set()
        self.luld_symbols: Set[str] = set()  # LULD subscriptions

        # Data buffers (last N items per symbol)
        self.trades: Dict[str, deque] = {}  # symbol -> deque of trades
        self.quotes: Dict[str, Dict] = {}   # symbol -> latest quote
        self.tape: deque = deque(maxlen=500)  # Combined tape for all symbols
        self.luld_bands: Dict[str, Dict] = {}  # symbol -> LULD bands

        # Callbacks
        self._trade_callbacks: List[Callable] = []
        self._quote_callbacks: List[Callable] = []
        self._luld_callbacks: List[Callable] = []

        # Stats
        self.stats = {
            'connected': False,
            'trades_received': 0,
            'quotes_received': 0,
            'luld_received': 0,
            'last_message': None,
            'reconnects': 0
        }

        logger.info("PolygonStream initialized")

    def on_trade(self, callback: Callable):
        """Register callback for trade events"""
        self._trade_callbacks.append(callback)

    def on_quote(self, callback: Callable):
        """Register callback for quote events"""
        self._quote_callbacks.append(callback)

    def subscribe_trades(self, symbol: str):
        """Subscribe to trade stream for symbol"""
        symbol = symbol.upper()
        is_new = symbol not in self.trade_symbols
        self.trade_symbols.add(symbol)
        if symbol not in self.trades:
            self.trades[symbol] = deque(maxlen=200)
        logger.info(f"Subscribed to trades: {symbol}")
        print(f"[PolygonStream] subscribe_trades({symbol}), is_new={is_new}, ws={self.ws is not None}, connected={self.stats.get('connected')}")
        # Send subscription to WebSocket if already connected
        if is_new and self.ws and self.stats.get('connected'):
            print(f"[PolygonStream] Sending live subscription for T.{symbol}")
            self._send_live_subscription(f"T.{symbol}")
        else:
            print(f"[PolygonStream] NOT sending live sub - is_new={is_new}, ws={self.ws is not None}, connected={self.stats.get('connected')}")

    def subscribe_quotes(self, symbol: str):
        """Subscribe to quote stream for symbol"""
        symbol = symbol.upper()
        is_new = symbol not in self.quote_symbols
        self.quote_symbols.add(symbol)
        logger.info(f"Subscribed to quotes: {symbol}")
        # Send subscription to WebSocket if already connected
        if is_new and self.ws and self.stats.get('connected'):
            self._send_live_subscription(f"Q.{symbol}")

    def _send_live_subscription(self, channel: str):
        """Send a subscription message to the live WebSocket"""
        print(f"[PolygonStream] _send_live_subscription({channel}), loop={self._loop is not None}, ws={self.ws is not None}")
        if self._loop and self.ws:
            try:
                msg = {"action": "subscribe", "params": channel}
                print(f"[PolygonStream] Sending msg: {msg}")
                future = asyncio.run_coroutine_threadsafe(
                    self.ws.send(json.dumps(msg)),
                    self._loop
                )
                # Wait briefly for the result to ensure it's sent
                try:
                    future.result(timeout=2)
                    print(f"[PolygonStream] Sent subscription: {channel}")
                except Exception as e:
                    print(f"[PolygonStream] Subscription send error: {e}")
                logger.info(f"Sent live subscription: {channel}")
            except Exception as e:
                print(f"[PolygonStream] Failed to send subscription: {e}")
                logger.error(f"Failed to send live subscription: {e}")
        else:
            print(f"[PolygonStream] Cannot send - no loop or ws")

    def subscribe_luld(self, symbol: str):
        """Subscribe to LULD (Limit Up/Limit Down) events for symbol"""
        symbol = symbol.upper()
        self.luld_symbols.add(symbol)
        logger.info(f"Subscribed to LULD: {symbol}")

    def on_luld(self, callback: Callable):
        """Register callback for LULD events"""
        self._luld_callbacks.append(callback)

    def unsubscribe(self, symbol: str):
        """Unsubscribe from all streams for symbol"""
        symbol = symbol.upper()
        self.trade_symbols.discard(symbol)
        self.quote_symbols.discard(symbol)
        self.luld_symbols.discard(symbol)
        logger.info(f"Unsubscribed: {symbol}")

    async def _connect(self):
        """Connect to Polygon WebSocket"""
        print(f"[PolygonStream] _connect() called, api_key len: {len(self.api_key)}")
        if not self.api_key:
            logger.error("Polygon API key not configured")
            print("[PolygonStream] No API key in _connect!")
            return

        try:
            print(f"[PolygonStream] Connecting to {POLYGON_WS_URL}...")
            self.ws = await websockets.connect(
                POLYGON_WS_URL,
                ping_interval=30,
                ping_timeout=10
            )
            logger.info("Connected to Polygon WebSocket")
            print("[PolygonStream] WebSocket connected!")

            # Authenticate
            auth_msg = {"action": "auth", "params": self.api_key}
            await self.ws.send(json.dumps(auth_msg))

            # Wait for connection confirmation and auth response
            authenticated = False
            for _ in range(3):  # Check up to 3 messages for auth
                try:
                    response = await asyncio.wait_for(self.ws.recv(), timeout=5)
                    data = json.loads(response)

                    if isinstance(data, list):
                        for msg in data:
                            status = msg.get("status", "")
                            if status == "connected":
                                logger.info("Polygon WebSocket connected")
                            elif status == "auth_success":
                                logger.info("Polygon authentication successful")
                                self.stats['connected'] = True
                                authenticated = True
                            elif status == "auth_failed":
                                logger.error(f"Polygon auth failed: {msg.get('message')}")
                                return

                    if authenticated:
                        break
                except asyncio.TimeoutError:
                    break

            if not authenticated:
                logger.warning("Polygon auth not confirmed, proceeding anyway")
                self.stats['connected'] = True  # Assume connected

            # Subscribe to symbols
            await self._send_subscriptions()

        except Exception as e:
            logger.error(f"Polygon connection error: {e}")
            self.stats['connected'] = False

    async def _send_subscriptions(self):
        """Send subscription messages"""
        if not self.ws:
            return

        subs = []
        for symbol in self.trade_symbols:
            subs.append(f"T.{symbol}")
        for symbol in self.quote_symbols:
            subs.append(f"Q.{symbol}")
        for symbol in self.luld_symbols:
            subs.append(f"LULD.{symbol}")

        if subs:
            msg = {"action": "subscribe", "params": ",".join(subs)}
            await self.ws.send(json.dumps(msg))
            logger.info(f"Subscribed to: {subs}")

    def _process_trade(self, data: dict):
        """Process incoming trade message"""
        symbol = data.get("sym", data.get("S", ""))
        if not symbol:
            return

        # Parse trade data
        trade = {
            "symbol": symbol,
            "price": data.get("p", 0),
            "size": data.get("s", 0),
            "exchange": data.get("x", ""),
            "timestamp": data.get("t", 0),
            "conditions": data.get("c", []),
            "time": datetime.now().strftime("%H:%M:%S.%f")[:-3]
        }

        # Add to symbol-specific buffer
        if symbol in self.trades:
            self.trades[symbol].append(trade)

        # Add to combined tape
        self.tape.append(trade)

        self.stats['trades_received'] += 1
        self.stats['last_message'] = datetime.now().isoformat()

        # Fire callbacks
        for cb in self._trade_callbacks:
            try:
                cb(trade)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")

    def _process_quote(self, data: dict):
        """Process incoming quote message"""
        symbol = data.get("sym", data.get("S", ""))
        if not symbol:
            return

        # Parse quote data
        quote = {
            "symbol": symbol,
            "bid": data.get("bp", data.get("p", 0)),
            "bid_size": data.get("bs", data.get("s", 0)),
            "ask": data.get("ap", data.get("P", 0)),
            "ask_size": data.get("as", data.get("S", 0)),
            "timestamp": data.get("t", 0),
            "time": datetime.now().strftime("%H:%M:%S.%f")[:-3]
        }

        # Calculate spread
        if quote['bid'] > 0 and quote['ask'] > 0:
            quote['spread'] = quote['ask'] - quote['bid']
            quote['spread_pct'] = (quote['spread'] / quote['bid']) * 100

        # Store latest quote
        self.quotes[symbol] = quote

        self.stats['quotes_received'] += 1
        self.stats['last_message'] = datetime.now().isoformat()

        # Fire callbacks
        for cb in self._quote_callbacks:
            try:
                cb(quote)
            except Exception as e:
                logger.error(f"Quote callback error: {e}")

    def _process_luld(self, data: dict):
        """Process incoming LULD (Limit Up/Limit Down) message"""
        symbol = data.get("T", data.get("sym", ""))
        if not symbol:
            return

        # Parse LULD data
        luld = {
            "symbol": symbol,
            "upper": data.get("h", 0),  # Limit up price
            "lower": data.get("l", 0),  # Limit down price
            "indicators": data.get("i", []),
            "tape": data.get("z", ""),
            "timestamp": data.get("t", 0),
            "time": datetime.now().strftime("%H:%M:%S")
        }

        # Store LULD bands
        self.luld_bands[symbol] = luld

        self.stats['luld_received'] += 1
        self.stats['last_message'] = datetime.now().isoformat()

        # Fire callbacks
        for cb in self._luld_callbacks:
            try:
                cb(luld)
            except Exception as e:
                logger.error(f"LULD callback error: {e}")

    async def _message_loop(self):
        """Main message processing loop"""
        print("[PolygonStream] Message loop started")
        msg_count = 0
        while self.running:
            try:
                if not self.ws or self.ws.close_code is not None:
                    print("[PolygonStream] WS not connected, calling _connect()")
                    await self._connect()
                    if not self.ws:
                        await asyncio.sleep(5)
                        continue

                print(f"[PolygonStream] Waiting for recv()... (count so far: {msg_count})")
                message = await asyncio.wait_for(self.ws.recv(), timeout=30)
                msg_count += 1
                data = json.loads(message)
                print(f"[PolygonStream] Got message, total: {msg_count}")

                # Debug first few messages
                if msg_count <= 5:
                    print(f"[PolygonStream] Received msg #{msg_count}: {str(data)[:200]}")

                # Handle array of messages
                if isinstance(data, list):
                    for msg in data:
                        ev = msg.get("ev", "")
                        if ev == "T":  # Trade
                            self._process_trade(msg)
                        elif ev == "Q":  # Quote
                            self._process_quote(msg)
                        elif ev == "LULD":  # Limit Up/Limit Down
                            self._process_luld(msg)
                        elif ev == "status":
                            print(f"[PolygonStream] Status: {msg.get('message')}")
                            logger.info(f"Polygon status: {msg.get('message')}")

            except asyncio.TimeoutError:
                print(f"[PolygonStream] Recv timeout - no data for 30s")
                # Keep connection alive, just no data

            except ConnectionClosed:
                print("[PolygonStream] WebSocket disconnected!")
                logger.warning("Polygon WebSocket disconnected, reconnecting...")
                self.stats['connected'] = False
                self.stats['reconnects'] += 1
                self.ws = None
                await asyncio.sleep(2)

            except Exception as e:
                print(f"[PolygonStream] Error in message loop: {e}")
                logger.error(f"Polygon stream error: {e}")
                await asyncio.sleep(1)

    def _run_loop(self):
        """Run the async event loop in a thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._message_loop())

    def start(self):
        """Start the streaming connection"""
        if self.running:
            logger.warning("PolygonStream already running")
            print("[PolygonStream] Already running")
            return

        if not self.api_key:
            logger.error("Cannot start - Polygon API key not configured")
            print(f"[PolygonStream] No API key! Key length: {len(self.api_key)}")
            return

        print(f"[PolygonStream] Starting with API key: {self.api_key[:5]}...")
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("PolygonStream started")
        print("[PolygonStream] Thread started")

    def stop(self):
        """Stop the streaming connection"""
        self.running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2)
        self.stats['connected'] = False
        logger.info("PolygonStream stopped")

    def get_trades(self, symbol: str, limit: int = 50) -> List[dict]:
        """Get recent trades for symbol"""
        symbol = symbol.upper()
        if symbol in self.trades:
            trades = list(self.trades[symbol])
            return trades[-limit:]
        return []

    def get_tape(self, limit: int = 100) -> List[dict]:
        """Get combined tape (all symbols)"""
        return list(self.tape)[-limit:]

    def get_quote(self, symbol: str) -> Optional[dict]:
        """Get latest quote for symbol"""
        return self.quotes.get(symbol.upper())

    def get_synthetic_depth(self, symbol: str, levels: int = 10) -> dict:
        """
        Build synthetic depth from recent trades.
        Groups trades by price level to show volume distribution.
        Returns bids (below current) and asks (above current).
        """
        symbol = symbol.upper()
        if symbol not in self.trades or len(self.trades[symbol]) < 5:
            return {'bids': [], 'asks': [], 'source': 'polygon_synthetic'}

        trades = list(self.trades[symbol])
        quote = self.quotes.get(symbol, {})
        mid_price = quote.get('bid', 0) if quote else 0
        if not mid_price and trades:
            mid_price = trades[-1].get('price', 0)

        # Aggregate volume by price
        price_volume = {}
        for t in trades:
            price = round(t.get('price', 0), 2)
            size = t.get('size', 0)
            if price > 0:
                price_volume[price] = price_volume.get(price, 0) + size

        # Split into bids (at or below mid) and asks (above mid)
        bids = []
        asks = []
        for price, volume in sorted(price_volume.items(), reverse=True):
            entry = {'price': price, 'size': volume, 'orders': 1}
            if price <= mid_price:
                bids.append(entry)
            else:
                asks.append(entry)

        # Sort: bids descending (highest first), asks ascending (lowest first)
        bids = sorted(bids, key=lambda x: x['price'], reverse=True)[:levels]
        asks = sorted(asks, key=lambda x: x['price'])[:levels]

        return {
            'bids': bids,
            'asks': asks,
            'mid_price': mid_price,
            'source': 'polygon_synthetic',
            'trade_count': len(trades)
        }

    def get_luld_bands(self, symbol: str) -> Optional[dict]:
        """Get current LULD bands for symbol"""
        return self.luld_bands.get(symbol.upper())

    def get_minute_bars(self, symbol: str, minutes: int = 30) -> Optional['pd.DataFrame']:
        """
        Build minute OHLCV bars from recent trade stream.
        Returns DataFrame with Open, High, Low, Close, Volume columns.
        """
        if not HAS_PANDAS:
            logger.warning("Pandas not available for minute bar aggregation")
            return None

        symbol = symbol.upper()
        if symbol not in self.trades or len(self.trades[symbol]) < 20:
            return None

        try:
            # Convert trades to DataFrame
            trades_list = list(self.trades[symbol])
            df = pd.DataFrame(trades_list)

            if 'timestamp' not in df.columns or 'price' not in df.columns:
                return None

            # Convert timestamp (milliseconds) to datetime
            df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('dt')

            # Resample to 1-minute bars
            ohlcv = df['price'].resample('1min').ohlc()
            ohlcv['Volume'] = df['size'].resample('1min').sum()
            ohlcv.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            # Drop NaN rows and return last N minutes
            result = ohlcv.dropna().tail(minutes)
            return result if len(result) > 0 else None

        except Exception as e:
            logger.debug(f"Error building minute bars for {symbol}: {e}")
            return None

    def get_status(self) -> dict:
        """Get stream status"""
        return {
            "connected": self.stats['connected'],
            "api_key_configured": bool(self.api_key),
            "trade_subscriptions": list(self.trade_symbols),
            "quote_subscriptions": list(self.quote_symbols),
            "luld_subscriptions": list(self.luld_symbols),
            "trades_received": self.stats['trades_received'],
            "quotes_received": self.stats['quotes_received'],
            "luld_received": self.stats['luld_received'],
            "last_message": self.stats['last_message'],
            "reconnects": self.stats['reconnects']
        }


# Singleton instance
_polygon_stream: Optional[PolygonStream] = None


def get_polygon_stream() -> PolygonStream:
    """Get or create the Polygon stream instance"""
    global _polygon_stream
    if _polygon_stream is None:
        _polygon_stream = PolygonStream()
    return _polygon_stream


def is_polygon_streaming_available() -> bool:
    """Check if Polygon streaming is available"""
    return bool(get_polygon_api_key())


# ============ HOD Scanner Integration ============

_hod_scanner_wired = False

def wire_hod_scanner():
    """
    Wire HOD Momentum Scanner to receive real-time trades from Polygon.

    This enables automatic HOD detection and A/B grade stock identification
    based on Ross Cameron's 5 criteria.
    """
    global _hod_scanner_wired

    if _hod_scanner_wired:
        logger.debug("HOD scanner already wired to Polygon stream")
        return

    try:
        from ai.hod_momentum_scanner import get_hod_scanner
        import httpx

        stream = get_polygon_stream()
        hod_scanner = get_hod_scanner()

        def on_polygon_trade(trade: dict):
            """Process each trade from Polygon and update HOD scanner"""
            try:
                symbol = trade.get('symbol', '')
                price = trade.get('price', 0)
                size = trade.get('size', 0)

                if not symbol or price <= 0:
                    return

                # Update HOD scanner - returns alert if new HOD made
                alert = hod_scanner.update_price(symbol, price, size)

                if alert:
                    logger.info(f"🚀 HOD Alert from Polygon: {alert.symbol} ${alert.price:.2f} "
                               f"[{alert.grade}] {alert.strategy_name}")

                    # Auto-add A and B grade stocks to watchlists
                    if alert.grade in ["A", "B"]:
                        _auto_add_to_watchlists(alert.symbol)

            except Exception as e:
                # Don't spam logs - HOD updates are very frequent
                pass

        # Register the callback
        stream.on_trade(on_polygon_trade)

        # Register HOD alert callback for additional processing
        def on_hod_alert(alert):
            """Called when HOD scanner generates an alert"""
            logger.info(f"📈 HOD Break: {alert.symbol} ${alert.price:.2f} ({alert.change_pct:+.1f}%) "
                       f"[{alert.grade}] {alert.criteria_met}/5 criteria")

        hod_scanner.on_alert(on_hod_alert)

        _hod_scanner_wired = True
        logger.info("✅ HOD scanner wired to Polygon stream - real-time HOD detection enabled")

    except Exception as e:
        logger.error(f"Failed to wire HOD scanner to Polygon: {e}")


def _auto_add_to_watchlists(symbol: str):
    """Auto-add symbol to dashboard and scalper watchlists"""
    import threading
    import httpx

    def _add():
        try:
            with httpx.Client(timeout=3.0) as client:
                # Add to dashboard watchlist
                try:
                    client.post(
                        "http://localhost:9100/api/worklist/add",
                        json={"symbol": symbol}
                    )
                except:
                    pass

                # Add to scalper watchlist
                try:
                    client.post(
                        f"http://localhost:9100/api/scanner/scalper/watchlist/add/{symbol}"
                    )
                except:
                    pass

                # Add to HOD tracker
                try:
                    client.post(
                        f"http://localhost:9100/api/scanner/hod/add/{symbol}"
                    )
                except:
                    pass

                logger.info(f"📝 Auto-added {symbol} to watchlists (HOD A/B grade)")
        except Exception as e:
            logger.debug(f"Auto-add failed for {symbol}: {e}")

    # Run in background thread to not block streaming
    threading.Thread(target=_add, daemon=True).start()


if __name__ == "__main__":
    # Test the stream
    logging.basicConfig(level=logging.INFO)

    stream = get_polygon_stream()

    # Define callbacks
    def on_trade(trade):
        print(f"TRADE: {trade['symbol']} ${trade['price']:.2f} x{trade['size']}")

    def on_quote(quote):
        print(f"QUOTE: {quote['symbol']} Bid: ${quote['bid']:.2f} Ask: ${quote['ask']:.2f}")

    stream.on_trade(on_trade)
    stream.on_quote(on_quote)

    # Subscribe to test symbols
    stream.subscribe_trades("SPY")
    stream.subscribe_quotes("SPY")

    # Start streaming
    stream.start()

    # Run for 30 seconds
    import time
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        pass

    stream.stop()
    print("\nStats:", stream.get_status())
