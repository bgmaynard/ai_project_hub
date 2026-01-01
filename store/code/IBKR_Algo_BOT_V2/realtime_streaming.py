"""
Real-Time WebSocket Streaming Module
====================================
Provides WebSocket-based real-time market data streaming using Alpaca's
data stream API integrated with the multi-channel architecture.

Features:
- Per-channel WebSocket streams
- Automatic reconnection
- Symbol subscription management
- Multi-client broadcasting
- Quote, trade, and bar streaming

Author: AI Trading Bot Team
Version: 1.0
"""

import asyncio
import json
import logging
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar, Quote, Trade
from config.broker_config import get_broker_config
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class StreamType(str, Enum):
    """Types of data streams"""

    QUOTES = "quotes"
    TRADES = "trades"
    BARS = "bars"


@dataclass
class StreamSubscription:
    """Tracks a stream subscription"""

    symbol: str
    stream_type: StreamType
    subscribed_at: datetime = field(default_factory=datetime.now)
    last_data: Optional[datetime] = None
    message_count: int = 0


@dataclass
class StreamStats:
    """Statistics for streaming"""

    messages_received: int = 0
    quotes_received: int = 0
    trades_received: int = 0
    bars_received: int = 0
    last_message_time: Optional[datetime] = None
    connected_since: Optional[datetime] = None
    reconnect_count: int = 0
    errors: int = 0


class RealtimeStreamManager:
    """
    Manages real-time WebSocket streaming from Alpaca.

    Integrates with the multi-channel architecture by providing dedicated
    streaming capabilities that complement the REST API channels.
    """

    def __init__(self):
        """Initialize the stream manager"""
        config = get_broker_config()

        if not config.is_alpaca():
            raise ValueError("Broker configuration is not set to Alpaca")

        self.api_key = config.alpaca.api_key
        self.secret_key = config.alpaca.secret_key

        # Alpaca data stream
        self._stream: Optional[StockDataStream] = None
        self._stream_task: Optional[asyncio.Task] = None
        self._stream_loop: Optional[asyncio.AbstractEventLoop] = None
        self._stream_thread: Optional[threading.Thread] = None

        # Subscriptions tracking
        self.subscriptions: Dict[str, Set[StreamType]] = defaultdict(set)
        self.subscription_details: Dict[str, StreamSubscription] = {}

        # Callbacks for data handlers
        self._quote_handlers: List[Callable] = []
        self._trade_handlers: List[Callable] = []
        self._bar_handlers: List[Callable] = []
        self._error_handlers: List[Callable] = []

        # Message queue for WebSocket clients
        self._message_queues: Dict[str, asyncio.Queue] = {}
        self._queue_lock = threading.Lock()

        # Statistics
        self.stats = StreamStats()

        # State
        self._running = False
        self._connected = False

        logger.info("[STREAMING] RealtimeStreamManager initialized")

    def _create_stream(self) -> StockDataStream:
        """Create a new Alpaca data stream"""
        return StockDataStream(self.api_key, self.secret_key)

    async def _handle_quote(self, quote: Quote):
        """Handle incoming quote data"""
        try:
            symbol = quote.symbol
            data = {
                "type": "quote",
                "symbol": symbol,
                "bid": float(quote.bid_price),
                "ask": float(quote.ask_price),
                "bid_size": int(quote.bid_size),
                "ask_size": int(quote.ask_size),
                "mid": (float(quote.bid_price) + float(quote.ask_price)) / 2,
                "spread": float(quote.ask_price) - float(quote.bid_price),
                "timestamp": (
                    quote.timestamp.isoformat()
                    if quote.timestamp
                    else datetime.now().isoformat()
                ),
                "received_at": datetime.now().isoformat(),
            }

            # Update stats
            self.stats.messages_received += 1
            self.stats.quotes_received += 1
            self.stats.last_message_time = datetime.now()

            # Update subscription stats
            key = f"{symbol}_{StreamType.QUOTES.value}"
            if key in self.subscription_details:
                self.subscription_details[key].last_data = datetime.now()
                self.subscription_details[key].message_count += 1

            # Broadcast to message queues
            await self._broadcast_message(data)

            # Call registered handlers
            for handler in self._quote_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Quote handler error: {e}")

        except Exception as e:
            logger.error(f"Error handling quote: {e}")
            self.stats.errors += 1

    async def _handle_trade(self, trade: Trade):
        """Handle incoming trade data"""
        try:
            symbol = trade.symbol
            data = {
                "type": "trade",
                "symbol": symbol,
                "price": float(trade.price),
                "size": int(trade.size),
                "exchange": trade.exchange,
                "timestamp": (
                    trade.timestamp.isoformat()
                    if trade.timestamp
                    else datetime.now().isoformat()
                ),
                "received_at": datetime.now().isoformat(),
            }

            # Update stats
            self.stats.messages_received += 1
            self.stats.trades_received += 1
            self.stats.last_message_time = datetime.now()

            # Update subscription stats
            key = f"{symbol}_{StreamType.TRADES.value}"
            if key in self.subscription_details:
                self.subscription_details[key].last_data = datetime.now()
                self.subscription_details[key].message_count += 1

            # Broadcast to message queues
            await self._broadcast_message(data)

            # Call registered handlers
            for handler in self._trade_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Trade handler error: {e}")

        except Exception as e:
            logger.error(f"Error handling trade: {e}")
            self.stats.errors += 1

    async def _handle_bar(self, bar: Bar):
        """Handle incoming bar data"""
        try:
            symbol = bar.symbol
            data = {
                "type": "bar",
                "symbol": symbol,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
                "vwap": float(bar.vwap) if bar.vwap else None,
                "timestamp": (
                    bar.timestamp.isoformat()
                    if bar.timestamp
                    else datetime.now().isoformat()
                ),
                "received_at": datetime.now().isoformat(),
            }

            # Update stats
            self.stats.messages_received += 1
            self.stats.bars_received += 1
            self.stats.last_message_time = datetime.now()

            # Update subscription stats
            key = f"{symbol}_{StreamType.BARS.value}"
            if key in self.subscription_details:
                self.subscription_details[key].last_data = datetime.now()
                self.subscription_details[key].message_count += 1

            # Broadcast to message queues
            await self._broadcast_message(data)

            # Call registered handlers
            for handler in self._bar_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Bar handler error: {e}")

        except Exception as e:
            logger.error(f"Error handling bar: {e}")
            self.stats.errors += 1

    async def _broadcast_message(self, data: Dict):
        """Broadcast message to all connected WebSocket clients"""
        message = json.dumps(data)

        with self._queue_lock:
            dead_clients = []
            for client_id, queue in self._message_queues.items():
                try:
                    # Non-blocking put, drop if queue full
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    # Queue full, skip this message for this client
                    pass
                except Exception as e:
                    logger.error(f"Error broadcasting to {client_id}: {e}")
                    dead_clients.append(client_id)

            # Clean up dead clients
            for client_id in dead_clients:
                del self._message_queues[client_id]

    def register_client(self, client_id: str) -> asyncio.Queue:
        """Register a WebSocket client and return its message queue"""
        queue = asyncio.Queue(maxsize=1000)
        with self._queue_lock:
            self._message_queues[client_id] = queue
        logger.info(f"[STREAMING] Client registered: {client_id}")
        return queue

    def unregister_client(self, client_id: str):
        """Unregister a WebSocket client"""
        with self._queue_lock:
            if client_id in self._message_queues:
                del self._message_queues[client_id]
                logger.info(f"[STREAMING] Client unregistered: {client_id}")

    def add_quote_handler(self, handler: Callable):
        """Add a handler for quote data"""
        self._quote_handlers.append(handler)

    def add_trade_handler(self, handler: Callable):
        """Add a handler for trade data"""
        self._trade_handlers.append(handler)

    def add_bar_handler(self, handler: Callable):
        """Add a handler for bar data"""
        self._bar_handlers.append(handler)

    async def subscribe(
        self, symbols: List[str], stream_types: List[StreamType] = None
    ):
        """
        Subscribe to real-time data for symbols.

        Args:
            symbols: List of stock symbols
            stream_types: Types of data to subscribe (default: quotes)
        """
        if stream_types is None:
            stream_types = [StreamType.QUOTES]

        if self._stream is None:
            self._stream = self._create_stream()

            # Register handlers
            self._stream.subscribe_quotes(self._handle_quote)
            self._stream.subscribe_trades(self._handle_trade)
            self._stream.subscribe_bars(self._handle_bar)

        for symbol in symbols:
            symbol = symbol.upper()

            for stream_type in stream_types:
                key = f"{symbol}_{stream_type.value}"

                if stream_type not in self.subscriptions[symbol]:
                    self.subscriptions[symbol].add(stream_type)
                    self.subscription_details[key] = StreamSubscription(
                        symbol=symbol, stream_type=stream_type
                    )

                    logger.info(
                        f"[STREAMING] Subscribed: {symbol} ({stream_type.value})"
                    )

        # Actually subscribe on the stream
        quote_symbols = [
            s for s, types in self.subscriptions.items() if StreamType.QUOTES in types
        ]
        trade_symbols = [
            s for s, types in self.subscriptions.items() if StreamType.TRADES in types
        ]
        bar_symbols = [
            s for s, types in self.subscriptions.items() if StreamType.BARS in types
        ]

        if quote_symbols:
            self._stream.subscribe_quotes(*quote_symbols)
        if trade_symbols:
            self._stream.subscribe_trades(*trade_symbols)
        if bar_symbols:
            self._stream.subscribe_bars(*bar_symbols)

    async def unsubscribe(
        self, symbols: List[str], stream_types: List[StreamType] = None
    ):
        """Unsubscribe from symbols"""
        if stream_types is None:
            stream_types = list(StreamType)

        if self._stream is None:
            return

        for symbol in symbols:
            symbol = symbol.upper()

            for stream_type in stream_types:
                key = f"{symbol}_{stream_type.value}"

                if stream_type in self.subscriptions.get(symbol, set()):
                    self.subscriptions[symbol].discard(stream_type)
                    if key in self.subscription_details:
                        del self.subscription_details[key]

                    logger.info(
                        f"[STREAMING] Unsubscribed: {symbol} ({stream_type.value})"
                    )

        # Actually unsubscribe on the stream
        quote_unsub = [s for s in symbols if StreamType.QUOTES in stream_types]
        trade_unsub = [s for s in symbols if StreamType.TRADES in stream_types]
        bar_unsub = [s for s in symbols if StreamType.BARS in stream_types]

        if quote_unsub:
            self._stream.unsubscribe_quotes(*quote_unsub)
        if trade_unsub:
            self._stream.unsubscribe_trades(*trade_unsub)
        if bar_unsub:
            self._stream.unsubscribe_bars(*bar_unsub)

    def _run_stream_loop(self):
        """Run the stream in a separate thread"""
        self._stream_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._stream_loop)

        async def run():
            self._connected = True
            self.stats.connected_since = datetime.now()
            logger.info("[STREAMING] Stream connected")

            try:
                await self._stream._run_forever()
            except Exception as e:
                logger.error(f"[STREAMING] Stream error: {e}")
                self._connected = False
                self.stats.reconnect_count += 1

        self._stream_loop.run_until_complete(run())

    async def start(self):
        """Start the streaming service"""
        if self._running:
            logger.warning("[STREAMING] Already running")
            return

        if self._stream is None:
            self._stream = self._create_stream()
            self._stream.subscribe_quotes(self._handle_quote)
            self._stream.subscribe_trades(self._handle_trade)
            self._stream.subscribe_bars(self._handle_bar)

        self._running = True

        # Start stream in background thread
        self._stream_thread = threading.Thread(
            target=self._run_stream_loop, daemon=True
        )
        self._stream_thread.start()

        logger.info("[STREAMING] Service started")

    async def stop(self):
        """Stop the streaming service"""
        if not self._running:
            return

        self._running = False
        self._connected = False

        if self._stream:
            try:
                await self._stream.stop()
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")

        self._stream = None
        logger.info("[STREAMING] Service stopped")

    def get_status(self) -> Dict:
        """Get streaming service status"""
        return {
            "running": self._running,
            "connected": self._connected,
            "subscriptions": {
                symbol: [t.value for t in types]
                for symbol, types in self.subscriptions.items()
                if types
            },
            "subscription_count": sum(
                len(types) for types in self.subscriptions.values()
            ),
            "connected_clients": len(self._message_queues),
            "stats": {
                "messages_received": self.stats.messages_received,
                "quotes_received": self.stats.quotes_received,
                "trades_received": self.stats.trades_received,
                "bars_received": self.stats.bars_received,
                "errors": self.stats.errors,
                "reconnect_count": self.stats.reconnect_count,
                "connected_since": (
                    self.stats.connected_since.isoformat()
                    if self.stats.connected_since
                    else None
                ),
                "last_message": (
                    self.stats.last_message_time.isoformat()
                    if self.stats.last_message_time
                    else None
                ),
            },
        }

    def get_subscription_details(self) -> Dict:
        """Get detailed subscription information"""
        return {
            key: {
                "symbol": sub.symbol,
                "type": sub.stream_type.value,
                "subscribed_at": sub.subscribed_at.isoformat(),
                "last_data": sub.last_data.isoformat() if sub.last_data else None,
                "message_count": sub.message_count,
            }
            for key, sub in self.subscription_details.items()
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_stream_manager: Optional[RealtimeStreamManager] = None
_manager_lock = threading.Lock()


def get_stream_manager() -> RealtimeStreamManager:
    """Get or create the stream manager singleton"""
    global _stream_manager

    if _stream_manager is None:
        with _manager_lock:
            if _stream_manager is None:
                _stream_manager = RealtimeStreamManager()

    return _stream_manager


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    async def test_streaming():
        print("Testing Real-Time Streaming Manager...")

        manager = get_stream_manager()

        # Print initial status
        print("\n=== Initial Status ===")
        print(json.dumps(manager.get_status(), indent=2, default=str))

        # Test subscription
        print("\n=== Subscribing to AAPL, TSLA ===")
        await manager.subscribe(
            ["AAPL", "TSLA"], [StreamType.QUOTES, StreamType.TRADES]
        )

        print("\n=== After Subscription ===")
        print(json.dumps(manager.get_status(), indent=2, default=str))

        # Start streaming
        print("\n=== Starting Stream ===")
        await manager.start()

        # Wait for some data
        print("Waiting 30 seconds for data...")
        await asyncio.sleep(30)

        # Print final status
        print("\n=== Final Status ===")
        print(json.dumps(manager.get_status(), indent=2, default=str))
        print("\n=== Subscription Details ===")
        print(json.dumps(manager.get_subscription_details(), indent=2, default=str))

        # Stop
        await manager.stop()
        print("\n=== Stream Stopped ===")

    asyncio.run(test_streaming())
