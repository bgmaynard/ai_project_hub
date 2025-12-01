"""
Multi-Channel Market Data Provider
==================================
Creates multiple parallel data connections to Alpaca for faster, non-blocking
access to market data across different platform components:

1. ORDERS Channel   - Dedicated for order execution and position updates
2. CHARTS Channel   - Dedicated for TradingView/charting data
3. AI Channel       - Dedicated for AI predictions and analysis
4. SCANNER Channel  - Dedicated for market scanning operations

Each channel has its own connection pool, preventing bottlenecks when
multiple components need data simultaneously.

Author: AI Trading Bot Team
Version: 1.0
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
import threading
import time

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestBarRequest,
    StockSnapshotRequest
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from config.broker_config import get_broker_config
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class DataChannel(str, Enum):
    """Data channel types"""
    ORDERS = "orders"      # Order execution, positions, account
    CHARTS = "charts"      # Charting data, historical bars
    AI = "ai"              # AI predictions, analysis
    SCANNER = "scanner"    # Market scanning, screening
    REALTIME = "realtime"  # Real-time quotes and snapshots


@dataclass
class ChannelStats:
    """Statistics for a data channel"""
    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    avg_latency_ms: float = 0.0
    last_request_time: Optional[datetime] = None
    latencies: List[float] = field(default_factory=list)

    def record_request(self, latency_ms: float, success: bool):
        """Record a request result"""
        self.requests_total += 1
        if success:
            self.requests_success += 1
        else:
            self.requests_failed += 1
        self.last_request_time = datetime.now()

        # Keep last 100 latencies for rolling average
        self.latencies.append(latency_ms)
        if len(self.latencies) > 100:
            self.latencies.pop(0)
        self.avg_latency_ms = sum(self.latencies) / len(self.latencies)


class DataChannelClient:
    """
    Individual data channel with its own connection.
    Each channel operates independently for parallel data access.
    """

    def __init__(self, channel: DataChannel, api_key: str, secret_key: str):
        """Initialize a data channel with its own connection"""
        self.channel = channel
        self.api_key = api_key
        self.secret_key = secret_key
        self.stats = ChannelStats()
        self._lock = threading.Lock()

        # Create dedicated clients for this channel
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        self.trading_client = TradingClient(api_key, secret_key, paper=True)

        # Thread pool for parallel requests within this channel
        self.executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix=f"{channel.value}_")

        logger.info(f"[CHANNEL] {channel.value.upper()} channel initialized")

    def _timed_request(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a request and track timing"""
        start = time.time()
        success = True
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            raise e
        finally:
            latency_ms = (time.time() - start) * 1000
            with self._lock:
                self.stats.record_request(latency_ms, success)

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get latest quote"""
        try:
            def _fetch():
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = self.data_client.get_stock_latest_quote(request)
                if symbol in quotes:
                    q = quotes[symbol]
                    return {
                        "symbol": symbol,
                        "bid": float(q.bid_price),
                        "ask": float(q.ask_price),
                        "bid_size": int(q.bid_size),
                        "ask_size": int(q.ask_size),
                        "mid": (float(q.bid_price) + float(q.ask_price)) / 2,
                        "spread": float(q.ask_price) - float(q.bid_price),
                        "timestamp": q.timestamp.isoformat() if q.timestamp else None,
                        "channel": self.channel.value
                    }
                return None
            return self._timed_request(_fetch)
        except Exception as e:
            logger.error(f"[{self.channel.value}] Quote error for {symbol}: {e}")
            return None

    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get quotes for multiple symbols in parallel"""
        results = {}
        try:
            def _fetch():
                request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
                return self.data_client.get_stock_latest_quote(request)

            quotes = self._timed_request(_fetch)
            for symbol, q in quotes.items():
                results[symbol] = {
                    "symbol": symbol,
                    "bid": float(q.bid_price),
                    "ask": float(q.ask_price),
                    "mid": (float(q.bid_price) + float(q.ask_price)) / 2,
                    "timestamp": q.timestamp.isoformat() if q.timestamp else None,
                    "channel": self.channel.value
                }
        except Exception as e:
            logger.error(f"[{self.channel.value}] Multi-quote error: {e}")
        return results

    def get_snapshot(self, symbol: str) -> Optional[Dict]:
        """Get full market snapshot"""
        try:
            def _fetch():
                request = StockSnapshotRequest(symbol_or_symbols=symbol)
                return self.data_client.get_stock_snapshot(request)

            snapshots = self._timed_request(_fetch)
            if symbol in snapshots:
                s = snapshots[symbol]
                result = {
                    "symbol": symbol,
                    "channel": self.channel.value,
                    "timestamp": datetime.now().isoformat()
                }

                if s.latest_quote:
                    result.update({
                        "bid": float(s.latest_quote.bid_price),
                        "ask": float(s.latest_quote.ask_price),
                        "mid": (float(s.latest_quote.bid_price) + float(s.latest_quote.ask_price)) / 2
                    })

                if s.latest_trade:
                    result.update({
                        "last": float(s.latest_trade.price),
                        "last_size": int(s.latest_trade.size)
                    })

                if s.daily_bar:
                    result.update({
                        "open": float(s.daily_bar.open),
                        "high": float(s.daily_bar.high),
                        "low": float(s.daily_bar.low),
                        "close": float(s.daily_bar.close),
                        "volume": int(s.daily_bar.volume),
                        "vwap": float(s.daily_bar.vwap) if s.daily_bar.vwap else None
                    })

                if s.previous_daily_bar:
                    result["prev_close"] = float(s.previous_daily_bar.close)
                    if "close" in result:
                        result["change"] = result["close"] - result["prev_close"]
                        result["change_pct"] = (result["change"] / result["prev_close"]) * 100

                return result
            return None
        except Exception as e:
            logger.error(f"[{self.channel.value}] Snapshot error for {symbol}: {e}")
            return None

    def get_bars(self, symbol: str, timeframe: str = "1Day",
                 days: int = 30, limit: int = None) -> List[Dict]:
        """Get historical bars"""
        try:
            tf_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "30Min": TimeFrame(30, TimeFrameUnit.Minute),
                "1Hour": TimeFrame.Hour,
                "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
                "1Day": TimeFrame.Day,
                "1Week": TimeFrame.Week
            }

            alpaca_tf = tf_map.get(timeframe, TimeFrame.Day)
            start = datetime.now() - timedelta(days=days)

            def _fetch():
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=alpaca_tf,
                    start=start,
                    limit=limit
                )
                return self.data_client.get_stock_bars(request)

            bars = self._timed_request(_fetch)
            if symbol in bars:
                result = []
                for bar in bars[symbol]:
                    result.append({
                        "timestamp": bar.timestamp.isoformat(),
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume),
                        "vwap": float(bar.vwap) if bar.vwap else None
                    })
                return result
            return []
        except Exception as e:
            logger.error(f"[{self.channel.value}] Bars error for {symbol}: {e}")
            return []

    def get_status(self) -> Dict:
        """Get channel status and statistics"""
        return {
            "channel": self.channel.value,
            "active": True,
            "requests_total": self.stats.requests_total,
            "requests_success": self.stats.requests_success,
            "requests_failed": self.stats.requests_failed,
            "success_rate": (self.stats.requests_success / self.stats.requests_total * 100)
                           if self.stats.requests_total > 0 else 0,
            "avg_latency_ms": round(self.stats.avg_latency_ms, 2),
            "last_request": self.stats.last_request_time.isoformat()
                           if self.stats.last_request_time else None
        }


class MultiChannelDataProvider:
    """
    Multi-channel data provider that manages multiple parallel connections
    to Alpaca for different platform components.

    This prevents bottlenecks by giving each component (orders, charts, AI)
    its own dedicated data channel.
    """

    def __init__(self):
        """Initialize multi-channel data provider"""
        config = get_broker_config()

        if not config.is_alpaca():
            raise ValueError("Broker configuration is not set to Alpaca")

        self.api_key = config.alpaca.api_key
        self.secret_key = config.alpaca.secret_key

        # Create dedicated channels
        self.channels: Dict[DataChannel, DataChannelClient] = {}

        for channel in DataChannel:
            try:
                self.channels[channel] = DataChannelClient(
                    channel, self.api_key, self.secret_key
                )
            except Exception as e:
                logger.error(f"Failed to create {channel.value} channel: {e}")

        # Global thread pool for cross-channel parallel operations
        self.global_executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix="global_")

        logger.info(f"[MULTI-CHANNEL] Initialized {len(self.channels)} data channels")

    def get_channel(self, channel: DataChannel) -> Optional[DataChannelClient]:
        """Get a specific channel"""
        return self.channels.get(channel)

    # ========================================================================
    # PARALLEL DATA FETCHING
    # ========================================================================

    def get_quotes_parallel(self, symbols: List[str],
                           channels: List[DataChannel] = None) -> Dict[str, Dict]:
        """
        Fetch quotes for symbols using multiple channels in parallel.
        Each channel handles a portion of the symbols for maximum speed.
        """
        if channels is None:
            channels = [DataChannel.REALTIME, DataChannel.CHARTS]

        available_channels = [c for c in channels if c in self.channels]
        if not available_channels:
            available_channels = list(self.channels.values())[:2]

        # Split symbols across channels
        chunk_size = max(1, len(symbols) // len(available_channels))
        symbol_chunks = [symbols[i:i+chunk_size] for i in range(0, len(symbols), chunk_size)]

        results = {}
        futures = []

        for i, chunk in enumerate(symbol_chunks):
            channel = available_channels[i % len(available_channels)]
            if isinstance(channel, DataChannel):
                channel = self.channels[channel]
            future = self.global_executor.submit(channel.get_quotes, chunk)
            futures.append(future)

        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                results.update(chunk_results)
            except Exception as e:
                logger.error(f"Parallel quote fetch error: {e}")

        return results

    def get_snapshots_parallel(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch snapshots for multiple symbols in parallel across all channels.
        """
        results = {}
        futures = {}

        channels = list(self.channels.values())

        for i, symbol in enumerate(symbols):
            channel = channels[i % len(channels)]
            future = self.global_executor.submit(channel.get_snapshot, symbol)
            futures[future] = symbol

        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result:
                    results[symbol] = result
            except Exception as e:
                logger.error(f"Parallel snapshot error for {symbol}: {e}")

        return results

    def get_multi_symbol_bars(self, symbols: List[str], timeframe: str = "1Day",
                              days: int = 30) -> Dict[str, List[Dict]]:
        """
        Fetch historical bars for multiple symbols in parallel.
        Ideal for AI training data collection.
        """
        results = {}
        futures = {}

        # Use AI and CHARTS channels for bars
        channels = [
            self.channels.get(DataChannel.AI),
            self.channels.get(DataChannel.CHARTS),
            self.channels.get(DataChannel.SCANNER)
        ]
        channels = [c for c in channels if c]

        for i, symbol in enumerate(symbols):
            channel = channels[i % len(channels)]
            future = self.global_executor.submit(
                channel.get_bars, symbol, timeframe, days
            )
            futures[future] = symbol

        for future in as_completed(futures):
            symbol = futures[future]
            try:
                bars = future.result()
                if bars:
                    results[symbol] = bars
            except Exception as e:
                logger.error(f"Parallel bars error for {symbol}: {e}")

        return results

    # ========================================================================
    # CHANNEL-SPECIFIC ACCESS
    # ========================================================================

    def orders_channel(self) -> DataChannelClient:
        """Get the orders channel for trading operations"""
        return self.channels[DataChannel.ORDERS]

    def charts_channel(self) -> DataChannelClient:
        """Get the charts channel for charting data"""
        return self.channels[DataChannel.CHARTS]

    def ai_channel(self) -> DataChannelClient:
        """Get the AI channel for predictions"""
        return self.channels[DataChannel.AI]

    def scanner_channel(self) -> DataChannelClient:
        """Get the scanner channel for market scanning"""
        return self.channels[DataChannel.SCANNER]

    def realtime_channel(self) -> DataChannelClient:
        """Get the real-time channel for live quotes"""
        return self.channels[DataChannel.REALTIME]

    # ========================================================================
    # STATUS AND MONITORING
    # ========================================================================

    def get_all_status(self) -> Dict:
        """Get status of all channels"""
        return {
            "provider": "MultiChannelDataProvider",
            "total_channels": len(self.channels),
            "channels": {
                channel.value: client.get_status()
                for channel, client in self.channels.items()
            },
            "timestamp": datetime.now().isoformat()
        }

    def get_aggregate_stats(self) -> Dict:
        """Get aggregate statistics across all channels"""
        total_requests = sum(c.stats.requests_total for c in self.channels.values())
        total_success = sum(c.stats.requests_success for c in self.channels.values())
        total_failed = sum(c.stats.requests_failed for c in self.channels.values())

        avg_latencies = [c.stats.avg_latency_ms for c in self.channels.values() if c.stats.avg_latency_ms > 0]
        overall_avg_latency = sum(avg_latencies) / len(avg_latencies) if avg_latencies else 0

        return {
            "total_requests": total_requests,
            "total_success": total_success,
            "total_failed": total_failed,
            "success_rate": (total_success / total_requests * 100) if total_requests > 0 else 0,
            "avg_latency_ms": round(overall_avg_latency, 2),
            "channels_active": len(self.channels)
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_provider: Optional[MultiChannelDataProvider] = None
_lock = threading.Lock()


def get_multi_channel_provider() -> MultiChannelDataProvider:
    """Get or create the multi-channel data provider singleton"""
    global _provider

    if _provider is None:
        with _lock:
            if _provider is None:
                _provider = MultiChannelDataProvider()

    return _provider


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_quote_fast(symbol: str, channel: DataChannel = DataChannel.REALTIME) -> Optional[Dict]:
    """Quick quote fetch using specified channel"""
    provider = get_multi_channel_provider()
    return provider.get_channel(channel).get_quote(symbol)


def get_quotes_fast(symbols: List[str]) -> Dict[str, Dict]:
    """Quick multi-symbol quote fetch using parallel channels"""
    provider = get_multi_channel_provider()
    return provider.get_quotes_parallel(symbols)


def get_snapshot_fast(symbol: str, channel: DataChannel = DataChannel.REALTIME) -> Optional[Dict]:
    """Quick snapshot fetch using specified channel"""
    provider = get_multi_channel_provider()
    return provider.get_channel(channel).get_snapshot(symbol)


def get_bars_for_ai(symbol: str, days: int = 365) -> List[Dict]:
    """Get historical bars for AI using AI channel"""
    provider = get_multi_channel_provider()
    return provider.ai_channel().get_bars(symbol, "1Day", days)


def get_bars_for_chart(symbol: str, timeframe: str = "5Min", days: int = 5) -> List[Dict]:
    """Get historical bars for charting using CHARTS channel"""
    provider = get_multi_channel_provider()
    return provider.charts_channel().get_bars(symbol, timeframe, days)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    print("Testing Multi-Channel Data Provider...")

    provider = get_multi_channel_provider()

    # Test status
    print("\n=== Channel Status ===")
    print(json.dumps(provider.get_all_status(), indent=2, default=str))

    # Test parallel quotes
    print("\n=== Testing Parallel Quotes ===")
    symbols = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT"]
    start = time.time()
    quotes = provider.get_quotes_parallel(symbols)
    elapsed = (time.time() - start) * 1000
    print(f"Fetched {len(quotes)} quotes in {elapsed:.0f}ms")
    for sym, q in quotes.items():
        print(f"  {sym}: ${q.get('mid', 'N/A'):.2f} via {q.get('channel', 'unknown')}")

    # Test aggregate stats
    print("\n=== Aggregate Stats ===")
    print(json.dumps(provider.get_aggregate_stats(), indent=2))
