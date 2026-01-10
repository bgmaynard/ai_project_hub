"""
Price / Volume Shock Detector
=============================
PRIMARY TRIGGER for the Hot Symbol Engine.

Detects abnormal market behavior WITHOUT waiting for news:
- Price spikes (>= 3% in 2-5 minutes)
- Volume shocks (>= 3x rolling average)
- Range expansion (candle range >= 2x average)
- Momentum chains (3+ consecutive green candles)

This module uses FREE data sources (Schwab, Yahoo, Polygon free tier)
to detect market reactions as they happen.

Author: AI Trading Bot Team
Version: 1.0
Created: 2026-01-10
"""

import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PriceBar:
    """Simple price bar representation"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def pct_change(self) -> float:
        if self.open == 0:
            return 0.0
        return ((self.close - self.open) / self.open) * 100

    @property
    def is_green(self) -> bool:
        return self.close >= self.open


@dataclass
class ShockMetrics:
    """Metrics about a detected shock"""
    symbol: str
    shock_type: str
    pct_change: float = 0.0
    volume_ratio: float = 0.0
    range_ratio: float = 0.0
    green_candle_streak: int = 0
    price: float = 0.0
    volume: int = 0
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "shock_type": self.shock_type,
            "pct_change": round(self.pct_change, 2),
            "volume_ratio": round(self.volume_ratio, 2),
            "range_ratio": round(self.range_ratio, 2),
            "green_candle_streak": self.green_candle_streak,
            "price": self.price,
            "volume": self.volume,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp.isoformat()
        }


class SymbolTracker:
    """Tracks price/volume history for a single symbol"""

    def __init__(self, symbol: str, window_size: int = 20):
        self.symbol = symbol
        self.window_size = window_size
        self.bars: deque = deque(maxlen=window_size)
        self.last_price: float = 0.0
        self.last_volume: int = 0
        self.last_update: datetime = datetime.now()

        # Rolling baselines
        self._volume_ewma: float = 0.0
        self._range_ewma: float = 0.0
        self._ewma_alpha: float = 0.2  # Smoothing factor

    def add_bar(self, bar: PriceBar):
        """Add a price bar and update baselines"""
        self.bars.append(bar)
        self.last_price = bar.close
        self.last_volume = bar.volume
        self.last_update = bar.timestamp

        # Update EWMA baselines
        if self._volume_ewma == 0:
            self._volume_ewma = float(bar.volume)
            self._range_ewma = bar.range
        else:
            self._volume_ewma = (
                self._ewma_alpha * bar.volume +
                (1 - self._ewma_alpha) * self._volume_ewma
            )
            self._range_ewma = (
                self._ewma_alpha * bar.range +
                (1 - self._ewma_alpha) * self._range_ewma
            )

    def get_pct_change(self, lookback_bars: int = 5) -> float:
        """Get % change over last N bars"""
        if len(self.bars) < lookback_bars:
            return 0.0

        start_price = self.bars[-lookback_bars].open
        end_price = self.bars[-1].close

        if start_price == 0:
            return 0.0
        return ((end_price - start_price) / start_price) * 100

    def get_volume_ratio(self) -> float:
        """Get current volume vs EWMA baseline"""
        if self._volume_ewma == 0 or len(self.bars) == 0:
            return 1.0
        return self.bars[-1].volume / self._volume_ewma

    def get_range_ratio(self) -> float:
        """Get current range vs EWMA baseline"""
        if self._range_ewma == 0 or len(self.bars) == 0:
            return 1.0
        return self.bars[-1].range / self._range_ewma

    def get_green_streak(self) -> int:
        """Count consecutive green candles from most recent"""
        streak = 0
        for bar in reversed(list(self.bars)):
            if bar.is_green:
                streak += 1
            else:
                break
        return streak

    def has_enough_data(self) -> bool:
        """Check if we have enough bars for analysis"""
        return len(self.bars) >= 3


class PriceVolumeShockDetector:
    """
    Detects price and volume shocks across multiple symbols.

    This is the PRIMARY TRIGGER for the Hot Symbol Engine.
    When a shock is detected, symbols are immediately injected
    into the HotSymbolQueue.
    """

    # Default thresholds
    DEFAULT_THRESHOLDS = {
        "min_pct_change": 3.0,          # >= 3% move
        "min_volume_ratio": 3.0,        # >= 3x average volume
        "min_range_ratio": 2.0,         # >= 2x average range
        "min_green_streak": 3,          # >= 3 consecutive green
        "lookback_bars": 5,             # Bars to check for % change
        "min_price": 1.0,               # Skip penny stocks
        "max_price": 50.0,              # Focus on tradable range
    }

    def __init__(self, thresholds: Optional[Dict] = None):
        """Initialize the shock detector"""
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self._trackers: Dict[str, SymbolTracker] = {}
        self._lock = threading.Lock()
        self._running = False
        self._scan_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_shock_callbacks: List[callable] = []

        # Stats
        self._total_scans = 0
        self._total_shocks_detected = 0
        self._shocks_by_type: Dict[str, int] = {}

        logger.info(f"PriceVolumeShockDetector initialized with thresholds: {self.thresholds}")

    def add_shock_callback(self, callback: callable):
        """Register a callback for when shocks are detected"""
        self._on_shock_callbacks.append(callback)

    def get_or_create_tracker(self, symbol: str) -> SymbolTracker:
        """Get or create a symbol tracker"""
        symbol = symbol.upper()
        with self._lock:
            if symbol not in self._trackers:
                self._trackers[symbol] = SymbolTracker(symbol)
            return self._trackers[symbol]

    def feed_bar(self, symbol: str, bar: PriceBar) -> Optional[ShockMetrics]:
        """
        Feed a price bar and check for shocks.

        Returns ShockMetrics if a shock is detected, None otherwise.
        """
        symbol = symbol.upper()
        tracker = self.get_or_create_tracker(symbol)
        tracker.add_bar(bar)

        if not tracker.has_enough_data():
            return None

        # Check all shock conditions
        return self._check_for_shocks(tracker, bar)

    def feed_quote(
        self,
        symbol: str,
        price: float,
        volume: int,
        timestamp: Optional[datetime] = None
    ) -> Optional[ShockMetrics]:
        """
        Feed a simple quote (creates a synthetic bar).
        Use this when you don't have OHLC data.
        """
        bar = PriceBar(
            timestamp=timestamp or datetime.now(),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=volume
        )
        return self.feed_bar(symbol, bar)

    def _check_for_shocks(self, tracker: SymbolTracker, bar: PriceBar) -> Optional[ShockMetrics]:
        """Check all shock conditions for a symbol"""
        shocks_detected = []

        # Price filter
        if bar.close < self.thresholds["min_price"] or bar.close > self.thresholds["max_price"]:
            return None

        pct_change = tracker.get_pct_change(self.thresholds["lookback_bars"])
        volume_ratio = tracker.get_volume_ratio()
        range_ratio = tracker.get_range_ratio()
        green_streak = tracker.get_green_streak()

        # Check PRICE_SPIKE
        if abs(pct_change) >= self.thresholds["min_pct_change"]:
            shocks_detected.append("PRICE_SPIKE")

        # Check VOLUME_SHOCK
        if volume_ratio >= self.thresholds["min_volume_ratio"]:
            shocks_detected.append("VOLUME_SHOCK")

        # Check RANGE_EXPANSION
        if range_ratio >= self.thresholds["min_range_ratio"]:
            shocks_detected.append("RANGE_EXPANSION")

        # Check MOMENTUM_CHAIN
        if green_streak >= self.thresholds["min_green_streak"]:
            shocks_detected.append("MOMENTUM_CHAIN")

        if not shocks_detected:
            return None

        # Calculate confidence based on how many conditions triggered
        base_confidence = 0.4 + (len(shocks_detected) * 0.15)

        # Boost confidence based on magnitude
        if pct_change > 5.0:
            base_confidence += 0.1
        if volume_ratio > 5.0:
            base_confidence += 0.1
        if pct_change > 0:  # Upward move
            base_confidence += 0.05

        confidence = min(1.0, base_confidence)

        metrics = ShockMetrics(
            symbol=tracker.symbol,
            shock_type=shocks_detected[0],  # Primary type
            pct_change=pct_change,
            volume_ratio=volume_ratio,
            range_ratio=range_ratio,
            green_candle_streak=green_streak,
            price=bar.close,
            volume=bar.volume,
            confidence=confidence,
            timestamp=bar.timestamp
        )

        # Update stats
        self._total_shocks_detected += 1
        for shock_type in shocks_detected:
            self._shocks_by_type[shock_type] = self._shocks_by_type.get(shock_type, 0) + 1

        # Log the detection
        logger.info(
            f"SHOCK DETECTED: {tracker.symbol} | "
            f"types={shocks_detected} | pct={pct_change:.1f}% | "
            f"vol_ratio={volume_ratio:.1f}x | confidence={confidence:.2f}"
        )

        # Fire callbacks
        for callback in self._on_shock_callbacks:
            try:
                callback(metrics, shocks_detected)
            except Exception as e:
                logger.error(f"Shock callback error: {e}")

        return metrics

    async def scan_symbols(
        self,
        symbols: List[str],
        data_provider: callable
    ) -> List[ShockMetrics]:
        """
        Scan a list of symbols for shocks.

        Args:
            symbols: List of ticker symbols
            data_provider: Async function that returns (price, volume) for a symbol

        Returns:
            List of detected shocks
        """
        self._total_scans += 1
        detected = []

        for symbol in symbols:
            try:
                price, volume = await data_provider(symbol)
                if price and volume:
                    shock = self.feed_quote(symbol, price, volume)
                    if shock:
                        detected.append(shock)
            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")

        return detected

    def get_status(self) -> Dict:
        """Get detector status for API/dashboard"""
        return {
            "running": self._running,
            "tracked_symbols": len(self._trackers),
            "total_scans": self._total_scans,
            "total_shocks_detected": self._total_shocks_detected,
            "shocks_by_type": self._shocks_by_type,
            "thresholds": self.thresholds
        }

    def get_tracked_symbols(self) -> List[str]:
        """Get list of symbols being tracked"""
        with self._lock:
            return list(self._trackers.keys())

    def clear_tracker(self, symbol: str):
        """Clear tracking data for a symbol"""
        symbol = symbol.upper()
        with self._lock:
            if symbol in self._trackers:
                del self._trackers[symbol]

    def clear_all_trackers(self):
        """Clear all tracking data"""
        with self._lock:
            self._trackers.clear()

    def update_thresholds(self, new_thresholds: Dict):
        """Update detection thresholds"""
        self.thresholds.update(new_thresholds)
        logger.info(f"Thresholds updated: {self.thresholds}")


# Singleton instance
_shock_detector: Optional[PriceVolumeShockDetector] = None


def get_shock_detector() -> PriceVolumeShockDetector:
    """Get or create the shock detector singleton"""
    global _shock_detector
    if _shock_detector is None:
        _shock_detector = PriceVolumeShockDetector()
    return _shock_detector


def wire_shock_detector_to_hot_queue():
    """
    Wire the shock detector to automatically inject into HotSymbolQueue.
    Call this during app startup.
    """
    from ai.hot_symbol_queue import get_hot_symbol_queue

    detector = get_shock_detector()
    queue = get_hot_symbol_queue()

    def on_shock(metrics: ShockMetrics, shock_types: List[str]):
        """Callback when shock detected - inject into hot queue"""
        for shock_type in shock_types:
            queue.add(
                symbol=metrics.symbol,
                reason=shock_type,
                confidence=metrics.confidence,
                metrics=metrics.to_dict()
            )

    detector.add_shock_callback(on_shock)
    logger.info("Shock detector wired to HotSymbolQueue")


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    detector = get_shock_detector()

    # Simulate feeding bars
    import random

    print("\n=== Testing Shock Detector ===\n")

    # Normal activity
    for i in range(5):
        bar = PriceBar(
            timestamp=datetime.now(),
            open=10.0 + i * 0.05,
            high=10.1 + i * 0.05,
            low=9.95 + i * 0.05,
            close=10.05 + i * 0.05,
            volume=100000
        )
        result = detector.feed_bar("TEST", bar)
        print(f"Bar {i+1}: close={bar.close:.2f}, shock={result}")

    # Spike!
    print("\n--- Simulating spike ---")
    bar = PriceBar(
        timestamp=datetime.now(),
        open=10.30,
        high=11.50,
        low=10.30,
        close=11.40,  # Big move!
        volume=500000  # Volume spike
    )
    result = detector.feed_bar("TEST", bar)
    if result:
        print(f"SHOCK: {result.to_dict()}")

    print(f"\nDetector status: {detector.get_status()}")
