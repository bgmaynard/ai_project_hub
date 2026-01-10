"""
SmartZone Pattern Detector

Identifies consolidation zones and breakout triggers.
Key patterns: tight consolidation, pullback to support, flag formations.
"""

from datetime import datetime
from typing import List, Optional, Tuple
from .types import Bar, SmartZoneSignal, ZoneType


class SmartZoneDetector:
    """
    Detect SmartZone consolidation patterns.

    A SmartZone is a tight price range where:
    1. Price consolidates for 3-10 bars
    2. Range compresses (volatility squeeze)
    3. Volume decreases during formation
    4. Breakout above zone high = EXPANSION (bullish)
    5. Breakdown below zone low = BREAKDOWN (bearish)
    """

    def __init__(
        self,
        min_bars: int = 3,
        max_bars: int = 10,
        max_zone_width_pct: float = 2.0,
        min_compression_ratio: float = 0.3,
        breakout_threshold_pct: float = 0.3,
    ):
        self.min_bars = min_bars
        self.max_bars = max_bars
        self.max_zone_width_pct = max_zone_width_pct
        self.min_compression_ratio = min_compression_ratio
        self.breakout_threshold_pct = breakout_threshold_pct

        # State per symbol
        self._zones: dict[str, SmartZoneSignal] = {}
        self._bar_buffers: dict[str, List[Bar]] = {}

    def add_bar(self, symbol: str, bar: Bar) -> Optional[SmartZoneSignal]:
        """
        Add a new bar and check for zone formation/resolution.

        Args:
            symbol: Stock symbol
            bar: New OHLCV bar

        Returns:
            SmartZoneSignal if zone detected or resolved, None otherwise
        """
        if symbol not in self._bar_buffers:
            self._bar_buffers[symbol] = []

        self._bar_buffers[symbol].append(bar)

        # Keep buffer at reasonable size
        if len(self._bar_buffers[symbol]) > self.max_bars * 2:
            self._bar_buffers[symbol] = self._bar_buffers[symbol][-self.max_bars * 2:]

        # Check for active zone resolution
        if symbol in self._zones:
            resolution = self._check_resolution(symbol, bar)
            if resolution:
                return resolution

        # Check for new zone formation
        zone = self._detect_zone(symbol)
        if zone:
            self._zones[symbol] = zone
            return zone

        return None

    def _detect_zone(self, symbol: str) -> Optional[SmartZoneSignal]:
        """Detect new SmartZone formation"""
        bars = self._bar_buffers.get(symbol, [])
        if len(bars) < self.min_bars:
            return None

        # Try different lookback windows
        for lookback in range(self.min_bars, min(len(bars), self.max_bars) + 1):
            recent_bars = bars[-lookback:]
            zone = self._analyze_zone(symbol, recent_bars)
            if zone:
                return zone

        return None

    def _analyze_zone(self, symbol: str, bars: List[Bar]) -> Optional[SmartZoneSignal]:
        """Analyze a set of bars for zone pattern"""
        if not bars:
            return None

        # Calculate zone boundaries
        zone_high = max(b.high for b in bars)
        zone_low = min(b.low for b in bars)
        zone_mid = (zone_high + zone_low) / 2

        # Check zone width
        zone_width_pct = ((zone_high - zone_low) / zone_mid) * 100
        if zone_width_pct > self.max_zone_width_pct:
            return None  # Too wide

        # Calculate compression ratio (current range vs initial range)
        if len(bars) >= 2:
            initial_range = bars[0].range
            final_range = bars[-1].range
            if initial_range > 0:
                compression_ratio = final_range / initial_range
            else:
                compression_ratio = 1.0
        else:
            compression_ratio = 1.0

        # Check for compression (volatility squeeze)
        if compression_ratio > (1 - self.min_compression_ratio):
            # Not compressed enough
            pass  # Still allow detection, just lower confidence

        # Determine zone type
        zone_type = self._classify_zone_type(bars)

        # Calculate breakout level (zone high + threshold)
        breakout_level = zone_high * (1 + self.breakout_threshold_pct / 100)

        # Calculate confidence
        confidence = self._calculate_confidence(
            bars, zone_width_pct, compression_ratio
        )

        return SmartZoneSignal(
            symbol=symbol,
            zone_type=zone_type,
            zone_high=zone_high,
            zone_low=zone_low,
            zone_mid=zone_mid,
            formation_bars=len(bars),
            compression_ratio=compression_ratio,
            break_level=breakout_level,
            confidence=confidence,
            timestamp=datetime.now(),
        )

    def _classify_zone_type(self, bars: List[Bar]) -> ZoneType:
        """Classify the type of zone pattern"""
        if len(bars) < 3:
            return ZoneType.CONSOLIDATION

        # Check for flag pattern (higher lows during consolidation)
        lows = [b.low for b in bars]
        if all(lows[i] >= lows[i - 1] for i in range(1, len(lows))):
            return ZoneType.FLAG

        # Check for triangle (converging highs and lows)
        highs = [b.high for b in bars]
        highs_declining = all(highs[i] <= highs[i - 1] for i in range(1, len(highs)))
        lows_rising = all(lows[i] >= lows[i - 1] for i in range(1, len(lows)))
        if highs_declining and lows_rising:
            return ZoneType.TRIANGLE

        # Check for pullback (initial drop then stabilization)
        if bars[0].close > bars[-1].close * 1.02:  # 2% pullback
            return ZoneType.PULLBACK

        return ZoneType.CONSOLIDATION

    def _calculate_confidence(
        self,
        bars: List[Bar],
        zone_width_pct: float,
        compression_ratio: float,
    ) -> float:
        """Calculate zone confidence score (0-100)"""
        confidence = 50.0  # Base

        # Tighter zone = higher confidence
        if zone_width_pct < 1.0:
            confidence += 20
        elif zone_width_pct < 1.5:
            confidence += 10

        # More compression = higher confidence
        if compression_ratio < 0.5:
            confidence += 15
        elif compression_ratio < 0.7:
            confidence += 10

        # More bars = more confirmed
        if len(bars) >= 5:
            confidence += 10
        elif len(bars) >= 3:
            confidence += 5

        # Volume declining during formation = bullish
        if len(bars) >= 3:
            avg_early = sum(b.volume for b in bars[:len(bars)//2]) / (len(bars)//2)
            avg_late = sum(b.volume for b in bars[len(bars)//2:]) / (len(bars) - len(bars)//2)
            if avg_early > 0 and avg_late < avg_early * 0.8:
                confidence += 5

        return min(100.0, max(0.0, confidence))

    def _check_resolution(self, symbol: str, bar: Bar) -> Optional[SmartZoneSignal]:
        """Check if active zone is resolved (breakout or breakdown)"""
        zone = self._zones.get(symbol)
        if not zone:
            return None

        # Check for EXPANSION (breakout above zone)
        if bar.close > zone.break_level:
            zone.is_resolved = True
            zone.resolution_type = "EXPANSION"
            zone.resolution_price = bar.close
            zone.resolution_time = bar.timestamp
            del self._zones[symbol]
            return zone

        # Check for BREAKDOWN (below zone low)
        breakdown_level = zone.zone_low * (1 - self.breakout_threshold_pct / 100)
        if bar.close < breakdown_level:
            zone.is_resolved = True
            zone.resolution_type = "BREAKDOWN"
            zone.resolution_price = bar.close
            zone.resolution_time = bar.timestamp
            del self._zones[symbol]
            return zone

        # Zone still forming
        return None

    def get_active_zone(self, symbol: str) -> Optional[SmartZoneSignal]:
        """Get active zone for symbol"""
        return self._zones.get(symbol)

    def get_all_active_zones(self) -> dict[str, SmartZoneSignal]:
        """Get all active zones"""
        return self._zones.copy()

    def clear_symbol(self, symbol: str):
        """Clear state for symbol"""
        if symbol in self._zones:
            del self._zones[symbol]
        if symbol in self._bar_buffers:
            del self._bar_buffers[symbol]

    def get_status(self) -> dict:
        """Get detector status"""
        return {
            "active_zones": len(self._zones),
            "tracked_symbols": len(self._bar_buffers),
            "zones": {
                sym: {
                    "zone_type": z.zone_type.value,
                    "zone_high": z.zone_high,
                    "zone_low": z.zone_low,
                    "break_level": z.break_level,
                    "formation_bars": z.formation_bars,
                    "confidence": z.confidence,
                }
                for sym, z in self._zones.items()
            }
        }


# Singleton instance
_detector: Optional[SmartZoneDetector] = None


def get_smartzone_detector() -> SmartZoneDetector:
    """Get singleton SmartZone detector"""
    global _detector
    if _detector is None:
        _detector = SmartZoneDetector()
    return _detector
