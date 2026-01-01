"""
Chart Pattern Recognition Module
================================
Detects common chart patterns using OHLC price data.
Uses technical analysis to identify trading opportunities.

Supported Patterns:
- Support/Resistance levels
- Double Top/Bottom
- Head and Shoulders
- Triangle patterns (ascending, descending, symmetrical)
- Bull/Bear Flags
- Breakout detection
"""

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of chart patterns"""

    SUPPORT = "support"
    RESISTANCE = "resistance"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_shoulders"
    INV_HEAD_SHOULDERS = "inv_head_shoulders"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    BREAKOUT_UP = "breakout_up"
    BREAKOUT_DOWN = "breakout_down"
    CONSOLIDATION = "consolidation"


class PatternStrength(Enum):
    """Strength of detected pattern"""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


@dataclass
class ChartPattern:
    """Detected chart pattern"""

    pattern_type: PatternType
    strength: PatternStrength
    start_index: int
    end_index: int
    price_level: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    confidence: float = 0.0
    description: str = ""

    def to_dict(self) -> Dict:
        return {
            "pattern": self.pattern_type.value,
            "strength": self.strength.value,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "price_level": round(self.price_level, 2),
            "target_price": round(self.target_price, 2) if self.target_price else None,
            "stop_loss": round(self.stop_loss, 2) if self.stop_loss else None,
            "confidence": round(self.confidence, 2),
            "description": self.description,
        }


@dataclass
class SupportResistance:
    """Support or resistance level"""

    level: float
    strength: int  # Number of touches
    is_support: bool
    last_tested: int  # Index of last test

    def to_dict(self) -> Dict:
        return {
            "level": round(self.level, 2),
            "strength": self.strength,
            "type": "support" if self.is_support else "resistance",
            "last_tested": self.last_tested,
        }


class ChartPatternRecognizer:
    """
    Recognizes chart patterns from OHLC price data.
    Uses peak/trough detection and pattern matching.
    """

    def __init__(self):
        self.min_pattern_bars = 5  # Minimum bars for pattern
        self.sr_tolerance = 0.02  # 2% tolerance for S/R levels

    def analyze(self, candles: List[Dict]) -> Dict:
        """
        Analyze price data and detect all patterns.

        Args:
            candles: List of OHLC candles with keys: open, high, low, close, volume

        Returns:
            Dictionary with detected patterns and levels
        """
        if not candles or len(candles) < self.min_pattern_bars:
            return {
                "patterns": [],
                "support_levels": [],
                "resistance_levels": [],
                "trend": "neutral",
                "message": "Insufficient data for pattern analysis",
            }

        # Extract price arrays
        highs = np.array([c.get("high", c.get("h", 0)) for c in candles])
        lows = np.array([c.get("low", c.get("l", 0)) for c in candles])
        closes = np.array([c.get("close", c.get("c", 0)) for c in candles])
        volumes = np.array([c.get("volume", c.get("v", 0)) for c in candles])

        # Find peaks and troughs
        peaks = self._find_peaks(highs)
        troughs = self._find_troughs(lows)

        # Detect S/R levels
        support_levels = self._find_support_levels(lows, closes)
        resistance_levels = self._find_resistance_levels(highs, closes)

        # Detect patterns
        patterns = []

        # Check for double patterns
        double_patterns = self._detect_double_patterns(highs, lows, peaks, troughs)
        patterns.extend(double_patterns)

        # Check for triangles
        triangles = self._detect_triangles(highs, lows, peaks, troughs)
        patterns.extend(triangles)

        # Check for flags
        flags = self._detect_flags(highs, lows, closes, volumes)
        patterns.extend(flags)

        # Check for breakouts
        breakouts = self._detect_breakouts(closes, support_levels, resistance_levels)
        patterns.extend(breakouts)

        # Determine overall trend
        trend = self._calculate_trend(closes)

        return {
            "patterns": [p.to_dict() for p in patterns],
            "support_levels": [s.to_dict() for s in support_levels],
            "resistance_levels": [r.to_dict() for r in resistance_levels],
            "trend": trend,
            "current_price": float(closes[-1]) if len(closes) > 0 else 0,
            "analyzed_bars": len(candles),
        }

    def _find_peaks(self, highs: np.ndarray, order: int = 3) -> List[int]:
        """Find local maxima (peaks) in price data"""
        peaks = []
        for i in range(order, len(highs) - order):
            if all(highs[i] >= highs[i - j] for j in range(1, order + 1)) and all(
                highs[i] >= highs[i + j] for j in range(1, order + 1)
            ):
                peaks.append(i)
        return peaks

    def _find_troughs(self, lows: np.ndarray, order: int = 3) -> List[int]:
        """Find local minima (troughs) in price data"""
        troughs = []
        for i in range(order, len(lows) - order):
            if all(lows[i] <= lows[i - j] for j in range(1, order + 1)) and all(
                lows[i] <= lows[i + j] for j in range(1, order + 1)
            ):
                troughs.append(i)
        return troughs

    def _find_support_levels(
        self, lows: np.ndarray, closes: np.ndarray
    ) -> List[SupportResistance]:
        """Find support levels from price data"""
        if len(lows) < 5:
            return []

        levels = []
        tolerance = self.sr_tolerance * np.mean(closes)

        # Find price clusters in lows
        troughs = self._find_troughs(lows)

        for trough_idx in troughs:
            level = lows[trough_idx]
            # Count how many times price touched this level
            touches = np.sum(np.abs(lows - level) <= tolerance)

            if touches >= 2:
                # Find last time level was tested
                tests = np.where(np.abs(lows - level) <= tolerance)[0]
                last_test = int(tests[-1]) if len(tests) > 0 else trough_idx

                # Check if not too close to existing levels
                is_unique = True
                for existing in levels:
                    if abs(existing.level - level) <= tolerance:
                        is_unique = False
                        if touches > existing.strength:
                            existing.strength = touches
                            existing.level = level
                        break

                if is_unique:
                    levels.append(
                        SupportResistance(
                            level=float(level),
                            strength=int(touches),
                            is_support=True,
                            last_tested=last_test,
                        )
                    )

        # Sort by strength
        levels.sort(key=lambda x: x.strength, reverse=True)
        return levels[:5]  # Return top 5 levels

    def _find_resistance_levels(
        self, highs: np.ndarray, closes: np.ndarray
    ) -> List[SupportResistance]:
        """Find resistance levels from price data"""
        if len(highs) < 5:
            return []

        levels = []
        tolerance = self.sr_tolerance * np.mean(closes)

        # Find price clusters in highs
        peaks = self._find_peaks(highs)

        for peak_idx in peaks:
            level = highs[peak_idx]
            # Count how many times price touched this level
            touches = np.sum(np.abs(highs - level) <= tolerance)

            if touches >= 2:
                tests = np.where(np.abs(highs - level) <= tolerance)[0]
                last_test = int(tests[-1]) if len(tests) > 0 else peak_idx

                is_unique = True
                for existing in levels:
                    if abs(existing.level - level) <= tolerance:
                        is_unique = False
                        if touches > existing.strength:
                            existing.strength = touches
                            existing.level = level
                        break

                if is_unique:
                    levels.append(
                        SupportResistance(
                            level=float(level),
                            strength=int(touches),
                            is_support=False,
                            last_tested=last_test,
                        )
                    )

        levels.sort(key=lambda x: x.strength, reverse=True)
        return levels[:5]

    def _detect_double_patterns(
        self, highs: np.ndarray, lows: np.ndarray, peaks: List[int], troughs: List[int]
    ) -> List[ChartPattern]:
        """Detect double top and double bottom patterns"""
        patterns = []
        tolerance = self.sr_tolerance

        # Double Top: Two peaks at similar levels
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                for j in range(i + 1, len(peaks)):
                    peak1, peak2 = peaks[i], peaks[j]

                    # Peaks should be at similar height
                    price_diff = abs(highs[peak1] - highs[peak2]) / highs[peak1]

                    if price_diff <= tolerance and (peak2 - peak1) >= 5:
                        avg_peak = (highs[peak1] + highs[peak2]) / 2
                        # Find the trough between peaks
                        between_troughs = [t for t in troughs if peak1 < t < peak2]

                        if between_troughs:
                            neckline = min(lows[t] for t in between_troughs)
                            height = avg_peak - neckline

                            patterns.append(
                                ChartPattern(
                                    pattern_type=PatternType.DOUBLE_TOP,
                                    strength=(
                                        PatternStrength.MODERATE
                                        if price_diff < tolerance / 2
                                        else PatternStrength.WEAK
                                    ),
                                    start_index=peak1,
                                    end_index=peak2,
                                    price_level=float(avg_peak),
                                    target_price=float(
                                        neckline - height
                                    ),  # Measure move
                                    stop_loss=float(avg_peak * 1.02),
                                    confidence=1.0 - price_diff,
                                    description=f"Double top at {avg_peak:.2f}, neckline at {neckline:.2f}",
                                )
                            )

        # Double Bottom: Two troughs at similar levels
        if len(troughs) >= 2:
            for i in range(len(troughs) - 1):
                for j in range(i + 1, len(troughs)):
                    trough1, trough2 = troughs[i], troughs[j]

                    price_diff = abs(lows[trough1] - lows[trough2]) / lows[trough1]

                    if price_diff <= tolerance and (trough2 - trough1) >= 5:
                        avg_trough = (lows[trough1] + lows[trough2]) / 2
                        between_peaks = [p for p in peaks if trough1 < p < trough2]

                        if between_peaks:
                            neckline = max(highs[p] for p in between_peaks)
                            height = neckline - avg_trough

                            patterns.append(
                                ChartPattern(
                                    pattern_type=PatternType.DOUBLE_BOTTOM,
                                    strength=(
                                        PatternStrength.MODERATE
                                        if price_diff < tolerance / 2
                                        else PatternStrength.WEAK
                                    ),
                                    start_index=trough1,
                                    end_index=trough2,
                                    price_level=float(avg_trough),
                                    target_price=float(neckline + height),
                                    stop_loss=float(avg_trough * 0.98),
                                    confidence=1.0 - price_diff,
                                    description=f"Double bottom at {avg_trough:.2f}, neckline at {neckline:.2f}",
                                )
                            )

        return patterns

    def _detect_triangles(
        self, highs: np.ndarray, lows: np.ndarray, peaks: List[int], troughs: List[int]
    ) -> List[ChartPattern]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        patterns = []

        if len(peaks) < 2 or len(troughs) < 2:
            return patterns

        # Get recent peaks and troughs
        recent_peaks = peaks[-4:] if len(peaks) >= 4 else peaks
        recent_troughs = troughs[-4:] if len(troughs) >= 4 else troughs

        if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
            # Calculate slopes
            peak_values = [highs[p] for p in recent_peaks]
            trough_values = [lows[t] for t in recent_troughs]

            peak_slope = (peak_values[-1] - peak_values[0]) / max(
                1, recent_peaks[-1] - recent_peaks[0]
            )
            trough_slope = (trough_values[-1] - trough_values[0]) / max(
                1, recent_troughs[-1] - recent_troughs[0]
            )

            start_idx = min(recent_peaks[0], recent_troughs[0])
            end_idx = max(recent_peaks[-1], recent_troughs[-1])

            # Ascending Triangle: flat top, rising bottom
            if abs(peak_slope) < 0.001 and trough_slope > 0.001:
                resistance = np.mean(peak_values)
                patterns.append(
                    ChartPattern(
                        pattern_type=PatternType.ASCENDING_TRIANGLE,
                        strength=PatternStrength.MODERATE,
                        start_index=start_idx,
                        end_index=end_idx,
                        price_level=float(resistance),
                        target_price=float(resistance * 1.05),  # 5% breakout target
                        stop_loss=float(trough_values[-1] * 0.98),
                        confidence=0.7,
                        description=f"Ascending triangle, resistance at {resistance:.2f}",
                    )
                )

            # Descending Triangle: falling top, flat bottom
            elif peak_slope < -0.001 and abs(trough_slope) < 0.001:
                support = np.mean(trough_values)
                patterns.append(
                    ChartPattern(
                        pattern_type=PatternType.DESCENDING_TRIANGLE,
                        strength=PatternStrength.MODERATE,
                        start_index=start_idx,
                        end_index=end_idx,
                        price_level=float(support),
                        target_price=float(support * 0.95),  # 5% breakdown target
                        stop_loss=float(peak_values[-1] * 1.02),
                        confidence=0.7,
                        description=f"Descending triangle, support at {support:.2f}",
                    )
                )

            # Symmetrical Triangle: converging lines
            elif peak_slope < -0.001 and trough_slope > 0.001:
                mid_price = (peak_values[-1] + trough_values[-1]) / 2
                patterns.append(
                    ChartPattern(
                        pattern_type=PatternType.SYMMETRICAL_TRIANGLE,
                        strength=PatternStrength.WEAK,
                        start_index=start_idx,
                        end_index=end_idx,
                        price_level=float(mid_price),
                        confidence=0.5,
                        description="Symmetrical triangle - watch for breakout direction",
                    )
                )

        return patterns

    def _detect_flags(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> List[ChartPattern]:
        """Detect bull and bear flag patterns"""
        patterns = []

        if len(closes) < 15:
            return patterns

        # Look at last 15 bars
        recent_closes = closes[-15:]
        recent_highs = highs[-15:]
        recent_lows = lows[-15:]
        recent_volumes = volumes[-15:]

        # Check for strong move in first 5 bars (flagpole)
        pole_change = (recent_closes[4] - recent_closes[0]) / recent_closes[0]
        pole_avg_volume = np.mean(recent_volumes[:5])

        # Check for consolidation in last 10 bars (flag)
        flag_range = np.max(recent_highs[5:]) - np.min(recent_lows[5:])
        flag_avg_range = flag_range / np.mean(recent_closes[5:])
        flag_avg_volume = np.mean(recent_volumes[5:])

        # Bull Flag: strong up move followed by tight consolidation
        if pole_change > 0.05 and flag_avg_range < 0.05:  # 5%+ up, <5% consolidation
            if flag_avg_volume < pole_avg_volume * 0.7:  # Volume drying up
                pole_height = recent_closes[4] - recent_closes[0]
                patterns.append(
                    ChartPattern(
                        pattern_type=PatternType.BULL_FLAG,
                        strength=PatternStrength.MODERATE,
                        start_index=len(closes) - 15,
                        end_index=len(closes) - 1,
                        price_level=float(recent_closes[-1]),
                        target_price=float(
                            recent_closes[-1] + pole_height
                        ),  # Measured move
                        stop_loss=float(np.min(recent_lows[5:])),
                        confidence=0.65,
                        description=f"Bull flag forming, target {recent_closes[-1] + pole_height:.2f}",
                    )
                )

        # Bear Flag: strong down move followed by tight consolidation
        elif pole_change < -0.05 and flag_avg_range < 0.05:
            if flag_avg_volume < pole_avg_volume * 0.7:
                pole_height = recent_closes[0] - recent_closes[4]
                patterns.append(
                    ChartPattern(
                        pattern_type=PatternType.BEAR_FLAG,
                        strength=PatternStrength.MODERATE,
                        start_index=len(closes) - 15,
                        end_index=len(closes) - 1,
                        price_level=float(recent_closes[-1]),
                        target_price=float(recent_closes[-1] - pole_height),
                        stop_loss=float(np.max(recent_highs[5:])),
                        confidence=0.65,
                        description=f"Bear flag forming, target {recent_closes[-1] - pole_height:.2f}",
                    )
                )

        return patterns

    def _detect_breakouts(
        self,
        closes: np.ndarray,
        support_levels: List[SupportResistance],
        resistance_levels: List[SupportResistance],
    ) -> List[ChartPattern]:
        """Detect breakouts through support/resistance"""
        patterns = []

        if len(closes) < 3:
            return patterns

        current_price = closes[-1]
        prev_price = closes[-2]

        # Check for breakout above resistance
        for level in resistance_levels:
            if prev_price < level.level and current_price > level.level:
                breakout_pct = (current_price - level.level) / level.level
                if breakout_pct > 0.005:  # Confirmed breakout > 0.5%
                    patterns.append(
                        ChartPattern(
                            pattern_type=PatternType.BREAKOUT_UP,
                            strength=(
                                PatternStrength.STRONG
                                if breakout_pct > 0.02
                                else PatternStrength.MODERATE
                            ),
                            start_index=len(closes) - 3,
                            end_index=len(closes) - 1,
                            price_level=float(level.level),
                            target_price=float(level.level * 1.05),
                            stop_loss=float(level.level * 0.98),
                            confidence=min(0.9, 0.6 + (level.strength * 0.1)),
                            description=f"Breakout above {level.level:.2f} resistance (tested {level.strength}x)",
                        )
                    )

        # Check for breakdown below support
        for level in support_levels:
            if prev_price > level.level and current_price < level.level:
                breakdown_pct = (level.level - current_price) / level.level
                if breakdown_pct > 0.005:
                    patterns.append(
                        ChartPattern(
                            pattern_type=PatternType.BREAKOUT_DOWN,
                            strength=(
                                PatternStrength.STRONG
                                if breakdown_pct > 0.02
                                else PatternStrength.MODERATE
                            ),
                            start_index=len(closes) - 3,
                            end_index=len(closes) - 1,
                            price_level=float(level.level),
                            target_price=float(level.level * 0.95),
                            stop_loss=float(level.level * 1.02),
                            confidence=min(0.9, 0.6 + (level.strength * 0.1)),
                            description=f"Breakdown below {level.level:.2f} support (tested {level.strength}x)",
                        )
                    )

        return patterns

    def _calculate_trend(self, closes: np.ndarray) -> str:
        """Calculate overall trend using moving averages"""
        if len(closes) < 20:
            return "neutral"

        sma_10 = np.mean(closes[-10:])
        sma_20 = np.mean(closes[-20:])
        current = closes[-1]

        if current > sma_10 > sma_20:
            return "bullish"
        elif current < sma_10 < sma_20:
            return "bearish"
        else:
            return "neutral"


# Global instance
_pattern_recognizer: Optional[ChartPatternRecognizer] = None


def get_pattern_recognizer() -> ChartPatternRecognizer:
    """Get or create the pattern recognizer instance"""
    global _pattern_recognizer
    if _pattern_recognizer is None:
        _pattern_recognizer = ChartPatternRecognizer()
    return _pattern_recognizer


logger.info("Chart Pattern Recognition module loaded")
