"""
Pattern Detector - Ross Cameron Setup Detection
================================================
Detects key momentum patterns from Warrior Trading:
- Bull Flag (Setup 1)
- ABCD Pattern (Setup 2)
- Micro Pullback (Setup 4)
- High of Day Break (Setup 5)

Based on: "The best days are when multiple setups align on the same stock"
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    """OHLCV candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    @property
    def is_green(self) -> bool:
        return self.close > self.open

    @property
    def is_red(self) -> bool:
        return self.close < self.open

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low


@dataclass
class PatternResult:
    """Result of pattern detection"""
    pattern_type: str  # BULL_FLAG, ABCD, MICRO_PB, HOD_BREAK
    detected: bool = False
    confidence: float = 0.0

    # Pattern points
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0

    # Pattern-specific data
    details: Dict = field(default_factory=dict)

    # Timing
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            'pattern_type': self.pattern_type,
            'detected': self.detected,
            'confidence': round(self.confidence, 3),
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target_price': self.target_price,
            'risk_reward': round((self.target_price - self.entry_price) /
                                 (self.entry_price - self.stop_loss), 2)
                          if self.stop_loss and self.entry_price else 0,
            'details': self.details,
            'detected_at': self.detected_at.isoformat()
        }


class PatternDetector:
    """
    Detects momentum patterns for entry signals.

    Key Patterns:
    1. Bull Flag - Consolidation after strong move
    2. ABCD - Measured move pattern
    3. Micro Pullback - Quick dip during momentum
    4. HOD Break - New high of day breakout
    """

    def __init__(self, max_candles: int = 100):
        self.candle_history: Dict[str, deque] = {}  # symbol -> candles
        self.max_candles = max_candles

        # Pattern thresholds
        self.flag_min_pole_pct = 10.0  # 10% min move for pole
        self.flag_max_retrace = 0.50   # 50% max retracement
        self.flag_min_candles = 2
        self.flag_max_candles = 8

        self.abcd_fib_levels = (0.50, 0.618)  # C retracement
        self.abcd_fib_tolerance = 0.10

        self.micro_pb_max_retrace = 0.30  # 30% max
        self.micro_pb_max_candles = 3

        logger.info("PatternDetector initialized")

    def add_candle(self, symbol: str, candle: Candle):
        """Add a candle to history"""
        if symbol not in self.candle_history:
            self.candle_history[symbol] = deque(maxlen=self.max_candles)
        self.candle_history[symbol].append(candle)

    def add_candles_batch(self, symbol: str, candles: List[Dict]):
        """Add multiple candles from API response"""
        for c in candles:
            candle = Candle(
                timestamp=datetime.fromisoformat(c.get('timestamp', datetime.now().isoformat())),
                open=float(c.get('open', 0)),
                high=float(c.get('high', 0)),
                low=float(c.get('low', 0)),
                close=float(c.get('close', 0)),
                volume=int(c.get('volume', 0))
            )
            self.add_candle(symbol, candle)

    def detect_bull_flag(self, symbol: str) -> PatternResult:
        """
        Detect Bull Flag pattern.

        Pattern Structure:
        - Pole: Strong move up 10-20%+ in 5-10 min
        - Flag: Consolidation 2-8 candles, under 50% retracement
        - Volume: Decreases during flag

        Entry: Break above flag high
        Stop: Low of flag
        Target: Pole height added to breakout
        """
        result = PatternResult(pattern_type="BULL_FLAG")

        candles = list(self.candle_history.get(symbol, []))
        if len(candles) < 10:
            return result

        # Look for pole (strong move up)
        # Scan backwards for a strong upward move
        pole_start_idx = None
        pole_end_idx = None
        best_pole_pct = 0

        for i in range(len(candles) - 5, 4, -1):
            # Check if candles i to some earlier point form a strong move
            for j in range(max(0, i - 15), i - 3):  # 3-15 candles for pole
                pole_low = min(c.low for c in candles[j:i+1])
                pole_high = max(c.high for c in candles[j:i+1])
                pole_pct = (pole_high - pole_low) / pole_low * 100

                if pole_pct >= self.flag_min_pole_pct and pole_pct > best_pole_pct:
                    # Check if mostly green candles (momentum)
                    green_count = sum(1 for c in candles[j:i+1] if c.is_green)
                    if green_count >= len(candles[j:i+1]) * 0.6:  # 60%+ green
                        pole_start_idx = j
                        pole_end_idx = i
                        best_pole_pct = pole_pct

        if pole_start_idx is None:
            return result

        # Look for flag (consolidation after pole)
        flag_candles = candles[pole_end_idx + 1:]
        if len(flag_candles) < self.flag_min_candles:
            return result
        if len(flag_candles) > self.flag_max_candles:
            flag_candles = flag_candles[:self.flag_max_candles]

        # Check flag characteristics
        pole_high = max(c.high for c in candles[pole_start_idx:pole_end_idx+1])
        pole_low = candles[pole_start_idx].low
        pole_height = pole_high - pole_low

        flag_low = min(c.low for c in flag_candles)
        flag_high = max(c.high for c in flag_candles)

        # Retracement check
        retrace_pct = (pole_high - flag_low) / pole_height
        if retrace_pct > self.flag_max_retrace:
            return result

        # Volume should decrease in flag
        pole_avg_vol = np.mean([c.volume for c in candles[pole_start_idx:pole_end_idx+1]])
        flag_avg_vol = np.mean([c.volume for c in flag_candles])
        volume_decreasing = flag_avg_vol < pole_avg_vol * 0.8

        # Pattern detected
        result.detected = True
        result.entry_price = flag_high  # Break above flag
        result.stop_loss = flag_low
        result.target_price = flag_high + pole_height  # Measure move

        # Confidence based on factors
        confidence = 0.5
        if volume_decreasing:
            confidence += 0.2
        if retrace_pct < 0.38:  # Shallow retrace
            confidence += 0.15
        if len(flag_candles) >= 3:  # Good consolidation
            confidence += 0.15

        result.confidence = min(confidence, 1.0)
        result.details = {
            'pole_pct': round(best_pole_pct, 1),
            'pole_height': round(pole_height, 2),
            'retrace_pct': round(retrace_pct * 100, 1),
            'flag_candles': len(flag_candles),
            'volume_decreasing': volume_decreasing
        }

        return result

    def detect_abcd(self, symbol: str) -> PatternResult:
        """
        Detect ABCD pattern.

        Pattern Structure:
        - A: Low (start)
        - B: High (first peak)
        - C: Pullback (50-61.8% of A→B, MUST hold above A)
        - D: Target (equal to A→B move from C)

        Entry: Break above B after C formed
        Stop: Below C
        Target: D level (equal move)
        """
        result = PatternResult(pattern_type="ABCD")

        candles = list(self.candle_history.get(symbol, []))
        if len(candles) < 15:
            return result

        # Find potential A-B-C points in recent candles
        # A = significant low, B = subsequent high, C = pullback

        # Look for pattern in last 20 candles
        recent = candles[-20:]

        # Find local lows and highs
        lows = []
        highs = []

        for i in range(2, len(recent) - 2):
            # Local low
            if recent[i].low <= min(recent[i-1].low, recent[i-2].low,
                                     recent[i+1].low, recent[i+2].low):
                lows.append((i, recent[i].low))
            # Local high
            if recent[i].high >= max(recent[i-1].high, recent[i-2].high,
                                      recent[i+1].high, recent[i+2].high):
                highs.append((i, recent[i].high))

        if len(lows) < 2 or len(highs) < 1:
            return result

        # Try to find A-B-C sequence
        for a_idx, a_price in lows[:-1]:
            for b_idx, b_price in highs:
                if b_idx <= a_idx:
                    continue

                # Find C after B
                for c_idx, c_price in lows:
                    if c_idx <= b_idx:
                        continue

                    # C must be above A
                    if c_price <= a_price:
                        continue

                    # Check Fibonacci retracement
                    ab_move = b_price - a_price
                    bc_retrace = (b_price - c_price) / ab_move

                    fib_low, fib_high = self.abcd_fib_levels
                    if fib_low - self.abcd_fib_tolerance <= bc_retrace <= fib_high + self.abcd_fib_tolerance:
                        # Valid ABCD pattern found
                        result.detected = True
                        result.entry_price = b_price  # Break above B
                        result.stop_loss = c_price - 0.05  # Just below C
                        result.target_price = c_price + ab_move  # D = C + AB

                        # Confidence based on how close to fib levels
                        fib_mid = (fib_low + fib_high) / 2
                        fib_accuracy = 1 - abs(bc_retrace - fib_mid) / 0.2
                        result.confidence = max(0.5, min(0.9, fib_accuracy))

                        result.details = {
                            'a_price': a_price,
                            'b_price': b_price,
                            'c_price': c_price,
                            'd_target': round(result.target_price, 2),
                            'ab_move': round(ab_move, 2),
                            'bc_retrace_pct': round(bc_retrace * 100, 1),
                            'fib_level': f"{fib_low*100:.0f}-{fib_high*100:.0f}%"
                        }
                        return result

        return result

    def detect_micro_pullback(self, symbol: str) -> PatternResult:
        """
        Detect Micro Pullback pattern.

        Pattern Structure:
        - Strong momentum up (MACD above signal)
        - Mini retracement 10-30% (1-3 candles)
        - First green candle = entry

        Entry: First green candle after red pullback
        Stop: Low of pullback
        Target: Previous high + extension

        "When something is really strong, you will have these
        micro pullbacks and they'll just keep going higher"
        """
        result = PatternResult(pattern_type="MICRO_PULLBACK")

        candles = list(self.candle_history.get(symbol, []))
        if len(candles) < 8:
            return result

        # Check last few candles for pattern
        recent = candles[-8:]

        # Need momentum phase (mostly green, making highs)
        momentum_candles = recent[:5]
        green_count = sum(1 for c in momentum_candles if c.is_green)
        if green_count < 3:  # Need 60%+ green
            return result

        # Find the high of momentum phase
        momentum_high = max(c.high for c in momentum_candles)
        momentum_low = min(c.low for c in momentum_candles)
        momentum_range = momentum_high - momentum_low

        # Check for pullback (1-3 red candles)
        pullback_candles = recent[5:]
        red_count = sum(1 for c in pullback_candles if c.is_red)

        if red_count == 0:
            return result  # No pullback

        if len(pullback_candles) > self.micro_pb_max_candles:
            return result  # Too long, not "micro"

        pullback_low = min(c.low for c in pullback_candles)
        pullback_retrace = (momentum_high - pullback_low) / momentum_range

        if pullback_retrace > self.micro_pb_max_retrace:
            return result  # Too deep

        # Check if last candle is green (entry signal)
        if pullback_candles[-1].is_green:
            result.detected = True
            result.entry_price = pullback_candles[-1].close
            result.stop_loss = pullback_low
            result.target_price = momentum_high + (momentum_high - momentum_low) * 0.5

            # Confidence
            confidence = 0.6
            if pullback_retrace < 0.15:  # Very shallow
                confidence += 0.2
            if red_count == 1:  # Single candle pullback
                confidence += 0.1

            result.confidence = min(confidence, 0.9)
            result.details = {
                'momentum_high': momentum_high,
                'pullback_low': pullback_low,
                'retrace_pct': round(pullback_retrace * 100, 1),
                'pullback_candles': len(pullback_candles),
                'green_entry': True
            }

        return result

    def detect_hod_break(self, symbol: str, current_hod: float = None) -> PatternResult:
        """
        Detect High of Day Break pattern.

        Pattern Structure:
        - Stock making new HOD
        - Consolidation near HOD
        - Break above with volume

        Entry: Break above HOD
        Stop: Low of consolidation or 5-10 cents
        Target: HOD + previous range

        "HOD Momentum is my go-to setup, when done correctly
        it gives me a beautiful 2:1 reward to risk ratio"
        """
        result = PatternResult(pattern_type="HOD_BREAK")

        candles = list(self.candle_history.get(symbol, []))
        if len(candles) < 5:
            return result

        # Calculate HOD from candles if not provided
        if current_hod is None:
            current_hod = max(c.high for c in candles)

        recent = candles[-5:]
        last_candle = recent[-1]

        # Check if we're near HOD (within 1%)
        distance_from_hod = (current_hod - last_candle.close) / current_hod
        if distance_from_hod > 0.01:  # More than 1% away
            return result

        # Check for consolidation (tight range near HOD)
        consolidation_range = max(c.high for c in recent[-3:]) - min(c.low for c in recent[-3:])
        avg_range = np.mean([c.range for c in candles[-10:]])

        tight_consolidation = consolidation_range < avg_range * 1.5

        # Check volume building
        recent_vol = np.mean([c.volume for c in recent[-3:]])
        prior_vol = np.mean([c.volume for c in candles[-10:-3]])
        volume_building = recent_vol >= prior_vol * 0.8

        if tight_consolidation:
            result.detected = True
            result.entry_price = current_hod + 0.01  # Pennies above HOD
            result.stop_loss = min(c.low for c in recent[-3:])

            # Target: HOD + average range
            avg_day_range = max(c.high for c in candles) - min(c.low for c in candles)
            result.target_price = current_hod + avg_day_range * 0.5

            confidence = 0.6
            if volume_building:
                confidence += 0.2
            if distance_from_hod < 0.005:  # Very close
                confidence += 0.1

            result.confidence = min(confidence, 0.9)
            result.details = {
                'current_hod': current_hod,
                'distance_from_hod_pct': round(distance_from_hod * 100, 2),
                'consolidation_range': round(consolidation_range, 2),
                'tight_consolidation': tight_consolidation,
                'volume_building': volume_building
            }

        return result

    def detect_all_patterns(self, symbol: str, current_hod: float = None) -> Dict[str, PatternResult]:
        """
        Run all pattern detectors and return results.

        Returns dict of pattern_type -> PatternResult
        """
        results = {
            'bull_flag': self.detect_bull_flag(symbol),
            'abcd': self.detect_abcd(symbol),
            'micro_pullback': self.detect_micro_pullback(symbol),
            'hod_break': self.detect_hod_break(symbol, current_hod)
        }
        return results

    def get_best_setup(self, symbol: str, current_hod: float = None) -> Optional[PatternResult]:
        """
        Get the highest confidence pattern currently detected.
        """
        patterns = self.detect_all_patterns(symbol, current_hod)

        detected = [p for p in patterns.values() if p.detected]
        if not detected:
            return None

        return max(detected, key=lambda x: x.confidence)

    def get_status(self) -> Dict:
        """Get detector status"""
        return {
            'symbols_tracked': list(self.candle_history.keys()),
            'candle_counts': {s: len(c) for s, c in self.candle_history.items()},
            'thresholds': {
                'flag_min_pole_pct': self.flag_min_pole_pct,
                'flag_max_retrace': self.flag_max_retrace,
                'abcd_fib_levels': self.abcd_fib_levels,
                'micro_pb_max_retrace': self.micro_pb_max_retrace
            }
        }


# Singleton instance
_pattern_detector: Optional[PatternDetector] = None


def get_pattern_detector() -> PatternDetector:
    """Get or create PatternDetector instance"""
    global _pattern_detector
    if _pattern_detector is None:
        _pattern_detector = PatternDetector()
    return _pattern_detector


# Convenience functions
async def detect_patterns(symbol: str) -> Dict[str, PatternResult]:
    """Detect all patterns for a symbol"""
    return get_pattern_detector().detect_all_patterns(symbol)


async def get_trading_setup(symbol: str) -> Optional[PatternResult]:
    """Get best trading setup for a symbol"""
    return get_pattern_detector().get_best_setup(symbol)
