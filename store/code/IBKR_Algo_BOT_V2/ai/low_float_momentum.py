"""
Low-Float Momentum Detector
===========================
Detects low-float momentum plays and provides CONFIDENCE INPUT for gating.

IMPORTANT: This module provides confidence signals ONLY.
All trades MUST still go through the Signal Gating Engine.
NO BYPASS PATHS are allowed.

These setups behave differently than normal stocks:
- Extreme gaps are GOOD (shows conviction)
- High volatility is EXPECTED
- Signal strength is used as CONFIDENCE INPUT to gating

Criteria for LOW_FLOAT_MOMENTUM:
- Float < 10M shares (tight supply)
- Volume > 5x average (demand surge)
- Gap > 10% (momentum)
- Price $1-$20 (scalp range)
- Float rotation > 50% (real interest)

Output: Signal strength (STRONG/MODERATE/WEAK/NONE) as confidence input.
"""

import yfinance as yf
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class MomentumSignal(Enum):
    """Momentum signal strength (CONFIDENCE INPUT - NO BYPASS)"""
    STRONG = "STRONG"          # All criteria met - high confidence input
    MODERATE = "MODERATE"      # Most criteria met - moderate confidence input
    WEAK = "WEAK"             # Some criteria - low confidence input
    NONE = "NONE"             # Not a low-float momentum play


@dataclass
class LowFloatAnalysis:
    """Analysis result for low-float momentum detection"""
    symbol: str
    signal: MomentumSignal

    # Metrics
    float_shares: float
    avg_volume: float
    current_volume: float
    volume_ratio: float
    gap_percent: float
    float_rotation: float
    current_price: float

    # Flags
    is_low_float: bool
    has_volume_surge: bool
    has_large_gap: bool
    has_float_rotation: bool
    is_in_price_range: bool

    # Confidence outputs (NO BYPASS - inputs to gating engine)
    confidence_score: float  # 0.0-1.0 confidence for gating
    reason: str

    def to_dict(self):
        return {
            'symbol': self.symbol,
            'signal': self.signal.value,
            'float_shares_m': round(self.float_shares / 1e6, 2),
            'avg_volume_m': round(self.avg_volume / 1e6, 2),
            'current_volume_m': round(self.current_volume / 1e6, 2),
            'volume_ratio': round(self.volume_ratio, 1),
            'gap_percent': round(self.gap_percent, 1),
            'float_rotation_pct': round(self.float_rotation, 1),
            'current_price': round(self.current_price, 2),
            'is_low_float': self.is_low_float,
            'has_volume_surge': self.has_volume_surge,
            'has_large_gap': self.has_large_gap,
            'has_float_rotation': self.has_float_rotation,
            'is_in_price_range': self.is_in_price_range,
            'confidence_score': round(self.confidence_score, 2),
            'reason': self.reason
        }


class LowFloatMomentumDetector:
    """Detects low-float momentum setups and provides CONFIDENCE INPUT for gating (NO BYPASS)"""

    # Thresholds
    MAX_FLOAT = 10_000_000      # 10M shares max
    MIN_VOLUME_RATIO = 5.0       # 5x average volume
    MIN_GAP_PERCENT = 10.0       # 10% minimum gap
    MIN_FLOAT_ROTATION = 50.0    # 50% of float traded
    MIN_PRICE = 1.0
    MAX_PRICE = 20.0

    # Strong signal thresholds (high confidence input to gating)
    STRONG_MIN_FLOAT_ROTATION = 100.0   # Float traded 1x
    STRONG_MIN_VOLUME_RATIO = 10.0      # 10x average volume

    def analyze(self, symbol: str,
                current_price: float = None,
                current_volume: float = None,
                gap_percent: float = None) -> LowFloatAnalysis:
        """
        Analyze a symbol for low-float momentum characteristics.

        Args:
            symbol: Stock symbol
            current_price: Optional current price (fetched if not provided)
            current_volume: Optional current volume (fetched if not provided)
            gap_percent: Optional gap percent (calculated if not provided)
        """
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='5d')

            if hist.empty or len(hist) < 2:
                return self._empty_analysis(symbol, "No price data")

            # Get metrics
            float_shares = info.get('floatShares', 0) or 0
            avg_volume = info.get('averageVolume', 0) or 1

            if current_price is None:
                current_price = hist.iloc[-1]['Close']
            if current_volume is None:
                current_volume = hist.iloc[-1]['Volume']
            if gap_percent is None:
                prior_close = hist.iloc[-2]['Close']
                gap_percent = ((current_price - prior_close) / prior_close) * 100 if prior_close > 0 else 0

            # Calculate metrics
            volume_ratio = current_volume / max(avg_volume, 1)
            float_rotation = (current_volume / max(float_shares, 1)) * 100 if float_shares > 0 else 0

            # Check flags
            is_low_float = float_shares > 0 and float_shares <= self.MAX_FLOAT
            has_volume_surge = volume_ratio >= self.MIN_VOLUME_RATIO
            has_large_gap = abs(gap_percent) >= self.MIN_GAP_PERCENT
            has_float_rotation = float_rotation >= self.MIN_FLOAT_ROTATION
            is_in_price_range = self.MIN_PRICE <= current_price <= self.MAX_PRICE

            # Determine signal strength
            signal = self._determine_signal(
                is_low_float, has_volume_surge, has_large_gap,
                has_float_rotation, is_in_price_range,
                volume_ratio, float_rotation
            )

            # Calculate confidence score based on signal strength (NO BYPASS - input to gating)
            confidence_score = {
                MomentumSignal.STRONG: 1.0,
                MomentumSignal.MODERATE: 0.7,
                MomentumSignal.WEAK: 0.3,
                MomentumSignal.NONE: 0.0
            }.get(signal, 0.0)

            # Build reason
            reason = self._build_reason(signal, float_shares, volume_ratio, gap_percent, float_rotation)

            return LowFloatAnalysis(
                symbol=symbol,
                signal=signal,
                float_shares=float_shares,
                avg_volume=avg_volume,
                current_volume=current_volume,
                volume_ratio=volume_ratio,
                gap_percent=gap_percent,
                float_rotation=float_rotation,
                current_price=current_price,
                is_low_float=is_low_float,
                has_volume_surge=has_volume_surge,
                has_large_gap=has_large_gap,
                has_float_rotation=has_float_rotation,
                is_in_price_range=is_in_price_range,
                confidence_score=confidence_score,
                reason=reason
            )

        except Exception as e:
            return self._empty_analysis(symbol, f"Error: {str(e)}")

    def _determine_signal(self, is_low_float: bool, has_volume_surge: bool,
                          has_large_gap: bool, has_float_rotation: bool,
                          is_in_price_range: bool, volume_ratio: float,
                          float_rotation: float) -> MomentumSignal:
        """Determine signal strength based on criteria"""

        # Must be in price range and have some volume
        if not is_in_price_range or not has_volume_surge:
            return MomentumSignal.NONE

        # Count criteria met
        criteria_met = sum([is_low_float, has_large_gap, has_float_rotation])

        # STRONG: Low float + all criteria + extreme metrics
        if (is_low_float and has_volume_surge and has_large_gap and has_float_rotation and
            volume_ratio >= self.STRONG_MIN_VOLUME_RATIO and
            float_rotation >= self.STRONG_MIN_FLOAT_ROTATION):
            return MomentumSignal.STRONG

        # MODERATE: Low float + most criteria
        if is_low_float and criteria_met >= 2:
            return MomentumSignal.MODERATE

        # WEAK: Some criteria but not enough
        if criteria_met >= 1:
            return MomentumSignal.WEAK

        return MomentumSignal.NONE

    def _build_reason(self, signal: MomentumSignal, float_shares: float,
                      volume_ratio: float, gap_percent: float, float_rotation: float) -> str:
        """Build explanation for the signal (CONFIDENCE INPUT - NO BYPASS)"""

        if signal == MomentumSignal.STRONG:
            return (f"LOW-FLOAT MOMENTUM: Float {float_shares/1e6:.1f}M, "
                   f"Vol {volume_ratio:.0f}x avg, Gap {gap_percent:+.0f}%, "
                   f"Rotation {float_rotation:.0f}% - HIGH CONFIDENCE (1.0)")

        elif signal == MomentumSignal.MODERATE:
            return (f"Moderate momentum: Float {float_shares/1e6:.1f}M, "
                   f"Vol {volume_ratio:.0f}x avg, Gap {gap_percent:+.0f}% - MODERATE CONFIDENCE (0.7)")

        elif signal == MomentumSignal.WEAK:
            return (f"Weak momentum: Vol {volume_ratio:.0f}x avg, "
                   f"Gap {gap_percent:+.0f}% - LOW CONFIDENCE (0.3)")

        else:
            return "Not a low-float momentum setup (confidence: 0.0)"

    def _empty_analysis(self, symbol: str, reason: str) -> LowFloatAnalysis:
        """Return empty analysis for error cases"""
        return LowFloatAnalysis(
            symbol=symbol,
            signal=MomentumSignal.NONE,
            float_shares=0,
            avg_volume=0,
            current_volume=0,
            volume_ratio=0,
            gap_percent=0,
            float_rotation=0,
            current_price=0,
            is_low_float=False,
            has_volume_surge=False,
            has_large_gap=False,
            has_float_rotation=False,
            is_in_price_range=False,
            confidence_score=0.0,
            reason=reason
        )


# Singleton instance
_detector: Optional[LowFloatMomentumDetector] = None


def get_low_float_detector() -> LowFloatMomentumDetector:
    """Get singleton detector instance"""
    global _detector
    if _detector is None:
        _detector = LowFloatMomentumDetector()
    return _detector


def check_low_float_momentum(symbol: str, **kwargs) -> LowFloatAnalysis:
    """Quick check for low-float momentum (CONFIDENCE INPUT for gating - NO BYPASS)"""
    return get_low_float_detector().analyze(symbol, **kwargs)


if __name__ == "__main__":
    # Test with ECDA and SOPA
    detector = LowFloatMomentumDetector()

    for sym in ['ECDA', 'SOPA', 'AAPL', 'TSLA']:
        print(f"\n{'='*60}")
        result = detector.analyze(sym)
        print(f"{sym}: {result.signal.value}")
        print(f"  Float: {result.float_shares/1e6:.2f}M")
        print(f"  Volume: {result.volume_ratio:.1f}x avg")
        print(f"  Gap: {result.gap_percent:+.1f}%")
        print(f"  Float Rotation: {result.float_rotation:.0f}%")
        print(f"  Confidence Score: {result.confidence_score}")
        print(f"  {result.reason}")
