"""
Momentum Score Calculator
=========================
Unified 0-100 momentum score from 3 buckets:
- Price/Urgency (0-40): ROC, range expansion, HOD proximity
- Participation (0-35): volume surge, float rotation
- Liquidity/Confirmation (0-25): order flow, spread, tape confirmation

Based on ChatGPT analysis: "Momentum = directional urgency + participation + liquidity"
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class MomentumGrade(Enum):
    """Momentum grade based on score"""
    A = "A"  # 80-100: Strong ignition candidate
    B = "B"  # 65-79: Good setup, watch for ignition
    C = "C"  # 50-64: Marginal, needs more confirmation
    D = "D"  # 35-49: Weak, likely to fade
    F = "F"  # 0-34: No momentum, avoid


@dataclass
class PriceUrgencyScore:
    """Price/Urgency component (0-40 points)"""
    roc_30s: float = 0.0       # Rate of change last 30 seconds
    roc_60s: float = 0.0       # Rate of change last 60 seconds
    roc_5m: float = 0.0        # Rate of change last 5 minutes
    range_expansion: float = 0.0  # Current bar range vs median
    hod_distance_pct: float = 0.0  # Distance from high of day
    hod_break: bool = False    # Just broke HOD
    vwap_position: str = "BELOW"  # ABOVE, AT, BELOW

    # Scores
    roc_score: int = 0         # 0-15: Speed of move
    range_score: int = 0       # 0-10: Unusual range
    hod_score: int = 0         # 0-10: HOD proximity/break
    vwap_score: int = 0        # 0-5: VWAP alignment

    @property
    def total(self) -> int:
        return min(40, self.roc_score + self.range_score + self.hod_score + self.vwap_score)

    def to_dict(self) -> Dict:
        return {
            'roc_30s': round(self.roc_30s, 3),
            'roc_60s': round(self.roc_60s, 3),
            'roc_5m': round(self.roc_5m, 3),
            'range_expansion': round(self.range_expansion, 2),
            'hod_distance_pct': round(self.hod_distance_pct, 2),
            'hod_break': self.hod_break,
            'vwap_position': self.vwap_position,
            'roc_score': self.roc_score,
            'range_score': self.range_score,
            'hod_score': self.hod_score,
            'vwap_score': self.vwap_score,
            'total': self.total
        }


@dataclass
class ParticipationScore:
    """Participation component (0-35 points)"""
    volume_surge: float = 0.0      # Current volume vs average
    relative_volume: float = 0.0   # RVol for the day
    float_rotation_pct: float = 0.0  # % of float traded
    consecutive_green: int = 0     # Consecutive green candles

    # Scores
    surge_score: int = 0       # 0-15: Volume surge
    rvol_score: int = 0        # 0-10: Relative volume
    rotation_score: int = 0    # 0-10: Float rotation

    @property
    def total(self) -> int:
        return min(35, self.surge_score + self.rvol_score + self.rotation_score)

    def to_dict(self) -> Dict:
        return {
            'volume_surge': round(self.volume_surge, 2),
            'relative_volume': round(self.relative_volume, 2),
            'float_rotation_pct': round(self.float_rotation_pct, 2),
            'consecutive_green': self.consecutive_green,
            'surge_score': self.surge_score,
            'rvol_score': self.rvol_score,
            'rotation_score': self.rotation_score,
            'total': self.total
        }


@dataclass
class LiquidityScore:
    """Liquidity/Confirmation component (0-25 points)"""
    spread_pct: float = 0.0        # Bid-ask spread %
    buy_pressure: float = 0.0      # Order flow buy pressure
    imbalance_ratio: float = 0.0   # Bid vs ask imbalance
    tape_signal: str = "NEUTRAL"   # BULLISH, NEUTRAL, BEARISH

    # Scores
    spread_score: int = 0      # 0-10: Tight spread = good
    flow_score: int = 0        # 0-10: Buy pressure
    tape_score: int = 0        # 0-5: Tape reading

    @property
    def total(self) -> int:
        return min(25, self.spread_score + self.flow_score + self.tape_score)

    def to_dict(self) -> Dict:
        return {
            'spread_pct': round(self.spread_pct, 3),
            'buy_pressure': round(self.buy_pressure, 3),
            'imbalance_ratio': round(self.imbalance_ratio, 2),
            'tape_signal': self.tape_signal,
            'spread_score': self.spread_score,
            'flow_score': self.flow_score,
            'tape_score': self.tape_score,
            'total': self.total
        }


@dataclass
class MomentumResult:
    """Complete momentum analysis result"""
    symbol: str
    timestamp: datetime
    score: int                 # 0-100 total score
    grade: MomentumGrade

    # Component scores
    price_urgency: PriceUrgencyScore
    participation: ParticipationScore
    liquidity: LiquidityScore

    # Flags
    is_tradeable: bool = False     # Score >= threshold
    ignition_ready: bool = False   # All components aligned
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'score': self.score,
            'grade': self.grade.value,
            'price_urgency': self.price_urgency.to_dict(),
            'participation': self.participation.to_dict(),
            'liquidity': self.liquidity.to_dict(),
            'is_tradeable': self.is_tradeable,
            'ignition_ready': self.ignition_ready,
            'reasons': self.reasons,
            'warnings': self.warnings
        }


class MomentumScorer:
    """
    Calculates unified momentum score from real-time data.

    Entry threshold: score >= 70 AND all component minimums met
    """

    # Score thresholds
    ENTRY_THRESHOLD = 70           # Minimum score to consider entry
    IGNITION_THRESHOLD = 75        # Score needed for immediate entry

    # Component minimums for ignition
    MIN_PRICE_URGENCY = 20         # Need at least 20/40 in price movement
    MIN_PARTICIPATION = 15         # Need at least 15/35 in volume
    MIN_LIQUIDITY = 10             # Need at least 10/25 in liquidity

    # Price/Urgency thresholds
    ROC_30S_THRESHOLD = 0.3        # 0.3% in 30 seconds = fast move
    ROC_60S_THRESHOLD = 0.5        # 0.5% in 60 seconds
    ROC_5M_THRESHOLD = 2.0         # 2% in 5 minutes
    RANGE_EXPANSION_THRESHOLD = 2.0  # 2x normal range

    # Participation thresholds
    VOLUME_SURGE_THRESHOLD = 3.0   # 3x average volume
    RVOL_THRESHOLD = 2.0           # 2x relative volume
    FLOAT_ROTATION_THRESHOLD = 10.0  # 10% float rotation

    # Liquidity thresholds
    MAX_SPREAD_PCT = 1.0           # Max 1% spread
    MIN_BUY_PRESSURE = 0.55        # 55% buy pressure minimum

    def __init__(self):
        self._cache: Dict[str, MomentumResult] = {}
        self._cache_ttl = 5  # Cache results for 5 seconds

    def calculate(self,
                  symbol: str,
                  current_price: float,
                  prices_30s: List[float] = None,
                  prices_60s: List[float] = None,
                  prices_5m: List[float] = None,
                  current_bar_range: float = 0,
                  median_bar_range: float = 0,
                  high_of_day: float = 0,
                  vwap: float = 0,
                  current_volume: int = 0,
                  avg_volume: int = 0,
                  float_shares: int = 0,
                  day_volume: int = 0,
                  spread_pct: float = 0,
                  buy_pressure: float = 0.5,
                  imbalance_ratio: float = 1.0,
                  tape_signal: str = "NEUTRAL",
                  consecutive_green: int = 0) -> MomentumResult:
        """
        Calculate momentum score from provided data.

        Args:
            symbol: Stock symbol
            current_price: Current price
            prices_30s: List of prices from last 30 seconds
            prices_60s: List of prices from last 60 seconds
            prices_5m: List of prices from last 5 minutes
            current_bar_range: Current candle high - low
            median_bar_range: Median bar range
            high_of_day: High of day
            vwap: VWAP price
            current_volume: Current bar volume
            avg_volume: Average daily volume
            float_shares: Float shares
            day_volume: Total volume today
            spread_pct: Bid-ask spread as percentage
            buy_pressure: Buy pressure from order flow (0-1)
            imbalance_ratio: Bid/ask imbalance ratio
            tape_signal: Tape reading signal
            consecutive_green: Consecutive green candles
        """
        now = datetime.now()
        reasons = []
        warnings = []

        # Calculate Price/Urgency Score
        price_urgency = self._calc_price_urgency(
            current_price=current_price,
            prices_30s=prices_30s or [],
            prices_60s=prices_60s or [],
            prices_5m=prices_5m or [],
            current_bar_range=current_bar_range,
            median_bar_range=median_bar_range,
            high_of_day=high_of_day,
            vwap=vwap,
            reasons=reasons,
            warnings=warnings
        )

        # Calculate Participation Score
        participation = self._calc_participation(
            current_volume=current_volume,
            avg_volume=avg_volume,
            float_shares=float_shares,
            day_volume=day_volume,
            consecutive_green=consecutive_green,
            reasons=reasons,
            warnings=warnings
        )

        # Calculate Liquidity Score
        liquidity = self._calc_liquidity(
            spread_pct=spread_pct,
            buy_pressure=buy_pressure,
            imbalance_ratio=imbalance_ratio,
            tape_signal=tape_signal,
            reasons=reasons,
            warnings=warnings
        )

        # Total score
        total_score = price_urgency.total + participation.total + liquidity.total

        # Grade
        grade = self._get_grade(total_score)

        # Tradeable check
        is_tradeable = total_score >= self.ENTRY_THRESHOLD

        # Ignition ready check (all components must meet minimums)
        ignition_ready = (
            total_score >= self.IGNITION_THRESHOLD and
            price_urgency.total >= self.MIN_PRICE_URGENCY and
            participation.total >= self.MIN_PARTICIPATION and
            liquidity.total >= self.MIN_LIQUIDITY
        )

        if ignition_ready:
            reasons.append("IGNITION READY - All components aligned")

        result = MomentumResult(
            symbol=symbol,
            timestamp=now,
            score=total_score,
            grade=grade,
            price_urgency=price_urgency,
            participation=participation,
            liquidity=liquidity,
            is_tradeable=is_tradeable,
            ignition_ready=ignition_ready,
            reasons=reasons,
            warnings=warnings
        )

        # Cache result
        self._cache[symbol] = result

        return result

    def _calc_price_urgency(self,
                            current_price: float,
                            prices_30s: List[float],
                            prices_60s: List[float],
                            prices_5m: List[float],
                            current_bar_range: float,
                            median_bar_range: float,
                            high_of_day: float,
                            vwap: float,
                            reasons: List[str],
                            warnings: List[str]) -> PriceUrgencyScore:
        """Calculate price urgency component"""
        score = PriceUrgencyScore()

        # ROC calculations
        if prices_30s and len(prices_30s) >= 2:
            score.roc_30s = ((current_price - prices_30s[0]) / prices_30s[0]) * 100
        if prices_60s and len(prices_60s) >= 2:
            score.roc_60s = ((current_price - prices_60s[0]) / prices_60s[0]) * 100
        if prices_5m and len(prices_5m) >= 2:
            score.roc_5m = ((current_price - prices_5m[0]) / prices_5m[0]) * 100

        # ROC score (0-15)
        max_roc = max(abs(score.roc_30s), abs(score.roc_60s) * 0.8, abs(score.roc_5m) * 0.5)
        if max_roc >= 1.0:
            score.roc_score = 15
            reasons.append(f"Strong ROC: {max_roc:.2f}%")
        elif max_roc >= 0.5:
            score.roc_score = 12
        elif max_roc >= 0.3:
            score.roc_score = 8
        elif max_roc >= 0.1:
            score.roc_score = 4
        else:
            score.roc_score = 0
            if max_roc < 0.05:
                warnings.append("Price not moving")

        # Range expansion (0-10)
        if median_bar_range > 0:
            score.range_expansion = current_bar_range / median_bar_range
            if score.range_expansion >= 3.0:
                score.range_score = 10
                reasons.append(f"Range expansion: {score.range_expansion:.1f}x")
            elif score.range_expansion >= 2.0:
                score.range_score = 8
            elif score.range_expansion >= 1.5:
                score.range_score = 5
            elif score.range_expansion >= 1.0:
                score.range_score = 2
            else:
                score.range_score = 0

        # HOD proximity/break (0-10)
        if high_of_day > 0 and current_price > 0:
            score.hod_distance_pct = ((high_of_day - current_price) / current_price) * 100
            score.hod_break = current_price >= high_of_day * 0.999  # Within 0.1% of HOD

            if score.hod_break:
                score.hod_score = 10
                reasons.append("HOD BREAK")
            elif score.hod_distance_pct <= 0.5:
                score.hod_score = 8
                reasons.append("Near HOD")
            elif score.hod_distance_pct <= 1.0:
                score.hod_score = 5
            elif score.hod_distance_pct <= 2.0:
                score.hod_score = 2
            else:
                score.hod_score = 0
                if score.hod_distance_pct > 5.0:
                    warnings.append(f"Far from HOD: {score.hod_distance_pct:.1f}%")

        # VWAP position (0-5)
        if vwap > 0:
            if current_price > vwap * 1.005:
                score.vwap_position = "ABOVE"
                score.vwap_score = 5
            elif current_price >= vwap * 0.995:
                score.vwap_position = "AT"
                score.vwap_score = 3
            else:
                score.vwap_position = "BELOW"
                score.vwap_score = 0
                warnings.append("Below VWAP")

        return score

    def _calc_participation(self,
                            current_volume: int,
                            avg_volume: int,
                            float_shares: int,
                            day_volume: int,
                            consecutive_green: int,
                            reasons: List[str],
                            warnings: List[str]) -> ParticipationScore:
        """Calculate participation component"""
        score = ParticipationScore()
        score.consecutive_green = consecutive_green

        # Volume surge (0-15)
        if avg_volume > 0:
            # Use per-minute average for surge calculation
            avg_per_minute = avg_volume / 390  # 6.5 hour trading day
            score.volume_surge = current_volume / max(avg_per_minute, 1)

            if score.volume_surge >= 10.0:
                score.surge_score = 15
                reasons.append(f"Volume surge: {score.volume_surge:.0f}x")
            elif score.volume_surge >= 5.0:
                score.surge_score = 12
            elif score.volume_surge >= 3.0:
                score.surge_score = 8
            elif score.volume_surge >= 2.0:
                score.surge_score = 4
            else:
                score.surge_score = 0
                if score.volume_surge < 1.0:
                    warnings.append("Low volume")

        # Relative volume for the day (0-10)
        if avg_volume > 0 and day_volume > 0:
            # Estimate what portion of day has passed (assume market hours)
            now = datetime.now()
            market_open = now.replace(hour=9, minute=30, second=0)
            minutes_elapsed = max(1, (now - market_open).total_seconds() / 60)
            expected_volume = avg_volume * (minutes_elapsed / 390)
            score.relative_volume = day_volume / max(expected_volume, 1)

            if score.relative_volume >= 5.0:
                score.rvol_score = 10
                reasons.append(f"RVol: {score.relative_volume:.1f}x")
            elif score.relative_volume >= 3.0:
                score.rvol_score = 8
            elif score.relative_volume >= 2.0:
                score.rvol_score = 5
            elif score.relative_volume >= 1.5:
                score.rvol_score = 2
            else:
                score.rvol_score = 0

        # Float rotation (0-10)
        if float_shares > 0 and day_volume > 0:
            score.float_rotation_pct = (day_volume / float_shares) * 100

            if score.float_rotation_pct >= 100:
                score.rotation_score = 10
                reasons.append(f"Float rotation: {score.float_rotation_pct:.0f}%")
            elif score.float_rotation_pct >= 50:
                score.rotation_score = 8
            elif score.float_rotation_pct >= 25:
                score.rotation_score = 5
            elif score.float_rotation_pct >= 10:
                score.rotation_score = 2
            else:
                score.rotation_score = 0

        return score

    def _calc_liquidity(self,
                        spread_pct: float,
                        buy_pressure: float,
                        imbalance_ratio: float,
                        tape_signal: str,
                        reasons: List[str],
                        warnings: List[str]) -> LiquidityScore:
        """Calculate liquidity component"""
        score = LiquidityScore()
        score.spread_pct = spread_pct
        score.buy_pressure = buy_pressure
        score.imbalance_ratio = imbalance_ratio
        score.tape_signal = tape_signal

        # Spread score (0-10) - lower spread = better
        if spread_pct <= 0.1:
            score.spread_score = 10
        elif spread_pct <= 0.25:
            score.spread_score = 8
        elif spread_pct <= 0.5:
            score.spread_score = 6
        elif spread_pct <= 1.0:
            score.spread_score = 3
        else:
            score.spread_score = 0
            warnings.append(f"Wide spread: {spread_pct:.2f}%")

        # Order flow score (0-10)
        if buy_pressure >= 0.70:
            score.flow_score = 10
            reasons.append(f"Strong buy pressure: {buy_pressure:.0%}")
        elif buy_pressure >= 0.60:
            score.flow_score = 8
        elif buy_pressure >= 0.55:
            score.flow_score = 5
        elif buy_pressure >= 0.50:
            score.flow_score = 2
        else:
            score.flow_score = 0
            warnings.append(f"Weak buy pressure: {buy_pressure:.0%}")

        # Tape score (0-5)
        if tape_signal == "BULLISH":
            score.tape_score = 5
            reasons.append("Bullish tape")
        elif tape_signal == "NEUTRAL":
            score.tape_score = 2
        else:
            score.tape_score = 0
            warnings.append("Bearish tape")

        return score

    def _get_grade(self, score: int) -> MomentumGrade:
        """Get grade from score"""
        if score >= 80:
            return MomentumGrade.A
        elif score >= 65:
            return MomentumGrade.B
        elif score >= 50:
            return MomentumGrade.C
        elif score >= 35:
            return MomentumGrade.D
        else:
            return MomentumGrade.F

    def get_cached(self, symbol: str) -> Optional[MomentumResult]:
        """Get cached result if still valid"""
        if symbol in self._cache:
            result = self._cache[symbol]
            age = (datetime.now() - result.timestamp).total_seconds()
            if age < self._cache_ttl:
                return result
        return None


# Singleton instance
_scorer: Optional[MomentumScorer] = None


def get_momentum_scorer() -> MomentumScorer:
    """Get singleton scorer instance"""
    global _scorer
    if _scorer is None:
        _scorer = MomentumScorer()
    return _scorer


def calculate_momentum_score(symbol: str, **kwargs) -> MomentumResult:
    """Quick calculation helper"""
    return get_momentum_scorer().calculate(symbol, **kwargs)


if __name__ == "__main__":
    # Test with sample data
    scorer = MomentumScorer()

    result = scorer.calculate(
        symbol="TEST",
        current_price=5.50,
        prices_30s=[5.40, 5.45, 5.50],
        prices_60s=[5.30, 5.40, 5.45, 5.50],
        prices_5m=[5.00, 5.20, 5.35, 5.45, 5.50],
        current_bar_range=0.15,
        median_bar_range=0.05,
        high_of_day=5.52,
        vwap=5.20,
        current_volume=50000,
        avg_volume=500000,
        float_shares=2000000,
        day_volume=400000,
        spread_pct=0.3,
        buy_pressure=0.65,
        imbalance_ratio=1.5,
        tape_signal="BULLISH",
        consecutive_green=3
    )

    print(f"\n{'='*60}")
    print(f"MOMENTUM SCORE: {result.symbol}")
    print(f"{'='*60}")
    print(f"Total Score: {result.score}/100 (Grade {result.grade.value})")
    print(f"\nPrice/Urgency: {result.price_urgency.total}/40")
    print(f"  ROC Score: {result.price_urgency.roc_score}/15")
    print(f"  Range Score: {result.price_urgency.range_score}/10")
    print(f"  HOD Score: {result.price_urgency.hod_score}/10")
    print(f"  VWAP Score: {result.price_urgency.vwap_score}/5")
    print(f"\nParticipation: {result.participation.total}/35")
    print(f"  Surge Score: {result.participation.surge_score}/15")
    print(f"  RVol Score: {result.participation.rvol_score}/10")
    print(f"  Rotation Score: {result.participation.rotation_score}/10")
    print(f"\nLiquidity: {result.liquidity.total}/25")
    print(f"  Spread Score: {result.liquidity.spread_score}/10")
    print(f"  Flow Score: {result.liquidity.flow_score}/10")
    print(f"  Tape Score: {result.liquidity.tape_score}/5")
    print(f"\nTradeable: {result.is_tradeable}")
    print(f"Ignition Ready: {result.ignition_ready}")
    print(f"\nReasons: {result.reasons}")
    print(f"Warnings: {result.warnings}")
