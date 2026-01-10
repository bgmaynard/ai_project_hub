"""
DERO Market Context Engine

Detects market regime (TREND/CHOP/NEWS/DEAD) from SPY/QQQ price data.
This is READ-ONLY, descriptive (not predictive), and does not affect trading.

Regime Categories:
- TREND: Clear directional move with momentum
- CHOP: Range-bound, indecisive price action
- NEWS: High volatility event-driven moves
- DEAD: Low volume, no clear direction
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime categories"""
    TREND = "TREND"
    CHOP = "CHOP"
    NEWS = "NEWS"
    DEAD = "DEAD"
    UNKNOWN = "UNKNOWN"


class MarketContextEngine:
    """
    Market awareness engine for DERO.

    Analyzes SPY/QQQ data to determine market regime.
    All operations are read-only and descriptive.
    """

    def __init__(self):
        self._cached_regime: Optional[Dict[str, Any]] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)

    def calculate_trend_slope(self, prices: List[float]) -> float:
        """
        Calculate trend slope using linear regression.

        Returns normalized slope (% change per bar).
        """
        if len(prices) < 2:
            return 0.0

        try:
            x = np.arange(len(prices))
            y = np.array(prices)

            # Linear regression
            slope, _ = np.polyfit(x, y, 1)

            # Normalize as percentage of mean price
            mean_price = np.mean(y)
            if mean_price > 0:
                return (slope / mean_price) * 100
            return 0.0
        except Exception as e:
            logger.warning(f"Error calculating trend slope: {e}")
            return 0.0

    def calculate_realized_volatility(self, prices: List[float]) -> float:
        """
        Calculate realized volatility (annualized).

        Returns volatility as decimal (0.20 = 20%).
        """
        if len(prices) < 2:
            return 0.0

        try:
            returns = np.diff(np.log(prices))
            if len(returns) == 0:
                return 0.0

            # Standard deviation of returns
            std = np.std(returns)

            # Annualize (assuming 1-minute bars, 390 minutes per day, 252 trading days)
            # For 5-minute bars: 78 bars per day
            bars_per_day = 78  # Assume 5-minute bars
            annualized = std * np.sqrt(bars_per_day * 252)

            return float(annualized)
        except Exception as e:
            logger.warning(f"Error calculating volatility: {e}")
            return 0.0

    def calculate_range_expansion(self, highs: List[float], lows: List[float]) -> float:
        """
        Calculate range expansion ratio.

        Compares recent range to average range.
        >1.5 suggests expansion (news/breakout), <0.5 suggests contraction (dead).
        """
        if len(highs) < 10 or len(lows) < 10:
            return 1.0

        try:
            # Calculate ATR-like measure
            ranges = [h - l for h, l in zip(highs, lows)]

            # Recent range (last 5 bars) vs average range
            recent_range = np.mean(ranges[-5:])
            avg_range = np.mean(ranges)

            if avg_range > 0:
                return float(recent_range / avg_range)
            return 1.0
        except Exception as e:
            logger.warning(f"Error calculating range expansion: {e}")
            return 1.0

    def calculate_choppiness(self, highs: List[float], lows: List[float], closes: List[float]) -> float:
        """
        Calculate Choppiness Index (0-100).

        High values (>61.8) = choppy/consolidating
        Low values (<38.2) = trending
        """
        if len(closes) < 14:
            return 50.0  # Neutral default

        try:
            period = min(14, len(closes))

            # True Range sum
            tr_sum = 0
            for i in range(1, period):
                high = highs[-(period-i)]
                low = lows[-(period-i)]
                prev_close = closes[-(period-i+1)]

                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                tr_sum += tr

            # Highest high - lowest low over period
            highest = max(highs[-period:])
            lowest = min(lows[-period:])
            hl_range = highest - lowest

            if hl_range > 0 and tr_sum > 0:
                chop = 100 * np.log10(tr_sum / hl_range) / np.log10(period)
                return float(np.clip(chop, 0, 100))
            return 50.0
        except Exception as e:
            logger.warning(f"Error calculating choppiness: {e}")
            return 50.0

    def detect_regime(
        self,
        closes: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        volumes: Optional[List[float]] = None
    ) -> Tuple[MarketRegime, float, Dict[str, Any]]:
        """
        Detect market regime from price data.

        Args:
            closes: List of close prices
            highs: Optional list of high prices
            lows: Optional list of low prices
            volumes: Optional list of volumes

        Returns:
            Tuple of (regime, confidence, features)
        """
        if not closes or len(closes) < 5:
            return MarketRegime.UNKNOWN, 0.0, {}

        # Use closes for highs/lows if not provided
        if highs is None:
            highs = closes
        if lows is None:
            lows = closes

        # Calculate features
        trend_slope = self.calculate_trend_slope(closes)
        rv = self.calculate_realized_volatility(closes)
        range_exp = self.calculate_range_expansion(highs, lows)
        choppiness = self.calculate_choppiness(highs, lows, closes)

        # Volume analysis
        vol_ratio = 1.0
        if volumes and len(volumes) >= 10:
            recent_vol = np.mean(volumes[-5:])
            avg_vol = np.mean(volumes)
            if avg_vol > 0:
                vol_ratio = recent_vol / avg_vol

        features = {
            "trend_slope": round(trend_slope, 4),
            "realized_volatility": round(rv, 4),
            "range_expansion": round(range_exp, 2),
            "choppiness_index": round(choppiness, 1),
            "volume_ratio": round(vol_ratio, 2),
        }

        # Regime classification logic
        regime = MarketRegime.UNKNOWN
        confidence = 0.5

        # NEWS: High volatility + range expansion
        if rv > 0.30 and range_exp > 1.5:
            regime = MarketRegime.NEWS
            confidence = min(0.9, 0.5 + (rv - 0.30) + (range_exp - 1.5) * 0.2)

        # TREND: Clear direction + low choppiness
        elif abs(trend_slope) > 0.05 and choppiness < 45:
            regime = MarketRegime.TREND
            confidence = min(0.9, 0.5 + abs(trend_slope) * 2 + (45 - choppiness) / 100)

        # DEAD: Low volatility + low volume
        elif rv < 0.10 and vol_ratio < 0.7:
            regime = MarketRegime.DEAD
            confidence = min(0.9, 0.5 + (0.10 - rv) * 5 + (0.7 - vol_ratio))

        # CHOP: High choppiness, moderate volatility
        elif choppiness > 55:
            regime = MarketRegime.CHOP
            confidence = min(0.9, 0.5 + (choppiness - 55) / 50)

        # Default to CHOP with lower confidence
        else:
            regime = MarketRegime.CHOP
            confidence = 0.4

        return regime, round(confidence, 2), features

    def build_context(
        self,
        spy_data: Optional[Dict[str, List[float]]] = None,
        qqq_data: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Build market context from SPY/QQQ data.

        Args:
            spy_data: Dict with 'close', 'high', 'low', 'volume' lists
            qqq_data: Dict with 'close', 'high', 'low', 'volume' lists

        Returns:
            Market context dictionary
        """
        result = {
            "regime": MarketRegime.UNKNOWN.value,
            "confidence": 0.0,
            "features": {},
            "spy_regime": None,
            "qqq_regime": None,
            "timestamp": datetime.now().isoformat(),
        }

        regimes = []
        confidences = []

        # Analyze SPY
        if spy_data and spy_data.get("close"):
            regime, conf, features = self.detect_regime(
                spy_data.get("close", []),
                spy_data.get("high"),
                spy_data.get("low"),
                spy_data.get("volume")
            )
            result["spy_regime"] = {
                "regime": regime.value,
                "confidence": conf,
                "features": features
            }
            regimes.append(regime)
            confidences.append(conf)

        # Analyze QQQ
        if qqq_data and qqq_data.get("close"):
            regime, conf, features = self.detect_regime(
                qqq_data.get("close", []),
                qqq_data.get("high"),
                qqq_data.get("low"),
                qqq_data.get("volume")
            )
            result["qqq_regime"] = {
                "regime": regime.value,
                "confidence": conf,
                "features": features
            }
            regimes.append(regime)
            confidences.append(conf)

        # Combined regime (prefer SPY, or average)
        if regimes:
            # If both agree, use that regime with higher confidence
            if len(regimes) == 2 and regimes[0] == regimes[1]:
                result["regime"] = regimes[0].value
                result["confidence"] = round(max(confidences), 2)
            elif regimes:
                # Use SPY regime primarily
                result["regime"] = regimes[0].value
                result["confidence"] = round(confidences[0], 2)

            # Combine features
            if result.get("spy_regime"):
                result["features"] = result["spy_regime"]["features"]

        return result

    def get_regime_description(self, regime: str) -> str:
        """Get human-readable description of a regime"""
        descriptions = {
            "TREND": "Clear directional move with momentum - favor trend-following",
            "CHOP": "Range-bound, indecisive price action - reduce position sizes",
            "NEWS": "High volatility event-driven - use caution, wider stops",
            "DEAD": "Low volume, no clear direction - avoid trading",
            "UNKNOWN": "Insufficient data to determine regime",
        }
        return descriptions.get(regime, "Unknown regime")


# Singleton instance
_market_context_engine: Optional[MarketContextEngine] = None


def get_market_context_engine() -> MarketContextEngine:
    """Get or create the singleton MarketContextEngine instance"""
    global _market_context_engine
    if _market_context_engine is None:
        _market_context_engine = MarketContextEngine()
    return _market_context_engine
