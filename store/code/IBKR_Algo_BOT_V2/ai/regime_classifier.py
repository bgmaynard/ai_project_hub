"""
Regime Classifier Module
Classifies market sessions as: TRENDING, RANGING, VOLATILE, or QUIET
Uses rolling statistics, price slope, and volatility metrics.

Part of the Next-Gen AI Logic Blueprint.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime classifications"""

    TRENDING_UP = "trending_up"  # Strong upward momentum
    TRENDING_DOWN = "trending_down"  # Strong downward momentum
    RANGING = "ranging"  # Sideways, mean-reverting
    VOLATILE = "volatile"  # High volatility, choppy
    QUIET = "quiet"  # Low volatility, low volume
    UNKNOWN = "unknown"


class MarketSession(str, Enum):
    """Time-based market sessions"""

    PRE_MARKET = "pre_market"  # 4:00 AM - 9:30 AM ET
    MARKET_OPEN = "market_open"  # 9:30 AM - 10:30 AM ET (most volatile)
    MIDDAY = "midday"  # 10:30 AM - 2:00 PM ET (often choppy)
    POWER_HOUR = "power_hour"  # 3:00 PM - 4:00 PM ET (trend moves)
    AFTER_HOURS = "after_hours"  # 4:00 PM - 8:00 PM ET
    CLOSED = "closed"


@dataclass
class RegimeAnalysis:
    """Result of regime classification"""

    regime: MarketRegime
    session: MarketSession
    confidence: float  # 0-1 confidence in classification
    trend_strength: float  # -1 (down) to +1 (up)
    volatility_percentile: float  # 0-100, relative to recent history
    volume_percentile: float  # 0-100, relative to recent history
    recommendation: str  # Trading recommendation
    details: Dict


class RegimeClassifier:
    """
    Classifies market regime using multiple indicators:
    - Price trend (slope of moving average)
    - Volatility (ATR, standard deviation)
    - Volume patterns
    - Time of day
    """

    def __init__(self):
        # Lookback periods
        self.short_period = 10  # Short-term trend
        self.medium_period = 20  # Medium-term trend
        self.long_period = 50  # Long-term trend
        self.volatility_period = 14  # ATR period

        # Thresholds
        self.trend_threshold = 0.002  # 0.2% slope = trending
        self.volatility_high = 75  # Percentile for "volatile"
        self.volatility_low = 25  # Percentile for "quiet"
        self.volume_high = 70  # Percentile for high volume

        # Historical data for percentile calculations
        self.volatility_history: List[float] = []
        self.volume_history: List[float] = []
        self.max_history = 100

        logger.info("RegimeClassifier initialized")

    def get_market_session(self, dt: datetime = None) -> MarketSession:
        """Determine current market session based on time"""
        if dt is None:
            dt = datetime.now()

        t = dt.time()

        # Pre-market: 4:00 AM - 9:30 AM
        if time(4, 0) <= t < time(9, 30):
            return MarketSession.PRE_MARKET

        # Market open: 9:30 AM - 10:30 AM
        if time(9, 30) <= t < time(10, 30):
            return MarketSession.MARKET_OPEN

        # Midday: 10:30 AM - 3:00 PM
        if time(10, 30) <= t < time(15, 0):
            return MarketSession.MIDDAY

        # Power hour: 3:00 PM - 4:00 PM
        if time(15, 0) <= t < time(16, 0):
            return MarketSession.POWER_HOUR

        # After hours: 4:00 PM - 8:00 PM
        if time(16, 0) <= t < time(20, 0):
            return MarketSession.AFTER_HOURS

        return MarketSession.CLOSED

    def calculate_trend_strength(self, prices: List[float]) -> float:
        """
        Calculate trend strength from -1 (strong down) to +1 (strong up)
        Uses linear regression slope normalized by price
        """
        if len(prices) < 5:
            return 0.0

        prices = np.array(prices[-self.medium_period :])
        n = len(prices)

        # Linear regression
        x = np.arange(n)
        slope = np.polyfit(x, prices, 1)[0]

        # Normalize by average price
        avg_price = np.mean(prices)
        if avg_price > 0:
            normalized_slope = slope / avg_price
        else:
            normalized_slope = 0

        # Scale to -1 to +1 (clip extremes)
        trend_strength = np.clip(normalized_slope * 100, -1, 1)

        return float(trend_strength)

    def calculate_volatility(
        self, highs: List[float], lows: List[float], closes: List[float]
    ) -> Tuple[float, float]:
        """
        Calculate ATR-based volatility and return (atr, percentile)
        """
        if len(highs) < self.volatility_period:
            return 0.0, 50.0

        # True Range calculation
        true_ranges = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i - 1])
            low_close = abs(lows[i] - closes[i - 1])
            tr = max(high_low, high_close, low_close)
            true_ranges.append(tr)

        # ATR
        atr = np.mean(true_ranges[-self.volatility_period :])

        # Normalize by price
        avg_price = np.mean(closes[-self.volatility_period :])
        if avg_price > 0:
            normalized_atr = (atr / avg_price) * 100  # As percentage
        else:
            normalized_atr = 0

        # Update history and calculate percentile
        self.volatility_history.append(normalized_atr)
        if len(self.volatility_history) > self.max_history:
            self.volatility_history.pop(0)

        if len(self.volatility_history) > 10:
            percentile = (
                np.searchsorted(np.sort(self.volatility_history), normalized_atr)
                / len(self.volatility_history)
            ) * 100
        else:
            percentile = 50.0

        return float(normalized_atr), float(percentile)

    def calculate_volume_percentile(self, volumes: List[float]) -> float:
        """Calculate volume percentile relative to recent history"""
        if not volumes:
            return 50.0

        current_volume = volumes[-1]

        # Update history
        self.volume_history.extend(volumes[-5:])  # Add recent volumes
        if len(self.volume_history) > self.max_history:
            self.volume_history = self.volume_history[-self.max_history :]

        if len(self.volume_history) > 10:
            percentile = (
                np.searchsorted(np.sort(self.volume_history), current_volume)
                / len(self.volume_history)
            ) * 100
        else:
            percentile = 50.0

        return float(percentile)

    def classify(
        self,
        prices: List[float],
        highs: List[float] = None,
        lows: List[float] = None,
        volumes: List[float] = None,
        timestamp: datetime = None,
    ) -> RegimeAnalysis:
        """
        Main classification method.

        Args:
            prices: List of closing prices (most recent last)
            highs: List of high prices (optional, for ATR)
            lows: List of low prices (optional, for ATR)
            volumes: List of volumes (optional)
            timestamp: Current timestamp (optional)

        Returns:
            RegimeAnalysis with classification results
        """
        if len(prices) < 10:
            return RegimeAnalysis(
                regime=MarketRegime.UNKNOWN,
                session=self.get_market_session(timestamp),
                confidence=0.0,
                trend_strength=0.0,
                volatility_percentile=50.0,
                volume_percentile=50.0,
                recommendation="Insufficient data",
                details={},
            )

        # Get market session
        session = self.get_market_session(timestamp)

        # Calculate trend strength
        trend_strength = self.calculate_trend_strength(prices)

        # Calculate volatility (use prices if highs/lows not provided)
        if highs is None or lows is None:
            highs = prices
            lows = prices
        atr, volatility_pct = self.calculate_volatility(highs, lows, prices)

        # Calculate volume percentile
        volume_pct = self.calculate_volume_percentile(volumes) if volumes else 50.0

        # Classify regime
        regime, confidence = self._determine_regime(
            trend_strength, volatility_pct, volume_pct
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            regime, session, trend_strength, volatility_pct
        )

        return RegimeAnalysis(
            regime=regime,
            session=session,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_percentile=volatility_pct,
            volume_percentile=volume_pct,
            recommendation=recommendation,
            details={
                "atr_percent": atr,
                "prices_analyzed": len(prices),
                "short_ma": (
                    float(np.mean(prices[-self.short_period :]))
                    if len(prices) >= self.short_period
                    else 0
                ),
                "long_ma": (
                    float(np.mean(prices[-self.long_period :]))
                    if len(prices) >= self.long_period
                    else 0
                ),
            },
        )

    def _determine_regime(
        self, trend: float, volatility_pct: float, volume_pct: float
    ) -> Tuple[MarketRegime, float]:
        """Determine regime based on indicators"""
        confidence = 0.5  # Base confidence

        # High volatility = volatile regime
        if volatility_pct >= self.volatility_high:
            if abs(trend) >= 0.5:
                # High vol + strong trend = trending (but risky)
                regime = (
                    MarketRegime.TRENDING_UP
                    if trend > 0
                    else MarketRegime.TRENDING_DOWN
                )
                confidence = 0.6
            else:
                regime = MarketRegime.VOLATILE
                confidence = 0.7

        # Low volatility = quiet or ranging
        elif volatility_pct <= self.volatility_low:
            if abs(trend) < 0.2:
                regime = MarketRegime.QUIET
                confidence = 0.7
            else:
                regime = MarketRegime.RANGING
                confidence = 0.6

        # Medium volatility = check trend
        else:
            if trend >= 0.4:
                regime = MarketRegime.TRENDING_UP
                confidence = 0.5 + (trend * 0.3)  # Higher trend = higher confidence
            elif trend <= -0.4:
                regime = MarketRegime.TRENDING_DOWN
                confidence = 0.5 + (abs(trend) * 0.3)
            else:
                regime = MarketRegime.RANGING
                confidence = 0.6

        # Volume confirmation
        if volume_pct >= self.volume_high:
            confidence = min(1.0, confidence + 0.1)
        elif volume_pct <= 30:
            confidence = max(0.3, confidence - 0.1)

        return regime, round(confidence, 2)

    def _generate_recommendation(
        self,
        regime: MarketRegime,
        session: MarketSession,
        trend: float,
        volatility_pct: float,
    ) -> str:
        """Generate trading recommendation based on regime and session"""

        recommendations = {
            MarketRegime.TRENDING_UP: "Favor long entries on pullbacks. Use momentum strategies.",
            MarketRegime.TRENDING_DOWN: "Favor short entries on rallies or stay flat. Avoid catching knives.",
            MarketRegime.RANGING: "Mean-reversion strategies. Buy support, sell resistance.",
            MarketRegime.VOLATILE: "Reduce position size. Wide stops. Quick profits.",
            MarketRegime.QUIET: "Low opportunity. Consider waiting for breakout.",
            MarketRegime.UNKNOWN: "Insufficient data. Wait for clarity.",
        }

        base_rec = recommendations.get(regime, "No recommendation")

        # Session-specific adjustments
        if session == MarketSession.MARKET_OPEN:
            base_rec += " [OPEN: High volatility expected, wait for direction]"
        elif session == MarketSession.MIDDAY:
            base_rec += " [MIDDAY: Watch for chop, tighten stops]"
        elif session == MarketSession.POWER_HOUR:
            base_rec += " [POWER HOUR: Trend moves likely, follow momentum]"
        elif session == MarketSession.PRE_MARKET:
            base_rec += " [PRE-MARKET: Low liquidity, wider spreads]"
        elif session == MarketSession.AFTER_HOURS:
            base_rec += " [AFTER-HOURS: Low liquidity, avoid large positions]"

        return base_rec

    def get_regime_adjustments(self, regime: MarketRegime) -> Dict:
        """
        Get trading parameter adjustments based on regime.
        Used by bot_manager to tune behavior.
        """
        adjustments = {
            MarketRegime.TRENDING_UP: {
                "confidence_multiplier": 0.9,  # Lower threshold (more trades)
                "position_size_multiplier": 1.2,  # Larger positions
                "trailing_stop_multiplier": 1.2,  # Wider stops (let it run)
                "take_profit_multiplier": 1.5,  # Higher targets
                "prefer_breakouts": True,
            },
            MarketRegime.TRENDING_DOWN: {
                "confidence_multiplier": 1.3,  # Higher threshold (fewer trades)
                "position_size_multiplier": 0.7,  # Smaller positions
                "trailing_stop_multiplier": 0.8,  # Tighter stops
                "take_profit_multiplier": 0.8,  # Lower targets
                "prefer_breakouts": False,
            },
            MarketRegime.RANGING: {
                "confidence_multiplier": 1.1,
                "position_size_multiplier": 0.9,
                "trailing_stop_multiplier": 0.9,
                "take_profit_multiplier": 0.7,  # Quick profits
                "prefer_breakouts": False,
            },
            MarketRegime.VOLATILE: {
                "confidence_multiplier": 1.4,  # Very selective
                "position_size_multiplier": 0.5,  # Half size
                "trailing_stop_multiplier": 1.5,  # Wide stops
                "take_profit_multiplier": 0.6,  # Quick exits
                "prefer_breakouts": False,
            },
            MarketRegime.QUIET: {
                "confidence_multiplier": 1.2,
                "position_size_multiplier": 0.8,
                "trailing_stop_multiplier": 0.8,
                "take_profit_multiplier": 0.5,  # Small targets
                "prefer_breakouts": True,  # Wait for breakout
            },
            MarketRegime.UNKNOWN: {
                "confidence_multiplier": 1.5,  # Very conservative
                "position_size_multiplier": 0.5,
                "trailing_stop_multiplier": 1.0,
                "take_profit_multiplier": 1.0,
                "prefer_breakouts": False,
            },
        }

        return adjustments.get(regime, adjustments[MarketRegime.UNKNOWN])


# Singleton instance
_regime_classifier: Optional[RegimeClassifier] = None


def get_regime_classifier() -> RegimeClassifier:
    """Get or create the regime classifier singleton"""
    global _regime_classifier
    if _regime_classifier is None:
        _regime_classifier = RegimeClassifier()
    return _regime_classifier
