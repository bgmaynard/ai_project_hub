"""
Warrior Trading Market Regime Detector

Identifies current market conditions and suggests strategy adjustments using Claude AI.

Market Regimes:
- TRENDING_BULL: Strong upward momentum
- TRENDING_BEAR: Downward pressure
- CHOPPY: Range-bound, low conviction
- HIGH_VOLATILITY: Large swings, news-driven
- LOW_VOLATILITY: Tight ranges

Features:
- Real-time regime detection
- Automatic parameter adjustments
- Regime change notifications
- Historical regime tracking
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from claude_integration import ClaudeRequest, get_claude_integration

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Market regime types"""

    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    CHOPPY = "CHOPPY"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNKNOWN = "UNKNOWN"


@dataclass
class MarketIndicators:
    """Market condition indicators"""

    spy_price: float
    spy_change_percent: float
    spy_20sma: Optional[float] = None
    spy_50sma: Optional[float] = None
    qqq_price: Optional[float] = None
    qqq_change_percent: Optional[float] = None
    vix: Optional[float] = None
    vix_change: Optional[float] = None
    advance_decline_ratio: Optional[float] = None
    volume_ratio: Optional[float] = None  # vs 20-day average
    gap_up_count: int = 0
    gap_down_count: int = 0
    timestamp: datetime = datetime.now()


@dataclass
class RegimeAdjustments:
    """Strategy adjustments for a regime"""

    position_size_multiplier: float
    min_confidence_threshold: float
    max_daily_trades: int
    preferred_patterns: List[str]
    stop_multiplier: float = 1.0
    halt_on_consecutive_losses: int = 3
    notes: str = ""


@dataclass
class RegimeDetection:
    """Detected market regime"""

    regime: RegimeType
    confidence: float
    reasoning: str
    indicators: MarketIndicators
    adjustments: RegimeAdjustments
    warnings: List[str]
    detection_time: datetime = datetime.now()


# Pre-defined regime adjustments
REGIME_DEFAULTS = {
    RegimeType.TRENDING_BULL: RegimeAdjustments(
        position_size_multiplier=1.2,
        min_confidence_threshold=0.60,
        max_daily_trades=6,
        preferred_patterns=["BULL_FLAG", "HOD_BREAKOUT", "MICRO_PULLBACK"],
        stop_multiplier=1.0,
        halt_on_consecutive_losses=3,
        notes="Strong trend - increase size, focus on momentum patterns",
    ),
    RegimeType.TRENDING_BEAR: RegimeAdjustments(
        position_size_multiplier=0.7,
        min_confidence_threshold=0.75,
        max_daily_trades=3,
        preferred_patterns=["SHORT_ONLY"],  # If shorting enabled
        stop_multiplier=1.1,
        halt_on_consecutive_losses=2,
        notes="Downtrend - reduce size, be selective, consider sitting out",
    ),
    RegimeType.CHOPPY: RegimeAdjustments(
        position_size_multiplier=0.7,
        min_confidence_threshold=0.75,
        max_daily_trades=3,
        preferred_patterns=["WHOLE_DOLLAR_BREAKOUT"],
        stop_multiplier=1.2,
        halt_on_consecutive_losses=2,
        notes="Choppy market - reduce size, widen stops, be patient",
    ),
    RegimeType.HIGH_VOLATILITY: RegimeAdjustments(
        position_size_multiplier=0.6,
        min_confidence_threshold=0.80,
        max_daily_trades=4,
        preferred_patterns=["BULL_FLAG"],  # Stick to high-probability setups
        stop_multiplier=1.3,
        halt_on_consecutive_losses=2,
        notes="High volatility - reduce size significantly, wider stops",
    ),
    RegimeType.LOW_VOLATILITY: RegimeAdjustments(
        position_size_multiplier=0.8,
        min_confidence_threshold=0.70,
        max_daily_trades=4,
        preferred_patterns=["WHOLE_DOLLAR_BREAKOUT", "MICRO_PULLBACK"],
        stop_multiplier=0.9,
        halt_on_consecutive_losses=3,
        notes="Low volatility - normal size, tighter stops",
    ),
}


class MarketRegimeDetector:
    """
    Detects market regime and suggests strategy adjustments

    Uses Claude AI to analyze market conditions and classify regime
    """

    def __init__(self):
        """Initialize market regime detector"""
        self.claude = get_claude_integration()
        self.current_regime: Optional[RegimeDetection] = None
        self.regime_history: List[RegimeDetection] = []
        self.last_detection_time: Optional[datetime] = None
        logger.info("Market regime detector initialized")

    def detect_regime(
        self, indicators: MarketIndicators, use_ai: bool = True
    ) -> RegimeDetection:
        """
        Detect current market regime

        Args:
            indicators: Current market indicators
            use_ai: Whether to use Claude AI (vs rule-based detection)

        Returns:
            RegimeDetection object
        """
        if use_ai:
            detection = self._detect_regime_with_ai(indicators)
        else:
            detection = self._detect_regime_rule_based(indicators)

        # Store detection
        self.current_regime = detection
        self.regime_history.append(detection)
        self.last_detection_time = datetime.now()

        # Keep history manageable (last 100 detections)
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]

        logger.info(
            f"Regime detected: {detection.regime.value} "
            f"(confidence: {detection.confidence:.2f})"
        )

        return detection

    def _detect_regime_rule_based(
        self, indicators: MarketIndicators
    ) -> RegimeDetection:
        """
        Rule-based regime detection (fast, no AI cost)

        Args:
            indicators: Market indicators

        Returns:
            RegimeDetection object
        """
        regime = RegimeType.UNKNOWN
        confidence = 0.5
        reasoning = "Rule-based detection"
        warnings = []

        # High volatility check (VIX > 25)
        if indicators.vix and indicators.vix > 25:
            regime = RegimeType.HIGH_VOLATILITY
            confidence = 0.8
            reasoning = f"VIX at {indicators.vix:.1f} indicates high volatility"
            warnings.append("High volatility - reduce position sizes")

        # Low volatility check (VIX < 12)
        elif indicators.vix and indicators.vix < 12:
            regime = RegimeType.LOW_VOLATILITY
            confidence = 0.7
            reasoning = f"VIX at {indicators.vix:.1f} indicates low volatility"

        # Trending bull (SPY > 20SMA and positive change)
        elif (
            indicators.spy_20sma
            and indicators.spy_price > indicators.spy_20sma
            and indicators.spy_change_percent > 0.5
        ):
            regime = RegimeType.TRENDING_BULL
            confidence = 0.75
            reasoning = "SPY above 20SMA with positive momentum"

        # Trending bear (SPY < 20SMA and negative change)
        elif (
            indicators.spy_20sma
            and indicators.spy_price < indicators.spy_20sma
            and indicators.spy_change_percent < -0.5
        ):
            regime = RegimeType.TRENDING_BEAR
            confidence = 0.75
            reasoning = "SPY below 20SMA with negative momentum"
            warnings.append("Bearish market - consider reducing activity")

        # Choppy (small moves, mixed signals)
        elif abs(indicators.spy_change_percent) < 0.3:
            regime = RegimeType.CHOPPY
            confidence = 0.65
            reasoning = "Small SPY movement indicates choppy conditions"
            warnings.append("Choppy market - be selective with trades")

        # Default to unknown
        else:
            warnings.append("Unable to confidently classify regime")

        # Get default adjustments for regime
        adjustments = REGIME_DEFAULTS.get(
            regime,
            RegimeAdjustments(
                position_size_multiplier=1.0,
                min_confidence_threshold=0.70,
                max_daily_trades=5,
                preferred_patterns=["ALL"],
                notes="Default settings",
            ),
        )

        return RegimeDetection(
            regime=regime,
            confidence=confidence,
            reasoning=reasoning,
            indicators=indicators,
            adjustments=adjustments,
            warnings=warnings,
        )

    def _detect_regime_with_ai(self, indicators: MarketIndicators) -> RegimeDetection:
        """
        AI-powered regime detection using Claude

        Args:
            indicators: Market indicators

        Returns:
            RegimeDetection object
        """
        # Build prompt with market data
        prompt = f"""Analyze current market conditions and classify the trading regime:

MARKET INDICATORS:
- SPY: ${indicators.spy_price:.2f} ({indicators.spy_change_percent:+.2f}%)
- SPY 20SMA: ${indicators.spy_20sma:.2f if indicators.spy_20sma else 'N/A'}
- SPY 50SMA: ${indicators.spy_50sma:.2f if indicators.spy_50sma else 'N/A'}
- QQQ: ${indicators.qqq_price:.2f if indicators.qqq_price else 'N/A'} ({indicators.qqq_change_percent:+.2f if indicators.qqq_change_percent else 'N/A'}%)
- VIX: {indicators.vix:.2f if indicators.vix else 'N/A'} (change: {indicators.vix_change:+.2f if indicators.vix_change else 'N/A'})
- Advance/Decline: {indicators.advance_decline_ratio:.2f if indicators.advance_decline_ratio else 'N/A'}
- Volume Ratio: {indicators.volume_ratio:.2f if indicators.volume_ratio else 'N/A'}x
- Gaps: {indicators.gap_up_count} up, {indicators.gap_down_count} down

REGIME CLASSIFICATIONS:
1. TRENDING_BULL - Strong upward momentum, healthy consolidations
2. TRENDING_BEAR - Downward pressure, distribution
3. CHOPPY - Range-bound, whipsaw, low conviction
4. HIGH_VOLATILITY - Large swings, news-driven, VIX elevated
5. LOW_VOLATILITY - Tight ranges, quiet market, VIX low

Analyze and provide JSON response:
{{
  "regime": "REGIME_TYPE",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentences explaining classification",
  "recommended_adjustments": {{
    "position_size_multiplier": 0.6-1.2,
    "min_confidence_threshold": 0.60-0.80,
    "max_daily_trades": 3-6,
    "preferred_patterns": ["PATTERN1", "PATTERN2"],
    "stop_multiplier": 0.9-1.3
  }},
  "warnings": ["specific risks or cautions"],
  "key_levels": ["important SPY levels to watch"]
}}

Focus on actionable adjustments for day trading the Warrior Trading patterns."""

        request = ClaudeRequest(
            request_type="market_regime",
            prompt=prompt,
            max_tokens=1024,
            temperature=0.3,  # Lower temperature for more consistent classification
            system_prompt="You are a market structure expert analyzing intraday conditions for day trading. Provide clear regime classification and specific trading adjustments.",
        )

        response = self.claude.request(request)

        if not response.success:
            logger.error(f"AI regime detection failed: {response.error}")
            # Fallback to rule-based
            return self._detect_regime_rule_based(indicators)

        # Parse AI response
        try:
            data = json.loads(response.content)

            # Map regime string to enum
            regime_str = data["regime"]
            try:
                regime = RegimeType[regime_str]
            except KeyError:
                logger.warning(
                    f"Unknown regime type: {regime_str}, defaulting to UNKNOWN"
                )
                regime = RegimeType.UNKNOWN

            # Create adjustments from AI recommendations
            adj_data = data.get("recommended_adjustments", {})
            adjustments = RegimeAdjustments(
                position_size_multiplier=adj_data.get("position_size_multiplier", 1.0),
                min_confidence_threshold=adj_data.get("min_confidence_threshold", 0.70),
                max_daily_trades=adj_data.get("max_daily_trades", 5),
                preferred_patterns=adj_data.get("preferred_patterns", []),
                stop_multiplier=adj_data.get("stop_multiplier", 1.0),
                halt_on_consecutive_losses=3,
                notes=data.get("reasoning", ""),
            )

            return RegimeDetection(
                regime=regime,
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", ""),
                indicators=indicators,
                adjustments=adjustments,
                warnings=data.get("warnings", []),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse AI regime response: {e}")
            logger.debug(f"Raw response: {response.content}")
            # Fallback to rule-based
            return self._detect_regime_rule_based(indicators)

    def get_current_regime(self) -> Optional[RegimeDetection]:
        """Get the most recent regime detection"""
        return self.current_regime

    def should_redetect(self, interval_minutes: int = 15) -> bool:
        """
        Check if it's time to re-detect regime

        Args:
            interval_minutes: Minutes between detections

        Returns:
            True if detection should be run
        """
        if not self.last_detection_time:
            return True

        elapsed = datetime.now() - self.last_detection_time
        return elapsed.total_seconds() > (interval_minutes * 60)

    def get_regime_history(self, hours: int = 4) -> List[RegimeDetection]:
        """
        Get regime detection history

        Args:
            hours: Number of hours of history to retrieve

        Returns:
            List of recent RegimeDetection objects
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        return [d for d in self.regime_history if d.detection_time > cutoff]

    def apply_regime_adjustments(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply regime adjustments to configuration

        Args:
            base_config: Base strategy configuration

        Returns:
            Adjusted configuration
        """
        if not self.current_regime:
            return base_config

        adjusted = base_config.copy()
        adj = self.current_regime.adjustments

        # Apply multipliers and adjustments
        if "default_position_size" in adjusted:
            adjusted["default_position_size"] *= adj.position_size_multiplier

        if "min_confidence" in adjusted:
            adjusted["min_confidence"] = adj.min_confidence_threshold

        if "max_daily_trades" in adjusted:
            adjusted["max_daily_trades"] = adj.max_daily_trades

        if "stop_loss_multiplier" in adjusted:
            adjusted["stop_loss_multiplier"] = adj.stop_multiplier

        adjusted["regime_adjusted"] = True
        adjusted["regime_type"] = self.current_regime.regime.value
        adjusted["regime_confidence"] = self.current_regime.confidence

        logger.info(
            f"Applied {self.current_regime.regime.value} adjustments "
            f"(size: {adj.position_size_multiplier}x, "
            f"min_conf: {adj.min_confidence_threshold})"
        )

        return adjusted

    def to_dict(self) -> Dict[str, Any]:
        """Convert current regime to dictionary"""
        if not self.current_regime:
            return {"regime": "UNKNOWN", "detected": False}

        return {
            "regime": self.current_regime.regime.value,
            "confidence": self.current_regime.confidence,
            "reasoning": self.current_regime.reasoning,
            "warnings": self.current_regime.warnings,
            "adjustments": {
                "position_size_multiplier": self.current_regime.adjustments.position_size_multiplier,
                "min_confidence": self.current_regime.adjustments.min_confidence_threshold,
                "max_daily_trades": self.current_regime.adjustments.max_daily_trades,
                "preferred_patterns": self.current_regime.adjustments.preferred_patterns,
                "stop_multiplier": self.current_regime.adjustments.stop_multiplier,
            },
            "indicators": {
                "spy_price": self.current_regime.indicators.spy_price,
                "spy_change_percent": self.current_regime.indicators.spy_change_percent,
                "vix": self.current_regime.indicators.vix,
            },
            "detection_time": self.current_regime.detection_time.isoformat(),
            "detected": True,
        }


# Global instance
_regime_detector: Optional[MarketRegimeDetector] = None


def get_regime_detector() -> MarketRegimeDetector:
    """Get or create global MarketRegimeDetector instance"""
    global _regime_detector
    if _regime_detector is None:
        _regime_detector = MarketRegimeDetector()
    return _regime_detector


if __name__ == "__main__":
    # Test the detector
    detector = MarketRegimeDetector()

    print("Market Regime Detector Test")
    print("=" * 50)

    # Create test indicators
    indicators = MarketIndicators(
        spy_price=450.25,
        spy_change_percent=1.2,
        spy_20sma=448.50,
        spy_50sma=445.00,
        qqq_price=385.75,
        qqq_change_percent=1.5,
        vix=15.5,
        vix_change=-2.1,
        advance_decline_ratio=1.8,
        volume_ratio=1.2,
        gap_up_count=45,
        gap_down_count=15,
    )

    # Test rule-based detection
    print("\nRule-Based Detection:")
    detection = detector.detect_regime(indicators, use_ai=False)
    print(f"Regime: {detection.regime.value}")
    print(f"Confidence: {detection.confidence:.2f}")
    print(f"Reasoning: {detection.reasoning}")
    print(f"Adjustments:")
    print(f"  Position Size: {detection.adjustments.position_size_multiplier}x")
    print(f"  Min Confidence: {detection.adjustments.min_confidence_threshold}")
    print(f"  Max Trades: {detection.adjustments.max_daily_trades}")
    print(
        f"  Preferred Patterns: {', '.join(detection.adjustments.preferred_patterns)}"
    )

    # Test AI detection (if API key configured)
    if detector.claude.client:
        print("\n\nAI-Powered Detection:")
        detection_ai = detector.detect_regime(indicators, use_ai=True)
        print(f"Regime: {detection_ai.regime.value}")
        print(f"Confidence: {detection_ai.confidence:.2f}")
        print(f"Reasoning: {detection_ai.reasoning}")
        if detection_ai.warnings:
            print(f"Warnings: {', '.join(detection_ai.warnings)}")

    print("\n" + "=" * 50)
    print("Test complete!")
