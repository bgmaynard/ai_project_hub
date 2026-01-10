"""
Market Condition Evaluator (Task H)
===================================
Lightweight evaluator that computes market conditions for profile selection.

Metrics computed:
- Market breadth score (advance/decline ratio)
- Small-cap participation ratio
- Gap continuation rate
- Chronos regime distribution

Output: market_condition (WEAK | MIXED | STRONG), confidence
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from enum import Enum
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Market condition classification"""
    WEAK = "WEAK"
    MIXED = "MIXED"
    STRONG = "STRONG"


@dataclass
class MarketMetrics:
    """Raw market metrics before classification"""
    # Breadth metrics
    advancers: int = 0
    decliners: int = 0
    unchanged: int = 0
    advance_decline_ratio: float = 1.0

    # Small-cap participation
    small_cap_gainers: int = 0
    small_cap_total: int = 0
    small_cap_participation: float = 0.0

    # Gap continuation
    gaps_attempted: int = 0
    gaps_continued: int = 0
    gap_continuation_rate: float = 0.0

    # Chronos regime distribution
    trending_up_count: int = 0
    trending_down_count: int = 0
    ranging_count: int = 0
    volatile_count: int = 0
    bullish_regime_ratio: float = 0.0

    # Momentum health
    symbols_with_momentum: int = 0
    total_symbols_tracked: int = 0
    momentum_participation: float = 0.0

    # Volume health
    avg_relative_volume: float = 1.0
    high_volume_count: int = 0


@dataclass
class ConditionResult:
    """Result of market condition evaluation"""
    market_condition: MarketCondition
    confidence: float  # 0.0 - 1.0
    metrics: MarketMetrics
    reasons: List[str]
    timestamp: datetime
    evaluation_duration_ms: float

    def to_dict(self) -> Dict:
        return {
            "market_condition": self.market_condition.value,
            "confidence": self.confidence,
            "metrics": {
                "advance_decline_ratio": self.metrics.advance_decline_ratio,
                "small_cap_participation": self.metrics.small_cap_participation,
                "gap_continuation_rate": self.metrics.gap_continuation_rate,
                "bullish_regime_ratio": self.metrics.bullish_regime_ratio,
                "momentum_participation": self.metrics.momentum_participation,
                "avg_relative_volume": self.metrics.avg_relative_volume
            },
            "reasons": self.reasons,
            "timestamp": self.timestamp.isoformat(),
            "evaluation_duration_ms": self.evaluation_duration_ms
        }


class MarketConditionEvaluator:
    """
    Evaluates market conditions to guide profile selection.

    This is intentionally lightweight - we don't want to slow down trading decisions.
    """

    def __init__(self):
        self.last_evaluation: Optional[ConditionResult] = None
        self.evaluation_history: List[ConditionResult] = []
        self.cache_duration_seconds = 60  # Cache results for 1 minute

        # Thresholds for classification
        self.thresholds = {
            # STRONG market thresholds
            "strong_ad_ratio": 1.5,  # Advancers 1.5x decliners
            "strong_participation": 0.60,  # 60% small caps participating
            "strong_gap_rate": 0.60,  # 60% gaps continue
            "strong_bullish_ratio": 0.60,  # 60% symbols in bullish regime

            # WEAK market thresholds
            "weak_ad_ratio": 0.67,  # Decliners 1.5x advancers
            "weak_participation": 0.30,  # Only 30% participating
            "weak_gap_rate": 0.30,  # Only 30% gaps continue
            "weak_bullish_ratio": 0.30  # Only 30% in bullish regime
        }

    async def evaluate(self, force: bool = False) -> ConditionResult:
        """
        Evaluate current market conditions.

        Uses cached result if recent enough, unless force=True.
        """
        start_time = datetime.now()

        # Check cache
        if not force and self.last_evaluation:
            age = (datetime.now() - self.last_evaluation.timestamp).total_seconds()
            if age < self.cache_duration_seconds:
                return self.last_evaluation

        # Gather metrics
        metrics = await self._gather_metrics()

        # Classify condition
        condition, confidence, reasons = self._classify(metrics)

        # Build result
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        result = ConditionResult(
            market_condition=condition,
            confidence=confidence,
            metrics=metrics,
            reasons=reasons,
            timestamp=datetime.now(),
            evaluation_duration_ms=duration_ms
        )

        # Cache result
        self.last_evaluation = result
        self.evaluation_history.append(result)

        # Keep history bounded
        if len(self.evaluation_history) > 100:
            self.evaluation_history = self.evaluation_history[-100:]

        logger.info(
            f"Market Condition: {condition.value} (confidence={confidence:.0%}) - "
            f"{', '.join(reasons[:3])}"
        )

        return result

    async def _gather_metrics(self) -> MarketMetrics:
        """Gather market metrics from various sources"""
        metrics = MarketMetrics()

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                # Get advance/decline data
                await self._get_breadth_metrics(client, metrics)

                # Get small-cap participation
                await self._get_participation_metrics(client, metrics)

                # Get gap continuation rate
                await self._get_gap_metrics(client, metrics)

                # Get Chronos regime distribution
                await self._get_regime_metrics(client, metrics)

                # Get momentum health
                await self._get_momentum_metrics(client, metrics)

        except Exception as e:
            logger.warning(f"Error gathering market metrics: {e}")

        return metrics

    async def _get_breadth_metrics(self, client, metrics: MarketMetrics):
        """Get market breadth (advance/decline)"""
        try:
            # Try to get from funnel metrics or scanner results
            resp = await client.get(
                "http://localhost:9100/api/scanner/finviz/status",
                timeout=3.0
            )
            if resp.status_code == 200:
                data = resp.json()
                # Count gainers vs losers from recent scans
                results = data.get('last_scan_results', [])
                for r in results:
                    change = r.get('change_percent', 0)
                    if change > 0.5:
                        metrics.advancers += 1
                    elif change < -0.5:
                        metrics.decliners += 1
                    else:
                        metrics.unchanged += 1

            # Calculate ratio
            if metrics.decliners > 0:
                metrics.advance_decline_ratio = metrics.advancers / metrics.decliners
            elif metrics.advancers > 0:
                metrics.advance_decline_ratio = 2.0  # Strong if no decliners

        except Exception as e:
            logger.debug(f"Breadth metrics error: {e}")

    async def _get_participation_metrics(self, client, metrics: MarketMetrics):
        """Get small-cap participation ratio"""
        try:
            # Get HOD scanner results (small-cap focused)
            resp = await client.get(
                "http://localhost:9100/api/scanner/hod/status",
                timeout=3.0
            )
            if resp.status_code == 200:
                data = resp.json()
                tracked = data.get('symbols_tracked', [])
                metrics.small_cap_total = len(tracked)

                # Count those with positive momentum
                for sym_data in data.get('symbol_data', {}).values():
                    if sym_data.get('change_percent', 0) > 2.0:
                        metrics.small_cap_gainers += 1

            if metrics.small_cap_total > 0:
                metrics.small_cap_participation = metrics.small_cap_gainers / metrics.small_cap_total

        except Exception as e:
            logger.debug(f"Participation metrics error: {e}")

    async def _get_gap_metrics(self, client, metrics: MarketMetrics):
        """Get gap continuation rate"""
        try:
            # Get gap grader results
            resp = await client.get(
                "http://localhost:9100/api/warrior/gap/status",
                timeout=3.0
            )
            if resp.status_code == 200:
                data = resp.json()
                grades = data.get('grade_distribution', {})

                # Count graded gaps
                metrics.gaps_attempted = sum(grades.values()) if grades else 0

                # A and B grades typically indicate continuation
                metrics.gaps_continued = grades.get('A', 0) + grades.get('B', 0)

            if metrics.gaps_attempted > 0:
                metrics.gap_continuation_rate = metrics.gaps_continued / metrics.gaps_attempted

        except Exception as e:
            logger.debug(f"Gap metrics error: {e}")

    async def _get_regime_metrics(self, client, metrics: MarketMetrics):
        """Get Chronos regime distribution"""
        try:
            # Get momentum states from validation API
            resp = await client.get(
                "http://localhost:9100/api/validation/momentum/states",
                timeout=3.0
            )
            if resp.status_code == 200:
                data = resp.json()
                states = data.get('states', {})

                for symbol, state_info in states.items():
                    regime = state_info.get('macro_regime', 'UNKNOWN')
                    if regime == 'TRENDING_UP':
                        metrics.trending_up_count += 1
                    elif regime == 'TRENDING_DOWN':
                        metrics.trending_down_count += 1
                    elif regime == 'RANGING':
                        metrics.ranging_count += 1
                    elif regime == 'VOLATILE':
                        metrics.volatile_count += 1

            total_regimes = (
                metrics.trending_up_count +
                metrics.trending_down_count +
                metrics.ranging_count +
                metrics.volatile_count
            )

            if total_regimes > 0:
                # Bullish = trending up + ranging (neutral is ok)
                bullish = metrics.trending_up_count + (metrics.ranging_count * 0.5)
                metrics.bullish_regime_ratio = bullish / total_regimes

        except Exception as e:
            logger.debug(f"Regime metrics error: {e}")

    async def _get_momentum_metrics(self, client, metrics: MarketMetrics):
        """Get momentum participation metrics"""
        try:
            resp = await client.get(
                "http://localhost:9100/api/validation/momentum/states",
                timeout=3.0
            )
            if resp.status_code == 200:
                data = resp.json()
                states = data.get('states', {})
                metrics.total_symbols_tracked = len(states)

                for symbol, state_info in states.items():
                    state = state_info.get('state', 'DEAD')
                    if state in ['IGNITING', 'ACTIVE', 'CONFIRMED']:
                        metrics.symbols_with_momentum += 1

            if metrics.total_symbols_tracked > 0:
                metrics.momentum_participation = (
                    metrics.symbols_with_momentum / metrics.total_symbols_tracked
                )

        except Exception as e:
            logger.debug(f"Momentum metrics error: {e}")

    def _classify(self, metrics: MarketMetrics) -> Tuple[MarketCondition, float, List[str]]:
        """
        Classify market condition based on metrics.

        Returns (condition, confidence, reasons)
        """
        reasons = []
        strong_signals = 0
        weak_signals = 0
        total_signals = 0

        # Evaluate breadth
        if metrics.advancers > 0 or metrics.decliners > 0:
            total_signals += 1
            if metrics.advance_decline_ratio >= self.thresholds["strong_ad_ratio"]:
                strong_signals += 1
                reasons.append(f"Strong breadth (A/D={metrics.advance_decline_ratio:.2f})")
            elif metrics.advance_decline_ratio <= self.thresholds["weak_ad_ratio"]:
                weak_signals += 1
                reasons.append(f"Weak breadth (A/D={metrics.advance_decline_ratio:.2f})")

        # Evaluate small-cap participation
        if metrics.small_cap_total >= 5:
            total_signals += 1
            if metrics.small_cap_participation >= self.thresholds["strong_participation"]:
                strong_signals += 1
                reasons.append(f"High small-cap participation ({metrics.small_cap_participation:.0%})")
            elif metrics.small_cap_participation <= self.thresholds["weak_participation"]:
                weak_signals += 1
                reasons.append(f"Low small-cap participation ({metrics.small_cap_participation:.0%})")

        # Evaluate gap continuation
        if metrics.gaps_attempted >= 3:
            total_signals += 1
            if metrics.gap_continuation_rate >= self.thresholds["strong_gap_rate"]:
                strong_signals += 1
                reasons.append(f"High gap continuation ({metrics.gap_continuation_rate:.0%})")
            elif metrics.gap_continuation_rate <= self.thresholds["weak_gap_rate"]:
                weak_signals += 1
                reasons.append(f"Low gap continuation ({metrics.gap_continuation_rate:.0%})")

        # Evaluate Chronos regimes
        total_regimes = (
            metrics.trending_up_count +
            metrics.trending_down_count +
            metrics.ranging_count +
            metrics.volatile_count
        )
        if total_regimes >= 5:
            total_signals += 1
            if metrics.bullish_regime_ratio >= self.thresholds["strong_bullish_ratio"]:
                strong_signals += 1
                reasons.append(f"Bullish regimes dominant ({metrics.bullish_regime_ratio:.0%})")
            elif metrics.bullish_regime_ratio <= self.thresholds["weak_bullish_ratio"]:
                weak_signals += 1
                reasons.append(f"Bearish regimes dominant ({metrics.bullish_regime_ratio:.0%})")

        # Evaluate momentum participation
        if metrics.total_symbols_tracked >= 5:
            total_signals += 1
            if metrics.momentum_participation >= 0.40:
                strong_signals += 1
                reasons.append(f"High momentum ({metrics.momentum_participation:.0%} active)")
            elif metrics.momentum_participation <= 0.10:
                weak_signals += 1
                reasons.append(f"Low momentum ({metrics.momentum_participation:.0%} active)")

        # Determine condition
        if total_signals == 0:
            # No data yet, default to MIXED
            return MarketCondition.MIXED, 0.5, ["Insufficient data for evaluation"]

        strong_ratio = strong_signals / total_signals
        weak_ratio = weak_signals / total_signals

        if strong_ratio >= 0.6:
            condition = MarketCondition.STRONG
            confidence = 0.5 + (strong_ratio * 0.5)
        elif weak_ratio >= 0.6:
            condition = MarketCondition.WEAK
            confidence = 0.5 + (weak_ratio * 0.5)
        else:
            condition = MarketCondition.MIXED
            confidence = 0.5 + abs(strong_ratio - weak_ratio) * 0.3

        # Add summary reason
        if not reasons:
            reasons.append(f"Mixed signals ({strong_signals} strong, {weak_signals} weak)")

        return condition, min(confidence, 1.0), reasons

    def get_status(self) -> Dict:
        """Get evaluator status"""
        return {
            "last_evaluation": self.last_evaluation.to_dict() if self.last_evaluation else None,
            "cache_duration_seconds": self.cache_duration_seconds,
            "thresholds": self.thresholds,
            "evaluation_count": len(self.evaluation_history)
        }


# Singleton instance
_evaluator: Optional[MarketConditionEvaluator] = None


def get_market_evaluator() -> MarketConditionEvaluator:
    """Get the singleton market condition evaluator"""
    global _evaluator
    if _evaluator is None:
        _evaluator = MarketConditionEvaluator()
    return _evaluator


async def evaluate_market_condition(force: bool = False) -> ConditionResult:
    """Convenience function to evaluate market condition"""
    return await get_market_evaluator().evaluate(force)
