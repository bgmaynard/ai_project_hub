"""
Phase Evaluation Engine (Task M)
================================
Lightweight evaluator that determines current market phase based on:
- Current time
- Market breadth
- Small-cap follow-through
- Chronos regime confidence

Emits: MARKET_PHASE_SELECTED
Lock phase for minimum duration (15-30 min).
"""

import logging
import asyncio
import pytz
from datetime import datetime, time, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class PhaseEvaluationResult:
    """Result of phase evaluation"""
    phase: str
    confidence: float  # 0.0 - 1.0
    reasons: List[str]
    metrics: Dict
    timestamp: datetime
    evaluation_duration_ms: float

    def to_dict(self) -> Dict:
        return {
            "phase": self.phase,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "evaluation_duration_ms": self.evaluation_duration_ms
        }


class PhaseEvaluator:
    """
    Evaluates market conditions to determine appropriate phase.

    Primary factor: Time of day
    Secondary factors: Market health metrics that can override/confirm
    """

    def __init__(self):
        self.last_evaluation: Optional[PhaseEvaluationResult] = None
        self.evaluation_history: List[PhaseEvaluationResult] = []
        self.min_lock_minutes = 15
        self.cache_duration_seconds = 60

    async def evaluate(self, force: bool = False) -> PhaseEvaluationResult:
        """
        Evaluate current market phase.

        Uses cached result if recent enough, unless force=True.
        """
        start_time = datetime.now()

        # Check cache
        if not force and self.last_evaluation:
            age = (datetime.now() - self.last_evaluation.timestamp).total_seconds()
            if age < self.cache_duration_seconds:
                return self.last_evaluation

        # Get time-based phase
        time_phase = self._get_time_based_phase()

        # Gather market metrics
        metrics = await self._gather_metrics()

        # Possibly override phase based on metrics
        final_phase, confidence, reasons = self._determine_phase(time_phase, metrics)

        # Build result
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        result = PhaseEvaluationResult(
            phase=final_phase,
            confidence=confidence,
            reasons=reasons,
            metrics=metrics,
            timestamp=datetime.now(),
            evaluation_duration_ms=duration_ms
        )

        # Cache result
        self.last_evaluation = result
        self.evaluation_history.append(result)

        if len(self.evaluation_history) > 100:
            self.evaluation_history = self.evaluation_history[-100:]

        logger.info(
            f"Phase Evaluation: {final_phase} (confidence={confidence:.0%}) - "
            f"{', '.join(reasons[:2])}"
        )

        return result

    def _get_time_based_phase(self) -> str:
        """Get phase based purely on time of day"""
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz).time()

        # Define time windows
        if time(4, 0) <= now_et < time(9, 30):
            return "PRE_MARKET"
        elif time(9, 30) <= now_et < time(9, 45):
            return "OPEN_IGNITION"
        elif time(9, 45) <= now_et < time(11, 30):
            return "STRUCTURED_MOMENTUM"
        elif time(11, 30) <= now_et < time(14, 0):
            return "MIDDAY_COMPRESSION"
        elif time(14, 0) <= now_et < time(16, 0):
            return "POWER_HOUR"
        elif time(16, 0) <= now_et < time(20, 0):
            return "AFTER_HOURS"
        else:
            return "CLOSED"

    async def _gather_metrics(self) -> Dict:
        """Gather market metrics for phase evaluation"""
        metrics = {
            "market_breadth": 0.5,
            "small_cap_follow_through": 0.5,
            "chronos_bullish_ratio": 0.5,
            "chronos_confidence": 0.5,
            "volume_participation": 0.5,
            "momentum_count": 0,
            "total_tracked": 0
        }

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                # Get momentum states for Chronos data
                try:
                    resp = await client.get(
                        "http://localhost:9100/api/validation/momentum/states",
                        timeout=3.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        states = data.get('states', {})

                        bullish = 0
                        total = len(states)
                        confident = 0

                        for symbol, state_info in states.items():
                            regime = state_info.get('macro_regime', 'UNKNOWN')
                            if regime in ['TRENDING_UP', 'RANGING']:
                                bullish += 1

                            conf = state_info.get('micro_confidence', 0)
                            if conf >= 0.6:
                                confident += 1

                            if state_info.get('state') in ['IGNITING', 'ACTIVE', 'CONFIRMED']:
                                metrics["momentum_count"] += 1

                        metrics["total_tracked"] = total
                        if total > 0:
                            metrics["chronos_bullish_ratio"] = bullish / total
                            metrics["chronos_confidence"] = confident / total
                except Exception as e:
                    logger.debug(f"Momentum metrics error: {e}")

                # Get market breadth from scanner
                try:
                    resp = await client.get(
                        "http://localhost:9100/api/baseline/market-condition",
                        timeout=3.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        mkt_metrics = data.get('metrics', {})
                        metrics["market_breadth"] = mkt_metrics.get('advance_decline_ratio', 1.0) / 2.0
                        metrics["small_cap_follow_through"] = mkt_metrics.get('small_cap_participation', 0.5)
                except Exception as e:
                    logger.debug(f"Market condition error: {e}")

        except Exception as e:
            logger.warning(f"Error gathering phase metrics: {e}")

        return metrics

    def _determine_phase(self, time_phase: str, metrics: Dict) -> Tuple[str, float, List[str]]:
        """
        Determine final phase based on time and metrics.

        Time is primary, but extreme conditions can override.
        """
        reasons = [f"Time-based: {time_phase}"]
        confidence = 0.7  # Base confidence from time

        # Extract metrics
        breadth = metrics.get("market_breadth", 0.5)
        follow_through = metrics.get("small_cap_follow_through", 0.5)
        bullish_ratio = metrics.get("chronos_bullish_ratio", 0.5)
        chronos_conf = metrics.get("chronos_confidence", 0.5)
        momentum_count = metrics.get("momentum_count", 0)

        # Check for extreme conditions that might override
        final_phase = time_phase

        # MIDDAY can be upgraded if market is very strong
        if time_phase == "MIDDAY_COMPRESSION":
            if breadth > 0.7 and follow_through > 0.6 and momentum_count >= 5:
                final_phase = "STRUCTURED_MOMENTUM"
                confidence = 0.65
                reasons.append("Strong momentum overrides midday compression")
            elif breadth < 0.3:
                confidence = 0.85
                reasons.append("Weak breadth confirms compression phase")

        # STRUCTURED_MOMENTUM can be downgraded if market is weak
        elif time_phase == "STRUCTURED_MOMENTUM":
            if breadth < 0.4 and bullish_ratio < 0.3:
                final_phase = "MIDDAY_COMPRESSION"
                confidence = 0.6
                reasons.append("Weak conditions - treating as compression")
            elif breadth > 0.6 and bullish_ratio > 0.6:
                confidence = 0.9
                reasons.append("Strong momentum confirmation")

        # OPEN_IGNITION is always time-based (too volatile to override)
        elif time_phase == "OPEN_IGNITION":
            confidence = 0.95
            reasons.append("Opening volatility - time-based only")

        # POWER_HOUR can be adjusted based on trend
        elif time_phase == "POWER_HOUR":
            if momentum_count >= 3 and bullish_ratio > 0.5:
                confidence = 0.85
                reasons.append("Momentum supports power hour plays")
            elif momentum_count == 0 and breadth < 0.4:
                final_phase = "MIDDAY_COMPRESSION"
                confidence = 0.55
                reasons.append("No momentum - extending compression mode")

        # Add confidence adjustments based on Chronos
        if chronos_conf > 0.7:
            confidence = min(confidence + 0.1, 1.0)
            reasons.append(f"High Chronos confidence ({chronos_conf:.0%})")
        elif chronos_conf < 0.3:
            confidence = max(confidence - 0.1, 0.3)
            reasons.append(f"Low Chronos confidence ({chronos_conf:.0%})")

        return final_phase, confidence, reasons

    async def evaluate_and_apply(self, force: bool = False) -> Dict:
        """
        Evaluate phase and apply to phase manager.

        Returns action result.
        """
        from ai.market_phases import get_phase_manager, MarketPhase

        result = await self.evaluate(force)

        manager = get_phase_manager()
        old_phase = manager.current_phase

        # Convert to enum
        try:
            new_phase = MarketPhase(result.phase)
        except ValueError:
            logger.error(f"Invalid phase: {result.phase}")
            return {"action": "ERROR", "error": f"Invalid phase: {result.phase}"}

        # Build reason string
        reason = f"Evaluation: {', '.join(result.reasons[:2])} (conf={result.confidence:.0%})"

        # Apply phase
        success = manager.set_phase(new_phase, reason, self.min_lock_minutes)

        return {
            "action": "PHASE_SELECTED" if success else "PHASE_LOCKED",
            "old_phase": old_phase.value if old_phase else None,
            "new_phase": new_phase.value,
            "phase_changed": success and (old_phase != new_phase if old_phase else True),
            "confidence": result.confidence,
            "reasons": result.reasons,
            "metrics": result.metrics,
            "lock_minutes": self.min_lock_minutes,
            "timestamp": datetime.now().isoformat()
        }

    def get_status(self) -> Dict:
        """Get evaluator status"""
        return {
            "last_evaluation": self.last_evaluation.to_dict() if self.last_evaluation else None,
            "min_lock_minutes": self.min_lock_minutes,
            "cache_duration_seconds": self.cache_duration_seconds,
            "evaluation_count": len(self.evaluation_history),
            "recent_evaluations": [
                e.to_dict() for e in self.evaluation_history[-5:]
            ] if self.evaluation_history else []
        }


# Singleton instance
_evaluator: Optional[PhaseEvaluator] = None


def get_phase_evaluator() -> PhaseEvaluator:
    """Get the singleton phase evaluator"""
    global _evaluator
    if _evaluator is None:
        _evaluator = PhaseEvaluator()
    return _evaluator


async def evaluate_phase(force: bool = False) -> PhaseEvaluationResult:
    """Convenience function to evaluate phase"""
    return await get_phase_evaluator().evaluate(force)


async def evaluate_and_apply_phase(force: bool = False) -> Dict:
    """Evaluate phase and apply to manager"""
    return await get_phase_evaluator().evaluate_and_apply(force)
