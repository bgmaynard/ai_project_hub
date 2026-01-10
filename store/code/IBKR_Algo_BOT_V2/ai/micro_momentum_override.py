"""
Micro-Momentum Override
=======================
Allows trades even in bad MACRO regime if MICRO momentum is strong.

Problem: REGIME_MISMATCH can veto ALL entries in TRENDING_DOWN even
when micro momentum is real and tradeable.

Solution: Allow controlled override with strict conditions and caps.

Conditions for override (ALL must be true):
1. ATS state is ACTIVE (or strongest micro trigger)
2. Chronos micro confidence >= threshold
3. Volume and/or float thresholds satisfied
4. Not exhausted/invalidated
5. Within rate limit (1 per 10 min)

Overrides are logged with GATING_MICRO_OVERRIDE_APPLIED event.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json
from pathlib import Path
import pytz

logger = logging.getLogger(__name__)


@dataclass
class OverrideConfig:
    """Configuration for micro-momentum override"""
    enabled: bool = True
    max_per_10min: int = 1
    size_multiplier: float = 0.5  # Reduce position size for override trades

    # Thresholds
    min_micro_confidence: float = 0.65
    min_rel_vol: float = 2.0
    min_volume: int = 500000
    max_float_millions: float = 50.0

    # ATS states that qualify for override
    qualifying_ats_states: List[str] = None

    def __post_init__(self):
        if self.qualifying_ats_states is None:
            self.qualifying_ats_states = ["ACTIVE", "CONFIRMED", "IGNITING"]


@dataclass
class OverrideEvent:
    """Record of a micro-momentum override"""
    timestamp: str
    symbol: str
    macro_regime: str
    micro_regime: str
    micro_confidence: float
    ats_state: str
    reasons: List[str]
    position_size_mult: float


class MicroMomentumOverride:
    """
    Manages micro-momentum override decisions.

    Allows trades in bad macro regime when micro momentum is strong.
    """

    def __init__(self, config: OverrideConfig = None):
        self.config = config or OverrideConfig()
        self.et_tz = pytz.timezone('US/Eastern')

        # Track recent overrides for rate limiting
        self.recent_overrides: List[OverrideEvent] = []
        self.max_history = 100

        # Stats
        self.override_requests = 0
        self.overrides_granted = 0
        self.overrides_denied = 0
        self.denial_reasons: Dict[str, int] = {}

        logger.info(f"MicroMomentumOverride initialized (enabled: {self.config.enabled})")

    def _count_recent_overrides(self, minutes: int = 10) -> int:
        """Count overrides in the last N minutes"""
        cutoff = datetime.now(self.et_tz) - timedelta(minutes=minutes)

        count = 0
        for event in self.recent_overrides:
            event_time = datetime.fromisoformat(event.timestamp)
            if event_time.tzinfo is None:
                event_time = self.et_tz.localize(event_time)
            if event_time > cutoff:
                count += 1

        return count

    def check_override(
        self,
        symbol: str,
        macro_regime: str,
        micro_regime: str,
        micro_confidence: float,
        ats_state: str,
        current_volume: int = 0,
        float_millions: float = 0,
        rel_vol: float = 0
    ) -> tuple[bool, str, float]:
        """
        Check if micro-momentum override is allowed.

        Args:
            symbol: Stock symbol
            macro_regime: Current macro (SPY) regime
            micro_regime: Current micro (symbol-specific) regime
            micro_confidence: Chronos confidence for micro regime
            ats_state: Current ATS state for the symbol
            current_volume: Today's volume
            float_millions: Float in millions
            rel_vol: Relative volume

        Returns:
            (allowed, reason, size_multiplier)
        """
        self.override_requests += 1

        # Not enabled
        if not self.config.enabled:
            return (False, "Override disabled", 1.0)

        reasons = []

        # Check 1: ATS state qualifies
        if ats_state not in self.config.qualifying_ats_states:
            self.overrides_denied += 1
            self.denial_reasons["ATS_STATE_NOT_QUALIFYING"] = self.denial_reasons.get("ATS_STATE_NOT_QUALIFYING", 0) + 1
            return (False, f"ATS state '{ats_state}' not in qualifying states {self.config.qualifying_ats_states}", 1.0)
        reasons.append(f"ATS state {ats_state} qualifies")

        # Check 2: Micro confidence meets threshold
        if micro_confidence < self.config.min_micro_confidence:
            self.overrides_denied += 1
            self.denial_reasons["MICRO_CONFIDENCE_LOW"] = self.denial_reasons.get("MICRO_CONFIDENCE_LOW", 0) + 1
            return (False, f"Micro confidence {micro_confidence:.0%} < {self.config.min_micro_confidence:.0%}", 1.0)
        reasons.append(f"Micro confidence {micro_confidence:.0%} OK")

        # Check 3: Volume threshold (if provided)
        if current_volume > 0 and current_volume < self.config.min_volume:
            self.overrides_denied += 1
            self.denial_reasons["VOLUME_LOW"] = self.denial_reasons.get("VOLUME_LOW", 0) + 1
            return (False, f"Volume {current_volume:,} < {self.config.min_volume:,}", 1.0)
        if current_volume > 0:
            reasons.append(f"Volume {current_volume:,} OK")

        # Check 4: RelVol threshold (if provided)
        if rel_vol > 0 and rel_vol < self.config.min_rel_vol:
            self.overrides_denied += 1
            self.denial_reasons["RELVOL_LOW"] = self.denial_reasons.get("RELVOL_LOW", 0) + 1
            return (False, f"RelVol {rel_vol:.1f}x < {self.config.min_rel_vol:.1f}x", 1.0)
        if rel_vol > 0:
            reasons.append(f"RelVol {rel_vol:.1f}x OK")

        # Check 5: Float threshold (if provided)
        if float_millions > 0 and float_millions > self.config.max_float_millions:
            self.overrides_denied += 1
            self.denial_reasons["FLOAT_TOO_HIGH"] = self.denial_reasons.get("FLOAT_TOO_HIGH", 0) + 1
            return (False, f"Float {float_millions:.1f}M > {self.config.max_float_millions:.1f}M", 1.0)
        if float_millions > 0:
            reasons.append(f"Float {float_millions:.1f}M OK")

        # Check 6: Rate limit
        recent_count = self._count_recent_overrides(10)
        if recent_count >= self.config.max_per_10min:
            self.overrides_denied += 1
            self.denial_reasons["RATE_LIMITED"] = self.denial_reasons.get("RATE_LIMITED", 0) + 1
            return (False, f"Rate limited: {recent_count} overrides in last 10 min (max {self.config.max_per_10min})", 1.0)
        reasons.append(f"Rate limit OK ({recent_count}/{self.config.max_per_10min})")

        # All checks passed - grant override
        self.overrides_granted += 1

        # Record the override
        event = OverrideEvent(
            timestamp=datetime.now(self.et_tz).isoformat(),
            symbol=symbol,
            macro_regime=macro_regime,
            micro_regime=micro_regime,
            micro_confidence=micro_confidence,
            ats_state=ats_state,
            reasons=reasons,
            position_size_mult=self.config.size_multiplier
        )
        self.recent_overrides.append(event)
        if len(self.recent_overrides) > self.max_history:
            self.recent_overrides = self.recent_overrides[-self.max_history:]

        logger.warning(
            f"GATING_MICRO_OVERRIDE_APPLIED: {symbol} | "
            f"macro={macro_regime}, micro={micro_regime} ({micro_confidence:.0%}), "
            f"ats={ats_state}, size_mult={self.config.size_multiplier}"
        )

        return (
            True,
            f"Override granted: {', '.join(reasons)}",
            self.config.size_multiplier
        )

    def get_status(self) -> Dict[str, Any]:
        """Get override status"""
        return {
            "enabled": self.config.enabled,
            "config": {
                "max_per_10min": self.config.max_per_10min,
                "size_multiplier": self.config.size_multiplier,
                "min_micro_confidence": self.config.min_micro_confidence,
                "min_rel_vol": self.config.min_rel_vol,
                "min_volume": self.config.min_volume,
                "max_float_millions": self.config.max_float_millions,
                "qualifying_ats_states": self.config.qualifying_ats_states
            },
            "stats": {
                "override_requests": self.override_requests,
                "overrides_granted": self.overrides_granted,
                "overrides_denied": self.overrides_denied,
                "grant_rate": (self.overrides_granted / self.override_requests * 100) if self.override_requests > 0 else 0,
                "denial_reasons": dict(self.denial_reasons),
                "recent_overrides_10min": self._count_recent_overrides(10)
            },
            "recent_overrides": [
                {
                    "timestamp": e.timestamp,
                    "symbol": e.symbol,
                    "macro_regime": e.macro_regime,
                    "micro_regime": e.micro_regime,
                    "micro_confidence": f"{e.micro_confidence:.0%}",
                    "ats_state": e.ats_state
                }
                for e in self.recent_overrides[-10:]
            ]
        }

    def update_config(
        self,
        enabled: bool = None,
        max_per_10min: int = None,
        size_multiplier: float = None,
        min_micro_confidence: float = None,
        min_rel_vol: float = None
    ):
        """Update override configuration"""
        if enabled is not None:
            self.config.enabled = enabled
        if max_per_10min is not None:
            self.config.max_per_10min = max(1, min(10, max_per_10min))
        if size_multiplier is not None:
            self.config.size_multiplier = max(0.1, min(1.0, size_multiplier))
        if min_micro_confidence is not None:
            self.config.min_micro_confidence = max(0.5, min(0.9, min_micro_confidence))
        if min_rel_vol is not None:
            self.config.min_rel_vol = max(1.0, min(10.0, min_rel_vol))

        logger.info(f"Override config updated: {self.config}")


# Singleton instance
_override_manager: Optional[MicroMomentumOverride] = None


def get_micro_override() -> MicroMomentumOverride:
    """Get or create the micro-momentum override singleton"""
    global _override_manager
    if _override_manager is None:
        _override_manager = MicroMomentumOverride()
    return _override_manager


def check_micro_override(
    symbol: str,
    macro_regime: str,
    micro_regime: str,
    micro_confidence: float,
    ats_state: str,
    **kwargs
) -> tuple[bool, str, float]:
    """Convenience function to check micro-momentum override"""
    return get_micro_override().check_override(
        symbol, macro_regime, micro_regime, micro_confidence, ats_state, **kwargs
    )
