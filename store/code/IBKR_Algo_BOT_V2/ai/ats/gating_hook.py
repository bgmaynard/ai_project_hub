"""
ATS Gating Hook

Integration with Signal Gating Engine for trade approval.
"""

from datetime import datetime
from typing import Optional, Tuple
import logging
from .types import AtsTrigger, AtsState
from .ats_registry import get_ats_registry
from .time_utils import should_allow_ats_trigger


logger = logging.getLogger(__name__)


class AtsGatingHook:
    """
    Hook for Signal Gating Engine.

    Provides ATS-based approval/veto for trade attempts.

    Approval Rules:
    1. Symbol must be in ACTIVE state
    2. Score must be >= 60
    3. Not on cooldown
    4. Time window must be valid
    5. R:R must be >= 1.5
    """

    # Gating thresholds
    MIN_SCORE = 60.0
    MIN_RR = 1.5

    def __init__(self):
        self._registry = get_ats_registry()
        self._approvals = 0
        self._vetoes = 0
        self._veto_reasons: dict[str, int] = {}

    def check_approval(
        self,
        symbol: str,
        entry_price: Optional[float] = None
    ) -> Tuple[bool, str, Optional[AtsTrigger]]:
        """
        Check if trade should be approved.

        Args:
            symbol: Stock symbol
            entry_price: Proposed entry price (optional)

        Returns:
            (approved, reason, trigger) tuple
        """
        # Check time window
        allowed, time_reason = should_allow_ats_trigger()
        if not allowed:
            return self._veto(symbol, f"TIME: {time_reason}")

        # Check cooldown
        if self._registry.is_on_cooldown(symbol):
            remaining = self._registry.get_cooldown_remaining(symbol)
            return self._veto(symbol, f"COOLDOWN: {remaining:.0f}s remaining")

        # Check state
        state = self._registry.get_state(symbol)
        if not state:
            return self._veto(symbol, "NO_STATE: Symbol not tracked by ATS")

        if state.state == AtsState.EXHAUSTION:
            return self._veto(symbol, "EXHAUSTION: Momentum exhausted")

        if state.state == AtsState.INVALIDATED:
            return self._veto(symbol, "INVALIDATED: Pattern failed")

        if state.state not in [AtsState.ACTIVE]:
            return self._veto(symbol, f"STATE: Not ACTIVE (is {state.state.value})")

        # Check score
        if state.score < self.MIN_SCORE:
            return self._veto(symbol, f"SCORE: {state.score:.0f} < {self.MIN_SCORE}")

        # Check trigger exists and is valid
        trigger = state.last_trigger
        if not trigger:
            return self._veto(symbol, "NO_TRIGGER: No trigger event")

        if trigger.permission == "BLOCKED":
            return self._veto(symbol, f"BLOCKED: {trigger.block_reason}")

        # Check R:R
        if trigger.risk_reward < self.MIN_RR:
            return self._veto(symbol, f"RR: {trigger.risk_reward:.1f} < {self.MIN_RR}")

        # All checks passed
        self._approvals += 1
        logger.info(f"ATS APPROVED: {symbol} (score: {state.score:.0f}, R:R: {trigger.risk_reward:.1f})")

        return True, "APPROVED", trigger

    def _veto(self, symbol: str, reason: str) -> Tuple[bool, str, None]:
        """Record veto and return result"""
        self._vetoes += 1

        # Track reason category
        category = reason.split(":")[0]
        self._veto_reasons[category] = self._veto_reasons.get(category, 0) + 1

        logger.debug(f"ATS VETO: {symbol} - {reason}")
        return False, reason, None

    def get_entry_recommendation(self, symbol: str) -> Optional[dict]:
        """
        Get entry recommendation if approved.

        Returns entry levels from ATS trigger.
        """
        state = self._registry.get_state(symbol)
        if not state or not state.last_trigger:
            return None

        trigger = state.last_trigger

        return {
            "symbol": symbol,
            "entry": trigger.entry_price,
            "stop": trigger.stop_loss,
            "target_1": trigger.target_1,
            "target_2": trigger.target_2,
            "size_boost": trigger.size_boost,
            "score": state.score,
            "risk_reward": trigger.risk_reward,
            "zone_low": trigger.zone_low,
            "zone_high": trigger.zone_high,
        }

    def get_stats(self) -> dict:
        """Get gating statistics"""
        total = self._approvals + self._vetoes
        approval_rate = (self._approvals / total * 100) if total > 0 else 0.0

        return {
            "total_checks": total,
            "approvals": self._approvals,
            "vetoes": self._vetoes,
            "approval_rate": f"{approval_rate:.1f}%",
            "veto_breakdown": self._veto_reasons,
        }

    def reset_stats(self):
        """Reset statistics"""
        self._approvals = 0
        self._vetoes = 0
        self._veto_reasons.clear()


# Singleton instance
_hook: Optional[AtsGatingHook] = None


def get_ats_gating_hook() -> AtsGatingHook:
    """Get singleton ATS gating hook"""
    global _hook
    if _hook is None:
        _hook = AtsGatingHook()
    return _hook
