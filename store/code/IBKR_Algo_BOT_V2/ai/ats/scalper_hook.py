"""
ATS Scalper Hook

Integration with HFT Scalper for trade execution.
"""

from datetime import datetime
from typing import Optional, Tuple, Dict
import logging
from .types import AtsTrigger, AtsState
from .ats_feed import get_ats_feed
from .ats_registry import get_ats_registry
from .gating_hook import get_ats_gating_hook


logger = logging.getLogger(__name__)


class AtsScalperHook:
    """
    Hook for HFT Scalper.

    Provides:
    - Entry permissions based on ATS state
    - Position size boosts for high-confidence setups
    - Exit recommendations based on exhaustion
    - Continuation permissions for scaling
    """

    def __init__(self):
        self._feed = get_ats_feed()
        self._registry = get_ats_registry()
        self._gating = get_ats_gating_hook()

        # Track positions we've approved
        self._active_positions: Dict[str, dict] = {}

    def check_entry_permission(
        self,
        symbol: str,
        proposed_entry: float,
        proposed_size: float
    ) -> Tuple[bool, str, dict]:
        """
        Check if entry should be allowed.

        Args:
            symbol: Stock symbol
            proposed_entry: Proposed entry price
            proposed_size: Proposed position size

        Returns:
            (allowed, reason, recommendation) tuple
        """
        # Check via gating hook
        approved, reason, trigger = self._gating.check_approval(symbol, proposed_entry)

        if not approved:
            return False, reason, {}

        # Get entry recommendation
        rec = self._gating.get_entry_recommendation(symbol)
        if not rec:
            return False, "NO_RECOMMENDATION", {}

        # Apply size boost
        adjusted_size = proposed_size * rec.get("size_boost", 1.0)

        recommendation = {
            "approved": True,
            "entry": rec["entry"],
            "stop": rec["stop"],
            "target_1": rec["target_1"],
            "target_2": rec["target_2"],
            "adjusted_size": adjusted_size,
            "size_boost": rec["size_boost"],
            "score": rec["score"],
            "risk_reward": rec["risk_reward"],
        }

        # Track position
        self._active_positions[symbol] = {
            "entry_price": proposed_entry,
            "entry_time": datetime.now(),
            "size": adjusted_size,
            "ats_score": rec["score"],
        }

        logger.info(
            f"ATS ENTRY APPROVED: {symbol} @ {proposed_entry:.2f} "
            f"(size: {adjusted_size:.0f}, boost: {rec['size_boost']:.2f}x)"
        )

        return True, "APPROVED", recommendation

    def check_exit_signal(self, symbol: str, current_price: float) -> Tuple[bool, str]:
        """
        Check if position should be exited based on ATS state.

        Args:
            symbol: Stock symbol
            current_price: Current price

        Returns:
            (should_exit, reason) tuple
        """
        state = self._registry.get_state(symbol)

        if not state:
            return False, "NO_STATE"

        # Exit on exhaustion
        if state.state == AtsState.EXHAUSTION:
            return True, "EXHAUSTION: Momentum fading"

        # Exit on invalidation
        if state.state == AtsState.INVALIDATED:
            return True, "INVALIDATED: Pattern failed"

        # Check for declining score
        if state.score_trend == "DECLINING" and state.score < 40:
            return True, "SCORE_DECLINING: Score dropped below 40"

        # Check position data
        pos = self._active_positions.get(symbol)
        if pos:
            entry = pos.get("entry_price", current_price)
            pnl_pct = ((current_price - entry) / entry) * 100

            # If we're up and momentum exhausting
            if pnl_pct > 1.0 and state.state == AtsState.EXHAUSTION:
                return True, f"TAKE_PROFIT: Up {pnl_pct:.1f}% with exhaustion"

        return False, "HOLD"

    def check_continuation_permission(
        self,
        symbol: str,
        current_price: float
    ) -> Tuple[bool, str, float]:
        """
        Check if position can be scaled (added to).

        Args:
            symbol: Stock symbol
            current_price: Current price

        Returns:
            (allowed, reason, size_boost) tuple
        """
        state = self._registry.get_state(symbol)

        if not state:
            return False, "NO_STATE", 1.0

        # Only allow continuation in ACTIVE state
        if state.state != AtsState.ACTIVE:
            return False, f"STATE: {state.state.value}", 1.0

        # Require high score
        if state.score < 70:
            return False, f"SCORE: {state.score:.0f} < 70", 1.0

        # Check score trend
        if state.score_trend == "DECLINING":
            return False, "SCORE_DECLINING", 1.0

        # Calculate size boost based on score
        size_boost = 1.0
        if state.score >= 85:
            size_boost = 0.5  # Add 50% more
        elif state.score >= 75:
            size_boost = 0.25  # Add 25% more

        return True, "CONTINUATION_APPROVED", size_boost

    def on_position_closed(self, symbol: str, pnl: float):
        """
        Notify when position is closed.

        Args:
            symbol: Stock symbol
            pnl: Realized P&L
        """
        if symbol in self._active_positions:
            pos = self._active_positions[symbol]
            hold_time = (datetime.now() - pos["entry_time"]).total_seconds()

            logger.info(
                f"ATS Position Closed: {symbol} "
                f"P&L: ${pnl:.2f} "
                f"Hold: {hold_time:.0f}s "
                f"Entry Score: {pos['ats_score']:.0f}"
            )

            del self._active_positions[symbol]

    def get_active_positions(self) -> Dict[str, dict]:
        """Get ATS-tracked positions"""
        return self._active_positions.copy()

    def get_tradeable_symbols(self) -> list[str]:
        """Get symbols that are tradeable via ATS"""
        tradeable = []
        for symbol in self._feed.get_active_symbols():
            allowed, _ = self._feed.is_symbol_tradeable(symbol)
            if allowed:
                tradeable.append(symbol)
        return tradeable

    def get_status(self) -> dict:
        """Get hook status"""
        return {
            "active_positions": len(self._active_positions),
            "positions": {
                sym: {
                    "entry": pos["entry_price"],
                    "score": pos["ats_score"],
                    "hold_seconds": (datetime.now() - pos["entry_time"]).total_seconds(),
                }
                for sym, pos in self._active_positions.items()
            },
            "tradeable_symbols": self.get_tradeable_symbols(),
            "gating_stats": self._gating.get_stats(),
        }


# Singleton instance
_hook: Optional[AtsScalperHook] = None


def get_ats_scalper_hook() -> AtsScalperHook:
    """Get singleton ATS scalper hook"""
    global _hook
    if _hook is None:
        _hook = AtsScalperHook()
    return _hook
