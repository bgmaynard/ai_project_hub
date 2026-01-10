"""
Scout → Strategy Handoff (Task Q)
=================================
Manages the transition from scout probes to full strategy execution.

When a scout:
- Holds for X bars, OR
- Triggers ATS SmartZone entry, OR
- Shows continuation volume

Then:
- Escalate symbol to ATS/Scalper logic
- Chronos becomes active AFTER scout success
- Normal gating resumes for scaling

If scout fails:
- Mark symbol SCOUT_FAILED
- Enforce cooldown
- No re-entry
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HandoffCondition(Enum):
    """What triggered the handoff"""
    BARS_HELD = "BARS_HELD"                 # Held for X bars
    ATS_SMARTZONE = "ATS_SMARTZONE"         # ATS SmartZone triggered
    CONTINUATION_VOLUME = "CONTINUATION_VOLUME"  # Volume continuation
    MOMENTUM_CONFIRMED = "MOMENTUM_CONFIRMED"    # General momentum confirmation


@dataclass
class HandoffRequest:
    """Request to hand off a scout to full strategy"""
    symbol: str
    scout_entry_price: float
    scout_size: int
    current_price: float
    condition: HandoffCondition
    target_strategy: str        # "ATS" or "SCALPER"
    chronos_required: bool      # Whether Chronos confirmation is needed
    scaling_allowed: bool       # Whether position scaling is allowed
    timestamp: datetime

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "scout_entry_price": self.scout_entry_price,
            "scout_size": self.scout_size,
            "current_price": self.current_price,
            "condition": self.condition.value,
            "target_strategy": self.target_strategy,
            "chronos_required": self.chronos_required,
            "scaling_allowed": self.scaling_allowed,
            "timestamp": self.timestamp.isoformat()
        }


class ScoutHandoffManager:
    """
    Manages scout → strategy transitions.

    Ensures:
    - Clean handoff from scout to ATS/Scalper
    - Chronos activation after scout success
    - Proper gating for scaling
    """

    def __init__(self):
        self.pending_handoffs: Dict[str, HandoffRequest] = {}
        self.completed_handoffs: List[HandoffRequest] = []
        self.failed_handoffs: List[Dict] = []

        # Handoff thresholds
        self.config = {
            "min_bars_for_handoff": 3,
            "min_gain_for_handoff_pct": 0.5,
            "continuation_volume_ratio": 1.5,
            "ats_smartzone_enabled": True,
            "scalper_handoff_enabled": True,
            "chronos_required_for_scaling": True,
            "max_pending_handoffs": 5
        }

        # Statistics
        self.stats = {
            "handoffs_attempted": 0,
            "handoffs_completed": 0,
            "handoffs_rejected": 0,
            "ats_handoffs": 0,
            "scalper_handoffs": 0
        }

    def check_handoff_conditions(
        self,
        symbol: str,
        scout_entry_price: float,
        current_price: float,
        bars_held: int,
        current_volume: float = 0,
        avg_volume: float = 0,
        ats_smartzone_triggered: bool = False
    ) -> Tuple[bool, Optional[HandoffCondition], str]:
        """
        Check if scout should be handed off to strategy.

        Returns:
            (should_handoff, condition, reason)
        """
        # Check ATS SmartZone first (highest priority)
        if self.config["ats_smartzone_enabled"] and ats_smartzone_triggered:
            return True, HandoffCondition.ATS_SMARTZONE, "ATS SmartZone triggered"

        # Check bars held
        gain_pct = ((current_price - scout_entry_price) / scout_entry_price) * 100
        if bars_held >= self.config["min_bars_for_handoff"]:
            if gain_pct >= self.config["min_gain_for_handoff_pct"]:
                return True, HandoffCondition.BARS_HELD, f"Held {bars_held} bars with {gain_pct:.2f}% gain"

        # Check continuation volume
        if avg_volume > 0 and current_volume > 0:
            vol_ratio = current_volume / avg_volume
            if vol_ratio >= self.config["continuation_volume_ratio"]:
                return True, HandoffCondition.CONTINUATION_VOLUME, f"Volume continuation {vol_ratio:.1f}x"

        return False, None, "No handoff condition met"

    def create_handoff_request(
        self,
        symbol: str,
        scout_entry_price: float,
        scout_size: int,
        current_price: float,
        condition: HandoffCondition
    ) -> HandoffRequest:
        """Create a handoff request"""
        # Determine target strategy based on condition
        if condition == HandoffCondition.ATS_SMARTZONE:
            target = "ATS"
            chronos_required = True
        else:
            target = "SCALPER"
            chronos_required = False  # Scalper can continue without Chronos initially

        request = HandoffRequest(
            symbol=symbol,
            scout_entry_price=scout_entry_price,
            scout_size=scout_size,
            current_price=current_price,
            condition=condition,
            target_strategy=target,
            chronos_required=chronos_required,
            scaling_allowed=True,
            timestamp=datetime.now()
        )

        self.pending_handoffs[symbol] = request
        self.stats["handoffs_attempted"] += 1

        logger.info(f"SCOUT_HANDOFF_REQUESTED: {symbol} -> {target} condition={condition.value}")

        return request

    async def execute_handoff(self, symbol: str) -> Tuple[bool, str]:
        """
        Execute the handoff to target strategy.

        Returns:
            (success, reason)
        """
        if symbol not in self.pending_handoffs:
            return False, f"No pending handoff for {symbol}"

        request = self.pending_handoffs[symbol]

        try:
            if request.target_strategy == "ATS":
                success, reason = await self._handoff_to_ats(request)
            elif request.target_strategy == "SCALPER":
                success, reason = await self._handoff_to_scalper(request)
            else:
                success, reason = False, f"Unknown target strategy: {request.target_strategy}"

            if success:
                self.completed_handoffs.append(request)
                del self.pending_handoffs[symbol]
                self.stats["handoffs_completed"] += 1

                if request.target_strategy == "ATS":
                    self.stats["ats_handoffs"] += 1
                else:
                    self.stats["scalper_handoffs"] += 1

                # Mark scout as handed off
                from ai.momentum_scout import get_momentum_scout
                scout = get_momentum_scout()
                scout.mark_handed_off(symbol)

                logger.info(f"SCOUT_HANDOFF_COMPLETE: {symbol} -> {request.target_strategy}")
            else:
                self.failed_handoffs.append({
                    "request": request.to_dict(),
                    "reason": reason,
                    "timestamp": datetime.now().isoformat()
                })
                del self.pending_handoffs[symbol]
                self.stats["handoffs_rejected"] += 1

                logger.warning(f"SCOUT_HANDOFF_FAILED: {symbol} reason={reason}")

            return success, reason

        except Exception as e:
            logger.error(f"Error executing handoff for {symbol}: {e}")
            return False, str(e)

    async def _handoff_to_ats(self, request: HandoffRequest) -> Tuple[bool, str]:
        """Hand off scout to ATS strategy"""
        try:
            # Check Chronos if required
            if request.chronos_required:
                from ai.chronos_adapter import get_chronos_adapter

                adapter = get_chronos_adapter()
                context = await adapter.get_context(request.symbol)

                if context is None:
                    return False, "Chronos context not available"

                # ATS requires favorable regime
                if context.market_regime in ["TRENDING_DOWN", "VOLATILE"]:
                    return False, f"ATS blocked by regime: {context.market_regime}"

                if context.regime_confidence < 0.5:
                    return False, f"Low Chronos confidence: {context.regime_confidence:.0%}"

            # Add to ATS priority queue
            # This would integrate with your existing ATS system
            logger.info(f"ATS_TAKEOVER: {request.symbol} from scout @ ${request.current_price:.2f}")

            return True, "Handed to ATS"

        except Exception as e:
            return False, f"ATS handoff error: {e}"

    async def _handoff_to_scalper(self, request: HandoffRequest) -> Tuple[bool, str]:
        """Hand off scout to HFT Scalper"""
        try:
            # Scalper doesn't require Chronos for initial takeover
            # but will use it for scaling decisions

            from ai.hft_scalper import get_hft_scalper

            scalper = get_hft_scalper()

            # Add to scalper's active positions with scout context
            # The scalper will treat this as an existing position to manage

            # For now, add to priority symbols for continuation monitoring
            if hasattr(scalper, 'priority_symbols'):
                if request.symbol not in scalper.priority_symbols:
                    scalper.priority_symbols.append(request.symbol)

            logger.info(f"SCALPER_TAKEOVER: {request.symbol} from scout @ ${request.current_price:.2f}")

            return True, "Handed to Scalper"

        except Exception as e:
            return False, f"Scalper handoff error: {e}"

    async def process_pending_handoffs(self):
        """Process all pending handoffs"""
        symbols = list(self.pending_handoffs.keys())
        results = []

        for symbol in symbols:
            success, reason = await self.execute_handoff(symbol)
            results.append({
                "symbol": symbol,
                "success": success,
                "reason": reason
            })

        return results

    def get_pending_handoffs(self) -> List[Dict]:
        """Get pending handoff requests"""
        return [h.to_dict() for h in self.pending_handoffs.values()]

    def get_completed_handoffs(self, limit: int = 20) -> List[Dict]:
        """Get completed handoffs"""
        return [h.to_dict() for h in self.completed_handoffs[-limit:]]

    def get_stats(self) -> Dict:
        """Get handoff statistics"""
        attempted = self.stats["handoffs_attempted"]
        completed = self.stats["handoffs_completed"]

        return {
            **self.stats,
            "success_rate": (completed / attempted * 100) if attempted > 0 else 0,
            "pending_count": len(self.pending_handoffs),
            "failed_count": len(self.failed_handoffs)
        }

    def get_status(self) -> Dict:
        """Get handoff manager status"""
        return {
            "pending_handoffs": self.get_pending_handoffs(),
            "stats": self.get_stats(),
            "config": self.config,
            "recent_completed": self.get_completed_handoffs(5),
            "timestamp": datetime.now().isoformat()
        }

    def update_config(self, updates: Dict):
        """Update configuration"""
        for key, value in updates.items():
            if key in self.config:
                self.config[key] = value
        return self.config


# Singleton instance
_manager: Optional[ScoutHandoffManager] = None


def get_handoff_manager() -> ScoutHandoffManager:
    """Get the singleton handoff manager"""
    global _manager
    if _manager is None:
        _manager = ScoutHandoffManager()
    return _manager
