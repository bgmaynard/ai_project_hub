"""
ATS Registry

Per-symbol state tracking with history and alerts.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from .types import AtsState, AtsTrigger, AtsSymbolState


@dataclass
class AtsAlert:
    """Alert event from ATS system"""
    symbol: str
    alert_type: str  # "FORMING", "ACTIVE", "EXHAUSTION", "TRIGGER"
    message: str
    score: float
    trigger: Optional[AtsTrigger] = None
    timestamp: datetime = field(default_factory=datetime.now)


class AtsRegistry:
    """
    Registry for ATS state tracking across all symbols.

    Maintains:
    - Current state per symbol
    - Trigger history
    - Alert log
    - Cooldown tracking
    """

    def __init__(
        self,
        max_alerts: int = 100,
        max_triggers: int = 50,
        cooldown_minutes: int = 5,
    ):
        self.max_alerts = max_alerts
        self.max_triggers = max_triggers
        self.cooldown_minutes = cooldown_minutes

        self._states: Dict[str, AtsSymbolState] = {}
        self._triggers: List[AtsTrigger] = []
        self._alerts: List[AtsAlert] = []
        self._cooldowns: Dict[str, datetime] = {}
        self._last_update: datetime = datetime.now()

    def update_state(
        self,
        symbol: str,
        new_state: AtsState,
        score: float,
        trigger: Optional[AtsTrigger] = None
    ):
        """Update symbol state and log changes"""
        old_state = self._states.get(symbol)

        # Create or update state
        if symbol not in self._states:
            self._states[symbol] = AtsSymbolState(symbol=symbol)

        state = self._states[symbol]
        old_state_value = state.state

        # Update state
        state.state = new_state
        state.add_score(score)
        state.last_update = datetime.now()

        if trigger:
            state.last_trigger = trigger
            self._add_trigger(trigger)

        # Check for state transitions worth alerting
        if old_state_value != new_state:
            self._create_transition_alert(symbol, old_state_value, new_state, score, trigger)

        self._last_update = datetime.now()

    def _create_transition_alert(
        self,
        symbol: str,
        old_state: AtsState,
        new_state: AtsState,
        score: float,
        trigger: Optional[AtsTrigger]
    ):
        """Create alert for state transition"""
        alert_type = new_state.value

        messages = {
            AtsState.FORMING: f"{symbol}: SmartZone forming (score: {score:.0f})",
            AtsState.ACTIVE: f"{symbol}: BREAKOUT ACTIVE - ready to trade (score: {score:.0f})",
            AtsState.EXHAUSTION: f"{symbol}: Momentum exhausting - avoid/exit (score: {score:.0f})",
            AtsState.INVALIDATED: f"{symbol}: Pattern invalidated - skip",
            AtsState.IDLE: f"{symbol}: Reset to idle",
        }

        message = messages.get(new_state, f"{symbol}: State changed to {new_state.value}")

        alert = AtsAlert(
            symbol=symbol,
            alert_type=alert_type,
            message=message,
            score=score,
            trigger=trigger,
        )

        self._add_alert(alert)

    def _add_trigger(self, trigger: AtsTrigger):
        """Add trigger to history"""
        self._triggers.append(trigger)
        if len(self._triggers) > self.max_triggers:
            self._triggers = self._triggers[-self.max_triggers:]

        # Set cooldown
        self._cooldowns[trigger.symbol] = datetime.now()

    def _add_alert(self, alert: AtsAlert):
        """Add alert to log"""
        self._alerts.append(alert)
        if len(self._alerts) > self.max_alerts:
            self._alerts = self._alerts[-self.max_alerts:]

    def is_on_cooldown(self, symbol: str) -> bool:
        """Check if symbol is on cooldown"""
        if symbol not in self._cooldowns:
            return False

        cooldown_end = self._cooldowns[symbol] + timedelta(minutes=self.cooldown_minutes)
        return datetime.now() < cooldown_end

    def get_cooldown_remaining(self, symbol: str) -> float:
        """Get remaining cooldown in seconds"""
        if symbol not in self._cooldowns:
            return 0.0

        cooldown_end = self._cooldowns[symbol] + timedelta(minutes=self.cooldown_minutes)
        remaining = (cooldown_end - datetime.now()).total_seconds()
        return max(0.0, remaining)

    def get_state(self, symbol: str) -> Optional[AtsSymbolState]:
        """Get symbol state"""
        return self._states.get(symbol)

    def get_all_states(self) -> Dict[str, AtsSymbolState]:
        """Get all states"""
        return self._states.copy()

    def get_symbols_by_state(self, state: AtsState) -> List[str]:
        """Get symbols in a specific state"""
        return [
            sym for sym, s in self._states.items()
            if s.state == state
        ]

    def get_active_symbols(self) -> List[str]:
        """Get symbols in ACTIVE state"""
        return self.get_symbols_by_state(AtsState.ACTIVE)

    def get_forming_symbols(self) -> List[str]:
        """Get symbols in FORMING state"""
        return self.get_symbols_by_state(AtsState.FORMING)

    def get_recent_triggers(self, minutes: int = 30) -> List[AtsTrigger]:
        """Get triggers from last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [t for t in self._triggers if t.timestamp >= cutoff]

    def get_recent_alerts(self, minutes: int = 30) -> List[AtsAlert]:
        """Get alerts from last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [a for a in self._alerts if a.timestamp >= cutoff]

    def get_trigger_history(self, symbol: Optional[str] = None) -> List[AtsTrigger]:
        """Get trigger history, optionally filtered by symbol"""
        if symbol:
            return [t for t in self._triggers if t.symbol == symbol]
        return self._triggers.copy()

    def clear_symbol(self, symbol: str):
        """Clear state for symbol"""
        if symbol in self._states:
            del self._states[symbol]
        if symbol in self._cooldowns:
            del self._cooldowns[symbol]

    def reset(self):
        """Reset all state"""
        self._states.clear()
        self._triggers.clear()
        self._alerts.clear()
        self._cooldowns.clear()

    def get_status(self) -> dict:
        """Get registry status"""
        state_counts = {}
        for state in self._states.values():
            state_name = state.state.value
            state_counts[state_name] = state_counts.get(state_name, 0) + 1

        cooldown_count = sum(1 for s in self._cooldowns if self.is_on_cooldown(s))

        return {
            "total_symbols": len(self._states),
            "state_distribution": state_counts,
            "active_symbols": self.get_active_symbols(),
            "forming_symbols": self.get_forming_symbols(),
            "trigger_count": len(self._triggers),
            "alert_count": len(self._alerts),
            "symbols_on_cooldown": cooldown_count,
            "last_update": self._last_update.isoformat(),
        }


# Singleton instance
_registry: Optional[AtsRegistry] = None


def get_ats_registry() -> AtsRegistry:
    """Get singleton ATS registry"""
    global _registry
    if _registry is None:
        _registry = AtsRegistry()
    return _registry
