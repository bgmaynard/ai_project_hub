"""
Momentum State Machine
======================
Per-symbol state management with clear transitions and handoffs.

States:
  IDLE -> ATTENTION -> SETUP -> IGNITION -> IN_POSITION -> EXIT

Entry logic owns: IDLE through IGNITION
Exit/Monitor logic owns: IN_POSITION through EXIT

Based on ChatGPT analysis: "The system stops arguing with itself
and starts behaving consistently."
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


class MomentumState(Enum):
    """Momentum state for a symbol"""
    IDLE = "IDLE"              # No action, not watching
    ATTENTION = "ATTENTION"    # Interesting but not buyable
    SETUP = "SETUP"            # Pattern exists, waiting for ignition
    IGNITION = "IGNITION"      # Entry is legal - BUY signal
    IN_POSITION = "IN_POSITION"  # Holding, monitoring
    EXIT = "EXIT"              # Exiting position
    COOLDOWN = "COOLDOWN"      # Post-exit cooldown


class TransitionReason(Enum):
    """Reason for state transition"""
    # Entry side
    SCANNER_ADD = "SCANNER_ADD"
    NEWS_CATALYST = "NEWS_CATALYST"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    PATTERN_DETECTED = "PATTERN_DETECTED"
    HOD_APPROACH = "HOD_APPROACH"
    MOMENTUM_SCORE = "MOMENTUM_SCORE"
    IGNITION_TRIGGER = "IGNITION_TRIGGER"
    ORDER_FILLED = "ORDER_FILLED"

    # Exit side
    STOP_LOSS = "STOP_LOSS"
    PROFIT_TARGET = "PROFIT_TARGET"
    TRAILING_STOP = "TRAILING_STOP"
    MOMENTUM_FAILED = "MOMENTUM_FAILED"
    REGIME_CHANGE = "REGIME_CHANGE"
    MANUAL_EXIT = "MANUAL_EXIT"
    MAX_HOLD_TIME = "MAX_HOLD_TIME"

    # Reset
    COOLDOWN_EXPIRED = "COOLDOWN_EXPIRED"
    MOMENTUM_LOST = "MOMENTUM_LOST"
    FILTER_REJECT = "FILTER_REJECT"
    TIMEOUT = "TIMEOUT"


@dataclass
class StateTransition:
    """Record of a state transition"""
    symbol: str
    from_state: MomentumState
    to_state: MomentumState
    reason: TransitionReason
    timestamp: datetime
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'from_state': self.from_state.value,
            'to_state': self.to_state.value,
            'reason': self.reason.value,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details
        }


@dataclass
class SymbolState:
    """Complete state for a symbol"""
    symbol: str
    state: MomentumState
    entered_state_at: datetime
    momentum_score: int = 0
    entry_price: float = 0
    current_price: float = 0
    high_since_entry: float = 0
    pnl_pct: float = 0

    # State-specific data
    attention_reason: str = ""
    setup_pattern: str = ""
    ignition_score: int = 0

    # Position data (when IN_POSITION)
    shares: int = 0
    stop_price: float = 0
    target_price: float = 0

    # History
    transitions: List[StateTransition] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'state': self.state.value,
            'entered_state_at': self.entered_state_at.isoformat(),
            'time_in_state_seconds': (datetime.now() - self.entered_state_at).total_seconds(),
            'momentum_score': self.momentum_score,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'high_since_entry': self.high_since_entry,
            'pnl_pct': round(self.pnl_pct, 2),
            'attention_reason': self.attention_reason,
            'setup_pattern': self.setup_pattern,
            'ignition_score': self.ignition_score,
            'shares': self.shares,
            'stop_price': self.stop_price,
            'target_price': self.target_price,
            'last_update': self.last_update.isoformat(),
            'transition_count': len(self.transitions)
        }


class MomentumStateMachine:
    """
    Manages momentum state for all symbols.

    Key principles:
    1. Entry logic owns IDLE -> IGNITION
    2. After fill, monitor/exit logic owns IN_POSITION -> EXIT
    3. Every transition is logged
    4. States have timeouts to prevent stale data
    """

    # Timeouts (seconds)
    ATTENTION_TIMEOUT = 300     # 5 minutes in ATTENTION before reset
    SETUP_TIMEOUT = 180         # 3 minutes in SETUP before reset
    IGNITION_TIMEOUT = 15       # 15 seconds to execute or lose it
    COOLDOWN_DURATION = 300     # 5 minute cooldown after exit

    # Score thresholds
    ATTENTION_SCORE = 40        # Score to enter ATTENTION
    SETUP_SCORE = 55            # Score to enter SETUP
    IGNITION_SCORE = 70         # Score to enter IGNITION

    def __init__(self):
        self._states: Dict[str, SymbolState] = {}
        self._transition_log: List[StateTransition] = []
        self._callbacks: Dict[str, List[Callable]] = {
            'on_ignition': [],
            'on_position_enter': [],
            'on_exit': [],
        }

    def get_state(self, symbol: str) -> Optional[SymbolState]:
        """Get current state for a symbol"""
        return self._states.get(symbol)

    def get_all_states(self) -> Dict[str, SymbolState]:
        """Get all symbol states"""
        return self._states.copy()

    def get_symbols_in_state(self, state: MomentumState) -> List[str]:
        """Get all symbols in a specific state"""
        return [s for s, st in self._states.items() if st.state == state]

    def transition(self,
                   symbol: str,
                   to_state: MomentumState,
                   reason: TransitionReason,
                   details: Dict = None) -> bool:
        """
        Transition a symbol to a new state.

        Returns True if transition was valid and executed.
        """
        now = datetime.now()
        details = details or {}

        # Get or create state
        if symbol not in self._states:
            self._states[symbol] = SymbolState(
                symbol=symbol,
                state=MomentumState.IDLE,
                entered_state_at=now
            )

        current = self._states[symbol]
        from_state = current.state

        # Validate transition
        if not self._is_valid_transition(from_state, to_state):
            logger.warning(
                f"Invalid transition: {symbol} {from_state.value} -> {to_state.value}"
            )
            return False

        # Record transition
        transition = StateTransition(
            symbol=symbol,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            timestamp=now,
            details=details
        )
        current.transitions.append(transition)
        self._transition_log.append(transition)

        # Update state
        current.state = to_state
        current.entered_state_at = now
        current.last_update = now

        # State-specific updates
        if to_state == MomentumState.ATTENTION:
            current.attention_reason = details.get('reason', '')

        elif to_state == MomentumState.SETUP:
            current.setup_pattern = details.get('pattern', '')

        elif to_state == MomentumState.IGNITION:
            current.ignition_score = details.get('score', 0)
            # Fire callbacks
            self._fire_callback('on_ignition', symbol, current)

        elif to_state == MomentumState.IN_POSITION:
            current.entry_price = details.get('entry_price', 0)
            current.shares = details.get('shares', 0)
            current.stop_price = details.get('stop_price', 0)
            current.target_price = details.get('target_price', 0)
            current.high_since_entry = current.entry_price
            # Fire callbacks
            self._fire_callback('on_position_enter', symbol, current)

        elif to_state == MomentumState.EXIT:
            current.pnl_pct = details.get('pnl_pct', 0)
            # Fire callbacks
            self._fire_callback('on_exit', symbol, current)

        logger.info(
            f"STATE TRANSITION: {symbol} {from_state.value} -> {to_state.value} "
            f"({reason.value})"
        )

        return True

    def _is_valid_transition(self, from_state: MomentumState, to_state: MomentumState) -> bool:
        """Check if transition is valid"""
        valid_transitions = {
            MomentumState.IDLE: [MomentumState.ATTENTION],
            MomentumState.ATTENTION: [MomentumState.SETUP, MomentumState.IDLE],
            MomentumState.SETUP: [MomentumState.IGNITION, MomentumState.ATTENTION, MomentumState.IDLE],
            MomentumState.IGNITION: [MomentumState.IN_POSITION, MomentumState.SETUP, MomentumState.IDLE],
            MomentumState.IN_POSITION: [MomentumState.EXIT],
            MomentumState.EXIT: [MomentumState.COOLDOWN, MomentumState.IDLE],
            MomentumState.COOLDOWN: [MomentumState.IDLE],
        }
        return to_state in valid_transitions.get(from_state, [])

    def update_momentum(self, symbol: str, score: int, details: Dict = None) -> Optional[MomentumState]:
        """
        Update momentum score and potentially trigger state transitions.

        Returns the new state if transitioned, None otherwise.
        """
        details = details or {}
        details['score'] = score

        state = self.get_state(symbol)
        current_state = state.state if state else MomentumState.IDLE

        # Determine target state based on score
        if score >= self.IGNITION_SCORE:
            target_state = MomentumState.IGNITION
        elif score >= self.SETUP_SCORE:
            target_state = MomentumState.SETUP
        elif score >= self.ATTENTION_SCORE:
            target_state = MomentumState.ATTENTION
        else:
            target_state = MomentumState.IDLE

        # Only transition forward (or backward if score drops)
        if current_state == MomentumState.IDLE and target_state != MomentumState.IDLE:
            self.transition(symbol, MomentumState.ATTENTION, TransitionReason.MOMENTUM_SCORE, details)
            current_state = MomentumState.ATTENTION

        if current_state == MomentumState.ATTENTION:
            if target_state == MomentumState.IDLE:
                self.transition(symbol, MomentumState.IDLE, TransitionReason.MOMENTUM_LOST, details)
                return MomentumState.IDLE
            elif target_state in [MomentumState.SETUP, MomentumState.IGNITION]:
                self.transition(symbol, MomentumState.SETUP, TransitionReason.MOMENTUM_SCORE, details)
                current_state = MomentumState.SETUP

        if current_state == MomentumState.SETUP:
            if target_state == MomentumState.IGNITION:
                self.transition(symbol, MomentumState.IGNITION, TransitionReason.IGNITION_TRIGGER, details)
                return MomentumState.IGNITION
            elif target_state == MomentumState.ATTENTION:
                self.transition(symbol, MomentumState.ATTENTION, TransitionReason.MOMENTUM_LOST, details)
                return MomentumState.ATTENTION

        # Update score in state
        if symbol in self._states:
            self._states[symbol].momentum_score = score
            self._states[symbol].last_update = datetime.now()

        return self._states[symbol].state if symbol in self._states else None

    def update_position(self, symbol: str, current_price: float):
        """Update position tracking data"""
        if symbol not in self._states:
            return

        state = self._states[symbol]
        if state.state != MomentumState.IN_POSITION:
            return

        state.current_price = current_price
        state.last_update = datetime.now()

        if current_price > state.high_since_entry:
            state.high_since_entry = current_price

        if state.entry_price > 0:
            state.pnl_pct = ((current_price - state.entry_price) / state.entry_price) * 100

    def check_timeouts(self) -> List[str]:
        """
        Check for state timeouts and reset stale symbols.

        Returns list of symbols that were reset.
        """
        now = datetime.now()
        reset_symbols = []

        for symbol, state in list(self._states.items()):
            time_in_state = (now - state.entered_state_at).total_seconds()

            if state.state == MomentumState.ATTENTION and time_in_state > self.ATTENTION_TIMEOUT:
                self.transition(symbol, MomentumState.IDLE, TransitionReason.TIMEOUT)
                reset_symbols.append(symbol)

            elif state.state == MomentumState.SETUP and time_in_state > self.SETUP_TIMEOUT:
                self.transition(symbol, MomentumState.ATTENTION, TransitionReason.TIMEOUT)
                reset_symbols.append(symbol)

            elif state.state == MomentumState.IGNITION and time_in_state > self.IGNITION_TIMEOUT:
                # Ignition window expired without fill
                self.transition(symbol, MomentumState.SETUP, TransitionReason.TIMEOUT)
                reset_symbols.append(symbol)

            elif state.state == MomentumState.COOLDOWN and time_in_state > self.COOLDOWN_DURATION:
                self.transition(symbol, MomentumState.IDLE, TransitionReason.COOLDOWN_EXPIRED)
                reset_symbols.append(symbol)

        return reset_symbols

    def enter_position(self, symbol: str, entry_price: float, shares: int,
                       stop_price: float = 0, target_price: float = 0) -> bool:
        """
        Mark symbol as entered position (after order fill).

        Returns True if successful.
        """
        state = self.get_state(symbol)
        if not state or state.state != MomentumState.IGNITION:
            logger.warning(f"Cannot enter position for {symbol} - not in IGNITION state")
            return False

        return self.transition(
            symbol,
            MomentumState.IN_POSITION,
            TransitionReason.ORDER_FILLED,
            {
                'entry_price': entry_price,
                'shares': shares,
                'stop_price': stop_price,
                'target_price': target_price
            }
        )

    def exit_position(self, symbol: str, reason: TransitionReason, pnl_pct: float = 0) -> bool:
        """
        Mark symbol as exiting position.

        Returns True if successful.
        """
        state = self.get_state(symbol)
        if not state or state.state != MomentumState.IN_POSITION:
            logger.warning(f"Cannot exit position for {symbol} - not in IN_POSITION state")
            return False

        success = self.transition(
            symbol,
            MomentumState.EXIT,
            reason,
            {'pnl_pct': pnl_pct}
        )

        if success:
            # Immediately transition to cooldown
            self.transition(symbol, MomentumState.COOLDOWN, reason)

        return success

    def add_to_attention(self, symbol: str, reason: str = "") -> bool:
        """
        Add a symbol to ATTENTION state (e.g., from scanner or news).

        Returns True if successful.
        """
        state = self.get_state(symbol)

        if state and state.state == MomentumState.COOLDOWN:
            logger.debug(f"Cannot add {symbol} to attention - in cooldown")
            return False

        if not state or state.state == MomentumState.IDLE:
            return self.transition(
                symbol,
                MomentumState.ATTENTION,
                TransitionReason.SCANNER_ADD,
                {'reason': reason}
            )

        return False  # Already in a higher state

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for state events"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _fire_callback(self, event: str, symbol: str, state: SymbolState):
        """Fire callbacks for an event"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(symbol, state)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    def get_ignition_symbols(self) -> List[str]:
        """Get all symbols currently in IGNITION state (ready to buy)"""
        return self.get_symbols_in_state(MomentumState.IGNITION)

    def get_position_symbols(self) -> List[str]:
        """Get all symbols currently in position"""
        return self.get_symbols_in_state(MomentumState.IN_POSITION)

    def get_transition_log(self, limit: int = 100) -> List[Dict]:
        """Get recent transitions"""
        return [t.to_dict() for t in self._transition_log[-limit:]]

    def get_summary(self) -> Dict:
        """Get summary of all states"""
        summary = {
            'total_symbols': len(self._states),
            'by_state': {},
            'ignition_ready': [],
            'in_position': [],
            'recent_transitions': len(self._transition_log)
        }

        for state in MomentumState:
            symbols = self.get_symbols_in_state(state)
            summary['by_state'][state.value] = len(symbols)

        summary['ignition_ready'] = self.get_ignition_symbols()
        summary['in_position'] = self.get_position_symbols()

        return summary

    def reset_symbol(self, symbol: str):
        """Force reset a symbol to IDLE"""
        if symbol in self._states:
            self.transition(symbol, MomentumState.IDLE, TransitionReason.MANUAL_EXIT)

    def clear_all(self):
        """Clear all states (for testing)"""
        self._states.clear()
        self._transition_log.clear()


# Singleton instance
_state_machine: Optional[MomentumStateMachine] = None


def get_state_machine() -> MomentumStateMachine:
    """Get singleton state machine instance"""
    global _state_machine
    if _state_machine is None:
        _state_machine = MomentumStateMachine()
    return _state_machine


if __name__ == "__main__":
    # Test the state machine
    sm = MomentumStateMachine()

    print("="*60)
    print("MOMENTUM STATE MACHINE TEST")
    print("="*60)

    # Simulate a trade lifecycle
    symbol = "TEST"

    print(f"\n1. Adding {symbol} to attention...")
    sm.add_to_attention(symbol, "Scanner detected gapper")
    print(f"   State: {sm.get_state(symbol).state.value}")

    print(f"\n2. Updating momentum score to 60 (SETUP)...")
    sm.update_momentum(symbol, 60, {'pattern': 'bull_flag'})
    print(f"   State: {sm.get_state(symbol).state.value}")

    print(f"\n3. Updating momentum score to 75 (IGNITION)...")
    sm.update_momentum(symbol, 75)
    print(f"   State: {sm.get_state(symbol).state.value}")

    print(f"\n4. Order filled - entering position...")
    sm.enter_position(symbol, entry_price=5.50, shares=100, stop_price=5.25, target_price=6.00)
    print(f"   State: {sm.get_state(symbol).state.value}")

    print(f"\n5. Updating position price...")
    sm.update_position(symbol, current_price=5.75)
    state = sm.get_state(symbol)
    print(f"   PnL: {state.pnl_pct:.2f}%")

    print(f"\n6. Exiting position (trailing stop)...")
    sm.exit_position(symbol, TransitionReason.TRAILING_STOP, pnl_pct=4.5)
    print(f"   State: {sm.get_state(symbol).state.value}")

    print(f"\n--- Summary ---")
    print(json.dumps(sm.get_summary(), indent=2))

    print(f"\n--- Transition Log ---")
    for t in sm.get_transition_log():
        print(f"  {t['from_state']} -> {t['to_state']} ({t['reason']})")
