"""
Momentum State Machine (v2 - ChatGPT FSM Spec)
===============================================
Per-symbol state management with clear transitions, handoffs, and ownership.

States:
  IDLE -> CANDIDATE -> IGNITING -> GATED -> IN_POSITION -> MONITORING -> EXITING -> COOLDOWN

Ownership:
  Entry logic owns:     IDLE through GATED
  Position logic owns:  IN_POSITION
  Exit/Monitor owns:    MONITORING through EXITING
  Cooldown logic owns:  COOLDOWN

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
    """Momentum state for a symbol - ChatGPT FSM Spec"""
    IDLE = "IDLE"                  # No action, not watching
    CANDIDATE = "CANDIDATE"        # Interesting but not igniting (was ATTENTION)
    IGNITING = "IGNITING"          # Momentum building, approaching entry (was SETUP)
    GATED = "GATED"                # NEW: Awaiting gating approval
    IN_POSITION = "IN_POSITION"    # Holding position
    MONITORING = "MONITORING"      # NEW: Exit manager monitoring (Chronos owns)
    EXITING = "EXITING"            # Actively exiting (was EXIT)
    COOLDOWN = "COOLDOWN"          # Post-exit cooldown


class StateOwner(Enum):
    """Which component owns each state"""
    ENTRY = "ENTRY"          # Entry logic: MomentumScorer, filters
    GATING = "GATING"        # Gating logic: GatedTradingManager
    POSITION = "POSITION"    # Position logic: Broker, order management
    EXIT = "EXIT"            # Exit logic: ChronosExitManager
    COOLDOWN = "COOLDOWN"    # Cooldown logic: Timer


# State ownership mapping
STATE_OWNERSHIP = {
    MomentumState.IDLE: StateOwner.ENTRY,
    MomentumState.CANDIDATE: StateOwner.ENTRY,
    MomentumState.IGNITING: StateOwner.ENTRY,
    MomentumState.GATED: StateOwner.GATING,
    MomentumState.IN_POSITION: StateOwner.POSITION,
    MomentumState.MONITORING: StateOwner.EXIT,
    MomentumState.EXITING: StateOwner.EXIT,
    MomentumState.COOLDOWN: StateOwner.COOLDOWN,
}


class TransitionReason(Enum):
    """Reason for state transition"""
    # Entry side (IDLE -> CANDIDATE -> IGNITING)
    SCANNER_ADD = "SCANNER_ADD"
    NEWS_CATALYST = "NEWS_CATALYST"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    PATTERN_DETECTED = "PATTERN_DETECTED"
    HOD_APPROACH = "HOD_APPROACH"
    MOMENTUM_SCORE = "MOMENTUM_SCORE"
    IGNITION_TRIGGER = "IGNITION_TRIGGER"

    # Gating side (IGNITING -> GATED -> IN_POSITION)
    GATING_REQUESTED = "GATING_REQUESTED"
    GATING_APPROVED = "GATING_APPROVED"
    GATING_REJECTED = "GATING_REJECTED"
    ORDER_FILLED = "ORDER_FILLED"

    # Exit side (IN_POSITION -> MONITORING -> EXITING)
    POSITION_ENTERED = "POSITION_ENTERED"
    STOP_LOSS = "STOP_LOSS"
    PROFIT_TARGET = "PROFIT_TARGET"
    TRAILING_STOP = "TRAILING_STOP"
    MOMENTUM_FAILED = "MOMENTUM_FAILED"
    CHRONOS_EXIT = "CHRONOS_EXIT"
    REGIME_CHANGE = "REGIME_CHANGE"
    MANUAL_EXIT = "MANUAL_EXIT"
    MAX_HOLD_TIME = "MAX_HOLD_TIME"

    # Reset
    COOLDOWN_EXPIRED = "COOLDOWN_EXPIRED"
    MOMENTUM_LOST = "MOMENTUM_LOST"
    VETO_TRIGGERED = "VETO_TRIGGERED"
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
    owner: StateOwner = None
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'from_state': self.from_state.value,
            'to_state': self.to_state.value,
            'reason': self.reason.value,
            'owner': self.owner.value if self.owner else None,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details
        }


@dataclass
class SymbolState:
    """Complete state for a symbol"""
    symbol: str
    state: MomentumState
    entered_state_at: datetime
    owner: StateOwner = StateOwner.ENTRY

    # Momentum data
    momentum_score: int = 0
    veto_reasons: List[str] = field(default_factory=list)
    is_vetoed: bool = False

    # Price data
    entry_price: float = 0
    current_price: float = 0
    high_since_entry: float = 0
    low_since_entry: float = 0
    pnl_pct: float = 0

    # State-specific data
    candidate_reason: str = ""    # Why in CANDIDATE
    igniting_score: int = 0       # Score when reached IGNITING
    gating_result: str = ""       # APPROVED, REJECTED, or reason
    gating_contract_id: str = ""  # Signal contract ID if applicable

    # Exit manager data
    exit_signal: str = ""         # Last exit signal from Chronos
    exit_confidence: float = 0    # Chronos confidence
    time_in_monitoring: float = 0 # Seconds in MONITORING

    # Position data (when IN_POSITION)
    shares: int = 0
    stop_price: float = 0
    target_price: float = 0
    trailing_stop_price: float = 0

    # History
    transitions: List[StateTransition] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'state': self.state.value,
            'owner': self.owner.value,
            'entered_state_at': self.entered_state_at.isoformat(),
            'time_in_state_seconds': (datetime.now() - self.entered_state_at).total_seconds(),
            'momentum_score': self.momentum_score,
            'is_vetoed': self.is_vetoed,
            'veto_reasons': self.veto_reasons,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'high_since_entry': self.high_since_entry,
            'pnl_pct': round(self.pnl_pct, 2),
            'candidate_reason': self.candidate_reason,
            'igniting_score': self.igniting_score,
            'gating_result': self.gating_result,
            'exit_signal': self.exit_signal,
            'shares': self.shares,
            'stop_price': self.stop_price,
            'target_price': self.target_price,
            'trailing_stop_price': self.trailing_stop_price,
            'last_update': self.last_update.isoformat(),
            'transition_count': len(self.transitions)
        }


class MomentumStateMachine:
    """
    Manages momentum state for all symbols.

    Key principles:
    1. Entry logic owns IDLE -> CANDIDATE -> IGNITING
    2. Gating logic owns IGNITING -> GATED -> IN_POSITION
    3. Exit logic owns IN_POSITION -> MONITORING -> EXITING
    4. Every transition is logged with owner
    5. States have timeouts to prevent stale data
    """

    # Timeouts (seconds)
    CANDIDATE_TIMEOUT = 300     # 5 minutes in CANDIDATE before reset
    IGNITING_TIMEOUT = 180      # 3 minutes in IGNITING before reset
    GATED_TIMEOUT = 15          # 15 seconds to get gating decision
    IN_POSITION_TIMEOUT = 10    # 10 seconds before auto-transition to MONITORING
    MONITORING_TIMEOUT = None   # No timeout - exit logic decides
    EXITING_TIMEOUT = 30        # 30 seconds to complete exit
    COOLDOWN_DURATION = 300     # 5 minute cooldown after exit

    # Default score thresholds (can be overridden via configure())
    DEFAULT_CANDIDATE_SCORE = 30   # Grid search optimal
    DEFAULT_IGNITING_SCORE = 45    # Grid search optimal
    DEFAULT_GATED_SCORE = 60       # Grid search optimal

    def __init__(self):
        self._states: Dict[str, SymbolState] = {}
        self._transition_log: List[StateTransition] = []
        self._callbacks: Dict[str, List[Callable]] = {
            'on_candidate': [],
            'on_igniting': [],
            'on_gating_requested': [],
            'on_gating_approved': [],
            'on_gating_rejected': [],
            'on_position_enter': [],
            'on_monitoring': [],
            'on_exit_signal': [],
            'on_exit': [],
            'on_cooldown': [],
        }

        # Gating manager reference (lazy loaded)
        self._gating_manager = None

        # Instance-level thresholds (configurable)
        self.candidate_score = self.DEFAULT_CANDIDATE_SCORE
        self.igniting_score = self.DEFAULT_IGNITING_SCORE
        self.gated_score = self.DEFAULT_GATED_SCORE

    def configure(self, candidate_score: int = None, igniting_score: int = None, gated_score: int = None):
        """Configure state machine thresholds"""
        if candidate_score is not None:
            self.candidate_score = candidate_score
        if igniting_score is not None:
            self.igniting_score = igniting_score
        if gated_score is not None:
            self.gated_score = gated_score
        logger.info(f"FSM thresholds configured: CANDIDATE={self.candidate_score}, IGNITING={self.igniting_score}, GATED={self.gated_score}")

    @property
    def gating_manager(self):
        """Lazy load gating manager"""
        if self._gating_manager is None:
            try:
                from ai.gated_trading import get_gated_trading_manager
                self._gating_manager = get_gated_trading_manager()
            except Exception as e:
                logger.debug(f"Gating manager not available: {e}")
        return self._gating_manager

    def get_state(self, symbol: str) -> Optional[SymbolState]:
        """Get current state for a symbol"""
        return self._states.get(symbol)

    def get_owner(self, symbol: str) -> Optional[StateOwner]:
        """Get the owner of a symbol's current state"""
        state = self.get_state(symbol)
        if state:
            return STATE_OWNERSHIP.get(state.state)
        return None

    def get_all_states(self) -> Dict[str, SymbolState]:
        """Get all symbol states"""
        return self._states.copy()

    def get_symbols_in_state(self, state: MomentumState) -> List[str]:
        """Get all symbols in a specific state"""
        return [s for s, st in self._states.items() if st.state == state]

    def get_symbols_by_owner(self, owner: StateOwner) -> List[str]:
        """Get all symbols owned by a specific component"""
        result = []
        for symbol, state in self._states.items():
            if STATE_OWNERSHIP.get(state.state) == owner:
                result.append(symbol)
        return result

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
                entered_state_at=now,
                owner=StateOwner.ENTRY
            )

        current = self._states[symbol]
        from_state = current.state

        # Validate transition
        if not self._is_valid_transition(from_state, to_state):
            logger.warning(
                f"Invalid transition: {symbol} {from_state.value} -> {to_state.value}"
            )
            return False

        # Get new owner
        new_owner = STATE_OWNERSHIP.get(to_state, StateOwner.ENTRY)

        # Record transition
        transition = StateTransition(
            symbol=symbol,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            timestamp=now,
            owner=new_owner,
            details=details
        )
        current.transitions.append(transition)
        self._transition_log.append(transition)

        # Update state
        current.state = to_state
        current.owner = new_owner
        current.entered_state_at = now
        current.last_update = now

        # State-specific updates
        if to_state == MomentumState.CANDIDATE:
            current.candidate_reason = details.get('reason', '')
            self._fire_callback('on_candidate', symbol, current)

        elif to_state == MomentumState.IGNITING:
            current.igniting_score = details.get('score', 0)
            self._fire_callback('on_igniting', symbol, current)

        elif to_state == MomentumState.GATED:
            self._fire_callback('on_gating_requested', symbol, current)

        elif to_state == MomentumState.IN_POSITION:
            current.entry_price = details.get('entry_price', 0)
            current.shares = details.get('shares', 0)
            current.stop_price = details.get('stop_price', 0)
            current.target_price = details.get('target_price', 0)
            current.high_since_entry = current.entry_price
            current.low_since_entry = current.entry_price
            current.gating_result = details.get('gating_result', 'APPROVED')
            current.gating_contract_id = details.get('contract_id', '')
            self._fire_callback('on_position_enter', symbol, current)

        elif to_state == MomentumState.MONITORING:
            current.time_in_monitoring = 0
            self._fire_callback('on_monitoring', symbol, current)

        elif to_state == MomentumState.EXITING:
            current.exit_signal = details.get('exit_signal', reason.value)
            current.exit_confidence = details.get('confidence', 0)
            current.pnl_pct = details.get('pnl_pct', 0)
            self._fire_callback('on_exit_signal', symbol, current)

        elif to_state == MomentumState.COOLDOWN:
            self._fire_callback('on_cooldown', symbol, current)

        logger.info(
            f"STATE: {symbol} {from_state.value} -> {to_state.value} "
            f"({reason.value}) [owner: {new_owner.value}]"
        )

        return True

    def _is_valid_transition(self, from_state: MomentumState, to_state: MomentumState) -> bool:
        """Check if transition is valid"""
        valid_transitions = {
            # Entry logic flow
            MomentumState.IDLE: [MomentumState.CANDIDATE],
            MomentumState.CANDIDATE: [MomentumState.IGNITING, MomentumState.IDLE],
            MomentumState.IGNITING: [MomentumState.GATED, MomentumState.CANDIDATE, MomentumState.IDLE],

            # Gating flow
            MomentumState.GATED: [MomentumState.IN_POSITION, MomentumState.IGNITING, MomentumState.IDLE],

            # Position flow
            MomentumState.IN_POSITION: [MomentumState.MONITORING, MomentumState.EXITING],

            # Exit flow
            MomentumState.MONITORING: [MomentumState.EXITING],
            MomentumState.EXITING: [MomentumState.COOLDOWN, MomentumState.IDLE],

            # Cooldown flow
            MomentumState.COOLDOWN: [MomentumState.IDLE],
        }
        return to_state in valid_transitions.get(from_state, [])

    def update_momentum(self, symbol: str, score: int, vetoed: bool = False,
                        veto_reasons: List[str] = None, details: Dict = None) -> Optional[MomentumState]:
        """
        Update momentum score and potentially trigger state transitions.

        Args:
            symbol: Stock symbol
            score: Momentum score 0-100
            vetoed: Whether score was vetoed
            veto_reasons: List of veto reasons
            details: Additional details

        Returns the new state if transitioned, None otherwise.
        """
        details = details or {}
        details['score'] = score
        details['vetoed'] = vetoed
        veto_reasons = veto_reasons or []

        state = self.get_state(symbol)
        current_state = state.state if state else MomentumState.IDLE

        # If vetoed, cannot progress (may regress)
        if vetoed:
            if symbol in self._states:
                self._states[symbol].is_vetoed = True
                self._states[symbol].veto_reasons = veto_reasons

            # If was in IGNITING or GATED, drop back
            if current_state in [MomentumState.IGNITING, MomentumState.GATED]:
                self.transition(symbol, MomentumState.CANDIDATE, TransitionReason.VETO_TRIGGERED, details)
                return MomentumState.CANDIDATE

            return current_state

        # Clear veto status
        if symbol in self._states:
            self._states[symbol].is_vetoed = False
            self._states[symbol].veto_reasons = []

        # Determine target state based on score (using instance thresholds)
        if score >= self.gated_score:
            target_state = MomentumState.GATED
        elif score >= self.igniting_score:
            target_state = MomentumState.IGNITING
        elif score >= self.candidate_score:
            target_state = MomentumState.CANDIDATE
        else:
            target_state = MomentumState.IDLE

        # Only transition forward (or backward if score drops)
        # IDLE -> CANDIDATE
        if current_state == MomentumState.IDLE and target_state != MomentumState.IDLE:
            self.transition(symbol, MomentumState.CANDIDATE, TransitionReason.MOMENTUM_SCORE, details)
            current_state = MomentumState.CANDIDATE

        # CANDIDATE transitions
        if current_state == MomentumState.CANDIDATE:
            if target_state == MomentumState.IDLE:
                self.transition(symbol, MomentumState.IDLE, TransitionReason.MOMENTUM_LOST, details)
                return MomentumState.IDLE
            elif target_state in [MomentumState.IGNITING, MomentumState.GATED]:
                self.transition(symbol, MomentumState.IGNITING, TransitionReason.MOMENTUM_SCORE, details)
                current_state = MomentumState.IGNITING

        # IGNITING transitions
        if current_state == MomentumState.IGNITING:
            if target_state == MomentumState.GATED:
                # Request gating approval
                self.transition(symbol, MomentumState.GATED, TransitionReason.GATING_REQUESTED, details)
                return MomentumState.GATED
            elif target_state == MomentumState.CANDIDATE:
                self.transition(symbol, MomentumState.CANDIDATE, TransitionReason.MOMENTUM_LOST, details)
                return MomentumState.CANDIDATE

        # Update score in state
        if symbol in self._states:
            self._states[symbol].momentum_score = score
            self._states[symbol].last_update = datetime.now()

        return self._states[symbol].state if symbol in self._states else None

    def request_gating(self, symbol: str, quote: Dict = None) -> Optional[Dict]:
        """
        Request gating decision for a symbol in IGNITING state.

        Returns gating result dict with 'approved', 'reason', etc.
        """
        state = self.get_state(symbol)
        if not state or state.state != MomentumState.IGNITING:
            return {'approved': False, 'reason': 'Not in IGNITING state'}

        # Transition to GATED
        self.transition(symbol, MomentumState.GATED, TransitionReason.GATING_REQUESTED)

        # Check with gating manager if available
        if self.gating_manager:
            try:
                approved, exec_request, reason = self.gating_manager.gate_trade_attempt(
                    symbol=symbol,
                    trigger_type='momentum_ignition',
                    quote=quote or {}
                )

                result = {
                    'approved': approved,
                    'reason': reason,
                    'exec_request': exec_request.to_dict() if exec_request else None
                }

                if approved:
                    self._fire_callback('on_gating_approved', symbol, state)
                else:
                    self._fire_callback('on_gating_rejected', symbol, state)
                    # Rejected - drop back to IGNITING
                    self.transition(symbol, MomentumState.IGNITING, TransitionReason.GATING_REJECTED,
                                    {'reason': reason})

                return result

            except Exception as e:
                logger.error(f"Gating error for {symbol}: {e}")
                return {'approved': False, 'reason': str(e)}
        else:
            # No gating manager - auto-approve
            self._fire_callback('on_gating_approved', symbol, state)
            return {'approved': True, 'reason': 'No gating manager - auto-approved'}

    def enter_position(self, symbol: str, entry_price: float, shares: int,
                       stop_price: float = 0, target_price: float = 0,
                       gating_result: str = "APPROVED") -> bool:
        """
        Mark symbol as entered position (after order fill).

        Returns True if successful.
        """
        state = self.get_state(symbol)
        if not state or state.state not in [MomentumState.GATED, MomentumState.IGNITING]:
            logger.warning(f"Cannot enter position for {symbol} - not in GATED/IGNITING state")
            return False

        return self.transition(
            symbol,
            MomentumState.IN_POSITION,
            TransitionReason.ORDER_FILLED,
            {
                'entry_price': entry_price,
                'shares': shares,
                'stop_price': stop_price,
                'target_price': target_price,
                'gating_result': gating_result
            }
        )

    def start_monitoring(self, symbol: str) -> bool:
        """
        Transition position to MONITORING state (exit manager takes over).

        Returns True if successful.
        """
        state = self.get_state(symbol)
        if not state or state.state != MomentumState.IN_POSITION:
            return False

        return self.transition(
            symbol,
            MomentumState.MONITORING,
            TransitionReason.POSITION_ENTERED
        )

    def update_position(self, symbol: str, current_price: float):
        """Update position tracking data"""
        if symbol not in self._states:
            return

        state = self._states[symbol]
        if state.state not in [MomentumState.IN_POSITION, MomentumState.MONITORING]:
            return

        state.current_price = current_price
        state.last_update = datetime.now()

        if current_price > state.high_since_entry:
            state.high_since_entry = current_price

        if current_price < state.low_since_entry or state.low_since_entry == 0:
            state.low_since_entry = current_price

        if state.entry_price > 0:
            state.pnl_pct = ((current_price - state.entry_price) / state.entry_price) * 100

        # Update monitoring time if in MONITORING
        if state.state == MomentumState.MONITORING:
            state.time_in_monitoring = (datetime.now() - state.entered_state_at).total_seconds()

    def signal_exit(self, symbol: str, exit_signal: str, confidence: float = 0,
                    pnl_pct: float = 0) -> bool:
        """
        Signal that exit manager wants to exit (transition to EXITING).

        Returns True if successful.
        """
        state = self.get_state(symbol)
        if not state or state.state not in [MomentumState.IN_POSITION, MomentumState.MONITORING]:
            logger.warning(f"Cannot signal exit for {symbol} - not in position/monitoring")
            return False

        # Map exit signal to reason
        reason_map = {
            'STOP_LOSS': TransitionReason.STOP_LOSS,
            'PROFIT_TARGET': TransitionReason.PROFIT_TARGET,
            'TRAILING_STOP': TransitionReason.TRAILING_STOP,
            'MOMENTUM_FAILED': TransitionReason.MOMENTUM_FAILED,
            'CHRONOS_EXIT': TransitionReason.CHRONOS_EXIT,
            'REGIME_CHANGE': TransitionReason.REGIME_CHANGE,
            'MAX_HOLD_TIME': TransitionReason.MAX_HOLD_TIME,
        }
        reason = reason_map.get(exit_signal, TransitionReason.MANUAL_EXIT)

        return self.transition(
            symbol,
            MomentumState.EXITING,
            reason,
            {
                'exit_signal': exit_signal,
                'confidence': confidence,
                'pnl_pct': pnl_pct
            }
        )

    def complete_exit(self, symbol: str, pnl_pct: float = 0) -> bool:
        """
        Complete exit and transition to COOLDOWN.

        Returns True if successful.
        """
        state = self.get_state(symbol)
        if not state or state.state != MomentumState.EXITING:
            return False

        state.pnl_pct = pnl_pct

        return self.transition(
            symbol,
            MomentumState.COOLDOWN,
            TransitionReason.MANUAL_EXIT,
            {'pnl_pct': pnl_pct}
        )

    def check_timeouts(self) -> List[str]:
        """
        Check for state timeouts and reset stale symbols.

        Returns list of symbols that were reset.
        """
        now = datetime.now()
        reset_symbols = []

        for symbol, state in list(self._states.items()):
            time_in_state = (now - state.entered_state_at).total_seconds()

            if state.state == MomentumState.CANDIDATE and time_in_state > self.CANDIDATE_TIMEOUT:
                self.transition(symbol, MomentumState.IDLE, TransitionReason.TIMEOUT)
                reset_symbols.append(symbol)

            elif state.state == MomentumState.IGNITING and time_in_state > self.IGNITING_TIMEOUT:
                self.transition(symbol, MomentumState.CANDIDATE, TransitionReason.TIMEOUT)
                reset_symbols.append(symbol)

            elif state.state == MomentumState.GATED and time_in_state > self.GATED_TIMEOUT:
                # Gating window expired without decision
                self.transition(symbol, MomentumState.IGNITING, TransitionReason.TIMEOUT)
                reset_symbols.append(symbol)

            elif state.state == MomentumState.IN_POSITION and time_in_state > self.IN_POSITION_TIMEOUT:
                # Auto-transition to MONITORING
                self.start_monitoring(symbol)

            elif state.state == MomentumState.EXITING and time_in_state > self.EXITING_TIMEOUT:
                # Force to cooldown if exiting too long
                self.complete_exit(symbol, state.pnl_pct)
                reset_symbols.append(symbol)

            elif state.state == MomentumState.COOLDOWN and time_in_state > self.COOLDOWN_DURATION:
                self.transition(symbol, MomentumState.IDLE, TransitionReason.COOLDOWN_EXPIRED)
                reset_symbols.append(symbol)

        return reset_symbols

    def add_to_candidate(self, symbol: str, reason: str = "") -> bool:
        """
        Add a symbol to CANDIDATE state (e.g., from scanner or news).

        Returns True if successful.
        """
        state = self.get_state(symbol)

        if state and state.state == MomentumState.COOLDOWN:
            logger.debug(f"Cannot add {symbol} to candidate - in cooldown")
            return False

        if not state or state.state == MomentumState.IDLE:
            return self.transition(
                symbol,
                MomentumState.CANDIDATE,
                TransitionReason.SCANNER_ADD,
                {'reason': reason}
            )

        return False  # Already in a higher state

    # Backwards compatibility aliases
    def add_to_attention(self, symbol: str, reason: str = "") -> bool:
        """Alias for add_to_candidate (backwards compatibility)"""
        return self.add_to_candidate(symbol, reason)

    def exit_position(self, symbol: str, reason: TransitionReason, pnl_pct: float = 0) -> bool:
        """Alias for signal_exit + complete_exit (backwards compatibility)"""
        state = self.get_state(symbol)
        if not state:
            return False

        # If in position/monitoring, signal exit
        if state.state in [MomentumState.IN_POSITION, MomentumState.MONITORING]:
            self.signal_exit(symbol, reason.value, pnl_pct=pnl_pct)

        # If in exiting, complete it
        if state.state == MomentumState.EXITING:
            return self.complete_exit(symbol, pnl_pct)

        return True

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

    def get_gated_symbols(self) -> List[str]:
        """Get all symbols currently in GATED state (awaiting approval)"""
        return self.get_symbols_in_state(MomentumState.GATED)

    def get_igniting_symbols(self) -> List[str]:
        """Get all symbols currently in IGNITING state"""
        return self.get_symbols_in_state(MomentumState.IGNITING)

    def get_position_symbols(self) -> List[str]:
        """Get all symbols currently in position (IN_POSITION or MONITORING)"""
        in_pos = self.get_symbols_in_state(MomentumState.IN_POSITION)
        monitoring = self.get_symbols_in_state(MomentumState.MONITORING)
        return in_pos + monitoring

    def get_monitoring_symbols(self) -> List[str]:
        """Get all symbols in MONITORING state"""
        return self.get_symbols_in_state(MomentumState.MONITORING)

    def get_transition_log(self, limit: int = 100) -> List[Dict]:
        """Get recent transitions"""
        return [t.to_dict() for t in self._transition_log[-limit:]]

    def get_summary(self) -> Dict:
        """Get summary of all states"""
        summary = {
            'total_symbols': len(self._states),
            'by_state': {},
            'by_owner': {},
            'gated': [],
            'igniting': [],
            'in_position': [],
            'monitoring': [],
            'exiting': [],
            'recent_transitions': len(self._transition_log)
        }

        for state in MomentumState:
            symbols = self.get_symbols_in_state(state)
            summary['by_state'][state.value] = len(symbols)

        for owner in StateOwner:
            symbols = self.get_symbols_by_owner(owner)
            summary['by_owner'][owner.value] = len(symbols)

        summary['gated'] = self.get_gated_symbols()
        summary['igniting'] = self.get_igniting_symbols()
        summary['in_position'] = self.get_symbols_in_state(MomentumState.IN_POSITION)
        summary['monitoring'] = self.get_monitoring_symbols()
        summary['exiting'] = self.get_symbols_in_state(MomentumState.EXITING)

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

    print("=" * 60)
    print("MOMENTUM STATE MACHINE v2 TEST (ChatGPT FSM Spec)")
    print("=" * 60)

    # Simulate a trade lifecycle
    symbol = "TEST"

    print(f"\n1. Adding {symbol} to CANDIDATE...")
    sm.add_to_candidate(symbol, "Scanner detected gapper")
    state = sm.get_state(symbol)
    print(f"   State: {state.state.value}, Owner: {state.owner.value}")

    print(f"\n2. Updating momentum score to 60 (IGNITING)...")
    sm.update_momentum(symbol, 60, details={'pattern': 'bull_flag'})
    state = sm.get_state(symbol)
    print(f"   State: {state.state.value}, Owner: {state.owner.value}")

    print(f"\n3. Updating momentum score to 75 (GATED)...")
    sm.update_momentum(symbol, 75)
    state = sm.get_state(symbol)
    print(f"   State: {state.state.value}, Owner: {state.owner.value}")

    print(f"\n4. Order filled - entering position...")
    sm.enter_position(symbol, entry_price=5.50, shares=100, stop_price=5.25, target_price=6.00)
    state = sm.get_state(symbol)
    print(f"   State: {state.state.value}, Owner: {state.owner.value}")

    print(f"\n5. Starting monitoring (Chronos takes over)...")
    sm.start_monitoring(symbol)
    state = sm.get_state(symbol)
    print(f"   State: {state.state.value}, Owner: {state.owner.value}")

    print(f"\n6. Updating position price...")
    sm.update_position(symbol, current_price=5.75)
    state = sm.get_state(symbol)
    print(f"   PnL: {state.pnl_pct:.2f}%")

    print(f"\n7. Exit signal from Chronos (trailing stop)...")
    sm.signal_exit(symbol, "TRAILING_STOP", confidence=0.85, pnl_pct=4.5)
    state = sm.get_state(symbol)
    print(f"   State: {state.state.value}, Exit Signal: {state.exit_signal}")

    print(f"\n8. Completing exit...")
    sm.complete_exit(symbol, pnl_pct=4.5)
    state = sm.get_state(symbol)
    print(f"   State: {state.state.value}")

    print(f"\n--- Summary ---")
    print(json.dumps(sm.get_summary(), indent=2))

    print(f"\n--- Transition Log ---")
    for t in sm.get_transition_log():
        print(f"  {t['from_state']} -> {t['to_state']} ({t['reason']}) [owner: {t['owner']}]")

    # Test veto
    print(f"\n--- VETO TEST ---")
    symbol2 = "VETO_TEST"
    sm.add_to_candidate(symbol2, "Test veto")
    sm.update_momentum(symbol2, 60)  # IGNITING
    print(f"Before veto: {sm.get_state(symbol2).state.value}")
    sm.update_momentum(symbol2, 75, vetoed=True, veto_reasons=["SPREAD_WIDE"])
    print(f"After veto: {sm.get_state(symbol2).state.value}")

    print("\n" + "=" * 60)
    print("STATE MACHINE v2 TEST COMPLETE")
    print("=" * 60)
