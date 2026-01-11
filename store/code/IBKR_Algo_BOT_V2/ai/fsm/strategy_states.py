"""
FSM Strategy States for ATS + 9 EMA Sniper
==========================================

States (from build doc):
- ATS_QUALIFIED: Symbol passed ATS qualification
- WAIT_9EMA_PULLBACK: Waiting for price to pull back to 9 EMA
- SNIPER_CONFIRMATION: Pullback detected, waiting for entry confirmation
- SCALP_ENTRY: Entry triggered, position opened
- EXIT_FAST: Exit signal triggered

Transitions are EVENT-DRIVEN, not time-driven.
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Callable

logger = logging.getLogger(__name__)


class SniperState(Enum):
    """
    FSM States for ATS + 9 EMA Sniper Strategy

    Flow:
    IDLE -> ATS_QUALIFIED -> WAIT_9EMA_PULLBACK -> SNIPER_CONFIRMATION -> SCALP_ENTRY -> EXIT_FAST -> COOLDOWN -> IDLE
    """
    IDLE = "IDLE"                           # Not trading this symbol
    ATS_QUALIFIED = "ATS_QUALIFIED"         # ATS approved, ready to hunt
    WAIT_9EMA_PULLBACK = "WAIT_9EMA_PULLBACK"  # Waiting for pullback to 9 EMA
    SNIPER_CONFIRMATION = "SNIPER_CONFIRMATION"  # In pullback zone, waiting for confirm
    SCALP_ENTRY = "SCALP_ENTRY"             # Position opened
    EXIT_FAST = "EXIT_FAST"                 # Exit triggered
    COOLDOWN = "COOLDOWN"                   # Post-trade cooldown
    DISABLED = "DISABLED"                   # Symbol disabled (2 failures)


@dataclass
class SniperStateData:
    """
    State data for a single symbol in the Sniper FSM.
    """
    symbol: str
    state: SniperState = SniperState.IDLE

    # ATS Qualification
    ats_score: float = 0.0
    ats_qualified_at: Optional[datetime] = None

    # Pullback tracking
    impulse_high: float = 0.0  # High before pullback started
    pullback_low: float = 0.0  # Low of pullback
    pullback_depth_pct: float = 0.0
    pullback_started_at: Optional[datetime] = None
    volume_decreasing: bool = False

    # Entry tracking
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    entry_reason: str = ""
    stop_loss: float = 0.0
    target_price: float = 0.0

    # Exit tracking
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""

    # Performance
    trade_result: str = ""  # WIN, LOSS, NO_TRADE
    pnl: float = 0.0

    # Anti-overtrading
    trades_today: int = 0
    failures_today: int = 0
    last_trade_time: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None

    # Timestamps
    state_entered_at: Optional[datetime] = None
    last_update: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "state": self.state.value,
            "ats_score": round(self.ats_score, 2),
            "ats_qualified_at": self.ats_qualified_at.isoformat() if self.ats_qualified_at else None,
            "impulse_high": round(self.impulse_high, 4),
            "pullback_low": round(self.pullback_low, 4),
            "pullback_depth_pct": round(self.pullback_depth_pct, 2),
            "entry_price": round(self.entry_price, 4),
            "entry_reason": self.entry_reason,
            "stop_loss": round(self.stop_loss, 4),
            "trade_result": self.trade_result,
            "trades_today": self.trades_today,
            "failures_today": self.failures_today,
            "state_entered_at": self.state_entered_at.isoformat() if self.state_entered_at else None,
        }


class SniperFSM:
    """
    Finite State Machine for ATS + 9 EMA Sniper Strategy.

    Core rules:
    - NO PULLBACK = NO TRADE
    - NO CONFIRMATION = NO TRADE
    - FLAT = SUCCESS
    """

    # Config
    MAX_TRADES_PER_SYMBOL = 2
    MAX_FAILURES_BEFORE_DISABLE = 2
    COOLDOWN_MINUTES = 5
    MAX_PULLBACK_PCT = 30.0  # Max pullback depth as % of impulse

    def __init__(self):
        self._states: Dict[str, SniperStateData] = {}
        self._on_state_change: Optional[Callable] = None
        self._on_trade_event: Optional[Callable] = None

    def set_callbacks(self,
                      on_state_change: Callable = None,
                      on_trade_event: Callable = None):
        """Set event callbacks"""
        self._on_state_change = on_state_change
        self._on_trade_event = on_trade_event

    def get_state(self, symbol: str) -> SniperStateData:
        """Get or create state for symbol"""
        symbol = symbol.upper()
        if symbol not in self._states:
            self._states[symbol] = SniperStateData(symbol=symbol)
        return self._states[symbol]

    def transition(self, symbol: str, new_state: SniperState, reason: str = "") -> bool:
        """
        Transition symbol to new state.

        Returns True if transition was valid.
        """
        symbol = symbol.upper()
        state_data = self.get_state(symbol)
        old_state = state_data.state

        # Validate transition
        if not self._is_valid_transition(old_state, new_state):
            logger.warning(
                f"[SNIPER_FSM] Invalid transition {symbol}: {old_state.value} -> {new_state.value}"
            )
            return False

        # Apply transition
        state_data.state = new_state
        state_data.state_entered_at = datetime.now()
        state_data.last_update = datetime.now()

        logger.info(
            f"[SNIPER_FSM] {symbol}: {old_state.value} -> {new_state.value} ({reason})"
        )

        # Fire callback
        if self._on_state_change:
            try:
                self._on_state_change(symbol, old_state, new_state, reason)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

        return True

    def _is_valid_transition(self, from_state: SniperState, to_state: SniperState) -> bool:
        """Check if state transition is valid"""
        valid_transitions = {
            SniperState.IDLE: [SniperState.ATS_QUALIFIED, SniperState.DISABLED],
            SniperState.ATS_QUALIFIED: [SniperState.WAIT_9EMA_PULLBACK, SniperState.IDLE],
            SniperState.WAIT_9EMA_PULLBACK: [SniperState.SNIPER_CONFIRMATION, SniperState.IDLE],
            SniperState.SNIPER_CONFIRMATION: [SniperState.SCALP_ENTRY, SniperState.WAIT_9EMA_PULLBACK, SniperState.IDLE],
            SniperState.SCALP_ENTRY: [SniperState.EXIT_FAST],
            SniperState.EXIT_FAST: [SniperState.COOLDOWN, SniperState.DISABLED],
            SniperState.COOLDOWN: [SniperState.IDLE, SniperState.ATS_QUALIFIED],
            SniperState.DISABLED: [SniperState.IDLE],  # Manual reset only
        }

        allowed = valid_transitions.get(from_state, [])
        return to_state in allowed

    # ========================================
    # STATE TRANSITION METHODS
    # ========================================

    def qualify_with_ats(self, symbol: str, ats_score: float) -> bool:
        """
        Symbol qualified by ATS. Move to ATS_QUALIFIED.

        Returns True if transition successful.
        """
        state = self.get_state(symbol)

        # Check if can qualify
        if state.state == SniperState.DISABLED:
            logger.debug(f"[SNIPER_FSM] {symbol} is DISABLED, cannot qualify")
            return False

        if state.trades_today >= self.MAX_TRADES_PER_SYMBOL:
            logger.debug(f"[SNIPER_FSM] {symbol} hit max trades ({self.MAX_TRADES_PER_SYMBOL})")
            return False

        # Transition
        state.ats_score = ats_score
        state.ats_qualified_at = datetime.now()

        return self.transition(symbol, SniperState.ATS_QUALIFIED, f"ATS score: {ats_score:.2f}")

    def start_pullback_watch(self, symbol: str, impulse_high: float) -> bool:
        """
        Start watching for 9 EMA pullback.
        """
        state = self.get_state(symbol)

        if state.state != SniperState.ATS_QUALIFIED:
            return False

        state.impulse_high = impulse_high
        state.pullback_low = impulse_high  # Will be updated as price drops
        state.pullback_started_at = datetime.now()

        return self.transition(symbol, SniperState.WAIT_9EMA_PULLBACK, "Watching for pullback")

    def detect_pullback(self, symbol: str, current_price: float, pullback_low: float) -> bool:
        """
        Pullback detected, move to confirmation phase.
        """
        state = self.get_state(symbol)

        if state.state != SniperState.WAIT_9EMA_PULLBACK:
            return False

        # Calculate pullback depth
        if state.impulse_high > 0:
            impulse_move = state.impulse_high - pullback_low
            pullback_depth = (state.impulse_high - current_price) / state.impulse_high * 100
            state.pullback_depth_pct = pullback_depth
            state.pullback_low = pullback_low

            # Check max pullback
            if pullback_depth > self.MAX_PULLBACK_PCT:
                logger.info(f"[SNIPER_FSM] {symbol} pullback too deep: {pullback_depth:.1f}%")
                return self.transition(symbol, SniperState.IDLE, "Pullback too deep")

        return self.transition(symbol, SniperState.SNIPER_CONFIRMATION, f"Pullback {state.pullback_depth_pct:.1f}%")

    def confirm_entry(self, symbol: str, entry_price: float, stop_loss: float,
                      target: float, reason: str) -> bool:
        """
        Entry confirmed. Execute scalp entry.
        """
        state = self.get_state(symbol)

        if state.state != SniperState.SNIPER_CONFIRMATION:
            return False

        state.entry_price = entry_price
        state.stop_loss = stop_loss
        state.target_price = target
        state.entry_reason = reason
        state.entry_time = datetime.now()
        state.trades_today += 1
        state.last_trade_time = datetime.now()

        # Fire trade event
        if self._on_trade_event:
            try:
                self._on_trade_event({
                    "event": "SNIPER_ENTRY",
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "target": target,
                    "reason": reason,
                    "ats_score": state.ats_score,
                    "pullback_depth_pct": state.pullback_depth_pct
                })
            except Exception as e:
                logger.error(f"Trade event callback error: {e}")

        return self.transition(symbol, SniperState.SCALP_ENTRY, reason)

    def trigger_exit(self, symbol: str, exit_price: float, reason: str, is_win: bool) -> bool:
        """
        Exit triggered. Close position.
        """
        state = self.get_state(symbol)

        if state.state != SniperState.SCALP_ENTRY:
            return False

        state.exit_price = exit_price
        state.exit_time = datetime.now()
        state.exit_reason = reason
        state.trade_result = "WIN" if is_win else "LOSS"
        state.pnl = exit_price - state.entry_price

        if not is_win:
            state.failures_today += 1

        # Fire trade event
        if self._on_trade_event:
            try:
                self._on_trade_event({
                    "event": "SNIPER_EXIT",
                    "symbol": symbol,
                    "entry_price": state.entry_price,
                    "exit_price": exit_price,
                    "pnl": state.pnl,
                    "result": state.trade_result,
                    "reason": reason
                })
            except Exception as e:
                logger.error(f"Trade event callback error: {e}")

        return self.transition(symbol, SniperState.EXIT_FAST, reason)

    def start_cooldown(self, symbol: str) -> bool:
        """
        Start cooldown period after trade.
        """
        state = self.get_state(symbol)

        if state.state != SniperState.EXIT_FAST:
            return False

        # Check for disable condition
        if state.failures_today >= self.MAX_FAILURES_BEFORE_DISABLE:
            logger.warning(f"[SNIPER_FSM] {symbol} disabled after {state.failures_today} failures")
            return self.transition(symbol, SniperState.DISABLED, "Max failures reached")

        from datetime import timedelta
        state.cooldown_until = datetime.now() + timedelta(minutes=self.COOLDOWN_MINUTES)

        return self.transition(symbol, SniperState.COOLDOWN, f"Cooldown {self.COOLDOWN_MINUTES}min")

    def end_cooldown(self, symbol: str) -> bool:
        """
        End cooldown period.
        """
        state = self.get_state(symbol)

        if state.state != SniperState.COOLDOWN:
            return False

        # Check if cooldown has elapsed
        if state.cooldown_until and datetime.now() < state.cooldown_until:
            return False  # Still in cooldown

        # Reset for next trade
        state.impulse_high = 0.0
        state.pullback_low = 0.0
        state.pullback_depth_pct = 0.0
        state.entry_price = 0.0
        state.exit_price = 0.0

        return self.transition(symbol, SniperState.IDLE, "Cooldown complete")

    def cancel(self, symbol: str, reason: str) -> bool:
        """
        Cancel current state, return to IDLE.
        """
        state = self.get_state(symbol)

        if state.state in [SniperState.IDLE, SniperState.DISABLED]:
            return False

        # Log NO_TRADE result
        if state.state in [SniperState.ATS_QUALIFIED, SniperState.WAIT_9EMA_PULLBACK, SniperState.SNIPER_CONFIRMATION]:
            state.trade_result = "NO_TRADE"

            if self._on_trade_event:
                try:
                    self._on_trade_event({
                        "event": "ATS_9EMA_SNIPER_ATTEMPT",
                        "symbol": symbol,
                        "ats_score": state.ats_score,
                        "pullback_depth_pct": state.pullback_depth_pct,
                        "result": "NO_TRADE",
                        "reason": reason
                    })
                except Exception as e:
                    logger.error(f"Trade event callback error: {e}")

        return self.transition(symbol, SniperState.IDLE, reason)

    def reset_daily(self, symbol: str = None):
        """Reset daily counters for symbol or all symbols"""
        if symbol:
            state = self.get_state(symbol)
            state.trades_today = 0
            state.failures_today = 0
            if state.state == SniperState.DISABLED:
                state.state = SniperState.IDLE
        else:
            for sym in self._states:
                self.reset_daily(sym)

    def get_all_states(self) -> Dict[str, Dict]:
        """Get all symbol states"""
        return {sym: state.to_dict() for sym, state in self._states.items()}

    def get_active_symbols(self) -> List[str]:
        """Get symbols currently in active states"""
        active_states = [
            SniperState.ATS_QUALIFIED,
            SniperState.WAIT_9EMA_PULLBACK,
            SniperState.SNIPER_CONFIRMATION,
            SniperState.SCALP_ENTRY
        ]
        return [sym for sym, state in self._states.items() if state.state in active_states]


# Singleton instance
_sniper_fsm: Optional[SniperFSM] = None


def get_sniper_fsm() -> SniperFSM:
    """Get singleton Sniper FSM instance"""
    global _sniper_fsm
    if _sniper_fsm is None:
        _sniper_fsm = SniperFSM()
    return _sniper_fsm
