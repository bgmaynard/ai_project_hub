"""
Exit Imperatives (Structural Exit Triggers)
============================================
Mandatory exit conditions that override normal exit logic.

Each exit imperative has:
- Clear trigger condition
- Required logging (exit_reason, momentum_state, regime, unrealized_pnl)
- Priority level

Exit Types:
1. REGIME_SHIFT - Chronos moved to invalid regime
2. MOMENTUM_DECAY - DECAY + failure to reclaim VWAP
3. TIME_STOP - No progress after defined time window
4. VOLATILITY_SPIKE - Volatility exceeds threshold

All exits are logged with full context.
"""

import logging
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

EXIT_LOG_FILE = os.path.join(os.path.dirname(__file__), "exit_imperatives_log.json")


class ExitReason(Enum):
    """Structural exit reasons"""
    REGIME_SHIFT = "REGIME_SHIFT"
    MOMENTUM_DECAY = "MOMENTUM_DECAY"
    TIME_STOP = "TIME_STOP"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    STOP_LOSS = "STOP_LOSS"
    PROFIT_TARGET = "PROFIT_TARGET"
    TRAILING_STOP = "TRAILING_STOP"
    MANUAL = "MANUAL"


class ExitPriority(Enum):
    """Exit priority levels"""
    CRITICAL = 1    # Must exit immediately (regime shift, volatility spike)
    HIGH = 2        # Exit within seconds (stop loss, decay)
    NORMAL = 3      # Exit when convenient (time stop, profit target)
    LOW = 4         # Soft suggestion (manual)


@dataclass
class ExitImperativeResult:
    """Result of exit imperative check"""
    should_exit: bool
    reason: ExitReason = None
    priority: ExitPriority = ExitPriority.NORMAL
    details: str = ""

    # Context for logging
    momentum_state: str = ""
    regime: str = ""
    unrealized_pnl: float = 0
    unrealized_pnl_pct: float = 0
    time_in_position_seconds: float = 0
    current_volatility: float = 0

    def to_dict(self) -> Dict:
        return {
            'should_exit': self.should_exit,
            'reason': self.reason.value if self.reason else None,
            'priority': self.priority.value if self.priority else None,
            'details': self.details,
            'momentum_state': self.momentum_state,
            'regime': self.regime,
            'unrealized_pnl': round(self.unrealized_pnl, 2),
            'unrealized_pnl_pct': round(self.unrealized_pnl_pct, 2),
            'time_in_position_seconds': round(self.time_in_position_seconds, 1),
            'current_volatility': round(self.current_volatility, 4),
        }


@dataclass
class ExitLog:
    """Log entry for an exit trigger"""
    timestamp: str
    symbol: str
    exit_reason: str
    priority: int
    momentum_state: str
    regime: str
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_price: float
    exit_price: float
    shares: int
    time_in_position_seconds: float
    details: str


@dataclass
class PositionContext:
    """Context for a position being monitored"""
    symbol: str
    entry_price: float
    current_price: float
    shares: int
    entry_time: datetime
    stop_price: float = 0
    target_price: float = 0
    trailing_stop_price: float = 0
    high_since_entry: float = 0


class ExitImperativeEngine:
    """
    Evaluates mandatory exit conditions for open positions.

    Key principles:
    1. Exit checks are run every monitoring loop
    2. Higher priority exits override lower priority
    3. All exits are logged with full context
    4. Fail-closed: on error, suggest exit
    """

    # Configuration thresholds
    def __init__(self):
        self.config = {
            # Regime-based exits
            'invalid_regimes': ['TRENDING_DOWN', 'VOLATILE'],
            'regime_shift_exit': True,

            # Momentum decay
            'decay_vwap_required': True,    # Must lose VWAP to trigger
            'decay_consecutive_checks': 3,  # N consecutive DECAY checks

            # Time stop
            'time_stop_enabled': True,
            'max_hold_seconds': 300,        # 5 minutes default
            'time_stop_only_if_flat': True, # Only exit if not profitable

            # Volatility spike
            'volatility_exit_enabled': True,
            'max_volatility': 0.05,         # 5% annualized volatility
            'volatility_window_seconds': 60,

            # Stop loss / profit target
            'stop_loss_pct': 0.03,          # 3%
            'profit_target_pct': 0.03,      # 3%
            'trailing_stop_pct': 0.015,     # 1.5% trailing
        }

        # Tracking state
        self._decay_counts: Dict[str, int] = {}  # symbol -> consecutive decay count
        self._exit_log: List[ExitLog] = []
        self._monitored_positions: Dict[str, PositionContext] = {}

    def register_position(self, symbol: str, entry_price: float, shares: int,
                         stop_price: float = 0, target_price: float = 0):
        """Register a position for exit monitoring"""
        self._monitored_positions[symbol] = PositionContext(
            symbol=symbol,
            entry_price=entry_price,
            current_price=entry_price,
            shares=shares,
            entry_time=datetime.now(),
            stop_price=stop_price or entry_price * (1 - self.config['stop_loss_pct']),
            target_price=target_price or entry_price * (1 + self.config['profit_target_pct']),
            high_since_entry=entry_price,
        )
        self._decay_counts[symbol] = 0
        logger.info(f"Registered position for exit monitoring: {symbol} @ {entry_price}")

    def unregister_position(self, symbol: str):
        """Unregister a position"""
        if symbol in self._monitored_positions:
            del self._monitored_positions[symbol]
        if symbol in self._decay_counts:
            del self._decay_counts[symbol]

    def update_position(self, symbol: str, current_price: float):
        """Update current price for a monitored position"""
        if symbol in self._monitored_positions:
            pos = self._monitored_positions[symbol]
            pos.current_price = current_price

            # Update high watermark
            if current_price > pos.high_since_entry:
                pos.high_since_entry = current_price

                # Update trailing stop
                new_trail = current_price * (1 - self.config['trailing_stop_pct'])
                if new_trail > pos.trailing_stop_price:
                    pos.trailing_stop_price = new_trail

    def check_exit(self, symbol: str, momentum_state: str = "",
                   regime: str = "", volatility: float = 0,
                   above_vwap: bool = True) -> ExitImperativeResult:
        """
        Check all exit imperatives for a position.

        Returns the highest priority exit if any triggered.
        """
        if symbol not in self._monitored_positions:
            return ExitImperativeResult(should_exit=False, details="Not monitored")

        pos = self._monitored_positions[symbol]
        now = datetime.now()

        time_in_position = (now - pos.entry_time).total_seconds()
        pnl = (pos.current_price - pos.entry_price) * pos.shares
        pnl_pct = ((pos.current_price - pos.entry_price) / pos.entry_price) * 100 if pos.entry_price > 0 else 0

        results = []

        # 1. REGIME_SHIFT check
        if self.config['regime_shift_exit'] and regime in self.config['invalid_regimes']:
            results.append(ExitImperativeResult(
                should_exit=True,
                reason=ExitReason.REGIME_SHIFT,
                priority=ExitPriority.CRITICAL,
                details=f"Regime shifted to {regime}",
                momentum_state=momentum_state,
                regime=regime,
                unrealized_pnl=pnl,
                unrealized_pnl_pct=pnl_pct,
                time_in_position_seconds=time_in_position,
                current_volatility=volatility,
            ))

        # 2. VOLATILITY_SPIKE check
        if self.config['volatility_exit_enabled'] and volatility > self.config['max_volatility']:
            results.append(ExitImperativeResult(
                should_exit=True,
                reason=ExitReason.VOLATILITY_SPIKE,
                priority=ExitPriority.CRITICAL,
                details=f"Volatility {volatility:.2%} > {self.config['max_volatility']:.2%}",
                momentum_state=momentum_state,
                regime=regime,
                unrealized_pnl=pnl,
                unrealized_pnl_pct=pnl_pct,
                time_in_position_seconds=time_in_position,
                current_volatility=volatility,
            ))

        # 3. STOP_LOSS check
        if pos.current_price <= pos.stop_price:
            results.append(ExitImperativeResult(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                priority=ExitPriority.HIGH,
                details=f"Price {pos.current_price:.2f} <= stop {pos.stop_price:.2f}",
                momentum_state=momentum_state,
                regime=regime,
                unrealized_pnl=pnl,
                unrealized_pnl_pct=pnl_pct,
                time_in_position_seconds=time_in_position,
                current_volatility=volatility,
            ))

        # 4. TRAILING_STOP check
        if pos.trailing_stop_price > 0 and pos.current_price <= pos.trailing_stop_price:
            results.append(ExitImperativeResult(
                should_exit=True,
                reason=ExitReason.TRAILING_STOP,
                priority=ExitPriority.HIGH,
                details=f"Price {pos.current_price:.2f} <= trail {pos.trailing_stop_price:.2f}",
                momentum_state=momentum_state,
                regime=regime,
                unrealized_pnl=pnl,
                unrealized_pnl_pct=pnl_pct,
                time_in_position_seconds=time_in_position,
                current_volatility=volatility,
            ))

        # 5. MOMENTUM_DECAY check
        if momentum_state == "DECAY":
            self._decay_counts[symbol] = self._decay_counts.get(symbol, 0) + 1

            if (self._decay_counts[symbol] >= self.config['decay_consecutive_checks'] and
                (not self.config['decay_vwap_required'] or not above_vwap)):
                results.append(ExitImperativeResult(
                    should_exit=True,
                    reason=ExitReason.MOMENTUM_DECAY,
                    priority=ExitPriority.HIGH,
                    details=f"DECAY x{self._decay_counts[symbol]}, VWAP={above_vwap}",
                    momentum_state=momentum_state,
                    regime=regime,
                    unrealized_pnl=pnl,
                    unrealized_pnl_pct=pnl_pct,
                    time_in_position_seconds=time_in_position,
                    current_volatility=volatility,
                ))
        else:
            self._decay_counts[symbol] = 0

        # 6. TIME_STOP check
        if self.config['time_stop_enabled']:
            if time_in_position > self.config['max_hold_seconds']:
                if not self.config['time_stop_only_if_flat'] or pnl_pct <= 0:
                    results.append(ExitImperativeResult(
                        should_exit=True,
                        reason=ExitReason.TIME_STOP,
                        priority=ExitPriority.NORMAL,
                        details=f"Held {time_in_position:.0f}s > {self.config['max_hold_seconds']}s",
                        momentum_state=momentum_state,
                        regime=regime,
                        unrealized_pnl=pnl,
                        unrealized_pnl_pct=pnl_pct,
                        time_in_position_seconds=time_in_position,
                        current_volatility=volatility,
                    ))

        # 7. PROFIT_TARGET check
        if pos.current_price >= pos.target_price:
            results.append(ExitImperativeResult(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                priority=ExitPriority.NORMAL,
                details=f"Price {pos.current_price:.2f} >= target {pos.target_price:.2f}",
                momentum_state=momentum_state,
                regime=regime,
                unrealized_pnl=pnl,
                unrealized_pnl_pct=pnl_pct,
                time_in_position_seconds=time_in_position,
                current_volatility=volatility,
            ))

        # Return highest priority exit
        if results:
            results.sort(key=lambda r: r.priority.value)
            return results[0]

        return ExitImperativeResult(
            should_exit=False,
            details="No exit conditions met",
            momentum_state=momentum_state,
            regime=regime,
            unrealized_pnl=pnl,
            unrealized_pnl_pct=pnl_pct,
            time_in_position_seconds=time_in_position,
            current_volatility=volatility,
        )

    def log_exit(self, symbol: str, result: ExitImperativeResult, exit_price: float):
        """Log an exit that was triggered"""
        if symbol not in self._monitored_positions:
            return

        pos = self._monitored_positions[symbol]

        entry = ExitLog(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            exit_reason=result.reason.value if result.reason else "UNKNOWN",
            priority=result.priority.value if result.priority else 3,
            momentum_state=result.momentum_state,
            regime=result.regime,
            unrealized_pnl=result.unrealized_pnl,
            unrealized_pnl_pct=result.unrealized_pnl_pct,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            shares=pos.shares,
            time_in_position_seconds=result.time_in_position_seconds,
            details=result.details,
        )

        self._exit_log.append(entry)

        # Persist to file
        try:
            log_data = []
            if os.path.exists(EXIT_LOG_FILE):
                with open(EXIT_LOG_FILE, 'r') as f:
                    log_data = json.load(f)

            log_data.append(asdict(entry))

            # Keep last 500
            if len(log_data) > 500:
                log_data = log_data[-500:]

            with open(EXIT_LOG_FILE, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to log exit: {e}")

        logger.warning(
            f"EXIT IMPERATIVE: {symbol} | {result.reason.value} | "
            f"PnL: ${result.unrealized_pnl:.2f} ({result.unrealized_pnl_pct:.1f}%) | "
            f"{result.details}"
        )

    def get_exit_log(self, symbol: str = None, limit: int = 50) -> List[Dict]:
        """Get recent exit log"""
        log = self._exit_log
        if symbol:
            log = [e for e in log if e.symbol == symbol]
        return [asdict(e) for e in log[-limit:]]

    def get_exit_stats(self) -> Dict:
        """Get exit statistics"""
        if not self._exit_log:
            return {"total_exits": 0}

        # Count by reason
        reason_counts = {}
        reason_pnl = {}
        for entry in self._exit_log:
            reason = entry.exit_reason
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            reason_pnl[reason] = reason_pnl.get(reason, 0) + entry.unrealized_pnl

        return {
            "total_exits": len(self._exit_log),
            "by_reason": reason_counts,
            "pnl_by_reason": {k: round(v, 2) for k, v in reason_pnl.items()},
            "total_pnl": round(sum(e.unrealized_pnl for e in self._exit_log), 2),
            "avg_time_in_position": round(
                sum(e.time_in_position_seconds for e in self._exit_log) / len(self._exit_log), 1
            ),
        }

    def update_config(self, new_config: Dict):
        """Update exit configuration"""
        for key, value in new_config.items():
            if key in self.config:
                self.config[key] = value
        logger.info(f"Exit imperative config updated: {new_config}")

    def get_config(self) -> Dict:
        """Get current configuration"""
        return self.config.copy()

    def get_monitored_positions(self) -> List[Dict]:
        """Get all monitored positions"""
        return [
            {
                'symbol': pos.symbol,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'shares': pos.shares,
                'entry_time': pos.entry_time.isoformat(),
                'stop_price': pos.stop_price,
                'target_price': pos.target_price,
                'trailing_stop_price': pos.trailing_stop_price,
                'high_since_entry': pos.high_since_entry,
                'time_in_position': (datetime.now() - pos.entry_time).total_seconds(),
                'pnl': (pos.current_price - pos.entry_price) * pos.shares,
                'pnl_pct': ((pos.current_price - pos.entry_price) / pos.entry_price * 100)
                          if pos.entry_price > 0 else 0,
            }
            for pos in self._monitored_positions.values()
        ]


# Singleton instance
_engine: Optional[ExitImperativeEngine] = None


def get_exit_imperative_engine() -> ExitImperativeEngine:
    """Get singleton exit imperative engine"""
    global _engine
    if _engine is None:
        _engine = ExitImperativeEngine()
    return _engine


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    print("=" * 60)
    print("EXIT IMPERATIVES ENGINE TEST")
    print("=" * 60)

    engine = ExitImperativeEngine()

    # Register a position
    symbol = "TEST"
    engine.register_position(symbol, entry_price=5.00, shares=100,
                            stop_price=4.85, target_price=5.15)

    print("\n1. Initial position check...")
    engine.update_position(symbol, 5.02)
    result = engine.check_exit(symbol, momentum_state="CONFIRMED",
                               regime="TRENDING_UP", volatility=0.02)
    print(f"   Should exit: {result.should_exit}")
    print(f"   Details: {result.details}")

    print("\n2. Testing stop loss...")
    engine.update_position(symbol, 4.80)
    result = engine.check_exit(symbol, momentum_state="DECAY",
                               regime="TRENDING_UP", volatility=0.02)
    print(f"   Should exit: {result.should_exit}")
    print(f"   Reason: {result.reason.value if result.reason else 'None'}")
    print(f"   Details: {result.details}")

    # Reset for next test
    engine.unregister_position(symbol)
    engine.register_position(symbol, entry_price=5.00, shares=100)

    print("\n3. Testing regime shift...")
    engine.update_position(symbol, 5.05)
    result = engine.check_exit(symbol, momentum_state="CONFIRMED",
                               regime="VOLATILE", volatility=0.06)
    print(f"   Should exit: {result.should_exit}")
    print(f"   Reason: {result.reason.value if result.reason else 'None'}")
    print(f"   Priority: {result.priority.value if result.priority else 'None'}")

    print("\n4. Testing momentum decay...")
    engine.unregister_position(symbol)
    engine.register_position(symbol, entry_price=5.00, shares=100)

    # Simulate 3 DECAY checks with lost VWAP
    for i in range(3):
        engine.update_position(symbol, 4.95 - (i * 0.02))
        result = engine.check_exit(symbol, momentum_state="DECAY",
                                   regime="TRENDING_UP", above_vwap=False)
        print(f"   DECAY check {i+1}: should_exit={result.should_exit}")

    # Log the exit
    if result.should_exit:
        engine.log_exit(symbol, result, exit_price=4.91)

    print("\n5. Exit stats:")
    stats = engine.get_exit_stats()
    print(f"   {json.dumps(stats, indent=2)}")

    print("\n" + "=" * 60)
    print("EXIT IMPERATIVES ENGINE TEST COMPLETE")
    print("=" * 60)
