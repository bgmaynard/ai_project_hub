"""
Safe Activation Mode
====================
Reduced-risk trading mode with kill-switches and observability.

Features:
- Position size multiplier: 0.25-0.50
- One strategy active at a time
- Optional symbol whitelist
- Kill-switch triggers:
  - Abnormal veto spikes
  - Repeated forced exits
  - Market data feed anomalies

Observability Export:
- Strategy enable/disable timeline
- Veto histogram by reason
- Momentum state transition counts
- Exit reason counts
- Time spent in each regime

Activation is EXPLICIT - never default.
"""

import logging
import json
import csv
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Set
import pytz
from enum import Enum
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)

# File paths
ACTIVATION_STATE_FILE = os.path.join(os.path.dirname(__file__), "safe_activation_state.json")
OBSERVABILITY_EXPORT_FILE = os.path.join(os.path.dirname(__file__), "observability_export.json")


class ActivationMode(Enum):
    """Trading activation modes"""
    DISABLED = "DISABLED"           # No trading at all
    SAFE = "SAFE"                   # Reduced risk mode
    NORMAL = "NORMAL"               # Full trading (not implemented here)


class GovernorHealthState(Enum):
    """Governor system health states for dashboard display"""
    OFFLINE = "OFFLINE"             # System not running or critical failure
    STANDBY = "STANDBY"             # Healthy but waiting (market closed)
    CONNECTED = "CONNECTED"         # All services connected, not yet active
    ACTIVE = "ACTIVE"               # Fully operational and trading enabled


class KillSwitchReason(Enum):
    """Reasons for kill-switch activation"""
    VETO_SPIKE = "VETO_SPIKE"
    FORCED_EXITS = "FORCED_EXITS"
    DATA_ANOMALY = "DATA_ANOMALY"
    MANUAL = "MANUAL"
    MAX_DAILY_LOSS = "MAX_DAILY_LOSS"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"


@dataclass
class KillSwitchEvent:
    """Kill-switch trigger event"""
    timestamp: str
    reason: str
    details: str
    metrics_snapshot: Dict


@dataclass
class StrategyEvent:
    """Strategy enable/disable event"""
    timestamp: str
    strategy_id: str
    action: str  # ENABLE, DISABLE, SUPPRESS
    reason: str
    duration_seconds: float = 0  # How long in this state


@dataclass
class SafeActivationConfig:
    """Configuration for safe activation mode"""
    enabled: bool = False
    mode: ActivationMode = ActivationMode.DISABLED

    # Position sizing
    position_size_multiplier: float = 0.25  # 25% of normal size
    max_position_size_usd: float = 500      # Max $500 per trade

    # Strategy limits
    max_concurrent_strategies: int = 1      # One at a time
    active_strategy: str = ""               # Currently active strategy

    # Symbol whitelist (empty = all allowed)
    symbol_whitelist: List[str] = field(default_factory=list)

    # Kill-switch thresholds
    veto_spike_threshold: int = 20          # Vetoes in 5 min window
    veto_spike_window_seconds: int = 300
    max_forced_exits: int = 5               # In 30 min window
    forced_exit_window_seconds: int = 1800
    max_daily_loss_usd: float = 100         # Max $100 daily loss
    data_stale_seconds: int = 30            # Market data staleness

    # Recovery
    kill_switch_cooldown_seconds: int = 1800  # 30 min cooldown

    def to_dict(self) -> Dict:
        return {
            'enabled': self.enabled,
            'mode': self.mode.value,
            'position_size_multiplier': self.position_size_multiplier,
            'max_position_size_usd': self.max_position_size_usd,
            'max_concurrent_strategies': self.max_concurrent_strategies,
            'active_strategy': self.active_strategy,
            'symbol_whitelist': self.symbol_whitelist,
            'veto_spike_threshold': self.veto_spike_threshold,
            'max_forced_exits': self.max_forced_exits,
            'max_daily_loss_usd': self.max_daily_loss_usd,
            'data_stale_seconds': self.data_stale_seconds,
            'kill_switch_cooldown_seconds': self.kill_switch_cooldown_seconds,
        }


@dataclass
class RuntimeMetrics:
    """Runtime metrics for observability"""
    # Vetoes
    veto_count_total: int = 0
    veto_by_reason: Dict[str, int] = field(default_factory=dict)
    veto_timestamps: List[str] = field(default_factory=list)

    # Momentum states
    momentum_state_counts: Dict[str, int] = field(default_factory=dict)
    momentum_transitions: int = 0

    # Exits
    exit_count_total: int = 0
    exit_by_reason: Dict[str, int] = field(default_factory=dict)
    forced_exit_count: int = 0
    forced_exit_timestamps: List[str] = field(default_factory=list)

    # Regimes
    regime_time_seconds: Dict[str, float] = field(default_factory=dict)
    current_regime: str = "UNKNOWN"
    regime_start_time: str = ""

    # Strategies
    strategy_events: List[Dict] = field(default_factory=list)
    strategy_state: Dict[str, str] = field(default_factory=dict)  # id -> ENABLED/DISABLED

    # P&L
    daily_pnl: float = 0
    trade_count: int = 0
    win_count: int = 0

    # Data quality
    last_data_update: str = ""
    data_gaps_count: int = 0


class SafeActivationMode:
    """
    Manages safe/reduced-risk trading mode.

    Key principles:
    1. Activation is EXPLICIT (never default to trading)
    2. Kill-switches are fail-closed (stop trading on anomaly)
    3. All state changes logged for observability
    4. Position sizing is strictly limited
    """

    def __init__(self):
        self.config = SafeActivationConfig()
        self.metrics = RuntimeMetrics()

        self._kill_switch_active: bool = False
        self._kill_switch_reason: Optional[KillSwitchReason] = None
        self._kill_switch_time: Optional[datetime] = None
        self._kill_switch_events: List[KillSwitchEvent] = []

        self._lock = threading.Lock()
        self._load_state()

    def _load_state(self):
        """Load persisted state"""
        try:
            if os.path.exists(ACTIVATION_STATE_FILE):
                with open(ACTIVATION_STATE_FILE, 'r') as f:
                    data = json.load(f)

                # Restore config
                cfg = data.get('config', {})
                self.config.enabled = cfg.get('enabled', False)
                self.config.mode = ActivationMode(cfg.get('mode', 'DISABLED'))
                self.config.position_size_multiplier = cfg.get('position_size_multiplier', 0.25)
                self.config.symbol_whitelist = cfg.get('symbol_whitelist', [])
                self.config.active_strategy = cfg.get('active_strategy', '')

                # Restore metrics
                metrics = data.get('metrics', {})
                self.metrics.daily_pnl = metrics.get('daily_pnl', 0)
                self.metrics.trade_count = metrics.get('trade_count', 0)
                self.metrics.veto_count_total = metrics.get('veto_count_total', 0)

                logger.info(f"Loaded safe activation state: mode={self.config.mode.value}")
        except Exception as e:
            logger.error(f"Failed to load safe activation state: {e}")

    def _save_state(self):
        """Persist state"""
        try:
            data = {
                'config': self.config.to_dict(),
                'metrics': {
                    'daily_pnl': self.metrics.daily_pnl,
                    'trade_count': self.metrics.trade_count,
                    'veto_count_total': self.metrics.veto_count_total,
                    'exit_count_total': self.metrics.exit_count_total,
                },
                'kill_switch_active': self._kill_switch_active,
                'kill_switch_reason': self._kill_switch_reason.value if self._kill_switch_reason else None,
                'last_updated': datetime.now().isoformat()
            }

            with open(ACTIVATION_STATE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save safe activation state: {e}")

    # ==================== Activation Control ====================

    def activate(self, mode: ActivationMode = ActivationMode.SAFE,
                 position_multiplier: float = 0.25,
                 symbol_whitelist: List[str] = None) -> Dict:
        """
        Explicitly activate safe trading mode.

        Args:
            mode: SAFE or NORMAL
            position_multiplier: 0.25 = 25% size
            symbol_whitelist: Optional list of allowed symbols

        Returns: Activation result
        """
        with self._lock:
            if self._kill_switch_active:
                return {
                    'success': False,
                    'error': f'Kill-switch active: {self._kill_switch_reason.value}'
                }

            self.config.enabled = True
            self.config.mode = mode
            self.config.position_size_multiplier = max(0.1, min(0.5, position_multiplier))

            if symbol_whitelist:
                self.config.symbol_whitelist = [s.upper() for s in symbol_whitelist]

            self._save_state()

            logger.warning(
                f"SAFE ACTIVATION: Mode={mode.value}, "
                f"Size={position_multiplier:.0%}, "
                f"Whitelist={len(self.config.symbol_whitelist)} symbols"
            )

            return {
                'success': True,
                'mode': mode.value,
                'position_multiplier': position_multiplier,
                'symbol_whitelist': self.config.symbol_whitelist
            }

    def deactivate(self, reason: str = "manual") -> Dict:
        """Deactivate trading"""
        with self._lock:
            self.config.enabled = False
            self.config.mode = ActivationMode.DISABLED
            self.config.active_strategy = ""
            self._save_state()

            logger.warning(f"SAFE DEACTIVATION: {reason}")

            return {'success': True, 'reason': reason}

    def is_active(self) -> bool:
        """Check if trading is active"""
        return (self.config.enabled and
                self.config.mode != ActivationMode.DISABLED and
                not self._kill_switch_active and
                self.is_trading_window())

    def is_trading_window(self) -> bool:
        """
        Check if we're in a valid trading window.

        Weekdays only (Mon-Fri):
        - Pre-market: 4:00 AM - 9:30 AM ET -> Trading allowed
        - Market hours: 9:30 AM - 4:00 PM ET -> Trading allowed
        - After hours: 4:00 PM - 8:00 PM ET -> Trading allowed

        Weekends: No trading
        """
        try:
            et_tz = pytz.timezone('US/Eastern')
            now_et = datetime.now(et_tz)
            now_time = now_et.time()

            # Check day of week (0=Monday, 6=Sunday)
            day_of_week = now_et.weekday()
            if day_of_week >= 5:  # Saturday (5) or Sunday (6)
                return False

            # Pre-market: 4:00 AM - 9:30 AM ET
            if time(4, 0) <= now_time < time(9, 30):
                return True

            # Regular market hours: 9:30 AM - 4:00 PM ET
            if time(9, 30) <= now_time < time(16, 0):
                return True

            # After hours: 4:00 PM - 8:00 PM ET
            if time(16, 0) <= now_time < time(20, 0):
                return True

            # Outside trading window
            return False

        except Exception as e:
            logger.error(f"Error checking trading window: {e}")
            # Fail safe: don't trade if we can't determine time
            return False

    def get_trading_window_status(self) -> dict:
        """Get current trading window status for UI display"""
        try:
            et_tz = pytz.timezone('US/Eastern')
            now_et = datetime.now(et_tz)
            now_time = now_et.time()
            day_of_week = now_et.weekday()
            day_name = now_et.strftime('%A')

            # Check weekend first
            if day_of_week >= 5:  # Saturday or Sunday
                window = "WEEKEND"
                window_detail = f"Market closed ({day_name})"
            elif time(4, 0) <= now_time < time(9, 30):
                window = "PRE_MARKET"
                window_detail = "Pre-market session (4:00 AM - 9:30 AM ET)"
            elif time(9, 30) <= now_time < time(16, 0):
                window = "MARKET_OPEN"
                window_detail = "Regular market hours (9:30 AM - 4:00 PM ET)"
            elif time(16, 0) <= now_time < time(20, 0):
                window = "AFTER_HOURS"
                window_detail = "After-hours session (4:00 PM - 8:00 PM ET)"
            else:
                window = "CLOSED"
                window_detail = "Market closed (outside trading hours)"

            return {
                'current_et_time': now_et.strftime('%I:%M %p ET'),
                'current_day': day_name,
                'window': window,
                'window_detail': window_detail,
                'trading_allowed': self.is_trading_window(),
                'is_active': self.is_active()
            }
        except Exception as e:
            return {
                'error': str(e),
                'trading_allowed': False,
                'is_active': False
            }

    def get_ai_posture(self) -> dict:
        """
        Get current AI trading posture for Governor UI.

        Posture reflects: mode + time window + system health
        """
        window_status = self.get_trading_window_status()

        # Determine posture
        if not self.config.enabled:
            posture = "DISABLED"
            posture_detail = "Trading disabled by configuration"
        elif self._kill_switch_active:
            posture = "KILLED"
            posture_detail = f"Kill switch active: {self._kill_switch_reason.value if self._kill_switch_reason else 'unknown'}"
        elif not window_status.get('trading_allowed', False):
            posture = "OUTSIDE_HOURS"
            posture_detail = window_status.get('window_detail', 'Outside trading hours')
        elif self.config.mode == ActivationMode.SAFE:
            posture = "LIVE_PAPER"
            posture_detail = "Paper trading active (SAFE mode)"
        elif self.config.mode == ActivationMode.NORMAL:
            posture = "LIVE_REAL"
            posture_detail = "Live trading active (NORMAL mode)"
        else:
            posture = "STANDBY"
            posture_detail = "System ready, awaiting activation"

        return {
            'posture': posture,
            'posture_detail': posture_detail,
            'mode': self.config.mode.value,
            'trading_window': window_status,
            'can_trade': self.is_active(),
            'kill_switch_active': self._kill_switch_active
        }

    # ==================== Trade Gating ====================

    def can_trade(self, symbol: str, strategy_id: str, size_usd: float) -> tuple:
        """
        Check if a trade is allowed.

        Returns: (allowed: bool, reason: str, adjusted_size: float)
        """
        with self._lock:
            # Check activation
            if not self.config.enabled:
                return False, "Trading not activated", 0

            if self._kill_switch_active:
                return False, f"Kill-switch: {self._kill_switch_reason.value}", 0

            # Check whitelist
            if self.config.symbol_whitelist and symbol not in self.config.symbol_whitelist:
                return False, f"Symbol {symbol} not in whitelist", 0

            # Check strategy limits
            if self.config.active_strategy and self.config.active_strategy != strategy_id:
                return False, f"Strategy {self.config.active_strategy} already active", 0

            # Check daily loss limit
            if self.metrics.daily_pnl <= -self.config.max_daily_loss_usd:
                self._trigger_kill_switch(KillSwitchReason.MAX_DAILY_LOSS,
                                         f"Daily loss ${abs(self.metrics.daily_pnl):.2f}")
                return False, "Daily loss limit reached", 0

            # Adjust position size
            adjusted_size = min(
                size_usd * self.config.position_size_multiplier,
                self.config.max_position_size_usd
            )

            # Mark strategy as active
            self.config.active_strategy = strategy_id

            return True, "Approved", adjusted_size

    def release_strategy(self, strategy_id: str):
        """Release a strategy (after trade completes)"""
        with self._lock:
            if self.config.active_strategy == strategy_id:
                self.config.active_strategy = ""

    # ==================== Kill-Switch ====================

    def _trigger_kill_switch(self, reason: KillSwitchReason, details: str):
        """Trigger the kill-switch"""
        if self._kill_switch_active:
            return  # Already triggered

        self._kill_switch_active = True
        self._kill_switch_reason = reason
        self._kill_switch_time = datetime.now()

        event = KillSwitchEvent(
            timestamp=datetime.now().isoformat(),
            reason=reason.value,
            details=details,
            metrics_snapshot={
                'daily_pnl': self.metrics.daily_pnl,
                'veto_count': self.metrics.veto_count_total,
                'exit_count': self.metrics.exit_count_total,
                'trade_count': self.metrics.trade_count,
            }
        )
        self._kill_switch_events.append(event)

        self._save_state()

        logger.critical(f"KILL-SWITCH TRIGGERED: {reason.value} - {details}")

    def check_kill_switch_conditions(self):
        """Check all kill-switch conditions"""
        now = datetime.now()

        # Check veto spike
        veto_window = now - timedelta(seconds=self.config.veto_spike_window_seconds)
        recent_vetoes = sum(
            1 for ts in self.metrics.veto_timestamps
            if datetime.fromisoformat(ts) > veto_window
        )

        if recent_vetoes >= self.config.veto_spike_threshold:
            self._trigger_kill_switch(
                KillSwitchReason.VETO_SPIKE,
                f"{recent_vetoes} vetoes in {self.config.veto_spike_window_seconds}s"
            )
            return

        # Check forced exits
        exit_window = now - timedelta(seconds=self.config.forced_exit_window_seconds)
        recent_forced = sum(
            1 for ts in self.metrics.forced_exit_timestamps
            if datetime.fromisoformat(ts) > exit_window
        )

        if recent_forced >= self.config.max_forced_exits:
            self._trigger_kill_switch(
                KillSwitchReason.FORCED_EXITS,
                f"{recent_forced} forced exits in {self.config.forced_exit_window_seconds}s"
            )
            return

        # Check data staleness
        if self.metrics.last_data_update:
            last_update = datetime.fromisoformat(self.metrics.last_data_update)
            if (now - last_update).total_seconds() > self.config.data_stale_seconds:
                self._trigger_kill_switch(
                    KillSwitchReason.DATA_ANOMALY,
                    f"No data for {(now - last_update).total_seconds():.0f}s"
                )

    def reset_kill_switch(self, force: bool = False) -> Dict:
        """Reset kill-switch (manual or after cooldown)"""
        with self._lock:
            if not self._kill_switch_active:
                return {'success': True, 'message': 'Kill-switch not active'}

            if not force:
                # Check cooldown
                if self._kill_switch_time:
                    elapsed = (datetime.now() - self._kill_switch_time).total_seconds()
                    if elapsed < self.config.kill_switch_cooldown_seconds:
                        remaining = self.config.kill_switch_cooldown_seconds - elapsed
                        return {
                            'success': False,
                            'error': f'Cooldown active: {remaining:.0f}s remaining'
                        }

            self._kill_switch_active = False
            self._kill_switch_reason = None
            self._kill_switch_time = None
            self._save_state()

            logger.warning("Kill-switch RESET")
            return {'success': True, 'message': 'Kill-switch reset'}

    # ==================== Metrics Recording ====================

    def record_veto(self, reason: str):
        """Record a veto event"""
        with self._lock:
            self.metrics.veto_count_total += 1
            self.metrics.veto_by_reason[reason] = self.metrics.veto_by_reason.get(reason, 0) + 1
            self.metrics.veto_timestamps.append(datetime.now().isoformat())

            # Keep last 100 timestamps
            if len(self.metrics.veto_timestamps) > 100:
                self.metrics.veto_timestamps = self.metrics.veto_timestamps[-100:]

            self.check_kill_switch_conditions()

    def record_exit(self, reason: str, is_forced: bool = False, pnl: float = 0):
        """Record an exit event"""
        with self._lock:
            self.metrics.exit_count_total += 1
            self.metrics.exit_by_reason[reason] = self.metrics.exit_by_reason.get(reason, 0) + 1
            self.metrics.daily_pnl += pnl

            if is_forced:
                self.metrics.forced_exit_count += 1
                self.metrics.forced_exit_timestamps.append(datetime.now().isoformat())

                if len(self.metrics.forced_exit_timestamps) > 50:
                    self.metrics.forced_exit_timestamps = self.metrics.forced_exit_timestamps[-50:]

            self.check_kill_switch_conditions()
            self._save_state()

    def record_trade(self, is_win: bool, pnl: float):
        """Record a completed trade"""
        with self._lock:
            self.metrics.trade_count += 1
            if is_win:
                self.metrics.win_count += 1
            self.metrics.daily_pnl += pnl
            self._save_state()

    def record_momentum_state(self, state: str):
        """Record momentum state observation"""
        self.metrics.momentum_state_counts[state] = \
            self.metrics.momentum_state_counts.get(state, 0) + 1

    def record_regime_change(self, new_regime: str):
        """Record regime change"""
        now = datetime.now()

        # Record time in previous regime
        if self.metrics.regime_start_time:
            prev_start = datetime.fromisoformat(self.metrics.regime_start_time)
            duration = (now - prev_start).total_seconds()
            prev_regime = self.metrics.current_regime
            self.metrics.regime_time_seconds[prev_regime] = \
                self.metrics.regime_time_seconds.get(prev_regime, 0) + duration

        self.metrics.current_regime = new_regime
        self.metrics.regime_start_time = now.isoformat()

    def record_strategy_event(self, strategy_id: str, action: str, reason: str):
        """Record strategy enable/disable event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'strategy_id': strategy_id,
            'action': action,
            'reason': reason
        }
        self.metrics.strategy_events.append(event)
        self.metrics.strategy_state[strategy_id] = action

        # Keep last 100 events
        if len(self.metrics.strategy_events) > 100:
            self.metrics.strategy_events = self.metrics.strategy_events[-100:]

    def record_data_update(self):
        """Record that market data was received"""
        self.metrics.last_data_update = datetime.now().isoformat()

    # ==================== Observability Export ====================

    def export_observability(self, format: str = "json") -> Any:
        """
        Export runtime observability data.

        Args:
            format: "json" or "csv"

        Returns:
            JSON dict or CSV string
        """
        export_data = {
            'export_time': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'status': {
                'is_active': self.is_active(),
                'kill_switch_active': self._kill_switch_active,
                'kill_switch_reason': self._kill_switch_reason.value if self._kill_switch_reason else None,
            },
            'metrics': {
                'vetoes': {
                    'total': self.metrics.veto_count_total,
                    'by_reason': self.metrics.veto_by_reason,
                },
                'exits': {
                    'total': self.metrics.exit_count_total,
                    'by_reason': self.metrics.exit_by_reason,
                    'forced_count': self.metrics.forced_exit_count,
                },
                'momentum_states': self.metrics.momentum_state_counts,
                'momentum_transitions': self.metrics.momentum_transitions,
                'regimes': {
                    'current': self.metrics.current_regime,
                    'time_by_regime': self.metrics.regime_time_seconds,
                },
                'pnl': {
                    'daily': round(self.metrics.daily_pnl, 2),
                    'trade_count': self.metrics.trade_count,
                    'win_count': self.metrics.win_count,
                    'win_rate': round(self.metrics.win_count / max(self.metrics.trade_count, 1), 3),
                },
            },
            'strategy_timeline': self.metrics.strategy_events[-50:],
            'kill_switch_history': [asdict(e) for e in self._kill_switch_events[-20:]],
        }

        if format == "json":
            # Save to file
            with open(OBSERVABILITY_EXPORT_FILE, 'w') as f:
                json.dump(export_data, f, indent=2)
            return export_data

        elif format == "csv":
            # Convert to CSV for spreadsheet analysis
            csv_lines = []

            # Veto histogram
            csv_lines.append("=== VETO HISTOGRAM ===")
            csv_lines.append("reason,count")
            for reason, count in self.metrics.veto_by_reason.items():
                csv_lines.append(f"{reason},{count}")

            # Exit histogram
            csv_lines.append("\n=== EXIT HISTOGRAM ===")
            csv_lines.append("reason,count")
            for reason, count in self.metrics.exit_by_reason.items():
                csv_lines.append(f"{reason},{count}")

            # Momentum states
            csv_lines.append("\n=== MOMENTUM STATES ===")
            csv_lines.append("state,count")
            for state, count in self.metrics.momentum_state_counts.items():
                csv_lines.append(f"{state},{count}")

            # Regime time
            csv_lines.append("\n=== REGIME TIME (seconds) ===")
            csv_lines.append("regime,seconds")
            for regime, seconds in self.metrics.regime_time_seconds.items():
                csv_lines.append(f"{regime},{seconds:.1f}")

            # Strategy timeline
            csv_lines.append("\n=== STRATEGY TIMELINE ===")
            csv_lines.append("timestamp,strategy,action,reason")
            for event in self.metrics.strategy_events[-50:]:
                csv_lines.append(f"{event['timestamp']},{event['strategy_id']},"
                               f"{event['action']},{event['reason']}")

            csv_content = "\n".join(csv_lines)
            csv_file = OBSERVABILITY_EXPORT_FILE.replace('.json', '.csv')
            with open(csv_file, 'w') as f:
                f.write(csv_content)

            return csv_content

    def get_governor_health_state(self, market_open: bool = None, services_healthy: bool = True) -> GovernorHealthState:
        """
        Determine current Governor health state for dashboard display.

        Args:
            market_open: Whether market is currently open (None = auto-detect)
            services_healthy: Whether all backend services are connected

        Returns:
            GovernorHealthState enum value
        """
        # Auto-detect market hours if not provided
        if market_open is None:
            from datetime import datetime
            now = datetime.now()
            # Market hours: 9:30 AM - 4:00 PM ET (simplified check)
            hour = now.hour
            minute = now.minute
            time_val = hour * 100 + minute
            # Include pre-market (4:00 AM - 9:30 AM) as "market accessible"
            market_open = 400 <= time_val <= 1600 and now.weekday() < 5

        # Check states in priority order
        if self._kill_switch_active:
            return GovernorHealthState.OFFLINE

        if not services_healthy:
            return GovernorHealthState.OFFLINE

        if not market_open:
            # Services healthy but market closed
            return GovernorHealthState.STANDBY

        if not self.config.enabled:
            # Market open, services healthy, but trading not enabled
            return GovernorHealthState.CONNECTED

        # All systems go
        return GovernorHealthState.ACTIVE

    def get_status(self) -> Dict:
        """Get current status summary"""
        health_state = self.get_governor_health_state()
        return {
            'activated': self.config.enabled,
            'mode': self.config.mode.value,
            'health_state': health_state.value,
            'trading_allowed': self.is_active(),
            'kill_switch_active': self._kill_switch_active,
            'kill_switch_reason': self._kill_switch_reason.value if self._kill_switch_reason else None,
            'position_multiplier': self.config.position_size_multiplier,
            'active_strategy': self.config.active_strategy,
            'symbol_whitelist': self.config.symbol_whitelist,
            'daily_pnl': round(self.metrics.daily_pnl, 2),
            'trade_count': self.metrics.trade_count,
            'veto_count': self.metrics.veto_count_total,
            'forced_exit_count': self.metrics.forced_exit_count,
            'current_regime': self.metrics.current_regime,
        }

    def reset_daily_metrics(self):
        """Reset daily metrics (call at market open)"""
        with self._lock:
            self.metrics.daily_pnl = 0
            self.metrics.trade_count = 0
            self.metrics.win_count = 0
            self.metrics.veto_count_total = 0
            self.metrics.veto_by_reason.clear()
            self.metrics.veto_timestamps.clear()
            self.metrics.exit_count_total = 0
            self.metrics.exit_by_reason.clear()
            self.metrics.forced_exit_count = 0
            self.metrics.forced_exit_timestamps.clear()
            self.metrics.momentum_state_counts.clear()
            self.metrics.momentum_transitions = 0
            self.metrics.regime_time_seconds.clear()
            self.metrics.strategy_events.clear()
            self._save_state()
            logger.info("Daily metrics reset")


# Singleton instance
_instance: Optional[SafeActivationMode] = None


def get_safe_activation() -> SafeActivationMode:
    """Get singleton safe activation instance"""
    global _instance
    if _instance is None:
        _instance = SafeActivationMode()
    return _instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    print("=" * 60)
    print("SAFE ACTIVATION MODE TEST")
    print("=" * 60)

    safe = SafeActivationMode()

    print("\n1. Initial status:")
    print(f"   {json.dumps(safe.get_status(), indent=2)}")

    print("\n2. Activating safe mode...")
    result = safe.activate(
        mode=ActivationMode.SAFE,
        position_multiplier=0.25,
        symbol_whitelist=['TSLA', 'NVDA', 'AAPL']
    )
    print(f"   Result: {result}")

    print("\n3. Testing trade approval...")
    allowed, reason, size = safe.can_trade('TSLA', 'scalper', 1000)
    print(f"   TSLA: allowed={allowed}, size=${size:.2f}, reason={reason}")

    allowed, reason, size = safe.can_trade('GME', 'scalper', 1000)
    print(f"   GME: allowed={allowed}, reason={reason}")

    print("\n4. Recording metrics...")
    safe.record_veto("SPREAD_WIDE")
    safe.record_veto("CONFIDENCE_LOW")
    safe.record_veto("SPREAD_WIDE")
    safe.record_exit("TRAILING_STOP", pnl=15.50)
    safe.record_exit("STOP_LOSS", is_forced=True, pnl=-25.00)
    safe.record_momentum_state("CONFIRMED")
    safe.record_momentum_state("DECAY")
    safe.record_regime_change("TRENDING_UP")
    safe.record_strategy_event("scalper", "ENABLE", "activated")
    safe.record_trade(is_win=True, pnl=15.50)
    safe.record_trade(is_win=False, pnl=-25.00)

    print("\n5. Exporting observability...")
    export = safe.export_observability("json")
    print(f"   Exported to {OBSERVABILITY_EXPORT_FILE}")

    print("\n6. Final status:")
    print(f"   {json.dumps(safe.get_status(), indent=2)}")

    print("\n7. Testing kill-switch...")
    # Simulate veto spike
    for i in range(25):
        safe.record_veto("TEST_VETO")

    print(f"   Kill-switch active: {safe._kill_switch_active}")

    if safe._kill_switch_active:
        print("\n8. Resetting kill-switch (force)...")
        result = safe.reset_kill_switch(force=True)
        print(f"   Result: {result}")

    print("\n9. Deactivating...")
    safe.deactivate("test complete")
    print(f"   Status: {safe.get_status()['mode']}")

    print("\n" + "=" * 60)
    print("SAFE ACTIVATION MODE TEST COMPLETE")
    print("=" * 60)
