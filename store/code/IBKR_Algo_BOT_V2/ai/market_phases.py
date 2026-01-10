"""
Market Phase Definition (Task K) & Strategy Compatibility Matrix (Task L)
=========================================================================
Defines discrete market phases and maps them to allowed strategies.

Phases:
- OPEN_IGNITION: First 15 minutes after open (9:30-9:45 ET) - high volatility
- STRUCTURED_MOMENTUM: Post-open momentum (9:45-11:30 ET) - trend development
- MIDDAY_COMPRESSION: Lunch doldrums (11:30-14:00 ET) - low volume, choppy
- POWER_HOUR: Final push (14:00-16:00 ET) - institutional activity

Each phase has different characteristics and appropriate strategies.
"""

import json
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set
from enum import Enum
from datetime import time, datetime
import pytz

logger = logging.getLogger(__name__)

# Config file location
PHASE_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "market_phases_config.json")


class MarketPhase(Enum):
    """Discrete market phases throughout the trading day"""
    PRE_MARKET = "PRE_MARKET"              # 4:00-9:30 ET
    OPEN_IGNITION = "OPEN_IGNITION"        # 9:30-9:45 ET
    STRUCTURED_MOMENTUM = "STRUCTURED_MOMENTUM"  # 9:45-11:30 ET
    MIDDAY_COMPRESSION = "MIDDAY_COMPRESSION"   # 11:30-14:00 ET
    POWER_HOUR = "POWER_HOUR"              # 14:00-16:00 ET
    AFTER_HOURS = "AFTER_HOURS"            # 16:00-20:00 ET
    CLOSED = "CLOSED"                       # Outside trading hours


class TradingStrategy(Enum):
    """Available trading strategies"""
    WARRIOR = "WARRIOR"                     # Ross Cameron fast scalping
    FAST_SCALPER = "FAST_SCALPER"          # HFT momentum scalper
    ATS = "ATS"                            # Adaptive Trading System (signal-gated)
    PULLBACK_SCALPER = "PULLBACK_SCALPER"  # Buy pullbacks in uptrends
    DEFENSIVE_SCALPER = "DEFENSIVE_SCALPER" # Conservative, tight stops
    SWING_ENTRY = "SWING_ENTRY"            # Longer hold, larger moves
    PROBE_ENTRY = "PROBE_ENTRY"            # Controlled initiation (Task F)
    NEWS_TRADER = "NEWS_TRADER"            # News-triggered trades


@dataclass
class PhaseConfig:
    """Configuration for a market phase"""
    name: str
    description: str
    start_time: str  # HH:MM format (ET)
    end_time: str    # HH:MM format (ET)

    # Market characteristics
    expected_volatility: str  # LOW, MEDIUM, HIGH, EXTREME
    expected_volume: str      # LOW, MEDIUM, HIGH
    trend_reliability: str    # LOW, MEDIUM, HIGH

    # Strategy settings
    allowed_strategies: List[str]  # Strategies enabled in this phase
    default_aggressiveness: int    # 1=conservative, 2=neutral, 3=aggressive

    # Risk adjustments
    position_size_multiplier: float = 1.0  # Scale position size
    stop_loss_multiplier: float = 1.0      # Scale stop distance
    max_trades_per_phase: int = 10         # Per-phase trade limit

    def get_start_time(self) -> time:
        h, m = map(int, self.start_time.split(':'))
        return time(h, m)

    def get_end_time(self) -> time:
        h, m = map(int, self.end_time.split(':'))
        return time(h, m)

    def to_dict(self) -> Dict:
        return asdict(self)


# Default phase configurations
DEFAULT_PHASES = {
    MarketPhase.PRE_MARKET.value: PhaseConfig(
        name="PRE_MARKET",
        description="Pre-market session. Gap plays, news reactions, low liquidity.",
        start_time="04:00",
        end_time="09:30",
        expected_volatility="HIGH",
        expected_volume="LOW",
        trend_reliability="LOW",
        allowed_strategies=["WARRIOR", "FAST_SCALPER", "NEWS_TRADER", "PROBE_ENTRY"],
        default_aggressiveness=2,
        position_size_multiplier=0.75,
        stop_loss_multiplier=1.5,
        max_trades_per_phase=8
    ),
    MarketPhase.OPEN_IGNITION.value: PhaseConfig(
        name="OPEN_IGNITION",
        description="First 15 minutes after open. Extreme volatility, gap fills, opening range.",
        start_time="09:30",
        end_time="09:45",
        expected_volatility="EXTREME",
        expected_volume="HIGH",
        trend_reliability="LOW",
        allowed_strategies=["WARRIOR", "FAST_SCALPER"],  # Only fast strategies
        default_aggressiveness=3,
        position_size_multiplier=0.5,  # Reduced due to volatility
        stop_loss_multiplier=2.0,       # Wider stops needed
        max_trades_per_phase=5
    ),
    MarketPhase.STRUCTURED_MOMENTUM.value: PhaseConfig(
        name="STRUCTURED_MOMENTUM",
        description="Post-open momentum phase. Trends develop, follow-through on gaps.",
        start_time="09:45",
        end_time="11:30",
        expected_volatility="MEDIUM",
        expected_volume="HIGH",
        trend_reliability="HIGH",
        allowed_strategies=["ATS", "PULLBACK_SCALPER", "PROBE_ENTRY", "NEWS_TRADER"],
        default_aggressiveness=2,
        position_size_multiplier=1.0,
        stop_loss_multiplier=1.0,
        max_trades_per_phase=15
    ),
    MarketPhase.MIDDAY_COMPRESSION.value: PhaseConfig(
        name="MIDDAY_COMPRESSION",
        description="Lunch doldrums. Low volume, choppy, false breakouts common.",
        start_time="11:30",
        end_time="14:00",
        expected_volatility="LOW",
        expected_volume="LOW",
        trend_reliability="LOW",
        allowed_strategies=["DEFENSIVE_SCALPER"],  # Only defensive strategy
        default_aggressiveness=1,
        position_size_multiplier=0.5,
        stop_loss_multiplier=0.75,  # Tighter stops
        max_trades_per_phase=3  # Severely limit trading
    ),
    MarketPhase.POWER_HOUR.value: PhaseConfig(
        name="POWER_HOUR",
        description="Final push before close. Institutional activity, trend continuation.",
        start_time="14:00",
        end_time="16:00",
        expected_volatility="MEDIUM",
        expected_volume="HIGH",
        trend_reliability="MEDIUM",
        allowed_strategies=["ATS", "SWING_ENTRY", "PULLBACK_SCALPER"],
        default_aggressiveness=2,
        position_size_multiplier=1.0,
        stop_loss_multiplier=1.0,
        max_trades_per_phase=10
    ),
    MarketPhase.AFTER_HOURS.value: PhaseConfig(
        name="AFTER_HOURS",
        description="After-hours session. Low liquidity, wide spreads.",
        start_time="16:00",
        end_time="20:00",
        expected_volatility="MEDIUM",
        expected_volume="LOW",
        trend_reliability="LOW",
        allowed_strategies=["NEWS_TRADER"],  # Only news-driven
        default_aggressiveness=1,
        position_size_multiplier=0.5,
        stop_loss_multiplier=1.5,
        max_trades_per_phase=3
    ),
    MarketPhase.CLOSED.value: PhaseConfig(
        name="CLOSED",
        description="Market closed. No trading allowed.",
        start_time="20:00",
        end_time="04:00",
        expected_volatility="LOW",
        expected_volume="LOW",
        trend_reliability="LOW",
        allowed_strategies=[],  # No strategies
        default_aggressiveness=1,
        position_size_multiplier=0.0,
        stop_loss_multiplier=1.0,
        max_trades_per_phase=0
    )
}


class MarketPhaseManager:
    """
    Manages market phases and strategy compatibility.

    Only one phase active at a time.
    All phase changes are logged.
    """

    def __init__(self):
        self.phases: Dict[str, PhaseConfig] = {}
        self.current_phase: Optional[MarketPhase] = None
        self.phase_locked_until: Optional[datetime] = None
        self.last_phase_change: Optional[datetime] = None
        self.phase_change_reason: str = ""
        self.phase_history: List[Dict] = []
        self.trades_this_phase: int = 0
        self.suppressed_signals: List[Dict] = []

        # Load configuration
        self._load_config()

        # Initialize current phase
        self._update_phase_by_time()

    def _load_config(self):
        """Load phase configuration from file or use defaults"""
        try:
            if os.path.exists(PHASE_CONFIG_FILE):
                with open(PHASE_CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    for name, params in data.get('phases', {}).items():
                        self.phases[name] = PhaseConfig(**params)
                logger.info(f"Loaded {len(self.phases)} market phases from config")
            else:
                # Use defaults
                self.phases = {k: v for k, v in DEFAULT_PHASES.items()}
                self._save_config()
                logger.info("Created default market phases config")
        except Exception as e:
            logger.error(f"Error loading phase config: {e}, using defaults")
            self.phases = {k: v for k, v in DEFAULT_PHASES.items()}

    def _save_config(self):
        """Save phase configuration to file"""
        try:
            data = {
                'phases': {k: v.to_dict() for k, v in self.phases.items()},
                'last_updated': datetime.now().isoformat()
            }
            with open(PHASE_CONFIG_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving phase config: {e}")

    def _update_phase_by_time(self) -> MarketPhase:
        """Determine current phase based on time of day"""
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz).time()

        for phase_name, config in self.phases.items():
            start = config.get_start_time()
            end = config.get_end_time()

            # Handle overnight phase (CLOSED: 20:00-04:00)
            if start > end:
                if now_et >= start or now_et < end:
                    return MarketPhase(phase_name)
            else:
                if start <= now_et < end:
                    return MarketPhase(phase_name)

        return MarketPhase.CLOSED

    def get_current_phase(self) -> MarketPhase:
        """Get the current market phase"""
        if self.current_phase is None:
            self._update_phase_by_time()
        return self.current_phase

    def get_phase_config(self, phase: MarketPhase = None) -> PhaseConfig:
        """Get configuration for a phase (current if not specified)"""
        if phase is None:
            phase = self.get_current_phase()
        return self.phases.get(phase.value, DEFAULT_PHASES[MarketPhase.CLOSED.value])

    def set_phase(self, phase: MarketPhase, reason: str, lock_minutes: int = 15) -> bool:
        """
        Set the current market phase.

        Args:
            phase: The phase to activate
            reason: Why this phase was selected
            lock_minutes: How long to lock this phase
        """
        # Check lock
        if self.phase_locked_until and datetime.now() < self.phase_locked_until:
            remaining = (self.phase_locked_until - datetime.now()).total_seconds()
            logger.info(f"Phase change blocked - locked for {remaining:.0f}s more")
            return False

        # Log change
        old_phase = self.current_phase
        self.current_phase = phase
        self.last_phase_change = datetime.now()
        self.phase_change_reason = reason
        self.trades_this_phase = 0  # Reset trade count

        # Set lock
        if lock_minutes > 0:
            from datetime import timedelta
            self.phase_locked_until = datetime.now() + timedelta(minutes=lock_minutes)

        # Record history
        change_record = {
            "timestamp": datetime.now().isoformat(),
            "from_phase": old_phase.value if old_phase else None,
            "to_phase": phase.value,
            "reason": reason,
            "lock_minutes": lock_minutes
        }
        self.phase_history.append(change_record)

        if len(self.phase_history) > 100:
            self.phase_history = self.phase_history[-100:]

        logger.info(
            f"MARKET_PHASE_SELECTED: {old_phase.value if old_phase else 'None'} -> {phase.value} | "
            f"Reason: {reason}"
        )

        return True

    def is_strategy_enabled(self, strategy: TradingStrategy) -> bool:
        """Check if a strategy is enabled for the current phase"""
        config = self.get_phase_config()
        return strategy.value in config.allowed_strategies

    def get_enabled_strategies(self) -> List[str]:
        """Get list of strategies enabled for current phase"""
        config = self.get_phase_config()
        return config.allowed_strategies

    def record_trade(self) -> bool:
        """
        Record a trade in the current phase.

        Returns False if phase trade limit exceeded.
        """
        config = self.get_phase_config()
        if self.trades_this_phase >= config.max_trades_per_phase:
            logger.warning(
                f"Phase trade limit reached: {self.trades_this_phase}/{config.max_trades_per_phase}"
            )
            return False

        self.trades_this_phase += 1
        return True

    def record_suppressed_signal(self, strategy: str, symbol: str, reason: str):
        """Record a signal that was suppressed due to phase/strategy mismatch"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "symbol": symbol,
            "phase": self.current_phase.value if self.current_phase else "UNKNOWN",
            "reason": reason
        }
        self.suppressed_signals.append(record)

        if len(self.suppressed_signals) > 200:
            self.suppressed_signals = self.suppressed_signals[-200:]

        logger.debug(f"Signal suppressed: {strategy} on {symbol} - {reason}")

    def get_status(self) -> Dict:
        """Get full phase manager status"""
        config = self.get_phase_config()

        lock_remaining = 0
        if self.phase_locked_until and datetime.now() < self.phase_locked_until:
            lock_remaining = (self.phase_locked_until - datetime.now()).total_seconds()

        return {
            "current_phase": self.current_phase.value if self.current_phase else None,
            "phase_description": config.description,
            "enabled_strategies": config.allowed_strategies,
            "disabled_strategies": [
                s.value for s in TradingStrategy
                if s.value not in config.allowed_strategies
            ],
            "phase_locked": lock_remaining > 0,
            "lock_remaining_seconds": lock_remaining,
            "next_change_allowed": self.phase_locked_until.isoformat() if self.phase_locked_until else None,
            "last_phase_change": self.last_phase_change.isoformat() if self.last_phase_change else None,
            "phase_change_reason": self.phase_change_reason,
            "trades_this_phase": self.trades_this_phase,
            "max_trades_this_phase": config.max_trades_per_phase,
            "position_size_multiplier": config.position_size_multiplier,
            "stop_loss_multiplier": config.stop_loss_multiplier,
            "expected_volatility": config.expected_volatility,
            "expected_volume": config.expected_volume,
            "suppressed_signals_count": len(self.suppressed_signals),
            "recent_suppressed": self.suppressed_signals[-10:] if self.suppressed_signals else []
        }


# Singleton instance
_manager: Optional[MarketPhaseManager] = None


def get_phase_manager() -> MarketPhaseManager:
    """Get the singleton market phase manager"""
    global _manager
    if _manager is None:
        _manager = MarketPhaseManager()
    return _manager


def get_current_phase() -> MarketPhase:
    """Get the current market phase"""
    return get_phase_manager().get_current_phase()


def is_strategy_enabled(strategy: TradingStrategy) -> bool:
    """Check if a strategy is enabled for the current phase"""
    return get_phase_manager().is_strategy_enabled(strategy)


def get_phase_config() -> PhaseConfig:
    """Get configuration for the current phase"""
    return get_phase_manager().get_phase_config()
