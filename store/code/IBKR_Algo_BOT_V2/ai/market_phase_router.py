"""
Market Phase Strategy Router

Routes trading strategy based on time-of-day market phases.
Each phase has different strategy bias, risk parameters, and trade allowances.

ChatGPT Directive Implementation - Jan 9, 2026

Phase Definitions:
- PREMARKET_EARLY (4:00-7:00 ET): Warrior Momentum - aggressive momentum plays
- PREMARKET_LATE (7:00-9:30 ET): ATS / Pullbacks - continuation and pullback setups
- OPEN (9:30-9:45 ET): Breakout / Halt - breakouts from open, halt plays
- POST_OPEN (9:45-11:00 ET): Scalper - standard HFT scalping
- MIDDAY (11:00-14:30 ET): No-trade / Light - reduced activity, only A-grade setups
- POWER_HOUR (15:00-16:00 ET): Continuation - ride established trends
- AFTER_HOURS (16:00-20:00 ET): Watch Only - no trading, just monitoring
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Dict, Optional, Any
import pytz

logger = logging.getLogger(__name__)


class MarketPhase(Enum):
    """Market phase enumeration"""
    PREMARKET_EARLY = "PREMARKET_EARLY"
    PREMARKET_LATE = "PREMARKET_LATE"
    OPEN = "OPEN"
    POST_OPEN = "POST_OPEN"
    MIDDAY = "MIDDAY"
    POWER_HOUR = "POWER_HOUR"
    AFTER_HOURS = "AFTER_HOURS"
    CLOSED = "CLOSED"


class StrategyBias(Enum):
    """Strategy bias for each phase"""
    WARRIOR_MOMENTUM = "WARRIOR_MOMENTUM"
    ATS_PULLBACKS = "ATS_PULLBACKS"
    BREAKOUT_HALT = "BREAKOUT_HALT"
    SCALPER = "SCALPER"
    LIGHT_ONLY = "LIGHT_ONLY"
    CONTINUATION = "CONTINUATION"
    WATCH_ONLY = "WATCH_ONLY"
    NO_TRADE = "NO_TRADE"


@dataclass
class PhaseConfig:
    """Configuration for a market phase"""
    phase: MarketPhase
    strategy_bias: StrategyBias
    trading_allowed: bool
    min_setup_grade: str  # A, B, or C
    max_position_pct: float  # % of account
    max_trades_per_hour: int
    profit_target_pct: float
    stop_loss_pct: float
    max_hold_seconds: int
    description: str


# Phase configurations based on ChatGPT directive
PHASE_CONFIGS: Dict[MarketPhase, PhaseConfig] = {
    MarketPhase.PREMARKET_EARLY: PhaseConfig(
        phase=MarketPhase.PREMARKET_EARLY,
        strategy_bias=StrategyBias.WARRIOR_MOMENTUM,
        trading_allowed=True,
        min_setup_grade="C",  # More permissive for momentum plays
        max_position_pct=0.25,  # 25% max per trade
        max_trades_per_hour=10,
        profit_target_pct=3.0,
        stop_loss_pct=2.0,
        max_hold_seconds=180,
        description="Aggressive momentum plays on gappers. Wide stops for volatility."
    ),
    MarketPhase.PREMARKET_LATE: PhaseConfig(
        phase=MarketPhase.PREMARKET_LATE,
        strategy_bias=StrategyBias.ATS_PULLBACKS,
        trading_allowed=True,
        min_setup_grade="C",
        max_position_pct=0.25,
        max_trades_per_hour=15,
        profit_target_pct=2.5,
        stop_loss_pct=1.5,
        max_hold_seconds=240,
        description="ATS pullback entries and continuation setups."
    ),
    MarketPhase.OPEN: PhaseConfig(
        phase=MarketPhase.OPEN,
        strategy_bias=StrategyBias.BREAKOUT_HALT,
        trading_allowed=True,
        min_setup_grade="B",  # Only B+ setups at open
        max_position_pct=0.20,  # Smaller size during chaos
        max_trades_per_hour=8,
        profit_target_pct=3.0,
        stop_loss_pct=2.0,
        max_hold_seconds=120,  # Quick exits
        description="Breakout plays and halt resumptions. High volatility window."
    ),
    MarketPhase.POST_OPEN: PhaseConfig(
        phase=MarketPhase.POST_OPEN,
        strategy_bias=StrategyBias.SCALPER,
        trading_allowed=True,
        min_setup_grade="C",
        max_position_pct=0.25,
        max_trades_per_hour=20,
        profit_target_pct=2.5,
        stop_loss_pct=1.5,
        max_hold_seconds=180,
        description="Standard HFT scalping. Trends established, cleaner moves."
    ),
    MarketPhase.MIDDAY: PhaseConfig(
        phase=MarketPhase.MIDDAY,
        strategy_bias=StrategyBias.LIGHT_ONLY,
        trading_allowed=True,
        min_setup_grade="A",  # Only A-grade setups
        max_position_pct=0.15,  # Reduced size
        max_trades_per_hour=5,  # Very selective
        profit_target_pct=2.0,
        stop_loss_pct=1.0,
        max_hold_seconds=300,
        description="Low volume chop. Only take A-grade setups with conviction."
    ),
    MarketPhase.POWER_HOUR: PhaseConfig(
        phase=MarketPhase.POWER_HOUR,
        strategy_bias=StrategyBias.CONTINUATION,
        trading_allowed=True,
        min_setup_grade="B",
        max_position_pct=0.25,
        max_trades_per_hour=15,
        profit_target_pct=3.0,
        stop_loss_pct=1.5,
        max_hold_seconds=240,
        description="Ride established trends. Volume returns, moves extend."
    ),
    MarketPhase.AFTER_HOURS: PhaseConfig(
        phase=MarketPhase.AFTER_HOURS,
        strategy_bias=StrategyBias.WATCH_ONLY,
        trading_allowed=False,
        min_setup_grade="A",
        max_position_pct=0.0,
        max_trades_per_hour=0,
        profit_target_pct=0.0,
        stop_loss_pct=0.0,
        max_hold_seconds=0,
        description="Monitor after-hours movers for next day. No trading."
    ),
    MarketPhase.CLOSED: PhaseConfig(
        phase=MarketPhase.CLOSED,
        strategy_bias=StrategyBias.NO_TRADE,
        trading_allowed=False,
        min_setup_grade="A",
        max_position_pct=0.0,
        max_trades_per_hour=0,
        profit_target_pct=0.0,
        stop_loss_pct=0.0,
        max_hold_seconds=0,
        description="Market closed. No trading."
    ),
}


class MarketPhaseRouter:
    """
    Routes trading strategy based on current market phase.

    Determines:
    - Which strategy bias to use
    - Whether trading is allowed
    - Position sizing limits
    - Profit/stop targets
    - Setup grade requirements
    """

    def __init__(self):
        self.et_tz = pytz.timezone('US/Eastern')
        self._current_phase: Optional[MarketPhase] = None
        self._phase_start_time: Optional[datetime] = None
        self._trades_this_hour: int = 0
        self._last_hour_reset: Optional[datetime] = None

        logger.info("MarketPhaseRouter initialized")

    def get_current_phase(self, now: Optional[datetime] = None) -> MarketPhase:
        """Get the current market phase based on time"""
        if now is None:
            now = datetime.now(self.et_tz)
        elif now.tzinfo is None:
            now = self.et_tz.localize(now)
        else:
            now = now.astimezone(self.et_tz)

        current_time = now.time()

        # Check if weekend
        if now.weekday() >= 5:
            return MarketPhase.CLOSED

        # Phase time boundaries (Eastern Time)
        if time(4, 0) <= current_time < time(7, 0):
            return MarketPhase.PREMARKET_EARLY
        elif time(7, 0) <= current_time < time(9, 30):
            return MarketPhase.PREMARKET_LATE
        elif time(9, 30) <= current_time < time(9, 45):
            return MarketPhase.OPEN
        elif time(9, 45) <= current_time < time(11, 0):
            return MarketPhase.POST_OPEN
        elif time(11, 0) <= current_time < time(14, 30):
            return MarketPhase.MIDDAY
        elif time(15, 0) <= current_time < time(16, 0):
            return MarketPhase.POWER_HOUR
        elif time(16, 0) <= current_time < time(20, 0):
            return MarketPhase.AFTER_HOURS
        else:
            return MarketPhase.CLOSED

    def get_phase_config(self, phase: Optional[MarketPhase] = None) -> PhaseConfig:
        """Get configuration for a market phase"""
        if phase is None:
            phase = self.get_current_phase()
        return PHASE_CONFIGS.get(phase, PHASE_CONFIGS[MarketPhase.CLOSED])

    def can_trade(self, setup_grade: str = "C") -> tuple[bool, str]:
        """
        Check if trading is allowed in current phase.

        Returns:
            (allowed: bool, reason: str)
        """
        phase = self.get_current_phase()
        config = self.get_phase_config(phase)

        if not config.trading_allowed:
            return False, f"Trading not allowed in {phase.value} phase"

        # Check setup grade requirement
        grade_order = {"A": 1, "B": 2, "C": 3}
        setup_rank = grade_order.get(setup_grade.upper(), 3)
        min_rank = grade_order.get(config.min_setup_grade, 3)

        if setup_rank > min_rank:
            return False, f"{phase.value} requires grade {config.min_setup_grade}+, got {setup_grade}"

        # Check hourly trade limit
        self._check_hour_reset()
        if self._trades_this_hour >= config.max_trades_per_hour:
            return False, f"Hourly trade limit reached ({config.max_trades_per_hour})"

        return True, f"Trading allowed in {phase.value}"

    def _check_hour_reset(self):
        """Reset hourly trade counter if new hour"""
        now = datetime.now(self.et_tz)
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        if self._last_hour_reset is None or current_hour > self._last_hour_reset:
            self._trades_this_hour = 0
            self._last_hour_reset = current_hour

    def record_trade(self):
        """Record a trade for hourly limit tracking"""
        self._check_hour_reset()
        self._trades_this_hour += 1

    def get_trade_params(self) -> Dict[str, Any]:
        """
        Get trading parameters for current phase.

        Returns dict with:
        - profit_target_pct
        - stop_loss_pct
        - max_hold_seconds
        - max_position_pct
        """
        config = self.get_phase_config()
        return {
            "profit_target_pct": config.profit_target_pct,
            "stop_loss_pct": config.stop_loss_pct,
            "max_hold_seconds": config.max_hold_seconds,
            "max_position_pct": config.max_position_pct,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current router status"""
        now = datetime.now(self.et_tz)
        phase = self.get_current_phase()
        config = self.get_phase_config(phase)

        self._check_hour_reset()

        return {
            "current_time_et": now.strftime("%H:%M:%S"),
            "current_phase": phase.value,
            "strategy_bias": config.strategy_bias.value,
            "trading_allowed": config.trading_allowed,
            "min_setup_grade": config.min_setup_grade,
            "max_position_pct": config.max_position_pct,
            "profit_target_pct": config.profit_target_pct,
            "stop_loss_pct": config.stop_loss_pct,
            "max_hold_seconds": config.max_hold_seconds,
            "trades_this_hour": self._trades_this_hour,
            "max_trades_per_hour": config.max_trades_per_hour,
            "phase_description": config.description,
        }

    def get_all_phases(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration for all phases"""
        return {
            phase.value: {
                "strategy_bias": config.strategy_bias.value,
                "trading_allowed": config.trading_allowed,
                "min_setup_grade": config.min_setup_grade,
                "max_position_pct": config.max_position_pct,
                "max_trades_per_hour": config.max_trades_per_hour,
                "profit_target_pct": config.profit_target_pct,
                "stop_loss_pct": config.stop_loss_pct,
                "max_hold_seconds": config.max_hold_seconds,
                "description": config.description,
            }
            for phase, config in PHASE_CONFIGS.items()
        }


# Singleton instance
_router: Optional[MarketPhaseRouter] = None


def get_market_phase_router() -> MarketPhaseRouter:
    """Get or create the MarketPhaseRouter singleton"""
    global _router
    if _router is None:
        _router = MarketPhaseRouter()
    return _router


# Convenience functions
def get_current_phase() -> MarketPhase:
    """Get current market phase"""
    return get_market_phase_router().get_current_phase()


def can_trade_now(setup_grade: str = "C") -> tuple[bool, str]:
    """Check if trading is allowed now"""
    return get_market_phase_router().can_trade(setup_grade)


def get_trade_params() -> Dict[str, Any]:
    """Get current trade parameters based on phase"""
    return get_market_phase_router().get_trade_params()
