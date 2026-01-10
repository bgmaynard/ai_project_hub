"""
Strategy Enable/Disable Hooks (Task N)
======================================
Gate layer that strategies must check before generating signals.

Each strategy module must:
- Check if it is enabled for current phase
- Refuse to generate signals if disabled
- Log suppressed signals for analysis

No strategy may bypass this layer.
"""

import logging
from functools import wraps
from typing import Dict, Optional, Callable, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class StrategyGate:
    """
    Gate that controls which strategies can generate signals.

    This is the enforcement layer for the phase/strategy matrix.
    All strategies MUST check this gate before generating signals.
    """

    def __init__(self):
        self.suppressed_count: Dict[str, int] = {}  # strategy -> count
        self.last_check: Dict[str, datetime] = {}   # strategy -> last check time
        self.override_enabled: bool = False          # Admin override (disable gate)
        self.override_strategies: set = set()        # Strategies with manual override

    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """
        Check if a strategy is enabled for the current phase.

        This is the main check that all strategies must call.
        """
        # Check admin override
        if self.override_enabled or strategy_name in self.override_strategies:
            logger.debug(f"Strategy {strategy_name} enabled via override")
            return True

        try:
            from ai.market_phases import get_phase_manager, TradingStrategy

            manager = get_phase_manager()

            # Convert string to enum if needed
            try:
                strategy_enum = TradingStrategy(strategy_name.upper())
            except ValueError:
                logger.warning(f"Unknown strategy: {strategy_name}")
                return False

            enabled = manager.is_strategy_enabled(strategy_enum)
            self.last_check[strategy_name] = datetime.now()

            return enabled

        except Exception as e:
            logger.error(f"Error checking strategy gate: {e}")
            # Fail-closed: if we can't check, don't allow
            return False

    def check_and_suppress(self, strategy_name: str, symbol: str, signal_data: Dict = None) -> bool:
        """
        Check if strategy is enabled and record suppression if not.

        Returns True if strategy should proceed, False if suppressed.
        """
        if self.is_strategy_enabled(strategy_name):
            return True

        # Record suppression
        try:
            from ai.market_phases import get_phase_manager

            manager = get_phase_manager()
            current_phase = manager.get_current_phase()

            reason = f"Strategy {strategy_name} not enabled in {current_phase.value} phase"

            manager.record_suppressed_signal(strategy_name, symbol, reason)

            # Track count
            self.suppressed_count[strategy_name] = self.suppressed_count.get(strategy_name, 0) + 1

            logger.info(f"SIGNAL_SUPPRESSED: {strategy_name} on {symbol} - {reason}")

        except Exception as e:
            logger.debug(f"Error recording suppression: {e}")

        return False

    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier for current phase"""
        try:
            from ai.market_phases import get_phase_config
            return get_phase_config().position_size_multiplier
        except Exception:
            return 1.0

    def get_stop_loss_multiplier(self) -> float:
        """Get stop loss multiplier for current phase"""
        try:
            from ai.market_phases import get_phase_config
            return get_phase_config().stop_loss_multiplier
        except Exception:
            return 1.0

    def can_trade(self) -> bool:
        """
        Check if trading is allowed in current phase.

        Returns False if:
        - Phase is CLOSED
        - Phase trade limit exceeded
        """
        try:
            from ai.market_phases import get_phase_manager, MarketPhase

            manager = get_phase_manager()
            config = manager.get_phase_config()

            # Check if closed
            if manager.current_phase == MarketPhase.CLOSED:
                return False

            # Check trade limit
            if manager.trades_this_phase >= config.max_trades_per_phase:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking can_trade: {e}")
            return False

    def record_trade(self) -> bool:
        """Record a trade in the current phase"""
        try:
            from ai.market_phases import get_phase_manager
            return get_phase_manager().record_trade()
        except Exception:
            return True

    def enable_override(self, strategies: list = None):
        """Enable admin override (bypass gate)"""
        if strategies:
            self.override_strategies.update(strategies)
            logger.warning(f"Admin override enabled for: {strategies}")
        else:
            self.override_enabled = True
            logger.warning("Admin override enabled for ALL strategies")

    def disable_override(self, strategies: list = None):
        """Disable admin override"""
        if strategies:
            self.override_strategies -= set(strategies)
            logger.info(f"Admin override disabled for: {strategies}")
        else:
            self.override_enabled = False
            self.override_strategies.clear()
            logger.info("Admin override disabled")

    def get_status(self) -> Dict:
        """Get gate status"""
        try:
            from ai.market_phases import get_phase_manager

            manager = get_phase_manager()
            config = manager.get_phase_config()

            return {
                "enabled_strategies": config.allowed_strategies,
                "suppressed_count": self.suppressed_count,
                "override_enabled": self.override_enabled,
                "override_strategies": list(self.override_strategies),
                "can_trade": self.can_trade(),
                "position_size_multiplier": config.position_size_multiplier,
                "stop_loss_multiplier": config.stop_loss_multiplier,
                "trades_this_phase": manager.trades_this_phase,
                "max_trades_this_phase": config.max_trades_per_phase
            }
        except Exception as e:
            return {"error": str(e)}


# Singleton instance
_gate: Optional[StrategyGate] = None


def get_strategy_gate() -> StrategyGate:
    """Get the singleton strategy gate"""
    global _gate
    if _gate is None:
        _gate = StrategyGate()
    return _gate


# ============================================================================
# DECORATOR FOR STRATEGY FUNCTIONS
# ============================================================================

def phase_gated(strategy_name: str):
    """
    Decorator that gates a strategy function by market phase.

    Usage:
        @phase_gated("WARRIOR")
        async def check_warrior_signal(symbol, quote):
            # This only runs if WARRIOR is enabled
            return signal

    If strategy is not enabled, returns None without executing the function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            gate = get_strategy_gate()

            # Extract symbol from args/kwargs for logging
            symbol = kwargs.get('symbol', args[0] if args else 'UNKNOWN')

            if not gate.check_and_suppress(strategy_name, symbol):
                return None  # Signal suppressed

            # Strategy is enabled, run the function
            return await func(*args, **kwargs)

        return wrapper
    return decorator


def phase_gated_sync(strategy_name: str):
    """
    Synchronous version of phase_gated decorator.

    Usage:
        @phase_gated_sync("DEFENSIVE_SCALPER")
        def check_signal(symbol, quote):
            return signal
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            gate = get_strategy_gate()

            # Extract symbol from args/kwargs for logging
            symbol = kwargs.get('symbol', args[0] if args else 'UNKNOWN')

            if not gate.check_and_suppress(strategy_name, symbol):
                return None  # Signal suppressed

            # Strategy is enabled, run the function
            return func(*args, **kwargs)

        return wrapper
    return decorator


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def is_strategy_enabled(strategy_name: str) -> bool:
    """Check if a strategy is enabled"""
    return get_strategy_gate().is_strategy_enabled(strategy_name)


def check_and_suppress(strategy_name: str, symbol: str) -> bool:
    """Check if strategy enabled, suppress if not"""
    return get_strategy_gate().check_and_suppress(strategy_name, symbol)


def can_trade() -> bool:
    """Check if trading is allowed in current phase"""
    return get_strategy_gate().can_trade()


def get_position_multiplier() -> float:
    """Get position size multiplier for current phase"""
    return get_strategy_gate().get_position_size_multiplier()


def get_stop_multiplier() -> float:
    """Get stop loss multiplier for current phase"""
    return get_strategy_gate().get_stop_loss_multiplier()
