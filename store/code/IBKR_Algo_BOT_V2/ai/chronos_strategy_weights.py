"""
Strategy-Specific Chronos Weighting (Task R)
=============================================
Chronos must be strategy-aware, not global.

Strategy     | Chronos Role
-------------|-------------------
Momentum Scout | Ignored (no blocking)
Fast Scalper   | Directional bias only
ATS            | Structural confirmation
Swing          | Hard requirement

Chronos confidence should:
- Scale position size
- Accelerate exits
- NOT block scout initiation
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ChronosRole(Enum):
    """How Chronos is applied to a strategy"""
    IGNORED = "IGNORED"                     # No influence at all
    DIRECTIONAL_BIAS = "DIRECTIONAL_BIAS"   # Only affects direction, doesn't block
    STRUCTURAL_CONFIRM = "STRUCTURAL_CONFIRM"  # Required for structure confirmation
    HARD_REQUIREMENT = "HARD_REQUIREMENT"   # Must pass or entry blocked


class ChronosInfluence(Enum):
    """What Chronos affects for this strategy"""
    NONE = "NONE"
    POSITION_SIZE = "POSITION_SIZE"
    EXIT_SPEED = "EXIT_SPEED"
    ENTRY_GATE = "ENTRY_GATE"
    ALL = "ALL"


@dataclass
class StrategyChronosConfig:
    """Chronos configuration for a specific strategy"""
    strategy_name: str
    role: ChronosRole
    influences: List[ChronosInfluence]

    # Thresholds
    min_confidence_for_entry: float = 0.0     # 0 = ignored
    min_confidence_for_scaling: float = 0.5
    size_scaling_enabled: bool = True
    exit_acceleration_enabled: bool = True

    # Size multipliers based on confidence
    high_confidence_size_mult: float = 1.2    # > 70% confidence
    medium_confidence_size_mult: float = 1.0  # 50-70% confidence
    low_confidence_size_mult: float = 0.7     # < 50% confidence

    # Exit acceleration based on regime
    favorable_regime_hold_mult: float = 1.5   # Hold longer in good regime
    unfavorable_regime_hold_mult: float = 0.5 # Exit faster in bad regime

    def to_dict(self) -> Dict:
        return {
            "strategy_name": self.strategy_name,
            "role": self.role.value,
            "influences": [i.value for i in self.influences],
            "min_confidence_for_entry": self.min_confidence_for_entry,
            "min_confidence_for_scaling": self.min_confidence_for_scaling,
            "size_scaling_enabled": self.size_scaling_enabled,
            "exit_acceleration_enabled": self.exit_acceleration_enabled,
            "high_confidence_size_mult": self.high_confidence_size_mult,
            "medium_confidence_size_mult": self.medium_confidence_size_mult,
            "low_confidence_size_mult": self.low_confidence_size_mult,
            "favorable_regime_hold_mult": self.favorable_regime_hold_mult,
            "unfavorable_regime_hold_mult": self.unfavorable_regime_hold_mult
        }


# Default strategy configurations
DEFAULT_STRATEGY_CONFIGS = {
    "MOMENTUM_SCOUT": StrategyChronosConfig(
        strategy_name="MOMENTUM_SCOUT",
        role=ChronosRole.IGNORED,
        influences=[ChronosInfluence.NONE],
        min_confidence_for_entry=0.0,
        min_confidence_for_scaling=0.0,
        size_scaling_enabled=False,
        exit_acceleration_enabled=False
    ),
    "FAST_SCALPER": StrategyChronosConfig(
        strategy_name="FAST_SCALPER",
        role=ChronosRole.DIRECTIONAL_BIAS,
        influences=[ChronosInfluence.POSITION_SIZE, ChronosInfluence.EXIT_SPEED],
        min_confidence_for_entry=0.0,  # Doesn't block entry
        min_confidence_for_scaling=0.4,
        size_scaling_enabled=True,
        exit_acceleration_enabled=True
    ),
    "WARRIOR": StrategyChronosConfig(
        strategy_name="WARRIOR",
        role=ChronosRole.DIRECTIONAL_BIAS,
        influences=[ChronosInfluence.POSITION_SIZE],
        min_confidence_for_entry=0.0,
        min_confidence_for_scaling=0.5,
        size_scaling_enabled=True,
        exit_acceleration_enabled=True
    ),
    "ATS": StrategyChronosConfig(
        strategy_name="ATS",
        role=ChronosRole.STRUCTURAL_CONFIRM,
        influences=[ChronosInfluence.ENTRY_GATE, ChronosInfluence.POSITION_SIZE, ChronosInfluence.EXIT_SPEED],
        min_confidence_for_entry=0.5,
        min_confidence_for_scaling=0.6,
        size_scaling_enabled=True,
        exit_acceleration_enabled=True
    ),
    "PULLBACK_SCALPER": StrategyChronosConfig(
        strategy_name="PULLBACK_SCALPER",
        role=ChronosRole.STRUCTURAL_CONFIRM,
        influences=[ChronosInfluence.ENTRY_GATE, ChronosInfluence.POSITION_SIZE],
        min_confidence_for_entry=0.45,
        min_confidence_for_scaling=0.5,
        size_scaling_enabled=True,
        exit_acceleration_enabled=True
    ),
    "SWING_ENTRY": StrategyChronosConfig(
        strategy_name="SWING_ENTRY",
        role=ChronosRole.HARD_REQUIREMENT,
        influences=[ChronosInfluence.ALL],
        min_confidence_for_entry=0.6,
        min_confidence_for_scaling=0.7,
        size_scaling_enabled=True,
        exit_acceleration_enabled=True,
        high_confidence_size_mult=1.3,
        favorable_regime_hold_mult=2.0
    ),
    "DEFENSIVE_SCALPER": StrategyChronosConfig(
        strategy_name="DEFENSIVE_SCALPER",
        role=ChronosRole.STRUCTURAL_CONFIRM,
        influences=[ChronosInfluence.ENTRY_GATE, ChronosInfluence.EXIT_SPEED],
        min_confidence_for_entry=0.5,
        min_confidence_for_scaling=0.6,
        size_scaling_enabled=True,
        exit_acceleration_enabled=True,
        low_confidence_size_mult=0.5,
        unfavorable_regime_hold_mult=0.3  # Exit very fast
    ),
    "NEWS_TRADER": StrategyChronosConfig(
        strategy_name="NEWS_TRADER",
        role=ChronosRole.DIRECTIONAL_BIAS,
        influences=[ChronosInfluence.EXIT_SPEED],
        min_confidence_for_entry=0.0,  # News drives entry, not Chronos
        min_confidence_for_scaling=0.4,
        size_scaling_enabled=True,
        exit_acceleration_enabled=True
    ),
    "PROBE_ENTRY": StrategyChronosConfig(
        strategy_name="PROBE_ENTRY",
        role=ChronosRole.DIRECTIONAL_BIAS,
        influences=[ChronosInfluence.POSITION_SIZE],
        min_confidence_for_entry=0.0,
        min_confidence_for_scaling=0.4,
        size_scaling_enabled=True,
        exit_acceleration_enabled=True,
        low_confidence_size_mult=0.5
    )
}


class ChronosStrategyManager:
    """
    Manages strategy-specific Chronos weighting.

    Ensures Chronos is applied appropriately for each strategy:
    - Scout mode: Chronos ignored
    - Fast strategies: Directional bias only
    - ATS: Full structural confirmation
    - Swing: Hard requirement
    """

    def __init__(self):
        self.configs: Dict[str, StrategyChronosConfig] = {}
        self._load_defaults()

    def _load_defaults(self):
        """Load default strategy configurations"""
        self.configs = {k: v for k, v in DEFAULT_STRATEGY_CONFIGS.items()}
        logger.info(f"Loaded Chronos weights for {len(self.configs)} strategies")

    def get_config(self, strategy_name: str) -> Optional[StrategyChronosConfig]:
        """Get Chronos configuration for a strategy"""
        return self.configs.get(strategy_name.upper())

    def should_block_entry(
        self,
        strategy_name: str,
        chronos_confidence: float,
        market_regime: str = None
    ) -> Tuple[bool, str]:
        """
        Check if Chronos should block entry for this strategy.

        Returns:
            (should_block, reason)
        """
        config = self.get_config(strategy_name)
        if config is None:
            return False, f"No config for {strategy_name}, allowing entry"

        # IGNORED role never blocks
        if config.role == ChronosRole.IGNORED:
            return False, "Chronos ignored for this strategy"

        # DIRECTIONAL_BIAS doesn't block, just influences
        if config.role == ChronosRole.DIRECTIONAL_BIAS:
            return False, "Directional bias only, not blocking"

        # STRUCTURAL_CONFIRM and HARD_REQUIREMENT can block
        if ChronosInfluence.ENTRY_GATE in config.influences or ChronosInfluence.ALL in config.influences:
            if chronos_confidence < config.min_confidence_for_entry:
                return True, f"Confidence {chronos_confidence:.0%} < required {config.min_confidence_for_entry:.0%}"

        # Check regime for HARD_REQUIREMENT
        if config.role == ChronosRole.HARD_REQUIREMENT:
            if market_regime in ["TRENDING_DOWN", "CRASH", "HALT_RISK"]:
                return True, f"Regime {market_regime} blocks {strategy_name}"

        return False, "Entry allowed"

    def get_position_size_multiplier(
        self,
        strategy_name: str,
        chronos_confidence: float
    ) -> float:
        """
        Get position size multiplier based on Chronos confidence.

        Returns multiplier (0.5 - 1.3 typically)
        """
        config = self.get_config(strategy_name)
        if config is None or not config.size_scaling_enabled:
            return 1.0

        if config.role == ChronosRole.IGNORED:
            return 1.0

        # Scale based on confidence
        if chronos_confidence >= 0.7:
            return config.high_confidence_size_mult
        elif chronos_confidence >= 0.5:
            return config.medium_confidence_size_mult
        else:
            return config.low_confidence_size_mult

    def get_hold_time_multiplier(
        self,
        strategy_name: str,
        market_regime: str
    ) -> float:
        """
        Get hold time multiplier based on regime.

        > 1.0 = hold longer (favorable)
        < 1.0 = exit faster (unfavorable)
        """
        config = self.get_config(strategy_name)
        if config is None or not config.exit_acceleration_enabled:
            return 1.0

        if config.role == ChronosRole.IGNORED:
            return 1.0

        # Favorable regimes
        if market_regime in ["TRENDING_UP", "RANGING"]:
            return config.favorable_regime_hold_mult

        # Unfavorable regimes
        if market_regime in ["TRENDING_DOWN", "VOLATILE", "CRASH"]:
            return config.unfavorable_regime_hold_mult

        return 1.0

    def should_allow_scaling(
        self,
        strategy_name: str,
        chronos_confidence: float
    ) -> Tuple[bool, str]:
        """
        Check if position scaling is allowed.

        Returns:
            (allowed, reason)
        """
        config = self.get_config(strategy_name)
        if config is None:
            return True, "No config, allowing scaling"

        if config.role == ChronosRole.IGNORED:
            return True, "Chronos ignored for scaling"

        if chronos_confidence >= config.min_confidence_for_scaling:
            return True, f"Confidence {chronos_confidence:.0%} >= required {config.min_confidence_for_scaling:.0%}"
        else:
            return False, f"Confidence {chronos_confidence:.0%} < required {config.min_confidence_for_scaling:.0%}"

    def get_strategy_role(self, strategy_name: str) -> Optional[ChronosRole]:
        """Get Chronos role for a strategy"""
        config = self.get_config(strategy_name)
        return config.role if config else None

    def get_all_configs(self) -> Dict[str, Dict]:
        """Get all strategy configurations"""
        return {k: v.to_dict() for k, v in self.configs.items()}

    def update_config(self, strategy_name: str, updates: Dict):
        """Update a strategy's Chronos configuration"""
        config = self.get_config(strategy_name)
        if config is None:
            logger.warning(f"No config for {strategy_name}")
            return None

        for key, value in updates.items():
            if hasattr(config, key):
                # Handle enum conversion
                if key == "role" and isinstance(value, str):
                    value = ChronosRole(value)
                elif key == "influences" and isinstance(value, list):
                    value = [ChronosInfluence(i) if isinstance(i, str) else i for i in value]

                setattr(config, key, value)

        return config.to_dict()

    def get_status(self) -> Dict:
        """Get manager status"""
        return {
            "strategies_configured": len(self.configs),
            "configs": self.get_all_configs(),
            "roles_summary": {
                role.value: [
                    s for s, c in self.configs.items()
                    if c.role == role
                ] for role in ChronosRole
            },
            "timestamp": datetime.now().isoformat()
        }


# Singleton instance
_manager: Optional[ChronosStrategyManager] = None


def get_chronos_strategy_manager() -> ChronosStrategyManager:
    """Get the singleton Chronos strategy manager"""
    global _manager
    if _manager is None:
        _manager = ChronosStrategyManager()
    return _manager


# Convenience functions
def should_chronos_block(strategy: str, confidence: float, regime: str = None) -> Tuple[bool, str]:
    """Check if Chronos should block entry for strategy"""
    return get_chronos_strategy_manager().should_block_entry(strategy, confidence, regime)


def get_chronos_size_mult(strategy: str, confidence: float) -> float:
    """Get position size multiplier from Chronos"""
    return get_chronos_strategy_manager().get_position_size_multiplier(strategy, confidence)


def get_chronos_hold_mult(strategy: str, regime: str) -> float:
    """Get hold time multiplier from Chronos regime"""
    return get_chronos_strategy_manager().get_hold_time_multiplier(strategy, regime)
