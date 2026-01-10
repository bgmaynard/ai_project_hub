"""
Phase-Controlled Exploration Policy (Task S)
=============================================
Market Phase controls how exploratory the bot is.

Examples:
- OPEN_IGNITION + AGGRESSIVE baseline: Scouts enabled, higher frequency
- MIDDAY_COMPRESSION: Scouts disabled
- STRUCTURED_MOMENTUM: Scouts allowed but capped

Phase changes must:
- Immediately enable/disable scout logic
- Log MARKET_PHASE_EXPLORATION_POLICY
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExplorationLevel(Enum):
    """How exploratory the system should be"""
    DISABLED = "DISABLED"           # No exploration (scouts off)
    MINIMAL = "MINIMAL"             # Very limited exploration
    NORMAL = "NORMAL"               # Standard exploration
    AGGRESSIVE = "AGGRESSIVE"       # High exploration mode


@dataclass
class ExplorationPolicy:
    """Exploration policy for a market phase"""
    phase: str
    level: ExplorationLevel
    scouts_enabled: bool
    max_scouts_per_hour: int
    scout_size_multiplier: float    # Relative to base scout size
    chronos_override_allowed: bool  # Can ignore Chronos for scouts
    reasons: List[str]

    def to_dict(self) -> Dict:
        return {
            "phase": self.phase,
            "level": self.level.value,
            "scouts_enabled": self.scouts_enabled,
            "max_scouts_per_hour": self.max_scouts_per_hour,
            "scout_size_multiplier": self.scout_size_multiplier,
            "chronos_override_allowed": self.chronos_override_allowed,
            "reasons": self.reasons
        }


# Default policies by phase and baseline
DEFAULT_POLICIES = {
    # OPEN_IGNITION - High volatility, scouts encouraged
    "OPEN_IGNITION": {
        "CONSERVATIVE": ExplorationPolicy(
            phase="OPEN_IGNITION",
            level=ExplorationLevel.MINIMAL,
            scouts_enabled=True,
            max_scouts_per_hour=2,
            scout_size_multiplier=0.15,
            chronos_override_allowed=True,
            reasons=["High volatility but conservative baseline"]
        ),
        "NEUTRAL": ExplorationPolicy(
            phase="OPEN_IGNITION",
            level=ExplorationLevel.NORMAL,
            scouts_enabled=True,
            max_scouts_per_hour=4,
            scout_size_multiplier=0.20,
            chronos_override_allowed=True,
            reasons=["Prime momentum discovery window"]
        ),
        "AGGRESSIVE": ExplorationPolicy(
            phase="OPEN_IGNITION",
            level=ExplorationLevel.AGGRESSIVE,
            scouts_enabled=True,
            max_scouts_per_hour=6,
            scout_size_multiplier=0.25,
            chronos_override_allowed=True,
            reasons=["Maximum momentum capture opportunity"]
        )
    },

    # STRUCTURED_MOMENTUM - Best for trend following
    "STRUCTURED_MOMENTUM": {
        "CONSERVATIVE": ExplorationPolicy(
            phase="STRUCTURED_MOMENTUM",
            level=ExplorationLevel.MINIMAL,
            scouts_enabled=True,
            max_scouts_per_hour=2,
            scout_size_multiplier=0.15,
            chronos_override_allowed=True,
            reasons=["Trend development phase, limited scouts"]
        ),
        "NEUTRAL": ExplorationPolicy(
            phase="STRUCTURED_MOMENTUM",
            level=ExplorationLevel.NORMAL,
            scouts_enabled=True,
            max_scouts_per_hour=4,
            scout_size_multiplier=0.20,
            chronos_override_allowed=True,
            reasons=["Good trend reliability, scout for breakouts"]
        ),
        "AGGRESSIVE": ExplorationPolicy(
            phase="STRUCTURED_MOMENTUM",
            level=ExplorationLevel.AGGRESSIVE,
            scouts_enabled=True,
            max_scouts_per_hour=5,
            scout_size_multiplier=0.22,
            chronos_override_allowed=True,
            reasons=["Best phase for momentum trades"]
        )
    },

    # PRE_MARKET - Low liquidity but gaps possible
    "PRE_MARKET": {
        "CONSERVATIVE": ExplorationPolicy(
            phase="PRE_MARKET",
            level=ExplorationLevel.MINIMAL,
            scouts_enabled=True,
            max_scouts_per_hour=1,
            scout_size_multiplier=0.10,
            chronos_override_allowed=True,
            reasons=["Low liquidity, very small scouts only"]
        ),
        "NEUTRAL": ExplorationPolicy(
            phase="PRE_MARKET",
            level=ExplorationLevel.MINIMAL,
            scouts_enabled=True,
            max_scouts_per_hour=2,
            scout_size_multiplier=0.15,
            chronos_override_allowed=True,
            reasons=["Gap plays possible but limited size"]
        ),
        "AGGRESSIVE": ExplorationPolicy(
            phase="PRE_MARKET",
            level=ExplorationLevel.NORMAL,
            scouts_enabled=True,
            max_scouts_per_hour=3,
            scout_size_multiplier=0.18,
            chronos_override_allowed=True,
            reasons=["Early gap discovery opportunity"]
        )
    },

    # MIDDAY_COMPRESSION - Choppy, avoid exploration
    "MIDDAY_COMPRESSION": {
        "CONSERVATIVE": ExplorationPolicy(
            phase="MIDDAY_COMPRESSION",
            level=ExplorationLevel.DISABLED,
            scouts_enabled=False,
            max_scouts_per_hour=0,
            scout_size_multiplier=0.0,
            chronos_override_allowed=False,
            reasons=["No scouts during choppy midday"]
        ),
        "NEUTRAL": ExplorationPolicy(
            phase="MIDDAY_COMPRESSION",
            level=ExplorationLevel.DISABLED,
            scouts_enabled=False,
            max_scouts_per_hour=0,
            scout_size_multiplier=0.0,
            chronos_override_allowed=False,
            reasons=["Avoid false breakouts in compression"]
        ),
        "AGGRESSIVE": ExplorationPolicy(
            phase="MIDDAY_COMPRESSION",
            level=ExplorationLevel.MINIMAL,
            scouts_enabled=True,
            max_scouts_per_hour=1,
            scout_size_multiplier=0.10,
            chronos_override_allowed=False,
            reasons=["Very limited, only for exceptional setups"]
        )
    },

    # POWER_HOUR - Late day momentum
    "POWER_HOUR": {
        "CONSERVATIVE": ExplorationPolicy(
            phase="POWER_HOUR",
            level=ExplorationLevel.MINIMAL,
            scouts_enabled=True,
            max_scouts_per_hour=1,
            scout_size_multiplier=0.12,
            chronos_override_allowed=False,
            reasons=["Late day, limited new entries"]
        ),
        "NEUTRAL": ExplorationPolicy(
            phase="POWER_HOUR",
            level=ExplorationLevel.NORMAL,
            scouts_enabled=True,
            max_scouts_per_hour=3,
            scout_size_multiplier=0.18,
            chronos_override_allowed=True,
            reasons=["Institutional flow can create opportunities"]
        ),
        "AGGRESSIVE": ExplorationPolicy(
            phase="POWER_HOUR",
            level=ExplorationLevel.NORMAL,
            scouts_enabled=True,
            max_scouts_per_hour=4,
            scout_size_multiplier=0.20,
            chronos_override_allowed=True,
            reasons=["Power hour breakouts possible"]
        )
    },

    # AFTER_HOURS - Low liquidity
    "AFTER_HOURS": {
        "CONSERVATIVE": ExplorationPolicy(
            phase="AFTER_HOURS",
            level=ExplorationLevel.DISABLED,
            scouts_enabled=False,
            max_scouts_per_hour=0,
            scout_size_multiplier=0.0,
            chronos_override_allowed=False,
            reasons=["Too illiquid for scouts"]
        ),
        "NEUTRAL": ExplorationPolicy(
            phase="AFTER_HOURS",
            level=ExplorationLevel.DISABLED,
            scouts_enabled=False,
            max_scouts_per_hour=0,
            scout_size_multiplier=0.0,
            chronos_override_allowed=False,
            reasons=["Avoid after-hours exploration"]
        ),
        "AGGRESSIVE": ExplorationPolicy(
            phase="AFTER_HOURS",
            level=ExplorationLevel.MINIMAL,
            scouts_enabled=True,
            max_scouts_per_hour=1,
            scout_size_multiplier=0.08,
            chronos_override_allowed=False,
            reasons=["News-driven only, very small"]
        )
    },

    # CLOSED - No trading
    "CLOSED": {
        "CONSERVATIVE": ExplorationPolicy(
            phase="CLOSED",
            level=ExplorationLevel.DISABLED,
            scouts_enabled=False,
            max_scouts_per_hour=0,
            scout_size_multiplier=0.0,
            chronos_override_allowed=False,
            reasons=["Market closed"]
        ),
        "NEUTRAL": ExplorationPolicy(
            phase="CLOSED",
            level=ExplorationLevel.DISABLED,
            scouts_enabled=False,
            max_scouts_per_hour=0,
            scout_size_multiplier=0.0,
            chronos_override_allowed=False,
            reasons=["Market closed"]
        ),
        "AGGRESSIVE": ExplorationPolicy(
            phase="CLOSED",
            level=ExplorationLevel.DISABLED,
            scouts_enabled=False,
            max_scouts_per_hour=0,
            scout_size_multiplier=0.0,
            chronos_override_allowed=False,
            reasons=["Market closed"]
        )
    }
}


class ExplorationPolicyManager:
    """
    Manages exploration policy based on market phase and baseline.

    Responsibilities:
    - Determine current exploration level
    - Apply policy to scout mode
    - Log policy changes
    """

    def __init__(self):
        self.policies = DEFAULT_POLICIES
        self.current_policy: Optional[ExplorationPolicy] = None
        self.policy_history: List[Dict] = []

    def get_current_policy(self) -> ExplorationPolicy:
        """Get current exploration policy based on phase and baseline"""
        try:
            # Get current phase
            from ai.market_phases import get_phase_manager
            phase_manager = get_phase_manager()
            phase = phase_manager.current_phase.value if phase_manager.current_phase else "CLOSED"

            # Get current baseline
            from ai.baseline_profiles import get_baseline_manager
            baseline_manager = get_baseline_manager()
            baseline = baseline_manager.current_profile.value

            # Look up policy
            phase_policies = self.policies.get(phase, self.policies.get("CLOSED"))
            policy = phase_policies.get(baseline, phase_policies.get("NEUTRAL"))

            # Log if policy changed
            if self.current_policy is None or self._policy_changed(policy):
                self._log_policy_change(policy, phase, baseline)

            self.current_policy = policy
            return policy

        except Exception as e:
            logger.warning(f"Error getting exploration policy: {e}")
            # Return disabled policy on error
            return ExplorationPolicy(
                phase="ERROR",
                level=ExplorationLevel.DISABLED,
                scouts_enabled=False,
                max_scouts_per_hour=0,
                scout_size_multiplier=0.0,
                chronos_override_allowed=False,
                reasons=[f"Error: {e}"]
            )

    def _policy_changed(self, new_policy: ExplorationPolicy) -> bool:
        """Check if policy has changed"""
        if self.current_policy is None:
            return True
        return (
            self.current_policy.phase != new_policy.phase or
            self.current_policy.level != new_policy.level
        )

    def _log_policy_change(self, policy: ExplorationPolicy, phase: str, baseline: str):
        """Log policy change"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "baseline": baseline,
            "level": policy.level.value,
            "scouts_enabled": policy.scouts_enabled,
            "max_scouts_per_hour": policy.max_scouts_per_hour,
            "reasons": policy.reasons
        }

        self.policy_history.append(record)
        if len(self.policy_history) > 100:
            self.policy_history = self.policy_history[-100:]

        logger.info(
            f"MARKET_PHASE_EXPLORATION_POLICY: phase={phase} baseline={baseline} "
            f"level={policy.level.value} scouts={policy.scouts_enabled}"
        )

    def apply_policy_to_scout(self):
        """Apply current policy to momentum scout"""
        try:
            from ai.momentum_scout import get_momentum_scout

            policy = self.get_current_policy()
            scout = get_momentum_scout()

            # Enable/disable scouts
            if policy.scouts_enabled and not scout.is_enabled():
                scout.enable()
                logger.info(f"Scouts ENABLED by exploration policy: {policy.phase}")
            elif not policy.scouts_enabled and scout.is_enabled():
                scout.disable()
                logger.info(f"Scouts DISABLED by exploration policy: {policy.phase}")

            # Update scout config
            scout.config.max_per_hour = policy.max_scouts_per_hour
            scout.config.size_multiplier = policy.scout_size_multiplier

            return True

        except Exception as e:
            logger.error(f"Error applying exploration policy: {e}")
            return False

    def is_exploration_allowed(self) -> Tuple[bool, str]:
        """Check if exploration (scouting) is currently allowed"""
        policy = self.get_current_policy()

        if policy.level == ExplorationLevel.DISABLED:
            return False, f"Exploration disabled in {policy.phase}"

        if not policy.scouts_enabled:
            return False, f"Scouts disabled for {policy.phase}"

        return True, f"Exploration level: {policy.level.value}"

    def get_exploration_level(self) -> ExplorationLevel:
        """Get current exploration level"""
        return self.get_current_policy().level

    def get_status(self) -> Dict:
        """Get exploration policy status"""
        policy = self.get_current_policy()

        return {
            "current_policy": policy.to_dict(),
            "exploration_allowed": self.is_exploration_allowed()[0],
            "phase_history": self.policy_history[-10:],
            "all_phases": list(self.policies.keys()),
            "timestamp": datetime.now().isoformat()
        }

    def get_policy_matrix(self) -> Dict:
        """Get full policy matrix for all phases and baselines"""
        matrix = {}
        for phase, baselines in self.policies.items():
            matrix[phase] = {}
            for baseline, policy in baselines.items():
                matrix[phase][baseline] = {
                    "level": policy.level.value,
                    "scouts_enabled": policy.scouts_enabled,
                    "max_scouts": policy.max_scouts_per_hour,
                    "size_mult": policy.scout_size_multiplier
                }
        return matrix


# Singleton instance
_manager: Optional[ExplorationPolicyManager] = None


def get_exploration_manager() -> ExplorationPolicyManager:
    """Get the singleton exploration policy manager"""
    global _manager
    if _manager is None:
        _manager = ExplorationPolicyManager()
    return _manager


# Convenience functions
def is_exploration_allowed() -> Tuple[bool, str]:
    """Check if exploration is allowed"""
    return get_exploration_manager().is_exploration_allowed()


def apply_exploration_policy():
    """Apply current exploration policy to scout"""
    return get_exploration_manager().apply_policy_to_scout()


def get_exploration_level() -> ExplorationLevel:
    """Get current exploration level"""
    return get_exploration_manager().get_exploration_level()
