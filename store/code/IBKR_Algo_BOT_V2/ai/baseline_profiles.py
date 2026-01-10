"""
Baseline Profile System (Task G)
================================
Adaptive baseline profiles that respond to market conditions.

Profiles:
- CONSERVATIVE: Tight parameters for weak/volatile markets
- NEUTRAL: Default balanced parameters
- AGGRESSIVE: Expanded parameters for strong trending markets

All profiles stored in config file, not hardcoded.
"""

import json
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

# Config file location
BASELINE_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "baseline_profiles_config.json")


class BaselineProfile(Enum):
    """Available baseline profiles"""
    CONSERVATIVE = "CONSERVATIVE"
    NEUTRAL = "NEUTRAL"
    AGGRESSIVE = "AGGRESSIVE"


@dataclass
class ProfileParameters:
    """
    Parameters controlled by baseline profile.

    These are the tunable knobs that change based on market conditions.
    Hard risk limits (max loss, position limits) are NOT included - those never change.
    """
    # Profile identity
    name: str
    description: str

    # Relative Volume floor (minimum rel vol to consider)
    rel_vol_floor: float = 2.0  # Bounded: 1.5 - 5.0
    rel_vol_ceiling: float = 10.0  # Max rel vol (avoid manipulation)

    # Chronos confidence thresholds
    chronos_micro_confidence_min: float = 0.60  # Min micro confidence
    chronos_macro_confidence_min: float = 0.50  # Min macro confidence

    # Probe entry parameters (Task F integration)
    probe_enabled: bool = True
    probe_size_multiplier: float = 0.30  # 25-33% of normal
    probe_max_per_hour: int = 3

    # Scalper aggressiveness tier (1=conservative, 2=neutral, 3=aggressive)
    scalper_aggressiveness: int = 2
    scalper_min_spike_percent: float = 3.0
    scalper_min_volume_surge: float = 2.0

    # Cooldown durations (seconds)
    cooldown_after_loss: int = 60
    cooldown_after_stopped_probe: int = 900  # 15 minutes
    cooldown_between_entries: int = 30

    # Entry quality thresholds
    min_entry_score: float = 0.50
    min_warrior_grade: str = "B"

    # Gap trading parameters
    gap_min_percent: float = 5.0
    gap_require_catalyst: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


# Default profile configurations
DEFAULT_PROFILES = {
    BaselineProfile.CONSERVATIVE.value: ProfileParameters(
        name="CONSERVATIVE",
        description="Tight parameters for weak/volatile markets. Prioritizes capital preservation.",
        rel_vol_floor=3.0,
        rel_vol_ceiling=8.0,
        chronos_micro_confidence_min=0.70,
        chronos_macro_confidence_min=0.60,
        probe_enabled=False,  # No probes in conservative mode
        probe_size_multiplier=0.25,
        probe_max_per_hour=1,
        scalper_aggressiveness=1,
        scalper_min_spike_percent=5.0,
        scalper_min_volume_surge=3.0,
        cooldown_after_loss=120,  # 2 minutes
        cooldown_after_stopped_probe=1800,  # 30 minutes
        cooldown_between_entries=60,
        min_entry_score=0.60,
        min_warrior_grade="A",
        gap_min_percent=7.0,
        gap_require_catalyst=True
    ),
    BaselineProfile.NEUTRAL.value: ProfileParameters(
        name="NEUTRAL",
        description="Default balanced parameters. Standard operating mode.",
        rel_vol_floor=2.0,
        rel_vol_ceiling=10.0,
        chronos_micro_confidence_min=0.60,
        chronos_macro_confidence_min=0.50,
        probe_enabled=True,
        probe_size_multiplier=0.30,
        probe_max_per_hour=3,
        scalper_aggressiveness=2,
        scalper_min_spike_percent=3.0,
        scalper_min_volume_surge=2.0,
        cooldown_after_loss=60,
        cooldown_after_stopped_probe=900,
        cooldown_between_entries=30,
        min_entry_score=0.50,
        min_warrior_grade="B",
        gap_min_percent=5.0,
        gap_require_catalyst=True
    ),
    BaselineProfile.AGGRESSIVE.value: ProfileParameters(
        name="AGGRESSIVE",
        description="Expanded parameters for strong trending markets. More opportunities, still safe.",
        rel_vol_floor=1.5,
        rel_vol_ceiling=15.0,
        chronos_micro_confidence_min=0.55,
        chronos_macro_confidence_min=0.45,
        probe_enabled=True,
        probe_size_multiplier=0.33,
        probe_max_per_hour=5,
        scalper_aggressiveness=3,
        scalper_min_spike_percent=2.5,
        scalper_min_volume_surge=1.5,
        cooldown_after_loss=45,
        cooldown_after_stopped_probe=600,  # 10 minutes
        cooldown_between_entries=20,
        min_entry_score=0.45,
        min_warrior_grade="B",
        gap_min_percent=4.0,
        gap_require_catalyst=False  # Allow technical gaps
    )
}


class BaselineProfileManager:
    """
    Manages baseline profiles and their application.

    Only one profile active at a time.
    All profile changes are logged.
    """

    def __init__(self):
        self.current_profile: BaselineProfile = BaselineProfile.NEUTRAL
        self.current_params: ProfileParameters = None
        self.profiles: Dict[str, ProfileParameters] = {}
        self.profile_history: List[Dict] = []
        self.last_change_time: Optional[datetime] = None
        self.lock_until: Optional[datetime] = None
        self.change_reason: str = "Initial default"

        # Load profiles from config (or use defaults)
        self._load_profiles()

        # Set initial profile
        self._apply_profile(BaselineProfile.NEUTRAL, "System startup - default profile")

    def _load_profiles(self):
        """Load profiles from config file or use defaults"""
        try:
            if os.path.exists(BASELINE_CONFIG_FILE):
                with open(BASELINE_CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    for name, params in data.get('profiles', {}).items():
                        self.profiles[name] = ProfileParameters(**params)
                logger.info(f"Loaded {len(self.profiles)} baseline profiles from config")
            else:
                # Use defaults
                self.profiles = {k: v for k, v in DEFAULT_PROFILES.items()}
                self._save_profiles()
                logger.info("Created default baseline profiles config")
        except Exception as e:
            logger.error(f"Error loading profiles: {e}, using defaults")
            self.profiles = {k: v for k, v in DEFAULT_PROFILES.items()}

    def _save_profiles(self):
        """Save profiles to config file"""
        try:
            data = {
                'profiles': {k: v.to_dict() for k, v in self.profiles.items()},
                'current_profile': self.current_profile.value,
                'last_updated': datetime.now().isoformat()
            }
            with open(BASELINE_CONFIG_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving profiles: {e}")

    def _apply_profile(self, profile: BaselineProfile, reason: str) -> bool:
        """
        Apply a profile change.

        Returns True if applied, False if locked or same profile.
        """
        # Check lock
        if self.lock_until and datetime.now() < self.lock_until:
            remaining = (self.lock_until - datetime.now()).total_seconds()
            logger.info(f"Profile change blocked - locked for {remaining:.0f}s more")
            return False

        # Check if same profile
        if profile == self.current_profile:
            logger.debug(f"Profile already {profile.value}, no change needed")
            return False

        # Get profile parameters
        if profile.value not in self.profiles:
            logger.error(f"Profile {profile.value} not found in config")
            return False

        old_profile = self.current_profile
        self.current_profile = profile
        self.current_params = self.profiles[profile.value]
        self.last_change_time = datetime.now()
        self.change_reason = reason

        # Log the change
        change_record = {
            "timestamp": datetime.now().isoformat(),
            "from_profile": old_profile.value,
            "to_profile": profile.value,
            "reason": reason
        }
        self.profile_history.append(change_record)

        # Keep history bounded
        if len(self.profile_history) > 100:
            self.profile_history = self.profile_history[-100:]

        logger.info(f"BASELINE_PROFILE_SELECTED: {old_profile.value} -> {profile.value} | Reason: {reason}")

        return True

    def set_profile(self, profile: BaselineProfile, reason: str, lock_minutes: int = 30) -> bool:
        """
        Set the active profile with optional lock duration.

        Args:
            profile: The profile to activate
            reason: Why this profile was selected
            lock_minutes: How long to lock this profile (prevents thrashing)
        """
        success = self._apply_profile(profile, reason)

        if success and lock_minutes > 0:
            self.lock_until = datetime.now() + timedelta(minutes=lock_minutes)
            logger.info(f"Profile locked for {lock_minutes} minutes")

        return success

    def get_current_params(self) -> ProfileParameters:
        """Get current profile parameters"""
        if self.current_params is None:
            self.current_params = self.profiles.get(
                self.current_profile.value,
                DEFAULT_PROFILES[BaselineProfile.NEUTRAL.value]
            )
        return self.current_params

    def get_status(self) -> Dict:
        """Get full status for observability"""
        params = self.get_current_params()

        lock_remaining = 0
        if self.lock_until and datetime.now() < self.lock_until:
            lock_remaining = (self.lock_until - datetime.now()).total_seconds()

        return {
            "current_profile": self.current_profile.value,
            "profile_description": params.description,
            "last_change_time": self.last_change_time.isoformat() if self.last_change_time else None,
            "change_reason": self.change_reason,
            "locked": lock_remaining > 0,
            "lock_remaining_seconds": lock_remaining,
            "next_change_allowed": self.lock_until.isoformat() if self.lock_until else None,
            "parameters": params.to_dict(),
            "recent_changes": self.profile_history[-10:] if self.profile_history else []
        }

    def get_param(self, param_name: str, default=None):
        """Get a specific parameter value from current profile"""
        params = self.get_current_params()
        return getattr(params, param_name, default)

    def update_profile_param(self, profile_name: str, param_name: str, value) -> bool:
        """Update a specific parameter in a profile (admin function)"""
        if profile_name not in self.profiles:
            return False

        profile = self.profiles[profile_name]
        if hasattr(profile, param_name):
            setattr(profile, param_name, value)
            self._save_profiles()

            # If updating current profile, refresh params
            if profile_name == self.current_profile.value:
                self.current_params = profile

            logger.info(f"Updated {profile_name}.{param_name} = {value}")
            return True
        return False

    def force_unlock(self) -> bool:
        """Force unlock profile (admin override)"""
        if self.lock_until:
            self.lock_until = None
            logger.warning("Profile lock forcibly removed by admin")
            return True
        return False


# Need to import timedelta
from datetime import timedelta

# Singleton instance
_manager: Optional[BaselineProfileManager] = None


def get_baseline_manager() -> BaselineProfileManager:
    """Get the singleton baseline profile manager"""
    global _manager
    if _manager is None:
        _manager = BaselineProfileManager()
    return _manager


def get_current_profile() -> BaselineProfile:
    """Get the currently active profile"""
    return get_baseline_manager().current_profile


def get_profile_param(param_name: str, default=None):
    """Get a parameter from the current profile"""
    return get_baseline_manager().get_param(param_name, default)
