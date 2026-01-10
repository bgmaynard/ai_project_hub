"""
Profile Selection Logic (Task I)
================================
Selects baseline profile based on market conditions at defined checkpoints.

Checkpoints:
- Pre-market (4:00 AM - 9:30 AM ET)
- Post-open (+15-30 min after open, 9:45-10:00 AM ET)
- Midday (12:00 PM ET)

Profile lock: Minimum 30 minutes between changes to prevent thrashing.
"""

import logging
import asyncio
import pytz
from datetime import datetime, time, timedelta
from typing import Dict, Optional, List
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Checkpoint(Enum):
    """Trading session checkpoints for profile evaluation"""
    PREMARKET = "PREMARKET"          # 4:00 AM - 9:30 AM ET
    POST_OPEN = "POST_OPEN"          # 9:45 AM - 10:00 AM ET (+15-30 min after open)
    MIDDAY = "MIDDAY"                # 12:00 PM ET
    AFTERNOON = "AFTERNOON"          # 2:00 PM ET
    CLOSE = "CLOSE"                  # 3:30 PM ET


@dataclass
class CheckpointConfig:
    """Configuration for a checkpoint"""
    name: Checkpoint
    start_time: time  # When this checkpoint window opens (ET)
    end_time: time    # When this checkpoint window closes (ET)
    priority: int     # Higher priority checkpoints override lower


# Define checkpoint windows
CHECKPOINT_WINDOWS = {
    Checkpoint.PREMARKET: CheckpointConfig(
        name=Checkpoint.PREMARKET,
        start_time=time(4, 0),
        end_time=time(9, 30),
        priority=1
    ),
    Checkpoint.POST_OPEN: CheckpointConfig(
        name=Checkpoint.POST_OPEN,
        start_time=time(9, 45),
        end_time=time(10, 0),
        priority=2  # Higher priority - important checkpoint
    ),
    Checkpoint.MIDDAY: CheckpointConfig(
        name=Checkpoint.MIDDAY,
        start_time=time(12, 0),
        end_time=time(12, 30),
        priority=1
    ),
    Checkpoint.AFTERNOON: CheckpointConfig(
        name=Checkpoint.AFTERNOON,
        start_time=time(14, 0),
        end_time=time(14, 30),
        priority=1
    ),
    Checkpoint.CLOSE: CheckpointConfig(
        name=Checkpoint.CLOSE,
        start_time=time(15, 30),
        end_time=time(16, 0),
        priority=1
    )
}


class ProfileSelector:
    """
    Selects baseline profile at checkpoints based on market conditions.

    Rules:
    - Only evaluate at defined checkpoints
    - Lock profile for minimum duration (default 30 min)
    - Map market condition to profile
    - Emit BASELINE_PROFILE_SELECTED event
    """

    def __init__(self):
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self.last_checkpoint: Optional[Checkpoint] = None
        self.last_evaluation_time: Optional[datetime] = None
        self.min_lock_minutes = 30
        self.evaluation_history: List[Dict] = []

        # Condition to profile mapping
        self.condition_profile_map = {
            "WEAK": "CONSERVATIVE",
            "MIXED": "NEUTRAL",
            "STRONG": "AGGRESSIVE"
        }

    def _get_current_checkpoint(self) -> Optional[Checkpoint]:
        """Get the current checkpoint based on time of day (ET)"""
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz).time()

        for checkpoint, config in CHECKPOINT_WINDOWS.items():
            if config.start_time <= now_et <= config.end_time:
                return checkpoint

        return None

    def _should_evaluate(self) -> bool:
        """Check if we should run an evaluation now"""
        current_checkpoint = self._get_current_checkpoint()

        if current_checkpoint is None:
            # Not in any checkpoint window
            return False

        if current_checkpoint == self.last_checkpoint:
            # Already evaluated this checkpoint
            if self.last_evaluation_time:
                # Unless it's been more than the lock duration
                elapsed = (datetime.now() - self.last_evaluation_time).total_seconds()
                if elapsed < self.min_lock_minutes * 60:
                    return False

        return True

    async def evaluate_and_select(self, force: bool = False) -> Dict:
        """
        Evaluate market conditions and select appropriate profile.

        Returns selection result dict.
        """
        from ai.market_condition_evaluator import get_market_evaluator, MarketCondition
        from ai.baseline_profiles import get_baseline_manager, BaselineProfile

        current_checkpoint = self._get_current_checkpoint()

        # Check if we should evaluate
        if not force and not self._should_evaluate():
            manager = get_baseline_manager()
            return {
                "action": "SKIPPED",
                "reason": "Not in checkpoint window or recently evaluated",
                "current_profile": manager.current_profile.value,
                "checkpoint": current_checkpoint.value if current_checkpoint else None
            }

        # Evaluate market conditions
        evaluator = get_market_evaluator()
        condition_result = await evaluator.evaluate(force=True)

        # Map condition to profile
        profile_name = self.condition_profile_map.get(
            condition_result.market_condition.value,
            "NEUTRAL"
        )
        target_profile = BaselineProfile(profile_name)

        # Get baseline manager
        manager = get_baseline_manager()
        old_profile = manager.current_profile

        # Build selection reason
        reason = (
            f"Checkpoint: {current_checkpoint.value if current_checkpoint else 'MANUAL'} | "
            f"Market: {condition_result.market_condition.value} "
            f"(confidence={condition_result.confidence:.0%})"
        )

        # Attempt to set profile
        success = manager.set_profile(
            target_profile,
            reason,
            lock_minutes=self.min_lock_minutes
        )

        # Update tracking
        self.last_checkpoint = current_checkpoint
        self.last_evaluation_time = datetime.now()

        # Build result
        result = {
            "action": "PROFILE_SELECTED" if success else "PROFILE_LOCKED",
            "checkpoint": current_checkpoint.value if current_checkpoint else "MANUAL",
            "market_condition": condition_result.market_condition.value,
            "market_confidence": condition_result.confidence,
            "market_reasons": condition_result.reasons,
            "old_profile": old_profile.value,
            "new_profile": target_profile.value,
            "profile_changed": success and old_profile != target_profile,
            "lock_minutes": self.min_lock_minutes,
            "timestamp": datetime.now().isoformat()
        }

        # Log the event
        if success:
            logger.info(
                f"BASELINE_PROFILE_SELECTED: {target_profile.value} | "
                f"Market={condition_result.market_condition.value} | "
                f"Checkpoint={current_checkpoint.value if current_checkpoint else 'MANUAL'}"
            )
        else:
            logger.info(f"Profile selection blocked (locked): would have selected {target_profile.value}")

        # Add to history
        self.evaluation_history.append(result)
        if len(self.evaluation_history) > 50:
            self.evaluation_history = self.evaluation_history[-50:]

        return result

    async def _selector_loop(self):
        """Background loop that checks checkpoints and evaluates"""
        logger.info("Profile selector started")

        while self.running:
            try:
                # Check if we're in a checkpoint window
                if self._should_evaluate():
                    await self.evaluate_and_select()

                # Sleep for a while (check every minute)
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Profile selector error: {e}")
                await asyncio.sleep(30)

        logger.info("Profile selector stopped")

    def start(self):
        """Start the profile selector background task"""
        if self.running:
            return

        self.running = True
        self._task = asyncio.create_task(self._selector_loop())
        logger.info("Profile selector starting...")

    def stop(self):
        """Stop the profile selector"""
        self.running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("Profile selector stopped")

    def get_status(self) -> Dict:
        """Get selector status"""
        from ai.baseline_profiles import get_baseline_manager

        manager = get_baseline_manager()
        current_checkpoint = self._get_current_checkpoint()

        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)

        # Find next checkpoint
        next_checkpoint = None
        next_checkpoint_time = None
        for checkpoint, config in CHECKPOINT_WINDOWS.items():
            checkpoint_dt = now_et.replace(
                hour=config.start_time.hour,
                minute=config.start_time.minute,
                second=0
            )
            if checkpoint_dt > now_et:
                if next_checkpoint_time is None or checkpoint_dt < next_checkpoint_time:
                    next_checkpoint = checkpoint
                    next_checkpoint_time = checkpoint_dt

        return {
            "running": self.running,
            "current_checkpoint": current_checkpoint.value if current_checkpoint else None,
            "in_checkpoint_window": current_checkpoint is not None,
            "last_checkpoint": self.last_checkpoint.value if self.last_checkpoint else None,
            "last_evaluation_time": self.last_evaluation_time.isoformat() if self.last_evaluation_time else None,
            "next_checkpoint": next_checkpoint.value if next_checkpoint else None,
            "next_checkpoint_time": next_checkpoint_time.isoformat() if next_checkpoint_time else None,
            "current_profile": manager.current_profile.value,
            "min_lock_minutes": self.min_lock_minutes,
            "condition_profile_map": self.condition_profile_map,
            "checkpoint_windows": {
                k.value: {
                    "start": v.start_time.strftime("%H:%M"),
                    "end": v.end_time.strftime("%H:%M")
                }
                for k, v in CHECKPOINT_WINDOWS.items()
            },
            "recent_evaluations": self.evaluation_history[-5:] if self.evaluation_history else []
        }


# Singleton instance
_selector: Optional[ProfileSelector] = None


def get_profile_selector() -> ProfileSelector:
    """Get the singleton profile selector"""
    global _selector
    if _selector is None:
        _selector = ProfileSelector()
    return _selector


def start_profile_selector():
    """Start the profile selector"""
    get_profile_selector().start()


def stop_profile_selector():
    """Stop the profile selector"""
    get_profile_selector().stop()
