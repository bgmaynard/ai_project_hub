"""
Trading Window Watchdog
=======================
Ensures scalper is running and enabled during trading windows.
Addresses the "no trades in 2 weeks" problem where scalper wasn't started.

Auto-enables scalper during configured trading windows.
Emits events and logs when intervention is needed.
"""

import asyncio
import logging
from datetime import datetime, time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import pytz

logger = logging.getLogger(__name__)


@dataclass
class WatchdogEvent:
    """Event emitted by the watchdog"""
    event_type: str
    timestamp: str
    details: Dict[str, Any]
    action_taken: str


@dataclass
class WatchdogConfig:
    """Watchdog configuration"""
    enabled: bool = True
    check_interval_seconds: int = 30
    auto_enable_scalper: bool = True
    auto_start_scalper: bool = True
    paper_mode_only: bool = True  # Only auto-enable in paper mode

    # Trading windows (Eastern Time)
    premarket_start: time = field(default_factory=lambda: time(4, 0))   # 4:00 AM ET
    premarket_end: time = field(default_factory=lambda: time(9, 30))    # 9:30 AM ET
    market_start: time = field(default_factory=lambda: time(9, 30))     # 9:30 AM ET
    market_end: time = field(default_factory=lambda: time(16, 0))       # 4:00 PM ET
    afterhours_start: time = field(default_factory=lambda: time(16, 0)) # 4:00 PM ET
    afterhours_end: time = field(default_factory=lambda: time(20, 0))   # 8:00 PM ET

    # Which windows to auto-enable
    enable_in_premarket: bool = True
    enable_in_market: bool = True
    enable_in_afterhours: bool = False  # Usually don't want AH trading


class TradingWindowWatchdog:
    """
    Watchdog that ensures scalper is running during trading windows.

    Checks every 30-60 seconds and auto-enables if needed.
    """

    def __init__(self, config: WatchdogConfig = None):
        self.config = config or WatchdogConfig()
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self.et_tz = pytz.timezone('US/Eastern')

        # Event history
        self.events: List[WatchdogEvent] = []
        self.max_events = 100

        # Stats
        self.check_count = 0
        self.interventions = 0
        self.last_check_time: Optional[datetime] = None
        self.last_intervention_time: Optional[datetime] = None

        logger.info("TradingWindowWatchdog initialized")

    def get_current_window(self) -> Dict[str, Any]:
        """Get current trading window status"""
        now_et = datetime.now(self.et_tz)
        current_time = now_et.time()

        window = "CLOSED"
        window_detail = "Market closed"
        trading_allowed = False
        auto_enable = False

        if self.config.premarket_start <= current_time < self.config.premarket_end:
            window = "PRE_MARKET"
            window_detail = f"Pre-market ({self.config.premarket_start.strftime('%H:%M')} - {self.config.premarket_end.strftime('%H:%M')} ET)"
            trading_allowed = True
            auto_enable = self.config.enable_in_premarket
        elif self.config.market_start <= current_time < self.config.market_end:
            window = "MARKET_HOURS"
            window_detail = f"Market hours ({self.config.market_start.strftime('%H:%M')} - {self.config.market_end.strftime('%H:%M')} ET)"
            trading_allowed = True
            auto_enable = self.config.enable_in_market
        elif self.config.afterhours_start <= current_time < self.config.afterhours_end:
            window = "AFTER_HOURS"
            window_detail = f"After hours ({self.config.afterhours_start.strftime('%H:%M')} - {self.config.afterhours_end.strftime('%H:%M')} ET)"
            trading_allowed = True
            auto_enable = self.config.enable_in_afterhours

        return {
            "current_time_et": now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
            "window": window,
            "window_detail": window_detail,
            "trading_allowed": trading_allowed,
            "auto_enable_configured": auto_enable,
            "is_weekend": now_et.weekday() >= 5
        }

    def is_trading_window(self) -> bool:
        """Check if currently in a trading window"""
        now_et = datetime.now(self.et_tz)

        # Skip weekends
        if now_et.weekday() >= 5:
            return False

        current_time = now_et.time()

        # Check pre-market
        if self.config.enable_in_premarket:
            if self.config.premarket_start <= current_time < self.config.premarket_end:
                return True

        # Check market hours
        if self.config.enable_in_market:
            if self.config.market_start <= current_time < self.config.market_end:
                return True

        # Check after hours
        if self.config.enable_in_afterhours:
            if self.config.afterhours_start <= current_time < self.config.afterhours_end:
                return True

        return False

    async def _get_scalper_status(self) -> Dict[str, Any]:
        """Get scalper status via internal API"""
        try:
            from ai.hft_scalper import get_hft_scalper
            scalper = get_hft_scalper()
            return {
                "is_running": scalper.is_running,
                "enabled": scalper.config.enabled,
                "paper_mode": scalper.config.paper_mode,
                "watchlist_count": len(scalper.watchlist)
            }
        except Exception as e:
            logger.error(f"Failed to get scalper status: {e}")
            return {
                "is_running": False,
                "enabled": False,
                "paper_mode": True,
                "error": str(e)
            }

    async def _start_scalper(self) -> bool:
        """Start the scalper"""
        try:
            from ai.hft_scalper import get_hft_scalper
            scalper = get_hft_scalper()

            if not scalper.is_running:
                await scalper.start()
                logger.info("Watchdog: Started scalper")
                return True
            return False
        except Exception as e:
            logger.error(f"Watchdog: Failed to start scalper: {e}")
            return False

    async def _enable_scalper(self) -> bool:
        """Enable the scalper for trading"""
        try:
            from ai.hft_scalper import get_hft_scalper
            scalper = get_hft_scalper()

            # Only enable in paper mode if configured
            if self.config.paper_mode_only and not scalper.config.paper_mode:
                logger.warning("Watchdog: Not enabling scalper - paper_mode_only is set but scalper is in live mode")
                return False

            if not scalper.config.enabled:
                scalper.config.enabled = True
                scalper._save_config()
                logger.info("Watchdog: Enabled scalper for trading")
                return True
            return False
        except Exception as e:
            logger.error(f"Watchdog: Failed to enable scalper: {e}")
            return False

    def _emit_event(self, event_type: str, details: Dict[str, Any], action_taken: str):
        """Emit and log a watchdog event"""
        event = WatchdogEvent(
            event_type=event_type,
            timestamp=datetime.now(self.et_tz).isoformat(),
            details=details,
            action_taken=action_taken
        )

        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        # Log based on severity
        if event_type == "SCALPER_NOT_RUNNING_DURING_WINDOW":
            logger.warning(f"WATCHDOG EVENT: {event_type} - {action_taken}")
        elif event_type == "SCALPER_NOT_ENABLED_DURING_WINDOW":
            logger.warning(f"WATCHDOG EVENT: {event_type} - {action_taken}")
        else:
            logger.info(f"WATCHDOG EVENT: {event_type} - {action_taken}")

    async def check_and_intervene(self) -> Dict[str, Any]:
        """
        Check scalper status and intervene if needed.

        Returns dict with check results and any actions taken.
        """
        self.check_count += 1
        self.last_check_time = datetime.now(self.et_tz)

        result = {
            "timestamp": self.last_check_time.isoformat(),
            "in_trading_window": False,
            "window_info": self.get_current_window(),
            "scalper_status": {},
            "intervention_needed": False,
            "actions_taken": []
        }

        # Check if in trading window
        if not self.is_trading_window():
            result["reason"] = "Not in trading window"
            return result

        result["in_trading_window"] = True

        # Get scalper status
        scalper_status = await self._get_scalper_status()
        result["scalper_status"] = scalper_status

        actions_taken = []

        # Check if scalper is running
        if not scalper_status.get("is_running", False):
            result["intervention_needed"] = True

            self._emit_event(
                "SCALPER_NOT_RUNNING_DURING_WINDOW",
                {"window": result["window_info"]["window"], "scalper_status": scalper_status},
                "Attempting to start scalper" if self.config.auto_start_scalper else "Alert only"
            )

            if self.config.auto_start_scalper:
                started = await self._start_scalper()
                if started:
                    actions_taken.append("Started scalper")
                    self.interventions += 1
                    self.last_intervention_time = datetime.now(self.et_tz)

        # Refresh status after potential start
        scalper_status = await self._get_scalper_status()
        result["scalper_status"] = scalper_status

        # Check if scalper is enabled
        if scalper_status.get("is_running") and not scalper_status.get("enabled", False):
            result["intervention_needed"] = True

            self._emit_event(
                "SCALPER_NOT_ENABLED_DURING_WINDOW",
                {"window": result["window_info"]["window"], "scalper_status": scalper_status},
                "Attempting to enable scalper" if self.config.auto_enable_scalper else "Alert only"
            )

            if self.config.auto_enable_scalper:
                enabled = await self._enable_scalper()
                if enabled:
                    actions_taken.append("Enabled scalper for trading")
                    self.interventions += 1
                    self.last_intervention_time = datetime.now(self.et_tz)

        result["actions_taken"] = actions_taken

        # Final status
        result["scalper_status"] = await self._get_scalper_status()

        return result

    async def _watchdog_loop(self):
        """Main watchdog loop"""
        logger.info(f"Watchdog loop started (interval: {self.config.check_interval_seconds}s)")

        while self.running:
            try:
                result = await self.check_and_intervene()

                if result.get("actions_taken"):
                    logger.info(f"Watchdog intervention: {result['actions_taken']}")

            except Exception as e:
                logger.error(f"Watchdog check failed: {e}")

            await asyncio.sleep(self.config.check_interval_seconds)

    async def start(self):
        """Start the watchdog"""
        if self.running:
            logger.warning("Watchdog already running")
            return

        self.running = True
        self._task = asyncio.create_task(self._watchdog_loop())
        logger.info("Trading Window Watchdog started")

    async def stop(self):
        """Stop the watchdog"""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Trading Window Watchdog stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get watchdog status"""
        return {
            "enabled": self.config.enabled,
            "running": self.running,
            "check_interval_seconds": self.config.check_interval_seconds,
            "auto_start_scalper": self.config.auto_start_scalper,
            "auto_enable_scalper": self.config.auto_enable_scalper,
            "paper_mode_only": self.config.paper_mode_only,
            "current_window": self.get_current_window(),
            "stats": {
                "check_count": self.check_count,
                "interventions": self.interventions,
                "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
                "last_intervention_time": self.last_intervention_time.isoformat() if self.last_intervention_time else None
            },
            "recent_events": [
                {
                    "event_type": e.event_type,
                    "timestamp": e.timestamp,
                    "action_taken": e.action_taken
                }
                for e in self.events[-10:]
            ],
            "trading_windows": {
                "premarket": f"{self.config.premarket_start.strftime('%H:%M')} - {self.config.premarket_end.strftime('%H:%M')} ET (enabled: {self.config.enable_in_premarket})",
                "market": f"{self.config.market_start.strftime('%H:%M')} - {self.config.market_end.strftime('%H:%M')} ET (enabled: {self.config.enable_in_market})",
                "afterhours": f"{self.config.afterhours_start.strftime('%H:%M')} - {self.config.afterhours_end.strftime('%H:%M')} ET (enabled: {self.config.enable_in_afterhours})"
            }
        }


# Singleton instance
_watchdog: Optional[TradingWindowWatchdog] = None


def get_trading_watchdog() -> TradingWindowWatchdog:
    """Get or create the trading window watchdog singleton"""
    global _watchdog
    if _watchdog is None:
        _watchdog = TradingWindowWatchdog()
    return _watchdog


async def start_watchdog():
    """Start the global watchdog"""
    watchdog = get_trading_watchdog()
    await watchdog.start()


async def stop_watchdog():
    """Stop the global watchdog"""
    watchdog = get_trading_watchdog()
    await watchdog.stop()
