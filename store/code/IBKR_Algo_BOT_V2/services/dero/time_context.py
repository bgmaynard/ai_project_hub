"""
DERO Time Context Engine

Converts timestamps to ET and assigns time window labels for market sessions.
This is READ-ONLY and does not affect trading execution.

Time Windows (Eastern Time):
- PREMARKET:   04:00 - 09:30
- OPEN_DRIVE:  09:30 - 10:30
- MIDDAY:      10:30 - 14:00
- POWER_HOUR:  14:00 - 16:00
- AFTER_HOURS: 16:00 - 20:00
- CLOSED:      20:00 - 04:00
"""

from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import pytz


class TimeWindow(Enum):
    """Trading session time windows"""
    PREMARKET = "PREMARKET"
    OPEN_DRIVE = "OPEN_DRIVE"
    MIDDAY = "MIDDAY"
    POWER_HOUR = "POWER_HOUR"
    AFTER_HOURS = "AFTER_HOURS"
    CLOSED = "CLOSED"


# Time window boundaries (ET)
TIME_WINDOWS = {
    TimeWindow.PREMARKET: (time(4, 0), time(9, 30)),
    TimeWindow.OPEN_DRIVE: (time(9, 30), time(10, 30)),
    TimeWindow.MIDDAY: (time(10, 30), time(14, 0)),
    TimeWindow.POWER_HOUR: (time(14, 0), time(16, 0)),
    TimeWindow.AFTER_HOURS: (time(16, 0), time(20, 0)),
    # CLOSED is everything else (20:00 - 04:00)
}


class TimeContextEngine:
    """
    Time awareness engine for DERO.

    Converts timestamps to ET and assigns session window labels.
    All operations are read-only.
    """

    def __init__(self):
        self.timezone = pytz.timezone("America/New_York")
        self._event_counts: Dict[str, Dict[TimeWindow, int]] = {}

    def to_et(self, dt: datetime) -> datetime:
        """Convert a datetime to Eastern Time"""
        if dt.tzinfo is None:
            # Assume UTC if no timezone
            dt = pytz.UTC.localize(dt)
        return dt.astimezone(self.timezone)

    def get_time_window(self, dt: datetime) -> TimeWindow:
        """Get the time window for a given datetime"""
        et_time = self.to_et(dt).time()

        for window, (start, end) in TIME_WINDOWS.items():
            if start <= et_time < end:
                return window

        return TimeWindow.CLOSED

    def get_window_label(self, dt: datetime) -> str:
        """Get human-readable window label"""
        return self.get_time_window(dt).value

    def is_market_hours(self, dt: datetime) -> bool:
        """Check if datetime is during regular market hours (9:30-16:00 ET)"""
        window = self.get_time_window(dt)
        return window in [TimeWindow.OPEN_DRIVE, TimeWindow.MIDDAY, TimeWindow.POWER_HOUR]

    def is_trading_session(self, dt: datetime) -> bool:
        """Check if datetime is during any trading session (premarket through close)"""
        window = self.get_time_window(dt)
        return window != TimeWindow.CLOSED

    def count_event(self, event_type: str, dt: datetime):
        """Count an event by time window (for aggregation)"""
        window = self.get_time_window(dt)
        if event_type not in self._event_counts:
            self._event_counts[event_type] = {w: 0 for w in TimeWindow}
        self._event_counts[event_type][window] += 1

    def get_event_counts(self, event_type: str) -> Dict[str, int]:
        """Get event counts by window for an event type"""
        if event_type not in self._event_counts:
            return {w.value: 0 for w in TimeWindow}
        return {w.value: count for w, count in self._event_counts[event_type].items()}

    def reset_counts(self):
        """Reset event counts (call at start of each day)"""
        self._event_counts = {}

    def get_session_windows(self) -> Dict[str, Dict[str, str]]:
        """Get session window definitions"""
        return {
            window.value: {
                "start": f"{start.strftime('%H:%M')} ET",
                "end": f"{end.strftime('%H:%M')} ET"
            }
            for window, (start, end) in TIME_WINDOWS.items()
        }

    def build_context(self, date: datetime, events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Build time context for a given date.

        Args:
            date: The date to build context for
            events: Optional list of events with 'timestamp' field

        Returns:
            Time context dictionary
        """
        # Reset counts for fresh aggregation
        self.reset_counts()

        # Process events if provided
        event_window_counts = {w.value: 0 for w in TimeWindow}
        if events:
            for event in events:
                ts = event.get("timestamp")
                if ts:
                    if isinstance(ts, str):
                        try:
                            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        except:
                            continue
                    window = self.get_time_window(ts)
                    event_window_counts[window.value] += 1

        # Build context
        et_date = self.to_et(date) if date.tzinfo else date

        return {
            "date": et_date.strftime("%Y-%m-%d"),
            "timezone": "America/New_York",
            "session_windows": self.get_session_windows(),
            "event_window_counts": event_window_counts,
            "total_events": sum(event_window_counts.values()),
            "market_hours_events": sum(
                event_window_counts.get(w.value, 0)
                for w in [TimeWindow.OPEN_DRIVE, TimeWindow.MIDDAY, TimeWindow.POWER_HOUR]
            ),
            "premarket_events": event_window_counts.get(TimeWindow.PREMARKET.value, 0),
        }

    def get_current_context(self) -> Dict[str, Any]:
        """Get time context for current moment"""
        now = datetime.now(self.timezone)
        return {
            "current_time_et": now.strftime("%Y-%m-%d %H:%M:%S ET"),
            "current_window": self.get_time_window(now).value,
            "is_market_hours": self.is_market_hours(now),
            "is_trading_session": self.is_trading_session(now),
        }


# Singleton instance
_time_context_engine: Optional[TimeContextEngine] = None


def get_time_context_engine() -> TimeContextEngine:
    """Get or create the singleton TimeContextEngine instance"""
    global _time_context_engine
    if _time_context_engine is None:
        _time_context_engine = TimeContextEngine()
    return _time_context_engine
