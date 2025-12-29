"""
Market Time Utility
===================
Centralized time handling for trading operations.

ALL trading logic uses Eastern Time (ET) as the reference.
Local time is only for display purposes.

Market Hours (ET):
- Pre-market: 4:00 AM - 9:30 AM
- Regular:    9:30 AM - 4:00 PM
- After-hours: 4:00 PM - 8:00 PM
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, Optional, Tuple
from enum import Enum

try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False

logger = logging.getLogger(__name__)

# Timezone definitions
ET = pytz.timezone('US/Eastern') if HAS_PYTZ else None
UTC = pytz.UTC if HAS_PYTZ else None


class MarketStatus(Enum):
    """Market status enum"""
    CLOSED = "CLOSED"
    PRE_MARKET = "PRE_MARKET"
    OPEN = "OPEN"
    AFTER_HOURS = "AFTER_HOURS"


# 2025 US Market Holidays (NYSE/NASDAQ)
MARKET_HOLIDAYS_2025 = [
    date(2025, 1, 1),    # New Year's Day
    date(2025, 1, 20),   # MLK Day
    date(2025, 2, 17),   # Presidents Day
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 26),   # Memorial Day
    date(2025, 6, 19),   # Juneteenth
    date(2025, 7, 4),    # Independence Day
    date(2025, 9, 1),    # Labor Day
    date(2025, 11, 27),  # Thanksgiving
    date(2025, 12, 25),  # Christmas
]

# Early close days (1:00 PM ET)
EARLY_CLOSE_2025 = [
    date(2025, 7, 3),    # Day before Independence Day
    date(2025, 11, 28),  # Day after Thanksgiving
    date(2025, 12, 24),  # Christmas Eve
]


def get_et_now() -> datetime:
    """Get current time in Eastern Time. This is the BOT'S reference time."""
    if HAS_PYTZ:
        return datetime.now(ET)
    else:
        # Fallback: assume local is ET (not ideal)
        return datetime.now()


def get_utc_now() -> datetime:
    """Get current UTC time."""
    if HAS_PYTZ:
        return datetime.now(UTC)
    else:
        return datetime.utcnow()


def get_local_now() -> datetime:
    """Get current local time (user's timezone)."""
    return datetime.now()


def et_to_local(et_time: datetime) -> datetime:
    """Convert ET time to local time."""
    if HAS_PYTZ and et_time.tzinfo:
        return et_time.astimezone(None)
    return et_time


def is_market_holiday(check_date: date = None) -> bool:
    """Check if a date is a market holiday."""
    if check_date is None:
        check_date = get_et_now().date()
    return check_date in MARKET_HOLIDAYS_2025


def is_early_close(check_date: date = None) -> bool:
    """Check if a date is an early close day (1:00 PM ET)."""
    if check_date is None:
        check_date = get_et_now().date()
    return check_date in EARLY_CLOSE_2025


def get_market_close_time(check_date: date = None) -> int:
    """Get market close time in HHMM format for a given date."""
    if is_early_close(check_date):
        return 1300  # 1:00 PM ET
    return 1600  # 4:00 PM ET


def get_market_status() -> Tuple[MarketStatus, str]:
    """
    Get current market status based on ET time.

    Returns:
        Tuple of (MarketStatus enum, description string)
    """
    now = get_et_now()
    weekday = now.weekday()
    hour = now.hour
    minute = now.minute
    time_val = hour * 100 + minute
    today = now.date()

    # Weekend check
    if weekday >= 5:  # Saturday = 5, Sunday = 6
        return MarketStatus.CLOSED, "Weekend"

    # Holiday check
    if is_market_holiday(today):
        return MarketStatus.CLOSED, "Holiday"

    # Get close time (regular or early)
    close_time = get_market_close_time(today)

    # Time-based status
    if time_val < 400:
        return MarketStatus.CLOSED, "Overnight (before 4 AM)"
    elif time_val < 930:
        return MarketStatus.PRE_MARKET, f"Pre-market ({now.strftime('%I:%M %p')} ET)"
    elif time_val < close_time:
        return MarketStatus.OPEN, f"Market open ({now.strftime('%I:%M %p')} ET)"
    elif time_val < 2000:
        return MarketStatus.AFTER_HOURS, f"After-hours ({now.strftime('%I:%M %p')} ET)"
    else:
        return MarketStatus.CLOSED, "Overnight (after 8 PM)"


def is_trading_hours() -> bool:
    """Check if we're in regular trading hours (9:30 AM - 4:00 PM ET)."""
    status, _ = get_market_status()
    return status == MarketStatus.OPEN


def is_premarket() -> bool:
    """Check if we're in pre-market hours (4:00 AM - 9:30 AM ET)."""
    status, _ = get_market_status()
    return status == MarketStatus.PRE_MARKET


def is_extended_hours() -> bool:
    """Check if we're in any extended hours (pre-market or after-hours)."""
    status, _ = get_market_status()
    return status in [MarketStatus.PRE_MARKET, MarketStatus.AFTER_HOURS]


def is_market_accessible() -> bool:
    """Check if market is accessible for trading (includes extended hours)."""
    status, _ = get_market_status()
    return status in [MarketStatus.OPEN, MarketStatus.PRE_MARKET, MarketStatus.AFTER_HOURS]


def get_next_market_event() -> Tuple[str, datetime]:
    """
    Get the next market event (open, close, etc.)

    Returns:
        Tuple of (event description, event time in ET)
    """
    now = get_et_now()
    today = now.date()
    time_val = now.hour * 100 + now.minute
    weekday = now.weekday()

    # Find next trading day
    next_trading_day = today
    days_ahead = 0
    while True:
        if weekday >= 5:  # Weekend
            days_ahead = 7 - weekday  # Skip to Monday
            next_trading_day = today + timedelta(days=days_ahead)
        elif is_market_holiday(next_trading_day):
            days_ahead += 1
            next_trading_day = today + timedelta(days=days_ahead)
        else:
            break
        weekday = next_trading_day.weekday()
        if days_ahead > 7:
            break

    close_time = get_market_close_time(today)

    # Determine next event
    if weekday >= 5 or is_market_holiday(today):
        # Next event is pre-market open on next trading day
        event_time = ET.localize(datetime.combine(next_trading_day, datetime.min.time().replace(hour=4)))
        return f"Pre-market opens", event_time
    elif time_val < 400:
        event_time = ET.localize(datetime.combine(today, datetime.min.time().replace(hour=4)))
        return "Pre-market opens", event_time
    elif time_val < 930:
        event_time = ET.localize(datetime.combine(today, datetime.min.time().replace(hour=9, minute=30)))
        return "Market opens", event_time
    elif time_val < close_time:
        close_hour = close_time // 100
        close_min = close_time % 100
        event_time = ET.localize(datetime.combine(today, datetime.min.time().replace(hour=close_hour, minute=close_min)))
        suffix = " (early close)" if is_early_close(today) else ""
        return f"Market closes{suffix}", event_time
    elif time_val < 2000:
        event_time = ET.localize(datetime.combine(today, datetime.min.time().replace(hour=20)))
        return "After-hours closes", event_time
    else:
        # Next pre-market
        next_day = today + timedelta(days=1)
        event_time = ET.localize(datetime.combine(next_day, datetime.min.time().replace(hour=4)))
        return "Pre-market opens", event_time


def get_time_status() -> Dict:
    """
    Get comprehensive time status for the trading system.

    This is the main function used by the API endpoint.
    """
    now_et = get_et_now()
    now_local = get_local_now()
    now_utc = get_utc_now()

    market_status, status_desc = get_market_status()
    next_event, next_event_time = get_next_market_event()

    # Calculate time until next event
    if next_event_time:
        time_until = next_event_time - now_et
        hours, remainder = divmod(int(time_until.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)
        time_until_str = f"{hours}h {minutes}m"
    else:
        time_until_str = "unknown"

    return {
        "reference_timezone": "US/Eastern (ET)",
        "et_time": now_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "et_date": now_et.strftime("%Y-%m-%d"),
        "et_time_only": now_et.strftime("%H:%M:%S"),
        "et_display": now_et.strftime("%I:%M %p ET"),
        "local_time": now_local.strftime("%Y-%m-%d %H:%M:%S"),
        "utc_time": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": now_et.strftime("%A"),
        "weekday_num": now_et.weekday(),
        "market_status": market_status.value,
        "market_status_detail": status_desc,
        "is_trading_day": not is_market_holiday() and now_et.weekday() < 5,
        "is_holiday": is_market_holiday(),
        "is_early_close": is_early_close(),
        "market_close_time": f"{get_market_close_time() // 100}:{get_market_close_time() % 100:02d} ET",
        "next_event": next_event,
        "next_event_time": next_event_time.strftime("%I:%M %p ET") if next_event_time else None,
        "time_until_next_event": time_until_str,
        "trading_windows": {
            "pre_market": "4:00 AM - 9:30 AM ET",
            "regular": "9:30 AM - 4:00 PM ET",
            "after_hours": "4:00 PM - 8:00 PM ET"
        }
    }


# Convenience exports
def now() -> datetime:
    """Alias for get_et_now() - returns current ET time."""
    return get_et_now()


if __name__ == "__main__":
    print("=" * 60)
    print("MARKET TIME UTILITY TEST")
    print("=" * 60)

    status = get_time_status()

    print(f"\nReference Timezone: {status['reference_timezone']}")
    print(f"ET Time:    {status['et_time']}")
    print(f"Local Time: {status['local_time']}")
    print(f"UTC Time:   {status['utc_time']}")
    print(f"\nWeekday: {status['weekday']}")
    print(f"Market Status: {status['market_status']} - {status['market_status_detail']}")
    print(f"Is Trading Day: {status['is_trading_day']}")
    print(f"Is Holiday: {status['is_holiday']}")
    print(f"Is Early Close: {status['is_early_close']}")
    print(f"\nNext Event: {status['next_event']} at {status['next_event_time']}")
    print(f"Time Until: {status['time_until_next_event']}")
