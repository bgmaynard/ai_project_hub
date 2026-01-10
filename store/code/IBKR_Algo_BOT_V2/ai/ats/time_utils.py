"""
ATS Time Utilities

Time-based guards and utilities for trading windows.
"""

from datetime import datetime, time
from typing import Tuple
import pytz


ET_TZ = pytz.timezone('US/Eastern')


def get_et_time() -> datetime:
    """Get current time in Eastern timezone"""
    return datetime.now(ET_TZ)


def get_et_time_only() -> time:
    """Get current time (time only) in Eastern timezone"""
    return get_et_time().time()


def is_premarket() -> bool:
    """Check if in pre-market window (4:00 AM - 9:30 AM ET)"""
    now = get_et_time_only()
    return time(4, 0) <= now < time(9, 30)


def is_market_hours() -> bool:
    """Check if in regular market hours (9:30 AM - 4:00 PM ET)"""
    now = get_et_time_only()
    return time(9, 30) <= now < time(16, 0)


def is_after_hours() -> bool:
    """Check if in after-hours (4:00 PM - 8:00 PM ET)"""
    now = get_et_time_only()
    return time(16, 0) <= now < time(20, 0)


def is_trading_hours() -> bool:
    """Check if in any trading window (pre-market, regular, after-hours)"""
    return is_premarket() or is_market_hours() or is_after_hours()


def is_post_open_guard(minutes: int = 5) -> bool:
    """
    Post-open guard: Block ATS triggers for first N minutes after open.

    Ross Cameron rule: First 5 minutes too chaotic, wait for direction.

    Args:
        minutes: Guard window in minutes (default 5)

    Returns:
        True if still in guard window (should block), False if safe
    """
    now = get_et_time_only()
    open_time = time(9, 30)
    guard_end = time(9, 30 + minutes)

    return open_time <= now < guard_end


def minutes_since_open() -> float:
    """Minutes since market open (9:30 AM ET)"""
    now = get_et_time()
    today = now.date()
    open_dt = datetime.combine(today, time(9, 30), tzinfo=ET_TZ)

    if now < open_dt:
        return 0.0

    delta = now - open_dt
    return delta.total_seconds() / 60.0


def minutes_to_close() -> float:
    """Minutes until market close (4:00 PM ET)"""
    now = get_et_time()
    today = now.date()
    close_dt = datetime.combine(today, time(16, 0), tzinfo=ET_TZ)

    if now >= close_dt:
        return 0.0

    delta = close_dt - now
    return delta.total_seconds() / 60.0


def get_trading_session() -> str:
    """Get current trading session name"""
    if is_premarket():
        return "PREMARKET"
    elif is_market_hours():
        if minutes_since_open() < 30:
            return "OPEN"
        elif minutes_to_close() < 30:
            return "CLOSE"
        else:
            return "MIDDAY"
    elif is_after_hours():
        return "AFTERHOURS"
    else:
        return "CLOSED"


def get_session_time_context() -> dict:
    """Get comprehensive time context for trading decisions"""
    now = get_et_time()
    return {
        "timestamp": now.isoformat(),
        "et_time": now.strftime("%H:%M:%S"),
        "session": get_trading_session(),
        "is_premarket": is_premarket(),
        "is_market_hours": is_market_hours(),
        "is_after_hours": is_after_hours(),
        "is_trading_hours": is_trading_hours(),
        "post_open_guard_active": is_post_open_guard(5),
        "minutes_since_open": round(minutes_since_open(), 1),
        "minutes_to_close": round(minutes_to_close(), 1),
    }


def should_allow_ats_trigger() -> Tuple[bool, str]:
    """
    Check if ATS triggers should be allowed based on time.

    Returns:
        (allowed, reason) tuple
    """
    if not is_trading_hours():
        return False, "Outside trading hours"

    if is_premarket():
        # Premarket: Allow triggers after 7:00 AM
        now = get_et_time_only()
        if now < time(7, 0):
            return False, "Too early in premarket (before 7:00 AM)"
        return True, "Premarket trading window"

    if is_market_hours():
        if is_post_open_guard(5):
            return False, "Post-open guard active (first 5 minutes)"
        if minutes_to_close() < 15:
            return False, "Too close to market close (last 15 minutes)"
        return True, "Regular market hours"

    if is_after_hours():
        # After hours: Only first 30 minutes
        now = get_et_time_only()
        if now > time(16, 30):
            return False, "After hours limited window ended"
        return True, "After hours trading window"

    return False, "Unknown session state"
