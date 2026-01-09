"""
Time Controls for Trading
=========================
Enforces trading windows and time-based restrictions.

WARRIOR TRADING WINDOW: 07:00 - 09:30 AM ET
- All NEW entries MUST occur within this window
- Exit management is allowed at any time
- No strategy-level overrides permitted
"""

import logging
from datetime import datetime, time
from typing import Tuple

try:
    import pytz

    ET_TZ = pytz.timezone("US/Eastern")
except ImportError:
    # Fallback if pytz not available
    from datetime import timedelta, timezone

    class ET_TZ:
        @staticmethod
        def localize(dt):
            return dt.replace(tzinfo=timezone(timedelta(hours=-5)))


logger = logging.getLogger(__name__)

# Trading Window Configuration
WARRIOR_WINDOW_START = time(7, 0)  # 07:00 AM ET
WARRIOR_WINDOW_END = time(16, 0)  # 04:00 PM ET (extended for full market day)

# Extended Pre-Market Window (for monitoring only)
PREMARKET_START = time(4, 0)  # 04:00 AM ET
PREMARKET_END = time(9, 30)  # 09:30 AM ET

# Regular Market Hours
MARKET_OPEN = time(9, 30)  # 09:30 AM ET
MARKET_CLOSE = time(16, 0)  # 04:00 PM ET


def get_eastern_time() -> datetime:
    """Get current time in Eastern timezone."""
    try:
        return datetime.now(ET_TZ)
    except:
        # Fallback
        return datetime.now()


def is_in_warrior_window() -> bool:
    """
    Check if current time is within the Warrior Trading window (07:00-09:30 AM ET).

    This is the ONLY window where new trade entries are allowed.
    Exit management is permitted at any time.

    Returns:
        bool: True if within trading window, False otherwise
    """
    now = get_eastern_time().time()
    in_window = WARRIOR_WINDOW_START <= now <= WARRIOR_WINDOW_END

    if not in_window:
        logger.debug(
            f"Outside trading window: {now.strftime('%H:%M:%S')} ET (window: 07:00-09:30)"
        )

    return in_window


def is_in_premarket() -> bool:
    """Check if current time is in extended pre-market hours (04:00-09:30 AM ET)."""
    now = get_eastern_time().time()
    return PREMARKET_START <= now <= PREMARKET_END


def is_market_open() -> bool:
    """Check if regular market hours (09:30 AM - 04:00 PM ET)."""
    now = get_eastern_time().time()
    return MARKET_OPEN <= now <= MARKET_CLOSE


def get_trading_window_status() -> Tuple[bool, str, str]:
    """
    Get detailed trading window status.

    Returns:
        Tuple of (is_entry_allowed, window_name, time_remaining_or_until)
    """
    now = get_eastern_time()
    current_time = now.time()

    if WARRIOR_WINDOW_START <= current_time <= WARRIOR_WINDOW_END:
        # Calculate time remaining
        end_dt = now.replace(hour=9, minute=30, second=0, microsecond=0)
        remaining = end_dt - now
        mins = int(remaining.total_seconds() / 60)
        return (True, "WARRIOR_WINDOW", f"{mins} min remaining")

    elif current_time < WARRIOR_WINDOW_START:
        # Before window
        start_dt = now.replace(hour=7, minute=0, second=0, microsecond=0)
        until = start_dt - now
        mins = int(until.total_seconds() / 60)
        return (False, "PRE_WINDOW", f"starts in {mins} min")

    elif current_time > WARRIOR_WINDOW_END and current_time <= MARKET_CLOSE:
        # After window, during market
        return (False, "POST_WINDOW", "window closed for today")

    else:
        # After market
        return (False, "AFTER_HOURS", "market closed")


def check_entry_allowed(action_type: str = "ENTRY") -> Tuple[bool, str]:
    """
    Check if a trade entry is allowed based on time controls.

    Args:
        action_type: Type of action ("ENTRY" or "EXIT")

    Returns:
        Tuple of (allowed, reason)
    """
    # Exits are ALWAYS allowed
    if action_type.upper() == "EXIT":
        return (True, "Exits allowed at any time")

    # Entries must be within warrior window
    if is_in_warrior_window():
        return (True, "Within trading window (07:00-09:30 AM ET)")

    now = get_eastern_time()
    return (
        False,
        f"TRADING_WINDOW_CLOSED: Current time {now.strftime('%H:%M:%S')} ET is outside 07:00-09:30 window",
    )


# Veto reason constant for gating engine
VETO_TRADING_WINDOW_CLOSED = "TRADING_WINDOW_CLOSED"


if __name__ == "__main__":
    # Test the module
    print(
        f"Current Eastern Time: {get_eastern_time().strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )
    print(f"In Warrior Window: {is_in_warrior_window()}")
    print(f"In Pre-Market: {is_in_premarket()}")
    print(f"Market Open: {is_market_open()}")

    status = get_trading_window_status()
    print(f"Window Status: {status}")

    entry_check = check_entry_allowed("ENTRY")
    print(f"Entry Allowed: {entry_check}")

    exit_check = check_entry_allowed("EXIT")
    print(f"Exit Allowed: {exit_check}")
