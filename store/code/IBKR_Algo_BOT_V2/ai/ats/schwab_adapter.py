"""
ATS Schwab Adapter

Wires Schwab streaming data into the ATS feed.
Converts Schwab quote updates to Bar objects for pattern detection.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Callable
from .types import Bar, MarketContext
from .ats_feed import get_ats_feed

logger = logging.getLogger(__name__)

# Track state per symbol for bar construction
_bar_builders: Dict[str, dict] = {}
_registered = False


def _build_bar_from_quote(symbol: str, quote: Dict) -> Optional[Bar]:
    """
    Build a Bar from Schwab quote data.

    Schwab quotes contain:
    - lastPrice: Last trade price
    - bidPrice, askPrice: Current bid/ask
    - highPrice, lowPrice: Day high/low
    - openPrice: Day open
    - totalVolume: Cumulative volume
    - mark: Mid price
    """
    try:
        last_price = quote.get("lastPrice") or quote.get("mark") or quote.get("bidPrice")
        if not last_price:
            return None

        # Get OHLC data
        open_price = quote.get("openPrice", last_price)
        high_price = quote.get("highPrice", last_price)
        low_price = quote.get("lowPrice", last_price)
        volume = quote.get("totalVolume", 0)

        # Track volume delta for this bar
        builder = _bar_builders.get(symbol, {"last_volume": 0})
        volume_delta = max(0, volume - builder.get("last_volume", 0))
        builder["last_volume"] = volume
        _bar_builders[symbol] = builder

        # Create bar
        bar = Bar(
            timestamp=datetime.now(),
            open=float(open_price),
            high=float(high_price),
            low=float(low_price),
            close=float(last_price),
            volume=float(volume_delta) if volume_delta > 0 else float(volume),
            vwap=quote.get("vwap"),  # If available
        )

        return bar

    except Exception as e:
        logger.debug(f"Error building bar for {symbol}: {e}")
        return None


def _build_context_from_quote(symbol: str, quote: Dict) -> MarketContext:
    """Build MarketContext from Schwab quote data"""
    last_price = quote.get("lastPrice") or quote.get("mark", 0)

    return MarketContext(
        symbol=symbol,
        current_price=float(last_price),
        vwap=quote.get("vwap"),
        rel_volume=1.0,  # TODO: Calculate from volume data
        timestamp=datetime.now(),
    )


def _on_schwab_quote(symbol: str, quote: Dict):
    """
    Callback for Schwab quote updates.

    Converts to Bar and feeds to ATS.
    """
    bar = _build_bar_from_quote(symbol, quote)
    if not bar:
        return

    context = _build_context_from_quote(symbol, quote)

    # Feed to ATS
    feed = get_ats_feed()
    feed.on_bar(symbol, bar)


def wire_schwab_to_ats() -> bool:
    """
    Wire Schwab streaming to ATS feed.

    Returns:
        True if successfully wired
    """
    global _registered

    if _registered:
        logger.info("Schwab-ATS adapter already registered")
        return True

    try:
        # Import Schwab streaming
        from schwab_streaming import register_callback

        # Register our callback
        register_callback(_on_schwab_quote)
        _registered = True

        logger.info("Schwab streaming wired to ATS feed")
        return True

    except ImportError as e:
        logger.warning(f"Cannot wire Schwab to ATS: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to wire Schwab to ATS: {e}")
        return False


def unwire_schwab_from_ats() -> bool:
    """Remove Schwab-ATS wiring"""
    global _registered

    if not _registered:
        return True

    try:
        from schwab_streaming import unregister_callback
        unregister_callback(_on_schwab_quote)
        _registered = False
        logger.info("Schwab streaming unwired from ATS feed")
        return True
    except Exception as e:
        logger.error(f"Failed to unwire Schwab from ATS: {e}")
        return False


def is_wired() -> bool:
    """Check if Schwab is wired to ATS"""
    return _registered


def get_status() -> dict:
    """Get adapter status"""
    return {
        "wired": _registered,
        "tracked_symbols": len(_bar_builders),
        "symbols": list(_bar_builders.keys()),
    }


# Auto-wire on module import (can be disabled)
def auto_wire():
    """Auto-wire on startup if Schwab streaming is available"""
    try:
        from schwab_streaming import get_status
        status = get_status()
        if status.get("connected", False):
            wire_schwab_to_ats()
    except ImportError:
        pass
    except Exception:
        pass
