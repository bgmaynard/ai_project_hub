"""
Trade Signals & Secondary Triggers
===================================
Captures secondary technical indicators for trade correlation analysis.
"""

import logging
from datetime import datetime
from typing import Dict, Optional
import httpx

logger = logging.getLogger(__name__)


def get_time_of_day() -> str:
    """Get current market time period"""
    now = datetime.now()
    hour = now.hour
    minute = now.minute

    # Convert to ET (assume local is ET or adjust as needed)
    if hour < 4:
        return "overnight"
    elif hour < 7:
        return "pre_market_early"
    elif hour < 9 or (hour == 9 and minute < 30):
        return "pre_market_late"
    elif hour == 9 and minute >= 30:
        return "open"
    elif hour < 12:
        return "morning"
    elif hour < 15:
        return "midday"
    elif hour < 16:
        return "power_hour"
    else:
        return "after_hours"


def get_day_of_week() -> str:
    """Get current day of week"""
    return datetime.now().strftime("%A")


async def get_secondary_triggers(symbol: str, quote: Dict, signal: Dict) -> Dict:
    """
    Capture secondary technical indicators at entry time.
    Returns dict of indicator values for correlation analysis.
    """
    triggers = {
        # Timing
        "time_of_day": get_time_of_day(),
        "day_of_week": get_day_of_week(),
        "entry_timestamp": datetime.now().isoformat(),

        # From quote
        "spread_at_entry": 0.0,
        "day_change_at_entry": quote.get("change_percent", 0),
        "volume_at_entry": quote.get("volume", 0),

        # From signal
        "momentum_strength": signal.get("momentum", 0),
        "volume_spike_percent": signal.get("volume_surge", 0),

        # Position in range
        "distance_from_hod": 0.0,
        "distance_from_lod": 0.0,

        # To be populated
        "float_shares": 0.0,
        "float_rotation": 0.0,
        "relative_volume": 0.0,
        "vwap_position": "unknown",
        "rsi_at_entry": 0.0,
        "macd_signal": "unknown",
        "spy_direction": "unknown",
        "has_news_catalyst": False,
        "borrow_status": "unknown",
    }

    # Calculate spread
    bid = quote.get("bid", 0)
    ask = quote.get("ask", 0)
    price = quote.get("price", 0) or quote.get("last", 0)

    if bid and ask and price > 0:
        triggers["spread_at_entry"] = round((ask - bid) / price * 100, 3)

    # Calculate distance from HOD/LOD
    high = quote.get("high", 0)
    low = quote.get("low", 0)

    if high > 0 and price > 0:
        triggers["distance_from_hod"] = round((high - price) / high * 100, 2)
    if low > 0 and price > 0:
        triggers["distance_from_lod"] = round((price - low) / low * 100, 2)

    # Try to get float data
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://localhost:9100/api/stock/float/{symbol}",
                timeout=2.0
            )
            if response.status_code == 200:
                float_data = response.json()
                float_shares = float_data.get("float", 0)
                triggers["float_shares"] = float_shares

                # Calculate float rotation
                volume = quote.get("volume", 0)
                if float_shares > 0 and volume > 0:
                    triggers["float_rotation"] = round(volume / float_shares * 100, 2)
    except:
        pass

    # Try to get SPY direction
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://localhost:9100/api/price/SPY",
                timeout=2.0
            )
            if response.status_code == 200:
                spy_data = response.json()
                spy_change = spy_data.get("change_percent", 0)
                if spy_change > 0.3:
                    triggers["spy_direction"] = "up"
                elif spy_change < -0.3:
                    triggers["spy_direction"] = "down"
                else:
                    triggers["spy_direction"] = "flat"
    except:
        pass

    # Try to check for news
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://localhost:9100/api/stock/news-check/{symbol}",
                timeout=2.0
            )
            if response.status_code == 200:
                news_data = response.json()
                triggers["has_news_catalyst"] = news_data.get("has_news", False)
    except:
        pass

    return triggers


def analyze_trigger_correlations(trades: list) -> Dict:
    """
    Analyze correlation between secondary triggers and trade outcomes.
    Returns insights on which triggers correlate with wins/losses.
    """
    if not trades:
        return {"message": "No trades to analyze"}

    # Separate wins and losses
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]

    analysis = {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "correlations": {}
    }

    # Analyze time of day
    time_stats = {}
    for t in trades:
        tod = t.get("time_of_day", "unknown")
        if tod not in time_stats:
            time_stats[tod] = {"wins": 0, "losses": 0, "total_pnl": 0}
        if t.get("pnl", 0) > 0:
            time_stats[tod]["wins"] += 1
        else:
            time_stats[tod]["losses"] += 1
        time_stats[tod]["total_pnl"] += t.get("pnl", 0)

    # Calculate win rates by time
    for tod, stats in time_stats.items():
        total = stats["wins"] + stats["losses"]
        stats["win_rate"] = round(stats["wins"] / total * 100, 1) if total > 0 else 0
        stats["avg_pnl"] = round(stats["total_pnl"] / total, 2) if total > 0 else 0

    analysis["correlations"]["time_of_day"] = time_stats

    # Analyze momentum strength
    if wins:
        avg_win_momentum = sum(t.get("momentum_strength", 0) for t in wins) / len(wins)
    else:
        avg_win_momentum = 0

    if losses:
        avg_loss_momentum = sum(t.get("momentum_strength", 0) for t in losses) / len(losses)
    else:
        avg_loss_momentum = 0

    analysis["correlations"]["momentum"] = {
        "avg_winning_momentum": round(avg_win_momentum, 2),
        "avg_losing_momentum": round(avg_loss_momentum, 2),
        "insight": "Higher momentum = better" if avg_win_momentum > avg_loss_momentum else "Lower momentum = better"
    }

    # Analyze SPY direction
    spy_stats = {"up": {"wins": 0, "losses": 0}, "down": {"wins": 0, "losses": 0}, "flat": {"wins": 0, "losses": 0}}
    for t in trades:
        spy = t.get("spy_direction", "unknown")
        if spy in spy_stats:
            if t.get("pnl", 0) > 0:
                spy_stats[spy]["wins"] += 1
            else:
                spy_stats[spy]["losses"] += 1

    for direction, stats in spy_stats.items():
        total = stats["wins"] + stats["losses"]
        stats["win_rate"] = round(stats["wins"] / total * 100, 1) if total > 0 else 0

    analysis["correlations"]["spy_direction"] = spy_stats

    # Analyze spread at entry
    if wins:
        avg_win_spread = sum(t.get("spread_at_entry", 0) for t in wins) / len(wins)
    else:
        avg_win_spread = 0

    if losses:
        avg_loss_spread = sum(t.get("spread_at_entry", 0) for t in losses) / len(losses)
    else:
        avg_loss_spread = 0

    analysis["correlations"]["spread"] = {
        "avg_winning_spread": round(avg_win_spread, 3),
        "avg_losing_spread": round(avg_loss_spread, 3),
        "insight": "Tighter spreads = better" if avg_win_spread < avg_loss_spread else "Spread less important"
    }

    return analysis


# Quick test
if __name__ == "__main__":
    import asyncio

    async def test():
        quote = {"price": 5.0, "bid": 4.98, "ask": 5.02, "volume": 100000, "high": 5.50, "low": 4.50, "change_percent": 10.0}
        signal = {"momentum": 8.5, "volume_surge": 3.0}

        triggers = await get_secondary_triggers("TEST", quote, signal)
        print(f"Secondary Triggers: {triggers}")

    asyncio.run(test())
