"""
PRE-MARKET SCANNER
==================
Finds pre-market movers using multiple data sources:
1. Schwab Movers API (best for real-time during market hours)
2. Yahoo Finance (yahooquery - free, works anytime)
3. Finviz (web scraping - backup)

Run: python premarket_scanner.py

Best used: 4:00 AM - 9:30 AM ET
"""

import json
import logging
from datetime import datetime
from typing import Dict, List

import requests

logging.basicConfig(level=logging.WARNING)


def get_schwab_movers() -> Dict[str, List]:
    """Get movers from Schwab API (requires market hours)"""
    try:
        from schwab_market_data import get_all_movers

        data = get_all_movers()
        return {
            "gainers": data.get("gainers", []),
            "losers": data.get("losers", []),
            "source": "Schwab",
        }
    except Exception as e:
        return {"gainers": [], "losers": [], "source": "Schwab", "error": str(e)}


def get_yahoo_movers() -> Dict[str, List]:
    """Get top movers from Yahoo Finance via yahooquery"""
    try:
        from yahooquery import Screener

        s = Screener()

        results = {"gainers": [], "losers": [], "active": [], "source": "Yahoo"}

        # Get day gainers
        data = s.get_screeners(["day_gainers"], count=30)
        for q in data.get("day_gainers", {}).get("quotes", []):
            results["gainers"].append(
                {
                    "symbol": q.get("symbol"),
                    "description": q.get("shortName", ""),
                    "price": q.get("regularMarketPrice", 0),
                    "change": q.get("regularMarketChange", 0),
                    "change_pct": q.get("regularMarketChangePercent", 0),
                    "volume": q.get("regularMarketVolume", 0),
                    "market_cap": q.get("marketCap", 0),
                }
            )

        # Get most active
        data = s.get_screeners(["most_actives"], count=30)
        for q in data.get("most_actives", {}).get("quotes", []):
            results["active"].append(
                {
                    "symbol": q.get("symbol"),
                    "description": q.get("shortName", ""),
                    "price": q.get("regularMarketPrice", 0),
                    "change": q.get("regularMarketChange", 0),
                    "change_pct": q.get("regularMarketChangePercent", 0),
                    "volume": q.get("regularMarketVolume", 0),
                }
            )

        # Get losers
        data = s.get_screeners(["day_losers"], count=15)
        for q in data.get("day_losers", {}).get("quotes", []):
            results["losers"].append(
                {
                    "symbol": q.get("symbol"),
                    "description": q.get("shortName", ""),
                    "price": q.get("regularMarketPrice", 0),
                    "change": q.get("regularMarketChange", 0),
                    "change_pct": q.get("regularMarketChangePercent", 0),
                    "volume": q.get("regularMarketVolume", 0),
                }
            )

        return results
    except Exception as e:
        return {
            "gainers": [],
            "losers": [],
            "active": [],
            "source": "Yahoo",
            "error": str(e),
        }


def get_news_movers() -> List[Dict]:
    """Get stocks with breaking news from Benzinga"""
    try:
        # Check the news trader API for recent news
        r = requests.get("http://localhost:9100/api/news/recent", timeout=5)
        if r.status_code == 200:
            data = r.json()
            news_items = data.get("news", [])

            symbols_with_news = []
            for item in news_items:
                for sym in item.get("symbols", []):
                    if sym not in [s["symbol"] for s in symbols_with_news]:
                        symbols_with_news.append(
                            {
                                "symbol": sym,
                                "headline": item.get("headline", "")[:50],
                                "catalyst": item.get("catalyst_type", "news"),
                                "sentiment": item.get("sentiment", "neutral"),
                                "urgency": item.get("urgency", "medium"),
                            }
                        )
            return symbols_with_news
    except:
        pass

    # Fallback: check Benzinga fast news directly
    try:
        from ai.benzinga_fast_news import get_benzinga_news

        return get_benzinga_news()
    except:
        pass

    return []


def filter_for_scalping(stocks: List[Dict]) -> List[Dict]:
    """Filter stocks for scalping criteria"""
    filtered = []
    for s in stocks:
        price = s.get("price", 0)
        change = abs(s.get("change_pct", 0))
        volume = s.get("volume", 0)

        # Scalping criteria (Warrior Trading style)
        if 1.0 <= price <= 20.0:  # Price range
            if change >= 5.0:  # Min 5% move
                if volume >= 500000:  # Min volume
                    s["scalp_score"] = int(change * 2 + (volume / 1000000))
                    filtered.append(s)

    # Sort by score
    return sorted(filtered, key=lambda x: x.get("scalp_score", 0), reverse=True)


def add_to_watchlist(symbol: str) -> bool:
    """Add symbol to Morpheus watchlist"""
    try:
        r = requests.post(
            f"http://localhost:9100/api/watchlist/add/{symbol}", timeout=5
        )
        return r.status_code == 200
    except:
        return False


def print_table(title: str, stocks: List[Dict], limit: int = 15):
    """Print formatted table of stocks"""
    print(f"\n[{title}]")
    print("-" * 65)
    print(f"{'Symbol':<8} {'Name':<20} {'Price':>8} {'Change':>8} {'Volume':>12}")
    print("-" * 65)

    if not stocks:
        print("  No data available")
        return

    for s in stocks[:limit]:
        symbol = s.get("symbol", "?")[:8]
        name = s.get("description", "")[:18]
        price = s.get("price", 0)
        change = s.get("change_pct", 0)
        volume = s.get("volume", 0)
        print(f"{symbol:<8} {name:<20} ${price:>7.2f} {change:>+7.1f}% {volume:>12,}")


def main():
    print("=" * 65)
    print("              PRE-MARKET / INTRADAY SCANNER")
    print(f"              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # Try Schwab first (best during market hours)
    print("\n[Checking Schwab API...]")
    schwab = get_schwab_movers()
    schwab_gainers = schwab.get("gainers", [])

    if schwab_gainers:
        print(f"  [OK] Schwab: {len(schwab_gainers)} gainers found")
    else:
        print("  [--] Schwab: No data (market closed or error)")

    # Get Yahoo data (always works)
    print("\n[Checking Yahoo Finance...]")
    yahoo = get_yahoo_movers()
    yahoo_gainers = yahoo.get("gainers", [])
    yahoo_active = yahoo.get("active", [])

    if yahoo_gainers:
        print(
            f"  [OK] Yahoo: {len(yahoo_gainers)} gainers, {len(yahoo_active)} most active"
        )
    else:
        print(f"  [--] Yahoo: {yahoo.get('error', 'No data')}")

    # Check for news catalysts
    print("\n[Checking News Catalysts...]")
    news_symbols = get_news_movers()
    if news_symbols:
        print(f"  [OK] News: {len(news_symbols)} symbols with breaking news")
    else:
        print("  [--] News: No breaking news detected")

    # Combine and dedupe
    all_gainers = schwab_gainers + yahoo_gainers
    seen = set()
    unique_gainers = []
    for s in all_gainers:
        sym = s.get("symbol")
        if sym and sym not in seen:
            seen.add(sym)
            unique_gainers.append(s)

    # Filter for scalping
    scalp_candidates = filter_for_scalping(unique_gainers)

    # Print results
    print_table(
        "SCALP CANDIDATES (Price $1-$20, Gap 5%+, Vol 500K+)", scalp_candidates, 15
    )

    if scalp_candidates:
        print(f"\n  Score = (Change% × 2) + (Volume in millions)")

    print_table(
        "ALL GAINERS (Top 15)",
        sorted(unique_gainers, key=lambda x: x.get("change_pct", 0), reverse=True),
        15,
    )

    print_table("MOST ACTIVE", yahoo_active, 10)

    # Summary
    print("\n" + "=" * 65)
    print("QUICK COMMANDS:")
    print("-" * 65)

    if scalp_candidates:
        top_symbols = [s["symbol"] for s in scalp_candidates[:5]]
        print(f"  Top Scalp Picks: {', '.join(top_symbols)}")
        print()
        print("  Add to watchlist:")
        for sym in top_symbols[:3]:
            print(f"    curl -X POST http://localhost:9100/api/watchlist/add/{sym}")
    else:
        print("  No scalp candidates found matching criteria")

    print()
    print("  API Endpoints:")
    print("    curl http://localhost:9100/api/market/movers?direction=all")
    print("    curl http://localhost:9100/api/market/movers/scalp")
    print("=" * 65)


if __name__ == "__main__":
    main()
