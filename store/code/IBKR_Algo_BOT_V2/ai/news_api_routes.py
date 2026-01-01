"""
News & Fundamentals API Routes
==============================
REST API endpoints for news monitoring, fundamental analysis,
and trading triggers.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from .fundamental_analysis import FundamentalAnalyzer, get_fundamental_analyzer
from .news_feed_monitor import (NewsCategory, NewsFeedMonitor, NewsImpact,
                                NewsSentiment, NewsTrigger, get_news_monitor)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/news")


# ============ Pydantic Models ============


class TriggerCreate(BaseModel):
    id: str
    name: str
    enabled: bool = True
    category_filters: List[str] = []
    impact_threshold: str = "medium"
    sentiment_threshold: Optional[str] = None
    symbol_filters: List[str] = []
    keyword_filters: List[str] = []
    action: str = "alert"
    action_params: Dict[str, Any] = {}
    cooldown_minutes: int = 5


class MonitorStartRequest(BaseModel):
    symbols: List[str] = []
    poll_interval: int = 30


class FundamentalsRequest(BaseModel):
    symbol: str
    force_refresh: bool = False


# ============ News Endpoints ============


@router.get("/info")
async def get_news_info():
    """Get news system information"""
    monitor = get_news_monitor()
    return {
        "status": "active",
        "is_monitoring": monitor.is_monitoring,
        "cached_news_count": len(monitor.news_cache),
        "active_triggers": len([t for t in monitor.triggers.values() if t.enabled]),
        "total_triggers": len(monitor.triggers),
        "watched_symbols": monitor.watched_symbols,
        "poll_interval": monitor.poll_interval,
        "features": [
            "Breaking news detection",
            "AI sentiment analysis",
            "Custom triggers and alerts",
            "Multi-source aggregation",
            "Claude AI deep analysis",
        ],
    }


# Removed duplicate /feed route - use the one below with small_cap_only filter


@router.get("/fetch")
async def fetch_fresh_news(symbols: Optional[str] = None, limit: int = 20):
    """Fetch fresh news from sources"""
    monitor = get_news_monitor()
    symbol_list = symbols.split(",") if symbols else None

    news = await monitor.fetch_news(symbol_list, limit)

    # Add to cache
    for item in news:
        if not any(cached.id == item.id for cached in monitor.news_cache):
            monitor.news_cache.insert(0, item)

    return {"count": len(news), "news": [item.to_dict() for item in news]}


@router.get("/feed")
async def get_news_feed(
    limit: int = 30,
    symbols: Optional[str] = None,
    small_cap_only: bool = True,
    max_price: float = 20.0,
    actionable_only: bool = True,
):
    """
    Get a clean news feed with time, symbol, headline, and URL.
    Perfect for dashboard display.

    small_cap_only: Filter to only small cap / penny stocks (default True - excludes AAPL, MSFT, etc.)
    max_price: Max price to include (default $20, filters out large caps)
    actionable_only: Filter out commentary/opinions, keep only material news (default True)

    Note: For small cap news, pass specific symbols or use SEC filings API.
    Benzinga's general feed is dominated by large caps.
    """
    monitor = get_news_monitor()
    symbol_list = symbols.split(",") if symbols else None

    # Known large cap symbols to always filter out
    LARGE_CAP_EXCLUSIONS = {
        "AAPL",
        "MSFT",
        "GOOGL",
        "GOOG",
        "AMZN",
        "META",
        "TSLA",
        "NVDA",
        "BRK.A",
        "BRK.B",
        "JPM",
        "JNJ",
        "V",
        "PG",
        "XOM",
        "HD",
        "CVX",
        "MA",
        "ABBV",
        "MRK",
        "PFE",
        "KO",
        "PEP",
        "COST",
        "WMT",
        "NKE",
        "DIS",
        "NFLX",
        "ADBE",
        "CRM",
        "TMO",
        "ABT",
        "DHR",
        "LLY",
        "UNH",
        "AVGO",
        "CSCO",
        "ACN",
        "ORCL",
        "TXN",
        "QCOM",
        "IBM",
        "AMD",
        "INTC",
        "BA",
        "CAT",
        "GE",
        "MMM",
        "HON",
        "UPS",
        "RTX",
        "LMT",
        "GS",
        "MS",
        "BAC",
        "WFC",
        "C",
        "AXP",
        "BLK",
        "SCHW",
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
    }

    # Commentary/opinion keywords to filter out (not actionable news)
    COMMENTARY_KEYWORDS = [
        "says",
        "believes",
        "thinks",
        "predicts",
        "expects",
        "sees",
        "according to",
        "analyst says",
        "jim cramer",
        "ed yardeni",
        "morning brief",
        "market wrap",
        "daily roundup",
        "top stories",
        "what to watch",
        "stocks to watch",
        "movers and shakers",
        "options activity",
        "options trading",
        "whale",
        "unusual options",
        "dark pool",
        "most active",
        "trending",
        "buzzing",
        "social sentiment",
        "reddit",
        "wallstreetbets",
        "meme stock",
        "retail traders",
        "check out what",
        "here's why",
        "here is why",
        "5 things",
        "3 things",
        "top picks",
        "best stocks",
        "worst stocks",
        "market outlook",
        "weekly preview",
        "monthly outlook",
        "kevin o'leary",
        "warren buffett",
        "elon musk",
        "cathie wood",
        "magnificent 7",
        "magnificent seven",
        "retail investor",
        "you should know",
        "need to know",
        "everything you need",
        "quick look",
        "in focus",
        "spotlight",
        "deep dive",
        "complete guide",
        "essential guide",
        "beginners guide",
    ]

    # Keywords that indicate momentum-worthy news (prioritize these)
    MOMENTUM_KEYWORDS = [
        "fda approval",
        "fda clears",
        "fda grants",
        "breakthrough",
        "earnings beat",
        "revenue beat",
        "guidance raise",
        "upgrades",
        "contract win",
        "partnership",
        "acquisition",
        "merger",
        "insider buy",
        "insider purchase",
        "buyback",
        "share repurchase",
        "patent",
        "clinical trial",
        "phase 3",
        "phase 2",
        "positive results",
        "sec filing",
        "form 4",
        "8-k",
        "s-1",
        "ipo prices",
        "halted",
        "resumes trading",
        "short squeeze",
        "gamma squeeze",
    ]

    # For small cap mode, fetch news for scalper watchlist symbols
    query_symbols = symbol_list
    if small_cap_only and not symbol_list:
        # Get scalper watchlist for targeted small cap news
        try:
            import json

            config_path = "ai/scalper_config.json"
            with open(config_path, "r") as f:
                scalper_config = json.load(f)
            watchlist = scalper_config.get("watchlist", [])
            if watchlist:
                # Use first 10 symbols for API query
                query_symbols = watchlist[:10]
                logger.info(f"Fetching news for small cap watchlist: {query_symbols}")
        except Exception as e:
            logger.warning(f"Could not load scalper watchlist: {e}")

    # Fetch fresh news if cache is empty or stale
    if not monitor.news_cache or len(monitor.news_cache) < 10:
        try:
            fresh_news = await monitor.fetch_news(query_symbols, limit=limit * 2)
            for item in fresh_news:
                if not any(cached.id == item.id for cached in monitor.news_cache):
                    monitor.news_cache.insert(0, item)
        except Exception as e:
            logger.warning(f"Could not fetch news: {e}")

    # Filter by symbols if specified
    news_items = monitor.news_cache[: limit * 5]  # Get more to filter from
    if symbol_list:
        news_items = [n for n in news_items if any(s in n.symbols for s in symbol_list)]

    # Format cleanly and apply filters
    feed = []
    for item in news_items:
        if len(feed) >= limit:
            break

        # Skip if any symbol is a known large cap
        if small_cap_only:
            symbols_in_news = set(item.symbols)
            if symbols_in_news & LARGE_CAP_EXCLUSIONS:
                continue

        # Skip commentary/opinion pieces (not actionable)
        if actionable_only:
            headline_lower = item.headline.lower()

            # Skip if headline contains commentary keywords
            if any(kw in headline_lower for kw in COMMENTARY_KEYWORDS):
                continue

            # Skip articles tagged to too many symbols (general market commentary)
            if len(item.symbols) > 3:
                continue

            # Skip articles with no symbols (general market news)
            if not item.symbols:
                continue

        # Format time based on type (string or datetime)
        if item.published_at:
            if isinstance(item.published_at, str):
                time_str = item.published_at[:19].replace("T", " ")
            else:
                time_str = item.published_at.strftime("%Y-%m-%d %H:%M:%S")
        else:
            time_str = ""

        feed.append(
            {
                "time": time_str,
                "symbol": item.symbols[0] if item.symbols else "",
                "symbols": item.symbols,
                "headline": item.headline,
                "summary": item.summary,
                "url": item.url,
                "sentiment": (
                    item.sentiment.value
                    if hasattr(item.sentiment, "value")
                    else str(item.sentiment)
                ),
                "impact": (
                    item.impact.value
                    if hasattr(item.impact, "value")
                    else str(item.impact)
                ),
                "id": item.id,
            }
        )

    return {
        "success": True,
        "feed": feed,
        "count": len(feed),
        "filters": {
            "small_cap_only": small_cap_only,
            "actionable_only": actionable_only,
        },
    }


@router.get("/sentiment")
async def get_sentiment_summary(symbols: Optional[str] = None, hours: int = 24):
    """Get aggregated sentiment summary"""
    monitor = get_news_monitor()
    symbol_list = symbols.split(",") if symbols else None

    # If cache is empty, fetch some news first
    if not monitor.news_cache:
        try:
            fresh_news = await monitor.fetch_news(symbol_list, limit=20)
            for item in fresh_news:
                if not any(cached.id == item.id for cached in monitor.news_cache):
                    monitor.news_cache.insert(0, item)
        except Exception as e:
            logger.warning(f"Could not fetch news for sentiment: {e}")

    summary = monitor.get_sentiment_summary(symbol_list, hours)
    return summary


@router.get("/analyze/{news_id}")
async def analyze_news_item(news_id: str):
    """Get AI analysis for a specific news item"""
    monitor = get_news_monitor()

    # Find news item
    item = next((n for n in monitor.news_cache if n.id == news_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="News item not found")

    # Get Claude analysis
    analysis = await monitor.analyze_with_claude(item)

    return {
        "news_id": news_id,
        "headline": item.headline,
        "ai_analysis": analysis,
        "sentiment": item.sentiment.value,
        "sentiment_score": item.sentiment_score,
        "impact": item.impact.value,
    }


# ============ Trigger Endpoints ============


@router.get("/triggers")
async def get_all_triggers():
    """Get all news triggers"""
    monitor = get_news_monitor()

    triggers = []
    for trigger_id, trigger in monitor.triggers.items():
        triggers.append(
            {
                "id": trigger.id,
                "name": trigger.name,
                "enabled": trigger.enabled,
                "category_filters": [c.value for c in trigger.category_filters],
                "impact_threshold": trigger.impact_threshold.value,
                "sentiment_threshold": (
                    trigger.sentiment_threshold.value
                    if trigger.sentiment_threshold
                    else None
                ),
                "symbol_filters": trigger.symbol_filters,
                "keyword_filters": trigger.keyword_filters,
                "action": trigger.action,
                "action_params": trigger.action_params,
                "cooldown_minutes": trigger.cooldown_minutes,
                "last_triggered": (
                    trigger.last_triggered.isoformat()
                    if trigger.last_triggered
                    else None
                ),
            }
        )

    return {"triggers": triggers}


@router.post("/triggers")
async def create_trigger(trigger_data: TriggerCreate):
    """Create a new news trigger"""
    monitor = get_news_monitor()

    # Parse enums
    category_filters = [NewsCategory(c) for c in trigger_data.category_filters]
    impact_threshold = NewsImpact(trigger_data.impact_threshold)
    sentiment_threshold = (
        NewsSentiment(trigger_data.sentiment_threshold)
        if trigger_data.sentiment_threshold
        else None
    )

    trigger = NewsTrigger(
        id=trigger_data.id,
        name=trigger_data.name,
        enabled=trigger_data.enabled,
        category_filters=category_filters,
        impact_threshold=impact_threshold,
        sentiment_threshold=sentiment_threshold,
        symbol_filters=trigger_data.symbol_filters,
        keyword_filters=trigger_data.keyword_filters,
        action=trigger_data.action,
        action_params=trigger_data.action_params,
        cooldown_minutes=trigger_data.cooldown_minutes,
    )

    monitor.add_trigger(trigger)

    return {"status": "created", "trigger_id": trigger.id}


@router.put("/triggers/{trigger_id}/enable")
async def enable_trigger(trigger_id: str, enabled: bool = True):
    """Enable or disable a trigger"""
    monitor = get_news_monitor()

    if trigger_id not in monitor.triggers:
        raise HTTPException(status_code=404, detail="Trigger not found")

    monitor.triggers[trigger_id].enabled = enabled

    return {"status": "updated", "trigger_id": trigger_id, "enabled": enabled}


@router.delete("/triggers/{trigger_id}")
async def delete_trigger(trigger_id: str):
    """Delete a trigger"""
    monitor = get_news_monitor()

    if trigger_id not in monitor.triggers:
        raise HTTPException(status_code=404, detail="Trigger not found")

    monitor.remove_trigger(trigger_id)

    return {"status": "deleted", "trigger_id": trigger_id}


# ============ Monitoring Endpoints ============


@router.post("/monitor/start")
async def start_monitoring(
    request: MonitorStartRequest, background_tasks: BackgroundTasks
):
    """Start news monitoring"""
    monitor = get_news_monitor()

    if monitor.is_monitoring:
        return {"status": "already_running", "watched_symbols": monitor.watched_symbols}

    # Start monitoring in background
    background_tasks.add_task(
        monitor.start_monitoring, request.symbols, request.poll_interval
    )

    return {
        "status": "started",
        "symbols": request.symbols or "ALL",
        "poll_interval": request.poll_interval,
    }


@router.post("/monitor/stop")
async def stop_monitoring():
    """Stop news monitoring"""
    monitor = get_news_monitor()
    monitor.stop_monitoring()

    return {"status": "stopped"}


@router.get("/monitor/status")
async def get_monitor_status():
    """Get monitoring status"""
    monitor = get_news_monitor()

    return {
        "is_monitoring": monitor.is_monitoring,
        "watched_symbols": monitor.watched_symbols,
        "poll_interval": monitor.poll_interval,
        "cache_size": len(monitor.news_cache),
        "active_triggers": len([t for t in monitor.triggers.values() if t.enabled]),
    }


# ============ Fundamentals Endpoints ============


@router.get("/fundamentals/{symbol}")
async def get_fundamentals(symbol: str, force_refresh: bool = False):
    """Get fundamental data for a symbol"""
    analyzer = get_fundamental_analyzer()
    metrics = await analyzer.get_fundamentals(symbol.upper(), force_refresh)

    return metrics.to_dict()


@router.get("/fundamentals/{symbol}/analysis")
async def get_fundamental_analysis(symbol: str):
    """Get AI analysis of fundamentals"""
    analyzer = get_fundamental_analyzer()
    analysis = await analyzer.get_ai_analysis(symbol.upper())

    return analysis


@router.get("/fundamentals/{symbol}/comparison")
async def get_sector_comparison(symbol: str):
    """Compare stock to sector averages"""
    analyzer = get_fundamental_analyzer()

    # Ensure fundamentals are loaded
    await analyzer.get_fundamentals(symbol.upper())
    comparison = analyzer.get_sector_comparison(symbol.upper())

    return comparison


@router.post("/fundamentals/batch")
async def get_batch_fundamentals(symbols: List[str]):
    """Get fundamentals for multiple symbols"""
    analyzer = get_fundamental_analyzer()
    results = {}

    for symbol in symbols:
        try:
            metrics = await analyzer.get_fundamentals(symbol.upper())
            results[symbol.upper()] = metrics.to_dict()
        except Exception as e:
            results[symbol.upper()] = {"error": str(e)}

    return results


# ============ Calendar Endpoints ============


@router.get("/calendar/earnings")
async def get_earnings_calendar(symbols: Optional[str] = None, days_ahead: int = 14):
    """Get upcoming earnings calendar"""
    analyzer = get_fundamental_analyzer()
    symbol_list = symbols.split(",") if symbols else None

    events = await analyzer.get_earnings_calendar(symbol_list, days_ahead)

    return {"count": len(events), "events": [event.to_dict() for event in events]}


@router.get("/calendar/economic")
async def get_economic_calendar(days_ahead: int = 7):
    """Get upcoming economic events"""
    analyzer = get_fundamental_analyzer()
    events = analyzer.get_economic_calendar(days_ahead)

    return {"count": len(events), "events": [event.to_dict() for event in events]}


# ============ Combined Analysis ============


@router.get("/combined/{symbol}")
async def get_combined_analysis(symbol: str):
    """Get combined news + fundamentals analysis for a symbol"""
    monitor = get_news_monitor()
    analyzer = get_fundamental_analyzer()

    # Get fundamentals
    fundamentals = await analyzer.get_ai_analysis(symbol.upper())

    # Get recent news
    news = monitor.get_recent_news(limit=10, symbols=[symbol.upper()])

    # Get sentiment
    sentiment = monitor.get_sentiment_summary([symbol.upper()], hours=24)

    return {
        "symbol": symbol.upper(),
        "fundamentals": fundamentals,
        "news_sentiment": sentiment,
        "recent_news": [item.to_dict() for item in news],
        "combined_outlook": _calculate_combined_outlook(fundamentals, sentiment),
    }


def _calculate_combined_outlook(fundamentals: Dict, sentiment: Dict) -> Dict:
    """Calculate combined outlook from fundamentals and news sentiment"""
    fundamental_score = fundamentals.get("overall_score", 50)

    # Convert sentiment to score
    sentiment_score_map = {"bullish": 70, "neutral": 50, "bearish": 30}
    news_score = sentiment_score_map.get(
        sentiment.get("overall_sentiment", "neutral"), 50
    )

    # Weight: 60% fundamentals, 40% news sentiment
    combined_score = (fundamental_score * 0.6) + (news_score * 0.4)

    if combined_score >= 70:
        outlook = "BULLISH"
        action = "Consider long position"
    elif combined_score <= 30:
        outlook = "BEARISH"
        action = "Consider short or avoid"
    else:
        outlook = "NEUTRAL"
        action = "Wait for clearer signal"

    return {
        "combined_score": round(combined_score, 1),
        "outlook": outlook,
        "suggested_action": action,
        "fundamental_weight": fundamental_score,
        "news_weight": news_score,
    }


# ============ Alerts History ============


@router.get("/alerts/history")
async def get_alert_history(limit: int = 50):
    """Get history of triggered alerts"""
    monitor = get_news_monitor()

    # Get news items that triggered actions
    triggered_news = [
        {
            "news_id": item.id,
            "headline": item.headline,
            "published_at": item.published_at.isoformat(),
            "symbols": item.symbols,
            "triggered_actions": item.triggered_actions,
            "impact": item.impact.value,
            "sentiment": item.sentiment.value,
        }
        for item in monitor.news_cache
        if item.triggered_actions
    ][:limit]

    return {"count": len(triggered_news), "alerts": triggered_news}


# ============ FinViz News Endpoints ============

import os

import aiohttp

# Import finvizfinance for news
try:
    from finvizfinance.news import News as FinvizNews
    from finvizfinance.quote import finvizfinance

    FINVIZ_AVAILABLE = True
except ImportError:
    FINVIZ_AVAILABLE = False
    logger.warning("finvizfinance not installed - FinViz news unavailable")

# FinViz Elite token from environment
FINVIZ_ELITE_TOKEN = os.getenv("FINVIZ_ELITE_TOKEN", "")


@router.get("/finviz/news")
async def get_finviz_news():
    """
    Get general news from FinViz - good for small cap coverage.
    Returns latest news headlines across all stocks.
    """
    if not FINVIZ_AVAILABLE:
        raise HTTPException(status_code=503, detail="finvizfinance not installed")

    try:
        fnews = FinvizNews()
        news_data = fnews.get_news()

        # FinViz returns dict with 'news' and 'blogs' keys, each containing DataFrames
        news_list = []

        # Handle different return formats
        if isinstance(news_data, dict):
            # Get news DataFrame
            news_df = news_data.get("news")
            if news_df is not None and hasattr(news_df, "iterrows"):
                for idx, row in news_df.head(50).iterrows():
                    news_list.append(
                        {
                            "time": str(row.get("Date", "")),
                            "headline": row.get("Title", ""),
                            "url": row.get("Link", ""),
                            "source": "FinViz",
                        }
                    )
        elif hasattr(news_data, "iterrows"):
            # Direct DataFrame
            for idx, row in news_data.head(50).iterrows():
                news_list.append(
                    {
                        "time": str(row.get("Date", "")),
                        "headline": row.get("Title", ""),
                        "url": row.get("Link", ""),
                        "source": "FinViz",
                    }
                )

        return {"success": True, "count": len(news_list), "news": news_list}
    except Exception as e:
        logger.error(f"FinViz news error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/finviz/stock/{symbol}")
async def get_finviz_stock_news(symbol: str):
    """
    Get news for a specific stock from FinViz.
    Better coverage for small caps than Benzinga.
    """
    if not FINVIZ_AVAILABLE:
        raise HTTPException(status_code=503, detail="finvizfinance not installed")

    try:
        stock = finvizfinance(symbol.upper())
        news_df = stock.ticker_news()

        news_list = []
        if news_df is not None and not news_df.empty:
            for idx, row in news_df.head(20).iterrows():
                news_list.append(
                    {
                        "time": str(row.get("Date", "")),
                        "headline": row.get("Title", ""),
                        "url": row.get("Link", ""),
                        "symbol": symbol.upper(),
                        "source": "FinViz",
                    }
                )

        return {
            "success": True,
            "symbol": symbol.upper(),
            "count": len(news_list),
            "news": news_list,
        }
    except Exception as e:
        logger.error(f"FinViz stock news error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/finviz/batch")
async def get_finviz_batch_news(symbols: str):
    """
    Get news for multiple symbols from FinViz.
    Pass comma-separated symbols: symbols=CETX,QUBT,SMR
    """
    if not FINVIZ_AVAILABLE:
        raise HTTPException(status_code=503, detail="finvizfinance not installed")

    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    all_news = []

    for sym in symbol_list[:10]:  # Limit to 10 symbols
        try:
            stock = finvizfinance(sym)
            news_df = stock.ticker_news()

            if news_df is not None and not news_df.empty:
                for idx, row in news_df.head(5).iterrows():
                    all_news.append(
                        {
                            "time": str(row.get("Date", "")),
                            "symbol": sym,
                            "headline": row.get("Title", ""),
                            "url": row.get("Link", ""),
                            "source": "FinViz",
                        }
                    )
        except Exception as e:
            logger.warning(f"FinViz news error for {sym}: {e}")
            continue

    # Sort by time (most recent first)
    all_news.sort(key=lambda x: x.get("time", ""), reverse=True)

    return {
        "success": True,
        "symbols": symbol_list,
        "count": len(all_news),
        "news": all_news,
    }


@router.get("/finviz/elite/movers")
async def get_finviz_elite_movers():
    """
    Get top penny stock movers from FinViz Elite.
    Uses Elite API for better coverage of small caps.

    Filters: Price $1-$20, Change > 5%, Volume > 500k
    """
    if not FINVIZ_ELITE_TOKEN:
        raise HTTPException(status_code=503, detail="FinViz Elite token not configured")

    try:
        # FinViz Elite export URL with penny stock filters
        # cap_small = Small Cap, sh_avgvol_o500 = Avg Vol > 500k
        # sh_price_u20 = Price under $20, ta_change_u = Up today
        filters = "cap_small,sh_avgvol_o500,sh_price_u20,ta_change_u5"
        url = f"https://elite.finviz.com/export.ashx?v=111&f={filters}&auth={FINVIZ_ELITE_TOKEN}"

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status == 200:
                    import csv
                    import io

                    text = await response.text()
                    reader = csv.DictReader(io.StringIO(text))
                    movers = []
                    for row in reader:
                        movers.append(
                            {
                                "symbol": row.get("Ticker", ""),
                                "company": row.get("Company", ""),
                                "sector": row.get("Sector", ""),
                                "industry": row.get("Industry", ""),
                                "price": row.get("Price", ""),
                                "change": row.get("Change", ""),
                                "volume": row.get("Volume", ""),
                                "market_cap": row.get("Market Cap", ""),
                                "float": row.get("Float", ""),
                                "short_float": row.get("Short Float", ""),
                            }
                        )
                    return {
                        "success": True,
                        "count": len(movers),
                        "movers": movers[:50],  # Limit to top 50
                    }
                else:
                    error = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"FinViz Elite error: {error}",
                    )

    except aiohttp.ClientError as e:
        logger.error(f"FinViz Elite network error: {e}")
        raise HTTPException(status_code=503, detail=f"Network error: {e}")
    except Exception as e:
        logger.error(f"FinViz Elite error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Import momentum scanner
try:
    from .finviz_momentum_scanner import (FinVizMomentumScanner,
                                          get_finviz_scanner)

    SCANNER_AVAILABLE = True
except ImportError:
    SCANNER_AVAILABLE = False


@router.get("/finviz/elite/top-plays")
async def get_finviz_top_plays(limit: int = 10, min_score: float = 15.0):
    """
    Get top momentum plays with news catalysts.
    Full pipeline: scan movers -> fetch news -> score -> rank.

    This is the main endpoint for finding tradeable setups!
    """
    if not SCANNER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Scanner not available")

    try:
        scanner = get_finviz_scanner()
        candidates = await scanner.get_top_plays(limit=limit * 2)

        # Filter by minimum score
        qualified = [c for c in candidates if c.combined_score >= min_score][:limit]

        return {
            "success": True,
            "count": len(qualified),
            "scan_time": scanner.last_scan.isoformat() if scanner.last_scan else None,
            "plays": [c.to_dict() for c in qualified],
        }
    except Exception as e:
        logger.error(f"Top plays error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/finviz/elite/sync-watchlist")
async def sync_finviz_to_watchlist(min_score: float = 20.0, max_add: int = 5):
    """
    Auto-add top FinViz momentum plays to scalper watchlist.
    Only adds stocks that score above min_score threshold.
    """
    if not SCANNER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Scanner not available")

    try:
        scanner = get_finviz_scanner()
        result = await scanner.sync_to_scalper_watchlist(
            min_score=min_score, max_add=max_add
        )
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Watchlist sync error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/finviz/elite/gappers")
async def get_finviz_elite_gappers():
    """
    Get gapping stocks from FinViz Elite.
    Perfect for pre-market momentum plays.

    Filters: Gap > 5%, Price $1-$20, Volume > 100k
    """
    if not FINVIZ_ELITE_TOKEN:
        raise HTTPException(status_code=503, detail="FinViz Elite token not configured")

    try:
        # Gap up filter with penny stock criteria
        filters = "sh_price_u20,sh_price_o1,ta_gap_u5,sh_curvol_o100"
        url = f"https://elite.finviz.com/export.ashx?v=111&f={filters}&auth={FINVIZ_ELITE_TOKEN}"

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status == 200:
                    import csv
                    import io

                    text = await response.text()
                    reader = csv.DictReader(io.StringIO(text))
                    gappers = []
                    for row in reader:
                        gappers.append(
                            {
                                "symbol": row.get("Ticker", ""),
                                "company": row.get("Company", ""),
                                "price": row.get("Price", ""),
                                "change": row.get("Change", ""),
                                "gap": row.get("Gap", ""),
                                "volume": row.get("Volume", ""),
                                "float": row.get("Float", ""),
                                "sector": row.get("Sector", ""),
                            }
                        )
                    return {
                        "success": True,
                        "count": len(gappers),
                        "gappers": gappers[:30],
                    }
                else:
                    error = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"FinViz Elite error: {error}",
                    )

    except Exception as e:
        logger.error(f"FinViz Elite gappers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


logger.info("News & Fundamentals API routes initialized")
