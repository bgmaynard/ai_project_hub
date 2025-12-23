"""
News & Fundamentals API Routes
==============================
REST API endpoints for news monitoring, fundamental analysis,
and trading triggers.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from .news_feed_monitor import (
    get_news_monitor, NewsFeedMonitor, NewsTrigger,
    NewsCategory, NewsImpact, NewsSentiment
)
from .fundamental_analysis import (
    get_fundamental_analyzer, FundamentalAnalyzer
)

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
            "Claude AI deep analysis"
        ]
    }


@router.get("/feed")
async def get_news_feed(
    limit: int = 20,
    category: Optional[str] = None,
    impact: Optional[str] = None,
    symbols: Optional[str] = None
):
    """Get recent news feed"""
    monitor = get_news_monitor()

    # Parse filters
    category_filter = NewsCategory(category) if category else None
    impact_filter = NewsImpact(impact) if impact else None
    symbol_list = symbols.split(",") if symbols else None

    # If cache is empty, fetch fresh news
    if not monitor.news_cache:
        fresh_news = await monitor.fetch_news(symbol_list, limit)
        for item in fresh_news:
            if not any(cached.id == item.id for cached in monitor.news_cache):
                monitor.news_cache.insert(0, item)

    news = monitor.get_recent_news(
        limit=limit,
        category=category_filter,
        impact=impact_filter,
        symbols=symbol_list
    )

    return {
        "count": len(news),
        "news": [item.to_dict() for item in news]
    }


@router.get("/fetch")
async def fetch_fresh_news(
    symbols: Optional[str] = None,
    limit: int = 20
):
    """Fetch fresh news from sources"""
    monitor = get_news_monitor()
    symbol_list = symbols.split(",") if symbols else None

    news = await monitor.fetch_news(symbol_list, limit)

    # Add to cache
    for item in news:
        if not any(cached.id == item.id for cached in monitor.news_cache):
            monitor.news_cache.insert(0, item)

    return {
        "count": len(news),
        "news": [item.to_dict() for item in news]
    }


@router.get("/feed")
async def get_news_feed(
    limit: int = 30,
    symbols: Optional[str] = None
):
    """
    Get a clean news feed with time, symbol, headline, and URL.
    Perfect for dashboard display.
    """
    monitor = get_news_monitor()
    symbol_list = symbols.split(",") if symbols else None

    # Fetch fresh news if cache is empty
    if not monitor.news_cache:
        try:
            fresh_news = await monitor.fetch_news(symbol_list, limit=limit)
            for item in fresh_news:
                if not any(cached.id == item.id for cached in monitor.news_cache):
                    monitor.news_cache.insert(0, item)
        except Exception as e:
            logger.warning(f"Could not fetch news: {e}")

    # Filter by symbols if specified
    news_items = monitor.news_cache[:limit]
    if symbol_list:
        news_items = [n for n in news_items if any(s in n.symbols for s in symbol_list)]

    # Format cleanly
    feed = []
    for item in news_items[:limit]:
        feed.append({
            "time": item.published_at[:19].replace("T", " ") if item.published_at else "",
            "symbol": item.symbols[0] if item.symbols else "",
            "symbols": item.symbols,
            "headline": item.headline,
            "url": item.url,
            "sentiment": item.sentiment,
            "impact": item.impact,
            "id": item.id
        })

    return {
        "success": True,
        "feed": feed,
        "count": len(feed)
    }


@router.get("/sentiment")
async def get_sentiment_summary(
    symbols: Optional[str] = None,
    hours: int = 24
):
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
        "impact": item.impact.value
    }


# ============ Trigger Endpoints ============

@router.get("/triggers")
async def get_all_triggers():
    """Get all news triggers"""
    monitor = get_news_monitor()

    triggers = []
    for trigger_id, trigger in monitor.triggers.items():
        triggers.append({
            "id": trigger.id,
            "name": trigger.name,
            "enabled": trigger.enabled,
            "category_filters": [c.value for c in trigger.category_filters],
            "impact_threshold": trigger.impact_threshold.value,
            "sentiment_threshold": trigger.sentiment_threshold.value if trigger.sentiment_threshold else None,
            "symbol_filters": trigger.symbol_filters,
            "keyword_filters": trigger.keyword_filters,
            "action": trigger.action,
            "action_params": trigger.action_params,
            "cooldown_minutes": trigger.cooldown_minutes,
            "last_triggered": trigger.last_triggered.isoformat() if trigger.last_triggered else None
        })

    return {"triggers": triggers}


@router.post("/triggers")
async def create_trigger(trigger_data: TriggerCreate):
    """Create a new news trigger"""
    monitor = get_news_monitor()

    # Parse enums
    category_filters = [NewsCategory(c) for c in trigger_data.category_filters]
    impact_threshold = NewsImpact(trigger_data.impact_threshold)
    sentiment_threshold = NewsSentiment(trigger_data.sentiment_threshold) if trigger_data.sentiment_threshold else None

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
        cooldown_minutes=trigger_data.cooldown_minutes
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

    return {
        "status": "updated",
        "trigger_id": trigger_id,
        "enabled": enabled
    }


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
async def start_monitoring(request: MonitorStartRequest, background_tasks: BackgroundTasks):
    """Start news monitoring"""
    monitor = get_news_monitor()

    if monitor.is_monitoring:
        return {
            "status": "already_running",
            "watched_symbols": monitor.watched_symbols
        }

    # Start monitoring in background
    background_tasks.add_task(
        monitor.start_monitoring,
        request.symbols,
        request.poll_interval
    )

    return {
        "status": "started",
        "symbols": request.symbols or "ALL",
        "poll_interval": request.poll_interval
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
        "active_triggers": len([t for t in monitor.triggers.values() if t.enabled])
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
async def get_earnings_calendar(
    symbols: Optional[str] = None,
    days_ahead: int = 14
):
    """Get upcoming earnings calendar"""
    analyzer = get_fundamental_analyzer()
    symbol_list = symbols.split(",") if symbols else None

    events = await analyzer.get_earnings_calendar(symbol_list, days_ahead)

    return {
        "count": len(events),
        "events": [event.to_dict() for event in events]
    }


@router.get("/calendar/economic")
async def get_economic_calendar(days_ahead: int = 7):
    """Get upcoming economic events"""
    analyzer = get_fundamental_analyzer()
    events = analyzer.get_economic_calendar(days_ahead)

    return {
        "count": len(events),
        "events": [event.to_dict() for event in events]
    }


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
        "combined_outlook": _calculate_combined_outlook(fundamentals, sentiment)
    }


def _calculate_combined_outlook(fundamentals: Dict, sentiment: Dict) -> Dict:
    """Calculate combined outlook from fundamentals and news sentiment"""
    fundamental_score = fundamentals.get("overall_score", 50)

    # Convert sentiment to score
    sentiment_score_map = {
        "bullish": 70,
        "neutral": 50,
        "bearish": 30
    }
    news_score = sentiment_score_map.get(sentiment.get("overall_sentiment", "neutral"), 50)

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
        "news_weight": news_score
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
            "sentiment": item.sentiment.value
        }
        for item in monitor.news_cache
        if item.triggered_actions
    ][:limit]

    return {
        "count": len(triggered_news),
        "alerts": triggered_news
    }


logger.info("News & Fundamentals API routes initialized")
