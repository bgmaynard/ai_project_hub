"""
News Trade Pipeline API Routes
==============================
REST endpoints for the news-to-trade pipeline, halt detection,
and news heat classification.
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/news-pipeline", tags=["News Pipeline"])


# ============================================================================
# NEWS HEAT CLASSIFICATION
# ============================================================================


class NewsHeat:
    """
    Classify news impact as HOT, WARM, or COLD.

    HOT (Trade immediately):
    - FDA approval/rejection
    - Merger/acquisition
    - Earnings beat/miss >20%
    - SEC investigation
    - CEO resignation

    WARM (Monitor closely):
    - Analyst upgrade/downgrade
    - Product launch
    - Partnership announcement
    - Guidance update

    COLD (Low impact):
    - Routine updates
    - Conference presentation
    - Industry news
    - General market commentary
    """

    HOT_KEYWORDS = [
        "fda approval",
        "fda approves",
        "fda rejects",
        "fda rejection",
        "merger",
        "acquisition",
        "buyout",
        "takeover",
        "earnings beat",
        "earnings miss",
        "guidance raise",
        "guidance cut",
        "sec investigation",
        "fraud",
        "lawsuit filed",
        "ceo resign",
        "ceo steps down",
        "cfo resign",
        "bankruptcy",
        "delisting",
        "halt",
        "halted",
        "breakthrough",
        "clinical trial success",
        "trial failure",
        "contract win",
        "major contract",
        "billion dollar",
    ]

    WARM_KEYWORDS = [
        "upgrade",
        "downgrade",
        "price target",
        "analyst",
        "rating change",
        "product launch",
        "new product",
        "partnership",
        "guidance",
        "outlook",
        "forecast update",
        "expansion",
        "new market",
        "strategic",
        "insider buying",
        "insider selling",
        "buyback",
    ]

    @classmethod
    def classify(cls, headline: str, summary: str = "") -> Dict:
        """Classify news heat level"""
        text = f"{headline} {summary}".lower()

        # Check for HOT keywords
        hot_matches = [kw for kw in cls.HOT_KEYWORDS if kw in text]
        if hot_matches:
            return {
                "heat": "HOT",
                "color": "#FF0000",
                "action": "TRADE NOW",
                "matched_keywords": hot_matches[:3],
                "priority": 1,
            }

        # Check for WARM keywords
        warm_matches = [kw for kw in cls.WARM_KEYWORDS if kw in text]
        if warm_matches:
            return {
                "heat": "WARM",
                "color": "#FFA500",
                "action": "MONITOR",
                "matched_keywords": warm_matches[:3],
                "priority": 2,
            }

        # Default to COLD
        return {
            "heat": "COLD",
            "color": "#808080",
            "action": "WATCH",
            "matched_keywords": [],
            "priority": 3,
        }

    @classmethod
    def get_sentiment_heat(cls, sentiment: str, urgency: str) -> str:
        """Convert sentiment/urgency to heat level"""
        if urgency in ["critical", "high"]:
            return "HOT"
        elif urgency == "medium":
            return "WARM"
        return "COLD"


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class PipelineConfigUpdate(BaseModel):
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    max_float: Optional[float] = None
    min_volume: Optional[int] = None
    min_change_percent: Optional[float] = None
    auto_add_to_watchlist: Optional[bool] = None
    auto_run_analysis: Optional[bool] = None
    auto_alert: Optional[bool] = None
    monitor_watchlist: Optional[bool] = None


class SymbolRequest(BaseModel):
    symbol: str


class NewsClassifyRequest(BaseModel):
    headline: str
    summary: Optional[str] = ""


# ============================================================================
# PIPELINE ENDPOINTS
# ============================================================================


@router.get("/status")
async def get_pipeline_status():
    """Get news trade pipeline status"""
    try:
        from ai.news_trade_pipeline import get_news_trade_pipeline

        pipeline = get_news_trade_pipeline()
        return pipeline.get_status()
    except Exception as e:
        logger.error(f"Pipeline status error: {e}")
        return {"error": str(e), "is_running": False}


@router.post("/start")
async def start_pipeline(watchlist: Optional[List[str]] = None):
    """Start the news trade pipeline"""
    try:
        from ai.news_trade_pipeline import start_news_trade_pipeline

        pipeline = start_news_trade_pipeline(watchlist or [])
        return {
            "success": True,
            "message": "Pipeline started",
            "status": pipeline.get_status(),
        }
    except Exception as e:
        logger.error(f"Pipeline start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_pipeline():
    """Stop the news trade pipeline"""
    try:
        from ai.news_trade_pipeline import stop_news_trade_pipeline

        stop_news_trade_pipeline()
        return {"success": True, "message": "Pipeline stopped"}
    except Exception as e:
        logger.error(f"Pipeline stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config")
async def update_pipeline_config(config: PipelineConfigUpdate):
    """Update pipeline configuration"""
    try:
        from ai.news_trade_pipeline import get_news_trade_pipeline

        pipeline = get_news_trade_pipeline()

        updates = config.dict(exclude_none=True)
        pipeline.update_config(**updates)

        return {
            "success": True,
            "message": f"Updated {len(updates)} settings",
            "config": pipeline.get_status()["config"],
        }
    except Exception as e:
        logger.error(f"Config update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/candidates")
async def get_candidates():
    """Get current trade candidates"""
    try:
        from ai.news_trade_pipeline import get_news_trade_pipeline

        pipeline = get_news_trade_pipeline()
        return {"success": True, "candidates": pipeline.get_candidates()}
    except Exception as e:
        logger.error(f"Candidates error: {e}")
        return {"success": False, "candidates": [], "error": str(e)}


@router.get("/alerts")
async def get_alerts(limit: int = 20):
    """Get recent trade alerts"""
    try:
        from ai.news_trade_pipeline import get_news_trade_pipeline

        pipeline = get_news_trade_pipeline()
        return {"success": True, "alerts": pipeline.get_alerts(limit)}
    except Exception as e:
        logger.error(f"Alerts error: {e}")
        return {"success": False, "alerts": [], "error": str(e)}


# ============================================================================
# HALT DETECTION ENDPOINTS
# ============================================================================


@router.get("/halts")
async def get_halts():
    """Get all currently halted stocks"""
    try:
        from ai.halt_detector import get_halt_detector

        detector = get_halt_detector()
        return {
            "success": True,
            "halted_count": len(detector.halted_stocks),
            "halts": detector.get_all_halts(),
        }
    except Exception as e:
        logger.error(f"Halts error: {e}")
        return {"success": False, "halts": [], "error": str(e)}


@router.get("/halts/{symbol}")
async def check_halt_status(symbol: str):
    """Check if a specific symbol is halted"""
    try:
        from ai.halt_detector import check_halt

        result = await check_halt(symbol.upper())
        return {
            "symbol": symbol.upper(),
            "is_halted": result is not None,
            "halt_info": result,
        }
    except Exception as e:
        logger.error(f"Halt check error: {e}")
        return {"symbol": symbol.upper(), "is_halted": False, "error": str(e)}


@router.get("/halts/history")
async def get_halt_history(limit: int = 20):
    """Get halt history"""
    try:
        from ai.halt_detector import get_halt_detector

        detector = get_halt_detector()
        return {"success": True, "history": detector.get_halt_history(limit)}
    except Exception as e:
        logger.error(f"Halt history error: {e}")
        return {"success": False, "history": [], "error": str(e)}


# ============================================================================
# HALT ANALYTICS ENDPOINTS
# ============================================================================


@router.get("/halt-analytics/stats")
async def get_halt_analytics_stats():
    """Get halt analytics statistics"""
    try:
        from ai.halt_analytics import get_halt_analytics

        analytics = get_halt_analytics()
        return {"success": True, **analytics.get_stats()}
    except Exception as e:
        logger.error(f"Halt analytics stats error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/halt-analytics/patterns")
async def analyze_halt_patterns():
    """Analyze halt patterns from historical data"""
    try:
        from ai.halt_analytics import get_halt_analytics

        analytics = get_halt_analytics()
        return analytics.analyze_patterns()
    except Exception as e:
        logger.error(f"Halt pattern analysis error: {e}")
        return {"status": "error", "error": str(e)}


@router.get("/halt-analytics/history")
async def get_halt_analytics_history(limit: int = 50, symbol: str = None):
    """Get detailed halt history with analytics"""
    try:
        from ai.halt_analytics import get_halt_analytics

        analytics = get_halt_analytics()
        return {"success": True, "history": analytics.get_history(limit, symbol)}
    except Exception as e:
        logger.error(f"Halt analytics history error: {e}")
        return {"success": False, "history": [], "error": str(e)}


@router.get("/halt-analytics/strategies")
async def get_halt_strategies():
    """Get available halt trading strategies"""
    try:
        from ai.halt_analytics import get_halt_analytics

        analytics = get_halt_analytics()
        return {
            "success": True,
            "strategies": {
                name: s.to_dict() for name, s in analytics.strategies.items()
            },
        }
    except Exception as e:
        logger.error(f"Halt strategies error: {e}")
        return {"success": False, "strategies": {}, "error": str(e)}


@router.post("/halts/monitor")
async def start_halt_monitoring(symbols: List[str]):
    """Start monitoring symbols for halts"""
    try:
        from ai.halt_detector import start_halt_detector

        detector = start_halt_detector([s.upper() for s in symbols])
        return {
            "success": True,
            "message": f"Monitoring {len(symbols)} symbols for halts",
            "symbols": symbols,
        }
    except Exception as e:
        logger.error(f"Halt monitor error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# NEWS HEAT CLASSIFICATION ENDPOINTS
# ============================================================================


@router.post("/classify")
async def classify_news(request: NewsClassifyRequest):
    """Classify news heat level (HOT/WARM/COLD)"""
    result = NewsHeat.classify(request.headline, request.summary)
    return {"headline": request.headline[:100], **result}


@router.get("/news/{symbol}")
async def get_symbol_news_summary(symbol: str):
    """Get news summary for a symbol with heat classification"""
    try:
        from ai.news_feed_monitor import get_news_monitor

        monitor = get_news_monitor()

        # Get recent news for this symbol
        news = monitor.get_recent_news(limit=10, symbols=[symbol.upper()])

        results = []
        for item in news:
            heat = NewsHeat.classify(item.headline, item.summary)
            results.append(
                {
                    "headline": item.headline[:100],
                    "published_at": item.published_at.isoformat(),
                    "sentiment": item.sentiment.value,
                    "impact": item.impact.value,
                    "heat": heat["heat"],
                    "heat_color": heat["color"],
                    "action": heat["action"],
                }
            )

        # Get hottest news
        hot_count = sum(1 for r in results if r["heat"] == "HOT")
        warm_count = sum(1 for r in results if r["heat"] == "WARM")

        return {
            "symbol": symbol.upper(),
            "news_count": len(results),
            "hot_count": hot_count,
            "warm_count": warm_count,
            "has_breaking": hot_count > 0,
            "news": results,
        }
    except Exception as e:
        logger.error(f"News summary error: {e}")
        return {
            "symbol": symbol.upper(),
            "news_count": 0,
            "hot_count": 0,
            "warm_count": 0,
            "has_breaking": False,
            "news": [],
            "error": str(e),
        }


@router.get("/breaking")
async def get_breaking_news():
    """Get all HOT breaking news across all symbols"""
    try:
        from ai.news_feed_monitor import get_news_monitor

        monitor = get_news_monitor()

        news = monitor.get_recent_news(limit=50)
        breaking = []

        for item in news:
            heat = NewsHeat.classify(item.headline, item.summary)
            if heat["heat"] == "HOT":
                breaking.append(
                    {
                        "symbols": item.symbols,
                        "headline": item.headline[:100],
                        "published_at": item.published_at.isoformat(),
                        "sentiment": item.sentiment.value,
                        "matched_keywords": heat["matched_keywords"],
                    }
                )

        return {"breaking_count": len(breaking), "breaking_news": breaking}
    except Exception as e:
        logger.error(f"Breaking news error: {e}")
        return {"breaking_count": 0, "breaking_news": [], "error": str(e)}


# ============================================================================
# WATCHLIST ENHANCEMENT
# ============================================================================


@router.get("/watchlist/enhanced")
async def get_enhanced_watchlist():
    """Get watchlist with halt status and news heat"""
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            # Get base watchlist
            response = await client.get(
                "http://localhost:9100/api/worklist", timeout=5.0
            )
            if response.status_code != 200:
                return {"success": False, "data": []}

            data = response.json()
            stocks = data.get("data", [])

        # Enhance with halt and news info
        from ai.halt_detector import get_halt_detector
        from ai.news_feed_monitor import get_news_monitor

        detector = get_halt_detector()
        monitor = get_news_monitor()

        enhanced = []
        for stock in stocks:
            symbol = stock.get("symbol")
            if not symbol:
                continue

            # Check halt status
            halt = detector.get_halt_status(symbol)

            # Get news heat
            news = monitor.get_recent_news(limit=5, symbols=[symbol])
            hot_news = []
            for n in news:
                heat = NewsHeat.classify(n.headline, n.summary)
                if heat["heat"] in ["HOT", "WARM"]:
                    hot_news.append(
                        {
                            "headline": n.headline[:60],
                            "heat": heat["heat"],
                            "time": n.published_at.strftime("%H:%M"),
                        }
                    )

            enhanced.append(
                {
                    **stock,
                    "is_halted": halt is not None,
                    "halt_info": halt,
                    "news_heat": hot_news[0]["heat"] if hot_news else "COLD",
                    "breaking_news": hot_news[:2],
                    "news_count": len(news),
                }
            )

        return {"success": True, "data": enhanced}

    except Exception as e:
        logger.error(f"Enhanced watchlist error: {e}")
        return {"success": False, "data": [], "error": str(e)}
