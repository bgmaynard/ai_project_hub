"""
Pre-Market Scanner API Routes
==============================
REST API endpoints for pre-market scanning and news logging.
Uses /api/scanner/premarket prefix to avoid conflicts with warrior autotrader.
"""

from fastapi import APIRouter
from typing import Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/scanner/premarket/status")
async def get_premarket_scanner_status():
    """Get pre-market scanner status"""
    try:
        from .premarket_scanner import get_premarket_scanner
        scanner = get_premarket_scanner()
        return scanner.get_status()
    except Exception as e:
        logger.error(f"Status error: {e}")
        return {"error": str(e)}


@router.post("/api/scanner/premarket/scan")
async def run_premarket_scan():
    """Run pre-market scan now (builds daily watchlist)"""
    try:
        from .premarket_scanner import get_premarket_scanner
        scanner = get_premarket_scanner()
        watchlist = await scanner.build_daily_watchlist()
        return {
            "success": True,
            "watchlist": watchlist,
            "count": len(watchlist)
        }
    except Exception as e:
        logger.error(f"Scan error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/api/scanner/premarket/watchlist")
async def get_premarket_watchlist():
    """Get current pre-market watchlist"""
    try:
        from .premarket_scanner import get_premarket_scanner
        scanner = get_premarket_scanner()
        return {
            "success": True,
            "watchlist": scanner.watchlist,
            "movers": scanner.premarket_movers,
            "continuations": scanner.continuations
        }
    except Exception as e:
        logger.error(f"Watchlist error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/api/scanner/premarket/news-monitor/start")
async def start_news_monitor():
    """Start continuous news monitoring"""
    try:
        from .premarket_scanner import get_premarket_scanner
        import asyncio
        scanner = get_premarket_scanner()

        if not scanner.is_running:
            asyncio.create_task(scanner.run_continuous_news_monitor())
            return {"success": True, "message": "News monitor started"}
        else:
            return {"success": True, "message": "Already running"}
    except Exception as e:
        logger.error(f"Start error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/api/scanner/premarket/news-monitor/stop")
async def stop_news_monitor():
    """Stop news monitoring"""
    try:
        from .premarket_scanner import get_premarket_scanner
        scanner = get_premarket_scanner()
        scanner.stop()
        return {"success": True, "message": "News monitor stopped"}
    except Exception as e:
        logger.error(f"Stop error: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# NEWS LOG ENDPOINTS
# ============================================================================

@router.get("/api/news-log")
async def get_news_log(limit: int = 50):
    """Get timestamped news log"""
    try:
        from .premarket_scanner import get_premarket_scanner
        scanner = get_premarket_scanner()
        news = scanner.get_news_log(limit)
        return {
            "success": True,
            "count": len(news),
            "news": news
        }
    except Exception as e:
        logger.error(f"News log error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/api/news-log/today")
async def get_today_news():
    """Get today's news only"""
    try:
        from .premarket_scanner import get_premarket_scanner
        scanner = get_premarket_scanner()
        news = scanner.news_logger.get_today_news()
        return {
            "success": True,
            "count": len(news),
            "news": news
        }
    except Exception as e:
        logger.error(f"Today news error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/api/news-log/formatted")
async def get_formatted_news(limit: int = 50):
    """Get formatted news log (like screenshot)"""
    try:
        from .premarket_scanner import get_premarket_scanner
        scanner = get_premarket_scanner()
        formatted = scanner.get_formatted_news(limit)
        return {
            "success": True,
            "formatted": formatted
        }
    except Exception as e:
        logger.error(f"Formatted news error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/api/news-log/symbol/{symbol}")
async def get_news_for_symbol(symbol: str):
    """Get all news for a specific symbol"""
    try:
        from .premarket_scanner import get_premarket_scanner
        scanner = get_premarket_scanner()
        news = scanner.news_logger.get_news_for_symbol(symbol)
        return {
            "success": True,
            "symbol": symbol.upper(),
            "count": len(news),
            "news": news
        }
    except Exception as e:
        logger.error(f"Symbol news error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/api/news-log/add")
async def add_news_entry(symbol: str, headline: str, source: str = "manual"):
    """Manually add a news entry"""
    try:
        from .premarket_scanner import get_premarket_scanner
        scanner = get_premarket_scanner()
        entry = scanner.news_logger.log_news(symbol, headline, source)
        return {
            "success": True,
            "entry": entry
        }
    except Exception as e:
        logger.error(f"Add news error: {e}")
        return {"success": False, "error": str(e)}
