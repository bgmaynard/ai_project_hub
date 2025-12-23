"""
EDGAR Monitor API Routes
========================
REST API endpoints for SEC EDGAR filing monitor.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/edgar", tags=["edgar"])


def get_monitor():
    """Get EDGAR monitor instance"""
    from ai.edgar_monitor import get_edgar_monitor
    return get_edgar_monitor()


@router.get("/status")
async def get_edgar_status():
    """Get EDGAR monitor status"""
    monitor = get_monitor()
    return monitor.get_status()


@router.post("/start")
async def start_edgar_monitor():
    """Start EDGAR monitor"""
    monitor = get_monitor()
    monitor.start()
    return {
        "success": True,
        "message": "EDGAR monitor started",
        "status": monitor.get_status()
    }


@router.post("/stop")
async def stop_edgar_monitor():
    """Stop EDGAR monitor"""
    monitor = get_monitor()
    monitor.stop()
    return {
        "success": True,
        "message": "EDGAR monitor stopped"
    }


@router.get("/filings")
async def get_recent_filings(limit: int = 20, priority: Optional[str] = None):
    """Get recent filings"""
    monitor = get_monitor()
    filings = monitor.get_recent_filings(limit=limit)

    if priority:
        filings = [f for f in filings if f.get('priority') == priority]

    return {
        "success": True,
        "filings": filings,
        "count": len(filings)
    }


@router.get("/filings/high-priority")
async def get_high_priority_filings(limit: int = 10):
    """Get high priority filings only"""
    monitor = get_monitor()
    filings = monitor.get_recent_filings(limit=50)
    high_priority = [f for f in filings if f.get('priority') == 'high']
    return {
        "success": True,
        "filings": high_priority[:limit],
        "count": len(high_priority[:limit])
    }


@router.get("/config")
async def get_edgar_config():
    """Get EDGAR monitor configuration"""
    monitor = get_monitor()
    return {
        "success": True,
        "config": monitor.config.to_dict()
    }


@router.post("/config")
async def update_edgar_config(data: Dict[str, Any]):
    """Update EDGAR monitor configuration"""
    monitor = get_monitor()
    monitor.update_config(**data)
    return {
        "success": True,
        "message": "Config updated",
        "config": monitor.config.to_dict()
    }
