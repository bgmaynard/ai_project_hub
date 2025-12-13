"""
Claude AI Watchlist Manager API Routes
======================================
API endpoints for AI-powered watchlist management.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from .claude_watchlist_manager import get_watchlist_manager, ClaudeWatchlistManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/watchlist-ai")


# ============ Pydantic Models ============

class MomentumScanRequest(BaseModel):
    min_change: float = 2.0
    min_volume_ratio: float = 1.5


class FullWorkflowRequest(BaseModel):
    min_change: float = 2.0
    max_selections: int = 10
    train_models: bool = True


class AddSymbolRequest(BaseModel):
    symbol: str
    reason: str = "Manual add"


class PerformanceUpdateRequest(BaseModel):
    symbol: str
    won: bool
    pnl: float


# ============ Endpoints ============

@router.get("/info")
async def get_watchlist_info():
    """Get watchlist manager information"""
    manager = get_watchlist_manager()
    working_list = manager.get_working_list()

    return {
        "status": "active",
        "ai_available": manager.ai_available,
        "working_list_count": len(working_list),
        "active_symbols": [e["symbol"] for e in working_list if e.get("active", True)],
        "nasdaq_universe_size": len(manager.nasdaq_universe),
        "features": [
            "After-hours momentum scanning",
            "Claude AI stock selection",
            "Automatic AI model training",
            "Performance tracking",
            "Watchlist sync"
        ]
    }


@router.get("/working-list")
async def get_working_list():
    """Get the current AI working list"""
    manager = get_watchlist_manager()
    working_list = manager.get_working_list()

    return {
        "count": len(working_list),
        "symbols": working_list
    }


@router.post("/scan-momentum")
async def scan_momentum(request: MomentumScanRequest):
    """Scan for after-hours momentum stocks"""
    manager = get_watchlist_manager()

    momentum_stocks = await manager.scan_after_hours_momentum(
        min_change=request.min_change,
        min_volume_ratio=request.min_volume_ratio
    )

    return {
        "status": "success",
        "count": len(momentum_stocks),
        "stocks": [
            {
                "symbol": s.symbol,
                "price": s.current_price,
                "change_percent": s.change_percent,
                "volume_ratio": s.volume_ratio,
                "momentum_score": s.momentum_score,
                "reason": s.reason
            }
            for s in momentum_stocks[:20]  # Return top 20
        ]
    }


@router.post("/select-stocks")
async def claude_select_stocks(max_selections: int = 10):
    """Have Claude AI analyze and select stocks from latest scan"""
    manager = get_watchlist_manager()

    # First do a scan
    momentum_stocks = await manager.scan_after_hours_momentum()

    if not momentum_stocks:
        return {"status": "no_momentum_found", "message": "No stocks with sufficient momentum"}

    # Have Claude select
    selected = await manager.claude_analyze_and_select(momentum_stocks, max_selections)

    return {
        "status": "success",
        "selected": selected,
        "count": len(selected),
        "message": f"Claude AI selected {len(selected)} stocks for the working list"
    }


@router.post("/run-workflow")
async def run_full_workflow(request: FullWorkflowRequest, background_tasks: BackgroundTasks):
    """
    Run the full momentum workflow:
    1. Scan for after-hours momentum
    2. Claude selects best stocks
    3. Add to working list
    4. Train AI models
    5. Sync to platform watchlist
    """
    manager = get_watchlist_manager()

    # Run workflow
    result = await manager.execute_full_workflow(
        min_change=request.min_change,
        max_selections=request.max_selections,
        train_models=request.train_models
    )

    return result


@router.post("/add")
async def add_to_working_list(request: AddSymbolRequest):
    """Add a symbol to the working list"""
    manager = get_watchlist_manager()

    success = manager.add_to_working_list(
        symbol=request.symbol,
        reason=request.reason,
        added_by="user"
    )

    if success:
        return {"status": "success", "symbol": request.symbol.upper(), "message": "Added to working list"}
    else:
        raise HTTPException(status_code=400, detail="Failed to add symbol")


@router.delete("/remove/{symbol}")
async def remove_from_working_list(symbol: str):
    """Remove a symbol from the working list"""
    manager = get_watchlist_manager()

    success = manager.remove_from_working_list(symbol)

    if success:
        return {"status": "success", "symbol": symbol.upper(), "message": "Removed from working list"}
    else:
        raise HTTPException(status_code=404, detail="Symbol not found in working list")


@router.post("/train")
async def train_working_list():
    """Train AI models on all symbols in the working list"""
    manager = get_watchlist_manager()

    result = await manager.train_on_working_list()

    return result


@router.get("/ranked")
async def get_ranked_list():
    """Get the working list ranked by AI predictions and performance"""
    manager = get_watchlist_manager()

    ranked = await manager.rank_and_filter_working_list()

    return {
        "count": len(ranked),
        "ranked_list": ranked
    }


@router.post("/sync")
async def sync_to_platform(watchlist_name: str = "AI_Working_List"):
    """Sync the AI working list to the platform's watchlist system"""
    manager = get_watchlist_manager()

    result = await manager.sync_to_platform_watchlist(watchlist_name)

    return result


@router.post("/update-performance")
async def update_symbol_performance(request: PerformanceUpdateRequest):
    """Update performance metrics after a trade"""
    manager = get_watchlist_manager()

    manager.update_performance(
        symbol=request.symbol,
        won=request.won,
        pnl=request.pnl
    )

    return {
        "status": "success",
        "symbol": request.symbol.upper(),
        "message": "Performance updated"
    }


@router.get("/top-picks")
async def get_top_picks(limit: int = 5):
    """Get top picks from the working list with buy signals"""
    manager = get_watchlist_manager()

    ranked = await manager.rank_and_filter_working_list()

    # Filter for BUY signals with high confidence
    top_picks = [
        r for r in ranked
        if r.get("signal") == "BUY" and r.get("confidence", 0) > 0.6
    ][:limit]

    return {
        "count": len(top_picks),
        "top_picks": top_picks
    }


logger.info("Watchlist AI API routes initialized")
