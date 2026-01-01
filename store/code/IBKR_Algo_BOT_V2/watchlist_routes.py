"""
Watchlist API Routes
RESTful endpoints for managing trading watchlists
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from watchlist_manager import get_watchlist_manager

router = APIRouter(prefix="/api/watchlists", tags=["Watchlists"])


# Pydantic models
class CreateWatchlistRequest(BaseModel):
    name: str
    symbols: List[str]


class UpdateWatchlistRequest(BaseModel):
    name: Optional[str] = None
    symbols: Optional[List[str]] = None


class AddSymbolsRequest(BaseModel):
    symbols: List[str]


class RemoveSymbolsRequest(BaseModel):
    symbols: List[str]


# ============================================================================
# WATCHLIST ENDPOINTS
# ============================================================================


@router.get("/")
async def get_all_watchlists():
    """Get all watchlists"""
    try:
        manager = get_watchlist_manager()
        watchlists = manager.get_all_watchlists()
        return {"watchlists": watchlists, "count": len(watchlists)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/default")
async def get_default_watchlist():
    """Get or create default watchlist"""
    try:
        manager = get_watchlist_manager()
        watchlist = manager.get_default_watchlist()
        return watchlist
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols")
async def get_all_symbols():
    """Get all unique symbols across all watchlists"""
    try:
        manager = get_watchlist_manager()
        symbols = manager.get_all_symbols()
        return {"symbols": symbols, "count": len(symbols)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{watchlist_id}")
async def get_watchlist(watchlist_id: str):
    """Get a specific watchlist by ID"""
    try:
        manager = get_watchlist_manager()
        watchlist = manager.get_watchlist(watchlist_id)

        if not watchlist:
            raise HTTPException(
                status_code=404, detail=f"Watchlist {watchlist_id} not found"
            )

        return watchlist
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/name/{name}")
async def get_watchlist_by_name(name: str):
    """Get a watchlist by name"""
    try:
        manager = get_watchlist_manager()
        watchlist = manager.get_watchlist_by_name(name)

        if not watchlist:
            raise HTTPException(status_code=404, detail=f"Watchlist '{name}' not found")

        return watchlist
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/")
async def create_watchlist(request: CreateWatchlistRequest):
    """Create a new watchlist"""
    try:
        manager = get_watchlist_manager()

        # Check if watchlist with same name exists
        existing = manager.get_watchlist_by_name(request.name)
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Watchlist with name '{request.name}' already exists",
            )

        watchlist = manager.create_watchlist(request.name, request.symbols)
        return watchlist
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{watchlist_id}")
async def update_watchlist(watchlist_id: str, request: UpdateWatchlistRequest):
    """Update a watchlist"""
    try:
        manager = get_watchlist_manager()
        watchlist = manager.update_watchlist(
            watchlist_id=watchlist_id, name=request.name, symbols=request.symbols
        )

        if not watchlist:
            raise HTTPException(
                status_code=404, detail=f"Watchlist {watchlist_id} not found"
            )

        return watchlist
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{watchlist_id}/symbols")
async def add_symbols(watchlist_id: str, request: AddSymbolsRequest):
    """Add symbols to a watchlist"""
    try:
        manager = get_watchlist_manager()
        watchlist = manager.add_symbols(watchlist_id, request.symbols)

        if not watchlist:
            raise HTTPException(
                status_code=404, detail=f"Watchlist {watchlist_id} not found"
            )

        return watchlist
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{watchlist_id}/symbols")
async def remove_symbols(watchlist_id: str, request: RemoveSymbolsRequest):
    """Remove symbols from a watchlist"""
    try:
        manager = get_watchlist_manager()
        watchlist = manager.remove_symbols(watchlist_id, request.symbols)

        if not watchlist:
            raise HTTPException(
                status_code=404, detail=f"Watchlist {watchlist_id} not found"
            )

        return watchlist
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{watchlist_id}")
async def delete_watchlist(watchlist_id: str):
    """Delete a watchlist"""
    try:
        manager = get_watchlist_manager()
        deleted = manager.delete_watchlist(watchlist_id)

        if not deleted:
            raise HTTPException(
                status_code=404, detail=f"Watchlist {watchlist_id} not found"
            )

        return {"success": True, "message": "Watchlist deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PATTERN BASELINE ENDPOINTS
# ============================================================================


@router.get("/baselines/all")
async def get_all_baselines():
    """Get all pattern baselines across all symbols"""
    try:
        manager = get_watchlist_manager()
        baselines = manager.get_all_baselines()
        return {"count": len(baselines), "baselines": baselines}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/baselines/{symbol}")
async def get_symbol_baseline(symbol: str):
    """Get pattern baselines for a specific symbol"""
    try:
        manager = get_watchlist_manager()
        baselines = manager.get_symbol_baseline(symbol)

        if not baselines:
            return {
                "symbol": symbol.upper(),
                "message": "No baselines found. Add symbol to a watchlist to trigger backtest.",
                "baselines": [],
            }

        return {
            "symbol": symbol.upper(),
            "pattern_count": len(baselines),
            "baselines": baselines,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/baselines/{symbol}/best")
async def get_best_patterns(symbol: str, min_win_rate: float = 0.5):
    """Get best performing patterns for a symbol (50%+ win rate, profit factor ranked)"""
    try:
        manager = get_watchlist_manager()
        best = manager.get_best_patterns_for_symbol(symbol, min_win_rate)

        return {
            "symbol": symbol.upper(),
            "min_win_rate": min_win_rate,
            "top_patterns": best,
            "recommendation": (
                "Use these patterns for trading signals"
                if best
                else "Not enough data - add to watchlist first"
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/baselines/{symbol}/refresh")
async def refresh_baseline(symbol: str):
    """Manually trigger a backtest refresh for a symbol"""
    try:
        manager = get_watchlist_manager()

        # Trigger background backtest
        manager._trigger_background_backtest([symbol.upper()])

        return {
            "symbol": symbol.upper(),
            "status": "backtest_triggered",
            "message": "Backtest running in background. Check baselines in 30-60 seconds.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AI TRAINING INTEGRATION
# ============================================================================


@router.post("/{watchlist_id}/train-all")
async def train_all_symbols(watchlist_id: str, test_size: float = 0.2):
    """Train AI model on all symbols in a watchlist"""
    try:
        from ai.alpaca_ai_predictor import get_alpaca_predictor

        manager = get_watchlist_manager()
        watchlist = manager.get_watchlist(watchlist_id)

        if not watchlist:
            raise HTTPException(
                status_code=404, detail=f"Watchlist {watchlist_id} not found"
            )

        predictor = get_alpaca_predictor()
        results = []

        for symbol in watchlist["symbols"]:
            try:
                result = predictor.train(symbol=symbol, test_size=test_size)
                results.append(
                    {
                        "symbol": symbol,
                        "success": True,
                        "accuracy": result["metrics"]["accuracy"],
                        "samples": result["samples"],
                    }
                )
            except Exception as e:
                results.append({"symbol": symbol, "success": False, "error": str(e)})

        success_count = sum(1 for r in results if r["success"])

        return {
            "watchlist": watchlist["name"],
            "total_symbols": len(watchlist["symbols"]),
            "trained": success_count,
            "failed": len(results) - success_count,
            "results": results,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{watchlist_id}/predict-all")
async def predict_all_symbols(watchlist_id: str, timeframe: str = "1Day"):
    """Get AI predictions for all symbols in a watchlist"""
    try:
        from ai.alpaca_ai_predictor import get_alpaca_predictor

        manager = get_watchlist_manager()
        watchlist = manager.get_watchlist(watchlist_id)

        if not watchlist:
            raise HTTPException(
                status_code=404, detail=f"Watchlist {watchlist_id} not found"
            )

        predictor = get_alpaca_predictor()
        predictions = []

        for symbol in watchlist["symbols"]:
            try:
                prediction = predictor.predict(symbol=symbol, timeframe=timeframe)
                predictions.append(prediction)
            except Exception as e:
                predictions.append({"symbol": symbol, "error": str(e)})

        # Sort by confidence (highest first)
        predictions.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        return {
            "watchlist": watchlist["name"],
            "total_symbols": len(watchlist["symbols"]),
            "predictions": predictions,
            "top_signals": [p for p in predictions if p.get("confidence", 0) > 0.15],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
