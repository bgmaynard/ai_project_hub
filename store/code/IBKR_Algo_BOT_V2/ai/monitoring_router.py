"""
Monitoring API Router - Backend for Trading Dashboard UI
Provides REST endpoints for trade tracking, error logs, and performance monitoring
"""

import asyncio
import json
import logging
# Import database manager
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (APIRouter, HTTPException, Query, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent.parent))
from database.db_manager import get_db_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/monitoring", tags=["Monitoring"])


# ==================== REQUEST/RESPONSE MODELS ====================


class TradeFilters(BaseModel):
    symbol: Optional[str] = None
    status: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=500)


class ErrorFilters(BaseModel):
    severity: Optional[str] = None
    module: Optional[str] = None
    resolved: Optional[bool] = None
    limit: int = Field(default=50, ge=1, le=200)


class ResolveErrorRequest(BaseModel):
    error_id: str
    resolution_notes: str = ""


class SaveLayoutRequest(BaseModel):
    layout_name: str
    layout_config: Dict[str, Any]
    is_default: bool = False
    ui_type: str = "monitor"  # 'monitor', 'platform', 'complete_platform'


class PerformanceQuery(BaseModel):
    time_period: str = "daily"  # 'daily', 'weekly', 'monthly'
    days: int = Field(default=30, ge=1, le=365)


# ==================== TRADE TRACKING ENDPOINTS ====================


@router.get("/trades")
async def get_trades(
    symbol: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    """
    Get trade history with optional filters

    Args:
        symbol: Filter by ticker symbol
        status: Filter by status (open, closed, stopped, target)
        date_from: Start date (YYYY-MM-DD)
        date_to: End date (YYYY-MM-DD)
        limit: Maximum number of records

    Returns:
        List of trade records with P&L, R multiples, patterns
    """
    try:
        db = get_db_manager()

        filters = {}
        if symbol:
            filters["symbol"] = symbol
        if status:
            filters["status"] = status
        if date_from:
            filters["date_from"] = date_from
        if date_to:
            filters["date_to"] = date_to

        trades = db.get_trades(filters=filters, limit=limit)

        return {"success": True, "count": len(trades), "trades": trades}

    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active-trades")
async def get_active_trades():
    """
    Get all currently open trades with unrealized P&L

    Returns:
        List of active positions with current status
    """
    try:
        db = get_db_manager()
        active_trades = db.get_active_trades()

        # Calculate unrealized P&L for each position
        # (requires real-time price data - would integrate with market data feed)

        return {
            "success": True,
            "count": len(active_trades),
            "active_trades": active_trades,
            "total_exposure": sum(
                t.get("shares", 0) * t.get("entry_price", 0) for t in active_trades
            ),
        }

    except Exception as e:
        logger.error(f"Error fetching active trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades/summary")
async def get_trades_summary(
    symbol: Optional[str] = Query(None), days: int = Query(30, ge=1, le=365)
):
    """
    Get aggregated trade statistics

    Args:
        symbol: Filter by symbol (optional)
        days: Number of days to analyze

    Returns:
        Win rate, profit factor, avg R, total P&L
    """
    try:
        db = get_db_manager()

        # Get trades from last N days
        date_from = (date.today() - timedelta(days=days)).isoformat()
        filters = {"date_from": date_from, "status": "closed"}
        if symbol:
            filters["symbol"] = symbol

        trades = db.get_trades(filters=filters, limit=1000)

        if not trades:
            return {
                "success": True,
                "summary": {
                    "total_trades": 0,
                    "win_rate": 0,
                    "total_pnl": 0,
                    "profit_factor": 0,
                    "avg_r_multiple": 0,
                },
            }

        # Calculate statistics
        total = len(trades)
        winners = [t for t in trades if t.get("pnl", 0) > 0]
        losers = [t for t in trades if t.get("pnl", 0) < 0]

        win_rate = (len(winners) / total * 100) if total > 0 else 0
        total_pnl = sum(t.get("pnl", 0) for t in trades)

        gross_profit = sum(t.get("pnl", 0) for t in winners) if winners else 0
        gross_loss = abs(sum(t.get("pnl", 0) for t in losers)) if losers else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

        r_multiples = [t.get("r_multiple", 0) for t in trades if t.get("r_multiple")]
        avg_r = (sum(r_multiples) / len(r_multiples)) if r_multiples else 0

        return {
            "success": True,
            "summary": {
                "total_trades": total,
                "winning_trades": len(winners),
                "losing_trades": len(losers),
                "win_rate": round(win_rate, 2),
                "total_pnl": round(total_pnl, 2),
                "gross_profit": round(gross_profit, 2),
                "gross_loss": round(gross_loss, 2),
                "profit_factor": round(profit_factor, 2),
                "avg_win": round(gross_profit / len(winners), 2) if winners else 0,
                "avg_loss": round(gross_loss / len(losers), 2) if losers else 0,
                "avg_r_multiple": round(avg_r, 2),
                "largest_win": (
                    max([t.get("pnl", 0) for t in winners]) if winners else 0
                ),
                "largest_loss": min([t.get("pnl", 0) for t in losers]) if losers else 0,
            },
        }

    except Exception as e:
        logger.error(f"Error calculating trade summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ERROR TRACKING ENDPOINTS ====================


@router.get("/errors")
async def get_errors(
    severity: Optional[str] = Query(None),
    module: Optional[str] = Query(None),
    resolved: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Get system error logs with filters

    Args:
        severity: Filter by severity (debug, info, warning, error, critical)
        module: Filter by module (scanner, ml, risk, etc.)
        resolved: Filter by resolution status
        limit: Maximum number of records

    Returns:
        List of error logs with stack traces and context
    """
    try:
        db = get_db_manager()

        filters = {}
        if severity:
            filters["severity"] = severity
        if module:
            filters["module"] = module
        if resolved is not None:
            filters["resolved"] = resolved

        errors = db.get_errors(filters=filters, limit=limit)

        return {"success": True, "count": len(errors), "errors": errors}

    except Exception as e:
        logger.error(f"Error fetching error logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/errors/resolve")
async def resolve_error(request: ResolveErrorRequest):
    """
    Mark an error as resolved

    Args:
        request: Error ID and resolution notes

    Returns:
        Success status
    """
    try:
        db = get_db_manager()
        success = db.resolve_error(request.error_id, request.resolution_notes)

        if not success:
            raise HTTPException(status_code=404, detail="Error not found")

        return {
            "success": True,
            "message": f"Error {request.error_id} marked as resolved",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving error log: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/errors/stats")
async def get_error_stats(days: int = Query(7, ge=1, le=30)):
    """
    Get error statistics for troubleshooting

    Args:
        days: Number of days to analyze

    Returns:
        Error counts by severity, module, trends
    """
    try:
        db = get_db_manager()

        # Get all errors from last N days
        errors = db.get_errors(filters={}, limit=1000)

        # Calculate statistics
        by_severity = {}
        by_module = {}

        for error in errors:
            severity = error.get("severity", "unknown")
            module = error.get("module", "unknown")

            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_module[module] = by_module.get(module, 0) + 1

        unresolved = [e for e in errors if not e.get("resolved", False)]

        return {
            "success": True,
            "stats": {
                "total_errors": len(errors),
                "unresolved": len(unresolved),
                "by_severity": by_severity,
                "by_module": by_module,
                "most_common_module": (
                    max(by_module, key=by_module.get) if by_module else None
                ),
            },
        }

    except Exception as e:
        logger.error(f"Error calculating error stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== PERFORMANCE METRICS ENDPOINTS ====================


@router.get("/performance/daily")
async def get_daily_performance(days: int = Query(30, ge=1, le=365)):
    """
    Get daily performance metrics

    Args:
        days: Number of days to retrieve

    Returns:
        Daily aggregated statistics (win rate, profit factor, etc.)
    """
    try:
        db = get_db_manager()
        metrics = db.get_performance_metrics(days=days)

        return {"success": True, "count": len(metrics), "metrics": metrics}

    except Exception as e:
        logger.error(f"Error fetching daily performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/performance/calculate")
async def calculate_performance(target_date: Optional[str] = None):
    """
    Calculate and store performance metrics for a specific date

    Args:
        target_date: Date to calculate (YYYY-MM-DD), defaults to today

    Returns:
        Calculated metrics
    """
    try:
        db = get_db_manager()

        if target_date:
            target = datetime.strptime(target_date, "%Y-%m-%d").date()
        else:
            target = date.today()

        metrics = db.calculate_daily_metrics(target)

        if not metrics:
            return {
                "success": True,
                "message": "No trades found for this date",
                "metrics": {},
            }

        return {"success": True, "metrics": metrics}

    except Exception as e:
        logger.error(f"Error calculating performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/slippage/stats")
async def get_slippage_stats(
    symbol: Optional[str] = Query(None), days: int = Query(7, ge=1, le=30)
):
    """
    Get slippage statistics for execution quality monitoring

    Args:
        symbol: Filter by symbol (optional)
        days: Number of days to analyze

    Returns:
        Avg slippage, critical count, total cost
    """
    try:
        db = get_db_manager()
        stats = db.get_slippage_stats(symbol=symbol, days=days)

        return {"success": True, "stats": stats}

    except Exception as e:
        logger.error(f"Error fetching slippage stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== LAYOUT MANAGEMENT ENDPOINTS ====================


@router.post("/layouts/save")
async def save_layout(request: SaveLayoutRequest):
    """
    Save a custom dashboard layout

    Args:
        request: Layout name, configuration (widget positions), default flag, UI type

    Returns:
        Layout ID
    """
    try:
        db = get_db_manager()
        layout_id = db.save_layout(
            layout_name=request.layout_name,
            layout_config=request.layout_config,
            is_default=request.is_default,
            ui_type=request.ui_type,
        )

        return {
            "success": True,
            "layout_id": layout_id,
            "message": f"Layout '{request.layout_name}' saved successfully for {request.ui_type}",
        }

    except Exception as e:
        logger.error(f"Error saving layout: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/layouts")
async def get_layouts(ui_type: Optional[str] = Query("monitor")):
    """
    Get all saved dashboard layouts for a specific UI type

    Args:
        ui_type: Filter by UI type (monitor, platform, complete_platform)

    Returns:
        List of layouts with configurations
    """
    try:
        db = get_db_manager()
        layouts = db.get_layouts(ui_type=ui_type)

        return {
            "success": True,
            "count": len(layouts),
            "ui_type": ui_type,
            "layouts": layouts,
        }

    except Exception as e:
        logger.error(f"Error fetching layouts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/layouts/default")
async def get_default_layout(ui_type: Optional[str] = Query("monitor")):
    """
    Get the default dashboard layout for a specific UI type

    Args:
        ui_type: Filter by UI type (monitor, platform, complete_platform)

    Returns:
        Default layout configuration or None
    """
    try:
        db = get_db_manager()
        layout = db.get_default_layout(ui_type=ui_type)

        if not layout:
            return {
                "success": True,
                "layout": None,
                "ui_type": ui_type,
                "message": f"No default layout configured for {ui_type}",
            }

        return {"success": True, "ui_type": ui_type, "layout": layout}

    except Exception as e:
        logger.error(f"Error fetching default layout: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== REAL-TIME WEBSOCKET ENDPOINT ====================


class ConnectionManager:
    """Manage WebSocket connections for real-time updates"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"WebSocket connected. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(
            f"WebSocket disconnected. Total connections: {len(self.active_connections)}"
        )

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)


manager = ConnectionManager()


@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time monitoring updates

    Streams:
        - New trade executions
        - Error alerts
        - Performance updates
        - Active position changes
    """
    await manager.connect(websocket)

    try:
        # Send initial data
        db = get_db_manager()
        active_trades = db.get_active_trades()

        await websocket.send_json(
            {
                "type": "initial_data",
                "data": {
                    "active_trades": active_trades,
                    "timestamp": datetime.now().isoformat(),
                },
            }
        )

        # Keep connection alive and listen for client messages
        while True:
            try:
                # Wait for client messages (if any)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)

                # Echo or process client messages if needed
                await websocket.send_json(
                    {"type": "pong", "timestamp": datetime.now().isoformat()}
                )

            except asyncio.TimeoutError:
                # Send heartbeat every 30 seconds
                await websocket.send_json(
                    {"type": "heartbeat", "timestamp": datetime.now().isoformat()}
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected from WebSocket")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ==================== HEALTH CHECK ====================


@router.get("/health")
async def health_check():
    """Check monitoring API health"""
    try:
        db = get_db_manager()
        # Try a simple query
        db.get_trades(limit=1)

        return {
            "success": True,
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"success": False, "status": "unhealthy", "error": str(e)},
        )


# Helper function to broadcast trade updates (called from trade execution code)
async def broadcast_trade_update(trade_data: dict):
    """Broadcast trade update to all connected WebSocket clients"""
    await manager.broadcast(
        {
            "type": "trade_update",
            "data": trade_data,
            "timestamp": datetime.now().isoformat(),
        }
    )


# Helper function to broadcast error alerts
async def broadcast_error_alert(error_data: dict):
    """Broadcast error alert to all connected WebSocket clients"""
    await manager.broadcast(
        {
            "type": "error_alert",
            "data": error_data,
            "timestamp": datetime.now().isoformat(),
        }
    )
