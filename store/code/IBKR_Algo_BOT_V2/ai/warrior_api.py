"""
Warrior Trading API Router
FastAPI endpoints for Warrior Trading system

Endpoints:
- Scanner (pre-market)
- Pattern detection
- Risk validation
- Daily statistics
- WebSocket alerts
"""

import asyncio
import json
import logging
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import (APIRouter, HTTPException, Query, WebSocket,
                     WebSocketDisconnect)
from pydantic import BaseModel

# Import Warrior Trading modules
try:
    from ai.warrior_database import WarriorDatabase, get_database
    from ai.warrior_pattern_detector import (SetupType, TradingSetup,
                                             WarriorPatternDetector)
    from ai.warrior_risk_manager import (ValidationResponse, ValidationResult,
                                         WarriorRiskManager)
    from ai.warrior_scanner import WarriorCandidate, WarriorScanner
    from config.config_loader import get_config

    WARRIOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Warrior Trading modules not available: {e}")
    WARRIOR_AVAILABLE = False

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/warrior", tags=["Warrior Trading"])

# Global instances (singleton pattern)
_scanner_instance: Optional["WarriorScanner"] = None
_detector_instance: Optional["WarriorPatternDetector"] = None
_risk_manager_instance: Optional["WarriorRiskManager"] = None
_database_instance: Optional["WarriorDatabase"] = None


# WebSocket connection manager
class ConnectionManager:
    """Manage WebSocket connections for real-time alerts"""

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
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")

    async def send_personal(self, message: dict, websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending to WebSocket: {e}")


manager = ConnectionManager()


# Pydantic models for request/response
class ScanRequest(BaseModel):
    """Pre-market scan request"""

    min_gap_percent: Optional[float] = None
    min_rvol: Optional[float] = None
    max_float: Optional[float] = None
    min_premarket_vol: Optional[int] = None


class PatternRequest(BaseModel):
    """Pattern detection request"""

    symbol: str
    candles_1m: List[Dict[str, Any]]  # OHLCV data
    candles_5m: List[Dict[str, Any]]  # OHLCV data
    vwap: float
    ema9_1m: float
    ema9_5m: float
    ema20_5m: float
    high_of_day: float


class ValidationRequest(BaseModel):
    """Trade validation request"""

    symbol: str
    setup_type: str
    entry_price: float
    stop_price: float
    target_price: float
    risk_dollars: Optional[float] = None


class TradeEntryRequest(BaseModel):
    """Record trade entry"""

    symbol: str
    setup_type: str
    entry_price: float
    shares: int
    stop_price: float
    target_price: float


class TradeExitRequest(BaseModel):
    """Record trade exit"""

    trade_id: str
    exit_price: float
    exit_reason: str = "MANUAL"


# Helper functions
def get_scanner() -> "WarriorScanner":
    """Get or create scanner instance"""
    global _scanner_instance

    if not WARRIOR_AVAILABLE:
        raise HTTPException(
            status_code=503, detail="Warrior Trading modules not available"
        )

    if _scanner_instance is None:
        _scanner_instance = WarriorScanner()
        logger.info("Created new WarriorScanner instance")

    return _scanner_instance


def get_detector() -> "WarriorPatternDetector":
    """Get or create detector instance"""
    global _detector_instance

    if not WARRIOR_AVAILABLE:
        raise HTTPException(
            status_code=503, detail="Warrior Trading modules not available"
        )

    if _detector_instance is None:
        _detector_instance = WarriorPatternDetector()
        logger.info("Created new WarriorPatternDetector instance")

    return _detector_instance


def get_risk_manager() -> "WarriorRiskManager":
    """Get or create risk manager instance"""
    global _risk_manager_instance

    if not WARRIOR_AVAILABLE:
        raise HTTPException(
            status_code=503, detail="Warrior Trading modules not available"
        )

    if _risk_manager_instance is None:
        _risk_manager_instance = WarriorRiskManager()
        logger.info("Created new WarriorRiskManager instance")

    return _risk_manager_instance


def get_db() -> "WarriorDatabase":
    """Get or create database instance"""
    global _database_instance

    if not WARRIOR_AVAILABLE:
        raise HTTPException(
            status_code=503, detail="Warrior Trading modules not available"
        )

    if _database_instance is None:
        _database_instance = get_database()
        logger.info("Created new WarriorDatabase instance")

    return _database_instance


# API Endpoints


@router.get("/status")
async def get_status():
    """
    Get Warrior Trading system status

    Returns:
        System availability and configuration
    """
    if not WARRIOR_AVAILABLE:
        return {"available": False, "error": "Warrior Trading modules not installed"}

    try:
        config = get_config()

        return {
            "available": True,
            "scanner_enabled": config.scanner.enabled,
            "patterns_enabled": config.patterns.enabled_patterns,
            "risk_config": {
                "daily_goal": config.risk.daily_profit_goal,
                "max_loss_per_trade": config.risk.max_loss_per_trade,
                "max_loss_per_day": config.risk.max_loss_per_day,
                "min_rr": config.risk.min_reward_to_risk,
            },
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting Warrior Trading status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan/premarket")
async def scan_premarket(request: ScanRequest = None):
    """
    Run pre-market scan for momentum candidates

    Args:
        request: Optional scan parameters

    Returns:
        List of WarriorCandidate objects
    """
    try:
        scanner = get_scanner()
        db = get_db()

        logger.info("Running pre-market scan...")
        start_time = datetime.now()

        if request:
            candidates = scanner.scan_premarket(
                min_gap_percent=request.min_gap_percent,
                min_rvol=request.min_rvol,
                max_float=request.max_float,
                min_premarket_vol=request.min_premarket_vol,
            )
        else:
            candidates = scanner.scan_premarket()

        scan_duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Scan complete: {len(candidates)} candidates found in {scan_duration:.2f}s"
        )

        # Save candidates to database
        if candidates:
            saved_count = db.save_watchlist_candidates(candidates)
            logger.info(f"Saved {saved_count} candidates to database")

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "count": len(candidates),
            "scan_time_seconds": scan_duration,
            "candidates": [c.to_dict() for c in candidates],
        }

    except Exception as e:
        logger.error(f"Error in pre-market scan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/watchlist")
async def get_watchlist():
    """
    Get current watchlist (from database or cache)

    Returns:
        Watchlist candidates from database or scanner cache
    """
    try:
        db = get_db()
        scanner = get_scanner()

        # Try database first
        db_candidates = db.get_watchlist()

        if db_candidates:
            return {
                "source": "database",
                "timestamp": db_candidates[0]["scan_time"] if db_candidates else None,
                "count": len(db_candidates),
                "watchlist": db_candidates,
            }

        # Fall back to scanner cache
        candidates = scanner.get_cached_results()
        is_valid = scanner.is_cache_valid()

        return {
            "source": "cache",
            "timestamp": (
                scanner.cache_timestamp.isoformat() if scanner.cache_timestamp else None
            ),
            "cache_valid": is_valid,
            "count": len(candidates),
            "watchlist": [c.to_dict() for c in candidates],
        }

    except Exception as e:
        logger.error(f"Error getting watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patterns/detect")
async def detect_patterns(request: PatternRequest):
    """
    Detect Warrior Trading patterns on a symbol

    Args:
        request: Symbol and market data

    Returns:
        List of detected TradingSetup objects
    """
    try:
        detector = get_detector()

        import pandas as pd

        # Convert dictionaries to DataFrames
        df_1m = pd.DataFrame(request.candles_1m)
        df_5m = pd.DataFrame(request.candles_5m)

        logger.info(f"Detecting patterns for {request.symbol}")

        setups = detector.detect_all_patterns(
            symbol=request.symbol,
            candles_1m=df_1m,
            candles_5m=df_5m,
            vwap=request.vwap,
            ema9_1m=request.ema9_1m,
            ema9_5m=request.ema9_5m,
            ema20_5m=request.ema20_5m,
            high_of_day=request.high_of_day,
        )

        logger.info(f"Found {len(setups)} patterns for {request.symbol}")

        return {
            "symbol": request.symbol,
            "timestamp": datetime.now().isoformat(),
            "count": len(setups),
            "setups": [s.to_dict() for s in setups],
        }

    except Exception as e:
        logger.error(f"Error detecting patterns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-trade")
async def validate_trade(request: ValidationRequest):
    """
    Validate a trade setup with risk management rules

    Args:
        request: Trade setup details

    Returns:
        ValidationResponse with approval/rejection
    """
    try:
        risk_mgr = get_risk_manager()

        # Create TradingSetup object from request
        risk_per_share = abs(request.entry_price - request.stop_price)
        reward_per_share = abs(request.target_price - request.entry_price)
        rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0

        setup = TradingSetup(
            setup_type=SetupType(request.setup_type),
            symbol=request.symbol,
            timeframe="5min",
            entry_price=request.entry_price,
            entry_condition="API validation",
            stop_price=request.stop_price,
            stop_reason="API validation",
            target_1r=request.entry_price + risk_per_share,
            target_2r=request.target_price,
            target_3r=request.entry_price + (risk_per_share * 3),
            risk_per_share=risk_per_share,
            reward_per_share=reward_per_share,
            risk_reward_ratio=rr_ratio,
            confidence=70.0,  # Default
            strength_factors=[],
            risk_factors=[],
            current_price=request.entry_price,
        )

        # Validate
        validation = risk_mgr.validate_trade(setup, request.risk_dollars)

        logger.info(
            f"Trade validation: {request.symbol} {request.setup_type} -> "
            f"{validation.result.value}"
        )

        return validation.to_dict()

    except Exception as e:
        logger.error(f"Error validating trade: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/status")
async def get_risk_status():
    """
    Get current risk manager status

    Returns:
        Daily statistics and risk limits
    """
    try:
        risk_mgr = get_risk_manager()

        stats = risk_mgr.get_daily_stats()

        return {
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "config": {
                "daily_goal": risk_mgr.risk_config.daily_profit_goal,
                "max_loss_per_trade": risk_mgr.risk_config.max_loss_per_trade,
                "max_loss_per_day": risk_mgr.risk_config.max_loss_per_day,
                "min_rr": risk_mgr.risk_config.min_reward_to_risk,
                "max_consecutive_losses": risk_mgr.risk_config.max_consecutive_losses,
            },
        }

    except Exception as e:
        logger.error(f"Error getting risk status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk/reset-daily")
async def reset_daily():
    """
    Reset daily statistics (use at market close)

    Returns:
        Confirmation message
    """
    try:
        risk_mgr = get_risk_manager()

        risk_mgr.reset_daily()

        logger.info("Daily risk statistics reset")

        return {
            "success": True,
            "message": "Daily statistics reset",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error resetting daily stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trades/enter")
async def record_trade_entry(request: TradeEntryRequest):
    """
    Record a trade entry

    Args:
        request: Trade entry details

    Returns:
        Trade ID
    """
    try:
        risk_mgr = get_risk_manager()
        db = get_db()

        # Record in risk manager (in-memory)
        trade_id = risk_mgr.record_trade_entry(
            symbol=request.symbol,
            setup_type=SetupType(request.setup_type),
            entry_price=request.entry_price,
            shares=request.shares,
            stop_price=request.stop_price,
            target_price=request.target_price,
        )

        # Save to database
        db.save_trade_entry(
            trade_id=trade_id,
            symbol=request.symbol,
            setup_type=request.setup_type,
            entry_time=datetime.now(),
            entry_price=request.entry_price,
            shares=request.shares,
            stop_price=request.stop_price,
            target_price=request.target_price,
        )

        logger.info(f"Trade entered and saved to database: {trade_id}")

        # Broadcast to WebSocket clients
        await manager.broadcast(
            {
                "type": "trade_entered",
                "trade_id": trade_id,
                "symbol": request.symbol,
                "entry_price": request.entry_price,
                "shares": request.shares,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return {
            "success": True,
            "trade_id": trade_id,
            "symbol": request.symbol,
            "entry_price": request.entry_price,
            "shares": request.shares,
            "stop_price": request.stop_price,
            "target_price": request.target_price,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error recording trade entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trades/exit")
async def record_trade_exit(request: TradeExitRequest):
    """
    Record a trade exit

    Args:
        request: Trade exit details

    Returns:
        Completed trade record
    """
    try:
        risk_mgr = get_risk_manager()
        db = get_db()

        # Record in risk manager (in-memory)
        completed_trade = risk_mgr.record_trade_exit(
            trade_id=request.trade_id,
            exit_price=request.exit_price,
            exit_reason=request.exit_reason,
        )

        if completed_trade is None:
            raise HTTPException(
                status_code=404, detail=f"Trade {request.trade_id} not found"
            )

        # Save to database
        db.save_trade_exit(
            trade_id=request.trade_id,
            exit_time=completed_trade.exit_time,
            exit_price=request.exit_price,
            exit_reason=request.exit_reason,
            pnl=completed_trade.pnl,
            pnl_percent=(
                completed_trade.pnl
                / (completed_trade.entry_price * completed_trade.shares)
            )
            * 100,
            r_multiple=completed_trade.r_multiple,
        )

        # Update daily stats in database
        stats = risk_mgr.get_daily_stats()
        db.save_daily_stats(date.today(), stats)

        logger.info(
            f"Trade exited and saved to database: {request.trade_id}, P&L: ${completed_trade.pnl:+.2f}"
        )

        # Broadcast to WebSocket clients
        await manager.broadcast(
            {
                "type": "trade_exited",
                "trade_id": request.trade_id,
                "symbol": completed_trade.symbol,
                "exit_price": request.exit_price,
                "pnl": completed_trade.pnl,
                "r_multiple": completed_trade.r_multiple,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return {
            "success": True,
            "trade_id": request.trade_id,
            "symbol": completed_trade.symbol,
            "entry_price": completed_trade.entry_price,
            "exit_price": completed_trade.exit_price,
            "shares": completed_trade.shares,
            "pnl": completed_trade.pnl,
            "pnl_percent": (
                completed_trade.pnl
                / (completed_trade.entry_price * completed_trade.shares)
            )
            * 100,
            "r_multiple": completed_trade.r_multiple,
            "exit_reason": request.exit_reason,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording trade exit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades/history")
async def get_trade_history(
    date_filter: Optional[str] = Query(None, description="Date filter (YYYY-MM-DD)"),
    status: Optional[str] = Query(None, description="Filter by status (OPEN, CLOSED)"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(100, description="Maximum number of trades to return"),
):
    """
    Get trade history from database

    Args:
        date_filter: Optional date filter (YYYY-MM-DD)
        status: Filter by status (OPEN, CLOSED)
        symbol: Filter by symbol
        limit: Maximum number of trades

    Returns:
        List of trades from database
    """
    try:
        db = get_db()

        # Parse date filter if provided
        start_date = None
        end_date = None
        if date_filter:
            filter_date = datetime.fromisoformat(date_filter).date()
            start_date = filter_date
            end_date = filter_date

        # Get trades from database
        trades = db.get_trades(
            status=status,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

        return {
            "success": True,
            "count": len(trades),
            "filters": {"date": date_filter, "status": status, "symbol": symbol},
            "trades": trades,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting trade history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time alerts
@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time pattern and trade alerts

    Sends:
    - Pattern detection alerts
    - Trade entry/exit notifications
    - Risk status updates
    """
    await manager.connect(websocket)

    try:
        # Send initial connection confirmation
        await manager.send_personal(
            {
                "type": "connected",
                "message": "Connected to Warrior Trading alerts",
                "timestamp": datetime.now().isoformat(),
            },
            websocket,
        )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle ping
                if message.get("type") == "ping":
                    await manager.send_personal(
                        {"type": "pong", "timestamp": datetime.now().isoformat()},
                        websocket,
                    )

                # Handle other message types as needed

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break

    finally:
        manager.disconnect(websocket)


# Helper function to broadcast pattern alerts
async def broadcast_pattern_alert(symbol: str, setup: "TradingSetup"):
    """
    Broadcast pattern detection alert to all connected clients

    Args:
        symbol: Stock ticker
        setup: Detected trading setup
    """
    await manager.broadcast(
        {
            "type": "pattern_detected",
            "symbol": symbol,
            "setup_type": setup.setup_type.value,
            "confidence": setup.confidence,
            "entry_price": setup.entry_price,
            "stop_price": setup.stop_price,
            "target_2r": setup.target_2r,
            "risk_reward_ratio": setup.risk_reward_ratio,
            "timestamp": datetime.now().isoformat(),
        }
    )


# Configuration endpoint
@router.get("/config")
async def get_configuration():
    """
    Get current Warrior Trading configuration

    Returns:
        Complete configuration
    """
    try:
        config = get_config()

        return {
            "scanner": {
                "min_gap_percent": config.scanner.min_gap_percent,
                "min_rvol": config.scanner.min_rvol,
                "max_float_millions": config.scanner.max_float_millions,
                "max_watchlist_size": config.scanner.max_watchlist_size,
            },
            "patterns": {
                "enabled_patterns": config.patterns.enabled_patterns,
                "bull_flag": config.patterns.bull_flag,
                "hod_breakout": config.patterns.hod_breakout,
                "whole_dollar_breakout": config.patterns.whole_dollar_breakout,
            },
            "risk": {
                "daily_profit_goal": config.risk.daily_profit_goal,
                "max_loss_per_trade": config.risk.max_loss_per_trade,
                "max_loss_per_day": config.risk.max_loss_per_day,
                "default_risk_per_trade": config.risk.default_risk_per_trade,
                "min_reward_to_risk": config.risk.min_reward_to_risk,
                "max_consecutive_losses": config.risk.max_consecutive_losses,
            },
        }

    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "warrior_available": WARRIOR_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
    }


# Export router and manager for use in main app
__all__ = ["router", "manager", "broadcast_pattern_alert"]
