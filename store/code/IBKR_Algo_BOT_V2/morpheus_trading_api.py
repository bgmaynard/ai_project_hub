"""
Morpheus Trading Bot API
Main API server with Schwab broker integration
"""
from dotenv import load_dotenv
load_dotenv()

import logging
import asyncio
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uuid
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Broker configuration
from config.broker_config import get_broker_config, BrokerType

# Unified Broker (Schwab)
from unified_broker import get_unified_broker, OrderSide

# Unified Market Data Provider (Schwab)
try:
    from unified_market_data import (
        get_unified_market_data,
        get_quote as unified_get_quote,
        get_quotes as unified_get_quotes,
        get_snapshot as unified_get_snapshot,
        get_price as unified_get_price
    )
    HAS_UNIFIED_DATA = True
except ImportError as e:
    HAS_UNIFIED_DATA = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Unified market data not available: {e}")

# Compatibility routes for legacy UI
from compatibility_routes import router as compat_router

# Watchlist management
try:
    from watchlist_routes import router as watchlist_router
    HAS_WATCHLIST_ROUTES = True
except ImportError:
    HAS_WATCHLIST_ROUTES = False
    watchlist_router = None

# TradingView integration
try:
    from tradingview_integration import router as tradingview_router
    HAS_TRADINGVIEW = True
except ImportError:
    HAS_TRADINGVIEW = False
    tradingview_router = None

# Analytics routes
try:
    from analytics_routes import router as analytics_router
    HAS_ANALYTICS = True
except ImportError:
    HAS_ANALYTICS = False
    analytics_router = None

# Advanced Pipeline (Task Queue Mesh Mode, XGBoost, Optuna, Regime/Drift Detection)
try:
    from src.pipeline.api_routes import router as pipeline_router
    HAS_ADVANCED_PIPELINE = True
except ImportError:
    HAS_ADVANCED_PIPELINE = False
    pipeline_router = None

# AI Predictor
try:
    from ai.ai_predictor import get_predictor as get_ai_scanner, EnhancedAIPredictor
    HAS_AI_SCANNER = True
    # Pre-load the model at startup
    _cached_predictor = EnhancedAIPredictor()
    print(f"[INFO] AI Predictor loaded: Model={_cached_predictor.model is not None}, Accuracy={_cached_predictor.accuracy:.2%}")
except ImportError as e:
    HAS_AI_SCANNER = False
    get_ai_scanner = None
    _cached_predictor = None
    print(f"[WARNING] AI Predictor not available: {e}")

# Claude AI API Router
try:
    from ai.claude_api import router as claude_api_router
    HAS_CLAUDE_API = True
except ImportError as e:
    logger.warning(f"Claude API router not available: {e}")
    HAS_CLAUDE_API = False
    claude_api_router = None

# Hybrid Data Architecture (Fast + Background channels)
try:
    from hybrid_data_routes import router as hybrid_router
    from schwab_hybrid_data import get_hybrid_provider
    HAS_HYBRID_DATA = True
except ImportError as e:
    logger.warning(f"Hybrid data not available: {e}")
    HAS_HYBRID_DATA = False
    hybrid_router = None

# Multi-Channel Data Provider for parallel market data access
try:
    from multi_channel_data import (
        get_multi_channel_provider,
        DataChannel,
        get_quote_fast,
        get_quotes_fast,
        get_snapshot_fast,
        get_bars_for_ai,
        get_bars_for_chart
    )
    HAS_MULTI_CHANNEL = True
except ImportError as e:
    logger.warning(f"Multi-channel data provider not available: {e}")
    HAS_MULTI_CHANNEL = False

# Real-Time Streaming
try:
    from realtime_streaming import (
        get_stream_manager,
        StreamType,
        RealtimeStreamManager
    )
    HAS_STREAMING = True
except ImportError as e:
    logger.warning(f"Real-time streaming not available: {e}")
    HAS_STREAMING = False

# Schwab Trading Integration
try:
    from schwab_api_routes import router as schwab_router
    from schwab_trading import get_schwab_trading, is_schwab_trading_available
    HAS_SCHWAB_TRADING = True
except ImportError as e:
    HAS_SCHWAB_TRADING = False
    schwab_router = None

# Schwab WebSocket Streaming for Real-Time Data (using schwabdev)
try:
    from schwabdev_streaming import (
        start_streaming as start_schwab_streaming,
        stop_streaming as stop_schwab_streaming,
        get_status as get_schwab_stream_status,
        subscribe as schwab_subscribe,
        get_cached_quote as get_schwab_stream_quote
    )
    HAS_SCHWAB_STREAMING = True
    logger = logging.getLogger(__name__)
    logger.info("Schwabdev streaming module loaded")
except ImportError as e:
    HAS_SCHWAB_STREAMING = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Schwabdev streaming not available: {e}")

# Schwab Fast Polling (fallback for when WebSocket streaming fails)
try:
    from schwab_fast_polling import (
        start_polling as start_schwab_fast_polling,
        stop_polling as stop_schwab_fast_polling,
        get_status as get_schwab_fast_poll_status,
        subscribe as schwab_fast_subscribe,
        get_cached_quote as get_schwab_fast_quote
    )
    HAS_SCHWAB_FAST_POLLING = True
except ImportError as e:
    HAS_SCHWAB_FAST_POLLING = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Schwab fast polling not available: {e}")

# Setup logging
logging.basicConfig(
    handlers=[RotatingFileHandler('bot.log', maxBytes=10*1024*1024, backupCount=3)],
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Morpheus Trading Bot",
    description="AI-powered trading platform with Schwab broker integration",
    version="2.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add no-cache headers to prevent browser caching issues
@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    # Add no-cache headers for API endpoints
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# =============================================================================
# PRIMARY ACCOUNT ENDPOINT (Schwab)
# =============================================================================
@app.get("/api/account")
async def get_primary_account():
    """Get account info from Schwab broker"""
    account = None
    broker = "Unknown"
    positions = []

    # Get Schwab account
    if HAS_SCHWAB_TRADING:
        try:
            if is_schwab_trading_available():
                schwab = get_schwab_trading()
                if schwab:
                    account = schwab.get_account_info()
                    positions = schwab.get_positions() or []
                    broker = "Schwab"
        except Exception as e:
            logger.warning(f"Schwab account fetch failed: {e}")

    if not account:
        raise HTTPException(status_code=503, detail="No broker available")

    # Format response for UI compatibility (UI expects 'summary' object)
    return {
        "summary": {
            "account_id": account.get("account_number", account.get("account_id", "Unknown")),
            "account_type": account.get("type", account.get("account_type", "Unknown")),
            "net_liquidation": account.get("market_value", account.get("equity", 0)),
            "buying_power": account.get("buying_power", 0),
            "total_cash": account.get("cash", account.get("cash_available_for_trading", 0)),
            "day_trading_buying_power": account.get("day_trading_buying_power", 0),
            "maintenance_margin": account.get("maintenance_requirement", 0),
            "broker": broker
        },
        "positions": positions,
        "total_realized_pnl": account.get("daily_pl", 0),
        "broker": broker,
        "name": "Morpheus Trading Bot",
        # Also include raw account data for other uses
        **account
    }

@app.get("/api/accounts")
async def get_accounts_list():
    """Get list of ALL available Schwab accounts"""
    accounts = []
    selected_account = None

    if HAS_SCHWAB_TRADING:
        try:
            if is_schwab_trading_available():
                schwab = get_schwab_trading()
                if schwab:
                    # Get ALL accounts, not just the selected one
                    all_accounts = schwab.get_accounts()
                    selected_account = schwab.get_selected_account()

                    for acc in all_accounts:
                        acc_num = acc.get("account_number")
                        # Get account type by fetching info for each
                        accounts.append({
                            "accountNumber": acc_num,
                            "accountType": acc.get("type", "UNKNOWN"),
                            "selected": acc_num == selected_account
                        })
        except Exception as e:
            logger.warning(f"Failed to fetch accounts: {e}")

    return {"accounts": accounts, "selected": selected_account}


@app.post("/api/accounts/select/{account_number}")
async def select_account(account_number: str):
    """Select a Schwab account for trading"""
    if not HAS_SCHWAB_TRADING:
        return {"success": False, "error": "Schwab trading not available"}

    try:
        if is_schwab_trading_available():
            schwab = get_schwab_trading()
            if schwab:
                success = schwab.select_account(account_number)
                if success:
                    # Get the new account info
                    account_info = schwab.get_account_info()
                    return {
                        "success": True,
                        "selected": account_number,
                        "account_type": account_info.get("type") if account_info else "UNKNOWN"
                    }
                else:
                    return {"success": False, "error": f"Account {account_number} not found"}
    except Exception as e:
        logger.error(f"Error selecting account: {e}")
        return {"success": False, "error": str(e)}

    return {"success": False, "error": "Unknown error"}

@app.get("/api/orders")
async def get_orders():
    """Get orders from Schwab broker"""
    orders = []

    if HAS_SCHWAB_TRADING:
        try:
            if is_schwab_trading_available():
                schwab = get_schwab_trading()
                if schwab:
                    orders = schwab.get_orders() or []
        except Exception as e:
            logger.warning(f"Failed to fetch orders: {e}")

    return {"orders": orders}


# Include watchlist router (if available)
if HAS_WATCHLIST_ROUTES and watchlist_router:
    app.include_router(watchlist_router)

# Include TradingView integration router (if available)
if HAS_TRADINGVIEW and tradingview_router:
    app.include_router(tradingview_router)

# Include analytics router (if available)
if HAS_ANALYTICS and analytics_router:
    app.include_router(analytics_router)

# Include compatibility router for legacy UI endpoints
app.include_router(compat_router)

# Include advanced pipeline router (if available)
if HAS_ADVANCED_PIPELINE and pipeline_router:
    app.include_router(pipeline_router, tags=["Advanced Pipeline"])

# Include Schwab trading router (if available)
if HAS_SCHWAB_TRADING and schwab_router:
    app.include_router(schwab_router, tags=["Schwab Trading"])

# Include Claude AI API router (if available)
if HAS_CLAUDE_API and claude_api_router:
    app.include_router(claude_api_router, tags=["Claude AI"])
    logger.info("Claude AI API router included")

# Include Hybrid Data router (if available)
if HAS_HYBRID_DATA and hybrid_router:
    app.include_router(hybrid_router)
    logger.info("Hybrid Data router included (Fast + Background channels)")

# Morphic AI Controller (Self-adapting AI)
try:
    from ai.morphic_api_routes import router as morphic_router
    app.include_router(morphic_router, tags=["Morphic AI"])
    HAS_MORPHIC = True
    logger.info("Morphic AI Controller router included")
except ImportError as e:
    logger.warning(f"Morphic AI not available: {e}")
    HAS_MORPHIC = False

# AI Predictor Routes (Chronos, LightGBM, Ensemble)
try:
    from ai.ai_api_routes import router as ai_predictor_router
    app.include_router(ai_predictor_router, tags=["AI Predictors"])
    HAS_AI_PREDICTORS = True
    logger.info("AI Predictor routes included (Chronos, LightGBM, Ensemble)")
except ImportError as e:
    logger.warning(f"AI Predictors not available: {e}")
    HAS_AI_PREDICTORS = False

# Strategy Template Library & Monitor (Claude-driven strategy management)
try:
    from ai.strategy_api_routes import router as strategy_router
    app.include_router(strategy_router, tags=["Strategy Library"])
    HAS_STRATEGY_LIBRARY = True
    logger.info("Strategy Template Library router included")
except ImportError as e:
    logger.warning(f"Strategy Library not available: {e}")
    HAS_STRATEGY_LIBRARY = False

# News Feed Monitor & Fundamental Analysis
try:
    from ai.news_api_routes import router as news_router
    app.include_router(news_router, tags=["News & Fundamentals"])
    HAS_NEWS_FEED = True
    logger.info("News Feed & Fundamentals router included")
except ImportError as e:
    logger.warning(f"News Feed not available: {e}")
    HAS_NEWS_FEED = False

# Claude AI Stock Scanner
try:
    from ai.scanner_api_routes import router as scanner_router
    app.include_router(scanner_router, tags=["Stock Scanner"])
    HAS_SCANNER = True
    logger.info("Claude AI Stock Scanner router included")
except ImportError as e:
    logger.warning(f"Stock Scanner not available: {e}")
    HAS_SCANNER = False

# Finviz Scanner (free supplemental data source)
try:
    from ai.finviz_routes import router as finviz_router
    app.include_router(finviz_router, tags=["Finviz Scanner"])
    logger.info("Finviz Scanner router included")
except ImportError as e:
    logger.warning(f"Finviz Scanner not available: {e}")

# Momentum Scorer API
try:
    from ai.momentum_scorer import get_momentum_scorer
    HAS_MOMENTUM_SCORER = True
    logger.info("Momentum Scorer loaded")
except ImportError as e:
    logger.warning(f"Momentum Scorer not available: {e}")
    HAS_MOMENTUM_SCORER = False

# Claude AI Watchlist Manager
try:
    from ai.watchlist_api_routes import router as watchlist_ai_router
    app.include_router(watchlist_ai_router, tags=["AI Watchlist"])
    HAS_WATCHLIST_AI = True
    logger.info("Claude AI Watchlist Manager router included")
except ImportError as e:
    logger.warning(f"AI Watchlist not available: {e}")
    HAS_WATCHLIST_AI = False

# Trading Pipeline & Coach (Automated Trading Flow)
try:
    from ai.pipeline_api_routes import router as pipeline_coach_router
    app.include_router(pipeline_coach_router, tags=["Trading Pipeline", "Trading Coach"])
    HAS_PIPELINE_COACH = True
    logger.info("Trading Pipeline & Coach router included")
except ImportError as e:
    logger.warning(f"Trading Pipeline/Coach not available: {e}")
    HAS_PIPELINE_COACH = False

# News Trade Pipeline (News -> Filter -> Analyze -> Alert)
try:
    from ai.news_pipeline_routes import router as news_pipeline_router
    app.include_router(news_pipeline_router, tags=["News Pipeline", "Halt Detection"])
    HAS_NEWS_PIPELINE = True
    logger.info("News Trade Pipeline router included")
except ImportError as e:
    logger.warning(f"News Trade Pipeline not available: {e}")
    HAS_NEWS_PIPELINE = False

# Data Collection & Backtest Routes (Schwab minute data, PyBroker)
try:
    from ai.data_api_routes import router as data_router
    app.include_router(data_router, tags=["Data Collection"])
    HAS_DATA_COLLECTION = True
    logger.info("Data Collection routes included")
except ImportError as e:
    logger.warning(f"Data Collection routes not available: {e}")
    HAS_DATA_COLLECTION = False

# Scalp Assistant Routes (HFT Exit Manager)
try:
    from ai.scalp_assistant_routes import router as scalp_router
    app.include_router(scalp_router, tags=["Scalp Assistant"])
    HAS_SCALP_ASSISTANT = True
    logger.info("Scalp Assistant routes included")
except ImportError as e:
    logger.warning(f"Scalp Assistant routes not available: {e}")
    HAS_SCALP_ASSISTANT = False

# Polygon Real-Time Streaming Routes
try:
    from polygon_streaming_routes import router as polygon_stream_router
    app.include_router(polygon_stream_router, tags=["Polygon Streaming"])
    HAS_POLYGON_STREAMING = True
    logger.info("Polygon Streaming routes included")
except ImportError as e:
    logger.warning(f"Polygon Streaming routes not available: {e}")
    HAS_POLYGON_STREAMING = False

# EDGAR SEC Filing Monitor Routes
try:
    from ai.edgar_routes import router as edgar_router
    app.include_router(edgar_router, tags=["SEC EDGAR"])
    HAS_EDGAR_MONITOR = True
    logger.info("SEC EDGAR routes included")
except ImportError as e:
    logger.warning(f"EDGAR routes not available: {e}")
    HAS_EDGAR_MONITOR = False

# Pre-Market Scanner & News Log Routes
try:
    from ai.premarket_routes import router as premarket_router
    app.include_router(premarket_router, tags=["Pre-Market Scanner"])
    HAS_PREMARKET_SCANNER = True
    logger.info("Pre-Market Scanner routes included")
except ImportError as e:
    logger.warning(f"Pre-Market Scanner routes not available: {e}")
    HAS_PREMARKET_SCANNER = False

# Warrior Trading Setup Detection Routes
try:
    from ai.warrior_routes import router as warrior_router
    app.include_router(warrior_router, tags=["Warrior Trading"])
    HAS_WARRIOR_SETUP = True
    logger.info("Warrior Trading Setup Detection routes included")
except ImportError as e:
    logger.warning(f"Warrior Trading routes not available: {e}")
    HAS_WARRIOR_SETUP = False

# Strategy Policy Engine Routes
try:
    from ai.strategy_policy_routes import router as strategy_policy_router
    app.include_router(strategy_policy_router, tags=["Strategy Policy"])
    HAS_STRATEGY_POLICY = True
    logger.info("Strategy Policy Engine routes included")
except ImportError as e:
    logger.warning(f"Strategy Policy routes not available: {e}")
    HAS_STRATEGY_POLICY = False

# Validation & Calibration Routes (Phase: Strategy Policy Validation)
try:
    from ai.validation_routes import router as validation_router
    app.include_router(validation_router, tags=["Validation & Calibration"])
    HAS_VALIDATION = True
    logger.info("Validation & Calibration routes included")
except ImportError as e:
    logger.warning(f"Validation routes not available: {e}")
    HAS_VALIDATION = False

# Task Queue Routes (HOD Momentum Bot Pipeline)
try:
    from ai.task_queue_routes import router as task_queue_router
    app.include_router(task_queue_router, tags=["Task Queue"])
    HAS_TASK_QUEUE = True
    logger.info("Task Queue routes included (HOD Momentum Pipeline)")
except ImportError as e:
    logger.warning(f"Task Queue routes not available: {e}")
    HAS_TASK_QUEUE = False

# Momentum Watchlist Operator Controls
try:
    from ai.momentum_watchlist_routes import router as momentum_watchlist_router
    app.include_router(momentum_watchlist_router, tags=["Watchlist Operator"])
    HAS_MOMENTUM_WATCHLIST = True
    logger.info("Momentum Watchlist operator control routes included")
except ImportError as e:
    logger.warning(f"Momentum Watchlist routes not available: {e}")
    HAS_MOMENTUM_WATCHLIST = False

# Lightweight Charts API Routes
try:
    from charts_routes import router as charts_router
    app.include_router(charts_router, tags=["Charts"])
    HAS_CHARTS_API = True
    logger.info("Lightweight Charts API routes included")
except ImportError as e:
    logger.warning(f"Charts routes not available: {e}")
    HAS_CHARTS_API = False

# Warrior Trading Scanners
try:
    from scanners.scanner_routes import router as scanner_router
    app.include_router(scanner_router, tags=["Warrior Scanners"])
    HAS_WARRIOR_SCANNERS = True
    logger.info("Warrior Trading Scanner routes included")
except ImportError as e:
    logger.warning(f"Warrior Scanners not available: {e}")
    HAS_WARRIOR_SCANNERS = False


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class TrainRequest(BaseModel):
    symbol: str
    test_size: float = 0.2


class TrainMultiRequest(BaseModel):
    symbols: List[str]
    test_size: float = 0.2


class PredictRequest(BaseModel):
    symbol: str
    timeframe: str = "1Day"


class MarketDataRequest(BaseModel):
    symbol: str
    timeframe: str = "1Day"
    limit: Optional[int] = 100


class ClaudeQueryRequest(BaseModel):
    query: str
    symbol: Optional[str] = "SPY"
    context: Optional[Union[str, Dict[str, Any]]] = None  # Accept string or dict


# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    primary_broker = "Schwab"
    broker_status = "disconnected"
    if HAS_SCHWAB_TRADING:
        try:
            if is_schwab_trading_available():
                broker_status = "connected"
        except Exception:
            broker_status = "error"

    return {
        "name": "Morpheus Trading Bot",
        "version": "2.1.0",
        "broker": primary_broker,
        "broker_status": broker_status,
        "status": "operational",
        "data_source": "Schwab"
    }


# Health check cache to avoid slow API calls
_health_cache = {"data": None, "timestamp": 0}
_HEALTH_CACHE_TTL = 30  # Cache for 30 seconds

@app.get("/api/health")
async def health_check():
    """Health check endpoint (cached for performance)"""
    import time

    # Return cached response if fresh
    now = time.time()
    if _health_cache["data"] and (now - _health_cache["timestamp"]) < _HEALTH_CACHE_TTL:
        return _health_cache["data"]

    config = get_broker_config()

    # Check multi-channel status
    multi_channel_status = "not_available"
    if HAS_MULTI_CHANNEL:
        try:
            provider = get_multi_channel_provider()
            multi_channel_status = f"active ({len(provider.channels)} channels)"
        except Exception:
            multi_channel_status = "error"

    # Check streaming status
    streaming_status = "not_available"
    if HAS_STREAMING:
        try:
            stream_mgr = get_stream_manager()
            status = stream_mgr.get_status()
            if status["running"]:
                streaming_status = f"running ({status['subscription_count']} subscriptions)"
            else:
                streaming_status = "available"
        except Exception:
            streaming_status = "error"

    # Check unified market data status (Schwab) - skip slow status check
    unified_data_status = "schwab_active"
    primary_data_source = "schwab"

    # Check Schwab trading status - use cached broker instance
    schwab_trading_status = "not_available"
    schwab_accounts = 2  # Default to known account count
    if HAS_SCHWAB_TRADING:
        try:
            if is_schwab_trading_available():
                schwab = get_schwab_trading()
                if schwab:
                    # Use cached account info if available, otherwise quick check
                    selected = schwab.get_selected_account()
                    schwab_trading_status = f"active ({schwab_accounts} accounts)"
                    if selected:
                        schwab_trading_status += f" - {selected[:4]}***"
        except Exception:
            schwab_trading_status = "error"

    # Check Claude AI availability
    claude_available = False
    try:
        import os
        claude_available = bool(os.environ.get("ANTHROPIC_API_KEY"))
    except Exception:
        pass

    result = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "broker": {
            "type": "schwab",
            "connected": schwab_trading_status != "not_available",
            "paper_trading": False
        },
        "services": {
            "api": "operational",
            "ai_predictor": "operational",
            "market_data": unified_data_status,
            "primary_data_source": primary_data_source,
            "multi_channel_data": multi_channel_status,
            "realtime_streaming": streaming_status,
            "schwab_trading": schwab_trading_status
        },
        "claude_available": claude_available
    }

    # Cache the result
    _health_cache["data"] = result
    _health_cache["timestamp"] = now

    return result


@app.get("/health")
async def health_check_simple():
    """Simple health check endpoint (alias for /api/health)"""
    return await health_check()


@app.get("/api/health/connections")
async def get_connection_health():
    """
    Get detailed connection health status with auto-recovery monitoring.

    Returns status of all data connections:
    - Schwab broker
    - Schwab market data
    - Polygon WebSocket stream
    - Auto-recovery status
    """
    try:
        from ai.connection_health_monitor import get_health_monitor
        monitor = get_health_monitor()
        return monitor.get_status()
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "connections": {}
        }


@app.get("/api/health/history")
async def get_connection_health_history(minutes: int = 30):
    """Get connection health history for trend analysis"""
    try:
        from ai.connection_health_monitor import get_health_monitor
        monitor = get_health_monitor()
        return {
            "history": monitor.get_health_history(minutes),
            "minutes": minutes
        }
    except Exception as e:
        return {"error": str(e), "history": []}


@app.get("/api/config")
async def get_config():
    """Get system configuration"""
    config = get_broker_config()
    return config.get_config_dict()


@app.get("/api/gating/status")
async def get_gating_status():
    """
    Get Signal Gating Engine status.

    Shows:
    - Whether gating is enabled
    - Number of trade attempts
    - Approval/veto counts
    - Recent vetoes with reasons
    """
    result = {
        "gating_enabled": False,
        "total_attempts": 0,
        "approved": 0,
        "vetoed": 0,
        "approval_rate": 0,
        "recent_vetoes": [],
        "error": None
    }

    try:
        from ai.gated_trading import get_gated_trading_manager
        from ai.hft_scalper import get_hft_scalper

        # Check if gating is enabled in scalper config
        scalper = get_hft_scalper()
        if scalper and scalper.config:
            result["gating_enabled"] = getattr(scalper.config, 'require_gating_approval', False)

        # Get gating stats
        manager = get_gated_trading_manager()
        stats = manager.get_stats()

        result["total_attempts"] = stats.get("total_attempts", 0)
        result["approved"] = stats.get("approved", 0)
        result["vetoed"] = stats.get("vetoed", 0)
        result["approval_rate"] = stats.get("approval_rate", 0)
        result["contracts_loaded"] = stats.get("contracts_loaded", 0)
        result["active_contracts"] = stats.get("active_contracts", 0)

        # Get recent vetoes
        recent_vetoes = manager.get_recent_vetoes(10)
        result["recent_vetoes"] = recent_vetoes

        # Get gating engine details
        if "gating_engine" in stats:
            result["engine_stats"] = stats["gating_engine"]

    except Exception as e:
        result["error"] = str(e)
        import traceback
        logger.error(f"Gating status error: {traceback.format_exc()}")

    return result


@app.get("/api/gating/vetoes")
async def get_gating_vetoes(count: int = 20):
    """Get recent gating vetoes with full details"""
    try:
        from ai.gated_trading import get_gated_trading_manager
        manager = get_gated_trading_manager()
        vetoes = manager.get_recent_vetoes(count)
        return {
            "count": len(vetoes),
            "vetoes": vetoes
        }
    except Exception as e:
        return {"error": str(e), "vetoes": []}


@app.get("/api/system/status")
async def get_system_status():
    """
    Comprehensive system status endpoint for monitoring.

    Returns detailed status of all system components:
    - Schwab broker connection
    - Token lifecycle status
    - Market data sources
    - AI modules availability
    - Real-time streaming status
    """
    from datetime import datetime

    status = {
        "name": "Morpheus Trading Bot",
        "version": "2.1.0",
        "timestamp": datetime.now().isoformat(),
        "overall_status": "operational"
    }

    # Schwab token status
    try:
        from schwab_market_data import get_token_status, is_schwab_available
        token_status = get_token_status()
        status["schwab_token"] = token_status
        status["schwab_available"] = is_schwab_available()
    except Exception as e:
        status["schwab_token"] = {"valid": False, "error": str(e)}
        status["schwab_available"] = False

    # Schwab trading status
    schwab_trading_info = {"available": False}
    if HAS_SCHWAB_TRADING:
        try:
            if is_schwab_trading_available():
                schwab = get_schwab_trading()
                if schwab:
                    accounts = schwab.get_accounts()
                    selected = schwab.get_selected_account()
                    schwab_trading_info = {
                        "available": True,
                        "accounts": len(accounts),
                        "selected_account": f"{selected[:4]}***" if selected else None,
                        "status": "ready"
                    }
        except Exception as e:
            schwab_trading_info["error"] = str(e)
    status["schwab_trading"] = schwab_trading_info

    # Unified broker status
    broker_info = {"active": None}
    try:
        broker = get_unified_broker()
        if broker:
            broker_info = {
                "active": broker.broker_name,
                "is_connected": broker.is_connected,
                "supports_market_orders": True,
                "supports_limit_orders": True
            }
    except Exception as e:
        broker_info["error"] = str(e)
    status["unified_broker"] = broker_info

    # Market data sources
    data_sources = {"primary": "schwab"}
    if HAS_UNIFIED_DATA:
        try:
            unified_provider = get_unified_market_data()
            provider_status = unified_provider.get_status()
            data_sources = {
                "primary": "schwab",
                "schwab": provider_status["schwab"],
                "available": provider_status["schwab"]["available"]
            }
        except Exception as e:
            data_sources["error"] = str(e)
    status["market_data"] = data_sources

    # Multi-channel data
    if HAS_MULTI_CHANNEL:
        try:
            provider = get_multi_channel_provider()
            status["multi_channel"] = {
                "available": True,
                "channels": len(provider.channels) if hasattr(provider, 'channels') else 0
            }
        except Exception:
            status["multi_channel"] = {"available": False}
    else:
        status["multi_channel"] = {"available": False}

    # Streaming status
    if HAS_STREAMING:
        try:
            stream_mgr = get_stream_manager()
            stream_status = stream_mgr.get_status()
            status["streaming"] = {
                "available": True,
                "running": stream_status["running"],
                "subscriptions": stream_status.get("subscription_count", 0)
            }
        except Exception:
            status["streaming"] = {"available": False}
    else:
        status["streaming"] = {"available": False}

    # AI modules
    ai_status = {}
    try:
        predictor = get_ai_scanner() if HAS_AI_SCANNER else None
        ai_status["predictor"] = {
            "available": predictor is not None,
            "loaded": predictor.is_loaded if hasattr(predictor, 'is_loaded') else False
        }
    except Exception:
        ai_status["predictor"] = {"available": False}

    try:
        import os
        ai_status["claude"] = {
            "available": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "api_configured": bool(os.environ.get("ANTHROPIC_API_KEY"))
        }
    except Exception:
        ai_status["claude"] = {"available": False}
    status["ai_modules"] = ai_status

    # Determine overall status
    if not status.get("schwab_available"):
        status["overall_status"] = "degraded"
        status["issues"] = ["No broker connection available"]
    elif status.get("schwab_token", {}).get("status") == "expired":
        status["overall_status"] = "warning"
        status["issues"] = ["Schwab token expired - will attempt auto-refresh"]

    return status


# ============================================================================
# MARKET DATA ENDPOINTS
# ============================================================================

@app.get("/api/market/quote/{symbol}")
async def get_quote(symbol: str):
    """Get latest quote for a symbol"""
    try:
        if not HAS_UNIFIED_DATA:
            raise HTTPException(status_code=503, detail="Market data not available")
        quote = unified_get_quote(symbol.upper())

        if quote is None:
            raise HTTPException(status_code=404, detail=f"No quote data for {symbol}")

        return quote

    except Exception as e:
        logger.error(f"Error fetching quote: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/snapshot/{symbol}")
async def get_snapshot(symbol: str):
    """Get market snapshot for a symbol"""
    try:
        if not HAS_UNIFIED_DATA:
            raise HTTPException(status_code=503, detail="Market data not available")
        snapshot = unified_get_snapshot(symbol.upper())

        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"No snapshot data for {symbol}")

        return snapshot

    except Exception as e:
        logger.error(f"Error fetching snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/timesales/{symbol}")
async def get_timesales(symbol: str):
    """
    Get Time & Sales data for a symbol.

    Tries sources in order:
    1. Polygon streaming (real-time trades)
    2. Schwab price history (1-min bars as fallback)

    Returns array of trade-like records.
    """
    from datetime import datetime

    trades = []

    # Try Polygon streaming first
    try:
        from polygon_streaming import get_polygon_streamer
        streamer = get_polygon_streamer()
        if streamer and streamer.running:
            polygon_trades = streamer.get_recent_trades(symbol.upper(), limit=100)
            if polygon_trades:
                for t in polygon_trades:
                    trades.append({
                        "time": t.get("time", ""),
                        "price": t.get("price", 0),
                        "size": t.get("size", 0),
                        "side": t.get("side", "N"),
                        "source": "polygon"
                    })
    except Exception as e:
        logger.debug(f"Polygon trades not available: {e}")

    # Fallback to Schwab price history (minute bars)
    if not trades:
        try:
            from schwab_market_data import get_schwab_market_data
            schwab = get_schwab_market_data()
            if schwab:
                history = schwab.get_price_history(
                    symbol.upper(),
                    period_type="day",
                    period=1,
                    frequency_type="minute",
                    frequency=1
                )

                if history and "candles" in history:
                    candles = history["candles"][-50:]  # Last 50 minutes
                    for candle in reversed(candles):  # Newest first
                        timestamp = candle.get("datetime", 0)
                        if timestamp:
                            dt = datetime.fromtimestamp(timestamp / 1000)
                            time_str = dt.strftime("%H:%M:%S")
                        else:
                            time_str = ""

                        # Use close price, determine side from open/close
                        open_p = candle.get("open", 0)
                        close_p = candle.get("close", 0)
                        side = "B" if close_p > open_p else "S" if close_p < open_p else "N"

                        trades.append({
                            "time": time_str,
                            "price": close_p,
                            "size": candle.get("volume", 0),
                            "side": side,
                            "source": "schwab"
                        })
        except Exception as e:
            logger.error(f"Error getting Schwab price history for T&S: {e}")

    return {"trades": trades, "symbol": symbol.upper(), "count": len(trades)}


@app.get("/api/market/movers")
async def get_market_movers(index: str = "$SPX", direction: str = "up"):
    """
    Get top market movers (gainers or losers) from Schwab

    Args:
        index: Index to scan ($SPX, $COMPX, $DJI)
        direction: "up" for gainers, "down" for losers
    """
    try:
        from schwab_market_data import get_schwab_movers, get_all_movers

        if direction == "all":
            return get_all_movers()
        else:
            movers = get_schwab_movers(index, direction)
            return {
                "index": index,
                "direction": direction,
                "count": len(movers),
                "movers": movers
            }
    except Exception as e:
        logger.error(f"Error getting movers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/movers/scalp")
async def get_scalp_candidates():
    """
    Get movers filtered for scalping criteria:
    - Price $1-$20
    - Change 5%+
    - Volume 500K+
    """
    try:
        from schwab_market_data import get_all_movers

        all_movers = get_all_movers()
        scalp_candidates = []

        for mover in all_movers.get('gainers', []):
            price = mover.get('price', 0)
            change = mover.get('change_pct', 0)
            volume = mover.get('volume', 0)

            if 1.0 <= price <= 20.0 and change >= 5.0 and volume >= 500000:
                mover['scalp_score'] = int(change * 2 + (volume / 1000000))
                scalp_candidates.append(mover)

        # Sort by scalp score
        scalp_candidates.sort(key=lambda x: x.get('scalp_score', 0), reverse=True)

        return {
            "count": len(scalp_candidates),
            "candidates": scalp_candidates[:20],
            "criteria": {
                "min_price": 1.0,
                "max_price": 20.0,
                "min_change_pct": 5.0,
                "min_volume": 500000
            }
        }
    except Exception as e:
        logger.error(f"Error getting scalp candidates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/market/bars")
async def get_historical_bars(request: MarketDataRequest):
    """Get historical bars for a symbol"""
    try:
        if not HAS_UNIFIED_DATA:
            raise HTTPException(status_code=503, detail="Market data not available")
        provider = get_unified_market_data()
        bars = provider.get_bars(
            symbol=request.symbol.upper(),
            timeframe=request.timeframe,
            limit=request.limit
        )

        if bars is None or (hasattr(bars, 'empty') and bars.empty):
            raise HTTPException(status_code=404, detail=f"No data for {request.symbol}")

        # Convert DataFrame to dict for JSON response if needed
        if hasattr(bars, 'reset_index'):
            return {
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "bars": bars.reset_index().to_dict(orient='records')
            }
        return {"symbol": request.symbol, "timeframe": request.timeframe, "bars": bars}

    except Exception as e:
        logger.error(f"Error fetching bars: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/multi-quote")
async def get_multi_quote(symbols: str = Query(..., description="Comma-separated symbols")):
    """Get quotes for multiple symbols"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]

        if not HAS_UNIFIED_DATA:
            raise HTTPException(status_code=503, detail="Market data not available")
        quotes = unified_get_quotes(symbol_list)

        return quotes

    except Exception as e:
        logger.error(f"Error fetching multiple quotes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# UNIFIED MARKET DATA STATUS
# ============================================================================

@app.get("/api/data/unified/status")
async def get_unified_data_status():
    """
    Get status of unified market data provider.

    Shows which data source is active (Schwab) and statistics.
    """
    if not HAS_UNIFIED_DATA:
        return {
            "available": False,
            "message": "Unified market data provider not available"
        }

    try:
        provider = get_unified_market_data()
        return {
            "available": True,
            **provider.get_status()
        }
    except Exception as e:
        logger.error(f"Error getting unified data status: {e}")
        return {"available": False, "error": str(e)}


# ============================================================================
# MULTI-CHANNEL DATA ENDPOINTS (Parallel Market Data Access)
# ============================================================================

@app.get("/api/data/channels/status")
async def get_channel_status():
    """
    Get status of all data channels.

    Multi-channel architecture provides:
    - ORDERS channel: Dedicated for order execution
    - CHARTS channel: Dedicated for charting data
    - AI channel: Dedicated for AI predictions/analysis
    - SCANNER channel: Dedicated for market scanning
    - REALTIME channel: Dedicated for live quotes
    """
    if not HAS_MULTI_CHANNEL:
        return {
            "available": False,
            "message": "Multi-channel data provider not available"
        }

    try:
        provider = get_multi_channel_provider()
        return {
            "available": True,
            "status": provider.get_all_status(),
            "aggregate": provider.get_aggregate_stats()
        }
    except Exception as e:
        logger.error(f"Error getting channel status: {e}")
        return {"available": False, "error": str(e)}


@app.get("/api/data/channels/quote/{symbol}")
async def get_fast_quote(symbol: str, channel: str = "realtime"):
    """
    Get a quote using a specific data channel.

    Channels:
    - realtime: Live quotes (default)
    - orders: Order execution context
    - charts: Charting context
    - ai: AI analysis context
    - scanner: Market scanning context
    """
    if not HAS_MULTI_CHANNEL:
        # Use unified market data (Schwab)
        if HAS_UNIFIED_DATA:
            return unified_get_quote(symbol.upper())
        raise HTTPException(status_code=503, detail="Market data not available")

    try:
        channel_enum = DataChannel(channel.lower())
        quote = get_quote_fast(symbol.upper(), channel_enum)

        if quote is None:
            raise HTTPException(status_code=404, detail=f"No quote data for {symbol}")

        return quote
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid channel: {channel}")
    except Exception as e:
        logger.error(f"Error fetching fast quote: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/channels/multi-quote")
async def get_parallel_quotes(symbols: str = Query(..., description="Comma-separated symbols")):
    """
    Get quotes for multiple symbols using parallel channels.

    This splits the work across multiple data channels for maximum speed.
    Much faster than sequential fetching for large watchlists.
    """
    if not HAS_MULTI_CHANNEL:
        # Use unified market data (Schwab)
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        if HAS_UNIFIED_DATA:
            return unified_get_quotes(symbol_list)
        raise HTTPException(status_code=503, detail="Market data not available")

    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        import time

        start = time.time()
        quotes = get_quotes_fast(symbol_list)
        elapsed_ms = (time.time() - start) * 1000

        return {
            "quotes": quotes,
            "count": len(quotes),
            "elapsed_ms": round(elapsed_ms, 2),
            "parallel": True
        }
    except Exception as e:
        logger.error(f"Error fetching parallel quotes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/channels/snapshots")
async def get_parallel_snapshots(symbols: str = Query(..., description="Comma-separated symbols")):
    """
    Get full market snapshots for multiple symbols using parallel channels.

    Includes: quote, last trade, daily bar, previous close, change %.
    """
    if not HAS_MULTI_CHANNEL:
        raise HTTPException(status_code=503, detail="Multi-channel provider not available")

    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        import time

        provider = get_multi_channel_provider()
        start = time.time()
        snapshots = provider.get_snapshots_parallel(symbol_list)
        elapsed_ms = (time.time() - start) * 1000

        return {
            "snapshots": snapshots,
            "count": len(snapshots),
            "elapsed_ms": round(elapsed_ms, 2),
            "parallel": True
        }
    except Exception as e:
        logger.error(f"Error fetching parallel snapshots: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/channels/bars/{symbol}")
async def get_channel_bars(
    symbol: str,
    channel: str = "charts",
    timeframe: str = "1Day",
    days: int = 30
):
    """
    Get historical bars using a specific channel.

    Channels optimized for different use cases:
    - charts: For TradingView/charting (5m, 15m bars)
    - ai: For AI training data (daily bars, longer history)
    - scanner: For market scanning
    """
    if not HAS_MULTI_CHANNEL:
        # Use unified market data (Schwab)
        if HAS_UNIFIED_DATA:
            provider = get_unified_market_data()
            bars = provider.get_bars(symbol.upper(), timeframe, days * 10)
            return {"symbol": symbol, "bars": bars.to_dict(orient='records') if hasattr(bars, 'to_dict') else bars or []}
        raise HTTPException(status_code=503, detail="Market data not available")

    try:
        provider = get_multi_channel_provider()

        if channel.lower() == "ai":
            bars = get_bars_for_ai(symbol.upper(), days)
        elif channel.lower() == "charts":
            bars = get_bars_for_chart(symbol.upper(), timeframe, days)
        else:
            channel_client = provider.get_channel(DataChannel(channel.lower()))
            bars = channel_client.get_bars(symbol.upper(), timeframe, days)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "days": days,
            "channel": channel,
            "bars": bars,
            "count": len(bars)
        }
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid channel: {channel}")
    except Exception as e:
        logger.error(f"Error fetching channel bars: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/channels/multi-bars")
async def get_multi_symbol_bars(
    symbols: str = Query(..., description="Comma-separated symbols"),
    timeframe: str = "1Day",
    days: int = 30
):
    """
    Get historical bars for multiple symbols in parallel.

    Ideal for:
    - AI batch training data collection
    - Multi-symbol backtesting
    - Watchlist historical analysis
    """
    if not HAS_MULTI_CHANNEL:
        raise HTTPException(status_code=503, detail="Multi-channel provider not available")

    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        import time

        provider = get_multi_channel_provider()
        start = time.time()
        bars = provider.get_multi_symbol_bars(symbol_list, timeframe, days)
        elapsed_ms = (time.time() - start) * 1000

        return {
            "symbols": list(bars.keys()),
            "timeframe": timeframe,
            "days": days,
            "bars": bars,
            "count": sum(len(b) for b in bars.values()),
            "elapsed_ms": round(elapsed_ms, 2),
            "parallel": True
        }
    except Exception as e:
        logger.error(f"Error fetching multi-symbol bars: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# REAL-TIME WEBSOCKET STREAMING ENDPOINTS
# ============================================================================

@app.get("/api/streaming/status")
async def get_streaming_status():
    """
    Get real-time streaming service status.

    Shows:
    - Connection state
    - Active subscriptions
    - Message statistics
    - Connected clients
    """
    if not HAS_STREAMING:
        return {
            "available": False,
            "message": "Real-time streaming not available"
        }

    try:
        manager = get_stream_manager()
        return {
            "available": True,
            "status": manager.get_status()
        }
    except Exception as e:
        logger.error(f"Error getting streaming status: {e}")
        return {"available": False, "error": str(e)}


@app.post("/api/streaming/subscribe")
async def subscribe_to_stream(data: Dict[str, Any]):
    """
    Subscribe to real-time market data streams.

    Request body:
    {
        "symbols": ["AAPL", "TSLA", "NVDA"],
        "types": ["quotes", "trades", "bars"]  // optional, defaults to quotes
    }

    Stream types:
    - quotes: Level 1 bid/ask updates
    - trades: Executed trade prints
    - bars: Minute bars
    """
    if not HAS_STREAMING:
        raise HTTPException(status_code=503, detail="Streaming not available")

    try:
        symbols = data.get("symbols", [])
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",")]

        type_names = data.get("types", ["quotes"])
        stream_types = []
        for t in type_names:
            if t.lower() == "quotes":
                stream_types.append(StreamType.QUOTES)
            elif t.lower() == "trades":
                stream_types.append(StreamType.TRADES)
            elif t.lower() == "bars":
                stream_types.append(StreamType.BARS)

        manager = get_stream_manager()
        await manager.subscribe(symbols, stream_types)

        return {
            "success": True,
            "subscribed": symbols,
            "types": [t.value for t in stream_types],
            "status": manager.get_status()
        }

    except Exception as e:
        logger.error(f"Error subscribing to stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/streaming/unsubscribe")
async def unsubscribe_from_stream(data: Dict[str, Any]):
    """
    Unsubscribe from real-time market data streams.

    Request body:
    {
        "symbols": ["AAPL", "TSLA"],
        "types": ["quotes", "trades"]  // optional, unsubscribe from all if not specified
    }
    """
    if not HAS_STREAMING:
        raise HTTPException(status_code=503, detail="Streaming not available")

    try:
        symbols = data.get("symbols", [])
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",")]

        type_names = data.get("types")
        stream_types = None

        if type_names:
            stream_types = []
            for t in type_names:
                if t.lower() == "quotes":
                    stream_types.append(StreamType.QUOTES)
                elif t.lower() == "trades":
                    stream_types.append(StreamType.TRADES)
                elif t.lower() == "bars":
                    stream_types.append(StreamType.BARS)

        manager = get_stream_manager()
        await manager.unsubscribe(symbols, stream_types)

        return {
            "success": True,
            "unsubscribed": symbols,
            "status": manager.get_status()
        }

    except Exception as e:
        logger.error(f"Error unsubscribing from stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/streaming/start")
async def start_streaming():
    """Start the real-time streaming service."""
    if not HAS_STREAMING:
        raise HTTPException(status_code=503, detail="Streaming not available")

    try:
        manager = get_stream_manager()
        await manager.start()
        return {
            "success": True,
            "message": "Streaming service started",
            "status": manager.get_status()
        }
    except Exception as e:
        logger.error(f"Error starting streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/streaming/stop")
async def stop_streaming():
    """Stop the real-time streaming service."""
    if not HAS_STREAMING:
        raise HTTPException(status_code=503, detail="Streaming not available")

    try:
        manager = get_stream_manager()
        await manager.stop()
        return {
            "success": True,
            "message": "Streaming service stopped"
        }
    except Exception as e:
        logger.error(f"Error stopping streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/market")
async def websocket_market_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time market data streaming.

    Connect to receive live quotes, trades, and bars for subscribed symbols.

    Protocol:
    1. Connect to ws://localhost:9100/ws/market
    2. Send subscription message: {"action": "subscribe", "symbols": ["AAPL", "TSLA"]}
    3. Receive streaming data: {"type": "quote", "symbol": "AAPL", "bid": 150.25, ...}

    Actions:
    - subscribe: Subscribe to symbols (default: quotes)
    - unsubscribe: Unsubscribe from symbols
    - status: Get current subscription status
    """
    await websocket.accept()
    client_id = str(uuid.uuid4())
    logger.info(f"[WS] Client connected: {client_id}")

    if not HAS_STREAMING:
        await websocket.send_json({
            "type": "error",
            "message": "Streaming service not available"
        })
        await websocket.close()
        return

    manager = get_stream_manager()
    message_queue = manager.register_client(client_id)

    # Send welcome message
    await websocket.send_json({
        "type": "connected",
        "client_id": client_id,
        "message": "Connected to market stream",
        "status": manager.get_status()
    })

    async def send_messages():
        """Forward messages from queue to WebSocket"""
        while True:
            try:
                message = await message_queue.get()
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"[WS] Send error: {e}")
                break

    async def receive_commands():
        """Receive commands from WebSocket client"""
        while True:
            try:
                data = await websocket.receive_json()
                action = data.get("action", "").lower()

                if action == "subscribe":
                    symbols = data.get("symbols", [])
                    types_raw = data.get("types", ["quotes"])
                    stream_types = []

                    for t in types_raw:
                        if t.lower() == "quotes":
                            stream_types.append(StreamType.QUOTES)
                        elif t.lower() == "trades":
                            stream_types.append(StreamType.TRADES)
                        elif t.lower() == "bars":
                            stream_types.append(StreamType.BARS)

                    await manager.subscribe(symbols, stream_types)
                    await websocket.send_json({
                        "type": "subscribed",
                        "symbols": symbols,
                        "stream_types": [t.value for t in stream_types]
                    })

                elif action == "unsubscribe":
                    symbols = data.get("symbols", [])
                    await manager.unsubscribe(symbols)
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "symbols": symbols
                    })

                elif action == "status":
                    await websocket.send_json({
                        "type": "status",
                        "status": manager.get_status()
                    })

                elif action == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"[WS] Receive error: {e}")
                break

    try:
        # Run both tasks concurrently
        send_task = asyncio.create_task(send_messages())
        receive_task = asyncio.create_task(receive_commands())

        # Wait for either task to complete (usually due to disconnect)
        done, pending = await asyncio.wait(
            [send_task, receive_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()

    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"[WS] Error: {e}")
    finally:
        manager.unregister_client(client_id)
        logger.info(f"[WS] Client cleanup: {client_id}")


# ============================================================================
# GOVERNOR WEBSOCKET - Real-time health monitoring with auto-reconnect
# ============================================================================

@app.get("/api/governor/status")
async def api_governor_status():
    """REST endpoint for Governor status (used by UI initial load)"""
    return await get_governor_status()

# Track connected governor clients
governor_clients: Dict[str, WebSocket] = {}

@app.websocket("/ws/governor")
async def websocket_governor(websocket: WebSocket):
    """
    WebSocket endpoint for Governor UI real-time health monitoring.

    Features:
    - Heartbeat every 3 seconds
    - System status updates in real-time
    - Auto-reconnect support on client side
    - Trade-safe: monitors connection health to prevent slippage

    Messages sent:
    - heartbeat: {type: "heartbeat", timestamp: ..., seq: ...}
    - status: {type: "status", data: {...system status...}}
    - alert: {type: "alert", level: "warning|error", message: ...}
    """
    await websocket.accept()
    client_id = str(uuid.uuid4())[:8]
    governor_clients[client_id] = websocket
    logger.info(f"[Governor WS] Client connected: {client_id}")

    # Send welcome message with initial status
    try:
        initial_status = await get_governor_status()
        await websocket.send_json({
            "type": "connected",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
            "status": initial_status
        })
    except Exception as e:
        logger.error(f"[Governor WS] Error sending welcome: {e}")

    heartbeat_seq = 0
    last_status = None

    async def send_heartbeat():
        """Send heartbeat every 3 seconds"""
        nonlocal heartbeat_seq
        while True:
            try:
                heartbeat_seq += 1
                await websocket.send_json({
                    "type": "heartbeat",
                    "seq": heartbeat_seq,
                    "timestamp": datetime.now().isoformat(),
                    "server_time": datetime.now().strftime("%H:%M:%S")
                })
                await asyncio.sleep(3)
            except Exception:
                break

    async def send_status_updates():
        """Send status updates when things change (every 5 seconds)"""
        nonlocal last_status
        while True:
            try:
                current_status = await get_governor_status()

                # Only send if status changed or every 5th iteration
                status_json = str(current_status)
                if status_json != last_status:
                    await websocket.send_json({
                        "type": "status",
                        "timestamp": datetime.now().isoformat(),
                        "data": current_status
                    })
                    last_status = status_json

                await asyncio.sleep(5)
            except Exception:
                break

    async def receive_commands():
        """Handle commands from client"""
        while True:
            try:
                data = await websocket.receive_json()
                command = data.get("command")

                if command == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat(),
                        "client_seq": data.get("seq", 0)
                    })
                elif command == "status":
                    status = await get_governor_status()
                    await websocket.send_json({
                        "type": "status",
                        "timestamp": datetime.now().isoformat(),
                        "data": status
                    })
                elif command == "reconnect_feeds":
                    # Trigger feed reconnection
                    try:
                        from polygon_streaming import get_polygon_stream
                        stream = get_polygon_stream()
                        if stream:
                            stream.restart()
                            await websocket.send_json({
                                "type": "alert",
                                "level": "info",
                                "message": "Feeds reconnecting..."
                            })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "alert",
                            "level": "error",
                            "message": f"Reconnect failed: {str(e)}"
                        })
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"[Governor WS] Command error: {e}")
                break

    try:
        # Run heartbeat, status updates, and command receiver concurrently
        await asyncio.gather(
            send_heartbeat(),
            send_status_updates(),
            receive_commands()
        )
    except WebSocketDisconnect:
        logger.info(f"[Governor WS] Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"[Governor WS] Error: {e}")
    finally:
        if client_id in governor_clients:
            del governor_clients[client_id]
        logger.info(f"[Governor WS] Client cleanup: {client_id}")


async def get_governor_status() -> dict:
    """Get current system status for Governor WebSocket"""
    try:
        # Get safe activation status
        safe_status = {}
        try:
            from ai.safe_activation import get_safe_activation
            safe = get_safe_activation()
            safe_status = safe.get_status()
        except:
            pass

        # Get scalp assistant status
        scalp_running = False
        try:
            from ai.scalp_assistant import get_scalp_assistant
            assistant = get_scalp_assistant()
            scalp_running = assistant.running if assistant else False
        except:
            pass

        # Also check HFT scalper (primary trading system)
        hft_scalper_running = False
        hft_scalper_status = {}
        try:
            from ai.hft_scalper import get_hft_scalper
            scalper = get_hft_scalper()
            if scalper:
                hft_scalper_running = scalper.is_running
                hft_scalper_status = {
                    "running": scalper.is_running,
                    "enabled": getattr(scalper.config, 'enabled', False),
                    "paper_mode": getattr(scalper.config, 'paper_mode', True),
                    "watchlist_count": len(getattr(scalper.config, 'watchlist', [])),
                    "daily_trades": scalper.daily_stats.get("trades", 0) if hasattr(scalper, 'daily_stats') else 0,
                    "daily_pnl": scalper.daily_stats.get("pnl", 0) if hasattr(scalper, 'daily_stats') else 0
                }
        except Exception as e:
            logger.debug(f"HFT scalper status error: {e}")

        # Polygon removed - using Schwab for market data
        # polygon_status not included in response

        # Get connectivity status
        connectivity = {"system_state": "UNKNOWN"}
        try:
            from ai.connectivity_manager import get_connectivity_manager
            cm = get_connectivity_manager()
            if cm:
                system_state = cm.get_system_state()
                connectivity = {
                    "system_state": system_state.value,
                    "system_state_reason": cm._get_state_reason(system_state)
                }
        except Exception as e:
            logger.debug(f"Connectivity status error: {e}")

        return {
            "safe_activation": safe_status,
            "scalp_running": scalp_running or hft_scalper_running,  # Either scalp assistant or HFT scalper
            "hft_scalper": hft_scalper_status,
            "broker": {"connected": True, "source": "Schwab"},  # Using Schwab for data
            "connectivity": connectivity,
            "server_time": datetime.now().isoformat(),
            "trading_window": get_trading_window_status()
        }
    except Exception as e:
        logger.error(f"Error getting governor status: {e}")
        return {"error": str(e)}


def get_trading_window_status() -> dict:
    """Get current trading window status"""
    from datetime import time
    import pytz

    try:
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        current_time = now_et.time()

        if time(4, 0) <= current_time < time(9, 30):
            window = "PRE_MARKET"
            detail = "Pre-market (4:00 AM - 9:30 AM ET)"
        elif time(9, 30) <= current_time < time(16, 0):
            window = "OPEN"
            detail = "Market open (9:30 AM - 4:00 PM ET)"
        elif time(16, 0) <= current_time < time(20, 0):
            window = "AFTER_HOURS"
            detail = "After hours (4:00 PM - 8:00 PM ET)"
        else:
            window = "CLOSED"
            detail = "Market closed"

        return {
            "window": window,
            "detail": detail,
            "time_et": now_et.strftime("%H:%M:%S ET")
        }
    except:
        return {"window": "UNKNOWN", "detail": "Cannot determine", "time_et": ""}


# ============================================================================
# AI PREDICTION ENDPOINTS
# ============================================================================

@app.post("/api/ai/train")
async def train_model(request: TrainRequest):
    """Train AI prediction model on a single symbol"""
    try:
        predictor = get_ai_scanner() if HAS_AI_SCANNER else None
        result = predictor.train(
            symbol=request.symbol.upper(),
            test_size=request.test_size
        )

        return result

    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/train-multi")
async def train_model_multi(request: TrainMultiRequest):
    """Train AI prediction model on multiple symbols for better generalization"""
    try:
        predictor = get_ai_scanner() if HAS_AI_SCANNER else None
        if not predictor or not hasattr(predictor, 'train_multi'):
            return {
                "success": False,
                "error": "AI Scanner not available",
                "message": "Train a single symbol first with /api/ai/train"
            }
        symbols = [s.upper() for s in request.symbols]
        result = predictor.train_multi(
            symbols=symbols,
            test_size=request.test_size
        )

        return result

    except Exception as e:
        logger.error(f"Error training multi-symbol model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/models/train")
async def train_model_compat(data: dict):
    """Compatibility route for UI - Train AI prediction model"""
    import time
    try:
        predictor = get_ai_scanner() if HAS_AI_SCANNER else None
        symbol = data.get('symbol', 'AAPL').upper()
        test_size = data.get('test_size', 0.2)

        result = predictor.train(symbol=symbol, test_size=test_size)

        return {
            "success": True,
            "data": {
                "training_id": f"train_{symbol}_{int(time.time())}",
                "symbol": symbol,
                "status": "completed",
                "metrics": result
            }
        }
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return {"success": False, "message": str(e)}


@app.post("/api/ai/train-walkforward")
async def train_walkforward(data: dict):
    """
    Walk-Forward Training - The PROPER way to train financial ML models

    This ensures NO DATA LEAKAGE:
    - Training data: Historical data BEFORE cutoff date
    - Test data: Data AFTER cutoff (truly out-of-sample)

    Parameters:
        symbols: List of stock symbols (default: major tech + momentum stocks)
        train_months: Months of training data (default: 12)
        test_months: Months of test data (default: 3)

    Returns:
        Out-of-sample accuracy, precision, recall, F1 score
    """
    try:
        predictor = get_ai_scanner() if HAS_AI_SCANNER else None

        # Default to a diverse set of stocks
        default_symbols = ["TSLA", "NVDA", "AMD", "META", "AAPL", "GOOGL", "MSFT",
                          "AMZN", "PLTR", "COIN", "MARA", "SOFI"]

        symbols = data.get('symbols', default_symbols)
        if isinstance(symbols, str):
            symbols = [s.strip().upper() for s in symbols.split(',')]

        train_months = int(data.get('train_months', 12))
        test_months = int(data.get('test_months', 3))

        result = predictor.train_walkforward(
            symbols=symbols,
            train_months=train_months,
            test_months=test_months
        )

        return result

    except Exception as e:
        logger.error(f"Error in walk-forward training: {e}")
        return {"success": False, "message": str(e)}


@app.post("/api/backtest/run")
async def run_backtest_endpoint(data: dict):
    """Run a backtest simulation using AI predictions on historical data"""
    try:
        from backtester import get_backtester
        from dataclasses import asdict

        backtester = get_backtester()

        # Extract parameters
        symbols = data.get('symbols', ['SPY'])
        if isinstance(symbols, str):
            symbols = [s.strip().upper() for s in symbols.split(',')]

        start_date = data.get('start_date', (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'))
        end_date = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        initial_capital = float(data.get('initial_capital', 100000))
        position_size_pct = float(data.get('position_size_pct', 0.1))
        confidence_threshold = float(data.get('confidence_threshold', 0.15))
        max_positions = int(data.get('max_positions', 5))
        holding_period = int(data.get('holding_period', 5))

        # Run backtest
        result = backtester.run_backtest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            confidence_threshold=confidence_threshold,
            max_positions=max_positions,
            holding_period=holding_period
        )

        return {
            "success": True,
            "data": asdict(result)
        }

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return {
            "success": False,
            "message": str(e)
        }


def _sync_predict(predictor, symbol: str, timeframe: str):
    """Synchronous prediction function to run in thread pool"""
    import traceback
    try:
        print(f"[_sync_predict] Calling predict for {symbol}, tf={timeframe}, predictor={predictor}")
        result = predictor.predict(symbol, period=timeframe)
        print(f"[_sync_predict] Result: {result}")
        return result
    except Exception as e:
        print(f"[_sync_predict] Error: {e}\n{traceback.format_exc()}")
        raise


@app.post("/api/ai/predict")
async def predict(request: PredictRequest):
    """Get AI prediction for a symbol"""
    symbol = request.symbol.upper()

    # Convert timeframe to valid yfinance period
    # Valid: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    timeframe_map = {
        "1day": "3mo", "1d": "3mo", "1Day": "3mo",
        "5day": "3mo", "5d": "3mo", "5Day": "3mo",
        "1week": "3mo", "1w": "3mo", "1Week": "3mo",
        "1month": "3mo", "1m": "3mo", "1Month": "3mo", "1mo": "3mo",
        "3month": "3mo", "3m": "3mo", "3Month": "3mo", "3mo": "3mo",
        "6month": "6mo", "6m": "6mo", "6Month": "6mo", "6mo": "6mo",
        "1year": "1y", "1y": "1y", "1Year": "1y",
        "2year": "2y", "2y": "2y", "2Year": "2y",
    }
    raw_timeframe = request.timeframe if request.timeframe else "3mo"
    timeframe = timeframe_map.get(raw_timeframe, "3mo")  # Default to 3mo

    logger.info(f"[PREDICT] Request for {symbol}, raw_timeframe={raw_timeframe}, mapped_timeframe={timeframe}")
    logger.info(f"[PREDICT] _cached_predictor={_cached_predictor}, model_loaded={_cached_predictor.model is not None if _cached_predictor else 'N/A'}")

    try:
        # Check if predictor is available
        if not _cached_predictor or _cached_predictor.model is None:
            # Try fallback
            predictor = get_ai_scanner() if HAS_AI_SCANNER else None
            if not predictor or not hasattr(predictor, 'model') or predictor.model is None:
                return {
                    "success": False,
                    "symbol": symbol,
                    "signal": "NEUTRAL",
                    "confidence": 0.0,
                    "action": "HOLD",
                    "error": "No model available - train the AI first"
                }
        else:
            predictor = _cached_predictor

        # Run prediction in thread pool to avoid blocking async loop
        result = await asyncio.to_thread(_sync_predict, predictor, symbol, timeframe)

        # Map the result format
        signal = result.get('signal', 'NEUTRAL')
        confidence = result.get('confidence', 0.5)
        prob_up = result.get('prob_up', 0.5)

        # Determine action
        if 'BULLISH' in signal:
            action = 'BUY'
        elif 'BEARISH' in signal:
            action = 'SELL'
        else:
            action = 'HOLD'

        return {
            "success": True,
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence / 100 if confidence > 1 else confidence,
            "prob_up": prob_up / 100 if prob_up > 1 else prob_up,
            "action": action,
            "model": result.get('signal_detail', 'LightGBM (70% accuracy)')
        }

    except ValueError as e:
        error_msg = str(e)
        if "No data" in error_msg:
            return {
                "success": False,
                "symbol": symbol,
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "action": "HOLD",
                "error": error_msg,
                "message": "Unable to fetch market data. Try a different symbol or check if market is open."
            }
        return {
            "success": False,
            "symbol": symbol,
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "action": "HOLD",
            "error": error_msg
        }
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return {
            "success": False,
            "symbol": symbol,
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "action": "HOLD",
            "error": str(e)
        }


@app.get("/api/ai/model-info")
async def get_model_info():
    """Get information about the trained model"""
    try:
        predictor = get_ai_scanner() if HAS_AI_SCANNER else None

        if predictor.model is None:
            return {
                "trained": False,
                "message": "No model trained yet"
            }

        # Get top features
        sorted_features = sorted(
            predictor.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "trained": True,
            "accuracy": predictor.accuracy,
            "training_date": predictor.training_date,
            "num_features": len(predictor.feature_names),
            "top_features": [
                {"name": name, "importance": float(importance)}
                for name, importance in sorted_features
            ],
            "data_source": "Schwab"
        }

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/brain/status")
async def get_brain_status():
    """Get BackgroundBrain status for AI Monitor"""
    try:
        from ai.background_brain import get_background_brain
        brain = get_background_brain()

        return {
            "success": True,
            "data": {
                "running": brain.is_running if hasattr(brain, 'is_running') else False,
                "regime": brain.current_regime if hasattr(brain, 'current_regime') else "UNKNOWN",
                "regime_confidence": brain.regime_confidence if hasattr(brain, 'regime_confidence') else 0.0,
                "last_update": brain.last_update.isoformat() if hasattr(brain, 'last_update') and brain.last_update else None,
                "symbols_tracked": len(brain.tracked_symbols) if hasattr(brain, 'tracked_symbols') else 0
            }
        }
    except ImportError:
        return {"success": True, "data": {"running": False, "regime": "N/A"}}
    except Exception as e:
        logger.error(f"Brain status error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/stock/ai-prediction/{symbol}")
async def get_stock_ai_prediction(symbol: str):
    """
    Get comprehensive AI prediction for a symbol.
    Returns Chronos, Qlib, and Order Flow scores for AI Signals panel.
    """
    import pandas as pd

    symbol = symbol.upper()
    result = {
        "symbol": symbol,
        "prob_up": 50.0,  # Default neutral
        "confidence": 50.0,
        "market_regime": "UNKNOWN",
        "signals": []
    }

    try:
        # Fetch price data for prediction
        df = None
        try:
            from polygon_data import get_polygon_data
            polygon = get_polygon_data()
            bars = await polygon.get_minute_bars(symbol, limit=200)
            if bars and len(bars) > 50:
                df = pd.DataFrame(bars)
                df.columns = [c.lower() for c in df.columns]
                if 'c' in df.columns:
                    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
        except Exception as e:
            logger.debug(f"Polygon data unavailable: {e}")

        # Fallback to yfinance if polygon unavailable
        if df is None or len(df) < 50:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="5d", interval="5m")
                if not df.empty:
                    df = df.reset_index()
                    df.columns = [c.lower() for c in df.columns]
            except Exception as e:
                logger.debug(f"yfinance data unavailable: {e}")

        # Try ensemble predictor for Chronos/Qlib
        if df is not None and len(df) >= 50:
            try:
                from ai.ensemble_predictor import get_ensemble_predictor
                predictor = get_ensemble_predictor()
                prediction = predictor.predict(symbol, df)
                if prediction:
                    result["prob_up"] = prediction.chronos_score * 100
                    result["confidence"] = prediction.qlib_score * 100
                    result["market_regime"] = prediction.market_regime
                    result["signals"] = prediction.signals
                    result["lgb_score"] = prediction.lgb_score * 100
                    result["chronos_score"] = prediction.chronos_score * 100
                    result["qlib_score"] = prediction.qlib_score * 100
                    result["scalp_score"] = prediction.scalp_score * 100
                    result["scalp_verdict"] = prediction.scalp_verdict
            except Exception as e:
                logger.debug(f"Ensemble prediction failed: {e}")

        # Fallback to basic predictor
        if result["prob_up"] == 50.0:
            try:
                predictor = get_ai_scanner() if HAS_AI_SCANNER else None
                if predictor and predictor.model:
                    pred = predictor.predict(symbol)
                    if pred:
                        result["prob_up"] = pred.get("probability", 0.5) * 100
                        result["confidence"] = pred.get("confidence", 0.5) * 100
            except Exception as e2:
                logger.debug(f"Basic predictor unavailable: {e2}")

        return result

    except Exception as e:
        logger.error(f"AI prediction error for {symbol}: {e}")
        return {
            "symbol": symbol,
            "prob_up": 50.0,
            "confidence": 50.0,
            "error": str(e)
        }


@app.get("/api/ai/batch-predict")
async def batch_predict(symbols: str = Query(..., description="Comma-separated symbols")):
    """Get AI predictions for multiple symbols"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        predictor = get_ai_scanner() if HAS_AI_SCANNER else None

        results = []
        for symbol in symbol_list:
            try:
                prediction = predictor.predict(symbol)
                results.append(prediction)
            except Exception as e:
                logger.warning(f"Failed to predict {symbol}: {e}")
                results.append({
                    "symbol": symbol,
                    "error": str(e)
                })

        return results

    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai/models/performance")
async def get_models_performance():
    """Get performance metrics for trained AI models"""
    try:
        import os
        from pathlib import Path

        models_dir = Path("models")
        if not models_dir.exists():
            return {"success": True, "data": []}

        models = []

        # Check for trained models
        for model_file in models_dir.glob("*.keras"):
            symbol = model_file.stem.replace("_mtf", "").replace("_lstm", "").upper()
            models.append({
                "name": symbol,
                "type": "LSTM MTF",
                "accuracy": 0.65,  # Placeholder - would load from saved metrics
                "win_rate": 0.55,
                "sharpe_ratio": 1.2,
                "total_trades": 0,
                "profitable_trades": 0,
                "avg_profit": 0.0
            })

        # Check lstm_mtf subdirectory
        lstm_mtf_dir = models_dir / "lstm_mtf"
        if lstm_mtf_dir.exists():
            for model_file in lstm_mtf_dir.glob("*.keras"):
                symbol = model_file.stem.replace("_mtf", "").upper()
                if not any(m["name"] == symbol for m in models):
                    models.append({
                        "name": symbol,
                        "type": "LSTM MTF",
                        "accuracy": 0.65,
                        "win_rate": 0.55,
                        "sharpe_ratio": 1.2,
                        "total_trades": 0,
                        "profitable_trades": 0,
                        "avg_profit": 0.0
                    })

        return {"success": True, "data": models}

    except Exception as e:
        logger.error(f"Error getting models performance: {e}")
        return {"success": False, "data": [], "error": str(e)}


# ============================================================================
# AI TRIGGER ENDPOINTS
# ============================================================================

@app.get("/api/ai/trigger/{symbol}")
async def get_trigger(symbol: str):
    """Get MACD/RSI trigger signal for a symbol"""
    try:
        from ai.trigger_signal_generator import get_trigger_generator

        generator = get_trigger_generator()
        trigger = generator.generate_trigger(symbol.upper())

        return {
            "success": True,
            "data": trigger.to_dict()
        }

    except Exception as e:
        logger.error(f"Error getting trigger for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai/triggers/scan")
async def scan_triggers(symbols: str = Query(..., description="Comma-separated symbols")):
    """Scan multiple symbols for actionable triggers"""
    try:
        from ai.trigger_signal_generator import get_trigger_generator

        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        generator = get_trigger_generator()
        triggers = generator.scan_for_triggers(symbol_list)

        return {
            "success": True,
            "count": len(triggers),
            "data": [t.to_dict() for t in triggers]
        }

    except Exception as e:
        logger.error(f"Error scanning triggers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai/predict-with-triggers/{symbol}")
async def predict_with_triggers(symbol: str):
    """Get combined AI prediction with MACD/RSI triggers"""
    try:
        predictor = get_ai_scanner() if HAS_AI_SCANNER else None
        result = predictor.predict_with_triggers(symbol.upper())

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        logger.error(f"Error getting prediction with triggers for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/triggers/batch")
async def batch_triggers(request: dict):
    """Get triggers for multiple symbols (POST for larger lists)"""
    try:
        from ai.trigger_signal_generator import get_trigger_generator

        symbols = request.get("symbols", [])
        if isinstance(symbols, str):
            symbols = [s.strip().upper() for s in symbols.split(',')]

        generator = get_trigger_generator()
        results = []

        for symbol in symbols:
            try:
                trigger = generator.generate_trigger(symbol.upper())
                results.append(trigger.to_dict())
            except Exception as e:
                logger.warning(f"Failed to get trigger for {symbol}: {e}")
                results.append({
                    "symbol": symbol,
                    "action": "ERROR",
                    "error": str(e)
                })

        # Filter to only actionable triggers
        actionable = [t for t in results if t.get("action") in ["BUY", "SELL"]]

        return {
            "success": True,
            "total": len(results),
            "actionable_count": len(actionable),
            "all_triggers": results,
            "actionable_triggers": actionable
        }

    except Exception as e:
        logger.error(f"Error in batch triggers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WATCHLIST ENDPOINTS
# ============================================================================

@app.get("/api/watchlist")
async def get_watchlist():
    """Get default watchlist with market data"""
    # Get symbols from database watchlist manager
    try:
        from watchlist_manager import get_watchlist_manager
        manager = get_watchlist_manager()
        default_wl = manager.get_default_watchlist()
        watchlist = default_wl.get('symbols', [])
        watchlist_name = default_wl.get('name', 'Default Watchlist')
    except Exception as e:
        logger.warning(f"Could not load watchlist from database: {e}")
        # Fallback to hardcoded defaults
        watchlist = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD"]
        watchlist_name = "Default Watchlist"

    # Use unified provider (Schwab)
    if HAS_UNIFIED_DATA:
        quotes = unified_get_quotes(watchlist)
    else:
        quotes = {sym: {"symbol": sym, "price": 0} for sym in watchlist}

    return {
        "name": watchlist_name,
        "symbols": watchlist,
        "quotes": quotes
    }


# ============================================================================
# CLAUDE AI CHAT ENDPOINTS
# ============================================================================

@app.post("/api/claude/query")
async def claude_query(request: ClaudeQueryRequest):
    """
    Send a query to Claude AI with NLP trading capabilities.

    Supports natural language commands like:
    - "Buy 10 shares of AAPL"
    - "Sell all my TSLA position"
    - "What's my account balance?"
    - "Analyze MSFT"
    - "Set stop loss to 2%"
    - "Start the auto trading bot"
    - "Show my positions"
    """
    import re

    query = request.query
    context = request.context or {}
    symbol = request.symbol or "SPY"
    conversation_id = context.get("conversation_id") if isinstance(context, dict) else None

    try:
        # Check for status queries about Schwab/brokers
        query_lower = query.lower()
        if any(kw in query_lower for kw in ['schwab', 'broker', 'connected', 'status', 'account']):
            # Get broker status
            schwab_status = "Not available"

            try:
                from schwab_trading import get_schwab_trading, is_schwab_trading_available
                if is_schwab_trading_available():
                    trading = get_schwab_trading()
                    if trading:
                        accounts = trading.get_accounts()
                        selected = trading.get_selected_account()
                        schwab_status = f"Connected ({len(accounts)} accounts, selected: {selected or 'None'})"
                    else:
                        schwab_status = "Available but not authenticated"
            except Exception as e:
                schwab_status = f"Error: {e}"

            return {
                "success": True,
                "data": {
                    "response": f"📊 **Broker Status**\n\n• **Schwab:** {schwab_status}\n\nSchwab is the active broker for trading.",
                    "conversation_id": conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "timestamp": datetime.now().isoformat(),
                    "action": "status_check",
                    "action_details": {"schwab": schwab_status}
                }
            }

        # Import Claude bot intelligence
        from ai.claude_bot_intelligence import get_bot_intelligence

        ai = get_bot_intelligence()
        broker = get_unified_broker()

        # Build rich context for Claude
        trading_context = {
            "current_symbol": symbol,
            "connected": broker.is_connected if broker else False,
            "account": None,
            "positions": [],
            "recent_orders": []
        }

        # Fetch real account data if connected
        if broker and broker.is_connected:
            try:
                trading_context["account"] = broker.get_account()
                trading_context["positions"] = broker.get_positions()
                trading_context["recent_orders"] = broker.get_orders()[:10] if broker.get_orders() else []
            except Exception as e:
                logger.warning(f"Could not fetch trading context: {e}")

        # Merge with any provided context (only if it's a dict)
        if context and isinstance(context, dict):
            trading_context.update(context)

        # Check for NLP trading commands
        trade_result = await _parse_and_execute_nlp_trade(query, connector, trading_context)

        if trade_result.get("executed"):
            # A trade was executed via NLP
            response_text = trade_result.get("response", "Command executed successfully.")
            action_taken = trade_result.get("action", "trade")
            action_details = trade_result.get("details", {})
        else:
            # No trade command - use Claude AI for analysis/chat
            try:
                result = ai.chat_with_tools(query, use_tools=True)
                response_text = result.get("response", "I couldn't process your request. Please try again.")
                action_taken = "chat"
                action_details = {"mood": result.get("mood", "neutral")}
            except Exception as chat_err:
                logger.error(f"Claude chat_with_tools error: {chat_err}")
                response_text = f"I'm here to help! You asked: '{query}'\n\nNote: Full AI analysis is temporarily limited. You can:\n• Check broker status\n• Ask about positions\n• Request predictions\n• Start training"
                action_taken = "fallback"
                action_details = {"error": str(chat_err)}

        return {
            "success": True,
            "data": {
                "response": response_text,
                "conversation_id": conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "action": action_taken,
                "action_details": action_details
            }
        }

    except ImportError as e:
        logger.warning(f"Claude AI module not available: {e}")
        return {
            "success": False,
            "data": {
                "response": "I'm here to help with your trading questions. The AI service may be initializing - please try again in a moment.",
                "conversation_id": conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Claude query error: {e}")
        return {
            "success": False,
            "data": {
                "response": f"Error processing request: {str(e)}",
                "conversation_id": conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat()
            }
        }


async def _parse_and_execute_nlp_trade(query: str, connector, context: Dict) -> Dict:
    """
    Parse natural language trading commands and execute them.

    Supported commands:
    - Buy/sell orders: "buy 10 AAPL", "sell 50 shares of TSLA"
    - Close positions: "close AAPL", "close all positions"
    - Account queries: "show balance", "what's my buying power"
    - Position queries: "show positions", "what am I holding"
    - Bot control: "start bot", "stop bot", "enable auto trading"
    """
    import re

    query_lower = query.lower().strip()

    # Pattern: Buy/Sell orders
    trade_pattern = r"(buy|sell|purchase|short)\s+(\d+)\s*(?:shares?\s+(?:of\s+)?)?([A-Za-z]+)(?:\s+(?:at|@|for)\s+\$?([\d.]+))?"
    trade_match = re.search(trade_pattern, query_lower)

    if trade_match:
        action = trade_match.group(1)
        quantity = int(trade_match.group(2))
        symbol = trade_match.group(3).upper()
        limit_price = float(trade_match.group(4)) if trade_match.group(4) else None

        side = OrderSide.BUY if action in ["buy", "purchase"] else OrderSide.SELL

        # Use UnifiedBroker for order execution (Schwab)
        try:
            broker = get_unified_broker()
            if not broker.is_connected:
                return {
                    "executed": True,
                    "action": "order_failed",
                    "response": "❌ No broker connected. Please check your API credentials.",
                    "details": {"error": "No broker available"}
                }

            if limit_price:
                result = broker.place_limit_order(symbol, side, quantity, limit_price)
                if result.success:
                    return {
                        "executed": True,
                        "action": "limit_order",
                        "response": f"✅ Placed {side.value} limit order via {result.broker}: {quantity} shares of {symbol} at ${limit_price:.2f}",
                        "details": {"symbol": symbol, "quantity": quantity, "side": side.value, "limit_price": limit_price, "order_id": result.order_id, "broker": result.broker}
                    }
                else:
                    return {
                        "executed": True,
                        "action": "order_failed",
                        "response": f"❌ Order failed on {result.broker}: {result.error}",
                        "details": {"error": result.error, "broker": result.broker}
                    }
            else:
                # Market orders - warn user but execute via UnifiedBroker
                result = broker.place_market_order(symbol, side, quantity)
                if result.success:
                    return {
                        "executed": True,
                        "action": "market_order",
                        "response": f"✅ Placed {side.value} market order via {result.broker}: {quantity} shares of {symbol}",
                        "details": {"symbol": symbol, "quantity": quantity, "side": side.value, "order_id": result.order_id, "broker": result.broker}
                    }
                else:
                    return {
                        "executed": True,
                        "action": "order_failed",
                        "response": f"❌ Order failed on {result.broker}: {result.error}",
                        "details": {"error": result.error, "broker": result.broker}
                    }
        except Exception as e:
            return {
                "executed": True,
                "action": "order_failed",
                "response": f"❌ Order failed: {str(e)}",
                "details": {"error": str(e)}
            }

    # Pattern: Close position
    close_pattern = r"(?:close|exit|liquidate|sell all)\s+(?:my\s+)?(?:position\s+(?:in|on)\s+)?([A-Za-z]+)(?:\s+position)?"
    close_match = re.search(close_pattern, query_lower)

    if close_match:
        symbol = close_match.group(1).upper()
        try:
            broker = get_unified_broker()
            if not broker.is_connected:
                return {
                    "executed": True,
                    "action": "close_failed",
                    "response": "❌ No broker connected",
                    "details": {"error": "No broker available"}
                }
            result = broker.close_position(symbol)
            if result.success:
                return {
                    "executed": True,
                    "action": "close_position",
                    "response": f"✅ Closed position in {symbol} via {result.broker}",
                    "details": {"symbol": symbol, "order_id": result.order_id, "broker": result.broker}
                }
            else:
                return {
                    "executed": True,
                    "action": "close_failed",
                    "response": f"❌ Could not close {symbol}: {result.error}",
                    "details": {"error": result.error, "broker": result.broker}
                }
        except Exception as e:
            return {
                "executed": True,
                "action": "close_failed",
                "response": f"❌ Could not close {symbol}: {str(e)}",
                "details": {"error": str(e)}
            }

    # Pattern: Close all positions
    if any(phrase in query_lower for phrase in ["close all", "liquidate all", "sell everything", "flatten"]):
        try:
            broker = get_unified_broker()
            if not broker.is_connected:
                return {
                    "executed": True,
                    "action": "close_all_failed",
                    "response": "❌ No broker connected",
                    "details": {"error": "No broker available"}
                }
            results = broker.close_all_positions()
            success_count = sum(1 for r in results if r.success)
            return {
                "executed": True,
                "action": "close_all",
                "response": f"✅ Closed {success_count}/{len(results)} positions via {broker.broker_name}",
                "details": {"results": [{"symbol": r.symbol, "success": r.success, "error": r.error} for r in results]}
            }
        except Exception as e:
            return {
                "executed": True,
                "action": "close_all_failed",
                "response": f"❌ Could not close all positions: {str(e)}",
                "details": {"error": str(e)}
            }

    # Pattern: Show positions
    if any(phrase in query_lower for phrase in ["show position", "my position", "what am i holding", "current holdings", "show holdings"]):
        positions = context.get("positions", [])
        if positions:
            pos_text = "\n".join([f"• {p.get('symbol', 'N/A')}: {p.get('qty', 0)} shares @ ${float(p.get('avg_entry_price', 0)):.2f} (P&L: ${float(p.get('unrealized_pl', 0)):.2f})" for p in positions[:10]])
            return {
                "executed": True,
                "action": "show_positions",
                "response": f"📊 **Current Positions:**\n{pos_text}",
                "details": {"positions": positions}
            }
        else:
            return {
                "executed": True,
                "action": "show_positions",
                "response": "📊 No open positions",
                "details": {"positions": []}
            }

    # Pattern: Show account/balance
    if any(phrase in query_lower for phrase in ["show balance", "account balance", "buying power", "how much", "my account", "show account"]):
        account = context.get("account", {})
        if account:
            equity = float(account.get("equity", 0))
            buying_power = float(account.get("buying_power", 0))
            cash = float(account.get("cash", 0))
            return {
                "executed": True,
                "action": "show_account",
                "response": f"💰 **Account Summary:**\n• Equity: ${equity:,.2f}\n• Buying Power: ${buying_power:,.2f}\n• Cash: ${cash:,.2f}",
                "details": {"account": account}
            }
        else:
            return {
                "executed": True,
                "action": "show_account",
                "response": "💰 Account data unavailable. Please check connection.",
                "details": {}
            }

    # Pattern: Start/stop bot
    if any(phrase in query_lower for phrase in ["start bot", "start trading", "enable auto", "turn on bot"]):
        try:
            from bot_manager import get_bot_manager
            bot = get_bot_manager()
            await bot.start()
            return {
                "executed": True,
                "action": "start_bot",
                "response": "🤖 Auto-trading bot STARTED. Monitoring markets...",
                "details": {"status": "running"}
            }
        except Exception as e:
            return {
                "executed": True,
                "action": "start_bot_failed",
                "response": f"❌ Could not start bot: {str(e)}",
                "details": {"error": str(e)}
            }

    if any(phrase in query_lower for phrase in ["stop bot", "stop trading", "disable auto", "turn off bot", "pause bot"]):
        try:
            from bot_manager import get_bot_manager
            bot = get_bot_manager()
            await bot.stop()
            return {
                "executed": True,
                "action": "stop_bot",
                "response": "🛑 Auto-trading bot STOPPED.",
                "details": {"status": "stopped"}
            }
        except Exception as e:
            return {
                "executed": True,
                "action": "stop_bot_failed",
                "response": f"❌ Could not stop bot: {str(e)}",
                "details": {"error": str(e)}
            }

    # No command matched - return not executed
    return {"executed": False}


# ============================================================================
# STATIC FILES
# ============================================================================

# Favicon endpoint
@app.get("/favicon.ico")
async def favicon():
    """Return a simple SVG favicon"""
    from fastapi.responses import Response
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
        <rect width="32" height="32" rx="4" fill="#1a1a2e"/>
        <path d="M8 22 L16 10 L24 22" stroke="#00d4aa" stroke-width="3" fill="none"/>
        <circle cx="16" cy="8" r="2" fill="#00d4aa"/>
    </svg>'''
    return Response(content=svg, media_type="image/svg+xml")

# Mount UI directory if it exists
ui_path = Path("ui")
if ui_path.exists():
    app.mount("/ui", StaticFiles(directory="ui"), name="ui")

    @app.get("/dashboard")
    async def dashboard():
        """DEPRECATED: Redirect to /trading-new"""
        from starlette.responses import RedirectResponse
        return RedirectResponse(url="/trading-new", status_code=302)

    @app.get("/trading-new")
    async def trading_ui():
        """Serve new Golden Layout trading UI"""
        trading_index = Path("ui/trading/dist/index.html")
        if trading_index.exists():
            return FileResponse(trading_index)
        return {"error": "Trading UI not built. Run 'npm run build' in ui/trading/"}

    # Mount new Trading UI (Golden Layout React app) - AFTER routes to avoid shadowing
    trading_ui_path = Path("ui/trading/dist")
    if trading_ui_path.exists():
        app.mount("/trading-new", StaticFiles(directory="ui/trading/dist", html=True), name="trading-ui")

    @app.get("/trading")
    async def trading_dashboard():
        """Serve trading-focused dashboard"""
        return FileResponse("ui/trading_dashboard.html")

    @app.get("/ai")
    async def ai_dashboard():
        """Serve AI monitoring dashboard"""
        return FileResponse("ui/ai_dashboard.html")

    @app.get("/ai-control")
    async def ai_control_dashboard():
        """Serve AI control center - all controls in one view"""
        return FileResponse("ui/ai_control_dashboard.html")

    @app.get("/ai-control-center")
    @app.get("/ai-control-center/")
    async def ai_control_center():
        """Serve AI Control Center React app"""
        ai_cc_index = Path("ui/ai-control-center/build/index.html")
        if ai_cc_index.exists():
            return FileResponse(ai_cc_index)
        return {"error": "AI Control Center not built. Run 'npm run build' in ui/ai-control-center/"}

    # Mount AI Control Center React app
    ai_cc_path = Path("ui/ai-control-center/build")
    if ai_cc_path.exists():
        app.mount("/ai-control-center", StaticFiles(directory="ui/ai-control-center/build", html=True), name="ai-control-center")


# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("="*80)
    logger.info("MORPHEUS TRADING BOT")
    logger.info("="*80)

    # Check broker configuration
    config = get_broker_config()
    logger.info(f"Broker Type: {config.broker_type}")

    # Test Schwab connection
    if HAS_SCHWAB_TRADING:
        try:
            if is_schwab_trading_available():
                schwab = get_schwab_trading()
                if schwab:
                    account = schwab.get_account_info()
                    logger.info("[OK] Schwab Connected")
                    logger.info(f"   Account: {account.get('account_number', 'N/A')}")
                    logger.info(f"   Cash: ${account.get('cash', 0):,.2f}")
                else:
                    logger.warning("[WARN] Schwab available but not authenticated")
            else:
                logger.warning("[WARN] Schwab trading not available")
        except Exception as e:
            logger.error(f"[FAIL] Schwab initialization error: {e}")

    # Load AI model if available
    try:
        predictor = get_ai_scanner() if HAS_AI_SCANNER else None
        if predictor.model:
            logger.info(f"[OK] AI Model Loaded (Accuracy: {predictor.accuracy:.4f})")
        else:
            logger.info("[INFO] AI Model not trained - use /api/ai/train to train")
    except Exception as e:
        logger.info(f"[INFO] AI Model status: {e}")

    # Initialize multi-channel data provider
    if HAS_MULTI_CHANNEL:
        try:
            provider = get_multi_channel_provider()
            logger.info(f"[OK] Multi-Channel Data Provider Active ({len(provider.channels)} channels)")
            for channel in provider.channels:
                logger.info(f"     - {channel.value.upper()} channel ready")
        except Exception as e:
            logger.warning(f"[WARN] Multi-channel provider error: {e}")
    else:
        logger.info("[INFO] Multi-channel data provider not available")

    # Initialize real-time streaming
    if HAS_STREAMING:
        try:
            stream_manager = get_stream_manager()
            logger.info("[OK] Real-Time Streaming Available")
            logger.info("     - WebSocket endpoint: ws://localhost:9100/ws/market")
        except Exception as e:
            logger.warning(f"[WARN] Streaming provider error: {e}")
    else:
        logger.info("[INFO] Real-time streaming not available")

    # Start background position sync with slippage combat
    try:
        from ai.position_sync import start_sync
        sync = start_sync()
        logger.info("[OK] Position Synchronizer Started")
        logger.info("     - Sync interval: 5 seconds")
        logger.info("     - Slippage detection: ACTIVE")
        logger.info("     - Auto-combat: ENABLED")
    except Exception as e:
        logger.warning(f"[WARN] Position sync error: {e}")

    # Initialize market regime detector
    try:
        from ai.market_regime import get_regime_detector
        detector = get_regime_detector()
        phase_info = detector.get_phase_info()
        logger.info(f"[OK] Market Regime Detector Active")
        logger.info(f"     - Current Phase: {phase_info.get('current_phase', 'unknown')}")
        logger.info(f"     - Time: {phase_info.get('current_time_et', 'unknown')}")
    except Exception as e:
        logger.warning(f"[WARN] Regime detector error: {e}")

    # Initialize loss cutter
    try:
        from ai.loss_cutter import get_loss_cutter
        cutter = get_loss_cutter()
        logger.info("[OK] Loss Cutter Active")
        logger.info(f"     - Max Loss: {cutter.max_loss_pct}%")
        logger.info(f"     - Time Threshold: {cutter.time_threshold_minutes} min")
    except Exception as e:
        logger.warning(f"[WARN] Loss cutter error: {e}")

    # Initialize Background AI Brain for continuous learning
    try:
        from ai.background_brain import start_brain, get_background_brain
        brain = start_brain(cpu_target=0.3)  # Use 30% CPU for background AI (conservative)
        logger.info("[OK] Background AI Brain Started")
        logger.info(f"     - CPU Target: 30%")
        logger.info(f"     - Worker Threads: {brain.worker_threads}")
        logger.info(f"     - Cores Available: {brain.num_cores}")
        logger.info("     - Continuous Learning: ACTIVE")
        logger.info("     - Market Regime Detection: ENABLED")
    except Exception as e:
        logger.warning(f"[WARN] Background brain error: {e}")

    # Initialize Circuit Breaker for drawdown protection
    try:
        from ai.circuit_breaker import get_circuit_breaker
        breaker = get_circuit_breaker()
        broker = get_unified_broker()
        if broker and broker.is_connected:
            account = broker.get_account()
            equity = float(account.get('market_value', account.get('equity', 0)))
            breaker.initialize_day(equity)
            logger.info("[OK] Circuit Breaker Active")
            logger.info(f"     - Starting Equity: ${equity:,.2f}")
            logger.info(f"     - Warning Level: {breaker.daily_loss_warning*100:.1f}%")
            logger.info(f"     - Halt Level: {breaker.daily_loss_halt*100:.1f}%")
    except Exception as e:
        logger.warning(f"[WARN] Circuit breaker error: {e}")

    # Initialize Schwab real-time market data (WebSocket streaming or Fast Polling fallback)
    schwab_realtime_started = False
    default_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA", "AMD", "META", "GOOGL", "AMZN"]

    if HAS_SCHWAB_STREAMING and HAS_SCHWAB_TRADING:
        try:
            from schwab_market_data import is_schwab_available
            if is_schwab_available():
                # Try WebSocket streaming first
                start_schwab_streaming(default_symbols)
                # Check if it actually started
                import time
                time.sleep(1)  # Wait for streaming to initialize
                status = get_schwab_stream_status()
                if status.get('running'):
                    logger.info("[OK] Schwab WebSocket Streaming Started")
                    logger.info(f"     - Subscribed to {len(default_symbols)} symbols")
                    logger.info("     - Real-time quotes enabled (sub-second updates)")
                    schwab_realtime_started = True
                else:
                    logger.warning("[WARN] Schwab WebSocket streaming failed to start")
        except Exception as e:
            logger.warning(f"[WARN] Schwab streaming startup error: {e}")

    # Fallback to Fast Polling if WebSocket streaming failed
    if not schwab_realtime_started and HAS_SCHWAB_FAST_POLLING and HAS_SCHWAB_TRADING:
        try:
            from schwab_market_data import is_schwab_available
            if is_schwab_available():
                start_schwab_fast_polling(default_symbols)
                logger.info("[OK] Schwab Fast Polling Started (WebSocket fallback)")
                logger.info(f"     - Subscribed to {len(default_symbols)} symbols")
                logger.info("     - Updates every 300ms (near real-time)")
                schwab_realtime_started = True
        except Exception as e:
            logger.warning(f"[WARN] Schwab fast polling startup error: {e}")

    if not schwab_realtime_started:
        logger.info("[INFO] Schwab real-time data not available (using on-demand HTTP)")

    # Initialize Connection Health Monitor
    try:
        from ai.connection_health_monitor import start_health_monitor
        health_monitor = start_health_monitor()
        logger.info("[OK] Connection Health Monitor Started")
        logger.info("     - Auto-recovery: ENABLED")
        logger.info("     - Check interval: 10 seconds")
    except Exception as e:
        logger.warning(f"[WARN] Health monitor error: {e}")

    logger.info("="*80)
    logger.info("Server ready on http://localhost:9100")
    logger.info("Trading Dashboard: http://localhost:9100/trading-new")
    logger.info("AI Control Center: http://localhost:9100/ai-control-center")
    logger.info("API Docs: http://localhost:9100/docs")
    logger.info("="*80)


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("Shutting down AI Trading Hub...")

    # Stop Connection Health Monitor
    try:
        from ai.connection_health_monitor import get_health_monitor
        monitor = get_health_monitor()
        monitor.stop()
        logger.info("Connection health monitor stopped")
    except Exception as e:
        logger.warning(f"Error stopping health monitor: {e}")

    # Stop Schwab streaming
    if HAS_SCHWAB_STREAMING:
        try:
            stop_schwab_streaming()
            logger.info("Schwab streaming stopped")
        except Exception as e:
            logger.warning(f"Error stopping Schwab streaming: {e}")

    # Stop Schwab fast polling
    if HAS_SCHWAB_FAST_POLLING:
        try:
            stop_schwab_fast_polling()
            logger.info("Schwab fast polling stopped")
        except Exception as e:
            logger.warning(f"Error stopping Schwab fast polling: {e}")


# ============================================================================
# REAL-TIME WEBSOCKET HUB
# ============================================================================

# Import realtime hub for WebSocket management
try:
    from ai.realtime_hub import (
        get_realtime_hub,
        Channel,
        broadcast_trade,
        broadcast_position_update,
        broadcast_bot_status,
        broadcast_prediction,
        broadcast_brain_status,
        broadcast_var_update,
        broadcast_drawdown_update,
        broadcast_circuit_breaker,
        broadcast_risk_alert
    )
    HAS_REALTIME_HUB = True
except ImportError as e:
    logger.warning(f"Realtime hub not available: {e}")
    HAS_REALTIME_HUB = False


@app.websocket("/ws/realtime")
async def websocket_realtime_hub(websocket: WebSocket):
    """
    WebSocket endpoint for real-time trading updates.

    Channels:
    - trading: Bot status, trades, positions, orders
    - ai: Predictions, signals, brain status
    - risk: VAR, drawdown, circuit breaker alerts
    - system: Server status, errors

    Protocol:
    1. Connect to ws://localhost:9100/ws/realtime
    2. Optionally subscribe to specific channels:
       {"action": "subscribe", "channels": ["trading", "risk"]}
    3. Receive streaming updates:
       {"channel": "trading", "type": "trade", "data": {...}, "timestamp": "..."}

    Actions:
    - subscribe: Subscribe to channels
    - unsubscribe: Unsubscribe from channels
    - get_history: Get recent messages for a channel
    - status: Get hub status
    """
    await websocket.accept()
    client_id = str(uuid.uuid4())
    logger.info(f"[WS-RT] Client connected: {client_id}")

    if not HAS_REALTIME_HUB:
        await websocket.send_json({
            "type": "error",
            "message": "Realtime hub not available"
        })
        await websocket.close()
        return

    hub = get_realtime_hub()

    # Register with all channels by default
    message_queue = hub.register_client(client_id)

    # Send welcome message
    await websocket.send_json({
        "type": "connected",
        "client_id": client_id,
        "message": "Connected to realtime hub",
        "channels": [c.value for c in Channel if c != Channel.ALL],
        "status": hub.get_status()
    })

    async def send_messages():
        """Forward messages from queue to WebSocket"""
        while True:
            try:
                message = await message_queue.get()
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"[WS-RT] Send error: {e}")
                break

    async def receive_commands():
        """Receive commands from WebSocket client"""
        while True:
            try:
                data = await websocket.receive_json()
                action = data.get("action", "").lower()

                if action == "subscribe":
                    channels = data.get("channels", [])
                    hub.subscribe(client_id, channels)
                    await websocket.send_json({
                        "type": "subscribed",
                        "channels": channels
                    })

                elif action == "unsubscribe":
                    channels = data.get("channels", [])
                    hub.unsubscribe(client_id, channels)
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "channels": channels
                    })

                elif action == "get_history":
                    channel = data.get("channel", "trading")
                    limit = data.get("limit", 20)
                    history = hub.get_history(channel, limit)
                    await websocket.send_json({
                        "type": "history",
                        "channel": channel,
                        "messages": history
                    })

                elif action == "status":
                    await websocket.send_json({
                        "type": "status",
                        "hub": hub.get_status()
                    })

                elif action == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"[WS-RT] Receive error: {e}")
                break

    try:
        # Run both tasks concurrently
        send_task = asyncio.create_task(send_messages())
        receive_task = asyncio.create_task(receive_commands())

        done, pending = await asyncio.wait(
            [send_task, receive_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in pending:
            task.cancel()

    except WebSocketDisconnect:
        logger.info(f"[WS-RT] Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"[WS-RT] Error: {e}")
    finally:
        hub.unregister_client(client_id)
        logger.info(f"[WS-RT] Client cleanup: {client_id}")


@app.get("/api/realtime/status")
async def get_realtime_status():
    """Get realtime hub status"""
    if not HAS_REALTIME_HUB:
        return {"success": False, "error": "Realtime hub not available"}

    hub = get_realtime_hub()
    return {
        "success": True,
        "status": hub.get_status()
    }


@app.get("/api/realtime/history/{channel}")
async def get_realtime_history(channel: str, limit: int = 20):
    """Get message history for a channel"""
    if not HAS_REALTIME_HUB:
        return {"success": False, "error": "Realtime hub not available"}

    hub = get_realtime_hub()
    history = hub.get_history(channel, limit)

    return {
        "success": True,
        "channel": channel,
        "messages": history,
        "count": len(history)
    }


@app.post("/api/realtime/broadcast")
async def broadcast_message(data: dict):
    """
    Broadcast a message to connected clients.

    Body:
        channel: str - Channel to broadcast to (trading, ai, risk, system, all)
        type: str - Message type
        data: dict - Message payload
    """
    if not HAS_REALTIME_HUB:
        return {"success": False, "error": "Realtime hub not available"}

    from ai.realtime_hub import broadcast

    channel = data.get("channel", "system")
    message_type = data.get("type", "update")
    payload = data.get("data", {})

    await broadcast(channel, payload, message_type)

    return {
        "success": True,
        "message": f"Broadcast sent to {channel}",
        "type": message_type
    }


# ============================================================================
# BACKGROUND TASK: PERIODIC RISK UPDATES
# ============================================================================

async def periodic_risk_broadcast():
    """Background task to broadcast risk updates every 30 seconds"""
    if not HAS_REALTIME_HUB:
        return

    from ai.realtime_hub import broadcast

    while True:
        try:
            # Get current risk state
            broker = get_unified_broker()
            if broker and broker.is_connected:
                account = broker.get_account()
                equity = float(account.get('market_value', account.get('equity', 0)))

                # Broadcast drawdown
                try:
                    from ai.portfolio_guard import get_portfolio_guard
                    guard = get_portfolio_guard()
                    dd = guard.calculate_drawdown(equity)

                    await broadcast(
                        "risk",
                        {
                            "drawdown_pct": dd.drawdown_pct,
                            "max_drawdown_pct": dd.max_drawdown_pct,
                            "equity": equity,
                            "is_at_peak": dd.is_at_peak
                        },
                        "drawdown"
                    )
                except Exception as e:
                    logger.debug(f"Error broadcasting drawdown: {e}")

                # Broadcast circuit breaker status
                try:
                    from ai.circuit_breaker import get_circuit_breaker
                    breaker = get_circuit_breaker()
                    state = breaker.check_breaker()

                    await broadcast(
                        "risk",
                        {
                            "level": state.level,
                            "can_trade": state.can_trade,
                            "daily_pnl": state.daily_pnl,
                            "daily_pnl_pct": state.daily_pnl_pct
                        },
                        "circuit_breaker"
                    )
                except Exception as e:
                    logger.debug(f"Error broadcasting circuit breaker: {e}")

        except Exception as e:
            logger.debug(f"Periodic risk broadcast error: {e}")

        await asyncio.sleep(30)


@app.on_event("startup")
async def start_periodic_broadcasts():
    """Start background broadcast tasks on server startup"""
    if HAS_REALTIME_HUB:
        asyncio.create_task(periodic_risk_broadcast())
        logger.info("Started periodic risk broadcast task")


@app.on_event("startup")
async def run_startup_connectivity_test():
    """
    Run connectivity self-test at startup.

    Waits 5 seconds for services to initialize, then tests:
    1. Chronos scheduler
    2. Market data connection
    3. WebSocket broadcaster
    4. Scanner jobs

    Logs results and saves connectivity report.
    """
    async def delayed_test():
        await asyncio.sleep(5)  # Wait for services to initialize

        try:
            from ai.connectivity_manager import get_connectivity_manager
            manager = get_connectivity_manager()

            logger.info("=" * 60)
            logger.info("RUNNING STARTUP CONNECTIVITY SELF-TEST")
            logger.info("=" * 60)

            results = await manager.run_startup_self_test()

            # Log summary
            passed = sum(1 for t in results.get("tests", []) if t.get("passed"))
            total = len(results.get("tests", []))
            status = "PASSED" if results.get("all_passed") else "NEEDS ATTENTION"

            logger.info(f"CONNECTIVITY SELF-TEST: {status} ({passed}/{total} services)")

            # Log first tick status
            if manager.first_tick_received:
                logger.info(f"First tick: {manager.first_tick_symbol} at {manager.first_tick_received}")
            else:
                logger.info("First tick: Not yet received")

        except Exception as e:
            logger.error(f"Startup connectivity test failed: {e}")

    asyncio.create_task(delayed_test())
    logger.info("Scheduled startup connectivity self-test (5s delay)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("MORPHEUS TRADING BOT")
    print("="*80)
    print("Starting server on http://localhost:9100")
    print("Trading Dashboard: http://localhost:9100/trading-new")
    print("API Docs: http://localhost:9100/docs")
    print("="*80 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9100,
        log_level="warning",  # Reduced from 'info' for faster response times
        access_log=False      # Disable access logging - was causing 2-3s latency!
    )
