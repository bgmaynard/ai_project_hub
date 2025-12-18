"""
Trading Pipeline & Coach API Routes
====================================
REST API endpoints for the automated trading pipeline and coaching system.

Endpoints:
- /api/pipeline/* - Trading pipeline control and monitoring
- /api/coach/* - Trading coach and emotional trading analysis
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import json
import math
from datetime import datetime

logger = logging.getLogger(__name__)


def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize object for JSON serialization (handle inf/nan)"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isinf(obj):
            return 999.99 if obj > 0 else -999.99
        elif math.isnan(obj):
            return 0.0
        return obj
    return obj

router = APIRouter(tags=["Trading Pipeline & Coach"])

# ============================================================================
# TRADING PIPELINE
# ============================================================================

try:
    from ai.trading_pipeline import get_trading_pipeline, PipelineConfig
    HAS_PIPELINE = True
except ImportError as e:
    logger.warning(f"Trading pipeline not available: {e}")
    HAS_PIPELINE = False

# ============================================================================
# TRADING COACH
# ============================================================================

try:
    from ai.trading_coach import get_trading_coach
    HAS_COACH = True
except ImportError as e:
    logger.warning(f"Trading coach not available: {e}")
    HAS_COACH = False

# ============================================================================
# AI WATCHLIST TRADER
# ============================================================================

try:
    from ai.ai_watchlist_trader import get_ai_watchlist_trader
    HAS_AI_TRADER = True
except ImportError as e:
    logger.warning(f"AI watchlist trader not available: {e}")
    HAS_AI_TRADER = False

# ============================================================================
# TRADE JOURNAL (direct access)
# ============================================================================

try:
    from ai.trade_journal import get_trade_journal
    HAS_JOURNAL = True
except ImportError as e:
    logger.warning(f"Trade journal not available: {e}")
    HAS_JOURNAL = False


# ============================================================================
# REQUEST MODELS
# ============================================================================

class PipelineConfigRequest(BaseModel):
    auto_add_to_watchlist: Optional[bool] = None
    min_scanner_score: Optional[float] = None
    max_watchlist_size: Optional[int] = None
    auto_analyze_on_add: Optional[bool] = None
    min_ai_confidence: Optional[float] = None
    auto_execute: Optional[bool] = None
    paper_mode: Optional[bool] = None
    scanner_interval: Optional[int] = None
    analysis_interval: Optional[int] = None
    execution_interval: Optional[int] = None


class AITraderConfigRequest(BaseModel):
    enabled: Optional[bool] = None
    paper_mode: Optional[bool] = None
    min_confidence: Optional[float] = None
    max_position_value: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    max_daily_trades: Optional[int] = None
    max_daily_loss: Optional[float] = None


class TradeCritiqueRequest(BaseModel):
    symbol: str
    entry_price: float
    exit_price: float
    quantity: int
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    your_reasoning: Optional[str] = ""
    exit_reasoning: Optional[str] = ""


class CoachQuestionRequest(BaseModel):
    question: str


# ============================================================================
# PIPELINE ENDPOINTS
# ============================================================================

@router.get("/api/pipeline/status")
async def get_pipeline_status():
    """Get current status of the trading pipeline"""
    if not HAS_PIPELINE:
        return {"available": False, "error": "Trading pipeline not available"}

    pipeline = get_trading_pipeline()
    status = pipeline.get_status()

    return {
        "available": True,
        **status
    }


@router.post("/api/pipeline/start")
async def start_pipeline():
    """Start the automated trading pipeline"""
    if not HAS_PIPELINE:
        raise HTTPException(status_code=503, detail="Trading pipeline not available")

    pipeline = get_trading_pipeline()

    if pipeline.is_running:
        return {"success": False, "message": "Pipeline already running", "status": pipeline.get_status()}

    pipeline.start()

    return {
        "success": True,
        "message": "Trading pipeline started",
        "status": pipeline.get_status()
    }


@router.post("/api/pipeline/stop")
async def stop_pipeline():
    """Stop the automated trading pipeline"""
    if not HAS_PIPELINE:
        raise HTTPException(status_code=503, detail="Trading pipeline not available")

    pipeline = get_trading_pipeline()

    if not pipeline.is_running:
        return {"success": False, "message": "Pipeline not running"}

    pipeline.stop()

    return {
        "success": True,
        "message": "Trading pipeline stopped",
        "status": pipeline.get_status()
    }


@router.post("/api/pipeline/scan")
async def run_scanner():
    """Run the scanner manually (one-time)"""
    if not HAS_PIPELINE:
        raise HTTPException(status_code=503, detail="Trading pipeline not available")

    pipeline = get_trading_pipeline()
    results = pipeline.run_scanner()

    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        **results
    }


@router.post("/api/pipeline/analyze")
async def analyze_watchlist():
    """Run AI analysis on watchlist (one-time)"""
    if not HAS_PIPELINE:
        raise HTTPException(status_code=503, detail="Trading pipeline not available")

    pipeline = get_trading_pipeline()
    results = pipeline.analyze_watchlist()

    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        **results
    }


@router.post("/api/pipeline/execute")
async def process_execution_queue():
    """Process the trade execution queue"""
    if not HAS_PIPELINE:
        raise HTTPException(status_code=503, detail="Trading pipeline not available")

    pipeline = get_trading_pipeline()
    results = pipeline.process_execution_queue()

    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        **results
    }


@router.post("/api/pipeline/config")
async def update_pipeline_config(config: PipelineConfigRequest):
    """Update pipeline configuration"""
    if not HAS_PIPELINE:
        raise HTTPException(status_code=503, detail="Trading pipeline not available")

    pipeline = get_trading_pipeline()

    # Update only provided fields
    updates = config.dict(exclude_none=True)
    for key, value in updates.items():
        if hasattr(pipeline.config, key):
            setattr(pipeline.config, key, value)

    return {
        "success": True,
        "message": f"Updated {len(updates)} config options",
        "config": pipeline.config.__dict__
    }


# ============================================================================
# AI TRADER ENDPOINTS
# ============================================================================

@router.get("/api/ai-trader/status")
async def get_ai_trader_status():
    """Get AI Watchlist Trader status"""
    if not HAS_AI_TRADER:
        return {"available": False, "error": "AI Trader not available"}

    trader = get_ai_watchlist_trader()

    return {
        "available": True,
        **trader.get_queue_status()
    }


@router.post("/api/ai-trader/analyze/{symbol}")
async def analyze_symbol(symbol: str):
    """Run AI analysis on a specific symbol"""
    if not HAS_AI_TRADER:
        raise HTTPException(status_code=503, detail="AI Trader not available")

    trader = get_ai_watchlist_trader()
    analysis = trader.analyze_symbol(symbol.upper())

    return {
        "symbol": analysis.symbol,
        "timestamp": analysis.timestamp,
        "ai_signal": analysis.ai_signal,
        "ai_confidence": analysis.ai_confidence,
        "ai_action": analysis.ai_action,
        "current_price": analysis.current_price,
        "predicted_direction": analysis.predicted_direction,
        "scores": {
            "momentum": analysis.momentum_score,
            "volume": analysis.volume_score,
            "technical": analysis.technical_score,
            "news_sentiment": analysis.news_sentiment,
            "overall": analysis.overall_score
        },
        "strategies_triggered": analysis.strategies_triggered,
        "has_opportunity": analysis.trade_recommendation is not None,
        "opportunity": analysis.trade_recommendation.__dict__ if analysis.trade_recommendation else None
    }


@router.get("/api/ai-trader/queue")
async def get_opportunity_queue():
    """Get the current trade opportunity queue"""
    if not HAS_AI_TRADER:
        raise HTTPException(status_code=503, detail="AI Trader not available")

    trader = get_ai_watchlist_trader()

    return {
        "queue": [
            {
                "id": opp.id,
                "symbol": opp.symbol,
                "signal": opp.signal,
                "confidence": opp.confidence,
                "entry_price": opp.entry_price,
                "target_price": opp.target_price,
                "stop_loss": opp.stop_loss,
                "position_size": opp.position_size,
                "strategy": opp.strategy,
                "priority": opp.priority,
                "status": opp.status,
                "created_at": opp.created_at,
                "expires_at": opp.expires_at
            }
            for opp in trader.opportunity_queue
            if opp.status in ["pending", "ready"]
        ],
        "count": len([o for o in trader.opportunity_queue if o.status in ["pending", "ready"]])
    }


@router.post("/api/ai-trader/config")
async def update_ai_trader_config(config: AITraderConfigRequest):
    """Update AI Trader configuration"""
    if not HAS_AI_TRADER:
        raise HTTPException(status_code=503, detail="AI Trader not available")

    trader = get_ai_watchlist_trader()

    updates = config.dict(exclude_none=True)
    trader.update_config(updates)

    return {
        "success": True,
        "message": f"Updated {len(updates)} config options",
        "config": trader.config
    }


@router.post("/api/ai-trader/start")
async def start_ai_trader_monitoring():
    """Start AI Trader background monitoring"""
    if not HAS_AI_TRADER:
        raise HTTPException(status_code=503, detail="AI Trader not available")

    trader = get_ai_watchlist_trader()
    trader.start_monitoring()

    return {
        "success": True,
        "message": "AI Trader monitoring started",
        "status": trader.get_queue_status()
    }


@router.post("/api/ai-trader/stop")
async def stop_ai_trader_monitoring():
    """Stop AI Trader background monitoring"""
    if not HAS_AI_TRADER:
        raise HTTPException(status_code=503, detail="AI Trader not available")

    trader = get_ai_watchlist_trader()
    trader.stop_monitoring()

    return {
        "success": True,
        "message": "AI Trader monitoring stopped"
    }


# ============================================================================
# TRADING COACH ENDPOINTS
# ============================================================================

@router.get("/api/coach/window")
async def get_trading_window():
    """Get current trading window info (pre-market, prime time, etc.)"""
    if not HAS_COACH:
        return {"available": False, "error": "Trading coach not available"}

    coach = get_trading_coach()

    return {
        "available": True,
        **coach.get_trading_window()
    }


@router.get("/api/coach/briefing")
async def get_morning_briefing():
    """Generate morning briefing with top setups"""
    if not HAS_COACH:
        raise HTTPException(status_code=503, detail="Trading coach not available")

    coach = get_trading_coach()
    briefing = coach.generate_morning_briefing()

    return {
        "date": briefing.date,
        "generated_at": briefing.generated_at,
        "market_sentiment": briefing.market_sentiment,
        "spy_premarket": briefing.spy_premarket,
        "vix_level": briefing.vix_level,
        "top_gappers": briefing.top_gappers,
        "a_grade_setups": briefing.a_grade_setups,
        "watchlist_additions": briefing.watchlist_additions,
        "catalysts": briefing.catalysts,
        "warnings": briefing.warnings,
        "trading_plan": {
            "max_trades_today": briefing.max_trades_today,
            "max_risk_today": briefing.max_risk_today,
            "focus_strategy": briefing.focus_strategy
        }
    }


@router.post("/api/coach/critique")
async def critique_trade(request: TradeCritiqueRequest):
    """Submit a trade for coaching critique"""
    if not HAS_COACH:
        raise HTTPException(status_code=503, detail="Trading coach not available")

    coach = get_trading_coach()

    trade_data = {
        "symbol": request.symbol.upper(),
        "entry_price": request.entry_price,
        "exit_price": request.exit_price,
        "quantity": request.quantity,
        "entry_time": request.entry_time,
        "exit_time": request.exit_time,
        "your_reasoning": request.your_reasoning,
        "exit_reasoning": request.exit_reasoning
    }

    critique = coach.critique_trade(trade_data)

    return {
        "trade_id": critique.trade_id,
        "symbol": critique.symbol,
        "pnl": critique.pnl,
        "grade": critique.grade,
        "analysis": {
            "setup_quality": critique.setup_quality,
            "entry_timing": critique.entry_timing,
            "exit_timing": critique.exit_timing,
            "position_size_assessment": critique.position_size_assessment
        },
        "emotional_analysis": {
            "flags": critique.emotional_flags,
            "score": critique.emotional_score
        },
        "optimal_trade": {
            "optimal_entry": critique.optimal_entry,
            "optimal_exit": critique.optimal_exit,
            "optimal_pnl": critique.optimal_pnl,
            "missed_profit": critique.missed_profit
        },
        "coaching": {
            "mistakes": critique.mistakes,
            "lessons": critique.lessons,
            "recommendations": critique.recommendations
        }
    }


@router.get("/api/coach/summary")
async def get_coaching_summary():
    """Get emotional trading patterns and coaching advice"""
    if not HAS_COACH:
        raise HTTPException(status_code=503, detail="Trading coach not available")

    coach = get_trading_coach()

    return coach.get_coaching_summary()


@router.post("/api/coach/ask")
async def ask_coach(request: CoachQuestionRequest):
    """Ask the trading coach a question"""
    if not HAS_COACH:
        raise HTTPException(status_code=503, detail="Trading coach not available")

    coach = get_trading_coach()
    answer = coach.ask_question(request.question)

    return {
        "question": request.question,
        "answer": answer,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# TRADE JOURNAL DIRECT ENDPOINTS
# ============================================================================

@router.get("/api/journal/performance")
async def get_performance_metrics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get comprehensive performance metrics"""
    if not HAS_JOURNAL:
        raise HTTPException(status_code=503, detail="Trade journal not available")

    journal = get_trade_journal()
    metrics = journal.get_performance_metrics(start_date, end_date)

    return sanitize_for_json(metrics)


@router.get("/api/journal/daily-analysis")
async def run_daily_analysis(date: Optional[str] = None):
    """Run comprehensive daily analysis"""
    if not HAS_JOURNAL:
        raise HTTPException(status_code=503, detail="Trade journal not available")

    journal = get_trade_journal()
    analysis = journal.run_daily_analysis(date)

    return sanitize_for_json(analysis)


@router.get("/api/journal/trends")
async def analyze_trends(days: int = Query(default=30, ge=7, le=365)):
    """Analyze trading trends over time"""
    if not HAS_JOURNAL:
        raise HTTPException(status_code=503, detail="Trade journal not available")

    journal = get_trade_journal()
    trends = journal.analyze_trends(days)

    return sanitize_for_json(trends)


# ============================================================================
# COMBINED DASHBOARD ENDPOINT
# ============================================================================

@router.get("/api/trading-dashboard")
async def get_trading_dashboard():
    """Get combined trading dashboard with pipeline, trader, and coach data"""
    dashboard = {
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }

    # Pipeline status
    if HAS_PIPELINE:
        try:
            pipeline = get_trading_pipeline()
            dashboard["components"]["pipeline"] = {
                "available": True,
                "is_running": pipeline.is_running,
                "stats": pipeline.stats
            }
        except Exception as e:
            dashboard["components"]["pipeline"] = {"available": True, "error": str(e)}
    else:
        dashboard["components"]["pipeline"] = {"available": False}

    # AI Trader status
    if HAS_AI_TRADER:
        try:
            trader = get_ai_watchlist_trader()
            dashboard["components"]["ai_trader"] = {
                "available": True,
                **trader.get_queue_status()
            }
        except Exception as e:
            dashboard["components"]["ai_trader"] = {"available": True, "error": str(e)}
    else:
        dashboard["components"]["ai_trader"] = {"available": False}

    # Trading Coach status
    if HAS_COACH:
        try:
            coach = get_trading_coach()
            dashboard["components"]["coach"] = {
                "available": True,
                **coach.get_trading_window(),
                "emotional_summary": coach.get_coaching_summary()
            }
        except Exception as e:
            dashboard["components"]["coach"] = {"available": True, "error": str(e)}
    else:
        dashboard["components"]["coach"] = {"available": False}

    # Trade Journal metrics
    if HAS_JOURNAL:
        try:
            journal = get_trade_journal()
            dashboard["components"]["journal"] = {
                "available": True,
                "today": journal.run_daily_analysis()
            }
        except Exception as e:
            dashboard["components"]["journal"] = {"available": True, "error": str(e)}
    else:
        dashboard["components"]["journal"] = {"available": False}

    # Sanitize for JSON (handle inf/nan values)
    return sanitize_for_json(dashboard)
