"""
Warrior Trading Claude AI API Endpoints

FastAPI endpoints for Claude AI features:
- Strategy optimization
- Market regime detection
- Performance analysis
- Self-healing system
- AI insights

Integrates with existing dashboard_api.py
"""

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from claude_integration import get_claude_integration
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from warrior_database import WarriorDatabase
from warrior_market_regime import (MarketIndicators, MarketRegimeDetector,
                                   get_regime_detector)
from warrior_self_healing import (ErrorCategory, ErrorContext, ErrorSeverity,
                                  SelfHealingSystem, get_self_healing)
from warrior_strategy_optimizer import StrategyOptimizer

logger = logging.getLogger(__name__)

# Create router for Claude AI endpoints
router = APIRouter(prefix="/api/warrior/ai", tags=["Claude AI"])

# ═════════════════════════════════════════════════════════════════
#                           REQUEST MODELS
# ═════════════════════════════════════════════════════════════════


class MarketIndicatorsRequest(BaseModel):
    """Market indicators for regime detection"""

    spy_price: float
    spy_change_percent: float
    spy_20sma: Optional[float] = None
    spy_50sma: Optional[float] = None
    qqq_price: Optional[float] = None
    qqq_change_percent: Optional[float] = None
    vix: Optional[float] = None
    vix_change: Optional[float] = None
    advance_decline_ratio: Optional[float] = None
    volume_ratio: Optional[float] = None
    gap_up_count: int = 0
    gap_down_count: int = 0


class OptimizationRequest(BaseModel):
    """Request for strategy optimization"""

    days: int = Field(default=30, ge=7, le=90)
    current_config: Optional[Dict[str, Any]] = None


class ErrorDetectionRequest(BaseModel):
    """Manual error detection request"""

    error_type: str
    error_message: str
    component: str
    additional_data: Optional[Dict[str, Any]] = None


# ═════════════════════════════════════════════════════════════════
#                       STRATEGY OPTIMIZATION
# ═════════════════════════════════════════════════════════════════


@router.get("/optimize/suggest")
async def get_optimization_suggestions(
    days: int = 30, background_tasks: BackgroundTasks = None
):
    """
    Get AI-powered strategy optimization suggestions

    Args:
        days: Number of days of data to analyze (7-90)

    Returns:
        List of optimization suggestions
    """
    try:
        optimizer = StrategyOptimizer()

        # Get current configuration (would load from config in production)
        current_config = {
            "default_position_size": 100,
            "max_position_size": 300,
            "min_confidence": 0.65,
            "max_daily_trades": 5,
            "halt_on_consecutive_losses": 3,
            "daily_profit_goal": 200.0,
            "daily_loss_limit": -100.0,
        }

        suggestions = optimizer.generate_optimization_suggestions(
            current_config=current_config, days=days
        )

        # Convert to dicts
        suggestions_data = []
        for suggestion in suggestions:
            suggestions_data.append(
                {
                    "parameter": suggestion.parameter,
                    "current_value": suggestion.current_value,
                    "suggested_value": suggestion.suggested_value,
                    "reasoning": suggestion.reasoning,
                    "expected_impact": suggestion.expected_impact,
                    "confidence": suggestion.confidence,
                    "priority": suggestion.priority,
                    "status": suggestion.status,
                }
            )

        return {
            "success": True,
            "suggestions": suggestions_data,
            "analysis_period_days": days,
            "generated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error generating optimization suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimize/performance")
async def get_performance_summary(days: int = 30):
    """
    Get comprehensive performance summary

    Args:
        days: Number of days to analyze

    Returns:
        Performance metrics and analysis
    """
    try:
        optimizer = StrategyOptimizer()
        summary = optimizer.get_performance_summary(days=days)

        return {"success": True, **summary}

    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/daily")
async def get_daily_review(review_date: Optional[str] = None):
    """
    Get AI-powered daily trading review

    Args:
        review_date: Date to review (YYYY-MM-DD, defaults to today)

    Returns:
        Daily review with insights and recommendations
    """
    try:
        optimizer = StrategyOptimizer()

        # Parse date
        if review_date:
            target_date = date.fromisoformat(review_date)
        else:
            target_date = date.today()

        review = optimizer.generate_daily_review(review_date=target_date)

        return {"success": True, **review}

    except Exception as e:
        logger.error(f"Error generating daily review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════════════════════════
#                        MARKET REGIME DETECTION
# ═════════════════════════════════════════════════════════════════


@router.post("/regime/detect")
async def detect_market_regime(indicators: MarketIndicatorsRequest):
    """
    Detect current market regime

    Args:
        indicators: Current market indicators

    Returns:
        Regime classification and recommended adjustments
    """
    try:
        detector = get_regime_detector()

        # Convert request to MarketIndicators
        market_indicators = MarketIndicators(
            spy_price=indicators.spy_price,
            spy_change_percent=indicators.spy_change_percent,
            spy_20sma=indicators.spy_20sma,
            spy_50sma=indicators.spy_50sma,
            qqq_price=indicators.qqq_price,
            qqq_change_percent=indicators.qqq_change_percent,
            vix=indicators.vix,
            vix_change=indicators.vix_change,
            advance_decline_ratio=indicators.advance_decline_ratio,
            volume_ratio=indicators.volume_ratio,
            gap_up_count=indicators.gap_up_count,
            gap_down_count=indicators.gap_down_count,
        )

        # Detect regime (use AI if available)
        detection = detector.detect_regime(market_indicators, use_ai=True)

        return {"success": True, **detector.to_dict()}

    except Exception as e:
        logger.error(f"Error detecting market regime: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regime/current")
async def get_current_regime():
    """
    Get current market regime (cached)

    Returns:
        Current regime or None if not detected
    """
    try:
        detector = get_regime_detector()
        current = detector.get_current_regime()

        if current:
            return {"success": True, **detector.to_dict()}
        else:
            return {
                "success": True,
                "regime": "UNKNOWN",
                "detected": False,
                "message": "No regime detected yet",
            }

    except Exception as e:
        logger.error(f"Error getting current regime: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regime/history")
async def get_regime_history(hours: int = 4):
    """
    Get regime detection history

    Args:
        hours: Number of hours of history to retrieve

    Returns:
        List of recent regime detections
    """
    try:
        detector = get_regime_detector()
        history = detector.get_regime_history(hours=hours)

        history_data = []
        for detection in history:
            history_data.append(
                {
                    "regime": detection.regime.value,
                    "confidence": detection.confidence,
                    "reasoning": detection.reasoning,
                    "detection_time": detection.detection_time.isoformat(),
                    "warnings": detection.warnings,
                }
            )

        return {"success": True, "history": history_data, "period_hours": hours}

    except Exception as e:
        logger.error(f"Error getting regime history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════════════════════════
#                           SELF-HEALING
# ═════════════════════════════════════════════════════════════════


@router.get("/health/status")
async def get_system_health():
    """
    Get system health status

    Returns:
        Health metrics and active errors
    """
    try:
        healing = get_self_healing()
        health = healing.get_system_health()

        return {"success": True, **health}

    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/errors")
async def get_active_errors():
    """
    Get active errors

    Returns:
        List of currently active errors
    """
    try:
        healing = get_self_healing()

        active_errors = []
        for error_key, context in healing.active_errors.items():
            active_errors.append(
                {
                    "key": error_key,
                    "type": context.error_type,
                    "message": context.error_message,
                    "category": context.category.value,
                    "severity": context.severity.value,
                    "component": context.component,
                    "timestamp": context.timestamp.isoformat(),
                }
            )

        return {
            "success": True,
            "active_errors": active_errors,
            "count": len(active_errors),
        }

    except Exception as e:
        logger.error(f"Error getting active errors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/history")
async def get_error_history(hours: int = 24):
    """
    Get error recovery history

    Args:
        hours: Number of hours of history

    Returns:
        List of recent error recoveries
    """
    try:
        healing = get_self_healing()
        history = healing.get_error_history(hours=hours)

        history_data = []
        for result in history:
            history_data.append(
                {
                    "error_type": result.error_context.error_type,
                    "error_message": result.error_context.error_message,
                    "category": result.error_context.category.value,
                    "severity": result.error_context.severity.value,
                    "diagnosis": result.diagnosis,
                    "recovery_status": result.status.value,
                    "attempted_actions": result.attempted_actions,
                    "recovery_time_seconds": result.recovery_time_seconds,
                    "requires_manual": result.requires_manual_intervention,
                    "timestamp": result.timestamp.isoformat(),
                }
            )

        return {
            "success": True,
            "history": history_data,
            "period_hours": hours,
            "count": len(history_data),
        }

    except Exception as e:
        logger.error(f"Error getting error history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/health/detect")
async def detect_error(request: ErrorDetectionRequest):
    """
    Manually detect and diagnose an error

    Args:
        request: Error detection request

    Returns:
        Diagnosis and recovery suggestions
    """
    try:
        healing = get_self_healing()

        # Create error context
        context = ErrorContext(
            error_type=request.error_type,
            error_message=request.error_message,
            category=ErrorCategory[request.component.upper()],  # Convert to enum
            severity=ErrorSeverity.MEDIUM,  # Default severity
            timestamp=datetime.now(),
            component=request.component,
            additional_data=request.additional_data,
        )

        # Diagnose
        diagnosis = healing.diagnose_error(context, use_ai=True)

        return {"success": True, "diagnosis": diagnosis}

    except Exception as e:
        logger.error(f"Error detecting/diagnosing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/health/recover")
async def recover_from_error(request: ErrorDetectionRequest):
    """
    Attempt automated error recovery

    Args:
        request: Error detection request

    Returns:
        Recovery result
    """
    try:
        healing = get_self_healing()

        # Create error context
        context = ErrorContext(
            error_type=request.error_type,
            error_message=request.error_message,
            category=ErrorCategory[request.component.upper()],
            severity=ErrorSeverity.MEDIUM,
            timestamp=datetime.now(),
            component=request.component,
            additional_data=request.additional_data,
        )

        # Attempt recovery
        result = healing.recover_from_error(context, auto_recover=True)

        return {
            "success": True,
            "recovery_status": result.status.value,
            "diagnosis": result.diagnosis,
            "attempted_actions": result.attempted_actions,
            "recovery_time_seconds": result.recovery_time_seconds,
            "requires_manual": result.requires_manual_intervention,
            "resolution_notes": result.resolution_notes,
        }

    except Exception as e:
        logger.error(f"Error attempting recovery: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════════════════════════
#                         CLAUDE AI CONFIG
# ═════════════════════════════════════════════════════════════════


@router.get("/config")
async def get_claude_config():
    """
    Get Claude AI configuration and usage stats

    Returns:
        Configuration and usage statistics
    """
    try:
        claude = get_claude_integration()
        usage_stats = claude.get_usage_stats()

        return {
            "success": True,
            "model": claude.model,
            "rate_limit_rpm": claude.max_requests_per_minute,
            "daily_limit_usd": claude.daily_cost_limit,
            "monthly_limit_usd": claude.monthly_cost_limit,
            "cache_enabled": claude.cache_enabled,
            "cache_ttl_seconds": claude.cache_ttl_seconds,
            "usage_stats": usage_stats,
        }

    except Exception as e:
        logger.error(f"Error getting Claude config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/clear-cache")
async def clear_claude_cache():
    """
    Clear Claude response cache

    Returns:
        Success confirmation
    """
    try:
        claude = get_claude_integration()
        claude.clear_cache()

        return {"success": True, "message": "Claude response cache cleared"}

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/reset-daily-stats")
async def reset_daily_stats():
    """
    Reset daily usage statistics

    Returns:
        Success confirmation
    """
    try:
        claude = get_claude_integration()
        claude.reset_daily_stats()

        return {"success": True, "message": "Daily statistics reset"}

    except Exception as e:
        logger.error(f"Error resetting daily stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════════════════════════
#                         UTILITY ENDPOINTS
# ═════════════════════════════════════════════════════════════════


@router.get("/insights/realtime")
async def get_realtime_insights(insight_type: str = "general"):
    """
    Get real-time AI insights

    Args:
        insight_type: Type of insights (general, risk, opportunity, improvement)

    Returns:
        List of insights
    """
    try:
        claude = get_claude_integration()

        # Build context (would gather from various sources in production)
        context = {
            "current_time": datetime.now().isoformat(),
            "market_hours": True,  # Would determine from actual market hours
            "active_trades": 0,  # Would get from risk manager
            "daily_pnl": 0.0,  # Would get from risk manager
        }

        insights = claude.generate_insights(context=context, insight_type=insight_type)

        return {
            "success": True,
            "insights": insights,
            "insight_type": insight_type,
            "generated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Export router
__all__ = ["router"]


if __name__ == "__main__":
    # Test endpoint responses
    print("Claude AI API Endpoints Module")
    print("=" * 50)
    print("Available endpoints:")
    print("\nStrategy Optimization:")
    print("  GET  /api/warrior/ai/optimize/suggest")
    print("  GET  /api/warrior/ai/optimize/performance")
    print("  GET  /api/warrior/ai/analysis/daily")
    print("\nMarket Regime:")
    print("  POST /api/warrior/ai/regime/detect")
    print("  GET  /api/warrior/ai/regime/current")
    print("  GET  /api/warrior/ai/regime/history")
    print("\nSelf-Healing:")
    print("  GET  /api/warrior/ai/health/status")
    print("  GET  /api/warrior/ai/health/errors")
    print("  GET  /api/warrior/ai/health/history")
    print("  POST /api/warrior/ai/health/detect")
    print("  POST /api/warrior/ai/health/recover")
    print("\nConfiguration:")
    print("  GET  /api/warrior/ai/config")
    print("  POST /api/warrior/ai/config/clear-cache")
    print("  POST /api/warrior/ai/config/reset-daily-stats")
    print("\nInsights:")
    print("  GET  /api/warrior/ai/insights/realtime")
    print("\n" + "=" * 50)
