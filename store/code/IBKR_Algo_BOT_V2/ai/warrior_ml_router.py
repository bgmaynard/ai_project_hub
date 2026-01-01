"""
Warrior Trading ML API Router
Phase 3: REST API endpoints for Advanced ML features

Provides endpoints for:
- Transformer pattern detection
- RL agent execution recommendations
- Model training status
- Predictions and confidence scores
"""

import logging
from datetime import datetime
from typing import List, Optional

from ai.warrior_rl_agent import TradingAction, TradingState, get_rl_agent
from ai.warrior_transformer_detector import (PatternDetection,
                                             get_transformer_detector)
from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/ml", tags=["Advanced ML"])


# Request/Response models


class CandleData(BaseModel):
    """Single candlestick data"""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class PatternDetectionRequest(BaseModel):
    """Request for pattern detection"""

    symbol: str
    candles: List[CandleData] = Field(..., min_length=20, max_length=500)
    timeframe: str = Field("5min", pattern="^(1min|5min|15min|1h|1d)$")


class PatternDetectionResponse(BaseModel):
    """Pattern detection result"""

    symbol: str
    pattern_type: Optional[str]
    confidence: float
    timeframe: str
    timestamp: datetime
    price_target: Optional[float]
    stop_loss: Optional[float]
    features: dict


class TradingStateRequest(BaseModel):
    """Request for RL agent action"""

    symbol: str
    price: float
    volume: float
    volatility: float
    trend: float
    position_size: float = 0.0
    entry_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    sentiment_score: float = 0.0
    pattern_confidence: float = 0.5
    time_in_position: int = 0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 1.0
    win_rate: float = 0.5


class RLActionResponse(BaseModel):
    """RL agent action recommendation"""

    symbol: str
    action_type: str
    size_change: float
    confidence: float
    reasoning: str


class MLHealthResponse(BaseModel):
    """ML system health status"""

    status: str
    transformer_loaded: bool
    rl_agent_loaded: bool
    models_trained: bool
    device: str


# Endpoints


@router.get("/health", response_model=MLHealthResponse)
async def get_ml_health():
    """
    Check ML system health

    Returns status of:
    - Transformer model
    - RL agent
    - Training status
    - Compute device (CPU/GPU)
    """
    try:
        # Check transformer
        try:
            detector = get_transformer_detector()
            transformer_loaded = True
            device = detector.device
        except Exception as e:
            logger.error(f"Transformer not loaded: {e}")
            transformer_loaded = False
            device = "unknown"

        # Check RL agent
        try:
            agent = get_rl_agent()
            rl_loaded = True
            if device == "unknown":
                device = agent.device
        except Exception as e:
            logger.error(f"RL agent not loaded: {e}")
            rl_loaded = False

        # Check if models are trained (basic check)
        models_trained = transformer_loaded and rl_loaded

        status = "healthy" if (transformer_loaded and rl_loaded) else "degraded"

        return MLHealthResponse(
            status=status,
            transformer_loaded=transformer_loaded,
            rl_agent_loaded=rl_loaded,
            models_trained=models_trained,
            device=device,
        )

    except Exception as e:
        logger.error(f"ML health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-pattern", response_model=PatternDetectionResponse)
async def detect_pattern(request: PatternDetectionRequest):
    """
    Detect chart pattern using Transformer model

    Args:
        request: Candlestick data and metadata

    Returns:
        Pattern detection result with confidence and targets

    Example:
        POST /api/ml/detect-pattern
        {
            "symbol": "AAPL",
            "candles": [...],
            "timeframe": "5min"
        }
    """
    try:
        detector = get_transformer_detector()

        # Convert candles to dict format
        candles = [
            {
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in request.candles
        ]

        # Detect pattern
        pattern = detector.detect_pattern(
            candles=candles, symbol=request.symbol, timeframe=request.timeframe
        )

        if pattern:
            return PatternDetectionResponse(
                symbol=request.symbol,
                pattern_type=pattern.pattern_type,
                confidence=pattern.confidence,
                timeframe=pattern.timeframe,
                timestamp=pattern.timestamp,
                price_target=pattern.price_target,
                stop_loss=pattern.stop_loss,
                features=pattern.features,
            )
        else:
            # No pattern detected
            return PatternDetectionResponse(
                symbol=request.symbol,
                pattern_type=None,
                confidence=0.0,
                timeframe=request.timeframe,
                timestamp=datetime.now(),
                price_target=None,
                stop_loss=None,
                features={},
            )

    except Exception as e:
        logger.error(f"Pattern detection failed for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend-action", response_model=RLActionResponse)
async def recommend_action(request: TradingStateRequest):
    """
    Get RL agent execution recommendation

    Args:
        request: Current trading state

    Returns:
        Recommended action with confidence and reasoning

    Example:
        POST /api/ml/recommend-action
        {
            "symbol": "AAPL",
            "price": 150.0,
            "volume": 1000000,
            ...
        }
    """
    try:
        agent = get_rl_agent()

        # Create trading state
        state = TradingState(
            price=request.price,
            volume=request.volume,
            volatility=request.volatility,
            trend=request.trend,
            position_size=request.position_size,
            entry_price=request.entry_price,
            unrealized_pnl=request.unrealized_pnl,
            sentiment_score=request.sentiment_score,
            pattern_confidence=request.pattern_confidence,
            time_in_position=request.time_in_position,
            current_drawdown=request.current_drawdown,
            sharpe_ratio=request.sharpe_ratio,
            win_rate=request.win_rate,
        )

        # Get action recommendation (inference mode)
        action = agent.select_action(state, training=False)

        # Generate reasoning
        reasoning = _generate_reasoning(state, action)

        return RLActionResponse(
            symbol=request.symbol,
            action_type=action.action_type,
            size_change=action.size_change,
            confidence=action.confidence,
            reasoning=reasoning,
        )

    except Exception as e:
        logger.error(f"RL recommendation failed for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/supported")
async def get_supported_patterns():
    """
    Get list of supported pattern types

    Returns:
        List of pattern names that the transformer can detect
    """
    try:
        detector = get_transformer_detector()
        return {
            "patterns": detector.PATTERN_NAMES,
            "count": len(detector.PATTERN_NAMES),
        }
    except Exception as e:
        logger.error(f"Failed to get patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/actions/available")
async def get_available_actions():
    """
    Get list of available RL agent actions

    Returns:
        List of action types the RL agent can recommend
    """
    try:
        agent = get_rl_agent()
        return {
            "actions": agent.ACTIONS,
            "count": len(agent.ACTIONS),
            "descriptions": {
                "enter": "Enter new position",
                "hold": "Hold current position",
                "exit": "Exit current position",
                "size_up": "Increase position size",
                "size_down": "Decrease position size",
            },
        }
    except Exception as e:
        logger.error(f"Failed to get actions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch/detect-patterns")
async def batch_detect_patterns(
    requests: List[PatternDetectionRequest] = Body(..., max_length=10)
):
    """
    Detect patterns for multiple symbols

    Args:
        requests: List of pattern detection requests (max 10)

    Returns:
        Dict of symbol -> pattern detection result
    """
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 symbols per batch")

    results = {}

    for req in requests:
        try:
            response = await detect_pattern(req)
            results[req.symbol] = response.dict()
        except Exception as e:
            logger.error(f"Batch detection failed for {req.symbol}: {e}")
            results[req.symbol] = {"error": str(e)}

    return results


# Helper functions


def _generate_reasoning(state: TradingState, action: TradingAction) -> str:
    """Generate human-readable reasoning for RL action"""
    reasons = []

    # Position status
    if state.position_size == 0:
        reasons.append("No current position.")
    else:
        pnl_pct = state.unrealized_pnl * 100
        reasons.append(
            f"Current position: {state.position_size:.1%}, P&L: {pnl_pct:+.1f}%"
        )

    # Action reasoning
    if action.action_type == "enter":
        if state.sentiment_score > 0.3:
            reasons.append(f"Positive sentiment ({state.sentiment_score:+.2f})")
        if state.pattern_confidence > 0.6:
            reasons.append(
                f"Strong pattern detected (confidence: {state.pattern_confidence:.1%})"
            )
        reasons.append("Recommending entry")

    elif action.action_type == "exit":
        if state.unrealized_pnl < 0 and state.time_in_position > 20:
            reasons.append("Cutting losses on underperforming position")
        elif state.unrealized_pnl > 0:
            reasons.append("Taking profits")
        else:
            reasons.append("Exiting position")

    elif action.action_type == "hold":
        if abs(state.unrealized_pnl) < 0.02:
            reasons.append("Position near breakeven, holding")
        else:
            reasons.append("Maintaining current position")

    elif action.action_type == "size_up":
        reasons.append("Increasing position size on favorable conditions")

    elif action.action_type == "size_down":
        reasons.append("Reducing risk exposure")

    return " ".join(reasons)


# Example usage documentation
"""
ML API ENDPOINTS

1. Health Check
   GET /api/ml/health

   Check ML system status

2. Pattern Detection
   POST /api/ml/detect-pattern

   Detect chart patterns using transformer model

3. RL Action Recommendation
   POST /api/ml/recommend-action

   Get execution recommendation from RL agent

4. Supported Patterns
   GET /api/ml/patterns/supported

   List all detectable patterns

5. Available Actions
   GET /api/ml/actions/available

   List all RL agent actions

6. Batch Pattern Detection
   POST /api/ml/batch/detect-patterns

   Detect patterns for multiple symbols (max 10)
"""
