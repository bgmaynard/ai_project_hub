"""
Warrior Trading Risk Management API Router
Enhanced Phase 4: REST API endpoints for risk management

Provides endpoints for:
- Position sizing (Ross Cameron method: RISK / STOP_DISTANCE)
- Trade validation (2:1 R:R minimum, risk limits)
- Risk status and portfolio metrics
"""

from fastapi import APIRouter, HTTPException, Body, Query
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/risk", tags=["Risk Management"])

class PositionSizeRequest(BaseModel):
    symbol: str
    entry_price: float = Field(..., gt=0)
    stop_distance: float = Field(..., gt=0)
    risk_amount: float = Field(50.0, gt=0)

class PositionSizeResponse(BaseModel):
    symbol: str
    shares: int
    position_value: float
    risk_amount: float

class TradeValidationRequest(BaseModel):
    symbol: str
    entry_price: float
    stop_loss: float
    target_price: float
    shares: int

class TradeValidationResponse(BaseModel):
    symbol: str
    result: str
    risk_reward_ratio: float
    risk_amount: float
    warnings: List[str]

@router.get("/health")
async def get_risk_health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.post("/calculate-position-size", response_model=PositionSizeResponse)
async def calculate_position_size(request: PositionSizeRequest):
    try:
        shares = int(request.risk_amount / request.stop_distance)
        position_value = shares * request.entry_price
        return PositionSizeResponse(
            symbol=request.symbol,
            shares=shares,
            position_value=position_value,
            risk_amount=request.risk_amount
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate-trade", response_model=TradeValidationResponse)
async def validate_trade(request: TradeValidationRequest):
    try:
        risk_per_share = abs(request.entry_price - request.stop_loss)
        reward_per_share = abs(request.target_price - request.entry_price)
        risk_amount = risk_per_share * request.shares
        rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
        
        warnings = []
        result = "APPROVED"
        if rr_ratio < 2.0:
            warnings.append(f"R:R {rr_ratio:.2f} below minimum 2:1")
            result = "REJECTED"
        if risk_amount > 50.0:
            warnings.append(f"Risk ${risk_amount:.2f} exceeds max $50")
            result = "REJECTED"
        
        return TradeValidationResponse(
            symbol=request.symbol,
            result=result,
            risk_reward_ratio=rr_ratio,
            risk_amount=risk_amount,
            warnings=warnings
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_risk_status():
    return {
        "daily_pnl": 0.0,
        "max_daily_loss": 200.0,
        "max_loss_per_trade": 50.0,
        "is_trading_halted": False,
        "trades_today": 0
    }

# ===== SLIPPAGE & REVERSAL MONITORING =====

try:
    from ai.warrior_slippage_monitor import get_slippage_monitor, SlippageLevel
    from ai.warrior_reversal_detector import get_reversal_detector, ReversalType
    SLIPPAGE_AVAILABLE = True
except ImportError:
    SLIPPAGE_AVAILABLE = False
    logging.warning("Slippage/Reversal modules not available")

class SlippageRecordRequest(BaseModel):
    symbol: str
    side: str = Field(pattern="^(buy|sell)$")
    expected_price: float = Field(gt=0)
    actual_price: float = Field(gt=0)
    shares: int = Field(gt=0)

class SlippageResponse(BaseModel):
    symbol: str
    slippage_pct: float
    slippage_level: str
    is_acceptable: bool

class ReversalCheckRequest(BaseModel):
    symbol: str
    current_price: float
    entry_price: float
    recent_prices: List[float]
    direction: str = Field("long", pattern="^(long|short)$")

class ReversalResponse(BaseModel):
    symbol: str
    reversal_detected: bool
    reversal_type: Optional[str]
    severity: Optional[str]
    recommendation: Optional[str]
    should_exit_fast: bool

@router.post("/record-slippage", response_model=SlippageResponse)
async def record_slippage(request: SlippageRecordRequest):
    """
    Record order execution slippage
    
    Tracks difference between expected and actual fill prices.
    Alerts on excessive slippage (>0.25%)
    
    Example:
        POST /api/risk/record-slippage
        {
            "symbol": "AAPL",
            "side": "buy",
            "expected_price": 150.00,
            "actual_price": 150.15,
            "shares": 100
        }
    """
    if not SLIPPAGE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Slippage monitor not available")
    
    try:
        monitor = get_slippage_monitor()
        execution = monitor.record_execution(
            request.symbol,
            request.side,
            request.expected_price,
            request.actual_price,
            request.shares
        )
        
        return SlippageResponse(
            symbol=request.symbol,
            slippage_pct=execution.slippage_pct,
            slippage_level=execution.slippage_level.value,
            is_acceptable=execution.slippage_level == SlippageLevel.ACCEPTABLE
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/slippage-stats")
async def get_slippage_stats(symbol: Optional[str] = Query(None)):
    """
    Get slippage statistics
    
    Example:
        GET /api/risk/slippage-stats?symbol=AAPL
    """
    if not SLIPPAGE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Slippage monitor not available")
    
    try:
        monitor = get_slippage_monitor()
        return monitor.get_stats(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/check-reversal", response_model=ReversalResponse)
async def check_reversal(request: ReversalCheckRequest):
    """
    Check for jacknife reversal pattern
    
    Detects violent price reversals requiring fast exit.
    Returns exit recommendation if critical reversal detected.
    
    Example:
        POST /api/risk/check-reversal
        {
            "symbol": "TSLA",
            "current_price": 245.50,
            "entry_price": 244.00,
            "recent_prices": [244.0, 245.5, 246.2, 245.5],
            "direction": "long"
        }
    """
    if not SLIPPAGE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Reversal detector not available")
    
    try:
        detector = get_reversal_detector()
        reversal = detector.detect_jacknife(
            request.symbol,
            request.current_price,
            request.entry_price,
            request.recent_prices,
            request.direction
        )
        
        if reversal:
            return ReversalResponse(
                symbol=request.symbol,
                reversal_detected=True,
                reversal_type=reversal.reversal_type.value,
                severity=reversal.severity.value,
                recommendation=reversal.recommendation,
                should_exit_fast=detector.should_exit_fast(reversal)
            )
        else:
            return ReversalResponse(
                symbol=request.symbol,
                reversal_detected=False,
                reversal_type=None,
                severity=None,
                recommendation=None,
                should_exit_fast=False
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
