"""
Warrior Trading Risk Management API Router
Enhanced Phase 4: REST API endpoints for risk management

Provides endpoints for:
- Position sizing (Ross Cameron method: RISK / STOP_DISTANCE)
- Trade validation (2:1 R:R minimum, risk limits)
- Risk status and portfolio metrics
"""

from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict
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
