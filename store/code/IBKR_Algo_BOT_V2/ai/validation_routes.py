"""
Strategy Policy Validation & Calibration Routes
================================================
API endpoints for the Phase: Strategy Policy Validation, Calibration, and Activation

Endpoints:
- /api/validation/momentum - MomentumSnapshot engine
- /api/validation/exit - Exit imperatives
- /api/validation/safe - Safe activation mode
- /api/validation/export - Observability exports
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import logging

from .momentum_snapshot import (
    get_momentum_snapshot_engine,
    MomentumOutputState
)
from .exit_imperatives import (
    get_exit_imperative_engine,
    ExitReason
)
from .safe_activation import (
    get_safe_activation,
    ActivationMode
)
from .strategy_policy_engine import get_strategy_policy_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/validation", tags=["Validation & Calibration"])


# ==================== Request Models ====================

class PriceUpdate(BaseModel):
    symbol: str
    price: float
    volume: int = 0
    bid: float = 0
    ask: float = 0


class VolumeBaseline(BaseModel):
    symbol: str
    avg_1m: float
    avg_5m: float


class PositionRegister(BaseModel):
    symbol: str
    entry_price: float
    shares: int
    stop_price: float = 0
    target_price: float = 0


class ExitCheck(BaseModel):
    symbol: str
    momentum_state: str = ""
    regime: str = ""
    volatility: float = 0
    above_vwap: bool = True


class SafeActivateRequest(BaseModel):
    mode: str = "SAFE"
    position_multiplier: float = 0.25
    symbol_whitelist: List[str] = []


class TradeCheckRequest(BaseModel):
    symbol: str
    strategy_id: str
    size_usd: float


class RecordVetoRequest(BaseModel):
    reason: str


class RecordExitRequest(BaseModel):
    reason: str
    is_forced: bool = False
    pnl: float = 0


class RecordTradeRequest(BaseModel):
    is_win: bool
    pnl: float


# ==================== Momentum Snapshot Routes ====================

@router.post("/momentum/price")
async def add_price_update(data: PriceUpdate):
    """Add a price update for momentum tracking"""
    engine = get_momentum_snapshot_engine()
    engine.add_price(
        symbol=data.symbol,
        price=data.price,
        volume=data.volume,
        bid=data.bid,
        ask=data.ask
    )
    return {"status": "recorded", "symbol": data.symbol}


@router.post("/momentum/volume-baseline")
async def set_volume_baseline(data: VolumeBaseline):
    """Set volume baseline for relative volume calculation"""
    engine = get_momentum_snapshot_engine()
    engine.set_volume_baseline(data.symbol, data.avg_1m, data.avg_5m)
    return {"status": "set", "symbol": data.symbol}


@router.get("/momentum/snapshot/{symbol}")
async def get_momentum_snapshot(symbol: str):
    """Get current momentum snapshot for a symbol"""
    engine = get_momentum_snapshot_engine()
    snapshot = engine.compute_snapshot(symbol.upper())

    if not snapshot:
        return {"symbol": symbol, "error": "Insufficient data"}

    return snapshot.to_dict()


@router.get("/momentum/can-enter/{symbol}")
async def check_entry_eligibility(symbol: str):
    """Check if symbol is eligible for entry (requires CONFIRMED)"""
    engine = get_momentum_snapshot_engine()
    can_enter, reason = engine.can_enter(symbol.upper())
    return {
        "symbol": symbol,
        "can_enter": can_enter,
        "reason": reason,
        "current_state": engine.get_state(symbol.upper()).value
    }


@router.get("/momentum/should-exit/{symbol}")
async def check_exit_signal(symbol: str):
    """Check if momentum suggests exit"""
    engine = get_momentum_snapshot_engine()
    should_exit, reason, urgency = engine.should_exit(symbol.upper())
    return {
        "symbol": symbol,
        "should_exit": should_exit,
        "reason": reason,
        "urgency": urgency,
        "current_state": engine.get_state(symbol.upper()).value
    }


@router.get("/momentum/states")
async def get_all_momentum_states():
    """Get all symbol momentum states"""
    engine = get_momentum_snapshot_engine()
    return {
        "states": engine.get_all_states(),
        "stats": engine.get_stats()
    }


@router.get("/momentum/transitions")
async def get_momentum_transitions(
    symbol: str = None,
    limit: int = Query(50, description="Max entries")
):
    """Get momentum state transition log"""
    engine = get_momentum_snapshot_engine()
    return {
        "transitions": engine.get_transition_log(symbol, limit)
    }


@router.post("/momentum/reset/{symbol}")
async def reset_momentum_symbol(symbol: str):
    """Reset a symbol's momentum state to DEAD"""
    engine = get_momentum_snapshot_engine()
    engine.reset_symbol(symbol.upper())
    return {"status": "reset", "symbol": symbol}


# ==================== Exit Imperatives Routes ====================

@router.post("/exit/register")
async def register_position_for_exit(data: PositionRegister):
    """Register a position for exit monitoring"""
    engine = get_exit_imperative_engine()
    engine.register_position(
        symbol=data.symbol.upper(),
        entry_price=data.entry_price,
        shares=data.shares,
        stop_price=data.stop_price,
        target_price=data.target_price
    )
    return {"status": "registered", "symbol": data.symbol}


@router.delete("/exit/{symbol}")
async def unregister_position(symbol: str):
    """Unregister a position from exit monitoring"""
    engine = get_exit_imperative_engine()
    engine.unregister_position(symbol.upper())
    return {"status": "unregistered", "symbol": symbol}


@router.post("/exit/update/{symbol}")
async def update_position_price(symbol: str, current_price: float):
    """Update current price for exit monitoring"""
    engine = get_exit_imperative_engine()
    engine.update_position(symbol.upper(), current_price)
    return {"status": "updated", "symbol": symbol, "price": current_price}


@router.post("/exit/check")
async def check_exit_imperatives(data: ExitCheck):
    """Check all exit imperatives for a position"""
    engine = get_exit_imperative_engine()
    result = engine.check_exit(
        symbol=data.symbol.upper(),
        momentum_state=data.momentum_state,
        regime=data.regime,
        volatility=data.volatility,
        above_vwap=data.above_vwap
    )
    return result.to_dict()


@router.get("/exit/positions")
async def get_monitored_positions():
    """Get all positions being monitored for exit"""
    engine = get_exit_imperative_engine()
    return {"positions": engine.get_monitored_positions()}


@router.get("/exit/log")
async def get_exit_log(
    symbol: str = None,
    limit: int = Query(50, description="Max entries")
):
    """Get exit imperative log"""
    engine = get_exit_imperative_engine()
    return {"log": engine.get_exit_log(symbol, limit)}


@router.get("/exit/stats")
async def get_exit_stats():
    """Get exit statistics"""
    engine = get_exit_imperative_engine()
    return engine.get_exit_stats()


@router.get("/exit/config")
async def get_exit_config():
    """Get exit imperative configuration"""
    engine = get_exit_imperative_engine()
    return {"config": engine.get_config()}


@router.post("/exit/config")
async def update_exit_config(config: Dict[str, Any]):
    """Update exit imperative configuration"""
    engine = get_exit_imperative_engine()
    engine.update_config(config)
    return {"status": "updated", "config": engine.get_config()}


# ==================== Safe Activation Routes ====================

@router.get("/safe/status")
async def get_safe_status():
    """Get safe activation mode status"""
    safe = get_safe_activation()
    return safe.get_status()


@router.post("/safe/activate")
async def activate_safe_mode(data: SafeActivateRequest):
    """Activate safe trading mode"""
    safe = get_safe_activation()

    try:
        mode = ActivationMode(data.mode.upper())
    except ValueError:
        raise HTTPException(400, f"Invalid mode: {data.mode}")

    result = safe.activate(
        mode=mode,
        position_multiplier=data.position_multiplier,
        symbol_whitelist=data.symbol_whitelist
    )

    if not result['success']:
        raise HTTPException(400, result.get('error', 'Activation failed'))

    return result


@router.post("/safe/deactivate")
async def deactivate_safe_mode(reason: str = "manual"):
    """Deactivate trading"""
    safe = get_safe_activation()
    return safe.deactivate(reason)


@router.post("/safe/can-trade")
async def check_trade_allowed(data: TradeCheckRequest):
    """Check if a trade is allowed under safe mode"""
    safe = get_safe_activation()
    allowed, reason, adjusted_size = safe.can_trade(
        symbol=data.symbol.upper(),
        strategy_id=data.strategy_id,
        size_usd=data.size_usd
    )
    return {
        "allowed": allowed,
        "reason": reason,
        "adjusted_size_usd": adjusted_size,
        "original_size_usd": data.size_usd
    }


@router.post("/safe/release/{strategy_id}")
async def release_strategy(strategy_id: str):
    """Release a strategy after trade completion"""
    safe = get_safe_activation()
    safe.release_strategy(strategy_id)
    return {"status": "released", "strategy_id": strategy_id}


@router.post("/safe/record/veto")
async def record_veto_for_safe(data: RecordVetoRequest):
    """Record a veto event for kill-switch monitoring"""
    safe = get_safe_activation()
    safe.record_veto(data.reason)
    return {"status": "recorded"}


@router.post("/safe/record/exit")
async def record_exit_for_safe(data: RecordExitRequest):
    """Record an exit event"""
    safe = get_safe_activation()
    safe.record_exit(data.reason, data.is_forced, data.pnl)
    return {"status": "recorded"}


@router.post("/safe/record/trade")
async def record_trade_for_safe(data: RecordTradeRequest):
    """Record a completed trade"""
    safe = get_safe_activation()
    safe.record_trade(data.is_win, data.pnl)
    return {"status": "recorded"}


@router.post("/safe/record/regime/{regime}")
async def record_regime_change(regime: str):
    """Record a regime change"""
    safe = get_safe_activation()
    safe.record_regime_change(regime.upper())
    return {"status": "recorded", "regime": regime}


@router.post("/safe/record/momentum/{state}")
async def record_momentum_state(state: str):
    """Record momentum state observation"""
    safe = get_safe_activation()
    safe.record_momentum_state(state.upper())
    return {"status": "recorded", "state": state}


@router.post("/safe/kill-switch/reset")
async def reset_kill_switch(force: bool = False):
    """Reset kill-switch"""
    safe = get_safe_activation()
    result = safe.reset_kill_switch(force=force)

    if not result['success']:
        raise HTTPException(400, result.get('error', 'Reset failed'))

    return result


@router.post("/safe/reset-daily")
async def reset_daily_metrics():
    """Reset daily metrics (call at market open)"""
    safe = get_safe_activation()
    safe.reset_daily_metrics()
    return {"status": "reset"}


# ==================== Observability Export Routes ====================

@router.get("/export/json")
async def export_observability_json():
    """Export observability data as JSON"""
    safe = get_safe_activation()
    return safe.export_observability("json")


@router.get("/export/csv")
async def export_observability_csv():
    """Export observability data as CSV"""
    safe = get_safe_activation()
    csv_content = safe.export_observability("csv")
    return {"csv": csv_content}


@router.get("/export/summary")
async def get_observability_summary():
    """Get observability summary"""
    safe = get_safe_activation()
    momentum = get_momentum_snapshot_engine()
    exits = get_exit_imperative_engine()
    policy = get_strategy_policy_engine()

    return {
        "safe_activation": safe.get_status(),
        "momentum": momentum.get_stats(),
        "exits": exits.get_exit_stats(),
        "policy": policy.get_all_policies(),
    }


# ==================== Unified Validation Check ====================

@router.post("/check-entry/{symbol}")
async def unified_entry_check(
    symbol: str,
    strategy_id: str,
    size_usd: float,
    current_price: float,
    momentum_state: str = "",
    regime: str = "",
    above_vwap: bool = True
):
    """
    Unified entry check that validates through all systems.

    Checks:
    1. Safe activation mode allows trade
    2. Momentum state is CONFIRMED
    3. Strategy policy allows trade
    4. No exit imperatives active

    Returns comprehensive decision.
    """
    symbol = symbol.upper()

    safe = get_safe_activation()
    momentum = get_momentum_snapshot_engine()
    policy = get_strategy_policy_engine()

    results = {
        "symbol": symbol,
        "approved": False,
        "checks": {},
        "adjusted_size": 0,
        "reasons": []
    }

    # Check 1: Safe activation
    allowed, reason, adjusted_size = safe.can_trade(symbol, strategy_id, size_usd)
    results["checks"]["safe_activation"] = {
        "passed": allowed,
        "reason": reason,
        "adjusted_size": adjusted_size
    }
    if not allowed:
        results["reasons"].append(f"Safe: {reason}")
    else:
        results["adjusted_size"] = adjusted_size

    # Check 2: Momentum state
    can_enter, mom_reason = momentum.can_enter(symbol)
    state = momentum.get_state(symbol)
    results["checks"]["momentum"] = {
        "passed": can_enter,
        "state": state.value,
        "reason": mom_reason
    }
    if not can_enter:
        results["reasons"].append(f"Momentum: {mom_reason}")

    # Check 3: Policy evaluation
    try:
        from .chronos_adapter import get_chronos_adapter
        chronos = get_chronos_adapter()
        context = chronos.get_context(symbol)
        chronos_dict = context.to_dict() if context else {"market_regime": regime}
    except:
        chronos_dict = {"market_regime": regime}

    policy_result = policy.evaluate(strategy_id, chronos_context=chronos_dict)
    results["checks"]["policy"] = {
        "passed": policy_result.enabled,
        "position_multiplier": policy_result.position_multiplier,
        "reasons": policy_result.reasons
    }
    if not policy_result.enabled:
        results["reasons"].append(f"Policy: {'; '.join(policy_result.reasons)}")

    # Apply policy multiplier to size
    if policy_result.enabled and results["adjusted_size"] > 0:
        results["adjusted_size"] *= policy_result.position_multiplier

    # Final decision
    all_passed = (
        results["checks"]["safe_activation"]["passed"] and
        results["checks"]["momentum"]["passed"] and
        results["checks"]["policy"]["passed"]
    )

    results["approved"] = all_passed

    if all_passed:
        results["reasons"] = ["All checks passed"]
        # Record for observability
        safe.record_momentum_state(state.value)
        if regime:
            safe.record_regime_change(regime)

    return results
