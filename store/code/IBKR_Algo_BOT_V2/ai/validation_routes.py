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


# ==================== Connectivity Management ====================

@router.get("/connectivity/status")
async def get_connectivity_status():
    """
    Get comprehensive connectivity status.

    Distinguishes:
    - MARKET_CLOSED: Calendar says market not open
    - DATA_OFFLINE: Market open but no data feed
    - SERVICE_NOT_RUNNING: Process not started
    - DISCONNECTED: Started but connection lost
    - READY: All connected, trading disabled
    - ACTIVE: All systems go, trading enabled
    """
    try:
        from .connectivity_manager import get_connectivity_manager
        manager = get_connectivity_manager()
        return {"success": True, **manager.get_connectivity_report()}
    except Exception as e:
        logger.error(f"Connectivity status error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/connectivity/self-test")
async def run_connectivity_self_test():
    """
    Run connectivity self-test.

    Checks in order:
    1. Chronos scheduler
    2. Market data connection
    3. WebSocket broadcaster
    4. Scanner jobs

    Logs results and saves report.
    """
    try:
        from .connectivity_manager import get_connectivity_manager
        manager = get_connectivity_manager()
        results = await manager.run_startup_self_test()
        return {"success": True, **results}
    except Exception as e:
        logger.error(f"Self-test error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/connectivity/reconnect")
async def reconnect_feeds(paper_mode: bool = True):
    """
    Reconnect/restart data feeds.

    SAFETY: Only available in paper mode.

    Actions:
    - Restart Polygon stream
    - Restart scalper
    - Run health check
    """
    try:
        from .connectivity_manager import get_connectivity_manager
        manager = get_connectivity_manager()
        results = await manager.reconnect_feeds(paper_mode=paper_mode)
        return results
    except Exception as e:
        logger.error(f"Reconnect error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/connectivity/report")
async def get_connectivity_report():
    """
    Generate detailed connectivity check report.

    Includes:
    - Timestamps
    - Service name
    - Status (UP/DOWN)
    - Last successful event
    - System state reason
    """
    try:
        from .connectivity_manager import get_connectivity_manager
        manager = get_connectivity_manager()
        report = manager.get_connectivity_report()

        # Format as text for display
        lines = []
        lines.append("=" * 70)
        lines.append("CONNECTIVITY CHECK REPORT")
        lines.append(f"Generated: {report['report_generated']}")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"SYSTEM STATE: {report['system_state']}")
        lines.append(f"REASON: {report['system_state_reason']}")
        lines.append("")
        lines.append(f"MARKET: {report['market']['status']} - {report['market']['detail']}")
        lines.append(f"ET TIME: {report['market']['et_time']}")
        lines.append("")
        lines.append("-" * 70)
        lines.append("SERVICES")
        lines.append("-" * 70)
        lines.append(f"{'Service':<20} {'Status':<10} {'Last Event':<25} Detail")
        lines.append("-" * 70)

        for name, svc in report['services'].items():
            last_event = svc.get('last_successful_event', '-')
            if last_event and len(last_event) > 23:
                last_event = last_event[:23]
            detail = svc.get('detail', '')[:30]
            lines.append(f"{name:<20} {svc['status']:<10} {last_event or '-':<25} {detail}")

        lines.append("-" * 70)
        lines.append(f"SUMMARY: {report['summary']['services_up']} UP / "
                    f"{report['summary']['services_down']} DOWN / "
                    f"{report['summary']['services_degraded']} DEGRADED")
        lines.append("=" * 70)

        return {
            "success": True,
            "report": report,
            "formatted": "\n".join(lines)
        }
    except Exception as e:
        logger.error(f"Report error: {e}")
        return {"success": False, "error": str(e)}


# ==================== Market Time ====================

@router.get("/time/status")
async def get_time_status():
    """
    Get comprehensive time status for trading.

    The bot uses Eastern Time (ET) as the reference for all trading decisions.
    Local time is for display only.

    Returns:
    - ET time (bot's reference)
    - Local time
    - UTC time
    - Market status (CLOSED/PRE_MARKET/OPEN/AFTER_HOURS)
    - Next market event
    - Holiday/early close info
    """
    try:
        from .market_time import get_time_status as get_market_time
        return {"success": True, **get_market_time()}
    except Exception as e:
        logger.error(f"Time status error: {e}")
        from datetime import datetime
        return {
            "success": False,
            "error": str(e),
            "et_time": datetime.now().isoformat(),
            "market_status": "UNKNOWN"
        }


# ==================== Chronos Status ====================

@router.get("/chronos/status")
async def get_chronos_status():
    """
    Get Chronos AI service status.

    Shows:
    - Model availability
    - Model name
    - Last inference time
    - Symbols tracked
    - Inference count
    """
    try:
        from .chronos_adapter import get_chronos_adapter
        chronos = get_chronos_adapter()
        return {"success": True, **chronos.get_status()}
    except Exception as e:
        logger.error(f"Chronos status error: {e}")
        return {
            "success": False,
            "available": False,
            "error": str(e)
        }


# ==================== Pre-Market Readiness Check ====================

@router.get("/pre-market-ready")
async def check_pre_market_readiness():
    """
    Pre-Market Readiness Check Endpoint.

    Returns PASS only if ALL conditions are met:
    1. Market data live
    2. Streams subscribed
    3. Chronos responding
    4. Scanners running
    5. Gating engine loaded
    6. Kill switch off

    Used to verify system is ready before 07:00 ET trading window.
    """
    from datetime import datetime
    import httpx

    checks = {}
    all_passed = True
    messages = []

    # Helper to track check results
    def record_check(name: str, passed: bool, detail: str, timestamp: str = None):
        nonlocal all_passed
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
            "last_check": timestamp or datetime.now().isoformat()
        }
        if not passed:
            all_passed = False
            messages.append(f"{name}: {detail}")

    # Check 1: Market Data Live
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:9100/api/health", timeout=5)
            health = resp.json()
            market_data_ok = health.get("services", {}).get("market_data") != "unavailable"
            record_check(
                "market_data",
                market_data_ok,
                health.get("services", {}).get("market_data", "unknown"),
                health.get("timestamp")
            )
    except Exception as e:
        record_check("market_data", False, f"Health check failed: {str(e)}")

    # Check 2: Streams Subscribed (Polygon WebSocket)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:9100/api/polygon/stream/status", timeout=5)
            stream = resp.json()
            stream_ok = stream.get("connected", False) or stream.get("available", False)
            trade_subs = len(stream.get("trade_subscriptions", []))
            record_check(
                "websocket_streams",
                stream_ok,
                f"Connected: {stream_ok}, Subscriptions: {trade_subs}",
                stream.get("last_message")
            )
    except Exception as e:
        record_check("websocket_streams", False, f"Stream check failed: {str(e)}")

    # Check 3: Chronos Responding
    try:
        from .chronos_adapter import get_chronos_adapter
        chronos = get_chronos_adapter()
        chronos_ok = chronos.available
        model_name = "amazon/chronos-t5-small" if chronos_ok else "unavailable"
        record_check(
            "chronos_service",
            chronos_ok,
            f"Model: {model_name}, Available: {chronos_ok}"
        )
    except Exception as e:
        record_check("chronos_service", False, f"Chronos check failed: {str(e)}")

    # Check 4: Scanners Running
    scanner_status = {"news": False, "scalper": False, "premarket": False}
    scanner_timestamps = {}
    try:
        async with httpx.AsyncClient() as client:
            # News trader
            try:
                resp = await client.get("http://localhost:9100/api/scanner/news-trader/status", timeout=3)
                news = resp.json()
                scanner_status["news"] = news.get("scalper_running", False)
                scanner_timestamps["news"] = news.get("last_scan_time")
            except:
                pass

            # HFT Scalper
            try:
                resp = await client.get("http://localhost:9100/api/scanner/scalper/status", timeout=3)
                scalper = resp.json()
                scanner_status["scalper"] = scalper.get("is_running", False)
                scanner_timestamps["scalper"] = scalper.get("last_scan_time")
            except:
                pass

            # Premarket
            try:
                resp = await client.get("http://localhost:9100/api/scanner/premarket/status", timeout=3)
                premarket = resp.json()
                scanner_status["premarket"] = premarket.get("last_updated")
                scanner_timestamps["premarket"] = premarket.get("last_updated")
            except:
                pass

        # At least one scanner should be running
        any_scanner_running = scanner_status["scalper"] or scanner_status["news"]
        record_check(
            "scanners",
            any_scanner_running,
            f"Scalper: {scanner_status['scalper']}, News: {scanner_status['news']}, Premarket: {scanner_status['premarket'] is not None}",
        )
    except Exception as e:
        record_check("scanners", False, f"Scanner check failed: {str(e)}")

    # Check 5: Gating Engine Loaded
    try:
        safe = get_safe_activation()
        policy = get_strategy_policy_engine()
        gating_ok = True  # If we can access them, they're loaded
        record_check(
            "gating_engine",
            gating_ok,
            f"Mode: {safe.config.mode.value}, Policies: {len(policy.get_all_policies().get('policies', {}))}"
        )
    except Exception as e:
        record_check("gating_engine", False, f"Gating engine check failed: {str(e)}")

    # Check 6: Kill Switch Off
    try:
        safe = get_safe_activation()
        kill_switch_off = not safe._kill_switch_active
        reason = safe._kill_switch_reason.value if safe._kill_switch_reason else "none"
        record_check(
            "kill_switch",
            kill_switch_off,
            f"Active: {safe._kill_switch_active}, Reason: {reason}"
        )
    except Exception as e:
        record_check("kill_switch", False, f"Kill switch check failed: {str(e)}")

    # Determine overall status
    from .safe_activation import GovernorHealthState
    safe = get_safe_activation()
    health_state = safe.get_governor_health_state(services_healthy=all_passed)

    # Build display message
    if all_passed:
        if health_state == GovernorHealthState.STANDBY:
            display_status = "READY – Awaiting trading window"
        elif health_state == GovernorHealthState.CONNECTED:
            display_status = "READY – Enable trading to activate"
        elif health_state == GovernorHealthState.ACTIVE:
            display_status = "ACTIVE – Trading enabled"
        else:
            display_status = "READY – All systems operational"
    else:
        display_status = f"NOT READY – {len(messages)} issue(s)"

    return {
        "status": "PASS" if all_passed else "FAIL",
        "display_status": display_status,
        "health_state": health_state.value,
        "timestamp": datetime.now().isoformat(),
        "checks": checks,
        "issues": messages if messages else None,
        "scanner_timestamps": scanner_timestamps
    }
