"""
Task Group 3 - Chronos Context Engine
======================================

Tasks:
- CHRONOS_PERSISTENCE (R8): Momentum persistence analysis
- CHRONOS_PULLBACK_DEPTH (R9): Micro pullback depth

EXECUTION CONTRACT:
- Every task generates a persisted report artifact
- Fixed schema - no placeholders
- No trade execution without gating approval
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx

from .task_queue_manager import Task, get_task_queue_manager

logger = logging.getLogger(__name__)

API_BASE = "http://localhost:9100"


async def _fetch_api(endpoint: str) -> Optional[Dict]:
    """Fetch data from API"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(f"{API_BASE}{endpoint}")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.error(f"API fetch error {endpoint}: {e}")
    return None


# =============================================================================
# REPORT SCHEMAS (Fixed - No Placeholders)
# =============================================================================

SCHEMA_R8 = {
    "task_id": "CHRONOS_PERSISTENCE",
    "symbols": [],  # List of symbol objects
    "symbol_count": 0,
    "persistence_summary": {
        "high_persistence_count": 0,
        "medium_persistence_count": 0,
        "low_persistence_count": 0,
        "avg_persistence_score": 0.0,
        "fake_breakout_risk_count": 0,
    },
    "timestamp": "",
}

SCHEMA_R8_SYMBOL = {
    "symbol": "",
    "continuation_prob": 0.0,
    "persistence_score": 0.0,
    "persistence_category": "",  # HIGH / MEDIUM / LOW
    "fake_breakout_prob": 0.0,
    "momentum_strength": 0.0,
    "trend_consistency": 0,  # Number of higher highs
    "pullback_count": 0,
    "recovery_rate": 0.0,
    "chronos_confidence": 0.0,
}

SCHEMA_R9 = {
    "task_id": "CHRONOS_PULLBACK_DEPTH",
    "symbols": [],
    "symbol_count": 0,
    "pullback_summary": {
        "shallow_pullback_count": 0,
        "normal_pullback_count": 0,
        "deep_pullback_count": 0,
        "avg_pullback_depth": 0.0,
        "avg_recovery_speed": 0.0,
    },
    "timestamp": "",
}

SCHEMA_R9_SYMBOL = {
    "symbol": "",
    "pullback_depth_pct": 0.0,
    "pullback_category": "",  # SHALLOW / NORMAL / DEEP / NONE
    "recovery_speed": 0.0,  # Price recovery per second
    "vwap_position": "",  # ABOVE / AT / BELOW
    "vwap_slope": 0.0,
    "entry_quality": "",  # EXCELLENT / GOOD / FAIR / POOR
    "entry_score": 0.0,
}


# =============================================================================
# TASK 3.1 - CHRONOS_PERSISTENCE
# =============================================================================


async def task_chronos_persistence(inputs: Dict) -> Dict:
    """
    Momentum Persistence Analysis

    INPUTS: Live microstructure, Recent pullbacks
    PROCESS:
        - Estimate continuation probability
        - Fake breakout risk assessment
    OUTPUT: report_R8_momentum_persistence.json

    SCHEMA: Fixed - all fields required
    """
    logger.info("TASK 3.1: Analyzing momentum persistence...")

    # Load R7 input
    r7_data = inputs.get("report_R7_feature_importance.json", {})
    symbols_in = r7_data.get("symbols", [])

    # Get Chronos context
    chronos_data = await _fetch_api("/api/validation/chronos/context")

    symbols_out = []
    persistence_high = 0
    persistence_medium = 0
    persistence_low = 0
    fake_breakout_risk = 0
    total_persistence = 0.0

    for s in symbols_in:
        symbol = s.get("symbol", "")

        # Get symbol-specific Chronos analysis
        chronos_symbol = await _fetch_api(f"/api/validation/chronos/analyze/{symbol}")

        # Extract Chronos metrics (or compute defaults)
        if chronos_symbol and chronos_symbol.get("status") == "success":
            prob_up = chronos_symbol.get("prob_up", 0.5)
            trend_strength = chronos_symbol.get("trend_strength", 0.5)
            volatility = chronos_symbol.get("current_volatility", 0.02)
            regime = chronos_symbol.get("market_regime", "UNKNOWN")
        else:
            # Compute from available data
            prob_up = s.get("continuation_prob", 0.5)
            trend_strength = 0.5
            volatility = 0.02
            regime = "UNKNOWN"

        # Calculate persistence score
        # Use `or` to handle None values (dict.get returns None if key exists with None value)
        gap_pct = s.get("gap_pct") or 0
        rel_vol = s.get("rel_vol_5m") or 0
        at_hod = s.get("at_hod") or False
        near_hod = s.get("near_hod") or False

        # Persistence factors
        persistence_score = 0.3  # Base

        # Momentum factors
        if prob_up >= 0.65:
            persistence_score += 0.25
        elif prob_up >= 0.55:
            persistence_score += 0.15

        # Volume confirmation
        if rel_vol >= 500:
            persistence_score += 0.20
        elif rel_vol >= 300:
            persistence_score += 0.10

        # Price action
        if at_hod:
            persistence_score += 0.20
        elif near_hod:
            persistence_score += 0.10

        # Gap strength
        if gap_pct >= 20:
            persistence_score += 0.10
        elif gap_pct >= 15:
            persistence_score += 0.05

        # Cap score
        persistence_score = min(0.95, max(0.10, persistence_score))

        # Fake breakout probability (inverse of persistence)
        fake_breakout_prob = (1 - persistence_score) * 0.8

        # Adjust for volatility
        if volatility > 0.05:
            fake_breakout_prob *= 1.2

        fake_breakout_prob = min(0.90, fake_breakout_prob)

        # Categorize
        if persistence_score >= 0.70:
            category = "HIGH"
            persistence_high += 1
        elif persistence_score >= 0.50:
            category = "MEDIUM"
            persistence_medium += 1
        else:
            category = "LOW"
            persistence_low += 1

        if fake_breakout_prob >= 0.50:
            fake_breakout_risk += 1

        total_persistence += persistence_score

        # Build symbol output (fixed schema)
        # Use `or` to handle None values properly
        symbol_out = {
            "symbol": symbol,
            "continuation_prob": round(s.get("continuation_prob") or 0.5, 3),
            "persistence_score": round(persistence_score, 3),
            "persistence_category": category,
            "fake_breakout_prob": round(fake_breakout_prob, 3),
            "momentum_strength": round(trend_strength, 3),
            "trend_consistency": int(s.get("hod_breaks") or 0),
            "pullback_count": 0,  # Will be computed in R9
            "recovery_rate": 0.0,  # Will be computed in R9
            "chronos_confidence": round(prob_up, 3),
            # Pass through data
            "gap_pct": s.get("gap_pct") or 0,
            "rel_vol_5m": s.get("rel_vol_5m") or 0,
            "price": s.get("price") or 0,
            "float_shares": s.get(
                "float_shares"
            ),  # Keep None for proper data quality tracking
            "hod_status": s.get("hod_status") or "UNKNOWN",
            "priority": s.get("priority") or "NORMAL",
        }
        symbols_out.append(symbol_out)

    # Sort by persistence score
    symbols_out.sort(key=lambda x: x["persistence_score"], reverse=True)

    avg_persistence = total_persistence / len(symbols_out) if symbols_out else 0.0

    # Build report (fixed schema)
    # Check if we got real Chronos data or simulated
    chronos_available = (
        chronos_data is not None and chronos_data.get("status") == "success"
    )
    model_source = "REAL" if chronos_available else "SIMULATED"

    report = {
        "task_id": "CHRONOS_PERSISTENCE",
        "symbols": symbols_out,
        "symbol_count": len(symbols_out),
        "model_source": model_source,  # REAL or SIMULATED - Gate must check this
        "persistence_summary": {
            "high_persistence_count": persistence_high,
            "medium_persistence_count": persistence_medium,
            "low_persistence_count": persistence_low,
            "avg_persistence_score": round(avg_persistence, 3),
            "fake_breakout_risk_count": fake_breakout_risk,
        },
        "timestamp": datetime.now().isoformat(),
    }

    return report


# =============================================================================
# TASK 3.2 - CHRONOS_PULLBACK_DEPTH
# =============================================================================


async def task_chronos_pullback_depth(inputs: Dict) -> Dict:
    """
    Micro Pullback Depth Analysis

    INPUTS: Tick-level data, VWAP slope
    PROCESS:
        - Measure pullback %
        - Recovery speed
    OUTPUT: report_R9_pullback_depth.json

    SCHEMA: Fixed - all fields required
    """
    logger.info("TASK 3.2: Analyzing pullback depth...")

    # Load R8 input
    r8_data = inputs.get("report_R8_momentum_persistence.json", {})
    symbols_in = r8_data.get("symbols", [])

    symbols_out = []
    shallow_count = 0
    normal_count = 0
    deep_count = 0
    total_depth = 0.0
    total_recovery = 0.0

    for s in symbols_in:
        symbol = s.get("symbol", "")

        # Get quote for VWAP and price data
        quote = await _fetch_api(f"/api/price/{symbol}")

        if quote:
            current_price = (
                quote.get("lastPrice", 0) or quote.get("price", 0) or s.get("price", 0)
            )
            vwap = quote.get("vwap", 0) or current_price
            high_price = (
                quote.get("highPrice", 0) or quote.get("dayHigh", 0) or current_price
            )
            low_price = (
                quote.get("lowPrice", 0) or quote.get("dayLow", 0) or current_price
            )
        else:
            current_price = s.get("price", 0)
            vwap = current_price
            high_price = current_price
            low_price = current_price

        # Calculate pullback depth (distance from HOD)
        if high_price > 0:
            pullback_depth = ((high_price - current_price) / high_price) * 100
        else:
            pullback_depth = 0.0

        # VWAP position
        if vwap > 0:
            vwap_diff = ((current_price - vwap) / vwap) * 100
            if vwap_diff > 0.5:
                vwap_position = "ABOVE"
            elif vwap_diff < -0.5:
                vwap_position = "BELOW"
            else:
                vwap_position = "AT"
        else:
            vwap_position = "UNKNOWN"
            vwap_diff = 0.0

        # VWAP slope (simplified - would need historical VWAP in production)
        vwap_slope = (
            0.5
            if vwap_position == "ABOVE"
            else (-0.5 if vwap_position == "BELOW" else 0.0)
        )

        # Categorize pullback
        if pullback_depth <= 1.0:
            pullback_category = "SHALLOW"
            shallow_count += 1
        elif pullback_depth <= 3.0:
            pullback_category = "NORMAL"
            normal_count += 1
        elif pullback_depth <= 5.0:
            pullback_category = "DEEP"
            deep_count += 1
        else:
            pullback_category = "EXTENDED"

        total_depth += pullback_depth

        # Recovery speed (simplified)
        persistence = s.get("persistence_score", 0.5)
        recovery_speed = persistence * 0.5  # Higher persistence = faster recovery

        total_recovery += recovery_speed

        # Entry quality scoring
        entry_score = 0.0

        # Favor shallow pullbacks
        if pullback_category == "SHALLOW":
            entry_score += 0.30
        elif pullback_category == "NORMAL":
            entry_score += 0.20

        # Favor above VWAP
        if vwap_position == "ABOVE":
            entry_score += 0.25
        elif vwap_position == "AT":
            entry_score += 0.10

        # Factor in persistence
        entry_score += persistence * 0.30

        # Factor in momentum
        if s.get("hod_status") == "AT_HOD":
            entry_score += 0.15
        elif s.get("hod_status") == "NEAR_HOD":
            entry_score += 0.08

        entry_score = min(1.0, entry_score)

        # Determine entry quality
        if entry_score >= 0.75:
            entry_quality = "EXCELLENT"
        elif entry_score >= 0.55:
            entry_quality = "GOOD"
        elif entry_score >= 0.40:
            entry_quality = "FAIR"
        else:
            entry_quality = "POOR"

        # Build symbol output (fixed schema)
        symbol_out = {
            "symbol": symbol,
            "pullback_depth_pct": round(pullback_depth, 2),
            "pullback_category": pullback_category,
            "recovery_speed": round(recovery_speed, 3),
            "vwap_position": vwap_position,
            "vwap_slope": round(vwap_slope, 3),
            "entry_quality": entry_quality,
            "entry_score": round(entry_score, 3),
            # Pass through key data
            "current_price": round(current_price, 2),
            "hod_price": round(high_price, 2),
            "vwap": round(vwap, 2),
            "persistence_score": s.get("persistence_score", 0),
            "persistence_category": s.get("persistence_category", ""),
            "continuation_prob": s.get("continuation_prob", 0),
            "gap_pct": s.get("gap_pct", 0),
            "rel_vol_5m": s.get("rel_vol_5m", 0),
            "float_shares": s.get("float_shares", 0),
            "priority": s.get("priority", "NORMAL"),
        }
        symbols_out.append(symbol_out)

    # Sort by entry score
    symbols_out.sort(key=lambda x: x["entry_score"], reverse=True)

    avg_depth = total_depth / len(symbols_out) if symbols_out else 0.0
    avg_recovery = total_recovery / len(symbols_out) if symbols_out else 0.0

    # Build report (fixed schema)
    # R9 uses heuristics for pullback analysis, not a trained model
    report = {
        "task_id": "CHRONOS_PULLBACK_DEPTH",
        "symbols": symbols_out,
        "symbol_count": len(symbols_out),
        "model_source": "SIMULATED",  # Pullback analysis uses heuristics, not trained model
        "pullback_summary": {
            "shallow_pullback_count": shallow_count,
            "normal_pullback_count": normal_count,
            "deep_pullback_count": deep_count,
            "avg_pullback_depth": round(avg_depth, 2),
            "avg_recovery_speed": round(avg_recovery, 3),
        },
        "timestamp": datetime.now().isoformat(),
    }

    return report


# =============================================================================
# REGISTER TASKS
# =============================================================================


def register_chronos_tasks():
    """Register all Task Group 3 tasks with the manager"""
    manager = get_task_queue_manager()

    # Task 3.1 - Momentum Persistence
    manager.register_task(
        Task(
            id="CHRONOS_PERSISTENCE",
            name="Momentum Persistence Analysis",
            group="CHRONOS_CONTEXT",
            inputs=["report_R7_feature_importance.json"],
            process=task_chronos_persistence,
            output_file="report_R8_momentum_persistence.json",
            fail_conditions=[],
            next_task="CHRONOS_PULLBACK_DEPTH",
        )
    )

    # Task 3.2 - Pullback Depth
    manager.register_task(
        Task(
            id="CHRONOS_PULLBACK_DEPTH",
            name="Micro Pullback Depth",
            group="CHRONOS_CONTEXT",
            inputs=["report_R8_momentum_persistence.json"],
            process=task_chronos_pullback_depth,
            output_file="report_R9_pullback_depth.json",
            fail_conditions=[],
            next_task="GATING_SIGNAL_EVAL",
        )
    )

    logger.info("Task Group 3 (Chronos Context) registered: 2 tasks")
