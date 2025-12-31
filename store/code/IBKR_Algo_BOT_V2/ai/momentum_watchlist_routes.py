"""
Momentum Watchlist Operator Control Routes
===========================================

API endpoints for operator oversight of the momentum watchlist.

Key Principles:
1. Purge removes symbols - does NOT approve trades
2. Refresh re-evaluates - does NOT force adds
3. All actions are logged to R0_operator_actions.json
4. Full transparency on rank, metrics, inclusion/exclusion
"""

import logging
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .momentum_watchlist import (
    get_momentum_watchlist,
    reset_momentum_watchlist,
    WatchlistConfig,
    ExclusionReason,
    get_rel_vol_floor
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/watchlist", tags=["watchlist"])

# Operator actions log file
OPERATOR_LOG_PATH = Path("reports/R0_operator_actions.json")


def _log_operator_action(action: Dict):
    """Persist operator action to R0_operator_actions.json"""
    try:
        actions = []
        if OPERATOR_LOG_PATH.exists():
            with open(OPERATOR_LOG_PATH, 'r') as f:
                data = json.load(f)
                actions = data.get("actions", [])

        actions.append(action)

        # Keep last 1000 actions
        if len(actions) > 1000:
            actions = actions[-1000:]

        OPERATOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OPERATOR_LOG_PATH, 'w') as f:
            json.dump({
                "last_updated": datetime.now().isoformat(),
                "action_count": len(actions),
                "actions": actions
            }, f, indent=2)

    except Exception as e:
        logger.error(f"Failed to log operator action: {e}")


class ConfigUpdate(BaseModel):
    max_active_symbols: Optional[int] = None
    min_rel_vol_floor: Optional[float] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_gap_pct: Optional[float] = None


# =============================================================================
# OPERATOR ENDPOINTS
# =============================================================================

@router.get("/status")
async def get_watchlist_status():
    """
    Get current watchlist status with full transparency.

    Returns:
        - Session date
        - Cycle count
        - Active symbols with rank + metrics
        - Excluded symbols with reasons
        - Configuration
    """
    watchlist = get_momentum_watchlist()
    status = watchlist.get_status()

    # Add detailed active list
    status["active_watchlist"] = watchlist.get_active_watchlist()

    return status


@router.get("/active")
async def get_active_symbols():
    """
    Get currently active symbols with full transparency.

    Each symbol includes:
        - rank: Current rank in dominance order
        - dominance_score: Computed score
        - gap_pct, rel_vol, price: Current metrics
        - exclusion_reason: Why excluded (if not active)
    """
    watchlist = get_momentum_watchlist()
    return {
        "symbols": watchlist.get_active_watchlist(),
        "count": len(watchlist.active_symbols),
        "session_date": watchlist.session_date.isoformat()
    }


@router.get("/symbol/{symbol}")
async def get_symbol_status(symbol: str):
    """
    Get detailed status for a specific symbol.

    Returns:
        - Current metrics
        - Rank (if in active list)
        - Exclusion reason (if excluded)
        - Data quality flags
    """
    watchlist = get_momentum_watchlist()
    status = watchlist.get_symbol_status(symbol.upper())

    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol {symbol} not found in current session"
        )

    return status


@router.post("/purge")
async def purge_watchlist(triggered_by: str = "operator"):
    """
    PURGE ALL WATCHLIST

    - Removes all ACTIVE symbols
    - Resets discovery cache
    - Preserves historical reports
    - Does NOT approve trades or alter thresholds

    Args:
        triggered_by: Who/what triggered the purge (for audit)

    Returns:
        Action log entry
    """
    watchlist = get_momentum_watchlist()
    action = watchlist.purge_all(triggered_by=triggered_by)

    # Log to persistent file
    _log_operator_action(action)

    logger.warning(f"OPERATOR PURGE: All watchlist cleared by {triggered_by}")

    return {
        "status": "success",
        "message": f"Purged {len(action.get('symbols_before', []))} symbols",
        "action": action
    }


@router.delete("/{symbol}")
async def delete_symbol(symbol: str, triggered_by: str = "operator"):
    """
    DELETE ONE SYMBOL from watchlist.

    - Removes symbol from active watchlist
    - Archives the symbol
    - Logs the action

    Args:
        symbol: Symbol to remove
        triggered_by: Who/what triggered the deletion

    Returns:
        Action log entry
    """
    watchlist = get_momentum_watchlist()
    symbol = symbol.upper()

    # Check if symbol exists
    if symbol not in watchlist.active_symbols:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol {symbol} not in active watchlist"
        )

    # Remove from active watchlist
    now = datetime.now()
    removed = None
    new_active = []
    for s in watchlist._active_watchlist:
        if s.symbol == symbol:
            s.is_active = False
            s.exclusion_reason = ExclusionReason.MANUAL_EXCLUSION
            watchlist._archived.append(s)
            removed = s
        else:
            new_active.append(s)

    watchlist._active_watchlist = new_active

    # Also remove from candidates
    if symbol in watchlist._all_candidates:
        del watchlist._all_candidates[symbol]

    action = {
        "action_type": "delete_symbol",
        "timestamp": now.isoformat(),
        "symbol": symbol,
        "triggered_by": triggered_by,
        "session_date": watchlist.session_date.isoformat()
    }
    watchlist._operator_actions.append(action)
    _log_operator_action(action)

    logger.warning(f"OPERATOR DELETE: Removed {symbol} by {triggered_by}")

    return {
        "status": "success",
        "message": f"Deleted {symbol} from watchlist",
        "action": action
    }


@router.post("/refresh")
async def refresh_watchlist():
    """
    REFRESH WATCHLIST

    Forces a fresh session start:
    - Clears current session data
    - Next R1 cycle will start fresh
    - Does NOT add symbols or bypass thresholds

    Returns:
        Status confirmation
    """
    # Reset the singleton to force fresh start
    reset_momentum_watchlist()

    action = {
        "action_type": "refresh",
        "timestamp": datetime.now().isoformat(),
        "triggered_by": "operator"
    }
    _log_operator_action(action)

    logger.info("OPERATOR REFRESH: Watchlist reset for fresh session")

    return {
        "status": "success",
        "message": "Watchlist refreshed - next cycle will start fresh",
        "action": action
    }


@router.get("/config")
async def get_watchlist_config():
    """Get current watchlist configuration"""
    watchlist = get_momentum_watchlist()
    return watchlist.config.to_dict()


@router.post("/config")
async def update_watchlist_config(update: ConfigUpdate):
    """
    Update watchlist configuration.

    Allowed updates:
        - max_active_symbols: Top N cutoff
        - min_rel_vol_floor: Relative volume floor (0.30 = 30%)
        - min_price / max_price: Price range filter
        - min_gap_pct: Minimum gap threshold
    """
    watchlist = get_momentum_watchlist()

    if update.max_active_symbols is not None:
        watchlist.config.max_active_symbols = update.max_active_symbols
    if update.min_rel_vol_floor is not None:
        watchlist.config.min_rel_vol_floor = update.min_rel_vol_floor
    if update.min_price is not None:
        watchlist.config.min_price = update.min_price
    if update.max_price is not None:
        watchlist.config.max_price = update.max_price
    if update.min_gap_pct is not None:
        watchlist.config.min_gap_pct = update.min_gap_pct

    action = {
        "action_type": "config_update",
        "timestamp": datetime.now().isoformat(),
        "new_config": watchlist.config.to_dict(),
        "triggered_by": "operator"
    }
    _log_operator_action(action)

    return {
        "status": "success",
        "config": watchlist.config.to_dict()
    }


@router.get("/actions")
async def get_operator_actions(limit: int = 50):
    """
    Get operator action log.

    Returns:
        List of recent operator actions (purge, refresh, config changes)
    """
    watchlist = get_momentum_watchlist()
    in_memory_actions = watchlist.get_operator_actions()

    # Also load from persistent file
    persisted_actions = []
    if OPERATOR_LOG_PATH.exists():
        try:
            with open(OPERATOR_LOG_PATH, 'r') as f:
                data = json.load(f)
                persisted_actions = data.get("actions", [])
        except Exception as e:
            logger.error(f"Failed to load operator log: {e}")

    # Combine and dedupe by timestamp
    all_actions = {}
    for a in persisted_actions + in_memory_actions:
        key = a.get("timestamp", "")
        all_actions[key] = a

    # Sort by timestamp descending
    sorted_actions = sorted(
        all_actions.values(),
        key=lambda x: x.get("timestamp", ""),
        reverse=True
    )

    return {
        "actions": sorted_actions[:limit],
        "total_count": len(sorted_actions)
    }


@router.get("/rank-snapshot")
async def get_rank_snapshot():
    """
    Get the latest rank snapshot.

    Shows:
        - How all symbols were ranked
        - Who made the cut (active)
        - Who was excluded and why
        - Proves full recompute happened
    """
    # This is embedded in R1 output, but provide direct access
    reports_dir = Path("reports")
    r1_path = reports_dir / "report_R1_daily_top_gappers.json"

    if not r1_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No R1 report found - run pipeline first"
        )

    try:
        with open(r1_path, 'r') as f:
            r1_data = json.load(f)

        rank_snapshot = r1_data.get("rank_snapshot", {})
        if not rank_snapshot:
            raise HTTPException(
                status_code=404,
                detail="R1 report exists but has no rank_snapshot - old format?"
            )

        return rank_snapshot

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse R1 report: {e}"
        )


@router.get("/exclusions")
async def get_exclusion_summary():
    """
    Get summary of all excluded symbols with reasons.

    This shows why symbols were NOT in the active watchlist:
        - REL_VOL_BELOW_FLOOR: Failed 30% relative volume requirement
        - RANK_BELOW_CUTOFF: Ranked below top N
        - PRICE_OUT_OF_RANGE: Outside $1-$20 range
        - DATA_QUALITY_ISSUE: Missing or invalid data
    """
    reports_dir = Path("reports")
    r1_path = reports_dir / "report_R1_daily_top_gappers.json"

    if not r1_path.exists():
        return {
            "message": "No R1 report found",
            "exclusions": []
        }

    try:
        with open(r1_path, 'r') as f:
            r1_data = json.load(f)

        rank_snapshot = r1_data.get("rank_snapshot", {})

        return {
            "excluded": rank_snapshot.get("excluded", []),
            "exclusion_summary": rank_snapshot.get("exclusion_summary", {}),
            "below_cutoff": rank_snapshot.get("below_cutoff", []),
            "timestamp": r1_data.get("timestamp")
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read exclusions: {e}"
        )


@router.get("/entry-windows")
async def get_entry_window_log(limit: int = 50):
    """
    Get ENTRY_WINDOW events log for validation.

    This shows every time a symbol reached the ENTRY_WINDOW state
    (first pullback reclaim). Use this to validate the state machine
    against actual market behavior.

    Returns:
        - Total events
        - Symbol counts
        - Recent events (most recent first)
    """
    log_path = Path("reports/entry_window_log.json")

    if not log_path.exists():
        return {
            "message": "No entry window events yet",
            "total_events": 0,
            "events": []
        }

    try:
        with open(log_path, 'r') as f:
            log_data = json.load(f)

        events = log_data.get("events", [])
        # Return most recent first
        events_reversed = list(reversed(events))[:limit]

        return {
            "last_updated": log_data.get("last_updated"),
            "total_events": log_data.get("total_events", 0),
            "symbol_counts": log_data.get("symbol_counts", {}),
            "events": events_reversed
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read entry window log: {e}"
        )


@router.delete("/entry-windows")
async def clear_entry_window_log():
    """
    Clear the entry window log.
    Use at start of session for clean validation.
    """
    log_path = Path("reports/entry_window_log.json")

    if log_path.exists():
        log_path.unlink()

    return {
        "status": "success",
        "message": "Entry window log cleared"
    }
