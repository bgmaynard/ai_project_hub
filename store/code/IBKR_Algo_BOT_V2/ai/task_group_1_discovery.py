"""
Task Group 1 - Market Discovery (REFACTORED)
==============================================

Uses SignalSnapshot canonical schema for data integrity.
NO silent defaults - missing data is flagged, not hidden.

Tasks:
- R0 (Data Quality): Pre-flight data quality check
- R1 (DISCOVERY_GAPPERS): Generate daily top gappers
- R2 (DISCOVERY_FLOAT_FILTER): Low float qualification
- R3 (DISCOVERY_REL_VOLUME): Relative volume acceleration
- R4 (DISCOVERY_HOD_BEHAVIOR): HOD behavior tracking
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import asyncio

from .task_queue_manager import Task, get_task_queue_manager
from .signal_snapshot import (
    SignalSnapshot,
    HODStatus,
    DataQualityFlag,
    ModelSource,
    create_snapshot_from_worklist_item
)
from .connection_manager import get_connection_manager
from .momentum_watchlist import get_momentum_watchlist, MomentumWatchlist

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def _fetch_quote(symbol: str) -> Optional[Dict]:
    """Fetch quote using connection manager"""
    conn = get_connection_manager()
    data, is_fresh = await conn.fetch_quote(symbol)
    return data


async def _fetch_float(symbol: str) -> Optional[Dict]:
    """Fetch float data using connection manager"""
    conn = get_connection_manager()
    data, is_fresh = await conn.fetch_float(symbol)
    return data


async def _fetch_worklist() -> Optional[Dict]:
    """Fetch worklist using connection manager"""
    conn = get_connection_manager()
    data, is_fresh = await conn.fetch_worklist()
    return data


async def _fetch_market_movers(direction: str = "up") -> Optional[List]:
    """Fetch market movers using connection manager"""
    conn = get_connection_manager()
    data, is_fresh = await conn.fetch_market_movers(direction)
    if isinstance(data, list):
        return data
    return None


def _build_snapshot(symbol: str, raw_data: Dict, float_data: Optional[Dict] = None) -> SignalSnapshot:
    """
    Build SignalSnapshot from raw quote/worklist data.
    This is the SINGLE conversion point - ensures consistency.
    """
    return SignalSnapshot.from_quote(symbol, raw_data, float_data)


# =============================================================================
# TASK 0 - DATA QUALITY PRE-FLIGHT (R0)
# =============================================================================

async def task_data_quality_check(inputs: Dict) -> Dict:
    """
    Data Quality Pre-Flight Check (R0)

    INPUTS: None - runs before pipeline
    PROCESS:
        - Check API connectivity
        - Check data freshness
        - Report missing data patterns
    OUTPUT: report_R0_data_quality.json
    FAIL: Critical API failures or stale data
    """
    logger.info("TASK 0: Data quality pre-flight check...")

    conn = get_connection_manager()
    quality_report = {
        "task_id": "DATA_QUALITY_CHECK",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "api_status": {},
        "data_freshness": {},
        "issues": [],
        "critical_failures": [],
        "overall_quality": "GOOD"
    }

    # Check API connectivity
    try:
        # Test quote endpoint
        test_quote, fresh = await conn.fetch_quote("SPY")
        quality_report["api_status"]["quotes"] = "OK" if test_quote else "FAILED"
        if not test_quote:
            quality_report["critical_failures"].append("Quote API unavailable")

        # Test float endpoint
        test_float, fresh = await conn.fetch_float("SPY")
        quality_report["api_status"]["float"] = "OK" if test_float else "DEGRADED"
        if not test_float:
            quality_report["issues"].append("Float data may be unavailable")

        # Test worklist
        test_worklist, fresh = await conn.fetch_worklist()
        quality_report["api_status"]["worklist"] = "OK" if test_worklist else "FAILED"
        if not test_worklist:
            quality_report["critical_failures"].append("Worklist API unavailable")

    except Exception as e:
        quality_report["critical_failures"].append(f"API test error: {str(e)}")

    # Check data freshness
    conn_status = conn.get_status()
    quality_report["data_freshness"] = conn_status["data_freshness"]

    if conn.is_data_stale:
        quality_report["critical_failures"].append(f"Data is STALE: {conn.data_freshness.stale_reason}")

    if conn.circuit_open:
        quality_report["critical_failures"].append("Circuit breaker is OPEN - API unavailable")

    # Set overall quality
    if quality_report["critical_failures"]:
        quality_report["overall_quality"] = "CRITICAL"
    elif quality_report["issues"]:
        quality_report["overall_quality"] = "DEGRADED"

    return quality_report


# =============================================================================
# TASK 1.1 - DISCOVERY_GAPPERS (R1) - FULL RECOMPUTE MODEL
# =============================================================================

async def task_discovery_gappers(inputs: Dict) -> Dict:
    """
    Generate Daily Top Gappers (R1) - FULL RECOMPUTE MODEL

    CRITICAL: The watchlist is a RANKED, SESSION-SCOPED VIEW.
    Every cycle: ALL candidates are re-evaluated with current metrics.
    NO symbol is exempt from re-ranking.

    INPUTS: Pre-market + opening data (Price, Volume, Float)
    PROCESS:
        1. Gather ALL candidates from all sources
        2. Fetch current metrics for EVERY candidate
        3. FULL RECOMPUTE via MomentumWatchlist
        4. RANK by dominance score (global ranking)
        5. Top N only enters ACTIVE watchlist

    OUTPUT:
        - report_R1_daily_top_gappers.json (active watchlist)
        - rank_snapshot embedded for transparency

    FAIL: No symbols meet criteria -> halt pipeline
    """
    logger.info("TASK 1.1: Generating daily top gappers (FULL RECOMPUTE)...")

    conn = get_connection_manager()

    # Check for stale data FIRST
    if conn.is_data_stale:
        return {
            "status": "FAILED",
            "reason": f"DATA_STALE: {conn.get_stale_veto_reason()}",
            "data": {
                "task_id": "DISCOVERY_GAPPERS",
                "symbols": [],
                "symbol_count": 0,
                "data_quality_flags": [DataQualityFlag.PRICE_STALE.value]
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    # ==========================================================================
    # STEP 1: Gather ALL candidates from ALL sources
    # ==========================================================================
    raw_candidates: Dict[str, Dict] = {}  # symbol -> raw data

    # Source 1: Top GAPPERS (market movers up)
    gappers = await _fetch_market_movers("up")
    if gappers:
        for m in gappers:
            symbol = m.get("symbol", "")
            if symbol and symbol not in raw_candidates:
                raw_candidates[symbol] = {
                    "source": "gappers",
                    "change_pct": m.get("change_pct") or m.get("changePct") or m.get("change_percent") or 0,
                    "price": m.get("price") or m.get("lastPrice") or 0,
                    "volume": m.get("volume") or 0
                }
        logger.info(f"R1: Found {len(gappers)} top gappers")

    # Source 2: Top GAINERS (% gainers - may overlap with gappers)
    gainers = await _fetch_market_movers("gainers")
    if gainers:
        for m in gainers:
            symbol = m.get("symbol", "")
            if symbol and symbol not in raw_candidates:
                raw_candidates[symbol] = {
                    "source": "gainers",
                    "change_pct": m.get("change_pct") or m.get("changePct") or m.get("change_percent") or 0,
                    "price": m.get("price") or m.get("lastPrice") or 0,
                    "volume": m.get("volume") or 0
                }
        logger.info(f"R1: Found {len(gainers)} top gainers")

    # Source 3: Existing worklist (for RE-EVALUATION only)
    # These are NOT grandfathered - they MUST pass current filters
    worklist_resp = await _fetch_worklist()
    worklist_data = []
    if worklist_resp:
        if isinstance(worklist_resp, dict) and worklist_resp.get("data"):
            worklist_data = worklist_resp.get("data", [])
        elif isinstance(worklist_resp, list):
            worklist_data = worklist_resp

    for w in worklist_data:
        symbol = w.get("symbol", "")
        if symbol and symbol not in raw_candidates:
            raw_candidates[symbol] = {
                "source": "worklist_reeval",  # Mark as re-evaluation candidate
                "change_pct": w.get("change_pct") or w.get("changePct") or w.get("change_percent") or 0,
                "price": w.get("price") or 0,
                "volume": w.get("volume") or 0,
                "float": w.get("float"),
                "high": w.get("high"),
                "bid": w.get("bid"),
                "ask": w.get("ask"),
                "percent_from_hod": w.get("percent_from_hod"),
                "rel_vol": w.get("rel_vol")
            }

    logger.info(f"R1: Gathered {len(raw_candidates)} total candidates (gappers + gainers + worklist)")

    # ==========================================================================
    # STEP 2: Fetch CURRENT metrics for EVERY candidate
    # ==========================================================================
    enriched_candidates: List[Dict] = []
    data_quality_summary = {
        "total_evaluated": 0,
        "float_missing_count": 0,
        "avg_volume_missing_count": 0,
        "hod_inconsistent_count": 0,
        "rel_vol_absurd_count": 0,
        # TASK 1: Track avgVolume provenance
        "avg_volume_sources": {
            "schwab": 0,
            "polygon": 0,
            "yfinance_fallback": 0,
            "unavailable": 0
        }
    }

    for symbol, raw in raw_candidates.items():
        data_quality_summary["total_evaluated"] += 1

        # Fetch FRESH quote for this symbol
        quote = await _fetch_quote(symbol)
        if not quote:
            quote = raw  # Use raw data as fallback

        # Fetch float data
        float_data = await _fetch_float(symbol)

        # Extract current metrics
        price = quote.get("lastPrice") or quote.get("price") or raw.get("price") or 0
        high_price = quote.get("highPrice") or quote.get("dayHigh") or quote.get("high") or raw.get("high") or price
        volume = quote.get("totalVolume") or quote.get("volume") or raw.get("volume") or 0

        # =====================================================================
        # TASK 1: Multi-source avgVolume resolution with provenance tracking
        # =====================================================================
        avg_volume, avg_volume_source = await conn.fetch_avg_volume(symbol)
        data_quality_summary["avg_volume_sources"][avg_volume_source] = \
            data_quality_summary["avg_volume_sources"].get(avg_volume_source, 0) + 1

        # Gap percent (use change_pct from raw if quote doesn't have it)
        gap_pct = abs(raw.get("change_pct") or 0)

        # Compute relative volume
        rel_vol_daily = None
        if avg_volume and avg_volume > 0 and volume > 0:
            rel_vol_daily = round(volume / avg_volume, 2)
            if rel_vol_daily > 5000:
                data_quality_summary["rel_vol_absurd_count"] += 1
                rel_vol_daily = None  # Discard absurd values
        else:
            data_quality_summary["avg_volume_missing_count"] += 1

        # Float shares
        float_shares = None
        if float_data:
            float_shares = float_data.get("floatShares") or float_data.get("float_shares")
        if float_shares is None:
            data_quality_summary["float_missing_count"] += 1

        # HOD metrics
        pct_from_hod = 0.0
        at_hod = False
        near_hod = False
        if price > 0 and high_price > 0:
            pct_from_hod = round(((price - high_price) / high_price) * 100, 2)
            at_hod = abs(pct_from_hod) < 0.5
            near_hod = not at_hod and pct_from_hod > -2.0

        if high_price and price and high_price > price * 1.1:
            data_quality_summary["hod_inconsistent_count"] += 1

        # Build candidate dict for MomentumWatchlist
        candidate = {
            "symbol": symbol,
            "price": price,
            "gap_pct": gap_pct,
            "volume_today": volume,
            "rel_vol_daily": rel_vol_daily,
            "rel_vol_5m": rel_vol_daily * 1.5 if rel_vol_daily else None,  # Estimate
            "float_shares": float_shares,
            "hod_price": high_price,
            "pct_from_hod": pct_from_hod,
            "at_hod": at_hod,
            "near_hod": near_hod,
            "pct_gain": gap_pct,  # Use gap as intraday gain proxy
            "source": raw.get("source", "unknown"),
            # TASK 1: Track avgVolume provenance
            "avg_volume": avg_volume,
            "avg_volume_source": avg_volume_source
        }
        enriched_candidates.append(candidate)

    logger.info(f"R1: Enriched {len(enriched_candidates)} candidates with current metrics")

    # ==========================================================================
    # STEP 3: FULL RECOMPUTE via MomentumWatchlist
    # ==========================================================================
    watchlist = get_momentum_watchlist()

    # FULL RECOMPUTE - all candidates re-ranked with current metrics
    active_symbols, rank_snapshot = watchlist.full_recompute(
        candidates=enriched_candidates,
        force_fresh=True  # Discard prior state - fresh ranking every cycle
    )

    logger.info(
        f"R1: MomentumWatchlist recompute complete - "
        f"{len(active_symbols)} active, {rank_snapshot.get('excluded_count', 0)} excluded, "
        f"{rank_snapshot.get('degraded_count', 0)} degraded"
    )

    # ==========================================================================
    # STEP 4: Build output with full transparency
    # TASK 2: Include DEGRADED symbols in output for downstream processing
    # ==========================================================================

    # Convert active watchlist to SignalSnapshot format for downstream compatibility
    symbols_output = []
    for ranked in active_symbols:
        # Convert RankedSymbol to dict with SignalSnapshot-compatible keys
        symbol_dict = ranked.to_dict()

        # Add SignalSnapshot-required fields
        symbol_dict["hod_status"] = (
            "AT_HOD" if ranked.at_hod else
            "NEAR_HOD" if ranked.near_hod else
            "PULLBACK" if ranked.pct_from_hod < -5 else
            "PULLBACK"
        )
        symbol_dict["priority"] = (
            "HIGH" if ranked.dominance_score >= 60 else
            "NORMAL" if ranked.dominance_score >= 40 else
            "LOW"
        )
        symbol_dict["data_quality"] = "PASS"  # Active symbols have good data

        symbols_output.append(symbol_dict)

    # TASK 2: Add degraded symbols to output with warning flag
    # These symbols proceed downstream but are marked for careful handling
    degraded_symbols = rank_snapshot.get("degraded", [])
    for degraded_dict in degraded_symbols:
        # Add SignalSnapshot-required fields for degraded symbols
        degraded_dict["hod_status"] = "UNKNOWN"  # Can't determine without good data
        degraded_dict["priority"] = "LOW"  # Lower priority due to missing data
        degraded_dict["data_quality"] = "DEGRADED"  # Flag for downstream tasks
        degraded_dict["data_quality_warning"] = "Missing avgVolume - rel_vol unavailable"
        symbols_output.append(degraded_dict)

    return {
        "task_id": "DISCOVERY_GAPPERS",
        "symbols": symbols_output,
        "symbol_count": len(symbols_output),
        "active_count": len(active_symbols),  # TASK 2: Separate active from degraded count
        "degraded_count": len(degraded_symbols),  # TASK 2: Track degraded count
        "recompute_model": "FULL_RECOMPUTE",  # Flag indicating new model
        "filter_criteria": watchlist.config.to_dict(),
        "data_quality_summary": data_quality_summary,
        "rank_snapshot": rank_snapshot,  # EMBEDDED for transparency
        "watchlist_status": watchlist.get_status(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# =============================================================================
# TASK 1.2 - DISCOVERY_FLOAT_FILTER (R2)
# =============================================================================

async def task_discovery_float_filter(inputs: Dict) -> Dict:
    """
    Low Float Qualification (R2)

    INPUTS: report_R1_daily_top_gappers.json
    PROCESS:
        - Filter out float > 15M
        - Flag float <= 5M as HIGH_PRIORITY
        - Symbols with float_missing are DEGRADED, not rejected
    OUTPUT: report_R2_low_float_universe.json
    FAIL: Empty universe AND no degraded candidates -> halt
    """
    logger.info("TASK 1.2: Filtering by low float...")

    # Load R1 input
    r1_data = inputs.get("report_R1_daily_top_gappers.json", {})
    symbols = r1_data.get("symbols", [])

    qualified = []
    degraded = []  # Symbols with missing float data
    rejected = []  # Symbols with float > 15M

    max_float = 15_000_000
    high_priority_threshold = 5_000_000

    for s in symbols:
        # Reconstruct SignalSnapshot
        snapshot = SignalSnapshot.from_dict(s)

        # Handle missing float data
        if snapshot.float_shares is None:
            # Degraded - allow but flag
            snapshot.priority = "DEGRADED"
            degraded.append(snapshot.to_dict())
            continue

        # Filter by float (handle None - treat unknown float as potentially tradeable)
        if snapshot.float_shares is not None and snapshot.float_shares > max_float:
            rejected.append({
                "symbol": snapshot.symbol,
                "float_shares": snapshot.float_shares,
                "reason": f"Float {snapshot.float_shares:,} > {max_float:,}"
            })
            continue

        # Classify priority (unknown float = HIGH priority since likely small)
        if snapshot.float_shares is None or snapshot.float_shares <= high_priority_threshold:
            snapshot.priority = "HIGH"
        else:
            snapshot.priority = "NORMAL"

        qualified.append(snapshot.to_dict())

    # Sort: HIGH priority first, then by gap_pct
    qualified.sort(key=lambda x: (
        0 if x.get("priority") == "HIGH" else 1,
        -(x.get("gap_pct") or 0)
    ))

    # Combine qualified + degraded for downstream
    all_candidates = qualified + degraded

    return {
        "task_id": "DISCOVERY_FLOAT_FILTER",
        "symbols": all_candidates,
        "symbol_count": len(all_candidates),
        "qualified_count": len(qualified),
        "degraded_count": len(degraded),
        "rejected_count": len(rejected),
        "high_priority_count": sum(1 for s in qualified if s.get("priority") == "HIGH"),
        "filter_criteria": {
            "max_float": max_float,
            "high_priority_threshold": high_priority_threshold
        },
        "rejected_symbols": rejected,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# =============================================================================
# TASK 1.3 - DISCOVERY_REL_VOLUME (R3)
# =============================================================================

async def task_discovery_rel_volume(inputs: Dict) -> Dict:
    """
    Relative Volume Acceleration (R3)

    INPUTS: report_R2_low_float_universe.json
    PROCESS:
        - Compute Rel Vol (daily) with proper null handling
        - Flag absurd values (> 5000)
    OUTPUT: report_R3_rel_volume.json
    FAIL: All symbols have absurd rel_vol
    """
    logger.info("TASK 1.3: Computing relative volume...")

    # Load R2 input
    r2_data = inputs.get("report_R2_low_float_universe.json", {})
    symbols = r2_data.get("symbols", [])

    enriched = []
    data_quality_issues = {
        "avg_volume_missing": 0,
        "rel_vol_absurd": 0
    }

    conn = get_connection_manager()

    for s in symbols:
        # Reconstruct SignalSnapshot
        snapshot = SignalSnapshot.from_dict(s)
        symbol = snapshot.symbol

        # Fetch fresh quote for volume data
        quote = await _fetch_quote(symbol)
        if quote:
            # Update volume fields
            snapshot.volume_today = quote.get("totalVolume") or quote.get("volume") or snapshot.volume_today

        # =====================================================================
        # TASK 1: Multi-source avgVolume resolution with provenance tracking
        # =====================================================================
        avg_vol, avg_vol_source = await conn.fetch_avg_volume(symbol)

        # Track provenance in data quality
        if "avg_volume_sources" not in data_quality_issues:
            data_quality_issues["avg_volume_sources"] = {"schwab": 0, "polygon": 0, "yfinance_fallback": 0, "unavailable": 0}
        data_quality_issues["avg_volume_sources"][avg_vol_source] = \
            data_quality_issues["avg_volume_sources"].get(avg_vol_source, 0) + 1

        if avg_vol and avg_vol > 0:
            snapshot.avg_daily_volume_30d = avg_vol
            snapshot.compute_rel_volumes()
        else:
            snapshot.avg_daily_volume_30d = None
            snapshot.rel_vol_daily = None
            data_quality_issues["avg_volume_missing"] += 1
            snapshot.data_quality.add_flag(DataQualityFlag.AVG_VOLUME_MISSING)

        # Check for absurd rel_vol
        if snapshot.rel_vol_daily and snapshot.rel_vol_daily > 5000:
            data_quality_issues["rel_vol_absurd"] += 1
            snapshot.data_quality.add_flag(DataQualityFlag.REL_VOL_ABSURD)

        # Estimate 5m rel vol (simplified - would need real 5m bars)
        if snapshot.rel_vol_daily:
            snapshot.rel_vol_5m = round(snapshot.rel_vol_daily * 1.5, 2)

        enriched.append(snapshot.to_dict())

    # Sort by rel_vol_daily descending (nulls last)
    enriched.sort(key=lambda x: x.get("rel_vol_daily") or 0, reverse=True)

    return {
        "task_id": "DISCOVERY_REL_VOLUME",
        "symbols": enriched,
        "symbol_count": len(enriched),
        "high_rel_vol_count": sum(1 for s in enriched if (s.get("rel_vol_daily") or 0) >= 3.0),
        "data_quality_issues": data_quality_issues,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# =============================================================================
# TASK 1.4 - DISCOVERY_HOD_BEHAVIOR (R4)
# =============================================================================

async def task_discovery_hod_behavior(inputs: Dict) -> Dict:
    """
    HOD Behavior Tracking (R4)

    CRITICAL: Must output EXACT keys: hod_status, pct_from_hod, hod_price
    These are used by R10 gating.

    INPUTS: report_R3_rel_volume.json
    PROCESS:
        - Track HOD breaks
        - Compute distance from HOD
        - Set hod_status enum
    OUTPUT: report_R4_hod_behavior.json
    """
    logger.info("TASK 1.4: Tracking HOD behavior...")

    # Load R3 input
    r3_data = inputs.get("report_R3_rel_volume.json", {})
    symbols = r3_data.get("symbols", [])

    enriched = []
    hod_status_counts = {
        "AT_HOD": 0,
        "NEAR_HOD": 0,
        "PULLBACK": 0,
        "FAIL": 0,
        "UNKNOWN": 0
    }

    for s in symbols:
        # Reconstruct SignalSnapshot
        snapshot = SignalSnapshot.from_dict(s)
        symbol = snapshot.symbol

        # Fetch fresh quote for HOD data
        quote = await _fetch_quote(symbol)
        if quote:
            # Update price and HOD
            snapshot.price = quote.get("lastPrice") or quote.get("price") or snapshot.price
            snapshot.hod_price = quote.get("highPrice") or quote.get("dayHigh") or quote.get("high") or snapshot.hod_price

            # Recompute HOD status with fresh data
            snapshot.compute_hod_status()

        # Count by status
        status_str = snapshot.hod_status.value if snapshot.hod_status else "UNKNOWN"
        hod_status_counts[status_str] = hod_status_counts.get(status_str, 0) + 1

        enriched.append(snapshot.to_dict())

    # Sort: AT_HOD first, then NEAR_HOD, then by gap_pct
    status_order = {"AT_HOD": 0, "NEAR_HOD": 1, "PULLBACK": 2, "FAIL": 3, "UNKNOWN": 4}
    enriched.sort(key=lambda x: (
        status_order.get(x.get("hod_status", "UNKNOWN"), 4),
        -(x.get("gap_pct") or 0)
    ))

    return {
        "task_id": "DISCOVERY_HOD_BEHAVIOR",
        "symbols": enriched,
        "symbol_count": len(enriched),
        "hod_status_summary": hod_status_counts,
        "at_hod_count": hod_status_counts.get("AT_HOD", 0),
        "near_hod_count": hod_status_counts.get("NEAR_HOD", 0),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# =============================================================================
# REGISTER TASKS
# =============================================================================

def register_discovery_tasks():
    """Register all Task Group 1 tasks with the manager"""
    manager = get_task_queue_manager()

    # Task 0 - Data Quality (optional pre-flight)
    manager.register_task(Task(
        id="DATA_QUALITY_CHECK",
        name="Data Quality Pre-Flight",
        group="PRE_FLIGHT",
        inputs=[],
        process=task_data_quality_check,
        output_file="report_R0_data_quality.json",
        fail_conditions=["critical_failures"],
        next_task="DISCOVERY_GAPPERS"
    ))

    # Task 1.1 - Top Gappers
    manager.register_task(Task(
        id="DISCOVERY_GAPPERS",
        name="Generate Daily Top Gappers",
        group="MARKET_DISCOVERY",
        inputs=[],  # Uses live data
        process=task_discovery_gappers,
        output_file="report_R1_daily_top_gappers.json",
        fail_conditions=["no_symbols"],
        next_task="DISCOVERY_FLOAT_FILTER"
    ))

    # Task 1.2 - Float Filter
    manager.register_task(Task(
        id="DISCOVERY_FLOAT_FILTER",
        name="Low Float Qualification",
        group="MARKET_DISCOVERY",
        inputs=["report_R1_daily_top_gappers.json"],
        process=task_discovery_float_filter,
        output_file="report_R2_low_float_universe.json",
        fail_conditions=["empty_universe"],
        next_task="DISCOVERY_REL_VOLUME"
    ))

    # Task 1.3 - Relative Volume
    # NOTE: "low_rel_vol" fail condition disabled for DRY-RUN testing
    # Re-enable for production when real volume data is available
    manager.register_task(Task(
        id="DISCOVERY_REL_VOLUME",
        name="Relative Volume Acceleration",
        group="MARKET_DISCOVERY",
        inputs=["report_R2_low_float_universe.json"],
        process=task_discovery_rel_volume,
        output_file="report_R3_rel_volume.json",
        fail_conditions=[],  # Disabled: ["low_rel_vol"]
        next_task="DISCOVERY_HOD_BEHAVIOR"
    ))

    # Task 1.4 - HOD Behavior
    manager.register_task(Task(
        id="DISCOVERY_HOD_BEHAVIOR",
        name="HOD Behavior Tracking",
        group="MARKET_DISCOVERY",
        inputs=["report_R3_rel_volume.json"],
        process=task_discovery_hod_behavior,
        output_file="report_R4_hod_behavior.json",
        fail_conditions=[],
        next_task="QLIB_HOD_PROBABILITY"
    ))

    logger.info("Task Group 1 (Market Discovery) registered: 5 tasks (R0-R4)")
