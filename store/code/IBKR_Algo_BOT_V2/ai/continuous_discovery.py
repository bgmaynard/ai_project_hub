"""
Continuous Discovery Service

Polls for new trading opportunities throughout the session.
Injects symbols into the pipeline when found.

Key Features:
- Runs every N minutes during trading hours
- Multiple data sources: movers, ATS, news, manual
- Injects symbols via task_queue_manager.inject_symbols()
- Respects pipeline WAITING state for auto-resume
- TASK 4: Time-aware filter scheduling (different filters by time of day)
"""

import asyncio
import logging
import json
from datetime import datetime, time
from typing import List, Dict, Optional, Set, Callable, Tuple
from pathlib import Path
import pytz
import threading

logger = logging.getLogger(__name__)

ET_TZ = pytz.timezone('US/Eastern')

# =============================================================================
# TASK 4: Time-Aware Discovery Schedule Configuration
# =============================================================================

DEFAULT_DISCOVERY_SCHEDULE = [
    {
        "name": "premarket",
        "time_range_et": ["04:00", "09:29"],
        "filters": {
            "min_price": 1.0,
            "max_price": 25.0,
            "min_gap_pct": 3.0,
            "min_rel_vol": 2.0,
            "min_premarket_volume": 250000
        }
    },
    {
        "name": "open",
        "time_range_et": ["09:30", "09:45"],
        "filters": {
            "min_price": 1.0,
            "max_price": 25.0,
            "min_gap_pct": 5.0,
            "min_rel_vol": 2.5,
            "cooldown_seconds": 120
        }
    },
    {
        "name": "post_open",
        "time_range_et": ["09:45", "11:30"],
        "filters": {
            "min_price": 1.0,
            "max_price": 25.0,
            "min_gap_pct": 8.0,
            "min_rel_vol": 3.0,
            "min_volume": 1000000
        }
    },
    {
        "name": "midday",
        "time_range_et": ["11:30", "15:00"],
        "filters": {
            "min_price": 1.0,
            "max_price": 25.0,
            "min_gap_pct": 10.0,
            "min_rel_vol": 2.0
        }
    }
]

# Default fallback filters (used when no schedule matches)
DEFAULT_FILTERS = {
    "min_price": 1.0,
    "max_price": 25.0,
    "min_gap_pct": 5.0,
    "min_rel_vol": 2.0
}

# Service state
_service_running = False
_service_thread: Optional[threading.Thread] = None
_discovered_symbols: Set[str] = set()  # Track what we've already found this session
_discovery_callbacks: List[Callable] = []
_poll_interval_seconds = 300  # 5 minutes default
_last_poll_time: Optional[datetime] = None
_discovery_schedule: List[Dict] = DEFAULT_DISCOVERY_SCHEDULE
_last_filter_stats: Dict = {}  # Track last poll filter statistics


def _parse_time(time_str: str) -> time:
    """Parse time string HH:MM to time object"""
    parts = time_str.split(":")
    return time(int(parts[0]), int(parts[1]))


def get_active_schedule() -> Tuple[Optional[Dict], str]:
    """
    TASK 4: Get the currently active schedule based on ET time.

    Returns:
        (schedule_dict, schedule_name) or (None, "none") if outside all windows
    """
    now = datetime.now(ET_TZ).time()

    for schedule in _discovery_schedule:
        time_range = schedule.get("time_range_et", [])
        if len(time_range) != 2:
            continue

        start_time = _parse_time(time_range[0])
        end_time = _parse_time(time_range[1])

        if start_time <= now <= end_time:
            return schedule, schedule.get("name", "unknown")

    return None, "none"


def get_active_filters() -> Dict:
    """
    TASK 4: Get the currently active filters based on schedule.

    Returns:
        Filter dict for current time window, or DEFAULT_FILTERS if none match
    """
    schedule, name = get_active_schedule()
    if schedule:
        return schedule.get("filters", DEFAULT_FILTERS)
    return DEFAULT_FILTERS


def is_discovery_window() -> bool:
    """Check if we're in a discovery window (any schedule active)"""
    schedule, name = get_active_schedule()
    return schedule is not None


async def _fetch_gappers() -> List[Dict]:
    """Fetch top gappers from Schwab"""
    try:
        from .connection_manager import get_connection_manager
        conn = get_connection_manager()
        data, _ = await conn.fetch_market_movers("up")
        if isinstance(data, list):
            return data
    except Exception as e:
        logger.debug(f"Gapper fetch failed: {e}")
    return []


async def _fetch_from_finviz() -> List[Dict]:
    """Fetch top gainers from Finviz"""
    try:
        from .finviz_scanner import get_finviz_scanner
        scanner = get_finviz_scanner()
        gainers = await scanner.scan_top_gainers(limit=20)
        return gainers or []
    except Exception as e:
        logger.debug(f"Finviz fetch failed: {e}")
    return []


async def _fetch_from_news() -> List[Dict]:
    """Get symbols with breaking news catalysts"""
    try:
        from .benzinga_fast_news import get_benzinga_fast_news
        news = get_benzinga_fast_news()
        catalysts = news.get_catalyst_symbols(min_urgency="high")
        return [{"symbol": s, "source": "news"} for s in catalysts]
    except Exception as e:
        logger.debug(f"News fetch failed: {e}")
    return []


def _filter_candidates(candidates: List[Dict]) -> Tuple[List[str], Dict]:
    """
    TASK 4: Filter candidates using time-aware filters.

    Returns:
        (filtered_symbols, filter_stats) - stats include counts and reasons
    """
    global _last_filter_stats

    # Get active schedule and filters
    schedule, schedule_name = get_active_schedule()
    filters = get_active_filters()

    filtered = []
    excluded = {
        "already_discovered": [],
        "price_too_low": [],
        "price_too_high": [],
        "gap_too_low": [],
        "rel_vol_too_low": [],
        "volume_too_low": [],
        "premarket_volume_too_low": [],
        "invalid_symbol": []
    }

    # Extract filter thresholds
    min_price = filters.get("min_price", 1.0)
    max_price = filters.get("max_price", 25.0)
    min_gap_pct = filters.get("min_gap_pct", 5.0)
    min_rel_vol = filters.get("min_rel_vol", 2.0)
    min_volume = filters.get("min_volume", 0)
    min_premarket_volume = filters.get("min_premarket_volume", 0)

    for c in candidates:
        symbol = c.get("symbol", "").upper()
        if not symbol or len(symbol) > 5:
            excluded["invalid_symbol"].append(symbol or "EMPTY")
            continue

        # Skip if already discovered
        if symbol in _discovered_symbols:
            excluded["already_discovered"].append(symbol)
            continue

        price = c.get("price") or c.get("lastPrice") or 0
        gap = abs(c.get("change_pct") or c.get("changePct") or c.get("gap_pct") or 0)
        rel_vol = c.get("rel_vol") or c.get("relativeVolume") or 0
        volume = c.get("volume") or c.get("totalVolume") or 0
        premarket_vol = c.get("premarket_volume") or c.get("premarketVolume") or 0

        # Price filter
        if price < min_price:
            excluded["price_too_low"].append(symbol)
            continue
        if price > max_price:
            excluded["price_too_high"].append(symbol)
            continue

        # Gap filter (news sources get 50% reduction in min gap)
        source = c.get("source", "")
        effective_min_gap = min_gap_pct * 0.5 if source == "news" else min_gap_pct
        if gap < effective_min_gap:
            excluded["gap_too_low"].append(symbol)
            continue

        # Relative volume filter (if we have the data)
        if rel_vol > 0 and rel_vol < min_rel_vol:
            excluded["rel_vol_too_low"].append(symbol)
            continue

        # Volume filter (if specified and we have data)
        if min_volume > 0 and volume > 0 and volume < min_volume:
            excluded["volume_too_low"].append(symbol)
            continue

        # Pre-market volume filter (if specified and we have data)
        if min_premarket_volume > 0 and premarket_vol > 0 and premarket_vol < min_premarket_volume:
            excluded["premarket_volume_too_low"].append(symbol)
            continue

        filtered.append(symbol)

    # Build filter statistics
    filter_stats = {
        "schedule_name": schedule_name,
        "schedule_active": schedule is not None,
        "filters_applied": filters,
        "candidates_evaluated": len(candidates),
        "candidates_passed": len(filtered),
        "candidates_excluded": sum(len(v) for v in excluded.values()),
        "exclusion_reasons": {k: len(v) for k, v in excluded.items() if v},
        "excluded_symbols": excluded,
        "timestamp": datetime.now(ET_TZ).isoformat()
    }

    _last_filter_stats = filter_stats
    return filtered, filter_stats


async def _discovery_poll():
    """Single discovery poll cycle with TASK 4 schedule-aware filtering"""
    global _last_poll_time, _discovered_symbols

    # TASK 4: Get active schedule info
    schedule, schedule_name = get_active_schedule()

    if not is_discovery_window():
        logger.debug(f"Outside discovery window (schedule={schedule_name}), skipping poll")
        return []

    _last_poll_time = datetime.now(ET_TZ)
    logger.info(f"Running discovery poll (schedule: {schedule_name})...")

    all_candidates = []

    # Fetch from multiple sources in parallel
    try:
        gappers, finviz, news = await asyncio.gather(
            _fetch_gappers(),
            _fetch_from_finviz(),
            _fetch_from_news(),
            return_exceptions=True
        )

        if isinstance(gappers, list):
            for g in gappers:
                g["source"] = "schwab_movers"
            all_candidates.extend(gappers)

        if isinstance(finviz, list):
            for f in finviz:
                f["source"] = "finviz"
            all_candidates.extend(finviz)

        if isinstance(news, list):
            all_candidates.extend(news)

    except Exception as e:
        logger.error(f"Discovery poll error: {e}")
        return []

    # TASK 4: Filter with time-aware filters and get statistics
    new_symbols, filter_stats = _filter_candidates(all_candidates)

    # Log filter statistics
    logger.info(
        f"Filter stats [{schedule_name}]: evaluated={filter_stats['candidates_evaluated']}, "
        f"passed={filter_stats['candidates_passed']}, excluded={filter_stats['candidates_excluded']}"
    )
    if filter_stats.get("exclusion_reasons"):
        logger.debug(f"Exclusion breakdown: {filter_stats['exclusion_reasons']}")

    if new_symbols:
        # Track as discovered
        _discovered_symbols.update(new_symbols)

        logger.info(f"Discovery found {len(new_symbols)} new symbols: {new_symbols}")

        # Inject into pipeline with provenance (Task 3) and schedule metadata (Task 4)
        try:
            from .task_queue_manager import get_task_queue_manager
            manager = get_task_queue_manager()

            success, result = manager.inject_symbols(
                symbols=new_symbols,
                source="CONTINUOUS_DISCOVERY",
                trigger_reason="GAP_SCAN",
                metadata={
                    "poll_time": datetime.now(ET_TZ).isoformat(),
                    "discovery_window": is_discovery_window(),
                    # TASK 4: Include schedule metadata
                    "schedule_name": schedule_name,
                    "filters_applied": filter_stats.get("filters_applied", {}),
                    "filter_stats": {
                        "evaluated": filter_stats["candidates_evaluated"],
                        "passed": filter_stats["candidates_passed"],
                        "excluded": filter_stats["candidates_excluded"]
                    }
                }
            )

            if success:
                accepted = result.get("accepted", [])
                deferred = result.get("deferred", [])
                logger.info(f"Discovery injection: accepted={len(accepted)}, deferred={len(deferred)}")

                # If pipeline is WAITING, auto-resume with the new symbols
                if manager.is_waiting() and accepted:
                    logger.info("Pipeline in WAITING state, auto-resuming with discovered symbols...")
                    try:
                        # Run resume in this event loop
                        results = await manager.resume_from_waiting()
                        logger.info(f"Pipeline resumed, completed: {len([r for r in results.values() if r.status.value == 'COMPLETED'])}")
                    except Exception as resume_err:
                        logger.error(f"Failed to resume pipeline: {resume_err}")
            else:
                logger.warning(f"Discovery injection throttled: {result.get('reason', 'unknown')}")

        except Exception as e:
            logger.error(f"Failed to inject symbols: {e}")

        # Notify callbacks
        for callback in _discovery_callbacks:
            try:
                callback(new_symbols)
            except Exception as e:
                logger.error(f"Discovery callback error: {e}")

    return new_symbols


async def _discovery_loop():
    """Main discovery loop"""
    global _service_running

    logger.info(f"Continuous discovery started (poll interval: {_poll_interval_seconds}s)")

    while _service_running:
        try:
            await _discovery_poll()
        except Exception as e:
            logger.error(f"Discovery loop error: {e}")

        # Wait for next poll
        await asyncio.sleep(_poll_interval_seconds)

    logger.info("Continuous discovery stopped")


def _run_loop_in_thread():
    """Run the discovery loop in a separate thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_discovery_loop())
    finally:
        loop.close()


def start_continuous_discovery(poll_interval_seconds: int = 300):
    """
    Start the continuous discovery service.

    Args:
        poll_interval_seconds: How often to poll (default 5 minutes)
    """
    global _service_running, _service_thread, _poll_interval_seconds

    if _service_running:
        logger.info("Continuous discovery already running")
        return

    _poll_interval_seconds = poll_interval_seconds
    _service_running = True

    _service_thread = threading.Thread(target=_run_loop_in_thread, daemon=True)
    _service_thread.start()

    logger.info(f"Continuous discovery service started (poll every {poll_interval_seconds}s)")


def stop_continuous_discovery():
    """Stop the continuous discovery service"""
    global _service_running, _service_thread

    _service_running = False

    if _service_thread and _service_thread.is_alive():
        _service_thread.join(timeout=5)

    _service_thread = None
    logger.info("Continuous discovery service stopped")


def register_discovery_callback(callback: Callable[[List[str]], None]):
    """Register a callback for when new symbols are discovered"""
    if callback not in _discovery_callbacks:
        _discovery_callbacks.append(callback)


def unregister_discovery_callback(callback: Callable):
    """Unregister a discovery callback"""
    if callback in _discovery_callbacks:
        _discovery_callbacks.remove(callback)


def reset_session():
    """Reset discovered symbols for a new session"""
    global _discovered_symbols
    _discovered_symbols.clear()
    logger.info("Discovery session reset")


def set_discovery_schedule(schedule: List[Dict]):
    """
    TASK 4: Set custom discovery schedule.

    Args:
        schedule: List of schedule dicts with name, time_range_et, filters
    """
    global _discovery_schedule
    _discovery_schedule = schedule
    logger.info(f"Discovery schedule updated: {[s['name'] for s in schedule]}")


def get_discovery_schedule() -> List[Dict]:
    """TASK 4: Get current discovery schedule"""
    return _discovery_schedule


def get_status() -> dict:
    """Get discovery service status with TASK 4 schedule info"""
    schedule, schedule_name = get_active_schedule()
    filters = get_active_filters()

    return {
        "running": _service_running,
        "poll_interval_seconds": _poll_interval_seconds,
        "last_poll_time": _last_poll_time.isoformat() if _last_poll_time else None,
        "discovered_count": len(_discovered_symbols),
        "discovered_symbols": list(_discovered_symbols),
        "in_discovery_window": is_discovery_window(),
        "callback_count": len(_discovery_callbacks),
        # TASK 4: Schedule info
        "schedule": {
            "active_name": schedule_name,
            "active_filters": filters,
            "all_schedules": [s["name"] for s in _discovery_schedule],
            "current_et_time": datetime.now(ET_TZ).strftime("%H:%M:%S")
        },
        "last_filter_stats": _last_filter_stats
    }


async def poll_now() -> List[str]:
    """Trigger an immediate discovery poll"""
    return await _discovery_poll()
