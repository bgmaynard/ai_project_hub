"""
Warrior Trading Scanner API Routes
===================================
REST endpoints for gap, gainer, and HOD scanners.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scanners", tags=["Warrior Scanners"])

try:
    from scanners import get_scanner_config, ScannerConfig
    from scanners.gap_scanner import get_gap_scanner
    from scanners.gainer_scanner import get_gainer_scanner
    from scanners.hod_scanner import get_hod_scanner
    from scanners.scanner_coordinator import get_scanner_coordinator
    HAS_SCANNERS = True
except ImportError as e:
    logger.warning(f"Warrior scanners not available: {e}")
    HAS_SCANNERS = False


class ScanRequest(BaseModel):
    symbols: List[str]


class ConfigUpdate(BaseModel):
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    max_spread_pct: Optional[float] = None
    gap_min_pct: Optional[float] = None
    gap_min_volume: Optional[int] = None
    gainer_min_pct: Optional[float] = None
    gainer_min_rel_vol: Optional[float] = None
    gainer_min_volume: Optional[int] = None
    hod_min_rel_vol: Optional[float] = None
    hod_min_volume: Optional[int] = None


# ============================================================================
# STATUS ENDPOINTS
# ============================================================================

@router.get("/status")
async def get_scanner_status():
    """Get status of all Warrior scanners"""
    if not HAS_SCANNERS:
        return {"available": False, "error": "Scanners not available"}

    coordinator = get_scanner_coordinator()
    return {
        "available": True,
        **coordinator.get_status()
    }


@router.get("/config")
async def get_config():
    """Get current scanner configuration"""
    if not HAS_SCANNERS:
        return {"error": "Scanners not available"}

    config = get_scanner_config()
    return {
        "min_price": config.min_price,
        "max_price": config.max_price,
        "max_spread_pct": config.max_spread_pct,
        "gap_min_pct": config.gap_min_pct,
        "gap_min_volume": config.gap_min_volume,
        "gainer_min_pct": config.gainer_min_pct,
        "gainer_min_rel_vol": config.gainer_min_rel_vol,
        "gainer_min_volume": config.gainer_min_volume,
        "hod_min_rel_vol": config.hod_min_rel_vol,
        "hod_min_volume": config.hod_min_volume
    }


@router.post("/config")
async def update_config(update: ConfigUpdate):
    """Update scanner configuration"""
    if not HAS_SCANNERS:
        return {"error": "Scanners not available"}

    config = get_scanner_config()

    # Update non-None fields
    if update.min_price is not None:
        config.min_price = update.min_price
    if update.max_price is not None:
        config.max_price = update.max_price
    if update.max_spread_pct is not None:
        config.max_spread_pct = update.max_spread_pct
    if update.gap_min_pct is not None:
        config.gap_min_pct = update.gap_min_pct
    if update.gap_min_volume is not None:
        config.gap_min_volume = update.gap_min_volume
    if update.gainer_min_pct is not None:
        config.gainer_min_pct = update.gainer_min_pct
    if update.gainer_min_rel_vol is not None:
        config.gainer_min_rel_vol = update.gainer_min_rel_vol
    if update.gainer_min_volume is not None:
        config.gainer_min_volume = update.gainer_min_volume
    if update.hod_min_rel_vol is not None:
        config.hod_min_rel_vol = update.hod_min_rel_vol
    if update.hod_min_volume is not None:
        config.hod_min_volume = update.hod_min_volume

    return {"success": True, "config": await get_config()}


# ============================================================================
# AUTO-DISCOVERY ENDPOINTS (Pull from Schwab movers)
# ============================================================================

@router.get("/discover")
async def discover_candidates():
    """
    Auto-discover candidates from Schwab movers.

    1. Pulls top gainers from Schwab ($SPX, $COMPX, $DJI)
    2. Filters through scanner criteria
    3. Returns qualified candidates
    """
    if not HAS_SCANNERS:
        return {"error": "Scanners not available"}

    try:
        from schwab_market_data import get_all_movers

        # Get movers from Schwab
        all_movers = get_all_movers()
        gainers = all_movers.get('gainers', [])

        if not gainers:
            return {
                "source": "schwab_movers",
                "movers_found": 0,
                "candidates": [],
                "message": "No movers returned from Schwab (market may be closed)"
            }

        # Extract symbols
        symbols = [m['symbol'] for m in gainers if m.get('symbol')]

        # Run through scanners
        coordinator = get_scanner_coordinator()
        results = coordinator.scan(symbols)

        # Also return raw movers for visibility
        return {
            "source": "schwab_movers",
            "movers_found": len(gainers),
            "symbols_scanned": len(symbols),
            "raw_movers": gainers[:20],  # Top 20 raw movers
            "scanner_results": {
                scanner: [r.to_dict() for r in candidates]
                for scanner, candidates in results.items()
            },
            "total_candidates": len(coordinator.get_all_candidates())
        }

    except ImportError as e:
        return {"error": f"Schwab market data not available: {e}"}
    except Exception as e:
        logger.error(f"Discovery error: {e}")
        return {"error": str(e)}


@router.post("/discover/scan")
async def discover_and_scan():
    """
    Discover movers and run full scan pipeline.
    Adds qualifying candidates to watchlist.
    """
    if not HAS_SCANNERS:
        return {"error": "Scanners not available"}

    try:
        from schwab_market_data import get_all_movers

        # Get movers
        all_movers = get_all_movers()
        gainers = all_movers.get('gainers', [])
        symbols = [m['symbol'] for m in gainers if m.get('symbol')]

        if not symbols:
            return {"movers_found": 0, "candidates_added": 0}

        # Run scanners
        coordinator = get_scanner_coordinator()
        coordinator.scan(symbols)

        # Feed to watchlist
        count = coordinator.feed_to_watchlist()

        return {
            "movers_found": len(gainers),
            "symbols_scanned": len(symbols),
            "candidates_added": count,
            "active_scanners": coordinator.get_active_scanners()
        }

    except Exception as e:
        logger.error(f"Discover and scan error: {e}")
        return {"error": str(e)}


# ============================================================================
# SCAN ENDPOINTS
# ============================================================================

@router.post("/scan")
async def run_all_scans(request: ScanRequest):
    """
    Run all active scanners on the given symbols.

    Active scanners depend on time of day (ET):
    - 04:00-07:00: GAPPER only
    - 07:00-09:15: GAPPER + GAINER
    - 09:15-09:30: GAINER + HOD
    - After 09:30: HOD only
    """
    if not HAS_SCANNERS:
        return {"error": "Scanners not available"}

    coordinator = get_scanner_coordinator()
    results = coordinator.scan(request.symbols)

    return {
        "active_scanners": coordinator.get_active_scanners(),
        "results": {
            scanner: [r.to_dict() for r in candidates]
            for scanner, candidates in results.items()
        },
        "total_candidates": len(coordinator.get_all_candidates())
    }


@router.post("/scan/gap")
async def scan_gaps(request: ScanRequest):
    """Run gap scanner on the given symbols"""
    if not HAS_SCANNERS:
        return {"error": "Scanners not available"}

    scanner = get_gap_scanner()
    results = scanner.scan_symbols(request.symbols)

    return {
        "scanner": "GAPPER",
        "active": scanner.is_active_window(),
        "results": [r.to_dict() for r in results],
        "count": len(results)
    }


@router.post("/scan/gainer")
async def scan_gainers(request: ScanRequest):
    """Run gainer scanner on the given symbols"""
    if not HAS_SCANNERS:
        return {"error": "Scanners not available"}

    scanner = get_gainer_scanner()
    results = scanner.scan_symbols(request.symbols)

    return {
        "scanner": "GAINER",
        "active": scanner.is_active_window(),
        "results": [r.to_dict() for r in results],
        "count": len(results)
    }


@router.post("/scan/hod")
async def scan_hod(request: ScanRequest):
    """Run HOD scanner on the given symbols"""
    if not HAS_SCANNERS:
        return {"error": "Scanners not available"}

    scanner = get_hod_scanner()
    results = scanner.scan_symbols(request.symbols)

    return {
        "scanner": "HOD",
        "active": scanner.is_active_window(),
        "results": [r.to_dict() for r in results],
        "count": len(results)
    }


# ============================================================================
# CANDIDATE ENDPOINTS
# ============================================================================

@router.get("/candidates")
async def get_all_candidates():
    """Get all current candidates from all scanners"""
    if not HAS_SCANNERS:
        return {"error": "Scanners not available"}

    coordinator = get_scanner_coordinator()
    candidates = coordinator.get_all_candidates()

    return {
        "candidates": [c.to_dict() for c in candidates],
        "count": len(candidates),
        "by_scanner": {
            "GAPPER": len([c for c in candidates if c.scanner.value == "GAPPER"]),
            "GAINER": len([c for c in candidates if c.scanner.value == "GAINER"]),
            "HOD": len([c for c in candidates if c.scanner.value == "HOD"])
        }
    }


@router.get("/candidates/gap")
async def get_gap_candidates():
    """Get current gap scanner candidates"""
    if not HAS_SCANNERS:
        return {"error": "Scanners not available"}

    scanner = get_gap_scanner()
    candidates = scanner.get_candidates()

    return {
        "scanner": "GAPPER",
        "candidates": [c.to_dict() for c in candidates],
        "count": len(candidates)
    }


@router.get("/candidates/gainer")
async def get_gainer_candidates():
    """Get current gainer scanner candidates"""
    if not HAS_SCANNERS:
        return {"error": "Scanners not available"}

    scanner = get_gainer_scanner()
    candidates = scanner.get_candidates()

    return {
        "scanner": "GAINER",
        "candidates": [c.to_dict() for c in candidates],
        "count": len(candidates)
    }


@router.get("/candidates/hod")
async def get_hod_candidates():
    """Get current HOD scanner candidates"""
    if not HAS_SCANNERS:
        return {"error": "Scanners not available"}

    scanner = get_hod_scanner()
    candidates = scanner.get_candidates()

    return {
        "scanner": "HOD",
        "candidates": [c.to_dict() for c in candidates],
        "count": len(candidates)
    }


# ============================================================================
# WATCHLIST INTEGRATION
# ============================================================================

@router.post("/feed-watchlist")
async def feed_to_watchlist():
    """Feed all scanner candidates to MomentumWatchlist"""
    if not HAS_SCANNERS:
        return {"error": "Scanners not available"}

    coordinator = get_scanner_coordinator()
    count = coordinator.feed_to_watchlist()

    return {
        "success": True,
        "candidates_fed": count
    }


# ============================================================================
# MANAGEMENT
# ============================================================================

@router.post("/clear")
async def clear_all_candidates():
    """Clear all scanner candidates (end of session)"""
    if not HAS_SCANNERS:
        return {"error": "Scanners not available"}

    coordinator = get_scanner_coordinator()
    coordinator.clear_all()

    return {"success": True, "message": "All candidates cleared"}


@router.post("/reset-daily")
async def reset_for_new_day():
    """Reset scanners for new trading day"""
    if not HAS_SCANNERS:
        return {"error": "Scanners not available"}

    coordinator = get_scanner_coordinator()
    coordinator.clear_all()
    coordinator.clear_volume_caches()

    return {"success": True, "message": "Scanners reset for new day"}
