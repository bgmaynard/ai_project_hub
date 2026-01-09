"""
Scanner Coordinator
===================
Manages all Warrior Trading scanners based on time of day.

Time Windows (ET):
- 04:00-07:00: GAPPER only
- 07:00-09:15: GAPPER + GAINER
- 09:15-09:30: GAINER + HOD
- After 09:30: HOD only (monitoring)

Feeds results into MomentumWatchlist.
"""

import logging
import threading
from datetime import datetime, time
from typing import Dict, List, Optional, Set

import pytz

logger = logging.getLogger(__name__)

from scanners import ScannerResult, ScannerType, get_scanner_config
from scanners.gainer_scanner import get_gainer_scanner
from scanners.gap_scanner import get_gap_scanner
from scanners.hod_scanner import get_hod_scanner


class ScannerCoordinator:
    """
    Coordinates all scanners based on time of day.

    Collects candidates and feeds them to MomentumWatchlist.
    """

    def __init__(self):
        self._et_tz = pytz.timezone("US/Eastern")
        self._last_scan: Optional[datetime] = None
        self._scan_results: Dict[str, List[ScannerResult]] = {
            "GAPPER": [],
            "GAINER": [],
            "HOD": [],
        }
        self._lock = threading.Lock()
        self._cutoff_time = time(
            16, 0
        )  # No new candidates after market close (4 PM ET)

    def get_active_scanners(self) -> List[str]:
        """
        Get list of scanners that should be active based on current time.

        All scanners stay active throughout the trading day since:
        - Gappers continue to update as pre-market data changes
        - Gainers reflect intraday momentum shifts
        - HOD tracks breakout leaders

        Time Windows (ET):
        - Before 04:00: None (too early)
        - 04:00-16:00: All scanners active
        - After cutoff: None (session ended)
        """
        now_et = datetime.now(self._et_tz).time()

        if now_et < time(4, 0):
            return []  # Too early
        elif now_et < self._cutoff_time:
            return ["GAPPER", "GAINER", "HOD"]  # All active during trading day
        else:
            return []  # Past cutoff

    def is_past_cutoff(self) -> bool:
        """Check if we're past the candidate cutoff time"""
        now_et = datetime.now(self._et_tz).time()
        return now_et >= self._cutoff_time

    def scan(self, symbols: List[str]) -> Dict[str, List[ScannerResult]]:
        """
        Run all active scanners on the given symbols.

        Returns dict of scanner type -> results.
        """
        if self.is_past_cutoff():
            logger.debug("[ScannerCoordinator] Past cutoff time - no new candidates")
            return {"GAPPER": [], "GAINER": [], "HOD": []}

        active = self.get_active_scanners()
        results = {"GAPPER": [], "GAINER": [], "HOD": []}

        with self._lock:
            if "GAPPER" in active:
                gap_scanner = get_gap_scanner()
                results["GAPPER"] = gap_scanner.scan_symbols(symbols)

            if "GAINER" in active:
                gainer_scanner = get_gainer_scanner()
                results["GAINER"] = gainer_scanner.scan_symbols(symbols)

            if "HOD" in active:
                hod_scanner = get_hod_scanner()
                results["HOD"] = hod_scanner.scan_symbols(symbols)

            self._scan_results = results
            self._last_scan = datetime.now()

        return results

    def get_all_candidates(self) -> List[ScannerResult]:
        """
        Get all candidates from all scanners, deduplicated by symbol.

        Returns list of ScannerResults, one per unique symbol (latest scanner wins).
        """
        with self._lock:
            # Combine all results
            all_results = []
            all_results.extend(self._scan_results.get("GAPPER", []))
            all_results.extend(self._scan_results.get("GAINER", []))
            all_results.extend(self._scan_results.get("HOD", []))

            # Deduplicate by symbol (keep latest)
            seen: Dict[str, ScannerResult] = {}
            for result in all_results:
                seen[result.symbol] = result

            return list(seen.values())

    def feed_to_watchlist(self) -> int:
        """
        Feed all candidates to MomentumWatchlist.

        Returns number of candidates ingested.
        """
        candidates = self.get_all_candidates()

        if not candidates:
            return 0

        try:
            # Import here to avoid circular
            from ai.momentum_watchlist import get_momentum_watchlist

            watchlist = get_momentum_watchlist()

            # Convert to format expected by watchlist
            candidate_dicts = [c.to_dict() for c in candidates]
            watchlist.ingest_candidates(candidate_dicts)

            logger.info(
                f"[ScannerCoordinator] Fed {len(candidates)} candidates to MomentumWatchlist"
            )
            return len(candidates)

        except ImportError:
            logger.warning("[ScannerCoordinator] MomentumWatchlist not available")
            return 0
        except Exception as e:
            logger.error(f"[ScannerCoordinator] Error feeding to watchlist: {e}")
            return 0

    def clear_all(self):
        """Clear all scanner candidates (call at end of session)"""
        with self._lock:
            get_gap_scanner().clear()
            get_gainer_scanner().clear()
            get_hod_scanner().clear()
            self._scan_results = {"GAPPER": [], "GAINER": [], "HOD": []}

    def clear_volume_caches(self):
        """Clear volume caches (call at start of new day)"""
        get_gainer_scanner().clear_volume_cache()
        get_hod_scanner().clear_volume_cache()

    def get_status(self) -> Dict:
        """Get coordinator status"""
        now_et = datetime.now(self._et_tz)
        active = self.get_active_scanners()

        return {
            "current_time_et": now_et.strftime("%H:%M:%S"),
            "active_scanners": active,
            "past_cutoff": self.is_past_cutoff(),
            "cutoff_time": self._cutoff_time.strftime("%H:%M"),
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
            "candidate_counts": {
                "GAPPER": len(self._scan_results.get("GAPPER", [])),
                "GAINER": len(self._scan_results.get("GAINER", [])),
                "HOD": len(self._scan_results.get("HOD", [])),
            },
            "total_candidates": len(self.get_all_candidates()),
            "scanner_status": {
                "gap": get_gap_scanner().get_status(),
                "gainer": get_gainer_scanner().get_status(),
                "hod": get_hod_scanner().get_status(),
            },
        }


# Singleton
_coordinator: Optional[ScannerCoordinator] = None


def get_scanner_coordinator() -> ScannerCoordinator:
    """Get the scanner coordinator instance"""
    global _coordinator
    if _coordinator is None:
        _coordinator = ScannerCoordinator()
    return _coordinator
