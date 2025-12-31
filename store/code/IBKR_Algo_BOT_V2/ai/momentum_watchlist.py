"""
Momentum Watchlist Manager - Session-Scoped Ranked View
========================================================

CRITICAL: The watchlist is NOT a persistent store.
It is a RANKED, SESSION-SCOPED VIEW of the strongest symbols RIGHT NOW.

Key Principles:
1. FULL RECOMPUTE every cycle - no symbol is exempt
2. RANK-FIRST, FILTER-SECOND - global ranking then top N
3. HARD REL_VOL FLOOR - symbols below threshold are excluded
4. SESSION BOUNDARY - new day = fresh start
5. DERIVED VIEW - watchlist is computed, not stored

Warrior Trading Model:
- Focus on the "front runner" - the stock with market attention NOW
- Relative volume is king - no volume = no play
- Gap + volume + momentum = dominance score
"""

import logging
from datetime import datetime, timezone, date, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Eastern timezone for market hours
ET = ZoneInfo('America/New_York')


def get_rel_vol_floor(now: Optional[datetime] = None) -> float:
    """
    Get dynamic relative volume floor based on time of day.

    Relative Volume is a MULTIPLIER, not a percentage.

    Time-based floors (Eastern Time):
    - 04:00-09:29 → rel_vol >= 5.0 (pre-market, need extreme volume)
    - 09:30-10:30 → rel_vol >= 3.0 (opening hour, high activity)
    - 10:30-12:00 → rel_vol >= 2.0 (mid-morning)
    - After 12:00  → rel_vol >= 1.5 (afternoon, lower bar)

    Returns:
        float: Minimum relative volume multiplier required
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Convert to Eastern time
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    et_now = now.astimezone(ET)
    current_time = et_now.time()

    # Pre-market: 04:00-09:29
    if current_time >= time(4, 0) and current_time < time(9, 30):
        return 5.0

    # Opening hour: 09:30-10:30
    if current_time >= time(9, 30) and current_time < time(10, 30):
        return 3.0

    # Mid-morning: 10:30-12:00
    if current_time >= time(10, 30) and current_time < time(12, 0):
        return 2.0

    # Afternoon and after-hours: 12:00+
    return 1.5


class ExclusionReason(Enum):
    """Why a symbol was excluded from active watchlist"""
    NONE = "NONE"  # Not excluded
    REL_VOL_BELOW_FLOOR = "REL_VOL_BELOW_FLOOR"
    RANK_BELOW_CUTOFF = "RANK_BELOW_CUTOFF"
    STALE_SESSION = "STALE_SESSION"
    PRICE_OUT_OF_RANGE = "PRICE_OUT_OF_RANGE"
    DATA_QUALITY_ISSUE = "DATA_QUALITY_ISSUE"
    MANUAL_EXCLUSION = "MANUAL_EXCLUSION"


@dataclass
class RankedSymbol:
    """A symbol with its current metrics and rank"""
    symbol: str

    # Core metrics (recomputed every cycle)
    gap_pct: float = 0.0
    rel_vol_daily: Optional[float] = None  # None if unavailable
    rel_vol_5m: Optional[float] = None
    volume_today: int = 0
    pct_gain: float = 0.0  # Intraday gain from open
    price: float = 0.0

    # HOD tracking
    hod_price: float = 0.0
    pct_from_hod: float = 0.0
    at_hod: bool = False
    near_hod: bool = False

    # Dominance score (computed)
    dominance_score: float = 0.0
    rank: int = 0

    # Status
    is_active: bool = False
    exclusion_reason: ExclusionReason = ExclusionReason.NONE

    # Timestamps
    first_seen: Optional[datetime] = None
    last_evaluated: Optional[datetime] = None
    session_date: Optional[date] = None

    # Data quality
    has_valid_rel_vol: bool = False
    has_valid_price: bool = False

    def compute_dominance_score(self) -> float:
        """
        Compute dominance score for ranking.

        Formula: Gap% * 0.3 + RelVol * 0.4 + Gain% * 0.2 + HOD_Bonus * 0.1

        This favors:
        - High gap (morning momentum)
        - High relative volume (market attention)
        - Intraday gains (continuation)
        - Near HOD (strength)
        """
        score = 0.0

        # Gap component (30% weight)
        gap_component = min(self.gap_pct, 50) / 50 * 30  # Cap at 50% gap
        score += gap_component

        # Relative volume component (40% weight) - MOST IMPORTANT
        if self.rel_vol_daily and self.rel_vol_daily > 0:
            # Normalize: 100% rel_vol = baseline, cap at 1000%
            rel_vol_normalized = min(self.rel_vol_daily, 10.0) / 10.0
            score += rel_vol_normalized * 40
            self.has_valid_rel_vol = True
        else:
            # No rel_vol data = severe penalty
            self.has_valid_rel_vol = False

        # Gain component (20% weight)
        gain_component = min(abs(self.pct_gain), 30) / 30 * 20  # Cap at 30% gain
        if self.pct_gain > 0:
            score += gain_component
        else:
            score -= gain_component * 0.5  # Penalize red stocks

        # HOD bonus (10% weight)
        if self.at_hod:
            score += 10
        elif self.near_hod:
            score += 5
        elif self.pct_from_hod < -5:
            score -= 5  # Penalty for extended pullback

        self.dominance_score = round(max(0, score), 2)
        return self.dominance_score

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "rank": self.rank,
            "dominance_score": self.dominance_score,
            "gap_pct": self.gap_pct,
            "rel_vol_daily": self.rel_vol_daily,
            "rel_vol_5m": self.rel_vol_5m,
            "volume_today": self.volume_today,
            "pct_gain": self.pct_gain,
            "price": self.price,
            "hod_price": self.hod_price,
            "pct_from_hod": self.pct_from_hod,
            "at_hod": self.at_hod,
            "near_hod": self.near_hod,
            "is_active": self.is_active,
            "exclusion_reason": self.exclusion_reason.value,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_evaluated": self.last_evaluated.isoformat() if self.last_evaluated else None,
            "session_date": self.session_date.isoformat() if self.session_date else None,
            "has_valid_rel_vol": self.has_valid_rel_vol,
            "has_valid_price": self.has_valid_price
        }


@dataclass
class WatchlistConfig:
    """Configuration for watchlist ranking"""
    # Top N cutoff - HARD CAP at 10
    max_active_symbols: int = 10

    # Relative volume schedule: (hour_start, hour_end, floor)
    # Times are ET. End hour is exclusive.
    rel_vol_schedule: Tuple[Tuple[int, int, float], ...] = (
        (4, 9, 5.0),    # 04:00-09:00 → rel_vol >= 5.0
        (9, 10, 3.0),   # 09:00-10:00 → rel_vol >= 3.0
        (10, 12, 2.0),  # 10:00-12:00 → rel_vol >= 2.0
        (12, 20, 1.5),  # 12:00-20:00 → rel_vol >= 1.5
    )

    # Fallback floor if no schedule matches
    min_rel_vol_floor: float = 1.5

    # Use dynamic rel_vol floor based on time of day
    use_dynamic_rel_vol: bool = True

    # Price range
    min_price: float = 1.0
    max_price: float = 20.0

    # Gap threshold
    min_gap_pct: float = 2.0  # DRY-RUN mode

    # Session boundary
    enforce_session_boundary: bool = True

    def current_rel_vol_floor(self, now: Optional[datetime] = None) -> float:
        """
        Get the current rel_vol floor from schedule.

        Uses rel_vol_schedule tuple to determine floor based on ET hour.
        Falls back to min_rel_vol_floor if no schedule matches.
        """
        if not self.use_dynamic_rel_vol:
            return self.min_rel_vol_floor

        if now is None:
            now = datetime.now(timezone.utc)

        # Convert to Eastern time
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        et_now = now.astimezone(ET)
        current_hour = et_now.hour

        # Find matching schedule entry
        for start_hr, end_hr, floor in self.rel_vol_schedule:
            if start_hr <= current_hour < end_hr:
                return floor

        return self.min_rel_vol_floor

    def get_current_rel_vol_floor(self) -> float:
        """Get the current rel_vol floor (backward compatible wrapper)"""
        return self.current_rel_vol_floor()

    def to_dict(self) -> Dict:
        current_floor = self.current_rel_vol_floor()
        return {
            "max_active_symbols": self.max_active_symbols,
            "min_rel_vol_floor": self.min_rel_vol_floor,
            "current_rel_vol_floor": current_floor,
            "use_dynamic_rel_vol": self.use_dynamic_rel_vol,
            "rel_vol_schedule": self.rel_vol_schedule,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "min_gap_pct": self.min_gap_pct,
            "enforce_session_boundary": self.enforce_session_boundary
        }


class MomentumWatchlist:
    """
    Session-scoped momentum watchlist.

    This is a DERIVED VIEW, not a persistent store.
    Every cycle, all symbols are re-evaluated and re-ranked.
    """

    def __init__(self, config: Optional[WatchlistConfig] = None):
        self.config = config or WatchlistConfig()
        self._current_session: Optional[date] = None
        self._all_candidates: Dict[str, RankedSymbol] = {}
        self._active_watchlist: List[RankedSymbol] = []
        self._archived: List[RankedSymbol] = []
        self._last_recompute: Optional[datetime] = None
        self._cycle_count: int = 0

        # Operator action log
        self._operator_actions: List[Dict] = []

    @property
    def session_date(self) -> date:
        """Current trading session date (Eastern Time)"""
        if self._current_session is None:
            # Use ET date to properly handle session boundary
            et_now = datetime.now(timezone.utc).astimezone(ET)
            self._current_session = et_now.date()
        return self._current_session

    @property
    def active_symbols(self) -> List[str]:
        """List of active symbol tickers"""
        return [s.symbol for s in self._active_watchlist]

    def _enforce_session_boundary(self):
        """Check if we've crossed into a new trading day (ET)"""
        et_now = datetime.now(timezone.utc).astimezone(ET)
        today = et_now.date()
        if self._current_session != today:
            logger.info(f"Session boundary crossed: {self._current_session} -> {today}")
            # Archive all current candidates
            for symbol in self._all_candidates.values():
                symbol.exclusion_reason = ExclusionReason.STALE_SESSION
                self._archived.append(symbol)

            # Reset for new session
            self._all_candidates = {}
            self._active_watchlist = []
            self._current_session = today
            self._cycle_count = 0

    def full_recompute(
        self,
        candidates: List[Dict],
        force_fresh: bool = True
    ) -> Tuple[List[RankedSymbol], Dict]:
        """
        FULL RECOMPUTE - Core operation.

        This is called every discovery cycle.
        ALL candidates are re-evaluated with current metrics.
        NO symbol is exempt because it already exists.

        Args:
            candidates: List of symbol dicts with current metrics
            force_fresh: If True, discard all prior state

        Returns:
            (active_watchlist, rank_snapshot)
        """
        self._cycle_count += 1
        now = datetime.now(timezone.utc)

        # Session boundary check
        if self.config.enforce_session_boundary:
            self._enforce_session_boundary()

        if force_fresh:
            self._all_candidates = {}

        # Step 1: Build/update RankedSymbol for each candidate
        evaluated = []
        for c in candidates:
            symbol = c.get("symbol", "")
            if not symbol:
                continue

            # Create fresh RankedSymbol (even if existed before)
            ranked = RankedSymbol(
                symbol=symbol,
                gap_pct=c.get("gap_pct") or 0,
                rel_vol_daily=c.get("rel_vol_daily"),
                rel_vol_5m=c.get("rel_vol_5m"),
                volume_today=c.get("volume_today") or c.get("volume") or 0,
                pct_gain=c.get("pct_gain") or c.get("change_pct") or 0,
                price=c.get("price") or 0,
                hod_price=c.get("hod_price") or c.get("high") or 0,
                pct_from_hod=c.get("pct_from_hod") or 0,
                at_hod=c.get("at_hod") or False,
                near_hod=c.get("near_hod") or False,
                first_seen=now,
                last_evaluated=now,
                session_date=self.session_date,
                has_valid_price=bool(c.get("price") and c.get("price") > 0)
            )

            # Compute dominance score
            ranked.compute_dominance_score()

            evaluated.append(ranked)

        # Step 2: Apply hard filters (exclusions)
        passed_filters = []
        excluded = []

        for ranked in evaluated:
            exclusion = self._check_exclusions(ranked)
            if exclusion != ExclusionReason.NONE:
                ranked.exclusion_reason = exclusion
                ranked.is_active = False
                excluded.append(ranked)
            else:
                passed_filters.append(ranked)

        # Step 3: RANK by dominance score (descending)
        passed_filters.sort(key=lambda x: x.dominance_score, reverse=True)

        # Assign ranks
        for i, ranked in enumerate(passed_filters):
            ranked.rank = i + 1

        # Step 4: Take TOP N only
        active = []
        below_cutoff = []

        for ranked in passed_filters:
            if ranked.rank <= self.config.max_active_symbols:
                ranked.is_active = True
                ranked.exclusion_reason = ExclusionReason.NONE
                active.append(ranked)
            else:
                ranked.is_active = False
                ranked.exclusion_reason = ExclusionReason.RANK_BELOW_CUTOFF
                below_cutoff.append(ranked)

        # Update internal state
        self._active_watchlist = active
        self._all_candidates = {s.symbol: s for s in evaluated}
        self._last_recompute = now

        # Get current rel_vol floor for reporting
        current_rel_vol_floor = self.config.get_current_rel_vol_floor()

        # Build rank snapshot report
        snapshot = {
            "cycle_number": self._cycle_count,
            "session_date": self.session_date.isoformat(),
            "timestamp": now.isoformat(),
            "config": self.config.to_dict(),
            "current_rel_vol_floor": current_rel_vol_floor,
            "total_candidates": len(evaluated),
            "passed_filters": len(passed_filters),
            "excluded_count": len(excluded),
            "active_count": len(active),
            "below_cutoff_count": len(below_cutoff),
            "active_watchlist": [s.to_dict() for s in active],
            "below_cutoff": [s.to_dict() for s in below_cutoff[:10]],  # Top 10 below cutoff
            "excluded": [
                {
                    "symbol": s.symbol,
                    "reason": s.exclusion_reason.value,
                    "dominance_score": s.dominance_score,
                    "rel_vol_daily": s.rel_vol_daily,
                    "gap_pct": s.gap_pct,
                    "floor_used": current_rel_vol_floor
                }
                for s in excluded
            ],
            "exclusion_summary": self._summarize_exclusions(excluded),
            "ranking_proof": {
                "top_5": [
                    {"rank": s.rank, "symbol": s.symbol, "score": s.dominance_score}
                    for s in active[:5]
                ],
                "recomputed_all": True,
                "force_fresh": force_fresh,
                "rel_vol_floor_applied": current_rel_vol_floor
            }
        }

        logger.info(
            f"Watchlist recompute cycle {self._cycle_count}: "
            f"{len(active)} active, {len(excluded)} excluded, "
            f"{len(below_cutoff)} below cutoff"
        )

        return active, snapshot

    def _check_exclusions(self, ranked: RankedSymbol) -> ExclusionReason:
        """
        Check if symbol should be excluded.

        HARD RULES - NO GRANDFATHERING:
        - Symbols below rel_vol floor are REJECTED immediately
        - Symbols without rel_vol data are REJECTED (no free passes)
        - Price must be in range
        """

        # Price range check
        if ranked.price < self.config.min_price or ranked.price > self.config.max_price:
            return ExclusionReason.PRICE_OUT_OF_RANGE

        # HARD RELATIVE VOLUME FLOOR - DYNAMIC BY TIME OF DAY
        # Get current floor based on market time
        current_floor = self.config.get_current_rel_vol_floor()

        if ranked.rel_vol_daily is None:
            # NO REL_VOL DATA = REJECTED
            # Cannot enter watchlist without volume confirmation
            logger.debug(f"{ranked.symbol}: REJECTED - no rel_vol data (floor={current_floor})")
            return ExclusionReason.REL_VOL_BELOW_FLOOR

        if ranked.rel_vol_daily < current_floor:
            # BELOW FLOOR = REJECTED
            logger.debug(
                f"{ranked.symbol}: REJECTED - rel_vol {ranked.rel_vol_daily:.2f} < floor {current_floor}"
            )
            return ExclusionReason.REL_VOL_BELOW_FLOOR

        # Gap threshold (not a hard exclusion, but logged)
        if abs(ranked.gap_pct) < self.config.min_gap_pct:
            # Low gap = lower score, but not excluded
            pass

        # Data quality check
        if not ranked.has_valid_price:
            return ExclusionReason.DATA_QUALITY_ISSUE

        return ExclusionReason.NONE

    def _summarize_exclusions(self, excluded: List[RankedSymbol]) -> Dict:
        """Summarize exclusion reasons"""
        summary = {}
        for s in excluded:
            reason = s.exclusion_reason.value
            if reason not in summary:
                summary[reason] = 0
            summary[reason] += 1
        return summary

    def get_active_watchlist(self) -> List[Dict]:
        """Get current active watchlist with full transparency"""
        return [s.to_dict() for s in self._active_watchlist]

    def get_symbol_status(self, symbol: str) -> Optional[Dict]:
        """Get detailed status for a specific symbol"""
        ranked = self._all_candidates.get(symbol)
        if ranked:
            return ranked.to_dict()
        return None

    # =========================================================================
    # OPERATOR CONTROLS
    # =========================================================================

    def purge_all(self, triggered_by: str = "manual") -> Dict:
        """
        PURGE ALL WATCHLIST

        - Removes all ACTIVE symbols
        - Resets discovery cache
        - Preserves historical reports (handled externally)
        - Does NOT approve trades or alter thresholds
        """
        now = datetime.now(timezone.utc)

        symbols_before = [s.symbol for s in self._active_watchlist]

        # Archive current active symbols
        for s in self._active_watchlist:
            s.is_active = False
            s.exclusion_reason = ExclusionReason.MANUAL_EXCLUSION
            self._archived.append(s)

        # Clear active watchlist
        self._active_watchlist = []
        self._all_candidates = {}

        action = {
            "action_type": "purge",
            "timestamp": now.isoformat(),
            "symbols_before": symbols_before,
            "symbols_after": [],
            "triggered_by": triggered_by,
            "session_date": self.session_date.isoformat()
        }
        self._operator_actions.append(action)

        logger.warning(f"OPERATOR: Purged all watchlist ({len(symbols_before)} symbols)")

        return action

    def get_operator_actions(self) -> List[Dict]:
        """Get log of all operator actions"""
        return self._operator_actions

    def get_status(self) -> Dict:
        """Get current watchlist status"""
        return {
            "session_date": self.session_date.isoformat(),
            "cycle_count": self._cycle_count,
            "last_recompute": self._last_recompute.isoformat() if self._last_recompute else None,
            "active_count": len(self._active_watchlist),
            "total_candidates": len(self._all_candidates),
            "archived_count": len(self._archived),
            "config": self.config.to_dict()
        }


# Global singleton
_momentum_watchlist: Optional[MomentumWatchlist] = None


def get_momentum_watchlist() -> MomentumWatchlist:
    """Get or create the global momentum watchlist"""
    global _momentum_watchlist
    if _momentum_watchlist is None:
        _momentum_watchlist = MomentumWatchlist()
    return _momentum_watchlist


def reset_momentum_watchlist():
    """Reset the watchlist (for testing or new session)"""
    global _momentum_watchlist
    _momentum_watchlist = None
