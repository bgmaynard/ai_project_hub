"""
Overnight Continuation Scanner
==============================
Detect after-hours runners that continue into pre-market.

Ross Cameron Insight:
- Stocks that move in after-hours on news often continue pre-market
- Continuation = confirmation of momentum
- Reversals (AH up, PM down) = warning sign
- Best setups: Strong AH move + PM continuation + volume

Sessions:
- After-Hours (AH): 4:00 PM - 8:00 PM ET
- Pre-Market (PM): 4:00 AM - 9:30 AM ET

Continuation Patterns:
1. STRONG_CONTINUATION - AH up + PM up (same direction, strong)
2. WEAK_CONTINUATION - AH up + PM flat (holding gains)
3. REVERSAL - AH up + PM down (warning, avoid)
4. ACCELERATION - PM move > AH move (increasing momentum)
5. FADE - AH big + PM losing ground (exhaustion)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ContinuationPattern(Enum):
    """Types of overnight continuation patterns"""

    STRONG_CONTINUATION = "STRONG_CONTINUATION"  # AH and PM same direction, strong
    WEAK_CONTINUATION = "WEAK_CONTINUATION"  # AH move, PM flat (holding)
    ACCELERATION = "ACCELERATION"  # PM move > AH move
    FADE = "FADE"  # PM losing AH gains
    REVERSAL = "REVERSAL"  # PM opposite of AH
    NO_MOVEMENT = "NO_MOVEMENT"  # Nothing significant


class ContinuationStrength(Enum):
    """Strength of continuation signal"""

    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NONE = "NONE"


@dataclass
class SessionMove:
    """Movement during a trading session"""

    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    close_price: float = 0.0
    volume: int = 0
    change_pct: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "open": round(self.open_price, 4),
            "high": round(self.high_price, 4),
            "low": round(self.low_price, 4),
            "close": round(self.close_price, 4),
            "volume": self.volume,
            "change_pct": round(self.change_pct, 2),
        }


@dataclass
class OvernightMover:
    """Stock with overnight movement data"""

    symbol: str

    # Session data
    regular_close: float = 0.0  # 4 PM close
    after_hours: SessionMove = field(default_factory=SessionMove)
    premarket: SessionMove = field(default_factory=SessionMove)

    # Pattern analysis
    pattern: ContinuationPattern = ContinuationPattern.NO_MOVEMENT
    strength: ContinuationStrength = ContinuationStrength.NONE
    continuation_score: float = 0.0  # 0-100

    # Combined metrics
    total_overnight_change: float = 0.0  # AH + PM combined
    ah_contribution: float = 0.0  # % of move from AH
    pm_contribution: float = 0.0  # % of move from PM

    # Volume analysis
    ah_volume_ratio: float = 0.0  # AH vol vs avg daily
    pm_volume_ratio: float = 0.0  # PM vol vs avg daily
    avg_daily_volume: int = 0

    # Catalyst
    has_catalyst: bool = False
    catalyst_headline: str = ""
    catalyst_time: str = ""

    # Trading levels
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # Timestamps
    last_update: str = ""

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "regular_close": round(self.regular_close, 4),
            "after_hours": self.after_hours.to_dict(),
            "premarket": self.premarket.to_dict(),
            "pattern": self.pattern.value,
            "strength": self.strength.value,
            "continuation_score": round(self.continuation_score, 1),
            "total_overnight_change": round(self.total_overnight_change, 2),
            "ah_contribution": round(self.ah_contribution, 1),
            "pm_contribution": round(self.pm_contribution, 1),
            "ah_volume_ratio": round(self.ah_volume_ratio, 2),
            "pm_volume_ratio": round(self.pm_volume_ratio, 2),
            "has_catalyst": self.has_catalyst,
            "catalyst_headline": self.catalyst_headline,
            "entry_price": round(self.entry_price, 4),
            "stop_loss": round(self.stop_loss, 4),
            "targets": {
                "target_1": round(self.target_1, 4),
                "target_2": round(self.target_2, 4),
            },
            "warnings": self.warnings,
            "last_update": self.last_update,
        }


class OvernightContinuationScanner:
    """
    Scan for overnight continuation patterns.

    Best setups have:
    1. Significant after-hours move (>3%)
    2. Pre-market continuation in same direction
    3. Strong volume in both sessions
    4. Clear catalyst (news, earnings, FDA)
    """

    # Thresholds
    MIN_AH_MOVE_PCT = 3.0  # Minimum AH move to track
    MIN_PM_CONTINUATION_PCT = 1.0  # Minimum PM move for continuation
    REVERSAL_THRESHOLD_PCT = -1.0  # PM move opposite of AH
    FADE_THRESHOLD_PCT = -0.5  # PM giving back AH gains

    # Volume thresholds
    MIN_AH_VOLUME = 50000  # Minimum AH volume
    MIN_PM_VOLUME = 100000  # Minimum PM volume
    GOOD_VOLUME_RATIO = 0.1  # 10% of avg daily = good

    # Scoring weights
    WEIGHT_AH_MOVE = 25
    WEIGHT_PM_CONTINUATION = 30
    WEIGHT_VOLUME = 20
    WEIGHT_CATALYST = 25

    def __init__(self):
        self.movers: Dict[str, OvernightMover] = {}
        self.ah_movers: Dict[str, SessionMove] = {}  # Track AH moves first
        self.scan_history: List[Dict] = []

        logger.info("OvernightContinuationScanner initialized")

    def record_after_hours_move(
        self,
        symbol: str,
        regular_close: float,
        ah_high: float,
        ah_low: float,
        ah_close: float,
        ah_volume: int,
        avg_daily_volume: int = 0,
        catalyst: str = "",
    ) -> Optional[OvernightMover]:
        """
        Record an after-hours move. Call this at end of AH session (~8 PM).
        """
        symbol = symbol.upper()

        if regular_close <= 0:
            return None

        ah_change = ((ah_close - regular_close) / regular_close) * 100

        # Only track significant moves
        if abs(ah_change) < self.MIN_AH_MOVE_PCT:
            return None

        if ah_volume < self.MIN_AH_VOLUME:
            return None

        # Create session move
        ah_move = SessionMove(
            open_price=regular_close,
            high_price=ah_high,
            low_price=ah_low,
            close_price=ah_close,
            volume=ah_volume,
            change_pct=ah_change,
        )

        # Calculate volume ratio
        vol_ratio = 0.0
        if avg_daily_volume > 0:
            vol_ratio = ah_volume / avg_daily_volume

        # Create overnight mover
        mover = OvernightMover(
            symbol=symbol,
            regular_close=regular_close,
            after_hours=ah_move,
            avg_daily_volume=avg_daily_volume,
            ah_volume_ratio=vol_ratio,
            has_catalyst=bool(catalyst),
            catalyst_headline=catalyst,
            last_update=datetime.now().isoformat(),
        )

        self.movers[symbol] = mover
        self.ah_movers[symbol] = ah_move

        logger.info(
            f"AH MOVE RECORDED: {symbol} {ah_change:+.1f}% | "
            f"Vol: {ah_volume:,} | Catalyst: {catalyst[:50] if catalyst else 'None'}"
        )

        return mover

    def update_premarket(
        self,
        symbol: str,
        pm_open: float,
        pm_high: float,
        pm_low: float,
        pm_current: float,
        pm_volume: int,
    ) -> Optional[OvernightMover]:
        """
        Update pre-market data and analyze continuation pattern.
        Call during pre-market session (4 AM - 9:30 AM).
        """
        symbol = symbol.upper()

        if symbol not in self.movers:
            return None

        mover = self.movers[symbol]

        # Calculate PM change from AH close
        if mover.after_hours.close_price > 0:
            pm_change = (
                (pm_current - mover.after_hours.close_price)
                / mover.after_hours.close_price
            ) * 100
        else:
            pm_change = 0.0

        # Update PM session data
        mover.premarket = SessionMove(
            open_price=pm_open,
            high_price=pm_high,
            low_price=pm_low,
            close_price=pm_current,
            volume=pm_volume,
            change_pct=pm_change,
        )

        # Calculate PM volume ratio
        if mover.avg_daily_volume > 0:
            mover.pm_volume_ratio = pm_volume / mover.avg_daily_volume

        # Calculate total overnight change
        mover.total_overnight_change = mover.after_hours.change_pct + pm_change

        # Calculate contribution ratios
        total_abs = abs(mover.after_hours.change_pct) + abs(pm_change)
        if total_abs > 0:
            mover.ah_contribution = (
                abs(mover.after_hours.change_pct) / total_abs
            ) * 100
            mover.pm_contribution = (abs(pm_change) / total_abs) * 100

        # Determine continuation pattern
        mover.pattern = self._determine_pattern(mover)

        # Score the continuation
        mover.continuation_score = self._score_continuation(mover)

        # Determine strength
        mover.strength = self._determine_strength(mover.continuation_score)

        # Calculate trading levels
        self._calculate_levels(mover)

        # Generate warnings
        mover.warnings = self._get_warnings(mover)

        mover.last_update = datetime.now().isoformat()

        logger.info(
            f"PM UPDATE: {symbol} | Pattern: {mover.pattern.value} | "
            f"Score: {mover.continuation_score:.0f} | "
            f"Total: {mover.total_overnight_change:+.1f}%"
        )

        return mover

    def _determine_pattern(self, mover: OvernightMover) -> ContinuationPattern:
        """Determine the continuation pattern"""
        ah_pct = mover.after_hours.change_pct
        pm_pct = mover.premarket.change_pct

        # Check if same direction
        same_direction = (ah_pct > 0 and pm_pct > 0) or (ah_pct < 0 and pm_pct < 0)

        if same_direction:
            # Check if PM is accelerating
            if abs(pm_pct) > abs(ah_pct):
                return ContinuationPattern.ACCELERATION

            # Strong continuation
            if abs(pm_pct) >= self.MIN_PM_CONTINUATION_PCT:
                return ContinuationPattern.STRONG_CONTINUATION

            # Weak continuation (just holding)
            return ContinuationPattern.WEAK_CONTINUATION

        else:
            # Opposite direction
            if ah_pct > 0:  # AH was up
                if pm_pct < self.REVERSAL_THRESHOLD_PCT:
                    return ContinuationPattern.REVERSAL
                elif pm_pct < self.FADE_THRESHOLD_PCT:
                    return ContinuationPattern.FADE
            else:  # AH was down
                if pm_pct > abs(self.REVERSAL_THRESHOLD_PCT):
                    return ContinuationPattern.REVERSAL
                elif pm_pct > abs(self.FADE_THRESHOLD_PCT):
                    return ContinuationPattern.FADE

        # Default - minimal movement
        if abs(pm_pct) < 0.5:
            return ContinuationPattern.WEAK_CONTINUATION

        return ContinuationPattern.NO_MOVEMENT

    def _score_continuation(self, mover: OvernightMover) -> float:
        """Score the continuation setup (0-100)"""
        score = 0.0

        # 1. After-hours move score (0-25)
        ah_abs = abs(mover.after_hours.change_pct)
        if ah_abs >= 10:
            score += 25
        elif ah_abs >= 7:
            score += 20
        elif ah_abs >= 5:
            score += 15
        elif ah_abs >= 3:
            score += 10

        # 2. Pre-market continuation score (0-30)
        pm_pct = mover.premarket.change_pct
        ah_pct = mover.after_hours.change_pct

        if mover.pattern == ContinuationPattern.ACCELERATION:
            score += 30  # Best pattern
        elif mover.pattern == ContinuationPattern.STRONG_CONTINUATION:
            score += 25
        elif mover.pattern == ContinuationPattern.WEAK_CONTINUATION:
            score += 15
        elif mover.pattern == ContinuationPattern.FADE:
            score += 5  # Losing ground but not reversing
        elif mover.pattern == ContinuationPattern.REVERSAL:
            score += 0  # Worst pattern

        # 3. Volume score (0-20)
        combined_vol_ratio = mover.ah_volume_ratio + mover.pm_volume_ratio
        if combined_vol_ratio >= 0.3:
            score += 20
        elif combined_vol_ratio >= 0.2:
            score += 15
        elif combined_vol_ratio >= 0.1:
            score += 10
        elif combined_vol_ratio >= 0.05:
            score += 5

        # 4. Catalyst score (0-25)
        if mover.has_catalyst:
            headline = mover.catalyst_headline.lower()

            # High-value catalysts
            if any(
                kw in headline
                for kw in ["fda", "approval", "earnings beat", "merger", "acquisition"]
            ):
                score += 25
            elif any(
                kw in headline for kw in ["contract", "earnings", "upgrade", "clinical"]
            ):
                score += 20
            elif any(kw in headline for kw in ["partnership", "deal", "agreement"]):
                score += 15
            else:
                score += 10  # Generic catalyst
        else:
            score += 0  # No catalyst = less reliable

        return min(100, score)

    def _determine_strength(self, score: float) -> ContinuationStrength:
        """Determine strength from score"""
        if score >= 75:
            return ContinuationStrength.STRONG
        elif score >= 50:
            return ContinuationStrength.MODERATE
        elif score >= 25:
            return ContinuationStrength.WEAK
        else:
            return ContinuationStrength.NONE

    def _calculate_levels(self, mover: OvernightMover):
        """Calculate entry, stop, and target levels"""
        current = mover.premarket.close_price
        ah_change = mover.after_hours.change_pct

        if current <= 0:
            return

        if ah_change > 0:  # Bullish overnight
            # Entry: current price or slight dip
            mover.entry_price = current * 0.995

            # Stop: Below pre-market low or AH low
            pm_low = mover.premarket.low_price or current * 0.95
            ah_low = mover.after_hours.low_price or current * 0.95
            mover.stop_loss = min(pm_low, ah_low) * 0.99

            # Targets based on total overnight move
            total_move = abs(mover.total_overnight_change)
            mover.target_1 = current * (1 + total_move / 200)  # Half extension
            mover.target_2 = current * (1 + total_move / 100)  # Full extension

        else:  # Bearish overnight (we don't short, but track anyway)
            mover.entry_price = current
            mover.stop_loss = current * 1.03
            mover.target_1 = current * 0.97
            mover.target_2 = current * 0.95

    def _get_warnings(self, mover: OvernightMover) -> List[str]:
        """Generate warnings for the mover"""
        warnings = []

        if mover.pattern == ContinuationPattern.REVERSAL:
            warnings.append("REVERSAL: PM moving opposite of AH - high risk")

        if mover.pattern == ContinuationPattern.FADE:
            warnings.append("FADE: Losing AH gains in PM - momentum exhausting")

        if mover.pm_volume_ratio < 0.05:
            warnings.append("Low PM volume - may lack conviction")

        if not mover.has_catalyst:
            warnings.append("No clear catalyst - gap may fade")

        if mover.after_hours.change_pct < 0:
            warnings.append("Bearish overnight move - avoid longs")

        if abs(mover.total_overnight_change) > 30:
            warnings.append("Extreme overnight move - very high risk")

        return warnings

    def get_continuations(
        self, min_strength: ContinuationStrength = ContinuationStrength.WEAK
    ) -> List[OvernightMover]:
        """Get all continuation setups above minimum strength"""
        strength_order = {
            ContinuationStrength.STRONG: 0,
            ContinuationStrength.MODERATE: 1,
            ContinuationStrength.WEAK: 2,
            ContinuationStrength.NONE: 3,
        }

        min_order = strength_order[min_strength]

        results = [
            m
            for m in self.movers.values()
            if strength_order[m.strength] <= min_order
            and m.pattern
            not in [ContinuationPattern.REVERSAL, ContinuationPattern.NO_MOVEMENT]
        ]

        # Sort by score descending
        results.sort(key=lambda x: x.continuation_score, reverse=True)

        return results

    def get_strong_continuations(self) -> List[OvernightMover]:
        """Get only strong continuations (STRONG_CONTINUATION or ACCELERATION)"""
        return [
            m
            for m in self.movers.values()
            if m.pattern
            in [
                ContinuationPattern.STRONG_CONTINUATION,
                ContinuationPattern.ACCELERATION,
            ]
            and m.strength
            in [ContinuationStrength.STRONG, ContinuationStrength.MODERATE]
        ]

    def get_accelerators(self) -> List[OvernightMover]:
        """Get stocks where PM is accelerating vs AH (increasing momentum)"""
        return [
            m
            for m in self.movers.values()
            if m.pattern == ContinuationPattern.ACCELERATION
        ]

    def get_reversals(self) -> List[OvernightMover]:
        """Get stocks that reversed in PM (for avoidance)"""
        return [
            m for m in self.movers.values() if m.pattern == ContinuationPattern.REVERSAL
        ]

    def get_faders(self) -> List[OvernightMover]:
        """Get stocks fading in PM (for caution)"""
        return [
            m for m in self.movers.values() if m.pattern == ContinuationPattern.FADE
        ]

    def get_mover(self, symbol: str) -> Optional[OvernightMover]:
        """Get overnight mover by symbol"""
        return self.movers.get(symbol.upper())

    def get_bullish_continuations(self) -> List[OvernightMover]:
        """Get bullish continuations only (AH up, PM continuing)"""
        return [
            m
            for m in self.get_continuations()
            if m.after_hours.change_pct > 0 and m.premarket.change_pct >= 0
        ]

    def clear_movers(self):
        """Clear all movers (call at end of day)"""
        self.movers.clear()
        self.ah_movers.clear()
        logger.info("Overnight continuation scanner cleared")

    def get_status(self) -> Dict:
        """Get scanner status"""
        pattern_counts = {p.value: 0 for p in ContinuationPattern}
        strength_counts = {s.value: 0 for s in ContinuationStrength}

        for m in self.movers.values():
            pattern_counts[m.pattern.value] += 1
            strength_counts[m.strength.value] += 1

        return {
            "total_movers": len(self.movers),
            "ah_movers_tracked": len(self.ah_movers),
            "pattern_distribution": pattern_counts,
            "strength_distribution": strength_counts,
            "strong_continuations": len(self.get_strong_continuations()),
            "accelerators": len(self.get_accelerators()),
            "reversals": len(self.get_reversals()),
            "faders": len(self.get_faders()),
            "bullish": len(self.get_bullish_continuations()),
            "top_movers": [
                {
                    "symbol": m.symbol,
                    "pattern": m.pattern.value,
                    "score": m.continuation_score,
                    "total_change": m.total_overnight_change,
                }
                for m in sorted(
                    self.movers.values(),
                    key=lambda x: x.continuation_score,
                    reverse=True,
                )[:5]
            ],
        }


# Singleton instance
_scanner: Optional[OvernightContinuationScanner] = None


def get_overnight_scanner() -> OvernightContinuationScanner:
    """Get singleton scanner instance"""
    global _scanner
    if _scanner is None:
        _scanner = OvernightContinuationScanner()
    return _scanner


# Convenience functions
def record_ah_move(
    symbol: str, regular_close: float, ah_close: float, ah_volume: int, **kwargs
) -> Optional[OvernightMover]:
    """Quick record of after-hours move"""
    return get_overnight_scanner().record_after_hours_move(
        symbol=symbol,
        regular_close=regular_close,
        ah_high=kwargs.get("ah_high", ah_close),
        ah_low=kwargs.get("ah_low", ah_close),
        ah_close=ah_close,
        ah_volume=ah_volume,
        avg_daily_volume=kwargs.get("avg_volume", 0),
        catalyst=kwargs.get("catalyst", ""),
    )


def update_pm(
    symbol: str, pm_current: float, pm_volume: int, **kwargs
) -> Optional[OvernightMover]:
    """Quick update of pre-market data"""
    return get_overnight_scanner().update_premarket(
        symbol=symbol,
        pm_open=kwargs.get("pm_open", pm_current),
        pm_high=kwargs.get("pm_high", pm_current),
        pm_low=kwargs.get("pm_low", pm_current),
        pm_current=pm_current,
        pm_volume=pm_volume,
    )


def get_tradeable_continuations() -> List[OvernightMover]:
    """Get bullish continuations suitable for trading"""
    return get_overnight_scanner().get_bullish_continuations()
