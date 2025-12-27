"""
Gap Grader - Ross Cameron Methodology
======================================
Grade pre-market gaps by quality for trading potential.

Ross Cameron Gap Criteria:
1. GAP SIZE - 5-20% is ideal (too small = no momentum, too big = risky)
2. VOLUME - Pre-market volume must be significant (>100K shares)
3. FLOAT - Low float (<20M) gaps harder, more potential
4. CATALYST - News-driven gaps more reliable than technical gaps
5. PRIOR DAY - Continuation gaps (prior day green) are stronger
6. PRICE RANGE - $2-$20 ideal for scalping

Gap Grades:
- A = Perfect setup (high probability)
- B = Good setup (trade with normal size)
- C = Okay setup (reduced size)
- D = Marginal (watch only)
- F = Avoid (false gap or too risky)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time, timedelta
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class GapGrade(Enum):
    """Gap quality grades"""
    A = "A"  # Perfect setup
    B = "B"  # Good setup
    C = "C"  # Okay setup
    D = "D"  # Marginal
    F = "F"  # Avoid


class GapType(Enum):
    """Type of gap"""
    GAP_UP = "GAP_UP"
    GAP_DOWN = "GAP_DOWN"
    CONTINUATION_UP = "CONTINUATION_UP"    # Prior day green + gap up
    CONTINUATION_DOWN = "CONTINUATION_DOWN" # Prior day red + gap down
    REVERSAL_UP = "REVERSAL_UP"            # Prior day red + gap up
    REVERSAL_DOWN = "REVERSAL_DOWN"        # Prior day green + gap down


class CatalystType(Enum):
    """Type of catalyst driving the gap"""
    FDA = "FDA"                  # FDA approval/news
    EARNINGS = "EARNINGS"        # Earnings beat/miss
    CONTRACT = "CONTRACT"        # Major contract
    MERGER = "MERGER"            # M&A news
    UPGRADE = "UPGRADE"          # Analyst upgrade
    DOWNGRADE = "DOWNGRADE"      # Analyst downgrade
    OFFERING = "OFFERING"        # Secondary offering (usually bearish)
    PARTNERSHIP = "PARTNERSHIP"  # Strategic partnership
    CLINICAL = "CLINICAL"        # Clinical trial results
    OTHER = "OTHER"              # Other catalyst
    NONE = "NONE"                # No clear catalyst (technical gap)


@dataclass
class GapScore:
    """Detailed gap scoring breakdown"""
    gap_size_score: int = 0      # 0-20 points
    volume_score: int = 0        # 0-20 points
    float_score: int = 0         # 0-15 points
    catalyst_score: int = 0      # 0-25 points
    prior_day_score: int = 0     # 0-10 points
    price_score: int = 0         # 0-10 points

    @property
    def total(self) -> int:
        return (self.gap_size_score + self.volume_score + self.float_score +
                self.catalyst_score + self.prior_day_score + self.price_score)

    def to_dict(self) -> Dict:
        return {
            "gap_size": self.gap_size_score,
            "volume": self.volume_score,
            "float": self.float_score,
            "catalyst": self.catalyst_score,
            "prior_day": self.prior_day_score,
            "price": self.price_score,
            "total": self.total
        }


@dataclass
class GradedGap:
    """A graded gap with full analysis"""
    symbol: str
    grade: GapGrade
    score: GapScore
    gap_type: GapType
    catalyst_type: CatalystType

    # Gap metrics
    gap_percent: float = 0.0
    current_price: float = 0.0
    prior_close: float = 0.0
    premarket_high: float = 0.0
    premarket_low: float = 0.0
    premarket_volume: int = 0

    # Stock info
    float_shares: int = 0
    avg_volume: int = 0
    relative_volume: float = 0.0

    # Prior day info
    prior_day_change: float = 0.0  # % change prior day
    prior_day_volume: int = 0

    # Catalyst info
    catalyst_headline: str = ""
    catalyst_time: str = ""

    # Trading recommendation
    entry_zone_low: float = 0.0
    entry_zone_high: float = 0.0
    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    position_size_pct: float = 100.0  # % of normal position

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # Timestamps
    scanned_at: str = ""
    market_session: str = ""  # "premarket", "open", "regular"

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "grade": self.grade.value,
            "score": self.score.to_dict(),
            "gap_type": self.gap_type.value,
            "catalyst_type": self.catalyst_type.value,
            "gap_percent": round(self.gap_percent, 2),
            "current_price": round(self.current_price, 4),
            "prior_close": round(self.prior_close, 4),
            "premarket_volume": self.premarket_volume,
            "float_millions": round(self.float_shares / 1_000_000, 2) if self.float_shares else 0,
            "relative_volume": round(self.relative_volume, 2),
            "prior_day_change": round(self.prior_day_change, 2),
            "catalyst_headline": self.catalyst_headline,
            "entry_zone": {
                "low": round(self.entry_zone_low, 4),
                "high": round(self.entry_zone_high, 4)
            },
            "stop_loss": round(self.stop_loss, 4),
            "targets": {
                "target_1": round(self.target_1, 4),
                "target_2": round(self.target_2, 4)
            },
            "position_size_pct": self.position_size_pct,
            "warnings": self.warnings,
            "scanned_at": self.scanned_at,
            "market_session": self.market_session
        }


class GapGrader:
    """
    Grade pre-market gaps using Ross Cameron methodology.

    Scoring Breakdown (100 points max):
    - Gap Size: 20 points (5-15% optimal)
    - Volume: 20 points (pre-market volume significance)
    - Float: 15 points (lower float = higher potential)
    - Catalyst: 25 points (news-driven gaps)
    - Prior Day: 10 points (continuation patterns)
    - Price: 10 points ($2-$20 optimal)

    Grades:
    - A: 80+ points
    - B: 65-79 points
    - C: 50-64 points
    - D: 35-49 points
    - F: <35 points
    """

    # Scoring thresholds
    GRADE_THRESHOLDS = {
        GapGrade.A: 80,
        GapGrade.B: 65,
        GapGrade.C: 50,
        GapGrade.D: 35,
        GapGrade.F: 0
    }

    # Optimal ranges
    OPTIMAL_GAP_MIN = 5.0
    OPTIMAL_GAP_MAX = 15.0
    MAX_SAFE_GAP = 30.0

    OPTIMAL_PRICE_MIN = 2.0
    OPTIMAL_PRICE_MAX = 20.0

    LOW_FLOAT_THRESHOLD = 20_000_000  # 20M
    MICRO_FLOAT_THRESHOLD = 5_000_000  # 5M

    MIN_PREMARKET_VOLUME = 50_000
    GOOD_PREMARKET_VOLUME = 200_000
    GREAT_PREMARKET_VOLUME = 500_000

    def __init__(self):
        self.graded_gaps: Dict[str, GradedGap] = {}
        self.scan_history: List[Dict] = []

        # Catalyst keywords for detection
        self.catalyst_keywords = {
            CatalystType.FDA: ['fda', 'approval', 'approved', 'clearance', 'pdufa'],
            CatalystType.EARNINGS: ['earnings', 'eps', 'revenue', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'beat', 'miss'],
            CatalystType.CONTRACT: ['contract', 'award', 'deal', 'agreement', 'order'],
            CatalystType.MERGER: ['merger', 'acquisition', 'acquire', 'buyout', 'takeover', 'm&a'],
            CatalystType.UPGRADE: ['upgrade', 'raises', 'price target', 'outperform', 'buy rating'],
            CatalystType.DOWNGRADE: ['downgrade', 'lowers', 'underperform', 'sell rating'],
            CatalystType.OFFERING: ['offering', 'secondary', 'dilution', 'shelf', 'atm'],
            CatalystType.PARTNERSHIP: ['partnership', 'collaboration', 'alliance', 'joint venture'],
            CatalystType.CLINICAL: ['clinical', 'trial', 'phase', 'study', 'data', 'results', 'efficacy']
        }

    def _detect_catalyst(self, headline: str) -> CatalystType:
        """Detect catalyst type from headline"""
        if not headline:
            return CatalystType.NONE

        headline_lower = headline.lower()

        for catalyst_type, keywords in self.catalyst_keywords.items():
            if any(kw in headline_lower for kw in keywords):
                return catalyst_type

        return CatalystType.OTHER if headline else CatalystType.NONE

    def _score_gap_size(self, gap_pct: float) -> int:
        """Score gap size (0-20 points)"""
        gap_abs = abs(gap_pct)

        if gap_abs < 3:
            return 2  # Too small
        elif gap_abs < 5:
            return 8  # Borderline
        elif gap_abs <= 10:
            return 20  # Optimal
        elif gap_abs <= 15:
            return 18  # Good
        elif gap_abs <= 20:
            return 12  # Getting risky
        elif gap_abs <= 30:
            return 6  # Very risky
        else:
            return 0  # Too extreme

    def _score_volume(self, premarket_vol: int, avg_vol: int) -> int:
        """Score pre-market volume (0-20 points)"""
        if premarket_vol < self.MIN_PREMARKET_VOLUME:
            return 2  # Too low
        elif premarket_vol < self.GOOD_PREMARKET_VOLUME:
            return 8  # Okay
        elif premarket_vol < self.GREAT_PREMARKET_VOLUME:
            return 14  # Good
        else:
            return 20  # Great

        # Bonus for relative volume
        # if avg_vol > 0:
        #     rvol = premarket_vol / (avg_vol / 6.5)  # Approx PM hours
        #     if rvol > 5:
        #         return min(20, score + 4)

    def _score_float(self, float_shares: int) -> int:
        """Score float size (0-15 points)"""
        if float_shares <= 0:
            return 5  # Unknown float

        if float_shares < 1_000_000:
            return 15  # Nano float - highest potential
        elif float_shares < self.MICRO_FLOAT_THRESHOLD:
            return 13  # Micro float
        elif float_shares < self.LOW_FLOAT_THRESHOLD:
            return 10  # Low float
        elif float_shares < 50_000_000:
            return 6  # Medium float
        else:
            return 2  # Large float

    def _score_catalyst(self, catalyst_type: CatalystType, headline: str) -> int:
        """Score catalyst quality (0-25 points)"""
        if catalyst_type == CatalystType.NONE:
            return 0  # No catalyst = unreliable

        if catalyst_type == CatalystType.FDA:
            return 25  # FDA = highest conviction
        elif catalyst_type == CatalystType.EARNINGS:
            return 22
        elif catalyst_type == CatalystType.MERGER:
            return 22
        elif catalyst_type == CatalystType.CONTRACT:
            return 20
        elif catalyst_type == CatalystType.CLINICAL:
            return 18
        elif catalyst_type == CatalystType.PARTNERSHIP:
            return 15
        elif catalyst_type == CatalystType.UPGRADE:
            return 15
        elif catalyst_type == CatalystType.OFFERING:
            return 5  # Offerings are usually bearish
        elif catalyst_type == CatalystType.DOWNGRADE:
            return 8
        else:
            return 10  # Other catalyst

    def _score_prior_day(self, prior_change: float, gap_pct: float) -> int:
        """Score prior day action (0-10 points)"""
        # Continuation patterns are stronger
        is_continuation = (prior_change > 0 and gap_pct > 0) or (prior_change < 0 and gap_pct < 0)

        if is_continuation:
            if abs(prior_change) > 10:
                return 10  # Strong continuation
            elif abs(prior_change) > 5:
                return 8
            else:
                return 6
        else:
            # Reversal gap
            if abs(prior_change) > 10:
                return 4  # Sharp reversal - risky
            else:
                return 5  # Mild reversal

    def _score_price(self, price: float) -> int:
        """Score price range (0-10 points)"""
        if price < 1:
            return 2  # Sub-penny risk
        elif price < 2:
            return 5  # Low liquidity risk
        elif price <= 20:
            return 10  # Optimal scalping range
        elif price <= 50:
            return 7  # Okay but wider stops needed
        elif price <= 100:
            return 4  # Need more capital
        else:
            return 2  # Too expensive for scalping

    def _get_grade(self, score: int) -> GapGrade:
        """Convert score to grade"""
        for grade, threshold in self.GRADE_THRESHOLDS.items():
            if score >= threshold:
                return grade
        return GapGrade.F

    def _get_gap_type(self, gap_pct: float, prior_change: float) -> GapType:
        """Determine gap type"""
        if gap_pct > 0:
            if prior_change > 0:
                return GapType.CONTINUATION_UP
            elif prior_change < -3:
                return GapType.REVERSAL_UP
            else:
                return GapType.GAP_UP
        else:
            if prior_change < 0:
                return GapType.CONTINUATION_DOWN
            elif prior_change > 3:
                return GapType.REVERSAL_DOWN
            else:
                return GapType.GAP_DOWN

    def _calculate_levels(self, graded: GradedGap):
        """Calculate entry, stop, and target levels"""
        price = graded.current_price
        gap_pct = graded.gap_percent
        prior_close = graded.prior_close

        if gap_pct > 0:  # Gap up
            # Entry zone: Current price to 1% above
            graded.entry_zone_low = price * 0.99
            graded.entry_zone_high = price * 1.01

            # Stop: Below pre-market low or prior close
            pm_low = graded.premarket_low or price * 0.95
            graded.stop_loss = max(pm_low * 0.99, prior_close * 0.98)

            # Targets based on gap size
            graded.target_1 = price * (1 + gap_pct / 200)  # Half the gap as first target
            graded.target_2 = price * (1 + gap_pct / 100)  # Full gap extension

        else:  # Gap down (for shorts, but we don't short)
            graded.entry_zone_low = price * 0.99
            graded.entry_zone_high = price * 1.01
            graded.stop_loss = price * 1.03
            graded.target_1 = price * 0.97
            graded.target_2 = price * 0.95

    def _calculate_position_size(self, grade: GapGrade) -> float:
        """Calculate recommended position size as % of normal"""
        if grade == GapGrade.A:
            return 100.0
        elif grade == GapGrade.B:
            return 75.0
        elif grade == GapGrade.C:
            return 50.0
        elif grade == GapGrade.D:
            return 25.0
        else:
            return 0.0  # Don't trade F grades

    def _get_warnings(self, graded: GradedGap) -> List[str]:
        """Generate warnings for the gap"""
        warnings = []

        if graded.premarket_volume < self.MIN_PREMARKET_VOLUME:
            warnings.append("Low pre-market volume")

        if abs(graded.gap_percent) > 25:
            warnings.append("Extreme gap - high risk")

        if graded.float_shares > 0 and graded.float_shares > 100_000_000:
            warnings.append("Large float - harder to move")

        if graded.catalyst_type == CatalystType.OFFERING:
            warnings.append("Offering announced - typically bearish")

        if graded.catalyst_type == CatalystType.NONE:
            warnings.append("No clear catalyst - gap may fade")

        if graded.gap_type in [GapType.REVERSAL_UP, GapType.REVERSAL_DOWN]:
            warnings.append("Reversal gap - higher failure rate")

        if graded.current_price < 1:
            warnings.append("Sub-$1 stock - high volatility risk")

        return warnings

    def grade_gap(
        self,
        symbol: str,
        gap_percent: float,
        current_price: float,
        prior_close: float,
        premarket_volume: int = 0,
        float_shares: int = 0,
        avg_volume: int = 0,
        prior_day_change: float = 0,
        catalyst_headline: str = "",
        premarket_high: float = 0,
        premarket_low: float = 0
    ) -> GradedGap:
        """
        Grade a gap with full analysis.

        Args:
            symbol: Stock symbol
            gap_percent: Gap size in percent
            current_price: Current pre-market price
            prior_close: Prior day close
            premarket_volume: Pre-market volume
            float_shares: Float in shares
            avg_volume: Average daily volume
            prior_day_change: Prior day % change
            catalyst_headline: News headline if any
            premarket_high: Pre-market high
            premarket_low: Pre-market low

        Returns:
            GradedGap with full analysis
        """
        symbol = symbol.upper()

        # Detect catalyst
        catalyst_type = self._detect_catalyst(catalyst_headline)

        # Calculate scores
        score = GapScore(
            gap_size_score=self._score_gap_size(gap_percent),
            volume_score=self._score_volume(premarket_volume, avg_volume),
            float_score=self._score_float(float_shares),
            catalyst_score=self._score_catalyst(catalyst_type, catalyst_headline),
            prior_day_score=self._score_prior_day(prior_day_change, gap_percent),
            price_score=self._score_price(current_price)
        )

        # Get grade
        grade = self._get_grade(score.total)

        # Get gap type
        gap_type = self._get_gap_type(gap_percent, prior_day_change)

        # Calculate relative volume
        rvol = 0.0
        if avg_volume > 0:
            # Pre-market is ~1/10th of regular session typically
            rvol = premarket_volume / (avg_volume / 10)

        # Determine market session
        now = datetime.now()
        if now.hour < 4:
            session = "overnight"
        elif now.hour < 9 or (now.hour == 9 and now.minute < 30):
            session = "premarket"
        elif now.hour < 16:
            session = "regular"
        else:
            session = "afterhours"

        # Create graded gap
        graded = GradedGap(
            symbol=symbol,
            grade=grade,
            score=score,
            gap_type=gap_type,
            catalyst_type=catalyst_type,
            gap_percent=gap_percent,
            current_price=current_price,
            prior_close=prior_close,
            premarket_high=premarket_high or current_price,
            premarket_low=premarket_low or current_price,
            premarket_volume=premarket_volume,
            float_shares=float_shares,
            avg_volume=avg_volume,
            relative_volume=rvol,
            prior_day_change=prior_day_change,
            catalyst_headline=catalyst_headline,
            position_size_pct=self._calculate_position_size(grade),
            scanned_at=datetime.now().isoformat(),
            market_session=session
        )

        # Calculate trading levels
        self._calculate_levels(graded)

        # Get warnings
        graded.warnings = self._get_warnings(graded)

        # Store
        self.graded_gaps[symbol] = graded

        # Log
        logger.info(
            f"GAP GRADED: {symbol} = {grade.value} ({score.total}/100) | "
            f"Gap: {gap_percent:+.1f}% | Vol: {premarket_volume:,} | "
            f"Type: {gap_type.value}"
        )

        return graded

    def get_grade(self, symbol: str) -> Optional[GradedGap]:
        """Get graded gap for symbol"""
        return self.graded_gaps.get(symbol.upper())

    def get_top_gaps(self, min_grade: GapGrade = GapGrade.C, limit: int = 10) -> List[GradedGap]:
        """Get top graded gaps"""
        grade_order = {GapGrade.A: 0, GapGrade.B: 1, GapGrade.C: 2, GapGrade.D: 3, GapGrade.F: 4}
        min_order = grade_order[min_grade]

        filtered = [
            g for g in self.graded_gaps.values()
            if grade_order[g.grade] <= min_order
        ]

        # Sort by score descending
        filtered.sort(key=lambda x: x.score.total, reverse=True)

        return filtered[:limit]

    def get_a_grades(self) -> List[GradedGap]:
        """Get only A-grade gaps"""
        return [g for g in self.graded_gaps.values() if g.grade == GapGrade.A]

    def get_tradeable_gaps(self) -> List[GradedGap]:
        """Get gaps that are tradeable (C grade or better)"""
        return self.get_top_gaps(min_grade=GapGrade.C, limit=20)

    def clear_gaps(self):
        """Clear all graded gaps (call at market close)"""
        self.graded_gaps.clear()
        logger.info("Gap grader cleared for new session")

    def get_status(self) -> Dict:
        """Get grader status"""
        grade_counts = {g.value: 0 for g in GapGrade}
        for gap in self.graded_gaps.values():
            grade_counts[gap.grade.value] += 1

        return {
            "total_gaps": len(self.graded_gaps),
            "grade_distribution": grade_counts,
            "a_grades": len(self.get_a_grades()),
            "tradeable": len(self.get_tradeable_gaps()),
            "top_gaps": [
                {"symbol": g.symbol, "grade": g.grade.value, "score": g.score.total}
                for g in self.get_top_gaps(limit=5)
            ]
        }


# Singleton instance
_gap_grader: Optional[GapGrader] = None


def get_gap_grader() -> GapGrader:
    """Get singleton gap grader"""
    global _gap_grader
    if _gap_grader is None:
        _gap_grader = GapGrader()
    return _gap_grader


# Convenience functions
def grade_gap(symbol: str, gap_pct: float, price: float, prior_close: float,
              **kwargs) -> GradedGap:
    """Quick gap grading"""
    return get_gap_grader().grade_gap(symbol, gap_pct, price, prior_close, **kwargs)


def get_top_gaps(min_grade: str = "C", limit: int = 10) -> List[GradedGap]:
    """Get top graded gaps"""
    grade = GapGrade[min_grade] if isinstance(min_grade, str) else min_grade
    return get_gap_grader().get_top_gaps(grade, limit)
