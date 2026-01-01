"""
Setup Classifier - Ross Cameron Trade Setup Classification
===========================================================
Combines pattern detection, tape reading, and stock grading
to classify the current trading opportunity.

Based on: "I always want to see my entry criteria align before
I put my money at risk"
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SetupType(Enum):
    """Trading setup types from Warrior Trading"""

    BULL_FLAG = "Bull Flag Breakout"
    ABCD = "ABCD Pattern"
    MICRO_PULLBACK = "Micro Pullback"
    HOD_BREAK = "High of Day Break"
    VWAP_BREAKOUT = "VWAP Breakout"
    DIP_BUY = "Dip Buy / Flush"
    HALT_RESUME = "Halt Resumption"
    NO_SETUP = "No Valid Setup"


class SetupGrade(Enum):
    """Position sizing grades"""

    A_PLUS = "A+"  # Perfect setup - full position + add
    A = "A"  # 5/5 criteria - full position
    B = "B"  # 4/5 criteria - half position
    C = "C"  # 3/5 or less - scalp only
    F = "F"  # No trade


@dataclass
class StockCriteria:
    """Ross Cameron's 5 criteria for stock selection"""

    symbol: str

    # The 5 criteria
    has_news: bool = False
    float_under_10m: bool = False
    price_in_range: bool = False  # $1-$20
    change_over_10pct: bool = False
    rvol_over_5x: bool = False

    # Raw values
    float_shares: float = 0
    price: float = 0
    change_pct: float = 0
    relative_volume: float = 0
    news_headline: str = ""

    @property
    def criteria_met(self) -> int:
        return sum(
            [
                self.has_news,
                self.float_under_10m,
                self.price_in_range,
                self.change_over_10pct,
                self.rvol_over_5x,
            ]
        )

    @property
    def grade(self) -> SetupGrade:
        met = self.criteria_met
        if met >= 5:
            return SetupGrade.A
        elif met == 4:
            return SetupGrade.B
        else:
            return SetupGrade.C

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "criteria_met": self.criteria_met,
            "grade": self.grade.value,
            "details": {
                "has_news": self.has_news,
                "float_under_10m": self.float_under_10m,
                "price_in_range": self.price_in_range,
                "change_over_10pct": self.change_over_10pct,
                "rvol_over_5x": self.rvol_over_5x,
            },
            "raw_values": {
                "float": self.float_shares,
                "price": self.price,
                "change_pct": self.change_pct,
                "rvol": self.relative_volume,
                "news": self.news_headline,
            },
        }


@dataclass
class SetupClassification:
    """Complete setup classification result"""

    symbol: str
    timestamp: datetime

    # Classification
    setup_type: SetupType = SetupType.NO_SETUP
    setup_grade: SetupGrade = SetupGrade.F
    confidence: float = 0.0

    # Trade parameters
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    position_size_pct: float = 0.0  # 0-100%

    # Component analysis
    stock_criteria: Optional[StockCriteria] = None
    pattern_detected: str = ""
    tape_signal: str = ""

    # Action recommendation
    action: str = "WAIT"  # BUY_NOW, PREPARE, WAIT, AVOID

    # Reasons
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "setup_type": self.setup_type.value,
            "setup_grade": self.setup_grade.value,
            "confidence": round(self.confidence, 3),
            "trade": {
                "entry": self.entry_price,
                "stop": self.stop_loss,
                "target": self.target_price,
                "risk_reward": (
                    round(
                        (self.target_price - self.entry_price)
                        / (self.entry_price - self.stop_loss),
                        2,
                    )
                    if self.stop_loss and self.entry_price
                    else 0
                ),
                "position_size_pct": self.position_size_pct,
            },
            "analysis": {
                "pattern": self.pattern_detected,
                "tape_signal": self.tape_signal,
                "stock_criteria": (
                    self.stock_criteria.to_dict() if self.stock_criteria else None
                ),
            },
            "action": self.action,
            "reasons": self.reasons,
        }


class SetupClassifier:
    """
    Classifies trading setups by combining multiple signals.

    Integration Points:
    - PatternDetector: Bull flag, ABCD, micro pullback, HOD break
    - TapeAnalyzer: First green, seller thinning, flush detection
    - Stock criteria: Float, news, relative volume, price, change%

    Output:
    - Setup type (which pattern)
    - Setup grade (A/B/C for position sizing)
    - Entry/stop/target levels
    - Action recommendation
    """

    def __init__(self):
        self.min_confidence = 0.5
        self.min_risk_reward = 1.5

        # Position sizing by grade
        self.position_sizes = {
            SetupGrade.A_PLUS: 100,
            SetupGrade.A: 75,
            SetupGrade.B: 50,
            SetupGrade.C: 25,
            SetupGrade.F: 0,
        }

        logger.info("SetupClassifier initialized")

    def classify(
        self,
        symbol: str,
        stock_criteria: StockCriteria,
        pattern_results: Dict = None,
        tape_analysis: Dict = None,
        current_price: float = 0,
    ) -> SetupClassification:
        """
        Classify the current trading setup.

        Args:
            symbol: Stock symbol
            stock_criteria: Ross Cameron's 5 criteria
            pattern_results: Output from PatternDetector.detect_all_patterns()
            tape_analysis: Output from TapeAnalyzer.get_entry_signal()
            current_price: Current stock price

        Returns:
            SetupClassification with complete analysis
        """
        result = SetupClassification(
            symbol=symbol, timestamp=datetime.now(), stock_criteria=stock_criteria
        )

        reasons = []

        # Check stock grade first
        stock_grade = stock_criteria.grade
        if stock_grade == SetupGrade.C:
            reasons.append(f"Stock grade C ({stock_criteria.criteria_met}/5 criteria)")

        # Analyze patterns
        best_pattern = None
        pattern_confidence = 0

        if pattern_results:
            detected_patterns = [
                (name, p) for name, p in pattern_results.items() if p.detected
            ]

            if detected_patterns:
                best_name, best_pattern = max(
                    detected_patterns, key=lambda x: x[1].confidence
                )
                pattern_confidence = best_pattern.confidence
                result.pattern_detected = best_pattern.pattern_type

                # Map pattern to setup type
                pattern_map = {
                    "BULL_FLAG": SetupType.BULL_FLAG,
                    "ABCD": SetupType.ABCD,
                    "MICRO_PULLBACK": SetupType.MICRO_PULLBACK,
                    "HOD_BREAK": SetupType.HOD_BREAK,
                }
                result.setup_type = pattern_map.get(
                    best_pattern.pattern_type, SetupType.NO_SETUP
                )

                # Set trade levels from pattern
                result.entry_price = best_pattern.entry_price
                result.stop_loss = best_pattern.stop_loss
                result.target_price = best_pattern.target_price

                reasons.append(
                    f"Pattern: {best_pattern.pattern_type} ({pattern_confidence:.0%})"
                )

        # Analyze tape
        tape_confidence = 0
        if tape_analysis:
            tape_signal = tape_analysis.get("signal", "NO_SIGNAL")
            result.tape_signal = tape_signal

            if tape_signal != "NO_SIGNAL":
                tape_confidence = tape_analysis.get("confidence", 0)

                # Check for dip buy setup from tape
                if tape_signal == "IRRATIONAL_FLUSH":
                    result.setup_type = SetupType.DIP_BUY
                    result.entry_price = current_price
                    result.stop_loss = current_price * 0.97  # 3% stop
                    result.target_price = current_price * 1.05  # 5% target
                    reasons.append(f"Tape: Irrational flush detected")

                elif tape_signal in ["FIRST_GREEN_PRINT", "SELLER_THINNING"]:
                    reasons.append(f"Tape: {tape_signal} ({tape_confidence:.0%})")

        # Calculate combined confidence
        combined_confidence = 0.0
        if pattern_confidence > 0 and tape_confidence > 0:
            # Both signals align - high confidence
            combined_confidence = (pattern_confidence + tape_confidence) / 2 + 0.1
        elif pattern_confidence > 0:
            combined_confidence = pattern_confidence * 0.8
        elif tape_confidence > 0:
            combined_confidence = tape_confidence * 0.7
        else:
            combined_confidence = 0.0

        # Adjust for stock grade
        grade_multiplier = {
            SetupGrade.A: 1.0,
            SetupGrade.B: 0.85,
            SetupGrade.C: 0.7,
            SetupGrade.F: 0.5,
        }
        combined_confidence *= grade_multiplier.get(stock_grade, 0.5)
        result.confidence = min(combined_confidence, 1.0)

        # Determine final grade
        if combined_confidence >= 0.8 and stock_grade == SetupGrade.A:
            result.setup_grade = SetupGrade.A_PLUS
            reasons.append("A+ setup: Perfect alignment")
        elif combined_confidence >= 0.6:
            result.setup_grade = stock_grade
        else:
            result.setup_grade = SetupGrade.C

        # Calculate risk/reward
        if result.entry_price and result.stop_loss and result.target_price:
            risk = result.entry_price - result.stop_loss
            reward = result.target_price - result.entry_price
            if risk > 0:
                rr_ratio = reward / risk
                if rr_ratio < self.min_risk_reward:
                    result.setup_grade = SetupGrade.C
                    reasons.append(f"Risk/Reward {rr_ratio:.1f}:1 below minimum")

        # Set position size
        result.position_size_pct = self.position_sizes.get(result.setup_grade, 0)

        # Determine action
        if result.setup_grade in [SetupGrade.A_PLUS, SetupGrade.A]:
            if result.confidence >= 0.7:
                result.action = "BUY_NOW"
            else:
                result.action = "PREPARE"
        elif result.setup_grade == SetupGrade.B:
            result.action = "PREPARE"
        elif result.setup_grade == SetupGrade.C:
            result.action = "SCALP_ONLY"
        else:
            result.action = "WAIT"
            result.setup_type = SetupType.NO_SETUP

        result.reasons = reasons
        return result

    def quick_classify(
        self,
        symbol: str,
        price: float,
        change_pct: float,
        volume: int,
        avg_volume: int,
        float_shares: float = 0,
        has_news: bool = False,
    ) -> SetupClassification:
        """
        Quick classification with minimal data.
        Useful for scanner integration.
        """
        # Build criteria
        criteria = StockCriteria(
            symbol=symbol,
            has_news=has_news,
            float_under_10m=0 < float_shares < 10_000_000,
            price_in_range=1.0 <= price <= 20.0,
            change_over_10pct=change_pct >= 10.0,
            rvol_over_5x=(volume / avg_volume >= 5.0) if avg_volume > 0 else False,
            float_shares=float_shares,
            price=price,
            change_pct=change_pct,
            relative_volume=volume / avg_volume if avg_volume > 0 else 0,
        )

        return self.classify(
            symbol=symbol, stock_criteria=criteria, current_price=price
        )

    def get_entry_rules(self, setup_type: SetupType) -> Dict:
        """
        Get entry rules for a specific setup type.
        Based on Ross Cameron's documented rules.
        """
        rules = {
            SetupType.BULL_FLAG: {
                "entry": "Break above flag high",
                "confirmation": "Volume surge on breakout",
                "stop": "Low of flag",
                "target": "Pole height added to breakout",
                "timeout": "5 minutes (breakout or bailout)",
            },
            SetupType.ABCD: {
                "entry": "Break above B after C formed",
                "confirmation": "C holds above A, volume on D break",
                "stop": "Below C",
                "target": "D level (AB=CD)",
                "timeout": "Pattern invalidates if C breaks A",
            },
            SetupType.MICRO_PULLBACK: {
                "entry": "First green candle after mini dip",
                "confirmation": "MACD still bullish, volume returning",
                "stop": "Low of pullback",
                "target": "Previous high + extension",
                "timeout": "2-3 candles max for pullback",
            },
            SetupType.HOD_BREAK: {
                "entry": "Break above HOD",
                "confirmation": "Volume surge, tight pre-break consolidation",
                "stop": "Low of consolidation",
                "target": "HOD + average range",
                "timeout": "5 minutes for follow-through",
            },
            SetupType.VWAP_BREAKOUT: {
                "entry": "First break above VWAP",
                "confirmation": "Hold above VWAP on retest",
                "stop": "Below VWAP",
                "target": "Pre-market high or HOD",
                "timeout": "Invalidates on close below VWAP",
            },
            SetupType.DIP_BUY: {
                "entry": "First green print after flush",
                "confirmation": "Seller thinning, tape turning green",
                "stop": "Low of flush",
                "target": "Back to pre-flush levels",
                "timeout": "Exit if no bounce in 30 seconds",
            },
            SetupType.HALT_RESUME: {
                "entry": "Resumption price confirmation",
                "confirmation": "Direction matches expected (up halt = up resume)",
                "stop": "3% from resume price",
                "target": "Next halt level or previous high",
                "timeout": "Exit quickly if opens wrong direction",
            },
        }
        return rules.get(setup_type, {})


# Singleton instance
_setup_classifier: Optional[SetupClassifier] = None


def get_setup_classifier() -> SetupClassifier:
    """Get or create SetupClassifier instance"""
    global _setup_classifier
    if _setup_classifier is None:
        _setup_classifier = SetupClassifier()
    return _setup_classifier


# Convenience functions
async def classify_setup(
    symbol: str,
    price: float,
    change_pct: float,
    volume: int,
    avg_volume: int,
    float_shares: float = 0,
    has_news: bool = False,
) -> SetupClassification:
    """Quick setup classification"""
    classifier = get_setup_classifier()
    return classifier.quick_classify(
        symbol, price, change_pct, volume, avg_volume, float_shares, has_news
    )


async def get_full_classification(
    symbol: str,
    stock_criteria: StockCriteria,
    pattern_results: Dict = None,
    tape_analysis: Dict = None,
    current_price: float = 0,
) -> SetupClassification:
    """Full classification with all signals"""
    classifier = get_setup_classifier()
    return classifier.classify(
        symbol, stock_criteria, pattern_results, tape_analysis, current_price
    )
