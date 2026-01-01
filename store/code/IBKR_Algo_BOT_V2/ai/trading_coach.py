"""
Trading Coach - Pre-Market Analysis & Emotional Trading Detection
=================================================================
Runs during Warrior Trading hours (4AM-9:30AM EST) to:
1. Scan for momentum setups
2. Generate morning briefings
3. Analyze and critique manual trades
4. Detect emotional trading patterns
5. Provide coaching feedback to overcome emotional trading

Author: Claude Code
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytz

logger = logging.getLogger(__name__)

# Global instance
_coach_instance = None

# Trading hours (Eastern Time)
ET = pytz.timezone("US/Eastern")
PREMARKET_START = time(4, 0)  # 4:00 AM ET
PRIME_TIME_START = time(7, 0)  # 7:00 AM ET - Most active
MARKET_OPEN = time(9, 30)  # 9:30 AM ET
MARKET_CLOSE = time(16, 0)  # 4:00 PM ET


@dataclass
class TradeCritique:
    """Critique of a manual trade"""

    trade_id: str
    symbol: str
    timestamp: str

    # What happened
    action: str  # BUY/SELL
    entry_price: float
    exit_price: float
    pnl: float

    # Analysis
    setup_quality: str  # A+, A, B, C, F
    entry_timing: str  # EARLY, GOOD, LATE, CHASED
    exit_timing: str  # EARLY, GOOD, LATE, PANIC
    position_size_assessment: str  # TOO_SMALL, GOOD, TOO_LARGE

    # Emotional indicators
    emotional_flags: List[str]  # FOMO, REVENGE, FEAR, GREED, IMPATIENCE
    emotional_score: float  # 0-100 (0=rational, 100=emotional)

    # What should have happened
    optimal_entry: float
    optimal_exit: float
    optimal_pnl: float
    missed_profit: float

    # Coaching feedback
    mistakes: List[str]
    lessons: List[str]
    recommendations: List[str]
    grade: str  # A, B, C, D, F


@dataclass
class MorningBriefing:
    """Pre-market morning briefing"""

    date: str
    generated_at: str

    # Market overview
    market_sentiment: str  # BULLISH, BEARISH, NEUTRAL
    spy_premarket: float
    vix_level: float

    # Top setups found
    top_gappers: List[Dict]
    a_grade_setups: List[Dict]
    watchlist_additions: List[str]

    # What to watch
    key_levels: Dict[str, Dict]  # symbol -> {support, resistance, vwap}
    catalysts: List[Dict]

    # Risk warnings
    warnings: List[str]

    # Trading plan
    max_trades_today: int
    max_risk_today: float
    focus_strategy: str


class TradingCoach:
    """
    AI Trading Coach - Analyzes trades and provides feedback
    to help overcome emotional trading
    """

    def __init__(self):
        self.critiques: List[TradeCritique] = []
        self.briefings: List[MorningBriefing] = []
        self.emotional_patterns: Dict[str, int] = {
            "FOMO": 0,
            "REVENGE": 0,
            "FEAR": 0,
            "GREED": 0,
            "IMPATIENCE": 0,
            "OVERTRADING": 0,
        }
        self.state_file = (
            Path(__file__).parent.parent / "store" / "trading_coach_state.json"
        )
        self._load_state()
        logger.info("[COACH] Trading Coach initialized")

    def _load_state(self):
        """Load saved state"""
        try:
            if self.state_file.exists():
                with open(self.state_file) as f:
                    data = json.load(f)
                    self.emotional_patterns = data.get(
                        "emotional_patterns", self.emotional_patterns
                    )
        except Exception as e:
            logger.warning(f"[COACH] Failed to load state: {e}")

    def _save_state(self):
        """Save state"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(
                    {
                        "emotional_patterns": self.emotional_patterns,
                        "last_updated": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.warning(f"[COACH] Failed to save state: {e}")

    # =========================================================================
    # PRE-MARKET OPERATIONS
    # =========================================================================

    def is_premarket_hours(self) -> bool:
        """Check if we're in pre-market hours"""
        now = datetime.now(ET)
        current_time = now.time()
        return PREMARKET_START <= current_time < MARKET_OPEN

    def is_prime_time(self) -> bool:
        """Check if we're in prime Warrior trading time (7-9:30 AM ET)"""
        now = datetime.now(ET)
        current_time = now.time()
        return PRIME_TIME_START <= current_time < MARKET_OPEN

    def get_trading_window(self) -> Dict:
        """Get current trading window info"""
        now = datetime.now(ET)
        current_time = now.time()

        if current_time < PREMARKET_START:
            window = "PRE_PREMARKET"
            status = "Markets closed. Pre-market starts at 4:00 AM ET"
            minutes_until = (
                datetime.combine(now.date(), PREMARKET_START)
                - datetime.combine(now.date(), current_time)
            ).seconds // 60
        elif current_time < PRIME_TIME_START:
            window = "EARLY_PREMARKET"
            status = "Early pre-market. Scanning for gappers..."
            minutes_until = (
                datetime.combine(now.date(), PRIME_TIME_START)
                - datetime.combine(now.date(), current_time)
            ).seconds // 60
        elif current_time < MARKET_OPEN:
            window = "PRIME_TIME"
            status = "PRIME WARRIOR TRADING TIME! Most active period."
            minutes_until = (
                datetime.combine(now.date(), MARKET_OPEN)
                - datetime.combine(now.date(), current_time)
            ).seconds // 60
        elif current_time < MARKET_CLOSE:
            window = "MARKET_OPEN"
            status = "Regular market hours. Warrior momentum slowing."
            minutes_until = 0
        else:
            window = "AFTER_HOURS"
            status = "After hours. Review today's trades."
            minutes_until = 0

        return {
            "window": window,
            "status": status,
            "current_time_et": now.strftime("%H:%M:%S ET"),
            "is_premarket": self.is_premarket_hours(),
            "is_prime_time": self.is_prime_time(),
            "minutes_until_next": minutes_until,
        }

    def generate_morning_briefing(self) -> MorningBriefing:
        """Generate morning briefing with top setups"""
        now = datetime.now(ET)

        briefing = MorningBriefing(
            date=now.strftime("%Y-%m-%d"),
            generated_at=now.isoformat(),
            market_sentiment="NEUTRAL",
            spy_premarket=0.0,
            vix_level=0.0,
            top_gappers=[],
            a_grade_setups=[],
            watchlist_additions=[],
            key_levels={},
            catalysts=[],
            warnings=[],
            max_trades_today=5,
            max_risk_today=500.0,
            focus_strategy="momentum",
        )

        # Get scanner results
        try:
            from ai.warrior_scanner import WarriorScanner

            scanner = WarriorScanner()
            candidates = scanner.scan_premarket()

            if candidates:
                # Sort by confidence
                sorted_candidates = sorted(
                    candidates, key=lambda x: x.confidence_score, reverse=True
                )

                # Top gappers
                briefing.top_gappers = [
                    {
                        "symbol": c.symbol,
                        "gap_percent": c.gap_percent,
                        "price": c.price,
                        "volume": c.pre_market_volume,
                        "float": c.float_shares,
                        "catalyst": c.catalyst,
                        "score": c.confidence_score,
                    }
                    for c in sorted_candidates[:10]
                ]

                # A-grade setups (score >= 80)
                briefing.a_grade_setups = [
                    asdict(c) for c in sorted_candidates if c.confidence_score >= 80
                ][:5]

                # Add to watchlist
                briefing.watchlist_additions = [c.symbol for c in sorted_candidates[:5]]

        except Exception as e:
            logger.warning(f"[COACH] Scanner error: {e}")
            briefing.warnings.append(f"Scanner unavailable: {e}")

        # Add trading plan based on conditions
        if len(briefing.top_gappers) == 0:
            briefing.warnings.append(
                "No clear setups found. Consider sitting out today."
            )
            briefing.max_trades_today = 2
        elif len(briefing.a_grade_setups) >= 3:
            briefing.focus_strategy = "momentum"
            briefing.max_trades_today = 5
        else:
            briefing.focus_strategy = "selective"
            briefing.max_trades_today = 3

        # Emotional trading warnings based on history
        if self.emotional_patterns.get("OVERTRADING", 0) > 3:
            briefing.warnings.append(
                "⚠️ You've been overtrading recently. Stick to max 3 trades today."
            )
            briefing.max_trades_today = 3

        if self.emotional_patterns.get("REVENGE", 0) > 2:
            briefing.warnings.append(
                "⚠️ Revenge trading detected recently. Take a break after any loss."
            )

        if self.emotional_patterns.get("FOMO", 0) > 2:
            briefing.warnings.append(
                "⚠️ FOMO entries detected. Wait for pullbacks, don't chase."
            )

        self.briefings.append(briefing)
        logger.info(
            f"[COACH] Generated morning briefing: {len(briefing.top_gappers)} gappers found"
        )

        return briefing

    # =========================================================================
    # TRADE CRITIQUE & ANALYSIS
    # =========================================================================

    def critique_trade(self, trade_data: Dict) -> TradeCritique:
        """
        Analyze a manual trade and provide detailed critique

        trade_data should include:
        - symbol, entry_price, exit_price, quantity
        - entry_time, exit_time
        - your_reasoning (why you entered/exited)
        """
        symbol = trade_data.get("symbol", "UNKNOWN")
        entry_price = trade_data.get("entry_price", 0)
        exit_price = trade_data.get("exit_price", 0)
        quantity = trade_data.get("quantity", 0)
        entry_time = trade_data.get("entry_time", "")
        exit_time = trade_data.get("exit_time", "")
        reasoning = trade_data.get("your_reasoning", "")

        # Calculate P&L
        pnl = (exit_price - entry_price) * quantity
        pnl_percent = (
            ((exit_price - entry_price) / entry_price * 100) if entry_price else 0
        )

        # Analyze entry timing
        entry_analysis = self._analyze_entry(trade_data)

        # Analyze exit timing
        exit_analysis = self._analyze_exit(trade_data)

        # Detect emotional flags
        emotional_flags, emotional_score = self._detect_emotional_trading(trade_data)

        # Update emotional patterns
        for flag in emotional_flags:
            self.emotional_patterns[flag] = self.emotional_patterns.get(flag, 0) + 1
        self._save_state()

        # Generate coaching feedback
        mistakes, lessons, recommendations = self._generate_feedback(
            trade_data, entry_analysis, exit_analysis, emotional_flags
        )

        # Calculate grade
        grade = self._calculate_grade(
            entry_analysis, exit_analysis, emotional_score, pnl
        )

        critique = TradeCritique(
            trade_id=f"{symbol}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            action="BUY",
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            setup_quality=entry_analysis.get("setup_quality", "C"),
            entry_timing=entry_analysis.get("timing", "UNKNOWN"),
            exit_timing=exit_analysis.get("timing", "UNKNOWN"),
            position_size_assessment=entry_analysis.get("size_assessment", "GOOD"),
            emotional_flags=emotional_flags,
            emotional_score=emotional_score,
            optimal_entry=entry_analysis.get("optimal_entry", entry_price),
            optimal_exit=exit_analysis.get("optimal_exit", exit_price),
            optimal_pnl=entry_analysis.get("optimal_pnl", pnl),
            missed_profit=max(0, entry_analysis.get("optimal_pnl", pnl) - pnl),
            mistakes=mistakes,
            lessons=lessons,
            recommendations=recommendations,
            grade=grade,
        )

        self.critiques.append(critique)
        logger.info(f"[COACH] Trade critique generated: {symbol} - Grade: {grade}")

        return critique

    def _analyze_entry(self, trade_data: Dict) -> Dict:
        """Analyze entry quality"""
        entry_price = trade_data.get("entry_price", 0)
        reasoning = trade_data.get("your_reasoning", "").lower()

        analysis = {
            "setup_quality": "B",
            "timing": "GOOD",
            "size_assessment": "GOOD",
            "optimal_entry": entry_price,
            "optimal_pnl": 0,
        }

        # Check for chase indicators
        chase_words = [
            "chased",
            "fomo",
            "had to get in",
            "running",
            "missed",
            "jumping",
        ]
        if any(word in reasoning for word in chase_words):
            analysis["timing"] = "CHASED"
            analysis["setup_quality"] = "C"

        # Check for proper setup
        setup_words = ["pullback", "flag", "breakout", "support", "vwap", "pattern"]
        if any(word in reasoning for word in setup_words):
            analysis["setup_quality"] = "A"
            analysis["timing"] = "GOOD"

        return analysis

    def _analyze_exit(self, trade_data: Dict) -> Dict:
        """Analyze exit quality"""
        exit_price = trade_data.get("exit_price", 0)
        pnl = trade_data.get("pnl", 0)
        reasoning = trade_data.get("exit_reasoning", "").lower()

        analysis = {"timing": "GOOD", "optimal_exit": exit_price}

        # Check for panic exit
        panic_words = [
            "scared",
            "panic",
            "couldn't take it",
            "had to get out",
            "nervous",
        ]
        if any(word in reasoning for word in panic_words):
            analysis["timing"] = "PANIC"

        # Check for greed (holding too long)
        greed_words = ["wanted more", "thought it would keep going", "greedy"]
        if any(word in reasoning for word in greed_words):
            analysis["timing"] = "LATE"

        # Check for proper exit
        proper_words = ["target", "stop", "plan", "resistance", "momentum fading"]
        if any(word in reasoning for word in proper_words):
            analysis["timing"] = "GOOD"

        return analysis

    def _detect_emotional_trading(self, trade_data: Dict) -> tuple:
        """Detect emotional trading patterns"""
        flags = []
        score = 0  # 0 = rational, 100 = emotional

        reasoning = trade_data.get("your_reasoning", "").lower()
        exit_reasoning = trade_data.get("exit_reasoning", "").lower()

        # FOMO detection
        fomo_words = [
            "chased",
            "fomo",
            "had to",
            "couldn't miss",
            "everyone",
            "running away",
        ]
        if any(word in reasoning for word in fomo_words):
            flags.append("FOMO")
            score += 25

        # REVENGE trading detection
        revenge_words = [
            "make back",
            "revenge",
            "recover",
            "lost earlier",
            "get it back",
        ]
        if any(word in reasoning for word in revenge_words):
            flags.append("REVENGE")
            score += 30

        # FEAR detection
        fear_words = ["scared", "nervous", "panic", "couldn't hold", "too risky"]
        if any(word in exit_reasoning for word in fear_words):
            flags.append("FEAR")
            score += 20

        # GREED detection
        greed_words = ["more", "greedy", "bigger", "keep going", "moon"]
        if any(word in reasoning + exit_reasoning for word in greed_words):
            flags.append("GREED")
            score += 20

        # IMPATIENCE detection
        impatience_words = [
            "quick",
            "fast",
            "couldn't wait",
            "bored",
            "tired of waiting",
        ]
        if any(word in reasoning for word in impatience_words):
            flags.append("IMPATIENCE")
            score += 15

        return flags, min(100, score)

    def _generate_feedback(
        self,
        trade_data: Dict,
        entry_analysis: Dict,
        exit_analysis: Dict,
        emotional_flags: List[str],
    ) -> tuple:
        """Generate coaching feedback"""
        mistakes = []
        lessons = []
        recommendations = []

        # Entry feedback
        if entry_analysis.get("timing") == "CHASED":
            mistakes.append("Chased entry - entered after significant move")
            lessons.append("Wait for pullbacks to moving averages or support levels")
            recommendations.append("Set alerts at key levels instead of market-buying")

        # Exit feedback
        if exit_analysis.get("timing") == "PANIC":
            mistakes.append("Panic exit - sold based on fear, not plan")
            lessons.append(
                "Trust your stop loss. If it's not hit, the trade is still valid."
            )
            recommendations.append(
                "Write down your stop before entering. Don't touch it."
            )

        if exit_analysis.get("timing") == "LATE":
            mistakes.append("Held too long - greed overrode the plan")
            lessons.append(
                "Take profits at planned targets. First target = lock in gains."
            )
            recommendations.append("Use a trailing stop after hitting first target")

        # Emotional feedback
        if "FOMO" in emotional_flags:
            mistakes.append("FOMO entry - fear of missing out drove the decision")
            lessons.append(
                "There's always another trade. Missing one is better than forcing one."
            )
            recommendations.append(
                "If you feel FOMO, wait 5 minutes. The urge usually passes."
            )

        if "REVENGE" in emotional_flags:
            mistakes.append("Revenge trade - trying to recover losses")
            lessons.append(
                "Each trade is independent. Past losses don't affect future probability."
            )
            recommendations.append(
                "After a loss, take a 15-minute break. Reset mentally."
            )

        if "FEAR" in emotional_flags:
            mistakes.append("Fear-based exit - let emotions override the plan")
            lessons.append(
                "Your stop loss is your risk. If it's acceptable, hold the trade."
            )
            recommendations.append(
                "Reduce position size until comfortable holding to stop"
            )

        return mistakes, lessons, recommendations

    def _calculate_grade(
        self,
        entry_analysis: Dict,
        exit_analysis: Dict,
        emotional_score: float,
        pnl: float,
    ) -> str:
        """Calculate overall trade grade"""
        score = 100

        # Deduct for entry issues
        if entry_analysis.get("timing") == "CHASED":
            score -= 20
        if entry_analysis.get("setup_quality") == "C":
            score -= 15
        if entry_analysis.get("setup_quality") == "F":
            score -= 30

        # Deduct for exit issues
        if exit_analysis.get("timing") == "PANIC":
            score -= 25
        if exit_analysis.get("timing") == "LATE":
            score -= 15

        # Deduct for emotional trading
        score -= emotional_score * 0.3

        # Bonus for profitable trade
        if pnl > 0:
            score += 10

        # Convert to letter grade
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    # =========================================================================
    # COACHING & IMPROVEMENT
    # =========================================================================

    def get_coaching_summary(self) -> Dict:
        """Get summary of emotional patterns and areas to improve"""
        total_flags = sum(self.emotional_patterns.values())

        # Find top issues
        sorted_patterns = sorted(
            self.emotional_patterns.items(), key=lambda x: x[1], reverse=True
        )

        top_issues = [p for p, count in sorted_patterns if count > 0][:3]

        # Generate personalized advice
        advice = []

        if "FOMO" in top_issues:
            advice.append(
                {
                    "issue": "FOMO (Fear of Missing Out)",
                    "frequency": self.emotional_patterns["FOMO"],
                    "impact": "Leads to chasing entries at poor prices",
                    "fix": "Wait for pullbacks. Set price alerts instead of watching constantly.",
                    "rule": "If a stock has moved more than 5% from your ideal entry, SKIP IT.",
                }
            )

        if "REVENGE" in top_issues:
            advice.append(
                {
                    "issue": "Revenge Trading",
                    "frequency": self.emotional_patterns["REVENGE"],
                    "impact": "Compounds losses by forcing bad trades",
                    "fix": "After any loss, take a mandatory 15-minute break.",
                    "rule": "Max 2 consecutive losses, then done for the day.",
                }
            )

        if "FEAR" in top_issues:
            advice.append(
                {
                    "issue": "Fear-Based Exits",
                    "frequency": self.emotional_patterns["FEAR"],
                    "impact": "Cuts winners short, realizes losses too early",
                    "fix": "Reduce position size until you can hold comfortably.",
                    "rule": "Trust your stop loss. It's there for a reason.",
                }
            )

        if "GREED" in top_issues:
            advice.append(
                {
                    "issue": "Greed (Holding Too Long)",
                    "frequency": self.emotional_patterns["GREED"],
                    "impact": "Turns winners into losers",
                    "fix": "Take partial profits at 1R, trail the rest.",
                    "rule": "A profit taken is better than a profit given back.",
                }
            )

        if "IMPATIENCE" in top_issues:
            advice.append(
                {
                    "issue": "Impatience",
                    "frequency": self.emotional_patterns["IMPATIENCE"],
                    "impact": "Enters before setup is complete",
                    "fix": "Wait for confirmation. The best trades come to you.",
                    "rule": "No trade is urgent. If it feels urgent, it's probably wrong.",
                }
            )

        return {
            "total_emotional_flags": total_flags,
            "patterns": self.emotional_patterns,
            "top_issues": top_issues,
            "personalized_advice": advice,
            "overall_assessment": self._get_overall_assessment(total_flags),
            "rules_to_follow": self._get_rules(top_issues),
        }

    def _get_overall_assessment(self, total_flags: int) -> str:
        """Get overall assessment of emotional trading"""
        if total_flags == 0:
            return "Excellent! No emotional trading detected. Keep it up!"
        elif total_flags <= 3:
            return "Good progress. Minor emotional patterns detected. Stay disciplined."
        elif total_flags <= 7:
            return "Caution: Emotional trading is affecting your results. Focus on the rules."
        else:
            return "Warning: Significant emotional trading. Consider reducing position size and trade frequency."

    def _get_rules(self, top_issues: List[str]) -> List[str]:
        """Get personalized rules based on issues"""
        rules = [
            "Always have a plan BEFORE entering (entry, stop, target)",
            "Risk max 1% of account per trade",
            "No trading during first 5 minutes after open (too volatile)",
        ]

        if "FOMO" in top_issues:
            rules.append("If you missed the move, WAIT for the next setup")
        if "REVENGE" in top_issues:
            rules.append("After 2 losses, STOP trading for the day")
        if "FEAR" in top_issues:
            rules.append("Cut position size in half until comfortable")
        if "GREED" in top_issues:
            rules.append("Take 50% off at 1R, trail the rest")

        return rules

    def ask_question(self, question: str) -> str:
        """
        Ask the coach a question about trading
        Returns coaching advice
        """
        question_lower = question.lower()

        # Entry questions
        if any(word in question_lower for word in ["enter", "buy", "entry", "get in"]):
            return self._answer_entry_question(question)

        # Exit questions
        if any(
            word in question_lower for word in ["exit", "sell", "out", "take profit"]
        ):
            return self._answer_exit_question(question)

        # Loss questions
        if any(word in question_lower for word in ["loss", "losing", "down", "red"]):
            return self._answer_loss_question(question)

        # General advice
        return self._answer_general_question(question)

    def _answer_entry_question(self, question: str) -> str:
        """Answer questions about entries"""
        return """
ENTRY CHECKLIST:
✓ Is this an A or A+ setup? (If not, skip it)
✓ Is there a clear catalyst? (Gap, news, volume)
✓ Is relative volume > 2x? (Confirms interest)
✓ Is price above VWAP? (Confirms strength)
✓ Do you have a clear stop loss level?
✓ Is risk/reward at least 2:1?

If any answer is NO, don't enter. Wait for a better setup.

Remember: The best trades are OBVIOUS. If you're unsure, that's your answer.
"""

    def _answer_exit_question(self, question: str) -> str:
        """Answer questions about exits"""
        return """
EXIT RULES:
1. STOP LOSS: Exit immediately if hit. No exceptions.
2. FIRST TARGET (1R): Take 50% off, move stop to breakeven
3. SECOND TARGET (2R): Take another 25%, trail the rest
4. MOMENTUM FADING: If you see lower highs, tighten stop

WARNING SIGNS TO EXIT:
- Volume dropping significantly
- Price fails to make new highs
- Broader market turning red
- You're up big and getting greedy (take profits!)

Remember: You can always re-enter. Protecting capital is #1.
"""

    def _answer_loss_question(self, question: str) -> str:
        """Answer questions about handling losses"""
        return """
HANDLING LOSSES:

1. ACCEPT IT: Losses are part of trading. Even the best win only 60-70%.

2. DON'T REVENGE TRADE: Your next trade should be BETTER, not BIGGER.

3. TAKE A BREAK: After a loss, step away for 15 minutes minimum.

4. REVIEW: What went wrong? Entry? Exit? Setup quality?

5. REDUCE SIZE: If you're on a losing streak, cut size in half.

MINDSET SHIFT:
- A loss doesn't make you a bad trader
- Following your plan on a losing trade is a WIN
- The goal is good process, not perfect results

What was your stop loss? Did you follow it? If yes, you did nothing wrong.
"""

    def _answer_general_question(self, question: str) -> str:
        """Answer general trading questions"""
        return """
WARRIOR TRADING FUNDAMENTALS:

1. TRADE THE BEST: Only A and A+ setups. Skip everything else.

2. MANAGE RISK: Never risk more than 1-2% per trade.

3. BE PATIENT: The market is open every day. No need to force trades.

4. CUT LOSSES FAST: Small losses are fine. Big losses kill accounts.

5. LET WINNERS RUN: Take partials, but don't exit too early.

6. REVIEW DAILY: What worked? What didn't? Improve 1% each day.

Your #1 job is CAPITAL PRESERVATION. Profits come from surviving.

What specific aspect would you like help with?
"""


def get_trading_coach() -> TradingCoach:
    """Get or create the trading coach instance"""
    global _coach_instance
    if _coach_instance is None:
        _coach_instance = TradingCoach()
    return _coach_instance
