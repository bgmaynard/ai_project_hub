"""
Warrior Trading Strategy Optimizer

Analyzes trading performance and suggests parameter optimizations using Claude AI.

Features:
- Pattern-specific performance analysis
- Parameter optimization suggestions
- Win rate improvement recommendations
- Risk/reward optimization
- Time-of-day analysis
- Market condition correlation
"""

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from claude_integration import ClaudeRequest, get_claude_integration
from warrior_database import WarriorDatabase

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for analysis"""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_r_multiple: float
    profit_factor: float
    net_pnl: float
    best_trade: float
    worst_trade: float
    avg_hold_time_minutes: float


@dataclass
class PatternPerformance:
    """Performance metrics for a specific pattern"""

    pattern_type: str
    total_trades: int
    win_rate: float
    avg_r_multiple: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    best_trade: Dict[str, Any]
    worst_trade: Dict[str, Any]


@dataclass
class OptimizationSuggestion:
    """Suggested parameter optimization"""

    parameter: str
    current_value: Any
    suggested_value: Any
    reasoning: str
    expected_impact: str
    confidence: float
    priority: str  # HIGH, MEDIUM, LOW
    status: str = "pending"  # pending, applied, rejected


class StrategyOptimizer:
    """
    Analyzes trading performance and suggests optimizations

    Uses Claude AI to:
    - Identify winning patterns
    - Detect underperforming setups
    - Suggest parameter adjustments
    - Recommend strategy improvements
    """

    def __init__(self, db_path: str = "data/warrior_trading.db"):
        """
        Initialize strategy optimizer

        Args:
            db_path: Path to SQLite database
        """
        self.db = WarriorDatabase(db_path)
        self.claude = get_claude_integration()
        logger.info("Strategy optimizer initialized")

    def analyze_overall_performance(
        self, days: int = 30, min_trades: int = 10
    ) -> Optional[PerformanceMetrics]:
        """
        Analyze overall trading performance

        Args:
            days: Number of days to analyze
            min_trades: Minimum trades required for analysis

        Returns:
            PerformanceMetrics or None if insufficient data
        """
        start_date = date.today() - timedelta(days=days)
        trades = self.db.get_trades(status="CLOSED", start_date=start_date)

        if len(trades) < min_trades:
            logger.warning(
                f"Insufficient trades for analysis ({len(trades)} < {min_trades})"
            )
            return None

        # Calculate metrics
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in trades if t.get("pnl", 0) < 0]

        total_wins = sum(t.get("pnl", 0) for t in winning_trades)
        total_losses = abs(sum(t.get("pnl", 0) for t in losing_trades))

        avg_win = total_wins / len(winning_trades) if winning_trades else 0
        avg_loss = total_losses / len(losing_trades) if losing_trades else 0

        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0

        # Calculate average hold time
        hold_times = []
        for trade in trades:
            if trade.get("entry_time") and trade.get("exit_time"):
                entry = datetime.fromisoformat(trade["entry_time"])
                exit_dt = datetime.fromisoformat(trade["exit_time"])
                hold_times.append((exit_dt - entry).total_seconds() / 60)

        avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0

        return PerformanceMetrics(
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=(len(winning_trades) / len(trades)) * 100 if trades else 0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_r_multiple=(
                sum(t.get("r_multiple", 0) for t in trades) / len(trades)
                if trades
                else 0
            ),
            profit_factor=profit_factor,
            net_pnl=sum(t.get("pnl", 0) for t in trades),
            best_trade=max((t.get("pnl", 0) for t in trades), default=0),
            worst_trade=min((t.get("pnl", 0) for t in trades), default=0),
            avg_hold_time_minutes=avg_hold_time,
        )

    def analyze_pattern_performance(self, days: int = 30) -> List[PatternPerformance]:
        """
        Analyze performance by pattern type

        Args:
            days: Number of days to analyze

        Returns:
            List of PatternPerformance objects
        """
        start_date = date.today() - timedelta(days=days)
        trades = self.db.get_trades(status="CLOSED", start_date=start_date)

        # Group by pattern
        pattern_trades = defaultdict(list)
        for trade in trades:
            pattern_type = trade.get("setup_type", "UNKNOWN")
            pattern_trades[pattern_type].append(trade)

        # Analyze each pattern
        results = []
        for pattern_type, pattern_trade_list in pattern_trades.items():
            if len(pattern_trade_list) < 3:  # Skip patterns with too few trades
                continue

            winning = [t for t in pattern_trade_list if t.get("pnl", 0) > 0]
            losing = [t for t in pattern_trade_list if t.get("pnl", 0) < 0]

            total_wins = sum(t.get("pnl", 0) for t in winning)
            total_losses = abs(sum(t.get("pnl", 0) for t in losing))

            best_trade = max(pattern_trade_list, key=lambda t: t.get("pnl", 0))
            worst_trade = min(pattern_trade_list, key=lambda t: t.get("pnl", 0))

            results.append(
                PatternPerformance(
                    pattern_type=pattern_type,
                    total_trades=len(pattern_trade_list),
                    win_rate=(len(winning) / len(pattern_trade_list)) * 100,
                    avg_r_multiple=sum(
                        t.get("r_multiple", 0) for t in pattern_trade_list
                    )
                    / len(pattern_trade_list),
                    avg_win=total_wins / len(winning) if winning else 0,
                    avg_loss=total_losses / len(losing) if losing else 0,
                    profit_factor=(
                        (total_wins / total_losses) if total_losses > 0 else 0
                    ),
                    best_trade={
                        "symbol": best_trade.get("symbol"),
                        "pnl": best_trade.get("pnl", 0),
                        "r_multiple": best_trade.get("r_multiple", 0),
                    },
                    worst_trade={
                        "symbol": worst_trade.get("symbol"),
                        "pnl": worst_trade.get("pnl", 0),
                        "r_multiple": worst_trade.get("r_multiple", 0),
                    },
                )
            )

        # Sort by profit factor (best performing first)
        results.sort(key=lambda x: x.profit_factor, reverse=True)

        return results

    def analyze_time_of_day_performance(
        self, days: int = 30
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze performance by time of day

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary mapping time periods to performance metrics
        """
        start_date = date.today() - timedelta(days=days)
        trades = self.db.get_trades(status="CLOSED", start_date=start_date)

        # Define time periods
        time_periods = {
            "open": (9, 30, 10, 30),  # 9:30-10:30 AM
            "mid_morning": (10, 30, 12, 0),  # 10:30-12:00 PM
            "midday": (12, 0, 14, 0),  # 12:00-2:00 PM
            "afternoon": (14, 0, 16, 0),  # 2:00-4:00 PM
        }

        period_trades = defaultdict(list)

        for trade in trades:
            if not trade.get("entry_time"):
                continue

            entry_time = datetime.fromisoformat(trade["entry_time"])
            hour = entry_time.hour
            minute = entry_time.minute

            for period_name, (start_h, start_m, end_h, end_m) in time_periods.items():
                start_minutes = start_h * 60 + start_m
                end_minutes = end_h * 60 + end_m
                entry_minutes = hour * 60 + minute

                if start_minutes <= entry_minutes < end_minutes:
                    period_trades[period_name].append(trade)
                    break

        # Calculate metrics for each period
        results = {}
        for period_name, period_trade_list in period_trades.items():
            if not period_trade_list:
                continue

            winning = [t for t in period_trade_list if t.get("pnl", 0) > 0]
            total_pnl = sum(t.get("pnl", 0) for t in period_trade_list)

            results[period_name] = {
                "total_trades": len(period_trade_list),
                "win_rate": (len(winning) / len(period_trade_list)) * 100,
                "net_pnl": total_pnl,
                "avg_r_multiple": sum(t.get("r_multiple", 0) for t in period_trade_list)
                / len(period_trade_list),
            }

        return results

    def generate_optimization_suggestions(
        self, current_config: Dict[str, Any], days: int = 30
    ) -> List[OptimizationSuggestion]:
        """
        Generate parameter optimization suggestions using Claude AI

        Args:
            current_config: Current strategy configuration
            days: Number of days of data to analyze

        Returns:
            List of OptimizationSuggestion objects
        """
        # Gather performance data
        overall_metrics = self.analyze_overall_performance(days=days)
        if not overall_metrics:
            logger.warning("Insufficient data for optimization suggestions")
            return []

        pattern_metrics = self.analyze_pattern_performance(days=days)
        time_metrics = self.analyze_time_of_day_performance(days=days)

        # Build context for Claude
        context = {
            "analysis_period_days": days,
            "overall_performance": asdict(overall_metrics),
            "pattern_performance": [asdict(p) for p in pattern_metrics],
            "time_of_day_performance": time_metrics,
            "current_configuration": current_config,
        }

        # Request Claude analysis
        prompt = f"""Analyze this Warrior Trading performance data and suggest specific parameter optimizations:

OVERALL PERFORMANCE ({days} days):
- Win Rate: {overall_metrics.win_rate:.1f}%
- Total Trades: {overall_metrics.total_trades}
- Net P&L: ${overall_metrics.net_pnl:.2f}
- Avg R-Multiple: {overall_metrics.avg_r_multiple:.2f}R
- Profit Factor: {overall_metrics.profit_factor:.2f}
- Avg Win: ${overall_metrics.avg_win:.2f}
- Avg Loss: ${overall_metrics.avg_loss:.2f}

PATTERN PERFORMANCE:
{json.dumps([asdict(p) for p in pattern_metrics], indent=2)}

TIME OF DAY PERFORMANCE:
{json.dumps(time_metrics, indent=2)}

CURRENT CONFIGURATION:
{json.dumps(current_config, indent=2)}

Based on this data, suggest 3-7 specific parameter optimizations that could improve performance.
For each suggestion, provide:

{{
  "parameter": "exact parameter name from config",
  "current_value": current value,
  "suggested_value": recommended new value,
  "reasoning": "data-driven explanation (2-3 sentences)",
  "expected_impact": "specific expected outcome",
  "confidence": 0.0-1.0,
  "priority": "HIGH|MEDIUM|LOW"
}}

Focus on:
1. Patterns with high win rates (increase allocation)
2. Patterns with low win rates (tighten filters or reduce allocation)
3. Time periods with better/worse performance
4. Risk parameter adjustments based on avg win/loss
5. Position sizing optimization

Return as JSON array of suggestions. Be specific and actionable."""

        request = ClaudeRequest(
            request_type="strategy_optimization",
            prompt=prompt,
            max_tokens=3000,
            temperature=0.5,
            system_prompt="You are a quantitative trading strategist. Analyze performance data and suggest specific, data-driven parameter optimizations for a day trading strategy.",
        )

        response = self.claude.request(request)

        if not response.success:
            logger.error(f"Failed to generate suggestions: {response.error}")
            return []

        # Parse suggestions
        try:
            suggestions_data = json.loads(response.content)
            suggestions = []

            for item in suggestions_data:
                suggestions.append(
                    OptimizationSuggestion(
                        parameter=item["parameter"],
                        current_value=item["current_value"],
                        suggested_value=item["suggested_value"],
                        reasoning=item["reasoning"],
                        expected_impact=item["expected_impact"],
                        confidence=item["confidence"],
                        priority=item["priority"],
                    )
                )

            logger.info(f"Generated {len(suggestions)} optimization suggestions")
            return suggestions

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse optimization suggestions: {e}")
            logger.debug(f"Raw response: {response.content}")
            return []

    def generate_daily_review(
        self, review_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Generate end-of-day performance review using Claude AI

        Args:
            review_date: Date to review (defaults to today)

        Returns:
            Dictionary containing review analysis
        """
        if review_date is None:
            review_date = date.today()

        # Get trades for the day
        trades = self.db.get_trades(
            status="CLOSED", start_date=review_date, end_date=review_date
        )

        if not trades:
            return {
                "date": review_date.isoformat(),
                "summary": "No trades taken today",
                "trades": [],
            }

        # Separate wins and losses
        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) < 0]

        # Calculate daily metrics
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        win_rate = (len(wins) / len(trades)) * 100 if trades else 0
        avg_r = (
            sum(t.get("r_multiple", 0) for t in trades) / len(trades) if trades else 0
        )

        # Build context
        prompt = f"""Provide a comprehensive end-of-day trading review for {review_date.strftime('%B %d, %Y')}:

DAILY SUMMARY:
- Total Trades: {len(trades)}
- Wins: {len(wins)} | Losses: {len(losses)}
- Win Rate: {win_rate:.1f}%
- Net P&L: ${total_pnl:.2f}
- Avg R-Multiple: {avg_r:.2f}R

WINNING TRADES:
{json.dumps(wins, indent=2)}

LOSING TRADES:
{json.dumps(losses, indent=2)}

Provide analysis in JSON format:
{{
  "summary": "2-3 sentence overall assessment",
  "wins_analysis": [
    {{
      "trade": "SYMBOL Pattern @ Time",
      "what_went_well": "specific positives",
      "key_learning": "lesson to remember"
    }}
  ],
  "losses_analysis": [
    {{
      "trade": "SYMBOL Pattern @ Time",
      "what_went_wrong": "specific issues",
      "key_learning": "lesson to remember"
    }}
  ],
  "strengths_today": ["strength 1", "strength 2", ...],
  "areas_for_improvement": ["improvement 1", "improvement 2", ...],
  "tomorrow_game_plan": ["action 1", "action 2", ...]
}}

Be specific, constructive, and actionable."""

        request = ClaudeRequest(
            request_type="daily_review",
            prompt=prompt,
            max_tokens=2048,
            temperature=0.6,
            system_prompt="You are an experienced day trading coach providing constructive daily feedback.",
        )

        response = self.claude.request(request)

        if response.success:
            try:
                review = json.loads(response.content)
                review["date"] = review_date.isoformat()
                review["trades_count"] = len(trades)
                review["win_rate"] = win_rate
                review["net_pnl"] = total_pnl
                return review
            except json.JSONDecodeError:
                logger.error("Failed to parse daily review")
                return {"raw_review": response.content}
        else:
            return {"error": response.error}

    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive performance summary

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary containing all performance metrics
        """
        overall = self.analyze_overall_performance(days=days)
        patterns = self.analyze_pattern_performance(days=days)
        time_metrics = self.analyze_time_of_day_performance(days=days)

        return {
            "period_days": days,
            "overall": asdict(overall) if overall else None,
            "patterns": [asdict(p) for p in patterns],
            "time_of_day": time_metrics,
            "analysis_timestamp": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    # Test the optimizer
    optimizer = StrategyOptimizer()

    print("Strategy Optimizer Test")
    print("=" * 50)

    # Analyze overall performance
    metrics = optimizer.analyze_overall_performance(days=7)
    if metrics:
        print("\nOVERALL PERFORMANCE (7 days):")
        print(f"Total Trades: {metrics.total_trades}")
        print(f"Win Rate: {metrics.win_rate:.1f}%")
        print(f"Net P&L: ${metrics.net_pnl:.2f}")
        print(f"Avg R: {metrics.avg_r_multiple:.2f}R")
        print(f"Profit Factor: {metrics.profit_factor:.2f}")

    # Analyze pattern performance
    patterns = optimizer.analyze_pattern_performance(days=7)
    if patterns:
        print("\nPATTERN PERFORMANCE:")
        for pattern in patterns:
            print(f"\n{pattern.pattern_type}:")
            print(f"  Trades: {pattern.total_trades}")
            print(f"  Win Rate: {pattern.win_rate:.1f}%")
            print(f"  Avg R: {pattern.avg_r_multiple:.2f}R")
            print(f"  Profit Factor: {pattern.profit_factor:.2f}")

    # Time of day analysis
    time_metrics = optimizer.analyze_time_of_day_performance(days=7)
    if time_metrics:
        print("\nTIME OF DAY PERFORMANCE:")
        for period, stats in time_metrics.items():
            print(f"\n{period.upper()}:")
            print(f"  Trades: {stats['total_trades']}")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            print(f"  Net P&L: ${stats['net_pnl']:.2f}")

    print("\n" + "=" * 50)
    print("Test complete!")
