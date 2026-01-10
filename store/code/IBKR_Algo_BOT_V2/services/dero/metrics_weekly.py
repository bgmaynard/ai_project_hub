"""
DERO Weekly Metrics Aggregator

Rolls up daily metrics into weekly summaries.
This is READ-ONLY and does not affect trading execution.

Weekly Computes:
- Totals + averages
- Best/worst regime days
- Most common pattern flags
- Top gating reasons
- One recommended investigation (NOT a tuning instruction)
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict
import json
import logging

logger = logging.getLogger(__name__)


class WeeklyMetricsAggregator:
    """
    Aggregates daily metrics into weekly summaries.

    All operations are read-only.
    """

    def __init__(self, reports_dir: Optional[Path] = None):
        self.reports_dir = reports_dir or Path("reports")
        self.daily_dir = self.reports_dir / "daily"

    def _load_daily_report(self, target_date: date) -> Optional[Dict[str, Any]]:
        """Load daily report for a specific date"""
        json_path = self.daily_dir / f"daily_eval_{target_date.isoformat()}.json"
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load daily report {json_path}: {e}")
        return None

    def _get_week_dates(self, year: int, week: int) -> List[date]:
        """Get all dates in a given ISO week"""
        # Find the first day of the week (Monday)
        jan1 = date(year, 1, 1)
        # ISO week starts on Monday
        jan1_weekday = jan1.isocalendar()[2]  # 1=Monday, 7=Sunday

        # Calculate days to the first Monday
        days_to_first_monday = (8 - jan1_weekday) % 7
        first_monday = jan1 + timedelta(days=days_to_first_monday)

        # Calculate the Monday of the target week
        target_monday = first_monday + timedelta(weeks=week - 1)

        # Return Monday through Friday (trading days)
        return [target_monday + timedelta(days=i) for i in range(5)]

    def aggregate_week(self, year: int, week: int) -> Dict[str, Any]:
        """
        Aggregate daily reports for a given week.

        Args:
            year: Year (e.g., 2026)
            week: ISO week number (1-52)

        Returns:
            Weekly aggregated metrics
        """
        week_dates = self._get_week_dates(year, week)
        daily_reports = []

        # Load all available daily reports for the week
        for d in week_dates:
            report = self._load_daily_report(d)
            if report:
                daily_reports.append(report)

        if not daily_reports:
            return self._empty_weekly_metrics(year, week)

        return self._compute_weekly_metrics(year, week, daily_reports)

    def _empty_weekly_metrics(self, year: int, week: int) -> Dict[str, Any]:
        """Return empty weekly metrics structure"""
        return {
            "year": year,
            "week": week,
            "summary": {
                "days_analyzed": 0,
                "green_days": 0,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate_pct": 0,
                "total_pnl": 0,
            },
            "best_day": None,
            "worst_day": None,
            "regime_performance": {},
            "top_gating_reasons": [],
            "common_patterns": [],
            "recommended_investigation": "No data available for analysis.",
            "daily_summaries": [],
            "generated_at": datetime.now().isoformat(),
        }

    def _compute_weekly_metrics(
        self,
        year: int,
        week: int,
        daily_reports: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute weekly metrics from daily reports"""

        # Initialize accumulators
        total_trades = 0
        total_wins = 0
        total_losses = 0
        total_pnl = 0.0
        green_days = 0

        best_day = None
        best_pnl = float('-inf')
        worst_day = None
        worst_pnl = float('inf')

        regime_stats = defaultdict(lambda: {
            "days": 0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "pnl": 0.0
        })

        gating_reasons = defaultdict(int)
        pattern_counts = defaultdict(lambda: {"count": 0, "severity": "NONE"})

        daily_summaries = []

        # Process each daily report
        for report in daily_reports:
            date_str = report.get("date", "Unknown")
            data_health = report.get("data_health", "UNKNOWN")
            outcomes = report.get("outcomes", {})
            gating = report.get("gating", {})
            market = report.get("market", {})
            patterns = report.get("patterns", {})

            if data_health == "GREEN":
                green_days += 1

            # Outcomes
            day_trades = outcomes.get("total_trades", 0)
            day_wins = outcomes.get("wins", 0)
            day_losses = outcomes.get("losses", 0)
            day_pnl = outcomes.get("total_pnl", 0)

            total_trades += day_trades
            total_wins += day_wins
            total_losses += day_losses
            total_pnl += day_pnl

            # Best/Worst day tracking
            regime = market.get("regime", "UNKNOWN")

            if day_pnl > best_pnl:
                best_pnl = day_pnl
                best_day = {"date": date_str, "pnl": day_pnl, "regime": regime}

            if day_pnl < worst_pnl:
                worst_pnl = day_pnl
                worst_day = {"date": date_str, "pnl": day_pnl, "regime": regime}

            # Regime performance
            regime_stats[regime]["days"] += 1
            regime_stats[regime]["trades"] += day_trades
            regime_stats[regime]["wins"] += day_wins
            regime_stats[regime]["losses"] += day_losses
            regime_stats[regime]["pnl"] += day_pnl

            # Gating reasons
            top_reasons = gating.get("top_block_reasons", [])
            for r in top_reasons:
                reason = r.get("reason", "Unknown")
                count = r.get("count", 0)
                gating_reasons[reason] += count

            # Pattern flags
            pattern_flags = patterns.get("pattern_flags", [])
            for p in pattern_flags:
                pid = p.get("id", "Unknown")
                severity = p.get("severity", "NONE")
                pattern_counts[pid]["count"] += 1
                # Keep highest severity
                if severity == "CONFIRMED" or (severity == "POSSIBLE" and pattern_counts[pid]["severity"] == "NONE"):
                    pattern_counts[pid]["severity"] = severity

            # Daily summary
            daily_summaries.append({
                "date": date_str,
                "data_health": data_health,
                "regime": regime,
                "trades": day_trades,
                "win_rate": outcomes.get("win_rate_pct", 0),
                "pnl": day_pnl,
            })

        # Compute final metrics
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

        # Regime performance with win rates
        regime_performance = {}
        for regime, stats in regime_stats.items():
            regime_trades = stats["trades"]
            regime_wins = stats["wins"]
            regime_performance[regime] = {
                "days": stats["days"],
                "trades": regime_trades,
                "wins": regime_wins,
                "losses": stats["losses"],
                "win_rate": (regime_wins / regime_trades * 100) if regime_trades > 0 else 0,
                "pnl": round(stats["pnl"], 2),
            }

        # Top gating reasons sorted by count
        top_gating = sorted(
            [{"reason": r, "count": c} for r, c in gating_reasons.items()],
            key=lambda x: x["count"],
            reverse=True
        )[:10]

        # Common patterns sorted by count
        common_patterns = sorted(
            [{"id": pid, "occurrences": data["count"], "severity": data["severity"]}
             for pid, data in pattern_counts.items()],
            key=lambda x: x["occurrences"],
            reverse=True
        )[:10]

        # Generate recommended investigation
        recommendation = self._generate_recommendation(
            regime_performance, top_gating, common_patterns, total_pnl
        )

        return {
            "year": year,
            "week": week,
            "summary": {
                "days_analyzed": len(daily_reports),
                "green_days": green_days,
                "total_trades": total_trades,
                "wins": total_wins,
                "losses": total_losses,
                "win_rate_pct": round(win_rate, 1),
                "total_pnl": round(total_pnl, 2),
            },
            "best_day": best_day,
            "worst_day": worst_day,
            "regime_performance": regime_performance,
            "top_gating_reasons": top_gating,
            "common_patterns": common_patterns,
            "recommended_investigation": recommendation,
            "daily_summaries": daily_summaries,
            "generated_at": datetime.now().isoformat(),
        }

    def _generate_recommendation(
        self,
        regime_performance: Dict[str, Any],
        top_gating: List[Dict[str, Any]],
        common_patterns: List[Dict[str, Any]],
        total_pnl: float
    ) -> str:
        """Generate a single recommended investigation based on weekly data"""

        recommendations = []

        # Check for regime-specific issues
        for regime, stats in regime_performance.items():
            if stats["trades"] >= 5 and stats["win_rate"] < 25:
                recommendations.append(
                    f"INVESTIGATE {regime} regime performance: {stats['win_rate']:.0f}% win rate with ${stats['pnl']:.2f} P&L"
                )

        # Check top gating reason
        if top_gating and top_gating[0]["count"] > 10:
            top_reason = top_gating[0]
            recommendations.append(
                f"REVIEW gating threshold for '{top_reason['reason']}': {top_reason['count']} blocks this week"
            )

        # Check confirmed patterns
        confirmed_patterns = [p for p in common_patterns if p["severity"] == "CONFIRMED"]
        if confirmed_patterns:
            top_pattern = confirmed_patterns[0]
            recommendations.append(
                f"ADDRESS confirmed pattern '{top_pattern['id']}': occurred {top_pattern['occurrences']} times"
            )

        # Overall P&L concern
        if total_pnl < -100:
            recommendations.append(
                f"CRITICAL: Weekly P&L is ${total_pnl:.2f} - review overall strategy alignment"
            )

        if recommendations:
            return recommendations[0]  # Return top priority recommendation
        elif total_pnl > 0:
            return "No specific investigation needed. Week was profitable - continue monitoring."
        else:
            return "No critical patterns detected. Review daily reports for incremental improvements."

    def aggregate_current_week(self) -> Dict[str, Any]:
        """Aggregate metrics for the current week"""
        today = date.today()
        year, week, _ = today.isocalendar()
        return self.aggregate_week(year, week)

    def aggregate_last_week(self) -> Dict[str, Any]:
        """Aggregate metrics for last week"""
        today = date.today()
        last_week = today - timedelta(days=7)
        year, week, _ = last_week.isocalendar()
        return self.aggregate_week(year, week)


# Singleton instance
_weekly_metrics_aggregator: Optional[WeeklyMetricsAggregator] = None


def get_weekly_metrics_aggregator(reports_dir: Optional[Path] = None) -> WeeklyMetricsAggregator:
    """Get or create the singleton WeeklyMetricsAggregator instance"""
    global _weekly_metrics_aggregator
    if _weekly_metrics_aggregator is None:
        _weekly_metrics_aggregator = WeeklyMetricsAggregator(reports_dir)
    return _weekly_metrics_aggregator
