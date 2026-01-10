"""
DERO Report Writer

Generates daily and weekly evaluation reports in JSON and Markdown formats.
This is READ-ONLY and does not affect trading execution.

Output Paths:
- reports/daily/daily_eval_YYYY-MM-DD.md
- reports/daily/daily_eval_YYYY-MM-DD.json
- reports/weekly/weekly_eval_YYYY-WW.md
- reports/weekly/weekly_eval_YYYY-WW.json

Markdown Report Sections (in order):
1. Session Context
2. Market Context (Regime)
3. Infrastructure Health
4. Discovery Summary (Scanners)
5. Momentum Flow (FSM)
6. Gating Summary
7. Outcomes (contextual)
8. Pattern Flags
9. Human Notes (blank section)
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class ReportWriter:
    """
    Generates DERO reports in JSON and Markdown formats.

    All operations are read-only.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("reports")
        self.daily_dir = self.output_dir / "daily"
        self.weekly_dir = self.output_dir / "weekly"

        # Ensure directories exist
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.weekly_dir.mkdir(parents=True, exist_ok=True)

    def _health_emoji(self, health: str) -> str:
        """Get emoji for health status"""
        return {
            "GREEN": "🟢",
            "YELLOW": "🟡",
            "RED": "🔴",
        }.get(health, "⚪")

    def _regime_emoji(self, regime: str) -> str:
        """Get emoji for market regime"""
        return {
            "TREND": "📈",
            "CHOP": "↔️",
            "NEWS": "📰",
            "DEAD": "💤",
        }.get(regime, "❓")

    def _severity_emoji(self, severity: str) -> str:
        """Get emoji for pattern severity"""
        return {
            "CONFIRMED": "🚨",
            "POSSIBLE": "⚠️",
            "NONE": "✅",
        }.get(severity, "❓")

    def generate_daily_markdown(self, metrics: Dict[str, Any]) -> str:
        """Generate daily report in Markdown format"""
        date_str = metrics.get("date", "Unknown")
        mode = metrics.get("mode", "UNKNOWN")
        data_health = metrics.get("data_health", "UNKNOWN")

        time_ctx = metrics.get("time_context", {})
        market_ctx = metrics.get("market", {})
        infra = metrics.get("infrastructure", {})
        discovery = metrics.get("discovery", {})
        pipeline = metrics.get("pipeline", {})
        gating = metrics.get("gating", {})
        outcomes = metrics.get("outcomes", {})
        patterns = metrics.get("patterns", {})

        md = []

        # Header
        md.append(f"# Daily Evaluation Report: {date_str}")
        md.append("")
        md.append(f"**Mode:** {mode} | **Data Health:** {self._health_emoji(data_health)} {data_health}")
        md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
        md.append("")
        md.append("---")
        md.append("")

        # 1. Session Context
        md.append("## 1. Session Context")
        md.append("")
        if time_ctx:
            md.append(f"- **Timezone:** {time_ctx.get('timezone', 'America/New_York')}")
            md.append(f"- **Total Events:** {time_ctx.get('total_events', 0)}")
            md.append(f"- **Market Hours Events:** {time_ctx.get('market_hours_events', 0)}")
            md.append(f"- **Pre-Market Events:** {time_ctx.get('premarket_events', 0)}")

            event_counts = time_ctx.get("event_window_counts", {})
            if event_counts:
                md.append("")
                md.append("**Events by Window:**")
                for window, count in event_counts.items():
                    if count > 0:
                        md.append(f"- {window}: {count}")
        else:
            md.append("_No session context available_")
        md.append("")

        # 2. Market Context (Regime)
        md.append("## 2. Market Context (Regime)")
        md.append("")
        if market_ctx:
            regime = market_ctx.get("regime", "UNKNOWN")
            confidence = market_ctx.get("confidence", 0)
            md.append(f"**Regime:** {self._regime_emoji(regime)} {regime} (confidence: {confidence:.0%})")
            md.append("")

            features = market_ctx.get("features", {})
            if features:
                md.append("**Features:**")
                md.append(f"- Trend Slope: {features.get('trend_slope', 0):.4f}")
                md.append(f"- Realized Volatility: {features.get('realized_volatility', 0):.2%}")
                md.append(f"- Range Expansion: {features.get('range_expansion', 1):.2f}x")
                md.append(f"- Choppiness Index: {features.get('choppiness_index', 50):.1f}")
        else:
            md.append("_No market context available_")
        md.append("")

        # 3. Infrastructure Health
        md.append("## 3. Infrastructure Health")
        md.append("")
        if infra:
            health = infra.get("data_health", "UNKNOWN")
            md.append(f"**Status:** {self._health_emoji(health)} {health}")
            md.append("")
            md.append(f"- Feed Uptime: {infra.get('feed_uptime_pct', 0):.1f}%")
            md.append(f"- Error Count: {infra.get('error_count', 0)}")
            md.append(f"- Reconnects: {infra.get('reconnect_count', 0)}")
            md.append(f"- Feed Drops: {infra.get('feed_drops', 0)}")
            md.append(f"- Total Events: {infra.get('total_events', 0)}")
        else:
            md.append("_No infrastructure data available_")
        md.append("")

        # 4. Discovery Summary (Scanners)
        md.append("## 4. Discovery Summary (Scanners)")
        md.append("")
        if discovery:
            md.append(f"- **Total Candidates:** {discovery.get('total_candidates', 0)}")
            md.append(f"- **Unique Symbols:** {discovery.get('unique_symbols', 0)}")
            md.append(f"- **Symbols Added:** {discovery.get('symbols_added', 0)}")
            md.append(f"- **Symbols Removed:** {discovery.get('symbols_removed', 0)}")
            md.append(f"- **Churn:** {discovery.get('churn', 0)}")

            by_scanner = discovery.get("candidates_by_scanner", {})
            if by_scanner:
                md.append("")
                md.append("**By Scanner:**")
                for scanner, count in by_scanner.items():
                    md.append(f"- {scanner}: {count}")

            symbols = discovery.get("symbol_list", [])
            if symbols:
                md.append("")
                md.append(f"**Symbols:** {', '.join(symbols[:20])}")
                if len(symbols) > 20:
                    md.append(f"_...and {len(symbols) - 20} more_")
        else:
            md.append("_No discovery data available_")
        md.append("")

        # 5. Momentum Flow (FSM)
        md.append("## 5. Momentum Flow (FSM)")
        md.append("")
        if pipeline:
            state_counts = pipeline.get("state_counts", {})
            funnel = pipeline.get("funnel", {})

            md.append("**State Progression:**")
            for state, count in state_counts.items():
                if count > 0:
                    md.append(f"- {state}: {count}")

            md.append("")
            md.append("**Pipeline Funnel:**")
            md.append(f"- Discovery → Ignition: {funnel.get('discovery_to_ignition', 0)}")
            md.append(f"- Ignition → Confirmed: {funnel.get('ignition_to_confirmed', 0)}")
            md.append(f"- Confirmed → Gated: {funnel.get('confirmed_to_gated', 0)}")
            md.append(f"- Total Transitions: {pipeline.get('total_transitions', 0)}")
        else:
            md.append("_No pipeline data available_")
        md.append("")

        # 6. Gating Summary
        md.append("## 6. Gating Summary")
        md.append("")
        if gating:
            md.append(f"- **Total Decisions:** {gating.get('total_decisions', 0)}")
            md.append(f"- **Allowed:** {gating.get('allowed', 0)}")
            md.append(f"- **Blocked:** {gating.get('blocked', 0)}")
            md.append(f"- **Approval Rate:** {gating.get('approval_rate_pct', 0):.1f}%")

            top_reasons = gating.get("top_block_reasons", [])
            if top_reasons:
                md.append("")
                md.append("**Top Block Reasons:**")
                for r in top_reasons:
                    md.append(f"- {r.get('reason', 'Unknown')}: {r.get('count', 0)}")
        else:
            md.append("_No gating data available_")
        md.append("")

        # 7. Outcomes (Contextual)
        md.append("## 7. Outcomes (Contextual)")
        md.append("")
        if outcomes:
            total_trades = outcomes.get("total_trades", 0)
            wins = outcomes.get("wins", 0)
            losses = outcomes.get("losses", 0)
            win_rate = outcomes.get("win_rate_pct", 0)
            total_pnl = outcomes.get("total_pnl", 0)

            pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"

            md.append(f"- **Total Trades:** {total_trades}")
            md.append(f"- **Wins/Losses:** {wins}W / {losses}L")
            md.append(f"- **Win Rate:** {win_rate:.1f}%")
            md.append(f"- **Total P&L:** {pnl_emoji} ${total_pnl:.2f}")
            md.append(f"- **Avg R:** {outcomes.get('avg_r', 0):.3f}")
            md.append(f"- **Avg MAE:** {outcomes.get('avg_mae', 0):.2f}%")
            md.append(f"- **Avg MFE:** {outcomes.get('avg_mfe', 0):.2f}%")
            md.append(f"- **Avg Slippage:** {outcomes.get('avg_slippage', 0):.4f}")

            trades = outcomes.get("trade_details", [])
            if trades:
                md.append("")
                md.append("**Recent Trades:**")
                md.append("| Symbol | P&L | R |")
                md.append("|--------|-----|---|")
                for t in trades[:10]:
                    pnl = t.get("pnl", 0)
                    r = t.get("r_multiple", 0)
                    symbol = t.get("symbol", "?")
                    md.append(f"| {symbol} | ${pnl:.2f} | {r:.2f}R |")
        else:
            md.append("_No outcome data available_")
        md.append("")

        # 8. Pattern Flags
        md.append("## 8. Pattern Flags")
        md.append("")
        if patterns:
            pattern_flags = patterns.get("pattern_flags", [])
            summary = patterns.get("summary", {})

            md.append(f"**Sessions Analyzed:** {summary.get('sessions_analyzed', 0)} ({summary.get('green_sessions', 0)} GREEN)")
            md.append(f"**Patterns Found:** {summary.get('total_patterns', 0)} ({summary.get('confirmed', 0)} confirmed, {summary.get('possible', 0)} possible)")
            md.append("")

            if pattern_flags:
                for p in pattern_flags:
                    severity = p.get("severity", "NONE")
                    md.append(f"### {self._severity_emoji(severity)} {p.get('id', 'Unknown')}")
                    md.append("")
                    md.append(f"- **Severity:** {severity}")
                    md.append(f"- **Category:** {p.get('category', 'Unknown')}")
                    md.append(f"- **Occurrences:** {p.get('count_hits', 0)}/{p.get('count_lookback', 10)} sessions")
                    md.append(f"- **Description:** {p.get('description', '')}")
                    md.append(f"- **Suggested Action:** {p.get('suggested_action', 'INVESTIGATE')}")
                    md.append("")
            else:
                md.append("✅ No patterns detected")
        else:
            md.append("_Pattern analysis not available_")
        md.append("")

        # 9. Human Notes
        md.append("## 9. Human Notes")
        md.append("")
        md.append("_Add your observations here:_")
        md.append("")
        md.append("```")
        md.append("")
        md.append("```")
        md.append("")

        # Footer
        md.append("---")
        md.append("")
        md.append(f"*Report generated by DERO v1.0 | {datetime.now().isoformat()}*")

        return "\n".join(md)

    def generate_daily_json(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate daily report in JSON format"""
        return {
            "date": metrics.get("date"),
            "mode": metrics.get("mode", "PAPER"),
            "data_health": metrics.get("data_health", "UNKNOWN"),
            "market": metrics.get("market", {}),
            "time_context": metrics.get("time_context", {}),
            "infrastructure": metrics.get("infrastructure", {}),
            "discovery": metrics.get("discovery", {}),
            "pipeline": metrics.get("pipeline", {}),
            "gating": metrics.get("gating", {}),
            "outcomes": metrics.get("outcomes", {}),
            "patterns": metrics.get("patterns", {}),
            "artifacts": {
                "raw_events_sources": metrics.get("artifacts", {}).get("raw_events_sources", []),
                "report_version": "1.0",
                "generated_at": datetime.now().isoformat(),
            }
        }

    def write_daily_report(self, metrics: Dict[str, Any]) -> Dict[str, Path]:
        """
        Write daily report to files.

        Returns dict with paths to generated files.
        """
        date_str = metrics.get("date", date.today().isoformat())

        # Generate reports
        md_content = self.generate_daily_markdown(metrics)
        json_content = self.generate_daily_json(metrics)

        # Write files
        md_path = self.daily_dir / f"daily_eval_{date_str}.md"
        json_path = self.daily_dir / f"daily_eval_{date_str}.json"

        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info(f"Daily MD report written to {md_path}")
        except Exception as e:
            logger.error(f"Failed to write MD report: {e}")

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_content, f, indent=2)
            logger.info(f"Daily JSON report written to {json_path}")
        except Exception as e:
            logger.error(f"Failed to write JSON report: {e}")

        return {
            "markdown": md_path,
            "json": json_path,
        }

    def generate_weekly_markdown(self, weekly_metrics: Dict[str, Any]) -> str:
        """Generate weekly report in Markdown format"""
        week = weekly_metrics.get("week", "Unknown")
        year = weekly_metrics.get("year", datetime.now().year)

        md = []
        md.append(f"# Weekly Evaluation Report: {year}-W{week:02d}")
        md.append("")
        md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
        md.append("")
        md.append("---")
        md.append("")

        # Summary
        md.append("## Summary")
        md.append("")
        summary = weekly_metrics.get("summary", {})
        md.append(f"- **Days Analyzed:** {summary.get('days_analyzed', 0)}")
        md.append(f"- **Green Days:** {summary.get('green_days', 0)}")
        md.append(f"- **Total Trades:** {summary.get('total_trades', 0)}")
        md.append(f"- **Win Rate:** {summary.get('win_rate_pct', 0):.1f}%")
        md.append(f"- **Total P&L:** ${summary.get('total_pnl', 0):.2f}")
        md.append("")

        # Best/Worst Days
        md.append("## Best/Worst Days")
        md.append("")
        best = weekly_metrics.get("best_day", {})
        worst = weekly_metrics.get("worst_day", {})

        if best:
            md.append(f"**Best Day:** {best.get('date', 'N/A')} (${best.get('pnl', 0):.2f}, {best.get('regime', 'Unknown')} regime)")
        if worst:
            md.append(f"**Worst Day:** {worst.get('date', 'N/A')} (${worst.get('pnl', 0):.2f}, {worst.get('regime', 'Unknown')} regime)")
        md.append("")

        # Regime Performance
        md.append("## Regime Performance")
        md.append("")
        regimes = weekly_metrics.get("regime_performance", {})
        if regimes:
            md.append("| Regime | Days | Trades | Win Rate | P&L |")
            md.append("|--------|------|--------|----------|-----|")
            for regime, data in regimes.items():
                md.append(f"| {regime} | {data.get('days', 0)} | {data.get('trades', 0)} | {data.get('win_rate', 0):.1f}% | ${data.get('pnl', 0):.2f} |")
        md.append("")

        # Top Gating Reasons
        md.append("## Top Gating Reasons")
        md.append("")
        gating_reasons = weekly_metrics.get("top_gating_reasons", [])
        if gating_reasons:
            for r in gating_reasons[:5]:
                md.append(f"- {r.get('reason', 'Unknown')}: {r.get('count', 0)}")
        else:
            md.append("_No gating data available_")
        md.append("")

        # Pattern Summary
        md.append("## Pattern Summary")
        md.append("")
        patterns = weekly_metrics.get("common_patterns", [])
        if patterns:
            for p in patterns[:5]:
                md.append(f"- **{p.get('id', 'Unknown')}** ({p.get('severity', 'NONE')}): {p.get('occurrences', 0)} occurrences")
        else:
            md.append("✅ No recurring patterns detected")
        md.append("")

        # Recommended Investigation
        md.append("## Recommended Investigation")
        md.append("")
        recommendation = weekly_metrics.get("recommended_investigation", "No specific investigation recommended.")
        md.append(f"> {recommendation}")
        md.append("")

        # Footer
        md.append("---")
        md.append("")
        md.append(f"*Report generated by DERO v1.0 | {datetime.now().isoformat()}*")

        return "\n".join(md)

    def write_weekly_report(self, weekly_metrics: Dict[str, Any]) -> Dict[str, Path]:
        """
        Write weekly report to files.

        Returns dict with paths to generated files.
        """
        year = weekly_metrics.get("year", datetime.now().year)
        week = weekly_metrics.get("week", datetime.now().isocalendar()[1])

        # Generate reports
        md_content = self.generate_weekly_markdown(weekly_metrics)

        # Write files
        md_path = self.weekly_dir / f"weekly_eval_{year}-W{week:02d}.md"
        json_path = self.weekly_dir / f"weekly_eval_{year}-W{week:02d}.json"

        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info(f"Weekly MD report written to {md_path}")
        except Exception as e:
            logger.error(f"Failed to write weekly MD report: {e}")

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(weekly_metrics, f, indent=2)
            logger.info(f"Weekly JSON report written to {json_path}")
        except Exception as e:
            logger.error(f"Failed to write weekly JSON report: {e}")

        return {
            "markdown": md_path,
            "json": json_path,
        }

    def get_latest_daily_report(self) -> Optional[Dict[str, Any]]:
        """Get the most recent daily report"""
        try:
            json_files = sorted(self.daily_dir.glob("daily_eval_*.json"), reverse=True)
            if json_files:
                with open(json_files[0], "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load latest daily report: {e}")
        return None

    def get_daily_report(self, target_date: date) -> Optional[Dict[str, Any]]:
        """Get daily report for a specific date"""
        json_path = self.daily_dir / f"daily_eval_{target_date.isoformat()}.json"
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    return json.load(f)
            except:
                pass
        return None


# Singleton instance
_report_writer: Optional[ReportWriter] = None


def get_report_writer(output_dir: Optional[Path] = None) -> ReportWriter:
    """Get or create the singleton ReportWriter instance"""
    global _report_writer
    if _report_writer is None:
        _report_writer = ReportWriter(output_dir)
    return _report_writer
