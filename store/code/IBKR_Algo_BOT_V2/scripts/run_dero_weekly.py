#!/usr/bin/env python
"""
DERO Weekly Report Runner

Generates weekly evaluation report rollup.
Designed to be run by Windows Task Scheduler on Fridays at 6:00 PM ET
or Sundays.

Usage:
    python scripts/run_dero_weekly.py                     # Current week
    python scripts/run_dero_weekly.py --year 2026 --week 1  # Specific week
    python scripts/run_dero_weekly.py --last-week         # Last week
"""

import sys
import os
import argparse
import logging
from datetime import datetime, date, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "dero_scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_weekly_report(year: int, week: int) -> bool:
    """
    Generate weekly report for the specified year and week.

    Returns True on success, False on failure.
    """
    try:
        logger.info(f"Starting DERO weekly report for {year}-W{week:02d}")

        # Import DERO components
        from services.dero.metrics_weekly import get_weekly_metrics_aggregator
        from services.dero.report_writer import get_report_writer

        # Get components
        weekly_agg = get_weekly_metrics_aggregator(project_root / "reports")
        report_writer = get_report_writer(project_root / "reports")

        # Aggregate weekly metrics
        logger.info("Aggregating weekly metrics...")
        metrics = weekly_agg.aggregate_week(year, week)

        # Check if we have data
        days_analyzed = metrics.get("summary", {}).get("days_analyzed", 0)
        if days_analyzed == 0:
            logger.warning(f"No daily reports found for week {year}-W{week:02d}")
            # Still generate the report (shows no data)

        # Write reports
        logger.info("Writing reports...")
        paths = report_writer.write_weekly_report(metrics)

        logger.info(f"Weekly report generated successfully:")
        logger.info(f"  Markdown: {paths['markdown']}")
        logger.info(f"  JSON: {paths['json']}")

        # Log summary
        summary = metrics.get("summary", {})
        logger.info(f"Summary: {summary.get('days_analyzed', 0)} days, "
                   f"{summary.get('total_trades', 0)} trades, "
                   f"${summary.get('total_pnl', 0):.2f} P&L, "
                   f"{summary.get('win_rate_pct', 0):.1f}% win rate")

        recommendation = metrics.get("recommended_investigation", "")
        if recommendation:
            logger.info(f"Recommended investigation: {recommendation}")

        return True

    except Exception as e:
        logger.error(f"Failed to generate weekly report: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate DERO weekly evaluation report")
    parser.add_argument(
        "--year",
        type=int,
        help="Year (default: current)",
        default=None
    )
    parser.add_argument(
        "--week",
        type=int,
        help="ISO week number (default: current)",
        default=None
    )
    parser.add_argument(
        "--last-week",
        action="store_true",
        help="Generate report for last week"
    )

    args = parser.parse_args()

    today = date.today()

    if args.last_week:
        last_week = today - timedelta(days=7)
        year, week, _ = last_week.isocalendar()
    elif args.year and args.week:
        year = args.year
        week = args.week
    else:
        year, week, _ = today.isocalendar()

    success = run_weekly_report(year, week)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
