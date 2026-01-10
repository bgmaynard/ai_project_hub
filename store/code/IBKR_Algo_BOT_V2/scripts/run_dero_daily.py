#!/usr/bin/env python
"""
DERO Daily Report Runner

Generates daily evaluation report for a specific date.
Designed to be run by Windows Task Scheduler at 5:15 PM ET.

Usage:
    python scripts/run_dero_daily.py                    # Today's report
    python scripts/run_dero_daily.py --date 2026-01-03  # Specific date
"""

import sys
import os
import argparse
import logging
from datetime import datetime, date
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


def run_daily_report(target_date: date) -> bool:
    """
    Generate daily report for the specified date.

    Returns True on success, False on failure.
    """
    try:
        logger.info(f"Starting DERO daily report for {target_date.isoformat()}")

        # Import DERO components
        from services.dero.time_context import get_time_context_engine
        from services.dero.market_context import get_market_context_engine
        from services.dero.metrics_daily import get_daily_metrics_aggregator
        from services.dero.patterns import get_pattern_analyzer
        from services.dero.report_writer import get_report_writer

        # Get components
        time_engine = get_time_context_engine()
        market_engine = get_market_context_engine()
        daily_agg = get_daily_metrics_aggregator(project_root)
        pattern_analyzer = get_pattern_analyzer()
        report_writer = get_report_writer(project_root / "reports")

        # Aggregate metrics
        logger.info("Aggregating daily metrics...")
        metrics = daily_agg.aggregate_all(target_date)

        # Add time context
        metrics["time_context"] = time_engine.build_context(
            datetime.combine(target_date, datetime.min.time())
        )

        # Try to get market context from API if server is running
        try:
            import httpx
            response = httpx.get("http://localhost:9100/api/market/spy-data", timeout=5)
            if response.status_code == 200:
                spy_data = response.json()
                metrics["market"] = market_engine.build_context(spy_data=spy_data)
            else:
                metrics["market"] = market_engine.build_context()
        except:
            metrics["market"] = market_engine.build_context()

        # Add pattern analysis
        logger.info("Analyzing patterns...")
        metrics["patterns"] = pattern_analyzer.analyze_all(metrics)

        # Write reports
        logger.info("Writing reports...")
        paths = report_writer.write_daily_report(metrics)

        logger.info(f"Daily report generated successfully:")
        logger.info(f"  Markdown: {paths['markdown']}")
        logger.info(f"  JSON: {paths['json']}")

        # Log summary
        outcomes = metrics.get("outcomes", {})
        logger.info(f"Summary: {outcomes.get('total_trades', 0)} trades, "
                   f"${outcomes.get('total_pnl', 0):.2f} P&L, "
                   f"{outcomes.get('win_rate_pct', 0):.1f}% win rate")

        return True

    except Exception as e:
        logger.error(f"Failed to generate daily report: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate DERO daily evaluation report")
    parser.add_argument(
        "--date",
        type=str,
        help="Target date in YYYY-MM-DD format (default: today)",
        default=None
    )

    args = parser.parse_args()

    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        target_date = date.today()

    success = run_daily_report(target_date)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
