"""
DERO - Daily Evaluation Reporting Overlay

A read-only, non-invasive overlay system for generating daily and weekly
evaluation reports from trading bot metrics.

IMPORTANT: This module is READ-ONLY and MUST NOT affect trading execution.
- No imports from execution modules that could mutate state
- If DERO fails, the bot must continue normally
- Default mode: LOG_ONLY (no auto-tuning or parameter edits)

Components:
- time_context: Time awareness engine (ET timezone, session windows)
- market_context: Market regime detection (TREND/CHOP/NEWS/DEAD)
- metrics_daily: Daily metrics aggregation
- metrics_weekly: Weekly rollup aggregation
- patterns: Pattern analyzer (non-invasive flagging)
- report_writer: Report generation (MD + JSON)
"""

__version__ = "1.0.0"
__author__ = "Morpheus Trading Bot"

from .time_context import TimeContextEngine
from .market_context import MarketContextEngine
from .metrics_daily import DailyMetricsAggregator
from .metrics_weekly import WeeklyMetricsAggregator
from .patterns import PatternAnalyzer
from .report_writer import ReportWriter

__all__ = [
    "TimeContextEngine",
    "MarketContextEngine",
    "DailyMetricsAggregator",
    "WeeklyMetricsAggregator",
    "PatternAnalyzer",
    "ReportWriter",
]
