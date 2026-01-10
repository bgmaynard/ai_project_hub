"""
DERO API Routes

Read-only endpoints for accessing DERO reports and status.
"""

from datetime import datetime, date, timedelta
from typing import Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
import json
import logging

from .time_context import get_time_context_engine
from .market_context import get_market_context_engine
from .metrics_daily import get_daily_metrics_aggregator
from .metrics_weekly import get_weekly_metrics_aggregator
from .patterns import get_pattern_analyzer
from .report_writer import get_report_writer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reports", tags=["DERO Reports"])


@router.get("/status")
async def get_dero_status():
    """Get DERO system status"""
    try:
        time_engine = get_time_context_engine()
        report_writer = get_report_writer()

        # Check for latest reports
        latest_daily = report_writer.get_latest_daily_report()
        latest_daily_date = latest_daily.get("date") if latest_daily else None

        return {
            "status": "operational",
            "version": "1.0",
            "current_time": time_engine.get_current_context(),
            "latest_daily_report": latest_daily_date,
            "reports_directory": str(report_writer.output_dir),
        }
    except Exception as e:
        logger.error(f"Error getting DERO status: {e}")
        return {"status": "error", "error": str(e)}


@router.get("/daily/latest")
async def get_latest_daily_report():
    """Get the most recent daily report"""
    try:
        report_writer = get_report_writer()
        report = report_writer.get_latest_daily_report()

        if report:
            return report
        else:
            raise HTTPException(status_code=404, detail="No daily reports found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest daily report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/daily/{date_str}")
async def get_daily_report_by_date(date_str: str):
    """Get daily report for a specific date (YYYY-MM-DD)"""
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        report_writer = get_report_writer()
        report = report_writer.get_daily_report(target_date)

        if report:
            return report
        else:
            raise HTTPException(status_code=404, detail=f"No report found for {date_str}")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting daily report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weekly/latest")
async def get_latest_weekly_report():
    """Get the most recent weekly report (current or last week)"""
    try:
        weekly_agg = get_weekly_metrics_aggregator()
        report_writer = get_report_writer()

        # Try current week first
        report = weekly_agg.aggregate_current_week()

        # If no data, try last week
        if report["summary"]["days_analyzed"] == 0:
            report = weekly_agg.aggregate_last_week()

        return report
    except Exception as e:
        logger.error(f"Error getting latest weekly report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weekly/{year_week}")
async def get_weekly_report_by_week(year_week: str):
    """
    Get weekly report for a specific week.

    Format: YYYY-WW (e.g., 2026-01 for week 1 of 2026)
    """
    try:
        parts = year_week.split("-")
        if len(parts) != 2:
            raise ValueError("Invalid format")

        year = int(parts[0])
        week = int(parts[1].replace("W", ""))

        weekly_agg = get_weekly_metrics_aggregator()
        report = weekly_agg.aggregate_week(year, week)

        return report
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid format. Use YYYY-WW (e.g., 2026-01)")
    except Exception as e:
        logger.error(f"Error getting weekly report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/daily")
async def generate_daily_report(
    date_str: Optional[str] = Query(None, description="Date in YYYY-MM-DD format (default: today)")
):
    """Generate a daily report for a specific date"""
    try:
        if date_str:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            target_date = date.today()

        # Get components
        time_engine = get_time_context_engine()
        market_engine = get_market_context_engine()
        daily_agg = get_daily_metrics_aggregator()
        pattern_analyzer = get_pattern_analyzer()
        report_writer = get_report_writer()

        # Aggregate metrics
        metrics = daily_agg.aggregate_all(target_date)

        # Add time context
        metrics["time_context"] = time_engine.build_context(datetime.combine(target_date, datetime.min.time()))

        # Add market context (would need real data in production)
        metrics["market"] = market_engine.build_context()

        # Add pattern analysis
        metrics["patterns"] = pattern_analyzer.analyze_all(metrics)

        # Write reports
        paths = report_writer.write_daily_report(metrics)

        return {
            "status": "success",
            "date": target_date.isoformat(),
            "files": {
                "markdown": str(paths["markdown"]),
                "json": str(paths["json"]),
            },
            "summary": {
                "data_health": metrics.get("data_health"),
                "total_trades": metrics.get("outcomes", {}).get("total_trades", 0),
                "total_pnl": metrics.get("outcomes", {}).get("total_pnl", 0),
            }
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Error generating daily report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/weekly")
async def generate_weekly_report(
    year: Optional[int] = Query(None, description="Year (default: current)"),
    week: Optional[int] = Query(None, description="ISO week number (default: current)")
):
    """Generate a weekly report"""
    try:
        today = date.today()
        if year is None:
            year = today.year
        if week is None:
            _, week, _ = today.isocalendar()

        weekly_agg = get_weekly_metrics_aggregator()
        report_writer = get_report_writer()

        # Aggregate weekly metrics
        metrics = weekly_agg.aggregate_week(year, week)

        # Write reports
        paths = report_writer.write_weekly_report(metrics)

        return {
            "status": "success",
            "year": year,
            "week": week,
            "files": {
                "markdown": str(paths["markdown"]),
                "json": str(paths["json"]),
            },
            "summary": metrics.get("summary", {}),
        }
    except Exception as e:
        logger.error(f"Error generating weekly report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/time-context")
async def get_time_context():
    """Get current time context"""
    try:
        time_engine = get_time_context_engine()
        return time_engine.get_current_context()
    except Exception as e:
        logger.error(f"Error getting time context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-context")
async def get_market_context():
    """Get current market context/regime"""
    try:
        market_engine = get_market_context_engine()
        # In production, this would fetch real SPY/QQQ data
        return market_engine.build_context()
    except Exception as e:
        logger.error(f"Error getting market context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def get_pattern_analysis():
    """Get current pattern analysis"""
    try:
        pattern_analyzer = get_pattern_analyzer()
        return pattern_analyzer.analyze_all()
    except Exception as e:
        logger.error(f"Error getting pattern analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def register_dero_routes(app):
    """Register DERO routes with the main FastAPI app"""
    app.include_router(router)
    logger.info("DERO routes registered")
