"""
Session Report Generator
========================
Generates comprehensive trading session reports with statistics
from both pre-market and regular session trading.
"""

import json
import logging
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional
import pytz

logger = logging.getLogger(__name__)

ET_TZ = pytz.timezone('US/Eastern')
REPORTS_DIR = Path("reports")


def get_session_stats() -> Dict:
    """Get trading statistics from all active systems"""
    stats = {
        "generated_at": datetime.now(ET_TZ).isoformat(),
        "date": datetime.now(ET_TZ).strftime("%Y-%m-%d"),
        "sessions": {}
    }

    # Get HFT Scalper stats (pre-market + all day)
    try:
        from ai.hft_scalper import get_hft_scalper
        scalper = get_hft_scalper()
        scalper_stats = scalper.get_stats()

        stats["hft_scalper"] = {
            "total_trades": scalper_stats.get("total_trades", 0),
            "wins": scalper_stats.get("wins", 0),
            "losses": scalper_stats.get("losses", 0),
            "win_rate": scalper_stats.get("win_rate", 0),
            "total_pnl": scalper_stats.get("total_pnl", 0),
            "avg_win": scalper_stats.get("avg_win", 0),
            "avg_loss": scalper_stats.get("avg_loss", 0),
            "profit_factor": scalper_stats.get("profit_factor", 0),
            "best_trade": scalper_stats.get("best_trade", 0),
            "worst_trade": scalper_stats.get("worst_trade", 0),
        }
    except Exception as e:
        logger.warning(f"Could not get HFT Scalper stats: {e}")
        stats["hft_scalper"] = {"error": str(e)}

    # Get ATS + 9 EMA Sniper stats
    try:
        from ai.strategies.ats_9ema_sniper import get_sniper_strategy
        sniper = get_sniper_strategy()
        sniper_status = sniper.get_status()

        stats["sniper_strategy"] = {
            "enabled": sniper_status.get("enabled", False),
            "active_window": sniper_status.get("active_window", False),
            "today_stats": sniper_status.get("today_stats", {}),
            "config": sniper_status.get("config", {}),
        }
    except Exception as e:
        logger.warning(f"Could not get Sniper stats: {e}")
        stats["sniper_strategy"] = {"error": str(e)}

    # Get trading event logs
    try:
        from ai.logging.events import get_event_logger
        event_logger = get_event_logger()

        today_events = event_logger.get_today_events()
        today_stats = event_logger.get_today_stats()

        # Categorize events by session
        premarket_events = []
        sniper_events = []

        for event in today_events:
            event_time = event.get("timestamp", "")
            if event_time:
                try:
                    dt = datetime.fromisoformat(event_time)
                    et_time = dt.astimezone(ET_TZ).time()

                    if dt_time(7, 0) <= et_time < dt_time(9, 30):
                        premarket_events.append(event)
                    elif dt_time(9, 40) <= et_time < dt_time(11, 0):
                        sniper_events.append(event)
                except:
                    pass

        stats["sessions"]["premarket_7_930"] = {
            "window": "7:00 AM - 9:30 AM ET",
            "event_count": len(premarket_events),
            "events": premarket_events[-20:],  # Last 20 events
        }

        stats["sessions"]["sniper_940_1100"] = {
            "window": "9:40 AM - 11:00 AM ET",
            "event_count": len(sniper_events),
            "events": sniper_events[-20:],  # Last 20 events
        }

        stats["event_summary"] = today_stats

    except Exception as e:
        logger.warning(f"Could not get event logs: {e}")
        stats["events"] = {"error": str(e)}

    # Get watchlist summary
    try:
        from ai.hft_scalper import get_hft_scalper
        scalper = get_hft_scalper()
        stats["watchlist"] = {
            "symbols": list(scalper.watchlist),
            "count": len(scalper.watchlist),
        }
    except Exception as e:
        stats["watchlist"] = {"error": str(e)}

    return stats


def calculate_session_metrics(stats: Dict) -> Dict:
    """Calculate comparative metrics between sessions"""
    metrics = {
        "comparison": {},
        "recommendations": [],
    }

    # Extract stats
    scalper = stats.get("hft_scalper", {})
    sniper = stats.get("sniper_strategy", {}).get("today_stats", {})

    # Pre-market metrics (from scalper during 7-9:30)
    premarket = stats.get("sessions", {}).get("premarket_7_930", {})
    sniper_session = stats.get("sessions", {}).get("sniper_940_1100", {})

    metrics["comparison"] = {
        "premarket": {
            "window": "7:00 - 9:30 AM",
            "strategy": "HFT Scalper",
            "events": premarket.get("event_count", 0),
        },
        "morning_prime": {
            "window": "9:40 - 11:00 AM",
            "strategy": "ATS + 9 EMA Sniper",
            "events": sniper_session.get("event_count", 0),
            "trades": sniper.get("trades", 0),
            "wins": sniper.get("wins", 0),
            "losses": sniper.get("losses", 0),
            "win_rate": sniper.get("win_rate", 0),
            "pnl": sniper.get("total_pnl", 0),
        }
    }

    # Generate recommendations
    scalper_wr = scalper.get("win_rate", 0)
    sniper_wr = sniper.get("win_rate", 0)

    if scalper_wr < 30:
        metrics["recommendations"].append("Pre-market win rate below 30% - consider tighter entry filters")

    if sniper.get("trades", 0) == 0:
        metrics["recommendations"].append("No Sniper trades - check if ATS qualification is too strict")
    elif sniper_wr < 40:
        metrics["recommendations"].append("Sniper win rate below 40% - review pullback validation")

    if scalper.get("total_pnl", 0) < 0:
        metrics["recommendations"].append(f"Pre-market session negative P&L: ${scalper.get('total_pnl', 0):.2f}")

    return metrics


def generate_report() -> Dict:
    """Generate full session report"""
    stats = get_session_stats()
    metrics = calculate_session_metrics(stats)

    report = {
        "report_type": "DAILY_SESSION_REPORT",
        "generated_at": datetime.now(ET_TZ).isoformat(),
        "date": datetime.now(ET_TZ).strftime("%Y-%m-%d"),
        "statistics": stats,
        "metrics": metrics,
    }

    return report


def save_report(report: Dict = None) -> str:
    """Save report to file and return path"""
    if report is None:
        report = generate_report()

    # Ensure reports directory exists
    REPORTS_DIR.mkdir(exist_ok=True)

    # Generate filename
    date_str = datetime.now(ET_TZ).strftime("%Y%m%d")
    time_str = datetime.now(ET_TZ).strftime("%H%M%S")
    filename = f"session_report_{date_str}_{time_str}.json"
    filepath = REPORTS_DIR / filename

    # Save report
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Session report saved: {filepath}")
    return str(filepath)


def print_report_summary(report: Dict = None):
    """Print formatted report summary to console"""
    if report is None:
        report = generate_report()

    print("\n" + "=" * 60)
    print("DAILY SESSION REPORT")
    print(f"Date: {report['date']}")
    print("=" * 60)

    stats = report.get("statistics", {})

    # HFT Scalper Summary
    scalper = stats.get("hft_scalper", {})
    if "error" not in scalper:
        print("\nHFT SCALPER (All Day):")
        print(f"  Trades: {scalper.get('total_trades', 0)}")
        print(f"  Wins: {scalper.get('wins', 0)} | Losses: {scalper.get('losses', 0)}")
        print(f"  Win Rate: {scalper.get('win_rate', 0):.1f}%")
        print(f"  Total P&L: ${scalper.get('total_pnl', 0):.2f}")
        print(f"  Best: ${scalper.get('best_trade', 0):.2f} | Worst: ${scalper.get('worst_trade', 0):.2f}")

    # Sniper Strategy Summary
    sniper = stats.get("sniper_strategy", {}).get("today_stats", {})
    print("\nATS + 9 EMA SNIPER (9:40-11:00 AM):")
    print(f"  Trades: {sniper.get('trades', 0)}")
    print(f"  Wins: {sniper.get('wins', 0)} | Losses: {sniper.get('losses', 0)}")
    print(f"  Win Rate: {sniper.get('win_rate', 0):.1f}%")
    print(f"  Total P&L: ${sniper.get('total_pnl', 0):.2f}")

    # Session Comparison
    metrics = report.get("metrics", {})
    comparison = metrics.get("comparison", {})

    print("\nSESSION COMPARISON:")
    print(f"  Pre-market (7:00-9:30):  {comparison.get('premarket', {}).get('events', 0)} events")
    print(f"  Morning Prime (9:40-11:00): {comparison.get('morning_prime', {}).get('events', 0)} events")

    # Recommendations
    recommendations = metrics.get("recommendations", [])
    if recommendations:
        print("\nRECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  - {rec}")

    print("\n" + "=" * 60)


# API Functions
async def get_session_report() -> Dict:
    """API endpoint to get session report"""
    return generate_report()


async def save_session_report() -> Dict:
    """API endpoint to save session report"""
    report = generate_report()
    filepath = save_report(report)
    return {"success": True, "filepath": filepath, "report": report}


if __name__ == "__main__":
    # Generate and print report
    report = generate_report()
    print_report_summary(report)

    # Save report
    filepath = save_report(report)
    print(f"\nReport saved to: {filepath}")
