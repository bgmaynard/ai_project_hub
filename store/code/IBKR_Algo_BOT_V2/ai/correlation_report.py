"""
Trade Correlation Report
========================
Analyzes secondary triggers to find patterns that predict winners vs losers.

Usage:
    from ai.correlation_report import generate_correlation_report
    report = generate_correlation_report()
    print(report['summary'])
"""

import json
import logging
import os
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SCALPER_TRADES_FILE = os.path.join(os.path.dirname(__file__), "scalper_trades.json")


def load_trades() -> List[Dict]:
    """Load all trades from scalper_trades.json"""
    try:
        if os.path.exists(SCALPER_TRADES_FILE):
            with open(SCALPER_TRADES_FILE, "r") as f:
                data = json.load(f)
                return data.get("trades", [])
    except Exception as e:
        logger.error(f"Error loading trades: {e}")
    return []


def categorize_trade(trade: Dict) -> str:
    """Categorize trade as WIN, LOSS, or BREAKEVEN"""
    pnl = trade.get("pnl", 0)
    if pnl > 0.5:  # More than $0.50 profit
        return "WIN"
    elif pnl < -0.5:  # More than $0.50 loss
        return "LOSS"
    return "BREAKEVEN"


def analyze_categorical(trades: List[Dict], field: str) -> Dict:
    """Analyze a categorical field (time_of_day, day_of_week, etc.)"""
    results = defaultdict(
        lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0, "trades": 0}
    )

    for trade in trades:
        triggers = trade.get("secondary_triggers", {})
        if not triggers:
            continue

        value = triggers.get(field, "unknown")
        if value is None:
            value = "unknown"

        category = categorize_trade(trade)
        results[value]["trades"] += 1
        results[value]["total_pnl"] += trade.get("pnl", 0)

        if category == "WIN":
            results[value]["wins"] += 1
        elif category == "LOSS":
            results[value]["losses"] += 1

    # Calculate win rates
    analysis = {}
    for value, stats in results.items():
        total = stats["trades"]
        if total > 0:
            win_rate = stats["wins"] / total * 100
            avg_pnl = stats["total_pnl"] / total
            analysis[value] = {
                "trades": total,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "win_rate": round(win_rate, 1),
                "total_pnl": round(stats["total_pnl"], 2),
                "avg_pnl": round(avg_pnl, 2),
            }

    # Sort by win rate
    sorted_analysis = dict(
        sorted(analysis.items(), key=lambda x: x[1]["win_rate"], reverse=True)
    )

    return sorted_analysis


def analyze_numeric(trades: List[Dict], field: str, buckets: int = 5) -> Dict:
    """Analyze a numeric field by bucketing values"""
    values_with_outcomes = []

    for trade in trades:
        triggers = trade.get("secondary_triggers", {})
        if not triggers:
            continue

        value = triggers.get(field)
        if value is None or not isinstance(value, (int, float)):
            continue

        values_with_outcomes.append(
            {
                "value": float(value),
                "pnl": trade.get("pnl", 0),
                "category": categorize_trade(trade),
            }
        )

    if not values_with_outcomes:
        return {"error": "No data available"}

    # Get value range
    all_values = [v["value"] for v in values_with_outcomes]
    min_val = min(all_values)
    max_val = max(all_values)

    if min_val == max_val:
        return {"error": "All values identical"}

    # Create buckets
    bucket_size = (max_val - min_val) / buckets
    bucket_results = defaultdict(
        lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0, "trades": 0, "values": []}
    )

    for item in values_with_outcomes:
        bucket_idx = min(int((item["value"] - min_val) / bucket_size), buckets - 1)
        bucket_start = min_val + bucket_idx * bucket_size
        bucket_end = bucket_start + bucket_size
        bucket_label = f"{bucket_start:.2f}-{bucket_end:.2f}"

        bucket_results[bucket_label]["trades"] += 1
        bucket_results[bucket_label]["total_pnl"] += item["pnl"]
        bucket_results[bucket_label]["values"].append(item["value"])

        if item["category"] == "WIN":
            bucket_results[bucket_label]["wins"] += 1
        elif item["category"] == "LOSS":
            bucket_results[bucket_label]["losses"] += 1

    # Calculate stats
    analysis = {}
    for label, stats in bucket_results.items():
        total = stats["trades"]
        if total > 0:
            win_rate = stats["wins"] / total * 100
            avg_pnl = stats["total_pnl"] / total
            avg_value = statistics.mean(stats["values"])
            analysis[label] = {
                "trades": total,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "win_rate": round(win_rate, 1),
                "total_pnl": round(stats["total_pnl"], 2),
                "avg_pnl": round(avg_pnl, 2),
                "avg_value": round(avg_value, 3),
            }

    return analysis


def analyze_boolean(trades: List[Dict], field: str) -> Dict:
    """Analyze a boolean field"""
    results = {
        True: {"wins": 0, "losses": 0, "total_pnl": 0.0, "trades": 0},
        False: {"wins": 0, "losses": 0, "total_pnl": 0.0, "trades": 0},
    }

    for trade in trades:
        triggers = trade.get("secondary_triggers", {})
        if not triggers:
            continue

        value = bool(triggers.get(field, False))
        category = categorize_trade(trade)

        results[value]["trades"] += 1
        results[value]["total_pnl"] += trade.get("pnl", 0)

        if category == "WIN":
            results[value]["wins"] += 1
        elif category == "LOSS":
            results[value]["losses"] += 1

    analysis = {}
    for value, stats in results.items():
        total = stats["trades"]
        if total > 0:
            win_rate = stats["wins"] / total * 100
            avg_pnl = stats["total_pnl"] / total
            label = "Yes" if value else "No"
            analysis[label] = {
                "trades": total,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "win_rate": round(win_rate, 1),
                "total_pnl": round(stats["total_pnl"], 2),
                "avg_pnl": round(avg_pnl, 2),
            }

    return analysis


def calculate_correlation(trades: List[Dict], field: str) -> Optional[float]:
    """Calculate Pearson correlation between a numeric field and P&L"""
    pairs = []

    for trade in trades:
        triggers = trade.get("secondary_triggers", {})
        if not triggers:
            continue

        value = triggers.get(field)
        if value is None or not isinstance(value, (int, float)):
            continue

        pairs.append((float(value), trade.get("pnl", 0)))

    if len(pairs) < 3:
        return None

    # Calculate Pearson correlation
    n = len(pairs)
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]

    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)

    sum_sq_x = sum((x - mean_x) ** 2 for x in x_vals)
    sum_sq_y = sum((y - mean_y) ** 2 for y in y_vals)

    denominator = (sum_sq_x * sum_sq_y) ** 0.5

    if denominator == 0:
        return 0.0

    return round(numerator / denominator, 4)


def analyze_exit_reasons(trades: List[Dict]) -> Dict:
    """Analyze which exit reasons are most profitable"""
    results = defaultdict(
        lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0, "trades": 0}
    )

    for trade in trades:
        if trade.get("status") != "closed":
            continue

        reason = trade.get("exit_reason", "unknown")
        category = categorize_trade(trade)

        results[reason]["trades"] += 1
        results[reason]["total_pnl"] += trade.get("pnl", 0)

        if category == "WIN":
            results[reason]["wins"] += 1
        elif category == "LOSS":
            results[reason]["losses"] += 1

    analysis = {}
    for reason, stats in results.items():
        total = stats["trades"]
        if total > 0:
            win_rate = stats["wins"] / total * 100
            avg_pnl = stats["total_pnl"] / total
            analysis[reason] = {
                "trades": total,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "win_rate": round(win_rate, 1),
                "total_pnl": round(stats["total_pnl"], 2),
                "avg_pnl": round(avg_pnl, 2),
            }

    return dict(sorted(analysis.items(), key=lambda x: x[1]["total_pnl"], reverse=True))


def analyze_entry_signals(trades: List[Dict]) -> Dict:
    """Analyze which entry signals perform best"""
    results = defaultdict(
        lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0, "trades": 0}
    )

    for trade in trades:
        signal = trade.get("entry_signal", "unknown")
        category = categorize_trade(trade)

        results[signal]["trades"] += 1
        results[signal]["total_pnl"] += trade.get("pnl", 0)

        if category == "WIN":
            results[signal]["wins"] += 1
        elif category == "LOSS":
            results[signal]["losses"] += 1

    analysis = {}
    for signal, stats in results.items():
        total = stats["trades"]
        if total > 0:
            win_rate = stats["wins"] / total * 100
            avg_pnl = stats["total_pnl"] / total
            analysis[signal] = {
                "trades": total,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "win_rate": round(win_rate, 1),
                "total_pnl": round(stats["total_pnl"], 2),
                "avg_pnl": round(avg_pnl, 2),
            }

    return dict(sorted(analysis.items(), key=lambda x: x[1]["win_rate"], reverse=True))


def analyze_symbols(trades: List[Dict]) -> Dict:
    """Analyze performance by symbol"""
    results = defaultdict(
        lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0, "trades": 0}
    )

    for trade in trades:
        symbol = trade.get("symbol", "unknown")
        category = categorize_trade(trade)

        results[symbol]["trades"] += 1
        results[symbol]["total_pnl"] += trade.get("pnl", 0)

        if category == "WIN":
            results[symbol]["wins"] += 1
        elif category == "LOSS":
            results[symbol]["losses"] += 1

    analysis = {}
    for symbol, stats in results.items():
        total = stats["trades"]
        if total > 0:
            win_rate = stats["wins"] / total * 100
            avg_pnl = stats["total_pnl"] / total
            analysis[symbol] = {
                "trades": total,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "win_rate": round(win_rate, 1),
                "total_pnl": round(stats["total_pnl"], 2),
                "avg_pnl": round(avg_pnl, 2),
            }

    return dict(sorted(analysis.items(), key=lambda x: x[1]["total_pnl"], reverse=True))


def generate_correlation_report(trades: List[Dict] = None) -> Dict:
    """
    Generate comprehensive correlation report.

    Returns dict with:
    - summary: High-level findings
    - timing: Time-based analysis
    - technicals: Technical indicator analysis
    - catalysts: News/SPY analysis
    - signals: Entry/exit signal analysis
    - correlations: Numeric correlations with P&L
    - recommendations: Actionable insights
    """
    if trades is None:
        trades = load_trades()

    # Filter to closed trades only
    closed_trades = [t for t in trades if t.get("status") == "closed"]

    # Count trades with secondary triggers
    trades_with_triggers = [t for t in closed_trades if t.get("secondary_triggers")]

    if not closed_trades:
        return {"error": "No closed trades found"}

    # Overall stats
    total_trades = len(closed_trades)
    wins = len([t for t in closed_trades if categorize_trade(t) == "WIN"])
    losses = len([t for t in closed_trades if categorize_trade(t) == "LOSS"])
    total_pnl = sum(t.get("pnl", 0) for t in closed_trades)

    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_trades": total_trades,
            "trades_with_triggers": len(trades_with_triggers),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total_trades * 100, 1) if total_trades > 0 else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / total_trades, 2) if total_trades > 0 else 0,
        },
    }

    # Only analyze triggers if we have data
    if trades_with_triggers:
        # Timing analysis
        report["timing"] = {
            "time_of_day": analyze_categorical(trades_with_triggers, "time_of_day"),
            "day_of_week": analyze_categorical(trades_with_triggers, "day_of_week"),
        }

        # Technical analysis
        report["technicals"] = {
            "spread_at_entry": analyze_numeric(trades_with_triggers, "spread_at_entry"),
            "momentum_strength": analyze_numeric(
                trades_with_triggers, "momentum_strength"
            ),
            "volume_spike_percent": analyze_numeric(
                trades_with_triggers, "volume_spike_percent"
            ),
            "distance_from_hod": analyze_numeric(
                trades_with_triggers, "distance_from_hod"
            ),
            "distance_from_lod": analyze_numeric(
                trades_with_triggers, "distance_from_lod"
            ),
            "rsi_at_entry": analyze_numeric(trades_with_triggers, "rsi_at_entry"),
            "float_rotation": analyze_numeric(trades_with_triggers, "float_rotation"),
            "relative_volume": analyze_numeric(trades_with_triggers, "relative_volume"),
        }

        # Categorical technicals
        report["technicals"]["vwap_position"] = analyze_categorical(
            trades_with_triggers, "vwap_position"
        )
        report["technicals"]["macd_signal"] = analyze_categorical(
            trades_with_triggers, "macd_signal"
        )

        # Catalyst analysis
        report["catalysts"] = {
            "has_news_catalyst": analyze_boolean(
                trades_with_triggers, "has_news_catalyst"
            ),
            "spy_direction": analyze_categorical(trades_with_triggers, "spy_direction"),
            "borrow_status": analyze_categorical(trades_with_triggers, "borrow_status"),
        }

        # Correlations
        numeric_fields = [
            "spread_at_entry",
            "momentum_strength",
            "volume_spike_percent",
            "distance_from_hod",
            "distance_from_lod",
            "rsi_at_entry",
            "float_rotation",
            "relative_volume",
            "day_change_at_entry",
        ]

        correlations = {}
        for field in numeric_fields:
            corr = calculate_correlation(trades_with_triggers, field)
            if corr is not None:
                correlations[field] = corr

        # Sort by absolute correlation
        report["correlations"] = dict(
            sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        )
    else:
        report["timing"] = {"note": "No trades with secondary triggers yet"}
        report["technicals"] = {"note": "No trades with secondary triggers yet"}
        report["catalysts"] = {"note": "No trades with secondary triggers yet"}
        report["correlations"] = {"note": "No trades with secondary triggers yet"}

    # Signal analysis (works on all trades)
    report["signals"] = {
        "entry_signals": analyze_entry_signals(closed_trades),
        "exit_reasons": analyze_exit_reasons(closed_trades),
    }

    # Symbol analysis
    report["symbols"] = analyze_symbols(closed_trades)

    # Generate recommendations
    report["recommendations"] = generate_recommendations(report)

    return report


def generate_recommendations(report: Dict) -> List[str]:
    """Generate actionable recommendations from the analysis"""
    recommendations = []

    # Check timing patterns
    timing = report.get("timing", {})
    if isinstance(timing.get("time_of_day"), dict):
        tod = timing["time_of_day"]
        best_times = [
            k
            for k, v in tod.items()
            if v.get("win_rate", 0) > 50 and v.get("trades", 0) >= 3
        ]
        worst_times = [
            k
            for k, v in tod.items()
            if v.get("win_rate", 0) < 30 and v.get("trades", 0) >= 3
        ]

        if best_times:
            recommendations.append(
                f"TIMING: Best time periods: {', '.join(best_times)}"
            )
        if worst_times:
            recommendations.append(
                f"TIMING: Avoid trading during: {', '.join(worst_times)}"
            )

    # Check exit reasons
    signals = report.get("signals", {})
    exit_reasons = signals.get("exit_reasons", {})

    stop_loss_stats = exit_reasons.get("STOP_LOSS", {})
    if stop_loss_stats.get("trades", 0) > 5:
        stop_pnl = stop_loss_stats.get("total_pnl", 0)
        recommendations.append(
            f"STOP LOSS: {stop_loss_stats['trades']} trades hit stop loss (${stop_pnl:.2f})"
        )

    trailing_stats = exit_reasons.get("TRAILING_STOP", {})
    if trailing_stats.get("trades", 0) > 0:
        trail_pnl = trailing_stats.get("total_pnl", 0)
        recommendations.append(
            f"TRAILING: {trailing_stats['trades']} trailing stops (${trail_pnl:.2f})"
        )

    # Check correlations
    correlations = report.get("correlations", {})
    if isinstance(correlations, dict) and correlations:
        for field, corr in correlations.items():
            if isinstance(corr, (int, float)) and abs(corr) > 0.3:
                direction = "positive" if corr > 0 else "negative"
                recommendations.append(
                    f"CORRELATION: {field} has {direction} correlation ({corr:.2f}) with P&L"
                )

    # Check symbols
    symbols = report.get("symbols", {})
    if symbols:
        profitable_symbols = [
            s for s, stats in symbols.items() if stats.get("total_pnl", 0) > 5
        ]
        losing_symbols = [
            s for s, stats in symbols.items() if stats.get("total_pnl", 0) < -10
        ]

        if profitable_symbols:
            recommendations.append(
                f"SYMBOLS: Profitable: {', '.join(profitable_symbols[:5])}"
            )
        if losing_symbols:
            recommendations.append(
                f"SYMBOLS: Avoid/review: {', '.join(losing_symbols[:5])}"
            )

    # Summary recommendation
    summary = report.get("summary", {})
    win_rate = summary.get("win_rate", 0)

    if win_rate < 35:
        recommendations.append(
            "STRATEGY: Win rate below 35% - consider tightening entry criteria"
        )
    elif win_rate > 50:
        recommendations.append(
            "STRATEGY: Win rate above 50% - strategy is working well"
        )

    if not recommendations:
        recommendations.append(
            "Need more trades with secondary triggers for detailed analysis"
        )

    return recommendations


def print_report(report: Dict):
    """Print a formatted text report"""
    print("\n" + "=" * 60)
    print("TRADE CORRELATION REPORT")
    print("=" * 60)
    print(f"Generated: {report.get('generated_at', 'N/A')}")

    # Summary
    summary = report.get("summary", {})
    print(f"\n--- SUMMARY ---")
    print(f"Total Trades: {summary.get('total_trades', 0)}")
    print(f"With Triggers: {summary.get('trades_with_triggers', 0)}")
    print(f"Wins: {summary.get('wins', 0)} | Losses: {summary.get('losses', 0)}")
    print(f"Win Rate: {summary.get('win_rate', 0)}%")
    print(f"Total P&L: ${summary.get('total_pnl', 0):.2f}")
    print(f"Avg P&L: ${summary.get('avg_pnl', 0):.2f}")

    # Timing
    timing = report.get("timing", {})
    if isinstance(timing.get("time_of_day"), dict) and timing["time_of_day"]:
        print(f"\n--- TIME OF DAY ---")
        for period, stats in timing["time_of_day"].items():
            print(
                f"  {period:20} | {stats['trades']:3} trades | {stats['win_rate']:5.1f}% win | ${stats['total_pnl']:7.2f}"
            )

    if isinstance(timing.get("day_of_week"), dict) and timing["day_of_week"]:
        print(f"\n--- DAY OF WEEK ---")
        for day, stats in timing["day_of_week"].items():
            print(
                f"  {day:20} | {stats['trades']:3} trades | {stats['win_rate']:5.1f}% win | ${stats['total_pnl']:7.2f}"
            )

    # Exit reasons
    signals = report.get("signals", {})
    exit_reasons = signals.get("exit_reasons", {})
    if exit_reasons:
        print(f"\n--- EXIT REASONS ---")
        for reason, stats in exit_reasons.items():
            print(
                f"  {reason:20} | {stats['trades']:3} trades | {stats['win_rate']:5.1f}% win | ${stats['total_pnl']:7.2f}"
            )

    # Entry signals
    entry_signals = signals.get("entry_signals", {})
    if entry_signals:
        print(f"\n--- ENTRY SIGNALS ---")
        for signal, stats in entry_signals.items():
            print(
                f"  {signal:20} | {stats['trades']:3} trades | {stats['win_rate']:5.1f}% win | ${stats['total_pnl']:7.2f}"
            )

    # Correlations
    correlations = report.get("correlations", {})
    if isinstance(correlations, dict) and correlations:
        print(f"\n--- CORRELATIONS WITH P&L ---")
        for field, corr in correlations.items():
            if isinstance(corr, (int, float)):
                bar = (
                    "+" * int(abs(corr) * 10) if corr > 0 else "-" * int(abs(corr) * 10)
                )
                print(f"  {field:25} | {corr:+.3f} | {bar}")
            else:
                print(f"  {field}: {corr}")

    # Top symbols
    symbols = report.get("symbols", {})
    if symbols:
        print(f"\n--- TOP 5 SYMBOLS ---")
        for i, (symbol, stats) in enumerate(list(symbols.items())[:5]):
            print(
                f"  {symbol:6} | {stats['trades']:3} trades | {stats['win_rate']:5.1f}% win | ${stats['total_pnl']:7.2f}"
            )

    # Recommendations
    recommendations = report.get("recommendations", [])
    if recommendations:
        print(f"\n--- RECOMMENDATIONS ---")
        for rec in recommendations:
            print(f"  • {rec}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = generate_correlation_report()
    print_report(report)
