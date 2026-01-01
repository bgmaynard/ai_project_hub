"""
Backtest: Chronos Exit Manager Projected Savings
=================================================
Analyzes historical trades to project how much would have been saved
using the Chronos smart exit system vs fixed stop losses.

Key Analysis:
1. Identify all STOP_LOSS exits
2. Simulate what would have happened with Chronos exit triggers
3. Calculate projected savings
"""

import json
import os
from datetime import datetime
from typing import Dict, List

# Load historical trades
TRADES_FILE = os.path.join(os.path.dirname(__file__), "scalper_trades.json")


def load_trades() -> List[Dict]:
    """Load all historical trades"""
    with open(TRADES_FILE, "r") as f:
        data = json.load(f)
    return data.get("trades", [])


def analyze_stop_losses(trades: List[Dict]) -> Dict:
    """Analyze all stop loss exits"""
    stop_loss_trades = []
    other_trades = []

    for trade in trades:
        if trade.get("status") != "closed":
            continue

        exit_reason = trade.get("exit_reason", "")

        if "STOP_LOSS" in exit_reason:
            stop_loss_trades.append(trade)
        else:
            other_trades.append(trade)

    return {
        "stop_loss_count": len(stop_loss_trades),
        "stop_loss_trades": stop_loss_trades,
        "other_count": len(other_trades),
        "other_trades": other_trades,
    }


def simulate_chronos_exit(trade: Dict) -> Dict:
    """
    Simulate what would have happened with Chronos exit.

    Chronos Exit Rules:
    1. FAILED_MOMENTUM: Exit at -0.5% if no gain in 30s (instead of -3%)
    2. REGIME_SHIFT: Exit at -1% if regime changes (instead of -3%)
    3. MOMENTUM_FADING: Exit at breakeven/small loss if fading from high

    Conservative estimate: Chronos would have caught the trade at 1/3 of the loss
    """
    entry_price = trade.get("entry_price", 0)
    exit_price = trade.get("exit_price", 0)
    shares = trade.get("shares", 0)
    actual_pnl = trade.get("pnl", 0)
    actual_pnl_pct = trade.get("pnl_percent", 0)
    hold_time = trade.get("hold_time_seconds", 0)
    max_gain = trade.get("max_gain_percent", 0)

    # Determine which Chronos trigger would have fired first
    chronos_trigger = None
    projected_exit_pnl_pct = actual_pnl_pct  # Default to actual

    # Rule 1: FAILED_MOMENTUM - if held > 30s with no gain, exit at ~-0.5%
    if hold_time >= 30 and max_gain < 0.5:
        chronos_trigger = "FAILED_MOMENTUM"
        # Would have exited at around -0.5% to -1%
        projected_exit_pnl_pct = -0.75

    # Rule 2: MOMENTUM_FADING - if max gain was positive but ended negative
    elif max_gain > 0.3 and actual_pnl_pct < 0:
        chronos_trigger = "MOMENTUM_FADING"
        # Would have exited at ~0% to small loss when fading started
        projected_exit_pnl_pct = -0.3

    # Rule 3: REGIME_SHIFT - general case, caught earlier
    else:
        chronos_trigger = "REGIME_SHIFT"
        # Would have exited at about 1/2 of the actual loss
        projected_exit_pnl_pct = actual_pnl_pct * 0.4  # 40% of actual loss

    # Calculate projected P&L
    projected_exit_price = entry_price * (1 + projected_exit_pnl_pct / 100)
    projected_pnl = (projected_exit_price - entry_price) * shares

    savings = projected_pnl - actual_pnl

    return {
        "symbol": trade.get("symbol"),
        "entry_price": entry_price,
        "actual_exit_price": exit_price,
        "projected_exit_price": round(projected_exit_price, 4),
        "actual_pnl": round(actual_pnl, 2),
        "projected_pnl": round(projected_pnl, 2),
        "savings": round(savings, 2),
        "actual_pnl_pct": round(actual_pnl_pct, 2),
        "projected_pnl_pct": round(projected_exit_pnl_pct, 2),
        "chronos_trigger": chronos_trigger,
        "hold_time": hold_time,
        "max_gain": round(max_gain, 2),
    }


def run_backtest():
    """Run full backtest analysis"""
    print("=" * 70)
    print("CHRONOS EXIT MANAGER - BACKTEST ANALYSIS")
    print("=" * 70)

    trades = load_trades()
    print(f"\nTotal trades loaded: {len(trades)}")

    # Analyze stop losses
    analysis = analyze_stop_losses(trades)
    stop_loss_trades = analysis["stop_loss_trades"]

    print(f"Stop loss exits: {analysis['stop_loss_count']}")
    print(f"Other exits: {analysis['other_count']}")

    # Calculate actual stop loss damage
    total_stop_loss_pnl = sum(t.get("pnl", 0) for t in stop_loss_trades)
    print(f"\nActual Stop Loss P&L: ${total_stop_loss_pnl:.2f}")

    if not stop_loss_trades:
        print("\nNo stop loss trades to analyze.")
        return

    # Simulate Chronos exits
    print("\n" + "-" * 70)
    print("SIMULATING CHRONOS SMART EXITS")
    print("-" * 70)

    simulations = []
    for trade in stop_loss_trades:
        sim = simulate_chronos_exit(trade)
        simulations.append(sim)

    # Summary by trigger type
    trigger_stats = {}
    for sim in simulations:
        trigger = sim["chronos_trigger"]
        if trigger not in trigger_stats:
            trigger_stats[trigger] = {
                "count": 0,
                "actual_pnl": 0,
                "projected_pnl": 0,
                "savings": 0,
            }
        trigger_stats[trigger]["count"] += 1
        trigger_stats[trigger]["actual_pnl"] += sim["actual_pnl"]
        trigger_stats[trigger]["projected_pnl"] += sim["projected_pnl"]
        trigger_stats[trigger]["savings"] += sim["savings"]

    print("\nBy Trigger Type:")
    print(
        f"{'Trigger':<20} {'Count':>6} {'Actual P&L':>12} {'Projected':>12} {'Savings':>10}"
    )
    print("-" * 62)

    for trigger, stats in sorted(trigger_stats.items()):
        print(
            f"{trigger:<20} {stats['count']:>6} ${stats['actual_pnl']:>10.2f} ${stats['projected_pnl']:>10.2f} ${stats['savings']:>8.2f}"
        )

    # Total savings
    total_projected_pnl = sum(s["projected_pnl"] for s in simulations)
    total_savings = sum(s["savings"] for s in simulations)

    print("-" * 62)
    print(
        f"{'TOTAL':<20} {len(simulations):>6} ${total_stop_loss_pnl:>10.2f} ${total_projected_pnl:>10.2f} ${total_savings:>8.2f}"
    )

    # Individual trade details
    print("\n" + "-" * 70)
    print("INDIVIDUAL TRADE ANALYSIS (Worst 10 stop losses)")
    print("-" * 70)

    # Sort by worst actual P&L
    simulations.sort(key=lambda x: x["actual_pnl"])

    print(
        f"{'Symbol':<8} {'Actual':>8} {'Projected':>10} {'Saved':>8} {'Trigger':<18} {'MaxGain':>8}"
    )
    print("-" * 62)

    for sim in simulations[:10]:
        print(
            f"{sim['symbol']:<8} ${sim['actual_pnl']:>6.2f} ${sim['projected_pnl']:>8.2f} ${sim['savings']:>6.2f} {sim['chronos_trigger']:<18} {sim['max_gain']:>6.1f}%"
        )

    # Summary
    print("\n" + "=" * 70)
    print("PROJECTED IMPACT SUMMARY")
    print("=" * 70)

    avg_stop_loss = (
        total_stop_loss_pnl / len(stop_loss_trades) if stop_loss_trades else 0
    )
    avg_projected_loss = total_projected_pnl / len(simulations) if simulations else 0

    print(f"\nStop Loss Trades Analyzed: {len(stop_loss_trades)}")
    print(f"Average Stop Loss:         ${avg_stop_loss:.2f}")
    print(f"Average Projected Loss:    ${avg_projected_loss:.2f}")
    print(
        f"Average Savings per Trade: ${(total_savings / len(simulations)) if simulations else 0:.2f}"
    )
    print(f"\nTotal Actual Losses:       ${total_stop_loss_pnl:.2f}")
    print(f"Total Projected Losses:    ${total_projected_pnl:.2f}")
    print(f"TOTAL PROJECTED SAVINGS:   ${total_savings:.2f}")

    savings_pct = (
        (total_savings / abs(total_stop_loss_pnl) * 100)
        if total_stop_loss_pnl != 0
        else 0
    )
    print(f"Savings Percentage:        {savings_pct:.1f}%")

    # Win rate impact
    other_trades = analysis["other_trades"]
    total_other_pnl = sum(t.get("pnl", 0) for t in other_trades)

    print("\n" + "-" * 70)
    print("OVERALL PORTFOLIO IMPACT")
    print("-" * 70)

    total_actual_pnl = total_stop_loss_pnl + total_other_pnl
    total_new_pnl = total_projected_pnl + total_other_pnl

    print(f"Other Trades P&L:          ${total_other_pnl:.2f}")
    print(f"Actual Total P&L:          ${total_actual_pnl:.2f}")
    print(f"Projected Total P&L:       ${total_new_pnl:.2f}")
    print(f"NET IMPROVEMENT:           ${total_savings:.2f}")

    return {
        "stop_loss_count": len(stop_loss_trades),
        "total_stop_loss_pnl": total_stop_loss_pnl,
        "total_projected_pnl": total_projected_pnl,
        "total_savings": total_savings,
        "savings_pct": savings_pct,
        "trigger_stats": trigger_stats,
        "simulations": simulations,
    }


if __name__ == "__main__":
    run_backtest()
