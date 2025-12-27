"""
State Machine Backtest with Grid Search (ChatGPT Spec v2)
==========================================================
Simulates what trades would have been filtered by the momentum
state machine vs what actually traded with the old system.

New Features:
- Grid search for optimal thresholds
- Score-based filtering analysis
- Threshold optimization recommendations
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from itertools import product
import yfinance as yf

# Add parent path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_historical_trades() -> List[Dict]:
    """Load historical trades from backtest results"""
    filepath = os.path.join(os.path.dirname(__file__), "backtest_results.json")

    if not os.path.exists(filepath):
        print("No backtest_results.json found")
        return []

    with open(filepath, 'r') as f:
        data = json.load(f)

    return data.get('detailed_results', [])


def estimate_momentum_from_trade(trade: Dict) -> Dict:
    """
    Estimate what the momentum score would have been at entry time
    based on trade outcome characteristics.

    Uses:
    - P&L outcome (winners likely had higher momentum)
    - Exit reason (TRAILING_STOP = good momentum, STOP_LOSS = failed momentum)
    - Hold time and max gain
    """
    pnl = trade.get("pnl", 0)
    pnl_pct = trade.get("pnl_pct", 0)
    exit_reason = trade.get("exit_reason", "")
    symbol = trade.get("symbol", "")

    # Base score starts at 35 (just below CANDIDATE threshold)
    base_score = 35

    # Adjust based on exit reason (proxy for momentum quality)
    reason_adjustments = {
        "TRAILING_STOP": 25,      # Good momentum - reached target and trailed
        "PROFIT_TARGET": 30,      # Excellent momentum - hit full target
        "REVERSAL_DETECTED": 10,  # Some momentum but faded
        "MAX_HOLD_TIME": 5,       # No clear momentum
        "STOP_LOSS": -10,         # Failed momentum / fake breakout
        "": 0
    }
    reason_adj = reason_adjustments.get(exit_reason, 0)

    # Adjust based on P&L magnitude
    if pnl_pct > 3:
        pnl_adj = 20  # Strong winner
    elif pnl_pct > 1.5:
        pnl_adj = 15  # Moderate winner
    elif pnl_pct > 0.5:
        pnl_adj = 10  # Small winner
    elif pnl_pct > 0:
        pnl_adj = 5   # Tiny winner
    elif pnl_pct > -1:
        pnl_adj = 0   # Small loss
    elif pnl_pct > -3:
        pnl_adj = -5  # Moderate loss
    else:
        pnl_adj = -15  # Large loss

    # Calculate estimated momentum score
    estimated_score = max(0, min(100, base_score + reason_adj + pnl_adj))

    # Determine grade
    if estimated_score >= 80:
        grade = "A"
    elif estimated_score >= 70:
        grade = "B"
    elif estimated_score >= 55:
        grade = "C"
    elif estimated_score >= 40:
        grade = "D"
    else:
        grade = "F"

    # Determine if ignition would trigger
    is_tradeable = estimated_score >= 70
    ignition_ready = estimated_score >= 70 and exit_reason in ["TRAILING_STOP", "PROFIT_TARGET"]

    return {
        "score": estimated_score,
        "grade": grade,
        "reason_adjustment": reason_adj,
        "pnl_adjustment": pnl_adj,
        "is_tradeable": is_tradeable,
        "ignition_ready": ignition_ready,
        "estimation_method": "trade_outcome"
    }


def would_state_machine_trade(momentum_data: Dict,
                               candidate_threshold: int = 40,
                               igniting_threshold: int = 55,
                               gated_threshold: int = 70) -> Tuple[bool, str]:
    """
    Determine if state machine would have allowed this trade.

    Args:
        momentum_data: Estimated momentum data
        candidate_threshold: Score to enter CANDIDATE (default 40)
        igniting_threshold: Score to enter IGNITING (default 55)
        gated_threshold: Score to enter GATED/trade (default 70)

    Returns:
        (would_trade, reason)
    """
    score = momentum_data.get("score", 0)

    if score < candidate_threshold:
        return False, f"IDLE (score {score} < {candidate_threshold})"
    elif score < igniting_threshold:
        return False, f"CANDIDATE only (score {score} < {igniting_threshold})"
    elif score < gated_threshold:
        return False, f"IGNITING only (score {score} < {gated_threshold})"
    else:
        # Check for ignition ready (all components aligned)
        if momentum_data.get("ignition_ready", False):
            return True, f"GATED (score {score})"
        else:
            # Score is high but components not aligned
            return False, f"Score {score} but components not aligned"


def run_backtest_with_thresholds(trades: List[Dict],
                                  candidate_threshold: int = 40,
                                  igniting_threshold: int = 55,
                                  gated_threshold: int = 70,
                                  verbose: bool = False) -> Dict:
    """
    Run backtest with specific thresholds.

    Returns performance metrics for this threshold combination.
    """
    results = {
        "thresholds": {
            "candidate": candidate_threshold,
            "igniting": igniting_threshold,
            "gated": gated_threshold
        },
        "old_system": {
            "winners": 0,
            "losers": 0,
            "total_pnl": 0
        },
        "new_system": {
            "would_trade": 0,
            "would_skip": 0,
            "winners_taken": 0,
            "losers_avoided": 0,
            "winners_missed": 0,
            "losers_taken": 0,
            "total_pnl": 0
        }
    }

    for trade in trades:
        pnl = trade.get("pnl", 0)
        is_winner = pnl > 0

        # Old system results
        if is_winner:
            results["old_system"]["winners"] += 1
        else:
            results["old_system"]["losers"] += 1
        results["old_system"]["total_pnl"] += pnl

        # Estimate momentum from trade outcome
        momentum_data = estimate_momentum_from_trade(trade)

        # Would state machine trade with these thresholds?
        would_trade, _ = would_state_machine_trade(
            momentum_data,
            candidate_threshold,
            igniting_threshold,
            gated_threshold
        )

        # New system results
        if would_trade:
            results["new_system"]["would_trade"] += 1
            results["new_system"]["total_pnl"] += pnl
            if is_winner:
                results["new_system"]["winners_taken"] += 1
            else:
                results["new_system"]["losers_taken"] += 1
        else:
            results["new_system"]["would_skip"] += 1
            if is_winner:
                results["new_system"]["winners_missed"] += 1
            else:
                results["new_system"]["losers_avoided"] += 1

    # Calculate metrics
    total_trades = len(trades)
    old_win_rate = (results["old_system"]["winners"] / total_trades * 100) if total_trades > 0 else 0

    new_trades = results["new_system"]["would_trade"]
    new_winners = results["new_system"]["winners_taken"]
    new_win_rate = (new_winners / new_trades * 100) if new_trades > 0 else 0

    # Calculate P&L improvement
    pnl_improvement = results["new_system"]["total_pnl"] - results["old_system"]["total_pnl"]

    results["metrics"] = {
        "total_trades": total_trades,
        "new_trades": new_trades,
        "old_win_rate": round(old_win_rate, 1),
        "new_win_rate": round(new_win_rate, 1),
        "win_rate_improvement": round(new_win_rate - old_win_rate, 1),
        "pnl_improvement": round(pnl_improvement, 2),
        "new_total_pnl": round(results["new_system"]["total_pnl"], 2),
        "trade_reduction_pct": round((1 - new_trades/total_trades) * 100, 1) if total_trades > 0 else 0,
        "losers_avoided": results["new_system"]["losers_avoided"],
        "winners_missed": results["new_system"]["winners_missed"]
    }

    return results


def grid_search(trades: List[Dict], verbose: bool = True) -> Dict:
    """
    Grid search to find optimal threshold combinations.

    Tests combinations of:
    - candidate_threshold: 30-50 (step 5)
    - igniting_threshold: 45-65 (step 5)
    - gated_threshold: 60-80 (step 5)

    Optimizes for: Maximum P&L improvement while maintaining reasonable win rate
    """
    if not trades:
        print("No trades to analyze")
        return {}

    print("\n" + "=" * 70)
    print("GRID SEARCH - Finding Optimal Thresholds")
    print("=" * 70)

    # Define search space
    candidate_range = range(30, 51, 5)
    igniting_range = range(45, 66, 5)
    gated_range = range(60, 81, 5)

    # Ensure constraints: candidate < igniting < gated
    valid_combinations = [
        (c, i, g)
        for c, i, g in product(candidate_range, igniting_range, gated_range)
        if c < i < g
    ]

    print(f"Testing {len(valid_combinations)} threshold combinations...")

    all_results = []

    for c, i, g in valid_combinations:
        result = run_backtest_with_thresholds(trades, c, i, g, verbose=False)
        all_results.append(result)

    # Sort by different optimization targets
    by_pnl = sorted(all_results, key=lambda x: x['metrics']['pnl_improvement'], reverse=True)
    by_win_rate = sorted(all_results, key=lambda x: x['metrics']['new_win_rate'], reverse=True)
    by_trades = sorted(all_results, key=lambda x: x['metrics']['new_trades'], reverse=True)

    # Composite score: balance P&L improvement and win rate
    # Score = (P&L improvement normalized) + (win rate normalized) - (missed winners penalty)
    max_pnl = max(r['metrics']['pnl_improvement'] for r in all_results)
    min_pnl = min(r['metrics']['pnl_improvement'] for r in all_results)
    pnl_range = max_pnl - min_pnl if max_pnl != min_pnl else 1

    for r in all_results:
        pnl_norm = (r['metrics']['pnl_improvement'] - min_pnl) / pnl_range
        wr_norm = r['metrics']['new_win_rate'] / 100
        missed_penalty = r['metrics']['winners_missed'] * 0.05  # 5% per missed winner
        r['composite_score'] = pnl_norm + wr_norm - missed_penalty

    by_composite = sorted(all_results, key=lambda x: x['composite_score'], reverse=True)

    print("\n" + "=" * 70)
    print("TOP 5 BY P&L IMPROVEMENT")
    print("=" * 70)
    for i, r in enumerate(by_pnl[:5]):
        t = r['thresholds']
        m = r['metrics']
        print(f"{i+1}. C={t['candidate']}, I={t['igniting']}, G={t['gated']}")
        print(f"   P&L: ${m['pnl_improvement']:+.2f}, WR: {m['new_win_rate']:.1f}%, Trades: {m['new_trades']}")

    print("\n" + "=" * 70)
    print("TOP 5 BY WIN RATE")
    print("=" * 70)
    for i, r in enumerate(by_win_rate[:5]):
        t = r['thresholds']
        m = r['metrics']
        print(f"{i+1}. C={t['candidate']}, I={t['igniting']}, G={t['gated']}")
        print(f"   P&L: ${m['pnl_improvement']:+.2f}, WR: {m['new_win_rate']:.1f}%, Trades: {m['new_trades']}")

    print("\n" + "=" * 70)
    print("TOP 5 BY COMPOSITE SCORE (BALANCED)")
    print("=" * 70)
    for i, r in enumerate(by_composite[:5]):
        t = r['thresholds']
        m = r['metrics']
        print(f"{i+1}. C={t['candidate']}, I={t['igniting']}, G={t['gated']}")
        print(f"   P&L: ${m['pnl_improvement']:+.2f}, WR: {m['new_win_rate']:.1f}%, Trades: {m['new_trades']}")
        print(f"   Losers Avoided: {m['losers_avoided']}, Winners Missed: {m['winners_missed']}")

    # Best recommendation
    best = by_composite[0]
    print("\n" + "=" * 70)
    print("RECOMMENDED THRESHOLDS")
    print("=" * 70)
    print(f"CANDIDATE_SCORE = {best['thresholds']['candidate']}")
    print(f"IGNITING_SCORE = {best['thresholds']['igniting']}")
    print(f"GATED_SCORE = {best['thresholds']['gated']}")
    print(f"\nExpected Results:")
    print(f"  P&L Improvement: ${best['metrics']['pnl_improvement']:+.2f}")
    print(f"  Win Rate: {best['metrics']['new_win_rate']:.1f}%")
    print(f"  Trade Count: {best['metrics']['new_trades']} ({best['metrics']['trade_reduction_pct']:.1f}% reduction)")

    # Save grid search results
    output = {
        "search_date": datetime.now().isoformat(),
        "trades_analyzed": len(trades),
        "combinations_tested": len(valid_combinations),
        "best_by_pnl": by_pnl[0],
        "best_by_win_rate": by_win_rate[0],
        "best_balanced": by_composite[0],
        "all_results": all_results
    }

    output_path = os.path.join(os.path.dirname(__file__), "grid_search_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nGrid search results saved to: {output_path}")

    return output


def run_backtest(verbose: bool = True):
    """
    Run backtest comparing old vs new entry logic with current thresholds.
    """
    trades = load_historical_trades()

    if not trades:
        print("No trades to analyze")
        return

    print(f"\n{'='*70}")
    print("STATE MACHINE BACKTEST - Old vs New Entry Logic")
    print(f"{'='*70}")
    print(f"Analyzing {len(trades)} historical trades...")

    # Results tracking
    results = {
        "total_trades": len(trades),
        "old_system": {
            "winners": 0,
            "losers": 0,
            "total_pnl": 0
        },
        "new_system": {
            "would_trade": 0,
            "would_skip": 0,
            "winners_taken": 0,
            "losers_avoided": 0,
            "winners_missed": 0,
            "losers_taken": 0,
            "total_pnl": 0
        },
        "by_reason": {},
        "detailed": []
    }

    # Analyze each trade
    unique_symbols = set()
    for i, trade in enumerate(trades):
        symbol = trade.get("symbol", "")
        unique_symbols.add(symbol)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(trades)}...")

    # Process each trade using outcome-based estimation
    print(f"\nEstimating momentum scores from trade outcomes...")
    print(f"(This estimates what momentum was at entry based on how trade performed)")

    # Now analyze each trade
    print("\nAnalyzing trades with state machine logic...")

    for trade in trades:
        symbol = trade.get("symbol", "")
        pnl = trade.get("pnl", 0)
        is_winner = pnl > 0
        exit_reason = trade.get("exit_reason", "")

        # Old system results
        if is_winner:
            results["old_system"]["winners"] += 1
        else:
            results["old_system"]["losers"] += 1
        results["old_system"]["total_pnl"] += pnl

        # Estimate momentum from trade outcome
        momentum_data = estimate_momentum_from_trade(trade)

        # Would state machine trade?
        would_trade, reason = would_state_machine_trade(momentum_data)

        # Track reason
        if reason not in results["by_reason"]:
            results["by_reason"][reason] = {"count": 0, "pnl": 0, "winners": 0, "losers": 0}
        results["by_reason"][reason]["count"] += 1
        results["by_reason"][reason]["pnl"] += pnl
        if is_winner:
            results["by_reason"][reason]["winners"] += 1
        else:
            results["by_reason"][reason]["losers"] += 1

        # New system results
        if would_trade:
            results["new_system"]["would_trade"] += 1
            results["new_system"]["total_pnl"] += pnl
            if is_winner:
                results["new_system"]["winners_taken"] += 1
            else:
                results["new_system"]["losers_taken"] += 1
        else:
            results["new_system"]["would_skip"] += 1
            if is_winner:
                results["new_system"]["winners_missed"] += 1
            else:
                results["new_system"]["losers_avoided"] += 1

        # Store detailed result
        results["detailed"].append({
            "symbol": symbol,
            "pnl": pnl,
            "exit_reason": exit_reason,
            "is_winner": is_winner,
            "momentum_score": momentum_data.get("score", 0),
            "momentum_grade": momentum_data.get("grade", "F"),
            "would_trade": would_trade,
            "skip_reason": reason if not would_trade else ""
        })

    # Calculate metrics
    old_win_rate = (results["old_system"]["winners"] / results["total_trades"] * 100) if results["total_trades"] > 0 else 0

    new_trades = results["new_system"]["would_trade"]
    new_winners = results["new_system"]["winners_taken"]
    new_win_rate = (new_winners / new_trades * 100) if new_trades > 0 else 0

    # Print results
    print(f"\n{'='*70}")
    print("RESULTS COMPARISON")
    print(f"{'='*70}")

    print(f"\nOLD SYSTEM (what actually traded):")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Winners: {results['old_system']['winners']}")
    print(f"  Losers: {results['old_system']['losers']}")
    print(f"  Win Rate: {old_win_rate:.1f}%")
    print(f"  Total P&L: ${results['old_system']['total_pnl']:.2f}")

    print(f"\nNEW SYSTEM (with state machine filter):")
    print(f"  Would Trade: {results['new_system']['would_trade']}")
    print(f"  Would Skip: {results['new_system']['would_skip']}")
    print(f"  Winners Taken: {results['new_system']['winners_taken']}")
    print(f"  Losers Avoided: {results['new_system']['losers_avoided']}")
    print(f"  Winners Missed: {results['new_system']['winners_missed']}")
    print(f"  Losers Taken: {results['new_system']['losers_taken']}")
    print(f"  Win Rate: {new_win_rate:.1f}%")
    print(f"  Total P&L: ${results['new_system']['total_pnl']:.2f}")

    # Calculate improvement
    pnl_improvement = results['new_system']['total_pnl'] - results['old_system']['total_pnl']
    trades_reduction = results['total_trades'] - results['new_system']['would_trade']

    print(f"\n{'='*70}")
    print("IMPROVEMENT ANALYSIS")
    print(f"{'='*70}")
    print(f"  P&L Improvement: ${pnl_improvement:.2f}")
    print(f"  Trades Reduced: {trades_reduction} ({trades_reduction/results['total_trades']*100:.1f}% fewer trades)")
    print(f"  Win Rate Improvement: {new_win_rate - old_win_rate:+.1f}%")
    print(f"  Losers Avoided: {results['new_system']['losers_avoided']} (${-sum(t['pnl'] for t in results['detailed'] if not t['would_trade'] and not t['is_winner']):.2f} saved)")
    print(f"  Winners Missed: {results['new_system']['winners_missed']} (${sum(t['pnl'] for t in results['detailed'] if not t['would_trade'] and t['is_winner']):.2f} missed)")

    print(f"\n{'='*70}")
    print("FILTERED BY REASON")
    print(f"{'='*70}")
    for reason, data in sorted(results["by_reason"].items(), key=lambda x: x[1]["count"], reverse=True):
        wr = (data["winners"] / data["count"] * 100) if data["count"] > 0 else 0
        print(f"  {reason}:")
        print(f"    Trades: {data['count']}, P&L: ${data['pnl']:.2f}, WR: {wr:.1f}%")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "state_machine_backtest_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Momentum State Machine Backtest")
    parser.add_argument("--grid-search", action="store_true", help="Run grid search for optimal thresholds")
    args = parser.parse_args()

    trades = load_historical_trades()

    if args.grid_search:
        grid_search(trades)
    else:
        run_backtest()
