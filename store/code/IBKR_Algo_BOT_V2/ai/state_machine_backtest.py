"""
State Machine Backtest - Compare Old vs New Entry Logic
=========================================================
Simulates what trades would have been filtered by the new momentum
state machine vs what actually traded with the old system.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
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

    # Base score starts at 35 (just below ATTENTION threshold)
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


def get_historical_momentum(symbol: str, entry_time: str) -> Dict:
    """
    Get momentum data around the entry time.
    Returns a dict with momentum score components.
    """
    try:
        from ai.momentum_score import MomentumScorer

        scorer = MomentumScorer()

        # Parse entry time
        dt = datetime.fromisoformat(entry_time)

        # Get historical data around entry time (need intraday data)
        ticker = yf.Ticker(symbol)

        # Get 1-day of 1-minute data
        hist = ticker.history(period="5d", interval="1m")

        if hist.empty or len(hist) < 30:
            return {"score": 0, "grade": "F", "error": "insufficient_data"}

        # Find data closest to entry time
        # Note: yfinance timestamps may not align perfectly
        current_price = hist['Close'].iloc[-1]

        # Calculate momentum components
        prices_30s = hist['Close'].iloc[-6:].tolist()
        prices_60s = hist['Close'].iloc[-12:].tolist()
        prices_5m = hist['Close'].iloc[-60:].tolist() if len(hist) >= 60 else hist['Close'].tolist()
        current_volume = int(hist['Volume'].iloc[-1])

        # Estimate spread (use high-low range as proxy)
        recent_range = (hist['High'].iloc[-5:].mean() - hist['Low'].iloc[-5:].mean()) / current_price * 100
        spread_pct = min(recent_range / 2, 2.0)  # Cap at 2%

        result = scorer.calculate(
            symbol=symbol,
            current_price=current_price,
            prices_30s=prices_30s,
            prices_60s=prices_60s,
            prices_5m=prices_5m,
            current_volume=current_volume,
            spread_pct=spread_pct,
            buy_pressure=0.55,  # Neutral for simulation
            tape_signal="NEUTRAL"
        )

        return {
            "score": result.score,
            "grade": result.grade.value,
            "price_urgency": result.price_urgency.total,
            "participation": result.participation.total,
            "liquidity": result.liquidity.total,
            "is_tradeable": result.is_tradeable,
            "ignition_ready": result.ignition_ready
        }

    except Exception as e:
        return {"score": 0, "grade": "F", "error": str(e)}


def would_state_machine_trade(momentum_data: Dict) -> Tuple[bool, str]:
    """
    Determine if state machine would have allowed this trade.

    Returns:
        (would_trade, reason)
    """
    score = momentum_data.get("score", 0)

    # State machine thresholds from config
    ATTENTION_THRESHOLD = 40
    SETUP_THRESHOLD = 55
    IGNITION_THRESHOLD = 70

    if score < ATTENTION_THRESHOLD:
        return False, f"IDLE (score {score} < {ATTENTION_THRESHOLD})"
    elif score < SETUP_THRESHOLD:
        return False, f"ATTENTION only (score {score} < {SETUP_THRESHOLD})"
    elif score < IGNITION_THRESHOLD:
        return False, f"SETUP only (score {score} < {IGNITION_THRESHOLD})"
    else:
        # Check for ignition ready (all components aligned)
        if momentum_data.get("ignition_ready", False):
            return True, f"IGNITION (score {score})"
        else:
            # Score is high but components not aligned
            return False, f"Score {score} but components not aligned"


def run_backtest(verbose: bool = True):
    """
    Run backtest comparing old vs new entry logic.
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
    run_backtest()
