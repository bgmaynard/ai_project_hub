"""
Backtest New Filters (Gap Grader + Overnight Continuation)
==========================================================
Analyze historical trades and simulate how the new filters would have improved performance.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import yfinance as yf

# Add parent path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.gap_grader import GapGrader, GapGrade


def load_trades(filepath: str = None) -> List[Dict]:
    """Load historical trades"""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "scalper_trades.json")

    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get("trades", [])


def get_historical_data(symbol: str, trade_date: str) -> Dict:
    """Get historical data for a trade date"""
    try:
        ticker = yf.Ticker(symbol)

        # Parse trade date
        dt = datetime.fromisoformat(trade_date.replace('Z', '+00:00'))
        start = (dt - timedelta(days=5)).strftime('%Y-%m-%d')
        end = (dt + timedelta(days=1)).strftime('%Y-%m-%d')

        hist = ticker.history(start=start, end=end)

        if len(hist) < 2:
            return {}

        # Get prior day data
        prior_close = hist.iloc[-2]['Close'] if len(hist) >= 2 else 0
        prior_volume = hist.iloc[-2]['Volume'] if len(hist) >= 2 else 0
        prior_change = 0
        if len(hist) >= 3:
            prior_change = ((hist.iloc[-2]['Close'] - hist.iloc[-3]['Close']) / hist.iloc[-3]['Close']) * 100

        # Get stock info
        info = ticker.info
        float_shares = info.get('floatShares', 0)
        avg_volume = info.get('averageVolume', 0)

        return {
            'prior_close': prior_close,
            'prior_volume': prior_volume,
            'prior_change': prior_change,
            'float_shares': float_shares,
            'avg_volume': avg_volume
        }
    except Exception as e:
        return {}


def simulate_gap_grade(trade: Dict, hist_data: Dict) -> Tuple[str, int]:
    """Simulate gap grading for a trade"""
    grader = GapGrader()

    entry_price = trade.get('entry_price', 0)
    prior_close = hist_data.get('prior_close', entry_price * 0.95)

    if prior_close <= 0:
        prior_close = entry_price * 0.95

    gap_pct = ((entry_price - prior_close) / prior_close) * 100 if prior_close > 0 else 5.0

    # Estimate pre-market volume (use trade's implied volume activity)
    # Since we don't have exact PM volume, estimate based on avg_volume
    avg_vol = hist_data.get('avg_volume', 1000000)
    pm_vol = int(avg_vol * 0.05)  # Assume 5% of daily volume in PM

    graded = grader.grade_gap(
        symbol=trade['symbol'],
        gap_percent=gap_pct,
        current_price=entry_price,
        prior_close=prior_close,
        premarket_volume=pm_vol,
        float_shares=hist_data.get('float_shares', 0),
        avg_volume=avg_vol,
        prior_day_change=hist_data.get('prior_change', 0),
        catalyst_headline=""  # No catalyst data available
    )

    return graded.grade.value, graded.score.total


def run_backtest(min_grade: str = "C") -> Dict:
    """
    Run backtest filtering trades by gap grade.

    Args:
        min_grade: Minimum gap grade to accept (A, B, C, D, F)

    Returns:
        Backtest results comparing filtered vs unfiltered performance
    """
    trades = load_trades()

    if not trades:
        return {"error": "No trades found"}

    # Grade order for comparison
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
    min_order = grade_order.get(min_grade.upper(), 2)

    # Original stats
    orig_trades = len(trades)
    orig_wins = len([t for t in trades if t.get('pnl', 0) > 0])
    orig_losses = len([t for t in trades if t.get('pnl', 0) < 0])
    orig_pnl = sum(t.get('pnl', 0) for t in trades)
    orig_win_rate = (orig_wins / orig_trades * 100) if orig_trades > 0 else 0

    # Analyze unique symbols
    symbols = list(set(t['symbol'] for t in trades))
    print(f"\nAnalyzing {len(symbols)} unique symbols from {orig_trades} trades...")

    # Cache historical data
    symbol_cache = {}

    # Process trades with gap grading
    filtered_trades = []
    rejected_trades = []
    graded_results = []

    for i, trade in enumerate(trades):
        symbol = trade['symbol']

        # Get or cache historical data
        if symbol not in symbol_cache:
            entry_time = trade.get('entry_time', '')
            symbol_cache[symbol] = get_historical_data(symbol, entry_time)

        hist_data = symbol_cache[symbol]

        # Grade the trade
        grade, score = simulate_gap_grade(trade, hist_data)

        graded_results.append({
            'symbol': symbol,
            'grade': grade,
            'score': score,
            'pnl': trade.get('pnl', 0),
            'exit_reason': trade.get('exit_reason', '')
        })

        # Filter by grade
        if grade_order.get(grade, 4) <= min_order:
            filtered_trades.append(trade)
        else:
            rejected_trades.append(trade)

        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{orig_trades} trades...")

    # Filtered stats
    filt_trades = len(filtered_trades)
    filt_wins = len([t for t in filtered_trades if t.get('pnl', 0) > 0])
    filt_losses = len([t for t in filtered_trades if t.get('pnl', 0) < 0])
    filt_pnl = sum(t.get('pnl', 0) for t in filtered_trades)
    filt_win_rate = (filt_wins / filt_trades * 100) if filt_trades > 0 else 0

    # Rejected stats
    rej_trades = len(rejected_trades)
    rej_wins = len([t for t in rejected_trades if t.get('pnl', 0) > 0])
    rej_pnl = sum(t.get('pnl', 0) for t in rejected_trades)

    # Grade distribution
    grade_dist = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for r in graded_results:
        grade_dist[r['grade']] += 1

    # PnL by grade
    pnl_by_grade = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for r in graded_results:
        pnl_by_grade[r['grade']] += r['pnl']

    # Win rate by grade
    wins_by_grade = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    count_by_grade = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for r in graded_results:
        count_by_grade[r['grade']] += 1
        if r['pnl'] > 0:
            wins_by_grade[r['grade']] += 1

    winrate_by_grade = {}
    for g in ["A", "B", "C", "D", "F"]:
        if count_by_grade[g] > 0:
            winrate_by_grade[g] = round(wins_by_grade[g] / count_by_grade[g] * 100, 1)
        else:
            winrate_by_grade[g] = 0

    results = {
        "min_grade_filter": min_grade,
        "original": {
            "trades": orig_trades,
            "wins": orig_wins,
            "losses": orig_losses,
            "win_rate": round(orig_win_rate, 1),
            "total_pnl": round(orig_pnl, 2)
        },
        "filtered": {
            "trades": filt_trades,
            "wins": filt_wins,
            "losses": filt_losses,
            "win_rate": round(filt_win_rate, 1),
            "total_pnl": round(filt_pnl, 2)
        },
        "rejected": {
            "trades": rej_trades,
            "wins": rej_wins,
            "total_pnl": round(rej_pnl, 2)
        },
        "improvement": {
            "win_rate_change": round(filt_win_rate - orig_win_rate, 1),
            "pnl_change": round(filt_pnl - orig_pnl, 2),
            "trades_filtered_out": rej_trades,
            "pnl_saved": round(-rej_pnl, 2) if rej_pnl < 0 else 0
        },
        "grade_distribution": grade_dist,
        "pnl_by_grade": {k: round(v, 2) for k, v in pnl_by_grade.items()},
        "winrate_by_grade": winrate_by_grade
    }

    return results


def print_results(results: Dict):
    """Print formatted backtest results"""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS - Gap Grader Filter")
    print("=" * 60)

    print(f"\nFilter: Only take grades {results['min_grade_filter']} or better\n")

    print("ORIGINAL PERFORMANCE:")
    orig = results['original']
    print(f"  Trades: {orig['trades']}")
    print(f"  Win Rate: {orig['win_rate']}%")
    print(f"  Total PnL: ${orig['total_pnl']:.2f}")

    print("\nFILTERED PERFORMANCE:")
    filt = results['filtered']
    print(f"  Trades: {filt['trades']}")
    print(f"  Win Rate: {filt['win_rate']}%")
    print(f"  Total PnL: ${filt['total_pnl']:.2f}")

    print("\nREJECTED TRADES:")
    rej = results['rejected']
    print(f"  Trades: {rej['trades']}")
    print(f"  PnL Lost: ${rej['total_pnl']:.2f}")

    print("\nIMPROVEMENT:")
    imp = results['improvement']
    wr_symbol = "+" if imp['win_rate_change'] >= 0 else ""
    pnl_symbol = "+" if imp['pnl_change'] >= 0 else ""
    print(f"  Win Rate: {wr_symbol}{imp['win_rate_change']}%")
    print(f"  PnL Change: {pnl_symbol}${imp['pnl_change']:.2f}")
    print(f"  PnL Saved by Filtering: ${imp['pnl_saved']:.2f}")

    print("\nGRADE DISTRIBUTION:")
    for g in ["A", "B", "C", "D", "F"]:
        count = results['grade_distribution'][g]
        pnl = results['pnl_by_grade'][g]
        wr = results['winrate_by_grade'][g]
        print(f"  {g}: {count:3d} trades | {wr:5.1f}% WR | ${pnl:>8.2f} PnL")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest Gap Grader filter")
    parser.add_argument("--min-grade", "-g", default="C",
                        help="Minimum grade to accept (A, B, C, D, F)")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output as JSON")

    args = parser.parse_args()

    print("Running Gap Grader Backtest...")
    results = run_backtest(min_grade=args.min_grade)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results)
