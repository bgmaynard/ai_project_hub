"""
Backtest Analysis Script
========================
Runs systematic backtests with different parameters to find optimal settings.
"""

import sys
sys.path.insert(0, 'C:/ai_project_hub/store/code/IBKR_Algo_BOT_V2')

from dotenv import load_dotenv
load_dotenv('C:/ai_project_hub/store/code/IBKR_Algo_BOT_V2/.env')

import json
from ai.pybroker_walkforward import _run_minute_strategy, SCALPER_PARAMS
from ai.pybroker_schwab_data import SchwabDataSource
import pandas as pd

def run_test(symbols, days, params):
    """Run backtest with specific parameters"""
    global SCALPER_PARAMS

    # Update params
    for k, v in params.items():
        SCALPER_PARAMS[k] = v

    # Fetch data
    source = SchwabDataSource()
    data = source.query(symbols, days=days)

    if data.empty:
        return None

    # Run strategy
    result = _run_minute_strategy(data, 1000.0)
    return result

def main():
    print("=" * 70)
    print("PARAMETER OPTIMIZATION BACKTEST")
    print("=" * 70)

    # Symbols to test
    symbols = ['YCBD']
    days = 5

    # Fetch data once
    print(f"\nFetching {days} days of minute data for {symbols}...")
    source = SchwabDataSource()
    data = source.query(symbols, days=days)

    if data.empty:
        print("No data available!")
        return

    print(f"Loaded {len(data)} bars")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")

    # Parameter variations to test
    test_configs = [
        {"name": "Current (7%/5x)", "min_spike_percent": 7.0, "min_volume_surge": 5.0, "stop_loss_percent": 3.0, "trailing_stop_percent": 2.0},
        {"name": "Lower spike (5%/5x)", "min_spike_percent": 5.0, "min_volume_surge": 5.0, "stop_loss_percent": 3.0, "trailing_stop_percent": 2.0},
        {"name": "Lower spike (6%/5x)", "min_spike_percent": 6.0, "min_volume_surge": 5.0, "stop_loss_percent": 3.0, "trailing_stop_percent": 2.0},
        {"name": "Lower volume (7%/3x)", "min_spike_percent": 7.0, "min_volume_surge": 3.0, "stop_loss_percent": 3.0, "trailing_stop_percent": 2.0},
        {"name": "Tighter stop (7%/5x/2%)", "min_spike_percent": 7.0, "min_volume_surge": 5.0, "stop_loss_percent": 2.0, "trailing_stop_percent": 1.5},
        {"name": "Wider trail (7%/5x/2.5%)", "min_spike_percent": 7.0, "min_volume_surge": 5.0, "stop_loss_percent": 3.0, "trailing_stop_percent": 2.5},
        {"name": "Balanced (6%/4x)", "min_spike_percent": 6.0, "min_volume_surge": 4.0, "stop_loss_percent": 3.0, "trailing_stop_percent": 2.0},
        {"name": "Aggressive (5%/3x)", "min_spike_percent": 5.0, "min_volume_surge": 3.0, "stop_loss_percent": 3.0, "trailing_stop_percent": 2.0},
        {"name": "Conservative (8%/6x)", "min_spike_percent": 8.0, "min_volume_surge": 6.0, "stop_loss_percent": 3.0, "trailing_stop_percent": 2.0},
    ]

    results = []

    print("\n" + "=" * 70)
    print("TESTING PARAMETER VARIATIONS")
    print("=" * 70)
    print(f"{'Config':<25} {'Trades':>7} {'Wins':>6} {'Win%':>7} {'P/L':>10} {'Return':>8}")
    print("-" * 70)

    for config in test_configs:
        name = config.pop("name")

        # Update global params
        for k, v in config.items():
            SCALPER_PARAMS[k] = v

        # Run backtest
        result = _run_minute_strategy(data, 1000.0)

        trades = result['total_trades']
        wins = result['wins']
        win_rate = result['win_rate']
        pnl = result['total_pnl']
        ret = result['return_pct']

        print(f"{name:<25} {trades:>7} {wins:>6} {win_rate:>6.1f}% ${pnl:>8.2f} {ret:>7.2f}%")

        results.append({
            "config": name,
            "params": config.copy(),
            "trades": trades,
            "wins": wins,
            "losses": result['losses'],
            "win_rate": round(win_rate, 1),
            "pnl": round(pnl, 2),
            "return_pct": round(ret, 2),
            "profit_factor": round(wins / max(1, result['losses']), 2)
        })

    print("=" * 70)

    # Find best configs
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Best by P/L
    best_pnl = max(results, key=lambda x: x['pnl'])
    print(f"\nBest P/L: {best_pnl['config']}")
    print(f"  P/L: ${best_pnl['pnl']}, Win Rate: {best_pnl['win_rate']}%, Trades: {best_pnl['trades']}")

    # Best by Win Rate (min 5 trades)
    filtered = [r for r in results if r['trades'] >= 5]
    if filtered:
        best_wr = max(filtered, key=lambda x: x['win_rate'])
        print(f"\nBest Win Rate (min 5 trades): {best_wr['config']}")
        print(f"  Win Rate: {best_wr['win_rate']}%, P/L: ${best_wr['pnl']}, Trades: {best_wr['trades']}")

    # Best risk-adjusted (P/L per trade)
    trades_filtered = [r for r in results if r['trades'] > 0]
    if trades_filtered:
        for r in trades_filtered:
            r['pnl_per_trade'] = r['pnl'] / r['trades']
        best_risk = max(trades_filtered, key=lambda x: x['pnl_per_trade'])
        print(f"\nBest P/L per Trade: {best_risk['config']}")
        print(f"  P/L/Trade: ${best_risk['pnl_per_trade']:.2f}, Total P/L: ${best_risk['pnl']}, Trades: {best_risk['trades']}")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Score each config (balance of win rate, P/L, and trade frequency)
    for r in results:
        if r['trades'] > 0:
            # Normalize scores
            pnl_score = (r['pnl'] + 50) / 100  # Shift to positive
            wr_score = r['win_rate'] / 100
            trade_score = min(r['trades'] / 20, 1.0)  # Cap at 20 trades
            r['score'] = pnl_score * 0.5 + wr_score * 0.3 + trade_score * 0.2
        else:
            r['score'] = 0

    best = max(results, key=lambda x: x['score'])
    print(f"\nRecommended Config: {best['config']}")
    print(f"  Parameters:")
    for k, v in best['params'].items():
        print(f"    {k}: {v}")
    print(f"  Expected: {best['trades']} trades, {best['win_rate']}% win rate, ${best['pnl']} P/L")

    # Save results
    with open('C:/ai_project_hub/store/code/IBKR_Algo_BOT_V2/ai/backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to ai/backtest_results.json")

    return best

if __name__ == "__main__":
    best = main()
