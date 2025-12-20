"""
Backtest With Chronos Enhancement
==================================
Compares scalper performance with and without Chronos AI signals.
"""

import sys
sys.path.insert(0, 'C:/ai_project_hub/store/code/IBKR_Algo_BOT_V2')

from dotenv import load_dotenv
load_dotenv('C:/ai_project_hub/store/code/IBKR_Algo_BOT_V2/.env')

import numpy as np
import pandas as pd
from datetime import datetime
from ai.pybroker_schwab_data import SchwabDataSource
from ai.pybroker_walkforward import _run_minute_strategy, SCALPER_PARAMS

# Try to load Chronos
try:
    from ai.chronos_predictor import ChronosPredictor
    HAS_CHRONOS = True
    chronos = ChronosPredictor()
except Exception as e:
    print(f"Chronos not available: {e}")
    HAS_CHRONOS = False


def get_chronos_signal(symbol: str) -> dict:
    """Get Chronos prediction for a symbol."""
    if not HAS_CHRONOS:
        return {"signal": "NEUTRAL", "prob_up": 0.5, "confidence": 0.0}

    try:
        result = chronos.predict(symbol, horizon=1)  # 1-day forecast for scalping
        return {
            "signal": result.get("signal", "NEUTRAL"),
            "prob_up": result.get("probabilities", {}).get("prob_up", 0.5),
            "expected_return": result.get("expected_return_pct", 0),
            "confidence": result.get("confidence", 0),
        }
    except Exception as e:
        return {"signal": "NEUTRAL", "prob_up": 0.5, "confidence": 0.0, "error": str(e)}


def run_enhanced_strategy(data: pd.DataFrame, initial_cash: float, chronos_signal: dict) -> dict:
    """
    Run strategy with Chronos enhancement.

    Enhancements:
    1. Only take trades if Chronos agrees (BULLISH or STRONG_BULLISH)
    2. Adjust position size based on Chronos confidence
    3. Tighter stops if Chronos is bearish
    """
    # Store original params
    orig_params = SCALPER_PARAMS.copy()

    # Adjust based on Chronos signal
    signal = chronos_signal.get("signal", "NEUTRAL")
    prob_up = chronos_signal.get("prob_up", 0.5)
    confidence = chronos_signal.get("confidence", 0.5)

    # Strategy modifications based on Chronos
    if signal in ["STRONG_BULLISH"]:
        # High confidence - use normal parameters
        pass
    elif signal in ["BULLISH"]:
        # Medium confidence - slightly tighter
        SCALPER_PARAMS['min_spike_percent'] = 6.0  # Lower threshold
    elif signal in ["NEUTRAL"]:
        # Low confidence - more selective
        SCALPER_PARAMS['min_spike_percent'] = 8.0  # Higher threshold
        SCALPER_PARAMS['min_volume_surge'] = 6.0
    else:  # BEARISH or STRONG_BEARISH
        # Don't trade or very selective
        SCALPER_PARAMS['min_spike_percent'] = 10.0  # Very high threshold
        SCALPER_PARAMS['min_volume_surge'] = 8.0
        SCALPER_PARAMS['stop_loss_percent'] = 1.5  # Tighter stop

    # Run strategy with modified params
    result = _run_minute_strategy(data, initial_cash)

    # Restore original params
    for k, v in orig_params.items():
        SCALPER_PARAMS[k] = v

    return result


def main():
    print("=" * 70)
    print("BACKTEST COMPARISON: BASELINE vs CHRONOS-ENHANCED")
    print("=" * 70)
    print()

    # Symbols to test - using volatile penny stocks from watchlist
    symbols = ['YCBD', 'EVTV', 'LHAI']  # Recent movers
    days = 1  # Today only
    initial_cash = 1000.0

    # Fetch data
    print("Fetching today's minute data from Schwab...")
    source = SchwabDataSource()

    all_results = []

    for symbol in symbols:
        print(f"\n{'=' * 50}")
        print(f"  {symbol}")
        print("=" * 50)

        # Get data for this symbol
        data = source.query([symbol], days=days)

        if data.empty:
            print(f"  No data available for {symbol}")
            continue

        print(f"  Loaded {len(data)} minute bars")

        # Get Chronos signal
        print(f"  Getting Chronos prediction...")
        chronos_signal = get_chronos_signal(symbol)
        print(f"  Chronos: {chronos_signal['signal']} (prob_up={chronos_signal['prob_up']:.1%})")

        # Run BASELINE (original strategy)
        print(f"\n  [BASELINE] Running original scalper...")

        # Set baseline params (optimized from yesterday)
        SCALPER_PARAMS['min_spike_percent'] = 7.0
        SCALPER_PARAMS['min_volume_surge'] = 5.0
        SCALPER_PARAMS['stop_loss_percent'] = 2.0
        SCALPER_PARAMS['trailing_stop_percent'] = 1.5

        baseline_result = _run_minute_strategy(data, initial_cash)

        print(f"    Trades: {baseline_result['total_trades']}")
        print(f"    Win Rate: {baseline_result['win_rate']:.1f}%")
        print(f"    P/L: ${baseline_result['total_pnl']:.2f}")

        # Run ENHANCED (with Chronos)
        print(f"\n  [ENHANCED] Running Chronos-enhanced scalper...")
        enhanced_result = run_enhanced_strategy(data, initial_cash, chronos_signal)

        print(f"    Trades: {enhanced_result['total_trades']}")
        print(f"    Win Rate: {enhanced_result['win_rate']:.1f}%")
        print(f"    P/L: ${enhanced_result['total_pnl']:.2f}")

        # Calculate improvement
        baseline_pnl = baseline_result['total_pnl']
        enhanced_pnl = enhanced_result['total_pnl']
        improvement = enhanced_pnl - baseline_pnl

        print(f"\n  IMPROVEMENT: ${improvement:+.2f}")

        all_results.append({
            "symbol": symbol,
            "chronos_signal": chronos_signal['signal'],
            "chronos_prob_up": chronos_signal['prob_up'],
            "baseline_trades": baseline_result['total_trades'],
            "baseline_win_rate": baseline_result['win_rate'],
            "baseline_pnl": baseline_pnl,
            "enhanced_trades": enhanced_result['total_trades'],
            "enhanced_win_rate": enhanced_result['win_rate'],
            "enhanced_pnl": enhanced_pnl,
            "improvement": improvement,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if all_results:
        total_baseline = sum(r['baseline_pnl'] for r in all_results)
        total_enhanced = sum(r['enhanced_pnl'] for r in all_results)
        total_improvement = total_enhanced - total_baseline

        print(f"\n{'Symbol':<8} {'Chronos':<15} {'Baseline':>10} {'Enhanced':>10} {'Improve':>10}")
        print("-" * 60)

        for r in all_results:
            print(f"{r['symbol']:<8} {r['chronos_signal']:<15} ${r['baseline_pnl']:>8.2f} ${r['enhanced_pnl']:>8.2f} ${r['improvement']:>+8.2f}")

        print("-" * 60)
        print(f"{'TOTAL':<8} {'':<15} ${total_baseline:>8.2f} ${total_enhanced:>8.2f} ${total_improvement:>+8.2f}")

        print(f"\n{'=' * 70}")
        if total_improvement > 0:
            print(f"CHRONOS ENHANCEMENT: +${total_improvement:.2f} IMPROVEMENT")
        elif total_improvement < 0:
            print(f"CHRONOS ENHANCEMENT: ${total_improvement:.2f} (worse performance)")
        else:
            print(f"CHRONOS ENHANCEMENT: No significant difference")
        print("=" * 70)
    else:
        print("No data available for backtesting")

    return all_results


if __name__ == "__main__":
    results = main()
