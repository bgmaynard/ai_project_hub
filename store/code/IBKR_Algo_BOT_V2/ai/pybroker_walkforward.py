"""
PyBroker Walkforward Analysis for Scalper Strategy
===================================================
Tests our momentum scalping strategy using walkforward analysis
to validate performance on out-of-sample data.

Supports both YFinance (daily data) and Schwab (minute data) sources.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd

# PyBroker imports
from pybroker import Strategy, StrategyConfig, YFinance

logger = logging.getLogger(__name__)

# Strategy parameters (matching our scalper config)
SCALPER_PARAMS = {
    "min_spike_percent": 5.0,  # Entry threshold (updated Dec 19)
    "profit_target_percent": 3.0,
    "stop_loss_percent": 3.0,
    "trailing_stop_percent": 1.5,  # Trailing after target hit
    "min_volume_surge": 3.0,
    "use_atr_stops": True,  # Use ATR-based dynamic stops
    "atr_multiplier": 1.5,  # Stop at 1.5x ATR
}


def momentum_scalp_strategy(ctx):
    """
    Momentum scalping strategy for PyBroker.

    Entry: Price spikes X% with volume surge
    Exit: Stop loss, profit target, or trailing stop
    """
    # Get current bar data
    close = ctx.close[-1]
    volume = ctx.volume[-1]

    # Need at least 5 bars for momentum calculation
    if len(ctx.close) < 5:
        return

    # Calculate momentum (5-bar price change)
    price_5_ago = ctx.close[-5]
    momentum = ((close - price_5_ago) / price_5_ago * 100) if price_5_ago > 0 else 0

    # Calculate volume surge (vs 20-bar average)
    if len(ctx.volume) >= 20:
        avg_volume = np.mean(ctx.volume[-20:])
        volume_surge = volume / avg_volume if avg_volume > 0 else 0
    else:
        volume_surge = 1.0

    # Current position
    pos = ctx.long_pos()

    if pos is None:
        # === ENTRY LOGIC ===
        # Check for momentum spike with volume confirmation
        if momentum >= SCALPER_PARAMS["min_spike_percent"]:
            if volume_surge >= SCALPER_PARAMS["min_volume_surge"]:
                # Calculate position size (risk 1% of portfolio)
                risk_per_trade = ctx.portfolio_value * 0.01
                stop_price = close * (1 - SCALPER_PARAMS["stop_loss_percent"] / 100)
                risk_per_share = close - stop_price

                if risk_per_share > 0:
                    shares = int(risk_per_trade / risk_per_share)
                    shares = min(shares, int(ctx.portfolio_value * 0.2 / close))  # Max 20% position

                    if shares > 0:
                        ctx.buy_shares = shares
                        ctx.stop_loss_pct = SCALPER_PARAMS["stop_loss_percent"]
                        ctx.hold_bars = 0  # Track hold time
                        ctx.entry_price = close
                        ctx.high_since_entry = close
    else:
        # === EXIT LOGIC ===
        entry_price = getattr(ctx, 'entry_price', pos.entry_price)
        high_since = getattr(ctx, 'high_since_entry', close)

        # Update high since entry
        if close > high_since:
            ctx.high_since_entry = close
            high_since = close

        # Calculate P/L
        pnl_pct = ((close - entry_price) / entry_price * 100)
        max_gain = ((high_since - entry_price) / entry_price * 100)

        should_exit = False
        exit_reason = ""

        # Stop loss
        if pnl_pct <= -SCALPER_PARAMS["stop_loss_percent"]:
            should_exit = True
            exit_reason = "STOP_LOSS"

        # Trailing stop (after hitting profit target)
        elif max_gain >= SCALPER_PARAMS["profit_target_percent"]:
            trailing_trigger = max_gain - SCALPER_PARAMS["trailing_stop_percent"]
            if pnl_pct <= trailing_trigger:
                should_exit = True
                exit_reason = "TRAILING_STOP"

        # Max hold time (10 bars ~ 10 minutes on 1-min data)
        hold_bars = getattr(ctx, 'hold_bars', 0) + 1
        ctx.hold_bars = hold_bars

        if hold_bars >= 10 and pnl_pct <= 1.0:
            should_exit = True
            exit_reason = "MAX_HOLD"

        if should_exit:
            ctx.sell_all_shares()


def run_walkforward_test(
    symbols: list = None,
    start_date: str = None,
    end_date: str = None,
    train_size: int = 30,  # Training window (days)
    test_size: int = 5,    # Test window (days)
    initial_cash: float = 1000.0
):
    """
    Run walkforward analysis on the scalper strategy.

    Args:
        symbols: List of symbols to test
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        train_size: Days in each training window
        test_size: Days in each test window
        initial_cash: Starting capital

    Returns:
        Dict with backtest results
    """
    if symbols is None:
        symbols = ["YCBD", "ADTX"]  # Default test symbols

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print("PyBroker Walkforward Analysis - Scalper Strategy")
    print(f"{'='*60}")
    print(f"Symbols: {symbols}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Train Window: {train_size} days | Test Window: {test_size} days")
    print(f"Initial Cash: ${initial_cash:,.2f}")
    print(f"{'='*60}\n")

    # Configure strategy
    config = StrategyConfig(
        initial_cash=initial_cash,
        fee_mode='order_percent',
        fee_amount=0.001,  # 0.1% commission estimate
    )

    # Create strategy
    strategy = Strategy(
        YFinance(),
        start_date=start_date,
        end_date=end_date,
        config=config
    )

    # Add symbols
    strategy.add_execution(
        momentum_scalp_strategy,
        symbols
    )

    # Run walkforward analysis
    print("Running walkforward analysis...")
    print("(This tests strategy on out-of-sample data to detect overfitting)\n")

    try:
        result = strategy.walkforward(
            windows=3,  # Number of time windows
            train_size=0.6,  # 60% train, 40% test per window
            lookahead=1,
            warmup=10
        )

        # Extract results
        print("\n" + "="*60)
        print("WALKFORWARD RESULTS")
        print("="*60)

        if result is not None and hasattr(result, 'metrics'):
            m = result.metrics
            print(f"\nPerformance Metrics:")
            print(f"  Total Return: {getattr(m, 'total_return_pct', 0):.2f}%")
            print(f"  Sharpe Ratio: {getattr(m, 'sharpe', 0):.2f}")
            print(f"  Max Drawdown: {getattr(m, 'max_drawdown_pct', 0):.2f}%")
            print(f"  Win Rate: {getattr(m, 'win_rate', 0):.1f}%")
            print(f"  Total Trades: {getattr(m, 'total_trades', 0)}")
            print(f"  Profit Factor: {getattr(m, 'profit_factor', 0):.2f}")

            # Extract to dict for return
            metrics_dict = {
                "total_return_pct": getattr(m, 'total_return_pct', 0),
                "sharpe": getattr(m, 'sharpe', 0),
                "max_drawdown_pct": getattr(m, 'max_drawdown_pct', 0),
                "win_rate": getattr(m, 'win_rate', 0),
                "total_trades": getattr(m, 'total_trades', 0),
                "profit_factor": getattr(m, 'profit_factor', 0),
            }

            return {
                "success": True,
                "metrics": metrics_dict,
                "result": result
            }
        else:
            # Try basic backtest if walkforward has issues
            print("Walkforward returned no results, running standard backtest...")
            result = strategy.backtest()

            if result is not None:
                print(f"\nBacktest completed")
                return {
                    "success": True,
                    "result": result,
                    "note": "Standard backtest (walkforward unavailable)"
                }

            return {
                "success": False,
                "error": "No results from walkforward or backtest"
            }

    except Exception as e:
        logger.error(f"Walkforward error: {e}")
        print(f"\nError during walkforward: {e}")

        # Try simple backtest as fallback
        print("\nAttempting simple backtest as fallback...")
        try:
            result = strategy.backtest()
            print("Simple backtest completed")
            return {
                "success": True,
                "result": result,
                "note": "Fallback to simple backtest"
            }
        except Exception as e2:
            return {
                "success": False,
                "error": str(e),
                "fallback_error": str(e2)
            }


def compare_params(base_params: dict, test_variations: list, symbols: list = None):
    """
    Compare different parameter sets using walkforward analysis.

    Args:
        base_params: Base scalper parameters
        test_variations: List of param dicts to test
        symbols: Symbols to test on

    Returns:
        Comparison results
    """
    results = []

    print("\n" + "="*60)
    print("PARAMETER COMPARISON TEST")
    print("="*60)

    for i, params in enumerate([base_params] + test_variations):
        print(f"\nTest {i+1}: {params}")

        # Update global params
        global SCALPER_PARAMS
        SCALPER_PARAMS = params

        # Run test
        result = run_walkforward_test(symbols=symbols)
        results.append({
            "params": params,
            "result": result
        })

    return results


def run_minute_backtest(
    symbols: list = None,
    days: int = 10,
    initial_cash: float = 1000.0,
    use_stored_data: bool = False
) -> dict:
    """
    Run backtest using Schwab minute-bar data.

    This is more appropriate for scalping strategies than daily data.

    Args:
        symbols: List of symbols to test
        days: Days of minute data to use (max 10)
        initial_cash: Starting capital
        use_stored_data: Use data from unified_data_collector instead of live API

    Returns:
        Dict with backtest results
    """
    if symbols is None:
        symbols = ["SPY"]

    print(f"\n{'='*60}")
    print("MINUTE-BAR BACKTEST - Schwab Data")
    print(f"{'='*60}")
    print(f"Symbols: {symbols}")
    print(f"Days: {days}")
    print(f"Initial Cash: ${initial_cash:,.2f}")
    print(f"{'='*60}\n")

    # Fetch data from Schwab
    print("Fetching minute data from Schwab...")

    try:
        if use_stored_data:
            from ai.unified_data_collector import get_data_collector
            collector = get_data_collector()
            all_data = []
            for symbol in symbols:
                bars = collector.get_minute_bars(symbol)
                if bars:
                    df = pd.DataFrame(bars)
                    df['symbol'] = symbol
                    all_data.append(df)
            if all_data:
                data = pd.concat(all_data, ignore_index=True)
            else:
                return {"success": False, "error": "No stored data found"}
        else:
            from ai.pybroker_schwab_data import SchwabDataSource
            source = SchwabDataSource()
            data = source.query(symbols, days=days)

        if data.empty:
            return {"success": False, "error": "No data returned from Schwab"}

        print(f"Loaded {len(data)} minute bars")
        print(f"Date range: {data['date'].min()} to {data['date'].max()}")

        # Run custom backtest on minute data
        results = _run_minute_strategy(data, initial_cash)

        print("\n" + "="*60)
        print("MINUTE BACKTEST RESULTS")
        print("="*60)
        print(f"Total Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['wins']}")
        print(f"Losing Trades: {results['losses']}")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Total P/L: ${results['total_pnl']:.2f}")
        print(f"Final Portfolio: ${results['final_value']:.2f}")
        print(f"Return: {results['return_pct']:.2f}%")

        return {
            "success": True,
            "metrics": results,
            "data_source": "schwab_minute"
        }

    except Exception as e:
        logger.error(f"Minute backtest error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def _run_minute_strategy(data: pd.DataFrame, initial_cash: float) -> dict:
    """
    Execute minute-bar strategy simulation.

    This is a simplified backtester that works with minute data.
    """
    portfolio = initial_cash
    cash = initial_cash
    position = None  # {symbol, shares, entry_price, entry_time, high_since}
    trades = []

    # Group by symbol and process each
    for symbol in data['symbol'].unique():
        symbol_data = data[data['symbol'] == symbol].sort_values('date')

        closes = symbol_data['close'].values
        volumes = symbol_data['volume'].values
        dates = symbol_data['date'].values

        for i in range(20, len(closes)):  # Need 20 bars for volume average
            close = closes[i]
            volume = volumes[i]
            timestamp = dates[i]

            # Calculate indicators
            price_5_ago = closes[i-5] if i >= 5 else close
            momentum = ((close - price_5_ago) / price_5_ago * 100) if price_5_ago > 0 else 0

            avg_volume = np.mean(volumes[i-20:i])
            volume_surge = volume / avg_volume if avg_volume > 0 else 1

            if position is None:
                # === ENTRY LOGIC ===
                if momentum >= SCALPER_PARAMS["min_spike_percent"]:
                    if volume_surge >= SCALPER_PARAMS["min_volume_surge"]:
                        # Calculate position size
                        risk_per_trade = portfolio * 0.01
                        stop_price = close * (1 - SCALPER_PARAMS["stop_loss_percent"] / 100)
                        risk_per_share = close - stop_price

                        if risk_per_share > 0 and close > 0:
                            shares = int(risk_per_trade / risk_per_share)
                            shares = min(shares, int(cash * 0.2 / close))  # Max 20% position

                            if shares > 0:
                                cost = shares * close
                                if cost <= cash:
                                    cash -= cost
                                    position = {
                                        'symbol': symbol,
                                        'shares': shares,
                                        'entry_price': close,
                                        'entry_time': timestamp,
                                        'high_since': close,
                                        'bars_held': 0
                                    }
            else:
                # === EXIT LOGIC ===
                if position['symbol'] == symbol:
                    position['bars_held'] += 1

                    # Update high since entry
                    if close > position['high_since']:
                        position['high_since'] = close

                    entry_price = position['entry_price']
                    high_since = position['high_since']

                    pnl_pct = ((close - entry_price) / entry_price * 100)
                    max_gain = ((high_since - entry_price) / entry_price * 100)

                    should_exit = False
                    exit_reason = ""

                    # Stop loss
                    if pnl_pct <= -SCALPER_PARAMS["stop_loss_percent"]:
                        should_exit = True
                        exit_reason = "STOP_LOSS"

                    # Trailing stop
                    elif max_gain >= SCALPER_PARAMS["profit_target_percent"]:
                        trailing_trigger = max_gain - SCALPER_PARAMS["trailing_stop_percent"]
                        if pnl_pct <= trailing_trigger:
                            should_exit = True
                            exit_reason = "TRAILING_STOP"

                    # Max hold time
                    if position['bars_held'] >= 10 and pnl_pct <= 1.0:
                        should_exit = True
                        exit_reason = "MAX_HOLD"

                    if should_exit:
                        proceeds = position['shares'] * close
                        pnl = proceeds - (position['shares'] * entry_price)
                        cash += proceeds

                        trades.append({
                            'symbol': symbol,
                            'entry_price': entry_price,
                            'exit_price': close,
                            'shares': position['shares'],
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'exit_reason': exit_reason,
                            'bars_held': position['bars_held']
                        })

                        position = None

    # Close any remaining position at last price
    if position:
        last_price = data[data['symbol'] == position['symbol']]['close'].iloc[-1]
        proceeds = position['shares'] * last_price
        pnl = proceeds - (position['shares'] * position['entry_price'])
        cash += proceeds
        trades.append({
            'symbol': position['symbol'],
            'entry_price': position['entry_price'],
            'exit_price': last_price,
            'shares': position['shares'],
            'pnl': pnl,
            'exit_reason': 'END_OF_DATA'
        })

    # Calculate results
    wins = len([t for t in trades if t['pnl'] > 0])
    losses = len([t for t in trades if t['pnl'] <= 0])
    total_pnl = sum(t['pnl'] for t in trades)

    return {
        'total_trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': (wins / len(trades) * 100) if trades else 0,
        'total_pnl': total_pnl,
        'final_value': cash,
        'return_pct': ((cash - initial_cash) / initial_cash * 100),
        'trades': trades
    }


def run_full_analysis(
    symbols: list = None,
    include_minute: bool = True,
    include_daily: bool = True,
    initial_cash: float = 1000.0
) -> dict:
    """
    Run comprehensive analysis using both minute and daily data.

    Args:
        symbols: List of symbols to test
        include_minute: Run minute-bar backtest with Schwab data
        include_daily: Run daily walkforward with YFinance
        initial_cash: Starting capital

    Returns:
        Combined analysis results
    """
    results = {}

    if symbols is None:
        symbols = ["SPY"]

    if include_minute:
        print("\n=== MINUTE DATA ANALYSIS ===")
        results['minute'] = run_minute_backtest(
            symbols=symbols,
            days=10,
            initial_cash=initial_cash
        )

    if include_daily:
        print("\n=== DAILY DATA ANALYSIS ===")
        results['daily'] = run_walkforward_test(
            symbols=symbols,
            initial_cash=initial_cash
        )

    return results


def sync_with_scalper_config():
    """
    Sync SCALPER_PARAMS with the current HFT scalper configuration.

    Call this before running backtests to ensure params match live trading.
    """
    global SCALPER_PARAMS
    try:
        from ai.hft_scalper import get_hft_scalper
        scalper = get_hft_scalper()
        config = scalper.config

        SCALPER_PARAMS.update({
            "min_spike_percent": config.min_spike_percent,
            "profit_target_percent": config.profit_target_percent,
            "stop_loss_percent": config.stop_loss_percent,
            "trailing_stop_percent": config.trailing_stop_percent,
            "min_volume_surge": config.min_volume_surge,
            "use_atr_stops": config.use_atr_stops,
            "atr_multiplier": config.atr_stop_multiplier,
        })
        print(f"Synced params from scalper config: {SCALPER_PARAMS}")
        return True
    except Exception as e:
        print(f"Could not sync with scalper config: {e}")
        return False


def validate_current_params(symbols: list = None, days: int = 30) -> dict:
    """
    Validate current scalper parameters using walkforward analysis.

    This helps detect if parameters are overfit or if they generalize well.

    Args:
        symbols: Symbols to test (defaults to recent movers)
        days: Days of history to test

    Returns:
        Validation report with recommendations
    """
    if symbols is None:
        symbols = ["SOUN", "AAPL", "TSLA", "NVDA"]  # Mix of volatile + stable

    # Sync with current config
    sync_with_scalper_config()

    print("\n" + "=" * 60)
    print("PARAMETER VALIDATION REPORT")
    print("=" * 60)
    print(f"Testing current scalper parameters on {days} days of data")
    print(f"Symbols: {symbols}")
    print(f"Parameters: {SCALPER_PARAMS}")
    print("=" * 60)

    # Run backtest
    result = run_walkforward_test(
        symbols=symbols,
        start_date=(datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
        end_date=datetime.now().strftime("%Y-%m-%d"),
        initial_cash=1000.0
    )

    # Generate recommendations
    recommendations = []

    if result.get('success') and result.get('metrics'):
        metrics = result['metrics']

        win_rate = metrics.get('win_rate', 0)
        total_return = metrics.get('total_return_pct', 0)
        max_dd = abs(metrics.get('max_drawdown_pct', 0))
        trades = metrics.get('total_trades', 0)

        print("\n--- ANALYSIS ---")

        if win_rate < 35:
            recommendations.append(f"WIN RATE LOW ({win_rate:.1f}%): Consider tighter entry filters")
        elif win_rate > 60:
            recommendations.append(f"WIN RATE HIGH ({win_rate:.1f}%): Parameters may be overfit")
        else:
            recommendations.append(f"WIN RATE OK ({win_rate:.1f}%)")

        if max_dd > 20:
            recommendations.append(f"MAX DRAWDOWN HIGH ({max_dd:.1f}%): Reduce position size or tighten stops")
        else:
            recommendations.append(f"MAX DRAWDOWN OK ({max_dd:.1f}%)")

        if trades < 5:
            recommendations.append(f"LOW TRADE COUNT ({trades}): Entry criteria may be too strict")
        elif trades > 50:
            recommendations.append(f"HIGH TRADE COUNT ({trades}): May be overtrading")

        if total_return < 0:
            recommendations.append(f"NEGATIVE RETURN ({total_return:.1f}%): Strategy needs adjustment")
        else:
            recommendations.append(f"POSITIVE RETURN ({total_return:.1f}%)")

    else:
        recommendations.append("VALIDATION FAILED: Check data availability")

    print("\n--- RECOMMENDATIONS ---")
    for rec in recommendations:
        print(f"  - {rec}")

    return {
        "success": result.get('success', False),
        "params_tested": SCALPER_PARAMS.copy(),
        "symbols": symbols,
        "days": days,
        "metrics": result.get('metrics', {}),
        "recommendations": recommendations
    }


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Running Parameter Validation...")

    # Validate current params
    report = validate_current_params(
        symbols=["SOUN", "AAPL", "TSLA"],
        days=30
    )

    print(f"\n\nFinal Report: {report}")
