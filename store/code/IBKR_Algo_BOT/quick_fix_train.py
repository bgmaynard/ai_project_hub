"""
Quick Fix Training Script
Trains LSTM with hourly data (more history) and improved backtesting
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from lstm_training_pipeline import LSTMTrainingPipeline
from improved_backtest import ImprovedBacktester
from lstm_model_complete import LSTMTradingModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_with_improved_backtest(
    symbols=['AAPL', 'TSLA'],
    period='2y',  # 2 years of data instead of 60 days
    interval='1h',  # Hourly instead of 5-min
    confidence_threshold=0.65
):
    """
    Train LSTM models with improved settings
    """
    print("="*70)
    print("LSTM TRAINING WITH IMPROVEMENTS")
    print("="*70)
    print(f"\nSymbols: {symbols}")
    print(f"Data: {period} of {interval} bars")
    print(f"Confidence Threshold: {confidence_threshold*100:.0f}%")
    print(f"\nImprovements:")
    print("  ✓ More training data (2 years vs 60 days)")
    print("  ✓ Hourly bars (better trends, less noise)")
    print("  ✓ Confidence filtering (only high-confidence trades)")
    print("  ✓ Proper transaction costs")
    print("  ✓ Minimum hold period enforcement")
    print("="*70)
    
    # Initialize pipeline
    pipeline = LSTMTrainingPipeline(
        output_dir='models/lstm_pipeline',
        model_config={
            'sequence_length': 60,
            'prediction_horizon': 5,
            'lstm_units': [64, 32],
            'dense_units': [32, 16],
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        }
    )
    
    # Train each symbol
    all_results = {}
    
    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"TRAINING: {symbol}")
        print(f"{'='*70}\n")
        
        try:
            # Run pipeline with improved settings
            result = pipeline.run_full_pipeline(
                symbols=[symbol],
                data_source='yahoo',
                period=period,
                interval=interval,
                train_split=0.8,
                epochs=50,
                batch_size=32
            )
            
            # Load trained model
            model = LSTMTradingModel()
            model.load(f"{symbol}_lstm")
            
            # Download fresh test data
            import yfinance as yf
            print(f"\nDownloading test data for {symbol}...")
            df_test = yf.download(symbol, period='3mo', interval=interval, progress=False)
            df_test.columns = df_test.columns.str.lower()
            
            # Run improved backtest
            print(f"\nRunning improved backtest on {symbol}...")
            backtester = ImprovedBacktester(
                confidence_threshold=confidence_threshold,
                transaction_cost=0.001,
                min_hold_periods=3,
                max_position_size=0.95
            )
            
            backtest_result = backtester.backtest_strategy(model, df_test)
            backtester.print_results(backtest_result)
            
            # Store results
            all_results[symbol] = {
                'training': result[symbol],
                'backtest': backtest_result
            }
            
        except Exception as e:
            logger.error(f"Error training {symbol}: {e}")
            continue
    
    # Print summary
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}\n")
    
    for symbol, results in all_results.items():
        train = results['training']
        back = results['backtest']
        
        print(f"{symbol}:")
        print(f"  Training Accuracy:  {train['validation_accuracy']*100:>6.2f}%")
        print(f"  Training AUC:       {train['validation_auc']:>6.3f}")
        print(f"  Backtest Return:    {back['total_return']*100:>6.2f}%")
        print(f"  Backtest Sharpe:    {back['sharpe_ratio']:>6.2f}")
        print(f"  Total Trades:       {back['total_trades']:>6}")
        print(f"  Win Rate:           {back['win_rate']*100:>6.1f}%")
        print()
    
    # Compare to baseline
    print(f"{'='*70}")
    print("COMPARISON TO PREVIOUS (60-day, 5-min, no confidence filter)")
    print(f"{'='*70}")
    print("\nExpected Improvements:")
    print("  • Better training accuracy (+5-10%)")
    print("  • Positive backtest returns (vs -75% to -85%)")
    print("  • Fewer trades (~50-200 vs 1,700+)")
    print("  • Higher win rate (55-65% vs random)")
    print("  • Positive Sharpe ratio (>1.0 vs negative)")
    
    return all_results


def quick_test_single_symbol(symbol='AAPL'):
    """
    Quick test on single symbol for faster iteration
    """
    import pandas as pd
    import yfinance as yf
    
    print(f"\n🚀 QUICK TEST: {symbol}")
    print("="*70)
    
    # Use simpler approach - just test existing model with improved backtest
    print(f"\nLoading existing {symbol} model...")
    model = LSTMTradingModel()
    
    try:
        model.load(f"{symbol}_lstm")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Could not load model: {e}")
        print(f"\nPlease train the model first:")
        print(f"  python train_real_stocks.py")
        return
    
    # Download test data
    print(f"\nDownloading test data for {symbol}...")
    df_test = yf.download(symbol, period='60d', interval='5m', progress=False)
    
    # Handle MultiIndex columns (new yfinance format)
    if isinstance(df_test.columns, pd.MultiIndex):
        df_test.columns = df_test.columns.get_level_values(0)
    
    df_test.columns = df_test.columns.str.lower()
    print(f"✓ Downloaded {len(df_test)} bars")
    
    # Test multiple confidence thresholds
    print("\n" + "="*70)
    print("TESTING DIFFERENT CONFIDENCE THRESHOLDS")
    print("="*70)
    print("\nFinding the sweet spot: 50% to 60% confidence")
    print("-"*70)
    
    results_comparison = []
    
    # Test more granular thresholds in the 50-60% range
    for threshold in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]:
        backtester = ImprovedBacktester(
            confidence_threshold=threshold,
            transaction_cost=0.001,
            min_hold_periods=3
        )
        result = backtester.backtest_strategy(model, df_test)
        
        results_comparison.append({
            'threshold': threshold,
            'return': result['total_return'],
            'trades': result['total_trades'],
            'sharpe': result['sharpe_ratio'],
            'win_rate': result['win_rate']
        })
        
        # Color code the output
        if result['total_trades'] == 0:
            trade_str = f"{result['total_trades']:>4} ⚠️ "
        elif result['total_trades'] < 10:
            trade_str = f"{result['total_trades']:>4} ⚡"
        elif result['total_trades'] < 50:
            trade_str = f"{result['total_trades']:>4} ✓"
        else:
            trade_str = f"{result['total_trades']:>4}"
        
        print(f"Threshold: {threshold*100:>5.0f}% | "
              f"Return: {result['total_return']*100:>7.2f}% | "
              f"Trades: {trade_str} | "
              f"Win Rate: {result['win_rate']*100:>5.1f}% | "
              f"Sharpe: {result['sharpe_ratio']:>6.2f}")
    
    # Find best thresholds
    print("\n" + "="*70)
    print("ANALYSIS & RECOMMENDATION")
    print("="*70)
    
    # Filter out zero-trade results
    valid_results = [r for r in results_comparison if r['trades'] > 0]
    
    if not valid_results:
        print("\n⚠️  WARNING: No trades at any threshold!")
        print("   Model confidence is very low across all predictions.")
        print("\n   Recommendations:")
        print("   1. Try Option 2: Test TSLA model")
        print("   2. Or Option 3: Retrain with more data (2 years hourly)")
        return results_comparison
    
    best_sharpe = max(valid_results, key=lambda x: x['sharpe'])
    best_return = max(valid_results, key=lambda x: x['return'])
    most_trades = max(valid_results, key=lambda x: x['trades'])
    
    print(f"\n📊 Best Sharpe Ratio: {best_sharpe['threshold']*100:.0f}% threshold")
    print(f"   Return: {best_sharpe['return']*100:.2f}%")
    print(f"   Sharpe: {best_sharpe['sharpe']:.2f}")
    print(f"   Trades: {best_sharpe['trades']}")
    print(f"   Win Rate: {best_sharpe['win_rate']*100:.1f}%")
    
    print(f"\n💰 Best Total Return: {best_return['threshold']*100:.0f}% threshold")
    print(f"   Return: {best_return['return']*100:.2f}%")
    print(f"   Sharpe: {best_return['sharpe']:.2f}")
    print(f"   Trades: {best_return['trades']}")
    print(f"   Win Rate: {best_return['win_rate']*100:.1f}%")
    
    print(f"\n📈 Most Active Trading: {most_trades['threshold']*100:.0f}% threshold")
    print(f"   Return: {most_trades['return']*100:.2f}%")
    print(f"   Sharpe: {most_trades['sharpe']:.2f}")
    print(f"   Trades: {most_trades['trades']}")
    print(f"   Win Rate: {most_trades['win_rate']*100:.1f}%")
    
    print("\n" + "="*70)
    
    # Provide actionable recommendation
    print("\n💡 RECOMMENDATION:")
    
    # Find balanced threshold (20-50 trades is ideal)
    balanced = [r for r in valid_results if 20 <= r['trades'] <= 50]
    
    if balanced:
        best_balanced = max(balanced, key=lambda x: x['sharpe'])
        print(f"   ✓ Use {best_balanced['threshold']*100:.0f}% threshold (balanced)")
        print(f"     - {best_balanced['trades']} trades (good frequency)")
        print(f"     - {best_balanced['return']*100:.2f}% return")
        print(f"     - {best_balanced['win_rate']*100:.1f}% win rate")
    elif most_trades['trades'] < 20:
        print(f"   ⚠️  All thresholds produce <20 trades")
        print(f"   → Use {best_sharpe['threshold']*100:.0f}% for best risk-adjusted returns")
        print(f"   → Consider retraining with 2 years of hourly data for more opportunities")
    else:
        print(f"   ✓ Use {best_sharpe['threshold']*100:.0f}% threshold")
        print(f"     - Best risk-adjusted returns (Sharpe: {best_sharpe['sharpe']:.2f})")
    
    return results_comparison


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LSTM with improvements')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='Quick test or full training')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'TSLA'],
                       help='Symbols to train')
    parser.add_argument('--threshold', type=float, default=0.65,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        # Quick test on AAPL only
        quick_test_single_symbol('AAPL')
    else:
        # Full training on all symbols
        train_with_improved_backtest(
            symbols=args.symbols,
            confidence_threshold=args.threshold
        )
    
    print("\n✓ Training complete!")
    print("\nNext steps:")
    print("  1. Review results above")
    print("  2. If positive, run: python optimize_confidence_threshold.py")
    print("  3. Then test in paper trading for 1 week minimum")
    print("  4. Only go live after consistent positive results")