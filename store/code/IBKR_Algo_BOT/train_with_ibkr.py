"""
Train LSTM Models with IBKR Historical Data
Uses high-quality data fetched from Interactive Brokers
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import logging
from lstm_training_pipeline import LSTMTrainingPipeline
from improved_backtest import ImprovedBacktester

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_with_ibkr_data(
    symbols=['AAPL', 'TSLA'],
    data_dir='data/historical',
    bar_size='1_hour'
):
    """
    Train LSTM models using IBKR historical data
    
    Args:
        symbols: List of symbols to train
        data_dir: Directory with CSV files
        bar_size: Bar size string (matches filename)
    """
    print("\n" + "="*70)
    print("LSTM TRAINING WITH IBKR DATA")
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
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"TRAINING: {symbol}")
        print(f"{'='*70}\n")
        
        # Load IBKR data
        csv_file = f"{data_dir}/{symbol}_{bar_size}.csv"
        
        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            print(f"✓ Loaded {len(df)} bars from {csv_file}")
            print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
            
            # Ensure correct column names
            df.columns = df.columns.str.lower()
            
        except FileNotFoundError:
            print(f"✗ File not found: {csv_file}")
            print(f"  Run: python ibkr_historical_fetcher.py --symbols {symbol}")
            continue
        
        # Train model
        print("\nTraining LSTM model...")
        
        model = pipeline.train_model(
            df=df,
            symbol=symbol,
            epochs=50,
            batch_size=32,
            validation_split=0.2
        )
        
        # Evaluate
        print("\nEvaluating model...")
        
        # Run backtest with improved logic
        backtester = ImprovedBacktester(
            confidence_threshold=0.55,
            transaction_cost=0.001,
            min_hold_periods=3
        )
        
        # Use last 20% for testing
        test_size = int(len(df) * 0.2)
        df_test = df.iloc[-test_size:]
        
        backtest_result = backtester.backtest_strategy(model, df_test)
        backtester.print_results(backtest_result)
        
        all_results[symbol] = {
            'training_bars': len(df),
            'test_bars': len(df_test),
            'backtest': backtest_result
        }
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    for symbol, results in all_results.items():
        back = results['backtest']
        print(f"\n{symbol}:")
        print(f"  Training Bars: {results['training_bars']}")
        print(f"  Test Return:   {back['total_return']*100:>7.2f}%")
        print(f"  Sharpe Ratio:  {back['sharpe_ratio']:>7.2f}")
        print(f"  Win Rate:      {back['win_rate']*100:>7.1f}%")
        print(f"  Total Trades:  {back['total_trades']:>7}")
    
    print("\n" + "="*70)
    print("✓ Training complete! Models saved.")
    print("\nNext step:")
    print("  python quick_fix_train.py --mode quick")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train with IBKR data')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'TSLA'],
                       help='Symbols to train')
    parser.add_argument('--data-dir', default='data/historical',
                       help='Directory with CSV files')
    parser.add_argument('--bar-size', default='1_hour',
                       help='Bar size (matches filename)')
    
    args = parser.parse_args()
    
    train_with_ibkr_data(
        symbols=args.symbols,
        data_dir=args.data_dir,
        bar_size=args.bar_size
    )
