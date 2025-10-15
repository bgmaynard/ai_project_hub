"""
Train LSTM Models on Real Market Data - FIXED VERSION
Now uses 2 YEARS of HOURLY data for better results
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from lstm_training_pipeline import LSTMTrainingPipeline
import time

def main():
    """Main training function with improved settings"""
    
    print("\n" + "="*60)
    print("LSTM TRAINING ON REAL MARKET DATA - IMPROVED")
    print("="*60)
    
    # Symbols to train
    symbols = ['AAPL', 'TSLA']
    
    print(f"\nTraining models for: {', '.join(symbols)}")
    print("⏱ Estimated time: ~10-15 minutes")
    print("\nThis will:")
    print("  1. Download 2 YEARS of HOURLY data from Yahoo Finance")
    print("     (Much better than 60 days of 5-min data!)")
    print("  2. Train LSTM neural network on each symbol")
    print("  3. Backtest with IMPROVED logic (confidence filtering)")
    print("  4. Generate evaluation plots and metrics")
    print("  5. Save trained models to disk")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    # Initialize pipeline (without model_config - it's handled internally)
    pipeline = LSTMTrainingPipeline(
        output_dir='models/lstm_pipeline'
    )
    
    # Run training with IMPROVED SETTINGS
    results = pipeline.run_full_pipeline(
        symbols=symbols,
        data_source='yahoo',
        period='2y',       # ← 2 YEARS (not 60 days!)
        interval='1h',     # ← HOURLY (not 5-min!)
    )
    
    # Print summary
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE - RESULTS SUMMARY")
    print("="*60)
    
    for symbol, result in results.items():
        quality = "✓ GOOD" if result['validation_accuracy'] > 0.6 else "⚠ NEEDS IMPROVEMENT"
        
        print(f"\n📊 {symbol}:")
        print(f"  Training Accuracy:   {result['validation_accuracy']*100:.1f}%")
        print(f"  Training AUC:        {result['validation_auc']:.3f}")
        print(f"  Backtest Return:     {result['backtest_return']*100:.2f}%")
        print(f"  Sharpe Ratio:        {result['sharpe_ratio']:.2f}")
        print(f"  Win Rate:            {result['win_rate']*100:.1f}%")
        print(f"  Max Drawdown:        {result['max_drawdown']*100:.2f}%")
        print(f"  Total Trades:        {result['total_trades']}")
        print(f"  Quality:             {quality}")
    
    print("\n" + "="*60)
    print("📁 OUTPUT FILES:")
    print("  Models:      models/lstm_pipeline/")
    print("  Plots:       models/lstm_pipeline/evaluation_plots.png")
    print("  Summary:     models/lstm_pipeline/pipeline_summary.json")
    print("="*60)
    
    print("\n✨ Next Steps:")
    print("  1. Run: python quick_fix_train.py --mode quick")
    print("  2. Should see 20-50 trades (not 2-4!)")
    print("  3. Positive returns expected")
    print("  4. If good, start paper trading")
    
    print("\n💡 Tips:")
    print("  - Models with accuracy > 60% and Sharpe > 1.5 are ready")
    print("  - Lower accuracy? Try ensemble mode or add more symbols")
    print("  - Retrain models weekly to maintain performance")


if __name__ == "__main__":
    main()
