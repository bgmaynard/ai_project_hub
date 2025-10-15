"""
Train LSTM models on real stock data from Yahoo Finance
Save as: train_real_stocks.py
Run with: python train_real_stocks.py

IMPORTANT: Yahoo Finance limits:
- 5-minute data: Last 60 days only
- 1-hour data: Last 2 years
- Daily data: Last 5+ years
"""
from lstm_training_pipeline import LSTMTrainingPipeline
import sys

print("\n" + "=" * 60)
print("LSTM TRAINING ON REAL MARKET DATA")
print("=" * 60)

# Initialize pipeline
pipeline = LSTMTrainingPipeline(output_dir="models/lstm_pipeline")

# Choose symbols to train on
# Start with just 2-3 symbols for first run (takes ~5 min per symbol)
symbols = ['AAPL', 'TSLA']  # Add more later: 'NVDA', 'AMD', 'SPY', 'QQQ'

print(f"\nTraining models for: {', '.join(symbols)}")
print("⏱ Estimated time: ~10-15 minutes")
print("This will:")
print("  1. Download 60 days of 5-minute data from Yahoo Finance")
print("     (Yahoo limit: 5-min data available for last 60 days only)")
print("  2. Train LSTM neural network on each symbol")
print("  3. Backtest the model on historical data")
print("  4. Generate evaluation plots and metrics")
print("  5. Save trained models to disk")
print("\nStarting in 3 seconds...\n")

import time
time.sleep(3)

try:
    # Run full pipeline
    # Yahoo Finance limits: 5m data = 60 days max, 1h data = 2 years max
    results = pipeline.run_full_pipeline(
        symbols=symbols,
        data_source='yahoo',
        period='2y',       # 60 days (MAXIMUM for 5-minute data)
        interval='1h'       # 5-minute bars (day trading timeframe)
    )
    
    # Print results summary
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE - RESULTS SUMMARY")
    print("=" * 60)
    
    for symbol, result in results.items():
        print(f"\n📊 {symbol}:")
        print(f"  Training Accuracy:   {result['training']['val_accuracy']:.1%}")
        print(f"  Training AUC:        {result['training']['val_auc']:.3f}")
        print(f"  Backtest Return:     {result['backtest']['total_return']:+.2%}")
        print(f"  Sharpe Ratio:        {result['backtest']['sharpe_ratio']:.2f}")
        print(f"  Win Rate:            {result['backtest']['win_rate']:.1%}")
        print(f"  Max Drawdown:        {result['backtest']['max_drawdown']:.2%}")
        print(f"  Total Trades:        {result['backtest']['total_trades']}")
        
        # Quality assessment
        acc = result['training']['val_accuracy']
        sharpe = result['backtest']['sharpe_ratio']
        
        if acc > 0.60 and sharpe > 1.5:
            quality = "✅ EXCELLENT - Ready for live trading"
        elif acc > 0.55 and sharpe > 1.0:
            quality = "✓ GOOD - Consider paper trading first"
        else:
            quality = "⚠ NEEDS IMPROVEMENT - More training data or tuning needed"
        
        print(f"  Quality:             {quality}")
    
    print("\n" + "=" * 60)
    print("📁 OUTPUT FILES:")
    print(f"  Models:      models/lstm_pipeline/")
    print(f"  Plots:       models/lstm_pipeline/evaluation_plots.png")
    print(f"  Summary:     models/lstm_pipeline/pipeline_summary.json")
    print("=" * 60)
    
    print("\n✨ Next Steps:")
    print("  1. Review evaluation_plots.png to see model performance")
    print("  2. Run 'python run_bot_with_lstm.py' to test live predictions")
    print("  3. If results are good, integrate with your IBKR bot")
    
    print("\n💡 Tips:")
    print("  - Models with accuracy > 60% and Sharpe > 1.5 are ready for live use")
    print("  - Lower accuracy? Try: symbols with higher volume, longer training period")
    print("  - Retrain models weekly to maintain performance")

except KeyboardInterrupt:
    print("\n\n⚠ Training interrupted by user")
    print("Partial results may be saved in models/lstm_pipeline/")
    sys.exit(0)

except Exception as e:
    print(f"\n✗ Error during training: {e}")
    print("\n🔧 Troubleshooting tips:")
    print("  1. Check internet connection (needs to download Yahoo Finance data)")
    print("  2. Try with fewer symbols first: symbols = ['AAPL']")
    print("  3. Try shorter period: period='3mo' instead of '1y'")
    print("  4. Verify yfinance is installed: pip install --upgrade yfinance")
    print("  5. Check if Yahoo Finance is accessible from your location")
    import traceback
    print("\nFull error details:")
    traceback.print_exc()
    sys.exit(1)
