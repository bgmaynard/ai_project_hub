"""
Before/After MTF Comparison Tool
=================================

This script compares your results before and after implementing MTF features.
Run this after training to see the improvement clearly.

Usage:
    python compare_before_after_mtf.py
"""

import json
from pathlib import Path


def load_results():
    """Load training results from both old and new runs"""
    
    results = {}
    
    # Old results (if they exist)
    old_path = Path('models/lstm_pipeline/pipeline_summary.json')
    if old_path.exists():
        with open(old_path, 'r') as f:
            results['old'] = json.load(f)
    
    # New MTF results
    new_path = Path('models/lstm_pipeline/mtf_training_summary.json')
    if new_path.exists():
        with open(new_path, 'r') as f:
            results['new'] = json.load(f)
    
    return results


def print_comparison(results):
    """Print side-by-side comparison"""
    
    print("\n" + "="*90)
    print("BEFORE vs AFTER MTF IMPLEMENTATION")
    print("="*90 + "\n")
    
    if 'old' not in results:
        print("⚠️  No old results found. Train without MTF first for comparison.")
        print("   (This will show how much MTF improves performance)")
    
    if 'new' not in results:
        print("❌ No new MTF results found!")
        print("   Run: python train_real_stocks_MTF_FIXED.py")
        return
    
    # Show new results
    print("📊 CURRENT RESULTS (With MTF)")
    print("-" * 90)
    
    for symbol, data in results['new'].items():
        print(f"\n{symbol}:")
        print(f"  Test Accuracy:    {data['test_accuracy']*100:6.2f}%")
        print(f"  Backtest Return:  {data['total_return']:+7.2f}%")
        print(f"  Win Rate:         {data['win_rate']:6.2f}%")
        print(f"  Number of Trades: {data['num_trades']:6d}")
        print(f"  Sharpe Ratio:     {data['sharpe']:6.2f}")
        print(f"  Features Used:    {data['num_features']:6d}")
        print(f"  Data Bars:        {data['num_bars']:6d}")
        
        # Quality assessment
        if data['total_return'] > 15:
            quality = "🌟 EXCELLENT"
        elif data['total_return'] > 10:
            quality = "✅ VERY GOOD"
        elif data['total_return'] > 5:
            quality = "✓ GOOD"
        elif data['total_return'] > 0:
            quality = "↗ POSITIVE"
        else:
            quality = "⚠️  NEEDS WORK"
        
        print(f"  Status:           {quality}")
    
    # If we have old results, show comparison
    if 'old' in results:
        print("\n" + "="*90)
        print("IMPROVEMENT ANALYSIS")
        print("="*90 + "\n")
        
        for symbol in results['new'].keys():
            if symbol not in results['old']:
                continue
            
            old_data = results['old'][symbol]
            new_data = results['new'][symbol]
            
            # Calculate improvements
            acc_improvement = (new_data['test_accuracy'] - old_data.get('validation_accuracy', 0.5)) * 100
            return_improvement = new_data['total_return'] - old_data.get('backtest_return', 0)
            wr_improvement = new_data['win_rate'] - old_data.get('win_rate', 50)
            sharpe_improvement = new_data['sharpe'] - old_data.get('sharpe_ratio', 0)
            
            print(f"{symbol}:")
            print(f"  Accuracy:         {old_data.get('validation_accuracy', 0)*100:.1f}% → {new_data['test_accuracy']*100:.1f}% ({acc_improvement:+.1f}%)")
            print(f"  Return:           {old_data.get('backtest_return', 0):.1f}% → {new_data['total_return']:.1f}% ({return_improvement:+.1f}%)")
            print(f"  Win Rate:         {old_data.get('win_rate', 50):.1f}% → {new_data['win_rate']:.1f}% ({wr_improvement:+.1f}%)")
            print(f"  Sharpe:           {old_data.get('sharpe_ratio', 0):.2f} → {new_data['sharpe']:.2f} ({sharpe_improvement:+.2f})")
            
            # Overall assessment
            improvements = []
            if acc_improvement > 5:
                improvements.append("accuracy")
            if return_improvement > 10:
                improvements.append("returns")
            if wr_improvement > 5:
                improvements.append("win rate")
            if sharpe_improvement > 0.5:
                improvements.append("risk-adjusted returns")
            
            if improvements:
                print(f"  ✅ Improved: {', '.join(improvements)}")
            else:
                print("  ⚠️  No significant improvement")
            
            print()
    
    # Recommendations
    print("="*90)
    print("RECOMMENDATIONS")
    print("="*90 + "\n")
    
    any_good = False
    for symbol, data in results['new'].items():
        if data['total_return'] > 10 and data['win_rate'] > 60:
            any_good = True
            break
    
    if any_good:
        print("🎉 CONGRATULATIONS! Your MTF implementation is working well!")
        print("\nNext steps:")
        print("  1. ✅ MTF features are improving performance")
        print("  2. Consider adding MTF gate filtering for even better results")
        print("  3. Try ensemble approach (train 3 models, combine predictions)")
        print("  4. Ready for paper trading validation")
    else:
        print("📊 Results are mixed. Consider these adjustments:")
        print("\nIf accuracy < 60%:")
        print("  - Increase training epochs (50 → 100)")
        print("  - Try different LSTM architecture (128→256 units)")
        print("  - Add more historical data (2y → 3y)")
        print("\nIf too few trades:")
        print("  - Lower confidence threshold (0.58 → 0.55)")
        print("  - Reduce min_hold_bars (3 → 2)")
        print("\nIf too many losing trades:")
        print("  - Raise confidence threshold (0.58 → 0.62)")
        print("  - Add MTF gate filtering (require strong_alignment = 1)")
    
    print("\n" + "="*90 + "\n")


def show_feature_comparison():
    """Show what features are being used"""
    
    print("="*90)
    print("FEATURE COMPARISON")
    print("="*90 + "\n")
    
    print("WITHOUT MTF (19 features):")
    print("  Price & Returns:")
    print("    - returns, log_returns")
    print("    - price_to_sma_5, price_to_sma_10, price_to_sma_20, price_to_sma_50")
    print("  Momentum:")
    print("    - momentum_5, momentum_10")
    print("    - rsi, macd, macd_signal, macd_diff")
    print("  Volume:")
    print("    - volume_ratio")
    print("  Volatility:")
    print("    - volatility_20")
    print("  Moving Averages:")
    print("    - sma_5, sma_10, sma_20, sma_50")
    
    print("\n" + "-"*90 + "\n")
    
    print("WITH MTF (+26 features = 45 total):")
    print("  All 19 base features PLUS:")
    print("\n  4-Hour Timeframe (tf1_*):")
    print("    - tf1_trend, tf1_macd, tf1_macd_signal, tf1_macd_positive")
    print("    - tf1_rsi, tf1_vol_trend")
    print("\n  Daily Timeframe (tf2_*):")
    print("    - tf2_trend, tf2_macd, tf2_macd_signal, tf2_macd_positive")
    print("    - tf2_rsi, tf2_vol_trend")
    print("\n  Weekly Timeframe (tf3_*):")
    print("    - tf3_trend, tf3_macd, tf3_macd_signal, tf3_macd_positive")
    print("    - tf3_rsi, tf3_vol_trend")
    print("\n  Alignment Signals:")
    print("    - strong_alignment (all 3 TFs bullish)")
    print("    - weak_alignment (2 out of 3 bullish)")
    print("    - all_macd_positive (all TFs positive momentum)")
    
    print("\n" + "="*90 + "\n")


def main():
    """Main comparison function"""
    
    print("\n" + "📊"*45)
    print("MTF IMPLEMENTATION - BEFORE/AFTER COMPARISON")
    print("📊"*45 + "\n")
    
    # Load results
    results = load_results()
    
    if not results:
        print("❌ No results found!")
        print("\nTo generate results:")
        print("  1. Run: python train_real_stocks_MTF_FIXED.py")
        print("  2. Wait for training to complete")
        print("  3. Run this script again")
        return
    
    # Show comparison
    print_comparison(results)
    
    # Show feature details
    show_feature_comparison()
    
    # Summary
    if 'new' in results:
        print("📄 Detailed results saved to:")
        print("   models/lstm_pipeline/mtf_training_summary.json")
        print("\n💡 Tip: Open this file to see all metrics in JSON format")
    
    print()


if __name__ == "__main__":
    main()
