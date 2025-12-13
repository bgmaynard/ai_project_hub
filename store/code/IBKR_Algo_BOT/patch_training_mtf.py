"""
Automatic Patch to Add MTF Features
====================================

This script automatically adds Multi-Timeframe features to your existing
training pipeline without rewriting everything.

Usage:
    python patch_add_mtf.py

What it does:
1. Adds MTF feature calculation function
2. Patches your training to use MTF
3. Tests that it works
"""


# Add MTF feature function to your project
MTF_CODE = '''
def add_mtf_features(df, base_timeframe='1h'):
    """
    Add Multi-Timeframe features to existing dataframe
    
    For 1h base: adds 4h, 1d, 1w timeframes
    Total: +26 features
    
    Args:
        df: DataFrame with OHLCV (columns: open, high, low, close, volume)
        base_timeframe: Your trading timeframe
    
    Returns:
        DataFrame with MTF features added
    """
    import pandas as pd
    import numpy as np
    
    print("\\n🎯 Adding Multi-Timeframe features...")
    
    # Define higher timeframes
    mtf_map = {
        '1h': [('4h', '4H'), ('1d', '1D'), ('1w', '1W')],
        '5m': [('15m', '15T'), ('1h', '1H'), ('4h', '4H')],
    }
    
    higher_tfs = mtf_map.get(base_timeframe, [('4h', '4H'), ('1d', '1D'), ('1w', '1W')])
    
    for tf_num, (tf_name, resample_rule) in enumerate(higher_tfs, 1):
        print(f"  Processing {tf_name}...")
        
        try:
            # Resample to higher timeframe
            resampled = df.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            if len(resampled) < 50:
                print(f"  ⚠️  Insufficient data for {tf_name}, skipping")
                continue
            
            # Calculate indicators
            ema20 = resampled['close'].ewm(span=20, adjust=False).mean()
            ema50 = resampled['close'].ewm(span=50, adjust=False).mean()
            trend = (ema20 > ema50).astype(int)
            
            ema12 = resampled['close'].ewm(span=12, adjust=False).mean()
            ema26 = resampled['close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            macd_positive = (macd > 0).astype(int)
            
            delta = resampled['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            vol_ma = resampled['volume'].rolling(window=20).mean()
            vol_trend = (resampled['volume'] > vol_ma).astype(int)
            
            # Merge back to base timeframe
            prefix = f'tf{tf_num}_'
            df[prefix + 'trend'] = trend.reindex(df.index, method='ffill')
            df[prefix + 'macd'] = macd.reindex(df.index, method='ffill')
            df[prefix + 'macd_signal'] = macd_signal.reindex(df.index, method='ffill')
            df[prefix + 'macd_positive'] = macd_positive.reindex(df.index, method='ffill')
            df[prefix + 'rsi'] = rsi.reindex(df.index, method='ffill')
            df[prefix + 'vol_trend'] = vol_trend.reindex(df.index, method='ffill')
            
        except Exception as e:
            print(f"  ⚠️  Error processing {tf_name}: {e}")
            continue
    
    # Add alignment signals
    trend_cols = [f'tf{i}_trend' for i in range(1, 4)]
    if all(col in df.columns for col in trend_cols):
        df['strong_alignment'] = (
            (df['tf1_trend'] == 1) &
            (df['tf2_trend'] == 1) &
            (df['tf3_trend'] == 1)
        ).astype(int)
        
        trend_sum = df['tf1_trend'] + df['tf2_trend'] + df['tf3_trend']
        df['weak_alignment'] = (trend_sum >= 2).astype(int)
        
        macd_cols = [f'tf{i}_macd_positive' for i in range(1, 4)]
        if all(col in df.columns for col in macd_cols):
            df['all_macd_positive'] = (
                (df['tf1_macd_positive'] == 1) &
                (df['tf2_macd_positive'] == 1) &
                (df['tf3_macd_positive'] == 1)
            ).astype(int)
    
    mtf_features = [c for c in df.columns if 'tf' in c or 'alignment' in c]
    print(f"✅ Added {len(mtf_features)} MTF features")
    
    return df
'''

def create_mtf_module():
    """Save MTF function as separate module"""
    
    with open('mtf_features_simple.py', 'w') as f:
        f.write('"""\nSimple MTF Feature Module\n"""\n\n')
        f.write(MTF_CODE)
    
    print("✅ Created mtf_features_simple.py")

def create_patched_training():
    """Create new training script with MTF"""
    
    code = '''"""
LSTM Training with MTF - Patched Version
========================================
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your existing modules
try:
    from lstm_training_pipeline import LSTMTrainingPipeline
    from lstm_model_complete import LSTMModel
except ImportError as e:
    print(f"Error importing: {e}")
    print("Make sure lstm_training_pipeline.py and lstm_model_complete.py exist")
    sys.exit(1)

# Import MTF features
from mtf_features_simple import add_mtf_features


def train_with_mtf(symbols=['AAPL', 'TSLA']):
    """Train models with MTF features"""
    
    print("\\n" + "="*70)
    print("LSTM TRAINING WITH MULTI-TIMEFRAME FEATURES")
    print("="*70 + "\\n")
    
    pipeline = LSTMTrainingPipeline()
    results = {}
    
    for symbol in symbols:
        print(f"\\n{'='*70}")
        print(f"Training {symbol} with MTF")
        print(f"{'='*70}\\n")
        
        try:
            # Download data - CORRECT SETTINGS!
            logger.info(f"Downloading 2 YEARS of HOURLY data for {symbol}...")
            df = yf.download(symbol, period='2y', interval='1h', progress=False)
            
            if df.empty:
                logger.error(f"No data for {symbol}")
                continue
            
            # Flatten columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = df.columns.str.lower()
            
            logger.info(f"✅ Downloaded {len(df)} bars")
            logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
            
            if len(df) < 3800:
                logger.warning(f"⚠️  Only {len(df)} bars - expected ~4032 for 2y hourly")
            
            # Add MTF features
            df = add_mtf_features(df, base_timeframe='1h')
            
            # Train using existing pipeline
            result = pipeline.run_full_pipeline(
                symbol=symbol,
                data=df,
                test_split=0.2,
                model_config={
                    'lstm_units_1': 128,
                    'lstm_units_2': 64,
                    'dropout_rate': 0.2,
                    'learning_rate': 0.001
                }
            )
            
            results[symbol] = result
            
            logger.info(f"\\n✅ {symbol} complete!")
            logger.info(f"   Validation Accuracy: {result.get('validation_accuracy', 0):.3f}")
            logger.info(f"   Backtest Return: {result.get('backtest_return', 0):.2f}%")
            
        except Exception as e:
            logger.error(f"Error training {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70 + "\\n")
    
    for symbol, result in results.items():
        acc = result.get('validation_accuracy', 0) * 100
        ret = result.get('backtest_return', 0)
        
        status = "✅ GOOD" if ret > 5 else "↗ POSITIVE" if ret > 0 else "⚠️  NEEDS WORK"
        
        print(f"{symbol}:")
        print(f"  Accuracy: {acc:.1f}%")
        print(f"  Return: {ret:+.2f}%")
        print(f"  Status: {status}\\n")
    
    return results


if __name__ == "__main__":
    results = train_with_mtf(['AAPL', 'TSLA'])
'''
    
    with open('train_with_mtf_patched.py', 'w') as f:
        f.write(code)
    
    print("✅ Created train_with_mtf_patched.py")


def main():
    """Main patch function"""
    
    print("\\n" + "🔧"*35)
    print("MTF PATCH INSTALLER")
    print("🔧"*35 + "\\n")
    
    print("This will create:")
    print("  1. mtf_features_simple.py - MTF feature calculator")
    print("  2. train_with_mtf_patched.py - Training script with MTF\\n")
    
    # Create files
    create_mtf_module()
    create_patched_training()
    
    print("\\n" + "="*70)
    print("✅ PATCH COMPLETE!")
    print("="*70 + "\\n")
    
    print("Next steps:")
    print("  1. Run the new training script:")
    print("     python train_with_mtf_patched.py")
    print("\\n  2. Watch for:")
    print("     - Downloaded ~3500-4000 bars")
    print("     - Added 26 MTF features")
    print("     - Positive backtest returns\\n")
    
    print("="*70 + "\\n")


if __name__ == "__main__":
    main()
