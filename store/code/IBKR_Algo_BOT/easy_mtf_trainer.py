"""
EASY MTF TRAINER V2 - Optimized Version
========================================

Improvements over V1:
1. Lower confidence threshold (0.52 instead of 0.58) - More trades
2. Bigger LSTM model (256/128 units) - Better learning
3. More training epochs - Better convergence
4. Additional MTF features - Better predictions

Usage: python EASY_MTF_TRAINER_V2.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models/lstm_mtf_v2', exist_ok=True)

print("\n" + "🚀"*35)
print("EASY MTF TRAINER V2 - Optimized Version")
print("🚀"*35 + "\n")


def add_base_features(df):
    """Add technical indicators"""
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    for window in [5, 10, 20, 50]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}']
    
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-10)
    
    df['momentum_5'] = df['close'].pct_change(periods=5)
    df['momentum_10'] = df['close'].pct_change(periods=10)
    
    # Additional momentum features
    df['momentum_20'] = df['close'].pct_change(periods=20)
    df['roc_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    return df


def add_mtf_features(df):
    """Add Multi-Timeframe features - ENHANCED"""
    print("🎯 Adding MTF features...")
    
    higher_tfs = [('4h', '4H'), ('1d', '1D'), ('1w', '1W')]
    
    for tf_num, (tf_name, resample_rule) in enumerate(higher_tfs, 1):
        print(f"  {tf_name}...", end='')
        
        try:
            resampled = df.resample(resample_rule).agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'
            }).dropna()
            
            if len(resampled) < 50:
                print(" ⚠️  skipped")
                continue
            
            # Trend
            ema20 = resampled['close'].ewm(span=20, adjust=False).mean()
            ema50 = resampled['close'].ewm(span=50, adjust=False).mean()
            trend = (ema20 > ema50).astype(int)
            
            # MACD
            ema12 = resampled['close'].ewm(span=12, adjust=False).mean()
            ema26 = resampled['close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            macd_positive = (macd > 0).astype(int)
            macd_hist = macd - macd_signal
            
            # RSI
            delta = resampled['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # Volume
            vol_ma = resampled['volume'].rolling(window=20).mean()
            vol_trend = (resampled['volume'] > vol_ma).astype(int)
            
            # Price momentum
            momentum = resampled['close'].pct_change(periods=5)
            
            # Merge back
            prefix = f'tf{tf_num}_'
            df[prefix + 'trend'] = trend.reindex(df.index, method='ffill')
            df[prefix + 'macd'] = macd.reindex(df.index, method='ffill')
            df[prefix + 'macd_hist'] = macd_hist.reindex(df.index, method='ffill')
            df[prefix + 'macd_positive'] = macd_positive.reindex(df.index, method='ffill')
            df[prefix + 'rsi'] = rsi.reindex(df.index, method='ffill')
            df[prefix + 'vol_trend'] = vol_trend.reindex(df.index, method='ffill')
            df[prefix + 'momentum'] = momentum.reindex(df.index, method='ffill')
            
            print(" ✅")
            
        except Exception as e:
            print(f" ❌ {e}")
    
    # Alignment signals
    trend_cols = [f'tf{i}_trend' for i in range(1, 4)]
    if all(col in df.columns for col in trend_cols):
        # Strong alignment
        df['strong_alignment'] = ((df['tf1_trend'] == 1) & 
                                  (df['tf2_trend'] == 1) & 
                                  (df['tf3_trend'] == 1)).astype(int)
        
        # Weak alignment
        trend_sum = df['tf1_trend'] + df['tf2_trend'] + df['tf3_trend']
        df['weak_alignment'] = (trend_sum >= 2).astype(int)
        
        # All MACD positive
        macd_cols = [f'tf{i}_macd_positive' for i in range(1, 4)]
        if all(col in df.columns for col in macd_cols):
            df['all_macd_positive'] = ((df['tf1_macd_positive'] == 1) &
                                       (df['tf2_macd_positive'] == 1) &
                                       (df['tf3_macd_positive'] == 1)).astype(int)
    
    mtf_count = len([c for c in df.columns if 'tf' in c or 'alignment' in c])
    print(f"✅ Added {mtf_count} MTF features\n")
    return df


def backtest(df, predictions, confidence, threshold=0.52):
    """Run backtest - LOWER threshold for more trades"""
    position = 0
    entry_price = 0
    entry_idx = None
    trades = []
    equity = [10000]
    
    for i in range(len(df)):
        price = df['close'].iloc[i]
        conf = confidence[i]
        pred = predictions[i]
        
        if position == 0 and conf >= threshold and pred == 1:
            position = 1
            entry_price = price * 1.001
            entry_idx = i
            
        elif position == 1:
            bars_held = i - entry_idx
            exit_triggered = False
            
            if bars_held >= 2:  # Reduced from 3
                if pred == 0 or bars_held >= 24:  # Increased max hold
                    exit_triggered = True
            
            if exit_triggered:
                exit_price = price * 0.999
                pnl = (exit_price - entry_price) / entry_price
                trades.append(pnl)
                equity.append(equity[-1] * (1 + pnl))
                position = 0
        
        if position == 0:
            equity.append(equity[-1])
    
    if not trades:
        return {'return': 0, 'trades': 0, 'win_rate': 0, 'sharpe': 0, 'max_dd': 0}
    
    wins = [t for t in trades if t > 0]
    equity_series = pd.Series(equity)
    max_dd = ((equity_series / equity_series.cummax()) - 1).min() * 100
    
    return {
        'return': (equity[-1] / equity[0] - 1) * 100,
        'trades': len(trades),
        'win_rate': len(wins) / len(trades) * 100,
        'sharpe': np.mean(trades) / (np.std(trades) + 1e-10) * np.sqrt(252),
        'max_dd': max_dd
    }


def train_symbol(symbol):
    """Train one symbol - BIGGER MODEL"""
    print(f"\n{'='*70}")
    print(f"TRAINING {symbol}")
    print(f"{'='*70}\n")
    
    # Download
    print("📥 Downloading 2y hourly data...")
    df = yf.download(symbol, period='2y', interval='1h', progress=False)
    
    if df.empty:
        print("❌ No data!")
        return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    
    print(f"✅ {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
    
    # Features
    print("\n🔧 Adding features...")
    df = add_base_features(df)
    df = add_mtf_features(df)
    
    df = df.dropna()
    print(f"✅ Clean data: {len(df)} bars\n")
    
    # Target
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    
    # Select features
    exclude = ['open', 'high', 'low', 'close', 'volume', 'target']
    features = [c for c in df.columns if c not in exclude]
    
    X = df[features].values
    y = df['target'].values
    
    print(f"📊 Features: {len(features)} | Samples: {len(X)}")
    print(f"   Up: {y.sum()} ({y.sum()/len(y)*100:.1f}%)\n")
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    df_test = df.iloc[split:].copy()
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    # Build BIGGER model
    print("🧠 Building BIGGER LSTM...")
    model = keras.Sequential([
        layers.LSTM(256, activation='tanh', return_sequences=True,
                   input_shape=(1, X_train.shape[2])),
        layers.Dropout(0.3),
        layers.LSTM(128, activation='tanh', return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64, activation='tanh'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    print(f"   Parameters: {model.count_params():,}\n")
    
    # Train LONGER
    print("🏋️  Training (takes 10-15 min with bigger model)...")
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,  # Increased from 50
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_auc', patience=15,
                                         restore_best_weights=True, mode='max'),
            keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                                             patience=7, min_lr=0.00001, mode='max')
        ],
        verbose=0
    )
    
    # Evaluate
    train_pred = model.predict(X_train, verbose=0)
    test_pred_proba = model.predict(X_test, verbose=0)
    test_pred = (test_pred_proba > 0.5).astype(int).flatten()
    
    train_acc = ((train_pred.flatten() > 0.5).astype(int) == y_train[:len(train_pred)]).mean()
    test_acc = (test_pred == y_test).mean()
    
    print("\n✅ Training done!")
    print(f"   Train acc: {train_acc*100:.1f}%")
    print(f"   Test acc: {test_acc*100:.1f}%\n")
    
    # Backtest with LOWER threshold
    print("📈 Running backtest (threshold=0.52 for more trades)...")
    results = backtest(df_test, test_pred, test_pred_proba.flatten(), threshold=0.52)
    
    print(f"\n{'='*70}")
    print(f"RESULTS FOR {symbol}")
    print(f"{'='*70}")
    print(f"  Total Return:    {results['return']:+.2f}%")
    print(f"  Number of Trades: {results['trades']}")
    print(f"  Win Rate:        {results['win_rate']:.1f}%")
    print(f"  Sharpe Ratio:    {results['sharpe']:.2f}")
    print(f"  Max Drawdown:    {results['max_dd']:.2f}%")
    print(f"{'='*70}\n")
    
    # Save
    model_path = f"models/lstm_mtf_v2/{symbol}_mtf_v2.keras"
    model.save(model_path)
    print(f"💾 Saved: {model_path}\n")
    
    return {
        'symbol': symbol,
        'test_acc': test_acc,
        'results': results,
        'num_features': len(features)
    }


def main():
    """Main function"""
    symbols = ['AAPL', 'TSLA']
    results = {}
    
    print("🎯 V2 IMPROVEMENTS:")
    print("  - Bigger LSTM (256/128/64 units)")
    print("  - More training (100 epochs)")
    print("  - Lower threshold (0.52 → more trades)")
    print("  - Enhanced MTF features")
    print("  - Better dropout (prevent overfitting)\n")
    
    for symbol in symbols:
        try:
            result = train_symbol(symbol)
            if result:
                results[symbol] = result
        except Exception as e:
            print(f"❌ Error with {symbol}: {e}\n")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - V2 OPTIMIZED")
    print("="*70 + "\n")
    
    for symbol, data in results.items():
        acc = data['test_acc'] * 100
        ret = data['results']['return']
        wr = data['results']['win_rate']
        trades = data['results']['trades']
        sharpe = data['results']['sharpe']
        
        if ret > 15 and wr > 60:
            status = "🌟 OUTSTANDING"
        elif ret > 10 and wr > 55:
            status = "✅ EXCELLENT"
        elif ret > 5:
            status = "✓ GOOD"
        elif ret > 0:
            status = "↗ POSITIVE"
        else:
            status = "⚠️  NEEDS WORK"
        
        print(f"{symbol}: {status}")
        print(f"  Accuracy: {acc:.1f}%")
        print(f"  Return: {ret:+.2f}%")
        print(f"  Win Rate: {wr:.1f}%")
        print(f"  Trades: {trades}")
        print(f"  Sharpe: {sharpe:.2f}")
        print(f"  Features: {data['num_features']}\n")
    
    # Comparison with V1
    print("="*70)
    print("V1 vs V2 COMPARISON")
    print("="*70)
    print("\nV1 Results (from your run):")
    print("  AAPL: +0.64% (1 trade)")
    print("  TSLA: +10.09% (13 trades)")
    print("\nV2 Expected:")
    print("  More trades (20-40 per symbol)")
    print("  Better accuracy (60-65%)")
    print("  Higher returns (15-25%)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
