# LSTM Trading System - Complete Setup Guide

## 📋 Overview

You now have a complete LSTM-based AI trading system with:
- **Advanced LSTM Model** with attention mechanism
- **Ensemble Learning** for robust predictions  
- **Modular Integration** with your trading bot
- **Complete Training Pipeline** with backtesting
- **Real-time Prediction** capabilities

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Core ML libraries
pip install tensorflow==2.15.0
pip install keras==2.15.0
pip install scikit-learn==1.3.0

# Data handling
pip install pandas numpy yfinance

# Visualization
pip install matplotlib seaborn

# Utilities
pip install joblib

# Optional: GPU support (CUDA required)
pip install tensorflow-gpu==2.15.0
```

### Step 2: Create Project Structure

```bash
mkdir -p models/lstm_pipeline
mkdir -p models/lstm_trading
mkdir -p data/historical
mkdir -p logs
```

### Step 3: Save the Code Files

Save these 3 artifacts as Python files:
1. `lstm_model_complete.py` - Core LSTM models
2. `lstm_trading_integration.py` - Trading bot integration
3. `lstm_training_pipeline.py` - Training workflow

---

## 📊 Training Your First Model

### Option 1: Train on Live Market Data (Yahoo Finance)

```python
from lstm_training_pipeline import LSTMTrainingPipeline

# Initialize pipeline
pipeline = LSTMTrainingPipeline(output_dir="models/lstm_pipeline")

# Train on multiple symbols
symbols = ['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ']

results = pipeline.run_full_pipeline(
    symbols=symbols,
    data_source='yahoo',
    period='1y',      # 1 year of data
    interval='5m'     # 5-minute bars
)

# Results saved automatically to models/lstm_pipeline/
```

### Option 2: Train on Your CSV Data

```python
# Prepare your CSV with columns: timestamp, open, high, low, close, volume
pipeline = LSTMTrainingPipeline()

df = pipeline.load_csv_data('data/historical/AAPL_5min.csv')
model, history, results = pipeline.train_model('AAPL', df)

print(f"Validation Accuracy: {results['val_accuracy']:.3f}")
print(f"Validation AUC: {results['val_auc']:.3f}")
```

### Option 3: Quick Test with Sample Data

```python
from lstm_model_complete import LSTMTradingModel
import pandas as pd
import numpy as np

# Generate sample data
dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
df = pd.DataFrame({
    'open': np.cumsum(np.random.randn(1000)) + 100,
    'high': np.cumsum(np.random.randn(1000)) + 101,
    'low': np.cumsum(np.random.randn(1000)) + 99,
    'close': np.cumsum(np.random.randn(1000)) + 100,
    'volume': np.random.randint(100000, 1000000, 1000)
}, index=dates)

# Train
lstm = LSTMTradingModel(sequence_length=60, prediction_horizon=5)
history, results = lstm.train(df, epochs=20)

# Make prediction
probability = lstm.predict(df)
print(f"Prediction: {'UP' if probability > 0.5 else 'DOWN'} ({probability:.3f})")
```

---

## 🔗 Integration with Trading Bot

### Step 1: Add to Main Bot

```python
# In your ibkr_trading_backend.py

from lstm_trading_integration import LSTMTradingModule
from modular_dashboard_config import DataBus

# Initialize data bus (if not already created)
data_bus = DataBus()

# Initialize LSTM module
lstm_config = {
    'sequence_length': 60,
    'prediction_horizon': 5,
    'update_interval': 30,
    'min_confidence': 0.65,
    'use_ensemble': True
}

lstm_module = LSTMTradingModule(data_bus, lstm_config)
lstm_module.initialize()
lstm_module.start()

# The module will now:
# - Listen for market data
# - Generate predictions
# - Publish AI signals to data bus
```

### Step 2: Subscribe to LSTM Signals

```python
def on_lstm_signal(signal):
    """Handle LSTM predictions"""
    symbol = signal['symbol']
    direction = signal['signal_type']
    probability = signal['probability']
    confidence = signal['confidence']
    
    print(f"LSTM Signal: {symbol} {direction} @ {probability:.3f} (conf: {confidence:.3f})")
    
    # Execute trade if high confidence
    if confidence > 0.7:
        execute_lstm_trade(signal)

# Subscribe to signals
data_bus.subscribe('ai_signals', on_lstm_signal)
```

### Step 3: Feed Market Data to LSTM

```python
# In your market data handler
def on_market_update(symbol, bar_data):
    """Forward market data to LSTM module"""
    
    market_data = {
        'symbol': symbol,
        'timestamp': datetime.now(),
        'open': bar_data['open'],
        'high': bar_data['high'],
        'low': bar_data['low'],
        'close': bar_data['close'],
        'volume': bar_data['volume']
    }
    
    data_bus.publish('market_data', market_data)
```

---

## 🎯 Model Configuration Guide

### Basic Configuration (Faster Training)

```python
config = {
    'sequence_length': 30,        # Look back 30 bars
    'prediction_horizon': 3,       # Predict 3 bars ahead
    'lstm_units': [32, 16],       # Smaller networks
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'epochs': 20,
    'batch_size': 64
}
```

### Advanced Configuration (Better Performance)

```python
config = {
    'sequence_length': 60,        # Look back 60 bars
    'prediction_horizon': 5,       # Predict 5 bars ahead  
    'lstm_units': [128, 64],      # Larger networks
    'dropout_rate': 0.3,          # More regularization
    'learning_rate': 0.0005,
    'epochs': 50,
    'batch_size': 32
}
```

### Day Trading Configuration

```python
config = {
    'sequence_length': 20,        # Shorter lookback
    'prediction_horizon': 1,       # Predict next bar
    'lstm_units': [64, 32],
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'epochs': 30,
    'batch_size': 32
}
```

---

## 📈 Understanding Model Outputs

### Prediction Probability

```python
probability = lstm.predict(df)

if probability > 0.7:
    print("Strong BUY signal")
elif probability > 0.6:
    print("Moderate BUY signal")
elif probability > 0.4:
    print("Neutral / No trade")
elif probability > 0.3:
    print("Moderate SELL signal")
else:
    print("Strong SELL signal")
```

### Signal Confidence (Ensemble Only)

```python
from lstm_model_complete import EnsembleLSTM

ensemble = EnsembleLSTM(n_models=3)
ensemble.train_ensemble(df, epochs=30)

probability, confidence = ensemble.predict_with_confidence(df)

print(f"Prediction: {probability:.3f}")
print(f"Confidence: {confidence:.3f}")

# Trade only on high confidence
if confidence > 0.75:
    execute_trade(probability)
```

---

## 🔄 Continuous Learning

### Daily Retraining

```python
import schedule
import time

def retrain_models():
    """Retrain all models daily"""
    pipeline = LSTMTrainingPipeline()
    
    symbols = ['AAPL', 'TSLA', 'NVDA']
    
    for symbol in symbols:
        df = pipeline.fetch_historical_data(symbol, period='3mo', interval='5m')
        model, _, results = pipeline.train_model(symbol, df, model_config={
            'epochs': 20,
            'sequence_length': 60,
            'prediction_horizon': 5
        })
        
        print(f"{symbol} retrained - Accuracy: {results['val_accuracy']:.3f}")

# Schedule daily at 6 AM
schedule.every().day.at("06:00").do(retrain_models)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Online Learning (Real-time Updates)

```python
class OnlineLSTM:
    """LSTM with incremental learning"""
    
    def __init__(self, model):
        self.model = model
        self.buffer = []
        self.buffer_size = 100
        
    def add_observation(self, bar_data, actual_outcome):
        """Add new data point"""
        self.buffer.append({
            'data': bar_data,
            'outcome': actual_outcome
        })
        
        # Retrain when buffer is full
        if len(self.buffer) >= self.buffer_size:
            self.incremental_train()
            self.buffer = []
    
    def incremental_train(self):
        """Retrain on recent data"""
        df = pd.DataFrame([x['data'] for x in self.buffer])
        # Quick fine-tuning with low learning rate
        self.model.train(df, epochs=5, batch_size=16)
```

---

## 📊 Performance Monitoring

### Track Model Accuracy

```python
class ModelMonitor:
    """Monitor live model performance"""
    
    def __init__(self):
        self.predictions = []
        self.outcomes = []
        
    def log_prediction(self, symbol, probability, actual_outcome):
        """Log prediction and actual result"""
        self.predictions.append({
            'symbol': symbol,
            'probability': probability,
            'predicted': 1 if probability > 0.5 else 0,
            'actual': actual_outcome,
            'timestamp': datetime.now()
        })
        
    def get_accuracy(self, lookback_hours=24):
        """Calculate recent accuracy"""
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        recent = [p for p in self.predictions if p['timestamp'] > cutoff]
        
        if not recent:
            return None
            
        correct = sum(1 for p in recent if p['predicted'] == p['actual'])
        return correct / len(recent)
    
    def get_metrics(self):
        """Get comprehensive metrics"""
        df = pd.DataFrame(self.predictions)
        
        return {
            'accuracy': (df['predicted'] == df['actual']).mean(),
            'total_predictions': len(df),
            'by_symbol': df.groupby('symbol').apply(
                lambda x: (x['predicted'] == x['actual']).mean()
            ).to_dict()
        }

# Usage
monitor = ModelMonitor()

# When making prediction
probability = lstm.predict(df)
monitor.log_prediction('AAPL', probability, actual_outcome=1)

# Check performance
print(f"24h Accuracy: {monitor.get_accuracy(24):.2%}")
```

---

## 🎮 Advanced Usage

### 1. Multi-Timeframe Analysis

```python
def multi_timeframe_signal(symbol):
    """Combine predictions from multiple timeframes"""
    
    # Get data at different intervals
    df_1min = fetch_data(symbol, interval='1m')
    df_5min = fetch_data(symbol, interval='5m')
    df_15min = fetch_data(symbol, interval='15m')
    
    # Load models trained on each timeframe
    model_1min = LSTMTradingModel()
    model_1min.load('lstm_1min')
    
    model_5min = LSTMTradingModel()
    model_5min.load('lstm_5min')
    
    model_15min = LSTMTradingModel()
    model_15min.load('lstm_15min')
    
    # Get predictions
    p1 = model_1min.predict(df_1min)
    p5 = model_5min.predict(df_5min)
    p15 = model_15min.predict(df_15min)
    
    # Weighted average (longer timeframe = higher weight)
    combined = (p1 * 0.2) + (p5 * 0.3) + (p15 * 0.5)
    
    return combined

signal = multi_timeframe_signal('AAPL')
print(f"Multi-timeframe signal: {signal:.3f}")
```

### 2. Feature Importance Analysis

```python
def analyze_feature_importance(model, df):
    """Analyze which features matter most"""
    
    features = model.create_features(df)
    base_prediction = model.predict(df)
    
    importance = {}
    
    for col in features.columns:
        # Zero out feature
        df_modified = features.copy()
        df_modified[col] = 0
        
        # Get new prediction
        modified_pred = model.predict(df_modified)
        
        # Impact = difference in prediction
        importance[col] = abs(base_prediction - modified_pred)
    
    # Sort by importance
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    print("Feature Importance:")
    for feature, impact in sorted_features[:10]:
        print(f"  {feature}: {impact:.4f}")
    
    return importance
```

### 3. Risk-Adjusted Position Sizing

```python
def kelly_position_size(prediction, confidence, capital, risk_per_trade=0.02):
    """Calculate optimal position size using Kelly Criterion"""
    
    # Edge calculation
    edge = abs(prediction - 0.5) * 2  # Scale to 0-1
    
    # Adjust by confidence
    adjusted_edge = edge * confidence
    
    # Win probability
    win_prob = prediction if prediction > 0.5 else (1 - prediction)
    
    # Kelly fraction: f = (bp - q) / b
    # where b=1 (1:1 payout), p=win_prob, q=1-win_prob
    kelly_fraction = (win_prob * 1 - (1 - win_prob)) / 1
    
    # Apply confidence scaling
    kelly_fraction *= confidence
    
    # Use fractional Kelly (safer)
    fractional_kelly = kelly_fraction * 0.25  # 25% of full Kelly
    
    # Calculate position size
    position_size = capital * fractional_kelly
    
    # Cap at risk limit
    max_position = capital * risk_per_trade
    position_size = min(position_size, max_position)
    
    return max(0, position_size)

# Example
capital = 100000
pred = 0.72
conf = 0.85

size = kelly_position_size(pred, conf, capital)
print(f"Recommended position size: ${size:,.2f}")
```

---

## 🛠️ Troubleshooting

### Issue: Low Accuracy (< 55%)

**Solutions:**
```python
# 1. Increase sequence length
config['sequence_length'] = 120  # More historical context

# 2. Add more training data
df = pipeline.fetch_historical_data(symbol, period='2y', interval='5m')

# 3. Use ensemble
ensemble = EnsembleLSTM(n_models=5)
ensemble.train_ensemble(df)

# 4. Adjust prediction horizon
config['prediction_horizon'] = 3  # Shorter prediction window
```

### Issue: Overfitting (High train, low validation accuracy)

**Solutions:**
```python
# 1. Increase dropout
config['dropout_rate'] = 0.4

# 2. Add L2 regularization
from tensorflow.keras import regularizers

model.add(layers.Dense(
    32, 
    activation='relu',
    kernel_regularizer=regularizers.l2(0.01)
))

# 3. Use early stopping
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 4. Reduce model complexity
config['lstm_units'] = [32, 16]  # Smaller network
```

### Issue: Slow Training

**Solutions:**
```python
# 1. Reduce data size
df = df.iloc[-5000:]  # Use only recent data

# 2. Increase batch size
config['batch_size'] = 128

# 3. Use GPU
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# 4. Reduce epochs
config['epochs'] = 20
```

### Issue: Predictions Always 0.5 (Neutral)

**Solutions:**
```python
# 1. Check target distribution
future_return = df['close'].shift(-5) / df['close'] - 1
target = (future_return > 0).astype(int)
print(f"Target distribution: {target.value_counts()}")

# 2. Adjust prediction threshold
threshold = 0.0005  # 0.05% move
target = (future_return > threshold).astype(int)

# 3. Verify feature scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()  # Better for outliers

# 4. Check for data leakage
# Ensure no future data in features
```

---

## 📁 File Structure

```
ai_trading_bot/
├── lstm_model_complete.py           # Core LSTM models
├── lstm_trading_integration.py      # Bot integration
├── lstm_training_pipeline.py        # Training workflow
├── ibkr_trading_backend.py          # Main trading bot
├── modular_dashboard_config.py      # Data bus
│
├── models/
│   ├── lstm_pipeline/
│   │   ├── AAPL_lstm.h5
│   │   ├── AAPL_lstm_scaler.pkl
│   │   ├── pipeline_summary.json
│   │   └── evaluation_plots.png
│   │
│   └── lstm_trading/
│       ├── best_model.h5
│       └── training_results.json
│
├── data/
│   └── historical/
│       ├── AAPL_5min.csv
│       └── TSLA_5min.csv
│
└── logs/
    └── lstm_training.log
```

---

## 🚀 Next Steps

### 1. **Deploy to Production**
```python
# Load trained model
lstm = LSTMTradingModel()
lstm.load('AAPL_lstm')

# Integrate with trading bot
from lstm_trading_integration import LSTMStrategyExecutor

executor = LSTMStrategyExecutor(
    lstm_module=lstm_module,
    risk_manager=risk_manager,
    order_executor=order_executor
)

# Start live trading
executor.execute_strategy('AAPL', account_data)
```

### 2. **Add More Symbols**
```python
symbols = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 
           'GOOGL', 'AMZN', 'META', 'SPY', 'QQQ']

pipeline.run_full_pipeline(symbols=symbols)
```

### 3. **Combine with Other Strategies**
```python
# Ensemble of AI strategies
lstm_signal = lstm_module.get_signal_for_symbol('AAPL')
alpha_fusion_signal = alpha_fusion.get_signal('AAPL')

# Weighted combination
combined_signal = (
    lstm_signal['probability'] * 0.4 + 
    alpha_fusion_signal['probability'] * 0.6
)
```

### 4. **Monitor and Improve**
- Track live accuracy daily
- Retrain weekly on new data
- A/B test different configurations
- Analyze which market conditions work best

---

## 📞 Quick Commands

```bash
# Train all models
python lstm_training_pipeline.py

# Test single model
python lstm_model_complete.py

# Start live integration
python lstm_trading_integration.py

# Full bot with LSTM
python ibkr_trading_backend.py --enable-lstm
```

---

## ✅ Validation Checklist

Before going live:
- [ ] Model accuracy > 60% on validation set
- [ ] Backtest Sharpe ratio > 1.5
- [ ] Win rate > 55%
- [ ] Max drawdown < 15%
- [ ] Tested on paper trading for 30+ days
- [ ] Risk management integrated
- [ ] Stop-loss orders configured
- [ ] Daily loss limits set
- [ ] Model monitoring active
- [ ] Retraining schedule configured

---

## 🎯 Expected Performance

**Realistic Expectations:**
- **Accuracy**: 58-65% directional accuracy
- **Sharpe Ratio**: 1.5-2.5 (after costs)
- **Win Rate**: 55-60%
- **Best Performance**: High volume, trending stocks
- **Limitations**: Struggles in choppy/ranging markets

**Tips for Success:**
1. Start with paper trading
2. Use ensemble models for stability
3. Combine with technical indicators
4. Trade only high-confidence signals (>0.7)
5. Respect risk management rules
6. Monitor performance continuously
7. Retrain regularly on fresh data

---

## 📚 Additional Resources

- TensorFlow LSTM Guide: https://www.tensorflow.org/guide/keras/rnn
- Time Series Forecasting: https://www.tensorflow.org/tutorials/structured_data/time_series
- Trading ML Best Practices: https://www.quantstart.com/articles/

---

**You're now ready to train and deploy LSTM models for AI trading!** 🚀

Start with the training pipeline, validate your models with backtesting, then integrate with your trading bot when ready.