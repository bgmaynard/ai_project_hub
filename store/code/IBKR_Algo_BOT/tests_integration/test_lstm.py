"""
Quick test of LSTM model - generates sample data and trains
Save as: test_lstm.py
Run with: python test_lstm.py
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("Testing LSTM Model...")
print("=" * 60)

# Import your LSTM model
try:
    from lstm_model_complete import LSTMTradingModel
    print("✓ LSTM module loaded successfully")
except Exception as e:
    print(f"✗ Error loading LSTM module: {e}")
    print("\nMake sure lstm_model_complete.py is in the same folder!")
    exit(1)

# Generate sample data
print("\n1. Generating sample market data...")
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=500, freq='5min')

price = 100
prices = []
for i in range(500):
    price += np.random.randn() * 0.5
    prices.append(price)

df = pd.DataFrame({
    'open': prices,
    'high': [p + abs(np.random.randn() * 0.3) for p in prices],
    'low': [p - abs(np.random.randn() * 0.3) for p in prices],
    'close': [p + np.random.randn() * 0.2 for p in prices],
    'volume': np.random.randint(100000, 1000000, 500)
}, index=dates)

# Ensure high >= low
df['high'] = df[['open', 'close', 'high']].max(axis=1)
df['low'] = df[['open', 'close', 'low']].min(axis=1)

print(f"✓ Created {len(df)} bars of sample data")

# Initialize and train model
print("\n2. Building LSTM model...")
lstm = LSTMTradingModel(sequence_length=30, prediction_horizon=5)
print("✓ Model initialized")

print("\n3. Training model (this will take 2-3 minutes)...")
print("   Be patient - this is normal for neural networks!")
history, results = lstm.train(df, epochs=10, batch_size=32)

print("\n4. Training Results:")
print(f"   Training Accuracy: {results['train_accuracy']:.3f}")
print(f"   Validation Accuracy: {results['val_accuracy']:.3f}")
print(f"   Validation AUC: {results['val_auc']:.3f}")

# Make a prediction
print("\n5. Testing prediction...")
probability = lstm.predict(df)
direction = "BUY" if probability > 0.5 else "SELL"
confidence = abs(probability - 0.5) * 2

print(f"   Prediction: {direction}")
print(f"   Probability: {probability:.3f}")
print(f"   Confidence: {confidence:.3f}")

# Save model
print("\n6. Saving model...")
lstm.save("test_model")
print(f"   ✓ Model saved to: models/lstm_trading/")

print("\n" + "=" * 60)
print("✓ LSTM Test Complete - Everything Works!")
print("=" * 60)
print("\nNext step: Run 'python train_real_stocks.py' to train on real data")
