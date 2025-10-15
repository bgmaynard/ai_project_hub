"""
LSTM Trading Model - Complete TensorFlow Implementation
Predicts price direction and momentum probability for AI trading bot
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMTradingModel:
    """
    LSTM-based trading model with attention mechanism
    Predicts directional probability for next N candles
    """
    
    def __init__(self, sequence_length=60, prediction_horizon=5, feature_count=None):
        self.sequence_length = sequence_length  # lookback period (e.g., 60 bars)
        self.prediction_horizon = prediction_horizon  # predict N bars ahead
        self.feature_count = feature_count  # Will be set after feature engineering
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = Path("models/lstm_trading")
        self.model_path.mkdir(parents=True, exist_ok=True)
        
    def build_model(self, lstm_units=[64, 32], dropout_rate=0.2, learning_rate=0.001):
        """
        Build LSTM model with attention layer
        """
        # Use Input layer for Keras 3.x compatibility
        from tensorflow.keras import Input
        
        model = keras.Sequential(name="LSTM_Trading_Model")
        
        # Input layer (required for Keras 3.x)
        model.add(Input(shape=(self.sequence_length, self.feature_count)))
        
        # First LSTM layer with return sequences
        model.add(layers.LSTM(
            lstm_units[0],
            return_sequences=True,
            name='lstm_1'
        ))
        model.add(layers.Dropout(dropout_rate, name='dropout_1'))
        
        # Second LSTM layer
        model.add(layers.LSTM(
            lstm_units[1],
            return_sequences=False,  # Changed to False - no attention needed
            name='lstm_2'
        ))
        model.add(layers.Dropout(dropout_rate, name='dropout_2'))
        
        # Dense layers
        model.add(layers.Dense(32, activation='relu', name='dense_1'))
        model.add(layers.Dropout(dropout_rate, name='dropout_3'))
        model.add(layers.Dense(16, activation='relu', name='dense_2'))
        
        # Output layer - sigmoid for probability
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        
        # Compile with Adam optimizer
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        self.model = model
        logger.info(f"Model built with {model.count_params():,} parameters")
        return model
    
    def create_features(self, df):
        """
        Engineer features from OHLCV data
        Returns 15 technical indicators
        """
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['high_low_pct'] = (df['high'] - df['low']) / df['close']
        features['close_open_pct'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages
        for period in [5, 10, 20]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean() / df['close'] - 1
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean() / df['close'] - 1
        
        # Volatility
        features['volatility_20'] = df['close'].pct_change().rolling(20).std()
        features['atr_14'] = self._calculate_atr(df, 14) / df['close']
        
        # Volume features
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_price_trend'] = (df['close'].pct_change() * df['volume']).rolling(10).sum()
        
        # Momentum indicators
        features['rsi_14'] = self._calculate_rsi(df['close'], 14) / 100
        features['macd'], features['macd_signal'] = self._calculate_macd(df['close'])
        features['macd_diff'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        features['bb_position'] = (df['close'] - sma_20) / (2 * std_20)
        
        return features.fillna(0)
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd / prices, macd_signal / prices
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def prepare_sequences(self, df, target_column='future_return'):
        """
        Create sequences for LSTM training
        Returns: X (sequences), y (targets)
        """
        features = self.create_features(df)
        
        # Set feature count based on actual features created
        if self.feature_count is None:
            self.feature_count = len(features.columns)
            logger.info(f"Feature count set to: {self.feature_count}")
        
        # Create target: 1 if price goes up in next N bars, 0 otherwise
        future_return = df['close'].shift(-self.prediction_horizon) / df['close'] - 1
        target = (future_return > 0).astype(int)
        
        # Drop NaN rows
        valid_idx = features.notna().all(axis=1) & target.notna()
        features = features[valid_idx]
        target = target[valid_idx]
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - self.sequence_length):
            X.append(features_scaled[i:i + self.sequence_length])
            y.append(target.iloc[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(self, df, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the LSTM model
        """
        logger.info("Preparing training data...")
        X, y = self.prepare_sequences(df)
        
        logger.info(f"Training on {len(X)} sequences")
        logger.info(f"Target distribution - Up: {y.sum()}, Down: {len(y) - y.sum()}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=False
        )
        
        # Build model if not exists
        if self.model is None:
            self.build_model()
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        checkpoint = callbacks.ModelCheckpoint(
            str(self.model_path / 'best_model.h5'),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        
        # Train
        logger.info("Training model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1
        )
        
        # Evaluate
        logger.info("Evaluating model...")
        train_metrics = self.model.evaluate(X_train, y_train, verbose=0)
        val_metrics = self.model.evaluate(X_val, y_val, verbose=0)
        
        results = {
            'train_loss': float(train_metrics[0]),
            'train_accuracy': float(train_metrics[1]),
            'train_auc': float(train_metrics[2]),
            'val_loss': float(val_metrics[0]),
            'val_accuracy': float(val_metrics[1]),
            'val_auc': float(val_metrics[2]),
            'epochs_trained': len(history.history['loss']),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Training complete - Val Accuracy: {results['val_accuracy']:.3f}, Val AUC: {results['val_auc']:.3f}")
        
        # Save results
        with open(self.model_path / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return history, results
    
    def predict(self, df):
        """
        Make predictions on new data
        Returns probability of upward movement
        """
        features = self.create_features(df)
        features_scaled = self.scaler.transform(features.fillna(0))
        
        # Get last sequence
        if len(features_scaled) >= self.sequence_length:
            sequence = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            probability = self.model.predict(sequence, verbose=0)[0][0]
            return float(probability)
        else:
            logger.warning("Not enough data for prediction")
            return 0.5  # neutral
    
    def predict_batch(self, df):
        """
        Make predictions for entire dataframe
        """
        features = self.create_features(df)
        features_scaled = self.scaler.transform(features.fillna(0))
        
        predictions = []
        for i in range(len(features_scaled) - self.sequence_length + 1):
            sequence = features_scaled[i:i + self.sequence_length].reshape(1, self.sequence_length, -1)
            prob = self.model.predict(sequence, verbose=0)[0][0]
            predictions.append(prob)
        
        return np.array(predictions)
    
    def save(self, name="lstm_model"):
        """Save model and scaler"""
        model_file = self.model_path / f"{name}.h5"
        scaler_file = self.model_path / f"{name}_scaler.pkl"
        
        self.model.save(model_file)
        joblib.dump(self.scaler, scaler_file)
        
        logger.info(f"Model saved to {model_file}")
        logger.info(f"Scaler saved to {scaler_file}")
    
    def load(self, name="lstm_model"):
        """Load model and scaler"""
        model_file = self.model_path / f"{name}.h5"
        scaler_file = self.model_path / f"{name}_scaler.pkl"
        
        self.model = keras.models.load_model(model_file)
        self.scaler = joblib.load(scaler_file)
        
        logger.info(f"Model loaded from {model_file}")
        return self


class EnsembleLSTM:
    """
    Ensemble of multiple LSTM models for robust predictions
    """
    
    def __init__(self, n_models=3, sequence_length=60, prediction_horizon=5):
        self.n_models = n_models
        self.models = []
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        for i in range(n_models):
            model = LSTMTradingModel(sequence_length, prediction_horizon)
            self.models.append(model)
    
    def train_ensemble(self, df, epochs=50):
        """Train all models in ensemble with different configurations"""
        configs = [
            {'lstm_units': [64, 32], 'dropout_rate': 0.2, 'learning_rate': 0.001},
            {'lstm_units': [128, 64], 'dropout_rate': 0.3, 'learning_rate': 0.0005},
            {'lstm_units': [96, 48], 'dropout_rate': 0.25, 'learning_rate': 0.0015}
        ]
        
        results = []
        for i, (model, config) in enumerate(zip(self.models, configs)):
            logger.info(f"\nTraining ensemble model {i+1}/{self.n_models}")
            model.build_model(**config)
            history, metrics = model.train(df, epochs=epochs)
            results.append(metrics)
            model.save(f"ensemble_model_{i}")
        
        return results
    
    def predict(self, df):
        """Average predictions from all models"""
        predictions = [model.predict(df) for model in self.models]
        return np.mean(predictions)
    
    def predict_with_confidence(self, df):
        """Return mean prediction and confidence (1 - std)"""
        predictions = [model.predict(df) for model in self.models]
        mean_pred = np.mean(predictions)
        confidence = 1 - np.std(predictions)
        return mean_pred, confidence


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=1000, freq='5min')
    df = pd.DataFrame({
        'open': np.cumsum(np.random.randn(1000)) + 100,
        'high': np.cumsum(np.random.randn(1000)) + 101,
        'low': np.cumsum(np.random.randn(1000)) + 99,
        'close': np.cumsum(np.random.randn(1000)) + 100,
        'volume': np.random.randint(100000, 1000000, 1000)
    }, index=dates)
    
    # Ensure high >= low
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    print("Testing LSTM Trading Model...")
    print("=" * 60)
    
    # Single model
    print("\n1. Training single LSTM model...")
    lstm = LSTMTradingModel(sequence_length=60, prediction_horizon=5)
    history, results = lstm.train(df, epochs=20, batch_size=32)
    
    print(f"\nResults:")
    print(f"  Validation Accuracy: {results['val_accuracy']:.3f}")
    print(f"  Validation AUC: {results['val_auc']:.3f}")
    
    # Make prediction
    prediction = lstm.predict(df)
    print(f"\nLatest prediction probability: {prediction:.3f}")
    
    # Save model
    lstm.save()
    
    # Ensemble model
    print("\n2. Training ensemble model...")
    ensemble = EnsembleLSTM(n_models=3, sequence_length=60, prediction_horizon=5)
    ensemble_results = ensemble.train_ensemble(df, epochs=15)
    
    # Ensemble prediction
    mean_pred, confidence = ensemble.predict_with_confidence(df)
    print(f"\nEnsemble prediction: {mean_pred:.3f} (confidence: {confidence:.3f})")
    
    print("\n" + "=" * 60)
    print("LSTM model training complete!")
    print(f"Models saved to: {lstm.model_path}")
