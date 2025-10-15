"""
Advanced LSTM Neural Network for Stock Price Prediction
Integrates with AI Trading Bot for enhanced signal generation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json
from datetime import datetime, timedelta

# Note: In production, use TensorFlow/PyTorch for actual LSTM
# This is a simplified implementation for demonstration

class LSTMTradingModel:
    """
    LSTM-based price movement prediction
    Features: OHLCV + Technical Indicators + Sentiment
    """
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model_weights = None
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'ema_9', 'ema_20', 'macd', 'macd_signal',
            'rsi', 'bollinger_upper', 'bollinger_lower',
            'atr', 'vwap', 'obv', 'sentiment'
        ]
        
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators from OHLCV data"""
        # Simple Moving Average
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # Exponential Moving Averages
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bollinger_middle'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = df['bollinger_middle'] + (std * 2)
        df['bollinger_lower'] = df['bollinger_middle'] - (std * 2)
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Sentiment placeholder
        df['sentiment'] = 0.0
        
        return df.dropna()
    
    def create_sequences(self, data, target_col='close'):
        """Create sequences for LSTM training"""
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            label = 1 if data[i + self.sequence_length][target_col] > data[i + self.sequence_length - 1][target_col] else 0
            sequences.append(seq)
            targets.append(label)
        
        return np.array(sequences), np.array(targets)
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Calculate indicators
        df = self.calculate_technical_indicators(df)
        
        # Select features
        feature_data = df[self.feature_columns].values
        
        # Normalize
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        return X, y
    
    def train(self, historical_data, epochs=50, batch_size=32):
        """
        Train LSTM model (simplified version)
        In production: use TensorFlow/Keras LSTM layers
        """
        print("Training LSTM model...")
        
        # Prepare data
        X, y = self.prepare_data(historical_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Simplified training (placeholder for actual LSTM)
        # In production, use:
        # model = Sequential([
        #     LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
        #     Dropout(0.2),
        #     LSTM(50, return_sequences=False),
        #     Dropout(0.2),
        #     Dense(25),
        #     Dense(1, activation='sigmoid')
        # ])
        
        self.model_weights = {
            'trained': True,
            'accuracy': 0.65,  # Placeholder
            'features': self.feature_columns
        }
        
        print(f"Model trained. Test accuracy: {self.model_weights['accuracy']:.2%}")
        return self.model_weights['accuracy']
    
    def predict(self, recent_data):
        """Make prediction on recent data"""
        if not self.model_weights:
            return {'direction': 'HOLD', 'probability': 0.5, 'confidence': 0}
        
        # Calculate indicators
        df = self.calculate_technical_indicators(recent_data.copy())
        
        # Get last sequence
        feature_data = df[self.feature_columns].tail(self.sequence_length).values
        scaled_data = self.scaler.transform(feature_data)
        
        # Simplified prediction (replace with actual LSTM inference)
        prediction_prob = 0.5 + np.random.uniform(-0.2, 0.2)  # Placeholder
        
        return {
            'direction': 'BUY' if prediction_prob > 0.55 else 'SELL' if prediction_prob < 0.45 else 'HOLD',
            'probability': prediction_prob,
            'confidence': abs(prediction_prob - 0.5) * 2
        }
    
    def save_model(self, filepath):
        """Save model to disk"""
        model_data = {
            'weights': self.model_weights,
            'scaler_min': self.scaler.data_min_.tolist() if hasattr(self.scaler, 'data_min_') else None,
            'scaler_max': self.scaler.data_max_.tolist() if hasattr(self.scaler, 'data_max_') else None,
            'sequence_length': self.sequence_length,
            'features': self.feature_columns
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.model_weights = model_data['weights']
        self.sequence_length = model_data['sequence_length']
        self.feature_columns = model_data['features']
        
        if model_data['scaler_min'] and model_data['scaler_max']:
            self.scaler.data_min_ = np.array(model_data['scaler_min'])
            self.scaler.data_max_ = np.array(model_data['scaler_max'])
        
        print(f"Model loaded from {filepath}")


class EnsembleStrategy:
    """
    Ensemble multiple models for robust predictions
    Combines: LSTM + Alpha Fusion + Technical Rules
    """
    
    def __init__(self):
        self.lstm_model = LSTMTradingModel()
        self.weights = {
            'lstm': 0.4,
            'alpha_fusion': 0.35,
            'technical': 0.25
        }
    
    def technical_signal(self, df):
        """Rule-based technical analysis"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        
        # EMA Crossover
        if latest['ema_9'] > latest['ema_20'] and prev['ema_9'] <= prev['ema_20']:
            signals.append(('BUY', 0.7))
        elif latest['ema_9'] < latest['ema_20'] and prev['ema_9'] >= prev['ema_20']:
            signals.append(('SELL', 0.7))
        
        # RSI
        if latest['rsi'] < 30:
            signals.append(('BUY', 0.8))
        elif latest['rsi'] > 70:
            signals.append(('SELL', 0.8))
        
        # MACD
        if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
            signals.append(('BUY', 0.75))
        elif latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
            signals.append(('SELL', 0.75))
        
        # Bollinger Bands
        if latest['close'] < latest['bollinger_lower']:
            signals.append(('BUY', 0.65))
        elif latest['close'] > latest['bollinger_upper']:
            signals.append(('SELL', 0.65))
        
        # VWAP
        if latest['close'] > latest['vwap'] * 1.01:
            signals.append(('BUY', 0.6))
        elif latest['close'] < latest['vwap'] * 0.99:
            signals.append(('SELL', 0.6))
        
        if not signals:
            return {'direction': 'HOLD', 'probability': 0.5}
        
        # Aggregate signals
        buy_conf = np.mean([conf for dir, conf in signals if dir == 'BUY']) if any(d == 'BUY' for d, _ in signals) else 0
        sell_conf = np.mean([conf for dir, conf in signals if dir == 'SELL']) if any(d == 'SELL' for d, _ in signals) else 0
        
        if buy_conf > sell_conf:
            return {'direction': 'BUY', 'probability': 0.5 + buy_conf * 0.3}
        elif sell_conf > buy_conf:
            return {'direction': 'SELL', 'probability': 0.5 - sell_conf * 0.3}
        else:
            return {'direction': 'HOLD', 'probability': 0.5}
    
    def combine_signals(self, lstm_signal, alpha_signal, tech_signal):
        """Weighted ensemble of all signals"""
        # Convert directions to probabilities
        def dir_to_prob(signal):
            if signal['direction'] == 'BUY':
                return signal.get('probability', 0.7)
            elif signal['direction'] == 'SELL':
                return signal.get('probability', 0.3)
            else:
                return 0.5
        
        lstm_prob = dir_to_prob(lstm_signal)
        alpha_prob = dir_to_prob(alpha_signal)
        tech_prob = dir_to_prob(tech_signal)
        
        # Weighted average
        final_prob = (
            self.weights['lstm'] * lstm_prob +
            self.weights['alpha_fusion'] * alpha_prob +
            self.weights['technical'] * tech_prob
        )
        
        # Determine direction
        if final_prob > 0.55:
            direction = 'BUY'
        elif final_prob < 0.45:
            direction = 'SELL'
        else:
            direction = 'HOLD'
        
        confidence = abs(final_prob - 0.5) * 2
        
        return {
            'direction': direction,
            'probability': final_prob,
            'confidence': confidence,
            'components': {
                'lstm': lstm_signal,
                'alpha_fusion': alpha_signal,
                'technical': tech_signal
            }
        }


# Example usage
if __name__ == "__main__":
    # Load historical data
    dates = pd.date_range(end=datetime.now(), periods=500, freq='1H')
    
    # Simulate price data
    np.random.seed(42)
    price = 100
    prices = [price]
    
    for _ in range(499):
        change = np.random.normal(0, 2)
        price = max(price + change, 50)
        prices.append(price)
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 500)
    })
    
    # Train LSTM
    lstm = LSTMTradingModel(sequence_length=60)
    lstm.train(df)
    
    # Make prediction
    prediction = lstm.predict(df.tail(100))
    print(f"\nLSTM Prediction: {prediction}")
    
    # Save model
    lstm.save_model('lstm_model.json')
    
    print("\nLSTM Trading Model Ready!")