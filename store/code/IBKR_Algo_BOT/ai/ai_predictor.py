"""
Enhanced AI Predictor with LightGBM and Advanced Technical Indicators
Upgraded: 2025-10-24
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import yfinance as yf
import ta
from datetime import datetime
import json
import logging
from pathlib import Path

class EnhancedAIPredictor:
    def __init__(self, model_path: str = "store/models/lgb_predictor.txt"):
        self.model = None
        self.model_path = model_path
        self.feature_names = []
        self.feature_importance = {}
        self.accuracy = 0.0
        self.training_date = None
        self.logger = logging.getLogger(__name__)
        
        # Model hyperparameters - optimized for binary classification
        self.params = {
            'objective': 'binary',
            'metric': 'auc',  # Use AUC instead of logloss for imbalanced data
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 7,
            'min_data_in_leaf': 20,
            'scale_pos_weight': 5.0  # Give more weight to positive class
        }
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        data = df.copy()
        
        # Moving Averages
        data['sma_10'] = ta.trend.sma_indicator(data['Close'], window=10)
        data['sma_20'] = ta.trend.sma_indicator(data['Close'], window=20)
        data['sma_50'] = ta.trend.sma_indicator(data['Close'], window=50)
        data['ema_12'] = ta.trend.ema_indicator(data['Close'], window=12)
        data['ema_26'] = ta.trend.ema_indicator(data['Close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(data['Close'])
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_diff'] = macd.macd_diff()
        
        # ADX - Trend Strength
        adx = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14)
        data['adx'] = adx.adx()
        data['adx_pos'] = adx.adx_pos()
        data['adx_neg'] = adx.adx_neg()
        
        # RSI
        data['rsi'] = ta.momentum.rsi(data['Close'], window=14)
        data['rsi_fast'] = ta.momentum.rsi(data['Close'], window=7)
        
        # ROC - Rate of Change
        data['roc'] = ta.momentum.roc(data['Close'], window=12)
        data['roc_fast'] = ta.momentum.roc(data['Close'], window=6)
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['Close'])
        data['bb_high'] = bollinger.bollinger_hband()
        data['bb_low'] = bollinger.bollinger_lband()
        data['bb_width'] = bollinger.bollinger_wband()
        
        # ATR
        data['atr'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
        
        # VWAP - Volume Weighted Average Price
        data['vwap'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
        
        # OBV - On Balance Volume
        data['obv'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        
        # MFI - Money Flow Index
        data['mfi'] = ta.volume.money_flow_index(data['High'], data['Low'], data['Close'], data['Volume'])
        
        # Price momentum
        data['momentum_1'] = data['Close'].pct_change(1)
        data['momentum_3'] = data['Close'].pct_change(3)
        data['momentum_5'] = data['Close'].pct_change(5)
        
        # Volatility
        data['volatility_10'] = data['Close'].pct_change().rolling(10).std()
        data['volatility_20'] = data['Close'].pct_change().rolling(20).std()
        
        # Volume analysis
        data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        
        # Price relative to indicators
        data['price_to_sma20'] = data['Close'] / data['sma_20']
        data['price_to_sma50'] = data['Close'] / data['sma_50']
        data['price_to_vwap'] = data['Close'] / data['vwap']
        
        # Crossovers
        data['sma_cross_10_20'] = (data['sma_10'] > data['sma_20']).astype(int)
        data['ema_cross'] = (data['ema_12'] > data['ema_26']).astype(int)
        
        # Drop NaN
        data = data.dropna()
        
        return data
    
    def prepare_training_data(self, symbol: str, period: str = "2y"):
        """Download and prepare training data"""
        self.logger.info(f"Downloading data for {symbol}...")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data for {symbol}")
        
        df = self.calculate_features(df)
        
        # Create target - predict if price goes up 1% in next 3 days
        df['future_return'] = df['Close'].shift(-3) / df['Close'] - 1
        df['target'] = (df['future_return'] > 0.01).astype(int)
        
        df = df.dropna()
        
        # Log class distribution
        class_counts = df['target'].value_counts()
        self.logger.info(f"Class distribution - 0: {class_counts.get(0, 0)}, 1: {class_counts.get(1, 0)}")
        
        # Select features
        feature_cols = [col for col in df.columns if col not in 
                       ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 
                        'Stock Splits', 'target', 'future_return']]
        
        X = df[feature_cols]
        y = df['target']
        
        self.feature_names = feature_cols
        
        self.logger.info(f"Data prepared: {len(X)} samples, {len(feature_cols)} features")
        
        return X, y
    
    def train(self, symbol: str, test_size: float = 0.2):
        """Train LightGBM model"""
        self.logger.info(f"Training enhanced predictor for {symbol}...")
        
        X, y = self.prepare_training_data(symbol)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        self.logger.info("Training LightGBM model...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=200,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)]
        )
        
        # Predict with optimal threshold
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.4).astype(int)  # Lower threshold for imbalanced data
        
        self.accuracy = accuracy_score(y_test, y_pred)
        
        importance = self.model.feature_importance(importance_type='gain')
        self.feature_importance = dict(zip(self.feature_names, importance))
        
        self.training_date = datetime.now().isoformat()
        
        self.logger.info(f"\nAccuracy: {self.accuracy:.4f}")
        self.logger.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")
        
        # Show top features
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        self.logger.info("\nTop 10 Features:")
        for i, (feat, imp) in enumerate(sorted_features, 1):
            self.logger.info(f"{i}. {feat}: {imp:.2f}")
        
        return self.accuracy
    
    def predict(self, symbol: str, period: str = "3mo") -> dict:
        """Make prediction for a symbol"""
        if self.model is None:
            raise ValueError("Model not trained!")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data for {symbol}")
        
        df = self.calculate_features(df)
        
        # Only use features that exist in both training and current data
        available_features = [f for f in self.feature_names if f in df.columns]
        missing_features = [f for f in self.feature_names if f not in df.columns]
        
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features[:5]}...")
        
        latest = df[available_features].iloc[-1:].values
        
        # If we're missing features, pad with zeros
        if len(available_features) < len(self.feature_names):
            padding = np.zeros((1, len(self.feature_names) - len(available_features)))
            latest = np.concatenate([latest, padding], axis=1)
        
        prob = self.model.predict(latest)[0]
        prediction = int(prob > 0.4)  # Use same threshold as training
        confidence = abs(prob - 0.5) * 2
        
        if prob > 0.7:
            signal = "STRONG_BULLISH"
        elif prob > 0.55:
            signal = "BULLISH"
        elif prob < 0.3:
            signal = "STRONG_BEARISH"
        elif prob < 0.45:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"
        
        return {
            "symbol": symbol,
            "prediction": prediction,
            "signal": signal,
            "prob_up": float(prob),
            "prob_down": float(1 - prob),
            "confidence": float(confidence),
            "signal_detail": f"LightGBM | Acc: {self.accuracy:.4f}",
            "timestamp": datetime.now().isoformat()
        }
    
    def save_model(self):
        """Save model and metadata"""
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(self.model_path)
        
        meta_path = self.model_path.replace('.txt', '_meta.json')
        metadata = {
            'accuracy': self.accuracy,
            'training_date': self.training_date,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load model and metadata"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = lgb.Booster(model_file=self.model_path)
        
        meta_path = self.model_path.replace('.txt', '_meta.json')
        if Path(meta_path).exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                self.accuracy = metadata.get('accuracy', 0.0)
                self.training_date = metadata.get('training_date')
                self.feature_names = metadata.get('feature_names', [])
                self.feature_importance = metadata.get('feature_importance', {})
        
        self.logger.info(f"Model loaded: Accuracy {self.accuracy:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    predictor = EnhancedAIPredictor()
    print("\n🚀 Training Enhanced AI Predictor with LightGBM...")
    
    accuracy = predictor.train("SPY", test_size=0.2)
    
    print(f"\n✅ Training Complete! Accuracy: {accuracy:.4f}")
    
    predictor.save_model()
    
    print("\n🔮 Testing predictions on multiple symbols...")
    for symbol in ["SPY", "AAPL", "TSLA"]:
        result = predictor.predict(symbol)
        print(f"\n{symbol}: {result['signal']} | Confidence: {result['confidence']:.2%} | Prob Up: {result['prob_up']:.2%}")

# Singleton predictor instance
_predictor_instance = None

def get_predictor():
    """Get or create the AI predictor singleton"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = EnhancedAIPredictor()
        try:
            _predictor_instance.load_model()
        except:
            pass  # Model not trained yet
    return _predictor_instance
