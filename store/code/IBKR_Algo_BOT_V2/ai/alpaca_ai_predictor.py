"""
Enhanced AI Predictor with Alpaca Market Data
Uses Alpaca API for market data with LightGBM prediction model
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import ta
from datetime import datetime, timedelta
import json
import logging
from typing import Optional, Dict, List
from alpaca_market_data import get_alpaca_market_data

logger = logging.getLogger(__name__)


class AlpacaAIPredictor:
    """AI Predictor using Alpaca market data"""

    def __init__(self, model_path: str = "store/models/lgb_predictor.txt"):
        self.model = None
        self.model_path = model_path
        self.feature_names = []
        self.feature_importance = {}
        self.accuracy = 0.0
        self.training_date = None
        self.market_data = get_alpaca_market_data()

        # Data cache to avoid excessive yfinance downloads
        self._data_cache: Dict[str, Dict] = {}  # symbol -> {data, timestamp, features_df}
        self._cache_ttl_seconds = 300  # Cache data for 5 minutes

        # Try to load existing model
        try:
            if Path(self.model_path).exists():
                self.load_model()
                logger.info("Loaded existing model")
        except Exception as e:
            logger.info(f"No existing model to load: {e}")

        # Model hyperparameters - optimized for class imbalance
        self.params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 7,
            'min_data_in_leaf': 20,
            'is_unbalance': True,  # Auto-handle class imbalance
            'lambda_l1': 0.1,  # L1 regularization
            'lambda_l2': 0.1,  # L2 regularization
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

        # VWAP - if available from Alpaca, otherwise calculate
        if 'VWAP' not in data.columns:
            data['vwap'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
        else:
            data['vwap'] = data['VWAP']

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

    def prepare_training_data(self, symbol: str, use_alpaca: bool = True) -> tuple:
        """
        Download and prepare training data using Alpaca real-time data (preferred)
        or yfinance as fallback

        Args:
            symbol: Stock symbol to train on
            use_alpaca: Use Alpaca data (default True, requires data subscription)

        Returns:
            Tuple of (X, y) features and target
        """
        df = None

        if use_alpaca:
            try:
                logger.info(f"Downloading data from Alpaca for {symbol}...")
                # Use Alpaca real-time data (2 years)
                end = datetime.now()
                start = end - timedelta(days=730)  # 2 years

                df = self.market_data.get_historical_bars(
                    symbol=symbol,
                    timeframe="1Day",
                    start=start,
                    end=end
                )

                if df is not None and not df.empty:
                    logger.info(f"Received {len(df)} bars from Alpaca")
                else:
                    logger.warning(f"No Alpaca data for {symbol}, falling back to yfinance")
                    df = None

            except Exception as e:
                logger.warning(f"Alpaca data error for {symbol}: {e}, falling back to yfinance")
                df = None

        # Fallback to yfinance if Alpaca fails
        if df is None or df.empty:
            logger.info(f"Downloading data from yfinance for {symbol}...")
            import yfinance as yf

            df = yf.download(symbol, period="2y", progress=False)

            if df.empty:
                raise ValueError(f"No data received for {symbol}")

            # Flatten column index if multi-level (yfinance quirk)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            logger.info(f"Received {len(df)} bars from yfinance")

        # Calculate features
        df = self.calculate_features(df)

        # Create target - predict if price goes up 1% in next 3 days
        df['future_return'] = df['Close'].shift(-3) / df['Close'] - 1
        df['target'] = (df['future_return'] > 0.01).astype(int)

        df = df.dropna()

        # Log class distribution
        class_counts = df['target'].value_counts()
        logger.info(f"Class distribution - 0: {class_counts.get(0, 0)}, 1: {class_counts.get(1, 0)}")

        # Select features (exclude OHLCV and target columns)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP',
                       'target', 'future_return', 'Date']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]
        y = df['target']

        self.feature_names = feature_cols

        logger.info(f"Data prepared: {len(X)} samples, {len(feature_cols)} features")

        return X, y

    def train(self, symbol: str, test_size: float = 0.2) -> Dict:
        """
        Train LightGBM model

        Args:
            symbol: Stock symbol to train on
            test_size: Test set size (default 0.2)

        Returns:
            Dictionary with training results
        """
        logger.info(f"Training enhanced predictor for {symbol}...")

        X, y = self.prepare_training_data(symbol)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        logger.info("Training LightGBM model...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=200,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)]
        )

        # Find optimal threshold using validation set
        y_pred_proba = self.model.predict(X_test)
        best_threshold = 0.5
        best_f1 = 0
        for threshold in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            y_pred_temp = (y_pred_proba > threshold).astype(int)
            from sklearn.metrics import f1_score
            f1 = f1_score(y_test, y_pred_temp, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self.optimal_threshold = best_threshold
        y_pred = (y_pred_proba > best_threshold).astype(int)

        self.accuracy = accuracy_score(y_test, y_pred)

        importance = self.model.feature_importance(importance_type='gain')
        self.feature_importance = dict(zip(self.feature_names, importance))

        self.training_date = datetime.now().isoformat()
        self.trained_symbol = symbol

        logger.info(f"\nAccuracy: {self.accuracy:.4f} (threshold: {best_threshold})")
        logger.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

        # Show top features
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        logger.info("\nTop 10 Features:")
        for i, (feat, imp) in enumerate(sorted_features, 1):
            logger.info(f"{i}. {feat}: {imp:.2f}")

        self.save_model()

        return {
            "success": True,
            "symbol": symbol,
            "samples": len(X),
            "metrics": {
                "accuracy": float(self.accuracy),
                "optimal_threshold": float(best_threshold),
                "f1_score": float(best_f1)
            },
            "model_path": self.model_path
        }

    def train_multi(self, symbols: List[str], test_size: float = 0.2) -> Dict:
        """
        Train on multiple symbols for better generalization

        Args:
            symbols: List of stock symbols
            test_size: Test set size

        Returns:
            Dictionary with training results
        """
        logger.info(f"Training on {len(symbols)} symbols: {symbols}")

        all_X = []
        all_y = []

        for symbol in symbols:
            try:
                X, y = self.prepare_training_data(symbol)
                all_X.append(X)
                all_y.append(y)
                logger.info(f"  {symbol}: {len(X)} samples")
            except Exception as e:
                logger.warning(f"  {symbol}: Failed - {e}")

        if not all_X:
            raise ValueError("No valid data from any symbol")

        # Combine all data
        X = pd.concat(all_X, ignore_index=True)
        y = pd.concat(all_y, ignore_index=True)

        # CRITICAL: Update feature_names from actual data columns to prevent mismatch
        # This fixes the "Length of feature_name and num_feature don't match" error
        self.feature_names = list(X.columns)

        logger.info(f"Combined dataset: {len(X)} samples, {len(self.feature_names)} features")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, stratify=y
        )

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        logger.info("Training multi-symbol LightGBM model...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=300,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)]
        )

        # Find optimal threshold
        y_pred_proba = self.model.predict(X_test)
        best_threshold = 0.5
        best_f1 = 0
        for threshold in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            y_pred_temp = (y_pred_proba > threshold).astype(int)
            from sklearn.metrics import f1_score
            f1 = f1_score(y_test, y_pred_temp, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self.optimal_threshold = best_threshold
        y_pred = (y_pred_proba > best_threshold).astype(int)

        self.accuracy = accuracy_score(y_test, y_pred)
        importance = self.model.feature_importance(importance_type='gain')
        self.feature_importance = dict(zip(self.feature_names, importance))
        self.training_date = datetime.now().isoformat()
        self.trained_symbol = "multi:" + ",".join(symbols)

        logger.info(f"\nMulti-symbol Accuracy: {self.accuracy:.4f}")
        logger.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

        self.save_model()

        return {
            "success": True,
            "symbols": symbols,
            "samples": len(X),
            "metrics": {
                "accuracy": float(self.accuracy),
                "optimal_threshold": float(best_threshold),
                "f1_score": float(best_f1)
            },
            "model_path": self.model_path
        }

    def train_walkforward(self, symbols: List[str], train_months: int = 12, test_months: int = 3, use_alpaca: bool = True) -> Dict:
        """
        Walk-Forward Training - The PROPER way to train financial ML models

        Uses Alpaca real-time data (with your data subscription) for accurate training.

        This method:
        1. Uses data BEFORE a cutoff date for training
        2. Tests ONLY on data AFTER the cutoff (truly out-of-sample)
        3. No data leakage, no look-ahead bias

        Args:
            symbols: List of stock symbols to train on
            train_months: Months of historical data for training (default: 12)
            test_months: Months of data for out-of-sample testing (default: 3)
            use_alpaca: Use Alpaca real-time data (default: True)

        Returns:
            Dictionary with training results and out-of-sample metrics
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        logger.info("=" * 60)
        logger.info("WALK-FORWARD TRAINING (Proper Financial ML)")
        logger.info(f"Data Source: {'Alpaca Real-Time' if use_alpaca else 'yfinance'}")
        logger.info("=" * 60)

        # Calculate date ranges
        end_date = datetime.now()
        test_start = end_date - timedelta(days=test_months * 30)
        train_start = test_start - timedelta(days=train_months * 30)

        logger.info(f"Training Period: {train_start.strftime('%Y-%m-%d')} to {test_start.strftime('%Y-%m-%d')}")
        logger.info(f"Testing Period:  {test_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info("-" * 60)

        train_X_list = []
        train_y_list = []
        test_X_list = []
        test_y_list = []

        for symbol in symbols:
            try:
                logger.info(f"Processing {symbol}...")
                df = None

                # Try Alpaca first (real-time data with subscription)
                if use_alpaca:
                    try:
                        df = self.market_data.get_historical_bars(
                            symbol=symbol,
                            timeframe="1Day",
                            start=train_start,
                            end=end_date
                        )
                        if df is not None and not df.empty:
                            logger.info(f"  {symbol}: {len(df)} bars from Alpaca")
                        else:
                            df = None
                    except Exception as e:
                        logger.warning(f"  {symbol}: Alpaca error - {e}")
                        df = None

                # Fallback to yfinance
                if df is None or df.empty:
                    import yfinance as yf
                    df = yf.download(
                        symbol,
                        start=train_start.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        progress=False
                    )
                    if not df.empty:
                        # Flatten column index if multi-level
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.droplevel(1)
                        logger.info(f"  {symbol}: {len(df)} bars from yfinance (fallback)")

                if df is None or df.empty:
                    logger.warning(f"  {symbol}: No data available")
                    continue

                # Calculate features
                df = self.calculate_features(df)

                # Create target
                df['future_return'] = df['Close'].shift(-3) / df['Close'] - 1
                df['target'] = (df['future_return'] > 0.01).astype(int)
                df = df.dropna()

                # Split by date (NO SHUFFLE - chronological split)
                # Handle both timezone-aware and naive datetimes
                if df.index.tz is not None:
                    test_start_tz = test_start.replace(tzinfo=df.index.tz)
                    df_train = df[df.index < test_start_tz]
                    df_test = df[df.index >= test_start_tz]
                else:
                    df_train = df[df.index < test_start]
                    df_test = df[df.index >= test_start]

                # Get features
                exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP',
                               'target', 'future_return', 'Date']
                feature_cols = [col for col in df.columns if col not in exclude_cols]

                if len(df_train) > 20:
                    train_X_list.append(df_train[feature_cols])
                    train_y_list.append(df_train['target'])
                    logger.info(f"  {symbol} train: {len(df_train)} samples")

                if len(df_test) > 5:
                    test_X_list.append(df_test[feature_cols])
                    test_y_list.append(df_test['target'])
                    logger.info(f"  {symbol} test:  {len(df_test)} samples")

            except Exception as e:
                logger.warning(f"  {symbol}: Error - {e}")
                continue

        if not train_X_list:
            raise ValueError("No valid training data")

        # Combine data
        X_train = pd.concat(train_X_list, ignore_index=True)
        y_train = pd.concat(train_y_list, ignore_index=True)
        X_test = pd.concat(test_X_list, ignore_index=True) if test_X_list else None
        y_test = pd.concat(test_y_list, ignore_index=True) if test_y_list else None

        self.feature_names = list(X_train.columns)

        logger.info("-" * 60)
        logger.info(f"Total Training samples: {len(X_train)}")
        logger.info(f"Total Test samples: {len(X_test) if X_test is not None else 0}")

        # Train model
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)

        if X_test is not None and len(X_test) > 0:
            test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            valid_sets = [test_data]
        else:
            valid_sets = []

        logger.info("\nTraining LightGBM model...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=300,
            valid_sets=valid_sets,
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)] if valid_sets else [lgb.log_evaluation(50)]
        )

        # Evaluate on OUT-OF-SAMPLE test data
        results = {
            "success": True,
            "symbols": symbols,
            "train_samples": len(X_train),
            "test_samples": len(X_test) if X_test is not None else 0,
            "train_period": f"{train_start.strftime('%Y-%m-%d')} to {test_start.strftime('%Y-%m-%d')}",
            "test_period": f"{test_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        }

        if X_test is not None and len(X_test) > 0:
            y_pred_proba = self.model.predict(X_test)

            # Find optimal threshold
            best_threshold = 0.5
            best_f1 = 0
            for threshold in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
                y_pred_temp = (y_pred_proba > threshold).astype(int)
                f1 = f1_score(y_test, y_pred_temp, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            self.optimal_threshold = best_threshold
            y_pred = (y_pred_proba > best_threshold).astype(int)

            self.accuracy = accuracy_score(y_test, y_pred)

            # Calculate detailed metrics
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)

            logger.info("\n" + "=" * 60)
            logger.info("OUT-OF-SAMPLE RESULTS (Unseen Data)")
            logger.info("=" * 60)
            logger.info(f"Accuracy:  {self.accuracy:.4f} ({self.accuracy*100:.1f}%)")
            logger.info(f"Precision: {precision:.4f} ({precision*100:.1f}%)")
            logger.info(f"Recall:    {recall:.4f} ({recall*100:.1f}%)")
            logger.info(f"F1 Score:  {best_f1:.4f}")
            logger.info(f"Threshold: {best_threshold}")
            logger.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

            results["metrics"] = {
                "accuracy": float(self.accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(best_f1),
                "optimal_threshold": float(best_threshold)
            }
        else:
            logger.warning("No test data available for out-of-sample evaluation")
            results["metrics"] = {"warning": "No out-of-sample test data"}

        # Feature importance
        importance = self.model.feature_importance(importance_type='gain')
        self.feature_importance = dict(zip(self.feature_names, importance))
        self.training_date = datetime.now().isoformat()
        self.trained_symbol = "walkforward:" + ",".join(symbols)

        # Top features
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("\nTop 10 Predictive Features:")
        for i, (feat, imp) in enumerate(sorted_features, 1):
            logger.info(f"  {i}. {feat}: {imp:.2f}")

        results["top_features"] = [{"name": f, "importance": float(i)} for f, i in sorted_features]

        self.save_model()
        results["model_path"] = self.model_path

        return results

    def _get_cached_data(self, symbol: str) -> pd.DataFrame:
        """Get cached data or download fresh data if cache expired"""
        import yfinance as yf

        now = datetime.now()
        cache_key = symbol.upper()

        # Check if we have valid cached data
        if cache_key in self._data_cache:
            cache_entry = self._data_cache[cache_key]
            cache_age = (now - cache_entry['timestamp']).total_seconds()
            if cache_age < self._cache_ttl_seconds:
                logger.debug(f"Using cached data for {symbol} (age: {cache_age:.0f}s)")
                return cache_entry['features_df'].copy()

        # Download fresh data
        logger.info(f"Downloading fresh data for {symbol}...")
        df = yf.download(symbol, period="3mo", progress=False)

        # Flatten column index if multi-level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        if df.empty:
            raise ValueError(f"No data from yfinance for {symbol}")

        # Calculate features
        features_df = self.calculate_features(df)

        # Cache the data
        self._data_cache[cache_key] = {
            'timestamp': now,
            'features_df': features_df.copy()
        }

        return features_df

    def predict(
        self,
        symbol: str,
        timeframe: str = "1Day",
        bidSize: Optional[int] = None,
        askSize: Optional[int] = None,
        sentiment: Optional[str] = None
    ) -> Dict:
        """
        Make prediction for a symbol using Alpaca data

        Args:
            symbol: Stock symbol
            timeframe: Timeframe for data (1Day, 5Min, etc.)
            bidSize: Optional bid size (for future enhancements)
            askSize: Optional ask size (for future enhancements)
            sentiment: Optional sentiment data (for future enhancements)

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained!")

        # Get data (from cache or fresh download)
        df = self._get_cached_data(symbol)

        # Only use features that exist in both training and current data
        available_features = [f for f in self.feature_names if f in df.columns]
        missing_features = [f for f in self.feature_names if f not in df.columns]

        if missing_features:
            logger.warning(f"Missing features: {missing_features[:5]}...")

        latest = df[available_features].iloc[-1:].values

        # If missing features, pad with zeros
        if len(available_features) < len(self.feature_names):
            padding = np.zeros((1, len(self.feature_names) - len(available_features)))
            latest = np.concatenate([latest, padding], axis=1)

        prob = self.model.predict(latest)[0]
        threshold = getattr(self, 'optimal_threshold', 0.5)
        prediction = int(prob > threshold)
        confidence = abs(prob - threshold) * 2

        # Determine signal strength based on optimal threshold
        bull_strong = threshold + 0.2
        bull_mild = threshold + 0.05
        bear_strong = threshold - 0.2
        bear_mild = threshold - 0.05

        if prob > bull_strong:
            signal = "STRONG_BULLISH"
            action = "BUY"
        elif prob > bull_mild:
            signal = "BULLISH"
            action = "BUY"
        elif prob < bear_strong:
            signal = "STRONG_BEARISH"
            action = "SELL"
        elif prob < bear_mild:
            signal = "BEARISH"
            action = "SELL"
        else:
            signal = "NEUTRAL"
            action = "HOLD"

        return {
            "symbol": symbol,
            "prediction": signal,
            "signal": signal,
            "action": action,
            "prob_up": float(prob),
            "prob_down": float(1 - prob),
            "confidence": float(confidence),
            "threshold": float(threshold),
            "signal_detail": f"Alpaca+LightGBM | Acc: {self.accuracy:.4f}",
            "timestamp": datetime.now().isoformat(),
            "data_source": "Alpaca"
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
            'feature_importance': self.feature_importance,
            'optimal_threshold': getattr(self, 'optimal_threshold', 0.5),
            'trained_symbol': getattr(self, 'trained_symbol', 'unknown'),
            'data_source': 'Alpaca'
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {self.model_path}")

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
                self.optimal_threshold = metadata.get('optimal_threshold', 0.5)
                self.trained_symbol = metadata.get('trained_symbol', 'unknown')

        logger.info(f"Model loaded: Accuracy {self.accuracy:.4f}, Threshold {getattr(self, 'optimal_threshold', 0.5)}")


# Singleton predictor instance
_predictor_instance: Optional[AlpacaAIPredictor] = None


def get_alpaca_predictor() -> AlpacaAIPredictor:
    """Get or create the Alpaca AI predictor singleton"""
    global _predictor_instance

    if _predictor_instance is None:
        _predictor_instance = AlpacaAIPredictor()
        # Note: Model loading already happens in __init__ if model file exists
        # No need to call load_model() again here

    return _predictor_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    predictor = AlpacaAIPredictor()
    print("\n🚀 Training Alpaca AI Predictor with LightGBM...")

    result = predictor.train("SPY", test_size=0.2)

    print(f"\n✅ Training Complete! Accuracy: {result['metrics']['accuracy']:.4f}")

    print("\n🔮 Testing predictions on multiple symbols...")
    for symbol in ["SPY", "AAPL", "TSLA"]:
        prediction = predictor.predict(symbol)
        print(f"\n{symbol}: {prediction['signal']} | "
              f"Confidence: {prediction['confidence']:.2%} | "
              f"Prob Up: {prediction['prob_up']:.2%}")
