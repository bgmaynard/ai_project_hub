"""
Qlib Alpha158 Predictor
=======================
Uses Microsoft Qlib's Alpha158 factor library for stock scoring.
Alpha158 provides 158 pre-computed technical indicators that have been
proven effective in quantitative trading research.

This module:
- Computes Alpha158 features from OHLCV data
- Uses LightGBM for prediction (trained on computed features)
- Provides stock ranking scores for the ensemble predictor

No external data provider needed - works directly with yfinance data.

Author: Morpheus Trading Bot
Created: December 2024
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Lazy load heavy imports
_model = None
_feature_calculator = None


@dataclass
class QlibPrediction:
    """Result from Qlib prediction"""

    symbol: str
    timestamp: str
    score: float  # 0.0 to 1.0 (bearish to bullish)
    rank_percentile: float  # Relative ranking vs other stocks
    signal: str  # STRONG_BULLISH, BULLISH, NEUTRAL, BEARISH, STRONG_BEARISH
    confidence: float
    top_features: List[Tuple[str, float]]  # Top contributing features
    feature_count: int  # Number of valid features computed


class Alpha158Calculator:
    """
    Computes Alpha158 features from OHLCV data.

    This is a standalone implementation that doesn't require Qlib's
    data provider infrastructure. Features are computed using pandas/numpy.
    """

    # Feature windows used in Alpha158
    WINDOWS = [5, 10, 20, 30, 60]

    def __init__(self):
        self.feature_names = []
        self._build_feature_names()

    def _build_feature_names(self):
        """Build list of all feature names"""
        # Price-based features
        for w in self.WINDOWS:
            self.feature_names.extend(
                [
                    f"ROC_{w}",  # Rate of change
                    f"MA_{w}",  # Moving average ratio
                    f"STD_{w}",  # Standard deviation
                    f"BETA_{w}",  # Beta vs market
                    f"MAX_{w}",  # Max ratio
                    f"MIN_{w}",  # Min ratio
                    f"QTLU_{w}",  # Upper quantile
                    f"QTLD_{w}",  # Lower quantile
                    f"RANK_{w}",  # Rank in window
                    f"RSV_{w}",  # Raw stochastic value
                    f"CORR_{w}",  # Correlation with volume
                    f"CORD_{w}",  # Correlation delta
                ]
            )

        # Volume-based features
        for w in self.WINDOWS:
            self.feature_names.extend(
                [
                    f"VMA_{w}",  # Volume MA ratio
                    f"VSTD_{w}",  # Volume std
                    f"WVMA_{w}",  # Weighted volume MA
                ]
            )

        # Cross-sectional features
        self.feature_names.extend(
            [
                "KMID",
                "KLEN",
                "KMID2",
                "KUP",
                "KUP2",
                "KLOW",
                "KLOW2",
                "KSFT",
                "KSFT2",
                "OPEN_RATIO",
                "HIGH_RATIO",
                "LOW_RATIO",
                "CLOSE_RATIO",
                "VWAP_RATIO",
                "TURN",
                "TURN_MA5",
                "TURN_MA10",
            ]
        )

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Alpha158 features from OHLCV DataFrame.

        Args:
            df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
                Index should be datetime

        Returns:
            DataFrame with computed features
        """
        if len(df) < 60:
            logger.warning(f"Need at least 60 bars for Alpha158, got {len(df)}")
            return pd.DataFrame()

        features = {}

        close = df["Close"]
        open_ = df["Open"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # Prevent division by zero
        close_safe = close.replace(0, np.nan)
        volume_safe = volume.replace(0, np.nan)

        # === Price-based features ===
        for w in self.WINDOWS:
            # ROC: Rate of change
            features[f"ROC_{w}"] = close / close.shift(w) - 1

            # MA: Moving average ratio
            ma = close.rolling(w).mean()
            features[f"MA_{w}"] = close / ma - 1

            # STD: Rolling standard deviation (normalized)
            features[f"STD_{w}"] = close.rolling(w).std() / close_safe

            # BETA: Linear regression slope
            features[f"BETA_{w}"] = self._rolling_beta(close, w)

            # MAX: Distance from rolling max
            roll_max = close.rolling(w).max()
            features[f"MAX_{w}"] = close / roll_max - 1

            # MIN: Distance from rolling min
            roll_min = close.rolling(w).min()
            features[f"MIN_{w}"] = close / roll_min - 1

            # QTLU: Upper quantile distance
            qtlu = close.rolling(w).quantile(0.8)
            features[f"QTLU_{w}"] = close / qtlu - 1

            # QTLD: Lower quantile distance
            qtld = close.rolling(w).quantile(0.2)
            features[f"QTLD_{w}"] = close / qtld - 1

            # RANK: Percentile rank in window
            features[f"RANK_{w}"] = close.rolling(w).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )

            # RSV: Raw stochastic value
            features[f"RSV_{w}"] = (close - roll_min) / (roll_max - roll_min + 1e-8)

            # CORR: Price-volume correlation
            features[f"CORR_{w}"] = close.rolling(w).corr(volume)

            # CORD: Correlation delta
            corr = features[f"CORR_{w}"]
            features[f"CORD_{w}"] = corr - corr.shift(w // 2)

        # === Volume-based features ===
        for w in self.WINDOWS:
            # VMA: Volume MA ratio
            vma = volume.rolling(w).mean()
            features[f"VMA_{w}"] = volume / vma - 1

            # VSTD: Volume std (normalized)
            features[f"VSTD_{w}"] = volume.rolling(w).std() / volume_safe

            # WVMA: Weighted volume MA
            wvma = (close * volume).rolling(w).sum() / volume.rolling(w).sum()
            features[f"WVMA_{w}"] = close / wvma - 1

        # === K-line features ===
        # KMID: Middle of bar relative to close
        features["KMID"] = (close - open_) / close_safe

        # KLEN: Bar length relative to close
        features["KLEN"] = (high - low) / close_safe

        # KMID2: Open-close range normalized
        features["KMID2"] = (close - open_) / (high - low + 1e-8)

        # KUP: Upper shadow
        features["KUP"] = (high - np.maximum(open_, close)) / close_safe

        # KUP2: Upper shadow normalized
        features["KUP2"] = (high - np.maximum(open_, close)) / (high - low + 1e-8)

        # KLOW: Lower shadow
        features["KLOW"] = (np.minimum(open_, close) - low) / close_safe

        # KLOW2: Lower shadow normalized
        features["KLOW2"] = (np.minimum(open_, close) - low) / (high - low + 1e-8)

        # KSFT: Bar shift (close position in range)
        features["KSFT"] = (2 * close - high - low) / close_safe

        # KSFT2: Bar shift normalized
        features["KSFT2"] = (2 * close - high - low) / (high - low + 1e-8)

        # === Ratio features ===
        features["OPEN_RATIO"] = open_ / close.shift(1) - 1
        features["HIGH_RATIO"] = high / close.shift(1) - 1
        features["LOW_RATIO"] = low / close.shift(1) - 1
        features["CLOSE_RATIO"] = close / close.shift(1) - 1

        # VWAP ratio
        vwap = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        features["VWAP_RATIO"] = close / vwap - 1

        # Turnover features (using volume as proxy)
        features["TURN"] = volume / volume.rolling(20).mean()
        features["TURN_MA5"] = features["TURN"].rolling(5).mean()
        features["TURN_MA10"] = features["TURN"].rolling(10).mean()

        # Build DataFrame
        result = pd.DataFrame(features, index=df.index)

        # Replace inf with nan
        result = result.replace([np.inf, -np.inf], np.nan)

        return result

    def _rolling_beta(self, series: pd.Series, window: int) -> pd.Series:
        """Compute rolling linear regression slope (beta)"""

        def calc_beta(x):
            if len(x) < window:
                return np.nan
            y = np.arange(len(x))
            try:
                slope = np.polyfit(y, x, 1)[0]
                return slope / (np.std(x) + 1e-8)
            except:
                return np.nan

        return series.rolling(window).apply(calc_beta, raw=True)


class QlibPredictor:
    """
    Stock predictor using Alpha158 features.

    Uses a lightweight LightGBM model trained on historical patterns.
    The model predicts probability of positive returns.
    """

    def __init__(self):
        self.calculator = Alpha158Calculator()
        self.model = None
        self.feature_importance = {}
        self.is_trained = False
        self._init_model()

    def _init_model(self):
        """Initialize or load the prediction model"""
        import os
        import pickle

        # Try to load trained model first
        model_path = os.path.join(os.path.dirname(__file__), "qlib_model.pkl")

        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                logger.info(f"Qlib predictor loaded trained model from {model_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load trained model: {e}")

        # Fallback to default untrained model
        try:
            import lightgbm as lgb

            # Create a default model with reasonable hyperparameters
            # This can be trained on historical data
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                force_col_wise=True,
            )

            logger.info(
                "Qlib predictor initialized with default LightGBM (not trained)"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Qlib model: {e}")
            self.model = None

    def _fetch_data(self, symbol: str, period: str = "6mo") -> pd.DataFrame:
        """Fetch historical data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            if df.empty:
                raise ValueError(f"No data for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()

    def compute_score(self, features: pd.DataFrame) -> float:
        """
        Compute a prediction score from features.

        If model is not trained, uses a heuristic based on feature values.
        Returns score from 0.0 (bearish) to 1.0 (bullish).
        """
        if features.empty:
            return 0.5

        # Get latest row
        latest = features.iloc[-1]
        valid_features = latest.dropna()

        if len(valid_features) < 20:
            return 0.5

        # If model is trained, use it
        if self.is_trained and self.model is not None:
            try:
                X = features.iloc[[-1]].fillna(0).replace([np.inf, -np.inf], 0)
                # Trained model predicts returns, convert to 0-1 score
                predicted_return = self.model.predict(X)[0]
                # Convert return prediction to score (sigmoid-like transformation)
                # Expected returns are typically -10% to +10%, scale to 0-1
                score = 1 / (1 + np.exp(-predicted_return * 20))  # sigmoid with scaling
                return float(np.clip(score, 0.05, 0.95))
            except Exception as e:
                logger.warning(f"Model prediction failed, using heuristic: {e}")

        # Heuristic scoring based on feature values
        score = 0.5

        # Momentum signals (ROC)
        for w in [5, 10, 20]:
            roc = valid_features.get(f"ROC_{w}", 0)
            if not np.isnan(roc):
                score += np.clip(roc * 5, -0.1, 0.1)

        # Trend signals (MA position)
        for w in [5, 10, 20]:
            ma_ratio = valid_features.get(f"MA_{w}", 0)
            if not np.isnan(ma_ratio):
                if ma_ratio > 0:  # Price above MA
                    score += 0.02
                elif ma_ratio < 0:  # Price below MA
                    score -= 0.02

        # RSV signals (stochastic)
        rsv_20 = valid_features.get("RSV_20", 0.5)
        if not np.isnan(rsv_20):
            if rsv_20 > 0.8:  # Overbought
                score -= 0.05
            elif rsv_20 < 0.2:  # Oversold
                score += 0.05

        # Rank signals
        rank_20 = valid_features.get("RANK_20", 0.5)
        if not np.isnan(rank_20):
            score += (rank_20 - 0.5) * 0.1

        # Volume confirmation
        vma_5 = valid_features.get("VMA_5", 0)
        roc_5 = valid_features.get("ROC_5", 0)
        if not np.isnan(vma_5) and not np.isnan(roc_5):
            if vma_5 > 0 and roc_5 > 0:  # Volume up with price up
                score += 0.05
            elif vma_5 > 0 and roc_5 < 0:  # Volume up with price down
                score -= 0.05

        # VWAP position
        vwap_ratio = valid_features.get("VWAP_RATIO", 0)
        if not np.isnan(vwap_ratio):
            if vwap_ratio > 0:  # Above VWAP
                score += 0.03
            else:  # Below VWAP
                score -= 0.03

        # K-line patterns
        kmid = valid_features.get("KMID", 0)
        if not np.isnan(kmid):
            score += kmid * 0.5  # Bullish candle adds, bearish subtracts

        return np.clip(score, 0.0, 1.0)

    def predict(self, symbol: str, period: str = "6mo") -> QlibPrediction:
        """
        Generate Alpha158-based prediction for a symbol.

        Args:
            symbol: Stock ticker
            period: Historical data period

        Returns:
            QlibPrediction with score, signal, and feature info
        """
        try:
            # Fetch data
            df = self._fetch_data(symbol, period)
            if df.empty:
                return self._error_prediction(symbol, "No data available")

            # Compute features
            features = self.calculator.compute_features(df)
            if features.empty:
                return self._error_prediction(symbol, "Insufficient data for features")

            # Get latest features
            latest = features.iloc[-1]
            valid_count = (~latest.isna()).sum()

            # Compute score
            score = self.compute_score(features)

            # Determine signal
            if score >= 0.7:
                signal = "STRONG_BULLISH"
            elif score >= 0.55:
                signal = "BULLISH"
            elif score <= 0.3:
                signal = "STRONG_BEARISH"
            elif score <= 0.45:
                signal = "BEARISH"
            else:
                signal = "NEUTRAL"

            # Confidence based on feature coverage
            confidence = min(1.0, valid_count / 100)

            # Get top features (by absolute value)
            top_features = []
            for col in features.columns:
                val = latest.get(col, np.nan)
                if not np.isnan(val) and abs(val) > 0.01:
                    top_features.append((col, val))

            top_features.sort(key=lambda x: abs(x[1]), reverse=True)
            top_features = top_features[:10]

            return QlibPrediction(
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                score=round(score, 4),
                rank_percentile=0.0,  # Would need multiple symbols to compute
                signal=signal,
                confidence=round(confidence, 4),
                top_features=top_features,
                feature_count=valid_count,
            )

        except Exception as e:
            logger.error(f"Qlib prediction failed for {symbol}: {e}")
            return self._error_prediction(symbol, str(e))

    def _error_prediction(self, symbol: str, error: str) -> QlibPrediction:
        """Return error prediction"""
        return QlibPrediction(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            score=0.5,
            rank_percentile=0.0,
            signal="ERROR",
            confidence=0.0,
            top_features=[],
            feature_count=0,
        )

    def rank_symbols(self, symbols: List[str]) -> List[Dict]:
        """
        Rank multiple symbols by their Alpha158 scores.

        Args:
            symbols: List of tickers to rank

        Returns:
            List of dicts with symbol, score, rank, signal
        """
        results = []

        for symbol in symbols:
            try:
                pred = self.predict(symbol)
                results.append(
                    {
                        "symbol": symbol,
                        "score": pred.score,
                        "signal": pred.signal,
                        "confidence": pred.confidence,
                        "features": pred.feature_count,
                    }
                )
            except Exception as e:
                logger.error(f"Failed to score {symbol}: {e}")
                results.append(
                    {
                        "symbol": symbol,
                        "score": 0.5,
                        "signal": "ERROR",
                        "confidence": 0.0,
                        "features": 0,
                    }
                )

        # Sort by score (highest first)
        results.sort(key=lambda x: x["score"], reverse=True)

        # Add rank percentiles
        n = len(results)
        for i, r in enumerate(results):
            r["rank"] = i + 1
            r["rank_percentile"] = round((n - i) / n, 4) if n > 0 else 0.5

        return results

    def train_on_history(self, symbols: List[str], lookback_days: int = 252) -> Dict:
        """
        Train the model on historical data.

        Uses next-day returns as labels (up/down classification).

        Args:
            symbols: List of symbols to train on
            lookback_days: Days of history to use

        Returns:
            Training metrics
        """
        try:
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split

            all_X = []
            all_y = []

            for symbol in symbols:
                try:
                    df = self._fetch_data(symbol, period="1y")
                    if len(df) < 100:
                        continue

                    features = self.calculator.compute_features(df)
                    if features.empty:
                        continue

                    # Create labels: 1 if next day return > 0, else 0
                    returns = df["Close"].pct_change().shift(-1)
                    labels = (returns > 0).astype(int)

                    # Align and drop NaN
                    valid_idx = features.dropna().index.intersection(
                        labels.dropna().index
                    )

                    if len(valid_idx) < 50:
                        continue

                    X = features.loc[valid_idx].fillna(0)
                    y = labels.loc[valid_idx]

                    all_X.append(X)
                    all_y.append(y)

                except Exception as e:
                    logger.warning(f"Skipping {symbol}: {e}")
                    continue

            if not all_X:
                return {"error": "No valid training data"}

            # Combine all data
            X = pd.concat(all_X, ignore_index=True)
            y = pd.concat(all_y, ignore_index=True)

            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train model
            self.model.fit(X_train, y_train)

            # Evaluate
            train_acc = self.model.score(X_train, y_train)
            test_acc = self.model.score(X_test, y_test)

            # Store feature importance
            importances = self.model.feature_importances_
            self.feature_importance = dict(zip(X.columns, importances))

            self.is_trained = True

            logger.info(
                f"Qlib model trained: train_acc={train_acc:.2%}, test_acc={test_acc:.2%}"
            )

            return {
                "status": "trained",
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_accuracy": round(train_acc, 4),
                "test_accuracy": round(test_acc, 4),
                "feature_count": len(X.columns),
                "symbols_used": len(all_X),
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"error": str(e)}


# Singleton instance
_predictor = None


def get_qlib_predictor() -> QlibPredictor:
    """Get or create the global Qlib predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = QlibPredictor()
    return _predictor


# Convenience functions
def predict(symbol: str) -> QlibPrediction:
    """Quick prediction for a single symbol."""
    return get_qlib_predictor().predict(symbol)


def rank(symbols: List[str]) -> List[Dict]:
    """Rank multiple symbols."""
    return get_qlib_predictor().rank_symbols(symbols)


if __name__ == "__main__":
    # Test the predictor
    logging.basicConfig(level=logging.INFO)

    predictor = get_qlib_predictor()

    # Test single prediction
    print("Testing AAPL prediction...")
    result = predictor.predict("AAPL")
    print(f"  Score: {result.score}")
    print(f"  Signal: {result.signal}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Features computed: {result.feature_count}")
    print(f"  Top features: {result.top_features[:5]}")

    # Test ranking
    print("\nRanking test symbols...")
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    rankings = predictor.rank_symbols(symbols)

    print("\nRankings:")
    for r in rankings:
        print(f"  {r['rank']}. {r['symbol']}: {r['score']:.2%} ({r['signal']})")
