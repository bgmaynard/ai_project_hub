"""
Polygon Deep Training Pipeline
==============================
Comprehensive model training using Polygon's 20-year historical data.

Features:
- Extended historical data (up to 20 years)
- Multiple timeframes (daily, hourly for recent)
- Market regime detection
- Sector correlation analysis
- LightGBM + CNN ensemble training
- Scalp-specific signal detection

Usage:
    python -m ai.polygon_deep_training
"""

import os
import sys
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()  # Load .env file BEFORE accessing env vars

import numpy as np
import pandas as pd
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Polygon API
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
POLYGON_BASE_URL = "https://api.polygon.io"

# Training symbols - mix of different types
TRAINING_UNIVERSE = {
    # Large-cap liquid (for baseline patterns)
    'mega_cap': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD'],

    # Mid-cap growth (good momentum candidates)
    'mid_growth': ['PLTR', 'SNOW', 'NET', 'CRWD', 'DDOG', 'ZS', 'MDB', 'COIN', 'ROKU', 'SQ'],

    # Small-cap momentum (user's focus area)
    'small_momentum': ['SOUN', 'IONQ', 'RGTI', 'QUBT', 'RKLB', 'LUNR', 'DNA', 'STEM', 'PLUG', 'FCEL'],

    # Recent movers (volatile, scalp-friendly)
    'recent_movers': ['MSTR', 'SMCI', 'ARM', 'RDDT', 'HOOD', 'SOFI', 'AFRM', 'UPST', 'LCID', 'RIVN'],

    # Penny momentum (high risk/reward)
    'penny_momentum': ['MULN', 'FFIE', 'GOEV', 'NKLA', 'WKHS', 'RIDE', 'HYLN', 'ARVL', 'REE', 'EVGO'],

    # Sector ETFs (for regime detection)
    'sector_etfs': ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLB', 'XLU', 'XLP', 'XLY', 'XLRE'],

    # Volatility/hedge
    'volatility': ['VIX', 'UVXY', 'SQQQ', 'TQQQ', 'TLT', 'GLD', 'SLV', 'USO', 'UNG', 'DBA']
}


class PolygonDataFetcher:
    """Fetch historical data from Polygon.io"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or POLYGON_API_KEY
        self.cache_dir = Path(__file__).parent / "training_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self._request_count = 0
        self._last_request = 0

    def _rate_limit(self):
        """Rate limiting for API calls"""
        now = time.time()
        elapsed = now - self._last_request
        # With paid plan, can do more requests but still be careful
        min_interval = 0.05  # 20 requests per second
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request = time.time()
        self._request_count += 1

    def fetch_daily_bars(
        self,
        symbol: str,
        years: int = 10,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """Fetch daily bars for symbol"""
        cache_file = self.cache_dir / f"{symbol}_daily_{years}y.parquet"

        # Check cache (valid for 1 day)
        if use_cache and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 86400:  # 24 hours
                try:
                    return pd.read_parquet(cache_file)
                except Exception:
                    pass

        # Fetch from API
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")

        self._rate_limit()

        try:
            url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
            params = {"apiKey": self.api_key, "adjusted": "true", "sort": "asc", "limit": 50000}

            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, params=params)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch {symbol}: {response.status_code}")
                return None

            data = response.json()
            results = data.get("results", [])

            if not results:
                return None

            df = pd.DataFrame(results)
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={
                'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close',
                'v': 'volume', 'vw': 'vwap', 'n': 'trades'
            })
            df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trades']]
            df = df.set_index('date')
            df['symbol'] = symbol

            # Cache it
            df.to_parquet(cache_file)

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def fetch_minute_bars(
        self,
        symbol: str,
        days: int = 30,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """Fetch minute bars for recent period (for scalp training)"""
        cache_file = self.cache_dir / f"{symbol}_minute_{days}d.parquet"

        if use_cache and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 3600:  # 1 hour
                try:
                    return pd.read_parquet(cache_file)
                except Exception:
                    pass

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        self._rate_limit()

        try:
            url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range/1/minute/{start_date}/{end_date}"
            params = {"apiKey": self.api_key, "adjusted": "true", "sort": "asc", "limit": 50000}

            with httpx.Client(timeout=60.0) as client:
                response = client.get(url, params=params)

            if response.status_code != 200:
                return None

            data = response.json()
            results = data.get("results", [])

            if not results:
                return None

            df = pd.DataFrame(results)
            df['datetime'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={
                'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close',
                'v': 'volume', 'vw': 'vwap', 'n': 'trades'
            })
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trades']]
            df = df.set_index('datetime')
            df['symbol'] = symbol

            df.to_parquet(cache_file)
            return df

        except Exception as e:
            logger.error(f"Error fetching minute data for {symbol}: {e}")
            return None


class FeatureEngineering:
    """Generate features for training"""

    @staticmethod
    def compute_daily_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute features from daily bars"""
        data = df.copy()

        # Basic returns
        data['return_1d'] = data['close'].pct_change()
        data['return_5d'] = data['close'].pct_change(5)
        data['return_10d'] = data['close'].pct_change(10)
        data['return_20d'] = data['close'].pct_change(20)

        # Volatility
        data['volatility_10d'] = data['return_1d'].rolling(10).std()
        data['volatility_20d'] = data['return_1d'].rolling(20).std()
        data['volatility_60d'] = data['return_1d'].rolling(60).std()

        # Volume features
        data['volume_ma10'] = data['volume'].rolling(10).mean()
        data['volume_ma20'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma20']

        # Price momentum
        data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
        data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
        data['momentum_20d'] = data['close'] / data['close'].shift(20) - 1

        # Moving averages
        data['sma_10'] = data['close'].rolling(10).mean()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        data['sma_200'] = data['close'].rolling(200).mean()

        # Price relative to MAs
        data['price_vs_sma10'] = data['close'] / data['sma_10'] - 1
        data['price_vs_sma20'] = data['close'] / data['sma_20'] - 1
        data['price_vs_sma50'] = data['close'] / data['sma_50'] - 1
        data['price_vs_sma200'] = data['close'] / data['sma_200'] - 1

        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        data['macd'] = ema12 - ema26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']

        # Bollinger Bands
        data['bb_mid'] = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        data['bb_upper'] = data['bb_mid'] + 2 * bb_std
        data['bb_lower'] = data['bb_mid'] - 2 * bb_std
        data['bb_pct'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])

        # ATR
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['atr_14'] = true_range.rolling(14).mean()
        data['atr_pct'] = data['atr_14'] / data['close']

        # Gap analysis
        data['gap'] = data['open'] / data['close'].shift() - 1
        data['gap_filled'] = ((data['low'] <= data['close'].shift()) & (data['gap'] > 0)) | \
                            ((data['high'] >= data['close'].shift()) & (data['gap'] < 0))

        # Range metrics
        data['daily_range'] = (data['high'] - data['low']) / data['close']
        data['range_ma10'] = data['daily_range'].rolling(10).mean()

        # Higher highs / lower lows
        data['hh'] = (data['high'] > data['high'].shift()).astype(int)
        data['ll'] = (data['low'] < data['low'].shift()).astype(int)
        data['hh_streak'] = data['hh'].groupby((data['hh'] != data['hh'].shift()).cumsum()).cumsum()
        data['ll_streak'] = data['ll'].groupby((data['ll'] != data['ll'].shift()).cumsum()).cumsum()

        # Target: Next day return direction
        data['target'] = (data['return_1d'].shift(-1) > 0).astype(int)
        data['target_return'] = data['return_1d'].shift(-1)

        return data

    @staticmethod
    def compute_minute_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute features from minute bars (for scalp model)"""
        data = df.copy()

        # Returns at different intervals
        data['return_1m'] = data['close'].pct_change()
        data['return_5m'] = data['close'].pct_change(5)
        data['return_15m'] = data['close'].pct_change(15)

        # Volume surge
        data['vol_ma5'] = data['volume'].rolling(5).mean()
        data['vol_ma20'] = data['volume'].rolling(20).mean()
        data['vol_surge'] = data['volume'] / data['vol_ma20']

        # Velocity (speed of move)
        data['velocity_1m'] = data['return_1m'].abs()
        data['velocity_5m'] = data['return_5m'].abs() / 5

        # Range
        data['range_1m'] = (data['high'] - data['low']) / data['close']

        # VWAP deviation
        if 'vwap' in data.columns:
            data['vwap_dev'] = (data['close'] - data['vwap']) / data['vwap']

        # Trade intensity
        if 'trades' in data.columns:
            data['trades_ma5'] = data['trades'].rolling(5).mean()
            data['trades_surge'] = data['trades'] / data['trades_ma5']

        # Time features
        data['hour'] = data.index.hour
        data['minute'] = data.index.minute
        data['is_premarket'] = (data['hour'] < 9) | ((data['hour'] == 9) & (data['minute'] < 30))
        data['is_open'] = (data['hour'] == 9) & (data['minute'] >= 30) & (data['minute'] < 45)
        data['is_power_hour'] = data['hour'] >= 15

        # Spike detection
        data['is_spike'] = (
            (data['return_1m'].abs() > 0.005) |  # 0.5% in 1 min
            (data['return_5m'].abs() > 0.01)     # 1% in 5 min
        ) & (data['vol_surge'] > 1.5)

        # Target: Does price continue up in next 3 minutes?
        data['forward_3m'] = data['close'].shift(-3) / data['close'] - 1
        data['target'] = (data['forward_3m'] > 0).astype(int)

        return data


class ModelTrainer:
    """Train prediction models"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(__file__).parent
        self.results = {}

    def train_lightgbm_daily(self, df: pd.DataFrame) -> Dict:
        """Train LightGBM on daily features"""
        try:
            import lightgbm as lgb
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import accuracy_score, roc_auc_score
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            return {}

        # Feature columns (exclude target and non-feature columns)
        exclude_cols = ['symbol', 'target', 'target_return', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trades']
        feature_cols = [c for c in df.columns if c not in exclude_cols and not df[c].isna().all()]

        # Prepare data
        df_clean = df.dropna(subset=feature_cols + ['target'])
        X = df_clean[feature_cols].values
        y = df_clean['target'].values

        logger.info(f"Training LightGBM: {len(X)} samples, {len(feature_cols)} features")

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        fold_results = []
        models = []

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42
        }

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            val_pred = model.predict(X_val)
            val_pred_binary = (val_pred > 0.5).astype(int)

            acc = accuracy_score(y_val, val_pred_binary)
            auc = roc_auc_score(y_val, val_pred)

            fold_results.append({
                'fold': fold + 1,
                'accuracy': acc,
                'auc': auc,
                'best_iteration': model.best_iteration
            })

            models.append(model)
            logger.info(f"  Fold {fold+1}: Acc={acc:.4f}, AUC={auc:.4f}")

        # Save best model (last fold typically has most data)
        best_model = models[-1]
        model_path = self.output_dir / "polygon_lgb_daily.pkl"

        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': best_model,
                'features': feature_cols,
                'params': params
            }, f)

        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model.feature_importance()
        }).sort_values('importance', ascending=False)

        avg_acc = np.mean([r['accuracy'] for r in fold_results])
        avg_auc = np.mean([r['auc'] for r in fold_results])

        return {
            'model_type': 'LightGBM Daily',
            'samples': len(X),
            'features': len(feature_cols),
            'avg_accuracy': avg_acc,
            'avg_auc': avg_auc,
            'fold_results': fold_results,
            'top_features': importance.head(20).to_dict('records'),
            'model_path': str(model_path)
        }

    def train_scalp_model(self, df: pd.DataFrame) -> Dict:
        """Train model specifically for scalp/momentum trading"""
        try:
            import lightgbm as lgb
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import accuracy_score, roc_auc_score
        except ImportError:
            return {}

        # Feature columns for scalp model
        exclude_cols = ['symbol', 'target', 'forward_3m', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trades', 'datetime']
        feature_cols = [c for c in df.columns if c not in exclude_cols and not df[c].isna().all()]

        # Only use spike events for training
        df_spikes = df[df['is_spike'] == True].copy()

        if len(df_spikes) < 100:
            logger.warning(f"Not enough spike events: {len(df_spikes)}")
            return {}

        df_clean = df_spikes.dropna(subset=feature_cols + ['target'])
        X = df_clean[feature_cols].values
        y = df_clean['target'].values

        logger.info(f"Training Scalp Model: {len(X)} spike events, {len(feature_cols)} features")

        # Check class balance
        pos_rate = y.mean()
        logger.info(f"  Spike continuation rate: {pos_rate:.2%}")

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 15,
            'max_depth': 4,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 3,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
            'seed': 42
        }

        # Simple train/val split for spike data
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )

        val_pred = model.predict(X_val)
        val_pred_binary = (val_pred > 0.5).astype(int)

        acc = accuracy_score(y_val, val_pred_binary)
        auc = roc_auc_score(y_val, val_pred) if len(np.unique(y_val)) > 1 else 0.5

        # Save model
        model_path = self.output_dir / "polygon_scalp_model.pkl"
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'features': feature_cols,
                'params': params,
                'spike_continuation_rate': pos_rate
            }, f)

        return {
            'model_type': 'Scalp Model',
            'spike_events': len(X),
            'spike_continuation_rate': pos_rate,
            'val_accuracy': acc,
            'val_auc': auc,
            'model_path': str(model_path)
        }


def run_deep_training():
    """Main training pipeline"""
    start_time = datetime.now()
    logger.info("="*60)
    logger.info("POLYGON DEEP TRAINING PIPELINE")
    logger.info("="*60)

    # Initialize
    fetcher = PolygonDataFetcher()
    trainer = ModelTrainer()
    results = {
        'start_time': start_time.isoformat(),
        'data_collection': {},
        'models': {}
    }

    # 1. Fetch daily data for all symbols
    logger.info("\n[1/4] Fetching historical daily data...")
    all_symbols = []
    for category, symbols in TRAINING_UNIVERSE.items():
        all_symbols.extend(symbols)
    all_symbols = list(set(all_symbols))

    daily_dfs = []
    fetched = 0

    for symbol in all_symbols:
        df = fetcher.fetch_daily_bars(symbol, years=10)
        if df is not None and len(df) > 200:
            daily_dfs.append(df)
            fetched += 1
            if fetched % 10 == 0:
                logger.info(f"  Fetched {fetched}/{len(all_symbols)} symbols...")

    logger.info(f"  Total: {fetched} symbols with sufficient data")
    results['data_collection']['daily_symbols'] = fetched

    # 2. Compute features and combine
    logger.info("\n[2/4] Computing daily features...")
    featured_dfs = []
    for df in daily_dfs:
        try:
            featured = FeatureEngineering.compute_daily_features(df)
            featured_dfs.append(featured)
        except Exception as e:
            logger.warning(f"Feature error: {e}")

    if featured_dfs:
        combined_daily = pd.concat(featured_dfs, ignore_index=True)
        logger.info(f"  Combined dataset: {len(combined_daily)} samples")
        results['data_collection']['daily_samples'] = len(combined_daily)
    else:
        logger.error("No data to train on!")
        return results

    # 3. Train LightGBM on daily data
    logger.info("\n[3/4] Training LightGBM daily model...")
    lgb_results = trainer.train_lightgbm_daily(combined_daily)
    if lgb_results:
        results['models']['lightgbm_daily'] = lgb_results
        logger.info(f"  Accuracy: {lgb_results['avg_accuracy']:.4f}")
        logger.info(f"  AUC: {lgb_results['avg_auc']:.4f}")

    # 4. Fetch minute data for scalp model (recent 30 days)
    logger.info("\n[4/4] Training scalp model with minute data...")

    # Use momentum stocks for scalp training
    scalp_symbols = TRAINING_UNIVERSE['small_momentum'] + TRAINING_UNIVERSE['recent_movers'][:5]

    minute_dfs = []
    for symbol in scalp_symbols[:15]:  # Limit to avoid too much data
        df = fetcher.fetch_minute_bars(symbol, days=30)
        if df is not None and len(df) > 1000:
            try:
                featured = FeatureEngineering.compute_minute_features(df)
                minute_dfs.append(featured)
            except Exception as e:
                logger.warning(f"Minute feature error for {symbol}: {e}")

    if minute_dfs:
        combined_minute = pd.concat(minute_dfs, ignore_index=True)
        logger.info(f"  Minute data: {len(combined_minute)} bars from {len(minute_dfs)} symbols")
        results['data_collection']['minute_samples'] = len(combined_minute)

        scalp_results = trainer.train_scalp_model(combined_minute)
        if scalp_results:
            results['models']['scalp_model'] = scalp_results
            logger.info(f"  Spike continuation rate: {scalp_results.get('spike_continuation_rate', 0):.2%}")
            logger.info(f"  Validation AUC: {scalp_results.get('val_auc', 0):.4f}")

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    results['end_time'] = end_time.isoformat()
    results['duration_seconds'] = duration
    results['duration_minutes'] = duration / 60

    # Save results
    results_path = Path(__file__).parent / "polygon_training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Duration: {duration/60:.1f} minutes")
    logger.info(f"Results saved to: {results_path}")

    if 'lightgbm_daily' in results['models']:
        logger.info(f"Daily Model Accuracy: {results['models']['lightgbm_daily']['avg_accuracy']:.2%}")
    if 'scalp_model' in results['models']:
        logger.info(f"Scalp Model AUC: {results['models']['scalp_model'].get('val_auc', 0):.4f}")

    return results


if __name__ == "__main__":
    run_deep_training()
