"""
Scalp Trainer - Minute-Level Model for 2-Minute Moves
======================================================
Trains on minute-bar data to predict immediate continuation vs reversal.

Target Question: "A spike just started - will it continue for 2+ minutes or reverse?"

Features (computed on minute bars):
- Volume surge ratio (current vs 5-min average)
- Price velocity (% move in last 1-2 minutes)
- Spread proxy (high-low range)
- Time of day (pre-market, open, midday)
- Previous 5-min trend
- Gap from prior close

Target:
- 1 = Spike continues (price goes up another 1%+ in next 2-5 min)
- 0 = Spike fades/reverses

Created: December 2024
"""

import json
import logging
import os
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")


# ============================================================================
# VOLATILE SYMBOLS FOR TRAINING
# ============================================================================

# Stocks known for big intraday moves
SCALP_SYMBOLS = [
    # Recent movers / quantum / AI hype
    "SOUN",
    "IONQ",
    "RGTI",
    "QUBT",
    "RKLB",
    # Volatile small caps
    "SOFI",
    "HOOD",
    "AFRM",
    "UPST",
    "FUBO",
    # Biotech volatility
    "NVAX",
    "MRNA",
    "SAVA",
    "OCGN",
    "INO",
    # EV plays
    "RIVN",
    "LCID",
    "NIO",
    "XPEV",
    "LI",
    # Cannabis
    "TLRY",
    "CGC",
    "SNDL",
    "ACB",
    # Clean energy
    "PLUG",
    "FCEL",
    "BE",
    "CHPT",
    # Meme / retail favorites
    "PLTR",
    "CLOV",
    "GRAB",
    # Leveraged ETFs (extreme moves)
    "TQQQ",
    "SQQQ",
    "SOXL",
    "SOXS",
]


# ============================================================================
# POLYGON DATA FETCHER
# ============================================================================


class PolygonMinuteData:
    """Fetch minute-level data from Polygon.io"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or POLYGON_API_KEY
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not set")

    def get_minute_bars(
        self, symbol: str, date: str, limit: int = 1000  # YYYY-MM-DD
    ) -> pd.DataFrame:
        """Get minute bars for a single day"""
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{date}/{date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": limit,
            "apiKey": self.api_key,
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()

            if not data.get("results"):
                return pd.DataFrame()

            df = pd.DataFrame(data["results"])
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
            df = df.rename(
                columns={
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "vw": "vwap",
                    "n": "trades",
                }
            )
            df = df[
                [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "vwap",
                    "trades",
                ]
            ]
            df = df.set_index("timestamp")

            return df

        except Exception as e:
            logger.warning(f"Error fetching {symbol} for {date}: {e}")
            return pd.DataFrame()

    def get_multiple_days(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Get minute bars for multiple days"""
        all_data = []

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        current = start
        while current <= end:
            # Skip weekends
            if current.weekday() < 5:
                date_str = current.strftime("%Y-%m-%d")
                df = self.get_minute_bars(symbol, date_str)
                if not df.empty:
                    all_data.append(df)

            current += timedelta(days=1)

        if all_data:
            return pd.concat(all_data)
        return pd.DataFrame()


# ============================================================================
# SCALPING FEATURES
# ============================================================================


class ScalpFeatureEngine:
    """Generate features for scalping prediction"""

    def __init__(
        self,
        spike_threshold: float = 0.01,  # 1% move = spike (lowered from 2%)
        lookback_minutes: int = 5,
        forward_minutes: int = 3,  # Predict next 3 minutes
    ):
        self.spike_threshold = spike_threshold
        self.lookback_minutes = lookback_minutes
        self.forward_minutes = forward_minutes

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute scalping features from minute bars.

        Returns DataFrame with features for each bar that shows a spike.
        """
        if len(df) < self.lookback_minutes + self.forward_minutes + 5:
            return pd.DataFrame()

        data = df.copy()

        # Basic price features
        data["return_1m"] = data["close"].pct_change(1)
        data["return_2m"] = data["close"].pct_change(2)
        data["return_5m"] = data["close"].pct_change(5)

        # Volume features
        data["vol_ma5"] = data["volume"].rolling(5).mean()
        data["vol_surge"] = data["volume"] / data["vol_ma5"]
        data["vol_surge"] = data["vol_surge"].clip(0, 20)  # Cap at 20x

        # Price velocity (rate of change)
        data["velocity_1m"] = data["return_1m"].abs()
        data["velocity_2m"] = data["return_2m"].abs() / 2  # Per minute

        # Spread proxy (intrabar range)
        data["range_pct"] = (data["high"] - data["low"]) / data["open"]

        # VWAP deviation
        data["vwap_dev"] = (data["close"] - data["vwap"]) / data["vwap"]

        # Trade count (liquidity)
        data["trades_ma5"] = data["trades"].rolling(5).mean()
        data["trades_surge"] = data["trades"] / data["trades_ma5"]
        data["trades_surge"] = data["trades_surge"].clip(0, 10)

        # Time of day features
        data["hour"] = data.index.hour
        data["minute"] = data.index.minute
        data["is_premarket"] = (data["hour"] < 9) | (
            (data["hour"] == 9) & (data["minute"] < 30)
        )
        data["is_open"] = (
            (data["hour"] == 9) & (data["minute"] >= 30) & (data["minute"] < 45)
        )
        data["is_power_hour"] = data["hour"] >= 15

        # Previous bars momentum
        data["prev_5m_trend"] = data["close"].shift(1) / data["close"].shift(6) - 1

        # Spike detection (this bar shows a spike)
        # More lenient: 0.5% in 1 min OR 1% in 2 min, with some volume surge
        data["is_spike"] = (
            (data["return_1m"] > 0.005)  # 0.5% in 1 min
            | (data["return_2m"] > 0.01)  # 1% in 2 min
        ) & (
            data["vol_surge"] > 1.5
        )  # With moderate volume surge

        # FORWARD TARGET: Does spike continue?
        # Look at return over next N minutes
        data["forward_return"] = (
            data["close"].shift(-self.forward_minutes) / data["close"] - 1
        )

        # Target: 1 if price goes up AT ALL in next N minutes (any green)
        data["target"] = (data["forward_return"] > 0).astype(int)

        # Alternative targets for analysis
        data["target_half_pct"] = (data["forward_return"] > 0.005).astype(int)  # 0.5%+
        data["target_one_pct"] = (data["forward_return"] > 0.01).astype(int)  # 1%+
        data["target_two_pct"] = (data["forward_return"] > 0.02).astype(int)  # 2%+

        # Feature columns
        feature_cols = [
            "return_1m",
            "return_2m",
            "return_5m",
            "vol_surge",
            "velocity_1m",
            "velocity_2m",
            "range_pct",
            "vwap_dev",
            "trades_surge",
            "is_premarket",
            "is_open",
            "is_power_hour",
            "prev_5m_trend",
            "hour",
            "minute",
        ]

        # Only keep rows where we detected a spike (training on spike events)
        spike_data = data[data["is_spike"]].copy()

        if spike_data.empty:
            return pd.DataFrame()

        # Clean up - keep features, target, and forward return
        keep_cols = feature_cols + ["target", "forward_return"]
        spike_data = spike_data[keep_cols]
        spike_data = spike_data.dropna()

        return spike_data


# ============================================================================
# SCALP MODEL TRAINER
# ============================================================================


class ScalpModelTrainer:
    """Train model for scalp continuation prediction"""

    def __init__(self):
        self.model = None
        self.feature_names = []
        self.metrics = {}

    def collect_training_data(
        self, symbols: List[str], days: int = 30
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Collect spike events from multiple symbols"""
        fetcher = PolygonMinuteData()
        feature_engine = ScalpFeatureEngine()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        all_features = []
        all_labels = []

        logger.info(
            f"Collecting {days} days of minute data for {len(symbols)} symbols..."
        )

        for i, symbol in enumerate(symbols):
            try:
                df = fetcher.get_multiple_days(
                    symbol,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                )

                if df.empty or len(df) < 100:
                    logger.debug(f"  {symbol}: insufficient data")
                    continue

                features = feature_engine.compute_features(df)

                if features.empty:
                    logger.debug(f"  {symbol}: no spikes detected")
                    continue

                feature_cols = [
                    c
                    for c in features.columns
                    if c
                    not in [
                        "target",
                        "target_continues",
                        "target_strong",
                        "forward_return",
                    ]
                ]

                all_features.append(features[feature_cols])
                all_labels.append(features["target"])  # Using 1% continuation as target

                logger.info(
                    f"  [{i+1}/{len(symbols)}] {symbol}: {len(features)} spike events"
                )

            except Exception as e:
                logger.warning(f"  {symbol}: error - {e}")
                continue

        if not all_features:
            raise ValueError("No training data collected")

        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)

        # Clean
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        # Convert boolean to int
        for col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)

        self.feature_names = list(X.columns)

        logger.info(f"\nTotal spike events: {len(X)}")
        logger.info(f"Spike continues (any green in 3min): {y.mean():.1%}")

        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> Dict:
        """Train LightGBM for spike continuation"""
        import lightgbm as lgb
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, roc_auc_score)
        from sklearn.model_selection import TimeSeriesSplit

        # Parameters optimized for quick prediction
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 15,  # Simple model for fast decisions
            "max_depth": 4,
            "learning_rate": 0.1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 3,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }

        tscv = TimeSeriesSplit(n_splits=n_folds)
        fold_metrics = []
        best_model = None
        best_auc = 0

        logger.info(f"Training scalp model with {n_folds}-fold validation...")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30),
                    lgb.log_evaluation(period=0),
                ],
            )

            y_pred_proba = model.predict(X_val)
            y_pred = (y_pred_proba > 0.5).astype(int)

            val_acc = accuracy_score(y_val, y_pred)
            val_auc = roc_auc_score(y_val, y_pred_proba)
            val_precision = precision_score(y_val, y_pred, zero_division=0)
            val_recall = recall_score(y_val, y_pred, zero_division=0)

            fold_metrics.append(
                {
                    "fold": fold + 1,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                }
            )

            logger.info(
                f"  Fold {fold+1}: Acc={val_acc:.3f}, AUC={val_auc:.3f}, Prec={val_precision:.3f}, Rec={val_recall:.3f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                best_model = model

        self.model = best_model

        # Average metrics
        avg_acc = np.mean([m["val_acc"] for m in fold_metrics])
        avg_auc = np.mean([m["val_auc"] for m in fold_metrics])
        avg_precision = np.mean([m["val_precision"] for m in fold_metrics])
        avg_recall = np.mean([m["val_recall"] for m in fold_metrics])

        # Feature importance
        importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": best_model.feature_importance(importance_type="gain"),
            }
        ).sort_values("importance", ascending=False)

        self.metrics = {
            "model_type": "LightGBM_Scalp",
            "total_samples": len(X),
            "positive_rate": float(y.mean()),
            "avg_val_accuracy": float(avg_acc),
            "avg_val_auc": float(avg_auc),
            "avg_precision": float(avg_precision),
            "avg_recall": float(avg_recall),
            "fold_metrics": fold_metrics,
            "top_features": importance.head(15).to_dict("records"),
            "trained_at": datetime.now().isoformat(),
        }

        logger.info(f"\nScalp Model Training Complete:")
        logger.info(f"  Accuracy: {avg_acc:.1%}")
        logger.info(f"  AUC: {avg_auc:.3f}")
        logger.info(f"  Precision: {avg_precision:.1%}")
        logger.info(f"  Recall: {avg_recall:.1%}")

        return self.metrics

    def save(
        self,
        model_path: str = "ai/scalp_model.pkl",
        meta_path: str = "ai/scalp_model_meta.json",
    ):
        """Save model"""
        if self.model is None:
            raise ValueError("No model to save")

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        meta = {
            "trained_at": self.metrics.get("trained_at"),
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "model_path": model_path,
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(f"Scalp model saved to {model_path}")

    def predict(self, features: Dict) -> Dict:
        """Predict spike continuation probability"""
        if self.model is None:
            return {"error": "Model not loaded"}

        X = pd.DataFrame([features])

        # Ensure correct column order
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_names]

        prob = self.model.predict(X)[0]

        return {
            "prob_continue": float(prob),
            "signal": "CONTINUE" if prob > 0.6 else "FADE" if prob < 0.4 else "NEUTRAL",
            "confidence": float(abs(prob - 0.5) * 2),
        }


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def run_scalp_training(symbols: List[str] = None, days: int = 30):
    """Run full scalp training pipeline"""
    if symbols is None:
        symbols = SCALP_SYMBOLS

    logger.info("=" * 60)
    logger.info("SCALP MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Training for 2-5 minute spike continuation")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Days of data: {days}")

    trainer = ScalpModelTrainer()

    # Collect data
    X, y = trainer.collect_training_data(symbols, days=days)

    # Train
    metrics = trainer.train(X, y)

    # Save
    trainer.save()

    # Summary
    print("\n" + "=" * 60)
    print("SCALP MODEL SUMMARY")
    print("=" * 60)
    print(f"Total spike events: {metrics['total_samples']:,}")
    print(f"Spikes that continued: {metrics['positive_rate']:.1%}")
    print(f"\nValidation Metrics:")
    print(f"  Accuracy: {metrics['avg_val_accuracy']:.1%}")
    print(f"  AUC: {metrics['avg_val_auc']:.3f}")
    print(f"  Precision: {metrics['avg_precision']:.1%}")
    print(f"  Recall: {metrics['avg_recall']:.1%}")
    print(f"\nTop Features:")
    for f in metrics["top_features"][:10]:
        print(f"  {f['feature']}: {f['importance']:.0f}")
    print("=" * 60)

    return trainer, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scalp Model Training")
    parser.add_argument("--days", type=int, default=30, help="Days of minute data")

    args = parser.parse_args()

    trainer, metrics = run_scalp_training(days=args.days)
