"""
Momentum Stock Training Pipeline
=================================
Focused training on small-cap, high-momentum stocks for realistic scalping.

Target characteristics:
- Price range: $1-$20
- Small/micro cap
- High volatility
- Low float (when data available)
- Recent momentum movers

Features:
1. Momentum-focused symbol universe
2. Correlation analysis between features and outcomes
3. Walk-forward validation
4. Feature importance for momentum trading

Created: December 2024
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# MOMENTUM/SMALL-CAP SYMBOL UNIVERSE
# ============================================================================

# Small-cap momentum stocks - recent movers, low-float runners, penny stocks
MOMENTUM_SYMBOLS = {
    # Recent momentum runners (from trading sessions)
    "recent_movers": [
        "LHAI",
        "NRXS",
        "JLHL",
        "WULF",  # From watchlist
        "ATHA",
        "EVTV",  # Recent trades
        "SOUN",
        "IONQ",
        "RGTI",
        "QUBT",
        "RKLB",  # Quantum/AI hype
    ],
    # Penny stocks / Low priced momentum
    "penny_momentum": [
        "MULN",
        "FFIE",
        "GOEV",
        "NKLA",
        "LCID",  # EV pennies
        "BBIG",
        "ATER",
        "PROG",
        "SDC",
        "WISH",  # Meme/squeeze plays
        "CLOV",
        "CLVS",
        "OCGN",
        "SRNE",
        "VXRT",  # Biotech pennies
        "GNUS",
        "CTRM",
        "NAKD",
        "SNDL",
        "TLRY",  # Other small caps
    ],
    # SPACs and recent de-SPACs (volatile)
    "spacs": [
        "DWAC",
        "BKKT",
        "DNA",
        "GRAB",
        "SOFI",
        "OPEN",
        "PLTR",
        "HOOD",
        "AFRM",
        "UPST",
        "RIVN",
        "LCID",
        "PTRA",
        "FSR",
        "RIDE",
    ],
    # Small-cap biotech (catalyst driven)
    "biotech_small": [
        "NVAX",
        "MRNA",
        "BNTX",
        "INO",
        "VXRT",
        "SAVA",
        "AGEN",
        "MRIN",
        "IMVT",
        "CTXR",
        "CRBP",
        "FBIO",
        "VERU",
        "ATHX",
        "APLS",
    ],
    # Small-cap tech/growth
    "tech_small": [
        "BIGC",
        "FVRR",
        "FSLY",
        "NET",
        "DDOG",
        "ZM",
        "DOCU",
        "PTON",
        "CHWY",
        "ETSY",
        "FUBO",
        "SKLZ",
        "PUBM",
        "MGNI",
        "TTD",
    ],
    # Cannabis (volatile sector)
    "cannabis": [
        "SNDL",
        "TLRY",
        "CGC",
        "ACB",
        "CRON",
        "HEXO",
        "OGI",
        "VFF",
        "GRWG",
        "CURLF",
    ],
    # Chinese ADRs (volatile)
    "china_adr": [
        "NIO",
        "XPEV",
        "LI",
        "BABA",
        "JD",
        "PDD",
        "BIDU",
        "BILI",
        "IQ",
        "TME",
    ],
    # Energy/Mining small caps
    "energy_small": [
        "INDO",
        "IMPP",
        "PTRA",
        "EVGO",
        "CHPT",
        "PLUG",
        "FCEL",
        "BE",
        "BLDP",
        "HYLN",
    ],
    # High beta / Leveraged exposure
    "high_beta": [
        "TQQQ",
        "SQQQ",
        "SPXL",
        "SPXU",
        "UVXY",
        "VXX",
        "LABU",
        "LABD",
        "SOXL",
        "SOXS",
    ],
}


def get_momentum_symbols() -> List[str]:
    """Get flattened list of all momentum symbols"""
    all_symbols = []
    for category, symbols in MOMENTUM_SYMBOLS.items():
        all_symbols.extend(symbols)
    return list(set(all_symbols))


# ============================================================================
# DATA COLLECTION WITH FILTERING
# ============================================================================


class MomentumDataCollector:
    """Collect data for momentum stocks with price/volume filtering"""

    def __init__(self, cache_dir: str = "ai/momentum_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def fetch_and_filter(
        self,
        symbols: List[str],
        period: str = "1y",
        min_price: float = 0.50,
        max_price: float = 50.0,
        min_avg_volume: int = 500000,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        """
        Fetch data and filter by price/volume criteria.

        Returns:
            (data_dict, stats_dict)
        """
        data = {}
        stats = {
            "requested": len(symbols),
            "fetched": 0,
            "filtered_price": 0,
            "filtered_volume": 0,
            "failed": 0,
            "passed": 0,
            "symbols_used": [],
            "symbols_failed": [],
        }

        logger.info(f"Fetching data for {len(symbols)} momentum symbols...")
        logger.info(
            f"Filters: price ${min_price}-${max_price}, min vol {min_avg_volume:,}"
        )

        for i, symbol in enumerate(symbols):
            try:
                cache_file = self.cache_dir / f"{symbol}_{period}.parquet"

                # Check cache
                if cache_file.exists():
                    cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if (datetime.now() - cache_time).days < 1:
                        df = pd.read_parquet(cache_file)
                        if len(df) > 50:
                            # Apply filters
                            avg_price = df["Close"].mean()
                            avg_volume = df["Volume"].mean()

                            if avg_price < min_price or avg_price > max_price:
                                stats["filtered_price"] += 1
                                continue
                            if avg_volume < min_avg_volume:
                                stats["filtered_volume"] += 1
                                continue

                            data[symbol] = df
                            stats["symbols_used"].append(symbol)
                            stats["passed"] += 1
                            continue

                # Fetch from yfinance
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)

                if df.empty or len(df) < 50:
                    stats["failed"] += 1
                    stats["symbols_failed"].append(symbol)
                    continue

                stats["fetched"] += 1

                # Standardize
                df = df.reset_index()
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                elif "Datetime" in df.columns:
                    df["Date"] = pd.to_datetime(df["Datetime"]).dt.tz_localize(None)
                    df = df.drop("Datetime", axis=1)

                # Apply filters
                avg_price = df["Close"].mean()
                avg_volume = df["Volume"].mean()

                if avg_price < min_price or avg_price > max_price:
                    stats["filtered_price"] += 1
                    continue
                if avg_volume < min_avg_volume:
                    stats["filtered_volume"] += 1
                    continue

                # Save to cache
                df.to_parquet(cache_file)

                data[symbol] = df
                stats["symbols_used"].append(symbol)
                stats["passed"] += 1

                if (i + 1) % 20 == 0:
                    logger.info(f"  Processed {i + 1}/{len(symbols)} symbols...")

            except Exception as e:
                logger.debug(f"  Error fetching {symbol}: {e}")
                stats["failed"] += 1
                stats["symbols_failed"].append(symbol)
                continue

        logger.info(f"\nData Collection Summary:")
        logger.info(f"  Requested: {stats['requested']}")
        logger.info(f"  Passed filters: {stats['passed']}")
        logger.info(f"  Filtered by price: {stats['filtered_price']}")
        logger.info(f"  Filtered by volume: {stats['filtered_volume']}")
        logger.info(f"  Failed to fetch: {stats['failed']}")

        return data, stats


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================


class CorrelationAnalyzer:
    """Analyze correlations between features and outcomes"""

    def __init__(self):
        self.correlations = {}
        self.feature_stats = {}

    def analyze(
        self, features: pd.DataFrame, labels: pd.Series, feature_names: List[str]
    ) -> Dict:
        """
        Compute correlations between each feature and the target.

        Returns:
            Dictionary with correlation results
        """
        logger.info("Computing feature-outcome correlations...")

        results = []

        for col in feature_names:
            if col not in features.columns:
                continue

            feature_values = features[col].values
            label_values = labels.values

            # Remove NaN/Inf
            mask = np.isfinite(feature_values) & np.isfinite(label_values)
            if mask.sum() < 100:
                continue

            x = feature_values[mask]
            y = label_values[mask]

            # Pearson correlation
            try:
                pearson_r, pearson_p = stats.pearsonr(x, y)
            except:
                pearson_r, pearson_p = 0, 1

            # Spearman correlation (rank-based, more robust)
            try:
                spearman_r, spearman_p = stats.spearmanr(x, y)
            except:
                spearman_r, spearman_p = 0, 1

            # Point-biserial for binary outcome
            try:
                # For binary labels
                binary_labels = (y > 0).astype(int)
                pb_r, pb_p = stats.pointbiserialr(binary_labels, x)
            except:
                pb_r, pb_p = 0, 1

            results.append(
                {
                    "feature": col,
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_r": float(spearman_r),
                    "spearman_p": float(spearman_p),
                    "pointbiserial_r": float(pb_r),
                    "abs_correlation": abs(float(spearman_r)),
                    "significant": pearson_p < 0.05,
                }
            )

        # Sort by absolute correlation
        results = sorted(results, key=lambda x: x["abs_correlation"], reverse=True)

        self.correlations = {
            "features": results,
            "top_positive": [r for r in results if r["spearman_r"] > 0][:10],
            "top_negative": [r for r in results if r["spearman_r"] < 0][:10],
            "significant_count": sum(1 for r in results if r["significant"]),
            "total_features": len(results),
        }

        # Log top correlations
        logger.info(f"\nTop 10 Positive Correlations (feature → higher returns):")
        for r in self.correlations["top_positive"][:10]:
            logger.info(
                f"  {r['feature']}: r={r['spearman_r']:.3f} (p={r['spearman_p']:.4f})"
            )

        logger.info(f"\nTop 10 Negative Correlations (feature → lower returns):")
        for r in self.correlations["top_negative"][:10]:
            logger.info(
                f"  {r['feature']}: r={r['spearman_r']:.3f} (p={r['spearman_p']:.4f})"
            )

        return self.correlations


# ============================================================================
# MOMENTUM-FOCUSED TRAINING
# ============================================================================


class MomentumTrainer:
    """Training pipeline optimized for momentum stocks"""

    def __init__(self):
        self.model = None
        self.metrics = {}
        self.correlations = {}
        self.feature_names = []

    def prepare_data(
        self,
        data: Dict[str, pd.DataFrame],
        forward_days: int = 3,  # Shorter horizon for momentum
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare features and labels from momentum stock data"""
        try:
            from ai.qlib_predictor import Alpha158Calculator
        except ImportError:
            from qlib_predictor import Alpha158Calculator

        calculator = Alpha158Calculator()

        all_features = []
        all_labels = []
        all_returns = []

        logger.info("Preparing momentum training data...")

        for symbol, df in data.items():
            try:
                df_prep = df.copy()
                if "Date" in df_prep.columns:
                    df_prep = df_prep.set_index("Date")

                # Ensure column names
                df_prep = df_prep.rename(
                    columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    }
                )

                # Add momentum-specific features
                df_prep = self._add_momentum_features(df_prep)

                # Compute Alpha158 features
                features = calculator.compute_features(df_prep)

                if features.empty or len(features) < 50:
                    continue

                # Forward return (shorter horizon for momentum)
                features["forward_return"] = (
                    df_prep["Close"].shift(-forward_days) / df_prep["Close"] - 1
                )
                features = features.dropna(subset=["forward_return"])

                if len(features) < 30:
                    continue

                # Add our momentum features
                for col in [
                    "momentum_5",
                    "momentum_10",
                    "vol_spike",
                    "price_accel",
                    "gap_pct",
                ]:
                    if col in df_prep.columns:
                        # Align index
                        features[col] = (
                            df_prep.loc[features.index, col] if col in df_prep else 0
                        )

                # Binary label
                labels = (features["forward_return"] > 0).astype(int)
                returns = features["forward_return"]

                feature_cols = [c for c in features.columns if c != "forward_return"]

                all_features.append(features[feature_cols])
                all_labels.append(labels)
                all_returns.append(returns)

            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue

        if not all_features:
            raise ValueError("No valid training data")

        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)
        returns = pd.concat(all_returns, ignore_index=True)

        # Clean data
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        self.feature_names = list(X.columns)

        logger.info(
            f"Prepared {len(X)} samples with {len(self.feature_names)} features"
        )

        return X, y, returns

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-specific features"""
        data = df.copy()

        # Price momentum
        data["momentum_5"] = data["Close"].pct_change(5)
        data["momentum_10"] = data["Close"].pct_change(10)
        data["momentum_20"] = data["Close"].pct_change(20)

        # Volume spike
        vol_ma = data["Volume"].rolling(20).mean()
        data["vol_spike"] = data["Volume"] / vol_ma

        # Price acceleration (momentum of momentum)
        data["price_accel"] = data["momentum_5"].diff(5)

        # Gap percentage
        data["gap_pct"] = (data["Open"] - data["Close"].shift(1)) / data["Close"].shift(
            1
        )

        # Intraday range
        data["intraday_range"] = (data["High"] - data["Low"]) / data["Open"]

        # Close position in range
        data["close_position"] = (data["Close"] - data["Low"]) / (
            data["High"] - data["Low"] + 0.0001
        )

        return data

    def train(self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> Dict:
        """Train LightGBM with momentum-optimized parameters"""
        import lightgbm as lgb
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, roc_auc_score)
        from sklearn.model_selection import TimeSeriesSplit

        # Momentum-optimized parameters
        # Less regularization since momentum patterns may be more noisy
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,  # More capacity for patterns
            "max_depth": 6,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "reg_alpha": 0.05,
            "reg_lambda": 0.05,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }

        tscv = TimeSeriesSplit(n_splits=n_folds)
        fold_metrics = []
        best_model = None
        best_auc = 0

        logger.info(f"Training on momentum data with {n_folds}-fold validation...")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=0),
                ],
            )

            # Evaluate
            y_pred_proba = model.predict(X_val)
            y_pred = (y_pred_proba > 0.5).astype(int)

            train_pred = model.predict(X_train)
            train_acc = accuracy_score(y_train, (train_pred > 0.5).astype(int))

            val_acc = accuracy_score(y_val, y_pred)
            val_auc = roc_auc_score(y_val, y_pred_proba)
            val_precision = precision_score(y_val, y_pred, zero_division=0)
            val_recall = recall_score(y_val, y_pred, zero_division=0)

            fold_metrics.append(
                {
                    "fold": fold + 1,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "overfit_gap": train_acc - val_acc,
                    "best_iteration": model.best_iteration,
                }
            )

            logger.info(
                f"  Fold {fold + 1}: Acc={val_acc:.3f}, AUC={val_auc:.3f}, Prec={val_precision:.3f}, Rec={val_recall:.3f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                best_model = model

        self.model = best_model

        # Compute averages
        avg_metrics = {
            "avg_train_acc": np.mean([m["train_acc"] for m in fold_metrics]),
            "avg_val_acc": np.mean([m["val_acc"] for m in fold_metrics]),
            "avg_val_auc": np.mean([m["val_auc"] for m in fold_metrics]),
            "avg_precision": np.mean([m["val_precision"] for m in fold_metrics]),
            "avg_recall": np.mean([m["val_recall"] for m in fold_metrics]),
            "avg_overfit_gap": np.mean([m["overfit_gap"] for m in fold_metrics]),
        }

        # Feature importance
        importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": best_model.feature_importance(importance_type="gain"),
            }
        ).sort_values("importance", ascending=False)

        self.metrics = {
            "model_type": "LightGBM_Momentum",
            "n_folds": n_folds,
            "total_samples": len(X),
            "n_features": len(self.feature_names),
            **avg_metrics,
            "fold_metrics": fold_metrics,
            "top_features": importance.head(25).to_dict("records"),
            "params": params,
            "trained_at": datetime.now().isoformat(),
        }

        logger.info(f"\nMomentum Training Complete:")
        logger.info(f"  Avg Validation Accuracy: {avg_metrics['avg_val_acc']:.3f}")
        logger.info(f"  Avg Validation AUC: {avg_metrics['avg_val_auc']:.3f}")
        logger.info(f"  Avg Precision: {avg_metrics['avg_precision']:.3f}")
        logger.info(f"  Avg Recall: {avg_metrics['avg_recall']:.3f}")

        return self.metrics

    def save(
        self,
        model_path: str = "ai/momentum_model.pkl",
        meta_path: str = "ai/momentum_model_meta.json",
    ):
        """Save momentum model"""
        import pickle

        if self.model is None:
            raise ValueError("No model to save")

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        meta = {
            "trained_at": self.metrics.get("trained_at"),
            "metrics": self.metrics,
            "correlations": self.correlations,
            "feature_names": self.feature_names,
            "model_path": model_path,
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(f"Momentum model saved to {model_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================


class MomentumTrainingPipeline:
    """Complete momentum training pipeline"""

    def __init__(self, output_dir: str = "ai"):
        self.output_dir = Path(output_dir)
        self.collector = MomentumDataCollector()
        self.analyzer = CorrelationAnalyzer()
        self.trainer = MomentumTrainer()
        self.results = {}

    def run(
        self,
        symbols: List[str] = None,
        period: str = "1y",
        min_price: float = 0.50,
        max_price: float = 50.0,
        forward_days: int = 3,
    ) -> Dict:
        """Run full momentum training pipeline"""
        start_time = datetime.now()

        logger.info("=" * 70)
        logger.info("MOMENTUM STOCK TRAINING PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Focus: Small-cap, high-momentum stocks")
        logger.info(f"Price range: ${min_price} - ${max_price}")
        logger.info(f"Prediction horizon: {forward_days} days")

        # Get symbols
        if symbols is None:
            symbols = get_momentum_symbols()

        logger.info(f"Symbol universe: {len(symbols)} momentum stocks")

        # Phase 1: Data Collection
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1: DATA COLLECTION")
        logger.info("=" * 70)

        data, data_stats = self.collector.fetch_and_filter(
            symbols, period=period, min_price=min_price, max_price=max_price
        )

        self.results["data_collection"] = data_stats

        if len(data) < 10:
            raise ValueError(f"Not enough symbols passed filters: {len(data)}")

        # Phase 2: Prepare Data
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 2: DATA PREPARATION")
        logger.info("=" * 70)

        X, y, returns = self.trainer.prepare_data(data, forward_days=forward_days)

        # Phase 3: Correlation Analysis
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 3: CORRELATION ANALYSIS")
        logger.info("=" * 70)

        correlations = self.analyzer.analyze(X, returns, self.trainer.feature_names)
        self.results["correlations"] = correlations
        self.trainer.correlations = correlations

        # Phase 4: Model Training
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 4: MODEL TRAINING")
        logger.info("=" * 70)

        metrics = self.trainer.train(X, y, n_folds=5)
        self.results["training"] = metrics

        # Phase 5: Save Model
        self.trainer.save()

        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self.results["summary"] = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "symbols_trained": len(data),
            "total_samples": len(X),
            "prediction_horizon_days": forward_days,
            "price_range": f"${min_price}-${max_price}",
        }

        # Save results
        results_path = self.output_dir / "momentum_training_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        self._print_summary()

        return self.results

    def _print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "=" * 70)
        print("MOMENTUM TRAINING SUMMARY")
        print("=" * 70)

        # Data stats
        ds = self.results.get("data_collection", {})
        print(f"\nData Collection:")
        print(f"  Symbols passed filters: {ds.get('passed', 0)}")
        print(f"  Filtered by price: {ds.get('filtered_price', 0)}")
        print(f"  Filtered by volume: {ds.get('filtered_volume', 0)}")

        # Correlation insights
        corr = self.results.get("correlations", {})
        print(f"\nCorrelation Analysis:")
        print(
            f"  Significant features: {corr.get('significant_count', 0)}/{corr.get('total_features', 0)}"
        )

        if corr.get("top_positive"):
            print(f"\n  Top Positive Correlations (bullish signals):")
            for r in corr["top_positive"][:5]:
                print(f"    {r['feature']}: r={r['spearman_r']:.3f}")

        if corr.get("top_negative"):
            print(f"\n  Top Negative Correlations (bearish signals):")
            for r in corr["top_negative"][:5]:
                print(f"    {r['feature']}: r={r['spearman_r']:.3f}")

        # Model performance
        tr = self.results.get("training", {})
        print(f"\nModel Performance:")
        print(f"  Validation Accuracy: {tr.get('avg_val_acc', 0):.1%}")
        print(f"  Validation AUC: {tr.get('avg_val_auc', 0):.3f}")
        print(f"  Precision: {tr.get('avg_precision', 0):.1%}")
        print(f"  Recall: {tr.get('avg_recall', 0):.1%}")
        print(f"  Overfit Gap: {tr.get('avg_overfit_gap', 0):.1%}")

        if tr.get("top_features"):
            print(f"\n  Top 10 Features (by importance):")
            for f in tr["top_features"][:10]:
                print(f"    {f['feature']}: {f['importance']:.0f}")

        print("\n" + "=" * 70)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Momentum Stock Training")
    parser.add_argument("--period", type=str, default="1y", help="Data period")
    parser.add_argument("--min-price", type=float, default=0.50, help="Min avg price")
    parser.add_argument("--max-price", type=float, default=50.0, help="Max avg price")
    parser.add_argument(
        "--horizon", type=int, default=3, help="Prediction horizon days"
    )

    args = parser.parse_args()

    pipeline = MomentumTrainingPipeline()
    results = pipeline.run(
        period=args.period,
        min_price=args.min_price,
        max_price=args.max_price,
        forward_days=args.horizon,
    )
