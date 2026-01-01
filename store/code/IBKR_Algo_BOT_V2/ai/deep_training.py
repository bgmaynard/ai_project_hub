"""
Deep Training Pipeline
======================
Comprehensive training for all AI models with expanded data and validation.

Features:
1. Expanded symbol universe (50+ symbols across sectors)
2. Multiple model training (LightGBM, CNN)
3. Walk-forward validation
4. Cross-validation
5. Ensemble weight optimization
6. Comprehensive metrics reporting

Usage:
    python ai/deep_training.py --all           # Train everything
    python ai/deep_training.py --lgb           # LightGBM only
    python ai/deep_training.py --cnn           # CNN only
    python ai/deep_training.py --validate      # Validation only

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

warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# TRAINING UNIVERSE
# ============================================================================

# Expanded symbol universe - diverse sectors and market caps
TRAINING_SYMBOLS = {
    # Large Cap Tech
    "mega_tech": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "AMD",
        "INTC",
        "CRM",
    ],
    # Large Cap Finance
    "finance": ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BLK", "SCHW"],
    # Large Cap Healthcare
    "healthcare": [
        "JNJ",
        "UNH",
        "PFE",
        "ABBV",
        "MRK",
        "LLY",
        "TMO",
        "ABT",
        "DHR",
        "BMY",
    ],
    # Large Cap Consumer
    "consumer": ["PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST"],
    # Large Cap Industrial
    "industrial": ["CAT", "DE", "UNP", "HON", "GE", "BA", "LMT", "RTX", "MMM", "UPS"],
    # Mid Cap Growth
    "mid_growth": [
        "PANW",
        "CRWD",
        "DDOG",
        "NET",
        "SNOW",
        "ZS",
        "TEAM",
        "MELI",
        "SQ",
        "SHOP",
    ],
    # Small Cap / Momentum
    "small_momentum": [
        "SOUN",
        "IONQ",
        "RGTI",
        "QUBT",
        "RKLB",
        "PLTR",
        "SOFI",
        "HOOD",
        "AFRM",
        "UPST",
    ],
    # ETFs (market exposure)
    "etfs": ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "XLE", "XLI", "XLP"],
    # Volatility / Special
    "volatility": ["UVXY", "VXX", "SQQQ", "TQQQ", "SPXU", "SPXL"],
}


def get_all_symbols() -> List[str]:
    """Get flattened list of all training symbols"""
    all_symbols = []
    for sector, symbols in TRAINING_SYMBOLS.items():
        all_symbols.extend(symbols)
    return list(set(all_symbols))


# ============================================================================
# DATA COLLECTION
# ============================================================================


class DataCollector:
    """Collect and prepare training data"""

    def __init__(self, cache_dir: str = "ai/data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def fetch_data(
        self, symbols: List[str], period: str = "2y", interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for symbols.

        Args:
            symbols: List of tickers
            period: Data period (1y, 2y, 5y, max)
            interval: Data interval (1d, 1h, etc.)

        Returns:
            Dictionary of symbol -> DataFrame
        """
        data = {}
        failed = []

        logger.info(f"Fetching data for {len(symbols)} symbols...")

        for i, symbol in enumerate(symbols):
            try:
                cache_file = self.cache_dir / f"{symbol}_{period}_{interval}.parquet"

                # Check cache (use if less than 1 day old)
                if cache_file.exists():
                    cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if (datetime.now() - cache_time).days < 1:
                        df = pd.read_parquet(cache_file)
                        if len(df) > 100:
                            data[symbol] = df
                            continue

                # Fetch from yfinance
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)

                if df.empty or len(df) < 100:
                    failed.append(symbol)
                    continue

                # Standardize columns
                df = df.reset_index()
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                elif "Datetime" in df.columns:
                    df["Date"] = pd.to_datetime(df["Datetime"]).dt.tz_localize(None)
                    df = df.drop("Datetime", axis=1)

                # Save to cache
                df.to_parquet(cache_file)

                data[symbol] = df

                if (i + 1) % 20 == 0:
                    logger.info(f"  Fetched {i + 1}/{len(symbols)} symbols...")

            except Exception as e:
                logger.warning(f"  Error fetching {symbol}: {e}")
                failed.append(symbol)
                continue

        logger.info(f"Successfully fetched {len(data)} symbols, {len(failed)} failed")
        if failed:
            logger.info(
                f"  Failed symbols: {failed[:10]}{'...' if len(failed) > 10 else ''}"
            )

        return data


# ============================================================================
# LIGHTGBM TRAINING
# ============================================================================


class LightGBMTrainer:
    """Enhanced LightGBM trainer with walk-forward validation"""

    def __init__(self):
        self.model = None
        self.metrics = {}
        self.feature_names = []

    def train(
        self,
        data: Dict[str, pd.DataFrame],
        forward_days: int = 5,
        n_folds: int = 5,
        use_regularization: bool = True,
    ) -> Dict:
        """
        Train LightGBM with walk-forward cross-validation.

        Args:
            data: Dictionary of symbol -> DataFrame
            forward_days: Days ahead to predict
            n_folds: Number of cross-validation folds
            use_regularization: Whether to use strong regularization

        Returns:
            Training results with metrics
        """
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import TimeSeriesSplit

        try:
            from ai.qlib_predictor import Alpha158Calculator
        except ImportError:
            from qlib_predictor import Alpha158Calculator

        calculator = Alpha158Calculator()

        # Collect features and labels
        all_features = []
        all_labels = []

        logger.info("Preparing training data...")

        for symbol, df in data.items():
            try:
                # Prepare data for Alpha158
                df_prep = df.copy()
                if "Date" in df_prep.columns:
                    df_prep = df_prep.set_index("Date")

                # Rename columns to expected format
                df_prep = df_prep.rename(
                    columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    }
                )

                # Compute Alpha158 features
                features = calculator.compute_features(df_prep)

                if features.empty or len(features) < 100:
                    continue

                # Compute forward return labels
                features["forward_return"] = (
                    df_prep["Close"].shift(-forward_days) / df_prep["Close"] - 1
                )
                features = features.dropna(subset=["forward_return"])

                if len(features) < 50:
                    continue

                # Binary label: 1 if positive return, 0 otherwise
                labels = (features["forward_return"] > 0).astype(int)

                # Remove label column from features
                feature_cols = [c for c in features.columns if c != "forward_return"]

                all_features.append(features[feature_cols])
                all_labels.append(labels)

            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue

        if not all_features:
            raise ValueError("No valid training data collected")

        # Combine data
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)

        # Handle NaN/Inf
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        self.feature_names = list(X.columns)

        logger.info(
            f"Training data: {len(X)} samples, {len(self.feature_names)} features"
        )

        # Model parameters
        if use_regularization:
            params = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": 20,  # Reduced from 31
                "max_depth": 5,  # Limit depth
                "learning_rate": 0.03,  # Lower learning rate
                "feature_fraction": 0.7,  # More dropout
                "bagging_fraction": 0.7,
                "bagging_freq": 5,
                "min_child_samples": 30,  # More samples per leaf
                "reg_alpha": 0.1,  # L1 regularization
                "reg_lambda": 0.1,  # L2 regularization
                "verbose": -1,
                "n_jobs": -1,
                "seed": 42,
            }
        else:
            params = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "n_jobs": -1,
                "seed": 42,
            }

        # Walk-forward cross-validation
        tscv = TimeSeriesSplit(n_splits=n_folds)

        fold_metrics = []
        best_model = None
        best_auc = 0

        logger.info(f"Running {n_folds}-fold walk-forward validation...")

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
                    lgb.log_evaluation(period=0),  # Suppress per-iteration logging
                ],
            )

            # Evaluate
            y_pred_proba = model.predict(X_val)
            y_pred = (y_pred_proba > 0.5).astype(int)

            train_pred = model.predict(X_train)
            train_acc = accuracy_score(y_train, (train_pred > 0.5).astype(int))

            val_acc = accuracy_score(y_val, y_pred)
            val_auc = roc_auc_score(y_val, y_pred_proba)

            fold_metrics.append(
                {
                    "fold": fold + 1,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "overfit_gap": train_acc - val_acc,
                    "best_iteration": model.best_iteration,
                }
            )

            logger.info(
                f"  Fold {fold + 1}: Train={train_acc:.3f}, Val={val_acc:.3f}, AUC={val_auc:.3f}, Gap={train_acc - val_acc:.3f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                best_model = model

        self.model = best_model

        # Final metrics
        avg_train_acc = np.mean([m["train_acc"] for m in fold_metrics])
        avg_val_acc = np.mean([m["val_acc"] for m in fold_metrics])
        avg_val_auc = np.mean([m["val_auc"] for m in fold_metrics])
        avg_overfit_gap = np.mean([m["overfit_gap"] for m in fold_metrics])

        # Feature importance
        importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": best_model.feature_importance(importance_type="gain"),
            }
        ).sort_values("importance", ascending=False)

        self.metrics = {
            "model_type": "LightGBM",
            "n_folds": n_folds,
            "total_samples": len(X),
            "n_features": len(self.feature_names),
            "avg_train_accuracy": float(avg_train_acc),
            "avg_val_accuracy": float(avg_val_acc),
            "avg_val_auc": float(avg_val_auc),
            "avg_overfit_gap": float(avg_overfit_gap),
            "fold_metrics": fold_metrics,
            "top_features": importance.head(20).to_dict("records"),
            "regularization": params if use_regularization else None,
            "trained_at": datetime.now().isoformat(),
        }

        logger.info(f"\nLightGBM Training Complete:")
        logger.info(f"  Avg Train Accuracy: {avg_train_acc:.3f}")
        logger.info(f"  Avg Val Accuracy: {avg_val_acc:.3f}")
        logger.info(f"  Avg Val AUC: {avg_val_auc:.3f}")
        logger.info(f"  Avg Overfit Gap: {avg_overfit_gap:.3f}")

        return self.metrics

    def save(
        self,
        model_path: str = "ai/qlib_model.pkl",
        meta_path: str = "ai/qlib_model_meta.json",
    ):
        """Save trained model and metadata"""
        import pickle

        if self.model is None:
            raise ValueError("No model to save")

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        # Save metadata
        meta = {
            "trained_at": self.metrics.get("trained_at", datetime.now().isoformat()),
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "model_path": model_path,
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {meta_path}")


# ============================================================================
# CNN TRAINING
# ============================================================================


class CNNTrainerPipeline:
    """CNN training pipeline using the cnn_stock_predictor module"""

    def __init__(self):
        self.predictor = None
        self.metrics = {}

    def train(
        self,
        symbols: List[str],
        days: int = 365,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> Dict:
        """
        Train CNN model.

        Args:
            symbols: List of symbols to train on
            days: Days of historical data
            epochs: Training epochs
            batch_size: Batch size

        Returns:
            Training metrics
        """
        try:
            from ai.cnn_stock_predictor import CNNStockPredictor
        except ImportError:
            logger.warning("CNN predictor not available")
            return {"error": "CNN predictor not available"}

        logger.info(f"Training CNN on {len(symbols)} symbols...")

        self.predictor = CNNStockPredictor(
            model_path="ai/cnn_model.pt", input_features=30, sequence_length=60
        )

        try:
            results = self.predictor.train(
                symbols=symbols, days=days, epochs=epochs, batch_size=batch_size
            )

            self.metrics = {
                "model_type": "CNN",
                "symbols_trained": len(symbols),
                "days": days,
                "epochs_trained": results.get("epochs_trained", 0),
                "final_accuracy": results.get("metrics", {}).get("accuracy", 0),
                "f1_score": results.get("metrics", {}).get("f1_score", 0),
                "directional_accuracy": results.get("metrics", {}).get(
                    "directional_accuracy", 0
                ),
                "best_val_loss": results.get("best_val_loss", 0),
                "trained_at": datetime.now().isoformat(),
            }

            logger.info(f"\nCNN Training Complete:")
            logger.info(f"  Final Accuracy: {self.metrics['final_accuracy']:.3f}")
            logger.info(f"  F1 Score: {self.metrics['f1_score']:.3f}")
            logger.info(
                f"  Directional Accuracy: {self.metrics['directional_accuracy']:.3f}"
            )

            return self.metrics

        except Exception as e:
            logger.error(f"CNN training failed: {e}")
            return {"error": str(e)}


# ============================================================================
# ENSEMBLE OPTIMIZATION
# ============================================================================


class EnsembleOptimizer:
    """Optimize ensemble weights based on validation performance"""

    def __init__(self):
        self.optimal_weights = None

    def optimize(self, data: Dict[str, pd.DataFrame], n_trials: int = 100) -> Dict:
        """
        Optimize ensemble component weights.

        Uses random search to find optimal weights that maximize
        prediction accuracy on validation data.
        """
        try:
            from ai.ensemble_predictor import get_ensemble_predictor
        except ImportError:
            logger.warning("Ensemble predictor not available")
            return {"error": "Ensemble predictor not available"}

        predictor = get_ensemble_predictor()

        logger.info(f"Optimizing ensemble weights with {n_trials} trials...")

        # Prepare validation data
        val_symbols = list(data.keys())[:10]  # Use first 10 symbols for validation

        best_accuracy = 0
        best_weights = None

        for trial in range(n_trials):
            # Generate random weights (sum to 1)
            raw_weights = np.random.dirichlet(np.ones(5))
            weights = {
                "lgb": raw_weights[0],
                "chronos": raw_weights[1],
                "qlib": raw_weights[2],
                "heuristic": raw_weights[3],
                "momentum": raw_weights[4],
            }

            # Set weights in predictor
            predictor.base_weights = weights

            # Evaluate on validation symbols
            correct = 0
            total = 0

            for symbol in val_symbols:
                try:
                    df = data[symbol]
                    if "Date" in df.columns:
                        df = df.set_index("Date")

                    # Get prediction
                    result = predictor.predict(symbol, df)

                    # Check against actual next-day return
                    actual_return = (df["Close"].iloc[-1] / df["Close"].iloc[-6]) - 1
                    actual_direction = 1 if actual_return > 0 else 0

                    if result.prediction == actual_direction:
                        correct += 1
                    total += 1

                except Exception:
                    continue

            if total > 0:
                accuracy = correct / total
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = weights.copy()

            if (trial + 1) % 20 == 0:
                logger.info(
                    f"  Trial {trial + 1}/{n_trials}, Best accuracy: {best_accuracy:.3f}"
                )

        self.optimal_weights = best_weights

        result = {
            "optimal_weights": best_weights,
            "best_accuracy": best_accuracy,
            "n_trials": n_trials,
            "optimized_at": datetime.now().isoformat(),
        }

        logger.info(f"\nEnsemble Optimization Complete:")
        logger.info(f"  Best Accuracy: {best_accuracy:.3f}")
        logger.info(f"  Optimal Weights: {best_weights}")

        return result


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================


class DeepTrainingPipeline:
    """Main deep training pipeline"""

    def __init__(self, output_dir: str = "ai"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.collector = DataCollector(cache_dir=str(self.output_dir / "data_cache"))
        self.lgb_trainer = LightGBMTrainer()
        self.cnn_trainer = CNNTrainerPipeline()
        self.ensemble_optimizer = EnsembleOptimizer()

        self.results = {}

    def run(
        self,
        train_lgb: bool = True,
        train_cnn: bool = True,
        optimize_ensemble: bool = True,
        symbols: List[str] = None,
        period: str = "2y",
    ) -> Dict:
        """
        Run full training pipeline.

        Args:
            train_lgb: Train LightGBM model
            train_cnn: Train CNN model
            optimize_ensemble: Optimize ensemble weights
            symbols: Symbols to use (None = use full universe)
            period: Data period

        Returns:
            Complete training results
        """
        start_time = datetime.now()

        logger.info("=" * 70)
        logger.info("DEEP TRAINING PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Started at: {start_time}")

        # Get symbols
        if symbols is None:
            symbols = get_all_symbols()

        logger.info(f"Training universe: {len(symbols)} symbols")

        # Collect data
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1: DATA COLLECTION")
        logger.info("=" * 70)

        data = self.collector.fetch_data(symbols, period=period)
        self.results["data_collection"] = {
            "symbols_requested": len(symbols),
            "symbols_fetched": len(data),
            "period": period,
        }

        # Train LightGBM
        if train_lgb:
            logger.info("\n" + "=" * 70)
            logger.info("PHASE 2: LIGHTGBM TRAINING")
            logger.info("=" * 70)

            lgb_metrics = self.lgb_trainer.train(
                data, n_folds=5, use_regularization=True
            )
            self.lgb_trainer.save()
            self.results["lightgbm"] = lgb_metrics

        # Train CNN
        if train_cnn:
            logger.info("\n" + "=" * 70)
            logger.info("PHASE 3: CNN TRAINING")
            logger.info("=" * 70)

            cnn_symbols = list(data.keys())[:50]  # Use top 50 symbols for CNN
            cnn_metrics = self.cnn_trainer.train(cnn_symbols, days=365, epochs=50)
            self.results["cnn"] = cnn_metrics

        # Optimize ensemble
        if optimize_ensemble:
            logger.info("\n" + "=" * 70)
            logger.info("PHASE 4: ENSEMBLE OPTIMIZATION")
            logger.info("=" * 70)

            ensemble_results = self.ensemble_optimizer.optimize(data, n_trials=50)
            self.results["ensemble"] = ensemble_results

        # Save results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self.results["summary"] = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "duration_minutes": duration / 60,
            "symbols_trained": len(data),
            "models_trained": [],
        }

        if train_lgb:
            self.results["summary"]["models_trained"].append("LightGBM")
        if train_cnn:
            self.results["summary"]["models_trained"].append("CNN")

        # Save full results
        results_path = self.output_dir / "deep_training_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration/60:.1f} minutes")
        logger.info(f"Symbols trained: {len(data)}")
        logger.info(f"Results saved to: {results_path}")

        # Print summary
        self._print_summary()

        return self.results

    def _print_summary(self):
        """Print training summary"""
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)

        if "lightgbm" in self.results:
            lgb = self.results["lightgbm"]
            print(f"\nLightGBM Model:")
            print(f"  Validation Accuracy: {lgb.get('avg_val_accuracy', 0):.1%}")
            print(f"  Validation AUC: {lgb.get('avg_val_auc', 0):.3f}")
            print(f"  Overfit Gap: {lgb.get('avg_overfit_gap', 0):.1%}")
            print(f"  Total Samples: {lgb.get('total_samples', 0):,}")

        if "cnn" in self.results:
            cnn = self.results["cnn"]
            if "error" not in cnn:
                print(f"\nCNN Model:")
                print(f"  Final Accuracy: {cnn.get('final_accuracy', 0):.1%}")
                print(f"  F1 Score: {cnn.get('f1_score', 0):.3f}")
                print(
                    f"  Directional Accuracy: {cnn.get('directional_accuracy', 0):.1%}"
                )
            else:
                print(f"\nCNN Model: {cnn['error']}")

        if "ensemble" in self.results:
            ens = self.results["ensemble"]
            if "error" not in ens:
                print(f"\nEnsemble Optimization:")
                print(f"  Best Accuracy: {ens.get('best_accuracy', 0):.1%}")
                if ens.get("optimal_weights"):
                    print(f"  Optimal Weights:")
                    for k, v in ens["optimal_weights"].items():
                        print(f"    {k}: {v:.3f}")

        print("\n" + "=" * 70)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deep Training Pipeline")
    parser.add_argument("--all", action="store_true", help="Train all models")
    parser.add_argument("--lgb", action="store_true", help="Train LightGBM only")
    parser.add_argument("--cnn", action="store_true", help="Train CNN only")
    parser.add_argument(
        "--ensemble", action="store_true", help="Optimize ensemble only"
    )
    parser.add_argument(
        "--period", type=str, default="2y", help="Data period (1y, 2y, 5y)"
    )
    parser.add_argument(
        "--symbols", type=str, default=None, help="Comma-separated symbols (or 'all')"
    )

    args = parser.parse_args()

    # Determine what to train
    if args.all or (not args.lgb and not args.cnn and not args.ensemble):
        train_lgb = True
        train_cnn = True
        optimize_ensemble = True
    else:
        train_lgb = args.lgb
        train_cnn = args.cnn
        optimize_ensemble = args.ensemble

    # Parse symbols
    symbols = None
    if args.symbols and args.symbols != "all":
        symbols = [s.strip() for s in args.symbols.split(",")]

    # Run pipeline
    pipeline = DeepTrainingPipeline()
    results = pipeline.run(
        train_lgb=train_lgb,
        train_cnn=train_cnn,
        optimize_ensemble=optimize_ensemble,
        symbols=symbols,
        period=args.period,
    )
