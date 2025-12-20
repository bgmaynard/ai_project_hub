"""
Qlib Model Trainer
==================
Trains a LightGBM model on Alpha158 features for stock prediction.

Usage:
    from ai.qlib_trainer import train_qlib_model
    model, metrics = train_qlib_model(symbols=['AAPL', 'TSLA', 'NVDA'])

The trained model predicts future returns (1-5 day) based on Alpha158 features.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import os
import pickle
import json

logger = logging.getLogger(__name__)

# Model save path
MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, "qlib_model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "qlib_model_meta.json")


def collect_training_data(
    symbols: List[str],
    period: str = "2y",
    forward_days: int = 5
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Collect training data for multiple symbols.

    Args:
        symbols: List of stock tickers
        period: Historical period (1y, 2y, etc.)
        forward_days: Days ahead for return prediction

    Returns:
        (features_df, labels_series)
    """
    try:
        from ai.qlib_predictor import Alpha158Calculator
    except ImportError:
        from qlib_predictor import Alpha158Calculator

    calculator = Alpha158Calculator()
    all_features = []
    all_labels = []

    print(f"Collecting data for {len(symbols)} symbols...")

    for i, symbol in enumerate(symbols):
        try:
            print(f"  [{i+1}/{len(symbols)}] {symbol}...", end=" ")

            # Get historical data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

            if df.empty or len(df) < 100:
                print("insufficient data")
                continue

            # Reset index to get Date as column
            df = df.reset_index()

            # Standardize column names - lowercase and replace spaces
            df.columns = [str(c).lower().replace(' ', '_') for c in df.columns]

            # Rename common variations
            rename_map = {
                'datetime': 'date',
                'adj_close': 'close',
                'stock_splits': 'splits',
            }
            df = df.rename(columns=rename_map)

            # Ensure we have required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"missing columns: {missing}")
                continue

            # Keep only needed columns and set date as index
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)  # Remove timezone
            df = df.set_index('date')

            # Alpha158Calculator expects capitalized columns
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Compute features
            features = calculator.compute_features(df)

            if features.empty:
                print("no features")
                continue

            # Compute forward returns (label)
            # Use forward_days return as the prediction target
            features['forward_return'] = df['Close'].shift(-forward_days) / df['Close'] - 1

            # Drop rows without labels
            features = features.dropna(subset=['forward_return'])

            if len(features) < 50:
                print("too few samples")
                continue

            # Add symbol column
            features['symbol'] = symbol

            all_features.append(features)
            print(f"{len(features)} samples")

        except Exception as e:
            print(f"error: {e}")
            continue

    if not all_features:
        raise ValueError("No training data collected")

    # Combine all data
    combined = pd.concat(all_features, ignore_index=True)
    print(f"\nTotal samples: {len(combined)}")

    # Extract labels
    labels = combined['forward_return']

    # Remove non-feature columns
    feature_cols = [c for c in combined.columns if c not in ['forward_return', 'symbol', 'date']]
    features_df = combined[feature_cols]

    # Fill NaN with 0 (some features may be undefined for early bars)
    features_df = features_df.fillna(0)

    # Replace inf with large values
    features_df = features_df.replace([np.inf, -np.inf], 0)

    return features_df, labels


def train_model(
    features: pd.DataFrame,
    labels: pd.Series,
    test_size: float = 0.2
) -> Tuple[object, Dict]:
    """
    Train LightGBM model on Alpha158 features.

    Args:
        features: Feature DataFrame
        labels: Target labels (forward returns)
        test_size: Fraction of data for testing

    Returns:
        (trained_model, metrics_dict)
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    print("\nTraining LightGBM model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, shuffle=False
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Model parameters
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }

    # Train with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[test_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    # Evaluate
    y_pred = model.predict(X_test)

    metrics = {
        'mse': float(mean_squared_error(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'r2': float(r2_score(y_test, y_pred)),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'best_iteration': model.best_iteration,
        'num_features': len(features.columns)
    }

    # Calculate directional accuracy (predicting up vs down)
    correct_direction = ((y_pred > 0) == (y_test > 0)).mean()
    metrics['directional_accuracy'] = float(correct_direction)

    print(f"\n  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Directional Accuracy: {metrics['directional_accuracy']:.1%}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    metrics['top_features'] = importance.head(20).to_dict('records')

    print("\n  Top 10 Features:")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.0f}")

    return model, metrics


def save_model(model, metrics: Dict, feature_names: List[str]):
    """Save trained model and metadata"""
    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'metrics': metrics,
        'feature_names': feature_names,
        'model_path': MODEL_PATH
    }

    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")


def load_model() -> Tuple[Optional[object], Optional[Dict]]:
    """Load trained model and metadata"""
    if not os.path.exists(MODEL_PATH):
        return None, None

    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)

        return model, metadata

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None


def train_qlib_model(
    symbols: List[str] = None,
    period: str = "2y",
    forward_days: int = 5,
    save: bool = True
) -> Tuple[object, Dict]:
    """
    Main training function.

    Args:
        symbols: List of symbols to train on (defaults to diverse set)
        period: Historical data period
        forward_days: Days ahead to predict
        save: Whether to save the model

    Returns:
        (model, metrics)
    """
    if symbols is None:
        # Default training universe - mix of market caps and sectors
        symbols = [
            # Large cap tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            # Large cap other
            'JPM', 'JNJ', 'V', 'UNH', 'HD', 'PG', 'MA',
            # Mid cap
            'PANW', 'CRWD', 'DDOG', 'NET', 'SNOW', 'ZS',
            # Small cap / momentum
            'SOUN', 'IONQ', 'RGTI', 'QUBT', 'RKLB',
            # ETFs
            'SPY', 'QQQ', 'IWM', 'XLK', 'XLF'
        ]

    print("=" * 60)
    print("QLIB MODEL TRAINING")
    print("=" * 60)
    print(f"Symbols: {len(symbols)}")
    print(f"Period: {period}")
    print(f"Forward Days: {forward_days}")
    print("=" * 60)

    # Collect data
    features, labels = collect_training_data(symbols, period, forward_days)

    # Train model
    model, metrics = train_model(features, labels)

    # Save if requested
    if save:
        save_model(model, metrics, list(features.columns))

    return model, metrics


def evaluate_on_recent(model, symbols: List[str] = None, days: int = 30) -> Dict:
    """
    Evaluate model on recent data (out-of-sample).

    Args:
        model: Trained model
        symbols: Symbols to evaluate
        days: Recent days to evaluate

    Returns:
        Evaluation metrics
    """
    try:
        from ai.qlib_predictor import Alpha158Calculator
    except ImportError:
        from qlib_predictor import Alpha158Calculator

    if symbols is None:
        symbols = ['AAPL', 'TSLA', 'NVDA', 'SOUN', 'SPY']

    calculator = Alpha158Calculator()
    predictions = []
    actuals = []

    print(f"\nEvaluating on last {days} days...")

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="3mo")

            if df.empty:
                continue

            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            if 'date' in df.columns:
                df = df.set_index('date')

            features = calculator.compute_features(df)

            if features.empty:
                continue

            # Get last N days of features
            recent = features.tail(days + 5)

            # Compute actual returns
            recent['actual_return'] = df['close'].shift(-5) / df['close'] - 1
            recent = recent.dropna(subset=['actual_return'])

            if len(recent) < 5:
                continue

            # Predict
            feature_cols = [c for c in recent.columns if c not in ['actual_return', 'symbol']]
            X = recent[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
            preds = model.predict(X)

            predictions.extend(preds)
            actuals.extend(recent['actual_return'].values)

        except Exception as e:
            logger.warning(f"Error evaluating {symbol}: {e}")
            continue

    if not predictions:
        return {'error': 'No predictions generated'}

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    results = {
        'rmse': float(np.sqrt(mean_squared_error(actuals, predictions))),
        'mae': float(mean_absolute_error(actuals, predictions)),
        'directional_accuracy': float(((predictions > 0) == (actuals > 0)).mean()),
        'samples': len(predictions)
    }

    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  MAE: {results['mae']:.4f}")
    print(f"  Directional Accuracy: {results['directional_accuracy']:.1%}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Train model
    model, metrics = train_qlib_model()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.1%}")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"Best Iteration: {metrics['best_iteration']}")

    # Evaluate on recent data
    eval_results = evaluate_on_recent(model)
