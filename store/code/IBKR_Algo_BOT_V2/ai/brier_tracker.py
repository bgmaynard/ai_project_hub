"""
Brier Score Tracker Module
Tracks prediction accuracy using Brier score and related metrics.
Enables self-healing by identifying degrading models or symbols.

Part of the Next-Gen AI Logic Blueprint.

Brier Score: BS = (1/N) * sum((prediction - outcome)^2)
- Score of 0 = perfect predictions
- Score of 0.25 = random guessing (50/50)
- Score > 0.25 = worse than random

This module tracks:
- Rolling Brier scores per symbol
- Rolling Brier scores per model/strategy
- Calibration curves (predicted vs actual probability)
- Model degradation alerts
"""

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A single prediction record"""

    symbol: str
    model_name: str
    predicted_probability: float  # 0-1, probability of "up" move
    prediction_direction: str  # "up" or "down"
    actual_outcome: Optional[int]  # 1 if up, 0 if down, None if pending
    entry_price: float
    exit_price: Optional[float]
    threshold_move: float  # % move to count as "up" (e.g., 0.5%)
    timestamp: str
    resolved_timestamp: Optional[str]
    brier_contribution: Optional[float]  # Individual Brier score


@dataclass
class ModelHealth:
    """Health status of a model/strategy"""

    model_name: str
    brier_score: float  # Lower is better (0 = perfect)
    calibration_error: float  # How far off are probabilities
    accuracy: float  # Simple accuracy %
    total_predictions: int
    recent_predictions: int  # Last 7 days
    is_degraded: bool  # True if performing poorly
    degradation_reason: Optional[str]
    last_updated: str


class BrierTracker:
    """
    Tracks prediction accuracy using Brier scores and calibration metrics.
    Provides model health monitoring and degradation detection.
    """

    def __init__(self):
        # Storage
        self.predictions: List[Prediction] = []
        self.model_health: Dict[str, ModelHealth] = {}

        # Thresholds
        self.brier_threshold = 0.30  # Worse than this = degraded
        self.calibration_threshold = 0.15  # Max acceptable calibration error
        self.min_predictions = 20  # Min predictions before judging
        self.rolling_window = 50  # Number of recent predictions to track
        self.degradation_window = 7  # Days to check for degradation

        # File paths
        self.storage_path = Path("store/brier_scores")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing data
        self._load_data()

        logger.info("BrierTracker initialized")

    def _load_data(self):
        """Load existing prediction history"""
        try:
            pred_file = self.storage_path / "predictions.json"
            if pred_file.exists():
                with open(pred_file, "r") as f:
                    data = json.load(f)
                    self.predictions = [Prediction(**p) for p in data]
                logger.info(f"Loaded {len(self.predictions)} historical predictions")
        except Exception as e:
            logger.warning(f"Could not load predictions: {e}")
            self.predictions = []

    def _save_data(self):
        """Save prediction history"""
        try:
            pred_file = self.storage_path / "predictions.json"
            data = [asdict(p) for p in self.predictions[-1000:]]  # Keep last 1000
            with open(pred_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save predictions: {e}")

    def record_prediction(
        self,
        symbol: str,
        model_name: str,
        predicted_prob: float,
        entry_price: float,
        threshold_move: float = 0.5,
    ) -> str:
        """
        Record a new prediction before outcome is known.

        Args:
            symbol: Stock symbol
            model_name: Name of model making prediction (e.g., "lstm_mtf", "momentum")
            predicted_prob: Probability of upward move (0-1)
            entry_price: Price at time of prediction
            threshold_move: % move to count as "up" (default 0.5%)

        Returns:
            Prediction ID (timestamp)
        """
        direction = "up" if predicted_prob >= 0.5 else "down"

        prediction = Prediction(
            symbol=symbol,
            model_name=model_name,
            predicted_probability=predicted_prob,
            prediction_direction=direction,
            actual_outcome=None,
            entry_price=entry_price,
            exit_price=None,
            threshold_move=threshold_move,
            timestamp=datetime.now().isoformat(),
            resolved_timestamp=None,
            brier_contribution=None,
        )

        self.predictions.append(prediction)
        self._save_data()

        logger.debug(
            f"Recorded prediction: {symbol} {direction} ({predicted_prob:.1%}) via {model_name}"
        )

        return prediction.timestamp

    def resolve_prediction(
        self, prediction_id: str, exit_price: float
    ) -> Optional[Prediction]:
        """
        Resolve a prediction with the actual outcome.

        Args:
            prediction_id: Timestamp ID of prediction
            exit_price: Price at resolution time

        Returns:
            Updated Prediction object or None if not found
        """
        for pred in self.predictions:
            if pred.timestamp == prediction_id and pred.actual_outcome is None:
                pred.exit_price = exit_price

                # Calculate actual move
                pct_move = ((exit_price - pred.entry_price) / pred.entry_price) * 100

                # Determine outcome based on threshold
                pred.actual_outcome = 1 if pct_move >= pred.threshold_move else 0

                # Calculate Brier contribution
                # Brier = (predicted_prob - actual_outcome)^2
                pred.brier_contribution = (
                    pred.predicted_probability - pred.actual_outcome
                ) ** 2

                pred.resolved_timestamp = datetime.now().isoformat()

                self._save_data()
                self._update_model_health(pred.model_name)

                logger.debug(
                    f"Resolved prediction {prediction_id}: outcome={pred.actual_outcome}, brier={pred.brier_contribution:.4f}"
                )

                return pred

        logger.warning(f"Prediction {prediction_id} not found or already resolved")
        return None

    def resolve_by_symbol(
        self, symbol: str, exit_price: float, model_name: str = None
    ) -> List[Prediction]:
        """
        Resolve all pending predictions for a symbol.
        Useful when a position is closed.
        """
        resolved = []

        for pred in self.predictions:
            if (
                pred.symbol == symbol
                and pred.actual_outcome is None
                and (model_name is None or pred.model_name == model_name)
            ):

                result = self.resolve_prediction(pred.timestamp, exit_price)
                if result:
                    resolved.append(result)

        return resolved

    def calculate_brier_score(
        self, model_name: str = None, symbol: str = None, days: int = None
    ) -> Tuple[float, int]:
        """
        Calculate Brier score for given filters.

        Returns:
            (brier_score, num_predictions)
        """
        # Filter resolved predictions
        preds = [p for p in self.predictions if p.actual_outcome is not None]

        if model_name:
            preds = [p for p in preds if p.model_name == model_name]
        if symbol:
            preds = [p for p in preds if p.symbol == symbol]
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            preds = [p for p in preds if datetime.fromisoformat(p.timestamp) >= cutoff]

        if not preds:
            return 0.25, 0  # Return random baseline if no data

        # Calculate Brier score
        brier = sum(p.brier_contribution for p in preds) / len(preds)

        return brier, len(preds)

    def calculate_calibration_error(
        self, model_name: str = None, num_bins: int = 10
    ) -> Tuple[float, Dict]:
        """
        Calculate calibration error - how well do predicted probabilities
        match actual frequencies?

        Returns:
            (calibration_error, bin_data)
        """
        preds = [p for p in self.predictions if p.actual_outcome is not None]

        if model_name:
            preds = [p for p in preds if p.model_name == model_name]

        if len(preds) < self.min_predictions:
            return 0.0, {}

        # Bin predictions by predicted probability
        bins = defaultdict(list)
        for pred in preds:
            bin_idx = min(int(pred.predicted_probability * num_bins), num_bins - 1)
            bins[bin_idx].append(pred)

        # Calculate calibration error
        total_error = 0
        total_weight = 0
        bin_data = {}

        for bin_idx, bin_preds in bins.items():
            if bin_preds:
                predicted_avg = sum(p.predicted_probability for p in bin_preds) / len(
                    bin_preds
                )
                actual_avg = sum(p.actual_outcome for p in bin_preds) / len(bin_preds)

                error = abs(predicted_avg - actual_avg)
                weight = len(bin_preds)

                total_error += error * weight
                total_weight += weight

                bin_data[bin_idx] = {
                    "predicted_avg": predicted_avg,
                    "actual_avg": actual_avg,
                    "count": len(bin_preds),
                    "error": error,
                }

        calibration_error = total_error / total_weight if total_weight > 0 else 0

        return calibration_error, bin_data

    def _update_model_health(self, model_name: str):
        """Update health status for a model"""
        brier, count = self.calculate_brier_score(model_name=model_name)
        brier_recent, count_recent = self.calculate_brier_score(
            model_name=model_name, days=self.degradation_window
        )
        calibration, _ = self.calculate_calibration_error(model_name=model_name)

        # Calculate accuracy
        preds = [
            p
            for p in self.predictions
            if p.model_name == model_name and p.actual_outcome is not None
        ]

        if preds:
            correct = sum(
                1
                for p in preds
                if (p.predicted_probability >= 0.5 and p.actual_outcome == 1)
                or (p.predicted_probability < 0.5 and p.actual_outcome == 0)
            )
            accuracy = correct / len(preds)
        else:
            accuracy = 0.5

        # Determine degradation
        is_degraded = False
        degradation_reason = None

        if count >= self.min_predictions:
            if brier > self.brier_threshold:
                is_degraded = True
                degradation_reason = (
                    f"Brier score {brier:.3f} > threshold {self.brier_threshold}"
                )
            elif calibration > self.calibration_threshold:
                is_degraded = True
                degradation_reason = f"Calibration error {calibration:.3f} > threshold {self.calibration_threshold}"
            elif brier_recent > brier * 1.5 and count_recent >= 10:
                is_degraded = True
                degradation_reason = (
                    f"Recent degradation: {brier_recent:.3f} vs historical {brier:.3f}"
                )

        health = ModelHealth(
            model_name=model_name,
            brier_score=brier,
            calibration_error=calibration,
            accuracy=accuracy,
            total_predictions=count,
            recent_predictions=count_recent,
            is_degraded=is_degraded,
            degradation_reason=degradation_reason,
            last_updated=datetime.now().isoformat(),
        )

        self.model_health[model_name] = health

        if is_degraded:
            logger.warning(f"Model {model_name} degraded: {degradation_reason}")

        return health

    def get_model_health(self, model_name: str) -> Optional[ModelHealth]:
        """Get health status for a model"""
        if model_name not in self.model_health:
            self._update_model_health(model_name)
        return self.model_health.get(model_name)

    def get_all_model_health(self) -> Dict[str, ModelHealth]:
        """Get health status for all tracked models"""
        # Update all known models
        model_names = set(p.model_name for p in self.predictions)
        for name in model_names:
            self._update_model_health(name)
        return self.model_health

    def should_use_model(self, model_name: str) -> Tuple[bool, str]:
        """
        Determine if a model should be used based on health.

        Returns:
            (should_use, reason)
        """
        health = self.get_model_health(model_name)

        if health is None or health.total_predictions < self.min_predictions:
            return True, "Insufficient data to evaluate"

        if health.is_degraded:
            return False, health.degradation_reason or "Model degraded"

        if health.accuracy < 0.45:
            return False, f"Accuracy too low: {health.accuracy:.1%}"

        return True, "Model healthy"

    def get_confidence_adjustment(
        self, model_name: str, base_confidence: float
    ) -> float:
        """
        Adjust confidence based on model health.
        Better models get slight confidence boost, worse get reduced.
        """
        health = self.get_model_health(model_name)

        if health is None or health.total_predictions < self.min_predictions:
            return base_confidence  # No adjustment

        # Brier score adjustment
        # 0.0 (perfect) -> 1.1x multiplier
        # 0.25 (random) -> 1.0x multiplier
        # 0.50 (very bad) -> 0.8x multiplier
        brier_factor = 1.1 - (health.brier_score * 0.6)
        brier_factor = max(0.7, min(1.2, brier_factor))

        adjusted = base_confidence * brier_factor

        # Cap at 0.95
        return min(0.95, max(0.1, adjusted))

    def get_symbol_performance(self, symbol: str) -> Dict:
        """Get prediction performance for a specific symbol"""
        preds = [
            p
            for p in self.predictions
            if p.symbol == symbol and p.actual_outcome is not None
        ]

        if not preds:
            return {"symbol": symbol, "predictions": 0}

        brier = sum(p.brier_contribution for p in preds) / len(preds)

        correct = sum(
            1
            for p in preds
            if (p.predicted_probability >= 0.5 and p.actual_outcome == 1)
            or (p.predicted_probability < 0.5 and p.actual_outcome == 0)
        )

        return {
            "symbol": symbol,
            "predictions": len(preds),
            "brier_score": brier,
            "accuracy": correct / len(preds),
            "avg_predicted_prob": sum(p.predicted_probability for p in preds)
            / len(preds),
            "actual_win_rate": sum(p.actual_outcome for p in preds) / len(preds),
        }

    def get_summary(self) -> Dict:
        """Get overall tracking summary"""
        total = len(self.predictions)
        resolved = len([p for p in self.predictions if p.actual_outcome is not None])
        pending = total - resolved

        if resolved > 0:
            overall_brier, _ = self.calculate_brier_score()
            overall_calibration, _ = self.calculate_calibration_error()
        else:
            overall_brier = 0.25
            overall_calibration = 0.0

        # Get degraded models
        all_health = self.get_all_model_health()
        degraded = [name for name, h in all_health.items() if h.is_degraded]

        return {
            "total_predictions": total,
            "resolved_predictions": resolved,
            "pending_predictions": pending,
            "overall_brier_score": overall_brier,
            "overall_calibration_error": overall_calibration,
            "tracked_models": len(all_health),
            "degraded_models": degraded,
            "interpretation": self._interpret_brier(overall_brier),
        }

    def _interpret_brier(self, brier: float) -> str:
        """Human-readable interpretation of Brier score"""
        if brier < 0.15:
            return "Excellent - predictions are well calibrated"
        elif brier < 0.20:
            return "Good - predictions are better than random"
        elif brier < 0.25:
            return "Fair - predictions are slightly better than random"
        elif brier < 0.30:
            return "Poor - predictions are about as good as random guessing"
        else:
            return "Very Poor - predictions are worse than random guessing"


# Singleton instance
_brier_tracker: Optional[BrierTracker] = None


def get_brier_tracker() -> BrierTracker:
    """Get or create the Brier tracker singleton"""
    global _brier_tracker
    if _brier_tracker is None:
        _brier_tracker = BrierTracker()
    return _brier_tracker
