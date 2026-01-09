"""
Adaptive Model Weighting System
===============================
Dynamically adjusts prediction model weights based on:
1. Recent prediction accuracy
2. Market regime performance
3. Statistical significance of performance differences
4. Time-decay of old observations

This creates a self-improving ensemble that automatically
shifts weight toward better-performing models.

Architecture:
- Exponential moving average of accuracy
- Regime-specific weight tracking
- Automatic rebalancing
- Performance-based promotion/demotion

Created: December 2025
"""

import json
import logging
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ModelName(Enum):
    """Available prediction models"""

    LIGHTGBM = "lightgbm"
    ENSEMBLE = "ensemble"
    HEURISTIC = "heuristic"
    MOMENTUM = "momentum"
    RL_AGENT = "rl_agent"
    SENTIMENT = "sentiment"


@dataclass
class ModelPerformance:
    """Track individual model performance"""

    name: str

    # Running statistics
    total_predictions: int = 0
    correct_predictions: int = 0

    # Exponential moving average (EMA) of accuracy
    ema_accuracy: float = 0.5  # Start neutral
    ema_alpha: float = 0.1  # EMA smoothing factor

    # Recent window accuracy
    recent_window: deque = field(default_factory=lambda: deque(maxlen=50))

    # Regime-specific performance
    regime_accuracy: Dict[str, float] = field(default_factory=dict)
    regime_counts: Dict[str, int] = field(default_factory=dict)

    # Confidence intervals
    last_ci_low: float = 0.0
    last_ci_high: float = 1.0

    # Weight assigned
    current_weight: float = 0.2

    # Timestamps
    last_prediction: str = ""
    last_update: str = ""

    def update(self, correct: bool, regime: Optional[str] = None):
        """Update performance with new prediction outcome"""
        outcome = 1.0 if correct else 0.0

        # Update counts
        self.total_predictions += 1
        if correct:
            self.correct_predictions += 1

        # Update EMA
        self.ema_accuracy = (
            self.ema_alpha * outcome + (1 - self.ema_alpha) * self.ema_accuracy
        )

        # Update recent window
        self.recent_window.append(outcome)

        # Update regime-specific stats
        if regime:
            if regime not in self.regime_accuracy:
                self.regime_accuracy[regime] = 0.5
                self.regime_counts[regime] = 0

            count = self.regime_counts[regime]
            alpha = min(0.2, 2 / (count + 2))  # Adaptive alpha
            self.regime_accuracy[regime] = (
                alpha * outcome + (1 - alpha) * self.regime_accuracy[regime]
            )
            self.regime_counts[regime] = count + 1

        # Update confidence interval (Wilson score interval)
        if self.total_predictions >= 5:
            n = self.total_predictions
            p = self.correct_predictions / n
            z = 1.96  # 95% CI

            denominator = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denominator
            margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

            self.last_ci_low = max(0, center - margin)
            self.last_ci_high = min(1, center + margin)

        self.last_update = datetime.now().isoformat()

    @property
    def overall_accuracy(self) -> float:
        """Overall accuracy rate"""
        if self.total_predictions == 0:
            return 0.5
        return self.correct_predictions / self.total_predictions

    @property
    def recent_accuracy(self) -> float:
        """Accuracy over recent window"""
        if not self.recent_window:
            return 0.5
        return sum(self.recent_window) / len(self.recent_window)

    def get_regime_accuracy(self, regime: str) -> float:
        """Get accuracy for specific regime"""
        return self.regime_accuracy.get(regime, 0.5)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "overall_accuracy": self.overall_accuracy,
            "ema_accuracy": self.ema_accuracy,
            "recent_accuracy": self.recent_accuracy,
            "current_weight": self.current_weight,
            "ci_95": (self.last_ci_low, self.last_ci_high),
            "regime_accuracy": self.regime_accuracy,
            "last_update": self.last_update,
        }


@dataclass
class WeightUpdate:
    """Record of weight update"""

    timestamp: str
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    reason: str
    regime: Optional[str]


class AdaptiveWeightManager:
    """
    Manages adaptive weights for ensemble models.
    Automatically adjusts based on performance.
    """

    def __init__(
        self,
        min_weight: float = 0.05,  # Minimum weight per model
        max_weight: float = 0.6,  # Maximum weight per model
        rebalance_threshold: float = 0.1,  # Min change to trigger rebalance
        min_predictions_for_adjust: int = 10,  # Min predictions before adjusting
        state_path: str = "store/adaptive_weights.json",
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.rebalance_threshold = rebalance_threshold
        self.min_predictions_for_adjust = min_predictions_for_adjust
        self.state_path = state_path

        # Model performance tracking
        self.models: Dict[str, ModelPerformance] = {}

        # Initialize default models
        for model in ModelName:
            self.models[model.value] = ModelPerformance(
                name=model.value,
                current_weight=1.0 / len(ModelName),  # Equal initial weights
            )

        # Weight history
        self.weight_history: List[WeightUpdate] = []
        self.max_history = 100

        # Current regime
        self.current_regime: Optional[str] = None

        # Load state
        self._load_state()

        logger.info("AdaptiveWeightManager initialized")

    def record_prediction(
        self, model: str, prediction: int, actual: int, regime: Optional[str] = None
    ):
        """
        Record a prediction outcome for a model.

        Args:
            model: Model name
            prediction: Model's prediction (1=up, 0=down)
            actual: Actual outcome (1=up, 0=down)
            regime: Optional market regime
        """
        if model not in self.models:
            self.models[model] = ModelPerformance(
                name=model, current_weight=self.min_weight
            )

        correct = prediction == actual
        self.models[model].update(correct, regime)
        self.models[model].last_prediction = datetime.now().isoformat()

        logger.debug(
            f"Recorded {model}: pred={prediction}, actual={actual}, correct={correct}"
        )

        # Check if rebalancing needed
        self._maybe_rebalance(regime)

    def get_weights(self, regime: Optional[str] = None) -> Dict[str, float]:
        """
        Get current weights for all models.
        Optionally regime-specific.

        Args:
            regime: Market regime for context

        Returns:
            Dict of model -> weight
        """
        if regime:
            return self._calculate_regime_weights(regime)
        return {name: model.current_weight for name, model in self.models.items()}

    def _calculate_regime_weights(self, regime: str) -> Dict[str, float]:
        """Calculate regime-specific weights"""
        weights = {}
        total = 0

        for name, model in self.models.items():
            # Use regime-specific accuracy if available
            acc = model.get_regime_accuracy(regime)

            # Blend with overall EMA
            blended_acc = 0.7 * acc + 0.3 * model.ema_accuracy

            # Transform to weight (softmax-like)
            weight = math.exp(2 * (blended_acc - 0.5))  # Exponential scaling
            weights[name] = weight
            total += weight

        # Normalize
        for name in weights:
            weights[name] = max(
                self.min_weight, min(self.max_weight, weights[name] / total)
            )

        # Renormalize to sum to 1
        total = sum(weights.values())
        return {name: w / total for name, w in weights.items()}

    def _maybe_rebalance(self, regime: Optional[str] = None):
        """Check if rebalancing is needed and do it"""
        # Only rebalance if enough predictions
        total_preds = sum(m.total_predictions for m in self.models.values())
        if total_preds < self.min_predictions_for_adjust * len(self.models):
            return

        # Calculate new weights based on performance
        new_weights = self._calculate_optimal_weights(regime)

        # Check if change is significant
        old_weights = {name: m.current_weight for name, m in self.models.items()}
        max_change = max(
            abs(new_weights[n] - old_weights.get(n, 0)) for n in new_weights
        )

        if max_change >= self.rebalance_threshold:
            self._apply_weights(new_weights, regime, "Performance-based rebalance")

    def _calculate_optimal_weights(
        self, regime: Optional[str] = None
    ) -> Dict[str, float]:
        """Calculate optimal weights based on performance"""
        weights = {}

        for name, model in self.models.items():
            # Base weight on EMA accuracy
            base_score = model.ema_accuracy

            # Adjust for regime if available
            if regime and regime in model.regime_accuracy:
                regime_score = model.regime_accuracy[regime]
                # Blend: 60% regime-specific, 40% overall
                base_score = 0.6 * regime_score + 0.4 * base_score

            # Adjust for confidence (penalize uncertainty)
            ci_width = model.last_ci_high - model.last_ci_low
            confidence_factor = 1 - (ci_width / 2)  # Narrower CI = higher factor

            # Adjust for recent performance (momentum)
            if len(model.recent_window) >= 10:
                recent_score = model.recent_accuracy
                # If recent is much worse than overall, penalize
                momentum = recent_score - base_score
                base_score += 0.3 * momentum

            # Final score
            score = base_score * confidence_factor

            # Transform to weight using softmax
            weights[name] = math.exp(3 * (score - 0.5))

        # Normalize
        total = sum(weights.values())
        if total == 0:
            # Equal weights if all zero
            return {name: 1.0 / len(self.models) for name in self.models}

        # Apply min/max constraints
        normalized = {}
        for name, w in weights.items():
            normalized[name] = max(self.min_weight, min(self.max_weight, w / total))

        # Renormalize to sum to 1
        total = sum(normalized.values())
        return {name: w / total for name, w in normalized.items()}

    def _apply_weights(
        self, new_weights: Dict[str, float], regime: Optional[str], reason: str
    ):
        """Apply new weights and record history"""
        old_weights = {name: m.current_weight for name, m in self.models.items()}

        # Apply
        for name, weight in new_weights.items():
            if name in self.models:
                self.models[name].current_weight = weight

        # Record
        update = WeightUpdate(
            timestamp=datetime.now().isoformat(),
            old_weights=old_weights,
            new_weights=new_weights,
            reason=reason,
            regime=regime,
        )
        self.weight_history.append(update)
        if len(self.weight_history) > self.max_history:
            self.weight_history.pop(0)

        logger.info(f"Rebalanced weights: {new_weights} (reason: {reason})")

        # Save state
        self._save_state()

    def force_rebalance(self, regime: Optional[str] = None):
        """Force an immediate rebalance"""
        new_weights = self._calculate_optimal_weights(regime)
        self._apply_weights(new_weights, regime, "Manual rebalance")

    def get_model_stats(self) -> Dict[str, Dict]:
        """Get detailed stats for all models"""
        return {name: model.to_dict() for name, model in self.models.items()}

    def get_best_model(self, regime: Optional[str] = None) -> str:
        """Get the currently best-performing model"""
        if regime:
            weights = self._calculate_regime_weights(regime)
        else:
            weights = self.get_weights()

        return max(weights.items(), key=lambda x: x[1])[0]

    def get_ranking(self, regime: Optional[str] = None) -> List[Dict]:
        """Get models ranked by performance"""
        rankings = []
        for name, model in self.models.items():
            acc = model.get_regime_accuracy(regime) if regime else model.ema_accuracy
            rankings.append(
                {
                    "name": name,
                    "weight": model.current_weight,
                    "accuracy": acc,
                    "ema_accuracy": model.ema_accuracy,
                    "recent_accuracy": model.recent_accuracy,
                    "predictions": model.total_predictions,
                    "ci_95": (model.last_ci_low, model.last_ci_high),
                }
            )

        rankings.sort(key=lambda x: x["accuracy"], reverse=True)
        return rankings

    def _save_state(self):
        """Save state to disk"""
        try:
            Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)

            state = {
                "models": {
                    name: {
                        "total_predictions": m.total_predictions,
                        "correct_predictions": m.correct_predictions,
                        "ema_accuracy": m.ema_accuracy,
                        "current_weight": m.current_weight,
                        "regime_accuracy": m.regime_accuracy,
                        "regime_counts": m.regime_counts,
                        "last_ci_low": m.last_ci_low,
                        "last_ci_high": m.last_ci_high,
                        "recent_window": list(m.recent_window),
                        "last_update": m.last_update,
                    }
                    for name, m in self.models.items()
                },
                "weight_history": [
                    {
                        "timestamp": u.timestamp,
                        "old_weights": u.old_weights,
                        "new_weights": u.new_weights,
                        "reason": u.reason,
                        "regime": u.regime,
                    }
                    for u in self.weight_history[-20:]  # Last 20
                ],
                "saved_at": datetime.now().isoformat(),
            }

            with open(self.state_path, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save adaptive weights state: {e}")

    def _load_state(self):
        """Load state from disk"""
        try:
            if Path(self.state_path).exists():
                with open(self.state_path, "r") as f:
                    state = json.load(f)

                for name, data in state.get("models", {}).items():
                    if name in self.models:
                        m = self.models[name]
                        m.total_predictions = data["total_predictions"]
                        m.correct_predictions = data["correct_predictions"]
                        m.ema_accuracy = data["ema_accuracy"]
                        m.current_weight = data["current_weight"]
                        m.regime_accuracy = data.get("regime_accuracy", {})
                        m.regime_counts = data.get("regime_counts", {})
                        m.last_ci_low = data.get("last_ci_low", 0.0)
                        m.last_ci_high = data.get("last_ci_high", 1.0)
                        m.recent_window = deque(
                            data.get("recent_window", []), maxlen=50
                        )
                        m.last_update = data.get("last_update", "")

                logger.info(f"Loaded adaptive weights state from {self.state_path}")
        except Exception as e:
            logger.warning(f"Could not load adaptive weights state: {e}")


class AdaptiveEnsemble:
    """
    Ensemble predictor with adaptive weights.
    Combines predictions from multiple models using learned weights.
    """

    def __init__(self, weight_manager: Optional[AdaptiveWeightManager] = None):
        self.weight_manager = weight_manager or get_weight_manager()

        # Model instances (lazy loaded)
        self._lgb_predictor = None
        self._ensemble_predictor = None
        self._rl_agent = None

    def predict(
        self, symbol: str, df, regime: Optional[str] = None  # pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate weighted ensemble prediction.

        Args:
            symbol: Stock symbol
            df: Price data DataFrame
            regime: Optional market regime

        Returns:
            Dict with prediction details
        """
        predictions = {}
        scores = {}

        # Get weights
        weights = self.weight_manager.get_weights(regime)

        # Get predictions from each model
        # LightGBM
        try:
            if self._lgb_predictor is None:
                from ai.alpaca_ai_predictor import get_alpaca_predictor

                self._lgb_predictor = get_alpaca_predictor()

            if self._lgb_predictor and self._lgb_predictor.model:
                result = self._lgb_predictor.predict(symbol, df=df)
                if result:
                    scores["lightgbm"] = result.get("prob_up", 0.5)
                    predictions["lightgbm"] = 1 if scores["lightgbm"] > 0.5 else 0
        except Exception as e:
            logger.debug(f"LightGBM prediction failed: {e}")
            scores["lightgbm"] = 0.5
            predictions["lightgbm"] = -1  # Unknown

        # Ensemble (includes heuristics and momentum)
        try:
            if self._ensemble_predictor is None:
                from ai.ensemble_predictor import get_ensemble_predictor

                self._ensemble_predictor = get_ensemble_predictor()

            result = self._ensemble_predictor.predict(symbol, df)
            scores["ensemble"] = (
                result.lgb_score * 0.4
                + result.heuristic_score * 0.35
                + result.momentum_score * 0.25
            )
            predictions["ensemble"] = result.prediction

            # Also record component scores
            scores["heuristic"] = result.heuristic_score
            scores["momentum"] = result.momentum_score
            predictions["heuristic"] = 1 if result.heuristic_score > 0.5 else 0
            predictions["momentum"] = 1 if result.momentum_score > 0.5 else 0
        except Exception as e:
            logger.debug(f"Ensemble prediction failed: {e}")
            for m in ["ensemble", "heuristic", "momentum"]:
                scores[m] = 0.5
                predictions[m] = -1

        # RL Agent
        try:
            if self._rl_agent is None:
                from ai.rl_trading_agent import get_rl_agent

                self._rl_agent = get_rl_agent()

            probs = self._rl_agent.get_action_probabilities(
                self._build_rl_state(df, scores)
            )
            # Convert action probs to bullish score
            # BUY prob contributes positively, SELL negatively
            rl_score = 0.5 + 0.5 * (probs["BUY"] - probs["SELL"])
            scores["rl_agent"] = rl_score
            predictions["rl_agent"] = (
                1
                if probs["recommended"] == "BUY"
                else (0 if probs["recommended"] == "SELL" else -1)
            )
        except Exception as e:
            logger.debug(f"RL agent prediction failed: {e}")
            scores["rl_agent"] = 0.5
            predictions["rl_agent"] = -1

        # Calculate weighted score
        weighted_score = 0
        total_weight = 0
        for model, score in scores.items():
            if model in weights:
                weighted_score += score * weights[model]
                total_weight += weights[model]

        if total_weight > 0:
            weighted_score /= total_weight

        # Final prediction
        final_prediction = 1 if weighted_score > 0.5 else 0
        confidence = abs(weighted_score - 0.5) * 2  # 0 to 1

        return {
            "symbol": symbol,
            "prediction": final_prediction,
            "prediction_label": "BULLISH" if final_prediction == 1 else "BEARISH",
            "weighted_score": weighted_score,
            "confidence": confidence,
            "regime": regime,
            "model_scores": scores,
            "model_predictions": predictions,
            "weights_used": weights,
            "timestamp": datetime.now().isoformat(),
        }

    def _build_rl_state(self, df, scores: Dict) -> Any:
        """Build state for RL agent"""
        import ta
        from ai.rl_trading_agent import TradingState

        close = df["Close"]

        # Calculate basic features
        returns = close.pct_change().dropna()
        rsi = ta.momentum.rsi(close, window=14).iloc[-1] if len(close) > 14 else 50

        return TradingState(
            price_change_1d=returns.iloc[-1] if len(returns) > 0 else 0,
            price_change_5d=(
                (close.iloc[-1] / close.iloc[-5] - 1) if len(close) > 5 else 0
            ),
            price_change_20d=(
                (close.iloc[-1] / close.iloc[-20] - 1) if len(close) > 20 else 0
            ),
            rsi_normalized=(rsi - 50) / 50,
            macd_normalized=0,
            bb_position=0,
            volume_ratio=1,
            trend_strength=0,
            trend_direction=1 if close.iloc[-1] > close.iloc[-5] else -1,
            volatility=returns.std() if len(returns) > 5 else 0.02,
            ensemble_score=scores.get("ensemble", 0.5),
            lgb_score=scores.get("lightgbm", 0.5),
            momentum_score=scores.get("momentum", 0.5),
            has_position=0,
            position_pnl=0,
            holding_time=0,
            regime_trending=0,
            regime_volatile=0,
        )

    def record_outcome(
        self,
        symbol: str,
        predictions: Dict[str, int],
        actual: int,
        regime: Optional[str] = None,
    ):
        """
        Record actual outcome and update model weights.

        Args:
            symbol: Stock symbol
            predictions: Dict of model -> prediction (from predict())
            actual: Actual outcome (1=up, 0=down)
            regime: Market regime
        """
        for model, prediction in predictions.items():
            if prediction >= 0:  # Valid prediction
                self.weight_manager.record_prediction(model, prediction, actual, regime)


# Global instances
_weight_manager: Optional[AdaptiveWeightManager] = None
_adaptive_ensemble: Optional[AdaptiveEnsemble] = None


def get_weight_manager() -> AdaptiveWeightManager:
    """Get or create global weight manager"""
    global _weight_manager
    if _weight_manager is None:
        _weight_manager = AdaptiveWeightManager()
    return _weight_manager


def get_adaptive_ensemble() -> AdaptiveEnsemble:
    """Get or create global adaptive ensemble"""
    global _adaptive_ensemble
    if _adaptive_ensemble is None:
        _adaptive_ensemble = AdaptiveEnsemble()
    return _adaptive_ensemble


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Adaptive Weight Manager")
    print("=" * 60)

    # Create manager
    manager = get_weight_manager()

    # Simulate predictions
    print("\nSimulating model predictions...")

    # LightGBM does well
    for _ in range(20):
        manager.record_prediction("lightgbm", 1, 1, "TRENDING_UP")  # Correct
    for _ in range(8):
        manager.record_prediction("lightgbm", 1, 0, "TRENDING_UP")  # Wrong

    # Ensemble also does well
    for _ in range(18):
        manager.record_prediction("ensemble", 1, 1, "TRENDING_UP")
    for _ in range(10):
        manager.record_prediction("ensemble", 0, 0, "TRENDING_UP")

    # Heuristic is mediocre
    for _ in range(15):
        manager.record_prediction("heuristic", 1, 1, "TRENDING_UP")
    for _ in range(15):
        manager.record_prediction("heuristic", 0, 1, "TRENDING_UP")

    # Momentum struggles in trending
    for _ in range(10):
        manager.record_prediction("momentum", 1, 1, "TRENDING_UP")
    for _ in range(15):
        manager.record_prediction("momentum", 0, 1, "TRENDING_UP")

    # RL agent is learning
    for _ in range(12):
        manager.record_prediction("rl_agent", 1, 1, "TRENDING_UP")
    for _ in range(8):
        manager.record_prediction("rl_agent", 0, 0, "TRENDING_UP")

    # Force rebalance
    manager.force_rebalance("TRENDING_UP")

    # Show results
    print("\n" + "=" * 60)
    print("Model Rankings (TRENDING_UP regime)")
    print("=" * 60)

    rankings = manager.get_ranking("TRENDING_UP")
    for i, r in enumerate(rankings, 1):
        print(
            f"{i}. {r['name']:12} | weight={r['weight']:.3f} | "
            f"acc={r['accuracy']:.3f} | recent={r['recent_accuracy']:.3f} | "
            f"n={r['predictions']}"
        )

    # Show weights
    print("\n" + "=" * 60)
    print("Current Weights")
    print("=" * 60)

    weights = manager.get_weights("TRENDING_UP")
    for name, weight in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {name:12}: {weight:.3f}")

    print(f"\nBest model for TRENDING_UP: {manager.get_best_model('TRENDING_UP')}")

    # Stats
    print("\n" + "=" * 60)
    print("Detailed Model Stats")
    print("=" * 60)

    stats = manager.get_model_stats()
    for name, s in stats.items():
        print(f"\n{name}:")
        print(f"  Overall accuracy: {s['overall_accuracy']:.3f}")
        print(f"  EMA accuracy: {s['ema_accuracy']:.3f}")
        print(f"  Recent accuracy: {s['recent_accuracy']:.3f}")
        print(f"  95% CI: ({s['ci_95'][0]:.3f}, {s['ci_95'][1]:.3f})")
        print(f"  Predictions: {s['predictions']}")
