"""
AlphaFusion V2 - Complete Implementation
========================================

Based on: ai_trading_bot_logic_mathematical_concept_sheet_lln_drift_sentiment_barriers.md

This module implements the full AlphaFusion algorithm with:
- LLN-style estimators (EWMA for drift/volatility)
- Online logistic regression with SGD
- Calibration & reliability scoring
- Similarity-based probability boosting
- Round/half barrier detection
- Fill probability & slippage modeling

Author: AI Trading Bot Team
Version: 2.0
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#                     DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class MarketData:
    """Real-time market data snapshot"""

    timestamp: float
    bid: float
    ask: float
    last: float
    bid_size: float
    ask_size: float
    volume: int
    vwap: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None

    @property
    def mid(self) -> float:
        """Mid price"""
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        """Absolute spread"""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Spread as % of mid"""
        return self.spread / self.mid if self.mid > 0 else 0.0


@dataclass
class Features:
    """Engineered features for prediction"""

    imbalance: float  # Order book imbalance [-1, 1]
    momentum: float  # Vol-normalized momentum
    drift: float  # Slow EWMA drift
    vwap_distance: float  # Distance from VWAP (z-score)
    spread: float  # Normalized spread
    barrier: float  # Proximity to round/half barriers [0, 1]
    sentiment: float  # External sentiment [-1, 1]

    def to_array(self) -> np.ndarray:
        """Convert to feature vector"""
        return np.array(
            [
                self.imbalance,
                self.momentum,
                self.drift,
                self.vwap_distance,
                self.spread,
                self.barrier,
                self.sentiment,
            ]
        )


@dataclass
class Prediction:
    """Model prediction output"""

    timestamp: float
    p_up: float  # Raw model probability
    p_calibrated: float  # After calibration
    p_final: float  # After reliability & similarity boost
    reliability: float  # Reliability score [0, 1]
    similarity_boost: float  # Similarity multiplier
    features: Features


@dataclass
class TradeLabel:
    """Realized outcome after horizon τ"""

    timestamp: float
    entry_price: float
    exit_price: float
    label: int  # 1 = up, 0 = down
    horizon_seconds: float


# ═══════════════════════════════════════════════════════════════════════
#                     EWMA ESTIMATORS (LLN-STYLE)
# ═══════════════════════════════════════════════════════════════════════


class EWMA:
    """Exponentially Weighted Moving Average estimator"""

    def __init__(self, alpha: float = 0.1, initial_value: float = 0.0):
        self.alpha = alpha
        self.value = initial_value
        self.initialized = False

    def update(self, x: float) -> float:
        """Update with new observation"""
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def get(self) -> float:
        return self.value


class RunningStats:
    """Welford's online algorithm for mean and variance"""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float):
        """Add new observation"""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def get_mean(self) -> float:
        return self.mean

    def get_variance(self) -> float:
        return self.m2 / self.n if self.n > 1 else 0.0

    def get_std(self) -> float:
        return math.sqrt(self.get_variance())


# ═══════════════════════════════════════════════════════════════════════
#                     FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════


class FeatureEngineer:
    """Compute all features from market data"""

    def __init__(self, volatility_alpha: float = 0.1, drift_alpha: float = 0.05):
        # EWMA estimators
        self.volatility_ewma = EWMA(alpha=volatility_alpha)
        self.drift_ewma = EWMA(alpha=drift_alpha)

        # Last values for computing returns
        self.last_mid = None
        self.last_timestamp = None

    def compute_features(self, data: MarketData, sentiment: float = 0.0) -> Features:
        """Compute all features from market snapshot"""

        # 1. Order Book Imbalance
        imbalance = self._compute_imbalance(data)

        # 2. Momentum (vol-normalized)
        momentum = self._compute_momentum(data)

        # 3. Drift (slow EWMA)
        drift = self.drift_ewma.get()

        # 4. VWAP Distance
        vwap_dist = self._compute_vwap_distance(data)

        # 5. Spread (normalized)
        spread = data.spread_pct

        # 6. Barrier proximity
        barrier = self._compute_barrier(data.mid)

        # 7. Sentiment
        sentiment = max(-1.0, min(1.0, sentiment))

        return Features(
            imbalance=imbalance,
            momentum=momentum,
            drift=drift,
            vwap_distance=vwap_dist,
            spread=spread,
            barrier=barrier,
            sentiment=sentiment,
        )

    def _compute_imbalance(self, data: MarketData) -> float:
        """Order book imbalance: (bid_size - ask_size) / (bid_size + ask_size)"""
        total = data.bid_size + data.ask_size
        if total < 1e-6:
            return 0.0
        return (data.bid_size - data.ask_size) / total

    def _compute_momentum(self, data: MarketData) -> float:
        """Vol-normalized momentum"""
        if self.last_mid is None:
            self.last_mid = data.mid
            self.last_timestamp = data.timestamp
            return 0.0

        # Compute return
        r = (data.mid - self.last_mid) / self.last_mid if self.last_mid > 0 else 0.0

        # Update volatility EWMA
        volatility = self.volatility_ewma.update(abs(r))

        # Update drift EWMA
        self.drift_ewma.update(r)

        # Update last values
        self.last_mid = data.mid
        self.last_timestamp = data.timestamp

        # Normalize by volatility
        if volatility < 1e-8:
            return 0.0
        return r / volatility

    def _compute_vwap_distance(self, data: MarketData) -> float:
        """Distance from VWAP as z-score vs 0.1% band"""
        if data.vwap is None or data.vwap <= 0:
            return 0.0

        dist = (data.mid - data.vwap) / (0.001 * data.mid)
        return max(-10.0, min(10.0, dist))  # Clip to ±10

    def _compute_barrier(self, mid_price: float) -> float:
        """Proximity to round/half barriers (x.00, x.50)"""
        fractional = mid_price - math.floor(mid_price)

        # Distance to .00
        dist_00 = abs(fractional - 0.00)
        p_00 = max(0.0, 1.0 - dist_00 / 0.05)

        # Distance to .50
        dist_50 = abs(fractional - 0.50)
        p_50 = max(0.0, 1.0 - dist_50 / 0.05)

        # Return max proximity
        return max(p_00, p_50)


# ═══════════════════════════════════════════════════════════════════════
#                     ONLINE LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════════════════


class OnlineLogisticRegression:
    """Online logistic regression with SGD"""

    def __init__(
        self, n_features: int = 7, learning_rate: float = 0.01, l2_lambda: float = 0.001
    ):
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda

        # Initialize weights (β)
        self.beta = np.zeros(n_features + 1)  # +1 for intercept
        self.n_updates = 0

    def predict(self, features: Features) -> float:
        """Predict probability of up move"""
        x = np.concatenate([[1.0], features.to_array()])  # Add intercept
        z = np.dot(self.beta, x)
        return self._sigmoid(z)

    def update(self, features: Features, label: int):
        """Update weights with SGD on logistic loss"""
        x = np.concatenate([[1.0], features.to_array()])

        # Prediction
        p = self.predict(features)

        # Error
        error = label - p

        # Gradient descent update with L2 regularization
        # β ← β - η * (-error * x + λ * β)
        gradient = -error * x + self.l2_lambda * self.beta
        self.beta -= self.learning_rate * gradient

        self.n_updates += 1

    @staticmethod
    def _sigmoid(z: float) -> float:
        """Sigmoid function"""
        return 1.0 / (1.0 + math.exp(-max(-50, min(50, z))))  # Clip for stability


# ═══════════════════════════════════════════════════════════════════════
#                     CALIBRATION & RELIABILITY
# ═══════════════════════════════════════════════════════════════════════


class Calibrator:
    """Probability calibration with binning"""

    def __init__(self, n_bins: int = 10, min_samples: int = 10):
        self.n_bins = n_bins
        self.min_samples = min_samples

        # Bins: store (prediction, label) pairs
        self.bins: List[List[Tuple[float, int]]] = [[] for _ in range(n_bins)]

        # Bin statistics
        self.bin_means: List[float] = [0.5] * n_bins  # Default to 0.5
        self.bin_hit_rates: List[float] = [0.5] * n_bins

    def add(self, prediction: float, label: int):
        """Add prediction-label pair"""
        bin_idx = int(prediction * self.n_bins)
        bin_idx = max(0, min(self.n_bins - 1, bin_idx))

        self.bins[bin_idx].append((prediction, label))

        # Keep only recent N samples per bin
        if len(self.bins[bin_idx]) > 100:
            self.bins[bin_idx] = self.bins[bin_idx][-100:]

        # Update statistics
        self._update_stats(bin_idx)

    def calibrate(self, prediction: float) -> float:
        """Apply calibration correction"""
        bin_idx = int(prediction * self.n_bins)
        bin_idx = max(0, min(self.n_bins - 1, bin_idx))

        if len(self.bins[bin_idx]) < self.min_samples:
            return prediction  # Not enough data

        # Calibration: p_cal = p * (hit_rate / mean_p)
        mean_p = self.bin_means[bin_idx]
        hit_rate = self.bin_hit_rates[bin_idx]

        if mean_p < 0.01:
            return prediction

        return max(0.0, min(1.0, prediction * (hit_rate / mean_p)))

    def get_reliability(self) -> float:
        """Compute reliability score from Brier error"""
        all_pairs = []
        for bin_data in self.bins:
            all_pairs.extend(bin_data)

        if len(all_pairs) < 10:
            return 0.5  # Not enough data

        # Compute Brier score
        brier = sum((p - y) ** 2 for p, y in all_pairs) / len(all_pairs)

        # Map to reliability [0, 1]: R = max(0, min(1, 1 - 2*Brier))
        reliability = max(0.0, min(1.0, 1.0 - 2.0 * brier))

        return reliability

    def _update_stats(self, bin_idx: int):
        """Update bin statistics"""
        if len(self.bins[bin_idx]) == 0:
            return

        predictions, labels = zip(*self.bins[bin_idx])

        self.bin_means[bin_idx] = sum(predictions) / len(predictions)
        self.bin_hit_rates[bin_idx] = sum(labels) / len(labels)


# ═══════════════════════════════════════════════════════════════════════
#                     SIMILARITY-BASED BOOSTING
# ═══════════════════════════════════════════════════════════════════════


class SimilarityBooster:
    """k-NN style similarity matching for probability boost"""

    def __init__(self, k: int = 20, max_history: int = 1000, boost_coeff: float = 0.4):
        self.k = k
        self.max_history = max_history
        self.boost_coeff = boost_coeff

        # History of (signature, outcome)
        self.history: deque = deque(maxlen=max_history)

    def add(self, features: Features, outcome: int):
        """Add historical outcome"""
        signature = features.to_array()
        self.history.append((signature, outcome))

    def compute_boost(self, features: Features) -> float:
        """Compute similarity-based probability multiplier"""
        if len(self.history) < self.k:
            return 1.0  # Not enough history

        current_sig = features.to_array()

        # Compute cosine similarities
        similarities = []
        for hist_sig, outcome in self.history:
            sim = self._cosine_similarity(current_sig, hist_sig)
            similarities.append((sim, outcome))

        # Get top-k neighbors
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_k = similarities[: self.k]

        # Win rate in top-k
        win_rate = sum(outcome for _, outcome in top_k) / self.k

        # Boost around 0.5
        boost = self.boost_coeff * (win_rate - 0.5)

        # Multiplier: M = exp(boost)
        return math.exp(boost)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        return np.dot(a, b) / (norm_a * norm_b)


# ═══════════════════════════════════════════════════════════════════════
#                     FILL PROBABILITY & SLIPPAGE
# ═══════════════════════════════════════════════════════════════════════


class SlippageTracker:
    """Track realized slippage statistics"""

    def __init__(self):
        self.market_slippage = RunningStats()
        self.limit_fills = RunningStats()  # Track fill success rate

    def add_market_slippage(self, intended: float, filled: float, side: str):
        """Record market order slippage"""
        if side == "BUY":
            slip = max(0.0, filled - intended)
        else:  # SELL
            slip = max(0.0, intended - filled)

        self.market_slippage.update(slip)

    def add_limit_fill(self, filled: bool):
        """Record limit order fill outcome"""
        self.limit_fills.update(1.0 if filled else 0.0)

    def get_expected_slippage(self) -> float:
        """Get mean slippage for market orders"""
        return self.market_slippage.get_mean()

    def get_limit_fill_rate(self) -> float:
        """Get limit order fill rate"""
        return self.limit_fills.get_mean()


# ═══════════════════════════════════════════════════════════════════════
#                     MAIN ALPHAFUSION ENGINE
# ═══════════════════════════════════════════════════════════════════════


class AlphaFusionEngine:
    """
    Complete AlphaFusion trading algorithm

    Combines:
    - Feature engineering
    - Online logistic regression
    - Calibration & reliability
    - Similarity-based boosting
    - Slippage tracking
    """

    def __init__(
        self,
        horizon_seconds: float = 2.0,
        learning_rate: float = 0.01,
        l2_lambda: float = 0.001,
        n_bins: int = 10,
        k_neighbors: int = 20,
        boost_coeff: float = 0.4,
    ):

        self.horizon_seconds = horizon_seconds

        # Components
        self.feature_engineer = FeatureEngineer()
        self.model = OnlineLogisticRegression(
            n_features=7, learning_rate=learning_rate, l2_lambda=l2_lambda
        )
        self.calibrator = Calibrator(n_bins=n_bins)
        self.similarity = SimilarityBooster(k=k_neighbors, boost_coeff=boost_coeff)
        self.slippage_tracker = SlippageTracker()

        # Pending predictions waiting for labels
        self.pending: deque = deque(maxlen=1000)

    def predict(self, data: MarketData, sentiment: float = 0.0) -> Prediction:
        """
        Generate prediction for current market state

        Returns:
            Prediction object with p_final ready for trading decisions
        """
        timestamp = data.timestamp

        # 1. Feature engineering
        features = self.feature_engineer.compute_features(data, sentiment)

        # 2. Model prediction
        p_model = self.model.predict(features)

        # 3. Calibration
        p_cal = self.calibrator.calibrate(p_model)

        # 4. Reliability score
        reliability = self.calibrator.get_reliability()

        # 5. Similarity boost
        sim_multiplier = self.similarity.compute_boost(features)

        # 6. Fused final probability
        p_final = max(0.0, min(1.0, p_cal * reliability * sim_multiplier))

        prediction = Prediction(
            timestamp=timestamp,
            p_up=p_model,
            p_calibrated=p_cal,
            p_final=p_final,
            reliability=reliability,
            similarity_boost=sim_multiplier,
            features=features,
        )

        # Store for later labeling
        self.pending.append((prediction, data.mid))

        return prediction

    def update(self, label: TradeLabel):
        """
        Update model with realized outcome

        This should be called after horizon τ has elapsed
        """
        # Find matching pending prediction
        matching = None
        for pred, entry_price in self.pending:
            if abs(pred.timestamp - label.timestamp) < 0.1:
                matching = (pred, entry_price)
                break

        if matching is None:
            return  # No matching prediction found

        prediction, _ = matching

        # Update model
        self.model.update(prediction.features, label.label)

        # Update calibrator
        self.calibrator.add(prediction.p_up, label.label)

        # Update similarity
        self.similarity.add(prediction.features, label.label)

    def record_execution(
        self,
        order_type: str,
        intended_price: float,
        filled_price: Optional[float],
        side: str,
    ):
        """Record execution for slippage tracking"""
        if order_type == "MARKET" and filled_price is not None:
            self.slippage_tracker.add_market_slippage(
                intended_price, filled_price, side
            )
        elif order_type == "LIMIT":
            self.slippage_tracker.add_limit_fill(filled_price is not None)

    def get_stats(self) -> Dict:
        """Get current engine statistics"""
        return {
            "model_updates": self.model.n_updates,
            "calibration_samples": sum(len(b) for b in self.calibrator.bins),
            "similarity_history": len(self.similarity.history),
            "reliability": self.calibrator.get_reliability(),
            "expected_slippage": self.slippage_tracker.get_expected_slippage(),
            "limit_fill_rate": self.slippage_tracker.get_limit_fill_rate(),
            "pending_predictions": len(self.pending),
        }

    def save_state(self, filepath: str):
        """Save engine state to disk"""
        state = {
            "model_beta": self.model.beta.tolist(),
            "model_updates": self.model.n_updates,
            "volatility_ewma": self.feature_engineer.volatility_ewma.value,
            "drift_ewma": self.feature_engineer.drift_ewma.value,
            "calibrator_bins": [
                [list(pair) for pair in bin_data] for bin_data in self.calibrator.bins
            ],
            "similarity_history": [
                (sig.tolist(), outcome) for sig, outcome in self.similarity.history
            ],
            "timestamp": time.time(),
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load engine state from disk"""
        with open(filepath, "r") as f:
            state = json.load(f)

        self.model.beta = np.array(state["model_beta"])
        self.model.n_updates = state["model_updates"]
        self.feature_engineer.volatility_ewma.value = state["volatility_ewma"]
        self.feature_engineer.drift_ewma.value = state["drift_ewma"]

        # Restore calibrator bins
        for i, bin_data in enumerate(state["calibrator_bins"]):
            self.calibrator.bins[i] = [tuple(pair) for pair in bin_data]
            if len(self.calibrator.bins[i]) > 0:
                self.calibrator._update_stats(i)

        # Restore similarity history
        for sig_list, outcome in state["similarity_history"]:
            self.similarity.history.append((np.array(sig_list), outcome))


# ═══════════════════════════════════════════════════════════════════════
#                     HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════


def market_data_from_ibkr(ticker) -> MarketData:
    """Convert ib_insync Ticker to MarketData"""
    return MarketData(
        timestamp=time.time(),
        bid=float(ticker.bid) if ticker.bid == ticker.bid else 0.0,
        ask=float(ticker.ask) if ticker.ask == ticker.ask else 0.0,
        last=float(ticker.last) if ticker.last == ticker.last else 0.0,
        bid_size=float(ticker.bidSize) if ticker.bidSize == ticker.bidSize else 0.0,
        ask_size=float(ticker.askSize) if ticker.askSize == ticker.askSize else 0.0,
        volume=int(ticker.volume) if ticker.volume == ticker.volume else 0,
        vwap=(
            float(ticker.vwap)
            if hasattr(ticker, "vwap") and ticker.vwap == ticker.vwap
            else None
        ),
        high=float(ticker.high) if ticker.high == ticker.high else None,
        low=float(ticker.low) if ticker.low == ticker.low else None,
    )


if __name__ == "__main__":
    # Demo usage
    print("AlphaFusion V2 Engine - Demo")
    print("=" * 60)

    # Create engine
    engine = AlphaFusionEngine(horizon_seconds=2.0, learning_rate=0.01, k_neighbors=20)

    # Simulate market data
    data = MarketData(
        timestamp=time.time(),
        bid=100.50,
        ask=100.52,
        last=100.51,
        bid_size=500,
        ask_size=300,
        volume=10000,
        vwap=100.45,
    )

    # Get prediction
    pred = engine.predict(data, sentiment=0.2)

    print(f"\nPrediction:")
    print(f"  p_final: {pred.p_final:.3f}")
    print(f"  p_calibrated: {pred.p_calibrated:.3f}")
    print(f"  reliability: {pred.reliability:.3f}")
    print(f"  similarity_boost: {pred.similarity_boost:.3f}")

    print(f"\nFeatures:")
    print(f"  imbalance: {pred.features.imbalance:.3f}")
    print(f"  momentum: {pred.features.momentum:.3f}")
    print(f"  barrier: {pred.features.barrier:.3f}")

    # Simulate label after horizon
    label = TradeLabel(
        timestamp=data.timestamp,
        entry_price=100.51,
        exit_price=100.53,
        label=1,  # Price went up
        horizon_seconds=2.0,
    )

    engine.update(label)

    print(f"\nEngine Stats:")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
