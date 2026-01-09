"""
CNN Stock Profile Predictor
===========================
Convolutional Neural Network for building predictive stock profiles.

Key Features:
1. CNN architecture with proper forward() method
2. Weight decay regularization to prevent overfitting
3. Temporal weight decay - recent data weighted more heavily
4. Multi-metric comparison for reliability assessment
5. Backtesting integration for validation
6. Stock "profile" building using pattern recognition

Architecture:
- 1D CNN layers to detect patterns in time series
- Temporal attention to weight recent data more
- Multi-head output for price direction, magnitude, and confidence
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta  # Technical analysis library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class StockProfile:
    """Profile of a stock's trading characteristics"""

    symbol: str
    volatility_regime: str  # low, medium, high
    trend_strength: float  # 0-1
    momentum_score: float  # -1 to 1
    mean_reversion_tendency: float  # 0-1
    volume_profile: str  # light, normal, heavy
    optimal_holding_period: int  # bars
    pattern_signatures: List[str]  # detected patterns
    confidence: float
    last_updated: str


@dataclass
class PredictionMetrics:
    """Metrics for evaluating prediction quality"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    profit_factor: float
    max_drawdown: float
    brier_score: float  # calibration metric
    directional_accuracy: float


# ============================================================================
# TEMPORAL WEIGHT DECAY
# ============================================================================


class TemporalWeightDecay:
    """
    Apply exponential weight decay to time series data.
    Recent data gets higher weights, older data gets lower weights.
    """

    def __init__(self, decay_rate: float = 0.995, min_weight: float = 0.1):
        """
        Args:
            decay_rate: Decay factor per time step (0.995 = 0.5% decay per step)
            min_weight: Minimum weight to apply (prevents zeroing out old data)
        """
        self.decay_rate = decay_rate
        self.min_weight = min_weight

    def compute_weights(self, sequence_length: int) -> torch.Tensor:
        """Compute temporal weights for a sequence"""
        # Weight decays exponentially from present (1.0) to past
        steps = torch.arange(sequence_length - 1, -1, -1, dtype=torch.float32)
        weights = torch.pow(torch.tensor(self.decay_rate), steps)
        weights = torch.clamp(weights, min=self.min_weight)
        # Normalize so weights sum to sequence_length (preserve scale)
        weights = weights * (sequence_length / weights.sum())
        return weights

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        """Apply temporal weights to data (batch, seq, features)"""
        seq_len = data.shape[1]
        weights = self.compute_weights(seq_len).to(data.device)
        # Expand weights to match data shape
        weights = weights.unsqueeze(0).unsqueeze(-1)
        return data * weights


# ============================================================================
# CNN MODEL ARCHITECTURE
# ============================================================================


class StockPatternCNN(nn.Module):
    """
    1D Convolutional Neural Network for stock pattern recognition.

    Architecture:
    - Multi-scale 1D convolutions to detect patterns at different time scales
    - Temporal attention mechanism
    - Weight decay regularization built into optimizer
    - Multi-head output: direction, magnitude, confidence
    """

    def __init__(
        self,
        input_features: int = 30,
        sequence_length: int = 60,
        hidden_channels: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [3, 5, 7],
        dropout_rate: float = 0.3,
        num_classes: int = 3,  # BUY, HOLD, SELL
    ):
        super(StockPatternCNN, self).__init__()

        self.input_features = input_features
        self.sequence_length = sequence_length
        self.num_classes = num_classes

        # =====================================================================
        # MULTI-SCALE CONVOLUTIONAL BLOCKS
        # Different kernel sizes capture patterns at different time scales
        # =====================================================================

        self.conv_blocks = nn.ModuleList()

        for i, (out_channels, kernel_size) in enumerate(
            zip(hidden_channels, kernel_sizes)
        ):
            in_channels = input_features if i == 0 else hidden_channels[i - 1]

            block = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,  # No bias when using BatchNorm
                ),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),  # GELU often works better than ReLU for financial data
                nn.Dropout(dropout_rate),
                # Second conv in block
                nn.Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            )
            self.conv_blocks.append(block)

        # =====================================================================
        # TEMPORAL ATTENTION
        # Learn to focus on important time steps
        # =====================================================================

        self.attention = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
            nn.Tanh(),
            nn.Linear(hidden_channels[-1] // 2, 1),
            nn.Softmax(dim=1),
        )

        # =====================================================================
        # PROFILE ENCODER
        # Encode stock characteristics into a profile vector
        # =====================================================================

        self.profile_encoder = nn.Sequential(
            nn.Linear(hidden_channels[-1], 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # =====================================================================
        # OUTPUT HEADS
        # Multiple outputs for comprehensive prediction
        # =====================================================================

        # Direction prediction (BUY/HOLD/SELL)
        self.direction_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, num_classes)
        )

        # Magnitude prediction (expected move %)
        self.magnitude_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1)
        )

        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid()
        )

        # Volatility regime prediction
        self.volatility_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3),  # low, medium, high
            nn.Softmax(dim=-1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, return_profile: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the CNN.

        Args:
            x: Input tensor of shape (batch, sequence_length, features)
            return_profile: If True, also return the profile embedding

        Returns:
            Dictionary containing:
                - direction: (batch, num_classes) logits
                - direction_probs: (batch, num_classes) probabilities
                - magnitude: (batch, 1) expected move
                - confidence: (batch, 1) prediction confidence
                - volatility: (batch, 3) volatility regime probs
                - profile: (batch, 64) profile embedding (if return_profile=True)
        """
        # Input shape: (batch, seq, features) -> (batch, features, seq) for Conv1d
        x = x.transpose(1, 2)

        # Apply convolutional blocks sequentially
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Transpose back: (batch, channels, seq) -> (batch, seq, channels)
        x = x.transpose(1, 2)

        # Apply temporal attention
        attention_weights = self.attention(x)  # (batch, seq, 1)
        x = (x * attention_weights).sum(dim=1)  # (batch, channels)

        # Encode into profile
        profile = self.profile_encoder(x)  # (batch, 64)

        # Generate predictions from profile
        direction_logits = self.direction_head(profile)
        direction_probs = torch.softmax(direction_logits, dim=-1)
        magnitude = self.magnitude_head(profile)
        confidence = self.confidence_head(profile)
        volatility = self.volatility_head(profile)

        outputs = {
            "direction": direction_logits,
            "direction_probs": direction_probs,
            "magnitude": magnitude,
            "confidence": confidence,
            "volatility": volatility,
        }

        if return_profile:
            outputs["profile"] = profile

        return outputs


# ============================================================================
# STOCK PROFILE DATASET
# ============================================================================


class StockProfileDataset(Dataset):
    """Dataset for training the CNN stock predictor"""

    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 60,
        prediction_horizon: int = 5,
        temporal_decay: Optional[TemporalWeightDecay] = None,
    ):
        """
        Args:
            data: DataFrame with OHLCV and technical indicators
            sequence_length: Number of time steps for input sequence
            prediction_horizon: How many bars ahead to predict
            temporal_decay: Optional temporal weight decay
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.temporal_decay = temporal_decay

        # Prepare features and labels
        self.features, self.labels = self._prepare_data(data)

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature sequences and labels"""
        # Feature columns (exclude target-related columns)
        exclude_cols = ["target", "target_return", "future_return", "Date", "date"]
        feature_cols = [c for c in data.columns if c not in exclude_cols]

        # Normalize features
        features = data[feature_cols].values
        features = (features - np.nanmean(features, axis=0)) / (
            np.nanstd(features, axis=0) + 1e-8
        )
        features = np.nan_to_num(features, nan=0.0)

        # Create sequences
        X, y = [], []

        for i in range(len(features) - self.sequence_length - self.prediction_horizon):
            # Input sequence
            seq = features[i : i + self.sequence_length]
            X.append(seq)

            # Target: future return direction
            current_close_idx = (
                feature_cols.index("Close") if "Close" in feature_cols else 0
            )
            current_price = data.iloc[i + self.sequence_length - 1]["Close"]
            future_price = data.iloc[
                i + self.sequence_length + self.prediction_horizon - 1
            ]["Close"]
            future_return = (future_price - current_price) / current_price

            # Classify into BUY (0), HOLD (1), SELL (2)
            if future_return > 0.02:  # >2% gain
                label = 0  # BUY
            elif future_return < -0.02:  # >2% loss
                label = 2  # SELL
            else:
                label = 1  # HOLD

            y.append(label)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        # Apply temporal decay if configured
        if self.temporal_decay is not None:
            x = self.temporal_decay.apply(x.unsqueeze(0)).squeeze(0)

        return x, y


# ============================================================================
# TRAINING PIPELINE
# ============================================================================


class CNNTrainer:
    """Training pipeline for the CNN stock predictor"""

    def __init__(
        self,
        model: StockPatternCNN,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,  # L2 regularization
        temporal_decay_rate: float = 0.995,
        device: str = "auto",
    ):
        """
        Args:
            model: StockPatternCNN model
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength (prevents overfitting)
            temporal_decay_rate: Rate for temporal weight decay
            device: "cuda", "cpu", or "auto"
        """
        self.model = model

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        logger.info(f"Training on device: {self.device}")

        # Optimizer with weight decay (L2 regularization)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,  # This is the key for weight decay
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler (verbose removed in PyTorch 2.x)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Loss functions
        self.direction_loss = nn.CrossEntropyLoss()
        self.magnitude_loss = nn.MSELoss()
        self.confidence_loss = nn.BCELoss()

        # Temporal weight decay for data
        self.temporal_decay = TemporalWeightDecay(decay_rate=temporal_decay_rate)

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rates": [],
        }

        # Metrics tracking
        self.metrics = None

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch_x)

            # Compute losses
            loss_direction = self.direction_loss(outputs["direction"], batch_y)

            # Total loss (can add magnitude and confidence losses if we have those targets)
            loss = loss_direction

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(outputs["direction"], 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.direction_loss(outputs["direction"], batch_y)

                total_loss += loss.item()

                _, predicted = torch.max(outputs["direction"], 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None,
    ) -> Dict:
        """
        Full training loop with early stopping

        Returns:
            Training history and final metrics
        """
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rates"].append(current_lr)

            # Logging
            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                    f"LR: {current_lr:.6f}"
                )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                if save_path:
                    self.save_model(save_path)
                    logger.info(f"Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Compute final metrics
        self.metrics = self._compute_final_metrics(val_loader)

        return {
            "history": self.history,
            "metrics": asdict(self.metrics) if self.metrics else None,
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
        }

    def _compute_final_metrics(self, val_loader: DataLoader) -> PredictionMetrics:
        """Compute comprehensive metrics on validation set"""
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_probs = []
        all_confidences = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)

                outputs = self.model(batch_x)

                _, predicted = torch.max(outputs["direction"], 1)
                probs = outputs["direction_probs"]
                confidence = outputs["confidence"]

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.numpy())
                all_probs.extend(probs.cpu().numpy())
                all_confidences.extend(confidence.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_confidences = np.array(all_confidences).flatten()

        # Accuracy
        accuracy = np.mean(all_predictions == all_labels)

        # Precision, Recall, F1 for each class
        from sklearn.metrics import f1_score, precision_score, recall_score

        precision = precision_score(
            all_labels, all_predictions, average="weighted", zero_division=0
        )
        recall = recall_score(
            all_labels, all_predictions, average="weighted", zero_division=0
        )
        f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)

        # Directional accuracy (did we get the sign right for non-HOLD predictions)
        buy_sell_mask = (all_labels != 1) | (all_predictions != 1)
        if buy_sell_mask.sum() > 0:
            directional_acc = np.mean(
                ((all_predictions == 0) & (all_labels == 0))  # Both say BUY
                | ((all_predictions == 2) & (all_labels == 2))  # Both say SELL
            )
        else:
            directional_acc = 0.5

        # Brier score (calibration)
        # For each sample, compare predicted probability to actual outcome
        brier_scores = []
        for i, label in enumerate(all_labels):
            prob_true_class = all_probs[i, label]
            brier_scores.append((1 - prob_true_class) ** 2)
        brier_score = np.mean(brier_scores)

        return PredictionMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            sharpe_ratio=0.0,  # Computed in backtesting
            profit_factor=0.0,  # Computed in backtesting
            max_drawdown=0.0,  # Computed in backtesting
            brier_score=brier_score,
            directional_accuracy=directional_acc,
        )

    def save_model(self, path: str):
        """Save model state"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
                "metrics": asdict(self.metrics) if self.metrics else None,
            },
            path,
        )

    def load_model(self, path: str):
        """Load model state"""
        # Use weights_only=False for compatibility with PyTorch 2.6+
        # The model was saved by this system so it's trusted
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        if checkpoint.get("metrics"):
            self.metrics = PredictionMetrics(**checkpoint["metrics"])


# ============================================================================
# STOCK PROFILE BUILDER
# ============================================================================


class StockProfileBuilder:
    """Build and maintain stock profiles using CNN predictions"""

    def __init__(self, model: StockPatternCNN, device: str = "auto"):
        self.model = model

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        self.profiles: Dict[str, StockProfile] = {}

    def build_profile(self, symbol: str, data: pd.DataFrame) -> StockProfile:
        """
        Build a trading profile for a stock based on its historical patterns.

        Args:
            symbol: Stock ticker
            data: DataFrame with OHLCV and indicators

        Returns:
            StockProfile with trading characteristics
        """
        # Prepare data
        feature_cols = [c for c in data.columns if c not in ["Date", "date"]]
        features = data[feature_cols].values
        features = (features - np.nanmean(features, axis=0)) / (
            np.nanstd(features, axis=0) + 1e-8
        )
        features = np.nan_to_num(features, nan=0.0)

        # Use last 60 bars for profile
        seq_length = min(60, len(features))
        seq = features[-seq_length:]

        # Pad if necessary
        if len(seq) < 60:
            padding = np.zeros((60 - len(seq), seq.shape[1]))
            seq = np.vstack([padding, seq])

        # Convert to tensor
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(x, return_profile=True)

        # Extract profile characteristics
        direction_probs = outputs["direction_probs"].cpu().numpy()[0]
        volatility_probs = outputs["volatility"].cpu().numpy()[0]
        confidence = outputs["confidence"].cpu().item()
        magnitude = outputs["magnitude"].cpu().item()

        # Determine volatility regime
        vol_labels = ["low", "medium", "high"]
        volatility_regime = vol_labels[np.argmax(volatility_probs)]

        # Calculate momentum score (-1 to 1)
        momentum_score = direction_probs[0] - direction_probs[2]  # BUY prob - SELL prob

        # Calculate trend strength (how confident the model is in a direction)
        trend_strength = max(direction_probs[0], direction_probs[2])

        # Mean reversion tendency (inverse of trend strength when in HOLD zone)
        mean_reversion = direction_probs[1]

        # Detect patterns from profile embedding
        pattern_signatures = self._detect_patterns(outputs.get("profile"))

        # Volume profile from recent data
        if "Volume" in data.columns:
            recent_vol = data["Volume"].iloc[-20:].mean()
            avg_vol = data["Volume"].mean()
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0

            if vol_ratio > 1.5:
                volume_profile = "heavy"
            elif vol_ratio < 0.5:
                volume_profile = "light"
            else:
                volume_profile = "normal"
        else:
            volume_profile = "normal"

        # Optimal holding period based on magnitude
        # Higher expected move = longer hold
        optimal_holding = max(1, min(20, int(abs(magnitude) * 100)))

        profile = StockProfile(
            symbol=symbol,
            volatility_regime=volatility_regime,
            trend_strength=float(trend_strength),
            momentum_score=float(momentum_score),
            mean_reversion_tendency=float(mean_reversion),
            volume_profile=volume_profile,
            optimal_holding_period=optimal_holding,
            pattern_signatures=pattern_signatures,
            confidence=float(confidence),
            last_updated=datetime.now().isoformat(),
        )

        self.profiles[symbol] = profile
        return profile

    def _detect_patterns(self, profile_embedding: Optional[torch.Tensor]) -> List[str]:
        """Detect pattern signatures from profile embedding"""
        patterns = []

        if profile_embedding is None:
            return patterns

        embedding = profile_embedding.cpu().numpy()[0]

        # Simple pattern detection based on embedding characteristics
        # In a full implementation, these would be learned cluster centers

        # High momentum pattern
        if np.mean(embedding[:16]) > 0.5:
            patterns.append("strong_momentum")

        # Reversal pattern
        if np.mean(embedding[16:32]) > 0.5:
            patterns.append("reversal_setup")

        # Consolidation pattern
        if np.std(embedding) < 0.3:
            patterns.append("consolidation")

        # Breakout pattern
        if np.max(embedding) > 2.0:
            patterns.append("breakout_potential")

        return patterns

    def get_profile(self, symbol: str) -> Optional[StockProfile]:
        """Get cached profile for a symbol"""
        return self.profiles.get(symbol)

    def predict(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Make a prediction for a symbol.

        Returns:
            Dictionary with action, confidence, magnitude, etc.
        """
        # Build/update profile
        profile = self.build_profile(symbol, data)

        # Determine action
        if profile.momentum_score > 0.3 and profile.confidence > 0.5:
            action = "BUY"
            signal = "BULLISH"
        elif profile.momentum_score < -0.3 and profile.confidence > 0.5:
            action = "SELL"
            signal = "BEARISH"
        else:
            action = "HOLD"
            signal = "NEUTRAL"

        return {
            "symbol": symbol,
            "action": action,
            "signal": signal,
            "confidence": profile.confidence,
            "momentum_score": profile.momentum_score,
            "trend_strength": profile.trend_strength,
            "volatility_regime": profile.volatility_regime,
            "patterns": profile.pattern_signatures,
            "optimal_holding_period": profile.optimal_holding_period,
            "profile": asdict(profile),
        }


# ============================================================================
# BACKTESTING INTEGRATION
# ============================================================================


class CNNBacktester:
    """Backtest the CNN predictor on historical data"""

    def __init__(self, profile_builder: StockProfileBuilder):
        self.profile_builder = profile_builder
        self.trades: List[Dict] = []

    def backtest(
        self,
        symbol: str,
        data: pd.DataFrame,
        initial_capital: float = 100000,
        position_size_pct: float = 0.1,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.06,
        lookback: int = 60,
        prediction_interval: int = 1,
    ) -> Dict:
        """
        Run backtest on historical data.

        Args:
            symbol: Stock ticker
            data: Historical OHLCV data with indicators
            initial_capital: Starting capital
            position_size_pct: Position size as % of capital
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            lookback: Bars to look back for prediction
            prediction_interval: Bars between predictions

        Returns:
            Backtest results with metrics
        """
        capital = initial_capital
        position = None
        self.trades = []

        peak_capital = initial_capital
        max_drawdown = 0
        daily_returns = []

        for i in range(lookback, len(data), prediction_interval):
            current_bar = data.iloc[i]
            current_price = current_bar["Close"]

            # Check existing position
            if position:
                # Check stop loss / take profit
                pnl_pct = (current_price - position["entry_price"]) / position[
                    "entry_price"
                ]

                if position["side"] == "SELL":
                    pnl_pct = -pnl_pct

                # Stop loss hit
                if pnl_pct <= -stop_loss_pct:
                    exit_pnl = (
                        position["quantity"]
                        * position["entry_price"]
                        * (-stop_loss_pct)
                    )
                    capital += position["value"] + exit_pnl

                    self.trades.append(
                        {
                            "symbol": symbol,
                            "entry_date": position["entry_date"],
                            "entry_price": position["entry_price"],
                            "exit_date": (
                                current_bar.name.isoformat()
                                if hasattr(current_bar.name, "isoformat")
                                else str(current_bar.name)
                            ),
                            "exit_price": current_price,
                            "side": position["side"],
                            "quantity": position["quantity"],
                            "pnl": exit_pnl,
                            "pnl_pct": -stop_loss_pct * 100,
                            "exit_reason": "stop_loss",
                        }
                    )
                    position = None
                    continue

                # Take profit hit
                if pnl_pct >= take_profit_pct:
                    exit_pnl = (
                        position["quantity"] * position["entry_price"] * take_profit_pct
                    )
                    capital += position["value"] + exit_pnl

                    self.trades.append(
                        {
                            "symbol": symbol,
                            "entry_date": position["entry_date"],
                            "entry_price": position["entry_price"],
                            "exit_date": (
                                current_bar.name.isoformat()
                                if hasattr(current_bar.name, "isoformat")
                                else str(current_bar.name)
                            ),
                            "exit_price": current_price,
                            "side": position["side"],
                            "quantity": position["quantity"],
                            "pnl": exit_pnl,
                            "pnl_pct": take_profit_pct * 100,
                            "exit_reason": "take_profit",
                        }
                    )
                    position = None
                    continue

            # Get prediction
            historical_slice = data.iloc[i - lookback : i]
            prediction = self.profile_builder.predict(symbol, historical_slice)

            # Enter new position if no current position
            if position is None and prediction["action"] != "HOLD":
                if prediction["confidence"] > 0.6:  # Only trade with high confidence
                    position_value = capital * position_size_pct
                    quantity = int(position_value / current_price)

                    if quantity > 0:
                        position = {
                            "entry_date": (
                                current_bar.name.isoformat()
                                if hasattr(current_bar.name, "isoformat")
                                else str(current_bar.name)
                            ),
                            "entry_price": current_price,
                            "side": prediction["action"],
                            "quantity": quantity,
                            "value": position_value,
                            "confidence": prediction["confidence"],
                        }
                        capital -= position_value

            # Track drawdown
            total_value = capital
            if position:
                current_value = position["quantity"] * current_price
                total_value += current_value

            if total_value > peak_capital:
                peak_capital = total_value
            else:
                drawdown = (peak_capital - total_value) / peak_capital
                max_drawdown = max(max_drawdown, drawdown)

            # Track daily returns for Sharpe ratio
            if len(daily_returns) == 0:
                daily_returns.append(0)
            else:
                daily_return = (total_value - initial_capital) / initial_capital
                daily_returns.append(daily_return)

        # Close any remaining position at end
        if position:
            final_price = data.iloc[-1]["Close"]
            pnl_pct = (final_price - position["entry_price"]) / position["entry_price"]
            if position["side"] == "SELL":
                pnl_pct = -pnl_pct
            exit_pnl = position["quantity"] * position["entry_price"] * pnl_pct
            capital += position["value"] + exit_pnl

            self.trades.append(
                {
                    "symbol": symbol,
                    "entry_date": position["entry_date"],
                    "entry_price": position["entry_price"],
                    "exit_date": (
                        data.index[-1].isoformat()
                        if hasattr(data.index[-1], "isoformat")
                        else str(data.index[-1])
                    ),
                    "exit_price": final_price,
                    "side": position["side"],
                    "quantity": position["quantity"],
                    "pnl": exit_pnl,
                    "pnl_pct": pnl_pct * 100,
                    "exit_reason": "end_of_backtest",
                }
            )

        # Calculate metrics
        final_capital = capital
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100

        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]

        win_rate = len(wins) / len(self.trades) if self.trades else 0
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t["pnl"]) for t in losses]) if losses else 0

        gross_profit = sum(t["pnl"] for t in wins) if wins else 0
        gross_loss = sum(abs(t["pnl"]) for t in losses) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio (annualized)
        if len(daily_returns) > 1:
            returns_std = np.std(daily_returns)
            if returns_std > 0:
                sharpe = np.mean(daily_returns) / returns_std * np.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = 0

        return {
            "symbol": symbol,
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown * 100,
            "sharpe_ratio": sharpe,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "trades": self.trades,
        }


# ============================================================================
# MAIN PREDICTOR CLASS (Integration with existing system)
# ============================================================================


class CNNStockPredictor:
    """
    Main predictor class that integrates with the existing trading system.
    Uses CNN for pattern recognition and profile building.
    """

    def __init__(
        self,
        model_path: str = "store/models/cnn_predictor.pt",
        input_features: int = 30,
        sequence_length: int = 60,
    ):
        self.model_path = model_path
        self.input_features = input_features
        self.sequence_length = sequence_length

        # Initialize model
        self.model = StockPatternCNN(
            input_features=input_features, sequence_length=sequence_length
        )

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Profile builder
        self.profile_builder = StockProfileBuilder(self.model, device=str(self.device))

        # Backtester
        self.backtester = CNNBacktester(self.profile_builder)

        # Training metrics
        self.training_metrics: Optional[PredictionMetrics] = None
        self.last_trained: Optional[str] = None

        # Try to load existing model
        self._load_if_exists()

    def _load_if_exists(self):
        """Load model if it exists"""
        if Path(self.model_path).exists():
            try:
                # Use weights_only=False for compatibility with PyTorch 2.6+
                checkpoint = torch.load(
                    self.model_path, map_location=self.device, weights_only=False
                )
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.last_trained = checkpoint.get("last_trained")
                if checkpoint.get("metrics"):
                    self.training_metrics = PredictionMetrics(**checkpoint["metrics"])
                logger.info(f"Loaded CNN model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

    def train(
        self,
        symbols: List[str],
        days: int = 365,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ) -> Dict:
        """
        Train the CNN model on historical data.

        Args:
            symbols: List of stock symbols to train on
            days: Days of historical data to use
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Fraction for validation

        Returns:
            Training results
        """
        import ta
        import yfinance as yf

        logger.info(
            f"Training CNN on {len(symbols)} symbols with {days} days of data..."
        )

        # Collect data from all symbols
        all_data = []

        for symbol in symbols:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)

                df = yf.download(
                    symbol,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    progress=False,
                )

                if len(df) < self.sequence_length + 10:
                    logger.warning(f"Insufficient data for {symbol}, skipping")
                    continue

                # Calculate features
                df = self._calculate_features(df)
                df = df.dropna()

                if len(df) > self.sequence_length:
                    all_data.append(df)
                    logger.info(f"Added {len(df)} bars from {symbol}")

            except Exception as e:
                logger.warning(f"Error loading {symbol}: {e}")
                continue

        if not all_data:
            raise ValueError("No valid training data collected")

        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total training samples: {len(combined_data)}")

        # Create dataset
        temporal_decay = TemporalWeightDecay(decay_rate=0.995)
        dataset = StockProfileDataset(
            combined_data,
            sequence_length=self.sequence_length,
            temporal_decay=temporal_decay,
        )

        # Split into train/val
        train_size = int(len(dataset) * (1 - validation_split))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Train
        trainer = CNNTrainer(
            self.model,
            learning_rate=0.001,
            weight_decay=0.01,  # L2 regularization
            device=str(self.device),
        )

        results = trainer.train(
            train_loader,
            val_loader,
            epochs=epochs,
            early_stopping_patience=10,
            save_path=self.model_path,
        )

        self.training_metrics = trainer.metrics
        self.last_trained = datetime.now().isoformat()

        # Save final model with metadata
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "last_trained": self.last_trained,
                "metrics": (
                    asdict(self.training_metrics) if self.training_metrics else None
                ),
                "symbols_trained": symbols,
                "training_days": days,
            },
            self.model_path,
        )

        logger.info(
            f"Training complete. Final accuracy: {results['metrics']['accuracy']:.4f}"
        )

        return results

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features for the model"""
        data = df.copy()

        # Handle multi-index columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten to just the price columns
            data.columns = data.columns.get_level_values(0)

        # Ensure we have proper 1D Series for each column
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in data.columns:
                if hasattr(data[col], "values") and len(data[col].values.shape) > 1:
                    data[col] = data[col].values.flatten()

        # Price features
        close = (
            data["Close"].squeeze()
            if hasattr(data["Close"], "squeeze")
            else data["Close"]
        )
        data["returns"] = close.pct_change()
        data["log_returns"] = np.log(close / close.shift(1))

        # Moving averages
        for window in [5, 10, 20, 50]:
            data[f"sma_{window}"] = ta.trend.sma_indicator(data["Close"], window=window)
            data[f"ema_{window}"] = ta.trend.ema_indicator(data["Close"], window=window)

        # MACD
        macd = ta.trend.MACD(data["Close"])
        data["macd"] = macd.macd()
        data["macd_signal"] = macd.macd_signal()
        data["macd_diff"] = macd.macd_diff()

        # RSI
        data["rsi"] = ta.momentum.rsi(data["Close"], window=14)
        data["rsi_7"] = ta.momentum.rsi(data["Close"], window=7)

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(data["Close"])
        data["bb_high"] = bb.bollinger_hband()
        data["bb_low"] = bb.bollinger_lband()
        data["bb_mid"] = bb.bollinger_mavg()
        data["bb_width"] = (data["bb_high"] - data["bb_low"]) / data["bb_mid"]

        # ATR
        data["atr"] = ta.volatility.average_true_range(
            data["High"], data["Low"], data["Close"]
        )

        # Volume features
        data["volume_sma"] = ta.volume.volume_weighted_average_price(
            data["High"], data["Low"], data["Close"], data["Volume"]
        )
        data["volume_ratio"] = data["Volume"] / data["Volume"].rolling(20).mean()

        # ADX
        adx = ta.trend.ADXIndicator(data["High"], data["Low"], data["Close"])
        data["adx"] = adx.adx()

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            data["High"], data["Low"], data["Close"]
        )
        data["stoch_k"] = stoch.stoch()
        data["stoch_d"] = stoch.stoch_signal()

        return data

    def predict(self, symbol: str, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Make a prediction for a symbol.

        Args:
            symbol: Stock ticker
            data: Optional DataFrame with OHLCV data. If None, will fetch.

        Returns:
            Prediction dictionary
        """
        if data is None:
            import yfinance as yf

            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)

        if len(data) < self.sequence_length:
            return {
                "symbol": symbol,
                "action": "HOLD",
                "signal": "INSUFFICIENT_DATA",
                "confidence": 0.0,
            }

        # Calculate features
        data = self._calculate_features(data)
        data = data.dropna()

        if len(data) < self.sequence_length:
            return {
                "symbol": symbol,
                "action": "HOLD",
                "signal": "INSUFFICIENT_DATA",
                "confidence": 0.0,
            }

        # Get prediction from profile builder
        return self.profile_builder.predict(symbol, data)

    def backtest(self, symbol: str, days: int = 180, **kwargs) -> Dict:
        """Run backtest on a symbol"""
        import yfinance as yf

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        data = self._calculate_features(data)
        data = data.dropna()

        return self.backtester.backtest(symbol, data, **kwargs)

    def get_status(self) -> Dict:
        """Get predictor status"""
        return {
            "model_type": "CNN Stock Profile Predictor",
            "model_path": self.model_path,
            "model_loaded": Path(self.model_path).exists(),
            "last_trained": self.last_trained,
            "device": str(self.device),
            "metrics": asdict(self.training_metrics) if self.training_metrics else None,
            "input_features": self.input_features,
            "sequence_length": self.sequence_length,
            "profiles_cached": len(self.profile_builder.profiles),
        }


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_cnn_predictor: Optional[CNNStockPredictor] = None


def get_cnn_predictor() -> CNNStockPredictor:
    """Get or create the CNN predictor singleton"""
    global _cnn_predictor
    if _cnn_predictor is None:
        _cnn_predictor = CNNStockPredictor()
    return _cnn_predictor


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="CNN Stock Predictor")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", type=str, help="Predict for a symbol")
    parser.add_argument("--backtest", type=str, help="Backtest a symbol")
    parser.add_argument(
        "--symbols",
        type=str,
        default="SPY,QQQ,AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA",
        help="Comma-separated symbols for training",
    )
    parser.add_argument("--days", type=int, default=365, help="Days of historical data")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")

    args = parser.parse_args()

    predictor = get_cnn_predictor()

    if args.train:
        symbols = [s.strip() for s in args.symbols.split(",")]
        results = predictor.train(symbols, days=args.days, epochs=args.epochs)
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Final Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"F1 Score: {results['metrics']['f1_score']:.4f}")
        print(f"Directional Accuracy: {results['metrics']['directional_accuracy']:.4f}")
        print(f"Brier Score: {results['metrics']['brier_score']:.4f}")

    elif args.predict:
        prediction = predictor.predict(args.predict)
        print("\n" + "=" * 60)
        print(f"PREDICTION FOR {args.predict}")
        print("=" * 60)
        print(json.dumps(prediction, indent=2, default=str))

    elif args.backtest:
        results = predictor.backtest(args.backtest, days=args.days)
        print("\n" + "=" * 60)
        print(f"BACKTEST RESULTS FOR {args.backtest}")
        print("=" * 60)
        print(
            f"Total Return: ${results['total_return']:.2f} ({results['total_return_pct']:.2f}%)"
        )
        print(f"Win Rate: {results['win_rate']*100:.1f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Total Trades: {results['total_trades']}")

    else:
        print("Status:", json.dumps(predictor.get_status(), indent=2))
