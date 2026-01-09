"""
Warrior Trading ML Training Pipeline
Phase 3: Training pipeline for historical data

Provides tools to:
- Load and preprocess historical data
- Engineer features for ML models
- Label patterns for supervised learning
- Train transformer and RL models
- Evaluate model performance
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ai.warrior_rl_agent import TradingAction, TradingState, get_rl_agent
from ai.warrior_transformer_detector import get_transformer_detector

logger = logging.getLogger(__name__)


class PatternDataset(Dataset):
    """
    PyTorch Dataset for pattern detection training

    Each sample contains:
    - Candlestick sequence (100 bars)
    - Pattern label (0-7)
    - Target price movement %
    """

    def __init__(
        self, data: pd.DataFrame, sequence_length: int = 100, features: List[str] = None
    ):
        """
        Args:
            data: DataFrame with OHLCV + indicators
            sequence_length: Number of bars per sample
            features: Feature columns to use
        """
        self.data = data
        self.sequence_length = sequence_length

        if features is None:
            # Default features
            self.features = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "sma_20",
                "rsi",
                "macd",
                "bb_width",
                "atr",
            ]
        else:
            self.features = features

        # Create sequences
        self.sequences = []
        self.labels = []
        self.targets = []

        self._create_sequences()

    def _create_sequences(self):
        """Create training sequences from data"""
        for i in range(len(self.data) - self.sequence_length - 10):
            # Extract sequence
            sequence = self.data.iloc[i : i + self.sequence_length][
                self.features
            ].values

            # Get label (pattern type) if available
            if "pattern_label" in self.data.columns:
                label = self.data.iloc[i + self.sequence_length]["pattern_label"]
            else:
                label = 0  # Default: no pattern

            # Calculate target (next 10-bar return)
            current_price = self.data.iloc[i + self.sequence_length]["close"]
            future_price = self.data.iloc[i + self.sequence_length + 10]["close"]
            target = (future_price - current_price) / current_price

            self.sequences.append(sequence)
            self.labels.append(label)
            self.targets.append(target)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        target = torch.FloatTensor([self.targets[idx]])[0]

        return sequence, label, target


class HistoricalDataLoader:
    """
    Loads and preprocesses historical market data

    Supports multiple data sources:
    - CSV files
    - Database
    - yfinance
    """

    def __init__(self, data_dir: str = "data/historical"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file"""
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df

    def load_from_yfinance(
        self, symbol: str, start_date: str, end_date: str, interval: str = "5m"
    ) -> pd.DataFrame:
        """
        Load data from yfinance

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1m, 5m, 15m, 1h, 1d)
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            # Rename columns to lowercase
            df.columns = [c.lower() for c in df.columns]

            return df

        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            return pd.DataFrame()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to dataframe

        Indicators:
        - SMA 20, 50
        - RSI 14
        - MACD
        - Bollinger Bands
        - ATR
        """
        # SMA
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()

        # Bollinger Bands
        sma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        df["bb_upper"] = sma_20 + (2 * std_20)
        df["bb_lower"] = sma_20 - (2 * std_20)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma_20

        # ATR
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(14).mean()

        # Drop NaN rows
        df.dropna(inplace=True)

        return df


class PatternLabeler:
    """
    Automatic pattern labeling for supervised learning

    Uses rule-based heuristics to identify patterns in historical data
    """

    PATTERN_LABELS = {
        "bull_flag": 0,
        "bear_flag": 1,
        "breakout": 2,
        "breakdown": 3,
        "bullish_reversal": 4,
        "bearish_reversal": 5,
        "consolidation": 6,
        "gap_and_go": 7,
    }

    @staticmethod
    def detect_bull_flag(df: pd.DataFrame, idx: int, lookback: int = 20) -> bool:
        """Detect bull flag pattern"""
        if idx < lookback:
            return False

        window = df.iloc[idx - lookback : idx]

        # Pole: strong uptrend
        pole_return = (window["close"].iloc[10] - window["close"].iloc[0]) / window[
            "close"
        ].iloc[0]

        # Flag: consolidation/slight pullback
        flag_high = window["high"].iloc[10:].max()
        flag_low = window["low"].iloc[10:].min()
        flag_range = (flag_high - flag_low) / window["close"].iloc[10]

        # Volume: decreasing in flag
        pole_volume = window["volume"].iloc[:10].mean()
        flag_volume = window["volume"].iloc[10:].mean()

        # Criteria
        return (
            pole_return > 0.05  # 5% pole
            and flag_range < 0.03  # Tight flag
            and flag_volume < pole_volume * 0.8  # Declining volume
        )

    @staticmethod
    def detect_breakout(df: pd.DataFrame, idx: int, lookback: int = 20) -> bool:
        """Detect breakout pattern"""
        if idx < lookback + 5:
            return False

        window = df.iloc[idx - lookback : idx]

        # Consolidation: tight range
        consolidation_high = window["high"].iloc[:-5].max()
        consolidation_low = window["low"].iloc[:-5].min()
        consolidation_range = (consolidation_high - consolidation_low) / window[
            "close"
        ].iloc[0]

        # Breakout: price breaks above consolidation
        breakout_price = window["close"].iloc[-1]
        broke_out = breakout_price > consolidation_high

        # Volume surge
        avg_volume = window["volume"].iloc[:-5].mean()
        breakout_volume = window["volume"].iloc[-5:].mean()

        return (
            consolidation_range < 0.05  # Tight consolidation
            and broke_out
            and breakout_volume > avg_volume * 1.5  # Volume surge
        )

    @classmethod
    def label_patterns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pattern labels to dataframe

        Returns DataFrame with 'pattern_label' column
        """
        df = df.copy()
        df["pattern_label"] = -1  # No pattern

        for i in range(50, len(df) - 10):
            # Check each pattern type
            if cls.detect_bull_flag(df, i):
                df.iloc[i, df.columns.get_loc("pattern_label")] = cls.PATTERN_LABELS[
                    "bull_flag"
                ]
            elif cls.detect_breakout(df, i):
                df.iloc[i, df.columns.get_loc("pattern_label")] = cls.PATTERN_LABELS[
                    "breakout"
                ]
            # Add more pattern detectors...

        return df


class ModelTrainer:
    """
    Orchestrates model training

    Trains both Transformer and RL models
    """

    def __init__(self, data_dir: str = "data/historical", model_dir: str = "models"):
        self.data_loader = HistoricalDataLoader(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train_transformer(
        self,
        data: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        val_split: float = 0.2,
    ) -> Dict:
        """
        Train transformer pattern detector

        Args:
            data: Historical data with pattern labels
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            val_split: Validation split ratio

        Returns:
            Training metrics
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for training")

        logger.info("Training Transformer Pattern Detector...")

        # Create dataset
        dataset = PatternDataset(data, sequence_length=100)

        # Split train/val
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Get model
        detector = get_transformer_detector()
        detector.model.train()

        # Optimizer
        optimizer = torch.optim.Adam(detector.model.parameters(), lr=learning_rate)

        # Loss functions
        criterion_pattern = torch.nn.CrossEntropyLoss()
        criterion_target = torch.nn.MSELoss()

        # Training loop
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        for epoch in range(epochs):
            # Training
            train_loss = 0
            for sequences, labels, targets in train_loader:
                sequences = sequences.to(detector.device)
                labels = labels.to(detector.device)
                targets = targets.to(detector.device)

                optimizer.zero_grad()

                # Forward
                pattern_logits, confidence, price_target = detector.model(sequences)

                # Losses
                loss_pattern = criterion_pattern(pattern_logits, labels)
                loss_target = criterion_target(price_target.squeeze(), targets)

                loss = loss_pattern + 0.5 * loss_target

                # Backward
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            val_loss = 0
            correct = 0
            total = 0

            detector.model.eval()
            with torch.no_grad():
                for sequences, labels, targets in val_loader:
                    sequences = sequences.to(detector.device)
                    labels = labels.to(detector.device)
                    targets = targets.to(detector.device)

                    pattern_logits, confidence, price_target = detector.model(sequences)

                    loss_pattern = criterion_pattern(pattern_logits, labels)
                    loss_target = criterion_target(price_target.squeeze(), targets)
                    loss = loss_pattern + 0.5 * loss_target

                    val_loss += loss.item()

                    # Accuracy
                    _, predicted = pattern_logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            detector.model.train()

            # Metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100.0 * correct / total

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_accuracy"].append(val_accuracy)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | "
                    f"Val Acc: {val_accuracy:.2f}%"
                )

        # Save model
        model_path = self.model_dir / "transformer_detector.pth"
        torch.save(detector.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

        return history

    def train_rl_agent(
        self, data: pd.DataFrame, episodes: int = 1000, max_steps: int = 500
    ) -> Dict:
        """
        Train RL agent via backtesting simulation

        Args:
            data: Historical market data
            episodes: Number of training episodes
            max_steps: Max steps per episode

        Returns:
            Training metrics
        """
        logger.info("Training RL Execution Agent...")

        agent = get_rl_agent()

        episode_rewards = []

        for episode in range(episodes):
            # Reset environment
            start_idx = np.random.randint(100, len(data) - max_steps)
            current_idx = start_idx

            position_size = 0.0
            entry_price = None
            total_reward = 0.0

            for step in range(max_steps):
                if current_idx >= len(data) - 1:
                    break

                # Current state
                row = data.iloc[current_idx]
                state = TradingState(
                    price=row["close"],
                    volume=row["volume"],
                    volatility=row["atr"] / row["close"],
                    trend=(row["sma_20"] - row["sma_50"]) / row["close"],
                    position_size=position_size,
                    entry_price=entry_price,
                    unrealized_pnl=(
                        ((row["close"] - entry_price) / entry_price * position_size)
                        if entry_price
                        else 0.0
                    ),
                    sentiment_score=0.0,  # Would come from sentiment module
                    pattern_confidence=0.7,
                    time_in_position=step if entry_price else 0,
                    current_drawdown=0.0,
                    sharpe_ratio=1.0,
                    win_rate=0.5,
                )

                # Select action
                action = agent.select_action(state, training=True)

                # Execute action
                if action.action_type == "enter" and position_size == 0:
                    position_size = 0.3
                    entry_price = row["close"]
                elif action.action_type == "exit" and position_size > 0:
                    position_size = 0.0
                    entry_price = None

                # Next state
                current_idx += 1
                next_row = data.iloc[current_idx]
                next_state = TradingState(
                    price=next_row["close"],
                    volume=next_row["volume"],
                    volatility=next_row["atr"] / next_row["close"],
                    trend=(next_row["sma_20"] - next_row["sma_50"]) / next_row["close"],
                    position_size=position_size,
                    entry_price=entry_price,
                    unrealized_pnl=(
                        (
                            (next_row["close"] - entry_price)
                            / entry_price
                            * position_size
                        )
                        if entry_price
                        else 0.0
                    ),
                    sentiment_score=0.0,
                    pattern_confidence=0.7,
                    time_in_position=step + 1 if entry_price else 0,
                    current_drawdown=0.0,
                    sharpe_ratio=1.0,
                    win_rate=0.5,
                )

                # Calculate reward
                reward = agent.calculate_reward(state, action, next_state)
                total_reward += reward

                # Store experience
                done = (current_idx >= len(data) - 1) or (
                    position_size == 0 and step > 50
                )
                agent.memory.push(state, action, reward, next_state, done)

                # Train
                if len(agent.memory) > 100:
                    agent.train_step(batch_size=32)

                if done:
                    break

            episode_rewards.append(total_reward)

            # Update target network
            if (episode + 1) % 10 == 0:
                agent.update_target_network()

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(
                    f"Episode {episode+1}/{episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Epsilon: {agent.epsilon:.3f}"
                )

        # Save agent
        agent_path = self.model_dir / "rl_agent.pth"
        agent.save(str(agent_path))

        return {"episode_rewards": episode_rewards}


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize trainer
    trainer = ModelTrainer()

    # Load historical data
    loader = HistoricalDataLoader()

    # Example: load from yfinance
    print("Loading historical data...")
    data = loader.load_from_yfinance("SPY", "2023-01-01", "2024-01-01", "5m")

    if not data.empty:
        # Add indicators
        print("Adding technical indicators...")
        data = loader.add_technical_indicators(data)

        # Label patterns
        print("Labeling patterns...")
        data = PatternLabeler.label_patterns(data)

        # Train transformer
        print("\nTraining Transformer...")
        transformer_metrics = trainer.train_transformer(data, epochs=10)

        # Train RL agent
        print("\nTraining RL Agent...")
        rl_metrics = trainer.train_rl_agent(data, episodes=100)

        print("\nTraining complete!")
    else:
        print("No data loaded. Install yfinance: pip install yfinance")
