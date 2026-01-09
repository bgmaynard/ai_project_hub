"""
Reinforcement Learning Trading Agent
====================================
Self-optimizing agent for entry/exit timing using Deep Q-Network (DQN).

The agent learns optimal trading actions by:
1. Observing market state (price, indicators, position)
2. Taking actions (BUY, SELL, HOLD)
3. Receiving rewards (profit/loss, risk-adjusted returns)
4. Learning from experience replay

Architecture:
- State: Technical indicators + position info + ensemble signals
- Actions: BUY, SELL, HOLD (with position sizing)
- Reward: Risk-adjusted returns (Sharpe-like)
- Model: DQN with target network and experience replay

Created: December 2025
"""

import json
import logging
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed - RL agent will use simple Q-table fallback")


class Action(Enum):
    """Trading actions"""

    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class TradingState:
    """State representation for RL agent"""

    # Price features (normalized)
    price_change_1d: float = 0.0
    price_change_5d: float = 0.0
    price_change_20d: float = 0.0

    # Technical indicators (normalized -1 to 1)
    rsi_normalized: float = 0.0  # (RSI - 50) / 50
    macd_normalized: float = 0.0
    bb_position: float = 0.0  # Position within Bollinger Bands

    # Volume
    volume_ratio: float = 0.0  # Current / 20-day avg

    # Trend
    trend_strength: float = 0.0  # ADX normalized
    trend_direction: float = 0.0  # +1 up, -1 down

    # Volatility
    volatility: float = 0.0  # Normalized ATR

    # Ensemble signals
    ensemble_score: float = 0.0  # 0 to 1
    lgb_score: float = 0.0
    momentum_score: float = 0.0

    # Position info
    has_position: float = 0.0  # 1 if holding, 0 if not
    position_pnl: float = 0.0  # Unrealized P&L normalized
    holding_time: float = 0.0  # Days held, normalized

    # Market regime
    regime_trending: float = 0.0
    regime_volatile: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for neural network"""
        return np.array(
            [
                self.price_change_1d,
                self.price_change_5d,
                self.price_change_20d,
                self.rsi_normalized,
                self.macd_normalized,
                self.bb_position,
                self.volume_ratio,
                self.trend_strength,
                self.trend_direction,
                self.volatility,
                self.ensemble_score,
                self.lgb_score,
                self.momentum_score,
                self.has_position,
                self.position_pnl,
                self.holding_time,
                self.regime_trending,
                self.regime_volatile,
            ],
            dtype=np.float32,
        )

    @property
    def size(self) -> int:
        return 18


@dataclass
class Experience:
    """Single experience for replay buffer"""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class TrainingMetrics:
    """Training metrics"""

    episode: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    epsilon: float = 1.0
    loss: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    trades: int = 0


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


if HAS_TORCH:

    class DQNetwork(nn.Module):
        """Deep Q-Network for trading decisions"""

        def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
            super(DQNetwork, self).__init__()

            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc4 = nn.Linear(hidden_size // 2, action_size)

            self.dropout = nn.Dropout(0.2)
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)

        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = F.relu(self.fc3(x))
            return self.fc4(x)


class TradingEnvironment:
    """
    Trading environment for RL agent.
    Simulates trading with historical data.
    """

    def __init__(
        self,
        initial_balance: float = 100000,
        transaction_cost: float = 0.001,  # 0.1% per trade
        max_position_pct: float = 0.1,  # Max 10% per position
        risk_free_rate: float = 0.02,  # 2% annual
    ):
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_pct = max_position_pct
        self.risk_free_rate = risk_free_rate / 252  # Daily

        # State
        self.balance = initial_balance
        self.position = 0  # Number of shares
        self.position_price = 0  # Entry price
        self.position_time = 0  # Days held

        # History
        self.equity_history = []
        self.trade_history = []
        self.returns_history = []

        # Data
        self.df = None
        self.current_idx = 0
        self.current_price = 0

    def reset(self, df: pd.DataFrame) -> TradingState:
        """Reset environment with new data"""
        self.df = df
        self.current_idx = 50  # Start after enough history for indicators

        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0
        self.position_time = 0

        self.equity_history = [self.initial_balance]
        self.trade_history = []
        self.returns_history = []

        return self._get_state()

    def _get_state(self) -> TradingState:
        """Get current state from dataframe"""
        if self.df is None or self.current_idx >= len(self.df):
            return TradingState()

        row = self.df.iloc[self.current_idx]
        close = row["Close"]
        self.current_price = close

        # Price changes
        price_1d = (
            (close / self.df.iloc[self.current_idx - 1]["Close"] - 1)
            if self.current_idx > 0
            else 0
        )
        price_5d = (
            (close / self.df.iloc[self.current_idx - 5]["Close"] - 1)
            if self.current_idx >= 5
            else 0
        )
        price_20d = (
            (close / self.df.iloc[self.current_idx - 20]["Close"] - 1)
            if self.current_idx >= 20
            else 0
        )

        # Get indicators (assume they exist in df)
        rsi = row.get("rsi", 50)
        macd = row.get("macd", 0)
        bb_high = row.get("bb_high", close * 1.02)
        bb_low = row.get("bb_low", close * 0.98)
        volume = row.get("Volume", 1)
        vol_avg = (
            self.df["Volume"]
            .iloc[max(0, self.current_idx - 20) : self.current_idx]
            .mean()
            or 1
        )
        adx = row.get("adx", 20)
        atr = row.get("atr", close * 0.02)

        # Normalize
        rsi_norm = (rsi - 50) / 50
        macd_norm = np.tanh(macd / (close * 0.01))  # Normalize by ~1% of price
        bb_range = bb_high - bb_low if bb_high != bb_low else 1
        bb_pos = (close - bb_low) / bb_range * 2 - 1  # -1 to 1
        vol_ratio = np.tanh((volume / vol_avg - 1))
        trend_strength = adx / 50  # Normalize ADX
        volatility = np.tanh(atr / close * 10)  # Normalize ATR

        # Position info
        has_pos = 1.0 if self.position > 0 else 0.0
        if self.position > 0:
            pnl = (close - self.position_price) / self.position_price
            pnl_norm = np.tanh(pnl * 10)  # Normalize P&L
            hold_time = min(self.position_time / 20, 1.0)  # Max 20 days normalized
        else:
            pnl_norm = 0.0
            hold_time = 0.0

        # Ensemble signals (if available)
        ensemble = row.get("ensemble_score", 0.5)
        lgb = row.get("lgb_score", 0.5)
        momentum = row.get("momentum_score", 0.5)

        # Regime
        regime_trend = 1.0 if adx > 25 else 0.0
        regime_vol = 1.0 if volatility > 0.5 else 0.0

        return TradingState(
            price_change_1d=np.tanh(price_1d * 20),
            price_change_5d=np.tanh(price_5d * 10),
            price_change_20d=np.tanh(price_20d * 5),
            rsi_normalized=rsi_norm,
            macd_normalized=macd_norm,
            bb_position=bb_pos,
            volume_ratio=vol_ratio,
            trend_strength=trend_strength,
            trend_direction=1.0 if price_5d > 0 else -1.0,
            volatility=volatility,
            ensemble_score=ensemble,
            lgb_score=lgb,
            momentum_score=momentum,
            has_position=has_pos,
            position_pnl=pnl_norm,
            holding_time=hold_time,
            regime_trending=regime_trend,
            regime_volatile=regime_vol,
        )

    def step(self, action: int) -> Tuple[TradingState, float, bool, Dict]:
        """
        Execute action and return (next_state, reward, done, info)
        """
        prev_equity = self._get_equity()

        # Execute action
        trade_info = self._execute_action(Action(action))

        # Move to next time step
        self.current_idx += 1
        self.position_time += 1 if self.position > 0 else 0

        # Check if done
        done = self.current_idx >= len(self.df) - 1

        # Calculate reward
        new_equity = self._get_equity()
        self.equity_history.append(new_equity)

        daily_return = (new_equity - prev_equity) / prev_equity
        self.returns_history.append(daily_return)

        reward = self._calculate_reward(daily_return, trade_info)

        # Get new state
        next_state = self._get_state()

        info = {
            "equity": new_equity,
            "position": self.position,
            "trade": trade_info,
            "daily_return": daily_return,
        }

        return next_state, reward, done, info

    def _execute_action(self, action: Action) -> Dict:
        """Execute trading action"""
        trade_info = {"action": action.name, "executed": False}

        if action == Action.BUY and self.position == 0:
            # Calculate position size
            max_value = self.balance * self.max_position_pct
            shares = int(max_value / self.current_price)

            if shares > 0:
                cost = shares * self.current_price * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.balance -= cost
                    self.position = shares
                    self.position_price = self.current_price
                    self.position_time = 0

                    trade_info["executed"] = True
                    trade_info["shares"] = shares
                    trade_info["price"] = self.current_price
                    trade_info["cost"] = cost

                    self.trade_history.append(
                        {
                            "idx": self.current_idx,
                            "action": "BUY",
                            "shares": shares,
                            "price": self.current_price,
                        }
                    )

        elif action == Action.SELL and self.position > 0:
            # Sell all
            proceeds = self.position * self.current_price * (1 - self.transaction_cost)
            pnl = proceeds - (self.position * self.position_price)
            pnl_pct = pnl / (self.position * self.position_price)

            self.balance += proceeds

            trade_info["executed"] = True
            trade_info["shares"] = self.position
            trade_info["price"] = self.current_price
            trade_info["pnl"] = pnl
            trade_info["pnl_pct"] = pnl_pct
            trade_info["holding_days"] = self.position_time

            self.trade_history.append(
                {
                    "idx": self.current_idx,
                    "action": "SELL",
                    "shares": self.position,
                    "price": self.current_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                }
            )

            self.position = 0
            self.position_price = 0
            self.position_time = 0

        return trade_info

    def _get_equity(self) -> float:
        """Get current portfolio equity"""
        position_value = self.position * self.current_price if self.position > 0 else 0
        return self.balance + position_value

    def _calculate_reward(self, daily_return: float, trade_info: Dict) -> float:
        """
        Calculate reward for RL agent.
        Uses risk-adjusted returns with trading penalties.
        """
        reward = 0.0

        # Base reward: daily return (scaled)
        reward += daily_return * 100

        # Risk adjustment: penalize volatility
        if len(self.returns_history) >= 5:
            recent_vol = np.std(self.returns_history[-5:])
            if recent_vol > 0:
                sharpe_component = daily_return / (recent_vol + 1e-8)
                reward += sharpe_component * 0.5

        # Trade rewards/penalties
        if trade_info.get("executed"):
            if trade_info["action"] == "SELL":
                pnl_pct = trade_info.get("pnl_pct", 0)
                # Reward winning trades, penalize losing trades
                if pnl_pct > 0:
                    reward += pnl_pct * 50  # Bonus for profit
                else:
                    reward += (
                        pnl_pct * 30
                    )  # Smaller penalty for loss (encourage cutting losses)

                # Bonus for quick profitable trades
                holding_days = trade_info.get("holding_days", 0)
                if pnl_pct > 0.02 and holding_days < 5:
                    reward += 1.0  # Bonus for quick wins

        # Holding penalty (opportunity cost)
        if self.position > 0 and self.position_time > 10:
            reward -= 0.1  # Small penalty for holding too long

        # Drawdown penalty
        if len(self.equity_history) > 1:
            peak = max(self.equity_history)
            drawdown = (peak - self._get_equity()) / peak
            if drawdown > 0.05:
                reward -= drawdown * 10  # Penalize drawdowns

        return reward

    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        if not self.trade_history:
            return {"trades": 0, "win_rate": 0, "sharpe": 0, "max_dd": 0}

        # Win rate
        sells = [t for t in self.trade_history if t["action"] == "SELL"]
        if sells:
            wins = sum(1 for t in sells if t.get("pnl", 0) > 0)
            win_rate = wins / len(sells)
        else:
            win_rate = 0

        # Sharpe ratio
        if len(self.returns_history) > 1:
            returns = np.array(self.returns_history)
            sharpe = (
                (returns.mean() - self.risk_free_rate)
                / (returns.std() + 1e-8)
                * np.sqrt(252)
            )
        else:
            sharpe = 0

        # Max drawdown
        equity = np.array(self.equity_history)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = drawdown.max()

        # Total return
        total_return = (
            self._get_equity() - self.initial_balance
        ) / self.initial_balance

        return {
            "trades": len(sells),
            "win_rate": win_rate,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "total_return": total_return,
            "final_equity": self._get_equity(),
        }


class RLTradingAgent:
    """
    Deep Q-Learning agent for trading.
    """

    def __init__(
        self,
        state_size: int = 18,
        action_size: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.99,  # Discount factor
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_size: int = 100000,
        target_update_freq: int = 100,
        model_path: str = "store/models/rl_trading_agent.pt",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.model_path = model_path

        # Memory
        self.memory = ReplayBuffer(memory_size)

        # Networks (if PyTorch available)
        self.device = None
        self.policy_net = None
        self.target_net = None
        self.optimizer = None

        if HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net = DQNetwork(state_size, action_size).to(self.device)
            self.target_net = DQNetwork(state_size, action_size).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        else:
            # Fallback: Simple Q-table (discretized)
            self.q_table = {}

        # Training stats
        self.training_step = 0
        self.episode = 0
        self.total_reward = 0
        self.losses = []

        # Try to load existing model
        self._load_model()

        logger.info(
            f"RLTradingAgent initialized (device: {self.device if HAS_TORCH else 'CPU/Q-table'})"
        )

    def select_action(self, state: TradingState, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        state_array = state.to_array()

        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        if HAS_TORCH:
            with torch.no_grad():
                state_tensor = (
                    torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
                )
                self.policy_net.eval()
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        else:
            # Q-table fallback
            state_key = self._discretize_state(state_array)
            if state_key in self.q_table:
                return np.argmax(self.q_table[state_key])
            return random.randint(0, self.action_size - 1)

    def _discretize_state(self, state_array: np.ndarray) -> tuple:
        """Discretize state for Q-table"""
        # Bin each feature into 5 levels
        bins = np.array([-0.6, -0.2, 0.2, 0.6])
        discretized = np.digitize(state_array, bins)
        return tuple(discretized)

    def remember(
        self,
        state: TradingState,
        action: int,
        reward: float,
        next_state: TradingState,
        done: bool,
    ):
        """Store experience in memory"""
        experience = Experience(
            state=state.to_array(),
            action=action,
            reward=reward,
            next_state=next_state.to_array(),
            done=done,
        )
        self.memory.push(experience)

    def learn(self) -> float:
        """Learn from experience replay"""
        if len(self.memory) < self.batch_size:
            return 0.0

        if HAS_TORCH:
            return self._learn_dqn()
        else:
            return self._learn_qtable()

    def _learn_dqn(self) -> float:
        """DQN learning step"""
        # Sample batch
        batch = self.memory.sample(self.batch_size)

        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)

        # Current Q values
        self.policy_net.train()
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q values (Double DQN)
        with torch.no_grad():
            # Get best actions from policy net
            best_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            # Evaluate with target net
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Loss
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def _learn_qtable(self) -> float:
        """Q-table learning step"""
        batch = self.memory.sample(self.batch_size)

        total_update = 0
        for exp in batch:
            state_key = self._discretize_state(exp.state)
            next_state_key = self._discretize_state(exp.next_state)

            # Initialize if not exist
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_size)

            # Q-learning update
            current_q = self.q_table[state_key][exp.action]
            max_next_q = np.max(self.q_table[next_state_key]) if not exp.done else 0
            target_q = exp.reward + self.gamma * max_next_q

            # Update
            update = self.learning_rate * (target_q - current_q)
            self.q_table[state_key][exp.action] += update
            total_update += abs(update)

        return total_update / len(batch)

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_episode(
        self, env: TradingEnvironment, df: pd.DataFrame
    ) -> TrainingMetrics:
        """Train for one episode"""
        state = env.reset(df)
        total_reward = 0
        losses = []

        done = False
        while not done:
            # Select action
            action = self.select_action(state, training=True)

            # Take step
            next_state, reward, done, info = env.step(action)

            # Remember
            self.remember(state, action, reward, next_state, done)

            # Learn
            loss = self.learn()
            if loss > 0:
                losses.append(loss)

            total_reward += reward
            state = next_state

        # Decay epsilon
        self.decay_epsilon()
        self.episode += 1

        # Get metrics
        env_metrics = env.get_metrics()

        return TrainingMetrics(
            episode=self.episode,
            total_reward=total_reward,
            avg_reward=total_reward / max(1, env.current_idx - 50),
            epsilon=self.epsilon,
            loss=np.mean(losses) if losses else 0,
            win_rate=env_metrics["win_rate"],
            sharpe_ratio=env_metrics["sharpe"],
            max_drawdown=env_metrics["max_dd"],
            trades=env_metrics["trades"],
        )

    def save_model(self):
        """Save model to disk"""
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)

        if HAS_TORCH:
            torch.save(
                {
                    "policy_net": self.policy_net.state_dict(),
                    "target_net": self.target_net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epsilon": self.epsilon,
                    "episode": self.episode,
                    "training_step": self.training_step,
                },
                self.model_path,
            )
        else:
            # Save Q-table
            with open(self.model_path.replace(".pt", ".json"), "w") as f:
                # Convert tuple keys to strings
                q_table_serializable = {
                    str(k): v.tolist() for k, v in self.q_table.items()
                }
                json.dump(
                    {
                        "q_table": q_table_serializable,
                        "epsilon": self.epsilon,
                        "episode": self.episode,
                    },
                    f,
                )

        logger.info(f"Model saved to {self.model_path}")

    def _load_model(self):
        """Load model from disk"""
        try:
            if HAS_TORCH and Path(self.model_path).exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint["policy_net"])
                self.target_net.load_state_dict(checkpoint["target_net"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
                self.episode = checkpoint.get("episode", 0)
                self.training_step = checkpoint.get("training_step", 0)
                logger.info(
                    f"Loaded model from {self.model_path} (episode {self.episode})"
                )
            elif not HAS_TORCH:
                json_path = self.model_path.replace(".pt", ".json")
                if Path(json_path).exists():
                    with open(json_path, "r") as f:
                        data = json.load(f)
                    # Convert string keys back to tuples
                    self.q_table = {
                        eval(k): np.array(v) for k, v in data["q_table"].items()
                    }
                    self.epsilon = data.get("epsilon", self.epsilon_end)
                    self.episode = data.get("episode", 0)
                    logger.info(f"Loaded Q-table from {json_path}")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")

    def get_action_probabilities(self, state: TradingState) -> Dict[str, float]:
        """Get action probabilities for display"""
        state_array = state.to_array()

        if HAS_TORCH:
            with torch.no_grad():
                state_tensor = (
                    torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
                )
                self.policy_net.eval()
                q_values = self.policy_net(state_tensor).squeeze().cpu().numpy()
        else:
            state_key = self._discretize_state(state_array)
            q_values = self.q_table.get(state_key, np.zeros(self.action_size))

        # Convert to probabilities using softmax
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / exp_q.sum()

        return {
            "HOLD": float(probs[0]),
            "BUY": float(probs[1]),
            "SELL": float(probs[2]),
            "q_values": q_values.tolist(),
            "recommended": Action(np.argmax(q_values)).name,
        }

    def get_action_with_probs(self, state: TradingState) -> Tuple[int, np.ndarray]:
        """Get best action and probability distribution"""
        state_array = state.to_array()

        if HAS_TORCH:
            with torch.no_grad():
                state_tensor = (
                    torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
                )
                self.policy_net.eval()
                q_values = self.policy_net(state_tensor).squeeze().cpu().numpy()
        else:
            state_key = self._discretize_state(state_array)
            q_values = self.q_table.get(state_key, np.zeros(self.action_size))

        # Convert to probabilities using softmax
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / exp_q.sum()

        # Return best action and all probabilities
        return int(np.argmax(q_values)), probs


# Global instance
_rl_agent: Optional[RLTradingAgent] = None
_rl_env: Optional[TradingEnvironment] = None


def get_rl_agent() -> RLTradingAgent:
    """Get or create global RL agent"""
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = RLTradingAgent()
    return _rl_agent


def get_rl_environment() -> TradingEnvironment:
    """Get or create global RL environment"""
    global _rl_env
    if _rl_env is None:
        _rl_env = TradingEnvironment()
    return _rl_env


# Quick test
if __name__ == "__main__":
    import ta
    import yfinance as yf

    logging.basicConfig(level=logging.INFO)

    print("Testing RL Trading Agent")
    print("=" * 60)

    # Download data
    print("\nDownloading AAPL data...")
    df = yf.download("AAPL", period="1y", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Add indicators
    df["rsi"] = ta.momentum.rsi(df["Close"], window=14)
    df["macd"] = ta.trend.MACD(df["Close"]).macd()
    bb = ta.volatility.BollingerBands(df["Close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["adx"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
    df["atr"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])
    df = df.dropna()

    print(f"Data: {len(df)} bars")

    # Create agent and environment
    agent = get_rl_agent()
    env = get_rl_environment()

    # Train for a few episodes
    print("\nTraining for 5 episodes...")
    for ep in range(5):
        metrics = agent.train_episode(env, df)
        print(
            f"Episode {metrics.episode}: reward={metrics.total_reward:.2f}, "
            f"epsilon={metrics.epsilon:.3f}, trades={metrics.trades}, "
            f"win_rate={metrics.win_rate:.1%}, sharpe={metrics.sharpe_ratio:.2f}"
        )

    # Save model
    agent.save_model()

    # Test inference
    print("\nTesting inference...")
    state = env.reset(df)
    probs = agent.get_action_probabilities(state)
    print(f"Action probabilities: {probs}")
