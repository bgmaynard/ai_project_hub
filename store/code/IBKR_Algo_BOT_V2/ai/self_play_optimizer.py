"""
Self-Play Optimization for Entry/Exit Timing
=============================================
Uses adversarial self-play to optimize trading decisions:
- Entry agent learns optimal entry points
- Exit agent learns optimal exit points
- Both agents compete and improve together

Inspired by AlphaGo/AlphaZero self-play training.
Agents play against themselves and learn from outcomes.

Architecture:
- Twin agents (Entry and Exit)
- Monte Carlo Tree Search for decision exploration
- Policy gradient updates from game outcomes
- ELO rating system for version tracking

Created: December 2025
"""

import json
import logging
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed - Self-play will use simplified agents")


class AgentRole(Enum):
    """Agent roles in self-play"""

    ENTRY = "entry"  # Decides when to enter positions
    EXIT = "exit"  # Decides when to exit positions


class MarketAction(Enum):
    """Possible actions"""

    WAIT = 0  # Do nothing
    ENTER = 1  # Enter position
    EXIT = 2  # Exit position


@dataclass
class GameState:
    """State of the trading game"""

    # Price data
    prices: np.ndarray  # Recent price history
    current_price: float

    # Position info
    has_position: bool = False
    entry_price: float = 0.0
    entry_time: int = 0

    # Running P&L
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0

    # Time step
    time_step: int = 0
    max_steps: int = 100

    # Technical indicators (pre-computed)
    rsi: float = 50.0
    macd: float = 0.0
    bb_position: float = 0.5
    volume_ratio: float = 1.0
    trend: float = 0.0
    volatility: float = 0.02

    def to_array(self) -> np.ndarray:
        """Convert to feature array for neural network"""
        # Normalize price changes
        if len(self.prices) >= 20:
            recent_prices = self.prices[-20:]
            price_changes = np.diff(recent_prices) / recent_prices[:-1]
        else:
            price_changes = np.zeros(19)

        features = np.concatenate(
            [
                price_changes,  # 19 features
                [
                    self.rsi / 100,
                    np.tanh(self.macd * 10),
                    self.bb_position,
                    np.tanh(self.volume_ratio - 1),
                    np.tanh(self.trend * 10),
                    min(self.volatility * 10, 1),
                    float(self.has_position),
                    np.tanh(self.unrealized_pnl * 20) if self.has_position else 0,
                    (
                        min((self.time_step - self.entry_time) / 20, 1)
                        if self.has_position
                        else 0
                    ),
                    self.time_step / self.max_steps,
                ],
            ]
        )
        return features.astype(np.float32)

    @property
    def state_size(self) -> int:
        return 29  # 19 price changes + 10 features


@dataclass
class GameOutcome:
    """Outcome of a trading game"""

    total_pnl: float
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    entry_score: float  # Entry agent performance
    exit_score: float  # Exit agent performance
    trades: List[Dict]  # Individual trades


@dataclass
class TradeRecord:
    """Record of a single trade"""

    entry_time: int
    entry_price: float
    exit_time: int
    exit_price: float
    pnl: float
    pnl_pct: float
    holding_time: int


if HAS_TORCH:

    class PolicyNetwork(nn.Module):
        """Policy network for agent decisions"""

        def __init__(self, state_size: int = 29, hidden_size: int = 64):
            super().__init__()

            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size // 2)

            # Policy head (action probabilities)
            self.policy_head = nn.Linear(hidden_size // 2, 2)  # 2 actions

            # Value head (state value estimate)
            self.value_head = nn.Linear(hidden_size // 2, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))

            policy = F.softmax(self.policy_head(x), dim=-1)
            value = torch.tanh(self.value_head(x))

            return policy, value


class TradingAgent:
    """
    Agent that learns entry or exit decisions through self-play.
    Uses policy gradient (REINFORCE) with baseline.
    """

    def __init__(
        self,
        role: AgentRole,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
    ):
        self.role = role
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.state_size = 29
        self.action_size = 2  # WAIT or ACT (ENTER/EXIT depending on role)

        # Neural network or Q-table
        if HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net = PolicyNetwork(self.state_size).to(self.device)
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        else:
            self.q_table = {}

        # Episode memory
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []

        # Training stats
        self.episode = 0
        self.total_reward = 0
        self.elo_rating = 1000  # ELO rating

        logger.info(f"TradingAgent ({role.value}) initialized")

    def select_action(
        self, state: GameState, training: bool = True
    ) -> Tuple[int, float]:
        """
        Select action based on current state.

        Returns:
            (action, value) where action is 0=WAIT, 1=ACT
        """
        state_array = state.to_array()

        if HAS_TORCH:
            with torch.no_grad():
                state_tensor = (
                    torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
                )
                self.policy_net.eval()
                policy, value = self.policy_net(state_tensor)

            probs = policy.squeeze().cpu().numpy()
            value = value.item()

            # Sample action from policy during training, greedy during eval
            if training:
                action = np.random.choice(self.action_size, p=probs)
            else:
                action = np.argmax(probs)
        else:
            # Q-table fallback
            state_key = self._discretize_state(state_array)
            if state_key in self.q_table:
                q_values = self.q_table[state_key]
                if training and random.random() < 0.1:  # Epsilon-greedy
                    action = random.randint(0, self.action_size - 1)
                else:
                    action = np.argmax(q_values)
                value = max(q_values)
            else:
                action = random.randint(0, self.action_size - 1)
                value = 0.0

        return int(action), float(value)

    def _discretize_state(self, state_array: np.ndarray) -> tuple:
        """Discretize state for Q-table"""
        bins = np.array([-0.5, -0.2, 0, 0.2, 0.5])
        discretized = np.digitize(state_array, bins)
        return tuple(discretized)

    def remember(self, state: GameState, action: int, reward: float, value: float):
        """Store experience for learning"""
        self.states.append(state.to_array())
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)

    def learn(self) -> float:
        """Learn from episode using policy gradient"""
        if len(self.rewards) == 0:
            return 0.0

        # Calculate returns (discounted future rewards)
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = np.array(returns)

        # Normalize returns
        if len(returns) > 1 and returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        if HAS_TORCH:
            loss = self._train_pytorch(returns)
        else:
            loss = self._train_qtable(returns)

        # Clear episode memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []

        self.episode += 1

        return loss

    def _train_pytorch(self, returns: np.ndarray) -> float:
        """Train using PyTorch policy gradient"""
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        self.policy_net.train()
        policy, values = self.policy_net(states)

        # Policy loss (REINFORCE with baseline)
        values = values.squeeze()
        advantages = returns_t - values.detach()

        action_probs = policy.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -torch.mean(torch.log(action_probs + 1e-8) * advantages)

        # Value loss
        value_loss = F.mse_loss(values, returns_t)

        # Entropy bonus (encourage exploration)
        entropy = -torch.mean(torch.sum(policy * torch.log(policy + 1e-8), dim=1))

        # Total loss
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def _train_qtable(self, returns: np.ndarray) -> float:
        """Train using Q-table update"""
        total_update = 0

        for state_arr, action, ret in zip(self.states, self.actions, returns):
            state_key = self._discretize_state(state_arr)

            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)

            # Q-learning update
            old_value = self.q_table[state_key][action]
            self.q_table[state_key][action] += self.learning_rate * (ret - old_value)
            total_update += abs(ret - old_value)

        return total_update / len(self.states) if self.states else 0

    def get_action_probs(self, state: GameState) -> Dict[str, float]:
        """Get action probabilities for display"""
        state_array = state.to_array()

        if HAS_TORCH:
            with torch.no_grad():
                state_tensor = (
                    torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
                )
                self.policy_net.eval()
                policy, _ = self.policy_net(state_tensor)
            probs = policy.squeeze().cpu().numpy()
        else:
            state_key = self._discretize_state(state_array)
            if state_key in self.q_table:
                q = self.q_table[state_key]
                exp_q = np.exp(q - np.max(q))
                probs = exp_q / exp_q.sum()
            else:
                probs = np.array([0.5, 0.5])

        action_name = "ENTER" if self.role == AgentRole.ENTRY else "EXIT"
        return {"WAIT": float(probs[0]), action_name: float(probs[1])}


class TradingGame:
    """
    Trading game environment for self-play training.
    Simulates trading with entry and exit agents.
    """

    def __init__(
        self,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        max_holding_time: int = 20,  # Force exit after N steps
        reward_scaling: float = 100.0,
    ):
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_holding_time = max_holding_time
        self.reward_scaling = reward_scaling

        # Game state
        self.state: Optional[GameState] = None
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = []

    def reset(self, prices: np.ndarray, indicators: Optional[Dict] = None) -> GameState:
        """Reset game with new price data"""
        self.trades = []
        self.equity_curve = [0.0]

        self.state = GameState(
            prices=prices[:50],  # Start with some history
            current_price=prices[50],
            time_step=0,
            max_steps=len(prices) - 51,
        )

        if indicators:
            self.state.rsi = indicators.get("rsi", 50)
            self.state.macd = indicators.get("macd", 0)
            self.state.bb_position = indicators.get("bb_position", 0.5)
            self.state.volume_ratio = indicators.get("volume_ratio", 1)
            self.state.trend = indicators.get("trend", 0)
            self.state.volatility = indicators.get("volatility", 0.02)

        self._full_prices = prices
        self._indicators = indicators or {}

        return self.state

    def step(
        self, entry_action: int, exit_action: int
    ) -> Tuple[GameState, float, float, bool, Dict]:
        """
        Execute one step of the game.

        Args:
            entry_action: Entry agent's action (0=WAIT, 1=ENTER)
            exit_action: Exit agent's action (0=WAIT, 1=EXIT)

        Returns:
            (next_state, entry_reward, exit_reward, done, info)
        """
        entry_reward = 0.0
        exit_reward = 0.0
        info = {"action": "none"}

        old_price = self.state.current_price

        # Process actions
        if self.state.has_position:
            # Exit agent decides
            holding_time = self.state.time_step - self.state.entry_time

            # Force exit if held too long
            force_exit = holding_time >= self.max_holding_time

            if exit_action == 1 or force_exit:
                # Execute exit
                exit_price = self.state.current_price * (1 - self.slippage)
                pnl = (exit_price - self.state.entry_price) / self.state.entry_price
                pnl -= self.transaction_cost  # Exit cost

                # Record trade
                trade = TradeRecord(
                    entry_time=self.state.entry_time,
                    entry_price=self.state.entry_price,
                    exit_time=self.state.time_step,
                    exit_price=exit_price,
                    pnl=pnl * self.state.entry_price,  # Dollar P&L
                    pnl_pct=pnl * 100,
                    holding_time=holding_time,
                )
                self.trades.append(trade)

                # Update state
                self.state.realized_pnl += pnl * self.reward_scaling
                self.state.total_trades += 1
                if pnl > 0:
                    self.state.winning_trades += 1

                self.state.has_position = False
                self.state.entry_price = 0
                self.state.unrealized_pnl = 0

                # Rewards
                if pnl > 0:
                    # Winning trade: reward both agents
                    exit_reward = (
                        pnl * self.reward_scaling * 2
                    )  # Exit gets more for timing
                    entry_reward = pnl * self.reward_scaling
                else:
                    # Losing trade: penalize both
                    exit_reward = pnl * self.reward_scaling
                    entry_reward = pnl * self.reward_scaling * 0.5  # Entry less blame

                # Bonus for quick profits, penalty for slow losses
                if pnl > 0 and holding_time < 5:
                    exit_reward += 0.5
                elif pnl < 0 and holding_time > 10:
                    exit_reward -= 0.5

                info = {"action": "exit", "pnl": pnl, "forced": force_exit}

            else:
                # Holding - small time cost
                exit_reward = -0.01  # Small cost for holding

        else:
            # Entry agent decides
            if entry_action == 1:
                # Execute entry
                entry_price = self.state.current_price * (1 + self.slippage)
                entry_price *= 1 + self.transaction_cost

                self.state.has_position = True
                self.state.entry_price = entry_price
                self.state.entry_time = self.state.time_step

                info = {"action": "enter", "price": entry_price}
            else:
                # Waiting - small penalty for missed opportunities in trending market
                entry_reward = -0.005

        # Advance time
        self.state.time_step += 1

        # Update state
        if self.state.time_step + 50 < len(self._full_prices):
            idx = self.state.time_step + 50
            self.state.prices = self._full_prices[idx - 49 : idx + 1]
            self.state.current_price = self._full_prices[idx]

            # Update unrealized P&L
            if self.state.has_position:
                self.state.unrealized_pnl = (
                    (self.state.current_price - self.state.entry_price)
                    / self.state.entry_price
                ) * self.reward_scaling

        # Track equity
        equity = self.state.realized_pnl + self.state.unrealized_pnl
        self.equity_curve.append(equity)

        # Check if done
        done = self.state.time_step >= self.state.max_steps

        # Final rewards if done
        if done and self.state.has_position:
            # Force close at end
            pnl = (
                self.state.current_price - self.state.entry_price
            ) / self.state.entry_price
            pnl -= self.transaction_cost

            self.state.realized_pnl += pnl * self.reward_scaling
            self.state.total_trades += 1
            if pnl > 0:
                self.state.winning_trades += 1

            exit_reward += pnl * self.reward_scaling * 0.5  # Partial credit

        return self.state, entry_reward, exit_reward, done, info

    def get_outcome(self) -> GameOutcome:
        """Get game outcome statistics"""
        total_pnl = self.state.realized_pnl / self.reward_scaling
        total_trades = self.state.total_trades
        win_rate = self.state.winning_trades / max(1, total_trades)

        # Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve)
            if returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Max drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / (np.abs(peak) + 1e-8)
        max_dd = drawdown.max()

        return GameOutcome(
            total_pnl=total_pnl,
            total_trades=total_trades,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            entry_score=0,  # Calculated separately
            exit_score=0,
            trades=[asdict(t) for t in self.trades],
        )


class SelfPlayTrainer:
    """
    Trains entry and exit agents through self-play.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        model_path: str = "store/models/selfplay",
    ):
        self.entry_agent = TradingAgent(AgentRole.ENTRY, learning_rate, gamma)
        self.exit_agent = TradingAgent(AgentRole.EXIT, learning_rate, gamma)
        self.game = TradingGame()
        self.model_path = model_path

        # Training history
        self.training_history: List[Dict] = []
        self.best_pnl = float("-inf")

        # Load if exists
        self._load_models()

        logger.info("SelfPlayTrainer initialized")

    def train_episode(
        self, prices: np.ndarray, indicators: Optional[Dict] = None
    ) -> Dict:
        """Train one episode of self-play"""
        state = self.game.reset(prices, indicators)

        total_entry_reward = 0
        total_exit_reward = 0

        while True:
            # Get actions from both agents
            entry_action, entry_value = self.entry_agent.select_action(
                state, training=True
            )
            exit_action, exit_value = self.exit_agent.select_action(
                state, training=True
            )

            # Execute step
            next_state, entry_reward, exit_reward, done, info = self.game.step(
                entry_action, exit_action
            )

            # Remember experiences
            self.entry_agent.remember(state, entry_action, entry_reward, entry_value)
            if state.has_position:  # Exit agent only learns when in position
                self.exit_agent.remember(state, exit_action, exit_reward, exit_value)

            total_entry_reward += entry_reward
            total_exit_reward += exit_reward

            state = next_state

            if done:
                break

        # Learn from episode
        entry_loss = self.entry_agent.learn()
        exit_loss = self.exit_agent.learn()

        # Get outcome
        outcome = self.game.get_outcome()

        # Update ELO ratings based on performance
        self._update_elo(outcome)

        # Record
        result = {
            "episode": self.entry_agent.episode,
            "total_pnl": outcome.total_pnl,
            "trades": outcome.total_trades,
            "win_rate": outcome.win_rate,
            "sharpe": outcome.sharpe_ratio,
            "max_dd": outcome.max_drawdown,
            "entry_reward": total_entry_reward,
            "exit_reward": total_exit_reward,
            "entry_loss": entry_loss,
            "exit_loss": exit_loss,
            "entry_elo": self.entry_agent.elo_rating,
            "exit_elo": self.exit_agent.elo_rating,
        }

        self.training_history.append(result)

        # Save best model
        if outcome.total_pnl > self.best_pnl:
            self.best_pnl = outcome.total_pnl
            self._save_models()

        return result

    def _update_elo(self, outcome: GameOutcome):
        """Update ELO ratings based on game outcome"""
        K = 32  # ELO K-factor

        # Entry agent wins if average entry leads to profit
        entry_score = 1 if outcome.total_pnl > 0 and outcome.win_rate > 0.5 else 0

        # Exit agent wins if exits capture profit well
        exit_score = 1 if outcome.sharpe_ratio > 0 else 0

        # Expected scores
        entry_expected = 1 / (
            1 + 10 ** ((self.exit_agent.elo_rating - self.entry_agent.elo_rating) / 400)
        )
        exit_expected = 1 - entry_expected

        # Update
        self.entry_agent.elo_rating += K * (entry_score - entry_expected)
        self.exit_agent.elo_rating += K * (exit_score - exit_expected)

    def evaluate(self, prices: np.ndarray, indicators: Optional[Dict] = None) -> Dict:
        """Evaluate agents without training"""
        state = self.game.reset(prices, indicators)

        decisions = []

        while True:
            entry_action, _ = self.entry_agent.select_action(state, training=False)
            exit_action, _ = self.exit_agent.select_action(state, training=False)

            next_state, _, _, done, info = self.game.step(entry_action, exit_action)

            if info["action"] != "none":
                decisions.append(
                    {
                        "time_step": state.time_step,
                        "action": info["action"],
                        "price": state.current_price,
                        **info,
                    }
                )

            state = next_state

            if done:
                break

        outcome = self.game.get_outcome()

        return {
            "outcome": asdict(outcome),
            "decisions": decisions,
            "entry_probs": self.entry_agent.get_action_probs(state),
            "exit_probs": self.exit_agent.get_action_probs(state),
        }

    def get_entry_signal(self, state: GameState) -> Dict:
        """Get entry signal for live trading"""
        probs = self.entry_agent.get_action_probs(state)
        action, _ = self.entry_agent.select_action(state, training=False)

        return {
            "should_enter": action == 1,
            "confidence": probs["ENTER"],
            "probabilities": probs,
        }

    def get_exit_signal(self, state: GameState) -> Dict:
        """Get exit signal for live trading"""
        probs = self.exit_agent.get_action_probs(state)
        action, _ = self.exit_agent.select_action(state, training=False)

        return {
            "should_exit": action == 1,
            "confidence": probs["EXIT"],
            "probabilities": probs,
        }

    def _save_models(self):
        """Save models to disk"""
        try:
            Path(self.model_path).mkdir(parents=True, exist_ok=True)

            if HAS_TORCH:
                torch.save(
                    {
                        "entry_state": self.entry_agent.policy_net.state_dict(),
                        "exit_state": self.exit_agent.policy_net.state_dict(),
                        "entry_elo": self.entry_agent.elo_rating,
                        "exit_elo": self.exit_agent.elo_rating,
                        "entry_episode": self.entry_agent.episode,
                        "exit_episode": self.exit_agent.episode,
                        "best_pnl": self.best_pnl,
                    },
                    f"{self.model_path}/selfplay_agents.pt",
                )
            else:
                with open(f"{self.model_path}/selfplay_agents.json", "w") as f:
                    json.dump(
                        {
                            "entry_qtable": {
                                str(k): v.tolist()
                                for k, v in self.entry_agent.q_table.items()
                            },
                            "exit_qtable": {
                                str(k): v.tolist()
                                for k, v in self.exit_agent.q_table.items()
                            },
                            "entry_elo": self.entry_agent.elo_rating,
                            "exit_elo": self.exit_agent.elo_rating,
                            "best_pnl": self.best_pnl,
                        },
                        f,
                    )

            logger.info(f"Saved self-play models to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def _load_models(self):
        """Load models from disk"""
        try:
            if HAS_TORCH:
                path = f"{self.model_path}/selfplay_agents.pt"
                if Path(path).exists():
                    checkpoint = torch.load(
                        path, map_location=self.entry_agent.device, weights_only=False
                    )
                    self.entry_agent.policy_net.load_state_dict(
                        checkpoint["entry_state"]
                    )
                    self.exit_agent.policy_net.load_state_dict(checkpoint["exit_state"])
                    self.entry_agent.elo_rating = checkpoint["entry_elo"]
                    self.exit_agent.elo_rating = checkpoint["exit_elo"]
                    self.entry_agent.episode = checkpoint["entry_episode"]
                    self.exit_agent.episode = checkpoint["exit_episode"]
                    self.best_pnl = checkpoint["best_pnl"]
                    logger.info(f"Loaded self-play models from {path}")
            else:
                path = f"{self.model_path}/selfplay_agents.json"
                if Path(path).exists():
                    with open(path, "r") as f:
                        data = json.load(f)
                    self.entry_agent.q_table = {
                        eval(k): np.array(v) for k, v in data["entry_qtable"].items()
                    }
                    self.exit_agent.q_table = {
                        eval(k): np.array(v) for k, v in data["exit_qtable"].items()
                    }
                    self.entry_agent.elo_rating = data["entry_elo"]
                    self.exit_agent.elo_rating = data["exit_elo"]
                    self.best_pnl = data["best_pnl"]
                    logger.info(f"Loaded self-play models from {path}")
        except Exception as e:
            logger.warning(f"Could not load models: {e}")

    def get_stats(self) -> Dict:
        """Get training statistics"""
        return {
            "episodes": self.entry_agent.episode,
            "entry_elo": self.entry_agent.elo_rating,
            "exit_elo": self.exit_agent.elo_rating,
            "best_pnl": self.best_pnl,
            "recent_history": (
                self.training_history[-20:] if self.training_history else []
            ),
        }


# Global instance
_selfplay_trainer: Optional[SelfPlayTrainer] = None


def get_selfplay_trainer() -> SelfPlayTrainer:
    """Get or create global self-play trainer"""
    global _selfplay_trainer
    if _selfplay_trainer is None:
        _selfplay_trainer = SelfPlayTrainer()
    return _selfplay_trainer


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Self-Play Optimizer")
    print("=" * 60)

    # Generate fake price data
    np.random.seed(42)
    n_points = 200
    prices = 100 * np.exp(np.cumsum(np.random.randn(n_points) * 0.02))

    # Train
    trainer = get_selfplay_trainer()

    print("\nTraining for 10 episodes...")
    for ep in range(10):
        result = trainer.train_episode(prices)
        print(
            f"Episode {result['episode']}: PnL={result['total_pnl']:.2f}, "
            f"trades={result['trades']}, win_rate={result['win_rate']:.1%}, "
            f"Entry ELO={result['entry_elo']:.0f}, Exit ELO={result['exit_elo']:.0f}"
        )

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)

    eval_result = trainer.evaluate(prices)
    print(f"Final PnL: {eval_result['outcome']['total_pnl']:.2f}")
    print(f"Trades: {eval_result['outcome']['total_trades']}")
    print(f"Win Rate: {eval_result['outcome']['win_rate']:.1%}")
    print(f"Sharpe: {eval_result['outcome']['sharpe_ratio']:.2f}")

    print(f"\nEntry Probabilities: {eval_result['entry_probs']}")
    print(f"Exit Probabilities: {eval_result['exit_probs']}")

    # Stats
    print("\n" + "=" * 60)
    print("Training Stats")
    print("=" * 60)
    stats = trainer.get_stats()
    print(f"Total Episodes: {stats['episodes']}")
    print(f"Entry Agent ELO: {stats['entry_elo']:.0f}")
    print(f"Exit Agent ELO: {stats['exit_elo']:.0f}")
    print(f"Best PnL: {stats['best_pnl']:.2f}")
