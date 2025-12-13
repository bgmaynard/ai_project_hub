"""
Warrior Trading RL Execution Agent
Phase 3: Advanced ML - Reinforcement Learning for optimal execution

Uses Deep Q-Network (DQN) to learn optimal trading execution:
- When to enter positions
- How to size positions
- When to exit (partial or full)
- Risk management adjustments

State: Market data, position, sentiment, patterns
Actions: Enter, Hold, Exit, Size Up/Down
Reward: Risk-adjusted returns (Sharpe ratio)
"""

import logging
from typing import Dict, List, Optional, Tuple, Deque
from dataclasses import dataclass
from collections import deque
from datetime import datetime
import numpy as np
import random

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - RL Agent disabled")


logger = logging.getLogger(__name__)


@dataclass
class TradingState:
    """Current trading state representation"""
    # Market data
    price: float
    volume: float
    volatility: float
    trend: float  # -1 to 1

    # Position
    position_size: float  # 0 to 1 (% of capital)
    entry_price: Optional[float]
    unrealized_pnl: float

    # Context
    sentiment_score: float  # -1 to 1
    pattern_confidence: float  # 0 to 1
    time_in_position: int  # bars

    # Risk metrics
    current_drawdown: float
    sharpe_ratio: float
    win_rate: float


@dataclass
class TradingAction:
    """Trading action"""
    action_type: str  # 'enter', 'hold', 'exit', 'size_up', 'size_down'
    size_change: float  # Position size change (-1 to 1)
    confidence: float  # Action confidence (0 to 1)


class ReplayBuffer:
    """Experience replay buffer for training"""

    def __init__(self, capacity: int = 10000):
        self.buffer: Deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List:
        """Sample random batch"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNetwork(nn.Module):
    """
    Deep Q-Network for action value estimation

    Estimates Q(s, a) - expected return for taking action a in state s
    """

    def __init__(
        self,
        state_dim: int = 12,  # TradingState features
        num_actions: int = 5,  # enter, hold, exit, size_up, size_down
        hidden_dims: List[int] = [256, 128, 64]
    ):
        super().__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        # Q-value head (one value per action)
        layers.append(nn.Linear(prev_dim, num_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """
        Args:
            state: (batch, state_dim)
        Returns:
            Q-values: (batch, num_actions)
        """
        return self.network(state)


class DuelingDQNetwork(nn.Module):
    """
    Dueling DQN architecture

    Separately estimates:
    - V(s): State value
    - A(s, a): Action advantage

    Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))

    More stable learning for action-value estimation
    """

    def __init__(
        self,
        state_dim: int = 12,
        num_actions: int = 5,
        hidden_dim: int = 256
    ):
        super().__init__()

        # Shared feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, state):
        """
        Args:
            state: (batch, state_dim)
        Returns:
            Q-values: (batch, num_actions)
        """
        features = self.feature_layer(state)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


class WarriorRLAgent:
    """
    Main RL Agent for trade execution

    Uses Double DQN with Dueling architecture and Prioritized Experience Replay
    """

    ACTIONS = ['enter', 'hold', 'exit', 'size_up', 'size_down']

    def __init__(
        self,
        state_dim: int = 12,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,  # Discount factor
        epsilon_start: float = 1.0,  # Exploration rate
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        device: str = None
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for RL Agent")

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_actions = len(self.ACTIONS)

        # Q-Networks (Double DQN)
        self.policy_net = DuelingDQNetwork(
            state_dim=state_dim,
            num_actions=self.num_actions
        ).to(self.device)

        self.target_net = DuelingDQNetwork(
            state_dim=state_dim,
            num_actions=self.num_actions
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.memory = ReplayBuffer(capacity=10000)

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Training stats
        self.steps_done = 0
        self.episode_rewards = []

        logger.info(f"RL Agent initialized on {self.device}")

    def state_to_tensor(self, state: TradingState) -> torch.Tensor:
        """Convert TradingState to tensor"""
        features = [
            state.price / 100.0,  # Normalize
            state.volume / 1e6,
            state.volatility,
            state.trend,
            state.position_size,
            state.entry_price / 100.0 if state.entry_price else 0.0,
            state.unrealized_pnl,
            state.sentiment_score,
            state.pattern_confidence,
            state.time_in_position / 100.0,
            state.current_drawdown,
            state.sharpe_ratio / 3.0  # Normalize assuming max ~3
        ]

        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def select_action(
        self,
        state: TradingState,
        training: bool = True
    ) -> TradingAction:
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current trading state
            training: If True, use exploration; if False, pure exploitation

        Returns:
            TradingAction
        """
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            # Explore: random action
            action_idx = random.randrange(self.num_actions)
            confidence = 0.5  # Low confidence for random actions
        else:
            # Exploit: best action from Q-network
            with torch.no_grad():
                state_tensor = self.state_to_tensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.argmax().item()

                # Confidence from Q-value distribution
                q_probs = F.softmax(q_values, dim=1)[0]
                confidence = q_probs[action_idx].item()

        action_type = self.ACTIONS[action_idx]

        # Determine size change based on action
        if action_type == 'enter':
            size_change = 0.3  # Enter with 30% of capital
        elif action_type == 'size_up':
            size_change = 0.1  # Increase by 10%
        elif action_type == 'size_down':
            size_change = -0.1  # Decrease by 10%
        elif action_type == 'exit':
            size_change = -1.0  # Exit fully
        else:  # hold
            size_change = 0.0

        return TradingAction(
            action_type=action_type,
            size_change=size_change,
            confidence=confidence
        )

    def calculate_reward(
        self,
        state: TradingState,
        action: TradingAction,
        next_state: TradingState
    ) -> float:
        """
        Calculate reward for state transition

        Reward components:
        1. P&L (risk-adjusted)
        2. Sharpe ratio improvement
        3. Win rate
        4. Drawdown penalty
        """
        # P&L reward (risk-adjusted by position size)
        pnl_change = next_state.unrealized_pnl - state.unrealized_pnl
        risk_adjusted_pnl = pnl_change / (state.position_size + 0.01)  # Avoid div by zero

        # Sharpe ratio improvement
        sharpe_change = next_state.sharpe_ratio - state.sharpe_ratio

        # Drawdown penalty
        drawdown_penalty = -abs(next_state.current_drawdown) * 2

        # Win rate bonus
        win_rate_bonus = next_state.win_rate * 0.5

        # Combine rewards
        reward = (
            risk_adjusted_pnl * 10.0 +  # Primary: P&L
            sharpe_change * 5.0 +        # Sharpe improvement
            win_rate_bonus +             # Win rate
            drawdown_penalty             # Drawdown penalty
        )

        # Bonus for exiting profitable trades
        if action.action_type == 'exit' and state.unrealized_pnl > 0:
            reward += 2.0

        # Penalty for holding losers too long
        if state.time_in_position > 50 and state.unrealized_pnl < 0:
            reward -= 1.0

        return reward

    def train_step(self, batch_size: int = 32):
        """
        Perform one training step

        Uses Double DQN update:
        1. Select action with policy network
        2. Evaluate with target network
        3. Minimize TD error
        """
        if len(self.memory) < batch_size:
            return

        # Sample batch
        batch = self.memory.sample(batch_size)

        # Unpack batch
        state_batch = torch.stack([self.state_to_tensor(s) for s, _, _, _, _ in batch])
        action_batch = torch.tensor([self.ACTIONS.index(a.action_type)
                                     for _, a, _, _, _ in batch], device=self.device)
        reward_batch = torch.tensor([r for _, _, r, _, _ in batch],
                                    dtype=torch.float32, device=self.device)
        next_state_batch = torch.stack([self.state_to_tensor(s)
                                        for _, _, _, s, _ in batch])
        done_batch = torch.tensor([d for _, _, _, _, d in batch],
                                  dtype=torch.float32, device=self.device)

        # Current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Double DQN: use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(1)
            next_q_values = self.target_net(next_state_batch).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze()

            # Target Q values
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Loss (Huber loss for stability)
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def update_target_network(self):
        """Copy policy network weights to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps_done
        }, path)
        logger.info(f"Agent saved to {path}")

    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps']
        logger.info(f"Agent loaded from {path}")


# Global instance
_rl_agent: Optional[WarriorRLAgent] = None


def get_rl_agent(model_path: Optional[str] = None) -> WarriorRLAgent:
    """Get or create global RL agent instance"""
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = WarriorRLAgent()
        if model_path:
            try:
                _rl_agent.load(model_path)
            except Exception as e:
                logger.warning(f"Could not load RL agent from {model_path}: {e}")
    return _rl_agent


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create agent
    agent = get_rl_agent()

    # Example state
    state = TradingState(
        price=150.0,
        volume=1000000,
        volatility=0.02,
        trend=0.5,
        position_size=0.0,
        entry_price=None,
        unrealized_pnl=0.0,
        sentiment_score=0.3,
        pattern_confidence=0.7,
        time_in_position=0,
        current_drawdown=0.0,
        sharpe_ratio=1.5,
        win_rate=0.6
    )

    # Select action
    action = agent.select_action(state, training=False)

    print(f"\nRL Agent Decision:")
    print(f"  Action: {action.action_type}")
    print(f"  Size Change: {action.size_change:+.1%}")
    print(f"  Confidence: {action.confidence:.2%}")
