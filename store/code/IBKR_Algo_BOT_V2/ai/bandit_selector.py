"""
Multi-Armed Bandit for Symbol & Model Selection
================================================
Uses Thompson Sampling to intelligently select:
1. Which symbols to trade (based on historical performance)
2. Which models to use for each symbol
3. Which strategies to deploy

The bandit learns from trade outcomes and adapts dynamically,
balancing exploration (trying new options) vs exploitation
(using what works best).

Architecture:
- Thompson Sampling with Beta distributions
- Contextual bandits for regime-aware selection
- Decay for non-stationarity (market changes)
- Per-symbol and per-model tracking

Created: December 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import json
import logging
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ArmType(Enum):
    """Types of arms (options) the bandit can select"""
    SYMBOL = "symbol"
    MODEL = "model"
    STRATEGY = "strategy"
    TIMEFRAME = "timeframe"


@dataclass
class BetaArm:
    """
    Single arm with Beta distribution for Thompson Sampling.
    Beta(alpha, beta) where:
    - alpha = successes + 1
    - beta = failures + 1
    """
    name: str
    arm_type: ArmType
    alpha: float = 1.0  # Successes + prior
    beta: float = 1.0   # Failures + prior
    total_pulls: int = 0
    total_reward: float = 0.0
    last_updated: str = ""

    # Contextual stats (per regime)
    regime_alpha: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 1.0))
    regime_beta: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 1.0))

    def sample(self, regime: Optional[str] = None) -> float:
        """Sample from Beta distribution (Thompson Sampling)"""
        if regime and regime in self.regime_alpha:
            a = self.regime_alpha[regime]
            b = self.regime_beta[regime]
        else:
            a = self.alpha
            b = self.beta
        return np.random.beta(a, b)

    def update(self, reward: float, regime: Optional[str] = None, decay: float = 0.999):
        """
        Update arm with reward (0 to 1).
        Uses decay to handle non-stationarity.
        """
        # Apply decay to old observations
        self.alpha = 1 + (self.alpha - 1) * decay
        self.beta = 1 + (self.beta - 1) * decay

        # Update with new observation
        # For binary outcomes: reward = 1 (success) or 0 (failure)
        # For continuous: reward in [0, 1] treated as probability
        self.alpha += reward
        self.beta += (1 - reward)

        self.total_pulls += 1
        self.total_reward += reward
        self.last_updated = datetime.now().isoformat()

        # Update contextual stats
        if regime:
            self.regime_alpha[regime] = 1 + (self.regime_alpha.get(regime, 1) - 1) * decay + reward
            self.regime_beta[regime] = 1 + (self.regime_beta.get(regime, 1) - 1) * decay + (1 - reward)

    @property
    def mean(self) -> float:
        """Expected value (mean of Beta distribution)"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance of Beta distribution"""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """95% confidence interval"""
        # Approximate using Beta quantiles
        from scipy import stats
        try:
            low = stats.beta.ppf(0.025, self.alpha, self.beta)
            high = stats.beta.ppf(0.975, self.alpha, self.beta)
            return (low, high)
        except:
            return (0.0, 1.0)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'arm_type': self.arm_type.value,
            'alpha': self.alpha,
            'beta': self.beta,
            'total_pulls': self.total_pulls,
            'total_reward': self.total_reward,
            'mean': self.mean,
            'variance': self.variance,
            'last_updated': self.last_updated
        }


@dataclass
class SelectionResult:
    """Result of bandit selection"""
    selected: str
    arm_type: ArmType
    sample_value: float
    expected_value: float
    confidence: float
    regime: Optional[str]
    alternatives: List[Dict]  # Other options with their scores
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            'selected': self.selected,
            'arm_type': self.arm_type.value,
            'sample_value': self.sample_value,
            'expected_value': self.expected_value,
            'confidence': self.confidence,
            'regime': self.regime,
            'alternatives': self.alternatives,
            'timestamp': self.timestamp
        }


class MultiarmedBandit:
    """
    Multi-Armed Bandit with Thompson Sampling.
    Handles symbol, model, and strategy selection.
    """

    def __init__(
        self,
        decay_rate: float = 0.999,  # Decay for non-stationarity
        min_pulls_for_confidence: int = 10,
        state_path: str = "store/bandit_state.json"
    ):
        self.decay_rate = decay_rate
        self.min_pulls_for_confidence = min_pulls_for_confidence
        self.state_path = state_path

        # Arms organized by type
        self.arms: Dict[ArmType, Dict[str, BetaArm]] = {
            ArmType.SYMBOL: {},
            ArmType.MODEL: {},
            ArmType.STRATEGY: {},
            ArmType.TIMEFRAME: {}
        }

        # Selection history
        self.selection_history: List[Dict] = []
        self.max_history = 1000

        # Load state
        self._load_state()

        logger.info("MultiarmedBandit initialized")

    def add_arm(self, name: str, arm_type: ArmType, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """Add a new arm (option) to the bandit"""
        if name not in self.arms[arm_type]:
            self.arms[arm_type][name] = BetaArm(
                name=name,
                arm_type=arm_type,
                alpha=prior_alpha,
                beta=prior_beta
            )
            logger.debug(f"Added {arm_type.value} arm: {name}")

    def select(
        self,
        arm_type: ArmType,
        candidates: Optional[List[str]] = None,
        regime: Optional[str] = None,
        top_k: int = 1,
        explore_boost: float = 0.0
    ) -> SelectionResult:
        """
        Select best arm(s) using Thompson Sampling.

        Args:
            arm_type: Type of arm to select (SYMBOL, MODEL, etc.)
            candidates: Optional list of candidates to consider (default: all)
            regime: Market regime for contextual selection
            top_k: Number of top arms to consider (returns best)
            explore_boost: Additional exploration bonus (0-1)

        Returns:
            SelectionResult with selected arm and metadata
        """
        available_arms = self.arms[arm_type]

        if candidates:
            # Filter to specified candidates, add if missing
            for name in candidates:
                if name not in available_arms:
                    self.add_arm(name, arm_type)
            available_arms = {k: v for k, v in available_arms.items() if k in candidates}

        if not available_arms:
            raise ValueError(f"No arms available for type {arm_type.value}")

        # Sample from each arm
        samples = []
        for name, arm in available_arms.items():
            sample = arm.sample(regime)

            # Add exploration bonus for under-explored arms
            if arm.total_pulls < self.min_pulls_for_confidence:
                explore_bonus = explore_boost * (1 - arm.total_pulls / self.min_pulls_for_confidence)
                sample = min(1.0, sample + explore_bonus)

            samples.append({
                'name': name,
                'sample': sample,
                'mean': arm.mean,
                'pulls': arm.total_pulls,
                'variance': arm.variance
            })

        # Sort by sample value (Thompson Sampling)
        samples.sort(key=lambda x: x['sample'], reverse=True)

        # Select best
        best = samples[0]
        selected_arm = available_arms[best['name']]

        # Calculate confidence
        confidence = 1.0 - selected_arm.variance if selected_arm.total_pulls >= self.min_pulls_for_confidence else 0.5

        result = SelectionResult(
            selected=best['name'],
            arm_type=arm_type,
            sample_value=best['sample'],
            expected_value=best['mean'],
            confidence=confidence,
            regime=regime,
            alternatives=samples[:5],  # Top 5 alternatives
            timestamp=datetime.now().isoformat()
        )

        # Record selection
        self.selection_history.append(result.to_dict())
        if len(self.selection_history) > self.max_history:
            self.selection_history.pop(0)

        return result

    def select_multiple(
        self,
        arm_type: ArmType,
        n: int,
        candidates: Optional[List[str]] = None,
        regime: Optional[str] = None
    ) -> List[str]:
        """Select top n arms without replacement"""
        available_arms = self.arms[arm_type]

        if candidates:
            for name in candidates:
                if name not in available_arms:
                    self.add_arm(name, arm_type)
            available_arms = {k: v for k, v in available_arms.items() if k in candidates}

        # Sample all
        samples = [(name, arm.sample(regime)) for name, arm in available_arms.items()]
        samples.sort(key=lambda x: x[1], reverse=True)

        return [name for name, _ in samples[:n]]

    def update(
        self,
        name: str,
        arm_type: ArmType,
        reward: float,
        regime: Optional[str] = None
    ):
        """
        Update arm with observed reward.

        Args:
            name: Arm name
            arm_type: Type of arm
            reward: Reward value (0 to 1, or binary 0/1)
            regime: Optional market regime for contextual update
        """
        if name not in self.arms[arm_type]:
            self.add_arm(name, arm_type)

        self.arms[arm_type][name].update(reward, regime, self.decay_rate)
        logger.debug(f"Updated {arm_type.value}/{name}: reward={reward:.3f}, mean={self.arms[arm_type][name].mean:.3f}")

    def record_trade_outcome(
        self,
        symbol: str,
        model: str,
        strategy: str,
        pnl_pct: float,
        regime: Optional[str] = None
    ):
        """
        Record trade outcome and update all relevant arms.

        Args:
            symbol: Traded symbol
            model: Model used for prediction
            strategy: Trading strategy used
            pnl_pct: Profit/loss percentage (-100 to +100)
            regime: Market regime at time of trade
        """
        # Convert P&L to reward (0 to 1)
        # Winning trade = 1, losing trade = 0
        # With magnitude: scale by P&L size

        if pnl_pct > 0:
            # Winner: reward proportional to gain (capped)
            reward = min(1.0, 0.5 + pnl_pct / 20)  # 10% gain = 1.0 reward
        else:
            # Loser: penalty proportional to loss
            reward = max(0.0, 0.5 + pnl_pct / 20)  # -10% loss = 0.0 reward

        # Update all relevant arms
        self.update(symbol, ArmType.SYMBOL, reward, regime)
        self.update(model, ArmType.MODEL, reward, regime)
        self.update(strategy, ArmType.STRATEGY, reward, regime)

        logger.info(f"Recorded trade: {symbol} ({model}/{strategy}) -> {pnl_pct:+.2f}% = reward {reward:.3f}")

    def get_rankings(self, arm_type: ArmType, regime: Optional[str] = None) -> List[Dict]:
        """Get ranked list of arms by expected value"""
        arms = self.arms[arm_type]

        rankings = []
        for name, arm in arms.items():
            rankings.append({
                'name': name,
                'mean': arm.mean,
                'variance': arm.variance,
                'pulls': arm.total_pulls,
                'total_reward': arm.total_reward,
                'confidence_interval': arm.confidence_interval,
                'last_updated': arm.last_updated
            })

        rankings.sort(key=lambda x: x['mean'], reverse=True)
        return rankings

    def get_stats(self) -> Dict:
        """Get overall bandit statistics"""
        stats = {
            'total_selections': len(self.selection_history),
            'arms_by_type': {}
        }

        for arm_type in ArmType:
            arms = self.arms[arm_type]
            if arms:
                stats['arms_by_type'][arm_type.value] = {
                    'count': len(arms),
                    'total_pulls': sum(a.total_pulls for a in arms.values()),
                    'best_arm': max(arms.values(), key=lambda a: a.mean).name if arms else None,
                    'best_mean': max(a.mean for a in arms.values()) if arms else 0
                }

        return stats

    def _save_state(self):
        """Save bandit state to disk"""
        try:
            Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)

            state = {
                'arms': {},
                'selection_history': self.selection_history[-100:],  # Last 100
                'saved_at': datetime.now().isoformat()
            }

            for arm_type in ArmType:
                state['arms'][arm_type.value] = {
                    name: {
                        'alpha': arm.alpha,
                        'beta': arm.beta,
                        'total_pulls': arm.total_pulls,
                        'total_reward': arm.total_reward,
                        'last_updated': arm.last_updated,
                        'regime_alpha': dict(arm.regime_alpha),
                        'regime_beta': dict(arm.regime_beta)
                    }
                    for name, arm in self.arms[arm_type].items()
                }

            with open(self.state_path, 'w') as f:
                json.dump(state, f, indent=2)

            logger.debug(f"Saved bandit state to {self.state_path}")
        except Exception as e:
            logger.error(f"Failed to save bandit state: {e}")

    def _load_state(self):
        """Load bandit state from disk"""
        try:
            if Path(self.state_path).exists():
                with open(self.state_path, 'r') as f:
                    state = json.load(f)

                for arm_type_str, arms_data in state.get('arms', {}).items():
                    arm_type = ArmType(arm_type_str)
                    for name, data in arms_data.items():
                        arm = BetaArm(
                            name=name,
                            arm_type=arm_type,
                            alpha=data['alpha'],
                            beta=data['beta'],
                            total_pulls=data['total_pulls'],
                            total_reward=data['total_reward'],
                            last_updated=data.get('last_updated', '')
                        )
                        # Load regime stats
                        arm.regime_alpha = defaultdict(lambda: 1.0, data.get('regime_alpha', {}))
                        arm.regime_beta = defaultdict(lambda: 1.0, data.get('regime_beta', {}))
                        self.arms[arm_type][name] = arm

                self.selection_history = state.get('selection_history', [])
                logger.info(f"Loaded bandit state from {self.state_path}")
        except Exception as e:
            logger.warning(f"Could not load bandit state: {e}")

    def save(self):
        """Explicit save"""
        self._save_state()


class SymbolSelector:
    """
    High-level symbol selector using bandit.
    Combines bandit selection with market filters.
    """

    def __init__(self, bandit: Optional[MultiarmedBandit] = None):
        self.bandit = bandit or get_bandit()

        # Default symbols to track
        self.default_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            'AMD', 'INTC', 'CRM', 'NFLX', 'PYPL', 'SQ', 'SHOP', 'COIN'
        ]

        # Initialize arms
        for symbol in self.default_symbols:
            self.bandit.add_arm(symbol, ArmType.SYMBOL)

    def select_symbols(
        self,
        n: int = 5,
        candidates: Optional[List[str]] = None,
        regime: Optional[str] = None,
        require_min_pulls: bool = False
    ) -> List[str]:
        """
        Select top n symbols for trading.

        Args:
            n: Number of symbols to select
            candidates: Optional list of candidates (default: all tracked)
            regime: Market regime for contextual selection
            require_min_pulls: Only include symbols with enough history

        Returns:
            List of selected symbols
        """
        if candidates is None:
            candidates = self.default_symbols

        # Filter by min pulls if required
        if require_min_pulls:
            candidates = [
                s for s in candidates
                if s in self.bandit.arms[ArmType.SYMBOL]
                and self.bandit.arms[ArmType.SYMBOL][s].total_pulls >= self.bandit.min_pulls_for_confidence
            ]

        if not candidates:
            return self.default_symbols[:n]

        return self.bandit.select_multiple(ArmType.SYMBOL, n, candidates, regime)

    def get_symbol_scores(self) -> Dict[str, Dict]:
        """Get current scores for all tracked symbols"""
        rankings = self.bandit.get_rankings(ArmType.SYMBOL)
        return {r['name']: r for r in rankings}


class ModelSelector:
    """
    Model selector using bandit.
    Selects best prediction model for each symbol/regime.
    """

    def __init__(self, bandit: Optional[MultiarmedBandit] = None):
        self.bandit = bandit or get_bandit()

        # Available models
        self.available_models = [
            'lightgbm',      # LightGBM ML model
            'ensemble',      # Ensemble predictor
            'heuristic',     # Pure heuristic rules
            'momentum',      # Momentum scoring
            'rl_agent'       # RL trading agent
        ]

        # Initialize arms
        for model in self.available_models:
            self.bandit.add_arm(model, ArmType.MODEL)

    def select_model(
        self,
        symbol: Optional[str] = None,
        regime: Optional[str] = None
    ) -> str:
        """
        Select best model for prediction.

        Args:
            symbol: Optional symbol (for per-symbol model tracking)
            regime: Market regime for contextual selection

        Returns:
            Selected model name
        """
        result = self.bandit.select(
            ArmType.MODEL,
            candidates=self.available_models,
            regime=regime
        )
        return result.selected

    def get_model_rankings(self, regime: Optional[str] = None) -> List[Dict]:
        """Get model rankings"""
        return self.bandit.get_rankings(ArmType.MODEL, regime)


class StrategySelector:
    """
    Trading strategy selector using bandit.
    """

    def __init__(self, bandit: Optional[MultiarmedBandit] = None):
        self.bandit = bandit or get_bandit()

        # Available strategies
        self.available_strategies = [
            'momentum_breakout',    # Buy breakouts on momentum
            'mean_reversion',       # Buy dips, sell rips
            'trend_following',      # Follow established trends
            'scalping',             # Quick in-and-out trades
            'swing_trading',        # Multi-day holds
            'warrior_small_cap'     # Warrior Trading small cap strategy
        ]

        # Initialize arms
        for strategy in self.available_strategies:
            self.bandit.add_arm(strategy, ArmType.STRATEGY)

    def select_strategy(self, regime: Optional[str] = None) -> str:
        """Select best strategy for current conditions"""
        result = self.bandit.select(
            ArmType.STRATEGY,
            candidates=self.available_strategies,
            regime=regime
        )
        return result.selected

    def get_strategy_rankings(self) -> List[Dict]:
        """Get strategy rankings"""
        return self.bandit.get_rankings(ArmType.STRATEGY)


# Global instances
_bandit: Optional[MultiarmedBandit] = None
_symbol_selector: Optional[SymbolSelector] = None
_model_selector: Optional[ModelSelector] = None
_strategy_selector: Optional[StrategySelector] = None


def get_bandit() -> MultiarmedBandit:
    """Get or create global bandit instance"""
    global _bandit
    if _bandit is None:
        _bandit = MultiarmedBandit()
    return _bandit


def get_symbol_selector() -> SymbolSelector:
    """Get or create global symbol selector"""
    global _symbol_selector
    if _symbol_selector is None:
        _symbol_selector = SymbolSelector()
    return _symbol_selector


def get_model_selector() -> ModelSelector:
    """Get or create global model selector"""
    global _model_selector
    if _model_selector is None:
        _model_selector = ModelSelector()
    return _model_selector


def get_strategy_selector() -> StrategySelector:
    """Get or create global strategy selector"""
    global _strategy_selector
    if _strategy_selector is None:
        _strategy_selector = StrategySelector()
    return _strategy_selector


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Multi-Armed Bandit Selector")
    print("=" * 60)

    # Create bandit
    bandit = get_bandit()

    # Add some symbols
    symbols = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT']
    for symbol in symbols:
        bandit.add_arm(symbol, ArmType.SYMBOL)

    # Simulate some trade outcomes
    print("\nSimulating trade outcomes...")
    trades = [
        ('AAPL', 'lightgbm', 'momentum_breakout', 2.5, 'TRENDING_UP'),
        ('AAPL', 'lightgbm', 'momentum_breakout', 1.8, 'TRENDING_UP'),
        ('TSLA', 'ensemble', 'momentum_breakout', -3.2, 'VOLATILE'),
        ('NVDA', 'ensemble', 'trend_following', 4.1, 'TRENDING_UP'),
        ('NVDA', 'lightgbm', 'trend_following', 2.9, 'TRENDING_UP'),
        ('AMD', 'heuristic', 'mean_reversion', -1.5, 'RANGING'),
        ('MSFT', 'lightgbm', 'swing_trading', 1.2, 'TRENDING_UP'),
        ('AAPL', 'ensemble', 'momentum_breakout', 3.1, 'TRENDING_UP'),
        ('TSLA', 'rl_agent', 'scalping', -0.8, 'VOLATILE'),
        ('NVDA', 'ensemble', 'trend_following', 5.2, 'TRENDING_UP'),
    ]

    for symbol, model, strategy, pnl, regime in trades:
        bandit.record_trade_outcome(symbol, model, strategy, pnl, regime)

    # Test selection
    print("\n" + "=" * 60)
    print("Symbol Selection (Thompson Sampling)")
    print("=" * 60)

    for _ in range(3):
        result = bandit.select(ArmType.SYMBOL, regime='TRENDING_UP')
        print(f"Selected: {result.selected} (sample={result.sample_value:.3f}, mean={result.expected_value:.3f})")

    # Rankings
    print("\n" + "=" * 60)
    print("Symbol Rankings")
    print("=" * 60)

    rankings = bandit.get_rankings(ArmType.SYMBOL)
    for i, r in enumerate(rankings, 1):
        print(f"{i}. {r['name']}: mean={r['mean']:.3f}, pulls={r['pulls']}, CI={r['confidence_interval']}")

    # Model rankings
    print("\n" + "=" * 60)
    print("Model Rankings")
    print("=" * 60)

    model_rankings = bandit.get_rankings(ArmType.MODEL)
    for i, r in enumerate(model_rankings, 1):
        print(f"{i}. {r['name']}: mean={r['mean']:.3f}, pulls={r['pulls']}")

    # Strategy rankings
    print("\n" + "=" * 60)
    print("Strategy Rankings")
    print("=" * 60)

    strategy_rankings = bandit.get_rankings(ArmType.STRATEGY)
    for i, r in enumerate(strategy_rankings, 1):
        print(f"{i}. {r['name']}: mean={r['mean']:.3f}, pulls={r['pulls']}")

    # Test selectors
    print("\n" + "=" * 60)
    print("Testing High-Level Selectors")
    print("=" * 60)

    symbol_selector = get_symbol_selector()
    selected_symbols = symbol_selector.select_symbols(n=3, regime='TRENDING_UP')
    print(f"Top 3 symbols for TRENDING_UP: {selected_symbols}")

    model_selector = get_model_selector()
    best_model = model_selector.select_model(regime='TRENDING_UP')
    print(f"Best model for TRENDING_UP: {best_model}")

    strategy_selector = get_strategy_selector()
    best_strategy = strategy_selector.select_strategy(regime='TRENDING_UP')
    print(f"Best strategy for TRENDING_UP: {best_strategy}")

    # Save state
    bandit.save()
    print("\nBandit state saved.")

    # Stats
    print("\n" + "=" * 60)
    print("Bandit Statistics")
    print("=" * 60)
    print(json.dumps(bandit.get_stats(), indent=2))
