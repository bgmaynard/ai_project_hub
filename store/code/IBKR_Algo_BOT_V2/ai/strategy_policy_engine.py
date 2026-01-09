"""
Strategy Policy Engine (SPE)

Dynamically adjusts strategy parameters based on:
- ChronosContext (market regime)
- Rolling performance metrics per strategy
- Signal gating veto logs

Constraints:
- MUST NOT modify SignalContracts
- MUST NOT bypass Signal Gating Engine
- All policy changes logged with timestamp & reason

Default state: fail-safe (no policy override)
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# File paths
POLICY_STATE_FILE = os.path.join(
    os.path.dirname(__file__), "strategy_policy_state.json"
)
POLICY_LOG_FILE = os.path.join(os.path.dirname(__file__), "strategy_policy_log.json")


class PolicyAction(Enum):
    ENABLE = "ENABLE"
    DISABLE = "DISABLE"
    ADJUST_CONFIDENCE = "ADJUST_CONFIDENCE"
    ADJUST_POSITION_SIZE = "ADJUST_POSITION_SIZE"
    RESET_TO_DEFAULT = "RESET_TO_DEFAULT"


@dataclass
class StrategyPolicy:
    """Policy state for a single strategy"""

    strategy_id: str
    enabled: bool = True
    confidence_threshold_override: Optional[float] = None  # None = use contract default
    position_size_multiplier: float = 1.0  # 1.0 = normal, 0.5 = half size, etc.
    reason: str = "default"
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Bounds (cannot be overridden)
    MIN_CONFIDENCE: float = field(default=0.4, repr=False)
    MAX_CONFIDENCE: float = field(default=0.9, repr=False)
    MIN_POSITION_MULT: float = field(default=0.25, repr=False)
    MAX_POSITION_MULT: float = field(default=2.0, repr=False)


@dataclass
class PerformanceMetrics:
    """Rolling performance metrics for a strategy"""

    strategy_id: str
    win_rate: float = 0.0
    profit_factor: float = 1.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    consecutive_losses: int = 0
    last_5_trades: List[str] = field(default_factory=list)  # "W" or "L"
    period_start: str = ""
    period_end: str = ""


@dataclass
class PolicyLogEntry:
    """Audit log entry for policy changes"""

    timestamp: str
    strategy_id: str
    action: str
    old_value: Any
    new_value: Any
    reason: str
    trigger_source: str  # "performance", "regime", "manual", "veto_pattern"
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyEvaluationResult:
    """Result of policy evaluation"""

    strategy_id: str
    enabled: bool
    confidence_override: Optional[float]
    position_multiplier: float
    actions_taken: List[str]
    reasons: List[str]


class StrategyPolicyEngine:
    """
    Dynamic strategy policy adjustment engine.

    Fail-safe: Default state allows all strategies with no overrides.
    """

    def __init__(self):
        self.policies: Dict[str, StrategyPolicy] = {}
        self.performance: Dict[str, PerformanceMetrics] = {}
        self.veto_counts: Dict[str, Dict[str, int]] = (
            {}
        )  # strategy -> {veto_reason: count}
        self.policy_log: List[PolicyLogEntry] = []

        # Policy thresholds (configurable)
        self.config = {
            # Performance-based disabling
            "disable_after_consecutive_losses": 5,
            "disable_below_win_rate": 0.25,
            "disable_below_profit_factor": 0.5,
            # Confidence adjustments
            "increase_confidence_after_losses": 3,  # Increase threshold after N losses
            "confidence_increase_step": 0.05,
            # Position size adjustments
            "reduce_size_after_losses": 2,
            "size_reduction_step": 0.25,
            "increase_size_after_wins": 5,  # Consecutive wins to increase
            "size_increase_step": 0.25,
            # Regime-based adjustments
            "reduce_size_in_volatile": 0.5,  # Multiplier in VOLATILE regime
            "disable_in_trending_down": True,  # Disable LONG strategies
            # Veto pattern detection
            "veto_pattern_threshold": 10,  # Disable after N same-reason vetoes
            # Recovery
            "auto_enable_after_hours": 24,  # Re-enable after N hours
        }

        self._load_state()

    def _load_state(self):
        """Load persisted policy state"""
        try:
            if os.path.exists(POLICY_STATE_FILE):
                with open(POLICY_STATE_FILE, "r") as f:
                    data = json.load(f)
                    for sid, pdata in data.get("policies", {}).items():
                        self.policies[sid] = StrategyPolicy(**pdata)
                    self.veto_counts = data.get("veto_counts", {})
                    logger.info(f"Loaded {len(self.policies)} strategy policies")
        except Exception as e:
            logger.error(f"Failed to load policy state: {e}")

    def _save_state(self):
        """Persist policy state"""
        try:
            data = {
                "policies": {sid: asdict(p) for sid, p in self.policies.items()},
                "veto_counts": self.veto_counts,
                "last_updated": datetime.utcnow().isoformat(),
            }
            with open(POLICY_STATE_FILE, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save policy state: {e}")

    def _log_policy_change(self, entry: PolicyLogEntry):
        """Log policy change for audit"""
        self.policy_log.append(entry)

        # Persist to file
        try:
            log_data = []
            if os.path.exists(POLICY_LOG_FILE):
                with open(POLICY_LOG_FILE, "r") as f:
                    log_data = json.load(f)

            log_data.append(asdict(entry))

            # Keep last 1000 entries
            if len(log_data) > 1000:
                log_data = log_data[-1000:]

            with open(POLICY_LOG_FILE, "w") as f:
                json.dump(log_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to log policy change: {e}")

        logger.info(
            f"Policy change: {entry.strategy_id} - {entry.action} - {entry.reason}"
        )

    def get_policy(self, strategy_id: str) -> StrategyPolicy:
        """Get policy for a strategy (creates default if not exists)"""
        if strategy_id not in self.policies:
            self.policies[strategy_id] = StrategyPolicy(strategy_id=strategy_id)
        return self.policies[strategy_id]

    def get_all_policies(self) -> Dict[str, Dict]:
        """Get all current policies"""
        return {
            "policies": {sid: asdict(p) for sid, p in self.policies.items()},
            "config": self.config,
            "veto_counts": self.veto_counts,
            "last_updated": datetime.utcnow().isoformat(),
        }

    def update_performance(self, strategy_id: str, metrics: PerformanceMetrics):
        """Update rolling performance metrics for a strategy"""
        self.performance[strategy_id] = metrics

    def record_veto(self, strategy_id: str, veto_reason: str):
        """Record a veto for pattern detection"""
        if strategy_id not in self.veto_counts:
            self.veto_counts[strategy_id] = {}

        if veto_reason not in self.veto_counts[strategy_id]:
            self.veto_counts[strategy_id][veto_reason] = 0

        self.veto_counts[strategy_id][veto_reason] += 1
        self._save_state()

    def record_trade_result(self, strategy_id: str, is_win: bool, pnl: float):
        """Record a trade result for performance tracking"""
        if strategy_id not in self.performance:
            self.performance[strategy_id] = PerformanceMetrics(strategy_id=strategy_id)

        metrics = self.performance[strategy_id]
        metrics.total_trades += 1
        metrics.total_pnl += pnl

        if is_win:
            metrics.winning_trades += 1
            metrics.consecutive_losses = 0
            metrics.last_5_trades.append("W")
        else:
            metrics.losing_trades += 1
            metrics.consecutive_losses += 1
            metrics.last_5_trades.append("L")

        # Keep only last 5
        metrics.last_5_trades = metrics.last_5_trades[-5:]

        # Update win rate
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades

    def evaluate(
        self,
        strategy_id: str,
        chronos_context: Optional[Dict] = None,
        force_reevaluate: bool = False,
    ) -> PolicyEvaluationResult:
        """
        Evaluate and potentially adjust policy for a strategy.

        This is the main entry point for policy evaluation.
        """
        policy = self.get_policy(strategy_id)
        metrics = self.performance.get(strategy_id)
        vetos = self.veto_counts.get(strategy_id, {})

        actions_taken = []
        reasons = []

        # Start with current state
        new_enabled = policy.enabled
        new_confidence = policy.confidence_threshold_override
        new_position_mult = policy.position_size_multiplier

        # === Performance-based evaluation ===
        if metrics and metrics.total_trades >= 5:  # Need minimum trades

            # Check consecutive losses
            if (
                metrics.consecutive_losses
                >= self.config["disable_after_consecutive_losses"]
            ):
                if new_enabled:
                    new_enabled = False
                    actions_taken.append(PolicyAction.DISABLE.value)
                    reasons.append(f"Consecutive losses: {metrics.consecutive_losses}")

            # Check win rate
            if metrics.win_rate < self.config["disable_below_win_rate"]:
                if new_enabled:
                    new_enabled = False
                    actions_taken.append(PolicyAction.DISABLE.value)
                    reasons.append(f"Low win rate: {metrics.win_rate:.1%}")

            # Increase confidence after losses
            if (
                metrics.consecutive_losses
                >= self.config["increase_confidence_after_losses"]
            ):
                current_conf = new_confidence or 0.6  # Default
                new_conf = min(
                    current_conf + self.config["confidence_increase_step"],
                    policy.MAX_CONFIDENCE,
                )
                if new_conf != new_confidence:
                    new_confidence = new_conf
                    actions_taken.append(PolicyAction.ADJUST_CONFIDENCE.value)
                    reasons.append(
                        f"Increased confidence after {metrics.consecutive_losses} losses"
                    )

            # Reduce position size after losses
            if metrics.consecutive_losses >= self.config["reduce_size_after_losses"]:
                new_mult = max(
                    new_position_mult - self.config["size_reduction_step"],
                    policy.MIN_POSITION_MULT,
                )
                if new_mult != new_position_mult:
                    new_position_mult = new_mult
                    actions_taken.append(PolicyAction.ADJUST_POSITION_SIZE.value)
                    reasons.append(
                        f"Reduced size after {metrics.consecutive_losses} losses"
                    )

            # Increase size after consecutive wins
            consecutive_wins = len([t for t in metrics.last_5_trades if t == "W"])
            if consecutive_wins >= self.config["increase_size_after_wins"]:
                new_mult = min(
                    new_position_mult + self.config["size_increase_step"],
                    policy.MAX_POSITION_MULT,
                )
                if new_mult != new_position_mult:
                    new_position_mult = new_mult
                    actions_taken.append(PolicyAction.ADJUST_POSITION_SIZE.value)
                    reasons.append(
                        f"Increased size after {consecutive_wins} consecutive wins"
                    )

        # === Regime-based evaluation ===
        if chronos_context:
            regime = chronos_context.get("market_regime", "UNKNOWN")

            # Reduce size in volatile markets
            if regime == "VOLATILE":
                volatile_mult = self.config["reduce_size_in_volatile"]
                if new_position_mult > volatile_mult:
                    new_position_mult = volatile_mult
                    actions_taken.append(PolicyAction.ADJUST_POSITION_SIZE.value)
                    reasons.append(f"Reduced size due to VOLATILE regime")

            # Disable LONG strategies in downtrend (if configured)
            if regime == "TRENDING_DOWN" and self.config["disable_in_trending_down"]:
                # Check if this is a LONG strategy (would need contract info)
                # For now, just add a warning
                pass

        # === Veto pattern detection ===
        for veto_reason, count in vetos.items():
            if count >= self.config["veto_pattern_threshold"]:
                if new_enabled:
                    new_enabled = False
                    actions_taken.append(PolicyAction.DISABLE.value)
                    reasons.append(f"Veto pattern detected: {veto_reason} ({count}x)")

        # === Apply changes if any ===
        if actions_taken:
            old_policy = asdict(policy)

            policy.enabled = new_enabled
            policy.confidence_threshold_override = new_confidence
            policy.position_size_multiplier = new_position_mult
            policy.reason = "; ".join(reasons)
            policy.last_updated = datetime.utcnow().isoformat()

            self.policies[strategy_id] = policy
            self._save_state()

            # Log the change
            self._log_policy_change(
                PolicyLogEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    strategy_id=strategy_id,
                    action=", ".join(actions_taken),
                    old_value=old_policy,
                    new_value=asdict(policy),
                    reason="; ".join(reasons),
                    trigger_source="evaluate",
                    context={
                        "chronos": chronos_context,
                        "metrics": asdict(metrics) if metrics else None,
                        "vetos": vetos,
                    },
                )
            )

        return PolicyEvaluationResult(
            strategy_id=strategy_id,
            enabled=new_enabled,
            confidence_override=new_confidence,
            position_multiplier=new_position_mult,
            actions_taken=actions_taken,
            reasons=reasons,
        )

    def enable_strategy(self, strategy_id: str, reason: str = "manual"):
        """Manually enable a strategy"""
        policy = self.get_policy(strategy_id)
        old_enabled = policy.enabled

        policy.enabled = True
        policy.reason = reason
        policy.last_updated = datetime.utcnow().isoformat()

        self._save_state()
        self._log_policy_change(
            PolicyLogEntry(
                timestamp=datetime.utcnow().isoformat(),
                strategy_id=strategy_id,
                action=PolicyAction.ENABLE.value,
                old_value=old_enabled,
                new_value=True,
                reason=reason,
                trigger_source="manual",
            )
        )

    def disable_strategy(self, strategy_id: str, reason: str = "manual"):
        """Manually disable a strategy"""
        policy = self.get_policy(strategy_id)
        old_enabled = policy.enabled

        policy.enabled = False
        policy.reason = reason
        policy.last_updated = datetime.utcnow().isoformat()

        self._save_state()
        self._log_policy_change(
            PolicyLogEntry(
                timestamp=datetime.utcnow().isoformat(),
                strategy_id=strategy_id,
                action=PolicyAction.DISABLE.value,
                old_value=old_enabled,
                new_value=False,
                reason=reason,
                trigger_source="manual",
            )
        )

    def reset_strategy(self, strategy_id: str, reason: str = "manual reset"):
        """Reset a strategy to default policy"""
        old_policy = (
            asdict(self.get_policy(strategy_id))
            if strategy_id in self.policies
            else None
        )

        self.policies[strategy_id] = StrategyPolicy(
            strategy_id=strategy_id, reason=reason
        )

        # Clear veto counts
        if strategy_id in self.veto_counts:
            del self.veto_counts[strategy_id]

        self._save_state()
        self._log_policy_change(
            PolicyLogEntry(
                timestamp=datetime.utcnow().isoformat(),
                strategy_id=strategy_id,
                action=PolicyAction.RESET_TO_DEFAULT.value,
                old_value=old_policy,
                new_value=asdict(self.policies[strategy_id]),
                reason=reason,
                trigger_source="manual",
            )
        )

    def get_audit_log(
        self, strategy_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict]:
        """Get policy audit log"""
        try:
            if os.path.exists(POLICY_LOG_FILE):
                with open(POLICY_LOG_FILE, "r") as f:
                    log_data = json.load(f)

                if strategy_id:
                    log_data = [
                        e for e in log_data if e.get("strategy_id") == strategy_id
                    ]

                return log_data[-limit:]
        except Exception as e:
            logger.error(f"Failed to read audit log: {e}")

        return []

    def update_config(self, new_config: Dict) -> Dict:
        """Update policy configuration"""
        old_config = self.config.copy()

        for key, value in new_config.items():
            if key in self.config:
                self.config[key] = value

        self._log_policy_change(
            PolicyLogEntry(
                timestamp=datetime.utcnow().isoformat(),
                strategy_id="__CONFIG__",
                action="CONFIG_UPDATE",
                old_value=old_config,
                new_value=self.config,
                reason="Manual config update",
                trigger_source="manual",
            )
        )

        return self.config


# Singleton instance
_engine_instance: Optional[StrategyPolicyEngine] = None


def get_strategy_policy_engine() -> StrategyPolicyEngine:
    """Get singleton instance of Strategy Policy Engine"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = StrategyPolicyEngine()
    return _engine_instance
