"""
Warrior Trading Configuration Loader
Handles loading and validation of configuration settings
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScannerConfig:
    """Scanner configuration settings"""

    enabled: bool
    min_gap_percent: float
    min_rvol: float
    max_float_millions: float
    min_premarket_volume: int
    scan_interval_minutes: int
    max_watchlist_size: int
    min_price: float
    max_price: float
    preferred_float_millions: float
    scan_schedule: Dict[str, str]


@dataclass
class RiskConfig:
    """Risk management configuration"""

    daily_profit_goal: float
    max_loss_per_trade: float
    max_loss_per_day: float
    default_risk_per_trade: float
    min_reward_to_risk: float
    max_reward_to_risk: float
    max_consecutive_losses: int
    max_position_size_dollars: float
    max_concurrent_positions: int
    position_sizing_method: str
    reduce_size_on_losses: bool
    size_reduction_factor: float


@dataclass
class PatternConfig:
    """Pattern detection configuration"""

    enabled_patterns: list
    bull_flag: Dict[str, Any]
    hod_breakout: Dict[str, Any]
    whole_dollar_breakout: Dict[str, Any]
    micro_pullback: Dict[str, Any]
    hammer_reversal: Dict[str, Any]


@dataclass
class ExecutionConfig:
    """Execution configuration"""

    auto_execute: bool
    require_claude_approval: bool
    require_user_approval: bool
    default_order_type: str
    limit_offset_cents: int
    max_slippage_percent: float
    partial_exit_at_2r: float
    move_stop_to_breakeven_at_r: float
    trailing_stop_percent: float
    use_bracket_orders: bool


@dataclass
class ClaudeConfig:
    """Claude AI integration configuration"""

    enabled: bool
    api_key_env_var: str
    model: str
    validate_all_setups: bool
    validate_confidence_threshold: int
    performance_monitoring: bool
    code_guardian: bool
    strategy_optimization: bool


class WarriorConfig:
    """
    Warrior Trading Configuration Manager

    Loads and provides access to all configuration settings
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader

        Args:
            config_path: Path to config file (default: config/warrior_config.json)
        """
        if config_path is None:
            # Default to config/warrior_config.json in project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "warrior_config.json"

        self.config_path = Path(config_path)
        self._raw_config: Dict[str, Any] = {}
        self._load_config()
        self._parse_config()

    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")

            with open(self.config_path, "r") as f:
                self._raw_config = json.load(f)

            logger.info(f"Loaded configuration from {self.config_path}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _parse_config(self):
        """Parse raw config into typed dataclasses"""
        try:
            # Scanner config
            scanner_data = self._raw_config.get("scanner", {})
            self.scanner = ScannerConfig(
                enabled=scanner_data.get("enabled", True),
                min_gap_percent=scanner_data.get("min_gap_percent", 5.0),
                min_rvol=scanner_data.get("min_rvol", 2.0),
                max_float_millions=scanner_data.get("max_float_millions", 50.0),
                min_premarket_volume=scanner_data.get("min_premarket_volume", 50000),
                scan_interval_minutes=scanner_data.get("scan_interval_minutes", 15),
                max_watchlist_size=scanner_data.get("max_watchlist_size", 10),
                min_price=scanner_data.get("min_price", 2.0),
                max_price=scanner_data.get("max_price", 500.0),
                preferred_float_millions=scanner_data.get(
                    "preferred_float_millions", 10.0
                ),
                scan_schedule=scanner_data.get("scan_schedule", {}),
            )

            # Risk config
            risk_data = self._raw_config.get("risk_management", {})
            self.risk = RiskConfig(
                daily_profit_goal=risk_data.get("daily_profit_goal", 200.0),
                max_loss_per_trade=risk_data.get("max_loss_per_trade", 50.0),
                max_loss_per_day=risk_data.get("max_loss_per_day", 200.0),
                default_risk_per_trade=risk_data.get("default_risk_per_trade", 100.0),
                min_reward_to_risk=risk_data.get("min_reward_to_risk", 2.0),
                max_reward_to_risk=risk_data.get("max_reward_to_risk", 5.0),
                max_consecutive_losses=risk_data.get("max_consecutive_losses", 3),
                max_position_size_dollars=risk_data.get(
                    "max_position_size_dollars", 10000.0
                ),
                max_concurrent_positions=risk_data.get("max_concurrent_positions", 3),
                position_sizing_method=risk_data.get(
                    "position_sizing_method", "FIXED_RISK"
                ),
                reduce_size_on_losses=risk_data.get("reduce_size_on_losses", True),
                size_reduction_factor=risk_data.get("size_reduction_factor", 0.5),
            )

            # Pattern config
            pattern_data = self._raw_config.get("pattern_detection", {})
            self.patterns = PatternConfig(
                enabled_patterns=pattern_data.get("enabled_patterns", []),
                bull_flag=pattern_data.get("bull_flag", {}),
                hod_breakout=pattern_data.get("hod_breakout", {}),
                whole_dollar_breakout=pattern_data.get("whole_dollar_breakout", {}),
                micro_pullback=pattern_data.get("micro_pullback", {}),
                hammer_reversal=pattern_data.get("hammer_reversal", {}),
            )

            # Execution config
            exec_data = self._raw_config.get("execution", {})
            self.execution = ExecutionConfig(
                auto_execute=exec_data.get("auto_execute", False),
                require_claude_approval=exec_data.get("require_claude_approval", True),
                require_user_approval=exec_data.get("require_user_approval", True),
                default_order_type=exec_data.get("default_order_type", "LIMIT"),
                limit_offset_cents=exec_data.get("limit_offset_cents", 2),
                max_slippage_percent=exec_data.get("max_slippage_percent", 0.5),
                partial_exit_at_2r=exec_data.get("partial_exit_at_2r", 0.5),
                move_stop_to_breakeven_at_r=exec_data.get(
                    "move_stop_to_breakeven_at_r", 1.0
                ),
                trailing_stop_percent=exec_data.get("trailing_stop_percent", 1.5),
                use_bracket_orders=exec_data.get("use_bracket_orders", True),
            )

            # Claude config
            claude_data = self._raw_config.get("claude_integration", {})
            self.claude = ClaudeConfig(
                enabled=claude_data.get("enabled", False),
                api_key_env_var=claude_data.get("api_key_env_var", "CLAUDE_API_KEY"),
                model=claude_data.get("model", "claude-sonnet-4-20250514"),
                validate_all_setups=claude_data.get("validate_all_setups", False),
                validate_confidence_threshold=claude_data.get(
                    "validate_confidence_threshold", 70
                ),
                performance_monitoring=claude_data.get("performance_monitoring", False),
                code_guardian=claude_data.get("code_guardian", False),
                strategy_optimization=claude_data.get("strategy_optimization", False),
            )

            # Raw data access for other configs
            self.trading_hours = self._raw_config.get("trading_hours", {})
            self.data_sources = self._raw_config.get("data_sources", {})
            self.logging_config = self._raw_config.get("logging", {})
            self.alerts = self._raw_config.get("alerts", {})
            self.backtesting = self._raw_config.get("backtesting", {})
            self.database = self._raw_config.get("database", {})

            logger.info("Configuration parsed successfully")

        except Exception as e:
            logger.error(f"Error parsing configuration: {e}")
            raise

    def get_claude_api_key(self) -> Optional[str]:
        """Get Claude API key from environment variable"""
        if not self.claude.enabled:
            return None

        api_key = os.getenv(self.claude.api_key_env_var)
        if not api_key:
            logger.warning(
                f"Claude integration enabled but {self.claude.api_key_env_var} "
                f"environment variable not set"
            )
        return api_key

    def is_pattern_enabled(self, pattern_name: str) -> bool:
        """Check if a specific pattern is enabled"""
        return pattern_name in self.patterns.enabled_patterns

    def get_pattern_config(self, pattern_name: str) -> Dict[str, Any]:
        """Get configuration for a specific pattern"""
        pattern_map = {
            "BULL_FLAG": self.patterns.bull_flag,
            "HOD_BREAKOUT": self.patterns.hod_breakout,
            "WHOLE_DOLLAR_BREAKOUT": self.patterns.whole_dollar_breakout,
            "MICRO_PULLBACK": self.patterns.micro_pullback,
            "HAMMER_REVERSAL": self.patterns.hammer_reversal,
        }
        return pattern_map.get(pattern_name, {})

    def validate(self) -> bool:
        """
        Validate configuration settings

        Returns:
            True if valid, raises ValueError if invalid
        """
        # Validate risk settings
        if self.risk.max_loss_per_trade > self.risk.max_loss_per_day:
            raise ValueError("max_loss_per_trade cannot exceed max_loss_per_day")

        if self.risk.min_reward_to_risk < 1.0:
            raise ValueError("min_reward_to_risk must be at least 1.0")

        if self.risk.max_concurrent_positions < 1:
            raise ValueError("max_concurrent_positions must be at least 1")

        # Validate scanner settings
        if self.scanner.min_gap_percent < 0:
            raise ValueError("min_gap_percent must be positive")

        if self.scanner.min_rvol < 1.0:
            raise ValueError("min_rvol must be at least 1.0")

        # Validate execution settings
        if (
            self.execution.partial_exit_at_2r < 0
            or self.execution.partial_exit_at_2r > 1
        ):
            raise ValueError("partial_exit_at_2r must be between 0 and 1")

        logger.info("Configuration validation passed")
        return True

    def reload(self):
        """Reload configuration from file"""
        logger.info("Reloading configuration")
        self._load_config()
        self._parse_config()
        self.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return self._raw_config

    def __repr__(self) -> str:
        return f"WarriorConfig(path={self.config_path})"


# Global config instance (singleton pattern)
_config_instance: Optional[WarriorConfig] = None


def get_config(config_path: Optional[str] = None) -> WarriorConfig:
    """
    Get global configuration instance (singleton)

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        WarriorConfig instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = WarriorConfig(config_path)
        _config_instance.validate()

    return _config_instance


def reload_config():
    """Reload global configuration instance"""
    global _config_instance

    if _config_instance is not None:
        _config_instance.reload()
    else:
        _config_instance = WarriorConfig()
        _config_instance.validate()


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config
    config = get_config()

    # Access settings
    print(f"Scanner enabled: {config.scanner.enabled}")
    print(f"Min gap: {config.scanner.min_gap_percent}%")
    print(f"Min RVOL: {config.scanner.min_rvol}")
    print(f"Max float: {config.scanner.max_float_millions}M")
    print(f"\nRisk settings:")
    print(f"  Daily goal: ${config.risk.daily_profit_goal}")
    print(f"  Max loss/trade: ${config.risk.max_loss_per_trade}")
    print(f"  Max loss/day: ${config.risk.max_loss_per_day}")
    print(f"  Min R:R: {config.risk.min_reward_to_risk}:1")
    print(f"\nEnabled patterns: {config.patterns.enabled_patterns}")
    print(f"\nClaude integration: {config.claude.enabled}")

    # Validate
    config.validate()
    print("\n✅ Configuration validated successfully")
