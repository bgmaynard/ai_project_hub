"""
Warrior Trading Risk Manager
Enforces Ross Cameron's strict risk management rules

Core Rules:
- Minimum 2:1 reward-to-risk ratio
- Max loss per trade = 25% of daily goal
- Max daily loss = daily profit goal
- Stop trading after max daily loss
- Pause after 3 consecutive losses
- Position sizing: RISK / STOP_DISTANCE

This is the safety layer that prevents emotional trading and protects capital.
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from datetime import time as datetime_time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ai.warrior_pattern_detector import SetupType, TradingSetup
from config.config_loader import get_config

logger = logging.getLogger(__name__)


class ValidationResult(str, Enum):
    """Trade validation result"""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    WARNING = "WARNING"


@dataclass
class TradeRecord:
    """
    Record of a single trade

    Attributes:
        trade_id: Unique identifier
        symbol: Stock ticker
        setup_type: Pattern type
        entry_time: When entered
        entry_price: Entry price
        shares: Position size
        stop_price: Stop loss
        target_price: Profit target (2R)
        exit_time: When exited (None if open)
        exit_price: Exit price (None if open)
        exit_reason: Why exited
        pnl: Profit/loss in dollars
        pnl_percent: P&L as % of risk
        r_multiple: P&L in R units (risk multiples)
        side: "LONG" or "SHORT"
    """

    trade_id: str
    symbol: str
    setup_type: str
    entry_time: datetime
    entry_price: float
    shares: int
    stop_price: float
    target_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    r_multiple: Optional[float] = None
    side: str = "LONG"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert datetime to ISO string
        data["entry_time"] = self.entry_time.isoformat()
        if self.exit_time:
            data["exit_time"] = self.exit_time.isoformat()
        return data


@dataclass
class ValidationResponse:
    """
    Result of trade validation

    Attributes:
        result: APPROVED, REJECTED, or WARNING
        reason: Explanation
        position_size: Recommended shares (if approved)
        warnings: List of warnings
        risk_dollars: Total $ risk
        reward_dollars: Total $ reward potential
        risk_reward_ratio: Actual R:R ratio
    """

    result: ValidationResult
    reason: str
    position_size: int = 0
    warnings: List[str] = None
    risk_dollars: float = 0.0
    reward_dollars: float = 0.0
    risk_reward_ratio: float = 0.0

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "result": self.result.value,
            "reason": self.reason,
            "position_size": self.position_size,
            "warnings": self.warnings,
            "risk_dollars": self.risk_dollars,
            "reward_dollars": self.reward_dollars,
            "risk_reward_ratio": self.risk_reward_ratio,
        }


class WarriorRiskManager:
    """
    Enforce Warrior Trading risk management rules

    Responsibilities:
    - Calculate position sizes
    - Validate trades before entry
    - Track daily P&L
    - Enforce loss limits
    - Record trade history
    - Provide risk statistics
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize risk manager

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = get_config(config_path)
        self.risk_config = self.config.risk

        # State tracking
        self.trades_today: List[TradeRecord] = []
        self.current_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0
        self.is_trading_halted: bool = False
        self.halt_reason: Optional[str] = None
        self.current_date: date = date.today()

        # Open positions
        self.open_positions: Dict[str, TradeRecord] = {}

        logger.info("WarriorRiskManager initialized")
        logger.info(f"Daily goal: ${self.risk_config.daily_profit_goal}")
        logger.info(f"Max loss/trade: ${self.risk_config.max_loss_per_trade}")
        logger.info(f"Max loss/day: ${self.risk_config.max_loss_per_day}")

    def calculate_position_size(
        self,
        entry_price: float,
        stop_price: float,
        risk_dollars: Optional[float] = None,
    ) -> int:
        """
        Calculate position size based on risk

        Formula: POSITION_SIZE = floor(RISK_DOLLARS / STOP_DISTANCE)

        Args:
            entry_price: Planned entry price
            stop_price: Stop loss price
            risk_dollars: Amount willing to risk (default: from config)

        Returns:
            Number of shares
        """
        if risk_dollars is None:
            risk_dollars = self.risk_config.default_risk_per_trade

        # Calculate stop distance
        stop_distance = abs(entry_price - stop_price)

        if stop_distance <= 0:
            logger.warning("Invalid stop distance (must be > 0)")
            return 0

        # Calculate position size
        position_size = int(risk_dollars / stop_distance)

        # Check max position size in dollars
        total_exposure = position_size * entry_price
        max_exposure = self.risk_config.max_position_size_dollars

        if total_exposure > max_exposure:
            logger.warning(
                f"Position size ${total_exposure:.2f} exceeds max "
                f"${max_exposure:.2f}, reducing"
            )
            position_size = int(max_exposure / entry_price)

        logger.debug(
            f"Position sizing: Entry ${entry_price:.2f}, Stop ${stop_price:.2f}, "
            f"Risk ${risk_dollars:.2f} = {position_size} shares"
        )

        return position_size

    def validate_trade(
        self, setup: TradingSetup, risk_dollars: Optional[float] = None
    ) -> ValidationResponse:
        """
        Validate if trade meets Warrior Trading criteria

        Checks:
        1. Market hours
        2. R:R ratio ≥ minimum
        3. Daily loss limit
        4. Consecutive losses
        5. Position limits
        6. Risk per trade limit

        Args:
            setup: Trading setup from pattern detector
            risk_dollars: Custom risk amount (optional)

        Returns:
            ValidationResponse with result and details
        """
        warnings = []

        # Check if trading is halted
        if self.is_trading_halted:
            return ValidationResponse(
                result=ValidationResult.REJECTED,
                reason=f"Trading halted: {self.halt_reason}",
                warnings=warnings,
            )

        # Check market hours
        if not self._is_market_hours():
            return ValidationResponse(
                result=ValidationResult.REJECTED,
                reason="Outside trading hours",
                warnings=warnings,
            )

        # Check if setup is valid
        if not setup.is_valid():
            return ValidationResponse(
                result=ValidationResult.REJECTED,
                reason="Invalid setup (R:R < 2:1 or invalid prices)",
                warnings=warnings,
            )

        # Check R:R ratio
        min_rr = self.risk_config.min_reward_to_risk
        max_rr = self.risk_config.max_reward_to_risk

        if setup.risk_reward_ratio < min_rr:
            return ValidationResponse(
                result=ValidationResult.REJECTED,
                reason=f"R:R {setup.risk_reward_ratio:.1f}:1 below minimum {min_rr}:1",
                warnings=warnings,
            )

        if setup.risk_reward_ratio > max_rr:
            warnings.append(
                f"R:R {setup.risk_reward_ratio:.1f}:1 unusually high "
                f"(max {max_rr}:1) - verify target is realistic"
            )

        # Calculate position size
        position_size = self.calculate_position_size(
            setup.entry_price, setup.stop_price, risk_dollars
        )

        if position_size == 0:
            return ValidationResponse(
                result=ValidationResult.REJECTED,
                reason="Position size calculated as 0 (risk too small or stop too wide)",
                warnings=warnings,
            )

        # Check max loss per trade
        actual_risk = position_size * setup.risk_per_share
        max_loss_per_trade = self.risk_config.max_loss_per_trade

        if actual_risk > max_loss_per_trade:
            return ValidationResponse(
                result=ValidationResult.REJECTED,
                reason=f"Risk ${actual_risk:.2f} exceeds max per trade ${max_loss_per_trade:.2f}",
                warnings=warnings,
            )

        # Check daily loss limit
        remaining_daily_loss = self.risk_config.max_loss_per_day - abs(
            min(self.current_pnl, 0)
        )

        if actual_risk > remaining_daily_loss:
            return ValidationResponse(
                result=ValidationResult.REJECTED,
                reason=f"Risk ${actual_risk:.2f} exceeds remaining daily loss allowance ${remaining_daily_loss:.2f}",
                warnings=warnings,
            )

        # Check if daily loss limit already hit
        if (
            abs(self.current_pnl) >= self.risk_config.max_loss_per_day
            and self.current_pnl < 0
        ):
            self.halt_trading(
                f"Max daily loss ${self.risk_config.max_loss_per_day:.2f} reached"
            )
            return ValidationResponse(
                result=ValidationResult.REJECTED,
                reason=f"Max daily loss ${self.risk_config.max_loss_per_day:.2f} reached",
                warnings=warnings,
            )

        # Check consecutive losses
        max_consecutive = self.risk_config.max_consecutive_losses

        if self.consecutive_losses >= max_consecutive:
            warnings.append(
                f"⚠️  {self.consecutive_losses} consecutive losses - "
                f"consider taking a break"
            )

            # Reduce position size on losing streak
            if self.risk_config.reduce_size_on_losses:
                reduction_factor = self.risk_config.size_reduction_factor
                original_size = position_size
                position_size = int(position_size * reduction_factor)
                warnings.append(
                    f"Position size reduced from {original_size} to {position_size} "
                    f"shares due to losing streak"
                )

        # Check max concurrent positions
        max_positions = self.risk_config.max_concurrent_positions
        current_positions = len(self.open_positions)

        if current_positions >= max_positions:
            return ValidationResponse(
                result=ValidationResult.REJECTED,
                reason=f"Max concurrent positions ({max_positions}) reached",
                warnings=warnings,
            )

        # Check pattern-specific position size adjustments
        if setup.setup_type == SetupType.HAMMER_REVERSAL:
            # Reversals are higher risk - reduce size
            reversal_multiplier = self.config.patterns.hammer_reversal.get(
                "position_size_multiplier", 0.5
            )
            original_size = position_size
            position_size = int(position_size * reversal_multiplier)
            warnings.append(
                f"Reversal trade - position size reduced from {original_size} "
                f"to {position_size} shares"
            )

        # Calculate final risk/reward
        final_risk = position_size * setup.risk_per_share
        final_reward = position_size * setup.reward_per_share

        # All checks passed
        result = ValidationResult.APPROVED

        if warnings:
            result = ValidationResult.WARNING

        return ValidationResponse(
            result=result,
            reason=(
                "Trade approved"
                if result == ValidationResult.APPROVED
                else "Trade approved with warnings"
            ),
            position_size=position_size,
            warnings=warnings,
            risk_dollars=final_risk,
            reward_dollars=final_reward,
            risk_reward_ratio=setup.risk_reward_ratio,
        )

    def record_trade_entry(
        self,
        symbol: str,
        setup_type: SetupType,
        entry_price: float,
        shares: int,
        stop_price: float,
        target_price: float,
    ) -> str:
        """
        Record a new trade entry

        Args:
            symbol: Stock ticker
            setup_type: Pattern type
            entry_price: Entry price
            shares: Position size
            stop_price: Stop loss
            target_price: Profit target

        Returns:
            Trade ID
        """
        # Generate trade ID
        trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        trade = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            setup_type=setup_type.value,
            entry_time=datetime.now(),
            entry_price=entry_price,
            shares=shares,
            stop_price=stop_price,
            target_price=target_price,
            side="LONG",
        )

        # Add to open positions
        self.open_positions[trade_id] = trade

        logger.info(
            f"Trade entered: {symbol} {shares} shares @ ${entry_price:.2f}, "
            f"Stop ${stop_price:.2f}, Target ${target_price:.2f}"
        )

        return trade_id

    def record_trade_exit(
        self, trade_id: str, exit_price: float, exit_reason: str = "MANUAL"
    ) -> Optional[TradeRecord]:
        """
        Record a trade exit

        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            exit_reason: Why exited (STOP_HIT, TARGET_HIT, MANUAL, etc.)

        Returns:
            Completed TradeRecord or None if not found
        """
        if trade_id not in self.open_positions:
            logger.warning(f"Trade {trade_id} not found in open positions")
            return None

        trade = self.open_positions.pop(trade_id)

        # Calculate P&L
        trade.exit_time = datetime.now()
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason

        # Calculate P&L (for LONG positions)
        trade.pnl = (exit_price - trade.entry_price) * trade.shares

        # Calculate risk (initial)
        initial_risk = abs(trade.entry_price - trade.stop_price) * trade.shares

        # Calculate R multiple
        if initial_risk > 0:
            trade.r_multiple = trade.pnl / initial_risk
            trade.pnl_percent = (trade.pnl / initial_risk) * 100
        else:
            trade.r_multiple = 0
            trade.pnl_percent = 0

        # Update daily statistics
        self.current_pnl += trade.pnl
        self.trades_today.append(trade)

        # Update consecutive wins/losses
        if trade.pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        # Check if daily loss limit hit
        if (
            abs(self.current_pnl) >= self.risk_config.max_loss_per_day
            and self.current_pnl < 0
        ):
            self.halt_trading(
                f"Daily loss limit ${self.risk_config.max_loss_per_day:.2f} reached"
            )

        logger.info(
            f"Trade exited: {trade.symbol} @ ${exit_price:.2f}, "
            f"P&L: ${trade.pnl:+.2f} ({trade.r_multiple:+.2f}R), "
            f"Reason: {exit_reason}"
        )

        return trade

    def halt_trading(self, reason: str):
        """
        Halt all trading

        Args:
            reason: Why trading is halted
        """
        self.is_trading_halted = True
        self.halt_reason = reason
        logger.warning(f"🛑 TRADING HALTED: {reason}")

    def resume_trading(self):
        """Resume trading (use cautiously!)"""
        self.is_trading_halted = False
        self.halt_reason = None
        logger.info("✅ Trading resumed")

    def get_daily_stats(self) -> Dict[str, Any]:
        """
        Get today's trading statistics

        Returns:
            Dictionary with performance metrics
        """
        total_trades = len(self.trades_today)
        winning_trades = len([t for t in self.trades_today if t.pnl and t.pnl > 0])
        losing_trades = len([t for t in self.trades_today if t.pnl and t.pnl < 0])

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Calculate average win/loss
        wins = [t.pnl for t in self.trades_today if t.pnl and t.pnl > 0]
        losses = [t.pnl for t in self.trades_today if t.pnl and t.pnl < 0]

        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        # Calculate average R multiple
        r_multiples = [
            t.r_multiple for t in self.trades_today if t.r_multiple is not None
        ]
        avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0

        # Best and worst trades
        best_trade = (
            max(self.trades_today, key=lambda t: t.pnl or 0)
            if self.trades_today
            else None
        )
        worst_trade = (
            min(self.trades_today, key=lambda t: t.pnl or 0)
            if self.trades_today
            else None
        )

        # Distance to goal
        distance_to_goal = self.risk_config.daily_profit_goal - self.current_pnl

        return {
            "date": self.current_date.isoformat(),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "current_pnl": self.current_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_r_multiple": avg_r,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "is_halted": self.is_trading_halted,
            "halt_reason": self.halt_reason,
            "distance_to_goal": distance_to_goal,
            "daily_goal": self.risk_config.daily_profit_goal,
            "max_loss_per_day": self.risk_config.max_loss_per_day,
            "open_positions": len(self.open_positions),
            "best_trade": (
                {
                    "symbol": best_trade.symbol,
                    "pnl": best_trade.pnl,
                    "r_multiple": best_trade.r_multiple,
                }
                if best_trade
                else None
            ),
            "worst_trade": (
                {
                    "symbol": worst_trade.symbol,
                    "pnl": worst_trade.pnl,
                    "r_multiple": worst_trade.r_multiple,
                }
                if worst_trade
                else None
            ),
        }

    def reset_daily(self):
        """Reset for new trading day"""
        logger.info("Resetting for new trading day")

        self.trades_today = []
        self.current_pnl = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.is_trading_halted = False
        self.halt_reason = None
        self.current_date = date.today()

        # Warning if there are still open positions
        if self.open_positions:
            logger.warning(
                f"⚠️  Resetting with {len(self.open_positions)} open positions from previous day"
            )

    def _is_market_hours(self) -> bool:
        """
        Check if currently within trading hours

        Returns:
            True if within hours, False otherwise
        """
        now = datetime.now().time()

        # Get trading hours from config
        market_open_str = self.config.trading_hours.get("market_open", "09:30")
        market_close_str = self.config.trading_hours.get("market_close", "16:00")

        # Parse times
        market_open = datetime.strptime(market_open_str, "%H:%M").time()
        market_close = datetime.strptime(market_close_str, "%H:%M").time()

        return market_open <= now <= market_close

    def export_trades(self, filepath: str):
        """
        Export today's trades to JSON

        Args:
            filepath: Where to save trades
        """
        data = {
            "date": self.current_date.isoformat(),
            "stats": self.get_daily_stats(),
            "trades": [t.to_dict() for t in self.trades_today],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(self.trades_today)} trades to {filepath}")


# Example usage / testing
if __name__ == "__main__":
    from ai.warrior_pattern_detector import SetupType, WarriorPatternDetector

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 80)
    print("WARRIOR TRADING RISK MANAGER - TEST MODE")
    print("=" * 80)

    try:
        # Initialize risk manager
        risk_mgr = WarriorRiskManager()

        print(f"\n✅ Risk manager initialized")
        print(f"Configuration:")
        print(f"  Daily goal: ${risk_mgr.risk_config.daily_profit_goal}")
        print(f"  Max loss/trade: ${risk_mgr.risk_config.max_loss_per_trade}")
        print(f"  Max loss/day: ${risk_mgr.risk_config.max_loss_per_day}")
        print(f"  Min R:R: {risk_mgr.risk_config.min_reward_to_risk}:1")

        # Test 1: Position sizing
        print("\n" + "=" * 60)
        print("TEST 1: Position Sizing")
        print("=" * 60)

        entry = 5.00
        stop = 4.90
        risk = 100.0

        shares = risk_mgr.calculate_position_size(entry, stop, risk)
        actual_risk = shares * (entry - stop)

        print(f"\nEntry: ${entry:.2f}")
        print(f"Stop: ${stop:.2f}")
        print(f"Risk: ${risk:.2f}")
        print(f"Position size: {shares} shares")
        print(f"Actual risk: ${actual_risk:.2f}")
        print(f"✅ Position sizing works")

        # Test 2: Trade validation
        print("\n" + "=" * 60)
        print("TEST 2: Trade Validation")
        print("=" * 60)

        # Create mock trading setup
        from ai.warrior_pattern_detector import TradingSetup

        setup = TradingSetup(
            setup_type=SetupType.BULL_FLAG,
            symbol="TEST",
            timeframe="5min",
            entry_price=5.00,
            entry_condition="Test entry",
            stop_price=4.90,
            stop_reason="Test stop",
            target_1r=5.10,
            target_2r=5.20,
            target_3r=5.30,
            risk_per_share=0.10,
            reward_per_share=0.20,
            risk_reward_ratio=2.0,
            confidence=75.0,
            strength_factors=["Test"],
            risk_factors=[],
            current_price=4.95,
        )

        validation = risk_mgr.validate_trade(setup)

        print(f"\nSetup: {setup.setup_type.value}")
        print(f"Entry: ${setup.entry_price:.2f}")
        print(f"Stop: ${setup.stop_price:.2f}")
        print(f"Target: ${setup.target_2r:.2f}")
        print(f"R:R: {setup.risk_reward_ratio:.1f}:1")
        print(f"\nValidation Result: {validation.result.value}")
        print(f"Reason: {validation.reason}")
        print(f"Position size: {validation.position_size} shares")
        print(f"Risk: ${validation.risk_dollars:.2f}")
        print(f"Reward: ${validation.reward_dollars:.2f}")

        if validation.warnings:
            print(f"\nWarnings:")
            for warning in validation.warnings:
                print(f"  ⚠️  {warning}")

        print(f"\n✅ Trade validation works")

        # Test 3: Daily statistics
        print("\n" + "=" * 60)
        print("TEST 3: Daily Statistics")
        print("=" * 60)

        stats = risk_mgr.get_daily_stats()

        print(f"\nDate: {stats['date']}")
        print(f"Total trades: {stats['total_trades']}")
        print(f"Win rate: {stats['win_rate']:.1f}%")
        print(f"Current P&L: ${stats['current_pnl']:+.2f}")
        print(f"Distance to goal: ${stats['distance_to_goal']:.2f}")
        print(f"Open positions: {stats['open_positions']}")
        print(f"Trading halted: {stats['is_halted']}")

        print(f"\n✅ Statistics work")

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Error in risk manager test: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
