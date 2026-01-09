"""
Trading Decision Engine
======================

Converts AlphaFusion predictions into trading decisions with:
- Kelly sizing
- Risk management (position size, daily limits, stop loss)
- Market vs Limit order logic based on fill probability
- Adaptive trailing stops

Author: AI Trading Bot Team
Version: 2.0
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    TRAIL_STOP = "TRAIL_STOP"


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class TradingSignal:
    """Trading signal with sizing and order type"""

    timestamp: float
    symbol: str
    side: Side
    order_type: OrderType
    quantity: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_amount: Optional[float] = None
    confidence: float = 0.0
    expected_edge: float = 0.0
    reason: str = ""


@dataclass
class Position:
    """Current position"""

    symbol: str
    quantity: int  # Positive = long, Negative = short
    entry_price: float
    current_price: float
    timestamp: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trail_stop: Optional[float] = None

    @property
    def side(self) -> Side:
        return Side.BUY if self.quantity > 0 else Side.SELL

    @property
    def market_value(self) -> float:
        return abs(self.quantity) * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.quantity > 0:  # Long
            return self.quantity * (self.current_price - self.entry_price)
        else:  # Short
            return abs(self.quantity) * (self.entry_price - self.current_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.quantity > 0:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100


@dataclass
class RiskLimits:
    """Risk management parameters with 3-5-7 strategy"""

    # Position sizing
    max_position_size_usd: float = 10000.0  # Max $ per position
    max_position_size_pct: float = 0.10  # Max 10% of account per position
    max_total_exposure_pct: float = 0.50  # Max 50% of account in positions
    max_positions: int = 5  # Max concurrent positions

    # 3-5-7 Risk Management Strategy
    max_risk_per_trade_pct: float = 0.03  # 3% max risk per trade
    daily_loss_limit_pct: float = 0.05  # 5% max daily loss
    weekly_loss_limit_pct: float = 0.07  # 7% max weekly loss

    # Legacy USD limits (override percentage if set)
    daily_loss_limit_usd: Optional[float] = None
    weekly_loss_limit_usd: Optional[float] = None

    # Trading thresholds
    min_probability_threshold: float = 0.55  # Min p_final to trade
    max_spread_pct: float = 0.005  # Max 0.5% spread

    # Exit management
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    trailing_stop_pct: float = 0.015  # 1.5% trailing stop


class TradingEngine:
    """
    Trading decision engine with risk management

    Uses AlphaFusion predictions to make trading decisions
    """

    def __init__(self, account_size: float, risk_limits: Optional[RiskLimits] = None):

        self.account_size = account_size
        self.risk_limits = risk_limits or RiskLimits()

        # Current state
        self.positions: Dict[str, Position] = {}

        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_start_time = time.time()
        self.trades_today = 0

        # Weekly tracking (3-5-7 strategy)
        self.weekly_pnl = 0.0
        self.weekly_start_time = time.time()
        self.trades_this_week = 0

        # Trading enabled flag
        self.trading_enabled = True

    def evaluate_signal(
        self,
        symbol: str,
        p_final: float,
        reliability: float,
        bid: float,
        ask: float,
        spread_pct: float,
        expected_slippage: float = 0.0,
        fill_probability: float = 0.8,
    ) -> Optional[TradingSignal]:
        """
        Evaluate if we should trade based on AlphaFusion prediction

        Returns:
            TradingSignal if should trade, None otherwise
        """

        # Check if trading is enabled
        if not self.trading_enabled:
            return None

        # Check 3-5-7 risk limits (daily & weekly loss limits)
        if self._check_risk_limits():
            return None

        # Gate: probability threshold
        if p_final < self.risk_limits.min_probability_threshold:
            return None

        # Gate: spread cap
        if spread_pct > self.risk_limits.max_spread_pct:
            return None

        # Gate: already have position in this symbol
        if symbol in self.positions:
            return None  # Don't add to existing position for now

        # Gate: max positions
        if len(self.positions) >= self.risk_limits.max_positions:
            return None

        # Determine side
        mid = (bid + ask) / 2.0
        side = Side.BUY if p_final > 0.5 else Side.SELL

        # Calculate expected edge
        edge = abs(p_final - 0.5) * 2.0  # Map [0.5, 1.0] to [0, 1.0]

        # Kelly sizing (fractional Kelly)
        kelly_fraction = self._calculate_kelly_size(
            p_win=p_final if side == Side.BUY else (1 - p_final),
            edge=edge,
            reliability=reliability,
        )

        # Position size in USD
        position_size_usd = kelly_fraction * self.account_size

        # Apply 3-5-7 Rule: 3% max risk per trade
        # Risk per trade = position_size * stop_loss_pct
        # Therefore: max_position_size = (account * 0.03) / stop_loss_pct
        max_risk_usd = self.account_size * self.risk_limits.max_risk_per_trade_pct
        max_position_from_risk = max_risk_usd / self.risk_limits.stop_loss_pct

        # Apply all position limits (including 3% risk limit)
        position_size_usd = min(
            position_size_usd,
            max_position_from_risk,  # 3% max risk per trade
            self.risk_limits.max_position_size_usd,
            self.account_size * self.risk_limits.max_position_size_pct,
        )

        # Check total exposure
        current_exposure = sum(pos.market_value for pos in self.positions.values())
        available_exposure = (
            self.account_size * self.risk_limits.max_total_exposure_pct
            - current_exposure
        )

        if position_size_usd > available_exposure:
            position_size_usd = available_exposure

        if position_size_usd < mid * 10:  # Minimum 10 shares
            return None

        # Calculate quantity
        quantity = int(position_size_usd / mid)
        if quantity == 0:
            return None

        # Decide order type: Market vs Limit
        order_type, limit_price = self._decide_order_type(
            side=side,
            bid=bid,
            ask=ask,
            fill_probability=fill_probability,
            expected_slippage=expected_slippage,
        )

        # Create signal
        signal = TradingSignal(
            timestamp=time.time(),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            confidence=p_final,
            expected_edge=edge,
            reason=f"p_final={p_final:.3f}, edge={edge:.3f}, reliability={reliability:.3f}",
        )

        return signal

    def add_position(self, symbol: str, quantity: int, entry_price: float):
        """Add new position after fill"""
        timestamp = time.time()

        # Calculate stop loss and take profit
        if quantity > 0:  # Long
            stop_loss = entry_price * (1 - self.risk_limits.stop_loss_pct)
            take_profit = entry_price * (1 + self.risk_limits.take_profit_pct)
            trail_stop = entry_price * (1 - self.risk_limits.trailing_stop_pct)
        else:  # Short
            stop_loss = entry_price * (1 + self.risk_limits.stop_loss_pct)
            take_profit = entry_price * (1 - self.risk_limits.take_profit_pct)
            trail_stop = entry_price * (1 + self.risk_limits.trailing_stop_pct)

        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            timestamp=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trail_stop=trail_stop,
        )

        self.positions[symbol] = position

    def update_position(self, symbol: str, current_price: float):
        """Update position with current market price"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        position.current_price = current_price

        # Update trailing stop
        if position.quantity > 0:  # Long
            # Trail stop up as price rises
            new_trail = current_price * (1 - self.risk_limits.trailing_stop_pct)
            if position.trail_stop is None or new_trail > position.trail_stop:
                position.trail_stop = new_trail
        else:  # Short
            # Trail stop down as price falls
            new_trail = current_price * (1 + self.risk_limits.trailing_stop_pct)
            if position.trail_stop is None or new_trail < position.trail_stop:
                position.trail_stop = new_trail

    def check_exits(self, symbol: str, current_price: float) -> Optional[TradingSignal]:
        """Check if position should be closed"""
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        position.current_price = current_price

        exit_reason = None

        if position.quantity > 0:  # Long position
            # Stop loss hit
            if position.stop_loss and current_price <= position.stop_loss:
                exit_reason = "stop_loss"

            # Take profit hit
            elif position.take_profit and current_price >= position.take_profit:
                exit_reason = "take_profit"

            # Trailing stop hit
            elif position.trail_stop and current_price <= position.trail_stop:
                exit_reason = "trail_stop"

        else:  # Short position
            # Stop loss hit
            if position.stop_loss and current_price >= position.stop_loss:
                exit_reason = "stop_loss"

            # Take profit hit
            elif position.take_profit and current_price <= position.take_profit:
                exit_reason = "take_profit"

            # Trailing stop hit
            elif position.trail_stop and current_price >= position.trail_stop:
                exit_reason = "trail_stop"

        if exit_reason:
            # Create exit signal
            signal = TradingSignal(
                timestamp=time.time(),
                symbol=symbol,
                side=Side.SELL if position.quantity > 0 else Side.BUY,
                order_type=OrderType.MARKET,
                quantity=abs(position.quantity),
                reason=f"Exit: {exit_reason}, P&L: ${position.unrealized_pnl:.2f}",
            )

            return signal

        return None

    def close_position(self, symbol: str, exit_price: float):
        """Close position and update P&L"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        pnl = position.unrealized_pnl

        # Update daily and weekly P&L (3-5-7 tracking)
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.trades_today += 1
        self.trades_this_week += 1

        # Remove position
        del self.positions[symbol]

        # Check if we hit risk limits (3-5-7 strategy)
        self._check_risk_limits()

    def _calculate_kelly_size(
        self, p_win: float, edge: float, reliability: float
    ) -> float:
        """
        Calculate Kelly fraction

        Kelly = (p_win * (1 + edge) - (1 - p_win)) / edge

        We use fractional Kelly (e.g., 0.25x) for safety
        """
        if edge < 0.001:
            return 0.0

        # Kelly formula
        kelly = (p_win * (1 + edge) - (1 - p_win)) / edge

        # Apply reliability discount
        kelly *= reliability

        # Use fractional Kelly (0.25x) for safety
        kelly *= 0.25

        # Clip to reasonable range
        return max(0.0, min(kelly, 0.20))  # Max 20% of account

    def _decide_order_type(
        self,
        side: Side,
        bid: float,
        ask: float,
        fill_probability: float,
        expected_slippage: float,
    ) -> tuple[OrderType, Optional[float]]:
        """
        Decide between market and limit order

        Returns:
            (order_type, limit_price)
        """
        # If fill probability is high enough, use limit order at touch
        if fill_probability >= 0.7:
            if side == Side.BUY:
                return OrderType.LIMIT, ask  # Buy at ask
            else:
                return OrderType.LIMIT, bid  # Sell at bid

        # Otherwise use market order
        return OrderType.MARKET, None

    def _check_risk_limits(self) -> bool:
        """
        Check if risk limits hit (3-5-7 strategy)
        - 3% max risk per trade (enforced in evaluate_signal)
        - 5% max daily loss
        - 7% max weekly loss

        Returns:
            True if should stop trading
        """
        current_time = time.time()

        # Reset daily counters at start of new day
        if current_time - self.daily_start_time > 86400:  # 24 hours
            self.daily_pnl = 0.0
            self.daily_start_time = current_time
            self.trades_today = 0
            # Don't re-enable if weekly limit is hit
            if not self._is_weekly_limit_hit():
                self.trading_enabled = True

        # Reset weekly counters at start of new week (7 days)
        if current_time - self.weekly_start_time > 604800:  # 7 days
            self.weekly_pnl = 0.0
            self.weekly_start_time = current_time
            self.trades_this_week = 0
            self.trading_enabled = True

        # Check 5% daily loss limit
        if self.risk_limits.daily_loss_limit_usd is not None:
            # Use absolute USD limit if set
            if self.daily_pnl < -self.risk_limits.daily_loss_limit_usd:
                self.trading_enabled = False
                return True
        else:
            # Use percentage limit (5% from 3-5-7 strategy)
            daily_loss_limit = self.account_size * self.risk_limits.daily_loss_limit_pct
            if self.daily_pnl < -daily_loss_limit:
                self.trading_enabled = False
                return True

        # Check 7% weekly loss limit
        if self.risk_limits.weekly_loss_limit_usd is not None:
            # Use absolute USD limit if set
            if self.weekly_pnl < -self.risk_limits.weekly_loss_limit_usd:
                self.trading_enabled = False
                return True
        else:
            # Use percentage limit (7% from 3-5-7 strategy)
            weekly_loss_limit = (
                self.account_size * self.risk_limits.weekly_loss_limit_pct
            )
            if self.weekly_pnl < -weekly_loss_limit:
                self.trading_enabled = False
                return True

        return False

    def _is_weekly_limit_hit(self) -> bool:
        """Check if weekly loss limit is currently exceeded"""
        if self.risk_limits.weekly_loss_limit_usd is not None:
            return self.weekly_pnl < -self.risk_limits.weekly_loss_limit_usd
        else:
            weekly_loss_limit = (
                self.account_size * self.risk_limits.weekly_loss_limit_pct
            )
            return self.weekly_pnl < -weekly_loss_limit

    def get_stats(self) -> Dict:
        """Get current trading statistics"""
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_exposure = sum(pos.market_value for pos in self.positions.values())
        exposure_pct = (
            (total_exposure / self.account_size) * 100 if self.account_size > 0 else 0
        )

        # Calculate 3-5-7 risk limits
        daily_loss_limit = (
            self.risk_limits.daily_loss_limit_usd
            if self.risk_limits.daily_loss_limit_usd is not None
            else self.account_size * self.risk_limits.daily_loss_limit_pct
        )
        weekly_loss_limit = (
            self.risk_limits.weekly_loss_limit_usd
            if self.risk_limits.weekly_loss_limit_usd is not None
            else self.account_size * self.risk_limits.weekly_loss_limit_pct
        )

        return {
            "trading_enabled": self.trading_enabled,
            "account_size": self.account_size,
            "open_positions": len(self.positions),
            "total_exposure_usd": total_exposure,
            "total_exposure_pct": exposure_pct,
            "unrealized_pnl": total_pnl,
            # Daily tracking
            "daily_pnl": self.daily_pnl,
            "trades_today": self.trades_today,
            "daily_loss_limit": daily_loss_limit,
            "daily_pnl_pct": (
                (self.daily_pnl / self.account_size * 100)
                if self.account_size > 0
                else 0
            ),
            # Weekly tracking (3-5-7 strategy)
            "weekly_pnl": self.weekly_pnl,
            "trades_this_week": self.trades_this_week,
            "weekly_loss_limit": weekly_loss_limit,
            "weekly_pnl_pct": (
                (self.weekly_pnl / self.account_size * 100)
                if self.account_size > 0
                else 0
            ),
            # 3-5-7 Risk Management
            "risk_357_enabled": True,
            "max_risk_per_trade_pct": self.risk_limits.max_risk_per_trade_pct * 100,
            "daily_loss_limit_pct": self.risk_limits.daily_loss_limit_pct * 100,
            "weekly_loss_limit_pct": self.risk_limits.weekly_loss_limit_pct * 100,
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
                    "trail_stop": pos.trail_stop,
                }
                for symbol, pos in self.positions.items()
            },
        }


if __name__ == "__main__":
    # Demo usage
    print("Trading Engine - Demo")
    print("=" * 60)

    # Create engine with $50k account
    engine = TradingEngine(
        account_size=50000.0,
        risk_limits=RiskLimits(
            max_position_size_usd=5000.0,
            daily_loss_limit_usd=500.0,
            min_probability_threshold=0.60,
        ),
    )

    # Evaluate a signal
    signal = engine.evaluate_signal(
        symbol="AAPL",
        p_final=0.72,  # 72% probability of up move
        reliability=0.85,
        bid=150.00,
        ask=150.05,
        spread_pct=0.0003,  # 0.03% spread
        expected_slippage=0.02,
        fill_probability=0.8,
    )

    if signal:
        print(f"\n✓ Trade Signal Generated:")
        print(f"  Symbol: {signal.symbol}")
        print(f"  Side: {signal.side.value}")
        print(f"  Order Type: {signal.order_type.value}")
        print(f"  Quantity: {signal.quantity}")
        print(
            f"  Limit Price: ${signal.limit_price:.2f}"
            if signal.limit_price
            else "  Market Order"
        )
        print(f"  Confidence: {signal.confidence:.3f}")
        print(f"  Expected Edge: {signal.expected_edge:.3f}")
        print(f"  Reason: {signal.reason}")

        # Simulate position entry
        engine.add_position(
            symbol=signal.symbol,
            quantity=signal.quantity if signal.side == Side.BUY else -signal.quantity,
            entry_price=signal.limit_price or 150.02,
        )

        print(f"\n✓ Position Opened:")
        stats = engine.get_stats()
        print(f"  Open Positions: {stats['open_positions']}")
        print(
            f"  Total Exposure: ${stats['total_exposure_usd']:.2f} ({stats['total_exposure_pct']:.1f}%)"
        )

        # Check exits at higher price
        exit_signal = engine.check_exits("AAPL", 156.00)  # +4% profit
        if exit_signal:
            print(f"\n✓ Exit Signal Generated:")
            print(f"  {exit_signal.reason}")
    else:
        print("\n✗ No trade signal generated")

    print(f"\nEngine Stats:")
    for key, value in engine.get_stats().items():
        if key != "positions":
            print(f"  {key}: {value}")
