"""
Kelly Criterion Position Sizer
Calculates optimal position sizes based on win rate, average win/loss,
account equity, and risk parameters.

Part of the Next-Gen AI Logic Blueprint.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    """Result of position sizing calculation"""

    symbol: str
    recommended_shares: int
    recommended_value: float
    kelly_fraction: float  # Raw Kelly %
    adjusted_fraction: float  # After safety adjustments
    position_percent: float  # % of portfolio
    confidence_factor: float  # 0-1 AI confidence impact
    risk_per_share: float  # Expected risk per share
    expected_value: float  # Expected profit
    reasoning: str
    timestamp: str


@dataclass
class TradeStats:
    """Historical trade statistics for Kelly calculation"""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float  # Average winning trade %
    avg_loss: float  # Average losing trade %
    profit_factor: float  # Total wins / Total losses
    sharpe_estimate: float  # Rough Sharpe ratio
    max_drawdown: float  # Maximum drawdown %
    period_days: int  # Days of data


class KellySizer:
    """
    Position sizer using Kelly Criterion with safety adjustments.

    Kelly Formula: f* = (bp - q) / b
    Where:
        f* = fraction of bankroll to wager
        b = net odds received on the wager (win/loss ratio)
        p = probability of winning
        q = probability of losing (1 - p)

    We use "half-Kelly" or less for safety, and cap maximum position sizes.
    """

    def __init__(self, account_value: float = 100000):
        self.account_value = account_value
        self.trade_history: List[Dict] = []
        self.stats_cache: Optional[TradeStats] = None
        self.cache_time: Optional[datetime] = None
        self.cache_duration = timedelta(minutes=30)

        # Safety parameters
        self.kelly_multiplier = 0.25  # Use quarter-Kelly (very conservative)
        self.max_position_percent = 0.10  # Max 10% of portfolio per position
        self.min_position_percent = 0.02  # Min 2% to make it worth trading
        self.max_risk_percent = 0.02  # Max 2% of portfolio at risk per trade

        # Default stats when no history available
        self.default_win_rate = 0.55
        self.default_win_loss_ratio = 1.3

        # File paths
        self.history_path = Path("store/trade_history")
        self.history_path.mkdir(parents=True, exist_ok=True)

        # Load existing trade history
        self._load_history()

        logger.info(f"KellySizer initialized with account value: ${account_value:,.2f}")

    def update_account_value(self, value: float):
        """Update the account value for sizing calculations"""
        self.account_value = value
        logger.debug(f"Account value updated to ${value:,.2f}")

    def _load_history(self):
        """Load trade history from file"""
        try:
            history_file = self.history_path / "completed_trades.json"
            if history_file.exists():
                with open(history_file, "r") as f:
                    self.trade_history = json.load(f)
                logger.info(f"Loaded {len(self.trade_history)} historical trades")
        except Exception as e:
            logger.warning(f"Could not load trade history: {e}")
            self.trade_history = []

    def _save_history(self):
        """Save trade history to file"""
        try:
            history_file = self.history_path / "completed_trades.json"
            with open(history_file, "w") as f:
                json.dump(self.trade_history[-500:], f, indent=2)  # Keep last 500
        except Exception as e:
            logger.warning(f"Could not save trade history: {e}")

    def record_trade(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        quantity: int,
        side: str = "long",
    ):
        """
        Record a completed trade for statistics tracking.

        Args:
            symbol: Stock symbol
            entry_price: Entry price per share
            exit_price: Exit price per share
            quantity: Number of shares
            side: "long" or "short"
        """
        if side == "long":
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_percent = ((entry_price - exit_price) / entry_price) * 100

        trade = {
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "side": side,
            "pnl_percent": pnl_percent,
            "pnl_dollars": (
                (exit_price - entry_price) * quantity
                if side == "long"
                else (entry_price - exit_price) * quantity
            ),
            "is_winner": pnl_percent > 0,
            "timestamp": datetime.now().isoformat(),
        }

        self.trade_history.append(trade)
        self.stats_cache = None  # Invalidate cache
        self._save_history()

        logger.info(
            f"Recorded trade: {symbol} {side} {'+' if pnl_percent > 0 else ''}{pnl_percent:.2f}%"
        )

    def calculate_stats(self, lookback_days: int = 30) -> TradeStats:
        """
        Calculate trading statistics from history.

        Args:
            lookback_days: Number of days to look back

        Returns:
            TradeStats object with calculated metrics
        """
        # Check cache
        if (
            self.stats_cache is not None
            and self.cache_time is not None
            and datetime.now() - self.cache_time < self.cache_duration
        ):
            return self.stats_cache

        # Filter trades by date
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent_trades = [
            t
            for t in self.trade_history
            if datetime.fromisoformat(t["timestamp"]) >= cutoff
        ]

        if len(recent_trades) < 5:
            # Not enough data, use defaults
            stats = TradeStats(
                total_trades=len(recent_trades),
                winning_trades=0,
                losing_trades=0,
                win_rate=self.default_win_rate,
                avg_win=2.0,  # Default 2% avg win
                avg_loss=1.5,  # Default 1.5% avg loss
                profit_factor=self.default_win_loss_ratio,
                sharpe_estimate=0.5,
                max_drawdown=5.0,
                period_days=lookback_days,
            )
            logger.debug("Using default stats (insufficient trade history)")
            return stats

        # Calculate actual stats
        winners = [t for t in recent_trades if t["is_winner"]]
        losers = [t for t in recent_trades if not t["is_winner"]]

        win_rate = len(winners) / len(recent_trades) if recent_trades else 0.5

        avg_win = (
            sum(t["pnl_percent"] for t in winners) / len(winners) if winners else 2.0
        )
        avg_loss = (
            abs(sum(t["pnl_percent"] for t in losers) / len(losers)) if losers else 1.5
        )

        total_wins = sum(t["pnl_percent"] for t in winners) if winners else 0
        total_losses = abs(sum(t["pnl_percent"] for t in losers)) if losers else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 1.0

        # Calculate max drawdown
        cumulative = []
        running_pnl = 0
        peak = 0
        max_dd = 0

        for t in recent_trades:
            running_pnl += t["pnl_percent"]
            peak = max(peak, running_pnl)
            dd = peak - running_pnl
            max_dd = max(max_dd, dd)

        # Rough Sharpe estimate
        if recent_trades:
            returns = [t["pnl_percent"] for t in recent_trades]
            import numpy as np

            mean_return = np.mean(returns)
            std_return = np.std(returns) if len(returns) > 1 else 1
            sharpe = mean_return / std_return if std_return > 0 else 0
        else:
            sharpe = 0

        stats = TradeStats(
            total_trades=len(recent_trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_estimate=float(sharpe),
            max_drawdown=max_dd,
            period_days=lookback_days,
        )

        # Cache the stats
        self.stats_cache = stats
        self.cache_time = datetime.now()

        logger.debug(
            f"Calculated stats: win_rate={win_rate:.1%}, avg_win={avg_win:.2f}%, avg_loss={avg_loss:.2f}%"
        )

        return stats

    def calculate_kelly(self, stats: TradeStats = None) -> float:
        """
        Calculate raw Kelly fraction.

        Kelly: f* = (bp - q) / b
        Where b = avg_win/avg_loss, p = win_rate, q = 1-p
        """
        if stats is None:
            stats = self.calculate_stats()

        p = stats.win_rate
        q = 1 - p
        b = stats.avg_win / stats.avg_loss if stats.avg_loss > 0 else 1

        kelly = (b * p - q) / b if b > 0 else 0

        # Kelly can be negative (don't bet)
        kelly = max(0, kelly)

        return kelly

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        confidence: float = 0.5,
        stop_loss_percent: float = 2.0,
        volatility_factor: float = 1.0,
        regime_multiplier: float = 1.0,
    ) -> SizingResult:
        """
        Calculate optimal position size for a trade.

        Args:
            symbol: Stock symbol
            price: Current stock price
            confidence: AI confidence (0-1)
            stop_loss_percent: Stop loss as % from entry
            volatility_factor: Market volatility adjustment (higher = smaller size)
            regime_multiplier: Regime-based adjustment from regime_classifier

        Returns:
            SizingResult with recommended position size
        """
        # Get trade statistics
        stats = self.calculate_stats()

        # Calculate raw Kelly
        raw_kelly = self.calculate_kelly(stats)

        # Apply fractional Kelly (conservative)
        base_fraction = raw_kelly * self.kelly_multiplier

        # Adjust for AI confidence
        # Higher confidence -> closer to base_fraction
        # Lower confidence -> reduce size
        confidence_factor = 0.5 + (confidence * 0.5)  # Range: 0.5 to 1.0

        # Adjust for volatility
        # Higher volatility -> smaller position
        volatility_adjustment = 1.0 / max(0.5, volatility_factor)

        # Apply regime multiplier (from regime_classifier)
        regime_adjustment = regime_multiplier

        # Calculate adjusted fraction
        adjusted_fraction = (
            base_fraction
            * confidence_factor
            * volatility_adjustment
            * regime_adjustment
        )

        # Apply min/max constraints
        adjusted_fraction = max(
            self.min_position_percent, min(self.max_position_percent, adjusted_fraction)
        )

        # Calculate position value
        position_value = self.account_value * adjusted_fraction

        # Risk-based cap: max 2% of portfolio at risk
        risk_per_share = price * (stop_loss_percent / 100)
        max_shares_by_risk = (
            self.account_value * self.max_risk_percent
        ) / risk_per_share

        # Calculate shares
        shares_by_value = int(position_value / price)
        shares_by_risk = int(max_shares_by_risk)

        recommended_shares = min(shares_by_value, shares_by_risk)
        recommended_shares = max(1, recommended_shares)  # At least 1 share

        # Final position value
        final_value = recommended_shares * price
        position_percent = (final_value / self.account_value) * 100

        # Expected value calculation
        expected_win = stats.win_rate * stats.avg_win
        expected_loss = (1 - stats.win_rate) * stats.avg_loss
        expected_value_pct = expected_win - expected_loss
        expected_value = final_value * (expected_value_pct / 100)

        # Build reasoning
        reasoning_parts = [
            f"Kelly: {raw_kelly:.1%} raw, {adjusted_fraction:.1%} adjusted",
            f"Confidence factor: {confidence_factor:.2f}",
            f"Win rate: {stats.win_rate:.1%}",
            f"Profit factor: {stats.profit_factor:.2f}",
        ]

        if regime_adjustment != 1.0:
            reasoning_parts.append(f"Regime adj: {regime_adjustment:.2f}x")
        if volatility_factor != 1.0:
            reasoning_parts.append(f"Vol adj: {volatility_adjustment:.2f}x")
        if shares_by_risk < shares_by_value:
            reasoning_parts.append("Risk-limited")

        result = SizingResult(
            symbol=symbol,
            recommended_shares=recommended_shares,
            recommended_value=final_value,
            kelly_fraction=raw_kelly,
            adjusted_fraction=adjusted_fraction,
            position_percent=position_percent,
            confidence_factor=confidence_factor,
            risk_per_share=risk_per_share,
            expected_value=expected_value,
            reasoning=" | ".join(reasoning_parts),
            timestamp=datetime.now().isoformat(),
        )

        logger.info(
            f"Position size for {symbol}: {recommended_shares} shares (${final_value:,.2f}, {position_percent:.1f}% of portfolio)"
        )

        return result

    def get_stats_summary(self) -> Dict:
        """Get summary of trading statistics"""
        stats = self.calculate_stats()
        kelly = self.calculate_kelly(stats)

        return {
            "total_trades": stats.total_trades,
            "win_rate": f"{stats.win_rate:.1%}",
            "avg_win": f"{stats.avg_win:.2f}%",
            "avg_loss": f"{stats.avg_loss:.2f}%",
            "profit_factor": f"{stats.profit_factor:.2f}",
            "max_drawdown": f"{stats.max_drawdown:.2f}%",
            "raw_kelly": f"{kelly:.1%}",
            "adjusted_kelly": f"{kelly * self.kelly_multiplier:.1%}",
            "sharpe_estimate": f"{stats.sharpe_estimate:.2f}",
            "account_value": f"${self.account_value:,.2f}",
        }


# Singleton instance
_kelly_sizer: Optional[KellySizer] = None


def get_kelly_sizer(account_value: float = None) -> KellySizer:
    """Get or create the Kelly sizer singleton"""
    global _kelly_sizer
    if _kelly_sizer is None:
        _kelly_sizer = KellySizer(account_value or 100000)
    elif account_value:
        _kelly_sizer.update_account_value(account_value)
    return _kelly_sizer
