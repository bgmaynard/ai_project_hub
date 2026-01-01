"""
Portfolio Analytics Module
Provides real-time P&L tracking, performance metrics, and trade analysis
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a completed or active trade"""

    symbol: str
    side: str  # BUY or SELL
    quantity: int
    entry_price: float
    entry_time: str
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    status: str = "OPEN"  # OPEN, CLOSED, CANCELLED
    order_id: Optional[str] = None
    ai_signal: Optional[str] = None
    ai_confidence: Optional[float] = None


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""

    total_value: float
    cash: float
    positions_value: float
    daily_pnl: float
    daily_pnl_percent: float
    total_pnl: float
    total_pnl_percent: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float


class PortfolioAnalytics:
    """Portfolio analytics and P&L tracking"""

    def __init__(self, data_path: str = "store/portfolio"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.trades_file = self.data_path / "trades.json"
        self.metrics_file = self.data_path / "metrics.json"
        self.trades: List[TradeRecord] = []
        self.load_trades()

    def load_trades(self):
        """Load trade history from file"""
        if self.trades_file.exists():
            try:
                with open(self.trades_file, "r") as f:
                    data = json.load(f)
                    self.trades = [TradeRecord(**t) for t in data]
                logger.info(f"Loaded {len(self.trades)} trades from history")
            except Exception as e:
                logger.error(f"Error loading trades: {e}")
                self.trades = []

    def save_trades(self):
        """Save trade history to file"""
        try:
            with open(self.trades_file, "w") as f:
                json.dump([asdict(t) for t in self.trades], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trades: {e}")

    def record_trade_entry(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        order_id: str = None,
        ai_signal: str = None,
        ai_confidence: float = None,
    ) -> TradeRecord:
        """Record a new trade entry"""
        trade = TradeRecord(
            symbol=symbol.upper(),
            side=side.upper(),
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now().isoformat(),
            order_id=order_id,
            ai_signal=ai_signal,
            ai_confidence=ai_confidence,
            status="OPEN",
        )
        self.trades.append(trade)
        self.save_trades()
        logger.info(f"Recorded trade entry: {side} {quantity} {symbol} @ ${price:.2f}")
        return trade

    def record_trade_exit(
        self, symbol: str, price: float, order_id: str = None
    ) -> Optional[TradeRecord]:
        """Record trade exit and calculate P&L"""
        # Find the open trade for this symbol
        for trade in reversed(self.trades):
            if trade.symbol == symbol.upper() and trade.status == "OPEN":
                trade.exit_price = price
                trade.exit_time = datetime.now().isoformat()
                trade.status = "CLOSED"

                # Calculate P&L
                if trade.side == "BUY":
                    trade.pnl = (price - trade.entry_price) * trade.quantity
                else:  # SELL (short)
                    trade.pnl = (trade.entry_price - price) * trade.quantity

                trade.pnl_percent = (
                    trade.pnl / (trade.entry_price * trade.quantity)
                ) * 100

                self.save_trades()
                logger.info(
                    f"Recorded trade exit: {symbol} @ ${price:.2f}, P&L: ${trade.pnl:.2f} ({trade.pnl_percent:.2f}%)"
                )
                return trade

        logger.warning(f"No open trade found for {symbol}")
        return None

    def get_open_trades(self) -> List[Dict]:
        """Get all open trades"""
        return [asdict(t) for t in self.trades if t.status == "OPEN"]

    def get_closed_trades(self, days: int = 30) -> List[Dict]:
        """Get closed trades from last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        closed = []
        for t in self.trades:
            if t.status == "CLOSED" and t.exit_time:
                exit_dt = datetime.fromisoformat(t.exit_time)
                if exit_dt >= cutoff:
                    closed.append(asdict(t))
        return closed

    def calculate_position_pnl(self, symbol: str, current_price: float) -> Dict:
        """Calculate unrealized P&L for a position"""
        for trade in reversed(self.trades):
            if trade.symbol == symbol.upper() and trade.status == "OPEN":
                if trade.side == "BUY":
                    unrealized_pnl = (
                        current_price - trade.entry_price
                    ) * trade.quantity
                else:
                    unrealized_pnl = (
                        trade.entry_price - current_price
                    ) * trade.quantity

                pnl_percent = (
                    unrealized_pnl / (trade.entry_price * trade.quantity)
                ) * 100

                return {
                    "symbol": symbol,
                    "entry_price": trade.entry_price,
                    "current_price": current_price,
                    "quantity": trade.quantity,
                    "side": trade.side,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_percent": pnl_percent,
                    "entry_time": trade.entry_time,
                }

        return {"symbol": symbol, "error": "No open position found"}

    def calculate_metrics(self, account_value: float = 100000) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        closed_trades = [t for t in self.trades if t.status == "CLOSED"]

        if not closed_trades:
            return {"total_trades": 0, "message": "No completed trades to analyze"}

        # Basic stats
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]

        total_pnl = sum(t.pnl for t in closed_trades)
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0

        # Win rate
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        # Average win/loss
        avg_win = gross_profit / len(winning_trades) if winning_trades else 0
        avg_loss = gross_loss / len(losing_trades) if losing_trades else 0

        # Profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Calculate drawdown
        cumulative_pnl = []
        running_pnl = 0
        peak = 0
        max_drawdown = 0
        for t in sorted(closed_trades, key=lambda x: x.exit_time or ""):
            running_pnl += t.pnl
            cumulative_pnl.append(running_pnl)
            if running_pnl > peak:
                peak = running_pnl
            drawdown = peak - running_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        max_drawdown_percent = (
            (max_drawdown / account_value) * 100 if account_value > 0 else 0
        )

        # Daily P&L (last trading day)
        today = datetime.now().date()
        today_trades = [
            t
            for t in closed_trades
            if t.exit_time and datetime.fromisoformat(t.exit_time).date() == today
        ]
        daily_pnl = sum(t.pnl for t in today_trades)

        # Expectancy
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)

        # Return metrics
        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_percent": round((total_pnl / account_value) * 100, 2),
            "daily_pnl": round(daily_pnl, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": (
                round(profit_factor, 2) if profit_factor != float("inf") else "∞"
            ),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "largest_win": (
                round(max(t.pnl for t in closed_trades), 2) if closed_trades else 0
            ),
            "largest_loss": (
                round(min(t.pnl for t in closed_trades), 2) if closed_trades else 0
            ),
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_percent": round(max_drawdown_percent, 2),
            "expectancy": round(expectancy, 2),
            "account_value": account_value,
        }

    def get_trade_by_symbol(self, symbol: str) -> List[Dict]:
        """Get all trades for a specific symbol"""
        return [asdict(t) for t in self.trades if t.symbol == symbol.upper()]

    def get_ai_performance(self) -> Dict:
        """Analyze AI signal performance"""
        ai_trades = [t for t in self.trades if t.status == "CLOSED" and t.ai_signal]

        if not ai_trades:
            return {"message": "No AI-signaled trades to analyze"}

        # Group by signal type
        signal_stats = {}
        for trade in ai_trades:
            signal = trade.ai_signal
            if signal not in signal_stats:
                signal_stats[signal] = {"trades": 0, "wins": 0, "total_pnl": 0}

            signal_stats[signal]["trades"] += 1
            if trade.pnl > 0:
                signal_stats[signal]["wins"] += 1
            signal_stats[signal]["total_pnl"] += trade.pnl

        # Calculate win rate per signal
        for signal, stats in signal_stats.items():
            stats["win_rate"] = round((stats["wins"] / stats["trades"]) * 100, 2)
            stats["avg_pnl"] = round(stats["total_pnl"] / stats["trades"], 2)

        return {
            "total_ai_trades": len(ai_trades),
            "signal_performance": signal_stats,
            "overall_ai_pnl": round(sum(t.pnl for t in ai_trades), 2),
        }


# Singleton instance
_analytics_instance: Optional[PortfolioAnalytics] = None


def get_portfolio_analytics() -> PortfolioAnalytics:
    """Get or create the portfolio analytics singleton"""
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = PortfolioAnalytics()
    return _analytics_instance
