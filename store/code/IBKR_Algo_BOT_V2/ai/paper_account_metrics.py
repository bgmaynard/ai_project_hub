"""
Paper Account Metrics Module
============================
Provides visibility into paper trading account status.

Tracks:
- Starting balance
- Realized P&L
- Unrealized P&L (open positions)
- Current equity
- Drawdown
- Trade history
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class PaperAccountState:
    """Paper account state snapshot"""
    starting_balance: float = 1000.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    current_equity: float = 1000.0
    high_water_mark: float = 1000.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    open_positions: int = 0
    last_updated: str = ""


class PaperAccountMetrics:
    """
    Track paper trading account metrics in real-time.
    """

    def __init__(self, starting_balance: float = 1000.0):
        self.starting_balance = starting_balance
        self.realized_pnl = 0.0
        self.high_water_mark = starting_balance
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Equity history for charting
        self.equity_history: List[Dict] = []

        # Trade log
        self.trade_log: List[Dict] = []

        # Persistence file
        self.state_file = Path("ai/paper_account_state.json")

        # Load saved state
        self._load_state()

        logger.info(f"Paper Account Metrics initialized: ${starting_balance:.2f} starting balance")

    def _load_state(self):
        """Load state from file if exists"""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                    self.starting_balance = data.get("starting_balance", 1000.0)
                    self.realized_pnl = data.get("realized_pnl", 0.0)
                    self.high_water_mark = data.get("high_water_mark", self.starting_balance)
                    self.max_drawdown = data.get("max_drawdown", 0.0)
                    self.total_trades = data.get("total_trades", 0)
                    self.winning_trades = data.get("winning_trades", 0)
                    self.losing_trades = data.get("losing_trades", 0)
                    self.equity_history = data.get("equity_history", [])[-100:]  # Keep last 100
                    self.trade_log = data.get("trade_log", [])[-50:]  # Keep last 50
                    logger.info(f"Loaded paper account state: ${self.get_current_equity():.2f} equity")
            except Exception as e:
                logger.warning(f"Could not load paper account state: {e}")

    def _save_state(self):
        """Save state to file"""
        try:
            state = {
                "starting_balance": self.starting_balance,
                "realized_pnl": self.realized_pnl,
                "high_water_mark": self.high_water_mark,
                "max_drawdown": self.max_drawdown,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "equity_history": self.equity_history[-100:],
                "trade_log": self.trade_log[-50:],
                "last_saved": datetime.now().isoformat()
            }
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save paper account state: {e}")

    def get_current_equity(self, unrealized_pnl: float = 0.0) -> float:
        """Get current account equity"""
        return self.starting_balance + self.realized_pnl + unrealized_pnl

    def record_trade(self, pnl: float, symbol: str, entry_price: float,
                     exit_price: float, shares: int, hold_time: float):
        """Record a completed trade"""
        self.total_trades += 1
        self.realized_pnl += pnl

        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Update high water mark and drawdown
        current_equity = self.get_current_equity()
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity

        drawdown = (self.high_water_mark - current_equity) / self.high_water_mark * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        # Log trade
        trade_record = {
            "time": datetime.now().isoformat(),
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "shares": shares,
            "pnl": pnl,
            "pnl_percent": ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0,
            "hold_time_seconds": hold_time,
            "equity_after": current_equity
        }
        self.trade_log.append(trade_record)

        # Record equity snapshot
        self.equity_history.append({
            "time": datetime.now().isoformat(),
            "equity": current_equity,
            "trade_num": self.total_trades
        })

        # Save state
        self._save_state()

        logger.info(f"Paper trade recorded: {symbol} ${pnl:+.2f}, equity now ${current_equity:.2f}")

    def sync_from_scalper(self, scalper_stats: dict, open_positions: List[dict]):
        """Sync metrics from HFT scalper stats"""
        # Update from scalper stats
        self.total_trades = scalper_stats.get("total_trades", 0)
        self.winning_trades = scalper_stats.get("wins", 0)
        self.losing_trades = scalper_stats.get("losses", 0)
        self.realized_pnl = scalper_stats.get("total_pnl", 0.0)

        # Update high water mark
        current_equity = self.get_current_equity()
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity

        # Calculate drawdown
        if self.high_water_mark > 0:
            drawdown = (self.high_water_mark - current_equity) / self.high_water_mark * 100
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

        self._save_state()

    def get_unrealized_pnl(self, positions: List[dict], prices: Dict[str, float]) -> float:
        """Calculate unrealized P&L from open positions"""
        unrealized = 0.0
        for pos in positions:
            symbol = pos.get("symbol", "")
            entry_price = pos.get("entry_price", 0)
            shares = pos.get("shares", 0)
            current_price = prices.get(symbol, entry_price)
            unrealized += (current_price - entry_price) * shares
        return unrealized

    def get_account_summary(self, open_positions: List[dict] = None,
                           current_prices: Dict[str, float] = None) -> dict:
        """Get complete account summary"""

        # Calculate unrealized P&L if positions provided
        unrealized_pnl = 0.0
        position_details = []

        if open_positions and current_prices:
            for pos in open_positions:
                symbol = pos.get("symbol", "")
                entry_price = pos.get("entry_price", 0)
                shares = pos.get("shares", 0)
                current_price = current_prices.get(symbol, entry_price)
                pos_pnl = (current_price - entry_price) * shares
                pos_pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                unrealized_pnl += pos_pnl

                position_details.append({
                    "symbol": symbol,
                    "shares": shares,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "pnl": round(pos_pnl, 2),
                    "pnl_percent": round(pos_pnl_pct, 2)
                })

        current_equity = self.get_current_equity(unrealized_pnl)

        # Calculate current drawdown
        current_drawdown = 0.0
        if self.high_water_mark > 0:
            current_drawdown = (self.high_water_mark - current_equity) / self.high_water_mark * 100

        # Win rate
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        # Return on account
        roi = ((current_equity - self.starting_balance) / self.starting_balance * 100) if self.starting_balance > 0 else 0

        return {
            "account": {
                "starting_balance": round(self.starting_balance, 2),
                "realized_pnl": round(self.realized_pnl, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "current_equity": round(current_equity, 2),
                "high_water_mark": round(self.high_water_mark, 2),
                "roi_percent": round(roi, 2)
            },
            "drawdown": {
                "current_drawdown_percent": round(current_drawdown, 2),
                "max_drawdown_percent": round(self.max_drawdown, 2)
            },
            "performance": {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": round(win_rate, 2),
                "open_positions": len(position_details)
            },
            "positions": position_details,
            "equity_history": self.equity_history[-20:],  # Last 20 points
            "recent_trades": self.trade_log[-10:],  # Last 10 trades
            "last_updated": datetime.now().isoformat()
        }

    def reset(self, new_starting_balance: float = None):
        """Reset account metrics"""
        if new_starting_balance:
            self.starting_balance = new_starting_balance
        self.realized_pnl = 0.0
        self.high_water_mark = self.starting_balance
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.equity_history = []
        self.trade_log = []
        self._save_state()
        logger.info(f"Paper account reset to ${self.starting_balance:.2f}")


# Singleton instance
_paper_metrics: Optional[PaperAccountMetrics] = None


def get_paper_metrics() -> PaperAccountMetrics:
    """Get singleton paper account metrics instance"""
    global _paper_metrics
    if _paper_metrics is None:
        _paper_metrics = PaperAccountMetrics(starting_balance=1000.0)
    return _paper_metrics
