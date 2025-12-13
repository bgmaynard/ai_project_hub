"""
Circuit Breaker - Drawdown Protection System
=============================================
Automatically halts trading when losses exceed thresholds.

WARRIOR TRADING RULE: Live to trade another day!
- Daily loss limit: Stop trading when hit
- Consecutive losses: Take a break after 3 in a row
- Drawdown from peak: Scale back or halt

CIRCUIT BREAKER LEVELS:
1. WARNING (Yellow): Slow down, reduce size
2. CAUTION (Orange): Half position sizes, no new trades
3. HALT (Red): Stop all trading, protect capital

TRIGGERS:
- Daily P&L drops below threshold
- Consecutive losing trades
- Account drawdown from peak
- Win rate drops below minimum
"""

import logging
import json
import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pytz

logger = logging.getLogger(__name__)

# Storage path
BREAKER_FILE = os.path.join(os.path.dirname(__file__), "..", "store", "circuit_breaker.json")


class BreakerLevel(Enum):
    """Circuit breaker levels"""
    NORMAL = "NORMAL"      # All systems go
    WARNING = "WARNING"    # Slow down
    CAUTION = "CAUTION"    # Reduce size, careful
    HALT = "HALT"          # Stop trading!


@dataclass
class BreakerState:
    """Current circuit breaker state"""
    level: str
    reason: str
    triggered_at: str
    daily_pnl: float
    daily_pnl_pct: float
    consecutive_losses: int
    drawdown_pct: float
    trades_today: int
    wins_today: int
    losses_today: int
    can_trade: bool
    size_multiplier: float  # 1.0 = full, 0.5 = half, 0 = none
    recommendations: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DailyStats:
    """Daily trading statistics"""
    date: str
    starting_equity: float
    current_equity: float
    peak_equity: float
    pnl: float
    pnl_pct: float
    trades: int
    wins: int
    losses: int
    consecutive_losses: int
    consecutive_wins: int
    max_drawdown_pct: float
    breaker_triggered: bool
    breaker_level: str

    def to_dict(self) -> Dict:
        return asdict(self)


class CircuitBreaker:
    """
    Portfolio-level risk management and circuit breaker.

    WARRIOR RULE: Protect your capital above all else!
    """

    def __init__(self):
        self.et_tz = pytz.timezone('US/Eastern')

        # ═══════════════════════════════════════════════════════════
        # CONFIGURABLE THRESHOLDS - Adjust based on account size/risk
        # ═══════════════════════════════════════════════════════════

        # Daily loss limits (as % of account)
        self.daily_loss_warning = -1.0     # -1% = warning
        self.daily_loss_caution = -2.0     # -2% = caution
        self.daily_loss_halt = -3.0        # -3% = halt trading

        # Consecutive loss limits
        self.consecutive_loss_warning = 2   # 2 losses = warning
        self.consecutive_loss_caution = 3   # 3 losses = caution
        self.consecutive_loss_halt = 5      # 5 losses = halt

        # Drawdown from peak (intraday)
        self.drawdown_warning = -2.0       # -2% from peak = warning
        self.drawdown_caution = -3.0       # -3% = caution
        self.drawdown_halt = -5.0          # -5% = halt

        # Win rate minimum (after N trades)
        self.min_win_rate = 30.0           # Below 30% = caution
        self.min_trades_for_winrate = 5    # Need at least 5 trades

        # Max trades per day
        self.max_trades_warning = 15       # Warning at 15 trades
        self.max_trades_halt = 25          # Halt at 25 trades

        # ═══════════════════════════════════════════════════════════

        # Current state
        self.current_level = BreakerLevel.NORMAL
        self.daily_stats: Optional[DailyStats] = None
        self.history: List[DailyStats] = []

        # Trade tracking
        self.trades_today: List[Dict] = []
        self.consecutive_losses = 0
        self.consecutive_wins = 0

        # Account tracking
        self.starting_equity = 0.0
        self.current_equity = 0.0
        self.peak_equity = 0.0

        # Load state
        self._load()

        logger.info("CircuitBreaker initialized")

    def _load(self):
        """Load state from disk"""
        try:
            if os.path.exists(BREAKER_FILE):
                with open(BREAKER_FILE, 'r') as f:
                    data = json.load(f)

                self.history = [DailyStats(**d) for d in data.get('history', [])]

                # Load today's stats if exists
                today = date.today().isoformat()
                today_data = data.get('today')
                if today_data and today_data.get('date') == today:
                    self.daily_stats = DailyStats(**today_data)
                    self.starting_equity = self.daily_stats.starting_equity
                    self.current_equity = self.daily_stats.current_equity
                    self.peak_equity = self.daily_stats.peak_equity
                    self.consecutive_losses = self.daily_stats.consecutive_losses

                logger.info(f"Loaded circuit breaker state: {len(self.history)} days history")
        except Exception as e:
            logger.error(f"Error loading circuit breaker state: {e}")

    def _save(self):
        """Save state to disk"""
        try:
            os.makedirs(os.path.dirname(BREAKER_FILE), exist_ok=True)

            data = {
                'history': [d.to_dict() for d in self.history[-30:]],  # Keep 30 days
                'today': self.daily_stats.to_dict() if self.daily_stats else None,
                'last_updated': datetime.now(self.et_tz).isoformat()
            }

            with open(BREAKER_FILE, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving circuit breaker state: {e}")

    def initialize_day(self, account_equity: float):
        """
        Initialize for a new trading day.

        Call this at market open with current account equity.
        """
        today = date.today().isoformat()

        # Check if already initialized today
        if self.daily_stats and self.daily_stats.date == today:
            logger.debug("Day already initialized")
            return

        # Archive yesterday's stats if exists
        if self.daily_stats:
            self.history.append(self.daily_stats)

        # Start fresh
        self.starting_equity = account_equity
        self.current_equity = account_equity
        self.peak_equity = account_equity
        self.trades_today = []
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.current_level = BreakerLevel.NORMAL

        self.daily_stats = DailyStats(
            date=today,
            starting_equity=account_equity,
            current_equity=account_equity,
            peak_equity=account_equity,
            pnl=0.0,
            pnl_pct=0.0,
            trades=0,
            wins=0,
            losses=0,
            consecutive_losses=0,
            consecutive_wins=0,
            max_drawdown_pct=0.0,
            breaker_triggered=False,
            breaker_level="NORMAL"
        )

        self._save()
        logger.info(f"Circuit breaker initialized for {today} with equity ${account_equity:,.2f}")

    def update_equity(self, current_equity: float):
        """Update current equity (call periodically)"""
        if not self.daily_stats:
            self.initialize_day(current_equity)
            return

        self.current_equity = current_equity

        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Update stats
        self.daily_stats.current_equity = current_equity
        self.daily_stats.peak_equity = self.peak_equity
        self.daily_stats.pnl = current_equity - self.starting_equity
        self.daily_stats.pnl_pct = (self.daily_stats.pnl / self.starting_equity * 100) if self.starting_equity > 0 else 0

        # Calculate drawdown
        drawdown_pct = ((current_equity - self.peak_equity) / self.peak_equity * 100) if self.peak_equity > 0 else 0
        if drawdown_pct < self.daily_stats.max_drawdown_pct:
            self.daily_stats.max_drawdown_pct = drawdown_pct

        self._save()

    def record_trade(self, pnl: float, symbol: str = ""):
        """
        Record a completed trade.

        Args:
            pnl: Trade profit/loss
            symbol: Optional symbol for tracking
        """
        if not self.daily_stats:
            logger.warning("Circuit breaker not initialized")
            return

        # Update counters
        self.daily_stats.trades += 1

        if pnl > 0:
            self.daily_stats.wins += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        elif pnl < 0:
            self.daily_stats.losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.daily_stats.consecutive_losses = self.consecutive_losses

        # Track trade
        self.trades_today.append({
            "time": datetime.now(self.et_tz).isoformat(),
            "symbol": symbol,
            "pnl": pnl
        })

        self._save()

        # Check breaker
        self.check_breaker()

    def check_breaker(self) -> BreakerState:
        """
        Check circuit breaker status and return current state.

        Returns BreakerState with level, reason, and recommendations.
        """
        if not self.daily_stats:
            return BreakerState(
                level="NORMAL",
                reason="Not initialized",
                triggered_at="",
                daily_pnl=0,
                daily_pnl_pct=0,
                consecutive_losses=0,
                drawdown_pct=0,
                trades_today=0,
                wins_today=0,
                losses_today=0,
                can_trade=True,
                size_multiplier=1.0,
                recommendations=["Initialize circuit breaker with account equity"]
            )

        level = BreakerLevel.NORMAL
        reasons = []
        recommendations = []

        pnl_pct = self.daily_stats.pnl_pct
        drawdown_pct = ((self.current_equity - self.peak_equity) / self.peak_equity * 100) if self.peak_equity > 0 else 0

        # ═══════════════════════════════════════════════════════════
        # CHECK HALT CONDITIONS (most severe)
        # ═══════════════════════════════════════════════════════════

        if pnl_pct <= self.daily_loss_halt:
            level = BreakerLevel.HALT
            reasons.append(f"Daily loss {pnl_pct:.1f}% exceeds halt threshold ({self.daily_loss_halt}%)")
            recommendations.append("STOP TRADING - Daily loss limit hit")

        if self.consecutive_losses >= self.consecutive_loss_halt:
            level = BreakerLevel.HALT
            reasons.append(f"{self.consecutive_losses} consecutive losses - halt threshold")
            recommendations.append("STOP TRADING - Too many consecutive losses")

        if drawdown_pct <= self.drawdown_halt:
            level = BreakerLevel.HALT
            reasons.append(f"Drawdown {drawdown_pct:.1f}% exceeds halt threshold ({self.drawdown_halt}%)")
            recommendations.append("STOP TRADING - Max drawdown hit")

        if self.daily_stats.trades >= self.max_trades_halt:
            level = BreakerLevel.HALT
            reasons.append(f"{self.daily_stats.trades} trades today - max reached")
            recommendations.append("STOP TRADING - Max trades for day reached")

        # ═══════════════════════════════════════════════════════════
        # CHECK CAUTION CONDITIONS (if not already HALT)
        # ═══════════════════════════════════════════════════════════

        if level != BreakerLevel.HALT:
            if pnl_pct <= self.daily_loss_caution:
                level = BreakerLevel.CAUTION
                reasons.append(f"Daily loss {pnl_pct:.1f}% at caution level")
                recommendations.append("Reduce position sizes by 50%")

            if self.consecutive_losses >= self.consecutive_loss_caution:
                level = BreakerLevel.CAUTION
                reasons.append(f"{self.consecutive_losses} consecutive losses")
                recommendations.append("Take a 15-minute break before next trade")

            if drawdown_pct <= self.drawdown_caution:
                level = BreakerLevel.CAUTION
                reasons.append(f"Drawdown {drawdown_pct:.1f}% at caution level")
                recommendations.append("Tighten stops, reduce risk")

            # Win rate check
            if self.daily_stats.trades >= self.min_trades_for_winrate:
                win_rate = (self.daily_stats.wins / self.daily_stats.trades * 100) if self.daily_stats.trades > 0 else 0
                if win_rate < self.min_win_rate:
                    level = BreakerLevel.CAUTION
                    reasons.append(f"Win rate {win_rate:.0f}% below minimum ({self.min_win_rate}%)")
                    recommendations.append("Review strategy, something may be off")

        # ═══════════════════════════════════════════════════════════
        # CHECK WARNING CONDITIONS (if not already CAUTION or HALT)
        # ═══════════════════════════════════════════════════════════

        if level == BreakerLevel.NORMAL:
            if pnl_pct <= self.daily_loss_warning:
                level = BreakerLevel.WARNING
                reasons.append(f"Daily loss {pnl_pct:.1f}% approaching limit")
                recommendations.append("Be more selective with trades")

            if self.consecutive_losses >= self.consecutive_loss_warning:
                level = BreakerLevel.WARNING
                reasons.append(f"{self.consecutive_losses} losses in a row")
                recommendations.append("Consider taking a short break")

            if drawdown_pct <= self.drawdown_warning:
                level = BreakerLevel.WARNING
                reasons.append(f"Drawdown {drawdown_pct:.1f}% from peak")
                recommendations.append("Tighten stops on open positions")

            if self.daily_stats.trades >= self.max_trades_warning:
                level = BreakerLevel.WARNING
                reasons.append(f"{self.daily_stats.trades} trades today - nearing limit")
                recommendations.append("Be very selective, limit remaining trades")

        # ═══════════════════════════════════════════════════════════
        # DETERMINE CAN_TRADE AND SIZE MULTIPLIER
        # ═══════════════════════════════════════════════════════════

        if level == BreakerLevel.HALT:
            can_trade = False
            size_multiplier = 0.0
        elif level == BreakerLevel.CAUTION:
            can_trade = True
            size_multiplier = 0.5
        elif level == BreakerLevel.WARNING:
            can_trade = True
            size_multiplier = 0.75
        else:
            can_trade = True
            size_multiplier = 1.0

        # Update state
        self.current_level = level
        self.daily_stats.breaker_level = level.value

        if level != BreakerLevel.NORMAL and not self.daily_stats.breaker_triggered:
            self.daily_stats.breaker_triggered = True
            logger.warning(f"CIRCUIT BREAKER TRIGGERED: {level.value} - {', '.join(reasons)}")

        self._save()

        # Default recommendations
        if not recommendations:
            if level == BreakerLevel.NORMAL:
                recommendations = ["Trading normally", "Continue following your strategy"]

        return BreakerState(
            level=level.value,
            reason="; ".join(reasons) if reasons else "All systems normal",
            triggered_at=datetime.now(self.et_tz).isoformat() if level != BreakerLevel.NORMAL else "",
            daily_pnl=round(self.daily_stats.pnl, 2),
            daily_pnl_pct=round(pnl_pct, 2),
            consecutive_losses=self.consecutive_losses,
            drawdown_pct=round(drawdown_pct, 2),
            trades_today=self.daily_stats.trades,
            wins_today=self.daily_stats.wins,
            losses_today=self.daily_stats.losses,
            can_trade=can_trade,
            size_multiplier=size_multiplier,
            recommendations=recommendations
        )

    def can_open_trade(self) -> Tuple[bool, str, float]:
        """
        Quick check: Can we open a new trade?

        Returns (can_trade, reason, size_multiplier)
        """
        state = self.check_breaker()
        return state.can_trade, state.reason, state.size_multiplier

    def reset_breaker(self):
        """Manually reset the circuit breaker (use with caution!)"""
        self.current_level = BreakerLevel.NORMAL
        self.consecutive_losses = 0

        if self.daily_stats:
            self.daily_stats.breaker_triggered = False
            self.daily_stats.breaker_level = "NORMAL"
            self.daily_stats.consecutive_losses = 0

        self._save()
        logger.warning("Circuit breaker manually reset!")

    def get_daily_summary(self) -> Dict:
        """Get summary of today's trading"""
        if not self.daily_stats:
            return {"message": "Not initialized"}

        state = self.check_breaker()

        return {
            "date": self.daily_stats.date,
            "starting_equity": self.starting_equity,
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "pnl": self.daily_stats.pnl,
            "pnl_pct": self.daily_stats.pnl_pct,
            "trades": self.daily_stats.trades,
            "wins": self.daily_stats.wins,
            "losses": self.daily_stats.losses,
            "win_rate": round(self.daily_stats.wins / self.daily_stats.trades * 100, 1) if self.daily_stats.trades > 0 else 0,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "max_drawdown_pct": self.daily_stats.max_drawdown_pct,
            "breaker_state": state.to_dict()
        }

    def get_history(self, days: int = 7) -> List[Dict]:
        """Get historical daily stats"""
        return [d.to_dict() for d in self.history[-days:]]

    def update_thresholds(self, **kwargs):
        """Update threshold settings"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated threshold: {key} = {value}")


# Singleton instance
_circuit_breaker: Optional[CircuitBreaker] = None


def get_circuit_breaker() -> CircuitBreaker:
    """Get or create the circuit breaker singleton"""
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = CircuitBreaker()
    return _circuit_breaker
