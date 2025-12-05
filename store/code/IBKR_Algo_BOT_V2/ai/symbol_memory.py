"""
Symbol Memory - Per-Ticker Trade Performance Tracker
=====================================================
Tracks historical performance for each symbol to help AI make better decisions.

FEATURES:
- Win/loss rate per symbol
- Average profit/loss per trade
- Best/worst time of day to trade
- Streak tracking (hot/cold symbols)
- Sector performance aggregation

USE CASES:
1. Avoid symbols where AI consistently loses
2. Size up on symbols with high win rate
3. Identify time-of-day patterns
4. Learn which sectors work best for the strategy
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import pytz

logger = logging.getLogger(__name__)

# Storage path
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "..", "store", "symbol_memory.json")


@dataclass
class TradeRecord:
    """Single trade record"""
    symbol: str
    side: str  # BUY or SELL
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_percent: float
    entry_time: str
    exit_time: str
    hold_duration_minutes: int
    prediction_confidence: float = 0.0
    indicators: Dict = field(default_factory=dict)  # MACD, RSI, etc at entry

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeRecord':
        return cls(**data)


@dataclass
class SymbolStats:
    """Aggregated stats for a symbol"""
    symbol: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_pnl_percent: float = 0.0
    win_rate: float = 0.0
    avg_hold_minutes: float = 0.0
    best_hour: int = -1  # Hour of day with best performance
    worst_hour: int = -1
    current_streak: int = 0  # Positive = wins, negative = losses
    max_win_streak: int = 0
    max_loss_streak: int = 0
    last_trade_date: str = ""
    avg_winner_pnl: float = 0.0
    avg_loser_pnl: float = 0.0
    profit_factor: float = 0.0  # Gross profit / Gross loss

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'SymbolStats':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SymbolMemory:
    """
    Persistent memory for per-symbol trade performance.

    Answers questions like:
    - Should I trade AAPL? (check win rate)
    - How much should I size? (based on confidence from history)
    - Is this a hot or cold symbol right now? (streak)
    """

    def __init__(self):
        self.et_tz = pytz.timezone('US/Eastern')

        # Trade history per symbol
        self.trades: Dict[str, List[TradeRecord]] = defaultdict(list)

        # Computed stats per symbol
        self.stats: Dict[str, SymbolStats] = {}

        # Hourly performance tracking
        self.hourly_pnl: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

        # Sector mapping (can be expanded)
        self.sector_map: Dict[str, str] = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
            "AMZN": "Consumer", "TSLA": "Automotive", "NVDA": "Technology",
            "META": "Technology", "AMD": "Technology", "INTC": "Technology",
            "JPM": "Financial", "BAC": "Financial", "GS": "Financial",
            "XOM": "Energy", "CVX": "Energy", "OXY": "Energy",
        }

        # Load existing data
        self._load()

        logger.info(f"SymbolMemory initialized with {len(self.trades)} symbols tracked")

    def _load(self):
        """Load memory from disk"""
        try:
            if os.path.exists(MEMORY_FILE):
                with open(MEMORY_FILE, 'r') as f:
                    data = json.load(f)

                # Load trades
                for symbol, trade_list in data.get('trades', {}).items():
                    self.trades[symbol] = [TradeRecord.from_dict(t) for t in trade_list]

                # Load stats
                for symbol, stat_data in data.get('stats', {}).items():
                    self.stats[symbol] = SymbolStats.from_dict(stat_data)

                # Load hourly data
                self.hourly_pnl = defaultdict(lambda: defaultdict(list))
                for symbol, hours in data.get('hourly_pnl', {}).items():
                    for hour, pnls in hours.items():
                        self.hourly_pnl[symbol][int(hour)] = pnls

                logger.info(f"Loaded memory for {len(self.trades)} symbols")
        except Exception as e:
            logger.error(f"Error loading symbol memory: {e}")

    def _save(self):
        """Save memory to disk"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)

            data = {
                'trades': {
                    symbol: [t.to_dict() for t in trades]
                    for symbol, trades in self.trades.items()
                },
                'stats': {
                    symbol: stats.to_dict()
                    for symbol, stats in self.stats.items()
                },
                'hourly_pnl': {
                    symbol: {str(hour): pnls for hour, pnls in hours.items()}
                    for symbol, hours in self.hourly_pnl.items()
                },
                'last_updated': datetime.now(self.et_tz).isoformat()
            }

            with open(MEMORY_FILE, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug("Symbol memory saved")
        except Exception as e:
            logger.error(f"Error saving symbol memory: {e}")

    def record_trade(self,
                     symbol: str,
                     side: str,
                     entry_price: float,
                     exit_price: float,
                     quantity: int,
                     entry_time: datetime,
                     exit_time: datetime,
                     prediction_confidence: float = 0.0,
                     indicators: Dict = None) -> TradeRecord:
        """
        Record a completed trade.

        Called when a position is closed.
        """
        symbol = symbol.upper()

        # Calculate P&L
        if side.upper() == "BUY":
            pnl = (exit_price - entry_price) * quantity
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
        else:  # SELL/SHORT
            pnl = (entry_price - exit_price) * quantity
            pnl_percent = ((entry_price - exit_price) / entry_price) * 100

        # Calculate hold duration
        hold_duration = int((exit_time - entry_time).total_seconds() / 60)

        # Create trade record
        trade = TradeRecord(
            symbol=symbol,
            side=side.upper(),
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=round(pnl, 2),
            pnl_percent=round(pnl_percent, 2),
            entry_time=entry_time.isoformat(),
            exit_time=exit_time.isoformat(),
            hold_duration_minutes=hold_duration,
            prediction_confidence=prediction_confidence,
            indicators=indicators or {}
        )

        # Store trade
        self.trades[symbol].append(trade)

        # Update hourly tracking
        entry_hour = entry_time.hour
        self.hourly_pnl[symbol][entry_hour].append(pnl_percent)

        # Recalculate stats
        self._update_stats(symbol)

        # Save to disk
        self._save()

        logger.info(f"Recorded trade: {symbol} {side} P&L: ${pnl:.2f} ({pnl_percent:.2f}%)")

        return trade

    def _update_stats(self, symbol: str):
        """Recalculate stats for a symbol"""
        trades = self.trades.get(symbol, [])

        if not trades:
            return

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]

        total_pnl = sum(t.pnl for t in trades)
        avg_pnl = total_pnl / len(trades)
        avg_pnl_percent = sum(t.pnl_percent for t in trades) / len(trades)

        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_hold = sum(t.hold_duration_minutes for t in trades) / len(trades)

        # Calculate streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        temp_streak = 0

        for trade in trades:
            if trade.pnl > 0:
                if temp_streak > 0:
                    temp_streak += 1
                else:
                    temp_streak = 1
                max_win_streak = max(max_win_streak, temp_streak)
            elif trade.pnl < 0:
                if temp_streak < 0:
                    temp_streak -= 1
                else:
                    temp_streak = -1
                max_loss_streak = max(max_loss_streak, abs(temp_streak))

        current_streak = temp_streak

        # Best/worst hour
        hourly = self.hourly_pnl.get(symbol, {})
        if hourly:
            hour_avgs = {h: sum(pnls)/len(pnls) for h, pnls in hourly.items() if pnls}
            if hour_avgs:
                best_hour = max(hour_avgs, key=hour_avgs.get)
                worst_hour = min(hour_avgs, key=hour_avgs.get)
            else:
                best_hour = worst_hour = -1
        else:
            best_hour = worst_hour = -1

        # Average winner/loser
        avg_winner = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loser = sum(t.pnl for t in losses) / len(losses) if losses else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Update stats
        self.stats[symbol] = SymbolStats(
            symbol=symbol,
            total_trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            total_pnl=round(total_pnl, 2),
            avg_pnl=round(avg_pnl, 2),
            avg_pnl_percent=round(avg_pnl_percent, 2),
            win_rate=round(win_rate, 1),
            avg_hold_minutes=round(avg_hold, 1),
            best_hour=best_hour,
            worst_hour=worst_hour,
            current_streak=current_streak,
            max_win_streak=max_win_streak,
            max_loss_streak=max_loss_streak,
            last_trade_date=trades[-1].exit_time if trades else "",
            avg_winner_pnl=round(avg_winner, 2),
            avg_loser_pnl=round(avg_loser, 2),
            profit_factor=round(profit_factor, 2)
        )

    def get_stats(self, symbol: str) -> Optional[SymbolStats]:
        """Get stats for a symbol"""
        return self.stats.get(symbol.upper())

    def get_all_stats(self) -> Dict[str, SymbolStats]:
        """Get stats for all symbols"""
        return self.stats.copy()

    def should_trade(self, symbol: str, min_trades: int = 5, min_win_rate: float = 40.0) -> Tuple[bool, str]:
        """
        Should we trade this symbol based on history?

        Returns (should_trade, reason)
        """
        symbol = symbol.upper()
        stats = self.stats.get(symbol)

        if not stats:
            return True, "No history - OK to trade (new symbol)"

        if stats.total_trades < min_trades:
            return True, f"Only {stats.total_trades} trades - need more data"

        # Check win rate
        if stats.win_rate < min_win_rate:
            return False, f"Low win rate: {stats.win_rate}% (min: {min_win_rate}%)"

        # Check for cold streak
        if stats.current_streak <= -3:
            return False, f"Cold streak: {abs(stats.current_streak)} losses in a row"

        # Check profit factor
        if stats.profit_factor < 0.8 and stats.total_trades >= 10:
            return False, f"Poor profit factor: {stats.profit_factor} (losing money overall)"

        # Hot streak bonus
        if stats.current_streak >= 3:
            return True, f"HOT STREAK: {stats.current_streak} wins! Consider sizing up"

        return True, f"OK to trade: {stats.win_rate}% win rate, PF: {stats.profit_factor}"

    def get_position_size_multiplier(self, symbol: str) -> float:
        """
        Get suggested position size multiplier based on history.

        Returns multiplier (0.5 to 1.5):
        - 0.5 = Half size (poor history)
        - 1.0 = Normal size
        - 1.5 = Increased size (great history)
        """
        symbol = symbol.upper()
        stats = self.stats.get(symbol)

        if not stats or stats.total_trades < 5:
            return 1.0  # Default for new symbols

        multiplier = 1.0

        # Win rate adjustment
        if stats.win_rate >= 70:
            multiplier += 0.3
        elif stats.win_rate >= 60:
            multiplier += 0.15
        elif stats.win_rate < 45:
            multiplier -= 0.3
        elif stats.win_rate < 50:
            multiplier -= 0.15

        # Streak adjustment
        if stats.current_streak >= 3:
            multiplier += 0.2  # Hot
        elif stats.current_streak <= -2:
            multiplier -= 0.2  # Cold

        # Profit factor adjustment
        if stats.profit_factor >= 2.0:
            multiplier += 0.2
        elif stats.profit_factor < 1.0:
            multiplier -= 0.2

        # Clamp to range
        return max(0.5, min(1.5, multiplier))

    def get_best_symbols(self, min_trades: int = 5, top_n: int = 10) -> List[Tuple[str, SymbolStats]]:
        """Get top performing symbols"""
        qualified = [
            (symbol, stats) for symbol, stats in self.stats.items()
            if stats.total_trades >= min_trades
        ]

        # Sort by profit factor, then win rate
        qualified.sort(key=lambda x: (x[1].profit_factor, x[1].win_rate), reverse=True)

        return qualified[:top_n]

    def get_worst_symbols(self, min_trades: int = 5, top_n: int = 10) -> List[Tuple[str, SymbolStats]]:
        """Get worst performing symbols (avoid these!)"""
        qualified = [
            (symbol, stats) for symbol, stats in self.stats.items()
            if stats.total_trades >= min_trades
        ]

        # Sort by profit factor ascending
        qualified.sort(key=lambda x: x[1].profit_factor)

        return qualified[:top_n]

    def get_sector_performance(self) -> Dict[str, Dict]:
        """Get aggregated performance by sector"""
        sectors = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})

        for symbol, stats in self.stats.items():
            sector = self.sector_map.get(symbol, "Unknown")
            sectors[sector]["trades"] += stats.total_trades
            sectors[sector]["wins"] += stats.wins
            sectors[sector]["pnl"] += stats.total_pnl

        # Calculate win rates
        for sector, data in sectors.items():
            if data["trades"] > 0:
                data["win_rate"] = round(data["wins"] / data["trades"] * 100, 1)
            else:
                data["win_rate"] = 0

        return dict(sectors)

    def get_time_of_day_analysis(self, symbol: str = None) -> Dict[int, Dict]:
        """
        Get performance by hour of day.

        If symbol provided, returns for that symbol.
        Otherwise, aggregates across all symbols.
        """
        if symbol:
            hourly = self.hourly_pnl.get(symbol.upper(), {})
        else:
            # Aggregate all symbols
            hourly = defaultdict(list)
            for sym, hours in self.hourly_pnl.items():
                for hour, pnls in hours.items():
                    hourly[hour].extend(pnls)

        result = {}
        for hour in range(9, 17):  # Market hours 9 AM - 4 PM
            pnls = hourly.get(hour, [])
            if pnls:
                result[hour] = {
                    "trades": len(pnls),
                    "avg_pnl_percent": round(sum(pnls) / len(pnls), 2),
                    "win_rate": round(len([p for p in pnls if p > 0]) / len(pnls) * 100, 1),
                    "total_pnl_percent": round(sum(pnls), 2)
                }
            else:
                result[hour] = {"trades": 0, "avg_pnl_percent": 0, "win_rate": 0, "total_pnl_percent": 0}

        return result

    def get_recent_trades(self, symbol: str = None, limit: int = 20) -> List[TradeRecord]:
        """Get recent trades, optionally filtered by symbol"""
        if symbol:
            trades = self.trades.get(symbol.upper(), [])
        else:
            # All trades, sorted by exit time
            trades = []
            for sym_trades in self.trades.values():
                trades.extend(sym_trades)
            trades.sort(key=lambda t: t.exit_time, reverse=True)

        return trades[-limit:] if symbol else trades[:limit]

    def get_summary(self) -> Dict:
        """Get overall summary across all symbols"""
        if not self.stats:
            return {
                "total_symbols": 0,
                "total_trades": 0,
                "overall_win_rate": 0,
                "total_pnl": 0,
                "best_symbol": None,
                "worst_symbol": None
            }

        total_trades = sum(s.total_trades for s in self.stats.values())
        total_wins = sum(s.wins for s in self.stats.values())
        total_pnl = sum(s.total_pnl for s in self.stats.values())

        best = max(self.stats.values(), key=lambda s: s.profit_factor) if self.stats else None
        worst = min(self.stats.values(), key=lambda s: s.profit_factor) if self.stats else None

        return {
            "total_symbols": len(self.stats),
            "total_trades": total_trades,
            "overall_win_rate": round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0,
            "total_pnl": round(total_pnl, 2),
            "best_symbol": best.symbol if best else None,
            "best_win_rate": best.win_rate if best else 0,
            "worst_symbol": worst.symbol if worst else None,
            "worst_win_rate": worst.win_rate if worst else 0
        }

    def clear_symbol(self, symbol: str):
        """Clear all data for a symbol"""
        symbol = symbol.upper()
        self.trades.pop(symbol, None)
        self.stats.pop(symbol, None)
        self.hourly_pnl.pop(symbol, None)
        self._save()
        logger.info(f"Cleared memory for {symbol}")

    def clear_all(self):
        """Clear all memory (use with caution!)"""
        self.trades.clear()
        self.stats.clear()
        self.hourly_pnl.clear()
        self._save()
        logger.warning("All symbol memory cleared!")


# Singleton instance
_symbol_memory: Optional[SymbolMemory] = None


def get_symbol_memory() -> SymbolMemory:
    """Get or create the symbol memory singleton"""
    global _symbol_memory
    if _symbol_memory is None:
        _symbol_memory = SymbolMemory()
    return _symbol_memory
