"""
Trade Analytics Module - TraderVue-style analysis
Comprehensive trade journaling, statistical analysis, and pattern recognition
"""
import sqlite3
import json
import logging
from datetime import datetime, timedelta, date
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent / "store" / "trade_journal.db"


@dataclass
class Trade:
    """Trade record"""
    trade_id: str
    account: str
    symbol: str
    side: str  # BUY or SELL
    quantity: float
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    commission: float = 0.0
    strategy: str = ""
    setup: str = ""
    notes: str = ""
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class TradeAnalytics:
    """Trade analytics and journaling system"""

    def __init__(self):
        self.db_path = DB_PATH
        self._init_database()

    def _init_database(self):
        """Initialize or upgrade database schema"""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create comprehensive trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                account TEXT,
                trade_date DATE,
                symbol TEXT,
                side TEXT,
                quantity REAL,
                entry_price REAL,
                entry_time DATETIME,
                exit_price REAL,
                exit_time DATETIME,
                pnl REAL DEFAULT 0,
                pnl_percent REAL DEFAULT 0,
                commission REAL DEFAULT 0,
                hold_time_minutes INTEGER DEFAULT 0,
                strategy TEXT,
                setup TEXT,
                time_of_day TEXT,
                day_of_week TEXT,
                is_winner INTEGER DEFAULT 0,
                is_scratch INTEGER DEFAULT 0,
                r_multiple REAL DEFAULT 0,
                mae REAL DEFAULT 0,
                mfe REAL DEFAULT 0,
                notes TEXT,
                tags TEXT,
                ai_signal TEXT,
                ai_confidence REAL,
                market_condition TEXT,
                sector TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create daily performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE,
                account TEXT,
                total_trades INTEGER DEFAULT 0,
                winners INTEGER DEFAULT 0,
                losers INTEGER DEFAULT 0,
                scratches INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                gross_pnl REAL DEFAULT 0,
                commissions REAL DEFAULT 0,
                net_pnl REAL DEFAULT 0,
                largest_win REAL DEFAULT 0,
                largest_loss REAL DEFAULT 0,
                avg_win REAL DEFAULT 0,
                avg_loss REAL DEFAULT 0,
                profit_factor REAL DEFAULT 0,
                expectancy REAL DEFAULT 0,
                best_trade_symbol TEXT,
                worst_trade_symbol TEXT,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create strategy performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT,
                setup TEXT,
                total_trades INTEGER DEFAULT 0,
                winners INTEGER DEFAULT 0,
                losers INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                avg_pnl REAL DEFAULT 0,
                profit_factor REAL DEFAULT 0,
                avg_hold_time INTEGER DEFAULT 0,
                best_time_of_day TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(strategy, setup)
            )
        """)

        # Create symbol performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbol_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE,
                total_trades INTEGER DEFAULT 0,
                winners INTEGER DEFAULT 0,
                losers INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                avg_pnl REAL DEFAULT 0,
                best_setup TEXT,
                avg_hold_time INTEGER DEFAULT 0,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create time analysis table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS time_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time_slot TEXT,
                day_of_week TEXT,
                total_trades INTEGER DEFAULT 0,
                winners INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                avg_pnl REAL DEFAULT 0,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(time_slot, day_of_week)
            )
        """)

        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_date ON trade_history(trade_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_symbol ON trade_history(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_account ON trade_history(account)")

        conn.commit()
        conn.close()
        logger.info("Trade analytics database initialized")

    def record_trade(self, trade: Dict) -> bool:
        """Record a trade to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Calculate derived fields
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')

            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
            if isinstance(exit_time, str) and exit_time:
                exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))

            # Calculate hold time
            hold_time = 0
            if entry_time and exit_time:
                hold_time = int((exit_time - entry_time).total_seconds() / 60)

            # Determine time of day
            time_of_day = "unknown"
            if entry_time:
                hour = entry_time.hour
                if hour < 10:
                    time_of_day = "morning_open"
                elif hour < 12:
                    time_of_day = "mid_morning"
                elif hour < 14:
                    time_of_day = "midday"
                elif hour < 16:
                    time_of_day = "afternoon"
                else:
                    time_of_day = "after_hours"

            # Determine winner/loser/scratch
            pnl = trade.get('pnl', 0)
            is_winner = 1 if pnl > 0.50 else 0
            is_scratch = 1 if abs(pnl) <= 0.50 else 0

            # Day of week
            dow = entry_time.strftime('%A') if entry_time else 'Unknown'

            cursor.execute("""
                INSERT OR REPLACE INTO trade_history (
                    trade_id, account, trade_date, symbol, side, quantity,
                    entry_price, entry_time, exit_price, exit_time,
                    pnl, pnl_percent, commission, hold_time_minutes,
                    strategy, setup, time_of_day, day_of_week,
                    is_winner, is_scratch, notes, tags,
                    ai_signal, ai_confidence, market_condition
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.get('trade_id'),
                trade.get('account', ''),
                trade.get('trade_date', date.today().isoformat()),
                trade.get('symbol', ''),
                trade.get('side', ''),
                trade.get('quantity', 0),
                trade.get('entry_price', 0),
                entry_time.isoformat() if entry_time else None,
                trade.get('exit_price'),
                exit_time.isoformat() if exit_time else None,
                pnl,
                trade.get('pnl_percent', 0),
                trade.get('commission', 0),
                hold_time,
                trade.get('strategy', ''),
                trade.get('setup', ''),
                time_of_day,
                dow,
                is_winner,
                is_scratch,
                trade.get('notes', ''),
                json.dumps(trade.get('tags', [])),
                trade.get('ai_signal', ''),
                trade.get('ai_confidence', 0),
                trade.get('market_condition', '')
            ))

            conn.commit()
            conn.close()

            # Update statistics
            self._update_statistics()

            return True
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            return False

    def sync_from_schwab(self, schwab_orders: List[Dict], account: str) -> int:
        """Sync filled orders from Schwab into trade journal"""
        imported = 0

        for order in schwab_orders:
            if order.get('status') != 'FILLED':
                continue

            trade_id = f"{account}-{order.get('order_id')}"

            # Check if already exists
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM trade_history WHERE trade_id = ?", (trade_id,))
            if cursor.fetchone():
                conn.close()
                continue
            conn.close()

            # Create trade record
            trade = {
                'trade_id': trade_id,
                'account': account,
                'symbol': order.get('symbol'),
                'side': order.get('side'),
                'quantity': order.get('filled_qty', order.get('quantity', 0)),
                'entry_price': order.get('price', 0),
                'entry_time': order.get('entered_time'),
                'exit_time': order.get('close_time'),
                'trade_date': datetime.now().date().isoformat()
            }

            if self.record_trade(trade):
                imported += 1

        return imported

    def get_daily_summary(self, target_date: date = None) -> Dict:
        """Get summary statistics for a specific day"""
        if target_date is None:
            target_date = date.today()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as winners,
                SUM(CASE WHEN is_winner = 0 AND is_scratch = 0 THEN 1 ELSE 0 END) as losers,
                SUM(CASE WHEN is_scratch = 1 THEN 1 ELSE 0 END) as scratches,
                SUM(pnl) as total_pnl,
                SUM(commission) as total_commission,
                MAX(pnl) as largest_win,
                MIN(pnl) as largest_loss,
                AVG(CASE WHEN is_winner = 1 THEN pnl END) as avg_win,
                AVG(CASE WHEN is_winner = 0 AND is_scratch = 0 THEN pnl END) as avg_loss,
                AVG(hold_time_minutes) as avg_hold_time
            FROM trade_history
            WHERE trade_date = ?
        """, (target_date.isoformat(),))

        row = cursor.fetchone()
        conn.close()

        if not row or row[0] == 0:
            return {
                "date": target_date.isoformat(),
                "total_trades": 0,
                "message": "No trades recorded for this day"
            }

        total, winners, losers, scratches, total_pnl, commission, largest_win, largest_loss, avg_win, avg_loss, avg_hold = row

        win_rate = (winners / total * 100) if total > 0 else 0
        gross_wins = (avg_win or 0) * (winners or 0)
        gross_losses = abs(avg_loss or 0) * (losers or 0)
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float('inf') if gross_wins > 0 else 0

        # Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
        expectancy = (win_rate/100 * (avg_win or 0)) - ((100-win_rate)/100 * abs(avg_loss or 0))

        return {
            "date": target_date.isoformat(),
            "total_trades": total,
            "winners": winners or 0,
            "losers": losers or 0,
            "scratches": scratches or 0,
            "win_rate": round(win_rate, 2),
            "gross_pnl": round(total_pnl or 0, 2),
            "commissions": round(commission or 0, 2),
            "net_pnl": round((total_pnl or 0) - (commission or 0), 2),
            "largest_win": round(largest_win or 0, 2),
            "largest_loss": round(largest_loss or 0, 2),
            "avg_win": round(avg_win or 0, 2),
            "avg_loss": round(avg_loss or 0, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "∞",
            "expectancy": round(expectancy, 2),
            "avg_hold_time_minutes": round(avg_hold or 0, 1)
        }

    def get_overall_stats(self, days: int = 30) -> Dict:
        """Get overall trading statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        start_date = (date.today() - timedelta(days=days)).isoformat()

        cursor.execute("""
            SELECT
                COUNT(*) as total_trades,
                COUNT(DISTINCT trade_date) as trading_days,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as winners,
                SUM(CASE WHEN is_winner = 0 AND is_scratch = 0 THEN 1 ELSE 0 END) as losers,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade,
                AVG(CASE WHEN is_winner = 1 THEN pnl END) as avg_win,
                AVG(CASE WHEN is_winner = 0 AND is_scratch = 0 THEN pnl END) as avg_loss,
                AVG(hold_time_minutes) as avg_hold_time
            FROM trade_history
            WHERE trade_date >= ?
        """, (start_date,))

        row = cursor.fetchone()
        conn.close()

        if not row or row[0] == 0:
            return {"message": "No trades in the specified period", "days": days}

        total, days_traded, winners, losers, total_pnl, avg_pnl, best, worst, avg_win, avg_loss, avg_hold = row

        win_rate = (winners / total * 100) if total > 0 else 0
        avg_trades_per_day = total / days_traded if days_traded > 0 else 0

        gross_wins = (avg_win or 0) * (winners or 0)
        gross_losses = abs(avg_loss or 0) * (losers or 0)
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float('inf') if gross_wins > 0 else 0

        return {
            "period_days": days,
            "trading_days": days_traded,
            "total_trades": total,
            "avg_trades_per_day": round(avg_trades_per_day, 1),
            "winners": winners or 0,
            "losers": losers or 0,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl or 0, 2),
            "avg_pnl_per_trade": round(avg_pnl or 0, 2),
            "best_trade": round(best or 0, 2),
            "worst_trade": round(worst or 0, 2),
            "avg_win": round(avg_win or 0, 2),
            "avg_loss": round(avg_loss or 0, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "∞",
            "avg_hold_time_minutes": round(avg_hold or 0, 1)
        }

    def get_symbol_performance(self, limit: int = 20) -> List[Dict]:
        """Get performance breakdown by symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                symbol,
                COUNT(*) as trades,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as winners,
                ROUND(SUM(CASE WHEN is_winner = 1 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 2) as win_rate,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                AVG(hold_time_minutes) as avg_hold
            FROM trade_history
            GROUP BY symbol
            ORDER BY total_pnl DESC
            LIMIT ?
        """, (limit,))

        results = []
        for row in cursor.fetchall():
            results.append({
                "symbol": row[0],
                "trades": row[1],
                "winners": row[2],
                "win_rate": row[3],
                "total_pnl": round(row[4] or 0, 2),
                "avg_pnl": round(row[5] or 0, 2),
                "avg_hold_minutes": round(row[6] or 0, 1)
            })

        conn.close()
        return results

    def get_strategy_performance(self) -> List[Dict]:
        """Get performance breakdown by strategy/setup"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COALESCE(strategy, 'Unknown') as strategy,
                COALESCE(setup, 'Unknown') as setup,
                COUNT(*) as trades,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as winners,
                ROUND(SUM(CASE WHEN is_winner = 1 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 2) as win_rate,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl
            FROM trade_history
            WHERE strategy IS NOT NULL AND strategy != ''
            GROUP BY strategy, setup
            ORDER BY total_pnl DESC
        """)

        results = []
        for row in cursor.fetchall():
            results.append({
                "strategy": row[0],
                "setup": row[1],
                "trades": row[2],
                "winners": row[3],
                "win_rate": row[4],
                "total_pnl": round(row[5] or 0, 2),
                "avg_pnl": round(row[6] or 0, 2)
            })

        conn.close()
        return results

    def get_time_analysis(self) -> Dict:
        """Analyze performance by time of day and day of week"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # By time of day
        cursor.execute("""
            SELECT
                time_of_day,
                COUNT(*) as trades,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as winners,
                ROUND(SUM(CASE WHEN is_winner = 1 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 2) as win_rate,
                SUM(pnl) as total_pnl
            FROM trade_history
            WHERE time_of_day IS NOT NULL
            GROUP BY time_of_day
            ORDER BY total_pnl DESC
        """)

        by_time = []
        for row in cursor.fetchall():
            by_time.append({
                "time_slot": row[0],
                "trades": row[1],
                "winners": row[2],
                "win_rate": row[3],
                "total_pnl": round(row[4] or 0, 2)
            })

        # By day of week
        cursor.execute("""
            SELECT
                day_of_week,
                COUNT(*) as trades,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as winners,
                ROUND(SUM(CASE WHEN is_winner = 1 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 2) as win_rate,
                SUM(pnl) as total_pnl
            FROM trade_history
            WHERE day_of_week IS NOT NULL
            GROUP BY day_of_week
            ORDER BY
                CASE day_of_week
                    WHEN 'Monday' THEN 1
                    WHEN 'Tuesday' THEN 2
                    WHEN 'Wednesday' THEN 3
                    WHEN 'Thursday' THEN 4
                    WHEN 'Friday' THEN 5
                    ELSE 6
                END
        """)

        by_day = []
        for row in cursor.fetchall():
            by_day.append({
                "day": row[0],
                "trades": row[1],
                "winners": row[2],
                "win_rate": row[3],
                "total_pnl": round(row[4] or 0, 2)
            })

        conn.close()

        return {
            "by_time_of_day": by_time,
            "by_day_of_week": by_day
        }

    def get_insights(self) -> Dict:
        """Generate actionable insights from trade data"""
        insights = {
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }

        # Get various stats
        overall = self.get_overall_stats(30)
        symbols = self.get_symbol_performance(10)
        strategies = self.get_strategy_performance()
        time_analysis = self.get_time_analysis()

        if overall.get('total_trades', 0) < 5:
            insights["recommendations"].append("Need more trade data for meaningful analysis (minimum 5 trades)")
            return insights

        # Analyze win rate
        win_rate = overall.get('win_rate', 0)
        if win_rate >= 60:
            insights["strengths"].append(f"Strong win rate of {win_rate}%")
        elif win_rate < 40:
            insights["weaknesses"].append(f"Low win rate of {win_rate}% - focus on trade selection")

        # Analyze profit factor
        pf = overall.get('profit_factor', 0)
        if isinstance(pf, (int, float)) and pf >= 2.0:
            insights["strengths"].append(f"Excellent profit factor of {pf}")
        elif isinstance(pf, (int, float)) and pf < 1.0:
            insights["weaknesses"].append(f"Profit factor below 1.0 ({pf}) - losing money overall")

        # Best performing symbols
        if symbols:
            best_symbols = [s for s in symbols if s['total_pnl'] > 0][:3]
            if best_symbols:
                names = ", ".join([s['symbol'] for s in best_symbols])
                insights["strengths"].append(f"Best performing symbols: {names}")

            worst_symbols = [s for s in symbols if s['total_pnl'] < 0][-3:]
            if worst_symbols:
                names = ", ".join([s['symbol'] for s in worst_symbols])
                insights["weaknesses"].append(f"Avoid or improve: {names}")

        # Time analysis
        by_time = time_analysis.get('by_time_of_day', [])
        if by_time:
            best_time = max(by_time, key=lambda x: x.get('total_pnl', 0))
            worst_time = min(by_time, key=lambda x: x.get('total_pnl', 0))

            if best_time['total_pnl'] > 0:
                insights["strengths"].append(f"Best performance during {best_time['time_slot']} (${best_time['total_pnl']})")

            if worst_time['total_pnl'] < 0:
                insights["weaknesses"].append(f"Struggling during {worst_time['time_slot']} (${worst_time['total_pnl']})")
                insights["recommendations"].append(f"Consider reducing size or avoiding trades during {worst_time['time_slot']}")

        # Day analysis
        by_day = time_analysis.get('by_day_of_week', [])
        if by_day:
            best_day = max(by_day, key=lambda x: x.get('total_pnl', 0))
            worst_day = min(by_day, key=lambda x: x.get('total_pnl', 0))

            if best_day['total_pnl'] > 0:
                insights["strengths"].append(f"Best day: {best_day['day']} (${best_day['total_pnl']})")

            if worst_day['total_pnl'] < 0:
                insights["recommendations"].append(f"Review {worst_day['day']} trades - currently losing ${abs(worst_day['total_pnl'])}")

        return insights

    def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        """Get recent trades"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                trade_id, account, trade_date, symbol, side, quantity,
                entry_price, exit_price, pnl, pnl_percent,
                strategy, setup, hold_time_minutes, is_winner, notes
            FROM trade_history
            ORDER BY entry_time DESC
            LIMIT ?
        """, (limit,))

        results = []
        for row in cursor.fetchall():
            results.append({
                "trade_id": row[0],
                "account": row[1],
                "date": row[2],
                "symbol": row[3],
                "side": row[4],
                "quantity": row[5],
                "entry_price": row[6],
                "exit_price": row[7],
                "pnl": round(row[8] or 0, 2),
                "pnl_percent": round(row[9] or 0, 2),
                "strategy": row[10],
                "setup": row[11],
                "hold_time": row[12],
                "is_winner": bool(row[13]),
                "notes": row[14]
            })

        conn.close()
        return results

    def _update_statistics(self):
        """Update aggregated statistics tables"""
        # This runs after each trade to keep stats current
        pass  # Implemented on-demand via queries for now

    def calculate_pnl_from_fills(self) -> Dict:
        """Calculate P&L by matching buy/sell fills for each symbol (FIFO method)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all trades grouped by account and symbol
        cursor.execute("""
            SELECT id, account, symbol, side, quantity, entry_price, entry_time
            FROM trade_history
            ORDER BY account, symbol, entry_time
        """)

        trades = cursor.fetchall()

        # Group by account + symbol
        from collections import defaultdict
        grouped = defaultdict(list)
        for t in trades:
            key = (t[1], t[2])  # account, symbol
            grouped[key].append({
                'id': t[0],
                'side': t[3],
                'qty': t[4],
                'price': t[5],
                'time': t[6]
            })

        total_pnl = 0
        matched_trades = 0

        for (account, symbol), fills in grouped.items():
            buys = []

            for fill in fills:
                if fill['side'] == 'BUY':
                    buys.append({'qty': fill['qty'], 'price': fill['price'], 'id': fill['id']})
                elif fill['side'] == 'SELL' and buys:
                    # Match FIFO
                    sell_qty = fill['qty']
                    sell_price = fill['price']

                    while sell_qty > 0 and buys:
                        buy = buys[0]
                        match_qty = min(buy['qty'], sell_qty)

                        # Calculate P&L for this match
                        pnl = (sell_price - buy['price']) * match_qty
                        total_pnl += pnl

                        # Update the sell trade's P&L in DB
                        cursor.execute("""
                            UPDATE trade_history
                            SET pnl = pnl + ?,
                                exit_price = ?,
                                is_winner = CASE WHEN (pnl + ?) > 0.50 THEN 1 ELSE 0 END,
                                is_scratch = CASE WHEN ABS(pnl + ?) <= 0.50 THEN 1 ELSE 0 END
                            WHERE id = ?
                        """, (pnl, buy['price'], pnl, pnl, fill['id']))

                        matched_trades += 1

                        buy['qty'] -= match_qty
                        sell_qty -= match_qty

                        if buy['qty'] <= 0:
                            buys.pop(0)

        conn.commit()
        conn.close()

        return {
            "total_pnl_calculated": round(total_pnl, 2),
            "matched_round_trips": matched_trades,
            "message": "P&L calculated from matched buy/sell fills"
        }


# Singleton instance
_analytics_instance = None

def get_trade_analytics() -> TradeAnalytics:
    """Get singleton instance of trade analytics"""
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = TradeAnalytics()
    return _analytics_instance
