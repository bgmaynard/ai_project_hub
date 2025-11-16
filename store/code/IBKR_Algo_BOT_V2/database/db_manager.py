"""
Database Manager for Warrior Trading Bot
Handles all database operations for trades, errors, and performance tracking
"""

import sqlite3
import json
import logging
from datetime import datetime, date
from typing import List, Dict, Optional, Any
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Centralized database management for trade tracking and monitoring"""

    def __init__(self, db_path: str = "database/warrior_trading.db"):
        self.db_path = db_path
        self._ensure_database()

    def _ensure_database(self):
        """Ensure database and tables exist"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # Create tables from schema
        schema_path = db_dir / "schema.sql"
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema = f.read()

            with self.get_connection() as conn:
                conn.executescript(schema)
                conn.commit()

            logger.info(f"Database initialized: {self.db_path}")
        else:
            logger.warning(f"Schema file not found: {schema_path}")

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        try:
            yield conn
        finally:
            conn.close()

    # ==================== TRADE OPERATIONS ====================

    def log_trade_entry(self, trade_data: Dict) -> str:
        """Log a new trade entry"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    trade_id, symbol, side, shares, entry_price,
                    stop_loss, take_profit, pattern_type, pattern_confidence,
                    sentiment_score, slippage_entry
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data['trade_id'],
                trade_data['symbol'],
                trade_data['side'],
                trade_data['shares'],
                trade_data['entry_price'],
                trade_data.get('stop_loss'),
                trade_data.get('take_profit'),
                trade_data.get('pattern_type'),
                trade_data.get('pattern_confidence'),
                trade_data.get('sentiment_score'),
                trade_data.get('slippage_entry', 0)
            ))
            conn.commit()
            return trade_data['trade_id']

    def log_trade_exit(self, trade_id: str, exit_data: Dict) -> bool:
        """Log trade exit and calculate P&L"""
        with self.get_connection() as conn:
            # Get trade entry data
            trade = dict(conn.execute(
                "SELECT * FROM trades WHERE trade_id = ?", (trade_id,)
            ).fetchone())

            # Calculate P&L
            if trade['side'] == 'buy':
                pnl = (exit_data['exit_price'] - trade['entry_price']) * trade['shares']
                pnl_percent = ((exit_data['exit_price'] - trade['entry_price']) / trade['entry_price']) * 100
            else:
                pnl = (trade['entry_price'] - exit_data['exit_price']) * trade['shares']
                pnl_percent = ((trade['entry_price'] - exit_data['exit_price']) / trade['entry_price']) * 100

            # Calculate R multiple
            r_multiple = 0
            if trade['stop_loss']:
                risk = abs(trade['entry_price'] - trade['stop_loss']) * trade['shares']
                if risk > 0:
                    r_multiple = pnl / risk

            # Update trade
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE trades SET
                    exit_price = ?, exit_time = ?, pnl = ?, pnl_percent = ?,
                    r_multiple = ?, status = ?, slippage_exit = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE trade_id = ?
            """, (
                exit_data['exit_price'],
                exit_data.get('exit_time', datetime.now()),
                pnl,
                pnl_percent,
                r_multiple,
                exit_data.get('status', 'closed'),
                exit_data.get('slippage_exit', 0),
                trade_id
            ))
            conn.commit()
            return True

    def get_trades(self, filters: Optional[Dict] = None, limit: int = 100) -> List[Dict]:
        """Get trades with optional filters"""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if filters:
            if filters.get('symbol'):
                query += " AND symbol = ?"
                params.append(filters['symbol'])
            if filters.get('status'):
                query += " AND status = ?"
                params.append(filters['status'])
            if filters.get('date_from'):
                query += " AND entry_time >= ?"
                params.append(filters['date_from'])
            if filters.get('date_to'):
                query += " AND entry_time <= ?"
                params.append(filters['date_to'])

        query += f" ORDER BY entry_time DESC LIMIT {limit}"

        with self.get_connection() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def get_active_trades(self) -> List[Dict]:
        """Get all currently open trades"""
        with self.get_connection() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT * FROM v_active_trades"
            ).fetchall()]

    # ==================== ERROR LOG OPERATIONS ====================

    def log_error(self, error_data: Dict) -> str:
        """Log a system error"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            error_id = error_data.get('error_id', f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")

            cursor.execute("""
                INSERT INTO error_logs (
                    error_id, severity, module, error_type, error_message,
                    stack_trace, context
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                error_id,
                error_data['severity'],
                error_data['module'],
                error_data['error_type'],
                error_data['error_message'],
                error_data.get('stack_trace'),
                json.dumps(error_data.get('context', {}))
            ))
            conn.commit()
            return error_id

    def get_errors(self, filters: Optional[Dict] = None, limit: int = 50) -> List[Dict]:
        """Get error logs with optional filters"""
        query = "SELECT * FROM error_logs WHERE 1=1"
        params = []

        if filters:
            if filters.get('severity'):
                query += " AND severity = ?"
                params.append(filters['severity'])
            if filters.get('module'):
                query += " AND module = ?"
                params.append(filters['module'])
            if filters.get('resolved') is not None:
                query += " AND resolved = ?"
                params.append(1 if filters['resolved'] else 0)

        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def resolve_error(self, error_id: str, notes: str = "") -> bool:
        """Mark an error as resolved"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE error_logs SET
                    resolved = 1, resolved_at = CURRENT_TIMESTAMP, resolution_notes = ?
                WHERE error_id = ?
            """, (notes, error_id))
            conn.commit()
            return cursor.rowcount > 0

    # ==================== PERFORMANCE METRICS ====================

    def calculate_daily_metrics(self, target_date: Optional[date] = None) -> Dict:
        """Calculate and store daily performance metrics"""
        if target_date is None:
            target_date = date.today()

        with self.get_connection() as conn:
            # Get all closed trades for the day
            trades = [dict(row) for row in conn.execute("""
                SELECT * FROM trades
                WHERE DATE(entry_time) = ? AND status != 'open'
            """, (target_date,)).fetchall()]

            if not trades:
                return {}

            # Calculate metrics
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t['pnl'] > 0)
            losing_trades = sum(1 for t in trades if t['pnl'] < 0)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            wins = [t['pnl'] for t in trades if t['pnl'] > 0]
            losses = [t['pnl'] for t in trades if t['pnl'] < 0]

            total_pnl = sum(t['pnl'] for t in trades)
            gross_profit = sum(wins) if wins else 0
            gross_loss = sum(losses) if losses else 0
            profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0

            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0

            r_multiples = [t['r_multiple'] for t in trades if t['r_multiple']]
            avg_r_multiple = sum(r_multiples) / len(r_multiples) if r_multiples else 0

            metrics = {
                'date': target_date,
                'time_period': 'daily',
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': max(wins) if wins else 0,
                'largest_loss': min(losses) if losses else 0,
                'avg_r_multiple': avg_r_multiple
            }

            # Store metrics
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO performance_metrics (
                    date, time_period, total_trades, winning_trades, losing_trades,
                    win_rate, total_pnl, gross_profit, gross_loss, profit_factor,
                    avg_win, avg_loss, largest_win, largest_loss, avg_r_multiple
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, tuple(metrics.values()))
            conn.commit()

            return metrics

    def get_performance_metrics(self, days: int = 30) -> List[Dict]:
        """Get performance metrics for the last N days"""
        with self.get_connection() as conn:
            return [dict(row) for row in conn.execute("""
                SELECT * FROM performance_metrics
                WHERE time_period = 'daily'
                ORDER BY date DESC LIMIT ?
            """, (days,)).fetchall()]

    # ==================== SLIPPAGE TRACKING ====================

    def log_slippage(self, slippage_data: Dict) -> str:
        """Log execution slippage"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            execution_id = slippage_data.get('execution_id', f"EXEC_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")

            # Calculate slippage
            expected = slippage_data['expected_price']
            actual = slippage_data['actual_price']
            shares = slippage_data['shares']

            if slippage_data['side'].lower() == 'buy':
                slippage_pct = (actual - expected) / expected
            else:
                slippage_pct = (expected - actual) / expected

            slippage_cost = abs(actual - expected) * shares

            # Determine severity
            abs_slip = abs(slippage_pct)
            if abs_slip <= 0.001:
                level = 'acceptable'
            elif abs_slip <= 0.0025:
                level = 'warning'
            else:
                level = 'critical'

            cursor.execute("""
                INSERT INTO slippage_log (
                    execution_id, symbol, side, expected_price, actual_price,
                    shares, slippage_pct, slippage_level, slippage_cost
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution_id,
                slippage_data['symbol'],
                slippage_data['side'],
                expected,
                actual,
                shares,
                slippage_pct,
                level,
                slippage_cost
            ))
            conn.commit()
            return execution_id

    def get_slippage_stats(self, symbol: Optional[str] = None, days: int = 7) -> Dict:
        """Get slippage statistics"""
        with self.get_connection() as conn:
            query = """
                SELECT
                    COUNT(*) as total_executions,
                    AVG(slippage_pct) as avg_slippage,
                    MAX(slippage_pct) as max_slippage,
                    SUM(CASE WHEN slippage_level = 'acceptable' THEN 1 ELSE 0 END) as acceptable_count,
                    SUM(CASE WHEN slippage_level = 'warning' THEN 1 ELSE 0 END) as warning_count,
                    SUM(CASE WHEN slippage_level = 'critical' THEN 1 ELSE 0 END) as critical_count,
                    SUM(slippage_cost) as total_cost
                FROM slippage_log
                WHERE timestamp >= datetime('now', '-{} days')
            """.format(days)

            if symbol:
                query += " AND symbol = ?"
                row = conn.execute(query, (symbol,)).fetchone()
            else:
                row = conn.execute(query).fetchone()

            return dict(row) if row else {}

    # ==================== LAYOUT MANAGEMENT ====================

    def save_layout(self, layout_name: str, layout_config: Dict, is_default: bool = False) -> str:
        """Save a dashboard layout"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            layout_id = f"LAYOUT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # If setting as default, unset other defaults
            if is_default:
                cursor.execute("UPDATE user_layouts SET is_default = 0")

            cursor.execute("""
                INSERT INTO user_layouts (
                    layout_id, layout_name, layout_config, is_default
                ) VALUES (?, ?, ?, ?)
            """, (layout_id, layout_name, json.dumps(layout_config), 1 if is_default else 0))
            conn.commit()
            return layout_id

    def get_layouts(self) -> List[Dict]:
        """Get all saved layouts"""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM user_layouts ORDER BY is_default DESC, layout_name").fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d['layout_config'] = json.loads(d['layout_config'])
                result.append(d)
            return result

    def get_default_layout(self) -> Optional[Dict]:
        """Get the default layout"""
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM user_layouts WHERE is_default = 1").fetchone()
            if row:
                d = dict(row)
                d['layout_config'] = json.loads(d['layout_config'])
                return d
            return None


# Global instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create global database manager"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    db = get_db_manager()

    # Example: Log a trade
    trade_data = {
        'trade_id': 'TEST_001',
        'symbol': 'AAPL',
        'side': 'buy',
        'shares': 100,
        'entry_price': 150.00,
        'stop_loss': 149.50,
        'take_profit': 151.00,
        'pattern_type': 'bull_flag',
        'pattern_confidence': 0.75
    }

    print("Logging trade entry...")
    db.log_trade_entry(trade_data)

    # Example: Log trade exit
    exit_data = {
        'exit_price': 150.75,
        'status': 'closed'
    }

    print("Logging trade exit...")
    db.log_trade_exit('TEST_001', exit_data)

    # Example: Get trades
    trades = db.get_trades(limit=10)
    print(f"\nRecent trades: {len(trades)}")
    for trade in trades:
        print(f"  {trade['trade_id']}: {trade['symbol']} {trade['side']} - P&L: ${trade['pnl']:.2f}")

    # Example: Calculate daily metrics
    print("\nCalculating daily metrics...")
    metrics = db.calculate_daily_metrics()
    if metrics:
        print(f"  Total trades: {metrics['total_trades']}")
        print(f"  Win rate: {metrics['win_rate']:.1f}%")
        print(f"  Total P&L: ${metrics['total_pnl']:.2f}")
