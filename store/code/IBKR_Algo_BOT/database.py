"""
Database management for Trading Bot Dashboard
SQLite database for storing trades, positions, logs, and watchlists
"""

import sqlite3
from datetime import datetime
from pathlib import Path
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class TradingDatabase:
    """Main database manager"""
    
    def __init__(self, db_path='dashboard_data/trading_bot.db'):
        """Initialize database connection"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.init_database()
        
    def init_database(self):
        """Create database tables if they don't exist"""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        cursor = self.conn.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                pnl REAL,
                strategy TEXT NOT NULL,
                status TEXT DEFAULT 'open',
                duration_seconds INTEGER,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Positions table (current open positions)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL,
                pnl REAL,
                strategy TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                last_update TEXT,
                status TEXT DEFAULT 'open'
            )
        ''')
        
        # Activity logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                source TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Watchlists table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Watchlist symbols table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlist_symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                watchlist_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (watchlist_id) REFERENCES watchlists (id) ON DELETE CASCADE,
                UNIQUE(watchlist_id, symbol)
            )
        ''')
        
        # Training history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                model_type TEXT NOT NULL,
                accuracy REAL,
                loss REAL,
                epochs INTEGER,
                training_time_seconds REAL,
                backtest_return REAL,
                backtest_sharpe REAL,
                win_rate REAL,
                model_path TEXT,
                config TEXT,
                trained_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                strategy TEXT NOT NULL,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                total_pnl REAL,
                win_rate REAL,
                avg_win REAL,
                avg_loss REAL,
                profit_factor REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, strategy)
            )
        ''')
        
        self.conn.commit()
        logger.info('Database initialized successfully')
    
    # ========================================================================
    # TRADES MANAGEMENT
    # ========================================================================
    
    def add_trade(self, trade_data: Dict) -> int:
        """Add a new trade to database"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (
                timestamp, symbol, action, quantity, entry_price, 
                exit_price, pnl, strategy, status, duration_seconds, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data.get('timestamp', datetime.now().isoformat()),
            trade_data['symbol'],
            trade_data['action'],
            trade_data['quantity'],
            trade_data['entry_price'],
            trade_data.get('exit_price'),
            trade_data.get('pnl'),
            trade_data['strategy'],
            trade_data.get('status', 'closed'),
            trade_data.get('duration_seconds'),
            trade_data.get('notes')
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_trades(self, limit=100, strategy=None, symbol=None) -> List[Dict]:
        """Get trade history"""
        cursor = self.conn.cursor()
        
        query = 'SELECT * FROM trades WHERE 1=1'
        params = []
        
        if strategy:
            query += ' AND strategy = ?'
            params.append(strategy)
        
        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_trade_statistics(self, strategy=None, start_date=None) -> Dict:
        """Get trade statistics"""
        cursor = self.conn.cursor()
        
        query = '''
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(pnl) as total_pnl,
                AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade
            FROM trades
            WHERE status = 'closed'
        '''
        
        params = []
        
        if strategy:
            query += ' AND strategy = ?'
            params.append(strategy)
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        
        cursor.execute(query, params)
        result = dict(cursor.fetchone())
        
        # Calculate derived metrics
        if result['total_trades'] > 0:
            result['win_rate'] = (result['winning_trades'] / result['total_trades']) * 100
        else:
            result['win_rate'] = 0
        
        if result['avg_loss'] and result['avg_loss'] != 0:
            result['profit_factor'] = abs(result['avg_win'] / result['avg_loss'])
        else:
            result['profit_factor'] = 0
        
        return result
    
    # ========================================================================
    # POSITIONS MANAGEMENT
    # ========================================================================
    
    def add_or_update_position(self, position_data: Dict):
        """Add or update a position"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO positions (
                symbol, quantity, entry_price, current_price, pnl,
                strategy, entry_time, last_update, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            position_data['symbol'],
            position_data['quantity'],
            position_data['entry_price'],
            position_data.get('current_price'),
            position_data.get('pnl'),
            position_data['strategy'],
            position_data.get('entry_time', datetime.now().isoformat()),
            datetime.now().isoformat(),
            position_data.get('status', 'open')
        ))
        
        self.conn.commit()
    
    def get_positions(self, strategy=None) -> List[Dict]:
        """Get current positions"""
        cursor = self.conn.cursor()
        
        query = 'SELECT * FROM positions WHERE status = "open"'
        params = []
        
        if strategy:
            query += ' AND strategy = ?'
            params.append(strategy)
        
        cursor.execute(query, params)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def close_position(self, symbol: str, exit_price: float, pnl: float):
        """Close a position"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            UPDATE positions 
            SET status = 'closed', current_price = ?, pnl = ?, last_update = ?
            WHERE symbol = ? AND status = 'open'
        ''', (exit_price, pnl, datetime.now().isoformat(), symbol))
        
        self.conn.commit()
    
    # ========================================================================
    # ACTIVITY LOGS
    # ========================================================================
    
    def add_log(self, level: str, source: str, message: str):
        """Add activity log entry"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO activity_logs (timestamp, level, source, message)
            VALUES (?, ?, ?, ?)
        ''', (datetime.now().isoformat(), level, source, message))
        
        self.conn.commit()
    
    def get_logs(self, limit=100, level=None, source=None) -> List[Dict]:
        """Get activity logs"""
        cursor = self.conn.cursor()
        
        query = 'SELECT * FROM activity_logs WHERE 1=1'
        params = []
        
        if level:
            query += ' AND level = ?'
            params.append(level)
        
        if source:
            query += ' AND source = ?'
            params.append(source)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        
        return [dict(row) for row in cursor.fetchall()]
    
    # ========================================================================
    # WATCHLISTS
    # ========================================================================
    
    def create_watchlist(self, name: str, description: str = '') -> int:
        """Create a new watchlist"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO watchlists (name, description)
            VALUES (?, ?)
        ''', (name, description))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_watchlists(self) -> List[Dict]:
        """Get all watchlists"""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT * FROM watchlists ORDER BY created_at DESC')
        
        watchlists = [dict(row) for row in cursor.fetchall()]
        
        # Get symbol count for each watchlist
        for wl in watchlists:
            cursor.execute('''
                SELECT COUNT(*) as count 
                FROM watchlist_symbols 
                WHERE watchlist_id = ?
            ''', (wl['id'],))
            wl['symbol_count'] = cursor.fetchone()['count']
        
        return watchlists
    
    def add_symbol_to_watchlist(self, watchlist_id: int, symbol: str, metadata: Dict = None):
        """Add symbol to watchlist"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT OR IGNORE INTO watchlist_symbols (watchlist_id, symbol, metadata)
            VALUES (?, ?, ?)
        ''', (watchlist_id, symbol, json.dumps(metadata) if metadata else None))
        
        self.conn.commit()
    
    def get_watchlist_symbols(self, watchlist_id: int) -> List[Dict]:
        """Get symbols in a watchlist"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT * FROM watchlist_symbols 
            WHERE watchlist_id = ? 
            ORDER BY added_at DESC
        ''', (watchlist_id,))
        
        symbols = [dict(row) for row in cursor.fetchall()]
        
        # Parse metadata JSON
        for sym in symbols:
            if sym['metadata']:
                sym['metadata'] = json.loads(sym['metadata'])
        
        return symbols
    
    def remove_symbol_from_watchlist(self, watchlist_id: int, symbol: str):
        """Remove symbol from watchlist"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            DELETE FROM watchlist_symbols 
            WHERE watchlist_id = ? AND symbol = ?
        ''', (watchlist_id, symbol))
        
        self.conn.commit()
    
    # ========================================================================
    # TRAINING HISTORY
    # ========================================================================
    
    def add_training_record(self, training_data: Dict) -> int:
        """Add training history record"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_history (
                symbol, model_type, accuracy, loss, epochs, training_time_seconds,
                backtest_return, backtest_sharpe, win_rate, model_path, config
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            training_data['symbol'],
            training_data['model_type'],
            training_data.get('accuracy'),
            training_data.get('loss'),
            training_data.get('epochs'),
            training_data.get('training_time_seconds'),
            training_data.get('backtest_return'),
            training_data.get('backtest_sharpe'),
            training_data.get('win_rate'),
            training_data.get('model_path'),
            json.dumps(training_data.get('config', {}))
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_training_history(self, symbol=None, limit=50) -> List[Dict]:
        """Get training history"""
        cursor = self.conn.cursor()
        
        query = 'SELECT * FROM training_history WHERE 1=1'
        params = []
        
        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
        
        query += ' ORDER BY trained_at DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        
        records = [dict(row) for row in cursor.fetchall()]
        
        # Parse config JSON
        for rec in records:
            if rec['config']:
                rec['config'] = json.loads(rec['config'])
        
        return records
    
    # ========================================================================
    # PERFORMANCE METRICS
    # ========================================================================
    
    def save_daily_metrics(self, date: str, strategy: str, metrics: Dict):
        """Save daily performance metrics"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO performance_metrics (
                date, strategy, total_trades, winning_trades, losing_trades,
                total_pnl, win_rate, avg_win, avg_loss, profit_factor,
                max_drawdown, sharpe_ratio
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            date,
            strategy,
            metrics.get('total_trades', 0),
            metrics.get('winning_trades', 0),
            metrics.get('losing_trades', 0),
            metrics.get('total_pnl', 0),
            metrics.get('win_rate', 0),
            metrics.get('avg_win', 0),
            metrics.get('avg_loss', 0),
            metrics.get('profit_factor', 0),
            metrics.get('max_drawdown', 0),
            metrics.get('sharpe_ratio', 0)
        ))
        
        self.conn.commit()
    
    def get_performance_history(self, strategy=None, days=30) -> List[Dict]:
        """Get performance history"""
        cursor = self.conn.cursor()
        
        query = 'SELECT * FROM performance_metrics WHERE 1=1'
        params = []
        
        if strategy:
            query += ' AND strategy = ?'
            params.append(strategy)
        
        query += ' ORDER BY date DESC LIMIT ?'
        params.append(days)
        
        cursor.execute(query, params)
        
        return [dict(row) for row in cursor.fetchall()]
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info('Database connection closed')
    
    def backup_database(self, backup_path: str):
        """Create database backup"""
        import shutil
        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.db_path, backup_path)
        logger.info(f'Database backed up to {backup_path}')
    
    def execute_query(self, query: str, params=None) -> List[Dict]:
        """Execute custom query"""
        cursor = self.conn.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        return [dict(row) for row in cursor.fetchall()]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Global database instance
_db_instance = None

def get_database() -> TradingDatabase:
    """Get or create database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = TradingDatabase()
    return _db_instance


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # Example usage
    db = TradingDatabase()
    
    # Create default watchlists
    try:
        mtf_id = db.create_watchlist('MTF Swing', 'Large cap stocks for swing trading')
        db.add_symbol_to_watchlist(mtf_id, 'AAPL', {'sector': 'Technology'})
        db.add_symbol_to_watchlist(mtf_id, 'TSLA', {'sector': 'Automotive'})
        print('Created MTF Swing watchlist')
    except:
        print('MTF Swing watchlist already exists')
    
    try:
        warrior_id = db.create_watchlist('Warrior Gappers', 'Small cap momentum stocks')
        print('Created Warrior Gappers watchlist')
    except:
        print('Warrior Gappers watchlist already exists')
    
    # Add example trade
    trade = {
        'symbol': 'AAPL',
        'action': 'BUY',
        'quantity': 100,
        'entry_price': 254.50,
        'exit_price': 256.20,
        'pnl': 170.0,
        'strategy': 'MTF',
        'status': 'closed',
        'duration_seconds': 7200,
        'notes': 'Test trade'
    }
    trade_id = db.add_trade(trade)
    print(f'Added trade with ID: {trade_id}')
    
    # Get trade statistics
    stats = db.get_trade_statistics()
    print(f'Trade statistics: {stats}')
    
    # Add activity log
    db.add_log('info', 'system', 'Database example completed')
    
    print('Database setup complete!')