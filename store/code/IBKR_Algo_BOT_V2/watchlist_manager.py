"""
Watchlist Manager - Database-backed symbol list management
Manages watchlists for AI training, backtesting, and automated trading
Automatically runs pattern backtesting when symbols are added for baseline performance
"""
import sqlite3
import json
import logging
import threading
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class WatchlistManager:
    """Manages watchlists in SQLite database"""

    def __init__(self, db_path: str = "database/warrior_trading.db"):
        """Initialize watchlist manager

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_database()

    def _ensure_database(self):
        """Ensure database and tables exist"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Create watchlists table if not exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS watchlists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    watchlist_id TEXT UNIQUE NOT NULL,
                    watchlist_name TEXT NOT NULL,
                    symbols TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create pattern backtest baselines table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pattern_baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    pattern_name TEXT NOT NULL,
                    total_signals INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    avg_return REAL DEFAULT 0,
                    profit_factor REAL DEFAULT 0,
                    best_return REAL DEFAULT 0,
                    worst_return REAL DEFAULT 0,
                    backtest_period TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, pattern_name)
                )
            """)
            conn.commit()

    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_watchlist(self, name: str, symbols: List[str]) -> Dict:
        """Create a new watchlist

        Args:
            name: Watchlist name
            symbols: List of stock symbols

        Returns:
            Created watchlist dict
        """
        watchlist_id = str(uuid.uuid4())
        symbols_json = json.dumps([s.upper() for s in symbols])

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO watchlists (watchlist_id, watchlist_name, symbols)
                VALUES (?, ?, ?)
            """, (watchlist_id, name, symbols_json))
            conn.commit()

            logger.info(f"Created watchlist '{name}' with {len(symbols)} symbols")

            return self.get_watchlist(watchlist_id)

    def get_watchlist(self, watchlist_id: str) -> Optional[Dict]:
        """Get a specific watchlist by ID

        Args:
            watchlist_id: Watchlist unique ID

        Returns:
            Watchlist dict or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM watchlists WHERE watchlist_id = ?
            """, (watchlist_id,))

            row = cursor.fetchone()

            if row:
                return {
                    'id': row['id'],
                    'watchlist_id': row['watchlist_id'],
                    'name': row['watchlist_name'],
                    'symbols': json.loads(row['symbols']),
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                }

            return None

    def get_watchlist_by_name(self, name: str) -> Optional[Dict]:
        """Get a watchlist by name

        Args:
            name: Watchlist name

        Returns:
            Watchlist dict or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM watchlists WHERE watchlist_name = ?
            """, (name,))

            row = cursor.fetchone()

            if row:
                return {
                    'id': row['id'],
                    'watchlist_id': row['watchlist_id'],
                    'name': row['watchlist_name'],
                    'symbols': json.loads(row['symbols']),
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                }

            return None

    def get_all_watchlists(self) -> List[Dict]:
        """Get all watchlists

        Returns:
            List of watchlist dicts
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM watchlists ORDER BY watchlist_name
            """)

            watchlists = []
            for row in cursor.fetchall():
                watchlists.append({
                    'id': row['id'],
                    'watchlist_id': row['watchlist_id'],
                    'name': row['watchlist_name'],
                    'symbols': json.loads(row['symbols']),
                    'symbol_count': len(json.loads(row['symbols'])),
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                })

            return watchlists

    def update_watchlist(self, watchlist_id: str, name: Optional[str] = None,
                        symbols: Optional[List[str]] = None) -> Optional[Dict]:
        """Update a watchlist

        Args:
            watchlist_id: Watchlist unique ID
            name: New name (optional)
            symbols: New symbols list (optional)

        Returns:
            Updated watchlist dict or None if not found
        """
        watchlist = self.get_watchlist(watchlist_id)
        if not watchlist:
            return None

        update_fields = []
        params = []

        if name is not None:
            update_fields.append("watchlist_name = ?")
            params.append(name)

        if symbols is not None:
            update_fields.append("symbols = ?")
            params.append(json.dumps([s.upper() for s in symbols]))

        if not update_fields:
            return watchlist

        update_fields.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(watchlist_id)

        with self._get_connection() as conn:
            conn.execute(f"""
                UPDATE watchlists
                SET {', '.join(update_fields)}
                WHERE watchlist_id = ?
            """, params)
            conn.commit()

            logger.info(f"Updated watchlist '{watchlist['name']}'")

            return self.get_watchlist(watchlist_id)

    def add_symbols(self, watchlist_id: str, symbols: List[str]) -> Optional[Dict]:
        """Add symbols to a watchlist

        Args:
            watchlist_id: Watchlist unique ID
            symbols: List of symbols to add

        Returns:
            Updated watchlist dict or None if not found
        """
        watchlist = self.get_watchlist(watchlist_id)
        if not watchlist:
            return None

        current_symbols = set(watchlist['symbols'])
        new_symbols = [s.upper() for s in symbols]

        # Identify truly new symbols (not already in watchlist)
        truly_new = [s for s in new_symbols if s not in current_symbols]

        # Add new symbols (avoid duplicates)
        for symbol in new_symbols:
            current_symbols.add(symbol)

        result = self.update_watchlist(watchlist_id, symbols=list(current_symbols))

        # Trigger background backtest for truly new symbols
        if truly_new:
            self._trigger_background_backtest(truly_new)

        return result

    def _trigger_background_backtest(self, symbols: List[str]):
        """Run pattern backtest in background thread for new symbols

        Args:
            symbols: List of symbols to backtest
        """
        def run_backtest():
            try:
                from ai.pattern_backtester import PatternBacktester

                logger.info(f"Starting background backtest for: {symbols}")

                backtester = PatternBacktester()

                for symbol in symbols:
                    try:
                        # Run backtest for 6 months of history
                        results = backtester.backtest_symbol(symbol, period="6mo")

                        if results and 'pattern_results' in results:
                            # Store results in database
                            self._store_backtest_baseline(symbol, results)
                            logger.info(f"Baseline backtest completed for {symbol}")
                        else:
                            logger.warning(f"No backtest results for {symbol}")

                    except Exception as e:
                        logger.error(f"Backtest failed for {symbol}: {e}")

            except ImportError:
                logger.warning("Pattern backtester not available")
            except Exception as e:
                logger.error(f"Background backtest error: {e}")

        # Run in background thread
        thread = threading.Thread(target=run_backtest, daemon=True)
        thread.start()
        logger.info(f"Background backtest triggered for {len(symbols)} symbols")

    def _store_backtest_baseline(self, symbol: str, results: Dict):
        """Store backtest results as baseline in database

        Args:
            symbol: Stock symbol
            results: Backtest results dict
        """
        pattern_results = results.get('pattern_results', {})
        period = results.get('period', '6mo')

        with self._get_connection() as conn:
            for pattern_name, stats in pattern_results.items():
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO pattern_baselines
                        (symbol, pattern_name, total_signals, wins, losses,
                         win_rate, avg_return, profit_factor, best_return,
                         worst_return, backtest_period, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol.upper(),
                        pattern_name,
                        stats.get('total_signals', 0),
                        stats.get('wins', 0),
                        stats.get('losses', 0),
                        stats.get('win_rate', 0),
                        stats.get('avg_return', 0),
                        stats.get('profit_factor', 0),
                        stats.get('best_return', 0),
                        stats.get('worst_return', 0),
                        period,
                        datetime.now().isoformat()
                    ))
                except Exception as e:
                    logger.error(f"Failed to store baseline for {symbol}/{pattern_name}: {e}")

            conn.commit()
            logger.info(f"Stored {len(pattern_results)} pattern baselines for {symbol}")

    def get_symbol_baseline(self, symbol: str) -> List[Dict]:
        """Get pattern baselines for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            List of pattern baseline dicts
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM pattern_baselines
                WHERE symbol = ?
                ORDER BY win_rate DESC
            """, (symbol.upper(),))

            baselines = []
            for row in cursor.fetchall():
                baselines.append({
                    'symbol': row['symbol'],
                    'pattern': row['pattern_name'],
                    'total_signals': row['total_signals'],
                    'wins': row['wins'],
                    'losses': row['losses'],
                    'win_rate': row['win_rate'],
                    'avg_return': row['avg_return'],
                    'profit_factor': row['profit_factor'],
                    'best_return': row['best_return'],
                    'worst_return': row['worst_return'],
                    'period': row['backtest_period'],
                    'updated_at': row['updated_at']
                })

            return baselines

    def get_best_patterns_for_symbol(self, symbol: str, min_win_rate: float = 0.5) -> List[Dict]:
        """Get best performing patterns for a symbol

        Args:
            symbol: Stock symbol
            min_win_rate: Minimum win rate filter (default 50%)

        Returns:
            List of top patterns
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM pattern_baselines
                WHERE symbol = ? AND win_rate >= ? AND total_signals >= 3
                ORDER BY profit_factor DESC, win_rate DESC
                LIMIT 5
            """, (symbol.upper(), min_win_rate))

            return [dict(row) for row in cursor.fetchall()]

    def get_all_baselines(self) -> List[Dict]:
        """Get all pattern baselines across all symbols

        Returns:
            List of all baselines
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM pattern_baselines
                ORDER BY symbol, win_rate DESC
            """)

            return [dict(row) for row in cursor.fetchall()]

    def remove_symbols(self, watchlist_id: str, symbols: List[str]) -> Optional[Dict]:
        """Remove symbols from a watchlist

        Args:
            watchlist_id: Watchlist unique ID
            symbols: List of symbols to remove

        Returns:
            Updated watchlist dict or None if not found
        """
        watchlist = self.get_watchlist(watchlist_id)
        if not watchlist:
            return None

        current_symbols = set(watchlist['symbols'])
        remove_symbols = set(s.upper() for s in symbols)

        # Remove symbols
        current_symbols -= remove_symbols

        return self.update_watchlist(watchlist_id, symbols=list(current_symbols))

    def delete_watchlist(self, watchlist_id: str) -> bool:
        """Delete a watchlist

        Args:
            watchlist_id: Watchlist unique ID

        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                DELETE FROM watchlists WHERE watchlist_id = ?
            """, (watchlist_id,))
            conn.commit()

            deleted = cursor.rowcount > 0

            if deleted:
                logger.info(f"Deleted watchlist {watchlist_id}")

            return deleted

    def get_default_watchlist(self) -> Dict:
        """Get or create default watchlist

        Returns:
            Default watchlist dict
        """
        # Check if default watchlist exists
        default = self.get_watchlist_by_name("Default")

        if default:
            return default

        # Create default watchlist
        default_symbols = [
            "SPY", "QQQ", "AAPL", "MSFT", "GOOGL",
            "AMZN", "TSLA", "NVDA", "META", "AMD"
        ]

        return self.create_watchlist("Default", default_symbols)

    def get_all_symbols(self) -> List[str]:
        """Get all unique symbols across all watchlists

        Returns:
            List of unique symbols
        """
        all_symbols = set()

        for watchlist in self.get_all_watchlists():
            all_symbols.update(watchlist['symbols'])

        return sorted(list(all_symbols))

    def clear_default_watchlist(self) -> Dict:
        """Clear all symbols from the default watchlist (PURGE)

        Returns:
            Dict with symbols_cleared count and success status
        """
        default = self.get_default_watchlist()
        if not default:
            return {"success": False, "symbols_cleared": 0, "error": "No default watchlist"}

        symbols_before = default.get("symbols", [])
        count = len(symbols_before)

        # Update to empty list
        self.update_watchlist(default["watchlist_id"], symbols=[])

        logger.warning(f"PURGE: Cleared {count} symbols from default watchlist")

        return {
            "success": True,
            "symbols_cleared": count,
            "symbols_before": symbols_before
        }


# Singleton instance
_watchlist_manager = None

def get_watchlist_manager() -> WatchlistManager:
    """Get singleton watchlist manager instance"""
    global _watchlist_manager
    if _watchlist_manager is None:
        _watchlist_manager = WatchlistManager()
    return _watchlist_manager
