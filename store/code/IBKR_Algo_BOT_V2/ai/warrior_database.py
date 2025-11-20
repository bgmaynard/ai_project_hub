"""
Warrior Trading Database Manager

SQLite database for persisting:
- Watchlist candidates (daily scans)
- Trade history (complete lifecycle)
- Daily statistics (performance tracking)
- Pattern detections (for backtesting)
- Scan results (historical analysis)
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, date, timedelta
from contextlib import contextmanager
from dataclasses import asdict

from ai.warrior_scanner import WarriorCandidate
from ai.warrior_pattern_detector import TradingSetup, SetupType
from ai.warrior_risk_manager import TradeRecord

logger = logging.getLogger(__name__)


class WarriorDatabase:
    """Database manager for Warrior Trading system"""

    def __init__(self, db_path: str = "data/warrior_trading.db"):
        """
        Initialize database manager

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database schema
        self._initialize_schema()

        logger.info(f"Warrior Trading database initialized: {self.db_path}")

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _initialize_schema(self):
        """Create database tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Table 1: Watchlist candidates
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    scan_date DATE NOT NULL,
                    scan_time TIMESTAMP NOT NULL,
                    price REAL NOT NULL,
                    gap_percent REAL NOT NULL,
                    relative_volume REAL NOT NULL,
                    float_shares REAL NOT NULL,
                    pre_market_volume INTEGER,
                    catalyst TEXT,
                    daily_chart_signal TEXT,
                    distance_to_resistance REAL,
                    confidence_score REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, scan_date)
                )
            """)

            # Index for fast date queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_watchlist_scan_date
                ON watchlist_candidates(scan_date DESC)
            """)

            # Table 2: Trades
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    setup_type TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    entry_price REAL NOT NULL,
                    shares INTEGER NOT NULL,
                    stop_price REAL NOT NULL,
                    target_price REAL NOT NULL,
                    exit_time TIMESTAMP,
                    exit_price REAL,
                    exit_reason TEXT,
                    pnl REAL,
                    pnl_percent REAL,
                    r_multiple REAL,
                    status TEXT DEFAULT 'OPEN',
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Indexes for trades
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol
                ON trades(symbol)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_entry_time
                ON trades(entry_time DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_status
                ON trades(status)
            """)

            # Table 3: Daily statistics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date DATE UNIQUE NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,
                    gross_profit REAL DEFAULT 0.0,
                    gross_loss REAL DEFAULT 0.0,
                    net_pnl REAL DEFAULT 0.0,
                    avg_win REAL DEFAULT 0.0,
                    avg_loss REAL DEFAULT 0.0,
                    avg_r_multiple REAL DEFAULT 0.0,
                    largest_win REAL DEFAULT 0.0,
                    largest_loss REAL DEFAULT 0.0,
                    consecutive_wins INTEGER DEFAULT 0,
                    consecutive_losses INTEGER DEFAULT 0,
                    max_drawdown REAL DEFAULT 0.0,
                    profit_factor REAL DEFAULT 0.0,
                    goal_reached BOOLEAN DEFAULT 0,
                    trading_halted BOOLEAN DEFAULT 0,
                    halt_reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Table 4: Pattern detections
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pattern_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    detection_time TIMESTAMP NOT NULL,
                    setup_type TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    target_1r REAL NOT NULL,
                    target_2r REAL NOT NULL,
                    target_3r REAL NOT NULL,
                    risk_per_share REAL NOT NULL,
                    reward_per_share REAL NOT NULL,
                    risk_reward_ratio REAL NOT NULL,
                    confidence REAL NOT NULL,
                    strength_factors TEXT,
                    risk_factors TEXT,
                    entry_condition TEXT,
                    stop_reason TEXT,
                    current_price REAL NOT NULL,
                    was_traded BOOLEAN DEFAULT 0,
                    trade_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
                )
            """)

            # Indexes for pattern detections
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_symbol
                ON pattern_detections(symbol)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_detection_time
                ON pattern_detections(detection_time DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_setup_type
                ON pattern_detections(setup_type)
            """)

            # Table 5: Scan results (historical)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_date DATE NOT NULL,
                    scan_time TIMESTAMP NOT NULL,
                    scan_type TEXT DEFAULT 'PREMARKET',
                    total_candidates INTEGER NOT NULL,
                    scan_duration_seconds REAL,
                    parameters TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # ═════════════════════════════════════════════════════════════════
            #                   CLAUDE AI TABLES (PHASE 7)
            # ═════════════════════════════════════════════════════════════════

            # Table 6: AI Optimization Suggestions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_suggestions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    suggestion_date DATE NOT NULL,
                    category TEXT NOT NULL,
                    parameter_name TEXT,
                    current_value TEXT,
                    suggested_value TEXT,
                    reasoning TEXT,
                    expected_impact TEXT,
                    confidence REAL,
                    priority TEXT,
                    status TEXT DEFAULT 'pending',
                    applied_date DATETIME,
                    actual_impact TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_suggestions_date
                ON ai_suggestions(suggestion_date DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_suggestions_status
                ON ai_suggestions(status)
            """)

            # Table 7: Market Regimes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_regimes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_time DATETIME NOT NULL,
                    regime_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT,
                    indicators TEXT,
                    adjustments TEXT,
                    warnings TEXT,
                    duration_minutes INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_regimes_detection_time
                ON market_regimes(detection_time DESC)
            """)

            # Table 8: AI Insights and Recommendations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    insight_date DATE NOT NULL,
                    insight_type TEXT NOT NULL,
                    insight_text TEXT NOT NULL,
                    context TEXT,
                    priority TEXT,
                    acknowledged BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_insights_date
                ON ai_insights(insight_date DESC)
            """)

            # Table 9: Error Recovery Log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS error_recovery_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_time DATETIME NOT NULL,
                    error_type TEXT NOT NULL,
                    error_message TEXT,
                    error_category TEXT,
                    error_severity TEXT,
                    component TEXT,
                    diagnosis TEXT,
                    recovery_actions TEXT,
                    attempted_actions TEXT,
                    recovery_status TEXT,
                    recovery_time_seconds REAL,
                    requires_manual BOOLEAN DEFAULT 0,
                    resolution_time DATETIME,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_errors_time
                ON error_recovery_log(error_time DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_errors_status
                ON error_recovery_log(recovery_status)
            """)

            # Table 10: AI Usage Tracking (for cost monitoring)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_time DATETIME NOT NULL,
                    request_type TEXT NOT NULL,
                    tokens_used INTEGER,
                    cost_usd REAL,
                    response_time_ms INTEGER,
                    success BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_time
                ON ai_usage_log(request_time DESC)
            """)

            logger.info("Database schema initialized successfully (includes Claude AI tables)")

    # ═════════════════════════════════════════════════════════════════
    #                     WATCHLIST OPERATIONS
    # ═════════════════════════════════════════════════════════════════

    def save_watchlist_candidates(
        self,
        candidates: List[WarriorCandidate],
        scan_date: Optional[date] = None
    ) -> int:
        """
        Save watchlist candidates to database

        Args:
            candidates: List of WarriorCandidate objects
            scan_date: Date of scan (defaults to today)

        Returns:
            Number of candidates saved
        """
        if not candidates:
            return 0

        scan_date = scan_date or date.today()
        scan_time = datetime.now()

        with self.get_connection() as conn:
            cursor = conn.cursor()

            saved_count = 0
            for candidate in candidates:
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO watchlist_candidates (
                            symbol, scan_date, scan_time, price, gap_percent,
                            relative_volume, float_shares, pre_market_volume,
                            catalyst, daily_chart_signal, distance_to_resistance,
                            confidence_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        candidate.symbol,
                        scan_date,
                        scan_time,
                        candidate.price,
                        candidate.gap_percent,
                        candidate.relative_volume,
                        candidate.float_shares,
                        candidate.pre_market_volume,
                        candidate.catalyst,
                        candidate.daily_chart_signal,
                        candidate.distance_to_resistance,
                        candidate.confidence_score
                    ))
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Error saving candidate {candidate.symbol}: {e}")

            logger.info(f"Saved {saved_count} watchlist candidates for {scan_date}")
            return saved_count

    def get_watchlist(
        self,
        scan_date: Optional[date] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get watchlist candidates

        Args:
            scan_date: Date to retrieve (defaults to today)
            limit: Maximum number of results

        Returns:
            List of watchlist candidates
        """
        scan_date = scan_date or date.today()

        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM watchlist_candidates
                WHERE scan_date = ?
                ORDER BY confidence_score DESC
                LIMIT ?
            """, (scan_date, limit))

            results = cursor.fetchall()
            return [dict(row) for row in results]

    def get_watchlist_history(
        self,
        days: int = 7,
        limit_per_day: int = 10
    ) -> Dict[str, List[Dict]]:
        """
        Get watchlist history for multiple days

        Args:
            days: Number of days to retrieve
            limit_per_day: Max candidates per day

        Returns:
            Dict mapping dates to candidate lists
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT DISTINCT scan_date
                FROM watchlist_candidates
                ORDER BY scan_date DESC
                LIMIT ?
            """, (days,))

            dates = [row['scan_date'] for row in cursor.fetchall()]

            history = {}
            for scan_date in dates:
                candidates = self.get_watchlist(scan_date, limit_per_day)
                history[scan_date] = candidates

            return history

    # ═════════════════════════════════════════════════════════════════
    #                     TRADE OPERATIONS
    # ═════════════════════════════════════════════════════════════════

    def save_trade_entry(
        self,
        trade_id: str,
        symbol: str,
        setup_type: str,
        entry_time: datetime,
        entry_price: float,
        shares: int,
        stop_price: float,
        target_price: float
    ) -> bool:
        """
        Save trade entry to database

        Args:
            trade_id: Unique trade identifier
            symbol: Stock symbol
            setup_type: Pattern type
            entry_time: Entry timestamp
            entry_price: Entry price
            shares: Number of shares
            stop_price: Stop loss price
            target_price: Target price

        Returns:
            True if saved successfully
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute("""
                    INSERT INTO trades (
                        trade_id, symbol, setup_type, entry_time, entry_price,
                        shares, stop_price, target_price, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
                """, (
                    trade_id, symbol, setup_type, entry_time, entry_price,
                    shares, stop_price, target_price
                ))

                logger.info(f"Trade entry saved: {trade_id}")
                return True

            except Exception as e:
                logger.error(f"Error saving trade entry: {e}")
                return False

    def save_trade_exit(
        self,
        trade_id: str,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        pnl_percent: float,
        r_multiple: float
    ) -> bool:
        """
        Save trade exit to database

        Args:
            trade_id: Trade identifier
            exit_time: Exit timestamp
            exit_price: Exit price
            exit_reason: Reason for exit
            pnl: Profit/loss in dollars
            pnl_percent: P&L as percentage
            r_multiple: R multiple

        Returns:
            True if saved successfully
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute("""
                    UPDATE trades
                    SET exit_time = ?,
                        exit_price = ?,
                        exit_reason = ?,
                        pnl = ?,
                        pnl_percent = ?,
                        r_multiple = ?,
                        status = 'CLOSED',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE trade_id = ?
                """, (
                    exit_time, exit_price, exit_reason, pnl,
                    pnl_percent, r_multiple, trade_id
                ))

                if cursor.rowcount > 0:
                    logger.info(f"Trade exit saved: {trade_id}")
                    return True
                else:
                    logger.warning(f"Trade not found: {trade_id}")
                    return False

            except Exception as e:
                logger.error(f"Error saving trade exit: {e}")
                return False

    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trade by ID

        Args:
            trade_id: Trade identifier

        Returns:
            Trade dict or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
            row = cursor.fetchone()

            return dict(row) if row else None

    def get_trades(
        self,
        status: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get trades with filters

        Args:
            status: Filter by status (OPEN, CLOSED)
            symbol: Filter by symbol
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum results

        Returns:
            List of trades
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM trades WHERE 1=1"
            params = []

            if status:
                query += " AND status = ?"
                params.append(status)

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            if start_date:
                query += " AND DATE(entry_time) >= ?"
                params.append(start_date)

            if end_date:
                query += " AND DATE(entry_time) <= ?"
                params.append(end_date)

            query += " ORDER BY entry_time DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            results = cursor.fetchall()

            return [dict(row) for row in results]

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Get all open trades"""
        return self.get_trades(status='OPEN')

    def get_trades_by_date(self, trade_date: date) -> List[Dict[str, Any]]:
        """Get all trades for a specific date"""
        return self.get_trades(start_date=trade_date, end_date=trade_date, limit=1000)

    # ═════════════════════════════════════════════════════════════════
    #                     DAILY STATISTICS
    # ═════════════════════════════════════════════════════════════════

    def save_daily_stats(
        self,
        trade_date: date,
        stats: Dict[str, Any]
    ) -> bool:
        """
        Save daily statistics

        Args:
            trade_date: Trading date
            stats: Statistics dictionary

        Returns:
            True if saved successfully
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO daily_stats (
                        trade_date, total_trades, winning_trades, losing_trades,
                        win_rate, gross_profit, gross_loss, net_pnl,
                        avg_win, avg_loss, avg_r_multiple,
                        largest_win, largest_loss,
                        consecutive_wins, consecutive_losses,
                        max_drawdown, profit_factor,
                        goal_reached, trading_halted, halt_reason,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    trade_date,
                    stats.get('total_trades', 0),
                    stats.get('winning_trades', 0),
                    stats.get('losing_trades', 0),
                    stats.get('win_rate', 0.0),
                    stats.get('gross_profit', 0.0),
                    stats.get('gross_loss', 0.0),
                    stats.get('current_pnl', 0.0),
                    stats.get('avg_win', 0.0),
                    stats.get('avg_loss', 0.0),
                    stats.get('avg_r_multiple', 0.0),
                    stats.get('largest_win', 0.0),
                    stats.get('largest_loss', 0.0),
                    stats.get('consecutive_wins', 0),
                    stats.get('consecutive_losses', 0),
                    stats.get('max_drawdown', 0.0),
                    stats.get('profit_factor', 0.0),
                    stats.get('goal_reached', False),
                    stats.get('is_halted', False),
                    stats.get('halt_reason')
                ))

                logger.info(f"Daily stats saved for {trade_date}")
                return True

            except Exception as e:
                logger.error(f"Error saving daily stats: {e}")
                return False

    def get_daily_stats(
        self,
        trade_date: Optional[date] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get daily statistics

        Args:
            trade_date: Trading date (defaults to today)

        Returns:
            Statistics dict or None
        """
        trade_date = trade_date or date.today()

        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM daily_stats WHERE trade_date = ?",
                (trade_date,)
            )
            row = cursor.fetchone()

            return dict(row) if row else None

    def get_stats_history(
        self,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get historical statistics

        Args:
            days: Number of days

        Returns:
            List of daily stats
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM daily_stats
                ORDER BY trade_date DESC
                LIMIT ?
            """, (days,))

            results = cursor.fetchall()
            return [dict(row) for row in results]

    # ═════════════════════════════════════════════════════════════════
    #                     PATTERN DETECTIONS
    # ═════════════════════════════════════════════════════════════════

    def save_pattern_detection(
        self,
        setup: TradingSetup,
        was_traded: bool = False,
        trade_id: Optional[str] = None
    ) -> int:
        """
        Save pattern detection

        Args:
            setup: TradingSetup object
            was_traded: Whether this pattern was traded
            trade_id: Associated trade ID if traded

        Returns:
            Pattern detection ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute("""
                    INSERT INTO pattern_detections (
                        symbol, detection_time, setup_type, timeframe,
                        entry_price, stop_price, target_1r, target_2r, target_3r,
                        risk_per_share, reward_per_share, risk_reward_ratio,
                        confidence, strength_factors, risk_factors,
                        entry_condition, stop_reason, current_price,
                        was_traded, trade_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    setup.symbol,
                    setup.timestamp,
                    setup.setup_type.value,
                    setup.timeframe,
                    setup.entry_price,
                    setup.stop_price,
                    setup.target_1r,
                    setup.target_2r,
                    setup.target_3r,
                    setup.risk_per_share,
                    setup.reward_per_share,
                    setup.risk_reward_ratio,
                    setup.confidence,
                    json.dumps(setup.strength_factors),
                    json.dumps(setup.risk_factors),
                    setup.entry_condition,
                    setup.stop_reason,
                    setup.current_price,
                    was_traded,
                    trade_id
                ))

                pattern_id = cursor.lastrowid
                logger.info(f"Pattern detection saved: {setup.symbol} {setup.setup_type.value}")
                return pattern_id

            except Exception as e:
                logger.error(f"Error saving pattern detection: {e}")
                return -1

    def get_pattern_detections(
        self,
        symbol: Optional[str] = None,
        setup_type: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        was_traded: Optional[bool] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get pattern detections with filters

        Args:
            symbol: Filter by symbol
            setup_type: Filter by setup type
            start_date: Start date
            end_date: End date
            was_traded: Filter by whether pattern was traded
            limit: Maximum results

        Returns:
            List of pattern detections
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM pattern_detections WHERE 1=1"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            if setup_type:
                query += " AND setup_type = ?"
                params.append(setup_type)

            if start_date:
                query += " AND DATE(detection_time) >= ?"
                params.append(start_date)

            if end_date:
                query += " AND DATE(detection_time) <= ?"
                params.append(end_date)

            if was_traded is not None:
                query += " AND was_traded = ?"
                params.append(was_traded)

            query += " ORDER BY detection_time DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            results = cursor.fetchall()

            patterns = []
            for row in results:
                pattern = dict(row)
                # Parse JSON fields
                pattern['strength_factors'] = json.loads(pattern['strength_factors'])
                pattern['risk_factors'] = json.loads(pattern['risk_factors'])
                patterns.append(pattern)

            return patterns

    # ═════════════════════════════════════════════════════════════════
    #                     ANALYTICS & REPORTING
    # ═════════════════════════════════════════════════════════════════

    def get_performance_summary(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive performance summary

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Performance summary dict
        """
        trades = self.get_trades(
            status='CLOSED',
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )

        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'net_pnl': 0.0,
                'avg_r_multiple': 0.0
            }

        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]

        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = sum(t['pnl'] for t in losing_trades)
        net_pnl = gross_profit + gross_loss

        win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0.0
        avg_win = (gross_profit / len(winning_trades)) if winning_trades else 0.0
        avg_loss = (gross_loss / len(losing_trades)) if losing_trades else 0.0
        avg_r = sum(t['r_multiple'] for t in trades) / len(trades) if trades else 0.0

        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0.0

        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_pnl': net_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_r_multiple': avg_r,
            'profit_factor': profit_factor,
            'largest_win': max((t['pnl'] for t in trades), default=0.0),
            'largest_loss': min((t['pnl'] for t in trades), default=0.0)
        }

    def get_pattern_success_rates(
        self,
        days: int = 30
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get success rates by pattern type

        Args:
            days: Number of days to analyze

        Returns:
            Dict mapping pattern types to success metrics
        """
        start_date = date.today() - timedelta(days=days)
        patterns = self.get_pattern_detections(
            start_date=start_date,
            was_traded=True,
            limit=10000
        )

        # Group by setup type
        pattern_stats = {}
        for pattern in patterns:
            setup_type = pattern['setup_type']

            if setup_type not in pattern_stats:
                pattern_stats[setup_type] = {
                    'total': 0,
                    'traded': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0.0
                }

            pattern_stats[setup_type]['total'] += 1

            if pattern['trade_id']:
                trade = self.get_trade(pattern['trade_id'])
                if trade and trade['status'] == 'CLOSED':
                    pattern_stats[setup_type]['traded'] += 1
                    if trade['pnl'] > 0:
                        pattern_stats[setup_type]['wins'] += 1
                    else:
                        pattern_stats[setup_type]['losses'] += 1
                    pattern_stats[setup_type]['total_pnl'] += trade['pnl']

        # Calculate success rates
        for setup_type, stats in pattern_stats.items():
            traded = stats['traded']
            if traded > 0:
                stats['win_rate'] = (stats['wins'] / traded * 100)
                stats['avg_pnl'] = stats['total_pnl'] / traded
            else:
                stats['win_rate'] = 0.0
                stats['avg_pnl'] = 0.0

        return pattern_stats


# Global database instance
_db_instance: Optional[WarriorDatabase] = None


def get_database(db_path: str = "data/warrior_trading.db") -> WarriorDatabase:
    """
    Get database instance (singleton)

    Args:
        db_path: Path to database file

    Returns:
        WarriorDatabase instance
    """
    global _db_instance

    if _db_instance is None:
        _db_instance = WarriorDatabase(db_path)

    return _db_instance
