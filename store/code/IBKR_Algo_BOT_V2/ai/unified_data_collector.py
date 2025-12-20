"""
Unified Data Collector
======================
Collects and stores market data from Schwab for:
1. Historical backtesting (minute bars)
2. Live session capture (cumulative analysis)
3. Cross-utilization between backtest and live

Data is stored in SQLite for persistence and fast querying.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)

# Data storage path
DATA_DIR = Path(__file__).parent / "market_data"
DB_FILE = DATA_DIR / "unified_data.db"


class UnifiedDataCollector:
    """
    Collects market data from multiple sources for analysis.

    - Historical: Schwab API minute bars (6 months)
    - Live: Real-time capture during sessions
    - Storage: SQLite for persistence
    """

    def __init__(self):
        self.db_path = DB_FILE
        self._ensure_data_dir()
        self._init_db()
        logger.info(f"UnifiedDataCollector initialized - DB: {self.db_path}")

    def _ensure_data_dir(self):
        """Create data directory if needed"""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _init_db(self):
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Minute bars table (historical + live)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS minute_bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                source TEXT DEFAULT 'schwab',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        """)

        # Daily bars table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                source TEXT DEFAULT 'schwab',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        """)

        # Live quotes capture (tick-level during sessions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                price REAL,
                bid REAL,
                ask REAL,
                volume INTEGER,
                change_percent REAL,
                session_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Trade signals captured
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                signal_type TEXT,
                price REAL,
                momentum REAL,
                volume_surge REAL,
                spread_percent REAL,
                rsi REAL,
                vwap_position TEXT,
                spy_direction TEXT,
                time_of_day TEXT,
                float_shares REAL,
                float_rotation REAL,
                outcome TEXT,
                pnl REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Session summary
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                date TEXT,
                start_time TEXT,
                end_time TEXT,
                symbols_watched INTEGER,
                total_signals INTEGER,
                trades_taken INTEGER,
                total_pnl REAL,
                win_rate REAL,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_minute_symbol_ts ON minute_bars(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_symbol_date ON daily_bars(symbol, date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quotes_symbol_ts ON live_quotes(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON trade_signals(symbol)")

        conn.commit()
        conn.close()
        logger.info("Database schema initialized")

    async def fetch_schwab_history(self, symbol: str, days: int = 30) -> List[Dict]:
        """
        Fetch historical minute bars from Schwab API.

        Args:
            symbol: Stock symbol
            days: Number of days to fetch

        Returns:
            List of bar dicts
        """
        import httpx

        bars = []
        try:
            # Use our existing Schwab market data endpoint
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:9100/api/schwab/history/{symbol}",
                    params={"days": days, "frequency": "minute"},
                    timeout=30.0
                )
                if response.status_code == 200:
                    data = response.json()
                    bars = data.get("bars", [])
                    logger.info(f"Fetched {len(bars)} minute bars for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching Schwab history for {symbol}: {e}")

        return bars

    def store_minute_bars(self, symbol: str, bars: List[Dict], source: str = "schwab"):
        """Store minute bars to database"""
        if not bars:
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        inserted = 0
        for bar in bars:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO minute_bars
                    (symbol, timestamp, open, high, low, close, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol.upper(),
                    bar.get("timestamp") or bar.get("datetime"),
                    bar.get("open"),
                    bar.get("high"),
                    bar.get("low"),
                    bar.get("close"),
                    bar.get("volume"),
                    source
                ))
                inserted += 1
            except Exception as e:
                logger.debug(f"Bar insert error: {e}")

        conn.commit()
        conn.close()
        logger.info(f"Stored {inserted} minute bars for {symbol}")
        return inserted

    def capture_live_quote(self, symbol: str, quote: Dict, session_id: str = None):
        """Capture a live quote during trading session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO live_quotes
                (symbol, timestamp, price, bid, ask, volume, change_percent, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol.upper(),
                datetime.now().isoformat(),
                quote.get("price") or quote.get("last"),
                quote.get("bid"),
                quote.get("ask"),
                quote.get("volume"),
                quote.get("change_percent"),
                session_id or datetime.now().strftime("%Y%m%d")
            ))
            conn.commit()
        except Exception as e:
            logger.debug(f"Quote capture error: {e}")
        finally:
            conn.close()

    def store_trade_signal(self, signal: Dict):
        """Store a trade signal with all secondary triggers"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO trade_signals
                (symbol, timestamp, signal_type, price, momentum, volume_surge,
                 spread_percent, rsi, vwap_position, spy_direction, time_of_day,
                 float_shares, float_rotation, outcome, pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.get("symbol"),
                signal.get("timestamp") or datetime.now().isoformat(),
                signal.get("signal_type") or signal.get("type"),
                signal.get("price"),
                signal.get("momentum"),
                signal.get("volume_surge"),
                signal.get("spread_percent") or signal.get("spread_at_entry"),
                signal.get("rsi") or signal.get("rsi_at_entry"),
                signal.get("vwap_position"),
                signal.get("spy_direction"),
                signal.get("time_of_day"),
                signal.get("float_shares"),
                signal.get("float_rotation"),
                signal.get("outcome"),
                signal.get("pnl")
            ))
            conn.commit()
            logger.debug(f"Stored trade signal for {signal.get('symbol')}")
        except Exception as e:
            logger.error(f"Signal store error: {e}")
        finally:
            conn.close()

    def get_minute_bars(self, symbol: str, start_date: str = None, end_date: str = None) -> List[Dict]:
        """
        Get minute bars from database for backtesting.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of bar dicts
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM minute_bars WHERE symbol = ?"
        params = [symbol.upper()]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date + "T23:59:59")

        query += " ORDER BY timestamp"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_trade_signals(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get trade signals for analysis"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if symbol:
            cursor.execute(
                "SELECT * FROM trade_signals WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                (symbol.upper(), limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM trade_signals ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def analyze_signal_correlations(self) -> Dict:
        """
        Analyze correlations between signal characteristics and outcomes.

        Returns insights on what predicts winning trades.
        """
        signals = self.get_trade_signals(limit=1000)

        if len(signals) < 10:
            return {"message": "Need at least 10 signals for analysis", "count": len(signals)}

        wins = [s for s in signals if (s.get("pnl") or 0) > 0]
        losses = [s for s in signals if (s.get("pnl") or 0) <= 0]

        analysis = {
            "total_signals": len(signals),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(signals) * 100, 1) if signals else 0,
            "correlations": {}
        }

        # Momentum analysis
        if wins and losses:
            avg_win_momentum = sum(s.get("momentum") or 0 for s in wins) / len(wins)
            avg_loss_momentum = sum(s.get("momentum") or 0 for s in losses) / len(losses)
            analysis["correlations"]["momentum"] = {
                "avg_winning": round(avg_win_momentum, 2),
                "avg_losing": round(avg_loss_momentum, 2),
                "insight": "Higher momentum better" if avg_win_momentum > avg_loss_momentum else "Check momentum threshold"
            }

        # Time of day analysis
        time_stats = {}
        for s in signals:
            tod = s.get("time_of_day") or "unknown"
            if tod not in time_stats:
                time_stats[tod] = {"wins": 0, "losses": 0, "pnl": 0}
            if (s.get("pnl") or 0) > 0:
                time_stats[tod]["wins"] += 1
            else:
                time_stats[tod]["losses"] += 1
            time_stats[tod]["pnl"] += s.get("pnl") or 0

        for tod, stats in time_stats.items():
            total = stats["wins"] + stats["losses"]
            stats["win_rate"] = round(stats["wins"] / total * 100, 1) if total > 0 else 0
            stats["avg_pnl"] = round(stats["pnl"] / total, 2) if total > 0 else 0

        analysis["correlations"]["time_of_day"] = time_stats

        # SPY direction analysis
        spy_stats = {}
        for s in signals:
            spy = s.get("spy_direction") or "unknown"
            if spy not in spy_stats:
                spy_stats[spy] = {"wins": 0, "losses": 0}
            if (s.get("pnl") or 0) > 0:
                spy_stats[spy]["wins"] += 1
            else:
                spy_stats[spy]["losses"] += 1

        for direction, stats in spy_stats.items():
            total = stats["wins"] + stats["losses"]
            stats["win_rate"] = round(stats["wins"] / total * 100, 1) if total > 0 else 0

        analysis["correlations"]["spy_direction"] = spy_stats

        return analysis

    def get_data_summary(self) -> Dict:
        """Get summary of stored data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Count records
        cursor.execute("SELECT COUNT(*) FROM minute_bars")
        minute_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM daily_bars")
        daily_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM live_quotes")
        quote_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM trade_signals")
        signal_count = cursor.fetchone()[0]

        # Unique symbols
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM minute_bars")
        symbols_with_bars = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM trade_signals")
        symbols_with_signals = cursor.fetchone()[0]

        # Date range
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM minute_bars")
        bar_range = cursor.fetchone()

        conn.close()

        return {
            "minute_bars": minute_count,
            "daily_bars": daily_count,
            "live_quotes": quote_count,
            "trade_signals": signal_count,
            "symbols_with_bars": symbols_with_bars,
            "symbols_with_signals": symbols_with_signals,
            "data_range": {
                "start": bar_range[0] if bar_range else None,
                "end": bar_range[1] if bar_range else None
            },
            "db_path": str(self.db_path)
        }

    def export_for_pybroker(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """
        Export data in format suitable for PyBroker backtesting.

        Returns dict with symbol -> DataFrame-ready data
        """
        import pandas as pd

        result = {}
        for symbol in symbols:
            bars = self.get_minute_bars(symbol, start_date, end_date)
            if bars:
                df = pd.DataFrame(bars)
                df['date'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('date')
                df = df[['open', 'high', 'low', 'close', 'volume']]
                result[symbol] = df
                logger.info(f"Exported {len(df)} bars for {symbol}")

        return result


# Singleton instance
_collector: Optional[UnifiedDataCollector] = None


def get_data_collector() -> UnifiedDataCollector:
    """Get or create data collector singleton"""
    global _collector
    if _collector is None:
        _collector = UnifiedDataCollector()
    return _collector


# API helper functions
async def capture_quote(symbol: str, quote: Dict):
    """Quick helper to capture a quote"""
    collector = get_data_collector()
    collector.capture_live_quote(symbol, quote)


def store_signal(signal: Dict):
    """Quick helper to store a signal"""
    collector = get_data_collector()
    collector.store_trade_signal(signal)


def get_correlations() -> Dict:
    """Quick helper to get correlation analysis"""
    collector = get_data_collector()
    return collector.analyze_signal_correlations()


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    collector = get_data_collector()
    print(f"\nData Summary: {collector.get_data_summary()}")

    # Test signal storage
    test_signal = {
        "symbol": "TEST",
        "signal_type": "momentum_spike",
        "price": 5.50,
        "momentum": 8.5,
        "volume_surge": 3.2,
        "time_of_day": "pre_market_late",
        "spy_direction": "up",
        "pnl": 15.50
    }
    collector.store_trade_signal(test_signal)

    print(f"\nUpdated Summary: {collector.get_data_summary()}")
    print(f"\nCorrelation Analysis: {collector.analyze_signal_correlations()}")
