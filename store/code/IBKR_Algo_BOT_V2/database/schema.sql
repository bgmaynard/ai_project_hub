-- Warrior Trading Bot - Database Schema
-- SQLite database for trade tracking, error logging, and performance monitoring

-- Trades Table - Track all trade executions
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL, -- 'buy' or 'sell'
    shares INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    exit_price REAL,
    exit_time TIMESTAMP,
    stop_loss REAL,
    take_profit REAL,
    pnl REAL,
    pnl_percent REAL,
    r_multiple REAL, -- How many R achieved
    status TEXT DEFAULT 'open', -- 'open', 'closed', 'stopped', 'target'
    pattern_type TEXT,
    pattern_confidence REAL,
    sentiment_score REAL,
    slippage_entry REAL,
    slippage_exit REAL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Error Log Table - Track all system errors
CREATE TABLE IF NOT EXISTS error_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    error_id TEXT UNIQUE NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    severity TEXT NOT NULL, -- 'debug', 'info', 'warning', 'error', 'critical'
    module TEXT NOT NULL, -- Which module: 'scanner', 'ml', 'risk', etc.
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    stack_trace TEXT,
    context TEXT, -- JSON string with additional context
    resolved BOOLEAN DEFAULT 0,
    resolved_at TIMESTAMP,
    resolution_notes TEXT
);

-- System Events Table - Track important system events
CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type TEXT NOT NULL, -- 'startup', 'shutdown', 'scan_complete', 'daily_reset', etc.
    event_data TEXT, -- JSON string
    duration_ms INTEGER
);

-- Performance Metrics Table - Daily/hourly aggregated stats
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    time_period TEXT NOT NULL, -- 'daily', 'hourly', 'weekly'
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0,
    total_pnl REAL DEFAULT 0,
    gross_profit REAL DEFAULT 0,
    gross_loss REAL DEFAULT 0,
    profit_factor REAL DEFAULT 0,
    avg_win REAL DEFAULT 0,
    avg_loss REAL DEFAULT 0,
    largest_win REAL DEFAULT 0,
    largest_loss REAL DEFAULT 0,
    avg_r_multiple REAL DEFAULT 0,
    max_consecutive_wins INTEGER DEFAULT 0,
    max_consecutive_losses INTEGER DEFAULT 0,
    sharpe_ratio REAL DEFAULT 0,
    max_drawdown REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, time_period)
);

-- Slippage Log Table - Track execution quality
CREATE TABLE IF NOT EXISTS slippage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT UNIQUE NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    expected_price REAL NOT NULL,
    actual_price REAL NOT NULL,
    shares INTEGER NOT NULL,
    slippage_pct REAL NOT NULL,
    slippage_level TEXT NOT NULL, -- 'acceptable', 'warning', 'critical'
    slippage_cost REAL NOT NULL
);

-- User Layouts Table - Save custom dashboard layouts
CREATE TABLE IF NOT EXISTS user_layouts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    layout_id TEXT UNIQUE NOT NULL,
    layout_name TEXT NOT NULL,
    user_id TEXT DEFAULT 'default',
    layout_config TEXT NOT NULL, -- JSON string with widget positions
    is_default BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Watchlist Table - Save custom watchlists
CREATE TABLE IF NOT EXISTS watchlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    watchlist_id TEXT UNIQUE NOT NULL,
    watchlist_name TEXT NOT NULL,
    symbols TEXT NOT NULL, -- JSON array of symbols
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Alerts Table - Track user alerts/notifications
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT UNIQUE NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    alert_type TEXT NOT NULL, -- 'trade', 'error', 'performance', 'system'
    severity TEXT NOT NULL, -- 'info', 'warning', 'critical'
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    acknowledged BOOLEAN DEFAULT 0,
    acknowledged_at TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_error_logs_severity ON error_logs(severity);
CREATE INDEX IF NOT EXISTS idx_error_logs_module ON error_logs(module);
CREATE INDEX IF NOT EXISTS idx_error_logs_timestamp ON error_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_date ON performance_metrics(date);
CREATE INDEX IF NOT EXISTS idx_slippage_symbol ON slippage_log(symbol);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged);

-- Create views for common queries

-- Active trades view
CREATE VIEW IF NOT EXISTS v_active_trades AS
SELECT
    trade_id, symbol, side, shares, entry_price, entry_time,
    stop_loss, take_profit, pattern_type, pattern_confidence,
    (CASE WHEN side = 'buy' THEN
        ((SELECT last_price FROM market_data WHERE market_data.symbol = trades.symbol) - entry_price) / entry_price * 100
    ELSE
        (entry_price - (SELECT last_price FROM market_data WHERE market_data.symbol = trades.symbol)) / entry_price * 100
    END) as unrealized_pnl_pct
FROM trades
WHERE status = 'open';

-- Daily performance summary view
CREATE VIEW IF NOT EXISTS v_daily_summary AS
SELECT
    DATE(entry_time) as trade_date,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
    ROUND(CAST(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 2) as win_rate,
    ROUND(SUM(pnl), 2) as total_pnl,
    ROUND(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 2) as gross_profit,
    ROUND(SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END), 2) as gross_loss,
    ROUND(AVG(CASE WHEN pnl > 0 THEN pnl END), 2) as avg_win,
    ROUND(AVG(CASE WHEN pnl < 0 THEN pnl END), 2) as avg_loss
FROM trades
WHERE status != 'open'
GROUP BY DATE(entry_time)
ORDER BY trade_date DESC;

-- Recent errors view
CREATE VIEW IF NOT EXISTS v_recent_errors AS
SELECT
    error_id, timestamp, severity, module, error_type, error_message,
    resolved
FROM error_logs
WHERE resolved = 0
ORDER BY timestamp DESC
LIMIT 50;
