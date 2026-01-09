"""
Trade Journal & Analytics System
=================================
Comprehensive trade logging, performance tracking, and improvement analysis.

Features:
- Full trade logging with context (strategy, pattern, signals, etc.)
- Performance metrics (win rate, profit factor, Sharpe ratio, etc.)
- Daily/weekly/monthly analytics
- Pattern analysis (what works, what doesn't)
- Failure point detection
- Improvement recommendations

Author: Claude Code
"""

import json
import logging
import sqlite3
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "store" / "trade_journal.db"

# Global instance
_journal_instance = None


@dataclass
class TradeEntry:
    """Complete trade record for journaling"""

    # Identification
    id: str = ""
    trade_date: str = ""
    symbol: str = ""

    # Entry details
    entry_time: str = ""
    entry_price: float = 0.0
    entry_reason: str = ""  # Why we entered

    # Exit details
    exit_time: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""  # Why we exited (stop, target, manual, etc.)

    # Position details
    quantity: int = 0
    position_size_usd: float = 0.0
    direction: str = "LONG"  # LONG or SHORT

    # Strategy & Pattern
    strategy: str = ""  # momentum, breakout, reversal, etc.
    pattern: str = ""  # bull_flag, HOD_breakout, etc.
    setup_quality: str = ""  # A+, A, B, C

    # AI Signals at entry
    ai_signal: str = ""  # BULLISH, BEARISH, NEUTRAL
    ai_confidence: float = 0.0
    ai_prediction: str = ""

    # Technical context at entry
    gap_percent: float = 0.0
    relative_volume: float = 0.0
    float_shares: float = 0.0
    market_cap: float = 0.0
    sector: str = ""

    # Price levels
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward_ratio: float = 0.0

    # Results
    pnl: float = 0.0
    pnl_percent: float = 0.0
    is_winner: bool = False
    hold_time_minutes: int = 0

    # Risk metrics
    max_drawdown: float = 0.0  # Worst point during trade
    max_profit: float = 0.0  # Best point during trade
    slippage: float = 0.0  # Entry slippage

    # Market conditions
    market_trend: str = ""  # BULLISH, BEARISH, NEUTRAL
    spy_change: float = 0.0
    vix_level: float = 0.0

    # Notes & Tags
    notes: str = ""
    tags: str = ""  # comma-separated
    mistakes: str = ""  # What went wrong
    lessons: str = ""  # What to learn

    # Metadata
    paper_trade: bool = True
    created_at: str = ""


class TradeJournal:
    """
    Comprehensive trade journaling and analytics system
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self._ensure_db()
        logger.info(f"[JOURNAL] Trade Journal initialized: {self.db_path}")

    def _ensure_db(self):
        """Create database and tables if they don't exist"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main trades table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                trade_date TEXT,
                symbol TEXT,
                entry_time TEXT,
                entry_price REAL,
                entry_reason TEXT,
                exit_time TEXT,
                exit_price REAL,
                exit_reason TEXT,
                quantity INTEGER,
                position_size_usd REAL,
                direction TEXT,
                strategy TEXT,
                pattern TEXT,
                setup_quality TEXT,
                ai_signal TEXT,
                ai_confidence REAL,
                ai_prediction TEXT,
                gap_percent REAL,
                relative_volume REAL,
                float_shares REAL,
                market_cap REAL,
                sector TEXT,
                stop_loss REAL,
                take_profit REAL,
                risk_reward_ratio REAL,
                pnl REAL,
                pnl_percent REAL,
                is_winner INTEGER,
                hold_time_minutes INTEGER,
                max_drawdown REAL,
                max_profit REAL,
                slippage REAL,
                market_trend TEXT,
                spy_change REAL,
                vix_level REAL,
                notes TEXT,
                tags TEXT,
                mistakes TEXT,
                lessons TEXT,
                paper_trade INTEGER,
                created_at TEXT
            )
        """
        )

        # Daily summaries table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_summaries (
                date TEXT PRIMARY KEY,
                total_trades INTEGER,
                winners INTEGER,
                losers INTEGER,
                win_rate REAL,
                total_pnl REAL,
                avg_win REAL,
                avg_loss REAL,
                profit_factor REAL,
                largest_win REAL,
                largest_loss REAL,
                best_strategy TEXT,
                worst_strategy TEXT,
                best_pattern TEXT,
                common_mistakes TEXT,
                market_condition TEXT,
                notes TEXT,
                created_at TEXT
            )
        """
        )

        # Strategy performance table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT,
                period TEXT,
                total_trades INTEGER,
                win_rate REAL,
                total_pnl REAL,
                avg_pnl REAL,
                profit_factor REAL,
                best_market_condition TEXT,
                updated_at TEXT
            )
        """
        )

        # Improvement recommendations table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                category TEXT,
                recommendation TEXT,
                priority TEXT,
                based_on TEXT,
                implemented INTEGER DEFAULT 0,
                created_at TEXT
            )
        """
        )

        conn.commit()
        conn.close()

    # =========================================================================
    # TRADE LOGGING
    # =========================================================================

    def log_trade(self, trade: TradeEntry) -> bool:
        """Log a completed trade to the journal"""
        try:
            # Generate ID if not provided
            if not trade.id:
                trade.id = f"{trade.symbol}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

            if not trade.trade_date:
                trade.trade_date = datetime.now().strftime("%Y-%m-%d")

            if not trade.created_at:
                trade.created_at = datetime.now().isoformat()

            # Calculate derived fields
            if trade.entry_price and trade.exit_price and trade.quantity:
                if trade.direction == "LONG":
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                else:
                    trade.pnl = (trade.entry_price - trade.exit_price) * trade.quantity

                trade.pnl_percent = (
                    (trade.exit_price - trade.entry_price) / trade.entry_price
                ) * 100
                trade.is_winner = trade.pnl > 0

            # Calculate risk/reward if stops are set
            if trade.stop_loss and trade.take_profit and trade.entry_price:
                risk = abs(trade.entry_price - trade.stop_loss)
                reward = abs(trade.take_profit - trade.entry_price)
                trade.risk_reward_ratio = reward / risk if risk > 0 else 0

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO trades VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """,
                (
                    trade.id,
                    trade.trade_date,
                    trade.symbol,
                    trade.entry_time,
                    trade.entry_price,
                    trade.entry_reason,
                    trade.exit_time,
                    trade.exit_price,
                    trade.exit_reason,
                    trade.quantity,
                    trade.position_size_usd,
                    trade.direction,
                    trade.strategy,
                    trade.pattern,
                    trade.setup_quality,
                    trade.ai_signal,
                    trade.ai_confidence,
                    trade.ai_prediction,
                    trade.gap_percent,
                    trade.relative_volume,
                    trade.float_shares,
                    trade.market_cap,
                    trade.sector,
                    trade.stop_loss,
                    trade.take_profit,
                    trade.risk_reward_ratio,
                    trade.pnl,
                    trade.pnl_percent,
                    1 if trade.is_winner else 0,
                    trade.hold_time_minutes,
                    trade.max_drawdown,
                    trade.max_profit,
                    trade.slippage,
                    trade.market_trend,
                    trade.spy_change,
                    trade.vix_level,
                    trade.notes,
                    trade.tags,
                    trade.mistakes,
                    trade.lessons,
                    1 if trade.paper_trade else 0,
                    trade.created_at,
                ),
            )

            conn.commit()
            conn.close()

            logger.info(
                f"[JOURNAL] Logged trade: {trade.id} - {trade.symbol} - P&L: ${trade.pnl:.2f}"
            )
            return True

        except Exception as e:
            logger.error(f"[JOURNAL] Error logging trade: {e}")
            return False

    def get_trades(
        self,
        start_date: str = None,
        end_date: str = None,
        symbol: str = None,
        strategy: str = None,
        winners_only: bool = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get trades with optional filters"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if start_date:
            query += " AND trade_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND trade_date <= ?"
            params.append(end_date)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if winners_only is not None:
            query += " AND is_winner = ?"
            params.append(1 if winners_only else 0)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    # =========================================================================
    # PERFORMANCE METRICS
    # =========================================================================

    def get_performance_metrics(
        self, start_date: str = None, end_date: str = None
    ) -> Dict:
        """Calculate comprehensive performance metrics"""
        trades = self.get_trades(start_date, end_date, limit=10000)

        if not trades:
            return {"error": "No trades found", "total_trades": 0}

        # Basic counts
        total = len(trades)
        winners = [t for t in trades if t["is_winner"]]
        losers = [t for t in trades if not t["is_winner"]]

        # P&L metrics
        pnls = [t["pnl"] for t in trades if t["pnl"] is not None]
        win_pnls = [t["pnl"] for t in winners if t["pnl"] is not None]
        loss_pnls = [t["pnl"] for t in losers if t["pnl"] is not None]

        total_pnl = sum(pnls) if pnls else 0
        avg_win = statistics.mean(win_pnls) if win_pnls else 0
        avg_loss = statistics.mean(loss_pnls) if loss_pnls else 0

        # Win rate
        win_rate = (len(winners) / total * 100) if total > 0 else 0

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else 999.99
        )  # Use finite value for JSON

        # Expectancy
        expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

        # Max drawdown (simplified)
        running_pnl = 0
        peak = 0
        max_dd = 0
        for t in sorted(trades, key=lambda x: x["created_at"]):
            running_pnl += t["pnl"] or 0
            if running_pnl > peak:
                peak = running_pnl
            dd = peak - running_pnl
            if dd > max_dd:
                max_dd = dd

        # Strategy breakdown
        strategy_stats = {}
        for t in trades:
            strat = t["strategy"] or "unknown"
            if strat not in strategy_stats:
                strategy_stats[strat] = {"trades": 0, "wins": 0, "pnl": 0}
            strategy_stats[strat]["trades"] += 1
            if t["is_winner"]:
                strategy_stats[strat]["wins"] += 1
            strategy_stats[strat]["pnl"] += t["pnl"] or 0

        for s in strategy_stats:
            strategy_stats[s]["win_rate"] = (
                strategy_stats[s]["wins"] / strategy_stats[s]["trades"] * 100
                if strategy_stats[s]["trades"] > 0
                else 0
            )

        # Pattern breakdown
        pattern_stats = {}
        for t in trades:
            pat = t["pattern"] or "unknown"
            if pat not in pattern_stats:
                pattern_stats[pat] = {"trades": 0, "wins": 0, "pnl": 0}
            pattern_stats[pat]["trades"] += 1
            if t["is_winner"]:
                pattern_stats[pat]["wins"] += 1
            pattern_stats[pat]["pnl"] += t["pnl"] or 0

        return {
            "period": {"start": start_date or "all_time", "end": end_date or "now"},
            "summary": {
                "total_trades": total,
                "winners": len(winners),
                "losers": len(losers),
                "win_rate": round(win_rate, 2),
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(total_pnl / total, 2) if total > 0 else 0,
            },
            "pnl_metrics": {
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "largest_win": round(max(pnls), 2) if pnls else 0,
                "largest_loss": round(min(pnls), 2) if pnls else 0,
                "profit_factor": round(profit_factor, 2),
                "expectancy": round(expectancy, 2),
            },
            "risk_metrics": {
                "max_drawdown": round(max_dd, 2),
                "avg_risk_reward": round(
                    (
                        statistics.mean(
                            [
                                t["risk_reward_ratio"]
                                for t in trades
                                if t["risk_reward_ratio"]
                            ]
                        )
                        if any(t["risk_reward_ratio"] for t in trades)
                        else 0
                    ),
                    2,
                ),
            },
            "strategy_performance": strategy_stats,
            "pattern_performance": pattern_stats,
            "ai_accuracy": {
                "bullish_signals": len(
                    [t for t in trades if t["ai_signal"] == "BULLISH"]
                ),
                "bullish_wins": len(
                    [
                        t
                        for t in trades
                        if t["ai_signal"] == "BULLISH" and t["is_winner"]
                    ]
                ),
                "avg_ai_confidence_winners": round(
                    (
                        statistics.mean(
                            [t["ai_confidence"] for t in winners if t["ai_confidence"]]
                        )
                        if any(t["ai_confidence"] for t in winners)
                        else 0
                    ),
                    2,
                ),
                "avg_ai_confidence_losers": round(
                    (
                        statistics.mean(
                            [t["ai_confidence"] for t in losers if t["ai_confidence"]]
                        )
                        if any(t["ai_confidence"] for t in losers)
                        else 0
                    ),
                    2,
                ),
            },
        }

    # =========================================================================
    # DAILY ANALYSIS
    # =========================================================================

    def run_daily_analysis(self, date: str = None) -> Dict:
        """Run comprehensive analysis for a day's trades"""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        trades = self.get_trades(start_date=date, end_date=date, limit=1000)

        if not trades:
            return {"date": date, "message": "No trades for this day"}

        analysis = {
            "date": date,
            "trade_count": len(trades),
            "metrics": self.get_performance_metrics(date, date),
            "patterns_used": {},
            "strategies_used": {},
            "common_mistakes": [],
            "what_worked": [],
            "what_didnt_work": [],
            "recommendations": [],
        }

        # Analyze patterns
        for t in trades:
            pat = t["pattern"] or "none"
            if pat not in analysis["patterns_used"]:
                analysis["patterns_used"][pat] = {"count": 0, "wins": 0}
            analysis["patterns_used"][pat]["count"] += 1
            if t["is_winner"]:
                analysis["patterns_used"][pat]["wins"] += 1

        # Analyze strategies
        for t in trades:
            strat = t["strategy"] or "none"
            if strat not in analysis["strategies_used"]:
                analysis["strategies_used"][strat] = {"count": 0, "wins": 0, "pnl": 0}
            analysis["strategies_used"][strat]["count"] += 1
            if t["is_winner"]:
                analysis["strategies_used"][strat]["wins"] += 1
            analysis["strategies_used"][strat]["pnl"] += t["pnl"] or 0

        # Find what worked (strategies with >60% win rate)
        for strat, data in analysis["strategies_used"].items():
            if data["count"] >= 2:  # At least 2 trades
                win_rate = data["wins"] / data["count"] * 100
                if win_rate >= 60:
                    analysis["what_worked"].append(
                        f"{strat}: {win_rate:.0f}% win rate ({data['count']} trades)"
                    )
                elif win_rate < 40:
                    analysis["what_didnt_work"].append(
                        f"{strat}: {win_rate:.0f}% win rate ({data['count']} trades)"
                    )

        # Collect mistakes from losing trades
        mistakes_count = {}
        for t in trades:
            if not t["is_winner"] and t["mistakes"]:
                for m in t["mistakes"].split(","):
                    m = m.strip()
                    if m:
                        mistakes_count[m] = mistakes_count.get(m, 0) + 1

        analysis["common_mistakes"] = [
            f"{m} ({c} times)"
            for m, c in sorted(mistakes_count.items(), key=lambda x: -x[1])[:5]
        ]

        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(trades, analysis)

        # Save daily summary
        self._save_daily_summary(date, analysis)

        return analysis

    def _generate_recommendations(
        self, trades: List[Dict], analysis: Dict
    ) -> List[str]:
        """Generate improvement recommendations based on analysis"""
        recommendations = []
        metrics = analysis.get("metrics", {}).get("summary", {})

        # Win rate recommendations
        win_rate = metrics.get("win_rate", 0)
        if win_rate < 50:
            recommendations.append(
                {
                    "category": "win_rate",
                    "priority": "HIGH",
                    "recommendation": f"Win rate is {win_rate:.0f}%. Consider being more selective with entries. Focus on A+ setups only.",
                }
            )
        elif win_rate > 70:
            recommendations.append(
                {
                    "category": "win_rate",
                    "priority": "INFO",
                    "recommendation": f"Win rate is excellent at {win_rate:.0f}%. Consider if you're taking enough risk for potential gains.",
                }
            )

        # Profit factor recommendations
        pf = analysis.get("metrics", {}).get("pnl_metrics", {}).get("profit_factor", 0)
        if pf < 1.5:
            recommendations.append(
                {
                    "category": "profit_factor",
                    "priority": "HIGH",
                    "recommendation": f"Profit factor is {pf:.2f}. Cut losses faster or let winners run longer.",
                }
            )

        # Average win vs loss
        avg_win = abs(
            analysis.get("metrics", {}).get("pnl_metrics", {}).get("avg_win", 0)
        )
        avg_loss = abs(
            analysis.get("metrics", {}).get("pnl_metrics", {}).get("avg_loss", 0)
        )
        if avg_loss > avg_win:
            recommendations.append(
                {
                    "category": "risk_management",
                    "priority": "HIGH",
                    "recommendation": f"Average loss (${avg_loss:.2f}) exceeds average win (${avg_win:.2f}). Tighten stop losses.",
                }
            )

        # Strategy-specific recommendations
        for strat, data in analysis.get("strategies_used", {}).items():
            if data["count"] >= 3:
                strat_win_rate = data["wins"] / data["count"] * 100
                if strat_win_rate < 40:
                    recommendations.append(
                        {
                            "category": "strategy",
                            "priority": "MEDIUM",
                            "recommendation": f"Strategy '{strat}' has only {strat_win_rate:.0f}% win rate. Review or reduce usage.",
                        }
                    )

        # AI accuracy recommendations
        ai_data = analysis.get("metrics", {}).get("ai_accuracy", {})
        if ai_data.get("bullish_signals", 0) > 0:
            ai_win_rate = (
                ai_data.get("bullish_wins", 0) / ai_data.get("bullish_signals", 1)
            ) * 100
            if ai_win_rate < 50:
                recommendations.append(
                    {
                        "category": "ai_signals",
                        "priority": "MEDIUM",
                        "recommendation": f"AI bullish signals have {ai_win_rate:.0f}% accuracy. Consider retraining model or raising confidence threshold.",
                    }
                )

        return recommendations

    def _save_daily_summary(self, date: str, analysis: Dict):
        """Save daily summary to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            metrics = analysis.get("metrics", {}).get("summary", {})
            pnl_metrics = analysis.get("metrics", {}).get("pnl_metrics", {})

            # Find best/worst strategy
            best_strat = max(
                analysis.get("strategies_used", {}).items(),
                key=lambda x: x[1].get("pnl", 0),
                default=("none", {}),
            )[0]
            worst_strat = min(
                analysis.get("strategies_used", {}).items(),
                key=lambda x: x[1].get("pnl", 0),
                default=("none", {}),
            )[0]

            cursor.execute(
                """
                INSERT OR REPLACE INTO daily_summaries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    date,
                    metrics.get("total_trades", 0),
                    metrics.get("winners", 0),
                    metrics.get("losers", 0),
                    metrics.get("win_rate", 0),
                    metrics.get("total_pnl", 0),
                    pnl_metrics.get("avg_win", 0),
                    pnl_metrics.get("avg_loss", 0),
                    pnl_metrics.get("profit_factor", 0),
                    pnl_metrics.get("largest_win", 0),
                    pnl_metrics.get("largest_loss", 0),
                    best_strat,
                    worst_strat,
                    "",  # best_pattern
                    json.dumps(analysis.get("common_mistakes", [])),
                    "",  # market_condition
                    json.dumps(analysis.get("recommendations", [])),
                    datetime.now().isoformat(),
                ),
            )

            conn.commit()
            conn.close()
            logger.info(f"[JOURNAL] Saved daily summary for {date}")

        except Exception as e:
            logger.error(f"[JOURNAL] Error saving daily summary: {e}")

    # =========================================================================
    # TREND ANALYSIS
    # =========================================================================

    def analyze_trends(self, days: int = 30) -> Dict:
        """Analyze trends over a period"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        trades = self.get_trades(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            limit=10000,
        )

        if not trades:
            return {"error": "No trades in period"}

        # Group by week
        weekly_stats = {}
        for t in trades:
            trade_date = datetime.strptime(t["trade_date"], "%Y-%m-%d")
            week_start = (trade_date - timedelta(days=trade_date.weekday())).strftime(
                "%Y-%m-%d"
            )

            if week_start not in weekly_stats:
                weekly_stats[week_start] = {
                    "trades": 0,
                    "wins": 0,
                    "pnl": 0,
                    "strategies": {},
                }

            weekly_stats[week_start]["trades"] += 1
            if t["is_winner"]:
                weekly_stats[week_start]["wins"] += 1
            weekly_stats[week_start]["pnl"] += t["pnl"] or 0

        # Calculate trends
        weeks = sorted(weekly_stats.keys())
        pnl_trend = [weekly_stats[w]["pnl"] for w in weeks]
        wr_trend = [
            (
                weekly_stats[w]["wins"] / weekly_stats[w]["trades"] * 100
                if weekly_stats[w]["trades"] > 0
                else 0
            )
            for w in weeks
        ]

        # Determine trend direction
        if len(pnl_trend) >= 2:
            pnl_direction = "IMPROVING" if pnl_trend[-1] > pnl_trend[0] else "DECLINING"
            wr_direction = "IMPROVING" if wr_trend[-1] > wr_trend[0] else "DECLINING"
        else:
            pnl_direction = "STABLE"
            wr_direction = "STABLE"

        return {
            "period_days": days,
            "total_trades": len(trades),
            "weekly_breakdown": {
                w: {
                    "trades": weekly_stats[w]["trades"],
                    "win_rate": (
                        round(
                            weekly_stats[w]["wins"] / weekly_stats[w]["trades"] * 100, 1
                        )
                        if weekly_stats[w]["trades"] > 0
                        else 0
                    ),
                    "pnl": round(weekly_stats[w]["pnl"], 2),
                }
                for w in weeks
            },
            "trends": {
                "pnl_direction": pnl_direction,
                "win_rate_direction": wr_direction,
                "consistency": (
                    "HIGH"
                    if statistics.stdev(pnl_trend) < 100
                    else "LOW" if len(pnl_trend) > 1 else "N/A"
                ),
            },
            "insights": self._generate_trend_insights(trades, weekly_stats),
        }

    def _generate_trend_insights(
        self, trades: List[Dict], weekly_stats: Dict
    ) -> List[str]:
        """Generate insights from trend analysis"""
        insights = []

        # Best performing week
        if weekly_stats:
            best_week = max(weekly_stats.items(), key=lambda x: x[1]["pnl"])
            insights.append(
                f"Best week: {best_week[0]} with ${best_week[1]['pnl']:.2f} P&L"
            )

            worst_week = min(weekly_stats.items(), key=lambda x: x[1]["pnl"])
            if worst_week[1]["pnl"] < 0:
                insights.append(
                    f"Worst week: {worst_week[0]} with ${worst_week[1]['pnl']:.2f} loss"
                )

        # Most profitable strategy overall
        strategy_pnl = {}
        for t in trades:
            strat = t["strategy"] or "unknown"
            strategy_pnl[strat] = strategy_pnl.get(strat, 0) + (t["pnl"] or 0)

        if strategy_pnl:
            best_strat = max(strategy_pnl.items(), key=lambda x: x[1])
            insights.append(
                f"Most profitable strategy: {best_strat[0]} (${best_strat[1]:.2f})"
            )

        return insights


def get_trade_journal() -> TradeJournal:
    """Get or create the trade journal instance"""
    global _journal_instance
    if _journal_instance is None:
        _journal_instance = TradeJournal()
    return _journal_instance
