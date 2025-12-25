"""
Signal Strategy Backtest
========================
Backtests the technical signal strategy to see if:
1. Confluence scoring improves trade selection
2. Which signal combinations predict winners
3. Entry/exit based on signals improves P&L

Uses historical Polygon data and simulates scalp trades.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import httpx
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Import our modules
try:
    from ai.technical_signals import TechnicalSignalAnalyzer, SignalState
    HAS_DEPS = True
except ImportError as e:
    print(f"Import error: {e}")
    HAS_DEPS = False

# API base URL
API_BASE = "http://localhost:9100"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """A simulated trade"""
    symbol: str
    entry_time: int
    entry_price: float
    exit_time: int
    exit_price: float
    pnl: float
    pnl_pct: float
    hold_bars: int
    entry_reason: str
    exit_reason: str
    # Signal state at entry
    confluence_score: float
    signal_bias: str
    ema_bullish: bool
    macd_bullish: bool
    price_above_vwap: bool
    ema_crossover: str
    macd_crossover: str
    vwap_crossover: str
    candle_momentum: str


class SignalBacktester:
    """
    Backtests trading strategies using technical signals.
    """

    def __init__(self):
        self.analyzer = TechnicalSignalAnalyzer()
        self.client = httpx.Client(timeout=30.0)

        # Trade parameters (matching scalper)
        self.profit_target_pct = 3.0
        self.stop_loss_pct = 3.0
        self.trailing_stop_pct = 1.5
        self.max_hold_bars = 36  # 3 min at 5s bars, or 36 bars at 5m = 3 hours

        # Results storage
        self.trades: List[BacktestTrade] = []
        self.signal_correlations = {}

    def fetch_ohlc(self, symbol: str, days: int = 30, timeframe: str = "5m") -> List[Dict]:
        """Fetch OHLC data from API server"""
        try:
            # Use the charts API which fetches from Polygon
            resp = self.client.get(
                f"{API_BASE}/api/charts/ohlc/{symbol}",
                params={"timeframe": timeframe, "days": days}
            )

            if resp.status_code != 200:
                logger.warning(f"API error for {symbol}: {resp.status_code}")
                return []

            data = resp.json()

            if not data.get("success"):
                logger.warning(f"API failed for {symbol}: {data.get('error')}")
                return []

            ohlc = data.get("data", [])
            volume_data = data.get("volume", [])

            # Merge volume into OHLC data
            vol_by_time = {v["time"]: v["value"] for v in volume_data}
            for bar in ohlc:
                bar["volume"] = vol_by_time.get(bar["time"], 0)

            return ohlc

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return []

    def simulate_trade(self, df: pd.DataFrame, entry_idx: int, signal: SignalState) -> Optional[BacktestTrade]:
        """
        Simulate a trade from entry point.
        Uses trailing stop logic similar to scalper.
        """
        if entry_idx >= len(df) - 1:
            return None

        entry_bar = df.iloc[entry_idx]
        entry_price = entry_bar['close']
        entry_time = int(entry_bar['time'])

        # Track position
        high_water = entry_price
        exit_idx = None
        exit_reason = "MAX_HOLD"

        # Simulate bar by bar
        for i in range(entry_idx + 1, min(entry_idx + self.max_hold_bars + 1, len(df))):
            bar = df.iloc[i]
            current_price = bar['close']
            bar_high = bar['high']
            bar_low = bar['low']

            # Update high water mark
            if current_price > high_water:
                high_water = current_price

            # Calculate P&L
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            drawdown_from_high = ((high_water - current_price) / high_water) * 100 if high_water > 0 else 0

            # Check stop loss (use bar low for realism)
            low_pnl = ((bar_low - entry_price) / entry_price) * 100
            if low_pnl <= -self.stop_loss_pct:
                exit_idx = i
                exit_reason = "STOP_LOSS"
                # Exit at stop price, not bar close
                current_price = entry_price * (1 - self.stop_loss_pct / 100)
                break

            # Check if we hit profit target (activates trailing)
            high_pnl = ((bar_high - entry_price) / entry_price) * 100
            if high_pnl >= self.profit_target_pct:
                # Now in trailing mode - check if trail triggered
                if drawdown_from_high >= self.trailing_stop_pct:
                    exit_idx = i
                    exit_reason = "TRAILING_STOP"
                    break

            # Check for signal-based exit
            if i >= entry_idx + 3:  # Give it a few bars
                # Calculate signal at this bar
                window_start = max(0, i - 200)
                window_data = df.iloc[window_start:i+1].to_dict('records')
                current_signal = self.analyzer.analyze(signal.symbol, window_data, signal.timeframe)

                if current_signal:
                    # Exit on bearish crossover if in profit
                    if pnl_pct > 0.5:
                        if current_signal.ema_crossover == "BEARISH":
                            exit_idx = i
                            exit_reason = "EMA_CROSS_EXIT"
                            break
                        if current_signal.macd_crossover == "BEARISH":
                            exit_idx = i
                            exit_reason = "MACD_CROSS_EXIT"
                            break
                        if current_signal.confluence_score < 30:
                            exit_idx = i
                            exit_reason = "CONFLUENCE_DROP"
                            break

        # If no exit found, exit at max hold
        if exit_idx is None:
            exit_idx = min(entry_idx + self.max_hold_bars, len(df) - 1)

        exit_bar = df.iloc[exit_idx]
        exit_price = exit_bar['close']
        exit_time = int(exit_bar['time'])

        pnl = exit_price - entry_price
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100

        return BacktestTrade(
            symbol=signal.symbol,
            entry_time=entry_time,
            entry_price=round(entry_price, 4),
            exit_time=exit_time,
            exit_price=round(exit_price, 4),
            pnl=round(pnl, 4),
            pnl_pct=round(pnl_pct, 2),
            hold_bars=exit_idx - entry_idx,
            entry_reason="SIGNAL",
            exit_reason=exit_reason,
            confluence_score=signal.confluence_score,
            signal_bias=signal.signal_bias,
            ema_bullish=signal.ema_bullish,
            macd_bullish=signal.macd_bullish,
            price_above_vwap=signal.price_above_vwap,
            ema_crossover=signal.ema_crossover,
            macd_crossover=signal.macd_crossover,
            vwap_crossover=signal.vwap_crossover,
            candle_momentum=signal.candle_momentum
        )

    def run_backtest(self, symbols: List[str], days: int = 30,
                     min_confluence: float = 0, timeframe: str = "5m") -> Dict:
        """
        Run backtest on multiple symbols.

        Args:
            symbols: List of stock symbols
            days: Days of history to test
            min_confluence: Minimum confluence score to take trade (0-100)
            timeframe: OHLC timeframe

        Returns:
            Backtest results dictionary
        """
        self.trades = []
        all_signals = []

        for symbol in symbols:
            logger.info(f"Backtesting {symbol}...")

            # Fetch data
            ohlc = self.fetch_ohlc(symbol, days, timeframe)
            if len(ohlc) < 100:
                logger.warning(f"Insufficient data for {symbol}: {len(ohlc)} bars")
                continue

            df = pd.DataFrame(ohlc)

            # Scan through data looking for entry signals
            lookback = 50  # Minimum bars needed for indicators

            for i in range(lookback, len(df) - self.max_hold_bars):
                # Get signal at this bar
                window = df.iloc[max(0, i-200):i+1].to_dict('records')
                signal = self.analyzer.analyze(symbol, window, timeframe)

                if not signal:
                    continue

                all_signals.append(signal)

                # Check entry conditions
                should_enter = False

                # Strategy 1: Enter on bullish crossover with confluence
                if signal.confluence_score >= min_confluence:
                    if signal.ema_crossover == "BULLISH":
                        should_enter = True
                    elif signal.macd_crossover == "BULLISH" and signal.ema_bullish:
                        should_enter = True
                    elif signal.vwap_crossover == "BULLISH" and signal.ema_bullish and signal.macd_bullish:
                        should_enter = True

                if should_enter:
                    # Simulate the trade
                    trade = self.simulate_trade(df, i, signal)
                    if trade:
                        self.trades.append(trade)
                        # Skip ahead to avoid overlapping trades
                        i += trade.hold_bars

        # Calculate results
        results = self.analyze_results(all_signals)
        return results

    def analyze_results(self, all_signals: List[SignalState]) -> Dict:
        """Analyze backtest results and correlate signals with outcomes"""

        if not self.trades:
            return {"error": "No trades generated", "trade_count": 0}

        # Basic stats
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in self.trades)
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0

        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Correlation analysis
        correlations = self.calculate_correlations()

        # Confluence bucket analysis
        confluence_analysis = self.analyze_by_confluence()

        # Signal combination analysis
        combo_analysis = self.analyze_signal_combinations()

        # Exit reason analysis
        exit_analysis = self.analyze_exits()

        results = {
            "summary": {
                "total_trades": len(self.trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(win_rate, 1),
                "total_pnl": round(total_pnl, 2),
                "avg_win_pct": round(avg_win, 2),
                "avg_loss_pct": round(avg_loss, 2),
                "profit_factor": round(profit_factor, 2),
                "avg_hold_bars": round(np.mean([t.hold_bars for t in self.trades]), 1)
            },
            "correlations": correlations,
            "confluence_buckets": confluence_analysis,
            "signal_combinations": combo_analysis,
            "exit_reasons": exit_analysis,
            "trades": [asdict(t) for t in self.trades[-20:]]  # Last 20 trades
        }

        return results

    def calculate_correlations(self) -> Dict:
        """Calculate correlation between signals and trade outcomes"""

        if len(self.trades) < 10:
            return {"error": "Not enough trades for correlation"}

        # Create DataFrame for correlation
        df = pd.DataFrame([{
            "pnl": t.pnl,
            "win": 1 if t.pnl > 0 else 0,
            "confluence": t.confluence_score,
            "ema_bullish": 1 if t.ema_bullish else 0,
            "macd_bullish": 1 if t.macd_bullish else 0,
            "above_vwap": 1 if t.price_above_vwap else 0,
            "ema_cross": 1 if t.ema_crossover == "BULLISH" else 0,
            "macd_cross": 1 if t.macd_crossover == "BULLISH" else 0,
            "vwap_cross": 1 if t.vwap_crossover == "BULLISH" else 0,
            "momentum_building": 1 if t.candle_momentum == "BUILDING" else 0
        } for t in self.trades])

        # Correlation with win
        win_corr = df.corr()['win'].drop('win').to_dict()

        # Correlation with P&L
        pnl_corr = df.corr()['pnl'].drop('pnl').to_dict()

        return {
            "win_correlation": {k: round(v, 3) for k, v in win_corr.items()},
            "pnl_correlation": {k: round(v, 3) for k, v in pnl_corr.items()},
            "interpretation": self.interpret_correlations(win_corr)
        }

    def interpret_correlations(self, correlations: Dict) -> List[str]:
        """Interpret correlation results"""
        insights = []

        for signal, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(corr) < 0.05:
                continue

            direction = "positively" if corr > 0 else "negatively"
            strength = "strongly" if abs(corr) > 0.3 else "moderately" if abs(corr) > 0.15 else "weakly"

            insights.append(f"{signal} is {strength} {direction} correlated with wins ({corr:.3f})")

        return insights[:5]  # Top 5 insights

    def analyze_by_confluence(self) -> Dict:
        """Analyze win rate by confluence score buckets"""

        buckets = {
            "0-40": {"trades": [], "label": "Low (0-40%)"},
            "40-60": {"trades": [], "label": "Medium (40-60%)"},
            "60-80": {"trades": [], "label": "High (60-80%)"},
            "80-100": {"trades": [], "label": "Very High (80-100%)"}
        }

        for t in self.trades:
            if t.confluence_score < 40:
                buckets["0-40"]["trades"].append(t)
            elif t.confluence_score < 60:
                buckets["40-60"]["trades"].append(t)
            elif t.confluence_score < 80:
                buckets["60-80"]["trades"].append(t)
            else:
                buckets["80-100"]["trades"].append(t)

        results = {}
        for key, data in buckets.items():
            trades = data["trades"]
            if not trades:
                results[key] = {"count": 0, "win_rate": 0, "avg_pnl": 0}
                continue

            wins = len([t for t in trades if t.pnl > 0])
            results[key] = {
                "label": data["label"],
                "count": len(trades),
                "win_rate": round(wins / len(trades) * 100, 1),
                "avg_pnl_pct": round(np.mean([t.pnl_pct for t in trades]), 2),
                "total_pnl": round(sum(t.pnl for t in trades), 2)
            }

        return results

    def analyze_signal_combinations(self) -> Dict:
        """Analyze which signal combinations work best"""

        combos = defaultdict(list)

        for t in self.trades:
            # Create combo key
            signals = []
            if t.ema_bullish:
                signals.append("EMA")
            if t.macd_bullish:
                signals.append("MACD")
            if t.price_above_vwap:
                signals.append("VWAP")
            if t.candle_momentum == "BUILDING":
                signals.append("MOM")

            key = "+".join(sorted(signals)) if signals else "NONE"
            combos[key].append(t)

        results = {}
        for key, trades in combos.items():
            if len(trades) < 3:
                continue

            wins = len([t for t in trades if t.pnl > 0])
            results[key] = {
                "count": len(trades),
                "win_rate": round(wins / len(trades) * 100, 1),
                "avg_pnl_pct": round(np.mean([t.pnl_pct for t in trades]), 2),
                "total_pnl": round(sum(t.pnl for t in trades), 2)
            }

        # Sort by win rate
        return dict(sorted(results.items(), key=lambda x: x[1]["win_rate"], reverse=True))

    def analyze_exits(self) -> Dict:
        """Analyze exit reasons"""

        exits = defaultdict(list)
        for t in self.trades:
            exits[t.exit_reason].append(t)

        results = {}
        for reason, trades in exits.items():
            wins = len([t for t in trades if t.pnl > 0])
            results[reason] = {
                "count": len(trades),
                "win_rate": round(wins / len(trades) * 100, 1) if trades else 0,
                "avg_pnl_pct": round(np.mean([t.pnl_pct for t in trades]), 2) if trades else 0
            }

        return results

    def compare_confluence_thresholds(self, symbols: List[str], days: int = 30) -> Dict:
        """
        Compare results at different confluence thresholds.
        This shows if higher confluence = better results.
        """
        thresholds = [0, 40, 50, 60, 70, 80]
        comparison = {}

        for thresh in thresholds:
            logger.info(f"Testing confluence threshold: {thresh}%")
            results = self.run_backtest(symbols, days, min_confluence=thresh)

            if "error" in results:
                comparison[f"{thresh}%"] = {"error": results["error"]}
            else:
                comparison[f"{thresh}%"] = {
                    "trades": results["summary"]["total_trades"],
                    "win_rate": results["summary"]["win_rate"],
                    "profit_factor": results["summary"]["profit_factor"],
                    "total_pnl": results["summary"]["total_pnl"],
                    "avg_pnl": results["summary"]["total_pnl"] / results["summary"]["total_trades"] if results["summary"]["total_trades"] > 0 else 0
                }

        return comparison


def run_backtest_analysis(symbols: List[str] = None, days: int = 30):
    """Main function to run backtest analysis"""

    if not HAS_DEPS:
        print("ERROR: Missing dependencies")
        return None

    if symbols is None:
        # Default momentum stocks
        symbols = ["AAPL", "TSLA", "NVDA", "AMD", "SPY", "QQQ", "META", "MSFT"]

    print("="*70)
    print("TECHNICAL SIGNAL BACKTEST")
    print("="*70)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Days: {days}")
    print(f"Timeframe: 5m")
    print("="*70)

    backtester = SignalBacktester()

    # Run main backtest with 50% confluence minimum
    print("\n[1] Running backtest with 50% minimum confluence...")
    results = backtester.run_backtest(symbols, days, min_confluence=50)

    if "error" in results:
        print(f"Error: {results['error']}")
        return results

    # Print summary
    s = results["summary"]
    print(f"\n[SUMMARY]")
    print(f"   Total Trades: {s['total_trades']}")
    print(f"   Win Rate: {s['win_rate']}%")
    print(f"   Profit Factor: {s['profit_factor']}")
    print(f"   Total P&L: ${s['total_pnl']:.2f}")
    print(f"   Avg Win: +{s['avg_win_pct']}%")
    print(f"   Avg Loss: {s['avg_loss_pct']}%")
    print(f"   Avg Hold: {s['avg_hold_bars']} bars")

    # Print correlations
    print(f"\n[SIGNAL CORRELATIONS]")
    if "interpretation" in results.get("correlations", {}):
        for insight in results["correlations"]["interpretation"]:
            print(f"   * {insight}")

    # Print confluence analysis
    print(f"\n[CONFLUENCE ANALYSIS]")
    for bucket, data in results.get("confluence_buckets", {}).items():
        if data.get("count", 0) > 0:
            print(f"   {data.get('label', bucket)}: {data['count']} trades, {data['win_rate']}% win rate, {data['avg_pnl_pct']:+.2f}% avg")

    # Print best signal combinations
    print(f"\n[BEST SIGNAL COMBINATIONS]")
    combos = results.get("signal_combinations", {})
    for combo, data in list(combos.items())[:5]:
        print(f"   {combo}: {data['count']} trades, {data['win_rate']}% win rate, ${data['total_pnl']:.2f} P&L")

    # Print exit analysis
    print(f"\n[EXIT ANALYSIS]")
    for reason, data in results.get("exit_reasons", {}).items():
        print(f"   {reason}: {data['count']} trades, {data['win_rate']}% win rate, {data['avg_pnl_pct']:+.2f}% avg")

    # Compare thresholds
    print(f"\n[2] Comparing confluence thresholds...")
    comparison = backtester.compare_confluence_thresholds(symbols, days)

    print(f"\n[THRESHOLD COMPARISON]")
    print(f"   {'Threshold':<12} {'Trades':<8} {'Win Rate':<10} {'PF':<6} {'Total P&L':<12}")
    print(f"   {'-'*50}")
    for thresh, data in comparison.items():
        if "error" not in data:
            print(f"   {thresh:<12} {data['trades']:<8} {data['win_rate']:<10.1f} {data['profit_factor']:<6.2f} ${data['total_pnl']:<12.2f}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "symbols": symbols,
        "days": days,
        "results": results,
        "threshold_comparison": comparison
    }

    output_path = os.path.join(os.path.dirname(__file__), "signal_backtest_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[OK] Results saved to {output_path}")

    return output


if __name__ == "__main__":
    # Run with default symbols
    symbols = ["AAPL", "TSLA", "NVDA", "AMD", "SPY", "QQQ"]
    run_backtest_analysis(symbols, days=30)
