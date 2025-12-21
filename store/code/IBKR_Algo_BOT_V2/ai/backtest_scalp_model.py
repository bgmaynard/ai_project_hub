"""
Backtest Scalp Model Against Actual Trades
==========================================
Compares Polygon scalp model predictions vs actual trade outcomes.

Analyzes:
- Would the model have correctly predicted winners/losers?
- What signals were missed?
- Optimal entry/exit timing

Usage:
    python -m ai.backtest_scalp_model
"""

import json
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import httpx
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
POLYGON_BASE_URL = "https://api.polygon.io"


class ScalpModelBacktester:
    """Backtests the Polygon scalp model against actual trades."""

    def __init__(self):
        self.model = None
        self.feature_names = []
        self._load_model()
        self.trades = []
        self.results = []

    def _load_model(self):
        """Load the trained scalp model."""
        model_path = Path(__file__).parent / "polygon_scalp_model.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.feature_names = data.get('features', [])
            logger.info("Scalp model loaded successfully")
        else:
            logger.error("Scalp model not found!")

    def load_trades(self, filepath: str = None):
        """Load trade history."""
        if filepath is None:
            filepath = Path(__file__).parent / "scalper_trades.json"

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.trades = data.get('trades', data) if isinstance(data, dict) else data
        logger.info(f"Loaded {len(self.trades)} trades")

    def fetch_minute_bars(self, symbol: str, timestamp: datetime, minutes_before: int = 30) -> Optional[pd.DataFrame]:
        """Fetch minute bars around trade entry time."""
        start_time = timestamp - timedelta(minutes=minutes_before)
        end_time = timestamp + timedelta(minutes=5)

        start_str = start_time.strftime('%Y-%m-%d')
        end_str = end_time.strftime('%Y-%m-%d')

        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range/1/minute/{start_str}/{end_str}"
        params = {
            "apiKey": POLYGON_API_KEY,
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000
        }

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('results'):
                        df = pd.DataFrame(data['results'])
                        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                        df = df.rename(columns={
                            'o': 'Open', 'h': 'High', 'l': 'Low',
                            'c': 'Close', 'v': 'Volume'
                        })
                        # Filter to before entry time
                        df = df[df['timestamp'] <= timestamp]
                        return df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            logger.debug(f"Error fetching {symbol}: {e}")
        return None

    def compute_features(self, df: pd.DataFrame) -> Optional[List[float]]:
        """Compute features for model prediction."""
        if df is None or len(df) < 20:
            return None

        try:
            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            volume = df['Volume'].values

            # Current spike characteristics
            spike_size = (close[-1] / close[-2] - 1) * 100 if len(close) > 1 else 0
            vol_surge = volume[-1] / np.mean(volume[-20:]) if len(volume) > 20 else 1.0

            # Velocity features
            velocity_1m = spike_size
            velocity_5m = ((close[-1] / close[-5]) - 1) * 100 if len(close) > 5 else 0
            accel = velocity_1m - velocity_5m / 5 if velocity_5m != 0 else 0

            # Price position
            day_range = (max(high[-20:]) - min(low[-20:])) if len(high) > 20 else 1
            price_position = (close[-1] - min(low[-20:])) / day_range if day_range > 0 else 0.5

            # Volume profile
            vol_consistency = np.std(volume[-10:]) / np.mean(volume[-10:]) if len(volume) > 10 else 1

            # RSI approximation
            gains = np.diff(close[-15:])
            avg_gain = np.mean(gains[gains > 0]) if np.any(gains > 0) else 0
            avg_loss = abs(np.mean(gains[gains < 0])) if np.any(gains < 0) else 1
            rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 50

            # Price momentum
            ma_5 = np.mean(close[-5:]) if len(close) > 5 else close[-1]
            ma_20 = np.mean(close[-20:]) if len(close) > 20 else close[-1]
            above_vwap = 1 if close[-1] > ma_5 else 0

            # Range analysis
            recent_range = (max(high[-5:]) - min(low[-5:])) / close[-1] * 100 if len(high) > 5 else 0

            features = [
                spike_size, vol_surge, velocity_1m, velocity_5m, accel,
                price_position, rsi, vol_consistency, recent_range, above_vwap,
                ma_5 / ma_20 - 1 if ma_20 > 0 else 0,
                close[-1],
                np.mean(volume[-10:]),
                max(high[-10:]) - min(low[-10:]) if len(high) > 10 else 0,
                len([g for g in gains if g > 0]) / len(gains) if len(gains) > 0 else 0.5,
                spike_size * vol_surge,
                1 if spike_size > 3 else 0,
                1 if vol_surge > 3 else 0,
            ]
            return features
        except Exception as e:
            logger.debug(f"Feature computation error: {e}")
            return None

    def predict(self, features: List[float]) -> Tuple[float, str]:
        """Get model prediction."""
        if self.model is None or features is None:
            return 0.5, "N/A"

        try:
            prob = self.model.predict_proba([features])[0][1]
            verdict = "CONTINUE" if prob > 0.55 else "FADE" if prob < 0.45 else "NEUTRAL"
            return prob, verdict
        except:
            return 0.5, "N/A"

    def backtest(self, limit: int = None):
        """Run backtest on all trades."""
        if not self.trades:
            self.load_trades()

        trades_to_test = self.trades[:limit] if limit else self.trades
        logger.info(f"Backtesting {len(trades_to_test)} trades...")

        self.results = []

        for i, trade in enumerate(trades_to_test):
            symbol = trade.get('symbol', '')
            entry_time_str = trade.get('entry_time', '')
            pnl = trade.get('pnl', 0) or 0
            exit_reason = trade.get('exit_reason', '')
            pnl_pct = trade.get('pnl_percent', 0) or 0

            if not symbol or not entry_time_str:
                continue

            try:
                entry_time = datetime.fromisoformat(entry_time_str.replace('Z', ''))
            except:
                continue

            # Actual outcome
            actual_continue = pnl > 0
            actual_label = "WIN" if actual_continue else "LOSS"

            # Model prediction (would need minute data from Polygon)
            # For backtest, we use the trade's own characteristics
            df = self.fetch_minute_bars(symbol, entry_time)

            if df is not None and len(df) >= 20:
                features = self.compute_features(df)
                prob, verdict = self.predict(features)
            else:
                # Use heuristics from trade data itself
                prob = 0.5
                verdict = "N/A"

            # Compare prediction vs actual
            predicted_continue = prob > 0.5
            correct = (predicted_continue == actual_continue)

            result = {
                'trade_id': trade.get('trade_id', f'{symbol}_{i}'),
                'symbol': symbol,
                'entry_time': entry_time_str,
                'entry_price': trade.get('entry_price', 0),
                'exit_reason': exit_reason,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'actual': actual_label,
                'model_prob': prob,
                'model_verdict': verdict,
                'predicted': "WIN" if predicted_continue else "LOSS",
                'correct': correct,
                'would_have_traded': verdict == "CONTINUE" or verdict == "NEUTRAL"
            }
            self.results.append(result)

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i+1}/{len(trades_to_test)} trades...")

        return self.results

    def generate_report(self) -> Dict:
        """Generate backtest comparison report."""
        if not self.results:
            return {}

        report = {
            'total_trades': len(self.results),
            'actual_trades': {},
            'model_performance': {},
            'filtered_analysis': {},
            'by_exit_reason': {},
            'recommendations': []
        }

        # Actual trade performance
        actual_winners = sum(1 for r in self.results if r['actual'] == 'WIN')
        actual_pnl = sum(r['pnl'] for r in self.results)
        report['actual_trades'] = {
            'total_pnl': actual_pnl,
            'winners': actual_winners,
            'losers': len(self.results) - actual_winners,
            'win_rate': actual_winners / len(self.results) * 100 if self.results else 0
        }

        # Model prediction accuracy
        valid_predictions = [r for r in self.results if r['model_verdict'] != 'N/A']
        correct_predictions = sum(1 for r in valid_predictions if r['correct'])
        report['model_performance'] = {
            'valid_predictions': len(valid_predictions),
            'correct': correct_predictions,
            'accuracy': correct_predictions / len(valid_predictions) * 100 if valid_predictions else 0,
            'avg_prob_winners': np.mean([r['model_prob'] for r in self.results if r['actual'] == 'WIN']) if any(r['actual'] == 'WIN' for r in self.results) else 0,
            'avg_prob_losers': np.mean([r['model_prob'] for r in self.results if r['actual'] == 'LOSS']) if any(r['actual'] == 'LOSS' for r in self.results) else 0
        }

        # What if we only took "CONTINUE" signals?
        continue_signals = [r for r in self.results if r['model_verdict'] == 'CONTINUE']
        if continue_signals:
            continue_pnl = sum(r['pnl'] for r in continue_signals)
            continue_winners = sum(1 for r in continue_signals if r['actual'] == 'WIN')
            report['filtered_analysis']['continue_only'] = {
                'trades': len(continue_signals),
                'pnl': continue_pnl,
                'winners': continue_winners,
                'win_rate': continue_winners / len(continue_signals) * 100
            }

        # What if we avoided "FADE" signals?
        non_fade = [r for r in self.results if r['model_verdict'] != 'FADE']
        if non_fade:
            non_fade_pnl = sum(r['pnl'] for r in non_fade)
            non_fade_winners = sum(1 for r in non_fade if r['actual'] == 'WIN')
            report['filtered_analysis']['avoid_fade'] = {
                'trades': len(non_fade),
                'pnl': non_fade_pnl,
                'winners': non_fade_winners,
                'win_rate': non_fade_winners / len(non_fade) * 100 if non_fade else 0
            }

        # Avoided losses (FADE signals that were actually losses)
        fade_losses = [r for r in self.results if r['model_verdict'] == 'FADE' and r['actual'] == 'LOSS']
        report['filtered_analysis']['avoided_losses'] = {
            'count': len(fade_losses),
            'saved_pnl': abs(sum(r['pnl'] for r in fade_losses))
        }

        # Analysis by exit reason
        by_reason = defaultdict(list)
        for r in self.results:
            by_reason[r['exit_reason']].append(r)

        for reason, trades in by_reason.items():
            correct = sum(1 for t in trades if t['correct'])
            report['by_exit_reason'][reason] = {
                'count': len(trades),
                'pnl': sum(t['pnl'] for t in trades),
                'model_accuracy': correct / len(trades) * 100 if trades else 0,
                'avg_model_prob': np.mean([t['model_prob'] for t in trades])
            }

        # Generate recommendations
        if report['model_performance']['avg_prob_losers'] < 0.45:
            report['recommendations'].append(
                "Model correctly identifies fades - use FADE signals to avoid bad entries"
            )

        if report['filtered_analysis'].get('avoid_fade', {}).get('pnl', 0) > actual_pnl:
            improvement = report['filtered_analysis']['avoid_fade']['pnl'] - actual_pnl
            report['recommendations'].append(
                f"Avoiding FADE signals would have improved P&L by ${improvement:.2f}"
            )

        return report


def main():
    """Run the backtest."""
    logger.info("=" * 60)
    logger.info("SCALP MODEL BACKTEST vs ACTUAL TRADES")
    logger.info("=" * 60)

    backtester = ScalpModelBacktester()

    # Run backtest
    results = backtester.backtest()

    if not results:
        logger.error("No results to report!")
        return

    # Generate report
    report = backtester.generate_report()

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print(f"\n[ACTUAL TRADING PERFORMANCE]")
    actual = report['actual_trades']
    print(f"  Total Trades: {actual.get('winners', 0) + actual.get('losers', 0)}")
    print(f"  Total P&L: ${actual.get('total_pnl', 0):.2f}")
    print(f"  Win Rate: {actual.get('win_rate', 0):.1f}%")

    print(f"\n[MODEL PREDICTION ACCURACY]")
    model = report['model_performance']
    print(f"  Valid Predictions: {model.get('valid_predictions', 0)}")
    print(f"  Accuracy: {model.get('accuracy', 0):.1f}%")
    print(f"  Avg Prob for Winners: {model.get('avg_prob_winners', 0):.2%}")
    print(f"  Avg Prob for Losers: {model.get('avg_prob_losers', 0):.2%}")

    print(f"\n[IF WE USED THE MODEL]")
    filtered = report.get('filtered_analysis', {})

    if 'continue_only' in filtered:
        cont = filtered['continue_only']
        print(f"  CONTINUE signals only:")
        print(f"    Trades: {cont['trades']}, P&L: ${cont['pnl']:.2f}, Win: {cont['win_rate']:.1f}%")

    if 'avoid_fade' in filtered:
        avoid = filtered['avoid_fade']
        print(f"  Avoiding FADE signals:")
        print(f"    Trades: {avoid['trades']}, P&L: ${avoid['pnl']:.2f}, Win: {avoid['win_rate']:.1f}%")

    if 'avoided_losses' in filtered:
        avoided = filtered['avoided_losses']
        print(f"  Would have avoided: {avoided['count']} losing trades (${avoided['saved_pnl']:.2f} saved)")

    print(f"\n[ANALYSIS BY EXIT REASON]")
    for reason, stats in sorted(report.get('by_exit_reason', {}).items(), key=lambda x: -x[1]['count']):
        print(f"  {reason:20s}: {stats['count']:3d} trades, P&L: ${stats['pnl']:8.2f}, Model Acc: {stats['model_accuracy']:.0f}%")

    print(f"\n[RECOMMENDATIONS]")
    for rec in report.get('recommendations', ['No specific recommendations']):
        print(f"  - {rec}")

    # Save detailed results
    output_path = Path(__file__).parent / "backtest_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            'report': report,
            'detailed_results': results
        }, f, indent=2, default=str)

    logger.info(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
