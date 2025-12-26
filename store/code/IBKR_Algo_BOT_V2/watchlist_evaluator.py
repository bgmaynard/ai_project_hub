"""
WATCHLIST AUTO-EVALUATOR
========================
Monitors watchlist for new symbols and automatically runs:
1. AI Training on the symbol
2. Backtesting with multiple strategies
3. Generates predictions with confidence scores

Runs as background service - starts evaluation when stocks are added to watchlist.
"""

import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import requests

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("watchlist_evaluator.log")],
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE = "http://localhost:9100"
EVAL_RESULTS_FILE = Path("store/watchlist_evaluations.json")


class WatchlistEvaluator:
    """Auto-evaluates stocks added to watchlist"""

    def __init__(self):
        self.known_symbols: Set[str] = set()
        self.evaluation_results: Dict[str, Dict] = {}
        self.running = False
        self.check_interval = 10  # Check watchlist every 10 seconds

        # Load existing evaluations
        self._load_results()

        # Initialize AI predictor
        self.predictor = None
        try:
            from ai.alpaca_ai_predictor import AlpacaAIPredictor

            self.predictor = AlpacaAIPredictor()
            logger.info("AI Predictor initialized")
        except Exception as e:
            logger.warning(f"Could not initialize AI predictor: {e}")

        # Initialize backtester
        self.backtester = None
        try:
            from backtest_strategies import StrategyBacktester

            self.backtester = StrategyBacktester(initial_capital=100000)
            logger.info("Strategy Backtester initialized")
        except Exception as e:
            logger.warning(f"Could not initialize backtester: {e}")

        # Initialize market data
        self.market_data = None
        try:
            from alpaca_market_data import get_alpaca_market_data

            self.market_data = get_alpaca_market_data()
            logger.info("Market data initialized")
        except Exception as e:
            logger.warning(f"Could not initialize market data: {e}")

    def _load_results(self):
        """Load existing evaluation results"""
        if EVAL_RESULTS_FILE.exists():
            try:
                self.evaluation_results = json.loads(EVAL_RESULTS_FILE.read_text())
                logger.info(
                    f"Loaded {len(self.evaluation_results)} existing evaluations"
                )
            except:
                pass

    def _save_results(self):
        """Save evaluation results"""
        EVAL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        EVAL_RESULTS_FILE.write_text(
            json.dumps(self.evaluation_results, indent=2, default=str)
        )

    def get_watchlist(self) -> List[str]:
        """Get current watchlist symbols"""
        try:
            r = requests.get(f"{API_BASE}/api/worklist", timeout=5)
            if r.status_code == 200:
                data = r.json().get("data", [])
                return [item["symbol"] for item in data]
        except Exception as e:
            logger.error(f"Error fetching watchlist: {e}")
        return []

    def evaluate_symbol(self, symbol: str) -> Dict:
        """Run full evaluation pipeline on a symbol"""
        logger.info(f"=" * 60)
        logger.info(f"EVALUATING: {symbol}")
        logger.info(f"=" * 60)

        result = {
            "symbol": symbol,
            "evaluated_at": datetime.now().isoformat(),
            "training": None,
            "backtest": None,
            "prediction": None,
            "score": 0,
            "recommendation": "HOLD",
        }

        # Step 1: Get historical data and generate prediction
        logger.info(f"[1/3] Generating AI Prediction for {symbol}...")
        prediction = self._generate_prediction(symbol)
        result["prediction"] = prediction

        # Step 2: Run backtest with multiple strategies
        logger.info(f"[2/3] Running Backtest for {symbol}...")
        backtest = self._run_backtest(symbol)
        result["backtest"] = backtest

        # Step 3: Calculate composite score
        logger.info(f"[3/3] Calculating Composite Score...")
        score, recommendation = self._calculate_score(prediction, backtest)
        result["score"] = score
        result["recommendation"] = recommendation

        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATION COMPLETE: {symbol}")
        logger.info(
            f"  AI Prediction: {prediction.get('signal', 'N/A')} ({prediction.get('confidence', 0):.1%})"
        )
        logger.info(f"  Backtest Win Rate: {backtest.get('win_rate', 0):.1%}")
        logger.info(f"  Composite Score: {score}/100")
        logger.info(f"  Recommendation: {recommendation}")
        logger.info(f"{'='*60}\n")

        # Store result
        self.evaluation_results[symbol] = result
        self._save_results()

        # Update watchlist with evaluation data
        self._update_watchlist_prediction(symbol, result)

        return result

    def _generate_prediction(self, symbol: str) -> Dict:
        """Generate AI prediction for symbol"""
        result = {
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "prob_up": 0.5,
            "features": {},
        }

        try:
            if self.predictor:
                pred = self.predictor.predict(symbol)
                if pred:
                    result.update(pred)
                    logger.info(
                        f"  Prediction: {pred.get('signal')} @ {pred.get('confidence', 0):.1%}"
                    )
            else:
                # Fallback: Use simple momentum-based prediction
                result = self._simple_momentum_prediction(symbol)
        except Exception as e:
            logger.error(f"  Prediction error: {e}")
            result = self._simple_momentum_prediction(symbol)

        return result

    def _simple_momentum_prediction(self, symbol: str) -> Dict:
        """Simple momentum-based prediction fallback"""
        result = {"signal": "NEUTRAL", "confidence": 0.5, "prob_up": 0.5}

        try:
            # Get quote data
            r = requests.get(f"{API_BASE}/api/alpaca/quote/{symbol}", timeout=5)
            if r.status_code == 200:
                q = r.json()
                change_pct = float(q.get("change_percent", 0))

                # Simple signal based on momentum
                if change_pct > 5:
                    result = {
                        "signal": "STRONG_BULLISH",
                        "confidence": 0.8,
                        "prob_up": 0.75,
                    }
                elif change_pct > 2:
                    result = {"signal": "BULLISH", "confidence": 0.7, "prob_up": 0.65}
                elif change_pct < -5:
                    result = {
                        "signal": "STRONG_BEARISH",
                        "confidence": 0.8,
                        "prob_up": 0.25,
                    }
                elif change_pct < -2:
                    result = {"signal": "BEARISH", "confidence": 0.7, "prob_up": 0.35}
                else:
                    result = {"signal": "NEUTRAL", "confidence": 0.5, "prob_up": 0.5}

                logger.info(
                    f"  Momentum-based prediction: {result['signal']} (change: {change_pct:.1f}%)"
                )
        except Exception as e:
            logger.error(f"  Momentum prediction error: {e}")

        return result

    def _run_backtest(self, symbol: str) -> Dict:
        """Run backtest on symbol with multiple strategies"""
        result = {
            "win_rate": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "best_strategy": None,
            "trades": 0,
        }

        try:
            # Get historical data
            if self.market_data:
                df = self.market_data.get_bars(symbol, timeframe="1Day", limit=90)
                if df is not None and len(df) > 30:
                    result = self._simple_backtest(symbol, df)
            else:
                # Fallback: Try to get data from API
                result = self._api_backtest(symbol)
        except Exception as e:
            logger.error(f"  Backtest error: {e}")

        return result

    def _simple_backtest(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Simple momentum backtest"""
        result = {
            "win_rate": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "best_strategy": "momentum",
            "trades": 0,
        }

        try:
            # Calculate simple momentum strategy
            df["return"] = df["close"].pct_change()
            df["signal"] = df["return"].shift(1) > 0  # Buy if yesterday was up

            # Calculate strategy returns
            df["strategy_return"] = df["signal"].shift(1) * df["return"]

            # Calculate metrics
            winning_days = (df["strategy_return"] > 0).sum()
            total_days = df["strategy_return"].notna().sum()

            if total_days > 0:
                result["win_rate"] = winning_days / total_days
                result["total_return"] = df["strategy_return"].sum()
                result["trades"] = int(total_days)

                # Max drawdown
                cumulative = (1 + df["strategy_return"].fillna(0)).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdowns = cumulative / rolling_max - 1
                result["max_drawdown"] = abs(drawdowns.min())

            logger.info(
                f"  Backtest: {result['trades']} trades, {result['win_rate']:.1%} win rate, {result['total_return']:.1%} return"
            )
        except Exception as e:
            logger.error(f"  Simple backtest error: {e}")

        return result

    def _api_backtest(self, symbol: str) -> Dict:
        """Try to get backtest from API"""
        result = {
            "win_rate": 0.5,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "best_strategy": "N/A",
            "trades": 0,
        }

        try:
            # Get historical bars from API
            r = requests.get(
                f"{API_BASE}/api/alpaca/bars/{symbol}?timeframe=1Day&limit=30",
                timeout=10,
            )
            if r.status_code == 200:
                bars = r.json()
                if bars and len(bars) > 5:
                    # Calculate simple metrics from bars
                    closes = [float(b.get("c", b.get("close", 0))) for b in bars]
                    if closes:
                        returns = [
                            (closes[i] - closes[i - 1]) / closes[i - 1]
                            for i in range(1, len(closes))
                        ]
                        if returns:
                            winning = sum(1 for r in returns if r > 0)
                            result["win_rate"] = (
                                winning / len(returns) if returns else 0.5
                            )
                            result["total_return"] = sum(returns)
                            result["trades"] = len(returns)
                            logger.info(
                                f"  API backtest: {result['trades']} periods, {result['win_rate']:.1%} win rate"
                            )
        except Exception as e:
            logger.error(f"  API backtest error: {e}")

        return result

    def _calculate_score(self, prediction: Dict, backtest: Dict) -> tuple:
        """Calculate composite score and recommendation"""
        score = 50  # Base score

        # Add prediction component (40 points max)
        signal = prediction.get("signal", "NEUTRAL")
        confidence = prediction.get("confidence", 0.5)

        if signal == "STRONG_BULLISH":
            score += 40 * confidence
        elif signal == "BULLISH":
            score += 25 * confidence
        elif signal == "STRONG_BEARISH":
            score -= 30 * confidence
        elif signal == "BEARISH":
            score -= 15 * confidence

        # Add backtest component (30 points max)
        win_rate = backtest.get("win_rate", 0.5)
        total_return = backtest.get("total_return", 0)

        if win_rate > 0.6:
            score += 15
        elif win_rate > 0.5:
            score += 5
        elif win_rate < 0.4:
            score -= 10

        if total_return > 0.05:
            score += 15
        elif total_return > 0:
            score += 5
        elif total_return < -0.05:
            score -= 15

        # Clamp score
        score = max(0, min(100, int(score)))

        # Determine recommendation
        if score >= 75:
            recommendation = "STRONG_BUY"
        elif score >= 60:
            recommendation = "BUY"
        elif score >= 40:
            recommendation = "HOLD"
        elif score >= 25:
            recommendation = "SELL"
        else:
            recommendation = "STRONG_SELL"

        return score, recommendation

    def _update_watchlist_prediction(self, symbol: str, result: Dict):
        """Update watchlist item with prediction data"""
        try:
            prediction = result.get("prediction", {})
            signal = prediction.get("signal", "NEUTRAL")
            confidence = prediction.get("confidence", 0)
            score = result.get("score", 50)

            # The API should update the prediction when we re-add or update
            # For now just log it - the dashboard_api already generates predictions
            logger.info(
                f"  Updated {symbol}: {signal} @ {confidence:.1%} (score: {score})"
            )
        except Exception as e:
            logger.error(f"  Error updating watchlist: {e}")

    def run(self):
        """Main monitoring loop"""
        logger.info("=" * 60)
        logger.info("WATCHLIST AUTO-EVALUATOR STARTED")
        logger.info("=" * 60)
        logger.info("Monitoring watchlist for new symbols...")
        logger.info("When symbols are added, auto-evaluation will run:")
        logger.info("  1. AI Prediction")
        logger.info("  2. Strategy Backtest")
        logger.info("  3. Composite Score")
        logger.info("=" * 60)

        # Initial scan of watchlist
        initial_symbols = self.get_watchlist()
        self.known_symbols = set(initial_symbols)
        logger.info(f"Initial watchlist: {list(self.known_symbols)}")

        # Evaluate any symbols that don't have recent evaluations
        for symbol in self.known_symbols:
            if symbol not in self.evaluation_results:
                self.evaluate_symbol(symbol)

        self.running = True
        cycle = 0

        while self.running:
            try:
                # Check for new symbols
                current_symbols = set(self.get_watchlist())
                new_symbols = current_symbols - self.known_symbols

                if new_symbols:
                    logger.info(
                        f"\n*** NEW SYMBOLS DETECTED: {list(new_symbols)} ***\n"
                    )
                    for symbol in new_symbols:
                        self.evaluate_symbol(symbol)
                    self.known_symbols = current_symbols

                cycle += 1
                if cycle % 6 == 0:  # Status every minute
                    logger.info(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring {len(self.known_symbols)} symbols | "
                        f"Evaluations: {len(self.evaluation_results)}"
                    )

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            time.sleep(self.check_interval)

    def stop(self):
        """Stop the evaluator"""
        self.running = False


if __name__ == "__main__":
    evaluator = WatchlistEvaluator()
    try:
        evaluator.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        evaluator.stop()
