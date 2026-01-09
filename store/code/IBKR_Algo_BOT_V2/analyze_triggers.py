#!/usr/bin/env python
"""
Breakout & Reversion Trigger Analysis
======================================
Compare ML model predictions with MACD/RSI indicators to find:
1. Breakout patterns (trend continuation)
2. Mean reversion patterns (overbought/oversold)
3. Optimal buy/sell trigger points

This analyzes when models agree with technical indicators
to identify high-probability trading signals.
"""

import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta
import yfinance as yf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class TechnicalSignal:
    """Technical indicator signal"""

    indicator: str
    signal_type: str  # BREAKOUT, REVERSION, NEUTRAL
    direction: str  # BULLISH, BEARISH, NEUTRAL
    strength: float  # 0-1
    value: float
    trigger_price: float


@dataclass
class TriggerPoint:
    """Buy/sell trigger point"""

    timestamp: str
    symbol: str
    action: str  # BUY, SELL
    trigger_type: str  # BREAKOUT, REVERSION, MODEL_AGREE
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    signals: Dict
    outcome: Optional[str] = None  # WIN, LOSS, PENDING
    actual_return: Optional[float] = None


class TechnicalAnalyzer:
    """Analyze technical indicators for breakout/reversion signals"""

    def __init__(self):
        # RSI thresholds
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.rsi_extreme_overbought = 80
        self.rsi_extreme_oversold = 20

        # MACD thresholds
        self.macd_threshold = 0  # Signal line cross

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = data.copy()

        # Handle yfinance multi-index
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index(drop=True)

        close = pd.Series(df["Close"].values.flatten(), index=df.index)
        high = pd.Series(df["High"].values.flatten(), index=df.index)
        low = pd.Series(df["Low"].values.flatten(), index=df.index)
        volume = pd.Series(df["Volume"].values.flatten(), index=df.index)

        # RSI
        df["RSI"] = ta.momentum.rsi(close, window=14)
        df["RSI_SMA"] = df["RSI"].rolling(window=5).mean()

        # MACD
        df["MACD"] = ta.trend.macd(close)
        df["MACD_Signal"] = ta.trend.macd_signal(close)
        df["MACD_Hist"] = ta.trend.macd_diff(close)

        # Bollinger Bands
        df["BB_High"] = ta.volatility.bollinger_hband(close, window=20)
        df["BB_Low"] = ta.volatility.bollinger_lband(close, window=20)
        df["BB_Mid"] = ta.volatility.bollinger_mavg(close, window=20)
        df["BB_Width"] = (df["BB_High"] - df["BB_Low"]) / df["BB_Mid"]
        df["BB_Pct"] = (close - df["BB_Low"]) / (df["BB_High"] - df["BB_Low"])

        # ATR for volatility
        df["ATR"] = ta.volatility.average_true_range(high, low, close, window=14)
        df["ATR_Pct"] = df["ATR"] / close * 100

        # Momentum
        df["MOM"] = ta.momentum.roc(close, window=10)
        df["MOM_5"] = ta.momentum.roc(close, window=5)

        # Stochastic
        df["Stoch_K"] = ta.momentum.stoch(high, low, close, window=14)
        df["Stoch_D"] = ta.momentum.stoch_signal(high, low, close, window=14)

        # ADX for trend strength
        df["ADX"] = ta.trend.adx(high, low, close, window=14)
        df["DI_Plus"] = ta.trend.adx_pos(high, low, close, window=14)
        df["DI_Minus"] = ta.trend.adx_neg(high, low, close, window=14)

        # Volume analysis
        df["Volume_SMA"] = volume.rolling(window=20).mean()
        df["Volume_Ratio"] = volume / df["Volume_SMA"]

        # Price action
        df["Returns"] = close.pct_change()
        df["Returns_5d"] = close.pct_change(5)
        df["SMA_20"] = close.rolling(window=20).mean()
        df["SMA_50"] = close.rolling(window=50).mean()
        df["Price_vs_SMA20"] = (close - df["SMA_20"]) / df["SMA_20"] * 100

        return df

    def get_rsi_signal(self, data: pd.DataFrame, idx: int) -> TechnicalSignal:
        """Analyze RSI for breakout/reversion"""
        rsi = data["RSI"].iloc[idx]
        rsi_prev = data["RSI"].iloc[idx - 1] if idx > 0 else rsi
        close = data["Close"].iloc[idx]

        # Mean reversion signals
        if rsi <= self.rsi_extreme_oversold:
            return TechnicalSignal(
                indicator="RSI",
                signal_type="REVERSION",
                direction="BULLISH",
                strength=min(1.0, (self.rsi_extreme_oversold - rsi) / 10),
                value=rsi,
                trigger_price=close,
            )
        elif rsi >= self.rsi_extreme_overbought:
            return TechnicalSignal(
                indicator="RSI",
                signal_type="REVERSION",
                direction="BEARISH",
                strength=min(1.0, (rsi - self.rsi_extreme_overbought) / 10),
                value=rsi,
                trigger_price=close,
            )

        # RSI breakout from oversold/overbought
        if rsi_prev <= self.rsi_oversold and rsi > self.rsi_oversold:
            return TechnicalSignal(
                indicator="RSI",
                signal_type="BREAKOUT",
                direction="BULLISH",
                strength=0.7,
                value=rsi,
                trigger_price=close,
            )
        elif rsi_prev >= self.rsi_overbought and rsi < self.rsi_overbought:
            return TechnicalSignal(
                indicator="RSI",
                signal_type="BREAKOUT",
                direction="BEARISH",
                strength=0.7,
                value=rsi,
                trigger_price=close,
            )

        return TechnicalSignal(
            indicator="RSI",
            signal_type="NEUTRAL",
            direction="NEUTRAL",
            strength=0.0,
            value=rsi,
            trigger_price=close,
        )

    def get_macd_signal(self, data: pd.DataFrame, idx: int) -> TechnicalSignal:
        """Analyze MACD for breakout signals"""
        macd = data["MACD"].iloc[idx]
        signal = data["MACD_Signal"].iloc[idx]
        hist = data["MACD_Hist"].iloc[idx]
        hist_prev = data["MACD_Hist"].iloc[idx - 1] if idx > 0 else hist
        close = data["Close"].iloc[idx]

        # MACD histogram crossover (momentum shift)
        if hist_prev < 0 and hist > 0:
            return TechnicalSignal(
                indicator="MACD",
                signal_type="BREAKOUT",
                direction="BULLISH",
                strength=min(1.0, abs(hist) * 50),
                value=hist,
                trigger_price=close,
            )
        elif hist_prev > 0 and hist < 0:
            return TechnicalSignal(
                indicator="MACD",
                signal_type="BREAKOUT",
                direction="BEARISH",
                strength=min(1.0, abs(hist) * 50),
                value=hist,
                trigger_price=close,
            )

        # Strong trend continuation
        if hist > 0 and hist > hist_prev:
            return TechnicalSignal(
                indicator="MACD",
                signal_type="BREAKOUT",
                direction="BULLISH",
                strength=min(0.6, abs(hist) * 30),
                value=hist,
                trigger_price=close,
            )
        elif hist < 0 and hist < hist_prev:
            return TechnicalSignal(
                indicator="MACD",
                signal_type="BREAKOUT",
                direction="BEARISH",
                strength=min(0.6, abs(hist) * 30),
                value=hist,
                trigger_price=close,
            )

        return TechnicalSignal(
            indicator="MACD",
            signal_type="NEUTRAL",
            direction="NEUTRAL",
            strength=0.0,
            value=hist,
            trigger_price=close,
        )

    def get_bollinger_signal(self, data: pd.DataFrame, idx: int) -> TechnicalSignal:
        """Analyze Bollinger Bands for breakout/reversion"""
        bb_pct = data["BB_Pct"].iloc[idx]
        close = data["Close"].iloc[idx]
        bb_high = data["BB_High"].iloc[idx]
        bb_low = data["BB_Low"].iloc[idx]

        # Mean reversion at bands
        if bb_pct <= 0.0:  # Below lower band
            return TechnicalSignal(
                indicator="BB",
                signal_type="REVERSION",
                direction="BULLISH",
                strength=min(1.0, abs(bb_pct)),
                value=bb_pct,
                trigger_price=bb_low,
            )
        elif bb_pct >= 1.0:  # Above upper band
            return TechnicalSignal(
                indicator="BB",
                signal_type="REVERSION",
                direction="BEARISH",
                strength=min(1.0, bb_pct - 1.0),
                value=bb_pct,
                trigger_price=bb_high,
            )

        # Breakout from bands
        if bb_pct < 0.2:
            return TechnicalSignal(
                indicator="BB",
                signal_type="BREAKOUT",
                direction="BULLISH",
                strength=0.5,
                value=bb_pct,
                trigger_price=close,
            )
        elif bb_pct > 0.8:
            return TechnicalSignal(
                indicator="BB",
                signal_type="BREAKOUT",
                direction="BEARISH",
                strength=0.5,
                value=bb_pct,
                trigger_price=close,
            )

        return TechnicalSignal(
            indicator="BB",
            signal_type="NEUTRAL",
            direction="NEUTRAL",
            strength=0.0,
            value=bb_pct,
            trigger_price=close,
        )

    def analyze_all(self, data: pd.DataFrame, idx: int) -> Dict[str, TechnicalSignal]:
        """Get all technical signals"""
        return {
            "RSI": self.get_rsi_signal(data, idx),
            "MACD": self.get_macd_signal(data, idx),
            "BB": self.get_bollinger_signal(data, idx),
        }


class TriggerFinder:
    """Find optimal buy/sell triggers by combining ML + technicals"""

    def __init__(self):
        self.tech_analyzer = TechnicalAnalyzer()
        self.triggers: List[TriggerPoint] = []

    def load_models(self):
        """Load ML models"""
        self.cnn = None
        self.lstm = None
        self.convlstm = None
        self.lgb = None

        try:
            from ai.cnn_stock_predictor import get_cnn_predictor

            self.cnn = get_cnn_predictor()
        except Exception as e:
            logger.warning(f"Could not load CNN model: {e}")

        try:
            from ai.lstm_stock_predictor import get_lstm_predictor

            self.lstm = get_lstm_predictor("lstm")
        except Exception as e:
            logger.warning(f"Could not load LSTM model: {e}")

        try:
            from ai.lstm_stock_predictor import get_lstm_predictor

            self.convlstm = get_lstm_predictor("convlstm")
        except Exception as e:
            logger.warning(f"Could not load ConvLSTM model: {e}")

        try:
            from ai.alpaca_ai_predictor import get_alpaca_predictor

            self.lgb = get_alpaca_predictor()
        except Exception as e:
            logger.warning(f"Could not load LightGBM model: {e}")

        loaded = sum(
            [1 for m in [self.cnn, self.lstm, self.convlstm, self.lgb] if m is not None]
        )
        logger.info(f"Models loaded successfully: {loaded}/4")
        return loaded > 0

    def get_model_predictions(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Get predictions from all ML models"""
        predictions = {}

        if self.cnn:
            try:
                cnn_pred = self.cnn.predict(symbol, data)
                predictions["CNN"] = {
                    "action": cnn_pred.get("action", "HOLD"),
                    "confidence": cnn_pred.get("confidence", 0),
                    "momentum": cnn_pred.get("momentum_score", 0),
                }
            except:
                predictions["CNN"] = {"action": "HOLD", "confidence": 0, "momentum": 0}

        if self.lstm:
            try:
                lstm_pred = self.lstm.predict(symbol, data)
                predictions["LSTM"] = {
                    "action": lstm_pred.get("action", "HOLD"),
                    "confidence": lstm_pred.get("confidence", 0),
                    "momentum": lstm_pred.get("momentum_score", 0),
                }
            except:
                predictions["LSTM"] = {"action": "HOLD", "confidence": 0, "momentum": 0}

        if self.convlstm:
            try:
                convlstm_pred = self.convlstm.predict(symbol, data)
                predictions["ConvLSTM"] = {
                    "action": convlstm_pred.get("action", "HOLD"),
                    "confidence": convlstm_pred.get("confidence", 0),
                    "momentum": convlstm_pred.get("momentum_score", 0),
                }
            except:
                predictions["ConvLSTM"] = {
                    "action": "HOLD",
                    "confidence": 0,
                    "momentum": 0,
                }

        if self.lgb:
            try:
                lgb_pred = self.lgb.predict(symbol)
                predictions["LightGBM"] = {
                    "action": lgb_pred.get("action", "HOLD"),
                    "confidence": lgb_pred.get("confidence", 0),
                    "momentum": 0,
                }
            except:
                predictions["LightGBM"] = {
                    "action": "HOLD",
                    "confidence": 0,
                    "momentum": 0,
                }

        return predictions

    def find_triggers(
        self, symbol: str, data: pd.DataFrame, lookback: int = 60
    ) -> List[TriggerPoint]:
        """Find buy/sell triggers by combining ML + technicals"""
        triggers = []

        # Calculate indicators
        data = self.tech_analyzer.calculate_indicators(data)
        data = data.dropna()

        if len(data) < lookback + 10:
            return triggers

        close = data["Close"].values
        if hasattr(close, "flatten"):
            close = close.flatten()

        # Walk through data
        for i in range(lookback, len(data) - 5, 1):
            # Get technical signals at this point
            tech_signals = self.tech_analyzer.analyze_all(data, i)

            # Get ML predictions using data up to this point
            subset = data.iloc[: i + 1].copy()
            ml_preds = self.get_model_predictions(symbol, subset)

            # Current price
            current_price = close[i]

            # Calculate actual outcome (5 days ahead)
            future_price = close[min(i + 5, len(close) - 1)]
            actual_return = (future_price - current_price) / current_price

            # Find trigger conditions
            trigger = self._evaluate_trigger(
                symbol, i, data, current_price, tech_signals, ml_preds, actual_return
            )

            if trigger:
                triggers.append(trigger)

        self.triggers = triggers
        return triggers

    def _evaluate_trigger(
        self,
        symbol: str,
        idx: int,
        data: pd.DataFrame,
        price: float,
        tech_signals: Dict[str, TechnicalSignal],
        ml_preds: Dict,
        actual_return: float,
    ) -> Optional[TriggerPoint]:
        """Evaluate if conditions warrant a trigger"""

        rsi_sig = tech_signals["RSI"]
        macd_sig = tech_signals["MACD"]
        bb_sig = tech_signals["BB"]

        # Count bullish/bearish signals
        bullish_count = 0
        bearish_count = 0
        trigger_type = None
        combined_strength = 0

        # Technical signals
        for sig in [rsi_sig, macd_sig, bb_sig]:
            if sig.direction == "BULLISH":
                bullish_count += 1
                combined_strength += sig.strength
            elif sig.direction == "BEARISH":
                bearish_count += 1
                combined_strength += sig.strength

        # ML model consensus
        ml_bullish = 0
        ml_bearish = 0
        ml_confidence = 0

        for name, pred in ml_preds.items():
            if pred["action"] == "BUY":
                ml_bullish += 1
                ml_confidence += pred["confidence"]
            elif pred["action"] == "SELL":
                ml_bearish += 1
                ml_confidence += pred["confidence"]

        # Determine trigger conditions
        num_ml_models = len(ml_preds)  # How many models we actually have
        ml_thresh = max(1, num_ml_models // 2)  # Require at least half to agree

        # BREAKOUT BUY: MACD strong bullish crossover + RSI momentum zone (40-60)
        # Only trigger on actual histogram crossovers (high strength), not just continuation
        if (
            macd_sig.signal_type == "BREAKOUT"
            and macd_sig.direction == "BULLISH"
            and macd_sig.strength >= 0.8  # Only strong MACD crossovers
            and 40 < rsi_sig.value < 65
        ):  # RSI in momentum zone, room to run
            ml_boost = 0.2 if ml_bullish >= ml_thresh else 0
            trigger_type = "BREAKOUT"
            action = "BUY"
            confidence = min(1.0, macd_sig.strength + ml_boost)

        # REVERSION BUY: RSI oversold (< 30) + BB at lower band
        elif (
            rsi_sig.signal_type == "REVERSION"
            and rsi_sig.direction == "BULLISH"
            and rsi_sig.value < 30  # True oversold
            and bb_sig.direction == "BULLISH"
        ):
            ml_boost = 0.2 if ml_bullish >= 1 else 0
            trigger_type = "REVERSION"
            action = "BUY"
            confidence = min(1.0, (rsi_sig.strength + bb_sig.strength) / 2 + ml_boost)

        # BREAKOUT SELL: MACD strong bearish crossover + RSI momentum zone
        elif (
            macd_sig.signal_type == "BREAKOUT"
            and macd_sig.direction == "BEARISH"
            and macd_sig.strength >= 0.8  # Only strong MACD crossovers
            and 35 < rsi_sig.value < 60
        ):  # RSI in momentum zone, room to fall
            ml_boost = 0.2 if ml_bearish >= ml_thresh else 0
            trigger_type = "BREAKOUT"
            action = "SELL"
            confidence = min(1.0, macd_sig.strength + ml_boost)

        # REVERSION SELL: RSI overbought (> 70) + BB at upper band
        elif (
            rsi_sig.signal_type == "REVERSION"
            and rsi_sig.direction == "BEARISH"
            and rsi_sig.value > 70  # True overbought
            and bb_sig.direction == "BEARISH"
        ):
            ml_boost = 0.2 if ml_bearish >= 1 else 0
            trigger_type = "REVERSION"
            action = "SELL"
            confidence = min(1.0, (rsi_sig.strength + bb_sig.strength) / 2 + ml_boost)

        # PURE RSI EXTREME: Very strong RSI reversion signal alone (RSI < 25 or > 75)
        elif rsi_sig.signal_type == "REVERSION" and rsi_sig.strength > 0.8:
            trigger_type = "RSI_EXTREME"
            action = "BUY" if rsi_sig.direction == "BULLISH" else "SELL"
            confidence = rsi_sig.strength

        # ML CONSENSUS: All loaded models agree with high confidence
        elif (
            num_ml_models >= 2
            and ml_bullish == num_ml_models
            and ml_confidence / num_ml_models > 0.6
        ):
            trigger_type = "MODEL_AGREE"
            action = "BUY"
            confidence = ml_confidence / num_ml_models

        elif (
            num_ml_models >= 2
            and ml_bearish == num_ml_models
            and ml_confidence / num_ml_models > 0.6
        ):
            trigger_type = "MODEL_AGREE"
            action = "SELL"
            confidence = ml_confidence / num_ml_models

        else:
            return None

        # Calculate stop loss and take profit based on ATR
        atr = data["ATR"].iloc[idx]
        if action == "BUY":
            stop_loss = price - (2.0 * atr)
            take_profit = price + (3.0 * atr)
            outcome = (
                "WIN"
                if actual_return > 0.01
                else ("LOSS" if actual_return < -0.01 else "NEUTRAL")
            )
        else:
            stop_loss = price + (2.0 * atr)
            take_profit = price - (3.0 * atr)
            outcome = (
                "WIN"
                if actual_return < -0.01
                else ("LOSS" if actual_return > 0.01 else "NEUTRAL")
            )

        return TriggerPoint(
            timestamp=(
                str(data.index[idx])
                if hasattr(data.index[idx], "strftime")
                else str(idx)
            ),
            symbol=symbol,
            action=action,
            trigger_type=trigger_type,
            confidence=min(1.0, confidence),
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signals={
                "RSI": asdict(rsi_sig),
                "MACD": asdict(macd_sig),
                "BB": asdict(bb_sig),
                "ML": ml_preds,
            },
            outcome=outcome,
            actual_return=actual_return,
        )

    def analyze_trigger_performance(self, triggers: List[TriggerPoint]) -> Dict:
        """Analyze how well triggers performed"""
        if not triggers:
            return {}

        results = {
            "total_triggers": len(triggers),
            "by_type": {},
            "by_action": {},
            "overall": {},
        }

        # By trigger type
        for ttype in ["BREAKOUT", "REVERSION", "MODEL_AGREE"]:
            type_triggers = [t for t in triggers if t.trigger_type == ttype]
            if type_triggers:
                wins = len([t for t in type_triggers if t.outcome == "WIN"])
                losses = len([t for t in type_triggers if t.outcome == "LOSS"])
                total = len(type_triggers)
                returns = [
                    t.actual_return
                    for t in type_triggers
                    if t.actual_return is not None
                ]

                results["by_type"][ttype] = {
                    "count": total,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": wins / total if total > 0 else 0,
                    "avg_return": np.mean(returns) * 100 if returns else 0,
                    "total_return": np.sum(returns) * 100 if returns else 0,
                }

        # By action
        for action in ["BUY", "SELL"]:
            action_triggers = [t for t in triggers if t.action == action]
            if action_triggers:
                wins = len([t for t in action_triggers if t.outcome == "WIN"])
                total = len(action_triggers)
                returns = [
                    t.actual_return
                    for t in action_triggers
                    if t.actual_return is not None
                ]

                # Adjust returns for sell (profit when price goes down)
                if action == "SELL":
                    returns = [-r for r in returns]

                results["by_action"][action] = {
                    "count": total,
                    "wins": wins,
                    "win_rate": wins / total if total > 0 else 0,
                    "avg_return": np.mean(returns) * 100 if returns else 0,
                    "total_return": np.sum(returns) * 100 if returns else 0,
                }

        # Overall
        all_wins = len([t for t in triggers if t.outcome == "WIN"])
        all_returns = []
        for t in triggers:
            if t.actual_return is not None:
                if t.action == "SELL":
                    all_returns.append(-t.actual_return)
                else:
                    all_returns.append(t.actual_return)

        results["overall"] = {
            "total": len(triggers),
            "wins": all_wins,
            "win_rate": all_wins / len(triggers) if triggers else 0,
            "avg_return": np.mean(all_returns) * 100 if all_returns else 0,
            "total_return": np.sum(all_returns) * 100 if all_returns else 0,
            "sharpe": (
                np.mean(all_returns) / (np.std(all_returns) + 1e-8) * np.sqrt(252)
                if all_returns
                else 0
            ),
        }

        return results


def main():
    """Run trigger analysis"""
    logger.info("=" * 60)
    logger.info("BREAKOUT & REVERSION TRIGGER ANALYSIS")
    logger.info("=" * 60)

    # Symbols to analyze
    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD"]

    # Initialize
    finder = TriggerFinder()
    if not finder.load_models():
        logger.error("Cannot proceed without models")
        return

    all_triggers = []
    symbol_results = {}

    for symbol in symbols:
        logger.info(f"\n{'='*40}")
        logger.info(f"Analyzing {symbol}")
        logger.info(f"{'='*40}")

        try:
            # Get data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if len(data) < 100:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            # Find triggers
            triggers = finder.find_triggers(symbol, data)
            all_triggers.extend(triggers)

            # Analyze
            results = finder.analyze_trigger_performance(triggers)
            symbol_results[symbol] = results

            logger.info(f"\nTriggers found: {len(triggers)}")

            if results.get("by_type"):
                logger.info("\nBy Trigger Type:")
                for ttype, stats in results["by_type"].items():
                    logger.info(
                        f"  {ttype:12}: {stats['count']:3} triggers, "
                        f"Win Rate: {stats['win_rate']*100:5.1f}%, "
                        f"Avg Return: {stats['avg_return']:+6.2f}%"
                    )

            if results.get("overall"):
                overall = results["overall"]
                logger.info(
                    f"\nOverall: Win Rate {overall['win_rate']*100:.1f}%, "
                    f"Total Return: {overall['total_return']:+.2f}%, "
                    f"Sharpe: {overall['sharpe']:.2f}"
                )

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")

    # Aggregate results
    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATE RESULTS")
    logger.info("=" * 60)

    if all_triggers:
        aggregate = finder.analyze_trigger_performance(all_triggers)

        logger.info(f"\nTotal Triggers: {aggregate['overall']['total']}")
        logger.info(f"Overall Win Rate: {aggregate['overall']['win_rate']*100:.1f}%")
        logger.info(f"Total Return: {aggregate['overall']['total_return']:+.2f}%")
        logger.info(f"Sharpe Ratio: {aggregate['overall']['sharpe']:.2f}")

        logger.info("\nBY TRIGGER TYPE:")
        for ttype, stats in aggregate.get("by_type", {}).items():
            logger.info(
                f"  {ttype:15}: Win Rate {stats['win_rate']*100:5.1f}%, "
                f"Return {stats['total_return']:+7.2f}%, "
                f"Count: {stats['count']}"
            )

        logger.info("\nBY ACTION:")
        for action, stats in aggregate.get("by_action", {}).items():
            logger.info(
                f"  {action:4}: Win Rate {stats['win_rate']*100:5.1f}%, "
                f"Return {stats['total_return']:+7.2f}%, "
                f"Count: {stats['count']}"
            )

        # Best triggers
        logger.info("\n" + "=" * 60)
        logger.info("TOP PERFORMING TRIGGER COMBINATIONS")
        logger.info("=" * 60)

        # Analyze best combinations
        breakout_buys = [
            t
            for t in all_triggers
            if t.trigger_type == "BREAKOUT" and t.action == "BUY"
        ]
        reversion_buys = [
            t
            for t in all_triggers
            if t.trigger_type == "REVERSION" and t.action == "BUY"
        ]
        model_agree_buys = [
            t
            for t in all_triggers
            if t.trigger_type == "MODEL_AGREE" and t.action == "BUY"
        ]

        for name, triggers in [
            ("BREAKOUT BUY", breakout_buys),
            ("REVERSION BUY", reversion_buys),
            ("MODEL AGREE BUY", model_agree_buys),
        ]:
            if triggers:
                wins = len([t for t in triggers if t.outcome == "WIN"])
                returns = [t.actual_return for t in triggers if t.actual_return]
                logger.info(f"\n{name}:")
                logger.info(f"  Count: {len(triggers)}")
                logger.info(f"  Win Rate: {wins/len(triggers)*100:.1f}%")
                logger.info(f"  Avg Return: {np.mean(returns)*100:.2f}%")

                # Show conditions that led to wins
                winning = [t for t in triggers if t.outcome == "WIN"]
                if winning:
                    avg_rsi = np.mean([t.signals["RSI"]["value"] for t in winning])
                    avg_macd = np.mean([t.signals["MACD"]["value"] for t in winning])
                    logger.info(f"  Avg RSI (winners): {avg_rsi:.1f}")
                    logger.info(f"  Avg MACD (winners): {avg_macd:.4f}")

        # Recommended triggers
        logger.info("\n" + "=" * 60)
        logger.info("RECOMMENDED TRIGGER RULES")
        logger.info("=" * 60)

        # Find best performing combination
        best_winrate = 0
        best_combo = None

        for ttype in ["BREAKOUT", "REVERSION", "MODEL_AGREE"]:
            for action in ["BUY", "SELL"]:
                combo_triggers = [
                    t
                    for t in all_triggers
                    if t.trigger_type == ttype and t.action == action
                ]
                if len(combo_triggers) >= 5:
                    wins = len([t for t in combo_triggers if t.outcome == "WIN"])
                    winrate = wins / len(combo_triggers)
                    if winrate > best_winrate:
                        best_winrate = winrate
                        best_combo = (ttype, action, combo_triggers)

        if best_combo:
            ttype, action, triggers = best_combo
            logger.info(f"\nBest Trigger: {ttype} {action}")
            logger.info(f"Win Rate: {best_winrate*100:.1f}%")
            logger.info(f"Sample Size: {len(triggers)}")

            # Extract common conditions from winners
            winners = [t for t in triggers if t.outcome == "WIN"]
            if winners:
                avg_rsi = np.mean([t.signals["RSI"]["value"] for t in winners])
                avg_conf = np.mean([t.confidence for t in winners])

                logger.info(f"\nOptimal Conditions for {ttype} {action}:")
                if ttype == "REVERSION" and action == "BUY":
                    logger.info(f"  - RSI < {avg_rsi + 5:.0f} (oversold zone)")
                    logger.info(f"  - Price at/below lower Bollinger Band")
                    logger.info(f"  - At least 1 ML model says BUY")
                elif ttype == "BREAKOUT" and action == "BUY":
                    logger.info(f"  - MACD histogram crossing above 0")
                    logger.info(f"  - RSI between 40-70 (not overbought)")
                    logger.info(f"  - 2+ ML models say BUY")
                elif ttype == "MODEL_AGREE":
                    logger.info(f"  - 3+ ML models agree on direction")
                    logger.info(f"  - At least 1 technical indicator confirms")

    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
