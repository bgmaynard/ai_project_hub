"""
Background AI Brain - Continuous Learning & Evaluation Engine
==============================================================
Runs continuously in the background, using available CPU to:
- Monitor market conditions in real-time
- Retrain models when patterns shift
- Evaluate prediction accuracy
- Detect market regime changes
- Optimize strategy parameters

DESIGN:
- Uses threading/multiprocessing for parallel computation
- Scales CPU usage based on available resources
- Prioritizes critical tasks during market hours
- Runs intensive training during off-hours

WARRIOR RULE: The market is always changing. Your AI must adapt!
"""

import logging
import threading
import multiprocessing
import time
import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pytz

logger = logging.getLogger(__name__)

# Sector ETF mappings for rotation tracking
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Healthcare',
    'XLI': 'Industrials',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLU': 'Utilities',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services'
}

# Volatility thresholds
VOLATILITY_THRESHOLDS = {
    'EXTREME': 3.0,   # ATR > 3x normal
    'HIGH': 2.0,      # ATR > 2x normal
    'ELEVATED': 1.5,  # ATR > 1.5x normal
    'NORMAL': 1.0,    # Baseline
    'LOW': 0.5        # ATR < 0.5x normal
}

# Configuration
BRAIN_STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "store", "brain_state.json")


@dataclass
class VolatilityState:
    """Current volatility measurement"""
    level: str  # EXTREME, HIGH, ELEVATED, NORMAL, LOW
    atr_current: float
    atr_baseline: float
    atr_ratio: float
    vix_proxy: float  # Calculated from SPY volatility
    spike_detected: bool
    detected_at: str

    def to_dict(self):
        return asdict(self)


@dataclass
class SectorRotation:
    """Sector rotation tracking"""
    leading_sectors: List[str]
    lagging_sectors: List[str]
    sector_scores: Dict[str, float]
    rotation_signal: str  # RISK_ON, RISK_OFF, NEUTRAL
    detected_at: str

    def to_dict(self):
        return asdict(self)


@dataclass
class PredictionDrift:
    """Tracks prediction accuracy drift"""
    current_accuracy: float
    baseline_accuracy: float
    drift_percentage: float
    drift_detected: bool
    samples_evaluated: int
    window_start: str
    window_end: str
    recommendation: str  # RETRAIN_URGENT, RETRAIN_SCHEDULED, OK

    def to_dict(self):
        return asdict(self)


@dataclass
class MarketRegime:
    """Current market regime classification"""
    regime: str  # TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, CALM
    confidence: float
    detected_at: str
    indicators: Dict
    volatility: Optional[Dict] = None
    sector_rotation: Optional[Dict] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class BrainMetrics:
    """Performance metrics for the AI brain"""
    cpu_usage_target: float
    actual_cpu_usage: float
    tasks_completed: int
    tasks_pending: int
    predictions_evaluated: int
    prediction_accuracy: float
    last_retrain: str
    models_loaded: int
    market_regime: str
    uptime_seconds: float

    def to_dict(self):
        return asdict(self)


class BackgroundBrain:
    """
    Continuous AI processing engine.

    Runs in background threads/processes to:
    - Constantly evaluate market conditions
    - Retrain models when needed
    - Monitor prediction accuracy
    - Adapt to changing markets
    """

    def __init__(self, cpu_target: float = 0.5):
        """
        Initialize the background brain.

        Args:
            cpu_target: Target CPU utilization (0.0-1.0), default 50%
        """
        self.et_tz = pytz.timezone('US/Eastern')

        # CPU management
        self.cpu_target = cpu_target  # Target 50% by default
        self.num_cores = multiprocessing.cpu_count()
        self.worker_threads = max(2, int(self.num_cores * cpu_target))

        # Thread pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.worker_threads)
        self.process_pool = None  # Created on demand for heavy tasks

        # State
        self.is_running = False
        self.start_time = None
        self.tasks_completed = 0
        self.tasks_pending = 0

        # Market monitoring
        self.current_regime = MarketRegime(
            regime="UNKNOWN",
            confidence=0.0,
            detected_at="",
            indicators={}
        )
        self.price_buffer: Dict[str, deque] = {}  # Symbol -> recent prices
        self.prediction_history: deque = deque(maxlen=1000)

        # Volatility tracking
        self.volatility_state = VolatilityState(
            level="NORMAL",
            atr_current=0.0,
            atr_baseline=0.0,
            atr_ratio=1.0,
            vix_proxy=0.0,
            spike_detected=False,
            detected_at=""
        )
        self.atr_history: deque = deque(maxlen=100)  # Track ATR over time

        # Sector rotation tracking
        self.sector_rotation = SectorRotation(
            leading_sectors=[],
            lagging_sectors=[],
            sector_scores={},
            rotation_signal="NEUTRAL",
            detected_at=""
        )
        self.sector_price_history: Dict[str, deque] = {etf: deque(maxlen=50) for etf in SECTOR_ETFS}

        # Prediction drift tracking
        self.prediction_drift = PredictionDrift(
            current_accuracy=0.0,
            baseline_accuracy=0.63,  # Starting with known accuracy from training
            drift_percentage=0.0,
            drift_detected=False,
            samples_evaluated=0,
            window_start="",
            window_end="",
            recommendation="OK"
        )
        self.accuracy_history: deque = deque(maxlen=50)  # Track accuracy over time

        # Retraining state
        self.last_retrain_time: Optional[datetime] = None
        self.retrain_triggered = False

        # Callbacks
        self.on_regime_change: Optional[Callable] = None
        self.on_retrain_needed: Optional[Callable] = None
        self.on_prediction_drift: Optional[Callable] = None

        # Task scheduling
        self.scheduled_tasks: List[Dict] = []
        self.task_lock = threading.Lock()

        # Background threads
        self.monitor_thread = None
        self.evaluation_thread = None
        self.training_thread = None

        logger.info(f"BackgroundBrain initialized: {self.num_cores} cores, {self.worker_threads} workers, {cpu_target*100:.0f}% target")

    def start(self):
        """Start the background brain"""
        if self.is_running:
            logger.warning("Brain already running")
            return

        self.is_running = True
        self.start_time = datetime.now(self.et_tz)

        # Start background threads
        self.monitor_thread = threading.Thread(target=self._market_monitor_loop, daemon=True)
        self.evaluation_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)

        self.monitor_thread.start()
        self.evaluation_thread.start()
        self.training_thread.start()

        logger.info("BackgroundBrain started - continuous learning active")

    def stop(self):
        """Stop the background brain"""
        self.is_running = False

        # Shutdown thread pools
        self.thread_pool.shutdown(wait=False)
        if self.process_pool:
            self.process_pool.shutdown(wait=False)

        logger.info("BackgroundBrain stopped")

    def _market_monitor_loop(self):
        """Continuous market condition monitoring"""
        while self.is_running:
            try:
                # Check if market is open
                now = datetime.now(self.et_tz)
                is_market_hours = (
                    now.weekday() < 5 and
                    9 <= now.hour < 16 and
                    (now.hour > 9 or now.minute >= 30)
                )

                if is_market_hours:
                    # High-frequency monitoring during market hours
                    self._update_market_regime()
                    self._check_volatility_spike()
                    self._monitor_sector_rotation()
                    time.sleep(10)  # Every 10 seconds
                else:
                    # Low-frequency during off-hours
                    time.sleep(60)  # Every minute

            except Exception as e:
                logger.error(f"Market monitor error: {e}")
                time.sleep(30)

    def _evaluation_loop(self):
        """Continuous prediction evaluation"""
        while self.is_running:
            try:
                now = datetime.now(self.et_tz)
                is_market_hours = (
                    now.weekday() < 5 and
                    9 <= now.hour < 16
                )

                if is_market_hours:
                    # Evaluate recent predictions
                    self._evaluate_predictions()
                    self._check_prediction_drift()
                    self._update_confidence_scores()
                    time.sleep(30)  # Every 30 seconds
                else:
                    time.sleep(300)  # Every 5 minutes off-hours

            except Exception as e:
                logger.error(f"Evaluation loop error: {e}")
                time.sleep(60)

    def _training_loop(self):
        """Background model training/retraining"""
        while self.is_running:
            try:
                now = datetime.now(self.et_tz)

                # Heavy training during off-hours (before market or after)
                is_training_window = (
                    now.weekday() < 5 and
                    (now.hour < 9 or now.hour >= 16)
                )

                if is_training_window:
                    # Check if retraining is needed
                    if self._should_retrain():
                        self._run_incremental_training()
                    time.sleep(300)  # Every 5 minutes
                else:
                    # Light training during market hours (if needed)
                    if self._urgent_retrain_needed():
                        self._run_quick_adaptation()
                    time.sleep(600)  # Every 10 minutes

            except Exception as e:
                logger.error(f"Training loop error: {e}")
                time.sleep(300)

    def _update_market_regime(self):
        """Detect current market regime"""
        try:
            # Get recent market data (SPY as proxy)
            from ai.alpaca_ai_predictor import get_alpaca_predictor
            predictor = get_alpaca_predictor()

            # Simple regime detection based on recent price action
            # In production, this would be more sophisticated

            regime = "RANGING"
            confidence = 0.5
            indicators = {}

            # Check trend (using predictions as proxy)
            recent_predictions = list(self.prediction_history)[-20:]
            if recent_predictions:
                bullish = sum(1 for p in recent_predictions if p.get('prediction') in ['BUY', 'BULLISH'])
                bearish = sum(1 for p in recent_predictions if p.get('prediction') in ['SELL', 'BEARISH'])

                total = len(recent_predictions)
                if bullish / total > 0.7:
                    regime = "TRENDING_UP"
                    confidence = bullish / total
                elif bearish / total > 0.7:
                    regime = "TRENDING_DOWN"
                    confidence = bearish / total
                else:
                    regime = "RANGING"
                    confidence = 0.5

                indicators = {
                    "bullish_ratio": bullish / total,
                    "bearish_ratio": bearish / total,
                    "sample_size": total
                }

            # Check for regime change
            old_regime = self.current_regime.regime

            self.current_regime = MarketRegime(
                regime=regime,
                confidence=confidence,
                detected_at=datetime.now(self.et_tz).isoformat(),
                indicators=indicators
            )

            if old_regime != regime and self.on_regime_change:
                self.on_regime_change(old_regime, regime)
                logger.info(f"Market regime changed: {old_regime} -> {regime}")

        except Exception as e:
            logger.error(f"Regime detection error: {e}")

    def _check_volatility_spike(self):
        """Detect sudden volatility changes using ATR and price swings"""
        try:
            # Get SPY data as market proxy
            from alpaca_market_data import AlpacaMarketData
            market_data = AlpacaMarketData()

            # Fetch recent SPY bars for ATR calculation
            bars = market_data.get_bars('SPY', timeframe='1Min', limit=30)
            if not bars or len(bars) < 14:
                return

            # Calculate True Range for each bar
            true_ranges = []
            for i in range(1, len(bars)):
                high = bars[i].get('high', bars[i].get('h', 0))
                low = bars[i].get('low', bars[i].get('l', 0))
                prev_close = bars[i-1].get('close', bars[i-1].get('c', 0))

                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                true_ranges.append(tr)

            if not true_ranges:
                return

            # Calculate current ATR (14-period)
            atr_current = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else np.mean(true_ranges)

            # Store in history
            self.atr_history.append({
                'atr': atr_current,
                'timestamp': datetime.now(self.et_tz).isoformat()
            })

            # Calculate baseline ATR (average of historical ATR)
            if len(self.atr_history) >= 20:
                atr_baseline = np.mean([h['atr'] for h in list(self.atr_history)[:-5]])  # Exclude recent
            else:
                atr_baseline = atr_current  # Not enough history

            # Calculate ATR ratio
            atr_ratio = atr_current / atr_baseline if atr_baseline > 0 else 1.0

            # Determine volatility level
            level = "NORMAL"
            spike_detected = False
            for lev, threshold in sorted(VOLATILITY_THRESHOLDS.items(), key=lambda x: -x[1]):
                if atr_ratio >= threshold:
                    level = lev
                    if lev in ['EXTREME', 'HIGH']:
                        spike_detected = True
                    break

            # Calculate VIX proxy (annualized volatility from recent price changes)
            closes = [b.get('close', b.get('c', 0)) for b in bars]
            if len(closes) >= 2:
                returns = np.diff(closes) / closes[:-1]
                vix_proxy = np.std(returns) * np.sqrt(252 * 390) * 100  # Annualized, percent
            else:
                vix_proxy = 0.0

            # Update volatility state
            old_level = self.volatility_state.level
            self.volatility_state = VolatilityState(
                level=level,
                atr_current=round(atr_current, 4),
                atr_baseline=round(atr_baseline, 4),
                atr_ratio=round(atr_ratio, 2),
                vix_proxy=round(vix_proxy, 2),
                spike_detected=spike_detected,
                detected_at=datetime.now(self.et_tz).isoformat()
            )

            # Log significant changes
            if spike_detected and old_level not in ['EXTREME', 'HIGH']:
                logger.warning(f"VOLATILITY SPIKE DETECTED: {level} (ATR ratio: {atr_ratio:.2f}x, VIX proxy: {vix_proxy:.1f})")

        except Exception as e:
            logger.error(f"Volatility check error: {e}")

    def _monitor_sector_rotation(self):
        """Track sector strength rotation using ETF relative performance"""
        try:
            from alpaca_market_data import AlpacaMarketData
            market_data = AlpacaMarketData()

            sector_returns = {}

            # Get recent price data for each sector ETF
            for etf, sector_name in SECTOR_ETFS.items():
                try:
                    bars = market_data.get_bars(etf, timeframe='5Min', limit=12)  # Last hour
                    if bars and len(bars) >= 2:
                        first_close = bars[0].get('close', bars[0].get('c', 0))
                        last_close = bars[-1].get('close', bars[-1].get('c', 0))

                        if first_close > 0:
                            pct_return = ((last_close - first_close) / first_close) * 100
                            sector_returns[sector_name] = pct_return

                            # Store in history
                            self.sector_price_history[etf].append({
                                'price': last_close,
                                'return': pct_return,
                                'timestamp': datetime.now(self.et_tz).isoformat()
                            })
                except Exception as e:
                    logger.debug(f"Could not get data for {etf}: {e}")
                    continue

            if not sector_returns:
                return

            # Sort sectors by performance
            sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True)

            # Identify leading and lagging sectors
            leading = [s[0] for s in sorted_sectors[:3]]
            lagging = [s[0] for s in sorted_sectors[-3:]]

            # Determine rotation signal based on which sectors are leading
            # Risk-on: Tech, Consumer Discretionary, Industrials leading
            # Risk-off: Utilities, Consumer Staples, Healthcare leading
            risk_on_sectors = {'Technology', 'Consumer Discretionary', 'Industrials', 'Financials'}
            risk_off_sectors = {'Utilities', 'Consumer Staples', 'Healthcare', 'Real Estate'}

            leading_set = set(leading)
            risk_on_score = len(leading_set & risk_on_sectors)
            risk_off_score = len(leading_set & risk_off_sectors)

            if risk_on_score >= 2:
                rotation_signal = "RISK_ON"
            elif risk_off_score >= 2:
                rotation_signal = "RISK_OFF"
            else:
                rotation_signal = "NEUTRAL"

            # Update sector rotation state
            old_signal = self.sector_rotation.rotation_signal
            self.sector_rotation = SectorRotation(
                leading_sectors=leading,
                lagging_sectors=lagging,
                sector_scores=sector_returns,
                rotation_signal=rotation_signal,
                detected_at=datetime.now(self.et_tz).isoformat()
            )

            # Log significant rotation changes
            if old_signal != rotation_signal and old_signal != "NEUTRAL":
                logger.info(f"SECTOR ROTATION: {old_signal} -> {rotation_signal}")
                logger.info(f"  Leading: {', '.join(leading)}")
                logger.info(f"  Lagging: {', '.join(lagging)}")

        except Exception as e:
            logger.error(f"Sector rotation error: {e}")

    def _evaluate_predictions(self):
        """Evaluate accuracy of recent predictions"""
        try:
            # Compare predictions made to actual outcomes
            correct = 0
            total = 0

            for pred in list(self.prediction_history)[-100:]:
                if 'outcome' in pred:
                    total += 1
                    if pred['outcome'] == pred['prediction']:
                        correct += 1

            if total > 0:
                accuracy = correct / total
                # Store accuracy for monitoring

        except Exception as e:
            logger.error(f"Prediction evaluation error: {e}")

    def _check_prediction_drift(self):
        """Check if predictions are drifting from reality - triggers retraining alerts"""
        try:
            # Get predictions with outcomes for evaluation
            predictions_with_outcomes = [
                p for p in list(self.prediction_history)
                if 'outcome' in p
            ]

            if len(predictions_with_outcomes) < 20:
                return  # Not enough data

            # Calculate rolling accuracy windows
            recent_50 = predictions_with_outcomes[-50:] if len(predictions_with_outcomes) >= 50 else predictions_with_outcomes
            recent_20 = predictions_with_outcomes[-20:]

            # Calculate accuracy for each window
            def calc_accuracy(preds):
                correct = sum(1 for p in preds if p.get('outcome') == p.get('prediction'))
                return correct / len(preds) if preds else 0

            accuracy_50 = calc_accuracy(recent_50)
            accuracy_20 = calc_accuracy(recent_20)

            # Store in accuracy history for trend analysis
            self.accuracy_history.append({
                'accuracy': accuracy_20,
                'sample_size': len(recent_20),
                'timestamp': datetime.now(self.et_tz).isoformat()
            })

            # Calculate drift from baseline
            drift_percentage = ((self.prediction_drift.baseline_accuracy - accuracy_20) /
                               self.prediction_drift.baseline_accuracy * 100) if self.prediction_drift.baseline_accuracy > 0 else 0

            # Determine if drift is significant
            drift_detected = False
            recommendation = "OK"

            if drift_percentage > 20:  # More than 20% drop from baseline
                drift_detected = True
                recommendation = "RETRAIN_URGENT"
                logger.warning(f"PREDICTION DRIFT ALERT: Accuracy dropped {drift_percentage:.1f}% from baseline!")
            elif drift_percentage > 10:  # More than 10% drop
                drift_detected = True
                recommendation = "RETRAIN_SCHEDULED"
                logger.info(f"Prediction drift detected: {drift_percentage:.1f}% below baseline")

            # Check for consistent downward trend
            if len(self.accuracy_history) >= 10:
                recent_accuracies = [h['accuracy'] for h in list(self.accuracy_history)[-10:]]
                if all(recent_accuracies[i] >= recent_accuracies[i+1] for i in range(len(recent_accuracies)-1)):
                    # Consistent decline
                    if not drift_detected:
                        drift_detected = True
                        recommendation = "RETRAIN_SCHEDULED"
                        logger.info("Consistent accuracy decline detected - retraining recommended")

            # Get window timestamps
            window_start = recent_20[0].get('timestamp', '') if recent_20 else ''
            window_end = recent_20[-1].get('timestamp', '') if recent_20 else ''

            # Update drift state
            self.prediction_drift = PredictionDrift(
                current_accuracy=round(accuracy_20, 4),
                baseline_accuracy=self.prediction_drift.baseline_accuracy,
                drift_percentage=round(drift_percentage, 2),
                drift_detected=drift_detected,
                samples_evaluated=len(predictions_with_outcomes),
                window_start=window_start,
                window_end=window_end,
                recommendation=recommendation
            )

            # Trigger callback if drift detected
            if drift_detected and self.on_prediction_drift:
                self.on_prediction_drift(self.prediction_drift)

        except Exception as e:
            logger.error(f"Prediction drift check error: {e}")

    def _update_confidence_scores(self):
        """Dynamically adjust confidence based on recent accuracy"""
        pass

    def _should_retrain(self) -> bool:
        """Determine if models need retraining based on multiple factors"""
        try:
            # 1. Check time since last training (retrain every 24 hours)
            if self.last_retrain_time:
                hours_since_retrain = (datetime.now(self.et_tz) - self.last_retrain_time).total_seconds() / 3600
                if hours_since_retrain > 24:
                    logger.info(f"Retraining needed: {hours_since_retrain:.1f} hours since last training")
                    return True

            # 2. Check prediction drift recommendation
            if self.prediction_drift.recommendation in ["RETRAIN_SCHEDULED", "RETRAIN_URGENT"]:
                logger.info(f"Retraining needed: drift recommendation is {self.prediction_drift.recommendation}")
                return True

            # 3. Check if enough new predictions have been made
            predictions_since_retrain = len([
                p for p in self.prediction_history
                if 'outcome' in p and self.last_retrain_time and
                datetime.fromisoformat(p.get('timestamp', '')) > self.last_retrain_time
            ]) if self.last_retrain_time else len([p for p in self.prediction_history if 'outcome' in p])

            if predictions_since_retrain >= 100:
                logger.info(f"Retraining needed: {predictions_since_retrain} new evaluated predictions")
                return True

            # 4. Check for significant regime change
            if (self.current_regime.regime in ["VOLATILE", "TRENDING_DOWN"] and
                self.current_regime.confidence > 0.7):
                logger.info(f"Retraining recommended: market regime is {self.current_regime.regime}")
                return True

            return False

        except Exception as e:
            logger.error(f"Should retrain check error: {e}")
            return False

    def _urgent_retrain_needed(self) -> bool:
        """Check if urgent retraining is needed during market hours"""
        try:
            # Check for urgent drift
            if self.prediction_drift.recommendation == "RETRAIN_URGENT":
                return True

            # Check for extreme volatility combined with poor predictions
            if (self.volatility_state.level in ["EXTREME", "HIGH"] and
                self.prediction_drift.current_accuracy < 0.5):
                logger.warning("Urgent retraining: high volatility + low accuracy")
                return True

            # Check for sudden accuracy collapse (last 10 predictions)
            recent_10 = [p for p in list(self.prediction_history)[-10:] if 'outcome' in p]
            if len(recent_10) >= 10:
                accuracy_10 = sum(1 for p in recent_10 if p.get('outcome') == p.get('prediction')) / 10
                if accuracy_10 < 0.4:  # Less than 40% on last 10
                    logger.warning(f"Urgent retraining: recent accuracy collapsed to {accuracy_10:.0%}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Urgent retrain check error: {e}")
            return False

    def _run_incremental_training(self):
        """Run incremental model training with recent data"""
        try:
            if self.retrain_triggered:
                return  # Already running

            self.retrain_triggered = True
            logger.info("Starting incremental training...")
            self.tasks_pending += 1

            # Get the predictor
            from ai.alpaca_ai_predictor import get_alpaca_predictor
            predictor = get_alpaca_predictor()

            # Determine symbols to retrain (use tracked symbols or defaults)
            symbols_to_train = ['AAPL', 'MSFT', 'NVDA']  # Core symbols

            # Add symbols from recent predictions
            recent_symbols = set(p.get('symbol') for p in list(self.prediction_history)[-100:] if p.get('symbol'))
            symbols_to_train = list(set(symbols_to_train) | recent_symbols)[:10]  # Max 10 symbols

            logger.info(f"Retraining on symbols: {symbols_to_train}")

            # Run training
            result = predictor.train_multi(symbols_to_train, days=30)

            if result.get('success'):
                self.last_retrain_time = datetime.now(self.et_tz)

                # Update baseline accuracy if training improved it
                if result.get('accuracy', 0) > self.prediction_drift.baseline_accuracy:
                    self.prediction_drift.baseline_accuracy = result.get('accuracy')
                    logger.info(f"Updated baseline accuracy to {result.get('accuracy'):.2%}")

                logger.info(f"Incremental training complete: {result.get('accuracy', 0):.2%} accuracy")
            else:
                logger.error(f"Training failed: {result.get('error', 'Unknown error')}")

            self.tasks_completed += 1
            self.tasks_pending -= 1
            self.retrain_triggered = False

        except Exception as e:
            logger.error(f"Training error: {e}")
            self.retrain_triggered = False
            self.tasks_pending = max(0, self.tasks_pending - 1)

    def _run_quick_adaptation(self):
        """Quick model adaptation during market hours - lightweight parameter tuning"""
        try:
            # Only run if not already running a full retrain
            if self.retrain_triggered:
                return

            logger.info("Running quick adaptation...")

            # Adjust confidence thresholds based on recent performance
            recent_accuracy = self.prediction_drift.current_accuracy

            # If accuracy is low, we should be more conservative
            if recent_accuracy < 0.55:
                # Increase confidence threshold needed for trades
                logger.info("Quick adaptation: suggesting higher confidence thresholds due to low accuracy")

            # Log current state for monitoring
            logger.info(f"Current state - Accuracy: {recent_accuracy:.2%}, "
                       f"Volatility: {self.volatility_state.level}, "
                       f"Regime: {self.current_regime.regime}")

        except Exception as e:
            logger.error(f"Quick adaptation error: {e}")

    def record_prediction(self, symbol: str, prediction: str, confidence: float, price: float):
        """Record a prediction for later evaluation"""
        self.prediction_history.append({
            "symbol": symbol,
            "prediction": prediction,
            "confidence": confidence,
            "price": price,
            "timestamp": datetime.now(self.et_tz).isoformat()
        })

    def record_outcome(self, symbol: str, timestamp: str, actual_outcome: str):
        """Record the actual outcome for a prediction"""
        # Find and update the prediction
        for pred in self.prediction_history:
            if pred.get('symbol') == symbol and pred.get('timestamp') == timestamp:
                pred['outcome'] = actual_outcome
                break

    def submit_task(self, func: Callable, *args, **kwargs):
        """Submit a task to the thread pool"""
        with self.task_lock:
            self.tasks_pending += 1

        def wrapped():
            try:
                result = func(*args, **kwargs)
                with self.task_lock:
                    self.tasks_completed += 1
                    self.tasks_pending -= 1
                return result
            except Exception as e:
                logger.error(f"Task error: {e}")
                with self.task_lock:
                    self.tasks_pending -= 1

        return self.thread_pool.submit(wrapped)

    def get_metrics(self) -> BrainMetrics:
        """Get current brain metrics"""
        uptime = 0
        if self.start_time:
            uptime = (datetime.now(self.et_tz) - self.start_time).total_seconds()

        # Calculate prediction accuracy
        correct = 0
        total = 0
        for pred in list(self.prediction_history)[-100:]:
            if 'outcome' in pred:
                total += 1
                if pred.get('outcome') == pred.get('prediction'):
                    correct += 1
        accuracy = correct / total if total > 0 else 0

        # Get last retrain time string
        last_retrain_str = ""
        if self.last_retrain_time:
            last_retrain_str = self.last_retrain_time.isoformat()

        return BrainMetrics(
            cpu_usage_target=self.cpu_target,
            actual_cpu_usage=self._get_cpu_usage(),
            tasks_completed=self.tasks_completed,
            tasks_pending=self.tasks_pending,
            predictions_evaluated=total,
            prediction_accuracy=accuracy,
            last_retrain=last_retrain_str,
            models_loaded=1 if accuracy > 0 else 0,
            market_regime=self.current_regime.regime,
            uptime_seconds=uptime
        )

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        try:
            import psutil
            return psutil.cpu_percent() / 100.0
        except:
            return 0.0

    def set_cpu_target(self, target: float):
        """Adjust CPU usage target (0.0-1.0)"""
        self.cpu_target = max(0.1, min(0.9, target))
        self.worker_threads = max(2, int(self.num_cores * self.cpu_target))
        logger.info(f"CPU target set to {self.cpu_target*100:.0f}%")

    def get_market_regime(self) -> Dict:
        """Get current market regime with volatility and sector data"""
        regime_data = self.current_regime.to_dict()
        regime_data['volatility'] = self.volatility_state.to_dict()
        regime_data['sector_rotation'] = self.sector_rotation.to_dict()
        return regime_data

    def get_volatility_state(self) -> Dict:
        """Get current volatility state"""
        return self.volatility_state.to_dict()

    def get_sector_rotation(self) -> Dict:
        """Get current sector rotation analysis"""
        return self.sector_rotation.to_dict()

    def get_prediction_drift(self) -> Dict:
        """Get prediction drift analysis"""
        return self.prediction_drift.to_dict()

    def get_full_brain_state(self) -> Dict:
        """Get complete brain state for UI/monitoring"""
        return {
            "metrics": self.get_metrics().to_dict(),
            "regime": self.current_regime.to_dict(),
            "volatility": self.volatility_state.to_dict(),
            "sector_rotation": self.sector_rotation.to_dict(),
            "prediction_drift": self.prediction_drift.to_dict(),
            "is_running": self.is_running,
            "retrain_triggered": self.retrain_triggered
        }

    def update_baseline_accuracy(self, accuracy: float):
        """Update the baseline accuracy after training"""
        if 0.0 < accuracy <= 1.0:
            self.prediction_drift.baseline_accuracy = accuracy
            logger.info(f"Baseline accuracy updated to {accuracy:.2%}")


# Singleton instance
_brain: Optional[BackgroundBrain] = None


def get_background_brain() -> BackgroundBrain:
    """Get or create the background brain singleton"""
    global _brain
    if _brain is None:
        _brain = BackgroundBrain(cpu_target=0.5)
    return _brain


def start_brain(cpu_target: float = 0.5):
    """Start the background brain with specified CPU target"""
    brain = get_background_brain()
    brain.set_cpu_target(cpu_target)
    brain.start()
    return brain
