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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pytz

logger = logging.getLogger(__name__)

# Configuration
BRAIN_STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "store", "brain_state.json")


@dataclass
class MarketRegime:
    """Current market regime classification"""
    regime: str  # TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, CALM
    confidence: float
    detected_at: str
    indicators: Dict

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
        """Detect sudden volatility changes"""
        # Would analyze ATR, VIX, or price swings
        pass

    def _monitor_sector_rotation(self):
        """Track sector strength rotation"""
        # Would analyze relative sector performance
        pass

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
        """Check if predictions are drifting from reality"""
        # Compare expected vs actual outcomes
        # Trigger retraining if drift exceeds threshold
        pass

    def _update_confidence_scores(self):
        """Dynamically adjust confidence based on recent accuracy"""
        pass

    def _should_retrain(self) -> bool:
        """Determine if models need retraining"""
        # Check:
        # 1. Time since last training
        # 2. Prediction accuracy degradation
        # 3. Market regime change
        # 4. New data availability
        return False  # Placeholder

    def _urgent_retrain_needed(self) -> bool:
        """Check if urgent retraining is needed"""
        # Significant accuracy drop or regime change
        return False

    def _run_incremental_training(self):
        """Run incremental model training"""
        try:
            logger.info("Starting incremental training...")
            self.tasks_pending += 1

            # Submit to process pool for CPU-intensive work
            if not self.process_pool:
                self.process_pool = ProcessPoolExecutor(max_workers=self.worker_threads)

            # Training logic would go here
            # This would update models with recent data

            self.tasks_completed += 1
            self.tasks_pending -= 1
            logger.info("Incremental training complete")

        except Exception as e:
            logger.error(f"Training error: {e}")

    def _run_quick_adaptation(self):
        """Quick model adaptation during market hours"""
        # Lightweight updates that don't impact trading
        pass

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

        return BrainMetrics(
            cpu_usage_target=self.cpu_target,
            actual_cpu_usage=self._get_cpu_usage(),
            tasks_completed=self.tasks_completed,
            tasks_pending=self.tasks_pending,
            predictions_evaluated=total,
            prediction_accuracy=accuracy,
            last_retrain="",  # Would track this
            models_loaded=0,  # Would count loaded models
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
        """Get current market regime"""
        return self.current_regime.to_dict()


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
