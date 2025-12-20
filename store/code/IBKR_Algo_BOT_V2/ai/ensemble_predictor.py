"""
Ensemble AI Predictor
=====================
Combines multiple prediction methods for robust trading signals:
- LightGBM (ML model)
- Chronos (Amazon foundation model for time series)
- Heuristic rules (technical analysis patterns)
- Momentum scoring
- Regime-aware weighting

The ensemble uses weighted voting based on market conditions
and recent performance of each component.

New Technologies (Dec 2024):
- Amazon Chronos: Zero-shot time series forecasting
- Google TimesFM: Foundation model (requires Python 3.11)
- Microsoft Qlib: Quant research platform (requires Python 3.11)
- FinRL: Reinforcement learning for trading

Created: December 2025
Updated: December 2024 - Added Chronos integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging
import ta

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Result from ensemble prediction"""
    symbol: str
    timestamp: str

    # Final prediction
    prediction: int  # 1 = bullish, 0 = bearish
    confidence: float  # 0.0 to 1.0

    # Component scores (0.0 to 1.0)
    lgb_score: float
    chronos_score: float  # Amazon Chronos foundation model
    heuristic_score: float
    momentum_score: float

    # Weights used
    lgb_weight: float
    chronos_weight: float
    heuristic_weight: float
    momentum_weight: float

    # Regime context
    market_regime: str  # TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE
    regime_confidence: float

    # RL agent recommendation
    rl_action: str = "N/A"  # HOLD, BUY, SELL
    rl_confidence: float = 0.0

    # Explanation
    signals: List[str] = field(default_factory=list)  # Human-readable signal descriptions

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class HeuristicSignal:
    """Individual heuristic signal"""
    name: str
    score: float  # -1.0 to 1.0 (bearish to bullish)
    confidence: float  # 0.0 to 1.0
    description: str


class MarketRegimeDetector:
    """
    Detect current market regime for adaptive weighting.
    Regimes: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE
    """

    def __init__(self):
        self.regime_history: List[str] = []

    def detect(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Detect market regime from price data.

        Returns:
            (regime, confidence)
        """
        if len(df) < 50:
            return "RANGING", 0.5

        close = df['Close'].values

        # Calculate indicators
        sma_20 = df['Close'].rolling(20).mean().iloc[-1]
        sma_50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
        current_price = close[-1]

        # ADX for trend strength
        try:
            adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
            adx_value = adx.adx().iloc[-1]
            adx_pos = adx.adx_pos().iloc[-1]
            adx_neg = adx.adx_neg().iloc[-1]
        except:
            adx_value = 20
            adx_pos = 20
            adx_neg = 20

        # Volatility
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        avg_volatility = 0.25  # Average market volatility ~25%

        # Determine regime
        regime = "RANGING"
        confidence = 0.5

        # High volatility regime
        if volatility > avg_volatility * 1.5:
            regime = "VOLATILE"
            confidence = min(volatility / avg_volatility - 0.5, 1.0)

        # Strong trend
        elif adx_value > 25:
            if adx_pos > adx_neg and current_price > sma_20:
                regime = "TRENDING_UP"
                confidence = min(adx_value / 50, 1.0)
            elif adx_neg > adx_pos and current_price < sma_20:
                regime = "TRENDING_DOWN"
                confidence = min(adx_value / 50, 1.0)
            else:
                regime = "RANGING"
                confidence = 0.6

        # Weak trend / ranging
        else:
            regime = "RANGING"
            confidence = 1.0 - (adx_value / 25)

        self.regime_history.append(regime)
        if len(self.regime_history) > 50:
            self.regime_history.pop(0)

        return regime, confidence


class HeuristicEngine:
    """
    Technical analysis heuristics for trading signals.
    Uses proven patterns and indicators.
    """

    def __init__(self):
        self.signal_weights = {
            'macd_crossover': 1.0,
            'rsi_oversold': 0.8,
            'rsi_overbought': 0.8,
            'golden_cross': 1.2,
            'death_cross': 1.2,
            'bollinger_squeeze': 0.7,
            'volume_breakout': 0.9,
            'momentum_divergence': 1.0,
            'support_bounce': 0.8,
            'resistance_rejection': 0.8,
        }

    def analyze(self, df: pd.DataFrame) -> Tuple[float, List[HeuristicSignal]]:
        """
        Analyze price data and return heuristic score.

        Returns:
            (score 0-1, list of signals)
        """
        signals = []

        if len(df) < 50:
            return 0.5, signals

        # Calculate indicators
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # MACD
        macd = ta.trend.MACD(close)
        macd_line = macd.macd().iloc[-1]
        macd_signal = macd.macd_signal().iloc[-1]
        macd_prev = macd.macd().iloc[-2]
        macd_signal_prev = macd.macd_signal().iloc[-2]

        # MACD Bullish Crossover
        if macd_prev < macd_signal_prev and macd_line > macd_signal:
            signals.append(HeuristicSignal(
                name='macd_crossover',
                score=0.7,
                confidence=0.8,
                description='MACD bullish crossover'
            ))
        # MACD Bearish Crossover
        elif macd_prev > macd_signal_prev and macd_line < macd_signal:
            signals.append(HeuristicSignal(
                name='macd_crossover',
                score=-0.7,
                confidence=0.8,
                description='MACD bearish crossover'
            ))

        # RSI
        rsi = ta.momentum.rsi(close, window=14).iloc[-1]

        if rsi < 30:
            signals.append(HeuristicSignal(
                name='rsi_oversold',
                score=0.6,
                confidence=0.7,
                description=f'RSI oversold ({rsi:.1f})'
            ))
        elif rsi > 70:
            signals.append(HeuristicSignal(
                name='rsi_overbought',
                score=-0.6,
                confidence=0.7,
                description=f'RSI overbought ({rsi:.1f})'
            ))

        # Moving Average Crossovers
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1] if len(df) >= 200 else sma_50
        sma_50_prev = close.rolling(50).mean().iloc[-2]
        sma_200_prev = close.rolling(200).mean().iloc[-2] if len(df) >= 200 else sma_50_prev

        # Golden Cross
        if sma_50_prev < sma_200_prev and sma_50 > sma_200:
            signals.append(HeuristicSignal(
                name='golden_cross',
                score=0.9,
                confidence=0.9,
                description='Golden cross (50 SMA > 200 SMA)'
            ))
        # Death Cross
        elif sma_50_prev > sma_200_prev and sma_50 < sma_200:
            signals.append(HeuristicSignal(
                name='death_cross',
                score=-0.9,
                confidence=0.9,
                description='Death cross (50 SMA < 200 SMA)'
            ))

        # Bollinger Bands Squeeze
        bb = ta.volatility.BollingerBands(close)
        bb_width = bb.bollinger_wband().iloc[-1]
        bb_width_avg = bb.bollinger_wband().rolling(20).mean().iloc[-1]

        if bb_width < bb_width_avg * 0.5:
            # Squeeze - breakout expected
            signals.append(HeuristicSignal(
                name='bollinger_squeeze',
                score=0.3,  # Neutral but noteworthy
                confidence=0.6,
                description='Bollinger Band squeeze - breakout expected'
            ))

        # Volume Breakout
        vol_avg = volume.rolling(20).mean().iloc[-1]
        vol_current = volume.iloc[-1]
        price_change = (close.iloc[-1] / close.iloc[-2] - 1) * 100

        if vol_current > vol_avg * 2 and price_change > 1:
            signals.append(HeuristicSignal(
                name='volume_breakout',
                score=0.7,
                confidence=0.8,
                description=f'Volume breakout ({vol_current/vol_avg:.1f}x avg)'
            ))
        elif vol_current > vol_avg * 2 and price_change < -1:
            signals.append(HeuristicSignal(
                name='volume_breakout',
                score=-0.7,
                confidence=0.8,
                description=f'Heavy selling volume ({vol_current/vol_avg:.1f}x avg)'
            ))

        # Support/Resistance (simple approach using recent lows/highs)
        recent_low = low.tail(20).min()
        recent_high = high.tail(20).max()
        current_price = close.iloc[-1]

        # Near support
        if current_price <= recent_low * 1.02:
            signals.append(HeuristicSignal(
                name='support_bounce',
                score=0.5,
                confidence=0.6,
                description='Price near 20-day support'
            ))
        # Near resistance
        elif current_price >= recent_high * 0.98:
            signals.append(HeuristicSignal(
                name='resistance_rejection',
                score=-0.3,
                confidence=0.5,
                description='Price near 20-day resistance'
            ))

        # Calculate aggregate score
        if not signals:
            return 0.5, signals

        total_weight = sum(self.signal_weights.get(s.name, 1.0) * s.confidence for s in signals)
        weighted_score = sum(
            s.score * self.signal_weights.get(s.name, 1.0) * s.confidence
            for s in signals
        )

        # Normalize to 0-1 range
        raw_score = weighted_score / total_weight if total_weight > 0 else 0
        normalized_score = (raw_score + 1) / 2  # Convert from [-1, 1] to [0, 1]

        return normalized_score, signals


class ChronosScorer:
    """
    Amazon Chronos foundation model scorer.
    Provides zero-shot time series forecasting.
    """

    def __init__(self):
        self.predictor = None
        self.available = False
        self._init_predictor()

    def _init_predictor(self):
        """Initialize Chronos predictor (lazy load on first use)."""
        try:
            from ai.chronos_predictor import get_chronos_predictor
            self.predictor = get_chronos_predictor()
            self.available = True
            logger.info("Chronos predictor initialized")
        except Exception as e:
            logger.warning(f"Chronos not available: {e}")
            self.available = False

    def calculate(self, symbol: str, df: pd.DataFrame = None) -> Tuple[float, Dict]:
        """
        Get Chronos prediction score.

        Returns:
            (score 0-1, details dict)
        """
        if not self.available or self.predictor is None:
            return 0.5, {"available": False}

        try:
            result = self.predictor.predict(symbol, horizon=5)

            if "error" in result:
                return 0.5, {"available": False, "error": result["error"]}

            prob_up = result.get("probabilities", {}).get("prob_up", 0.5)
            expected_return = result.get("expected_return_pct", 0)
            confidence = result.get("confidence", 0.5)

            return prob_up, {
                "available": True,
                "signal": result.get("signal", "NEUTRAL"),
                "prob_up": prob_up,
                "expected_return": expected_return,
                "confidence": confidence,
                "forecast": result.get("forecast", {}),
            }
        except Exception as e:
            logger.warning(f"Chronos calculation failed: {e}")
            return 0.5, {"available": False, "error": str(e)}


class MomentumScorer:
    """
    Multi-timeframe momentum scoring.
    """

    def calculate(self, df: pd.DataFrame) -> float:
        """
        Calculate momentum score (0-1).
        Higher = stronger bullish momentum.
        """
        if len(df) < 50:
            return 0.5

        close = df['Close']

        # Multi-timeframe returns
        ret_1d = (close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 1 else 0
        ret_5d = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) > 5 else 0
        ret_20d = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) > 20 else 0

        # ROC (Rate of Change)
        roc = ta.momentum.roc(close, window=12).iloc[-1]

        # Momentum indicator
        mom = ta.momentum.awesome_oscillator(df['High'], df['Low']).iloc[-1]
        mom_normalized = np.tanh(mom / 5)  # Normalize

        # Combine with weights (more recent = more weight)
        momentum_score = (
            0.4 * np.tanh(ret_1d * 50) +  # 1-day (high weight, normalize strong moves)
            0.3 * np.tanh(ret_5d * 20) +  # 5-day
            0.2 * np.tanh(ret_20d * 10) + # 20-day
            0.1 * mom_normalized          # Awesome oscillator
        )

        # Convert from [-1, 1] to [0, 1]
        return (momentum_score + 1) / 2


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple methods.

    Components:
    1. LightGBM ML model (when available)
    2. Heuristic rules (technical analysis)
    3. Momentum scoring
    4. RL agent (entry/exit timing optimization)

    Weights are adjusted based on market regime and recent performance.
    """

    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.heuristic_engine = HeuristicEngine()
        self.momentum_scorer = MomentumScorer()
        self.chronos_scorer = ChronosScorer()

        # Try to load LightGBM predictor
        self.lgb_predictor = None
        try:
            from ai.alpaca_ai_predictor import get_alpaca_predictor
            self.lgb_predictor = get_alpaca_predictor()
            logger.info("LightGBM predictor loaded")
        except Exception as e:
            logger.warning(f"LightGBM predictor not available: {e}")

        # Try to load RL agent
        self.rl_agent = None
        self.rl_env = None
        try:
            from ai.rl_trading_agent import get_rl_agent, get_rl_environment
            self.rl_agent = get_rl_agent()
            self.rl_env = get_rl_environment()
            logger.info("RL agent loaded")
        except Exception as e:
            logger.warning(f"RL agent not available: {e}")

        # Base weights (adjusted by regime)
        # Chronos gets high weight as it's a foundation model with zero-shot capability
        self.base_weights = {
            'lgb': 0.30,
            'chronos': 0.35,  # Foundation model - most sophisticated
            'heuristic': 0.20,
            'momentum': 0.15
        }

        # Regime-specific weight adjustments
        self.regime_adjustments = {
            'TRENDING_UP': {'lgb': 0.25, 'chronos': 0.35, 'heuristic': 0.20, 'momentum': 0.20},
            'TRENDING_DOWN': {'lgb': 0.25, 'chronos': 0.30, 'heuristic': 0.30, 'momentum': 0.15},
            'RANGING': {'lgb': 0.30, 'chronos': 0.35, 'heuristic': 0.25, 'momentum': 0.10},
            'VOLATILE': {'lgb': 0.20, 'chronos': 0.25, 'heuristic': 0.40, 'momentum': 0.15},
        }

        # Performance tracking for adaptive weighting
        self.recent_predictions: List[Dict] = []
        self.max_history = 100

        logger.info("EnsemblePredictor initialized")

    def predict(self, symbol: str, df: pd.DataFrame) -> EnsemblePrediction:
        """
        Generate ensemble prediction for a symbol.

        Args:
            symbol: Stock symbol
            df: Price data with OHLCV columns

        Returns:
            EnsemblePrediction with all component scores
        """
        timestamp = datetime.now().isoformat()
        signals = []

        # Detect market regime
        regime, regime_confidence = self.regime_detector.detect(df)

        # Get regime-specific weights
        weights = self.regime_adjustments.get(regime, self.base_weights).copy()

        # 1. LightGBM Score
        lgb_score = 0.5  # Default neutral
        if self.lgb_predictor and self.lgb_predictor.model is not None:
            try:
                lgb_result = self.lgb_predictor.predict(symbol, df=df)
                if lgb_result and 'prob_up' in lgb_result:
                    lgb_score = lgb_result['prob_up']
                    signals.append(f"LightGBM: {lgb_score:.2%} bullish ({lgb_result.get('signal', 'N/A')})")
            except Exception as e:
                logger.debug(f"LightGBM prediction failed: {e}")
                weights['lgb'] = 0  # Disable LGB weight
        else:
            weights['lgb'] = 0  # No model available

        # Redistribute weights if LGB not available
        if weights['lgb'] == 0:
            remaining = self.base_weights['lgb']
            weights['chronos'] += remaining * 0.5
            weights['heuristic'] += remaining * 0.3
            weights['momentum'] += remaining * 0.2

        # 2. Chronos Score (Amazon Foundation Model)
        chronos_score = 0.5
        chronos_details = {}
        if self.chronos_scorer.available:
            try:
                chronos_score, chronos_details = self.chronos_scorer.calculate(symbol, df)
                if chronos_details.get('available'):
                    signals.append(f"Chronos: {chronos_score:.2%} bullish ({chronos_details.get('signal', 'N/A')}, exp_ret: {chronos_details.get('expected_return', 0):.1f}%)")
                else:
                    weights['chronos'] = 0
            except Exception as e:
                logger.debug(f"Chronos prediction failed: {e}")
                weights['chronos'] = 0
        else:
            weights['chronos'] = 0

        # Redistribute weights if Chronos not available
        if weights.get('chronos', 0) == 0:
            remaining = self.base_weights.get('chronos', 0.35)
            weights['heuristic'] += remaining * 0.5
            weights['momentum'] += remaining * 0.5

        # 3. Heuristic Score
        heuristic_score, heuristic_signals = self.heuristic_engine.analyze(df)
        for hs in heuristic_signals:
            signals.append(f"Heuristic: {hs.description}")

        # 4. Momentum Score
        momentum_score = self.momentum_scorer.calculate(df)
        if momentum_score > 0.6:
            signals.append(f"Momentum: Strong bullish ({momentum_score:.2%})")
        elif momentum_score < 0.4:
            signals.append(f"Momentum: Bearish ({momentum_score:.2%})")
        else:
            signals.append(f"Momentum: Neutral ({momentum_score:.2%})")

        # Calculate weighted ensemble score (includes Chronos)
        total_weight = weights['lgb'] + weights.get('chronos', 0) + weights['heuristic'] + weights['momentum']

        ensemble_score = (
            lgb_score * weights['lgb'] +
            chronos_score * weights.get('chronos', 0) +
            heuristic_score * weights['heuristic'] +
            momentum_score * weights['momentum']
        ) / total_weight

        # Determine prediction and confidence
        prediction = 1 if ensemble_score > 0.5 else 0

        # Confidence based on:
        # 1. Distance from 0.5 (strength of signal)
        # 2. Agreement between components
        # 3. Regime confidence

        signal_strength = abs(ensemble_score - 0.5) * 2  # 0 to 1

        # Agreement score (how much components agree) - includes Chronos
        scores = [lgb_score, chronos_score, heuristic_score, momentum_score]
        score_std = np.std(scores)
        agreement = 1 - min(score_std * 2, 1)  # Lower std = higher agreement

        confidence = (
            signal_strength * 0.4 +
            agreement * 0.3 +
            regime_confidence * 0.3
        )

        signals.append(f"Regime: {regime} ({regime_confidence:.0%} confidence)")

        # 4. RL Agent recommendation (if available)
        rl_action = "N/A"
        rl_confidence = 0.0

        if self.rl_agent is not None and self.rl_env is not None:
            try:
                from ai.rl_trading_agent import TradingState

                # Build state for RL agent
                close = df['Close']
                returns = close.pct_change().dropna()

                # Calculate state features
                rsi = ta.momentum.rsi(close, window=14).iloc[-1] if len(close) > 14 else 50

                macd_obj = ta.trend.MACD(close)
                macd_line = macd_obj.macd().iloc[-1] if len(close) > 26 else 0
                macd_signal = macd_obj.macd_signal().iloc[-1] if len(close) > 26 else 0

                bb = ta.volatility.BollingerBands(close)
                bb_high = bb.bollinger_hband().iloc[-1]
                bb_low = bb.bollinger_lband().iloc[-1]
                bb_position = (close.iloc[-1] - bb_low) / (bb_high - bb_low) if bb_high != bb_low else 0.5

                vol_ratio = df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1] if len(df) > 20 else 1.0

                # Trend and volatility
                sma_20 = close.rolling(20).mean().iloc[-1] if len(close) > 20 else close.iloc[-1]
                trend = (close.iloc[-1] - sma_20) / sma_20 if sma_20 > 0 else 0
                volatility = returns.std() * np.sqrt(252) if len(returns) > 5 else 0.25

                # Create trading state with correct field names
                state = TradingState(
                    price_change_1d=returns.iloc[-1] if len(returns) > 0 else 0,
                    price_change_5d=(close.iloc[-1] / close.iloc[-5] - 1) if len(close) > 5 else 0,
                    price_change_20d=(close.iloc[-1] / close.iloc[-20] - 1) if len(close) > 20 else 0,
                    rsi_normalized=(rsi - 50) / 50,  # (RSI - 50) / 50
                    macd_normalized=np.tanh(macd_line),  # Normalize MACD
                    bb_position=bb_position,
                    volume_ratio=min(vol_ratio, 5) / 5,  # Cap at 5x, normalize
                    trend_strength=abs(trend),  # Strength is absolute
                    trend_direction=np.sign(trend),  # Direction is sign
                    volatility=min(volatility, 1),  # Cap at 100% vol
                    ensemble_score=ensemble_score,
                    lgb_score=lgb_score,
                    momentum_score=momentum_score,
                    has_position=0,  # No position info in this context
                    position_pnl=0,
                    holding_time=0,
                    regime_trending=1.0 if regime in ['TRENDING_UP', 'TRENDING_DOWN'] else 0.0,
                    regime_volatile=1.0 if regime == 'VOLATILE' else 0.0
                )

                # Get RL action
                action, action_probs = self.rl_agent.get_action_with_probs(state)

                action_names = ['HOLD', 'BUY', 'SELL']
                rl_action = action_names[action]
                rl_confidence = float(action_probs[action])

                signals.append(f"RL Agent: {rl_action} ({rl_confidence:.1%} confidence)")

            except Exception as e:
                logger.debug(f"RL agent prediction failed: {e}")
                rl_action = "N/A"
                rl_confidence = 0.0

        return EnsemblePrediction(
            symbol=symbol,
            timestamp=timestamp,
            prediction=prediction,
            confidence=confidence,
            lgb_score=lgb_score,
            chronos_score=chronos_score,
            heuristic_score=heuristic_score,
            momentum_score=momentum_score,
            lgb_weight=weights['lgb'] / total_weight,
            chronos_weight=weights.get('chronos', 0) / total_weight,
            heuristic_weight=weights['heuristic'] / total_weight,
            momentum_weight=weights['momentum'] / total_weight,
            market_regime=regime,
            regime_confidence=regime_confidence,
            rl_action=rl_action,
            rl_confidence=rl_confidence,
            signals=signals
        )

    def record_outcome(self, symbol: str, prediction: int, actual: int):
        """Record prediction outcome for adaptive learning"""
        self.recent_predictions.append({
            'symbol': symbol,
            'prediction': prediction,
            'actual': actual,
            'timestamp': datetime.now().isoformat(),
            'correct': prediction == actual
        })

        if len(self.recent_predictions) > self.max_history:
            self.recent_predictions.pop(0)

    def get_performance_stats(self) -> Dict:
        """Get recent performance statistics"""
        if not self.recent_predictions:
            return {'accuracy': 0.0, 'total': 0}

        correct = sum(1 for p in self.recent_predictions if p['correct'])
        total = len(self.recent_predictions)

        return {
            'accuracy': correct / total if total > 0 else 0,
            'total': total,
            'correct': correct,
            'recent_10': sum(1 for p in self.recent_predictions[-10:] if p['correct']) / min(10, total)
        }


# Global instance
_ensemble_predictor: Optional[EnsemblePredictor] = None


def get_ensemble_predictor() -> EnsemblePredictor:
    """Get or create global ensemble predictor instance"""
    global _ensemble_predictor
    if _ensemble_predictor is None:
        _ensemble_predictor = EnsemblePredictor()
    return _ensemble_predictor


# Quick test
if __name__ == "__main__":
    import yfinance as yf

    logging.basicConfig(level=logging.INFO)

    # Test with some symbols
    symbols = ['AAPL', 'TSLA', 'NVDA']

    predictor = get_ensemble_predictor()

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Ensemble Prediction for {symbol}")
        print('='*60)

        # Get data
        df = yf.download(symbol, period="6mo", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Predict
        result = predictor.predict(symbol, df)

        print(f"\nPrediction: {'BULLISH' if result.prediction == 1 else 'BEARISH'}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"\nComponent Scores:")
        print(f"  LightGBM:  {result.lgb_score:.2%} (weight: {result.lgb_weight:.1%})")
        print(f"  Chronos:   {result.chronos_score:.2%} (weight: {result.chronos_weight:.1%})")
        print(f"  Heuristic: {result.heuristic_score:.2%} (weight: {result.heuristic_weight:.1%})")
        print(f"  Momentum:  {result.momentum_score:.2%} (weight: {result.momentum_weight:.1%})")
        print(f"\nRL Agent: {result.rl_action} ({result.rl_confidence:.1%} confidence)")
        print(f"\nMarket Regime: {result.market_regime} ({result.regime_confidence:.0%})")
        print(f"\nSignals:")
        for signal in result.signals:
            print(f"  - {signal}")
