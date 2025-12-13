"""
Trigger Signal Generator
========================
Real-time trigger generation combining MACD/RSI/Bollinger Band signals with ML model predictions.

This module generates actionable BUY/SELL triggers based on:
1. MACD histogram crossovers (BREAKOUT signals)
2. RSI overbought/oversold levels (REVERSION signals)
3. Bollinger Band touches (REVERSION signals)
4. ML model consensus (CNN, LightGBM)

Trigger Types:
- BREAKOUT: Strong MACD crossover + RSI in momentum zone
- REVERSION: RSI extreme + BB touch
- RSI_EXTREME: Very strong RSI signal alone
- MODEL_AGREE: All ML models agree with high confidence
"""

import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import logging
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class TriggerSignal:
    """A trading trigger signal"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    trigger_type: str  # BREAKOUT, REVERSION, RSI_EXTREME, MODEL_AGREE
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    rsi_value: float
    macd_value: float
    bb_position: float  # 0-1, position within Bollinger Bands
    ml_signal: str  # LightGBM signal
    ml_confidence: float
    cnn_signal: Optional[str] = None
    cnn_confidence: Optional[float] = None
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class TechnicalIndicators:
    """Calculate technical indicators for trigger detection"""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI, MACD, and Bollinger Bands"""
        data = df.copy()

        # Flatten columns if multi-index
        if hasattr(data.columns, 'get_level_values'):
            data.columns = data.columns.get_level_values(0)

        close = data['Close']
        if hasattr(close, 'values'):
            close = pd.Series(close.values.flatten(), index=data.index)

        high = data['High']
        if hasattr(high, 'values'):
            high = pd.Series(high.values.flatten(), index=data.index)

        low = data['Low']
        if hasattr(low, 'values'):
            low = pd.Series(low.values.flatten(), index=data.index)

        # RSI
        data['RSI'] = ta.momentum.rsi(close, window=14)

        # MACD
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Hist'] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        data['BB_High'] = bb.bollinger_hband()
        data['BB_Low'] = bb.bollinger_lband()
        data['BB_Mid'] = bb.bollinger_mavg()
        data['BB_Pct'] = bb.bollinger_pband()  # % position within bands

        # ATR for stop loss/take profit
        data['ATR'] = ta.volatility.average_true_range(high, low, close, window=14)

        return data


class TriggerSignalGenerator:
    """
    Generate real-time trading triggers by combining technical analysis with ML predictions.

    Usage:
        generator = TriggerSignalGenerator()
        trigger = generator.generate_trigger("AAPL")
        if trigger.action != "HOLD":
            print(f"Signal: {trigger.action} {trigger.symbol} @ ${trigger.entry_price}")
    """

    def __init__(self):
        self.indicators = TechnicalIndicators()
        self._lgb_predictor = None
        self._cnn_predictor = None
        self._data_cache: Dict[str, Dict] = {}
        self._cache_ttl = 300  # 5 minute cache

    def _get_lgb_predictor(self):
        """Lazy load LightGBM predictor"""
        if self._lgb_predictor is None:
            try:
                from ai.alpaca_ai_predictor import get_alpaca_predictor
                self._lgb_predictor = get_alpaca_predictor()
            except Exception as e:
                logger.warning(f"Could not load LightGBM predictor: {e}")
        return self._lgb_predictor

    def _get_cnn_predictor(self):
        """Lazy load CNN predictor"""
        if self._cnn_predictor is None:
            try:
                from ai.cnn_stock_predictor import get_cnn_predictor
                self._cnn_predictor = get_cnn_predictor()
            except Exception as e:
                logger.warning(f"Could not load CNN predictor: {e}")
        return self._cnn_predictor

    def _get_data(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """Get price data with caching"""
        cache_key = f"{symbol}_{days}"
        now = datetime.now()

        if cache_key in self._data_cache:
            entry = self._data_cache[cache_key]
            if (now - entry['timestamp']).total_seconds() < self._cache_ttl:
                return entry['data'].copy()

        # Download fresh data
        end_date = now
        start_date = end_date - timedelta(days=days)

        df = yf.download(symbol, start=start_date, end=end_date, progress=False)

        if df.empty:
            raise ValueError(f"No data available for {symbol}")

        # Flatten columns
        if hasattr(df.columns, 'get_level_values'):
            df.columns = df.columns.get_level_values(0)

        # Calculate indicators
        df = self.indicators.calculate_all(df)

        # Cache
        self._data_cache[cache_key] = {
            'timestamp': now,
            'data': df.copy()
        }

        return df

    def generate_trigger(self, symbol: str) -> TriggerSignal:
        """
        Generate a trading trigger for a symbol.

        Returns a TriggerSignal with:
        - action: BUY, SELL, or HOLD
        - trigger_type: BREAKOUT, REVERSION, RSI_EXTREME, MODEL_AGREE
        - confidence: 0.0 to 1.0
        - entry_price, stop_loss, take_profit levels
        """
        # Get data with indicators
        df = self._get_data(symbol)

        # Check if we have enough raw data (before dropna)
        if len(df) < 30:
            return self._no_signal(symbol, df)

        # Only need the last few rows to have valid indicators
        # Don't dropna the whole dataframe - just check the last row
        required_cols = ['RSI', 'MACD_Hist', 'BB_Pct', 'ATR', 'Close']
        latest = df.iloc[-1]
        if any(pd.isna(latest[col]) for col in required_cols):
            return self._no_signal(symbol, df)

        prev = df.iloc[-2] if len(df) > 1 else latest

        close = float(latest['Close'])
        rsi = float(latest['RSI'])
        macd_hist = float(latest['MACD_Hist'])
        macd_hist_prev = float(prev['MACD_Hist'])
        bb_pct = float(latest['BB_Pct'])
        atr = float(latest['ATR'])

        # Get ML predictions
        lgb_pred = self._get_ml_prediction('lgb', symbol)
        cnn_pred = self._get_ml_prediction('cnn', symbol, df)

        # Determine trigger
        trigger = self._evaluate_trigger(
            symbol=symbol,
            close=close,
            rsi=rsi,
            macd_hist=macd_hist,
            macd_hist_prev=macd_hist_prev,
            bb_pct=bb_pct,
            atr=atr,
            lgb_pred=lgb_pred,
            cnn_pred=cnn_pred
        )

        return trigger

    def _get_ml_prediction(self, model_type: str, symbol: str, data: pd.DataFrame = None) -> Dict:
        """Get prediction from ML model"""
        default = {'action': 'HOLD', 'confidence': 0.0, 'signal': 'NEUTRAL'}

        try:
            if model_type == 'lgb':
                predictor = self._get_lgb_predictor()
                if predictor:
                    pred = predictor.predict(symbol)
                    return {
                        'action': pred.get('action', 'HOLD'),
                        'confidence': pred.get('confidence', 0),
                        'signal': pred.get('signal', 'NEUTRAL')
                    }
            elif model_type == 'cnn':
                predictor = self._get_cnn_predictor()
                if predictor and data is not None:
                    pred = predictor.predict(symbol, data)
                    return {
                        'action': pred.get('action', 'HOLD'),
                        'confidence': pred.get('confidence', 0),
                        'signal': pred.get('signal', 'NEUTRAL')
                    }
        except Exception as e:
            logger.debug(f"ML prediction error ({model_type}): {e}")

        return default

    def _evaluate_trigger(
        self,
        symbol: str,
        close: float,
        rsi: float,
        macd_hist: float,
        macd_hist_prev: float,
        bb_pct: float,
        atr: float,
        lgb_pred: Dict,
        cnn_pred: Dict
    ) -> TriggerSignal:
        """Evaluate conditions and return trigger signal"""

        action = "HOLD"
        trigger_type = "NONE"
        confidence = 0.0

        # Check for MACD crossover (histogram sign change)
        macd_bullish_cross = macd_hist_prev < 0 and macd_hist > 0
        macd_bearish_cross = macd_hist_prev > 0 and macd_hist < 0
        macd_strength = min(1.0, abs(macd_hist) * 50)  # Normalize strength

        # ML consensus
        ml_bullish = sum([1 for p in [lgb_pred, cnn_pred] if p['action'] == 'BUY'])
        ml_bearish = sum([1 for p in [lgb_pred, cnn_pred] if p['action'] == 'SELL'])
        ml_conf_avg = (lgb_pred['confidence'] + cnn_pred['confidence']) / 2

        # === TRIGGER CONDITIONS ===

        # 1. BREAKOUT BUY: Strong MACD bullish crossover + RSI momentum zone (40-65)
        if macd_bullish_cross and macd_strength >= 0.5 and 40 < rsi < 65:
            action = "BUY"
            trigger_type = "BREAKOUT"
            ml_boost = 0.15 if ml_bullish >= 1 else 0
            confidence = min(1.0, macd_strength + ml_boost)

        # 2. BREAKOUT SELL: Strong MACD bearish crossover + RSI momentum zone
        elif macd_bearish_cross and macd_strength >= 0.5 and 35 < rsi < 60:
            action = "SELL"
            trigger_type = "BREAKOUT"
            ml_boost = 0.15 if ml_bearish >= 1 else 0
            confidence = min(1.0, macd_strength + ml_boost)

        # 3. REVERSION BUY: RSI oversold + BB at lower band
        elif rsi < 30 and bb_pct < 0.1:
            action = "BUY"
            trigger_type = "REVERSION"
            ml_boost = 0.2 if ml_bullish >= 1 else 0
            confidence = min(1.0, (30 - rsi) / 30 + ml_boost)

        # 4. REVERSION SELL: RSI overbought + BB at upper band
        elif rsi > 70 and bb_pct > 0.9:
            action = "SELL"
            trigger_type = "REVERSION"
            ml_boost = 0.2 if ml_bearish >= 1 else 0
            confidence = min(1.0, (rsi - 70) / 30 + ml_boost)

        # 5. RSI EXTREME: Very strong RSI signal alone
        elif rsi < 25:
            action = "BUY"
            trigger_type = "RSI_EXTREME"
            confidence = (25 - rsi) / 25

        elif rsi > 75:
            action = "SELL"
            trigger_type = "RSI_EXTREME"
            confidence = (rsi - 75) / 25

        # 6. ML CONSENSUS: Both models agree with high confidence
        elif ml_bullish == 2 and ml_conf_avg > 0.6:
            action = "BUY"
            trigger_type = "MODEL_AGREE"
            confidence = ml_conf_avg

        elif ml_bearish == 2 and ml_conf_avg > 0.6:
            action = "SELL"
            trigger_type = "MODEL_AGREE"
            confidence = ml_conf_avg

        # Calculate stop loss and take profit
        if action == "BUY":
            stop_loss = close - (2.0 * atr)
            take_profit = close + (3.0 * atr)
        elif action == "SELL":
            stop_loss = close + (2.0 * atr)
            take_profit = close - (3.0 * atr)
        else:
            stop_loss = close
            take_profit = close

        return TriggerSignal(
            symbol=symbol,
            action=action,
            trigger_type=trigger_type,
            confidence=confidence,
            entry_price=close,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            rsi_value=round(rsi, 2),
            macd_value=round(macd_hist, 4),
            bb_position=round(bb_pct, 2),
            ml_signal=lgb_pred['signal'],
            ml_confidence=lgb_pred['confidence'],
            cnn_signal=cnn_pred.get('signal'),
            cnn_confidence=cnn_pred.get('confidence'),
            timestamp=datetime.now().isoformat()
        )

    def _no_signal(self, symbol: str, df: pd.DataFrame) -> TriggerSignal:
        """Return a HOLD signal when no trigger is detected"""
        close = float(df['Close'].iloc[-1]) if len(df) > 0 else 0
        return TriggerSignal(
            symbol=symbol,
            action="HOLD",
            trigger_type="NONE",
            confidence=0.0,
            entry_price=close,
            stop_loss=close,
            take_profit=close,
            rsi_value=50.0,
            macd_value=0.0,
            bb_position=0.5,
            ml_signal="NEUTRAL",
            ml_confidence=0.0,
            timestamp=datetime.now().isoformat()
        )

    def scan_for_triggers(self, symbols: List[str]) -> List[TriggerSignal]:
        """
        Scan multiple symbols for triggers.
        Returns only actionable signals (BUY or SELL).
        """
        triggers = []

        for symbol in symbols:
            try:
                trigger = self.generate_trigger(symbol)
                if trigger.action != "HOLD":
                    triggers.append(trigger)
                    logger.info(f"Trigger: {trigger.action} {symbol} ({trigger.trigger_type}) "
                               f"@ ${trigger.entry_price:.2f} conf={trigger.confidence:.2f}")
            except Exception as e:
                logger.warning(f"Error scanning {symbol}: {e}")

        # Sort by confidence
        triggers.sort(key=lambda x: x.confidence, reverse=True)

        return triggers


# Singleton instance
_trigger_generator: Optional[TriggerSignalGenerator] = None


def get_trigger_generator() -> TriggerSignalGenerator:
    """Get or create the trigger generator singleton"""
    global _trigger_generator
    if _trigger_generator is None:
        _trigger_generator = TriggerSignalGenerator()
    return _trigger_generator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("TRIGGER SIGNAL GENERATOR TEST")
    print("=" * 60)

    generator = TriggerSignalGenerator()

    # Test on some stocks
    test_symbols = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "PLTR", "RGTI"]

    print("\nScanning for triggers...")

    for symbol in test_symbols:
        try:
            trigger = generator.generate_trigger(symbol)
            status = "SIGNAL" if trigger.action != "HOLD" else "no signal"
            print(f"\n{symbol}: {status}")
            print(f"  Action: {trigger.action} ({trigger.trigger_type})")
            print(f"  Price: ${trigger.entry_price:.2f}")
            print(f"  RSI: {trigger.rsi_value:.1f}, MACD: {trigger.macd_value:.4f}, BB%: {trigger.bb_position:.2f}")
            print(f"  ML: {trigger.ml_signal} (conf: {trigger.ml_confidence:.2f})")
            if trigger.cnn_signal:
                print(f"  CNN: {trigger.cnn_signal} (conf: {trigger.cnn_confidence:.2f})")
            if trigger.action != "HOLD":
                print(f"  Entry: ${trigger.entry_price:.2f}")
                print(f"  Stop Loss: ${trigger.stop_loss:.2f}")
                print(f"  Take Profit: ${trigger.take_profit:.2f}")
                print(f"  Confidence: {trigger.confidence:.2f}")
        except Exception as e:
            print(f"\n{symbol}: Error - {e}")

    print("\n" + "=" * 60)
