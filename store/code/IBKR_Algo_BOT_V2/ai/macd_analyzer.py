"""
MACD Analyzer for Two-Phase Strategy
=====================================
Real-time MACD analysis for Phase 2 continuation entries.

Provides:
- MACD crossover detection (bullish/bearish)
- MACD divergence detection
- Signal strength analysis
- Phase 2 entry confirmation

Integration with Two-Phase Strategy:
- Phase 2 Entry: Wait for MACD bullish crossover after consolidation
- Exit Indicator: MACD bearish divergence or histogram weakening
"""

import pandas as pd
import numpy as np
import ta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class MACDSignal:
    """MACD signal data"""
    symbol: str
    signal_type: str  # bullish, bearish, neutral
    crossover: bool  # True if crossover just occurred
    crossover_type: str  # bullish_cross, bearish_cross, none
    histogram: float  # Current histogram value
    histogram_trend: str  # expanding, contracting, flat
    macd_line: float
    signal_line: float
    divergence: Optional[str]  # bullish_div, bearish_div, none
    strength: float  # 0-1 signal strength
    momentum_score: float  # Combined momentum indicator
    phase2_ready: bool  # True if conditions good for Phase 2 entry
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)


class MACDAnalyzer:
    """
    Real-time MACD analysis for trading signals.

    Optimized for scalping/momentum trading with:
    - Fast detection of crossovers
    - Divergence identification
    - Signal strength scoring
    """

    def __init__(self,
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self._data_cache: Dict[str, Dict] = {}
        self._cache_ttl = 60  # 1 minute cache for real-time

    def _get_price_data(self, symbol: str, period: str = "5d",
                        interval: str = "1m") -> Optional[pd.DataFrame]:
        """Fetch price data for MACD calculation"""
        cache_key = f"{symbol}_{period}_{interval}"
        now = datetime.now()

        # Check cache
        if cache_key in self._data_cache:
            entry = self._data_cache[cache_key]
            if (now - entry['timestamp']).total_seconds() < self._cache_ttl:
                return entry['data'].copy()

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty or len(df) < self.slow_period + 10:
                logger.warning(f"Insufficient data for {symbol}")
                return None

            # Cache result
            self._data_cache[cache_key] = {
                'data': df,
                'timestamp': now
            }

            return df.copy()

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators"""
        data = df.copy()

        # Handle multi-index columns
        if hasattr(data.columns, 'get_level_values'):
            data.columns = data.columns.get_level_values(0)

        close = data['Close']
        if hasattr(close, 'values') and len(close.values.shape) > 1:
            close = pd.Series(close.values.flatten(), index=data.index)

        # Calculate MACD
        macd = ta.trend.MACD(
            close,
            window_slow=self.slow_period,
            window_fast=self.fast_period,
            window_sign=self.signal_period
        )

        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Hist'] = macd.macd_diff()

        # Calculate histogram change
        data['MACD_Hist_Change'] = data['MACD_Hist'].diff()

        # Crossover signals
        data['MACD_Cross'] = np.where(
            (data['MACD'] > data['MACD_Signal']) &
            (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1)),
            1,  # Bullish crossover
            np.where(
                (data['MACD'] < data['MACD_Signal']) &
                (data['MACD'].shift(1) >= data['MACD_Signal'].shift(1)),
                -1,  # Bearish crossover
                0
            )
        )

        return data

    def detect_divergence(self, df: pd.DataFrame, lookback: int = 20) -> Optional[str]:
        """
        Detect bullish or bearish divergence.

        Bullish divergence: Price makes lower low, MACD makes higher low
        Bearish divergence: Price makes higher high, MACD makes lower high
        """
        if len(df) < lookback:
            return None

        recent = df.tail(lookback)
        close = recent['Close'].values
        macd_hist = recent['MACD_Hist'].values

        # Find local extremes
        price_lows = []
        price_highs = []
        macd_lows = []
        macd_highs = []

        for i in range(2, len(close) - 2):
            # Local low
            if close[i] < close[i-1] and close[i] < close[i-2] and \
               close[i] < close[i+1] and close[i] < close[i+2]:
                price_lows.append((i, close[i]))
                macd_lows.append((i, macd_hist[i]))
            # Local high
            if close[i] > close[i-1] and close[i] > close[i-2] and \
               close[i] > close[i+1] and close[i] > close[i+2]:
                price_highs.append((i, close[i]))
                macd_highs.append((i, macd_hist[i]))

        # Check for bullish divergence (price lower low, MACD higher low)
        if len(price_lows) >= 2 and len(macd_lows) >= 2:
            if price_lows[-1][1] < price_lows[-2][1] and \
               macd_lows[-1][1] > macd_lows[-2][1]:
                return "bullish_div"

        # Check for bearish divergence (price higher high, MACD lower high)
        if len(price_highs) >= 2 and len(macd_highs) >= 2:
            if price_highs[-1][1] > price_highs[-2][1] and \
               macd_highs[-1][1] < macd_highs[-2][1]:
                return "bearish_div"

        return None

    def get_histogram_trend(self, df: pd.DataFrame, bars: int = 5) -> str:
        """Determine if histogram is expanding or contracting"""
        if len(df) < bars:
            return "flat"

        recent_hist = df['MACD_Hist'].tail(bars).values

        # Check if absolute values are increasing (expanding)
        abs_hist = np.abs(recent_hist)
        if all(abs_hist[i] < abs_hist[i+1] for i in range(len(abs_hist)-1)):
            return "expanding"

        # Check if absolute values are decreasing (contracting)
        if all(abs_hist[i] > abs_hist[i+1] for i in range(len(abs_hist)-1)):
            return "contracting"

        return "flat"

    def calculate_signal_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate overall MACD signal strength (0-1).

        Factors:
        - Histogram magnitude (vs recent average)
        - Histogram momentum (expanding vs contracting)
        - Distance from zero line
        - Crossover recency
        """
        if len(df) < 26:
            return 0.0

        latest = df.iloc[-1]
        hist = latest['MACD_Hist']
        macd_line = latest['MACD']

        # Average histogram magnitude (last 20 bars)
        avg_hist = df['MACD_Hist'].tail(20).abs().mean()
        if avg_hist == 0:
            avg_hist = 0.001

        # Histogram strength relative to average
        hist_strength = min(1.0, abs(hist) / (avg_hist * 2))

        # Histogram momentum (expanding = stronger)
        hist_trend = self.get_histogram_trend(df)
        trend_bonus = 0.2 if hist_trend == "expanding" else (-0.1 if hist_trend == "contracting" else 0)

        # Recent crossover bonus
        recent_cross = df['MACD_Cross'].tail(5).abs().sum()
        cross_bonus = 0.2 if recent_cross > 0 else 0

        # Combine factors
        strength = hist_strength + trend_bonus + cross_bonus
        return min(1.0, max(0.0, strength))

    def analyze(self, symbol: str) -> Optional[MACDSignal]:
        """
        Perform full MACD analysis for a symbol.

        Returns MACDSignal with all indicators for Phase 2 decision.
        """
        # Get price data
        df = self._get_price_data(symbol)
        if df is None:
            return None

        # Calculate MACD
        df = self.calculate_macd(df)

        if df['MACD'].isna().all():
            logger.warning(f"MACD calculation failed for {symbol}")
            return None

        latest = df.iloc[-1]
        macd_line = float(latest['MACD'])
        signal_line = float(latest['MACD_Signal'])
        histogram = float(latest['MACD_Hist'])
        crossover = int(latest['MACD_Cross'])

        # Determine signal type
        if macd_line > signal_line:
            signal_type = "bullish"
        elif macd_line < signal_line:
            signal_type = "bearish"
        else:
            signal_type = "neutral"

        # Crossover detection
        crossover_occurred = crossover != 0
        crossover_type = "bullish_cross" if crossover == 1 else \
                        ("bearish_cross" if crossover == -1 else "none")

        # Divergence
        divergence = self.detect_divergence(df)

        # Histogram trend
        hist_trend = self.get_histogram_trend(df)

        # Signal strength
        strength = self.calculate_signal_strength(df)

        # Momentum score (combine MACD with histogram direction)
        momentum = 0.0
        if signal_type == "bullish" and hist_trend == "expanding":
            momentum = strength
        elif signal_type == "bearish" and hist_trend == "expanding":
            momentum = -strength
        elif signal_type == "bullish":
            momentum = strength * 0.5
        elif signal_type == "bearish":
            momentum = -strength * 0.5

        # Phase 2 readiness check
        # Ready when: Bullish signal + strength > 0.5 + (crossover OR expanding histogram)
        phase2_ready = (
            signal_type == "bullish" and
            strength >= 0.5 and
            (crossover_occurred or hist_trend == "expanding") and
            divergence != "bearish_div"
        )

        return MACDSignal(
            symbol=symbol,
            signal_type=signal_type,
            crossover=crossover_occurred,
            crossover_type=crossover_type,
            histogram=histogram,
            histogram_trend=hist_trend,
            macd_line=macd_line,
            signal_line=signal_line,
            divergence=divergence,
            strength=strength,
            momentum_score=momentum,
            phase2_ready=phase2_ready,
            timestamp=datetime.now().isoformat()
        )

    def check_phase2_entry(self, symbol: str) -> Tuple[bool, str, Dict]:
        """
        Check if conditions are right for Phase 2 entry.

        Returns:
            (should_enter, reason, details)
        """
        signal = self.analyze(symbol)

        if signal is None:
            return False, "no_data", {}

        details = {
            "macd_signal": signal.signal_type,
            "crossover": signal.crossover_type,
            "histogram": signal.histogram,
            "histogram_trend": signal.histogram_trend,
            "strength": signal.strength,
            "momentum": signal.momentum_score,
            "divergence": signal.divergence
        }

        # Entry conditions for Phase 2
        if not signal.phase2_ready:
            if signal.signal_type == "bearish":
                return False, "bearish_macd", details
            if signal.divergence == "bearish_div":
                return False, "bearish_divergence", details
            if signal.strength < 0.5:
                return False, "weak_signal", details
            return False, "waiting_for_confirmation", details

        # Bullish conditions met
        reason = []
        if signal.crossover and signal.crossover_type == "bullish_cross":
            reason.append("bullish_crossover")
        if signal.histogram_trend == "expanding":
            reason.append("momentum_expanding")
        if signal.divergence == "bullish_div":
            reason.append("bullish_divergence")

        return True, "_".join(reason) if reason else "conditions_met", details

    def check_exit_signal(self, symbol: str) -> Tuple[bool, str, Dict]:
        """
        Check if MACD indicates exit for Phase 2 position.

        Returns:
            (should_exit, reason, details)
        """
        signal = self.analyze(symbol)

        if signal is None:
            return False, "no_data", {}

        details = {
            "macd_signal": signal.signal_type,
            "crossover": signal.crossover_type,
            "histogram_trend": signal.histogram_trend,
            "strength": signal.strength,
            "divergence": signal.divergence
        }

        # Exit conditions
        if signal.crossover_type == "bearish_cross":
            return True, "bearish_crossover", details

        if signal.divergence == "bearish_div":
            return True, "bearish_divergence", details

        if signal.signal_type == "bearish" and signal.histogram_trend == "expanding":
            return True, "strong_bearish", details

        if signal.histogram_trend == "contracting" and signal.strength < 0.3:
            return True, "momentum_fading", details

        return False, "hold", details


# Singleton instance
_analyzer: Optional[MACDAnalyzer] = None


def get_macd_analyzer() -> MACDAnalyzer:
    """Get or create the MACD analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = MACDAnalyzer()
    return _analyzer


# API function for scanner routes
def analyze_macd(symbol: str) -> Dict:
    """Analyze MACD for a symbol - for API use."""
    analyzer = get_macd_analyzer()
    signal = analyzer.analyze(symbol)

    if signal is None:
        return {"error": f"Could not analyze {symbol}"}

    return signal.to_dict()


def check_phase2_conditions(symbol: str) -> Dict:
    """Check Phase 2 entry conditions - for API use."""
    analyzer = get_macd_analyzer()
    should_enter, reason, details = analyzer.check_phase2_entry(symbol)

    return {
        "symbol": symbol,
        "should_enter": should_enter,
        "reason": reason,
        "details": details,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Test the analyzer
    logging.basicConfig(level=logging.INFO)

    analyzer = get_macd_analyzer()

    # Test symbols
    test_symbols = ["AAPL", "TSLA", "NVDA"]

    for symbol in test_symbols:
        print(f"\n{'='*50}")
        print(f"Analyzing {symbol}")
        print('='*50)

        signal = analyzer.analyze(symbol)
        if signal:
            print(f"Signal Type: {signal.signal_type}")
            print(f"Crossover: {signal.crossover_type}")
            print(f"Histogram: {signal.histogram:.4f} ({signal.histogram_trend})")
            print(f"Strength: {signal.strength:.2f}")
            print(f"Momentum: {signal.momentum_score:.2f}")
            print(f"Divergence: {signal.divergence or 'None'}")
            print(f"Phase 2 Ready: {signal.phase2_ready}")

            # Check Phase 2 conditions
            should_enter, reason, details = analyzer.check_phase2_entry(symbol)
            print(f"\nPhase 2 Entry: {should_enter} ({reason})")
