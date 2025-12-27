"""
Multi-Timeframe Confirmation Module
====================================
Ross Cameron methodology: 1-minute and 5-minute charts must align before entry.

Entry Criteria:
- Both timeframes show bullish structure (higher highs, higher lows)
- Both MACD above signal line
- Both EMAs aligned (9 > 20)
- Price above VWAP on both timeframes
- Volume confirming the move

This prevents entering on false breakouts where the higher timeframe
doesn't support the move.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class TimeframeTrend(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


class MTFSignal(Enum):
    CONFIRMED_LONG = "CONFIRMED_LONG"
    WEAK_LONG = "WEAK_LONG"
    NO_CONFIRMATION = "NO_CONFIRMATION"
    WEAK_SHORT = "WEAK_SHORT"
    CONFIRMED_SHORT = "CONFIRMED_SHORT"


@dataclass
class TimeframeAnalysis:
    """Analysis for a single timeframe"""
    timeframe: str  # "1m" or "5m"
    trend: TimeframeTrend = TimeframeTrend.NEUTRAL

    # Price structure
    higher_high: bool = False
    higher_low: bool = False
    lower_high: bool = False
    lower_low: bool = False

    # MACD
    macd_bullish: bool = False  # MACD > Signal
    macd_histogram_positive: bool = False
    macd_crossover: str = "NONE"  # BULLISH, BEARISH, NONE

    # EMAs
    ema_bullish: bool = False  # EMA9 > EMA20
    ema_9: float = 0.0
    ema_20: float = 0.0
    price_above_ema9: bool = False
    price_above_ema20: bool = False

    # VWAP
    above_vwap: bool = False
    vwap: float = 0.0
    vwap_distance_pct: float = 0.0

    # Volume
    volume_increasing: bool = False
    relative_volume: float = 0.0

    # RSI
    rsi: float = 50.0
    rsi_bullish: bool = False  # RSI > 50
    rsi_overbought: bool = False  # RSI > 70
    rsi_oversold: bool = False  # RSI < 30

    # Candle analysis
    last_candle_bullish: bool = False
    candle_body_pct: float = 0.0  # Body as % of range

    # Score
    bullish_score: int = 0  # 0-10

    def calculate_score(self):
        """Calculate bullish score out of 10"""
        score = 0
        if self.higher_high and self.higher_low:
            score += 2
        if self.macd_bullish:
            score += 2
        if self.ema_bullish:
            score += 1
        if self.price_above_ema9:
            score += 1
        if self.above_vwap:
            score += 2
        if self.volume_increasing:
            score += 1
        if self.rsi_bullish and not self.rsi_overbought:
            score += 1
        self.bullish_score = min(score, 10)
        return self.bullish_score


@dataclass
class MTFConfirmation:
    """Multi-timeframe confirmation result"""
    symbol: str
    timestamp: str = ""

    # Individual timeframe analysis
    tf_1m: Optional[TimeframeAnalysis] = None
    tf_5m: Optional[TimeframeAnalysis] = None

    # Alignment
    trend_aligned: bool = False
    macd_aligned: bool = False
    ema_aligned: bool = False
    vwap_aligned: bool = False

    # Overall signal
    signal: MTFSignal = MTFSignal.NO_CONFIRMATION
    confidence: float = 0.0  # 0-100

    # Recommendation
    recommendation: str = "WAIT"  # ENTER, WAIT, AVOID
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "signal": self.signal.value,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
            "reasons": self.reasons,
            "alignment": {
                "trend": self.trend_aligned,
                "macd": self.macd_aligned,
                "ema": self.ema_aligned,
                "vwap": self.vwap_aligned
            },
            "tf_1m": {
                "trend": self.tf_1m.trend.value if self.tf_1m else "UNKNOWN",
                "score": self.tf_1m.bullish_score if self.tf_1m else 0,
                "macd_bullish": self.tf_1m.macd_bullish if self.tf_1m else False,
                "above_vwap": self.tf_1m.above_vwap if self.tf_1m else False
            } if self.tf_1m else None,
            "tf_5m": {
                "trend": self.tf_5m.trend.value if self.tf_5m else "UNKNOWN",
                "score": self.tf_5m.bullish_score if self.tf_5m else 0,
                "macd_bullish": self.tf_5m.macd_bullish if self.tf_5m else False,
                "above_vwap": self.tf_5m.above_vwap if self.tf_5m else False
            } if self.tf_5m else None
        }


class MTFConfirmationEngine:
    """
    Multi-Timeframe Confirmation Engine

    Analyzes 1-minute and 5-minute charts to confirm entry signals.
    Ross Cameron rule: Only enter when both timeframes agree.
    """

    def __init__(self):
        self.cache: Dict[str, MTFConfirmation] = {}
        self.cache_ttl = 30  # seconds
        self.last_update: Dict[str, datetime] = {}

        # Configuration
        self.min_confidence_to_enter = 70.0  # Minimum confidence %
        self.require_both_above_vwap = True
        self.require_macd_alignment = True

    def analyze(self, symbol: str, candles_1m: List[Dict] = None,
                candles_5m: List[Dict] = None) -> MTFConfirmation:
        """
        Analyze multi-timeframe confirmation for a symbol.

        Args:
            symbol: Stock symbol
            candles_1m: 1-minute candle data (optional, will fetch if not provided)
            candles_5m: 5-minute candle data (optional, will fetch if not provided)

        Returns:
            MTFConfirmation with signal and recommendation
        """
        result = MTFConfirmation(
            symbol=symbol,
            timestamp=datetime.now().isoformat()
        )

        try:
            # Fetch candle data if not provided
            if candles_1m is None:
                candles_1m = self._fetch_candles(symbol, "1m")
            if candles_5m is None:
                candles_5m = self._fetch_candles(symbol, "5m")

            if not candles_1m or len(candles_1m) < 20:
                result.reasons.append("Insufficient 1M data")
                return result
            if not candles_5m or len(candles_5m) < 20:
                result.reasons.append("Insufficient 5M data")
                return result

            # Analyze each timeframe
            result.tf_1m = self._analyze_timeframe(candles_1m, "1m")
            result.tf_5m = self._analyze_timeframe(candles_5m, "5m")

            # Check alignment
            result.trend_aligned = self._check_trend_alignment(result.tf_1m, result.tf_5m)
            result.macd_aligned = result.tf_1m.macd_bullish == result.tf_5m.macd_bullish
            result.ema_aligned = result.tf_1m.ema_bullish == result.tf_5m.ema_bullish
            result.vwap_aligned = result.tf_1m.above_vwap == result.tf_5m.above_vwap

            # Calculate confidence and signal
            result = self._calculate_signal(result)

            # Cache result
            self.cache[symbol] = result
            self.last_update[symbol] = datetime.now()

        except Exception as e:
            logger.error(f"MTF analysis failed for {symbol}: {e}")
            result.reasons.append(f"Analysis error: {str(e)}")

        return result

    def _fetch_candles(self, symbol: str, timeframe: str) -> List[Dict]:
        """Fetch candle data from market data provider"""
        try:
            # Try Polygon first (better data)
            from polygon_data import get_polygon_client
            client = get_polygon_client()

            if timeframe == "1m":
                multiplier, span = 1, "minute"
                days = 1
            else:
                multiplier, span = 5, "minute"
                days = 2

            from datetime import datetime, timedelta
            end = datetime.now()
            start = end - timedelta(days=days)

            bars = client.get_bars(
                symbol,
                multiplier=multiplier,
                timespan=span,
                from_date=start.strftime("%Y-%m-%d"),
                to_date=end.strftime("%Y-%m-%d")
            )

            if bars:
                return [
                    {
                        "open": b.get("o", b.get("open", 0)),
                        "high": b.get("h", b.get("high", 0)),
                        "low": b.get("l", b.get("low", 0)),
                        "close": b.get("c", b.get("close", 0)),
                        "volume": b.get("v", b.get("volume", 0)),
                        "timestamp": b.get("t", b.get("timestamp", 0))
                    }
                    for b in bars
                ]
        except Exception as e:
            logger.warning(f"Polygon fetch failed for {symbol} {timeframe}: {e}")

        # Fallback to Schwab
        try:
            from schwab_market_data import get_schwab_market_data
            schwab = get_schwab_market_data()

            period = "1d" if timeframe == "1m" else "5d"
            freq = "1" if timeframe == "1m" else "5"

            data = schwab.get_price_history(
                symbol,
                period_type="day",
                period=int(period[0]),
                frequency_type="minute",
                frequency=int(freq)
            )

            if data and "candles" in data:
                return data["candles"]
        except Exception as e:
            logger.warning(f"Schwab fetch failed for {symbol} {timeframe}: {e}")

        return []

    def _analyze_timeframe(self, candles: List[Dict], timeframe: str) -> TimeframeAnalysis:
        """Analyze a single timeframe"""
        analysis = TimeframeAnalysis(timeframe=timeframe)

        if len(candles) < 20:
            return analysis

        # Extract price arrays
        closes = np.array([c.get("close", 0) for c in candles])
        highs = np.array([c.get("high", 0) for c in candles])
        lows = np.array([c.get("low", 0) for c in candles])
        volumes = np.array([c.get("volume", 0) for c in candles])

        current_price = closes[-1]

        # Price structure (last 5 candles)
        if len(highs) >= 5:
            recent_highs = highs[-5:]
            recent_lows = lows[-5:]
            analysis.higher_high = recent_highs[-1] > np.max(recent_highs[:-1])
            analysis.higher_low = recent_lows[-1] > np.min(recent_lows[:-1])
            analysis.lower_high = recent_highs[-1] < np.max(recent_highs[:-1])
            analysis.lower_low = recent_lows[-1] < np.min(recent_lows[:-1])

        # EMAs
        analysis.ema_9 = self._calculate_ema(closes, 9)
        analysis.ema_20 = self._calculate_ema(closes, 20)
        analysis.ema_bullish = analysis.ema_9 > analysis.ema_20
        analysis.price_above_ema9 = current_price > analysis.ema_9
        analysis.price_above_ema20 = current_price > analysis.ema_20

        # MACD
        macd, signal, histogram = self._calculate_macd(closes)
        analysis.macd_bullish = macd > signal
        analysis.macd_histogram_positive = histogram > 0

        # Check for crossover in last 3 candles
        if len(closes) >= 15:
            macd_prev, signal_prev, _ = self._calculate_macd(closes[:-1])
            if macd > signal and macd_prev <= signal_prev:
                analysis.macd_crossover = "BULLISH"
            elif macd < signal and macd_prev >= signal_prev:
                analysis.macd_crossover = "BEARISH"

        # VWAP (simplified - calculate from available data)
        analysis.vwap = self._calculate_vwap(candles)
        analysis.above_vwap = current_price > analysis.vwap
        if analysis.vwap > 0:
            analysis.vwap_distance_pct = ((current_price - analysis.vwap) / analysis.vwap) * 100

        # Volume
        if len(volumes) >= 20:
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1]
            if avg_volume > 0:
                analysis.relative_volume = current_volume / avg_volume
            analysis.volume_increasing = current_volume > np.mean(volumes[-5:])

        # RSI
        analysis.rsi = self._calculate_rsi(closes, 14)
        analysis.rsi_bullish = analysis.rsi > 50
        analysis.rsi_overbought = analysis.rsi > 70
        analysis.rsi_oversold = analysis.rsi < 30

        # Last candle analysis
        last_candle = candles[-1]
        candle_open = last_candle.get("open", 0)
        candle_close = last_candle.get("close", 0)
        candle_high = last_candle.get("high", 0)
        candle_low = last_candle.get("low", 0)

        analysis.last_candle_bullish = candle_close > candle_open
        candle_range = candle_high - candle_low
        if candle_range > 0:
            analysis.candle_body_pct = abs(candle_close - candle_open) / candle_range * 100

        # Determine trend
        bullish_count = sum([
            analysis.higher_high,
            analysis.higher_low,
            analysis.macd_bullish,
            analysis.ema_bullish,
            analysis.above_vwap,
            analysis.rsi_bullish
        ])

        if bullish_count >= 5:
            analysis.trend = TimeframeTrend.STRONG_BULLISH
        elif bullish_count >= 4:
            analysis.trend = TimeframeTrend.BULLISH
        elif bullish_count <= 1:
            analysis.trend = TimeframeTrend.STRONG_BEARISH
        elif bullish_count <= 2:
            analysis.trend = TimeframeTrend.BEARISH
        else:
            analysis.trend = TimeframeTrend.NEUTRAL

        # Calculate score
        analysis.calculate_score()

        return analysis

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0

        multiplier = 2 / (period + 1)
        ema = prices[:period].mean()

        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """Calculate MACD, Signal, and Histogram"""
        if len(prices) < 26:
            return 0, 0, 0

        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        macd = ema_12 - ema_26

        # For signal line, we need MACD history
        macd_history = []
        for i in range(26, len(prices)):
            subset = prices[:i+1]
            e12 = self._calculate_ema(subset, 12)
            e26 = self._calculate_ema(subset, 26)
            macd_history.append(e12 - e26)

        if len(macd_history) >= 9:
            signal = self._calculate_ema(np.array(macd_history), 9)
        else:
            signal = macd

        histogram = macd - signal

        return macd, signal, histogram

    def _calculate_vwap(self, candles: List[Dict]) -> float:
        """Calculate Volume Weighted Average Price"""
        total_volume = 0
        total_vwap = 0

        for c in candles:
            typical_price = (c.get("high", 0) + c.get("low", 0) + c.get("close", 0)) / 3
            volume = c.get("volume", 0)
            total_vwap += typical_price * volume
            total_volume += volume

        return total_vwap / total_volume if total_volume > 0 else 0

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _check_trend_alignment(self, tf1: TimeframeAnalysis, tf2: TimeframeAnalysis) -> bool:
        """Check if both timeframes have aligned trends"""
        bullish_trends = [TimeframeTrend.BULLISH, TimeframeTrend.STRONG_BULLISH]
        bearish_trends = [TimeframeTrend.BEARISH, TimeframeTrend.STRONG_BEARISH]

        both_bullish = tf1.trend in bullish_trends and tf2.trend in bullish_trends
        both_bearish = tf1.trend in bearish_trends and tf2.trend in bearish_trends

        return both_bullish or both_bearish

    def _calculate_signal(self, result: MTFConfirmation) -> MTFConfirmation:
        """Calculate overall signal and confidence"""
        tf1 = result.tf_1m
        tf2 = result.tf_5m

        if not tf1 or not tf2:
            result.signal = MTFSignal.NO_CONFIRMATION
            result.confidence = 0
            result.recommendation = "WAIT"
            return result

        # Calculate alignment score
        alignment_score = sum([
            result.trend_aligned * 30,
            result.macd_aligned * 25,
            result.ema_aligned * 20,
            result.vwap_aligned * 25
        ])

        # Add individual timeframe scores
        tf_score = (tf1.bullish_score + tf2.bullish_score) * 5  # Max 100

        # Combined confidence
        result.confidence = min((alignment_score + tf_score) / 2, 100)

        # Determine signal
        bullish_trends = [TimeframeTrend.BULLISH, TimeframeTrend.STRONG_BULLISH]
        bearish_trends = [TimeframeTrend.BEARISH, TimeframeTrend.STRONG_BEARISH]

        if result.trend_aligned:
            if tf1.trend in bullish_trends and tf2.trend in bullish_trends:
                if result.macd_aligned and tf1.macd_bullish:
                    if result.vwap_aligned and tf1.above_vwap:
                        result.signal = MTFSignal.CONFIRMED_LONG
                        result.recommendation = "ENTER"
                        result.reasons.append("Both timeframes bullish with MACD and VWAP confirmation")
                    else:
                        result.signal = MTFSignal.WEAK_LONG
                        result.recommendation = "WAIT"
                        result.reasons.append("Bullish but missing VWAP confirmation")
                else:
                    result.signal = MTFSignal.WEAK_LONG
                    result.recommendation = "WAIT"
                    result.reasons.append("Trend aligned but MACD not confirmed")
            elif tf1.trend in bearish_trends and tf2.trend in bearish_trends:
                result.signal = MTFSignal.CONFIRMED_SHORT
                result.recommendation = "AVOID"  # We don't short
                result.reasons.append("Both timeframes bearish - avoid longs")
        else:
            # Trends not aligned
            if tf2.trend in bearish_trends:
                result.signal = MTFSignal.NO_CONFIRMATION
                result.recommendation = "AVOID"
                result.reasons.append("5M timeframe bearish - no confirmation")
            else:
                result.signal = MTFSignal.NO_CONFIRMATION
                result.recommendation = "WAIT"
                result.reasons.append("Timeframes not aligned")

        # Additional checks
        if tf1.rsi_overbought or tf2.rsi_overbought:
            result.reasons.append("Warning: RSI overbought")
            if result.recommendation == "ENTER":
                result.recommendation = "WAIT"
                result.confidence *= 0.8

        return result

    def get_cached(self, symbol: str) -> Optional[MTFConfirmation]:
        """Get cached result if still valid"""
        if symbol in self.cache and symbol in self.last_update:
            age = (datetime.now() - self.last_update[symbol]).total_seconds()
            if age < self.cache_ttl:
                return self.cache[symbol]
        return None

    def is_confirmed_long(self, symbol: str) -> bool:
        """Quick check if symbol has confirmed long signal"""
        cached = self.get_cached(symbol)
        if cached:
            return cached.signal == MTFSignal.CONFIRMED_LONG

        result = self.analyze(symbol)
        return result.signal == MTFSignal.CONFIRMED_LONG


# Singleton instance
_mtf_engine: Optional[MTFConfirmationEngine] = None


def get_mtf_engine() -> MTFConfirmationEngine:
    """Get singleton MTF confirmation engine"""
    global _mtf_engine
    if _mtf_engine is None:
        _mtf_engine = MTFConfirmationEngine()
    return _mtf_engine


# Convenience function
def check_mtf_confirmation(symbol: str) -> MTFConfirmation:
    """Check multi-timeframe confirmation for a symbol"""
    return get_mtf_engine().analyze(symbol)
