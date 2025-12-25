"""
Technical Signal Analyzer
=========================
Calculates trading signals based on:
- EMA 9/20/200 crossovers
- MACD crossovers
- VWAP position
- Volume/candle momentum
- Multi-timeframe alignment

Provides confluence scoring for entry/exit decisions.
"""

import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SignalState:
    """Current state of all technical signals"""
    symbol: str
    timeframe: str
    timestamp: int

    # EMA signals
    ema9: float
    ema20: float
    ema200: float
    ema_9_above_20: bool
    ema_bullish: bool  # 9 > 20
    ema_crossover: str  # "BULLISH", "BEARISH", or "NONE"
    price_above_200: bool

    # MACD signals
    macd_line: float
    macd_signal: float
    macd_histogram: float
    macd_bullish: bool  # MACD > Signal
    macd_crossover: str  # "BULLISH", "BEARISH", or "NONE"
    macd_histogram_growing: bool

    # VWAP signals
    vwap: float
    price_above_vwap: bool
    vwap_crossover: str  # "BULLISH", "BEARISH", or "NONE"

    # Volume/Candle momentum
    volume_increasing: bool
    green_candle_dominance: float  # 0-1, ratio of green vs red
    candle_momentum: str  # "BUILDING", "FADING", "NEUTRAL"

    # Confluence
    bullish_signals: int  # Count of bullish signals
    bearish_signals: int  # Count of bearish signals
    confluence_score: float  # 0-100
    signal_bias: str  # "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"

    def to_dict(self):
        return asdict(self)


@dataclass
class SignalEvent:
    """A signal event for charting"""
    time: int
    event_type: str  # "EMA_CROSS", "MACD_CROSS", "VWAP_CROSS", "MOMENTUM_SHIFT"
    direction: str  # "BULLISH" or "BEARISH"
    description: str
    strength: float  # 0-1


class TechnicalSignalAnalyzer:
    """
    Analyzes price data and generates technical signals.
    """

    def __init__(self):
        self.signal_cache = {}  # symbol -> SignalState
        self.event_cache = {}   # symbol -> List[SignalEvent]

    def analyze(self, symbol: str, ohlc_data: List[Dict], timeframe: str = "5m") -> SignalState:
        """
        Analyze OHLC data and return current signal state.

        Args:
            symbol: Stock symbol
            ohlc_data: List of {time, open, high, low, close, volume} dicts
            timeframe: Timeframe string (1m, 5m, etc.)

        Returns:
            SignalState with all calculated signals
        """
        if not ohlc_data or len(ohlc_data) < 50:
            logger.warning(f"Insufficient data for {symbol}: {len(ohlc_data) if ohlc_data else 0} bars")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(ohlc_data)
        df = df.sort_values('time').reset_index(drop=True)

        # Calculate EMAs
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean() if len(df) >= 200 else df['close'].ewm(span=len(df), adjust=False).mean()

        # Calculate MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_line'] = ema12 - ema26
        df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']

        # Calculate VWAP
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_volume'] = df['typical_price'] * df['volume']
        df['cum_tp_vol'] = df['tp_volume'].cumsum()
        df['cum_vol'] = df['volume'].cumsum()
        df['vwap'] = df['cum_tp_vol'] / df['cum_vol']

        # Get current and previous values
        current = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else current
        prev2 = df.iloc[-3] if len(df) > 2 else prev

        # EMA analysis
        ema9 = current['ema9']
        ema20 = current['ema20']
        ema200 = current['ema200']
        ema_9_above_20 = ema9 > ema20
        ema_bullish = ema_9_above_20

        # EMA crossover detection
        prev_ema9_above = prev['ema9'] > prev['ema20']
        if ema_9_above_20 and not prev_ema9_above:
            ema_crossover = "BULLISH"
        elif not ema_9_above_20 and prev_ema9_above:
            ema_crossover = "BEARISH"
        else:
            ema_crossover = "NONE"

        price_above_200 = current['close'] > ema200

        # MACD analysis
        macd_line = current['macd_line']
        macd_signal_val = current['macd_signal']
        macd_histogram = current['macd_histogram']
        macd_bullish = macd_line > macd_signal_val

        # MACD crossover detection
        prev_macd_bullish = prev['macd_line'] > prev['macd_signal']
        if macd_bullish and not prev_macd_bullish:
            macd_crossover = "BULLISH"
        elif not macd_bullish and prev_macd_bullish:
            macd_crossover = "BEARISH"
        else:
            macd_crossover = "NONE"

        # MACD histogram momentum
        macd_histogram_growing = abs(macd_histogram) > abs(prev['macd_histogram'])

        # VWAP analysis
        vwap = current['vwap']
        price_above_vwap = current['close'] > vwap

        # VWAP crossover detection
        prev_above_vwap = prev['close'] > prev['vwap']
        if price_above_vwap and not prev_above_vwap:
            vwap_crossover = "BULLISH"
        elif not price_above_vwap and prev_above_vwap:
            vwap_crossover = "BEARISH"
        else:
            vwap_crossover = "NONE"

        # Volume/Candle momentum analysis
        volume_increasing = current['volume'] > prev['volume']

        # Analyze last 10 candles for green/red dominance
        recent = df.tail(10).copy()  # Use .copy() to avoid SettingWithCopyWarning
        green_candles = (recent['close'] > recent['open']).sum()
        red_candles = (recent['close'] < recent['open']).sum()
        total_candles = green_candles + red_candles
        green_candle_dominance = green_candles / total_candles if total_candles > 0 else 0.5

        # Candle size analysis for momentum
        recent.loc[:, 'candle_size'] = abs(recent['close'] - recent['open'])
        recent.loc[:, 'is_green'] = recent['close'] > recent['open']

        # Compare first half to second half
        first_half = recent.head(5)
        second_half = recent.tail(5)

        first_green_size = first_half[first_half['is_green']]['candle_size'].mean() or 0
        second_green_size = second_half[second_half['is_green']]['candle_size'].mean() or 0
        first_red_size = first_half[~first_half['is_green']]['candle_size'].mean() or 0
        second_red_size = second_half[~second_half['is_green']]['candle_size'].mean() or 0

        # Momentum: greens growing + reds shrinking = BUILDING
        if second_green_size > first_green_size * 1.1 and second_red_size < first_red_size * 0.9:
            candle_momentum = "BUILDING"
        elif second_red_size > first_red_size * 1.1 and second_green_size < first_green_size * 0.9:
            candle_momentum = "FADING"
        else:
            candle_momentum = "NEUTRAL"

        # Calculate confluence score
        bullish_signals = 0
        bearish_signals = 0

        # EMA signals (weight: 2)
        if ema_bullish:
            bullish_signals += 2
        else:
            bearish_signals += 2

        if ema_crossover == "BULLISH":
            bullish_signals += 1
        elif ema_crossover == "BEARISH":
            bearish_signals += 1

        if price_above_200:
            bullish_signals += 1
        else:
            bearish_signals += 1

        # MACD signals (weight: 2)
        if macd_bullish:
            bullish_signals += 2
        else:
            bearish_signals += 2

        if macd_crossover == "BULLISH":
            bullish_signals += 1
        elif macd_crossover == "BEARISH":
            bearish_signals += 1

        if macd_histogram_growing and macd_bullish:
            bullish_signals += 1
        elif macd_histogram_growing and not macd_bullish:
            bearish_signals += 1

        # VWAP signals (weight: 2)
        if price_above_vwap:
            bullish_signals += 2
        else:
            bearish_signals += 2

        if vwap_crossover == "BULLISH":
            bullish_signals += 1
        elif vwap_crossover == "BEARISH":
            bearish_signals += 1

        # Volume/candle signals (weight: 1)
        if green_candle_dominance > 0.6:
            bullish_signals += 1
        elif green_candle_dominance < 0.4:
            bearish_signals += 1

        if candle_momentum == "BUILDING":
            bullish_signals += 1
        elif candle_momentum == "FADING":
            bearish_signals += 1

        if volume_increasing:
            if ema_bullish and macd_bullish:
                bullish_signals += 1
            elif not ema_bullish and not macd_bullish:
                bearish_signals += 1

        # Calculate confluence score (0-100)
        total_signals = bullish_signals + bearish_signals
        if total_signals > 0:
            confluence_score = (bullish_signals / total_signals) * 100
        else:
            confluence_score = 50

        # Determine signal bias
        if confluence_score >= 80:
            signal_bias = "STRONG_BUY"
        elif confluence_score >= 60:
            signal_bias = "BUY"
        elif confluence_score >= 40:
            signal_bias = "NEUTRAL"
        elif confluence_score >= 20:
            signal_bias = "SELL"
        else:
            signal_bias = "STRONG_SELL"

        # Convert numpy types to Python native types for JSON serialization
        state = SignalState(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=int(current['time']),
            ema9=float(round(ema9, 4)),
            ema20=float(round(ema20, 4)),
            ema200=float(round(ema200, 4)),
            ema_9_above_20=bool(ema_9_above_20),
            ema_bullish=bool(ema_bullish),
            ema_crossover=ema_crossover,
            price_above_200=bool(price_above_200),
            macd_line=float(round(macd_line, 4)),
            macd_signal=float(round(macd_signal_val, 4)),
            macd_histogram=float(round(macd_histogram, 4)),
            macd_bullish=bool(macd_bullish),
            macd_crossover=macd_crossover,
            macd_histogram_growing=bool(macd_histogram_growing),
            vwap=float(round(vwap, 4)),
            price_above_vwap=bool(price_above_vwap),
            vwap_crossover=vwap_crossover,
            volume_increasing=bool(volume_increasing),
            green_candle_dominance=float(round(green_candle_dominance, 2)),
            candle_momentum=candle_momentum,
            bullish_signals=int(bullish_signals),
            bearish_signals=int(bearish_signals),
            confluence_score=float(round(confluence_score, 1)),
            signal_bias=signal_bias
        )

        self.signal_cache[symbol] = state
        return state

    def get_signal_events(self, symbol: str, ohlc_data: List[Dict], timeframe: str = "5m") -> List[SignalEvent]:
        """
        Scan historical data and return all signal events for charting.

        Returns list of SignalEvent objects that can be plotted on chart.
        """
        if not ohlc_data or len(ohlc_data) < 50:
            return []

        # Convert to DataFrame
        df = pd.DataFrame(ohlc_data)
        df = df.sort_values('time').reset_index(drop=True)

        # Calculate indicators
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()

        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_line'] = ema12 - ema26
        df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()

        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_volume'] = df['typical_price'] * df['volume']
        df['cum_tp_vol'] = df['tp_volume'].cumsum()
        df['cum_vol'] = df['volume'].cumsum()
        df['vwap'] = df['cum_tp_vol'] / df['cum_vol']

        # Detect crossover events
        events = []

        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            time = int(curr['time'])

            # EMA 9/20 crossover
            curr_ema_bullish = curr['ema9'] > curr['ema20']
            prev_ema_bullish = prev['ema9'] > prev['ema20']

            if curr_ema_bullish and not prev_ema_bullish:
                events.append(SignalEvent(
                    time=time,
                    event_type="EMA_CROSS",
                    direction="BULLISH",
                    description="EMA 9 crossed above 20",
                    strength=0.8
                ))
            elif not curr_ema_bullish and prev_ema_bullish:
                events.append(SignalEvent(
                    time=time,
                    event_type="EMA_CROSS",
                    direction="BEARISH",
                    description="EMA 9 crossed below 20",
                    strength=0.8
                ))

            # MACD crossover
            curr_macd_bullish = curr['macd_line'] > curr['macd_signal']
            prev_macd_bullish = prev['macd_line'] > prev['macd_signal']

            if curr_macd_bullish and not prev_macd_bullish:
                events.append(SignalEvent(
                    time=time,
                    event_type="MACD_CROSS",
                    direction="BULLISH",
                    description="MACD crossed above signal",
                    strength=0.7
                ))
            elif not curr_macd_bullish and prev_macd_bullish:
                events.append(SignalEvent(
                    time=time,
                    event_type="MACD_CROSS",
                    direction="BEARISH",
                    description="MACD crossed below signal",
                    strength=0.7
                ))

            # VWAP crossover
            curr_above_vwap = curr['close'] > curr['vwap']
            prev_above_vwap = prev['close'] > prev['vwap']

            if curr_above_vwap and not prev_above_vwap:
                events.append(SignalEvent(
                    time=time,
                    event_type="VWAP_CROSS",
                    direction="BULLISH",
                    description="Price crossed above VWAP",
                    strength=0.6
                ))
            elif not curr_above_vwap and prev_above_vwap:
                events.append(SignalEvent(
                    time=time,
                    event_type="VWAP_CROSS",
                    direction="BEARISH",
                    description="Price crossed below VWAP",
                    strength=0.6
                ))

        self.event_cache[symbol] = events
        return events

    def get_multi_timeframe_alignment(self, symbol: str,
                                       ohlc_1m: List[Dict],
                                       ohlc_5m: List[Dict]) -> Dict:
        """
        Check alignment between 1M and 5M timeframes.

        Returns alignment status and recommendation.
        """
        state_1m = self.analyze(symbol, ohlc_1m, "1m")
        state_5m = self.analyze(symbol, ohlc_5m, "5m")

        if not state_1m or not state_5m:
            return {"aligned": False, "recommendation": "INSUFFICIENT_DATA"}

        # Check alignment
        ema_aligned = state_1m.ema_bullish == state_5m.ema_bullish
        macd_aligned = state_1m.macd_bullish == state_5m.macd_bullish
        vwap_aligned = state_1m.price_above_vwap == state_5m.price_above_vwap

        alignment_count = sum([ema_aligned, macd_aligned, vwap_aligned])

        # Both bullish
        if state_1m.ema_bullish and state_5m.ema_bullish and alignment_count >= 2:
            recommendation = "STRONG_BUY"
            aligned = True
        # Both bearish
        elif not state_1m.ema_bullish and not state_5m.ema_bullish and alignment_count >= 2:
            recommendation = "STRONG_SELL"
            aligned = True
        # Mixed signals
        elif alignment_count >= 2:
            recommendation = "WEAK_" + ("BUY" if state_1m.ema_bullish else "SELL")
            aligned = False
        else:
            recommendation = "NEUTRAL"
            aligned = False

        return {
            "aligned": aligned,
            "alignment_count": alignment_count,
            "recommendation": recommendation,
            "1m_bias": state_1m.signal_bias,
            "5m_bias": state_5m.signal_bias,
            "1m_confluence": state_1m.confluence_score,
            "5m_confluence": state_5m.confluence_score,
            "details": {
                "ema_aligned": ema_aligned,
                "macd_aligned": macd_aligned,
                "vwap_aligned": vwap_aligned
            }
        }

    def should_enter(self, state: SignalState) -> Tuple[bool, str]:
        """
        Determine if conditions are favorable for entry.

        Returns (should_enter, reason)
        """
        if not state:
            return False, "No signal state"

        # Strong buy conditions
        if state.confluence_score >= 70:
            if state.ema_crossover == "BULLISH" or state.macd_crossover == "BULLISH":
                return True, f"Confluence {state.confluence_score}% with crossover trigger"
            elif state.vwap_crossover == "BULLISH":
                return True, f"Confluence {state.confluence_score}% with VWAP breakout"
            elif state.ema_bullish and state.macd_bullish and state.price_above_vwap:
                return True, f"Confluence {state.confluence_score}% - all signals aligned"

        # Moderate buy with fresh crossover
        if state.confluence_score >= 55:
            if state.ema_crossover == "BULLISH" and state.macd_bullish:
                return True, f"EMA crossover with MACD confirmation"
            if state.macd_crossover == "BULLISH" and state.ema_bullish:
                return True, f"MACD crossover with EMA confirmation"

        return False, f"Confluence {state.confluence_score}% - waiting for better setup"

    def should_exit(self, state: SignalState, entry_price: float, current_price: float) -> Tuple[bool, str]:
        """
        Determine if conditions suggest exit.

        Returns (should_exit, reason)
        """
        if not state:
            return False, "No signal state"

        pnl_pct = ((current_price - entry_price) / entry_price) * 100

        # Strong exit signals
        if state.ema_crossover == "BEARISH":
            return True, "EMA 9/20 bearish crossover"

        if state.macd_crossover == "BEARISH" and pnl_pct > 0:
            return True, "MACD bearish crossover - locking profit"

        if state.vwap_crossover == "BEARISH" and pnl_pct > 0:
            return True, "VWAP breakdown - locking profit"

        if state.candle_momentum == "FADING" and pnl_pct > 1:
            return True, "Momentum fading - taking profit"

        # Confluence collapse
        if state.confluence_score < 30 and pnl_pct > 0:
            return True, f"Confluence dropped to {state.confluence_score}%"

        # Strong bearish signal
        if state.signal_bias in ["SELL", "STRONG_SELL"] and pnl_pct > 0:
            return True, f"Signal turned {state.signal_bias}"

        return False, f"Holding - confluence {state.confluence_score}%"


# Singleton instance
_analyzer = None

def get_signal_analyzer() -> TechnicalSignalAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = TechnicalSignalAnalyzer()
    return _analyzer
