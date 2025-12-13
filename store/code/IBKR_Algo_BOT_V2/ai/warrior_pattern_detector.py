"""
Warrior Trading Pattern Detector
Detects Ross Cameron's momentum patterns in real-time

Patterns:
1. Bull Flag - Strong move up → controlled pullback → breakout
2. HOD Breakout - New intraday high with volume surge
3. Whole Dollar Breakout - Psychological level breaks (5.00, 10.00, etc.)
4. Micro Pullback - Brief retrace to moving average in strong trend
5. Hammer Reversal - Bottoming tail at support (higher risk)
"""

import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.config_loader import get_config

logger = logging.getLogger(__name__)


class SetupType(str, Enum):
    """Warrior Trading setup types"""
    BULL_FLAG = "BULL_FLAG"
    HOD_BREAKOUT = "HOD_BREAKOUT"
    WHOLE_DOLLAR_BREAKOUT = "WHOLE_DOLLAR_BREAKOUT"
    MICRO_PULLBACK = "MICRO_PULLBACK"
    HAMMER_REVERSAL = "HAMMER_REVERSAL"


@dataclass
class TradingSetup:
    """
    Detected Warrior Trading setup with entry/stop/target levels

    Attributes:
        setup_type: Type of pattern detected
        symbol: Stock ticker
        timeframe: Chart timeframe ("1min", "5min")
        entry_price: Suggested entry price
        entry_condition: Human-readable entry trigger
        stop_price: Stop loss price
        stop_reason: Why stop is placed there
        target_1r: 1:1 risk/reward target
        target_2r: 2:1 risk/reward target
        target_3r: 3:1 risk/reward target
        risk_per_share: $ risk per share
        reward_per_share: $ reward per share (to 2R)
        risk_reward_ratio: Actual R:R ratio
        confidence: Pattern quality score (0-100)
        strength_factors: List of bullish factors
        risk_factors: List of bearish factors
        current_price: Price when detected
        timestamp: Detection time
    """
    setup_type: SetupType
    symbol: str
    timeframe: str

    # Entry
    entry_price: float
    entry_condition: str

    # Stop
    stop_price: float
    stop_reason: str

    # Targets
    target_1r: float
    target_2r: float
    target_3r: float

    # Risk metrics
    risk_per_share: float
    reward_per_share: float
    risk_reward_ratio: float

    # Confidence
    confidence: float
    strength_factors: List[str]
    risk_factors: List[str]

    # Context
    current_price: float
    timestamp: str = ""

    def __post_init__(self):
        """Set timestamp if not provided"""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['setup_type'] = self.setup_type.value
        return data

    def is_valid(self) -> bool:
        """Check if setup has valid risk/reward"""
        return (
            self.risk_reward_ratio >= 2.0 and
            self.risk_per_share > 0 and
            self.entry_price > self.stop_price
        )


class WarriorPatternDetector:
    """
    Detect Warrior Trading patterns on live market data

    Uses candlestick data, volume, and indicators to identify high-probability setups
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pattern detector

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = get_config(config_path)
        self.pattern_config = self.config.patterns

        logger.info("WarriorPatternDetector initialized")
        logger.info(f"Enabled patterns: {self.pattern_config.enabled_patterns}")

    def detect_all_patterns(
        self,
        symbol: str,
        candles_1m: pd.DataFrame,
        candles_5m: pd.DataFrame,
        vwap: float,
        ema9_1m: float,
        ema9_5m: float,
        ema20_5m: float,
        high_of_day: float
    ) -> List[TradingSetup]:
        """
        Run all enabled pattern detectors

        Args:
            symbol: Stock ticker
            candles_1m: 1-minute candle data (OHLCV)
            candles_5m: 5-minute candle data (OHLCV)
            vwap: Volume-weighted average price
            ema9_1m: 9 EMA on 1-min chart
            ema9_5m: 9 EMA on 5-min chart
            ema20_5m: 20 EMA on 5-min chart
            high_of_day: Current high of day

        Returns:
            List of detected setups, sorted by confidence
        """
        setups = []

        # Bull Flag (5-min chart)
        if self._is_pattern_enabled(SetupType.BULL_FLAG):
            bull_flag = self.detect_bull_flag(
                symbol, candles_5m, vwap, ema9_5m, ema20_5m
            )
            if bull_flag:
                setups.append(bull_flag)

        # HOD Breakout (1-min chart)
        if self._is_pattern_enabled(SetupType.HOD_BREAKOUT):
            hod = self.detect_hod_breakout(
                symbol, candles_1m, high_of_day, vwap, ema9_1m
            )
            if hod:
                setups.append(hod)

        # Whole Dollar (1-min chart)
        if self._is_pattern_enabled(SetupType.WHOLE_DOLLAR_BREAKOUT):
            whole_dollar = self.detect_whole_dollar_breakout(
                symbol, candles_1m, vwap, ema9_1m
            )
            if whole_dollar:
                setups.append(whole_dollar)

        # Micro Pullback (5-min chart)
        if self._is_pattern_enabled(SetupType.MICRO_PULLBACK):
            pullback = self.detect_micro_pullback(
                symbol, candles_5m, vwap, ema9_5m, ema20_5m
            )
            if pullback:
                setups.append(pullback)

        # Hammer Reversal (5-min chart) - Optional, higher risk
        if self._is_pattern_enabled(SetupType.HAMMER_REVERSAL):
            hammer = self.detect_hammer_reversal(
                symbol, candles_5m, vwap, ema9_5m
            )
            if hammer:
                setups.append(hammer)

        # Sort by confidence (descending)
        setups.sort(key=lambda x: x.confidence, reverse=True)

        logger.info(f"{symbol}: Found {len(setups)} patterns")
        for setup in setups:
            logger.debug(
                f"  {setup.setup_type.value}: {setup.confidence:.0f}% confidence"
            )

        return setups

    def detect_bull_flag(
        self,
        symbol: str,
        candles_5m: pd.DataFrame,
        vwap: float,
        ema9: float,
        ema20: float
    ) -> Optional[TradingSetup]:
        """
        Detect bull flag pattern

        Pattern:
        1. Strong upward move (pole) - 3+ green candles
        2. Controlled pullback (flag) - 1-4 candles, lower volume
        3. Price stays above 9 EMA or VWAP
        4. Volume decreases on pullback
        5. No major resistance overhead

        Args:
            symbol: Stock ticker
            candles_5m: 5-minute candle data
            vwap: Volume-weighted average price
            ema9: 9-period EMA
            ema20: 20-period EMA

        Returns:
            TradingSetup or None
        """
        try:
            config = self.pattern_config.bull_flag

            if len(candles_5m) < 10:
                return None

            # Analyze last 10 candles
            recent = candles_5m.iloc[-10:].copy()

            # Identify pole (strong upward move)
            pole_start_idx = -10
            pole_end_idx = -5
            pole_candles = recent.iloc[pole_start_idx:pole_end_idx]

            # Check for strong upward pole
            green_candles = (pole_candles['Close'] > pole_candles['Open']).sum()
            pole_range = pole_candles['High'].max() - pole_candles['Low'].min()
            pole_avg_price = pole_candles['Close'].mean()

            min_candles = config.get('pole_min_candles', 3)
            min_range_pct = config.get('pole_min_range_percent', 2.0) / 100

            if green_candles < min_candles:
                return None

            if pole_range / pole_avg_price < min_range_pct:
                return None

            # Check for controlled pullback (flag)
            max_pullback_candles = config.get('pullback_max_candles', 4)
            pullback_candles = recent.iloc[-max_pullback_candles:]
            pullback_low = pullback_candles['Low'].min()
            pullback_high = pullback_candles['High'].max()

            # Support level (9 EMA or VWAP, whichever is higher)
            support_level = max(ema9, vwap)

            # Pullback should stay above support
            if config.get('require_above_ema9', True):
                if pullback_low < support_level * 0.99:  # 1% cushion
                    return None

            # Check volume (pullback volume should be lower than pole)
            pole_avg_volume = pole_candles['Volume'].mean()
            pullback_avg_volume = pullback_candles['Volume'].mean()

            max_volume_ratio = config.get('pullback_max_volume_ratio', 0.8)
            if pullback_avg_volume > pole_avg_volume * max_volume_ratio:
                return None

            # Calculate entry/stop/targets
            current_price = recent['Close'].iloc[-1]
            entry = pullback_high + 0.02  # Entry above flag high
            stop = max(pullback_low - 0.05, support_level * 0.99)
            risk = entry - stop

            if risk <= 0:
                return None

            target_1r = entry + risk
            target_2r = entry + (risk * 2)
            target_3r = entry + (risk * 3)

            # Confidence factors
            strength = []
            risks = []

            if current_price > vwap:
                strength.append("Price above VWAP")
            else:
                risks.append("Below VWAP")

            if current_price > ema9:
                strength.append("Above 9 EMA")

            if green_candles >= 4:
                strength.append(f"Strong pole ({green_candles} green candles)")

            pole_range_pct = (pole_range / pole_avg_price) * 100
            if pole_range_pct > 5:
                strength.append(f"Large pole range ({pole_range_pct:.1f}%)")

            volume_reduction = (1 - pullback_avg_volume / pole_avg_volume) * 100
            if volume_reduction > 30:
                strength.append(f"Volume decreased {volume_reduction:.0f}%")

            # Calculate confidence
            base_confidence = config.get('min_confidence', 60)
            confidence = base_confidence + len(strength) * 5 - len(risks) * 10
            confidence = max(min(confidence, 95.0), 40.0)

            setup = TradingSetup(
                setup_type=SetupType.BULL_FLAG,
                symbol=symbol,
                timeframe="5min",
                entry_price=entry,
                entry_condition=f"Break above flag high ${pullback_high:.2f}",
                stop_price=stop,
                stop_reason=f"Below pullback low / support at ${stop:.2f}",
                target_1r=target_1r,
                target_2r=target_2r,
                target_3r=target_3r,
                risk_per_share=risk,
                reward_per_share=risk * 2,
                risk_reward_ratio=2.0,
                confidence=confidence,
                strength_factors=strength,
                risk_factors=risks,
                current_price=current_price
            )

            if setup.is_valid():
                logger.info(f"{symbol}: Bull flag detected (confidence: {confidence:.0f}%)")
                return setup

        except Exception as e:
            logger.error(f"Error detecting bull flag for {symbol}: {e}")

        return None

    def detect_hod_breakout(
        self,
        symbol: str,
        candles_1m: pd.DataFrame,
        high_of_day: float,
        vwap: float,
        ema9: float
    ) -> Optional[TradingSetup]:
        """
        Detect high-of-day breakout

        Pattern:
        - Stock approaching or breaking HOD
        - Volume increasing
        - Price above VWAP (preferred)

        Args:
            symbol: Stock ticker
            candles_1m: 1-minute candle data
            high_of_day: Current high of day
            vwap: VWAP
            ema9: 9 EMA

        Returns:
            TradingSetup or None
        """
        try:
            config = self.pattern_config.hod_breakout

            if len(candles_1m) < 5:
                return None

            recent = candles_1m.iloc[-5:].copy()
            current_price = recent['Close'].iloc[-1]
            current_high = recent['High'].iloc[-1]

            # Check if we're near HOD
            proximity_pct = config.get('proximity_percent', 1.0) / 100
            distance_to_hod = (high_of_day - current_price) / current_price

            if distance_to_hod > proximity_pct or distance_to_hod < -0.005:
                return None  # Too far or already broken significantly

            # Check volume trend (should be increasing)
            volume_trend = recent['Volume'].iloc[-3:].mean() / recent['Volume'].iloc[:2].mean()
            min_surge = config.get('volume_surge_ratio', 1.5)

            if volume_trend < min_surge:
                return None

            # Entry above HOD
            entry = high_of_day + 0.02

            # Stop below recent consolidation
            consolidation_low = recent['Low'].iloc[-3:].min()
            stop = consolidation_low - 0.05

            risk = entry - stop
            if risk <= 0:
                return None

            target_2r = entry + (risk * 2)
            target_3r = entry + (risk * 3)

            # Strength factors
            strength = []
            risks = []

            if current_price > vwap:
                strength.append("Price above VWAP")
            else:
                risks.append("Below VWAP")

            if volume_trend > 2.0:
                strength.append(f"Strong volume surge ({volume_trend:.1f}x)")
            elif volume_trend > 1.5:
                strength.append(f"Volume increasing ({volume_trend:.1f}x)")

            if current_high >= high_of_day * 0.999:
                strength.append("Already tested HOD")

            if current_price > ema9:
                strength.append("Above 9 EMA")

            # Calculate confidence
            base_confidence = config.get('min_confidence', 55)
            confidence = base_confidence + len(strength) * 7 - len(risks) * 10
            confidence = max(min(confidence, 90.0), 40.0)

            setup = TradingSetup(
                setup_type=SetupType.HOD_BREAKOUT,
                symbol=symbol,
                timeframe="1min",
                entry_price=entry,
                entry_condition=f"Break above HOD ${high_of_day:.2f}",
                stop_price=stop,
                stop_reason=f"Below consolidation low ${stop:.2f}",
                target_1r=entry + risk,
                target_2r=target_2r,
                target_3r=target_3r,
                risk_per_share=risk,
                reward_per_share=risk * 2,
                risk_reward_ratio=2.0,
                confidence=confidence,
                strength_factors=strength,
                risk_factors=risks,
                current_price=current_price
            )

            if setup.is_valid():
                logger.info(f"{symbol}: HOD breakout detected (confidence: {confidence:.0f}%)")
                return setup

        except Exception as e:
            logger.error(f"Error detecting HOD breakout for {symbol}: {e}")

        return None

    def detect_whole_dollar_breakout(
        self,
        symbol: str,
        candles_1m: pd.DataFrame,
        vwap: float,
        ema9: float
    ) -> Optional[TradingSetup]:
        """
        Detect whole dollar level breakout

        Pattern:
        - Stock approaching whole number (2, 3, 5, 10, etc.)
        - Uptrend intact
        - Volume increasing (optional)

        Args:
            symbol: Stock ticker
            candles_1m: 1-minute candle data
            vwap: VWAP
            ema9: 9 EMA

        Returns:
            TradingSetup or None
        """
        try:
            config = self.pattern_config.whole_dollar_breakout

            if len(candles_1m) < 5:
                return None

            recent = candles_1m.iloc[-5:].copy()
            current_price = recent['Close'].iloc[-1]

            # Find nearest whole dollar above current price
            whole_levels = config.get('whole_dollar_levels', [
                2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 75, 100
            ])

            nearest_whole = None
            for level in whole_levels:
                if level > current_price:
                    nearest_whole = float(level)
                    break

            if nearest_whole is None:
                return None

            # Check proximity
            proximity_pct = config.get('proximity_percent', 2.0) / 100
            distance = (nearest_whole - current_price) / current_price

            if distance > proximity_pct or distance < -0.005:
                return None  # Too far or already broken

            # Check uptrend
            if config.get('require_uptrend', True):
                if current_price < ema9:
                    return None

            # Entry above whole dollar
            entry = nearest_whole + 0.01

            # Stop below psychological support
            if nearest_whole >= 10:
                stop = nearest_whole - 0.20
            else:
                stop = nearest_whole - 0.10

            risk = entry - stop
            if risk <= 0:
                return None

            target_2r = entry + (risk * 2)
            target_3r = entry + (risk * 3)

            # Strength factors
            strength = []
            risks = []

            if current_price > vwap:
                strength.append("Above VWAP")
            else:
                risks.append("Below VWAP")

            if current_price > ema9:
                strength.append("Above 9 EMA (uptrend)")
            else:
                risks.append("Below 9 EMA")

            strength.append(f"Approaching ${nearest_whole:.0f} whole dollar")

            # Calculate confidence
            base_confidence = config.get('min_confidence', 50)
            confidence = base_confidence + len(strength) * 8 - len(risks) * 12
            confidence = max(min(confidence, 85.0), 35.0)

            setup = TradingSetup(
                setup_type=SetupType.WHOLE_DOLLAR_BREAKOUT,
                symbol=symbol,
                timeframe="1min",
                entry_price=entry,
                entry_condition=f"Break ${nearest_whole:.0f} whole dollar",
                stop_price=stop,
                stop_reason=f"Below psychological support ${stop:.2f}",
                target_1r=entry + risk,
                target_2r=target_2r,
                target_3r=target_3r,
                risk_per_share=risk,
                reward_per_share=risk * 2,
                risk_reward_ratio=2.0,
                confidence=confidence,
                strength_factors=strength,
                risk_factors=risks,
                current_price=current_price
            )

            if setup.is_valid():
                logger.info(
                    f"{symbol}: Whole dollar breakout detected at "
                    f"${nearest_whole:.0f} (confidence: {confidence:.0f}%)"
                )
                return setup

        except Exception as e:
            logger.error(f"Error detecting whole dollar breakout for {symbol}: {e}")

        return None

    def detect_micro_pullback(
        self,
        symbol: str,
        candles_5m: pd.DataFrame,
        vwap: float,
        ema9: float,
        ema20: float
    ) -> Optional[TradingSetup]:
        """
        Detect micro pullback in strong trend

        Pattern:
        - Strong uptrend (price > 9 EMA > 20 EMA)
        - Brief 1-3 candle pullback
        - Pullback to 9 EMA or 20 EMA
        - Low volume on pullback

        Args:
            symbol: Stock ticker
            candles_5m: 5-minute candle data
            vwap: VWAP
            ema9: 9 EMA
            ema20: 20 EMA

        Returns:
            TradingSetup or None
        """
        try:
            config = self.pattern_config.micro_pullback

            if len(candles_5m) < 10:
                return None

            recent = candles_5m.iloc[-10:].copy()
            current_price = recent['Close'].iloc[-1]

            # Check for uptrend
            if config.get('require_trend', True):
                if not (current_price > ema9 > ema20):
                    return None

            # Look for pullback (last 1-3 candles)
            max_pullback_candles = config.get('max_pullback_candles', 3)
            pullback = recent.iloc[-max_pullback_candles:]

            # Check if pullback touched EMA
            pullback_low = pullback['Low'].min()
            touched_ema9 = abs(pullback_low - ema9) / ema9 < 0.01  # Within 1%
            touched_ema20 = abs(pullback_low - ema20) / ema20 < 0.01

            if config.get('pullback_to_ema', True):
                if not (touched_ema9 or touched_ema20):
                    return None

            # Entry on bounce
            entry = current_price + 0.02

            # Stop below EMA
            if touched_ema9:
                stop = ema9 - 0.10
            else:
                stop = ema20 - 0.10

            risk = entry - stop
            if risk <= 0:
                return None

            target_2r = entry + (risk * 2)
            target_3r = entry + (risk * 3)

            # Strength factors
            strength = []
            risks = []

            if current_price > vwap:
                strength.append("Above VWAP")

            if current_price > ema9 > ema20:
                strength.append("Strong uptrend (9 EMA > 20 EMA)")

            if touched_ema9:
                strength.append("Pullback to 9 EMA")
            elif touched_ema20:
                strength.append("Pullback to 20 EMA")

            # Calculate confidence
            base_confidence = config.get('min_confidence', 65)
            confidence = base_confidence + len(strength) * 6 - len(risks) * 10
            confidence = max(min(confidence, 90.0), 45.0)

            setup = TradingSetup(
                setup_type=SetupType.MICRO_PULLBACK,
                symbol=symbol,
                timeframe="5min",
                entry_price=entry,
                entry_condition="First green candle after pullback",
                stop_price=stop,
                stop_reason=f"Below moving average ${stop:.2f}",
                target_1r=entry + risk,
                target_2r=target_2r,
                target_3r=target_3r,
                risk_per_share=risk,
                reward_per_share=risk * 2,
                risk_reward_ratio=2.0,
                confidence=confidence,
                strength_factors=strength,
                risk_factors=risks,
                current_price=current_price
            )

            if setup.is_valid():
                logger.info(
                    f"{symbol}: Micro pullback detected (confidence: {confidence:.0f}%)"
                )
                return setup

        except Exception as e:
            logger.error(f"Error detecting micro pullback for {symbol}: {e}")

        return None

    def detect_hammer_reversal(
        self,
        symbol: str,
        candles_5m: pd.DataFrame,
        vwap: float,
        ema9: float
    ) -> Optional[TradingSetup]:
        """
        Detect hammer reversal pattern (HIGHER RISK)

        Pattern:
        - Extended down move (3+ red candles)
        - Hammer candle (long lower wick)
        - Support level nearby
        - Volume climax

        Args:
            symbol: Stock ticker
            candles_5m: 5-minute candle data
            vwap: VWAP
            ema9: 9 EMA

        Returns:
            TradingSetup or None
        """
        try:
            config = self.pattern_config.hammer_reversal

            if len(candles_5m) < 10:
                return None

            recent = candles_5m.iloc[-10:].copy()
            current_price = recent['Close'].iloc[-1]

            # Check for extended down move
            min_red_candles = config.get('min_red_candles', 3)
            last_n = recent.iloc[-(min_red_candles + 1):-1]
            red_candles = (last_n['Close'] < last_n['Open']).sum()

            if red_candles < min_red_candles:
                return None

            # Check for hammer candle (last candle)
            last_candle = recent.iloc[-1]
            body = abs(last_candle['Close'] - last_candle['Open'])
            lower_wick = min(last_candle['Open'], last_candle['Close']) - last_candle['Low']
            total_range = last_candle['High'] - last_candle['Low']

            # Hammer criteria: lower wick > 2x body
            if lower_wick < body * 2 or total_range == 0:
                return None

            # Entry above hammer high
            entry = last_candle['High'] + 0.02

            # Stop below hammer low
            stop = last_candle['Low'] - 0.05

            risk = entry - stop
            if risk <= 0:
                return None

            target_2r = entry + (risk * 2)
            target_3r = entry + (risk * 3)

            # Strength factors
            strength = []
            risks = []

            risks.append("Reversal trade (higher risk)")

            if lower_wick > body * 3:
                strength.append("Strong hammer (long wick)")

            if red_candles >= 4:
                strength.append(f"Extended selloff ({red_candles} red candles)")

            # Calculate confidence (lower for reversals)
            base_confidence = config.get('min_confidence', 70)
            confidence = base_confidence + len(strength) * 4 - len(risks) * 5
            confidence = max(min(confidence, 80.0), 40.0)

            setup = TradingSetup(
                setup_type=SetupType.HAMMER_REVERSAL,
                symbol=symbol,
                timeframe="5min",
                entry_price=entry,
                entry_condition=f"Break above hammer high ${last_candle['High']:.2f}",
                stop_price=stop,
                stop_reason=f"Below hammer low ${stop:.2f}",
                target_1r=entry + risk,
                target_2r=target_2r,
                target_3r=target_3r,
                risk_per_share=risk,
                reward_per_share=risk * 2,
                risk_reward_ratio=2.0,
                confidence=confidence,
                strength_factors=strength,
                risk_factors=risks,
                current_price=current_price
            )

            if setup.is_valid():
                logger.info(
                    f"{symbol}: Hammer reversal detected (confidence: {confidence:.0f}%)"
                )
                return setup

        except Exception as e:
            logger.error(f"Error detecting hammer reversal for {symbol}: {e}")

        return None

    def _is_pattern_enabled(self, pattern_type: SetupType) -> bool:
        """Check if pattern is enabled in configuration"""
        return pattern_type.value in self.pattern_config.enabled_patterns


# Example usage / testing
if __name__ == "__main__":
    import yfinance as yf

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("WARRIOR TRADING PATTERN DETECTOR - TEST MODE")
    print("=" * 80)

    try:
        # Initialize detector
        detector = WarriorPatternDetector()
        print(f"\n✅ Pattern detector initialized")
        print(f"Enabled patterns: {detector.pattern_config.enabled_patterns}")

        # Test with real data
        symbol = "TSLA"
        print(f"\nFetching data for {symbol}...")

        ticker = yf.Ticker(symbol)
        hist_5m = ticker.history(period="1d", interval="5m")
        hist_1m = ticker.history(period="1d", interval="1m")

        if hist_5m.empty or hist_1m.empty:
            print("❌ No data available (markets may be closed)")
        else:
            # Calculate indicators
            vwap = (hist_5m['Close'] * hist_5m['Volume']).sum() / hist_5m['Volume'].sum()
            ema9_5m = hist_5m['Close'].ewm(span=9, adjust=False).mean().iloc[-1]
            ema20_5m = hist_5m['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
            ema9_1m = hist_1m['Close'].ewm(span=9, adjust=False).mean().iloc[-1]
            hod = hist_1m['High'].max()

            print(f"  VWAP: ${vwap:.2f}")
            print(f"  9 EMA (5m): ${ema9_5m:.2f}")
            print(f"  20 EMA (5m): ${ema20_5m:.2f}")
            print(f"  High of Day: ${hod:.2f}")

            # Detect patterns
            print(f"\nDetecting patterns...")
            setups = detector.detect_all_patterns(
                symbol=symbol,
                candles_1m=hist_1m,
                candles_5m=hist_5m,
                vwap=vwap,
                ema9_1m=ema9_1m,
                ema9_5m=ema9_5m,
                ema20_5m=ema20_5m,
                high_of_day=hod
            )

            if not setups:
                print("  No patterns detected")
            else:
                print(f"\n✅ Found {len(setups)} pattern(s):\n")

                for i, setup in enumerate(setups, 1):
                    print(f"{i}. {setup.setup_type.value}")
                    print(f"   Entry: ${setup.entry_price:.2f}")
                    print(f"   Stop: ${setup.stop_price:.2f}")
                    print(f"   Target (2R): ${setup.target_2r:.2f}")
                    print(f"   Risk/Share: ${setup.risk_per_share:.2f}")
                    print(f"   R:R: {setup.risk_reward_ratio:.1f}:1")
                    print(f"   Confidence: {setup.confidence:.0f}%")
                    print(f"   Strengths: {', '.join(setup.strength_factors)}")
                    if setup.risk_factors:
                        print(f"   Risks: {', '.join(setup.risk_factors)}")
                    print()

    except Exception as e:
        logger.error(f"Error in pattern detection test: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
