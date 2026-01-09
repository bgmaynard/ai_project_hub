"""
VWAP Manager
============
Volume Weighted Average Price management for Ross Cameron trading methodology.

Ross Cameron Rules:
1. ENTRY: Only enter ABOVE VWAP (VWAP = institutional support)
2. STOP: Use VWAP as dynamic support - if price breaks below, exit
3. TRAIL: Trail stop to VWAP as price rises
4. AVOID: Don't enter if price is extended too far above VWAP (>3%)

VWAP Concepts:
- VWAP = Sum(Price * Volume) / Sum(Volume) for the day
- Resets at market open each day
- Institutional traders use VWAP for large order execution
- Price above VWAP = bullish (buyers in control)
- Price below VWAP = bearish (sellers in control)
- VWAP acts as magnet - extended prices tend to return to VWAP
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VWAPPosition(Enum):
    """Position relative to VWAP"""

    FAR_ABOVE = "FAR_ABOVE"  # >3% above - extended, risky entry
    ABOVE = "ABOVE"  # 0-3% above - ideal entry zone
    AT_VWAP = "AT_VWAP"  # Within 0.5% - at VWAP
    BELOW = "BELOW"  # 0-3% below - weak, avoid longs
    FAR_BELOW = "FAR_BELOW"  # >3% below - very weak


class VWAPSignal(Enum):
    """VWAP-based trading signals"""

    STRONG_BUY = "STRONG_BUY"  # Just crossed above VWAP with volume
    BUY_SUPPORTED = "BUY_SUPPORTED"  # Above VWAP, holding support
    NEUTRAL = "NEUTRAL"  # At VWAP, waiting for direction
    SELL_WARNING = "SELL_WARNING"  # Approaching VWAP from above
    SELL = "SELL"  # Broke below VWAP
    AVOID = "AVOID"  # Extended or below VWAP


@dataclass
class VWAPData:
    """VWAP data for a symbol"""

    symbol: str
    vwap: float = 0.0
    current_price: float = 0.0

    # Position relative to VWAP
    position: VWAPPosition = VWAPPosition.AT_VWAP
    distance_pct: float = 0.0  # Positive = above, negative = below

    # VWAP bands (standard deviation bands)
    upper_band_1: float = 0.0  # +1 std dev
    upper_band_2: float = 0.0  # +2 std dev
    lower_band_1: float = 0.0  # -1 std dev
    lower_band_2: float = 0.0  # -2 std dev

    # Crossover detection
    crossed_above: bool = False
    crossed_below: bool = False
    candles_since_cross: int = 0

    # Volume analysis
    volume_at_vwap: int = 0  # Volume when at VWAP
    avg_volume: float = 0.0
    relative_volume: float = 0.0

    # Trading signals
    signal: VWAPSignal = VWAPSignal.NEUTRAL
    entry_valid: bool = False
    stop_price: float = 0.0  # VWAP-based stop

    # Timestamps
    last_update: str = ""
    market_open: str = ""

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "vwap": round(self.vwap, 4),
            "current_price": round(self.current_price, 4),
            "position": self.position.value,
            "distance_pct": round(self.distance_pct, 2),
            "bands": {
                "upper_1": round(self.upper_band_1, 4),
                "upper_2": round(self.upper_band_2, 4),
                "lower_1": round(self.lower_band_1, 4),
                "lower_2": round(self.lower_band_2, 4),
            },
            "crossover": {
                "crossed_above": self.crossed_above,
                "crossed_below": self.crossed_below,
                "candles_since": self.candles_since_cross,
            },
            "signal": self.signal.value,
            "entry_valid": self.entry_valid,
            "stop_price": round(self.stop_price, 4),
            "relative_volume": round(self.relative_volume, 2),
            "last_update": self.last_update,
        }


@dataclass
class VWAPTrailingStop:
    """VWAP-based trailing stop for an open position"""

    symbol: str
    entry_price: float
    entry_vwap: float
    current_vwap: float = 0.0
    current_price: float = 0.0

    # Stop levels
    initial_stop: float = 0.0  # VWAP at entry
    current_stop: float = 0.0  # Updated VWAP or trailed
    stop_triggered: bool = False

    # Trailing mode
    trailing_active: bool = False  # Activated when price rises
    trail_from_vwap: bool = True  # Trail from VWAP vs fixed %
    trail_offset_pct: float = 0.3  # Trail 0.3% below VWAP

    # Stats
    max_price: float = 0.0
    max_distance_from_vwap: float = 0.0

    def update(self, price: float, vwap: float) -> Tuple[bool, str]:
        """
        Update trailing stop with new price/VWAP.
        Returns (should_exit, reason)
        """
        self.current_price = price
        self.current_vwap = vwap

        # Track max
        if price > self.max_price:
            self.max_price = price
            self.max_distance_from_vwap = (
                ((price - vwap) / vwap * 100) if vwap > 0 else 0
            )

        # Calculate current stop
        if self.trail_from_vwap:
            # Stop is VWAP minus small offset
            self.current_stop = vwap * (1 - self.trail_offset_pct / 100)
        else:
            # Fixed percentage trail from high
            self.current_stop = max(self.current_stop, self.max_price * 0.97)

        # Never lower the stop (only raise it)
        self.current_stop = max(self.current_stop, self.initial_stop)

        # Check if stop triggered
        if price <= self.current_stop:
            self.stop_triggered = True
            return True, f"VWAP stop triggered at ${self.current_stop:.2f}"

        # Check if broke below VWAP significantly
        if price < vwap * 0.995:  # 0.5% below VWAP
            self.stop_triggered = True
            return True, f"Price broke below VWAP (${vwap:.2f})"

        return False, ""


class VWAPManager:
    """
    VWAP Manager for Ross Cameron Trading

    Tracks VWAP for multiple symbols and provides:
    - Entry validation (must be above VWAP)
    - VWAP-based trailing stops
    - Crossover detection
    - VWAP bands for targets
    """

    def __init__(self):
        # VWAP data per symbol
        self.vwap_data: Dict[str, VWAPData] = {}

        # Cumulative data for VWAP calculation
        self.cumulative_pv: Dict[str, float] = {}  # Price * Volume
        self.cumulative_volume: Dict[str, float] = {}
        self.price_history: Dict[str, List[float]] = {}  # For std dev bands

        # Previous prices for crossover detection
        self.prev_prices: Dict[str, float] = {}
        self.prev_vwaps: Dict[str, float] = {}

        # Active trailing stops
        self.trailing_stops: Dict[str, VWAPTrailingStop] = {}

        # Configuration
        self.extended_threshold_pct = 3.0  # >3% above = extended
        self.at_vwap_threshold_pct = 0.5  # Within 0.5% = "at VWAP"
        self.min_volume_for_signal = 1000  # Min volume for crossover signal

        # Market hours (ET)
        self.market_open_time = time(9, 30)
        self.premarket_start = time(4, 0)

    def reset_daily(self, symbol: str = None):
        """Reset VWAP data at market open"""
        if symbol:
            symbols = [symbol]
        else:
            symbols = list(self.vwap_data.keys())

        for sym in symbols:
            self.cumulative_pv[sym] = 0.0
            self.cumulative_volume[sym] = 0.0
            self.price_history[sym] = []
            self.prev_prices[sym] = 0.0
            self.prev_vwaps[sym] = 0.0
            if sym in self.vwap_data:
                self.vwap_data[sym] = VWAPData(symbol=sym)

        logger.info(f"VWAP reset for {len(symbols)} symbols")

    def update_from_candle(self, symbol: str, candle: Dict) -> VWAPData:
        """
        Update VWAP from candle data.

        Candle should have: open, high, low, close, volume
        """
        symbol = symbol.upper()

        # Initialize if needed
        if symbol not in self.cumulative_pv:
            self.cumulative_pv[symbol] = 0.0
            self.cumulative_volume[symbol] = 0.0
            self.price_history[symbol] = []

        # Typical price = (H + L + C) / 3
        high = candle.get("high", candle.get("h", 0))
        low = candle.get("low", candle.get("l", 0))
        close = candle.get("close", candle.get("c", 0))
        volume = candle.get("volume", candle.get("v", 0))

        if not all([high, low, close, volume]):
            return self.vwap_data.get(symbol, VWAPData(symbol=symbol))

        typical_price = (high + low + close) / 3

        # Update cumulative
        self.cumulative_pv[symbol] += typical_price * volume
        self.cumulative_volume[symbol] += volume
        self.price_history[symbol].append(typical_price)

        # Keep only recent prices for std dev (last 100)
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]

        # Calculate VWAP
        if self.cumulative_volume[symbol] > 0:
            vwap = self.cumulative_pv[symbol] / self.cumulative_volume[symbol]
        else:
            vwap = close

        # Calculate VWAP bands (using std dev of prices)
        if len(self.price_history[symbol]) >= 10:
            std_dev = np.std(self.price_history[symbol])
        else:
            std_dev = vwap * 0.01  # Default 1% if not enough data

        # Update VWAPData
        data = self._update_vwap_data(symbol, close, vwap, std_dev, volume)

        return data

    def update_from_trade(self, symbol: str, price: float, volume: int) -> VWAPData:
        """Update VWAP from individual trade tick"""
        symbol = symbol.upper()

        if symbol not in self.cumulative_pv:
            self.cumulative_pv[symbol] = 0.0
            self.cumulative_volume[symbol] = 0.0
            self.price_history[symbol] = []

        # Update cumulative
        self.cumulative_pv[symbol] += price * volume
        self.cumulative_volume[symbol] += volume
        self.price_history[symbol].append(price)

        # Keep recent
        if len(self.price_history[symbol]) > 500:
            self.price_history[symbol] = self.price_history[symbol][-500:]

        # Calculate VWAP
        if self.cumulative_volume[symbol] > 0:
            vwap = self.cumulative_pv[symbol] / self.cumulative_volume[symbol]
        else:
            vwap = price

        # Std dev
        if len(self.price_history[symbol]) >= 20:
            std_dev = np.std(self.price_history[symbol][-100:])
        else:
            std_dev = vwap * 0.01

        return self._update_vwap_data(symbol, price, vwap, std_dev, volume)

    def _update_vwap_data(
        self, symbol: str, price: float, vwap: float, std_dev: float, volume: int
    ) -> VWAPData:
        """Update VWAPData with calculated values"""

        # Get or create data
        if symbol not in self.vwap_data:
            self.vwap_data[symbol] = VWAPData(symbol=symbol)

        data = self.vwap_data[symbol]
        prev_price = self.prev_prices.get(symbol, price)
        prev_vwap = self.prev_vwaps.get(symbol, vwap)

        # Update basic values
        data.vwap = vwap
        data.current_price = price
        data.last_update = datetime.now().isoformat()

        # Calculate distance from VWAP
        if vwap > 0:
            data.distance_pct = ((price - vwap) / vwap) * 100

        # Determine position
        if data.distance_pct > self.extended_threshold_pct:
            data.position = VWAPPosition.FAR_ABOVE
        elif data.distance_pct > self.at_vwap_threshold_pct:
            data.position = VWAPPosition.ABOVE
        elif data.distance_pct > -self.at_vwap_threshold_pct:
            data.position = VWAPPosition.AT_VWAP
        elif data.distance_pct > -self.extended_threshold_pct:
            data.position = VWAPPosition.BELOW
        else:
            data.position = VWAPPosition.FAR_BELOW

        # Calculate bands
        data.upper_band_1 = vwap + std_dev
        data.upper_band_2 = vwap + (2 * std_dev)
        data.lower_band_1 = vwap - std_dev
        data.lower_band_2 = vwap - (2 * std_dev)

        # Detect crossovers
        data.crossed_above = prev_price <= prev_vwap and price > vwap
        data.crossed_below = prev_price >= prev_vwap and price < vwap

        if data.crossed_above or data.crossed_below:
            data.candles_since_cross = 0
        else:
            data.candles_since_cross += 1

        # Volume analysis
        data.volume_at_vwap = volume if data.position == VWAPPosition.AT_VWAP else 0
        if len(self.price_history.get(symbol, [])) >= 20:
            data.avg_volume = self.cumulative_volume[symbol] / len(
                self.price_history[symbol]
            )
            if data.avg_volume > 0:
                data.relative_volume = volume / data.avg_volume

        # Determine signal
        data.signal = self._calculate_signal(data, volume)

        # Entry validation
        data.entry_valid = self._is_entry_valid(data)

        # Calculate stop price
        data.stop_price = self._calculate_stop(data)

        # Store previous values
        self.prev_prices[symbol] = price
        self.prev_vwaps[symbol] = vwap

        return data

    def _calculate_signal(self, data: VWAPData, volume: int) -> VWAPSignal:
        """Calculate VWAP-based trading signal"""

        # Just crossed above with volume = strong buy
        if data.crossed_above and volume >= self.min_volume_for_signal:
            return VWAPSignal.STRONG_BUY

        # Above VWAP and holding
        if data.position == VWAPPosition.ABOVE:
            return VWAPSignal.BUY_SUPPORTED

        # At VWAP - neutral
        if data.position == VWAPPosition.AT_VWAP:
            return VWAPSignal.NEUTRAL

        # Just crossed below = sell
        if data.crossed_below:
            return VWAPSignal.SELL

        # Below VWAP
        if data.position in [VWAPPosition.BELOW, VWAPPosition.FAR_BELOW]:
            return VWAPSignal.AVOID

        # Extended above = risky
        if data.position == VWAPPosition.FAR_ABOVE:
            return VWAPSignal.SELL_WARNING  # Don't chase extended moves

        return VWAPSignal.NEUTRAL

    def _is_entry_valid(self, data: VWAPData) -> bool:
        """
        Check if entry is valid based on VWAP.

        Ross Cameron rules:
        - Must be ABOVE VWAP
        - Not too extended (>3% is risky)
        - Ideally near VWAP with volume support
        """
        # Must be above VWAP
        if data.position in [VWAPPosition.BELOW, VWAPPosition.FAR_BELOW]:
            return False

        # Extended is risky but not invalid
        if data.position == VWAPPosition.FAR_ABOVE:
            return False  # Too risky, wait for pullback

        # ABOVE or AT_VWAP is valid
        return data.position in [VWAPPosition.ABOVE, VWAPPosition.AT_VWAP]

    def _calculate_stop(self, data: VWAPData) -> float:
        """Calculate VWAP-based stop loss"""
        # Stop just below VWAP (0.3% buffer)
        return data.vwap * 0.997

    def get_vwap(self, symbol: str) -> Optional[VWAPData]:
        """Get current VWAP data for symbol"""
        return self.vwap_data.get(symbol.upper())

    def is_above_vwap(self, symbol: str) -> bool:
        """Quick check if price is above VWAP"""
        data = self.get_vwap(symbol)
        if not data:
            return False
        return data.position in [VWAPPosition.ABOVE, VWAPPosition.FAR_ABOVE]

    def is_entry_valid(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if entry is valid for symbol.
        Returns (valid, reason)
        """
        data = self.get_vwap(symbol)
        if not data:
            return False, "No VWAP data"

        if data.position == VWAPPosition.FAR_BELOW:
            return False, f"Price far below VWAP ({data.distance_pct:.1f}%)"

        if data.position == VWAPPosition.BELOW:
            return False, f"Price below VWAP ({data.distance_pct:.1f}%)"

        if data.position == VWAPPosition.FAR_ABOVE:
            return (
                False,
                f"Price extended above VWAP ({data.distance_pct:.1f}%) - wait for pullback",
            )

        if data.position == VWAPPosition.AT_VWAP:
            return True, "At VWAP - ideal entry zone"

        if data.position == VWAPPosition.ABOVE:
            return True, f"Above VWAP ({data.distance_pct:.1f}%) - entry valid"

        return False, "Unknown position"

    def create_trailing_stop(self, symbol: str, entry_price: float) -> VWAPTrailingStop:
        """Create a VWAP-based trailing stop for a new position"""
        symbol = symbol.upper()
        data = self.get_vwap(symbol)

        vwap = data.vwap if data else entry_price * 0.99

        stop = VWAPTrailingStop(
            symbol=symbol,
            entry_price=entry_price,
            entry_vwap=vwap,
            current_vwap=vwap,
            current_price=entry_price,
            initial_stop=vwap * 0.997,  # 0.3% below VWAP
            current_stop=vwap * 0.997,
            max_price=entry_price,
        )

        self.trailing_stops[symbol] = stop
        logger.info(
            f"VWAP trailing stop created: {symbol} entry=${entry_price:.2f}, stop=${stop.initial_stop:.2f}"
        )

        return stop

    def update_trailing_stop(self, symbol: str, price: float) -> Tuple[bool, str]:
        """
        Update trailing stop for symbol.
        Returns (should_exit, reason)
        """
        symbol = symbol.upper()

        if symbol not in self.trailing_stops:
            return False, "No trailing stop"

        stop = self.trailing_stops[symbol]
        data = self.get_vwap(symbol)
        vwap = data.vwap if data else stop.current_vwap

        return stop.update(price, vwap)

    def remove_trailing_stop(self, symbol: str):
        """Remove trailing stop for symbol"""
        symbol = symbol.upper()
        if symbol in self.trailing_stops:
            del self.trailing_stops[symbol]

    def load_from_candles(self, symbol: str, candles: List[Dict]) -> VWAPData:
        """Load VWAP from historical candles (for initialization)"""
        symbol = symbol.upper()
        self.reset_daily(symbol)

        for candle in candles:
            self.update_from_candle(symbol, candle)

        return self.vwap_data.get(symbol, VWAPData(symbol=symbol))

    def get_all_data(self) -> Dict[str, Dict]:
        """Get all VWAP data"""
        return {sym: data.to_dict() for sym, data in self.vwap_data.items()}


# Singleton instance
_vwap_manager: Optional[VWAPManager] = None


def get_vwap_manager() -> VWAPManager:
    """Get singleton VWAP manager"""
    global _vwap_manager
    if _vwap_manager is None:
        _vwap_manager = VWAPManager()
    return _vwap_manager


# Convenience functions
def check_vwap_entry(symbol: str) -> Tuple[bool, str]:
    """Check if entry is valid based on VWAP"""
    return get_vwap_manager().is_entry_valid(symbol)


def get_vwap_stop(symbol: str, entry_price: float) -> float:
    """Get VWAP-based stop price"""
    manager = get_vwap_manager()
    data = manager.get_vwap(symbol)
    if data:
        return data.stop_price
    return entry_price * 0.97  # Fallback 3% stop
