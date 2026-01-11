"""
EMA (Exponential Moving Average) Indicator
==========================================
Used by ATS + 9 EMA Sniper Strategy for pullback detection.

Key Concept:
- 9 EMA acts as dynamic support in strong trends
- Price pullbacks to 9 EMA = potential entry zone
- Break below 9 EMA = trend exhaustion
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from collections import deque

logger = logging.getLogger(__name__)


def calculate_ema(prices: List[float], period: int) -> float:
    """
    Calculate Exponential Moving Average.

    Args:
        prices: List of prices (oldest to newest)
        period: EMA period (e.g., 9 for 9 EMA)

    Returns:
        Current EMA value
    """
    if not prices or len(prices) < period:
        return prices[-1] if prices else 0.0

    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period  # SMA for initial value

    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))

    return ema


@dataclass
class EMAState:
    """Current state of EMA relative to price"""
    symbol: str
    ema_9: float = 0.0
    ema_20: float = 0.0
    current_price: float = 0.0

    # Position relative to EMA
    price_above_ema9: bool = False
    price_at_ema9: bool = False  # Within 0.2% of EMA
    price_below_ema9: bool = False

    # Trend indicators
    ema9_above_ema20: bool = False  # Bullish trend
    ema_slope_positive: bool = False  # EMA trending up

    # Pullback detection
    distance_from_ema9_pct: float = 0.0
    is_pullback_zone: bool = False  # Price near but above EMA9

    # History for slope calculation
    ema9_history: List[float] = field(default_factory=list)

    last_update: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "ema_9": round(self.ema_9, 4),
            "ema_20": round(self.ema_20, 4),
            "current_price": round(self.current_price, 4),
            "price_above_ema9": self.price_above_ema9,
            "price_at_ema9": self.price_at_ema9,
            "price_below_ema9": self.price_below_ema9,
            "ema9_above_ema20": self.ema9_above_ema20,
            "ema_slope_positive": self.ema_slope_positive,
            "distance_from_ema9_pct": round(self.distance_from_ema9_pct, 2),
            "is_pullback_zone": self.is_pullback_zone,
            "last_update": self.last_update.isoformat() if self.last_update else None
        }


class EMATracker:
    """
    Tracks EMA values and detects pullback setups for 9 EMA Sniper strategy.

    Key behaviors:
    - Maintains rolling price history for EMA calculation
    - Detects when price pulls back to 9 EMA (entry zone)
    - Identifies trend direction via EMA slope and EMA9 vs EMA20
    """

    # Pullback zone: price within this % above EMA9
    PULLBACK_ZONE_PCT = 0.5  # 0.5% above EMA = pullback zone

    # At EMA threshold
    AT_EMA_THRESHOLD_PCT = 0.2  # Within 0.2% = "at" EMA

    # Slope calculation periods
    SLOPE_LOOKBACK = 3  # Compare current vs 3 bars ago

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self._states: Dict[str, EMAState] = {}
        self._price_history: Dict[str, deque] = {}
        self._ema9_history: Dict[str, deque] = {}

    def update(self, symbol: str, price: float, timestamp: Optional[datetime] = None) -> EMAState:
        """
        Update EMA calculations with new price.

        Args:
            symbol: Stock symbol
            price: Current price
            timestamp: Optional timestamp

        Returns:
            Updated EMAState
        """
        symbol = symbol.upper()

        # Initialize if needed
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self.max_history)
            self._ema9_history[symbol] = deque(maxlen=20)
            self._states[symbol] = EMAState(symbol=symbol)

        # Add price to history
        self._price_history[symbol].append(price)
        prices = list(self._price_history[symbol])

        # Calculate EMAs
        ema_9 = calculate_ema(prices, 9) if len(prices) >= 9 else price
        ema_20 = calculate_ema(prices, 20) if len(prices) >= 20 else price

        # Track EMA9 history for slope
        self._ema9_history[symbol].append(ema_9)

        # Update state
        state = self._states[symbol]
        state.ema_9 = ema_9
        state.ema_20 = ema_20
        state.current_price = price
        state.last_update = timestamp or datetime.now()

        # Position relative to EMA9
        if ema_9 > 0:
            distance_pct = ((price - ema_9) / ema_9) * 100
            state.distance_from_ema9_pct = distance_pct

            state.price_above_ema9 = distance_pct > self.AT_EMA_THRESHOLD_PCT
            state.price_at_ema9 = abs(distance_pct) <= self.AT_EMA_THRESHOLD_PCT
            state.price_below_ema9 = distance_pct < -self.AT_EMA_THRESHOLD_PCT

            # Pullback zone: just above EMA (0 to 0.5%)
            state.is_pullback_zone = 0 <= distance_pct <= self.PULLBACK_ZONE_PCT

        # Trend indicators
        state.ema9_above_ema20 = ema_9 > ema_20

        # EMA slope (is it trending up?)
        ema9_list = list(self._ema9_history[symbol])
        if len(ema9_list) >= self.SLOPE_LOOKBACK:
            state.ema_slope_positive = ema9_list[-1] > ema9_list[-self.SLOPE_LOOKBACK]

        state.ema9_history = ema9_list[-10:]  # Keep last 10 for reference

        return state

    def get_state(self, symbol: str) -> Optional[EMAState]:
        """Get current EMA state for symbol"""
        return self._states.get(symbol.upper())

    def is_in_pullback_zone(self, symbol: str) -> bool:
        """Check if symbol is in pullback zone (just above 9 EMA)"""
        state = self.get_state(symbol)
        if not state:
            return False
        return state.is_pullback_zone and state.ema9_above_ema20

    def is_bullish_structure(self, symbol: str) -> bool:
        """Check if symbol has bullish EMA structure"""
        state = self.get_state(symbol)
        if not state:
            return False
        return state.ema9_above_ema20 and state.ema_slope_positive

    def get_distance_from_ema9(self, symbol: str) -> float:
        """Get distance from 9 EMA as percentage"""
        state = self.get_state(symbol)
        return state.distance_from_ema9_pct if state else 0.0

    def clear(self, symbol: str = None):
        """Clear history for symbol or all symbols"""
        if symbol:
            symbol = symbol.upper()
            self._price_history.pop(symbol, None)
            self._ema9_history.pop(symbol, None)
            self._states.pop(symbol, None)
        else:
            self._price_history.clear()
            self._ema9_history.clear()
            self._states.clear()

    def get_all_states(self) -> Dict[str, Dict]:
        """Get all EMA states as dict"""
        return {sym: state.to_dict() for sym, state in self._states.items()}


# Singleton instance
_ema_tracker: Optional[EMATracker] = None


def get_ema_tracker() -> EMATracker:
    """Get singleton EMA tracker instance"""
    global _ema_tracker
    if _ema_tracker is None:
        _ema_tracker = EMATracker()
    return _ema_tracker
