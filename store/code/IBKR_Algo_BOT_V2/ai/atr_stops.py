"""
ATR-Based Dynamic Stop Loss Calculator
======================================
Calculates volatility-adjusted stop losses using Average True Range (ATR).

Why ATR stops?
- Fixed % stops don't account for stock volatility
- A 3% stop on a calm stock = good | 3% on volatile stock = too tight
- ATR adapts to each stock's natural movement range

Usage:
    from ai.atr_stops import get_dynamic_stops
    stops = await get_dynamic_stops("AAPL", entry_price=180.50)
    print(f"Stop: ${stops['stop_price']:.2f}, Target: ${stops['target_price']:.2f}")
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DynamicStops:
    """Dynamic stop loss and target levels"""
    symbol: str
    entry_price: float

    # ATR data
    atr: float  # Average True Range (dollar amount)
    atr_percent: float  # ATR as percentage of price

    # Stop levels
    stop_price: float
    stop_distance: float  # Dollars from entry
    stop_percent: float  # Percentage from entry

    # Target levels
    target_price: float
    target_distance: float
    target_percent: float

    # Risk/Reward
    risk_reward_ratio: float

    # Trailing stop
    trailing_atr_multiplier: float
    trailing_stop_distance: float

    # Metadata
    atr_period: int
    atr_multiplier: float
    volatility_regime: str  # LOW, NORMAL, HIGH, EXTREME
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'entry_price': round(self.entry_price, 2),
            'atr': round(self.atr, 3),
            'atr_percent': round(self.atr_percent, 2),
            'stop_price': round(self.stop_price, 2),
            'stop_distance': round(self.stop_distance, 3),
            'stop_percent': round(self.stop_percent, 2),
            'target_price': round(self.target_price, 2),
            'target_distance': round(self.target_distance, 3),
            'target_percent': round(self.target_percent, 2),
            'risk_reward_ratio': round(self.risk_reward_ratio, 2),
            'trailing_atr_multiplier': self.trailing_atr_multiplier,
            'trailing_stop_distance': round(self.trailing_stop_distance, 3),
            'atr_period': self.atr_period,
            'atr_multiplier': self.atr_multiplier,
            'volatility_regime': self.volatility_regime,
            'timestamp': self.timestamp
        }


class ATRStopCalculator:
    """
    Calculates dynamic stops based on ATR.

    Default settings:
    - ATR Period: 14 bars
    - Stop: 1.5x ATR below entry
    - Target: 2.0x ATR above entry (1.33 R:R)
    - Trailing: 1.0x ATR

    Volatility adjustments:
    - LOW vol: Tighter stops (1.0x ATR)
    - HIGH vol: Wider stops (2.0x ATR)
    """

    def __init__(self):
        self.atr_period = 14
        self.stop_multiplier = 1.5  # Stop at 1.5x ATR
        self.target_multiplier = 2.0  # Target at 2.0x ATR
        self.trailing_multiplier = 1.0  # Trail at 1.0x ATR

        # Min/max stop percentages (safety bounds)
        self.min_stop_percent = 1.0  # Never less than 1%
        self.max_stop_percent = 8.0  # Never more than 8%

        # Cache for ATR values
        self._atr_cache: Dict[str, Tuple[float, datetime]] = {}
        self._cache_ttl = 300  # 5 minutes

    def calculate_atr(self, symbol: str, period: str = "5d", interval: str = "5m") -> Optional[float]:
        """
        Calculate ATR for a symbol.

        Args:
            symbol: Stock ticker
            period: Data period (5d, 1mo, etc.)
            interval: Bar interval (1m, 5m, 15m, etc.)

        Returns:
            ATR value in dollars, or None if failed
        """
        # Check cache
        if symbol in self._atr_cache:
            cached_atr, cached_time = self._atr_cache[symbol]
            if (datetime.now() - cached_time).total_seconds() < self._cache_ttl:
                return cached_atr

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty or len(df) < self.atr_period + 1:
                logger.warning(f"Insufficient data for ATR calculation: {symbol}")
                return None

            # Calculate True Range
            df['prev_close'] = df['Close'].shift(1)
            df['tr1'] = df['High'] - df['Low']
            df['tr2'] = abs(df['High'] - df['prev_close'])
            df['tr3'] = abs(df['Low'] - df['prev_close'])
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

            # Calculate ATR (EMA of True Range)
            atr = df['true_range'].ewm(span=self.atr_period, adjust=False).mean().iloc[-1]

            # Cache result
            self._atr_cache[symbol] = (atr, datetime.now())

            return atr

        except Exception as e:
            logger.error(f"Error calculating ATR for {symbol}: {e}")
            return None

    def get_volatility_regime(self, atr_percent: float) -> str:
        """
        Classify volatility regime based on ATR%.

        Returns:
            LOW, NORMAL, HIGH, or EXTREME
        """
        if atr_percent < 1.0:
            return "LOW"
        elif atr_percent < 2.5:
            return "NORMAL"
        elif atr_percent < 5.0:
            return "HIGH"
        else:
            return "EXTREME"

    def adjust_multipliers(self, volatility_regime: str) -> Tuple[float, float, float]:
        """
        Adjust ATR multipliers based on volatility regime.

        Returns:
            (stop_mult, target_mult, trail_mult)
        """
        if volatility_regime == "LOW":
            # Tighter stops for calm stocks
            return (1.0, 1.5, 0.75)
        elif volatility_regime == "NORMAL":
            # Standard settings
            return (1.5, 2.0, 1.0)
        elif volatility_regime == "HIGH":
            # Wider stops for volatile stocks
            return (2.0, 3.0, 1.5)
        else:  # EXTREME
            # Very wide stops, conservative targets
            return (2.5, 3.0, 2.0)

    def calculate_stops(
        self,
        symbol: str,
        entry_price: float,
        atr: float = None
    ) -> Optional[DynamicStops]:
        """
        Calculate dynamic stop and target levels.

        Args:
            symbol: Stock ticker
            entry_price: Entry price
            atr: Pre-calculated ATR (optional)

        Returns:
            DynamicStops object with all levels
        """
        # Get ATR if not provided
        if atr is None:
            atr = self.calculate_atr(symbol)

        if atr is None or atr <= 0:
            # Fallback to fixed 3% if ATR fails
            logger.warning(f"ATR unavailable for {symbol}, using 3% fallback")
            atr = entry_price * 0.03

        # Calculate ATR as percentage
        atr_percent = (atr / entry_price) * 100

        # Get volatility regime
        volatility_regime = self.get_volatility_regime(atr_percent)

        # Adjust multipliers based on volatility
        stop_mult, target_mult, trail_mult = self.adjust_multipliers(volatility_regime)

        # Calculate stop distance
        stop_distance = atr * stop_mult
        stop_percent = (stop_distance / entry_price) * 100

        # Apply safety bounds
        if stop_percent < self.min_stop_percent:
            stop_percent = self.min_stop_percent
            stop_distance = entry_price * (stop_percent / 100)
        elif stop_percent > self.max_stop_percent:
            stop_percent = self.max_stop_percent
            stop_distance = entry_price * (stop_percent / 100)

        # Calculate target distance
        target_distance = atr * target_mult
        target_percent = (target_distance / entry_price) * 100

        # Calculate trailing stop distance
        trailing_distance = atr * trail_mult

        # Calculate prices
        stop_price = entry_price - stop_distance
        target_price = entry_price + target_distance

        # Risk/Reward ratio
        risk_reward = target_distance / stop_distance if stop_distance > 0 else 0

        return DynamicStops(
            symbol=symbol,
            entry_price=entry_price,
            atr=atr,
            atr_percent=atr_percent,
            stop_price=stop_price,
            stop_distance=stop_distance,
            stop_percent=stop_percent,
            target_price=target_price,
            target_distance=target_distance,
            target_percent=target_percent,
            risk_reward_ratio=risk_reward,
            trailing_atr_multiplier=trail_mult,
            trailing_stop_distance=trailing_distance,
            atr_period=self.atr_period,
            atr_multiplier=stop_mult,
            volatility_regime=volatility_regime,
            timestamp=datetime.now().isoformat()
        )

    def get_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        high_price: float,
        atr: float,
        trail_multiplier: float = None
    ) -> float:
        """
        Calculate current trailing stop level.

        Args:
            entry_price: Original entry price
            current_price: Current market price
            high_price: Highest price since entry
            atr: ATR value
            trail_multiplier: ATR multiplier for trail

        Returns:
            Current trailing stop price
        """
        if trail_multiplier is None:
            trail_multiplier = self.trailing_multiplier

        trail_distance = atr * trail_multiplier
        trailing_stop = high_price - trail_distance

        # Never trail below entry (lock in breakeven at minimum)
        initial_stop = entry_price - (atr * self.stop_multiplier)

        return max(trailing_stop, initial_stop)


# Singleton instance
_calculator = None


def get_atr_calculator() -> ATRStopCalculator:
    """Get singleton ATR calculator instance"""
    global _calculator
    if _calculator is None:
        _calculator = ATRStopCalculator()
    return _calculator


async def get_dynamic_stops(symbol: str, entry_price: float) -> Dict:
    """
    Get dynamic stop/target levels for a symbol.

    Args:
        symbol: Stock ticker
        entry_price: Entry price

    Returns:
        Dict with stop_price, target_price, and metadata
    """
    calculator = get_atr_calculator()
    stops = calculator.calculate_stops(symbol, entry_price)

    if stops:
        return stops.to_dict()
    else:
        # Fallback
        return {
            'symbol': symbol,
            'entry_price': entry_price,
            'stop_price': entry_price * 0.97,  # 3% stop
            'stop_percent': 3.0,
            'target_price': entry_price * 1.03,  # 3% target
            'target_percent': 3.0,
            'error': 'ATR calculation failed, using 3% fallback'
        }


if __name__ == "__main__":
    import asyncio

    async def test():
        test_symbols = [
            ("AAPL", 180.50),   # Low volatility blue chip
            ("TSLA", 250.00),   # High volatility
            ("SPY", 450.00),    # Very low volatility ETF
            ("SOUN", 5.25),     # Penny stock, high vol
        ]

        print("\nDynamic ATR Stop Calculator Test")
        print("=" * 70)

        for symbol, price in test_symbols:
            stops = await get_dynamic_stops(symbol, price)
            print(f"\n{symbol} @ ${price:.2f}")
            print(f"  ATR: ${stops.get('atr', 0):.3f} ({stops.get('atr_percent', 0):.1f}%)")
            print(f"  Volatility: {stops.get('volatility_regime', 'N/A')}")
            print(f"  Stop: ${stops.get('stop_price', 0):.2f} (-{stops.get('stop_percent', 0):.1f}%)")
            print(f"  Target: ${stops.get('target_price', 0):.2f} (+{stops.get('target_percent', 0):.1f}%)")
            print(f"  R:R Ratio: {stops.get('risk_reward_ratio', 0):.2f}")
            print(f"  Trail Distance: ${stops.get('trailing_stop_distance', 0):.3f}")

    asyncio.run(test())
