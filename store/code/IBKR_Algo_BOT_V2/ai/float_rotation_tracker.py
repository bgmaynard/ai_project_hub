"""
Float Rotation Tracker
======================
Track when cumulative volume exceeds float (float rotation = explosive moves).

Ross Cameron Rule:
- Low float stocks (<20M shares) can make explosive moves
- When volume exceeds float, stock is "rotating" - every share has changed hands
- Multiple float rotations (2x, 3x) signal extreme momentum
- Float rotation is a leading indicator of continued momentum

Example:
- Stock has 5M float
- Morning volume hits 10M = 2x float rotation
- This means every share has theoretically changed hands TWICE
- Indicates extreme interest and potential for continued momentum
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RotationLevel(Enum):
    """Float rotation levels"""

    NONE = "NONE"  # < 0.25x
    WARMING = "WARMING"  # 0.25x - 0.5x
    ACTIVE = "ACTIVE"  # 0.5x - 1.0x
    ROTATING = "ROTATING"  # 1.0x - 2.0x
    HOT = "HOT"  # 2.0x - 3.0x
    EXTREME = "EXTREME"  # 3.0x+


@dataclass
class FloatData:
    """Float and volume data for a symbol"""

    symbol: str
    float_shares: int = 0  # Free float in shares
    shares_outstanding: int = 0  # Total shares

    # Daily volume tracking
    cumulative_volume: int = 0  # Total volume today
    premarket_volume: int = 0  # Pre-market volume
    regular_volume: int = 0  # Regular hours volume

    # Rotation calculations
    rotation_ratio: float = 0.0  # cumulative_volume / float_shares
    rotation_level: RotationLevel = RotationLevel.NONE
    rotations_completed: int = 0  # Number of full rotations (1x, 2x, 3x...)

    # Thresholds hit
    thresholds_hit: List[float] = field(default_factory=list)  # [0.5, 1.0, 2.0, ...]

    # Timing
    last_update: str = ""
    date: str = ""
    time_to_first_rotation: Optional[float] = None  # Minutes to hit 1x

    # Quality indicators
    is_low_float: bool = False  # < 20M shares
    is_micro_float: bool = False  # < 5M shares
    is_nano_float: bool = False  # < 1M shares

    # Average metrics
    avg_daily_volume: int = 0
    relative_volume: float = 0.0  # Today vs average

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "float_shares": self.float_shares,
            "float_millions": (
                round(self.float_shares / 1_000_000, 2) if self.float_shares else 0
            ),
            "cumulative_volume": self.cumulative_volume,
            "volume_millions": (
                round(self.cumulative_volume / 1_000_000, 2)
                if self.cumulative_volume
                else 0
            ),
            "rotation_ratio": round(self.rotation_ratio, 2),
            "rotation_level": self.rotation_level.value,
            "rotations_completed": self.rotations_completed,
            "thresholds_hit": self.thresholds_hit,
            "is_low_float": self.is_low_float,
            "is_micro_float": self.is_micro_float,
            "is_nano_float": self.is_nano_float,
            "relative_volume": round(self.relative_volume, 2),
            "time_to_first_rotation": self.time_to_first_rotation,
            "last_update": self.last_update,
            "date": self.date,
        }


@dataclass
class RotationAlert:
    """Alert when rotation threshold is crossed"""

    symbol: str
    threshold: float  # 0.5, 1.0, 2.0, etc.
    rotation_ratio: float  # Actual ratio when crossed
    volume: int
    float_shares: int
    timestamp: str
    is_low_float: bool

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "threshold": self.threshold,
            "rotation_ratio": round(self.rotation_ratio, 2),
            "volume_millions": round(self.volume / 1_000_000, 2),
            "float_millions": round(self.float_shares / 1_000_000, 2),
            "timestamp": self.timestamp,
            "is_low_float": self.is_low_float,
            "message": f"{self.symbol} hit {self.threshold}x float rotation ({self.rotation_ratio:.1f}x)",
        }


class FloatRotationTracker:
    """
    Track float rotation for momentum stocks.

    Ross Cameron emphasizes:
    - Low float (<20M) stocks can make huge moves
    - Float rotation (volume > float) indicates extreme momentum
    - Multiple rotations = parabolic potential
    """

    # Rotation thresholds to alert on
    ROTATION_THRESHOLDS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    # Float classifications (in shares)
    NANO_FLOAT_MAX = 1_000_000  # < 1M = nano float
    MICRO_FLOAT_MAX = 5_000_000  # < 5M = micro float
    LOW_FLOAT_MAX = 20_000_000  # < 20M = low float

    def __init__(self):
        # Float data per symbol
        self.float_data: Dict[str, FloatData] = {}

        # Float shares cache (from fundamental analysis)
        self.float_cache: Dict[str, int] = {}
        self.avg_volume_cache: Dict[str, int] = {}

        # Alerts history
        self.alerts: List[RotationAlert] = []
        self.max_alerts = 500

        # Callbacks for alerts
        self.on_rotation_alert: Optional[callable] = None

        # Tracking
        self.today = date.today().isoformat()
        self.market_open_time: Optional[datetime] = None

    def reset_daily(self, symbol: str = None):
        """Reset daily tracking (call at market open)"""
        self.today = date.today().isoformat()
        self.market_open_time = datetime.now()

        if symbol:
            symbols = [symbol]
        else:
            symbols = list(self.float_data.keys())

        for sym in symbols:
            if sym in self.float_data:
                data = self.float_data[sym]
                data.cumulative_volume = 0
                data.premarket_volume = 0
                data.regular_volume = 0
                data.rotation_ratio = 0.0
                data.rotation_level = RotationLevel.NONE
                data.rotations_completed = 0
                data.thresholds_hit = []
                data.time_to_first_rotation = None
                data.date = self.today

        logger.info(f"Float rotation tracker reset for {len(symbols)} symbols")

    async def load_float_data(self, symbol: str) -> Optional[int]:
        """Load float data from fundamental analysis"""
        symbol = symbol.upper()

        # Check cache first
        if symbol in self.float_cache:
            return self.float_cache[symbol]

        try:
            from ai.fundamental_analysis import get_fundamentals

            fundamentals = await get_fundamentals(symbol)

            if fundamentals and fundamentals.float_shares:
                self.float_cache[symbol] = int(fundamentals.float_shares)

                if fundamentals.avg_volume:
                    self.avg_volume_cache[symbol] = int(fundamentals.avg_volume)

                logger.info(
                    f"Loaded float for {symbol}: {fundamentals.float_shares:,.0f} shares"
                )
                return int(fundamentals.float_shares)

        except Exception as e:
            logger.warning(f"Failed to load float for {symbol}: {e}")

        return None

    def set_float(self, symbol: str, float_shares: int, avg_volume: int = 0):
        """Manually set float data for a symbol"""
        symbol = symbol.upper()
        self.float_cache[symbol] = float_shares
        if avg_volume:
            self.avg_volume_cache[symbol] = avg_volume

        # Update existing data if present
        if symbol in self.float_data:
            data = self.float_data[symbol]
            data.float_shares = float_shares
            data.avg_daily_volume = avg_volume
            self._classify_float(data)
            self._update_rotation(data)

    def _classify_float(self, data: FloatData):
        """Classify float size"""
        if data.float_shares <= self.NANO_FLOAT_MAX:
            data.is_nano_float = True
            data.is_micro_float = True
            data.is_low_float = True
        elif data.float_shares <= self.MICRO_FLOAT_MAX:
            data.is_nano_float = False
            data.is_micro_float = True
            data.is_low_float = True
        elif data.float_shares <= self.LOW_FLOAT_MAX:
            data.is_nano_float = False
            data.is_micro_float = False
            data.is_low_float = True
        else:
            data.is_nano_float = False
            data.is_micro_float = False
            data.is_low_float = False

    def _get_rotation_level(self, ratio: float) -> RotationLevel:
        """Determine rotation level from ratio"""
        if ratio < 0.25:
            return RotationLevel.NONE
        elif ratio < 0.5:
            return RotationLevel.WARMING
        elif ratio < 1.0:
            return RotationLevel.ACTIVE
        elif ratio < 2.0:
            return RotationLevel.ROTATING
        elif ratio < 3.0:
            return RotationLevel.HOT
        else:
            return RotationLevel.EXTREME

    def _update_rotation(self, data: FloatData):
        """Update rotation calculations"""
        if data.float_shares <= 0:
            return

        # Calculate rotation ratio
        data.rotation_ratio = data.cumulative_volume / data.float_shares
        data.rotation_level = self._get_rotation_level(data.rotation_ratio)
        data.rotations_completed = int(data.rotation_ratio)

        # Calculate relative volume
        if data.avg_daily_volume > 0:
            data.relative_volume = data.cumulative_volume / data.avg_daily_volume

    def _check_thresholds(self, data: FloatData) -> List[RotationAlert]:
        """Check if new rotation thresholds were crossed"""
        alerts = []

        for threshold in self.ROTATION_THRESHOLDS:
            if (
                threshold not in data.thresholds_hit
                and data.rotation_ratio >= threshold
            ):
                data.thresholds_hit.append(threshold)

                # Track time to first rotation
                if threshold == 1.0 and data.time_to_first_rotation is None:
                    if self.market_open_time:
                        minutes = (
                            datetime.now() - self.market_open_time
                        ).total_seconds() / 60
                        data.time_to_first_rotation = round(minutes, 1)

                alert = RotationAlert(
                    symbol=data.symbol,
                    threshold=threshold,
                    rotation_ratio=data.rotation_ratio,
                    volume=data.cumulative_volume,
                    float_shares=data.float_shares,
                    timestamp=datetime.now().isoformat(),
                    is_low_float=data.is_low_float,
                )
                alerts.append(alert)

                # Log alert
                level_str = "LOW FLOAT " if data.is_low_float else ""
                logger.info(
                    f"FLOAT ROTATION: {level_str}{data.symbol} hit {threshold}x "
                    f"({data.rotation_ratio:.2f}x actual, "
                    f"{data.cumulative_volume:,} vol / {data.float_shares:,} float)"
                )

        return alerts

    def update_volume(
        self, symbol: str, volume: int, is_premarket: bool = False
    ) -> Optional[FloatData]:
        """
        Update volume for a symbol.

        Args:
            symbol: Stock symbol
            volume: Trade volume to add
            is_premarket: Whether this is pre-market volume

        Returns:
            Updated FloatData or None if no float data
        """
        symbol = symbol.upper()

        # Initialize if needed
        if symbol not in self.float_data:
            float_shares = self.float_cache.get(symbol, 0)
            avg_vol = self.avg_volume_cache.get(symbol, 0)

            if float_shares <= 0:
                # No float data yet, can't track
                return None

            self.float_data[symbol] = FloatData(
                symbol=symbol,
                float_shares=float_shares,
                avg_daily_volume=avg_vol,
                date=self.today,
            )
            self._classify_float(self.float_data[symbol])

        data = self.float_data[symbol]

        # Check for new day
        if data.date != self.today:
            self.reset_daily(symbol)

        # Add volume
        data.cumulative_volume += volume
        if is_premarket:
            data.premarket_volume += volume
        else:
            data.regular_volume += volume

        data.last_update = datetime.now().isoformat()

        # Update rotation calculations
        self._update_rotation(data)

        # Check thresholds
        alerts = self._check_thresholds(data)

        # Store alerts
        for alert in alerts:
            self.alerts.append(alert)
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts :]

            # Fire callback if registered
            if self.on_rotation_alert:
                try:
                    self.on_rotation_alert(alert)
                except Exception as e:
                    logger.error(f"Rotation alert callback error: {e}")

        return data

    def get_float_data(self, symbol: str) -> Optional[FloatData]:
        """Get float rotation data for a symbol"""
        return self.float_data.get(symbol.upper())

    def get_rotation_ratio(self, symbol: str) -> float:
        """Get current rotation ratio for symbol"""
        data = self.get_float_data(symbol)
        return data.rotation_ratio if data else 0.0

    def is_rotating(self, symbol: str) -> bool:
        """Check if symbol has completed at least 1x float rotation"""
        data = self.get_float_data(symbol)
        return data.rotation_ratio >= 1.0 if data else False

    def get_rotation_boost(self, symbol: str) -> float:
        """
        Get confidence boost based on rotation level.

        Used by HFT Scalper to increase confidence on rotating stocks.

        Returns:
            Boost factor (0.0 to 0.3)
        """
        data = self.get_float_data(symbol)
        if not data:
            return 0.0

        # Base boost on rotation level
        boost = 0.0

        if data.rotation_level == RotationLevel.ACTIVE:
            boost = 0.05
        elif data.rotation_level == RotationLevel.ROTATING:
            boost = 0.10
        elif data.rotation_level == RotationLevel.HOT:
            boost = 0.20
        elif data.rotation_level == RotationLevel.EXTREME:
            boost = 0.30

        # Extra boost for low float
        if data.is_micro_float:
            boost += 0.05
        elif data.is_nano_float:
            boost += 0.10

        return min(boost, 0.40)  # Cap at 40% boost

    def get_all_rotating(self, min_rotation: float = 1.0) -> List[FloatData]:
        """Get all symbols with rotation >= threshold"""
        rotating = []
        for data in self.float_data.values():
            if data.rotation_ratio >= min_rotation:
                rotating.append(data)

        # Sort by rotation ratio descending
        rotating.sort(key=lambda x: x.rotation_ratio, reverse=True)
        return rotating

    def get_low_float_movers(self) -> List[FloatData]:
        """Get low float stocks with significant volume"""
        movers = []
        for data in self.float_data.values():
            if data.is_low_float and data.rotation_ratio >= 0.25:
                movers.append(data)

        movers.sort(key=lambda x: x.rotation_ratio, reverse=True)
        return movers

    def get_recent_alerts(self, limit: int = 20) -> List[RotationAlert]:
        """Get recent rotation alerts"""
        return self.alerts[-limit:][::-1]

    def get_status(self) -> Dict:
        """Get tracker status"""
        rotating = [d for d in self.float_data.values() if d.rotation_ratio >= 1.0]
        active = [d for d in self.float_data.values() if 0.5 <= d.rotation_ratio < 1.0]
        low_float = [d for d in self.float_data.values() if d.is_low_float]

        return {
            "symbols_tracked": len(self.float_data),
            "symbols_rotating": len(rotating),
            "symbols_active": len(active),
            "low_float_count": len(low_float),
            "total_alerts": len(self.alerts),
            "float_cache_size": len(self.float_cache),
            "date": self.today,
            "top_rotators": [
                {"symbol": d.symbol, "rotation": round(d.rotation_ratio, 2)}
                for d in sorted(
                    self.float_data.values(),
                    key=lambda x: x.rotation_ratio,
                    reverse=True,
                )[:5]
            ],
        }

    def get_all_data(self) -> Dict[str, Dict]:
        """Get all float rotation data"""
        return {sym: data.to_dict() for sym, data in self.float_data.items()}


# Singleton instance
_float_tracker: Optional[FloatRotationTracker] = None


def get_float_tracker() -> FloatRotationTracker:
    """Get singleton float rotation tracker"""
    global _float_tracker
    if _float_tracker is None:
        _float_tracker = FloatRotationTracker()
    return _float_tracker


# Convenience functions
def update_float_volume(
    symbol: str, volume: int, is_premarket: bool = False
) -> Optional[FloatData]:
    """Update volume for float tracking"""
    return get_float_tracker().update_volume(symbol, volume, is_premarket)


def get_rotation_ratio(symbol: str) -> float:
    """Get float rotation ratio"""
    return get_float_tracker().get_rotation_ratio(symbol)


def is_float_rotating(symbol: str) -> bool:
    """Check if stock is rotating (>= 1x float)"""
    return get_float_tracker().is_rotating(symbol)


def get_rotation_boost(symbol: str) -> float:
    """Get confidence boost from float rotation"""
    return get_float_tracker().get_rotation_boost(symbol)
