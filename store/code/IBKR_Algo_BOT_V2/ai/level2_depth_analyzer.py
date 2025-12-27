"""
Level 2 Depth Analyzer
======================
Analyze order book depth for trading signals.

Ross Cameron Tape Reading Concepts:
1. BID WALLS - Large buy orders = support levels
2. ASK WALLS - Large sell orders = resistance levels
3. ABSORPTION - Wall getting eaten = breakout imminent
4. SPOOFING - Walls that appear/disappear = fake orders
5. IMBALANCE - More bids than asks = bullish pressure

Key Insights:
- Large walls often act as magnets (price gravitates to them)
- Walls getting absorbed signal strong momentum
- Sudden wall disappearance = spoofing/manipulation
- Consistent imbalance predicts direction
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class WallType(Enum):
    """Type of order wall"""
    BID_WALL = "BID_WALL"       # Large buy order (support)
    ASK_WALL = "ASK_WALL"       # Large sell order (resistance)


class WallStatus(Enum):
    """Status of a detected wall"""
    ACTIVE = "ACTIVE"           # Wall is present
    ABSORBING = "ABSORBING"     # Wall being eaten
    ABSORBED = "ABSORBED"       # Wall was fully eaten (breakout)
    DISAPPEARED = "DISAPPEARED" # Wall vanished (spoofing)
    REINFORCED = "REINFORCED"   # Wall got bigger


class DepthSignal(Enum):
    """Trading signals from depth analysis"""
    STRONG_BID = "STRONG_BID"           # Strong bid wall support
    WEAK_BID = "WEAK_BID"               # Bid wall weakening
    STRONG_ASK = "STRONG_ASK"           # Strong ask wall resistance
    WEAK_ASK = "WEAK_ASK"               # Ask wall weakening
    BID_ABSORPTION = "BID_ABSORPTION"   # Bids being eaten (bearish)
    ASK_ABSORPTION = "ASK_ABSORPTION"   # Asks being eaten (bullish breakout)
    BULLISH_IMBALANCE = "BULLISH_IMBALANCE"  # More bids than asks
    BEARISH_IMBALANCE = "BEARISH_IMBALANCE"  # More asks than bids
    SPOOFING_DETECTED = "SPOOFING_DETECTED"  # Fake wall detected
    NEUTRAL = "NEUTRAL"


@dataclass
class OrderWall:
    """A detected order wall"""
    symbol: str
    wall_type: WallType
    price: float
    size: int                   # Total shares at this level
    status: WallStatus = WallStatus.ACTIVE

    # Tracking
    initial_size: int = 0       # Size when first detected
    peak_size: int = 0          # Maximum size observed
    current_size: int = 0       # Current size

    # Absorption tracking
    absorbed_volume: int = 0    # Volume eaten from wall
    absorption_rate: float = 0  # Shares/second being absorbed

    # Timing
    first_seen: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0

    # Quality
    is_significant: bool = False    # Large enough to matter
    distance_from_price: float = 0  # % distance from current price

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "wall_type": self.wall_type.value,
            "price": round(self.price, 4),
            "size": self.size,
            "status": self.status.value,
            "initial_size": self.initial_size,
            "peak_size": self.peak_size,
            "absorbed_volume": self.absorbed_volume,
            "absorption_rate": round(self.absorption_rate, 1),
            "duration_seconds": round(self.duration_seconds, 1),
            "is_significant": self.is_significant,
            "distance_from_price": round(self.distance_from_price, 2)
        }


@dataclass
class DepthLevel:
    """A single price level in the order book"""
    price: float
    size: int
    order_count: int = 1
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DepthSnapshot:
    """Snapshot of order book depth"""
    symbol: str
    timestamp: datetime

    # Bid/ask levels
    bids: List[DepthLevel] = field(default_factory=list)
    asks: List[DepthLevel] = field(default_factory=list)

    # Summary stats
    total_bid_volume: int = 0
    total_ask_volume: int = 0
    bid_ask_ratio: float = 1.0
    spread: float = 0.0
    spread_pct: float = 0.0
    mid_price: float = 0.0

    # Detected features
    bid_walls: List[OrderWall] = field(default_factory=list)
    ask_walls: List[OrderWall] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "total_bid_volume": self.total_bid_volume,
            "total_ask_volume": self.total_ask_volume,
            "bid_ask_ratio": round(self.bid_ask_ratio, 2),
            "spread": round(self.spread, 4),
            "spread_pct": round(self.spread_pct, 3),
            "mid_price": round(self.mid_price, 4),
            "bid_levels": len(self.bids),
            "ask_levels": len(self.asks),
            "bid_walls": [w.to_dict() for w in self.bid_walls],
            "ask_walls": [w.to_dict() for w in self.ask_walls]
        }


@dataclass
class DepthAnalysis:
    """Analysis result for a symbol"""
    symbol: str
    signal: DepthSignal
    confidence: float           # 0-100%

    # Imbalance
    imbalance_ratio: float      # >1 = bullish, <1 = bearish
    imbalance_strength: str     # STRONG, MODERATE, WEAK

    # Walls
    nearest_bid_wall: Optional[OrderWall] = None
    nearest_ask_wall: Optional[OrderWall] = None

    # Absorption
    active_absorption: Optional[str] = None  # "BID" or "ASK"
    absorption_progress: float = 0  # 0-100%

    # Entry/exit recommendation
    entry_valid: bool = True
    entry_reason: str = ""
    suggested_stop: float = 0   # Based on bid wall
    suggested_target: float = 0 # Based on ask wall

    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "signal": self.signal.value,
            "confidence": round(self.confidence, 1),
            "imbalance_ratio": round(self.imbalance_ratio, 2),
            "imbalance_strength": self.imbalance_strength,
            "nearest_bid_wall": self.nearest_bid_wall.to_dict() if self.nearest_bid_wall else None,
            "nearest_ask_wall": self.nearest_ask_wall.to_dict() if self.nearest_ask_wall else None,
            "active_absorption": self.active_absorption,
            "absorption_progress": round(self.absorption_progress, 1),
            "entry_valid": self.entry_valid,
            "entry_reason": self.entry_reason,
            "suggested_stop": round(self.suggested_stop, 4) if self.suggested_stop else None,
            "suggested_target": round(self.suggested_target, 4) if self.suggested_target else None,
            "timestamp": self.timestamp
        }


class Level2DepthAnalyzer:
    """
    Analyze Level 2 order book depth for trading signals.

    Features:
    - Bid/ask wall detection
    - Order absorption tracking
    - Spoofing detection
    - Imbalance analysis
    - Support/resistance from order book
    """

    def __init__(self):
        # Depth data per symbol
        self.snapshots: Dict[str, deque] = {}  # History of snapshots
        self.current_depth: Dict[str, DepthSnapshot] = {}

        # Wall tracking
        self.active_walls: Dict[str, List[OrderWall]] = {}
        self.wall_history: Dict[str, List[OrderWall]] = {}

        # Imbalance history
        self.imbalance_history: Dict[str, deque] = {}

        # Configuration
        self.config = {
            # Wall detection
            "wall_size_multiplier": 3.0,    # Wall = X times average level size
            "min_wall_size": 5000,          # Minimum shares to be a wall
            "wall_proximity_pct": 2.0,      # Consider walls within 2% of price

            # Absorption detection
            "absorption_threshold": 0.3,     # 30% eaten = absorbing
            "absorbed_threshold": 0.8,       # 80% eaten = absorbed

            # Spoofing detection
            "spoof_time_threshold": 5.0,     # Wall disappears within 5 seconds
            "spoof_size_threshold": 0.5,     # Wall drops by 50%+ suddenly

            # Imbalance
            "strong_imbalance": 2.0,         # 2:1 ratio = strong
            "moderate_imbalance": 1.5,       # 1.5:1 = moderate

            # History
            "max_snapshots": 100,
            "max_wall_history": 50
        }

    def update_depth(self, symbol: str, bids: List[Dict], asks: List[Dict],
                     current_price: float = None) -> DepthAnalysis:
        """
        Update order book depth for a symbol.

        Args:
            symbol: Stock symbol
            bids: List of {price, size} dicts
            asks: List of {price, size} dicts
            current_price: Current market price

        Returns:
            DepthAnalysis with signals
        """
        symbol = symbol.upper()
        now = datetime.now()

        # Parse bid/ask levels
        bid_levels = [
            DepthLevel(
                price=b.get('price', b.get('p', 0)),
                size=b.get('size', b.get('s', 0)),
                order_count=b.get('count', 1),
                timestamp=now
            )
            for b in bids if b.get('price', b.get('p', 0)) > 0
        ]

        ask_levels = [
            DepthLevel(
                price=a.get('price', a.get('p', 0)),
                size=a.get('size', a.get('s', 0)),
                order_count=a.get('count', 1),
                timestamp=now
            )
            for a in asks if a.get('price', a.get('p', 0)) > 0
        ]

        # Sort by price
        bid_levels.sort(key=lambda x: x.price, reverse=True)  # Highest first
        ask_levels.sort(key=lambda x: x.price)  # Lowest first

        # Calculate summary stats
        total_bid = sum(l.size for l in bid_levels)
        total_ask = sum(l.size for l in ask_levels)

        best_bid = bid_levels[0].price if bid_levels else 0
        best_ask = ask_levels[0].price if ask_levels else 0

        spread = best_ask - best_bid if best_bid and best_ask else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else current_price or 0
        spread_pct = (spread / mid_price * 100) if mid_price > 0 else 0

        ratio = total_bid / total_ask if total_ask > 0 else 1.0

        # Create snapshot
        snapshot = DepthSnapshot(
            symbol=symbol,
            timestamp=now,
            bids=bid_levels,
            asks=ask_levels,
            total_bid_volume=total_bid,
            total_ask_volume=total_ask,
            bid_ask_ratio=ratio,
            spread=spread,
            spread_pct=spread_pct,
            mid_price=mid_price
        )

        # Store snapshot
        if symbol not in self.snapshots:
            self.snapshots[symbol] = deque(maxlen=self.config["max_snapshots"])
        self.snapshots[symbol].append(snapshot)

        # Store current
        prev_depth = self.current_depth.get(symbol)
        self.current_depth[symbol] = snapshot

        # Detect walls
        bid_walls = self._detect_walls(symbol, bid_levels, WallType.BID_WALL, mid_price)
        ask_walls = self._detect_walls(symbol, ask_levels, WallType.ASK_WALL, mid_price)

        snapshot.bid_walls = bid_walls
        snapshot.ask_walls = ask_walls

        # Track walls over time
        self._track_walls(symbol, bid_walls + ask_walls, prev_depth)

        # Track imbalance
        if symbol not in self.imbalance_history:
            self.imbalance_history[symbol] = deque(maxlen=50)
        self.imbalance_history[symbol].append(ratio)

        # Generate analysis
        return self._analyze(symbol, snapshot, current_price or mid_price)

    def _detect_walls(self, symbol: str, levels: List[DepthLevel],
                      wall_type: WallType, current_price: float) -> List[OrderWall]:
        """Detect order walls in price levels"""
        if not levels:
            return []

        walls = []

        # Calculate average level size
        sizes = [l.size for l in levels]
        avg_size = np.mean(sizes) if sizes else 0

        # Wall threshold
        min_wall = max(
            self.config["min_wall_size"],
            avg_size * self.config["wall_size_multiplier"]
        )

        for level in levels:
            if level.size >= min_wall:
                # Calculate distance from price
                distance = abs(level.price - current_price) / current_price * 100 if current_price > 0 else 0

                # Only consider walls within proximity
                if distance <= self.config["wall_proximity_pct"]:
                    wall = OrderWall(
                        symbol=symbol,
                        wall_type=wall_type,
                        price=level.price,
                        size=level.size,
                        initial_size=level.size,
                        peak_size=level.size,
                        current_size=level.size,
                        is_significant=level.size >= min_wall * 1.5,
                        distance_from_price=distance
                    )
                    walls.append(wall)

        # Sort by size (largest first)
        walls.sort(key=lambda w: w.size, reverse=True)

        return walls[:5]  # Keep top 5 walls per side

    def _track_walls(self, symbol: str, new_walls: List[OrderWall],
                     prev_depth: Optional[DepthSnapshot]):
        """Track wall changes over time for absorption/spoofing detection"""
        if symbol not in self.active_walls:
            self.active_walls[symbol] = []

        now = datetime.now()
        updated_walls = []

        for new_wall in new_walls:
            # Find matching existing wall
            existing = None
            for aw in self.active_walls[symbol]:
                if (aw.wall_type == new_wall.wall_type and
                    abs(aw.price - new_wall.price) < 0.01):  # Same price level
                    existing = aw
                    break

            if existing:
                # Update existing wall
                existing.last_update = now
                existing.duration_seconds = (now - existing.first_seen).total_seconds()
                existing.current_size = new_wall.size

                # Track peak
                if new_wall.size > existing.peak_size:
                    existing.peak_size = new_wall.size
                    existing.status = WallStatus.REINFORCED

                # Check absorption
                absorbed_pct = 1 - (new_wall.size / existing.peak_size) if existing.peak_size > 0 else 0
                existing.absorbed_volume = existing.peak_size - new_wall.size

                if absorbed_pct >= self.config["absorbed_threshold"]:
                    existing.status = WallStatus.ABSORBED
                elif absorbed_pct >= self.config["absorption_threshold"]:
                    existing.status = WallStatus.ABSORBING
                    # Calculate absorption rate
                    if existing.duration_seconds > 0:
                        existing.absorption_rate = existing.absorbed_volume / existing.duration_seconds

                updated_walls.append(existing)
            else:
                # New wall
                new_wall.first_seen = now
                new_wall.last_update = now
                updated_walls.append(new_wall)

        # Check for disappeared walls (spoofing)
        for aw in self.active_walls[symbol]:
            found = any(
                w.wall_type == aw.wall_type and abs(w.price - aw.price) < 0.01
                for w in new_walls
            )
            if not found:
                # Wall disappeared
                duration = (now - aw.first_seen).total_seconds()
                if duration < self.config["spoof_time_threshold"]:
                    # Likely spoofing
                    aw.status = WallStatus.DISAPPEARED
                    logger.warning(f"SPOOF DETECTED: {symbol} {aw.wall_type.value} at ${aw.price:.2f} disappeared after {duration:.1f}s")

                    # Add to history
                    if symbol not in self.wall_history:
                        self.wall_history[symbol] = []
                    self.wall_history[symbol].append(aw)
                    if len(self.wall_history[symbol]) > self.config["max_wall_history"]:
                        self.wall_history[symbol] = self.wall_history[symbol][-self.config["max_wall_history"]:]

        self.active_walls[symbol] = updated_walls

    def _analyze(self, symbol: str, snapshot: DepthSnapshot,
                 current_price: float) -> DepthAnalysis:
        """Generate trading analysis from depth data"""

        # Determine imbalance strength
        ratio = snapshot.bid_ask_ratio
        if ratio >= self.config["strong_imbalance"]:
            imbalance_strength = "STRONG"
            signal = DepthSignal.BULLISH_IMBALANCE
        elif ratio >= self.config["moderate_imbalance"]:
            imbalance_strength = "MODERATE"
            signal = DepthSignal.BULLISH_IMBALANCE
        elif ratio <= 1 / self.config["strong_imbalance"]:
            imbalance_strength = "STRONG"
            signal = DepthSignal.BEARISH_IMBALANCE
        elif ratio <= 1 / self.config["moderate_imbalance"]:
            imbalance_strength = "MODERATE"
            signal = DepthSignal.BEARISH_IMBALANCE
        else:
            imbalance_strength = "WEAK"
            signal = DepthSignal.NEUTRAL

        # Find nearest significant walls
        nearest_bid_wall = None
        nearest_ask_wall = None

        if snapshot.bid_walls:
            nearest_bid_wall = min(snapshot.bid_walls,
                                   key=lambda w: w.distance_from_price)
        if snapshot.ask_walls:
            nearest_ask_wall = min(snapshot.ask_walls,
                                   key=lambda w: w.distance_from_price)

        # Check for absorption
        active_absorption = None
        absorption_progress = 0

        for wall in self.active_walls.get(symbol, []):
            if wall.status == WallStatus.ABSORBING:
                if wall.wall_type == WallType.ASK_WALL:
                    active_absorption = "ASK"
                    signal = DepthSignal.ASK_ABSORPTION  # Bullish
                else:
                    active_absorption = "BID"
                    signal = DepthSignal.BID_ABSORPTION  # Bearish

                if wall.peak_size > 0:
                    absorption_progress = (wall.absorbed_volume / wall.peak_size) * 100
                break

        # Check for spoofing
        recent_spoofs = [
            w for w in self.wall_history.get(symbol, [])
            if w.status == WallStatus.DISAPPEARED and
            (datetime.now() - w.last_update).total_seconds() < 60
        ]
        if recent_spoofs:
            signal = DepthSignal.SPOOFING_DETECTED

        # Entry validation based on depth
        entry_valid = True
        entry_reason = ""

        # Check spread
        if snapshot.spread_pct > 1.0:
            entry_valid = False
            entry_reason = f"Spread too wide ({snapshot.spread_pct:.2f}%)"

        # Check for strong resistance nearby
        if nearest_ask_wall and nearest_ask_wall.is_significant:
            if nearest_ask_wall.distance_from_price < 0.5:  # Within 0.5%
                entry_reason = f"Strong ask wall at ${nearest_ask_wall.price:.2f}"

        # Calculate suggested levels
        suggested_stop = 0
        suggested_target = 0

        if nearest_bid_wall:
            # Stop just below bid wall
            suggested_stop = nearest_bid_wall.price * 0.998

        if nearest_ask_wall:
            # Target at ask wall
            suggested_target = nearest_ask_wall.price

        # Calculate confidence
        confidence = 50.0

        # Boost for strong imbalance
        if imbalance_strength == "STRONG":
            confidence += 20
        elif imbalance_strength == "MODERATE":
            confidence += 10

        # Boost for absorption
        if active_absorption == "ASK":
            confidence += 15  # Bullish
        elif active_absorption == "BID":
            confidence -= 15  # Bearish

        # Penalize for spoofing
        if signal == DepthSignal.SPOOFING_DETECTED:
            confidence -= 20

        confidence = max(0, min(100, confidence))

        return DepthAnalysis(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            imbalance_ratio=ratio,
            imbalance_strength=imbalance_strength,
            nearest_bid_wall=nearest_bid_wall,
            nearest_ask_wall=nearest_ask_wall,
            active_absorption=active_absorption,
            absorption_progress=absorption_progress,
            entry_valid=entry_valid,
            entry_reason=entry_reason,
            suggested_stop=suggested_stop,
            suggested_target=suggested_target,
            timestamp=datetime.now().isoformat()
        )

    def get_depth(self, symbol: str) -> Optional[DepthSnapshot]:
        """Get current depth snapshot for symbol"""
        return self.current_depth.get(symbol.upper())

    def get_analysis(self, symbol: str) -> Optional[DepthAnalysis]:
        """Get latest analysis for symbol"""
        symbol = symbol.upper()
        snapshot = self.current_depth.get(symbol)
        if not snapshot:
            return None

        return self._analyze(symbol, snapshot, snapshot.mid_price)

    def get_walls(self, symbol: str) -> Dict:
        """Get active walls for symbol"""
        symbol = symbol.upper()
        walls = self.active_walls.get(symbol, [])

        bid_walls = [w for w in walls if w.wall_type == WallType.BID_WALL]
        ask_walls = [w for w in walls if w.wall_type == WallType.ASK_WALL]

        return {
            "symbol": symbol,
            "bid_walls": [w.to_dict() for w in bid_walls],
            "ask_walls": [w.to_dict() for w in ask_walls],
            "absorbing": [w.to_dict() for w in walls if w.status == WallStatus.ABSORBING]
        }

    def get_imbalance_trend(self, symbol: str) -> Dict:
        """Get imbalance trend over time"""
        symbol = symbol.upper()
        history = list(self.imbalance_history.get(symbol, []))

        if not history:
            return {"symbol": symbol, "trend": "UNKNOWN", "ratios": []}

        # Calculate trend
        if len(history) >= 5:
            recent = np.mean(history[-5:])
            older = np.mean(history[:-5]) if len(history) > 5 else recent

            if recent > older * 1.1:
                trend = "BULLISH"
            elif recent < older * 0.9:
                trend = "BEARISH"
            else:
                trend = "STABLE"
        else:
            trend = "UNKNOWN"

        return {
            "symbol": symbol,
            "trend": trend,
            "current_ratio": round(history[-1], 2) if history else 1.0,
            "avg_ratio": round(np.mean(history), 2),
            "samples": len(history)
        }

    def is_entry_valid(self, symbol: str) -> Tuple[bool, str]:
        """Check if entry is valid based on depth"""
        analysis = self.get_analysis(symbol)
        if not analysis:
            return True, "No depth data"

        return analysis.entry_valid, analysis.entry_reason

    def get_entry_boost(self, symbol: str) -> float:
        """
        Get confidence boost from depth analysis.

        Returns boost factor (-0.2 to +0.2)
        """
        analysis = self.get_analysis(symbol)
        if not analysis:
            return 0.0

        boost = 0.0

        # Bullish signals
        if analysis.signal == DepthSignal.BULLISH_IMBALANCE:
            boost += 0.05 if analysis.imbalance_strength == "MODERATE" else 0.10
        elif analysis.signal == DepthSignal.ASK_ABSORPTION:
            boost += 0.15  # Ask wall being eaten = breakout

        # Bearish signals
        elif analysis.signal == DepthSignal.BEARISH_IMBALANCE:
            boost -= 0.05 if analysis.imbalance_strength == "MODERATE" else 0.10
        elif analysis.signal == DepthSignal.BID_ABSORPTION:
            boost -= 0.15  # Bid wall being eaten = breakdown

        # Spoofing penalty
        if analysis.signal == DepthSignal.SPOOFING_DETECTED:
            boost -= 0.10

        return max(-0.2, min(0.2, boost))

    def get_status(self) -> Dict:
        """Get analyzer status"""
        return {
            "symbols_tracked": len(self.current_depth),
            "total_walls": sum(len(w) for w in self.active_walls.values()),
            "symbols_with_absorption": len([
                s for s, walls in self.active_walls.items()
                if any(w.status == WallStatus.ABSORBING for w in walls)
            ]),
            "recent_spoofs": sum(
                len([w for w in walls if w.status == WallStatus.DISAPPEARED])
                for walls in self.wall_history.values()
            ),
            "config": self.config
        }


# Singleton instance
_depth_analyzer: Optional[Level2DepthAnalyzer] = None


def get_depth_analyzer() -> Level2DepthAnalyzer:
    """Get singleton depth analyzer"""
    global _depth_analyzer
    if _depth_analyzer is None:
        _depth_analyzer = Level2DepthAnalyzer()
    return _depth_analyzer


# Convenience functions
def analyze_depth(symbol: str, bids: List[Dict], asks: List[Dict],
                  price: float = None) -> DepthAnalysis:
    """Analyze order book depth"""
    return get_depth_analyzer().update_depth(symbol, bids, asks, price)


def get_depth_signal(symbol: str) -> Optional[DepthSignal]:
    """Get current depth signal for symbol"""
    analysis = get_depth_analyzer().get_analysis(symbol)
    return analysis.signal if analysis else None


def is_depth_entry_valid(symbol: str) -> Tuple[bool, str]:
    """Check if entry is valid based on depth"""
    return get_depth_analyzer().is_entry_valid(symbol)
