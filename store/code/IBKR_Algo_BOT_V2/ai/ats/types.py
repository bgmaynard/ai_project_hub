"""
ATS Shared Types

Bar, SmartZoneSignal, MarketContext, AtsTrigger
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List


class AtsState(Enum):
    """ATS State Machine States"""
    IDLE = "IDLE"
    FORMING = "FORMING"
    ACTIVE = "ACTIVE"
    EXHAUSTION = "EXHAUSTION"
    INVALIDATED = "INVALIDATED"


class ZoneType(Enum):
    """SmartZone pattern type"""
    CONSOLIDATION = "CONSOLIDATION"
    PULLBACK = "PULLBACK"
    FLAG = "FLAG"
    TRIANGLE = "TRIANGLE"


@dataclass
class Bar:
    """OHLCV bar data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_green(self) -> bool:
        return self.close >= self.open

    @property
    def is_red(self) -> bool:
        return self.close < self.open

    @property
    def body_ratio(self) -> float:
        """Body as percentage of range"""
        if self.range == 0:
            return 0.0
        return self.body / self.range


@dataclass
class SmartZoneSignal:
    """SmartZone pattern detection signal"""
    symbol: str
    zone_type: ZoneType
    zone_high: float
    zone_low: float
    zone_mid: float
    formation_bars: int
    compression_ratio: float  # How tight the zone is
    break_level: float  # Price that triggers breakout
    is_resolved: bool = False
    resolution_type: Optional[str] = None  # "EXPANSION" or "BREAKDOWN"
    resolution_price: Optional[float] = None
    resolution_time: Optional[datetime] = None
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def zone_width_pct(self) -> float:
        """Zone width as percentage of mid"""
        if self.zone_mid == 0:
            return 0.0
        return ((self.zone_high - self.zone_low) / self.zone_mid) * 100


@dataclass
class MarketContext:
    """Current market context for ATS decisions"""
    symbol: str
    current_price: float
    vwap: Optional[float] = None
    ema9: Optional[float] = None
    ema20: Optional[float] = None
    rel_volume: float = 1.0
    float_shares: Optional[float] = None
    float_rotation: float = 0.0
    has_catalyst: bool = False
    catalyst_type: Optional[str] = None
    regime: str = "UNKNOWN"  # From Chronos
    trend_direction: int = 0  # -1, 0, 1
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_above_vwap(self) -> bool:
        if self.vwap is None:
            return True  # No VWAP data, allow
        return self.current_price >= self.vwap

    @property
    def is_above_ema9(self) -> bool:
        if self.ema9 is None:
            return True
        return self.current_price >= self.ema9

    @property
    def is_bullish_structure(self) -> bool:
        """Price above key MAs in bullish order"""
        if self.ema9 is None or self.ema20 is None:
            return True
        return self.current_price >= self.ema9 >= self.ema20


@dataclass
class AtsTrigger:
    """ATS trigger event for scalper"""
    symbol: str
    trigger_type: str  # "BREAKOUT", "CONTINUATION", "REVERSAL"
    score: float  # 0-100
    break_level: float
    zone_low: float
    zone_high: float
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    size_boost: float = 1.0  # Position size multiplier
    permission: str = "ALLOWED"  # "ALLOWED", "BLOCKED", "CAUTION"
    block_reason: Optional[str] = None
    context: Optional[MarketContext] = None
    zone_signal: Optional[SmartZoneSignal] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def risk_reward(self) -> float:
        """Risk/reward ratio to target 1"""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.target_1 - self.entry_price)
        if risk == 0:
            return 0.0
        return reward / risk

    @property
    def is_valid(self) -> bool:
        """Check if trigger is valid for execution"""
        return (
            self.permission == "ALLOWED" and
            self.score >= 60 and
            self.risk_reward >= 1.5
        )


@dataclass
class AtsSymbolState:
    """Per-symbol ATS state tracking"""
    symbol: str
    state: AtsState = AtsState.IDLE
    current_zone: Optional[SmartZoneSignal] = None
    last_trigger: Optional[AtsTrigger] = None
    score: float = 0.0
    score_history: List[float] = field(default_factory=list)
    consecutive_greens: int = 0
    consecutive_reds: int = 0
    bars_in_state: int = 0
    last_update: datetime = field(default_factory=datetime.now)

    def add_score(self, score: float, max_history: int = 20):
        """Add score to history"""
        self.score = score
        self.score_history.append(score)
        if len(self.score_history) > max_history:
            self.score_history = self.score_history[-max_history:]

    @property
    def avg_score(self) -> float:
        if not self.score_history:
            return 0.0
        return sum(self.score_history) / len(self.score_history)

    @property
    def score_trend(self) -> str:
        """Is score improving or declining?"""
        if len(self.score_history) < 3:
            return "UNKNOWN"
        recent = self.score_history[-3:]
        if recent[-1] > recent[0]:
            return "IMPROVING"
        elif recent[-1] < recent[0]:
            return "DECLINING"
        return "FLAT"
