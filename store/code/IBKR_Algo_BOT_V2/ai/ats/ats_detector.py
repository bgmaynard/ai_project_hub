"""
ATS (Advanced Trading Signal) Detector

Scoring engine and state machine for trade signals.

States:
- IDLE: No activity
- FORMING: SmartZone detected, watching
- ACTIVE: Breakout detected, ready to trade
- EXHAUSTION: Momentum fading, close/avoid
- INVALIDATED: Pattern failed, skip
"""

from datetime import datetime
from typing import Optional, Tuple
from .types import (
    Bar, SmartZoneSignal, MarketContext, AtsTrigger,
    AtsState, AtsSymbolState
)
from .smartzone import SmartZoneDetector, get_smartzone_detector
from .time_utils import should_allow_ats_trigger


class AtsDetector:
    """
    ATS Scoring and State Machine.

    Scoring Components (0-100):
    - SmartZone quality (0-30)
    - Breakout strength (0-25)
    - Volume confirmation (0-20)
    - Market structure (0-15)
    - Time alignment (0-10)
    """

    # State transition thresholds
    FORMING_THRESHOLD = 30
    ACTIVE_THRESHOLD = 60
    EXHAUSTION_THRESHOLD = 40

    # Scoring weights
    WEIGHT_ZONE = 30
    WEIGHT_BREAKOUT = 25
    WEIGHT_VOLUME = 20
    WEIGHT_STRUCTURE = 15
    WEIGHT_TIME = 10

    def __init__(self):
        self._smartzone = get_smartzone_detector()
        self._states: dict[str, AtsSymbolState] = {}

    def process_bar(
        self,
        symbol: str,
        bar: Bar,
        context: Optional[MarketContext] = None
    ) -> Tuple[AtsState, Optional[AtsTrigger]]:
        """
        Process new bar and update state.

        Args:
            symbol: Stock symbol
            bar: New OHLCV bar
            context: Optional market context

        Returns:
            (new_state, trigger) - Current state and any trigger event
        """
        # Get or create symbol state
        if symbol not in self._states:
            self._states[symbol] = AtsSymbolState(symbol=symbol)

        state = self._states[symbol]

        # Update consecutive candle tracking
        if bar.is_green:
            state.consecutive_greens += 1
            state.consecutive_reds = 0
        else:
            state.consecutive_reds += 1
            state.consecutive_greens = 0

        # Process through SmartZone detector
        zone_signal = self._smartzone.add_bar(symbol, bar)

        # Calculate score
        score = self._calculate_score(symbol, bar, zone_signal, context)
        state.add_score(score)

        # State machine transitions
        trigger = None
        new_state = self._transition_state(state, score, zone_signal, bar, context)

        # Generate trigger on ACTIVE state
        if new_state == AtsState.ACTIVE and state.state != AtsState.ACTIVE:
            trigger = self._generate_trigger(symbol, bar, zone_signal, context, score)

        state.state = new_state
        state.bars_in_state = state.bars_in_state + 1 if state.state == new_state else 1
        state.current_zone = zone_signal if zone_signal and not zone_signal.is_resolved else state.current_zone
        state.last_update = datetime.now()

        return new_state, trigger

    def _calculate_score(
        self,
        symbol: str,
        bar: Bar,
        zone_signal: Optional[SmartZoneSignal],
        context: Optional[MarketContext]
    ) -> float:
        """Calculate ATS score (0-100)"""
        score = 0.0

        # Zone quality (0-30)
        zone = zone_signal or self._smartzone.get_active_zone(symbol)
        if zone:
            zone_score = zone.confidence * 0.3
            if zone.is_resolved and zone.resolution_type == "EXPANSION":
                zone_score += 15  # Bonus for breakout
            score += min(self.WEIGHT_ZONE, zone_score)

        # Breakout strength (0-25)
        if zone and bar.close > zone.zone_high:
            breakout_pct = ((bar.close - zone.zone_high) / zone.zone_high) * 100
            breakout_score = min(25, breakout_pct * 5)
            score += breakout_score

        # Volume confirmation (0-20)
        if context and context.rel_volume > 1.0:
            vol_score = min(20, (context.rel_volume - 1) * 5)
            score += vol_score

        # Market structure (0-15)
        if context:
            structure_score = 0
            if context.is_above_vwap:
                structure_score += 5
            if context.is_above_ema9:
                structure_score += 5
            if context.is_bullish_structure:
                structure_score += 5
            score += min(self.WEIGHT_STRUCTURE, structure_score)

        # Time alignment (0-10)
        allowed, _ = should_allow_ats_trigger()
        if allowed:
            score += self.WEIGHT_TIME

        return min(100.0, max(0.0, score))

    def _transition_state(
        self,
        state: AtsSymbolState,
        score: float,
        zone_signal: Optional[SmartZoneSignal],
        bar: Bar,
        context: Optional[MarketContext]
    ) -> AtsState:
        """Determine state transition based on score and conditions"""

        current = state.state

        # Check for exhaustion conditions
        if self._is_exhausted(state, bar, context):
            return AtsState.EXHAUSTION

        # Check for invalidation
        if self._is_invalidated(state, zone_signal):
            return AtsState.INVALIDATED

        # Score-based transitions
        if current == AtsState.IDLE:
            if zone_signal and not zone_signal.is_resolved:
                return AtsState.FORMING
            if score >= self.FORMING_THRESHOLD:
                return AtsState.FORMING

        elif current == AtsState.FORMING:
            if score >= self.ACTIVE_THRESHOLD:
                # Need breakout confirmation
                if zone_signal and zone_signal.is_resolved and zone_signal.resolution_type == "EXPANSION":
                    return AtsState.ACTIVE
            if score < self.FORMING_THRESHOLD:
                return AtsState.IDLE

        elif current == AtsState.ACTIVE:
            if score < self.EXHAUSTION_THRESHOLD:
                return AtsState.EXHAUSTION
            if state.bars_in_state > 10:
                # Active too long without trigger
                return AtsState.EXHAUSTION

        elif current == AtsState.EXHAUSTION:
            if state.bars_in_state > 5:
                return AtsState.IDLE
            if score >= self.ACTIVE_THRESHOLD:
                return AtsState.ACTIVE

        elif current == AtsState.INVALIDATED:
            if state.bars_in_state > 10:
                return AtsState.IDLE

        return current

    def _is_exhausted(
        self,
        state: AtsSymbolState,
        bar: Bar,
        context: Optional[MarketContext]
    ) -> bool:
        """Check for momentum exhaustion"""
        # 4+ consecutive red candles
        if state.consecutive_reds >= 4:
            return True

        # Below VWAP with declining score
        if context and not context.is_above_vwap:
            if state.score_trend == "DECLINING":
                return True

        return False

    def _is_invalidated(
        self,
        state: AtsSymbolState,
        zone_signal: Optional[SmartZoneSignal]
    ) -> bool:
        """Check for pattern invalidation"""
        # Zone broke down
        if zone_signal and zone_signal.is_resolved:
            if zone_signal.resolution_type == "BREAKDOWN":
                return True

        return False

    def _generate_trigger(
        self,
        symbol: str,
        bar: Bar,
        zone_signal: Optional[SmartZoneSignal],
        context: Optional[MarketContext],
        score: float
    ) -> AtsTrigger:
        """Generate ATS trigger for scalper"""

        zone = zone_signal or self._smartzone.get_active_zone(symbol)

        # Calculate levels
        entry_price = bar.close
        if zone:
            zone_low = zone.zone_low
            zone_high = zone.zone_high
            break_level = zone.break_level
            stop_loss = zone_low * 0.995  # Just below zone low
        else:
            zone_low = bar.low
            zone_high = bar.high
            break_level = bar.high
            stop_loss = bar.low * 0.995

        # Risk-based targets
        risk = entry_price - stop_loss
        target_1 = entry_price + (risk * 2)  # 2:1 R/R
        target_2 = entry_price + (risk * 3)  # 3:1 R/R

        # Size boost based on score
        size_boost = 1.0
        if score >= 80:
            size_boost = 1.25  # Strong signal
        elif score >= 70:
            size_boost = 1.1

        # Check permission
        allowed, reason = should_allow_ats_trigger()
        permission = "ALLOWED" if allowed else "BLOCKED"

        return AtsTrigger(
            symbol=symbol,
            trigger_type="BREAKOUT",
            score=score,
            break_level=break_level,
            zone_low=zone_low,
            zone_high=zone_high,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            size_boost=size_boost,
            permission=permission,
            block_reason=reason if not allowed else None,
            context=context,
            zone_signal=zone,
            timestamp=datetime.now(),
        )

    def get_state(self, symbol: str) -> Optional[AtsSymbolState]:
        """Get symbol state"""
        return self._states.get(symbol)

    def get_all_states(self) -> dict[str, AtsSymbolState]:
        """Get all symbol states"""
        return self._states.copy()

    def get_active_symbols(self) -> list[str]:
        """Get symbols in ACTIVE state"""
        return [
            sym for sym, state in self._states.items()
            if state.state == AtsState.ACTIVE
        ]

    def get_forming_symbols(self) -> list[str]:
        """Get symbols in FORMING state"""
        return [
            sym for sym, state in self._states.items()
            if state.state == AtsState.FORMING
        ]

    def reset_symbol(self, symbol: str):
        """Reset symbol state"""
        if symbol in self._states:
            del self._states[symbol]
        self._smartzone.clear_symbol(symbol)

    def reset_all(self):
        """Reset all state"""
        self._states.clear()

    def get_status(self) -> dict:
        """Get detector status"""
        state_counts = {}
        for state in self._states.values():
            state_name = state.state.value
            state_counts[state_name] = state_counts.get(state_name, 0) + 1

        return {
            "total_symbols": len(self._states),
            "state_distribution": state_counts,
            "active_count": len(self.get_active_symbols()),
            "forming_count": len(self.get_forming_symbols()),
            "smartzone": self._smartzone.get_status(),
        }


# Singleton instance
_detector: Optional[AtsDetector] = None


def get_ats_detector() -> AtsDetector:
    """Get singleton ATS detector"""
    global _detector
    if _detector is None:
        _detector = AtsDetector()
    return _detector
