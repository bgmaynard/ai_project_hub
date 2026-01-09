"""
Momentum Exhaustion Detector
============================
Detect when a momentum move is fading before the big drop.

Ross Cameron Exit Signals:
1. RSI Bearish Divergence - Price making higher highs, RSI making lower highs
2. Volume Exhaustion - Declining volume on rising price (buyers drying up)
3. Consecutive Red Candles - 3+ red candles after a run-up
4. Price Stalling - Multiple attempts to break level failing
5. Spread Widening - Market makers pulling bids (liquidity drying up)

Key Insight:
- Momentum exhaustion is a LEADING indicator
- Exit BEFORE the big drop, not during it
- Better to exit early with smaller profit than ride it down to a loss
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ExhaustionSignal(Enum):
    """Types of exhaustion signals"""

    NONE = "NONE"
    RSI_DIVERGENCE = "RSI_DIVERGENCE"  # Bearish RSI divergence
    VOLUME_EXHAUSTION = "VOLUME_EXHAUSTION"  # Declining volume on rise
    RED_CANDLES = "RED_CANDLES"  # Consecutive red candles
    PRICE_STALLING = "PRICE_STALLING"  # Failed breakout attempts
    SPREAD_WIDENING = "SPREAD_WIDENING"  # Liquidity drying up
    MOMENTUM_FADE = "MOMENTUM_FADE"  # General momentum fading
    CLIMAX_TOP = "CLIMAX_TOP"  # Blow-off top pattern


class ExhaustionSeverity(Enum):
    """Severity of exhaustion signal"""

    LOW = "LOW"  # Watch closely
    MEDIUM = "MEDIUM"  # Consider exit
    HIGH = "HIGH"  # Exit recommended
    CRITICAL = "CRITICAL"  # Exit immediately


@dataclass
class ExhaustionAlert:
    """Alert when exhaustion is detected"""

    symbol: str
    signal: ExhaustionSignal
    severity: ExhaustionSeverity
    confidence: float  # 0-100%
    details: str
    should_exit: bool
    timestamp: str

    # Technical data
    current_price: float = 0.0
    rsi: float = 0.0
    volume_trend: float = 0.0  # Negative = declining
    consecutive_red: int = 0
    spread_pct: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "signal": self.signal.value,
            "severity": self.severity.value,
            "confidence": round(self.confidence, 1),
            "details": self.details,
            "should_exit": self.should_exit,
            "timestamp": self.timestamp,
            "current_price": round(self.current_price, 4),
            "rsi": round(self.rsi, 1),
            "volume_trend": round(self.volume_trend, 2),
            "consecutive_red": self.consecutive_red,
            "spread_pct": round(self.spread_pct, 3),
        }


@dataclass
class PricePoint:
    """Price point for divergence detection"""

    price: float
    rsi: float
    volume: int
    timestamp: datetime
    is_high: bool = False
    is_low: bool = False


@dataclass
class SymbolState:
    """Tracking state for a symbol"""

    symbol: str

    # Price/candle history
    candles: deque = field(default_factory=lambda: deque(maxlen=50))
    prices: deque = field(default_factory=lambda: deque(maxlen=100))
    volumes: deque = field(default_factory=lambda: deque(maxlen=50))

    # RSI calculation
    gains: deque = field(default_factory=lambda: deque(maxlen=14))
    losses: deque = field(default_factory=lambda: deque(maxlen=14))
    current_rsi: float = 50.0

    # Divergence detection
    price_highs: List[PricePoint] = field(default_factory=list)
    rsi_at_highs: List[float] = field(default_factory=list)

    # Pattern detection
    consecutive_red_candles: int = 0
    failed_breakout_attempts: int = 0
    last_high: float = 0.0

    # Spread tracking
    recent_spreads: deque = field(default_factory=lambda: deque(maxlen=20))
    avg_spread: float = 0.0

    # Alerts
    active_alerts: List[ExhaustionAlert] = field(default_factory=list)
    last_alert_time: Optional[datetime] = None

    # Entry tracking (for position monitoring)
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    high_since_entry: float = 0.0


class MomentumExhaustionDetector:
    """
    Detect momentum exhaustion for early exit signals.

    Combines multiple indicators:
    - RSI divergence (most reliable)
    - Volume exhaustion
    - Price patterns (red candles, stalling)
    - Spread analysis
    """

    def __init__(self):
        # State per symbol
        self.symbols: Dict[str, SymbolState] = {}

        # Alert history
        self.alert_history: List[ExhaustionAlert] = []
        self.max_history = 500

        # Callbacks
        self.on_exhaustion_alert: Optional[callable] = None

        # Configuration
        self.config = {
            # RSI settings
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_divergence_threshold": 5,  # RSI must drop this much for divergence
            # Volume settings
            "volume_decline_threshold": 0.7,  # 30% decline = exhaustion
            "volume_lookback": 5,
            # Red candles
            "red_candle_threshold": 3,  # 3 consecutive reds = warning
            "red_candle_critical": 5,  # 5 reds = critical
            # Spread
            "spread_widen_threshold": 1.5,  # 50% wider than average
            "spread_critical": 2.0,  # 100% wider = critical
            # Stalling
            "stall_attempts_threshold": 3,  # 3 failed breakouts
            "stall_range_pct": 0.5,  # Within 0.5% of high
            # Alert cooldown
            "alert_cooldown_seconds": 30,
        }

    def get_or_create_state(self, symbol: str) -> SymbolState:
        """Get or create symbol state"""
        symbol = symbol.upper()
        if symbol not in self.symbols:
            self.symbols[symbol] = SymbolState(symbol=symbol)
        return self.symbols[symbol]

    def register_position(self, symbol: str, entry_price: float):
        """Register a position for monitoring"""
        state = self.get_or_create_state(symbol)
        state.entry_price = entry_price
        state.entry_time = datetime.now()
        state.high_since_entry = entry_price
        logger.info(f"Exhaustion detector: Registered {symbol} @ ${entry_price:.2f}")

    def unregister_position(self, symbol: str):
        """Unregister a position"""
        symbol = symbol.upper()
        if symbol in self.symbols:
            state = self.symbols[symbol]
            state.entry_price = 0.0
            state.entry_time = None
            state.high_since_entry = 0.0
            state.active_alerts = []

    def update_candle(self, symbol: str, candle: Dict) -> Optional[ExhaustionAlert]:
        """
        Update with new candle data.

        Candle should have: open, high, low, close, volume
        """
        symbol = symbol.upper()
        state = self.get_or_create_state(symbol)

        # Extract candle data
        open_price = candle.get("open", candle.get("o", 0))
        high = candle.get("high", candle.get("h", 0))
        low = candle.get("low", candle.get("l", 0))
        close = candle.get("close", candle.get("c", 0))
        volume = candle.get("volume", candle.get("v", 0))

        if not all([open_price, high, low, close]):
            return None

        # Store candle
        state.candles.append(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "time": datetime.now(),
            }
        )
        state.prices.append(close)
        state.volumes.append(volume)

        # Update high since entry
        if state.entry_price > 0 and high > state.high_since_entry:
            state.high_since_entry = high

        # Update RSI
        self._update_rsi(state, close)

        # Check for red candle
        is_red = close < open_price
        if is_red:
            state.consecutive_red_candles += 1
        else:
            state.consecutive_red_candles = 0

        # Detect price highs for divergence
        self._detect_highs(state, high, close)

        # Check for exhaustion signals
        return self._check_exhaustion(state, close, volume)

    def update_quote(
        self, symbol: str, bid: float, ask: float
    ) -> Optional[ExhaustionAlert]:
        """Update with quote data for spread analysis"""
        symbol = symbol.upper()
        state = self.get_or_create_state(symbol)

        if bid <= 0 or ask <= 0:
            return None

        # Calculate spread
        spread = ask - bid
        mid = (bid + ask) / 2
        spread_pct = (spread / mid) * 100 if mid > 0 else 0

        state.recent_spreads.append(spread_pct)

        # Calculate average spread
        if len(state.recent_spreads) >= 5:
            state.avg_spread = np.mean(list(state.recent_spreads)[:10])

        # Check for spread widening
        if state.avg_spread > 0 and len(state.recent_spreads) >= 10:
            current_spread = spread_pct
            spread_ratio = current_spread / state.avg_spread

            if spread_ratio >= self.config["spread_critical"]:
                return self._create_alert(
                    state,
                    ExhaustionSignal.SPREAD_WIDENING,
                    ExhaustionSeverity.HIGH,
                    85.0,
                    f"Spread widened {spread_ratio:.1f}x (${spread:.4f})",
                    should_exit=True,
                    spread_pct=spread_pct,
                )
            elif spread_ratio >= self.config["spread_widen_threshold"]:
                return self._create_alert(
                    state,
                    ExhaustionSignal.SPREAD_WIDENING,
                    ExhaustionSeverity.MEDIUM,
                    65.0,
                    f"Spread widening {spread_ratio:.1f}x",
                    should_exit=False,
                    spread_pct=spread_pct,
                )

        return None

    def _update_rsi(self, state: SymbolState, close: float):
        """Update RSI calculation"""
        if len(state.prices) < 2:
            return

        prev_close = state.prices[-2]
        change = close - prev_close

        gain = max(0, change)
        loss = max(0, -change)

        state.gains.append(gain)
        state.losses.append(loss)

        if len(state.gains) >= self.config["rsi_period"]:
            avg_gain = np.mean(list(state.gains))
            avg_loss = np.mean(list(state.losses))

            if avg_loss == 0:
                state.current_rsi = 100
            else:
                rs = avg_gain / avg_loss
                state.current_rsi = 100 - (100 / (1 + rs))

    def _detect_highs(self, state: SymbolState, high: float, close: float):
        """Detect price highs for divergence analysis"""
        if len(state.candles) < 3:
            return

        # Check if previous candle was a local high
        candles = list(state.candles)
        if len(candles) >= 3:
            prev = candles[-2]
            prev_prev = candles[-3]
            current = candles[-1]

            # Previous candle is a high if its high is greater than surrounding
            if prev["high"] > prev_prev["high"] and prev["high"] > current["high"]:
                point = PricePoint(
                    price=prev["high"],
                    rsi=state.current_rsi,
                    volume=prev["volume"],
                    timestamp=prev["time"],
                    is_high=True,
                )
                state.price_highs.append(point)
                state.rsi_at_highs.append(state.current_rsi)

                # Keep only recent highs
                if len(state.price_highs) > 10:
                    state.price_highs = state.price_highs[-10:]
                    state.rsi_at_highs = state.rsi_at_highs[-10:]

        # Track last high
        if high > state.last_high:
            state.last_high = high

    def _check_exhaustion(
        self, state: SymbolState, price: float, volume: int
    ) -> Optional[ExhaustionAlert]:
        """Check all exhaustion signals"""
        alerts = []

        # 1. Check RSI divergence
        rsi_alert = self._check_rsi_divergence(state, price)
        if rsi_alert:
            alerts.append(rsi_alert)

        # 2. Check volume exhaustion
        vol_alert = self._check_volume_exhaustion(state, price, volume)
        if vol_alert:
            alerts.append(vol_alert)

        # 3. Check consecutive red candles
        red_alert = self._check_red_candles(state, price)
        if red_alert:
            alerts.append(red_alert)

        # 4. Check price stalling
        stall_alert = self._check_price_stalling(state, price)
        if stall_alert:
            alerts.append(stall_alert)

        # 5. Check for climax top
        climax_alert = self._check_climax_top(state, price, volume)
        if climax_alert:
            alerts.append(climax_alert)

        # Return highest severity alert
        if alerts:
            # Sort by severity
            severity_order = {
                ExhaustionSeverity.CRITICAL: 0,
                ExhaustionSeverity.HIGH: 1,
                ExhaustionSeverity.MEDIUM: 2,
                ExhaustionSeverity.LOW: 3,
            }
            alerts.sort(key=lambda x: severity_order[x.severity])
            return alerts[0]

        return None

    def _check_rsi_divergence(
        self, state: SymbolState, price: float
    ) -> Optional[ExhaustionAlert]:
        """Check for bearish RSI divergence"""
        if len(state.price_highs) < 2:
            return None

        # Get last two highs
        recent_highs = state.price_highs[-2:]

        # Bearish divergence: price making higher high, RSI making lower high
        if len(recent_highs) >= 2:
            prev_high = recent_highs[-2]
            curr_high = recent_highs[-1]

            price_higher = curr_high.price > prev_high.price
            rsi_lower = (
                curr_high.rsi < prev_high.rsi - self.config["rsi_divergence_threshold"]
            )

            if price_higher and rsi_lower:
                confidence = min(90, 60 + (prev_high.rsi - curr_high.rsi))
                severity = (
                    ExhaustionSeverity.HIGH
                    if confidence > 75
                    else ExhaustionSeverity.MEDIUM
                )

                return self._create_alert(
                    state,
                    ExhaustionSignal.RSI_DIVERGENCE,
                    severity,
                    confidence,
                    f"Bearish divergence: Price higher, RSI {prev_high.rsi:.0f} -> {curr_high.rsi:.0f}",
                    should_exit=True,
                    rsi=state.current_rsi,
                )

        return None

    def _check_volume_exhaustion(
        self, state: SymbolState, price: float, volume: int
    ) -> Optional[ExhaustionAlert]:
        """Check for declining volume on rising price"""
        if len(state.volumes) < self.config["volume_lookback"] + 2:
            return None

        if len(state.prices) < self.config["volume_lookback"] + 2:
            return None

        # Check if price is rising
        prices = list(state.prices)
        recent_prices = prices[-self.config["volume_lookback"] :]
        price_rising = recent_prices[-1] > recent_prices[0]

        if not price_rising:
            return None

        # Check volume trend
        volumes = list(state.volumes)
        recent_vols = volumes[-self.config["volume_lookback"] :]

        # Calculate volume trend (negative = declining)
        if len(recent_vols) >= 3:
            vol_change = (
                (recent_vols[-1] - recent_vols[0]) / recent_vols[0]
                if recent_vols[0] > 0
                else 0
            )

            if vol_change < -(1 - self.config["volume_decline_threshold"]):
                confidence = min(85, 50 + abs(vol_change) * 50)
                severity = ExhaustionSeverity.MEDIUM

                if vol_change < -0.5:  # 50%+ decline
                    severity = ExhaustionSeverity.HIGH

                return self._create_alert(
                    state,
                    ExhaustionSignal.VOLUME_EXHAUSTION,
                    severity,
                    confidence,
                    f"Volume declining {abs(vol_change)*100:.0f}% while price rising",
                    should_exit=severity == ExhaustionSeverity.HIGH,
                    volume_trend=vol_change,
                )

        return None

    def _check_red_candles(
        self, state: SymbolState, price: float
    ) -> Optional[ExhaustionAlert]:
        """Check for consecutive red candles"""
        red_count = state.consecutive_red_candles

        if red_count >= self.config["red_candle_critical"]:
            return self._create_alert(
                state,
                ExhaustionSignal.RED_CANDLES,
                ExhaustionSeverity.HIGH,
                85.0,
                f"{red_count} consecutive red candles - momentum dead",
                should_exit=True,
                consecutive_red=red_count,
            )
        elif red_count >= self.config["red_candle_threshold"]:
            return self._create_alert(
                state,
                ExhaustionSignal.RED_CANDLES,
                ExhaustionSeverity.MEDIUM,
                65.0,
                f"{red_count} consecutive red candles - momentum fading",
                should_exit=False,
                consecutive_red=red_count,
            )

        return None

    def _check_price_stalling(
        self, state: SymbolState, price: float
    ) -> Optional[ExhaustionAlert]:
        """Check for price stalling at resistance"""
        if state.last_high <= 0:
            return None

        # Check if price is near high but not breaking through
        distance_from_high = (state.last_high - price) / state.last_high * 100

        if distance_from_high <= self.config["stall_range_pct"]:
            # Count failed attempts to break high
            state.failed_breakout_attempts += 1

            if (
                state.failed_breakout_attempts
                >= self.config["stall_attempts_threshold"]
            ):
                return self._create_alert(
                    state,
                    ExhaustionSignal.PRICE_STALLING,
                    ExhaustionSeverity.MEDIUM,
                    70.0,
                    f"{state.failed_breakout_attempts} failed breakout attempts at ${state.last_high:.2f}",
                    should_exit=False,
                )
        else:
            # Reset if price moves away
            state.failed_breakout_attempts = 0

        return None

    def _check_climax_top(
        self, state: SymbolState, price: float, volume: int
    ) -> Optional[ExhaustionAlert]:
        """Check for climax/blow-off top pattern"""
        if len(state.candles) < 5:
            return None

        candles = list(state.candles)[-5:]

        # Climax top characteristics:
        # 1. Current candle has huge volume (2x+ average)
        # 2. Long upper wick (rejection)
        # 3. RSI overbought

        current = candles[-1]
        avg_vol = np.mean([c["volume"] for c in candles[:-1]])

        if avg_vol <= 0:
            return None

        vol_ratio = current["volume"] / avg_vol

        # Calculate upper wick ratio
        body = abs(current["close"] - current["open"])
        upper_wick = current["high"] - max(current["open"], current["close"])
        total_range = current["high"] - current["low"]

        wick_ratio = upper_wick / total_range if total_range > 0 else 0

        # Climax conditions
        is_climax = (
            vol_ratio > 2.0
            and wick_ratio > 0.4
            and state.current_rsi > self.config["rsi_overbought"]
        )

        if is_climax:
            return self._create_alert(
                state,
                ExhaustionSignal.CLIMAX_TOP,
                ExhaustionSeverity.CRITICAL,
                90.0,
                f"Climax top: {vol_ratio:.1f}x volume, {wick_ratio*100:.0f}% wick, RSI {state.current_rsi:.0f}",
                should_exit=True,
                rsi=state.current_rsi,
                volume_trend=vol_ratio,
            )

        return None

    def _create_alert(
        self,
        state: SymbolState,
        signal: ExhaustionSignal,
        severity: ExhaustionSeverity,
        confidence: float,
        details: str,
        should_exit: bool,
        rsi: float = 0,
        volume_trend: float = 0,
        consecutive_red: int = 0,
        spread_pct: float = 0,
    ) -> Optional[ExhaustionAlert]:
        """Create an exhaustion alert with cooldown check"""

        # Check cooldown
        if state.last_alert_time:
            elapsed = (datetime.now() - state.last_alert_time).total_seconds()
            if elapsed < self.config["alert_cooldown_seconds"]:
                return None

        alert = ExhaustionAlert(
            symbol=state.symbol,
            signal=signal,
            severity=severity,
            confidence=confidence,
            details=details,
            should_exit=should_exit,
            timestamp=datetime.now().isoformat(),
            current_price=float(state.prices[-1]) if state.prices else 0,
            rsi=rsi or state.current_rsi,
            volume_trend=volume_trend,
            consecutive_red=consecutive_red or state.consecutive_red_candles,
            spread_pct=spread_pct,
        )

        state.last_alert_time = datetime.now()
        state.active_alerts.append(alert)

        # Store in history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history :]

        # Log alert
        logger.info(
            f"EXHAUSTION {severity.value}: {state.symbol} - {signal.value} - {details} "
            f"(exit={should_exit}, conf={confidence:.0f}%)"
        )

        # Fire callback
        if self.on_exhaustion_alert:
            try:
                self.on_exhaustion_alert(alert)
            except Exception as e:
                logger.error(f"Exhaustion alert callback error: {e}")

        return alert

    def check_exit(
        self, symbol: str, current_price: float
    ) -> Optional[ExhaustionAlert]:
        """
        Quick check if should exit based on current exhaustion signals.

        Used by HFT Scalper for exit decisions.
        """
        symbol = symbol.upper()
        state = self.symbols.get(symbol)

        if not state or not state.active_alerts:
            return None

        # Return most recent high-severity alert
        for alert in reversed(state.active_alerts):
            if alert.should_exit:
                return alert

        return None

    def get_exhaustion_score(self, symbol: str) -> Tuple[float, List[str]]:
        """
        Get overall exhaustion score (0-100) and reasons.

        Higher score = more exhausted = more likely to reverse
        """
        symbol = symbol.upper()
        state = self.symbols.get(symbol)

        if not state:
            return 0.0, []

        score = 0.0
        reasons = []

        # RSI component
        if state.current_rsi > 70:
            rsi_score = (state.current_rsi - 70) * 1.5
            score += rsi_score
            reasons.append(f"RSI overbought ({state.current_rsi:.0f})")

        # Red candles component
        if state.consecutive_red_candles >= 2:
            red_score = state.consecutive_red_candles * 10
            score += red_score
            reasons.append(f"{state.consecutive_red_candles} red candles")

        # Volume trend
        if len(state.volumes) >= 5:
            vols = list(state.volumes)[-5:]
            if vols[0] > 0:
                vol_change = (vols[-1] - vols[0]) / vols[0]
                if vol_change < -0.3:
                    vol_score = abs(vol_change) * 30
                    score += vol_score
                    reasons.append(f"Volume declining {abs(vol_change)*100:.0f}%")

        # Spread component
        if state.avg_spread > 0 and len(state.recent_spreads) >= 5:
            current = state.recent_spreads[-1] if state.recent_spreads else 0
            spread_ratio = current / state.avg_spread if state.avg_spread > 0 else 1
            if spread_ratio > 1.3:
                spread_score = (spread_ratio - 1) * 20
                score += spread_score
                reasons.append(f"Spread widening {spread_ratio:.1f}x")

        # Active alerts component
        for alert in state.active_alerts:
            if alert.severity == ExhaustionSeverity.CRITICAL:
                score += 30
            elif alert.severity == ExhaustionSeverity.HIGH:
                score += 20
            elif alert.severity == ExhaustionSeverity.MEDIUM:
                score += 10

        return min(100, score), reasons

    def get_status(self) -> Dict:
        """Get detector status"""
        exhausted = [
            s
            for s in self.symbols.values()
            if any(a.should_exit for a in s.active_alerts)
        ]

        return {
            "symbols_tracked": len(self.symbols),
            "symbols_with_positions": len(
                [s for s in self.symbols.values() if s.entry_price > 0]
            ),
            "exhausted_symbols": len(exhausted),
            "total_alerts": len(self.alert_history),
            "config": self.config,
            "exhausted_list": [
                {"symbol": s.symbol, "score": self.get_exhaustion_score(s.symbol)[0]}
                for s in exhausted
            ],
        }

    def get_recent_alerts(self, limit: int = 20) -> List[ExhaustionAlert]:
        """Get recent alerts"""
        return self.alert_history[-limit:][::-1]


# Singleton instance
_exhaustion_detector: Optional[MomentumExhaustionDetector] = None


def get_exhaustion_detector() -> MomentumExhaustionDetector:
    """Get singleton exhaustion detector"""
    global _exhaustion_detector
    if _exhaustion_detector is None:
        _exhaustion_detector = MomentumExhaustionDetector()
    return _exhaustion_detector


# Convenience functions
def check_exhaustion(symbol: str, price: float) -> Optional[ExhaustionAlert]:
    """Quick check for exhaustion exit signal"""
    return get_exhaustion_detector().check_exit(symbol, price)


def get_exhaustion_score(symbol: str) -> Tuple[float, List[str]]:
    """Get exhaustion score (0-100) and reasons"""
    return get_exhaustion_detector().get_exhaustion_score(symbol)
