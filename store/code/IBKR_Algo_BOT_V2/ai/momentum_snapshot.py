"""
Momentum Snapshot Engine (Execution-Local)
==========================================
Real-time momentum measurement for execution decisions.

NOT Chronos-based - uses raw price/volume data to compute:
- Rate of Change: 5s / 30s / 2m
- Relative Volume: 1m / 5m
- VWAP distance and reclaim flag
- Micro-trend (higher highs/lows or lower highs/lows)
- Spread and liquidity score

Output States:
- IGNITION: Initial momentum detected, not yet confirmed
- CONFIRMED: Momentum confirmed with follow-through
- DECAY: Momentum weakening, prepare for exit
- DEAD: Momentum failed, forced exit or no-trade

Rules:
- Entry requires CONFIRMED state
- IGNITION without CONFIRMED within timeout → veto
- DECAY → prepare exit
- DEAD → forced exit or no-trade

All state transitions are logged.
"""

import logging
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)

SNAPSHOT_LOG_FILE = os.path.join(os.path.dirname(__file__), "momentum_snapshot_log.json")


class MomentumOutputState(Enum):
    """Momentum output states for execution decisions"""
    IGNITION = "IGNITION"      # Initial momentum detected
    CONFIRMED = "CONFIRMED"    # Momentum confirmed with follow-through
    DECAY = "DECAY"            # Momentum weakening
    DEAD = "DEAD"              # Momentum failed


@dataclass
class PricePoint:
    """Single price observation"""
    timestamp: datetime
    price: float
    volume: int
    bid: float = 0
    ask: float = 0


@dataclass
class MomentumSnapshot:
    """
    Point-in-time momentum measurement for a symbol.

    All metrics are computed from recent price/volume data.
    """
    symbol: str
    timestamp: datetime

    # Current prices
    current_price: float = 0
    bid: float = 0
    ask: float = 0

    # Rate of Change (percentage)
    roc_5s: float = 0       # 5-second rate of change
    roc_30s: float = 0      # 30-second rate of change
    roc_2m: float = 0       # 2-minute rate of change

    # Relative Volume
    rvol_1m: float = 1.0    # 1-minute relative volume
    rvol_5m: float = 1.0    # 5-minute relative volume

    # VWAP metrics
    vwap: float = 0
    vwap_distance_pct: float = 0    # % distance from VWAP
    above_vwap: bool = False
    vwap_reclaim: bool = False      # Just reclaimed VWAP

    # Micro-trend
    higher_highs: bool = False      # Making higher highs
    higher_lows: bool = False       # Making higher lows
    lower_highs: bool = False       # Making lower highs
    lower_lows: bool = False        # Making lower lows
    micro_trend: str = "NEUTRAL"    # BULLISH, BEARISH, NEUTRAL

    # Spread and liquidity
    spread_pct: float = 0           # Bid-ask spread %
    spread_score: int = 100         # 0-100 liquidity score (100=tight)

    # Computed output state
    output_state: MomentumOutputState = MomentumOutputState.IGNITION
    state_confidence: float = 0.5

    # Timing
    time_in_state_seconds: float = 0
    state_entered_at: datetime = None

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'current_price': self.current_price,
            'roc_5s': round(self.roc_5s, 3),
            'roc_30s': round(self.roc_30s, 3),
            'roc_2m': round(self.roc_2m, 3),
            'rvol_1m': round(self.rvol_1m, 2),
            'rvol_5m': round(self.rvol_5m, 2),
            'vwap': round(self.vwap, 2),
            'vwap_distance_pct': round(self.vwap_distance_pct, 2),
            'above_vwap': self.above_vwap,
            'vwap_reclaim': self.vwap_reclaim,
            'micro_trend': self.micro_trend,
            'higher_highs': self.higher_highs,
            'higher_lows': self.higher_lows,
            'spread_pct': round(self.spread_pct, 3),
            'spread_score': self.spread_score,
            'output_state': self.output_state.value,
            'state_confidence': round(self.state_confidence, 2),
            'time_in_state_seconds': round(self.time_in_state_seconds, 1),
        }


@dataclass
class StateTransitionLog:
    """Log entry for momentum state transition"""
    symbol: str
    timestamp: str
    from_state: str
    to_state: str
    reason: str
    snapshot: Dict


class MomentumSnapshotEngine:
    """
    Computes MomentumSnapshot for symbols based on price/volume history.

    Key principles:
    1. Execution-local only - no external predictions
    2. All state transitions logged
    3. CONFIRMED required for entry
    4. DECAY/DEAD trigger exit preparation
    """

    # Thresholds for state determination
    IGNITION_ROC_5S = 0.3           # 0.3% move in 5s = ignition
    CONFIRMED_ROC_30S = 0.5         # 0.5% in 30s with volume
    CONFIRMED_RVOL = 2.0            # 2x relative volume for confirmation
    DECAY_ROC_REVERSAL = -0.2       # -0.2% = decay starting
    DEAD_ROC_THRESHOLD = -0.5       # -0.5% from high = dead

    IGNITION_TIMEOUT_SECONDS = 30   # Must confirm within 30s
    VWAP_RECLAIM_THRESHOLD = 0.1    # % above VWAP to count as reclaim

    SPREAD_TIGHT = 0.5              # < 0.5% = tight spread
    SPREAD_WIDE = 2.0               # > 2% = wide spread (danger)

    def __init__(self):
        # Price history per symbol (last 5 minutes of ticks)
        self._price_history: Dict[str, deque] = {}

        # Current state per symbol
        self._states: Dict[str, MomentumOutputState] = {}
        self._state_entered: Dict[str, datetime] = {}
        self._last_snapshot: Dict[str, MomentumSnapshot] = {}

        # Volume baselines (average volume for comparison)
        self._volume_baselines: Dict[str, Dict[str, float]] = {}

        # Previous VWAP position for reclaim detection
        self._prev_above_vwap: Dict[str, bool] = {}

        # Transition log
        self._transition_log: List[StateTransitionLog] = []

        # High watermarks for decay detection
        self._session_highs: Dict[str, float] = {}

    def add_price(self, symbol: str, price: float, volume: int = 0,
                  bid: float = 0, ask: float = 0):
        """
        Add a new price observation for a symbol.

        Should be called frequently (every tick or every few seconds).
        """
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=600)  # 5 min @ 2/sec
            self._states[symbol] = MomentumOutputState.DEAD
            self._state_entered[symbol] = datetime.now()

        point = PricePoint(
            timestamp=datetime.now(),
            price=price,
            volume=volume,
            bid=bid,
            ask=ask
        )
        self._price_history[symbol].append(point)

        # Update session high
        if symbol not in self._session_highs or price > self._session_highs[symbol]:
            self._session_highs[symbol] = price

    def set_volume_baseline(self, symbol: str, avg_1m: float, avg_5m: float):
        """Set baseline volumes for relative volume calculation"""
        self._volume_baselines[symbol] = {
            'avg_1m': avg_1m,
            'avg_5m': avg_5m
        }

    def compute_snapshot(self, symbol: str) -> Optional[MomentumSnapshot]:
        """
        Compute current MomentumSnapshot for a symbol.

        Returns None if insufficient data.
        """
        if symbol not in self._price_history:
            return None

        history = self._price_history[symbol]
        if len(history) < 5:  # Need minimum data
            return None

        now = datetime.now()
        current = history[-1]

        # Compute ROC at different intervals
        roc_5s = self._compute_roc(history, 5)
        roc_30s = self._compute_roc(history, 30)
        roc_2m = self._compute_roc(history, 120)

        # Compute relative volume
        rvol_1m, rvol_5m = self._compute_rvol(symbol, history)

        # Compute VWAP metrics
        vwap = self._compute_vwap(history)
        above_vwap = current.price > vwap if vwap > 0 else False
        vwap_distance = ((current.price - vwap) / vwap * 100) if vwap > 0 else 0

        # Detect VWAP reclaim
        prev_above = self._prev_above_vwap.get(symbol, False)
        vwap_reclaim = above_vwap and not prev_above and vwap_distance > self.VWAP_RECLAIM_THRESHOLD
        self._prev_above_vwap[symbol] = above_vwap

        # Compute micro-trend
        micro_trend, hh, hl, lh, ll = self._compute_micro_trend(history)

        # Compute spread metrics
        spread_pct = 0
        spread_score = 100
        if current.bid > 0 and current.ask > 0:
            spread_pct = (current.ask - current.bid) / current.bid * 100
            if spread_pct < self.SPREAD_TIGHT:
                spread_score = 100
            elif spread_pct > self.SPREAD_WIDE:
                spread_score = 0
            else:
                spread_score = int(100 * (1 - (spread_pct - self.SPREAD_TIGHT) /
                                          (self.SPREAD_WIDE - self.SPREAD_TIGHT)))

        # Determine output state
        prev_state = self._states.get(symbol, MomentumOutputState.DEAD)
        new_state, confidence, reason = self._determine_state(
            symbol, roc_5s, roc_30s, roc_2m, rvol_1m, above_vwap,
            vwap_reclaim, micro_trend, spread_score, prev_state
        )

        # Handle state transition
        if new_state != prev_state:
            self._log_transition(symbol, prev_state, new_state, reason, current.price)
            self._states[symbol] = new_state
            self._state_entered[symbol] = now

        time_in_state = (now - self._state_entered.get(symbol, now)).total_seconds()

        snapshot = MomentumSnapshot(
            symbol=symbol,
            timestamp=now,
            current_price=current.price,
            bid=current.bid,
            ask=current.ask,
            roc_5s=roc_5s,
            roc_30s=roc_30s,
            roc_2m=roc_2m,
            rvol_1m=rvol_1m,
            rvol_5m=rvol_5m,
            vwap=vwap,
            vwap_distance_pct=vwap_distance,
            above_vwap=above_vwap,
            vwap_reclaim=vwap_reclaim,
            higher_highs=hh,
            higher_lows=hl,
            lower_highs=lh,
            lower_lows=ll,
            micro_trend=micro_trend,
            spread_pct=spread_pct,
            spread_score=spread_score,
            output_state=new_state,
            state_confidence=confidence,
            time_in_state_seconds=time_in_state,
            state_entered_at=self._state_entered.get(symbol, now),
        )

        self._last_snapshot[symbol] = snapshot
        return snapshot

    def _compute_roc(self, history: deque, seconds: int) -> float:
        """Compute rate of change over specified seconds"""
        if len(history) < 2:
            return 0

        now = datetime.now()
        cutoff = now - timedelta(seconds=seconds)

        # Find price at cutoff time
        old_price = None
        for point in history:
            if point.timestamp >= cutoff:
                old_price = point.price
                break

        if old_price is None or old_price == 0:
            old_price = history[0].price

        current = history[-1].price
        if old_price == 0:
            return 0

        return ((current - old_price) / old_price) * 100

    def _compute_rvol(self, symbol: str, history: deque) -> tuple:
        """Compute relative volume for 1m and 5m"""
        now = datetime.now()

        # Sum volume in last 1m and 5m
        vol_1m = 0
        vol_5m = 0
        cutoff_1m = now - timedelta(minutes=1)
        cutoff_5m = now - timedelta(minutes=5)

        for point in history:
            if point.timestamp >= cutoff_1m:
                vol_1m += point.volume
            if point.timestamp >= cutoff_5m:
                vol_5m += point.volume

        # Get baselines
        baselines = self._volume_baselines.get(symbol, {'avg_1m': 1, 'avg_5m': 1})

        rvol_1m = vol_1m / max(baselines['avg_1m'], 1)
        rvol_5m = vol_5m / max(baselines['avg_5m'], 1)

        return rvol_1m, rvol_5m

    def _compute_vwap(self, history: deque) -> float:
        """Compute VWAP from history"""
        total_pv = 0
        total_v = 0

        for point in history:
            if point.volume > 0:
                total_pv += point.price * point.volume
                total_v += point.volume

        if total_v == 0:
            # No volume data - use simple average
            if len(history) > 0:
                return sum(p.price for p in history) / len(history)
            return 0

        return total_pv / total_v

    def _compute_micro_trend(self, history: deque) -> tuple:
        """
        Compute micro-trend from recent price action.

        Returns: (trend, higher_highs, higher_lows, lower_highs, lower_lows)
        """
        if len(history) < 10:
            return "NEUTRAL", False, False, False, False

        # Get last 10 prices, split into 3 groups
        prices = [p.price for p in list(history)[-10:]]

        # Find highs and lows in each third
        thirds = [prices[0:3], prices[3:7], prices[7:10]]
        highs = [max(t) for t in thirds]
        lows = [min(t) for t in thirds]

        # Check for higher highs / higher lows (bullish)
        higher_highs = highs[2] > highs[1] > highs[0]
        higher_lows = lows[2] > lows[1] > lows[0]

        # Check for lower highs / lower lows (bearish)
        lower_highs = highs[2] < highs[1] < highs[0]
        lower_lows = lows[2] < lows[1] < lows[0]

        # Determine trend
        if higher_highs and higher_lows:
            trend = "BULLISH"
        elif lower_highs and lower_lows:
            trend = "BEARISH"
        elif higher_highs or higher_lows:
            trend = "BULLISH"
        elif lower_highs or lower_lows:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        return trend, higher_highs, higher_lows, lower_highs, lower_lows

    def _determine_state(self, symbol: str, roc_5s: float, roc_30s: float,
                         roc_2m: float, rvol: float, above_vwap: bool,
                         vwap_reclaim: bool, micro_trend: str,
                         spread_score: int, prev_state: MomentumOutputState
                        ) -> tuple:
        """
        Determine momentum output state.

        Returns: (state, confidence, reason)
        """
        # Check for DEAD first (failed momentum)
        session_high = self._session_highs.get(symbol, 0)
        current_price = self._last_snapshot[symbol].current_price if symbol in self._last_snapshot else 0

        pct_from_high = 0
        if session_high > 0 and current_price > 0:
            pct_from_high = ((current_price - session_high) / session_high) * 100

        # DEAD: Significant drop from high OR wide spread OR bearish trend
        if pct_from_high < self.DEAD_ROC_THRESHOLD:
            return MomentumOutputState.DEAD, 0.9, f"Dropped {pct_from_high:.1f}% from high"

        if spread_score == 0:
            return MomentumOutputState.DEAD, 0.85, "Spread too wide"

        if micro_trend == "BEARISH" and roc_30s < -0.3:
            return MomentumOutputState.DEAD, 0.8, "Bearish trend with negative ROC"

        # DECAY: Momentum weakening
        if roc_5s < self.DECAY_ROC_REVERSAL and prev_state == MomentumOutputState.CONFIRMED:
            return MomentumOutputState.DECAY, 0.75, f"ROC reversal: {roc_5s:.2f}%"

        if not above_vwap and prev_state == MomentumOutputState.CONFIRMED:
            return MomentumOutputState.DECAY, 0.7, "Lost VWAP"

        if micro_trend == "BEARISH" and prev_state == MomentumOutputState.CONFIRMED:
            return MomentumOutputState.DECAY, 0.65, "Micro-trend turned bearish"

        # Check for IGNITION timeout
        if prev_state == MomentumOutputState.IGNITION:
            time_in_ignition = (datetime.now() -
                               self._state_entered.get(symbol, datetime.now())).total_seconds()
            if time_in_ignition > self.IGNITION_TIMEOUT_SECONDS:
                return MomentumOutputState.DEAD, 0.8, f"Ignition timeout ({time_in_ignition:.0f}s)"

        # CONFIRMED: Strong momentum with follow-through
        if (roc_30s >= self.CONFIRMED_ROC_30S and
            rvol >= self.CONFIRMED_RVOL and
            micro_trend == "BULLISH" and
            above_vwap):
            confidence = min(0.5 + (roc_30s / 2) + (rvol / 10), 0.95)
            return MomentumOutputState.CONFIRMED, confidence, \
                   f"ROC={roc_30s:.1f}%, RVol={rvol:.1f}x, Bullish, Above VWAP"

        # Partial confirmation (3 of 4 criteria)
        criteria_met = sum([
            roc_30s >= self.CONFIRMED_ROC_30S,
            rvol >= self.CONFIRMED_RVOL,
            micro_trend == "BULLISH",
            above_vwap
        ])

        if criteria_met >= 3 and prev_state in [MomentumOutputState.IGNITION,
                                                 MomentumOutputState.CONFIRMED]:
            return MomentumOutputState.CONFIRMED, 0.7, f"Partial confirmation ({criteria_met}/4 criteria)"

        # IGNITION: Initial momentum detected
        if roc_5s >= self.IGNITION_ROC_5S or vwap_reclaim:
            reason = []
            if roc_5s >= self.IGNITION_ROC_5S:
                reason.append(f"ROC 5s={roc_5s:.2f}%")
            if vwap_reclaim:
                reason.append("VWAP reclaim")
            return MomentumOutputState.IGNITION, 0.6, ", ".join(reason)

        # Stay in current state if nothing triggered
        if prev_state == MomentumOutputState.CONFIRMED:
            return MomentumOutputState.CONFIRMED, 0.6, "Holding confirmation"
        if prev_state == MomentumOutputState.IGNITION:
            return MomentumOutputState.IGNITION, 0.5, "Awaiting confirmation"

        # Default: DEAD
        return MomentumOutputState.DEAD, 0.5, "No momentum detected"

    def _log_transition(self, symbol: str, from_state: MomentumOutputState,
                        to_state: MomentumOutputState, reason: str, price: float):
        """Log a state transition"""
        entry = StateTransitionLog(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            from_state=from_state.value,
            to_state=to_state.value,
            reason=reason,
            snapshot={'price': price}
        )
        self._transition_log.append(entry)

        # Keep last 1000 entries
        if len(self._transition_log) > 1000:
            self._transition_log = self._transition_log[-1000:]

        # Log to file
        try:
            log_data = []
            if os.path.exists(SNAPSHOT_LOG_FILE):
                with open(SNAPSHOT_LOG_FILE, 'r') as f:
                    log_data = json.load(f)

            log_data.append(asdict(entry))

            # Keep last 500 in file
            if len(log_data) > 500:
                log_data = log_data[-500:]

            with open(SNAPSHOT_LOG_FILE, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to log transition: {e}")

        logger.info(f"MOMENTUM STATE: {symbol} {from_state.value} -> {to_state.value} ({reason})")

    def can_enter(self, symbol: str) -> tuple:
        """
        Check if symbol is eligible for entry.

        Returns: (can_enter: bool, reason: str)

        Entry requires CONFIRMED state.
        """
        state = self._states.get(symbol, MomentumOutputState.DEAD)

        if state == MomentumOutputState.CONFIRMED:
            return True, "Momentum CONFIRMED"

        if state == MomentumOutputState.IGNITION:
            time_in_state = (datetime.now() -
                            self._state_entered.get(symbol, datetime.now())).total_seconds()
            return False, f"Awaiting confirmation (in IGNITION {time_in_state:.0f}s)"

        if state == MomentumOutputState.DECAY:
            return False, "Momentum DECAY - no entry"

        return False, f"Momentum DEAD - no entry ({state.value})"

    def should_exit(self, symbol: str) -> tuple:
        """
        Check if position should exit based on momentum.

        Returns: (should_exit: bool, reason: str, urgency: str)

        Urgency: "IMMEDIATE" for DEAD, "PREPARE" for DECAY
        """
        state = self._states.get(symbol, MomentumOutputState.DEAD)

        if state == MomentumOutputState.DEAD:
            return True, "Momentum DEAD", "IMMEDIATE"

        if state == MomentumOutputState.DECAY:
            snapshot = self._last_snapshot.get(symbol)
            if snapshot and not snapshot.above_vwap:
                return True, "DECAY + lost VWAP", "IMMEDIATE"
            return True, "Momentum DECAY", "PREPARE"

        return False, "Momentum healthy", "NONE"

    def get_state(self, symbol: str) -> MomentumOutputState:
        """Get current momentum state for symbol"""
        return self._states.get(symbol, MomentumOutputState.DEAD)

    def get_snapshot(self, symbol: str) -> Optional[MomentumSnapshot]:
        """Get last computed snapshot for symbol"""
        return self._last_snapshot.get(symbol)

    def get_all_states(self) -> Dict[str, str]:
        """Get all symbol states"""
        return {s: state.value for s, state in self._states.items()}

    def get_transition_log(self, symbol: str = None, limit: int = 50) -> List[Dict]:
        """Get recent state transitions"""
        log = self._transition_log
        if symbol:
            log = [t for t in log if t.symbol == symbol]
        return [asdict(t) for t in log[-limit:]]

    def get_stats(self) -> Dict:
        """Get momentum engine statistics"""
        state_counts = {}
        for state in MomentumOutputState:
            state_counts[state.value] = sum(1 for s in self._states.values() if s == state)

        # Count transitions by type
        transition_counts = {}
        for t in self._transition_log:
            key = f"{t.from_state}->{t.to_state}"
            transition_counts[key] = transition_counts.get(key, 0) + 1

        return {
            'total_symbols': len(self._states),
            'state_counts': state_counts,
            'transition_counts': transition_counts,
            'total_transitions': len(self._transition_log),
            'symbols_confirmed': [s for s, st in self._states.items()
                                  if st == MomentumOutputState.CONFIRMED],
            'symbols_igniting': [s for s, st in self._states.items()
                                 if st == MomentumOutputState.IGNITION],
        }

    def reset_symbol(self, symbol: str):
        """Reset a symbol to DEAD state"""
        if symbol in self._states:
            prev_state = self._states[symbol]
            self._states[symbol] = MomentumOutputState.DEAD
            self._state_entered[symbol] = datetime.now()
            self._log_transition(symbol, prev_state, MomentumOutputState.DEAD,
                               "Manual reset", 0)

    def clear_all(self):
        """Clear all state (for testing)"""
        self._price_history.clear()
        self._states.clear()
        self._state_entered.clear()
        self._last_snapshot.clear()
        self._session_highs.clear()


# Singleton instance
_engine: Optional[MomentumSnapshotEngine] = None


def get_momentum_snapshot_engine() -> MomentumSnapshotEngine:
    """Get singleton momentum snapshot engine"""
    global _engine
    if _engine is None:
        _engine = MomentumSnapshotEngine()
    return _engine


if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    print("=" * 60)
    print("MOMENTUM SNAPSHOT ENGINE TEST")
    print("=" * 60)

    engine = MomentumSnapshotEngine()
    symbol = "TEST"

    # Simulate price action
    base_price = 5.00
    volume_base = 10000

    engine.set_volume_baseline(symbol, avg_1m=50000, avg_5m=250000)

    print("\n1. Simulating price buildup (DEAD -> IGNITION)...")
    for i in range(20):
        price = base_price + (i * 0.01) + random.uniform(-0.02, 0.03)
        volume = volume_base + random.randint(5000, 15000)
        engine.add_price(symbol, price, volume, bid=price-0.01, ask=price+0.01)

    snapshot = engine.compute_snapshot(symbol)
    print(f"   State: {snapshot.output_state.value}")
    print(f"   ROC 5s: {snapshot.roc_5s:.2f}%")

    print("\n2. Simulating momentum spike (IGNITION -> CONFIRMED)...")
    for i in range(30):
        # Strong upward move with volume
        price = base_price + 0.20 + (i * 0.02) + random.uniform(0, 0.01)
        volume = volume_base * 3 + random.randint(10000, 30000)
        engine.add_price(symbol, price, volume, bid=price-0.01, ask=price+0.01)
        snapshot = engine.compute_snapshot(symbol)

    print(f"   State: {snapshot.output_state.value}")
    print(f"   ROC 30s: {snapshot.roc_30s:.2f}%")
    print(f"   RVol 1m: {snapshot.rvol_1m:.2f}x")
    print(f"   Micro-trend: {snapshot.micro_trend}")

    print("\n3. Checking entry eligibility...")
    can_enter, reason = engine.can_enter(symbol)
    print(f"   Can enter: {can_enter}")
    print(f"   Reason: {reason}")

    print("\n4. Simulating decay (CONFIRMED -> DECAY)...")
    for i in range(15):
        # Price starts dropping
        price = base_price + 0.60 - (i * 0.03)
        volume = volume_base
        engine.add_price(symbol, price, volume, bid=price-0.02, ask=price+0.02)
        snapshot = engine.compute_snapshot(symbol)

    print(f"   State: {snapshot.output_state.value}")
    should_exit, exit_reason, urgency = engine.should_exit(symbol)
    print(f"   Should exit: {should_exit}")
    print(f"   Reason: {exit_reason}")
    print(f"   Urgency: {urgency}")

    print("\n5. Getting stats...")
    stats = engine.get_stats()
    print(f"   State counts: {stats['state_counts']}")
    print(f"   Transitions: {stats['transition_counts']}")

    print("\n6. Transition log:")
    for t in engine.get_transition_log(limit=10):
        print(f"   {t['from_state']} -> {t['to_state']}: {t['reason']}")

    print("\n" + "=" * 60)
    print("MOMENTUM SNAPSHOT ENGINE TEST COMPLETE")
    print("=" * 60)
