"""
Two-Phase Entry/Exit Strategy
==============================
Based on user's trading approach - combines volume spike entry with continuation trades.

PHASE 1: VOLUME SPIKE ENTRY (Breaking News / Pop)
-------------------------------------------------
Trigger: Volume surge after breaking news or sudden price pop

Entry Rules:
1. DON'T buy the initial spike - wait for first pullback
2. Watch for pullback to see if it's going to dump
3. Buy at first candle making new HIGH after pullback
4. Entry = price when candle breaks above previous candle high

Stop Loss:
- Stop = difference from bottom of last candle to entry point
- Example: Entry $5.00, last candle low $4.80 -> Stop = $0.20 -> Stop price = $4.80

Targets:
- Target 1 = 2x the stop distance
- Example: Stop = $0.20 -> Target 1 = $5.00 + $0.40 = $5.40

Exit Rules:
- After hitting Target 1: Activate 3% trailing stop
- If price action is erratic: Look for exit indicator in pattern or momentum swing
- Goal: Manage risk but don't take profit too early

PHASE 2: CONTINUATION TRADE
---------------------------
Trigger: After exiting Phase 1 trade, stock shows consolidation then resumes climb

Entry Rules:
1. Wait for pullback or consolidation after Phase 1 exit
2. Look for signs of new move starting:
   - Chart patterns (flags, pennants, wedges)
   - MACD crossover/divergence
   - Volume pickup
   - Momentum indicator turning positive
3. Enter when climb starts with confirmation

Exit Rules:
- Grinding slow rise: Use 3% trailing stop
- If price gets past 2x profit: Continue trading until exit indicator
- Exit indicators: MACD divergence, volume dry up, reversal pattern

INDICATORS TO MONITOR
---------------------
- MACD: Crossovers, divergence
- Volume: Surge on entry, dry up on exit
- Price patterns: Higher highs, higher lows for uptrend
- Momentum: RSI, rate of change
- Support/Resistance levels
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
_macd_analyzer = None


def get_macd_analyzer():
    """Lazy load MACD analyzer"""
    global _macd_analyzer
    if _macd_analyzer is None:
        try:
            from ai.macd_analyzer import get_macd_analyzer as _get

            _macd_analyzer = _get()
        except Exception as e:
            logger.warning(f"Could not load MACD analyzer: {e}")
    return _macd_analyzer


class TradePhase(Enum):
    WAITING = "waiting"  # No position, watching for setup
    PHASE1_WATCHING = "phase1_watching"  # Saw spike, waiting for pullback
    PHASE1_ENTRY = "phase1_entry"  # Looking for first new high after pullback
    PHASE1_ACTIVE = "phase1_active"  # In Phase 1 trade
    PHASE1_TRAILING = "phase1_trailing"  # Hit target 1, now trailing
    PHASE2_WATCHING = "phase2_watching"  # Exited Phase 1, watching for continuation
    PHASE2_ACTIVE = "phase2_active"  # In Phase 2 trade
    PHASE2_TRAILING = "phase2_trailing"  # Trailing in Phase 2


@dataclass
class TwoPhasePosition:
    """Tracks a position through both phases"""

    symbol: str
    phase: TradePhase

    # Phase 1 data
    spike_high: float = 0.0  # Initial spike high
    pullback_low: float = 0.0  # Pullback low
    entry_price: float = 0.0  # Actual entry price
    last_candle_low: float = 0.0  # For calculating stop
    stop_distance: float = 0.0  # Entry - last candle low
    stop_price: float = 0.0  # Actual stop price
    target1_price: float = 0.0  # 2x stop distance target
    trailing_stop_pct: float = 3.0  # Trailing stop percentage
    high_since_entry: float = 0.0  # Track high for trailing

    # Phase 2 data
    phase2_entry: float = 0.0  # Phase 2 entry price
    phase2_high: float = 0.0  # High since Phase 2 entry

    # Indicators at entry
    macd_signal: str = ""  # bullish/bearish/neutral
    volume_surge: float = 0.0  # Volume vs average
    momentum_score: float = 0.0  # Composite momentum


class TwoPhaseStrategy:
    """
    Implements the two-phase entry/exit strategy.

    Phase 1: Volume spike entry with 2x risk/reward target, then trailing stop
    Phase 2: Continuation trade after pullback/consolidation
    """

    def __init__(self):
        self.positions: Dict[str, TwoPhasePosition] = {}
        self.min_pullback_pct = 2.0  # Minimum pullback % to trigger Phase 1 entry watch
        self.min_volume_surge = 2.0  # Minimum volume surge (vs average)
        self.trailing_stop_pct = 3.0  # Default trailing stop %
        self.phase2_profit_multiple = 2.0  # Let Phase 2 run past 2x

    def on_spike_detected(
        self, symbol: str, spike_high: float, volume_surge: float
    ) -> Dict:
        """
        Called when a volume spike is detected. Start watching for pullback.
        """
        if symbol not in self.positions:
            self.positions[symbol] = TwoPhasePosition(
                symbol=symbol,
                phase=TradePhase.PHASE1_WATCHING,
                spike_high=spike_high,
                volume_surge=volume_surge,
            )
            logger.info(
                f"[PHASE1] {symbol} spike detected at ${spike_high:.2f}, watching for pullback"
            )
            return {
                "action": "WATCH",
                "message": f"Spike detected, waiting for pullback",
            }
        return {"action": "NONE", "message": "Already tracking"}

    def on_pullback(self, symbol: str, pullback_low: float) -> Dict:
        """
        Called when price pulls back after spike. Now looking for new high.
        """
        if symbol in self.positions:
            pos = self.positions[symbol]
            if pos.phase == TradePhase.PHASE1_WATCHING:
                pullback_pct = ((pos.spike_high - pullback_low) / pos.spike_high) * 100

                if pullback_pct >= self.min_pullback_pct:
                    pos.pullback_low = pullback_low
                    pos.phase = TradePhase.PHASE1_ENTRY
                    logger.info(
                        f"[PHASE1] {symbol} pullback to ${pullback_low:.2f} ({pullback_pct:.1f}%), looking for new high"
                    )
                    return {
                        "action": "READY",
                        "message": f"Pullback {pullback_pct:.1f}%, ready for entry on new high",
                    }
        return {"action": "NONE"}

    def check_entry(
        self, symbol: str, current_high: float, last_candle_low: float
    ) -> Dict:
        """
        Check if we should enter - first candle making new high after pullback.
        """
        if symbol in self.positions:
            pos = self.positions[symbol]
            if pos.phase == TradePhase.PHASE1_ENTRY:
                # Check if current high exceeds previous candle high (new high)
                if current_high > pos.pullback_low * 1.005:  # Small buffer
                    entry_price = current_high
                    stop_distance = entry_price - last_candle_low
                    stop_price = last_candle_low
                    target1 = entry_price + (stop_distance * 2)  # 2x risk/reward

                    pos.entry_price = entry_price
                    pos.last_candle_low = last_candle_low
                    pos.stop_distance = stop_distance
                    pos.stop_price = stop_price
                    pos.target1_price = target1
                    pos.high_since_entry = entry_price
                    pos.phase = TradePhase.PHASE1_ACTIVE

                    logger.info(f"[PHASE1 ENTRY] {symbol} @ ${entry_price:.2f}")
                    logger.info(
                        f"  Stop: ${stop_price:.2f} (${stop_distance:.2f} risk)"
                    )
                    logger.info(
                        f"  Target 1: ${target1:.2f} (2x = ${stop_distance * 2:.2f} reward)"
                    )

                    return {
                        "action": "BUY",
                        "price": entry_price,
                        "stop": stop_price,
                        "target": target1,
                        "risk_reward": 2.0,
                        "message": f"Phase 1 entry at ${entry_price:.2f}, stop ${stop_price:.2f}, target ${target1:.2f}",
                    }
        return {"action": "NONE"}

    def check_exit(
        self, symbol: str, current_price: float, current_high: float
    ) -> Dict:
        """
        Check exit conditions for active positions.
        """
        if symbol not in self.positions:
            return {"action": "NONE"}

        pos = self.positions[symbol]

        # Phase 1 Active - check stop and target
        if pos.phase == TradePhase.PHASE1_ACTIVE:
            # Stop loss hit
            if current_price <= pos.stop_price:
                logger.info(
                    f"[PHASE1 STOP] {symbol} stopped out at ${current_price:.2f}"
                )
                del self.positions[symbol]
                return {"action": "SELL", "reason": "STOP_LOSS", "price": current_price}

            # Target 1 hit - activate trailing stop
            if current_price >= pos.target1_price:
                pos.phase = TradePhase.PHASE1_TRAILING
                pos.high_since_entry = current_price
                logger.info(
                    f"[PHASE1 TARGET] {symbol} hit target at ${current_price:.2f}, activating 3% trailing stop"
                )
                return {
                    "action": "TRAIL",
                    "message": f"Target hit, trailing 3% from ${current_price:.2f}",
                }

            # Update high
            pos.high_since_entry = max(pos.high_since_entry, current_high)
            return {"action": "HOLD"}

        # Phase 1 Trailing
        if pos.phase == TradePhase.PHASE1_TRAILING:
            pos.high_since_entry = max(pos.high_since_entry, current_high)
            trailing_stop = pos.high_since_entry * (1 - pos.trailing_stop_pct / 100)

            if current_price <= trailing_stop:
                pnl = current_price - pos.entry_price
                pnl_pct = (pnl / pos.entry_price) * 100
                logger.info(
                    f"[PHASE1 EXIT] {symbol} trailing stop at ${current_price:.2f} (+{pnl_pct:.1f}%)"
                )

                # Move to Phase 2 watching
                pos.phase = TradePhase.PHASE2_WATCHING
                return {
                    "action": "SELL",
                    "reason": "TRAILING_STOP",
                    "price": current_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "next_phase": "PHASE2_WATCHING",
                }
            return {"action": "TRAIL", "trailing_stop": trailing_stop}

        # Phase 2 similar logic...
        if pos.phase == TradePhase.PHASE2_ACTIVE:
            pos.phase2_high = max(pos.phase2_high, current_high)
            trailing_stop = pos.phase2_high * (1 - pos.trailing_stop_pct / 100)

            if current_price <= trailing_stop:
                pnl = current_price - pos.phase2_entry
                logger.info(f"[PHASE2 EXIT] {symbol} at ${current_price:.2f}")
                del self.positions[symbol]
                return {
                    "action": "SELL",
                    "reason": "PHASE2_TRAILING",
                    "price": current_price,
                }
            return {"action": "HOLD"}

        return {"action": "NONE"}

    def check_phase2_entry(
        self,
        symbol: str,
        price: float,
        macd_signal: str = "",
        volume_vs_avg: float = 0.0,
        momentum: float = 0.0,
    ) -> Dict:
        """
        Check for Phase 2 continuation entry after consolidation.

        If MACD analyzer available, uses advanced signal detection.
        Otherwise falls back to simple indicator checks.
        """
        if symbol in self.positions:
            pos = self.positions[symbol]
            if pos.phase == TradePhase.PHASE2_WATCHING:
                conditions = []
                macd_details = {}

                # Try to use MACD analyzer for advanced signal detection
                analyzer = get_macd_analyzer()
                if analyzer:
                    should_enter, reason, details = analyzer.check_phase2_entry(symbol)
                    macd_details = details

                    if should_enter:
                        # MACD analyzer says enter
                        conditions.append(f"MACD: {reason}")
                        if details.get("crossover") == "bullish_cross":
                            conditions.append("Bullish crossover")
                        if details.get("histogram_trend") == "expanding":
                            conditions.append("Momentum expanding")
                        if details.get("divergence") == "bullish_div":
                            conditions.append("Bullish divergence")
                    else:
                        # MACD analyzer says wait
                        logger.debug(f"[PHASE2] {symbol} waiting - {reason}")
                        return {
                            "action": "WAIT",
                            "reason": reason,
                            "macd_details": macd_details,
                        }
                else:
                    # Fallback to simple checks
                    if macd_signal == "bullish":
                        conditions.append("MACD bullish")

                # Additional indicators
                if volume_vs_avg > 1.5:
                    conditions.append("Volume picking up")
                if momentum > 0:
                    conditions.append("Momentum positive")

                # Require at least 2 conditions (or MACD analyzer approval)
                if len(conditions) >= 2:
                    pos.phase = TradePhase.PHASE2_ACTIVE
                    pos.phase2_entry = price
                    pos.phase2_high = price
                    pos.macd_signal = macd_details.get("macd_signal", macd_signal)
                    pos.momentum_score = macd_details.get("momentum", momentum)

                    logger.info(f"[PHASE2 ENTRY] {symbol} @ ${price:.2f}")
                    logger.info(f"  Signals: {', '.join(conditions)}")

                    return {
                        "action": "BUY",
                        "price": price,
                        "phase": "PHASE2",
                        "signals": conditions,
                        "macd_details": macd_details,
                        "message": f"Phase 2 continuation entry at ${price:.2f}",
                    }
        return {"action": "NONE"}

    def check_phase2_exit_macd(self, symbol: str, current_price: float) -> Dict:
        """
        Check if MACD indicates we should exit Phase 2 position.

        Uses MACD divergence, bearish crossover, and momentum fading as exit signals.
        """
        if symbol not in self.positions:
            return {"action": "NONE"}

        pos = self.positions[symbol]
        if pos.phase not in [TradePhase.PHASE2_ACTIVE, TradePhase.PHASE2_TRAILING]:
            return {"action": "NONE"}

        analyzer = get_macd_analyzer()
        if not analyzer:
            return {"action": "NONE"}

        should_exit, reason, details = analyzer.check_exit_signal(symbol)

        if should_exit:
            pnl = current_price - pos.phase2_entry
            pnl_pct = (pnl / pos.phase2_entry) * 100 if pos.phase2_entry > 0 else 0

            logger.info(
                f"[PHASE2 MACD EXIT] {symbol} at ${current_price:.2f} - {reason}"
            )

            return {
                "action": "SELL",
                "reason": f"MACD_{reason.upper()}",
                "price": current_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "macd_details": details,
            }

        return {"action": "HOLD", "macd_details": details}

    def evaluate_phase2_opportunity(
        self, symbol: str, current_price: float, volume_vs_avg: float = 1.0
    ) -> Dict:
        """
        Full evaluation of Phase 2 entry opportunity.

        Combines MACD analysis with volume and returns comprehensive assessment.
        """
        if symbol not in self.positions:
            return {
                "eligible": False,
                "reason": "not_tracking",
                "message": f"{symbol} is not being tracked for Phase 2",
            }

        pos = self.positions[symbol]
        if pos.phase != TradePhase.PHASE2_WATCHING:
            return {
                "eligible": False,
                "reason": "wrong_phase",
                "phase": pos.phase.value,
                "message": f"{symbol} is in {pos.phase.value}, not Phase 2 watching",
            }

        # Get MACD analysis
        analyzer = get_macd_analyzer()
        macd_signal = None
        if analyzer:
            macd_signal = analyzer.analyze(symbol)

        result = {
            "eligible": True,
            "symbol": symbol,
            "current_price": current_price,
            "phase1_entry": pos.entry_price,
            "phase1_profit_pct": (
                ((current_price - pos.entry_price) / pos.entry_price * 100)
                if pos.entry_price > 0
                else 0
            ),
            "volume_vs_avg": volume_vs_avg,
            "conditions_met": [],
        }

        if macd_signal:
            result["macd"] = {
                "signal_type": macd_signal.signal_type,
                "crossover": macd_signal.crossover_type,
                "histogram": macd_signal.histogram,
                "histogram_trend": macd_signal.histogram_trend,
                "strength": macd_signal.strength,
                "momentum": macd_signal.momentum_score,
                "divergence": macd_signal.divergence,
                "phase2_ready": macd_signal.phase2_ready,
            }

            if macd_signal.phase2_ready:
                result["conditions_met"].append("MACD bullish setup")
            if macd_signal.crossover_type == "bullish_cross":
                result["conditions_met"].append("Bullish crossover")
            if macd_signal.histogram_trend == "expanding":
                result["conditions_met"].append("Momentum expanding")
            if macd_signal.divergence == "bullish_div":
                result["conditions_met"].append("Bullish divergence")

        if volume_vs_avg >= 1.5:
            result["conditions_met"].append("Volume surge")
        elif volume_vs_avg >= 1.2:
            result["conditions_met"].append("Volume picking up")

        # Recommendation
        if len(result["conditions_met"]) >= 2:
            result["recommendation"] = "ENTER"
            result["confidence"] = min(1.0, len(result["conditions_met"]) * 0.3)
        elif len(result["conditions_met"]) == 1:
            result["recommendation"] = "WAIT"
            result["confidence"] = 0.3
        else:
            result["recommendation"] = "NOT_READY"
            result["confidence"] = 0.0

        return result

    def get_position_status(self, symbol: str) -> Optional[Dict]:
        """Get current status of a position."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            return {
                "symbol": symbol,
                "phase": pos.phase.value,
                "entry_price": pos.entry_price,
                "stop_price": pos.stop_price,
                "target1": pos.target1_price,
                "high_since_entry": pos.high_since_entry,
                "trailing_stop_pct": pos.trailing_stop_pct,
            }
        return None


# Singleton instance
_strategy: Optional[TwoPhaseStrategy] = None


def get_two_phase_strategy() -> TwoPhaseStrategy:
    """Get or create the two-phase strategy instance."""
    global _strategy
    if _strategy is None:
        _strategy = TwoPhaseStrategy()
    return _strategy
