"""
Chronos Exit Manager - Smart Exit Signals
==========================================
Uses Chronos regime detection to provide early exit signals
and avoid getting caught in momentum swings/stop losses.

PROBLEM SOLVED:
- Historical data shows 28 stop losses = -$460 (0% win rate)
- Meanwhile trailing stops = 85% WR, +$96
- Solution: Exit BEFORE hitting stop loss when Chronos detects:
  1. Regime shift (TRENDING_UP -> VOLATILE or TRENDING_DOWN)
  2. Momentum fading (trend strength dropping)
  3. Volatility spike (risk of whipsaw)
  4. Confidence degradation

KEY INSIGHT:
Stop losses don't work for momentum plays because by the time
price drops 3%, the momentum is already dead. Chronos can detect
the regime shift BEFORE the price drop catches up.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChronosExitSignal:
    """Exit signal from Chronos analysis"""
    should_exit: bool = False
    reason: str = ""
    urgency: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    regime_before: str = ""
    regime_after: str = ""
    trend_strength: float = 0.0
    confidence: float = 0.0
    volatility: float = 0.0
    details: Dict = field(default_factory=dict)


@dataclass
class PositionContext:
    """Track Chronos context while holding a position"""
    symbol: str
    entry_time: datetime
    entry_price: float = 0.0
    entry_regime: str = ""
    entry_confidence: float = 0.0
    entry_trend_strength: float = 0.0

    # Running context (updated on each check)
    current_regime: str = ""
    current_confidence: float = 0.0
    current_trend_strength: float = 0.0
    current_volatility: float = 0.0
    current_prob_up: float = 0.5

    # Change tracking
    regime_changes: int = 0
    confidence_drops: int = 0
    last_update: Optional[datetime] = None

    # History for trend detection
    regime_history: List[str] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    trend_history: List[float] = field(default_factory=list)

    # Failed momentum tracking (Ross Cameron rule)
    high_since_entry: float = 0.0
    price_history: List[Tuple[datetime, float]] = field(default_factory=list)
    momentum_stalled: bool = False
    stall_count: int = 0  # Consecutive stall readings


class ChronosExitManager:
    """
    Monitors open positions with Chronos context.
    Provides early exit signals when momentum fades.

    STATE MACHINE INTEGRATION:
    - Entry logic owns: IDLE → ATTENTION → SETUP → IGNITION
    - Exit/Monitor logic (Chronos) owns: IN_POSITION → EXIT
    - On exit signal, state machine transitions to EXIT → COOLDOWN
    """

    def __init__(self):
        self.positions: Dict[str, PositionContext] = {}

        # State machine integration (lazy loaded)
        self._state_machine = None

        # Exit thresholds (tuned for momentum scalping)
        self.config = {
            # Regime-based exits
            "exit_on_regime_change": True,
            "favorable_regimes": ["TRENDING_UP", "RANGING"],
            "danger_regimes": ["TRENDING_DOWN", "VOLATILE"],

            # Confidence-based exits
            "min_confidence": 0.4,  # Exit if confidence drops below this
            "confidence_drop_threshold": 0.2,  # Exit if drops 20% from entry

            # Trend-based exits
            "min_trend_strength": 0.3,  # Exit if trend weakens below this
            "trend_fade_threshold": 0.3,  # Exit if drops 30% from entry

            # Volatility-based exits
            "max_volatility": 0.5,  # 50% annualized = danger zone
            "volatility_spike_threshold": 2.0,  # 2x avg = exit

            # Probability-based exits
            "min_prob_up": 0.45,  # Exit if prob drops below 45%

            # Timing
            "check_interval_seconds": 5,  # How often to check
            "min_hold_before_exit": 10,  # Don't exit within 10s of entry

            # ===== FAILED MOMENTUM DETECTION (Ross Cameron Rule) =====
            # "If it doesn't move immediately, get out"
            "use_failed_momentum_exit": True,
            "momentum_check_seconds": 30,  # Check momentum after 30s
            "expected_gain_30s": 0.5,  # Expect at least 0.5% gain in 30s
            "momentum_stall_threshold": 0.2,  # < 0.2% gain = stalled
            "momentum_fade_exit": True,  # Exit if price fading from high
            "fade_from_high_percent": 0.5,  # Exit if down 0.5% from high
            "consecutive_stall_checks": 3,  # 3 consecutive stalls = exit
        }

        # Import chronos adapter
        self._chronos_adapter = None
        self._init_chronos()

        logger.info("ChronosExitManager initialized with state machine support")

    @property
    def state_machine(self):
        """Lazy load state machine for position tracking"""
        if self._state_machine is None:
            try:
                from ai.momentum_state_machine import get_state_machine
                self._state_machine = get_state_machine()
            except Exception as e:
                logger.warning(f"State machine init failed: {e}")
        return self._state_machine

    def _init_chronos(self):
        """Initialize Chronos adapter"""
        try:
            from ai.chronos_adapter import get_chronos_adapter
            self._chronos_adapter = get_chronos_adapter()
            if self._chronos_adapter.available:
                logger.info("Chronos backend available for exit management")
            else:
                logger.warning("Chronos backend not available - using technical fallback")
        except Exception as e:
            logger.warning(f"Chronos init failed: {e}")

    def register_position(self, symbol: str, entry_price: float) -> PositionContext:
        """
        Register a new position for Chronos monitoring.
        Called when scalper enters a trade.

        STATE MACHINE HANDOFF:
        - Entry logic (HFT Scalper) has transitioned to IN_POSITION
        - Chronos Exit Manager now takes over position monitoring
        - Exit signals will trigger state machine EXIT → COOLDOWN
        """
        # Verify state machine handoff (state should be IN_POSITION)
        state_info = None
        if self.state_machine:
            try:
                from ai.momentum_state_machine import MomentumState
                symbol_state = self.state_machine.get_state(symbol)
                if symbol_state and symbol_state.state == MomentumState.IN_POSITION:
                    logger.info(f"[STATE HANDOFF] {symbol}: Entry logic → Chronos Exit Manager")
                    state_info = self.state_machine.get_state_info(symbol)
                elif symbol_state is not None:
                    logger.warning(
                        f"[STATE HANDOFF] {symbol}: Expected IN_POSITION, got {symbol_state.state.value}. "
                        f"Chronos will still monitor."
                    )
            except Exception as e:
                logger.debug(f"State machine check failed: {e}")

        # Get initial context
        initial_context = self._get_chronos_context(symbol)

        ctx = PositionContext(
            symbol=symbol,
            entry_time=datetime.now(),
            entry_price=entry_price,
            entry_regime=initial_context.get("regime", "UNKNOWN"),
            entry_confidence=initial_context.get("confidence", 0.5),
            entry_trend_strength=initial_context.get("trend_strength", 0.5),
            current_regime=initial_context.get("regime", "UNKNOWN"),
            current_confidence=initial_context.get("confidence", 0.5),
            current_trend_strength=initial_context.get("trend_strength", 0.5),
            current_volatility=initial_context.get("volatility", 0.25),
            current_prob_up=initial_context.get("prob_up", 0.5),
            last_update=datetime.now(),
            high_since_entry=entry_price
        )

        ctx.regime_history.append(ctx.entry_regime)
        ctx.confidence_history.append(ctx.entry_confidence)
        ctx.trend_history.append(ctx.entry_trend_strength)

        self.positions[symbol] = ctx

        # Sync position info from state machine if available
        if state_info:
            ctx.entry_price = state_info.get("entry_price", entry_price)
            logger.debug(f"[STATE SYNC] {symbol}: Entry price synced from state machine")

        logger.info(
            f"[CHRONOS EXIT] Registered {symbol}: "
            f"Regime={ctx.entry_regime}, "
            f"Conf={ctx.entry_confidence:.0%}, "
            f"Trend={ctx.entry_trend_strength:.0%}"
        )

        return ctx

    def unregister_position(self, symbol: str):
        """
        Remove position from monitoring (on exit).

        STATE MACHINE NOTE:
        The HFT Scalper handles state machine transition to EXIT → COOLDOWN.
        Chronos just cleans up its internal tracking here.
        """
        if symbol in self.positions:
            ctx = self.positions[symbol]
            hold_seconds = (datetime.now() - ctx.entry_time).total_seconds()
            max_gain = (ctx.high_since_entry - ctx.entry_price) / ctx.entry_price * 100 if ctx.entry_price > 0 else 0

            del self.positions[symbol]

            logger.info(
                f"[CHRONOS EXIT] Unregistered {symbol}: "
                f"Hold={hold_seconds:.0f}s, MaxGain={max_gain:.1f}%, "
                f"Stalls={ctx.stall_count}"
            )

    def get_state_machine_status(self, symbol: str) -> Optional[Dict]:
        """Get state machine status for a symbol"""
        if not self.state_machine:
            return None
        try:
            symbol_state = self.state_machine.get_state(symbol)
            info = self.state_machine.get_state_info(symbol)
            return {
                "state": symbol_state.state.value if symbol_state else "UNKNOWN",
                "info": info
            }
        except Exception as e:
            logger.debug(f"State machine status failed for {symbol}: {e}")
            return None

    def _get_chronos_context(self, symbol: str) -> Dict:
        """Get current Chronos context for a symbol"""
        try:
            if self._chronos_adapter and self._chronos_adapter.available:
                ctx = self._chronos_adapter.get_context(symbol)
                return {
                    "regime": ctx.market_regime,
                    "confidence": ctx.regime_confidence,
                    "trend_strength": ctx.trend_strength,
                    "trend_direction": ctx.trend_direction,
                    "volatility": ctx.current_volatility,
                    "prob_up": ctx.prob_up,
                    "prob_down": ctx.prob_down
                }
        except Exception as e:
            logger.debug(f"Chronos context failed for {symbol}: {e}")

        # Fallback to technical analysis
        return self._get_technical_context(symbol)

    def _get_technical_context(self, symbol: str) -> Dict:
        """Fallback: Get context from technical indicators"""
        try:
            import yfinance as yf
            import ta

            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5d", interval="5m")

            if len(df) < 50:
                return {"regime": "UNKNOWN", "confidence": 0.5, "trend_strength": 0.5}

            close = df['Close']
            high = df['High']
            low = df['Low']

            # ADX for trend strength
            adx = ta.trend.ADXIndicator(high, low, close, window=14)
            adx_value = adx.adx().iloc[-1]
            adx_pos = adx.adx_pos().iloc[-1]
            adx_neg = adx.adx_neg().iloc[-1]

            # Volatility
            returns = close.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252 * 78)  # Annualized from 5min

            # Determine regime
            if volatility > 0.4:  # High volatility
                regime = "VOLATILE"
            elif adx_value > 25:
                regime = "TRENDING_UP" if adx_pos > adx_neg else "TRENDING_DOWN"
            else:
                regime = "RANGING"

            # Recent direction for prob_up
            recent_change = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
            prob_up = 0.5 + (recent_change * 5)  # Scale to probability
            prob_up = max(0.2, min(0.8, prob_up))

            return {
                "regime": regime,
                "confidence": min(adx_value / 50, 1.0),
                "trend_strength": min(adx_value / 50, 1.0),
                "trend_direction": 1 if adx_pos > adx_neg else -1,
                "volatility": volatility,
                "prob_up": prob_up,
                "prob_down": 1 - prob_up
            }

        except Exception as e:
            logger.debug(f"Technical context failed for {symbol}: {e}")
            return {
                "regime": "UNKNOWN",
                "confidence": 0.5,
                "trend_strength": 0.5,
                "volatility": 0.25,
                "prob_up": 0.5
            }

    def check_exit(self, symbol: str, current_price: float, entry_price: float) -> ChronosExitSignal:
        """
        Check if Chronos recommends an early exit.

        This is the main entry point - called by scalper's check_exit_signal.

        Returns:
            ChronosExitSignal with should_exit=True if we should exit early
        """
        signal = ChronosExitSignal()

        # Get or create position context
        if symbol not in self.positions:
            self.register_position(symbol, entry_price)

        ctx = self.positions[symbol]

        # Check minimum hold time
        hold_time = (datetime.now() - ctx.entry_time).total_seconds()
        if hold_time < self.config["min_hold_before_exit"]:
            return signal  # Don't exit yet

        # Update context
        new_context = self._get_chronos_context(symbol)

        ctx.current_regime = new_context.get("regime", ctx.current_regime)
        ctx.current_confidence = new_context.get("confidence", ctx.current_confidence)
        ctx.current_trend_strength = new_context.get("trend_strength", ctx.current_trend_strength)
        ctx.current_volatility = new_context.get("volatility", ctx.current_volatility)
        ctx.current_prob_up = new_context.get("prob_up", ctx.current_prob_up)
        ctx.last_update = datetime.now()

        # Track history
        ctx.regime_history.append(ctx.current_regime)
        ctx.confidence_history.append(ctx.current_confidence)
        ctx.trend_history.append(ctx.current_trend_strength)

        # Keep history limited
        ctx.regime_history = ctx.regime_history[-20:]
        ctx.confidence_history = ctx.confidence_history[-20:]
        ctx.trend_history = ctx.trend_history[-20:]

        # Current P/L
        pnl_pct = (current_price - entry_price) / entry_price * 100

        # ===== EXIT CHECKS =====

        # 0. FAILED MOMENTUM CHECK (Ross Cameron Rule - MOST IMPORTANT)
        # "If it doesn't move immediately, get out"
        signal = self._check_failed_momentum(ctx, current_price, pnl_pct, signal)
        if signal.should_exit:
            return signal

        # 1. REGIME SHIFT CHECK
        if self.config["exit_on_regime_change"]:
            signal = self._check_regime_shift(ctx, pnl_pct, signal)
            if signal.should_exit:
                return signal

        # 2. CONFIDENCE DROP CHECK
        signal = self._check_confidence_drop(ctx, pnl_pct, signal)
        if signal.should_exit:
            return signal

        # 3. TREND FADE CHECK
        signal = self._check_trend_fade(ctx, pnl_pct, signal)
        if signal.should_exit:
            return signal

        # 4. VOLATILITY SPIKE CHECK
        signal = self._check_volatility_spike(ctx, pnl_pct, signal)
        if signal.should_exit:
            return signal

        # 5. PROBABILITY DROP CHECK
        signal = self._check_probability_drop(ctx, pnl_pct, signal)
        if signal.should_exit:
            return signal

        # 6. CONSECUTIVE REGIME WARNINGS
        signal = self._check_regime_warnings(ctx, pnl_pct, signal)
        if signal.should_exit:
            return signal

        return signal

    def _check_regime_shift(self, ctx: PositionContext, pnl_pct: float,
                            signal: ChronosExitSignal) -> ChronosExitSignal:
        """Check for regime shift from favorable to danger"""
        if ctx.entry_regime in self.config["favorable_regimes"]:
            if ctx.current_regime in self.config["danger_regimes"]:
                ctx.regime_changes += 1

                # If we're down and regime shifted to dangerous, exit
                if pnl_pct < 1.0:  # Not in profit
                    signal.should_exit = True
                    signal.reason = "REGIME_SHIFT"
                    signal.urgency = "HIGH"
                    signal.regime_before = ctx.entry_regime
                    signal.regime_after = ctx.current_regime
                    signal.details = {
                        "message": f"Regime changed {ctx.entry_regime} -> {ctx.current_regime}",
                        "pnl_pct": pnl_pct
                    }
                    logger.warning(
                        f"[CHRONOS EXIT] {ctx.symbol}: REGIME SHIFT "
                        f"{ctx.entry_regime}->{ctx.current_regime} @ {pnl_pct:+.1f}%"
                    )

        return signal

    def _check_confidence_drop(self, ctx: PositionContext, pnl_pct: float,
                               signal: ChronosExitSignal) -> ChronosExitSignal:
        """Check for confidence drop below threshold"""
        # Absolute threshold
        if ctx.current_confidence < self.config["min_confidence"]:
            if pnl_pct < 1.0:  # Not in profit
                signal.should_exit = True
                signal.reason = "CONFIDENCE_LOW"
                signal.urgency = "MEDIUM"
                signal.confidence = ctx.current_confidence
                signal.details = {
                    "message": f"Confidence {ctx.current_confidence:.0%} < {self.config['min_confidence']:.0%} min",
                    "pnl_pct": pnl_pct
                }
                logger.warning(
                    f"[CHRONOS EXIT] {ctx.symbol}: LOW CONFIDENCE "
                    f"{ctx.current_confidence:.0%} @ {pnl_pct:+.1f}%"
                )
                return signal

        # Relative drop from entry
        if ctx.entry_confidence > 0:
            drop = (ctx.entry_confidence - ctx.current_confidence) / ctx.entry_confidence
            if drop >= self.config["confidence_drop_threshold"]:
                if pnl_pct < 0.5:  # Flat or losing
                    signal.should_exit = True
                    signal.reason = "CONFIDENCE_FADING"
                    signal.urgency = "MEDIUM"
                    signal.confidence = ctx.current_confidence
                    signal.details = {
                        "message": f"Confidence dropped {drop:.0%} from entry",
                        "entry_conf": ctx.entry_confidence,
                        "current_conf": ctx.current_confidence,
                        "pnl_pct": pnl_pct
                    }
                    logger.warning(
                        f"[CHRONOS EXIT] {ctx.symbol}: CONFIDENCE FADING "
                        f"{ctx.entry_confidence:.0%}->{ctx.current_confidence:.0%} @ {pnl_pct:+.1f}%"
                    )

        return signal

    def _check_trend_fade(self, ctx: PositionContext, pnl_pct: float,
                          signal: ChronosExitSignal) -> ChronosExitSignal:
        """Check for trend strength fading"""
        # Absolute threshold
        if ctx.current_trend_strength < self.config["min_trend_strength"]:
            if pnl_pct < 0.5:  # Flat or losing
                signal.should_exit = True
                signal.reason = "TREND_WEAK"
                signal.urgency = "MEDIUM"
                signal.trend_strength = ctx.current_trend_strength
                signal.details = {
                    "message": f"Trend strength {ctx.current_trend_strength:.0%} < {self.config['min_trend_strength']:.0%} min",
                    "pnl_pct": pnl_pct
                }
                logger.warning(
                    f"[CHRONOS EXIT] {ctx.symbol}: WEAK TREND "
                    f"{ctx.current_trend_strength:.0%} @ {pnl_pct:+.1f}%"
                )
                return signal

        # Check for declining trend (last 3 readings)
        if len(ctx.trend_history) >= 3:
            recent = ctx.trend_history[-3:]
            if all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                # Trend is consistently declining
                decline = (recent[0] - recent[-1]) / recent[0] if recent[0] > 0 else 0
                if decline >= 0.2 and pnl_pct < 1.0:  # 20% decline
                    signal.should_exit = True
                    signal.reason = "TREND_DECLINING"
                    signal.urgency = "MEDIUM"
                    signal.trend_strength = ctx.current_trend_strength
                    signal.details = {
                        "message": f"Trend declining: {recent[0]:.0%} -> {recent[-1]:.0%}",
                        "decline": decline,
                        "pnl_pct": pnl_pct
                    }
                    logger.warning(
                        f"[CHRONOS EXIT] {ctx.symbol}: TREND DECLINING "
                        f"{recent[0]:.0%}->{recent[-1]:.0%} @ {pnl_pct:+.1f}%"
                    )

        return signal

    def _check_volatility_spike(self, ctx: PositionContext, pnl_pct: float,
                                signal: ChronosExitSignal) -> ChronosExitSignal:
        """Check for dangerous volatility spike"""
        if ctx.current_volatility > self.config["max_volatility"]:
            if pnl_pct < 1.0:  # Not solidly in profit
                signal.should_exit = True
                signal.reason = "VOLATILITY_SPIKE"
                signal.urgency = "HIGH"
                signal.volatility = ctx.current_volatility
                signal.details = {
                    "message": f"Volatility {ctx.current_volatility:.0%} > {self.config['max_volatility']:.0%} max",
                    "pnl_pct": pnl_pct
                }
                logger.warning(
                    f"[CHRONOS EXIT] {ctx.symbol}: VOLATILITY SPIKE "
                    f"{ctx.current_volatility:.0%} @ {pnl_pct:+.1f}%"
                )

        return signal

    def _check_probability_drop(self, ctx: PositionContext, pnl_pct: float,
                                signal: ChronosExitSignal) -> ChronosExitSignal:
        """Check for probability of up movement dropping"""
        if ctx.current_prob_up < self.config["min_prob_up"]:
            if pnl_pct < 0.5:  # Flat or losing
                signal.should_exit = True
                signal.reason = "PROBABILITY_BEARISH"
                signal.urgency = "HIGH"
                signal.details = {
                    "message": f"Prob up {ctx.current_prob_up:.0%} < {self.config['min_prob_up']:.0%} min",
                    "prob_up": ctx.current_prob_up,
                    "pnl_pct": pnl_pct
                }
                logger.warning(
                    f"[CHRONOS EXIT] {ctx.symbol}: BEARISH PROBABILITY "
                    f"{ctx.current_prob_up:.0%} @ {pnl_pct:+.1f}%"
                )

        return signal

    def _check_failed_momentum(self, ctx: PositionContext, current_price: float,
                               pnl_pct: float, signal: ChronosExitSignal) -> ChronosExitSignal:
        """
        ROSS CAMERON RULE: If stock doesn't move immediately, get out.

        This is the most important exit check - catches failed momentum
        BEFORE it becomes a stop loss situation.
        """
        if not self.config["use_failed_momentum_exit"]:
            return signal

        hold_seconds = (datetime.now() - ctx.entry_time).total_seconds()

        # Update high since entry
        if current_price > ctx.high_since_entry:
            ctx.high_since_entry = current_price

        # Track price history
        ctx.price_history.append((datetime.now(), current_price))
        ctx.price_history = ctx.price_history[-60:]  # Keep last 60 readings

        # Check 1: After 30s, did we get expected gain?
        check_time = self.config["momentum_check_seconds"]
        expected_gain = self.config["expected_gain_30s"]

        if hold_seconds >= check_time:
            # Check if we've reached expected gain at any point
            max_gain_pct = (ctx.high_since_entry - ctx.entry_price) / ctx.entry_price * 100

            if max_gain_pct < expected_gain:
                # Never hit expected gain - momentum failed
                ctx.stall_count += 1

                if ctx.stall_count >= self.config["consecutive_stall_checks"]:
                    signal.should_exit = True
                    signal.reason = "FAILED_MOMENTUM"
                    signal.urgency = "HIGH"
                    signal.details = {
                        "message": f"No momentum after {hold_seconds:.0f}s - max gain was {max_gain_pct:.2f}%",
                        "expected_gain": expected_gain,
                        "actual_max_gain": max_gain_pct,
                        "hold_seconds": hold_seconds,
                        "stall_count": ctx.stall_count
                    }
                    logger.warning(
                        f"[ROSS CAMERON EXIT] {ctx.symbol}: FAILED MOMENTUM - "
                        f"Only {max_gain_pct:.2f}% gain in {hold_seconds:.0f}s (expected {expected_gain}%)"
                    )
                    return signal
            else:
                # Reset stall count if we hit a new high
                ctx.stall_count = 0

        # Check 2: Momentum stall - price not moving up
        if hold_seconds >= 15 and len(ctx.price_history) >= 5:
            recent_prices = [p[1] for p in ctx.price_history[-5:]]
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100

            if price_change < self.config["momentum_stall_threshold"]:
                ctx.momentum_stalled = True

                # If stalled AND not in profit, exit
                if pnl_pct < 0.5:
                    ctx.stall_count += 1

                    if ctx.stall_count >= self.config["consecutive_stall_checks"]:
                        signal.should_exit = True
                        signal.reason = "MOMENTUM_STALLED"
                        signal.urgency = "MEDIUM"
                        signal.details = {
                            "message": f"Price stalled - only {price_change:.2f}% in last 5 checks",
                            "recent_change": price_change,
                            "pnl_pct": pnl_pct,
                            "stall_count": ctx.stall_count
                        }
                        logger.warning(
                            f"[ROSS CAMERON EXIT] {ctx.symbol}: STALLED - "
                            f"{price_change:.2f}% change in last 5 checks @ {pnl_pct:+.1f}%"
                        )
                        return signal
            else:
                ctx.momentum_stalled = False
                ctx.stall_count = max(0, ctx.stall_count - 1)  # Slow decay

        # Check 3: Fading from high - price dropping from the high
        if self.config["momentum_fade_exit"] and ctx.high_since_entry > ctx.entry_price:
            fade_from_high = (ctx.high_since_entry - current_price) / ctx.high_since_entry * 100
            fade_threshold = self.config["fade_from_high_percent"]

            # Only exit on fade if we're flat or losing AND we had a gain
            max_gain_pct = (ctx.high_since_entry - ctx.entry_price) / ctx.entry_price * 100

            if fade_from_high >= fade_threshold and max_gain_pct > 0.3 and pnl_pct < 0.5:
                signal.should_exit = True
                signal.reason = "MOMENTUM_FADING"
                signal.urgency = "MEDIUM"
                signal.details = {
                    "message": f"Fading {fade_from_high:.2f}% from high",
                    "high_price": ctx.high_since_entry,
                    "current_price": current_price,
                    "max_gain_was": max_gain_pct,
                    "current_pnl": pnl_pct
                }
                logger.warning(
                    f"[ROSS CAMERON EXIT] {ctx.symbol}: FADING - "
                    f"Down {fade_from_high:.2f}% from high ${ctx.high_since_entry:.2f}"
                )
                return signal

        return signal

    def _check_regime_warnings(self, ctx: PositionContext, pnl_pct: float,
                               signal: ChronosExitSignal) -> ChronosExitSignal:
        """Check for accumulating warning signs"""
        # Count danger regimes in recent history
        if len(ctx.regime_history) >= 3:
            recent_regimes = ctx.regime_history[-3:]
            danger_count = sum(1 for r in recent_regimes if r in self.config["danger_regimes"])

            # If 2+ of last 3 readings are danger regimes, exit
            if danger_count >= 2 and pnl_pct < 1.0:
                signal.should_exit = True
                signal.reason = "MULTIPLE_WARNINGS"
                signal.urgency = "HIGH"
                signal.details = {
                    "message": f"{danger_count}/3 recent readings in danger regime",
                    "regimes": recent_regimes,
                    "pnl_pct": pnl_pct
                }
                logger.warning(
                    f"[CHRONOS EXIT] {ctx.symbol}: MULTIPLE WARNINGS "
                    f"{recent_regimes} @ {pnl_pct:+.1f}%"
                )

        return signal

    def get_position_contexts(self) -> Dict[str, Dict]:
        """Get all position contexts for dashboard display (includes state machine status)"""
        result = {}
        for symbol, ctx in self.positions.items():
            hold_seconds = (datetime.now() - ctx.entry_time).total_seconds()
            max_gain_pct = (ctx.high_since_entry - ctx.entry_price) / ctx.entry_price * 100 if ctx.entry_price > 0 else 0

            # Get state machine status
            sm_status = self.get_state_machine_status(symbol)

            result[symbol] = {
                "symbol": symbol,
                "entry_price": ctx.entry_price,
                "entry_regime": ctx.entry_regime,
                "current_regime": ctx.current_regime,
                "entry_confidence": ctx.entry_confidence,
                "current_confidence": ctx.current_confidence,
                "trend_strength": ctx.current_trend_strength,
                "volatility": ctx.current_volatility,
                "prob_up": ctx.current_prob_up,
                "regime_changes": ctx.regime_changes,
                "hold_seconds": hold_seconds,
                "last_update": ctx.last_update.isoformat() if ctx.last_update else None,
                # Failed momentum tracking
                "high_since_entry": ctx.high_since_entry,
                "max_gain_pct": max_gain_pct,
                "momentum_stalled": ctx.momentum_stalled,
                "stall_count": ctx.stall_count,
                # State machine integration
                "state_machine": sm_status
            }
        return result

    def get_config(self) -> Dict:
        """Get current configuration"""
        return self.config.copy()

    def update_config(self, **kwargs):
        """Update configuration"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                logger.info(f"ChronosExitManager config: {key}={value}")


# Singleton
_exit_manager: Optional[ChronosExitManager] = None


def get_chronos_exit_manager() -> ChronosExitManager:
    """Get or create the Chronos exit manager singleton"""
    global _exit_manager
    if _exit_manager is None:
        _exit_manager = ChronosExitManager()
    return _exit_manager


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("CHRONOS EXIT MANAGER + STATE MACHINE TEST")
    print("=" * 60)

    # First, set up state machine to IN_POSITION (simulate entry)
    print("\n1. Setting up state machine...")
    from ai.momentum_state_machine import get_state_machine, MomentumState
    sm = get_state_machine()

    # Transition AAPL through states to IN_POSITION
    sm.update_momentum("AAPL", 45, {"grade": "D"})  # -> ATTENTION
    sm.update_momentum("AAPL", 60, {"grade": "C"})  # -> SETUP
    sm.update_momentum("AAPL", 75, {"grade": "B"})  # -> IGNITION
    sm.enter_position("AAPL", 150.0, 10, 147.0, 154.5)  # -> IN_POSITION

    symbol_state = sm.get_state("AAPL")
    print(f"   State machine state: {symbol_state.state.value if symbol_state else 'None'}")
    assert symbol_state.state == MomentumState.IN_POSITION, f"Expected IN_POSITION, got {symbol_state.state}"

    # Now register with Chronos Exit Manager
    print("\n2. Registering position with Chronos Exit Manager...")
    manager = get_chronos_exit_manager()
    manager.register_position("AAPL", 150.0)

    # Check state machine status is visible
    sm_status = manager.get_state_machine_status("AAPL")
    print(f"   State machine status: {sm_status}")

    # Simulate exit check
    print("\n3. Testing exit signal detection...")
    import time
    time.sleep(11)  # Wait past min hold time

    signal = manager.check_exit("AAPL", 148.5, 150.0)  # -1% loss
    print(f"   Exit Signal: should_exit={signal.should_exit}, reason={signal.reason}")

    # Check position context includes state machine
    print("\n4. Checking position contexts...")
    contexts = manager.get_position_contexts()
    for sym, ctx in contexts.items():
        print(f"   {sym}:")
        print(f"      Entry: ${ctx['entry_price']:.2f}")
        print(f"      Regime: {ctx['current_regime']}")
        print(f"      State Machine: {ctx.get('state_machine', 'N/A')}")

    # Unregister (simulating exit)
    print("\n5. Unregistering position (exit)...")
    manager.unregister_position("AAPL")

    print("\n" + "=" * 60)
    print("STATE MACHINE HANDOFF TEST PASSED!")
    print("=" * 60)
