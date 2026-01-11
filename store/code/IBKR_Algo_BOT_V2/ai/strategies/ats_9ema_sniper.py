"""
ATS + 9 EMA SNIPER STRATEGY
===========================
Morning Scalping Strategy (9:40 AM - 11:00 AM ET)

Status: Approved for implementation
Priority: HIGH
Mode: Paper-first, automation-ready
Risk Profile: Capital preservation first

CORE RULES (CODED VERBATIM):
- NO PULLBACK = NO TRADE
- NO CONFIRMATION = NO TRADE
- FLAT = SUCCESS

Strategy Intent:
- Trades CONTINUATION, not breakouts
- Uses ATS for QUALIFICATION
- Uses 9 EMA pullbacks for EXECUTION
- Avoids overtrading in changing market conditions

Role Split:
- ATS = QUALIFICATION (Is this symbol worth trading?)
- 9 EMA Sniper = EXECUTION (Where is the lowest-risk entry?)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Optional, Dict, List, Tuple
import pytz

from ai.indicators.ema import EMATracker, get_ema_tracker
from ai.fsm.strategy_states import SniperFSM, SniperState, get_sniper_fsm
from ai.logging.events import get_event_logger, log_sniper_attempt

logger = logging.getLogger(__name__)

ET_TZ = pytz.timezone('US/Eastern')


@dataclass
class SniperConfig:
    """Configuration for ATS + 9 EMA Sniper Strategy"""

    # Time gate (HARD)
    start_time_et: dt_time = dt_time(9, 40)   # 9:40 AM ET
    end_time_et: dt_time = dt_time(11, 0)     # 11:00 AM ET

    # ATS qualification thresholds
    ats_min_confidence: float = 0.6           # Min ATS score to qualify
    rvol_min: float = 2.0                     # Min relative volume

    # Pullback requirements
    max_pullback_pct: float = 30.0            # Max pullback as % of impulse
    ema_pullback_zone_pct: float = 0.5        # Within 0.5% of EMA = pullback zone
    require_volume_decrease: bool = True       # Volume must decrease in pullback

    # Entry confirmation
    reclaim_threshold_pct: float = 0.3        # Break above pullback high by 0.3%
    volume_expansion_ratio: float = 1.2       # Volume must expand 20% on entry

    # Risk management
    stop_below_pullback_low: bool = True      # Place stop below pullback low
    stop_below_ema9: bool = True              # Also use EMA9 as stop reference
    stop_buffer_pct: float = 0.2              # Buffer below stop level

    # Profit targets
    target_prior_high: bool = True            # First target at prior high
    trail_with_ema9: bool = True              # Trail with 9 EMA

    # Anti-overtrading
    max_trades_per_symbol: int = 2            # Max trades per symbol per day
    cooldown_after_stop_minutes: int = 5      # Cooldown after stop-out
    disable_after_failures: int = 2           # Disable symbol after X failures

    def to_dict(self) -> Dict:
        return {
            "start_time_et": self.start_time_et.strftime("%H:%M"),
            "end_time_et": self.end_time_et.strftime("%H:%M"),
            "ats_min_confidence": self.ats_min_confidence,
            "rvol_min": self.rvol_min,
            "max_pullback_pct": self.max_pullback_pct,
            "max_trades_per_symbol": self.max_trades_per_symbol,
            "cooldown_after_stop_minutes": self.cooldown_after_stop_minutes,
        }


@dataclass
class QualificationResult:
    """Result of ATS qualification check"""
    symbol: str
    qualified: bool = False
    ats_score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    disqualify_reasons: List[str] = field(default_factory=list)

    # Context data
    hrdc_mode: str = ""
    vwap_status: str = ""
    rvol: float = 0.0
    structure_bullish: bool = False

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "qualified": self.qualified,
            "ats_score": round(self.ats_score, 2),
            "reasons": self.reasons,
            "disqualify_reasons": self.disqualify_reasons,
            "hrdc_mode": self.hrdc_mode,
            "vwap_status": self.vwap_status,
            "rvol": self.rvol,
        }


@dataclass
class EntrySignal:
    """Entry confirmation signal"""
    symbol: str
    confirmed: bool = False
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    reason: str = ""  # RECLAIM_PULLBACK_HIGH, BULLISH_CANDLE_OFF_EMA

    pullback_low: float = 0.0
    pullback_depth_pct: float = 0.0
    ema9_at_entry: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "confirmed": self.confirmed,
            "entry_price": round(self.entry_price, 4),
            "stop_loss": round(self.stop_loss, 4),
            "target": round(self.target, 4),
            "reason": self.reason,
            "pullback_depth_pct": round(self.pullback_depth_pct, 2),
        }


class ATS9EMASniperStrategy:
    """
    ATS + 9 EMA Sniper Strategy Implementation.

    This strategy is PRECISION-FIRST, not frequency-first.

    Optimize for:
    - correctness
    - discipline
    - capital preservation

    NOT:
    - more trades
    - more activity
    - more signals
    """

    def __init__(self, config: SniperConfig = None):
        self.config = config or SniperConfig()
        self.fsm = get_sniper_fsm()
        self.ema_tracker = get_ema_tracker()
        self.event_logger = get_event_logger()

        self._enabled = False
        self._last_check_time: Optional[datetime] = None

        # Track impulse highs per symbol for pullback calculation
        self._impulse_highs: Dict[str, float] = {}
        self._pullback_lows: Dict[str, float] = {}
        self._volume_history: Dict[str, List[float]] = {}

        # Setup FSM callbacks
        self.fsm.set_callbacks(
            on_state_change=self._on_state_change,
            on_trade_event=self._on_trade_event
        )

    # ========================================
    # TIME GATE (HARD EXCLUSIONS)
    # ========================================

    def is_active_window(self) -> bool:
        """
        Check if current time is within strategy active window.

        HARD TIME GATE:
        - Active only when market_session == OPEN
        - time >= 9:40 AM ET
        - time <= 11:00 AM ET
        """
        now_et = datetime.now(ET_TZ)
        current_time = now_et.time()

        # Check weekday
        if now_et.weekday() >= 5:  # Saturday or Sunday
            return False

        # Check time window
        return self.config.start_time_et <= current_time <= self.config.end_time_et

    def get_market_session(self) -> str:
        """Get current market session"""
        try:
            from ai.halt_detector import get_market_session
            return get_market_session()
        except ImportError:
            # Fallback
            now_et = datetime.now(ET_TZ).time()
            if dt_time(9, 30) <= now_et < dt_time(16, 0):
                return "OPEN"
            elif dt_time(4, 0) <= now_et < dt_time(9, 30):
                return "PRE_MARKET"
            elif dt_time(16, 0) <= now_et < dt_time(20, 0):
                return "AFTER_HOURS"
            return "CLOSED"

    def check_hard_exclusions(self, symbol: str) -> Tuple[bool, str]:
        """
        Check hard exclusions. Strategy MUST NOT run when:
        - HRDC mode = FLAT_ONLY or DEFENSIVE
        - Symbol is HALTED or RESUMING
        - Market session != OPEN
        - Symbol lost VWAP
        """
        # Check market session
        session = self.get_market_session()
        if session != "OPEN":
            return False, f"Market session is {session}, not OPEN"

        # Check time window
        if not self.is_active_window():
            now_et = datetime.now(ET_TZ).time()
            return False, f"Outside active window ({self.config.start_time_et} - {self.config.end_time_et}), current: {now_et}"

        # Check HRDC mode
        try:
            from ai.halt_detector import get_halt_detector
            detector = get_halt_detector()
            hrdc_status = detector.get_hrdc_status(symbol)
            if hrdc_status:
                mode = hrdc_status.get("mode", "")
                if mode in ["FLAT_ONLY", "DEFENSIVE"]:
                    return False, f"HRDC mode is {mode}"
        except Exception:
            pass  # HRDC not available, continue

        # Check if halted
        try:
            from ai.halt_detector import get_halt_detector
            detector = get_halt_detector()
            if symbol in detector.current_halts:
                return False, "Symbol is currently HALTED"
        except Exception:
            pass

        return True, "Passed hard exclusions"

    # ========================================
    # ATS QUALIFICATION (NON-NEGOTIABLE)
    # ========================================

    def check_ats_qualification(self, symbol: str, quote: Dict) -> QualificationResult:
        """
        ATS Qualification Check.

        Symbol is eligible only if ALL are true:
        - ATS score >= ATS_MIN_CONFIDENCE
        - Price above VWAP
        - Structure = higher highs + higher lows
        - RVOL >= RVOL_MIN
        - HRDC mode != FLAT_ONLY, DEFENSIVE

        If ATS fails → strategy is disabled for that symbol.
        """
        result = QualificationResult(symbol=symbol)

        # Get ATS score
        try:
            from ai.ats import get_ats_feed
            ats = get_ats_feed()
            ats_state = ats.get_state(symbol)
            if ats_state:
                result.ats_score = ats_state.get("score", 0) / 100.0  # Normalize to 0-1
            else:
                result.ats_score = 0.0
        except Exception:
            # ATS not available - use quote data to estimate
            result.ats_score = self._estimate_ats_score(symbol, quote)

        # Check ATS threshold
        if result.ats_score >= self.config.ats_min_confidence:
            result.reasons.append(f"ATS score {result.ats_score:.2f} >= {self.config.ats_min_confidence}")
        else:
            result.disqualify_reasons.append(f"ATS score {result.ats_score:.2f} < {self.config.ats_min_confidence}")

        # Check VWAP
        vwap = quote.get("vwap", 0)
        price = quote.get("price", quote.get("last", 0))
        if vwap > 0 and price > vwap:
            result.vwap_status = "ABOVE"
            result.reasons.append("Price above VWAP")
        elif vwap > 0:
            result.vwap_status = "BELOW"
            result.disqualify_reasons.append("Price below VWAP - lost support")
        else:
            result.vwap_status = "UNKNOWN"

        # Check RVOL
        result.rvol = quote.get("rvol", quote.get("relative_volume", 1.0))
        if result.rvol >= self.config.rvol_min:
            result.reasons.append(f"RVOL {result.rvol:.1f} >= {self.config.rvol_min}")
        else:
            result.disqualify_reasons.append(f"RVOL {result.rvol:.1f} < {self.config.rvol_min}")

        # Check HRDC mode
        try:
            from ai.halt_detector import get_halt_detector
            detector = get_halt_detector()
            hrdc_status = detector.get_hrdc_status(symbol)
            if hrdc_status:
                result.hrdc_mode = hrdc_status.get("mode", "")
                if result.hrdc_mode in ["FLAT_ONLY", "DEFENSIVE"]:
                    result.disqualify_reasons.append(f"HRDC mode {result.hrdc_mode}")
                elif result.hrdc_mode == "MOMENTUM_ALLOWED":
                    result.reasons.append("HRDC = MOMENTUM_ALLOWED")
        except Exception:
            result.hrdc_mode = "N/A"

        # Check structure (higher highs, higher lows)
        ema_state = self.ema_tracker.get_state(symbol)
        if ema_state and ema_state.ema9_above_ema20 and ema_state.ema_slope_positive:
            result.structure_bullish = True
            result.reasons.append("Bullish structure (EMA9 > EMA20, slope positive)")
        else:
            result.disqualify_reasons.append("Structure not bullish")

        # Final qualification decision
        result.qualified = len(result.disqualify_reasons) == 0

        return result

    def _estimate_ats_score(self, symbol: str, quote: Dict) -> float:
        """Estimate ATS score from quote data when ATS module not available"""
        score = 0.0

        # Price change component (up to 0.3)
        change_pct = quote.get("change_percent", 0)
        if change_pct > 10:
            score += 0.3
        elif change_pct > 5:
            score += 0.2
        elif change_pct > 2:
            score += 0.1

        # Volume component (up to 0.3)
        rvol = quote.get("rvol", quote.get("relative_volume", 1))
        if rvol > 5:
            score += 0.3
        elif rvol > 3:
            score += 0.2
        elif rvol > 2:
            score += 0.1

        # VWAP component (up to 0.2)
        vwap = quote.get("vwap", 0)
        price = quote.get("price", quote.get("last", 0))
        if vwap > 0 and price > vwap:
            score += 0.2

        # EMA structure component (up to 0.2)
        ema_state = self.ema_tracker.get_state(symbol)
        if ema_state and ema_state.ema9_above_ema20:
            score += 0.2

        return min(score, 1.0)

    # ========================================
    # 9 EMA PULLBACK DETECTION
    # ========================================

    def update_price(self, symbol: str, price: float, volume: float = 0) -> Dict:
        """
        Update price data and check for pullback setup.

        Returns dict with pullback status.
        """
        symbol = symbol.upper()

        # Update EMA tracker
        ema_state = self.ema_tracker.update(symbol, price)

        # Track volume history
        if symbol not in self._volume_history:
            self._volume_history[symbol] = []
        self._volume_history[symbol].append(volume)
        if len(self._volume_history[symbol]) > 20:
            self._volume_history[symbol] = self._volume_history[symbol][-20:]

        # Track impulse high
        if symbol not in self._impulse_highs or price > self._impulse_highs[symbol]:
            self._impulse_highs[symbol] = price

        # Track pullback low
        fsm_state = self.fsm.get_state(symbol)
        if fsm_state.state == SniperState.WAIT_9EMA_PULLBACK:
            if symbol not in self._pullback_lows or price < self._pullback_lows[symbol]:
                self._pullback_lows[symbol] = price

        return {
            "symbol": symbol,
            "price": price,
            "ema_9": ema_state.ema_9,
            "distance_from_ema9_pct": ema_state.distance_from_ema9_pct,
            "is_pullback_zone": ema_state.is_pullback_zone,
            "bullish_structure": ema_state.ema9_above_ema20 and ema_state.ema_slope_positive
        }

    def check_pullback_setup(self, symbol: str, price: float) -> Tuple[bool, str]:
        """
        Check if pullback setup requirements are met.

        Pullback Requirements:
        - Price pulls back toward 9 EMA
        - Pullback depth <= MAX_PULLBACK_PCT of last impulse
        - Volume decreases during pullback
        - VWAP is not lost
        - Candles compress (no wide range flush)

        HARD RULE: If price does not pull back → DO NOTHING
        """
        symbol = symbol.upper()
        ema_state = self.ema_tracker.get_state(symbol)

        if not ema_state:
            return False, "No EMA data"

        # Check if in pullback zone (near but above 9 EMA)
        if not ema_state.is_pullback_zone and not ema_state.price_at_ema9:
            if ema_state.distance_from_ema9_pct > 0.5:
                return False, f"Price too far from EMA9 ({ema_state.distance_from_ema9_pct:.1f}%)"
            elif ema_state.price_below_ema9:
                return False, "Price broke below EMA9"

        # Check pullback depth
        impulse_high = self._impulse_highs.get(symbol, price)
        if impulse_high > 0:
            pullback_depth = ((impulse_high - price) / impulse_high) * 100
            if pullback_depth > self.config.max_pullback_pct:
                return False, f"Pullback too deep ({pullback_depth:.1f}% > {self.config.max_pullback_pct}%)"

        # Check volume decrease
        if self.config.require_volume_decrease:
            volumes = self._volume_history.get(symbol, [])
            if len(volumes) >= 3:
                recent_avg = sum(volumes[-3:]) / 3
                prior_avg = sum(volumes[-6:-3]) / 3 if len(volumes) >= 6 else recent_avg
                if recent_avg > prior_avg * 1.1:  # Volume should decrease, not increase
                    return False, "Volume not decreasing during pullback"

        # Check bullish structure maintained
        if not ema_state.ema9_above_ema20:
            return False, "Lost bullish EMA structure"

        return True, "Pullback setup valid"

    # ========================================
    # ENTRY CONFIRMATION (SNIPER LOGIC)
    # ========================================

    def check_entry_confirmation(self, symbol: str, price: float, volume: float,
                                  high: float, low: float, is_green: bool) -> EntrySignal:
        """
        Check for entry confirmation.

        Entry triggers only if ONE occurs:
        - Reclaim of pullback high with volume expansion
        OR
        - Strong bullish candle off 9 EMA with volume

        NO prediction. NO anticipation.
        """
        signal = EntrySignal(symbol=symbol)
        symbol = symbol.upper()

        ema_state = self.ema_tracker.get_state(symbol)
        if not ema_state:
            return signal

        pullback_low = self._pullback_lows.get(symbol, low)
        impulse_high = self._impulse_highs.get(symbol, high)

        signal.pullback_low = pullback_low
        signal.ema9_at_entry = ema_state.ema_9

        # Calculate pullback depth
        if impulse_high > 0 and pullback_low < impulse_high:
            signal.pullback_depth_pct = ((impulse_high - pullback_low) / impulse_high) * 100

        # Check for volume expansion
        volumes = self._volume_history.get(symbol, [])
        avg_volume = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else volume
        volume_expanding = volume > avg_volume * self.config.volume_expansion_ratio

        # CONFIRMATION 1: Reclaim of pullback high with volume expansion
        pullback_high = pullback_low * (1 + self.config.reclaim_threshold_pct / 100)
        if price > pullback_high and volume_expanding and is_green:
            signal.confirmed = True
            signal.entry_price = price
            signal.reason = "RECLAIM_PULLBACK_HIGH"

            # Set stop below pullback low
            stop_level = pullback_low
            if self.config.stop_below_ema9:
                stop_level = min(stop_level, ema_state.ema_9)
            signal.stop_loss = stop_level * (1 - self.config.stop_buffer_pct / 100)

            # Set target at prior high
            signal.target = impulse_high

            return signal

        # CONFIRMATION 2: Strong bullish candle off 9 EMA
        candle_range = high - low
        candle_body = abs(price - low) if is_green else abs(high - price)
        is_strong_candle = candle_body > candle_range * 0.6  # Body > 60% of range

        if (ema_state.price_at_ema9 or ema_state.is_pullback_zone) and is_strong_candle and is_green and volume_expanding:
            signal.confirmed = True
            signal.entry_price = price
            signal.reason = "BULLISH_CANDLE_OFF_EMA"

            # Set stop just below EMA9
            signal.stop_loss = ema_state.ema_9 * (1 - self.config.stop_buffer_pct / 100)

            # Set target
            signal.target = impulse_high

            return signal

        return signal

    # ========================================
    # MAIN PROCESS LOOP
    # ========================================

    def process(self, symbol: str, quote: Dict) -> Dict:
        """
        Main processing loop for a symbol.

        This is the entry point called by the scalper/monitor.

        Returns dict with:
        - action: QUALIFY, WAIT_PULLBACK, ENTER, EXIT, NO_ACTION
        - details: Action-specific details
        """
        symbol = symbol.upper()
        result = {
            "symbol": symbol,
            "action": "NO_ACTION",
            "details": {},
            "timestamp": datetime.now().isoformat()
        }

        # Check hard exclusions first
        can_trade, exclusion_reason = self.check_hard_exclusions(symbol)
        if not can_trade:
            result["action"] = "EXCLUDED"
            result["details"]["reason"] = exclusion_reason
            return result

        # Get current FSM state
        fsm_state = self.fsm.get_state(symbol)
        price = quote.get("price", quote.get("last", 0))
        volume = quote.get("volume", 0)

        # Update price tracking
        self.update_price(symbol, price, volume)

        # State-specific processing
        if fsm_state.state == SniperState.IDLE:
            # Check ATS qualification
            qual = self.check_ats_qualification(symbol, quote)
            result["details"]["qualification"] = qual.to_dict()

            if qual.qualified:
                self.fsm.qualify_with_ats(symbol, qual.ats_score)
                result["action"] = "QUALIFIED"
            else:
                result["action"] = "NOT_QUALIFIED"

        elif fsm_state.state == SniperState.ATS_QUALIFIED:
            # Start watching for pullback
            impulse_high = self._impulse_highs.get(symbol, price)
            self.fsm.start_pullback_watch(symbol, impulse_high)
            result["action"] = "WATCHING_PULLBACK"

        elif fsm_state.state == SniperState.WAIT_9EMA_PULLBACK:
            # Check if pullback setup is valid
            setup_valid, setup_reason = self.check_pullback_setup(symbol, price)
            result["details"]["pullback_status"] = setup_reason

            if setup_valid:
                pullback_low = self._pullback_lows.get(symbol, price)
                self.fsm.detect_pullback(symbol, price, pullback_low)
                result["action"] = "PULLBACK_DETECTED"
            else:
                # Check if we should cancel (e.g., lost VWAP)
                if "broke below" in setup_reason.lower() or "lost" in setup_reason.lower():
                    self.fsm.cancel(symbol, setup_reason)
                    result["action"] = "CANCELLED"

        elif fsm_state.state == SniperState.SNIPER_CONFIRMATION:
            # Check for entry confirmation
            high = quote.get("high", price)
            low = quote.get("low", price)
            is_green = quote.get("is_green", price >= quote.get("open", price))

            entry = self.check_entry_confirmation(symbol, price, volume, high, low, is_green)
            result["details"]["entry_signal"] = entry.to_dict()

            if entry.confirmed:
                self.fsm.confirm_entry(
                    symbol,
                    entry.entry_price,
                    entry.stop_loss,
                    entry.target,
                    entry.reason
                )
                result["action"] = "ENTER"
                result["details"]["entry_price"] = entry.entry_price
                result["details"]["stop_loss"] = entry.stop_loss
                result["details"]["target"] = entry.target

        elif fsm_state.state == SniperState.SCALP_ENTRY:
            # Check exit conditions
            exit_result = self._check_exit_conditions(symbol, price, fsm_state)
            if exit_result["should_exit"]:
                is_win = price > fsm_state.entry_price
                self.fsm.trigger_exit(symbol, price, exit_result["reason"], is_win)
                self.fsm.start_cooldown(symbol)
                result["action"] = "EXIT"
                result["details"]["exit_price"] = price
                result["details"]["exit_reason"] = exit_result["reason"]
                result["details"]["pnl"] = price - fsm_state.entry_price

        elif fsm_state.state == SniperState.COOLDOWN:
            # Check if cooldown elapsed
            if self.fsm.end_cooldown(symbol):
                result["action"] = "COOLDOWN_COMPLETE"

        return result

    def _check_exit_conditions(self, symbol: str, price: float, fsm_state) -> Dict:
        """Check if exit conditions are met"""
        result = {"should_exit": False, "reason": ""}

        # Stop loss hit
        if price <= fsm_state.stop_loss:
            result["should_exit"] = True
            result["reason"] = "STOP_LOSS"
            return result

        # Target hit
        if price >= fsm_state.target_price:
            result["should_exit"] = True
            result["reason"] = "TARGET_HIT"
            return result

        # Trail with 9 EMA
        if self.config.trail_with_ema9:
            ema_state = self.ema_tracker.get_state(symbol)
            if ema_state and price < ema_state.ema_9:
                result["should_exit"] = True
                result["reason"] = "BROKE_EMA9_TRAIL"
                return result

        return result

    # ========================================
    # CALLBACKS
    # ========================================

    def _on_state_change(self, symbol: str, old_state: SniperState,
                          new_state: SniperState, reason: str):
        """Handle FSM state changes"""
        logger.info(
            f"[ATS_9EMA_SNIPER] {symbol}: {old_state.value} -> {new_state.value} | {reason}"
        )

    def _on_trade_event(self, event: Dict):
        """Handle trade events - log to event logger"""
        event_type = event.get("event", "")

        if event_type == "SNIPER_ENTRY":
            log_sniper_attempt(
                symbol=event.get("symbol", ""),
                ats_score=event.get("ats_score", 0),
                hrdc_mode=event.get("hrdc_mode", ""),
                result="PENDING",
                entry_reason=event.get("reason", ""),
                entry_price=event.get("entry_price", 0),
                pullback_depth_pct=event.get("pullback_depth_pct", 0)
            )

        elif event_type == "SNIPER_EXIT":
            result = event.get("result", "LOSS")
            log_sniper_attempt(
                symbol=event.get("symbol", ""),
                ats_score=0,  # Already logged on entry
                hrdc_mode="",
                result=result,
                entry_price=event.get("entry_price", 0),
                exit_price=event.get("exit_price", 0),
                pnl=event.get("pnl", 0),
                exit_reason=event.get("reason", "")
            )

        elif event_type == "ATS_9EMA_SNIPER_ATTEMPT":
            # NO_TRADE result
            log_sniper_attempt(**event)

    # ========================================
    # STATUS & CONTROL
    # ========================================

    def enable(self):
        """Enable the strategy"""
        self._enabled = True
        logger.info("[ATS_9EMA_SNIPER] Strategy ENABLED")

    def disable(self):
        """Disable the strategy"""
        self._enabled = False
        logger.info("[ATS_9EMA_SNIPER] Strategy DISABLED")

    def is_enabled(self) -> bool:
        """Check if strategy is enabled"""
        return self._enabled

    def get_status(self) -> Dict:
        """Get strategy status"""
        return {
            "enabled": self._enabled,
            "active_window": self.is_active_window(),
            "market_session": self.get_market_session(),
            "config": self.config.to_dict(),
            "active_symbols": self.fsm.get_active_symbols(),
            "fsm_states": self.fsm.get_all_states(),
            "today_stats": self.event_logger.get_today_stats()
        }

    def reset_daily(self):
        """Reset daily counters"""
        self.fsm.reset_daily()
        self._impulse_highs.clear()
        self._pullback_lows.clear()
        self._volume_history.clear()
        logger.info("[ATS_9EMA_SNIPER] Daily reset complete")


# Singleton instance
_sniper_strategy: Optional[ATS9EMASniperStrategy] = None


def get_sniper_strategy() -> ATS9EMASniperStrategy:
    """Get singleton sniper strategy instance"""
    global _sniper_strategy
    if _sniper_strategy is None:
        _sniper_strategy = ATS9EMASniperStrategy()
    return _sniper_strategy
