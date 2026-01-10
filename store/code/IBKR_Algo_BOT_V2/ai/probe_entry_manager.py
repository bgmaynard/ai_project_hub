"""
Probe Entry Execution Layer
===========================
Controlled initiation layer for scalper execution.

Problem: Chronos + ATS grant permission, but scalper rarely finds a "perfect"
trigger, resulting in zero executions.

Solution: Add a Probe Entry execution type with:
- Size: 25-33% of normal position
- Tighter stops than normal
- No averaging down
- Cooldown between probes

Probe Entry Conditions (ALL required):
1. ATS state in {IGNITING, ACTIVE}
2. Chronos micro confidence >= threshold
3. Volume acceleration detected
4. Price crosses a micro level (VWAP reclaim, premarket high, range break)

Behavior After Probe:
- If stopped → mark PROBE_FAILED, block re-entry for cooldown
- If holds + continuation → allow normal scalper logic

Safety Caps:
- Max 1 probe per symbol per session
- Max N probes per hour (configurable)
- Disabled if macro regime == CRASH / HALT_RISK
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pytz

logger = logging.getLogger(__name__)


class ProbeState(Enum):
    """State of a probe entry"""
    PENDING = "PENDING"          # Conditions met, awaiting entry
    ACTIVE = "ACTIVE"            # Probe position open
    CONFIRMED = "CONFIRMED"      # Probe held, continuation detected
    STOPPED = "STOPPED"          # Probe hit stop loss
    FAILED = "FAILED"            # Probe failed (exhausted, invalidated)
    COOLDOWN = "COOLDOWN"        # Symbol in cooldown after probe


class ProbeTriggerType(Enum):
    """Types of micro-level triggers for probe entry"""
    VWAP_RECLAIM = "VWAP_RECLAIM"
    PREMARKET_HIGH_BREAK = "PREMARKET_HIGH_BREAK"
    RANGE_BREAK_1M = "RANGE_BREAK_1M"
    RANGE_BREAK_3M = "RANGE_BREAK_3M"
    HOD_BREAK = "HOD_BREAK"
    VOLUME_SPIKE = "VOLUME_SPIKE"


@dataclass
class ProbeConfig:
    """Configuration for probe entries"""
    enabled: bool = True

    # Position sizing
    size_multiplier: float = 0.30  # 30% of normal position (25-33% range)

    # Stop loss (tighter than normal)
    stop_loss_percent: float = 1.5  # Tighter than normal scalper stop

    # Thresholds
    min_micro_confidence: float = 0.60
    min_volume_accel: float = 1.5  # Volume acceleration threshold
    qualifying_ats_states: List[str] = field(default_factory=lambda: ["IGNITING", "ACTIVE"])

    # Safety caps
    max_probes_per_symbol_per_session: int = 1
    max_probes_per_hour: int = 3
    cooldown_minutes: int = 15

    # Blocked regimes
    blocked_regimes: List[str] = field(default_factory=lambda: ["CRASH", "HALT_RISK", "TRENDING_DOWN"])

    # Confirmation thresholds
    confirmation_gain_percent: float = 0.5  # Gain needed to confirm probe
    confirmation_time_seconds: int = 30  # Time to hold for confirmation


@dataclass
class ProbeEntry:
    """Record of a probe entry"""
    symbol: str
    trigger_type: str
    entry_price: float
    entry_time: str
    size: int
    stop_price: float
    state: ProbeState

    # Context at entry
    ats_state: str
    micro_confidence: float
    macro_regime: str
    volume_accel: float

    # Outcome (filled after exit)
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    high_water: float = 0.0  # Highest price during probe

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "trigger_type": self.trigger_type,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "size": self.size,
            "stop_price": self.stop_price,
            "state": self.state.value,
            "ats_state": self.ats_state,
            "micro_confidence": f"{self.micro_confidence:.0%}",
            "macro_regime": self.macro_regime,
            "volume_accel": f"{self.volume_accel:.1f}x",
            "exit_price": self.exit_price,
            "exit_time": self.exit_time,
            "exit_reason": self.exit_reason,
            "pnl": self.pnl,
            "high_water": self.high_water
        }


class ProbeEntryManager:
    """
    Manages probe entry execution layer.

    Allows controlled initiation when ATS + Chronos grant permission
    but normal triggers are not "perfect" enough.
    """

    def __init__(self, config: ProbeConfig = None):
        self.config = config or ProbeConfig()
        self.et_tz = pytz.timezone('US/Eastern')

        # Active probes by symbol
        self.active_probes: Dict[str, ProbeEntry] = {}

        # Probe history for session
        self.probe_history: List[ProbeEntry] = []
        self.max_history = 100

        # Session tracking
        self.session_date: str = ""
        self.probes_per_symbol: Dict[str, int] = {}  # Symbol → count this session
        self.hourly_probes: List[datetime] = []  # Timestamps of recent probes

        # Cooldown tracking
        self.cooldowns: Dict[str, datetime] = {}  # Symbol → cooldown expires at

        # Stats
        self.probe_attempts = 0
        self.probe_entries = 0
        self.probe_confirmations = 0
        self.probe_stops = 0
        self.total_pnl = 0.0

        logger.info(f"ProbeEntryManager initialized (enabled: {self.config.enabled})")

    def _reset_session_if_needed(self):
        """Reset session tracking if new day"""
        today = datetime.now(self.et_tz).strftime("%Y-%m-%d")
        if self.session_date != today:
            self.session_date = today
            self.probes_per_symbol.clear()
            self.cooldowns.clear()
            self.hourly_probes.clear()
            logger.info(f"Probe session reset for {today}")

    def _count_hourly_probes(self) -> int:
        """Count probes in the last hour"""
        cutoff = datetime.now(self.et_tz) - timedelta(hours=1)
        self.hourly_probes = [t for t in self.hourly_probes if t > cutoff]
        return len(self.hourly_probes)

    def _is_in_cooldown(self, symbol: str) -> Tuple[bool, Optional[str]]:
        """Check if symbol is in cooldown"""
        if symbol not in self.cooldowns:
            return False, None

        expires = self.cooldowns[symbol]
        now = datetime.now(self.et_tz)

        if now >= expires:
            del self.cooldowns[symbol]
            return False, None

        remaining = (expires - now).total_seconds() / 60
        return True, f"Cooldown: {remaining:.1f} min remaining"

    def _detect_volume_acceleration(
        self,
        current_volume: int,
        avg_volume: int,
        recent_volume: int = 0
    ) -> Tuple[bool, float]:
        """
        Detect volume acceleration.

        Args:
            current_volume: Today's total volume
            avg_volume: Average daily volume
            recent_volume: Volume in last N minutes (optional)

        Returns:
            (detected, acceleration_ratio)
        """
        if avg_volume <= 0:
            return False, 0.0

        # Calculate relative volume
        rel_vol = current_volume / avg_volume

        # Simple acceleration: RelVol >= threshold
        detected = rel_vol >= self.config.min_volume_accel

        return detected, rel_vol

    def _detect_micro_trigger(
        self,
        symbol: str,
        current_price: float,
        vwap: float = 0,
        premarket_high: float = 0,
        range_high_1m: float = 0,
        range_high_3m: float = 0,
        day_high: float = 0
    ) -> Tuple[bool, Optional[ProbeTriggerType], str]:
        """
        Detect if price crosses a micro level.

        Returns:
            (triggered, trigger_type, reason)
        """
        reasons = []

        # Check VWAP reclaim
        if vwap > 0 and current_price > vwap:
            pct_above = (current_price - vwap) / vwap * 100
            if 0 < pct_above < 1.0:  # Just crossed above (within 1%)
                return True, ProbeTriggerType.VWAP_RECLAIM, f"Price reclaimed VWAP (${vwap:.2f})"

        # Check premarket high break
        if premarket_high > 0 and current_price > premarket_high:
            pct_above = (current_price - premarket_high) / premarket_high * 100
            if 0 < pct_above < 0.5:  # Just broke (within 0.5%)
                return True, ProbeTriggerType.PREMARKET_HIGH_BREAK, f"Broke premarket high (${premarket_high:.2f})"

        # Check 1-minute range break
        if range_high_1m > 0 and current_price > range_high_1m:
            pct_above = (current_price - range_high_1m) / range_high_1m * 100
            if 0 < pct_above < 0.3:
                return True, ProbeTriggerType.RANGE_BREAK_1M, f"Broke 1m range high (${range_high_1m:.2f})"

        # Check 3-minute range break
        if range_high_3m > 0 and current_price > range_high_3m:
            pct_above = (current_price - range_high_3m) / range_high_3m * 100
            if 0 < pct_above < 0.5:
                return True, ProbeTriggerType.RANGE_BREAK_3M, f"Broke 3m range high (${range_high_3m:.2f})"

        # Check HOD break
        if day_high > 0 and current_price > day_high:
            pct_above = (current_price - day_high) / day_high * 100
            if 0 < pct_above < 0.3:
                return True, ProbeTriggerType.HOD_BREAK, f"Broke HOD (${day_high:.2f})"

        return False, None, "No micro trigger detected"

    def check_probe_eligibility(
        self,
        symbol: str,
        ats_state: str,
        micro_confidence: float,
        macro_regime: str,
        current_price: float,
        current_volume: int = 0,
        avg_volume: int = 0,
        vwap: float = 0,
        premarket_high: float = 0,
        range_high_1m: float = 0,
        range_high_3m: float = 0,
        day_high: float = 0
    ) -> Tuple[bool, str, Optional[ProbeTriggerType]]:
        """
        Check if a probe entry is eligible.

        Args:
            symbol: Stock symbol
            ats_state: Current ATS state
            micro_confidence: Chronos micro confidence
            macro_regime: Current macro regime
            current_price: Current price
            current_volume: Today's volume
            avg_volume: Average daily volume
            vwap: Current VWAP
            premarket_high: Premarket high
            range_high_1m: 1-minute range high
            range_high_3m: 3-minute range high
            day_high: Day's high

        Returns:
            (eligible, reason, trigger_type)
        """
        self._reset_session_if_needed()
        self.probe_attempts += 1

        # Check if enabled
        if not self.config.enabled:
            return False, "Probe entries disabled", None

        # Check if already have active probe
        if symbol in self.active_probes:
            return False, f"Active probe already exists for {symbol}", None

        # Check blocked regimes (CRITICAL SAFETY)
        if macro_regime.upper() in [r.upper() for r in self.config.blocked_regimes]:
            return False, f"Blocked regime: {macro_regime}", None

        # Check ATS state
        if ats_state.upper() not in [s.upper() for s in self.config.qualifying_ats_states]:
            return False, f"ATS state '{ats_state}' not in {self.config.qualifying_ats_states}", None

        # Check micro confidence
        if micro_confidence < self.config.min_micro_confidence:
            return False, f"Micro confidence {micro_confidence:.0%} < {self.config.min_micro_confidence:.0%}", None

        # Check session limit per symbol
        symbol_probes = self.probes_per_symbol.get(symbol, 0)
        if symbol_probes >= self.config.max_probes_per_symbol_per_session:
            return False, f"Max probes per symbol ({self.config.max_probes_per_symbol_per_session}) reached", None

        # Check hourly limit
        hourly_count = self._count_hourly_probes()
        if hourly_count >= self.config.max_probes_per_hour:
            return False, f"Max probes per hour ({self.config.max_probes_per_hour}) reached", None

        # Check cooldown
        in_cooldown, cooldown_reason = self._is_in_cooldown(symbol)
        if in_cooldown:
            return False, cooldown_reason, None

        # Check volume acceleration
        vol_detected, vol_accel = self._detect_volume_acceleration(current_volume, avg_volume)
        if not vol_detected and current_volume > 0:
            return False, f"Volume acceleration {vol_accel:.1f}x < {self.config.min_volume_accel:.1f}x", None

        # Check micro trigger
        trigger_detected, trigger_type, trigger_reason = self._detect_micro_trigger(
            symbol, current_price, vwap, premarket_high, range_high_1m, range_high_3m, day_high
        )

        if not trigger_detected:
            return False, trigger_reason, None

        # All conditions met!
        logger.info(
            f"PROBE_ENTRY_ELIGIBLE: {symbol} | ATS={ats_state}, conf={micro_confidence:.0%}, "
            f"trigger={trigger_type.value if trigger_type else 'NONE'}"
        )

        return True, f"Probe eligible: {trigger_reason}", trigger_type

    def create_probe_entry(
        self,
        symbol: str,
        trigger_type: ProbeTriggerType,
        entry_price: float,
        normal_position_size: int,
        ats_state: str,
        micro_confidence: float,
        macro_regime: str,
        volume_accel: float
    ) -> ProbeEntry:
        """
        Create a probe entry.

        Args:
            symbol: Stock symbol
            trigger_type: What triggered the probe
            entry_price: Entry price
            normal_position_size: What normal scalper would use
            ats_state: Current ATS state
            micro_confidence: Chronos micro confidence
            macro_regime: Current macro regime
            volume_accel: Volume acceleration ratio

        Returns:
            ProbeEntry object
        """
        now = datetime.now(self.et_tz)

        # Calculate probe size (25-33% of normal)
        probe_size = max(1, int(normal_position_size * self.config.size_multiplier))

        # Calculate tighter stop
        stop_price = entry_price * (1 - self.config.stop_loss_percent / 100)

        probe = ProbeEntry(
            symbol=symbol,
            trigger_type=trigger_type.value,
            entry_price=entry_price,
            entry_time=now.isoformat(),
            size=probe_size,
            stop_price=stop_price,
            state=ProbeState.ACTIVE,
            ats_state=ats_state,
            micro_confidence=micro_confidence,
            macro_regime=macro_regime,
            volume_accel=volume_accel,
            high_water=entry_price
        )

        # Track probe
        self.active_probes[symbol] = probe
        self.probes_per_symbol[symbol] = self.probes_per_symbol.get(symbol, 0) + 1
        self.hourly_probes.append(now)
        self.probe_entries += 1

        logger.warning(
            f"PROBE_ENTRY_ATTEMPT: {symbol} | "
            f"trigger={trigger_type.value}, price=${entry_price:.2f}, "
            f"size={probe_size}, stop=${stop_price:.2f}, "
            f"ATS={ats_state}, conf={micro_confidence:.0%}"
        )

        return probe

    def update_probe(
        self,
        symbol: str,
        current_price: float
    ) -> Tuple[Optional[ProbeState], Optional[str]]:
        """
        Update a probe with current price.

        Returns:
            (new_state, action) - action is "STOP", "CONFIRM", or None
        """
        if symbol not in self.active_probes:
            return None, None

        probe = self.active_probes[symbol]

        if probe.state != ProbeState.ACTIVE:
            return probe.state, None

        # Update high water mark
        if current_price > probe.high_water:
            probe.high_water = current_price

        # Check stop loss
        if current_price <= probe.stop_price:
            return self._stop_probe(probe, current_price, "STOP_LOSS")

        # Check for confirmation
        gain_pct = (current_price - probe.entry_price) / probe.entry_price * 100
        entry_time = datetime.fromisoformat(probe.entry_time)
        if entry_time.tzinfo is None:
            entry_time = self.et_tz.localize(entry_time)
        elapsed = (datetime.now(self.et_tz) - entry_time).total_seconds()

        if gain_pct >= self.config.confirmation_gain_percent and elapsed >= self.config.confirmation_time_seconds:
            return self._confirm_probe(probe, current_price)

        return ProbeState.ACTIVE, None

    def _stop_probe(
        self,
        probe: ProbeEntry,
        exit_price: float,
        reason: str
    ) -> Tuple[ProbeState, str]:
        """Mark probe as stopped"""
        now = datetime.now(self.et_tz)

        probe.state = ProbeState.STOPPED
        probe.exit_price = exit_price
        probe.exit_time = now.isoformat()
        probe.exit_reason = reason
        probe.pnl = (exit_price - probe.entry_price) * probe.size

        self.total_pnl += probe.pnl
        self.probe_stops += 1

        # Set cooldown
        self.cooldowns[probe.symbol] = now + timedelta(minutes=self.config.cooldown_minutes)

        # Move to history
        if probe.symbol in self.active_probes:
            del self.active_probes[probe.symbol]
        self.probe_history.append(probe)
        if len(self.probe_history) > self.max_history:
            self.probe_history.pop(0)

        logger.warning(
            f"PROBE_ENTRY_STOPPED: {probe.symbol} | "
            f"entry=${probe.entry_price:.2f}, exit=${exit_price:.2f}, "
            f"pnl=${probe.pnl:.2f}, reason={reason}"
        )

        return ProbeState.STOPPED, "STOP"

    def _confirm_probe(
        self,
        probe: ProbeEntry,
        current_price: float
    ) -> Tuple[ProbeState, str]:
        """Mark probe as confirmed (continuation detected)"""
        probe.state = ProbeState.CONFIRMED

        self.probe_confirmations += 1

        logger.warning(
            f"PROBE_ENTRY_CONFIRMED: {probe.symbol} | "
            f"entry=${probe.entry_price:.2f}, current=${current_price:.2f}, "
            f"gain={((current_price - probe.entry_price) / probe.entry_price * 100):.2f}%"
        )

        return ProbeState.CONFIRMED, "CONFIRM"

    def close_probe(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "MANUAL"
    ) -> Optional[ProbeEntry]:
        """
        Close an active or confirmed probe.

        Returns:
            The closed probe, or None if not found
        """
        if symbol not in self.active_probes:
            return None

        probe = self.active_probes[symbol]
        now = datetime.now(self.et_tz)

        probe.exit_price = exit_price
        probe.exit_time = now.isoformat()
        probe.exit_reason = reason
        probe.pnl = (exit_price - probe.entry_price) * probe.size

        if probe.state == ProbeState.ACTIVE:
            probe.state = ProbeState.STOPPED if probe.pnl < 0 else ProbeState.CONFIRMED
            if probe.pnl < 0:
                self.probe_stops += 1
            else:
                self.probe_confirmations += 1

        self.total_pnl += probe.pnl

        # Move to history
        del self.active_probes[symbol]
        self.probe_history.append(probe)
        if len(self.probe_history) > self.max_history:
            self.probe_history.pop(0)

        logger.info(
            f"PROBE_CLOSED: {symbol} | pnl=${probe.pnl:.2f}, reason={reason}"
        )

        return probe

    def get_active_probe(self, symbol: str) -> Optional[ProbeEntry]:
        """Get active probe for symbol"""
        return self.active_probes.get(symbol)

    def get_status(self) -> Dict[str, Any]:
        """Get probe manager status"""
        return {
            "enabled": self.config.enabled,
            "config": {
                "size_multiplier": self.config.size_multiplier,
                "stop_loss_percent": self.config.stop_loss_percent,
                "min_micro_confidence": self.config.min_micro_confidence,
                "min_volume_accel": self.config.min_volume_accel,
                "qualifying_ats_states": self.config.qualifying_ats_states,
                "max_probes_per_symbol_per_session": self.config.max_probes_per_symbol_per_session,
                "max_probes_per_hour": self.config.max_probes_per_hour,
                "cooldown_minutes": self.config.cooldown_minutes,
                "blocked_regimes": self.config.blocked_regimes
            },
            "stats": {
                "probe_attempts": self.probe_attempts,
                "probe_entries": self.probe_entries,
                "probe_confirmations": self.probe_confirmations,
                "probe_stops": self.probe_stops,
                "confirmation_rate": f"{(self.probe_confirmations / self.probe_entries * 100):.1f}%" if self.probe_entries > 0 else "0%",
                "total_pnl": f"${self.total_pnl:.2f}",
                "hourly_probes": self._count_hourly_probes(),
                "active_probes": len(self.active_probes)
            },
            "active_probes": [p.to_dict() for p in self.active_probes.values()],
            "cooldowns": {
                sym: expires.isoformat()
                for sym, expires in self.cooldowns.items()
            },
            "session_probes_per_symbol": dict(self.probes_per_symbol)
        }

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get probe history"""
        return [p.to_dict() for p in self.probe_history[-limit:]]

    def update_config(
        self,
        enabled: bool = None,
        size_multiplier: float = None,
        stop_loss_percent: float = None,
        min_micro_confidence: float = None,
        max_probes_per_hour: int = None,
        cooldown_minutes: int = None
    ):
        """Update probe configuration"""
        if enabled is not None:
            self.config.enabled = enabled
        if size_multiplier is not None:
            self.config.size_multiplier = max(0.1, min(0.5, size_multiplier))
        if stop_loss_percent is not None:
            self.config.stop_loss_percent = max(0.5, min(5.0, stop_loss_percent))
        if min_micro_confidence is not None:
            self.config.min_micro_confidence = max(0.4, min(0.9, min_micro_confidence))
        if max_probes_per_hour is not None:
            self.config.max_probes_per_hour = max(1, min(10, max_probes_per_hour))
        if cooldown_minutes is not None:
            self.config.cooldown_minutes = max(5, min(60, cooldown_minutes))

        logger.info(f"Probe config updated: {self.config}")


# Singleton instance
_probe_manager: Optional[ProbeEntryManager] = None


def get_probe_manager() -> ProbeEntryManager:
    """Get or create the probe entry manager singleton"""
    global _probe_manager
    if _probe_manager is None:
        _probe_manager = ProbeEntryManager()
    return _probe_manager


def check_probe_eligibility(
    symbol: str,
    ats_state: str,
    micro_confidence: float,
    macro_regime: str,
    current_price: float,
    **kwargs
) -> Tuple[bool, str, Optional[ProbeTriggerType]]:
    """Convenience function to check probe eligibility"""
    return get_probe_manager().check_probe_eligibility(
        symbol, ats_state, micro_confidence, macro_regime, current_price, **kwargs
    )
