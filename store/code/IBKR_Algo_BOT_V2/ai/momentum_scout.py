"""
Momentum Scout Mode (Task P)
============================
Lightweight execution path for early momentum discovery.

Key Principles:
- Ignores macro regime vetoes
- Ignores deep structure confirmation
- Operates with very small size (15-25% of normal)
- Hands successful probes to ATS/Scalper for continuation

Scout Trigger Conditions (ALL required):
1. Market Phase in {OPEN_IGNITION, STRUCTURED_MOMENTUM}
2. Symbol liquidity > minimum threshold
3. Volume acceleration detected
4. Price breaks a micro-level (PMH, VWAP reclaim, range expansion)
5. Symbol not exhausted or invalidated

Chronos confidence is NOT required to initiate.
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)

# Config file
SCOUT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "scout_config.json")


class ScoutState(Enum):
    """Scout probe states"""
    PENDING = "PENDING"           # Waiting for trigger
    ACTIVE = "ACTIVE"             # Scout position active
    CONFIRMED = "CONFIRMED"       # Scout succeeded, handed off
    STOPPED = "STOPPED"           # Scout hit stop
    FAILED = "FAILED"             # Scout failed validation
    COOLDOWN = "COOLDOWN"         # Symbol in cooldown


class ScoutTrigger(Enum):
    """What triggered the scout entry"""
    PMH_BREAK = "PMH_BREAK"                 # Premarket high break
    VWAP_RECLAIM = "VWAP_RECLAIM"           # Price reclaims VWAP
    RANGE_1M_BREAK = "RANGE_1M_BREAK"       # 1-minute range high break
    RANGE_3M_BREAK = "RANGE_3M_BREAK"       # 3-minute range high break
    HOD_BREAK = "HOD_BREAK"                 # High of day break
    VOLUME_SPIKE = "VOLUME_SPIKE"           # Volume acceleration


@dataclass
class ScoutConfig:
    """Scout mode configuration"""
    enabled: bool = True

    # Size controls
    size_multiplier: float = 0.20           # 20% of normal size (15-25% range)
    stop_loss_percent: float = 1.0          # Tight 1% stop

    # Rate limits
    max_per_symbol_per_session: int = 1     # Max 1 scout per symbol
    max_per_hour: int = 5                   # Max N scouts per hour
    cooldown_minutes: int = 30              # Cooldown after failed scout

    # Trigger thresholds
    min_volume_acceleration: float = 1.5    # 1.5x short-term avg
    min_liquidity_shares: int = 50000       # Min recent volume
    min_price: float = 1.0                  # Min stock price
    max_price: float = 20.0                 # Max stock price
    max_spread_percent: float = 1.0         # Max bid-ask spread

    # Confirmation thresholds
    confirm_hold_bars: int = 3              # Bars to hold for confirmation
    confirm_gain_percent: float = 0.5       # Gain needed for confirmation

    # Allowed phases
    allowed_phases: List[str] = field(default_factory=lambda: [
        "OPEN_IGNITION", "STRUCTURED_MOMENTUM", "PRE_MARKET"
    ])

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ScoutEntry:
    """A scout probe entry"""
    symbol: str
    trigger: ScoutTrigger
    entry_price: float
    entry_time: datetime
    size_shares: int
    stop_price: float

    # Tracking
    state: ScoutState = ScoutState.ACTIVE
    high_price: float = 0.0
    low_price: float = 0.0
    bars_held: int = 0
    current_price: float = 0.0

    # Outcomes
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    handed_off: bool = False

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "trigger": self.trigger.value,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "size_shares": self.size_shares,
            "stop_price": self.stop_price,
            "state": self.state.value,
            "high_price": self.high_price,
            "low_price": self.low_price,
            "bars_held": self.bars_held,
            "current_price": self.current_price,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_reason": self.exit_reason,
            "pnl": self.pnl,
            "handed_off": self.handed_off
        }


class MomentumScout:
    """
    Momentum Scout Mode - Early discovery without Chronos confirmation.

    Purpose: Enable the bot to discover real momentum early while preserving
    hard risk limits and gating authority.
    """

    def __init__(self):
        self.config = ScoutConfig()
        self.active_scouts: Dict[str, ScoutEntry] = {}
        self.scout_history: List[ScoutEntry] = []
        self.failed_symbols: Set[str] = set()           # Symbols that failed today
        self.cooldowns: Dict[str, datetime] = {}        # Symbol -> cooldown end time
        self.hourly_counts: List[datetime] = []         # Timestamps of scout attempts
        self.session_counts: Dict[str, int] = {}        # Symbol -> count today

        # Statistics
        self.stats = {
            "scout_attempts": 0,
            "scout_confirmed": 0,
            "scout_stopped": 0,
            "scout_failed": 0,
            "scout_to_trade": 0,
            "total_pnl": 0.0
        }

        self._load_config()

    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(SCOUT_CONFIG_FILE):
                with open(SCOUT_CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                logger.info(f"Loaded scout config: enabled={self.config.enabled}")
        except Exception as e:
            logger.warning(f"Error loading scout config: {e}")

    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(SCOUT_CONFIG_FILE, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving scout config: {e}")

    def is_enabled(self) -> bool:
        """Check if scout mode is enabled"""
        return self.config.enabled

    def enable(self):
        """Enable scout mode"""
        self.config.enabled = True
        self._save_config()
        logger.info("SCOUT_MODE_ENABLED")

    def disable(self):
        """Disable scout mode"""
        self.config.enabled = False
        self._save_config()
        logger.info("SCOUT_MODE_DISABLED")

    def _is_phase_allowed(self) -> Tuple[bool, str]:
        """Check if current market phase allows scouting"""
        try:
            from ai.market_phases import get_phase_manager

            manager = get_phase_manager()
            current_phase = manager.current_phase

            if current_phase is None:
                return False, "No phase set"

            phase_name = current_phase.value
            if phase_name in self.config.allowed_phases:
                return True, phase_name
            else:
                return False, f"Phase {phase_name} not in allowed list"

        except Exception as e:
            logger.debug(f"Error checking phase: {e}")
            return False, str(e)

    def _check_rate_limits(self, symbol: str) -> Tuple[bool, str]:
        """Check hourly and session rate limits"""
        now = datetime.now()

        # Check session limit for symbol
        session_count = self.session_counts.get(symbol, 0)
        if session_count >= self.config.max_per_symbol_per_session:
            return False, f"Symbol {symbol} hit session limit ({session_count})"

        # Check cooldown
        if symbol in self.cooldowns:
            cooldown_end = self.cooldowns[symbol]
            if now < cooldown_end:
                remaining = (cooldown_end - now).total_seconds()
                return False, f"Symbol {symbol} in cooldown ({remaining:.0f}s remaining)"

        # Check hourly limit
        hour_ago = now - timedelta(hours=1)
        self.hourly_counts = [t for t in self.hourly_counts if t > hour_ago]
        if len(self.hourly_counts) >= self.config.max_per_hour:
            return False, f"Hourly limit reached ({len(self.hourly_counts)}/{self.config.max_per_hour})"

        # Check if symbol already has active scout
        if symbol in self.active_scouts:
            return False, f"Symbol {symbol} already has active scout"

        # Check if symbol failed today
        if symbol in self.failed_symbols:
            return False, f"Symbol {symbol} failed earlier today"

        return True, "OK"

    def _check_liquidity(self, quote: Dict) -> Tuple[bool, str]:
        """Check if symbol has sufficient liquidity"""
        try:
            price = quote.get('lastPrice', 0) or quote.get('last', 0)
            volume = quote.get('totalVolume', 0) or quote.get('volume', 0)
            bid = quote.get('bidPrice', 0) or quote.get('bid', 0)
            ask = quote.get('askPrice', 0) or quote.get('ask', 0)

            # Price range check
            if price < self.config.min_price:
                return False, f"Price ${price:.2f} below min ${self.config.min_price}"
            if price > self.config.max_price:
                return False, f"Price ${price:.2f} above max ${self.config.max_price}"

            # Volume check
            if volume < self.config.min_liquidity_shares:
                return False, f"Volume {volume:,} below min {self.config.min_liquidity_shares:,}"

            # Spread check
            if bid > 0 and ask > 0:
                spread_pct = ((ask - bid) / bid) * 100
                if spread_pct > self.config.max_spread_percent:
                    return False, f"Spread {spread_pct:.2f}% above max {self.config.max_spread_percent}%"

            return True, "OK"

        except Exception as e:
            return False, f"Liquidity check error: {e}"

    def _detect_trigger(self, symbol: str, quote: Dict, market_data: Dict = None) -> Tuple[Optional[ScoutTrigger], str]:
        """
        Detect if any scout trigger condition is met.

        Trigger conditions:
        - PMH break
        - VWAP reclaim
        - 1-3 minute range break
        - HOD break
        - Volume spike
        """
        try:
            price = quote.get('lastPrice', 0) or quote.get('last', 0)

            if market_data is None:
                market_data = {}

            # Check PMH break
            pmh = market_data.get('premarket_high', 0)
            if pmh > 0 and price > pmh:
                return ScoutTrigger.PMH_BREAK, f"Price ${price:.2f} broke PMH ${pmh:.2f}"

            # Check VWAP reclaim
            vwap = market_data.get('vwap', 0)
            prev_price = market_data.get('prev_price', price)
            if vwap > 0 and prev_price < vwap and price > vwap:
                return ScoutTrigger.VWAP_RECLAIM, f"Price reclaimed VWAP ${vwap:.2f}"

            # Check HOD break
            hod = market_data.get('high_of_day', 0)
            if hod > 0 and price > hod:
                return ScoutTrigger.HOD_BREAK, f"Price ${price:.2f} broke HOD ${hod:.2f}"

            # Check volume spike
            avg_volume = market_data.get('avg_volume_5m', 0)
            current_volume = market_data.get('current_volume_5m', 0)
            if avg_volume > 0 and current_volume > 0:
                vol_ratio = current_volume / avg_volume
                if vol_ratio >= self.config.min_volume_acceleration:
                    return ScoutTrigger.VOLUME_SPIKE, f"Volume spike {vol_ratio:.1f}x"

            # Check range breaks
            range_high_1m = market_data.get('range_high_1m', 0)
            if range_high_1m > 0 and price > range_high_1m:
                return ScoutTrigger.RANGE_1M_BREAK, f"1min range break ${range_high_1m:.2f}"

            range_high_3m = market_data.get('range_high_3m', 0)
            if range_high_3m > 0 and price > range_high_3m:
                return ScoutTrigger.RANGE_3M_BREAK, f"3min range break ${range_high_3m:.2f}"

            return None, "No trigger detected"

        except Exception as e:
            return None, f"Trigger detection error: {e}"

    def check_scout_entry(
        self,
        symbol: str,
        quote: Dict,
        market_data: Dict = None,
        base_position_size: int = 100
    ) -> Tuple[bool, Optional[ScoutEntry], str]:
        """
        Check if a scout entry should be initiated.

        Args:
            symbol: Stock symbol
            quote: Current quote data
            market_data: Additional market data (PMH, VWAP, HOD, etc.)
            base_position_size: Normal position size (scout will use % of this)

        Returns:
            (should_enter, scout_entry, reason)
        """
        if not self.config.enabled:
            return False, None, "Scout mode disabled"

        # Check phase
        phase_ok, phase_reason = self._is_phase_allowed()
        if not phase_ok:
            return False, None, f"Phase blocked: {phase_reason}"

        # Check rate limits
        rate_ok, rate_reason = self._check_rate_limits(symbol)
        if not rate_ok:
            return False, None, f"Rate limit: {rate_reason}"

        # Check liquidity
        liq_ok, liq_reason = self._check_liquidity(quote)
        if not liq_ok:
            return False, None, f"Liquidity: {liq_reason}"

        # Check for trigger
        trigger, trigger_reason = self._detect_trigger(symbol, quote, market_data)
        if trigger is None:
            return False, None, f"No trigger: {trigger_reason}"

        # Build scout entry
        price = quote.get('lastPrice', 0) or quote.get('last', 0)
        scout_size = int(base_position_size * self.config.size_multiplier)
        stop_price = price * (1 - self.config.stop_loss_percent / 100)

        scout = ScoutEntry(
            symbol=symbol,
            trigger=trigger,
            entry_price=price,
            entry_time=datetime.now(),
            size_shares=scout_size,
            stop_price=stop_price,
            high_price=price,
            low_price=price,
            current_price=price
        )

        # Record attempt
        self.stats["scout_attempts"] += 1
        self.hourly_counts.append(datetime.now())
        self.session_counts[symbol] = self.session_counts.get(symbol, 0) + 1

        logger.info(f"SCOUT_ENTRY_ATTEMPT: {symbol} @ ${price:.2f} trigger={trigger.value} size={scout_size}")

        return True, scout, trigger_reason

    def register_scout(self, scout: ScoutEntry):
        """Register an active scout"""
        self.active_scouts[scout.symbol] = scout
        logger.info(f"SCOUT_REGISTERED: {scout.symbol} @ ${scout.entry_price:.2f}")

    def update_scout(self, symbol: str, current_price: float) -> Tuple[ScoutState, Optional[str]]:
        """
        Update a scout's state based on current price.

        Returns:
            (new_state, action) where action is None, "STOP", or "CONFIRM"
        """
        if symbol not in self.active_scouts:
            return ScoutState.FAILED, None

        scout = self.active_scouts[symbol]
        scout.current_price = current_price
        scout.bars_held += 1

        # Update high/low
        if current_price > scout.high_price:
            scout.high_price = current_price
        if current_price < scout.low_price:
            scout.low_price = current_price

        # Check stop loss
        if current_price <= scout.stop_price:
            return self._stop_scout(symbol, current_price, "STOP_LOSS")

        # Check for confirmation
        gain_pct = ((current_price - scout.entry_price) / scout.entry_price) * 100

        if scout.bars_held >= self.config.confirm_hold_bars and gain_pct >= self.config.confirm_gain_percent:
            return self._confirm_scout(symbol, current_price)

        return ScoutState.ACTIVE, None

    def _stop_scout(self, symbol: str, exit_price: float, reason: str) -> Tuple[ScoutState, str]:
        """Stop a scout (hit stop or failed)"""
        if symbol not in self.active_scouts:
            return ScoutState.FAILED, "NOT_FOUND"

        scout = self.active_scouts[symbol]
        scout.state = ScoutState.STOPPED
        scout.exit_price = exit_price
        scout.exit_time = datetime.now()
        scout.exit_reason = reason
        scout.pnl = (exit_price - scout.entry_price) * scout.size_shares

        # Move to history
        self.scout_history.append(scout)
        del self.active_scouts[symbol]

        # Set cooldown
        self.cooldowns[symbol] = datetime.now() + timedelta(minutes=self.config.cooldown_minutes)
        self.failed_symbols.add(symbol)

        # Update stats
        self.stats["scout_stopped"] += 1
        self.stats["total_pnl"] += scout.pnl

        logger.info(f"SCOUT_ENTRY_STOPPED: {symbol} @ ${exit_price:.2f} reason={reason} PnL=${scout.pnl:.2f}")

        return ScoutState.STOPPED, "STOP"

    def _confirm_scout(self, symbol: str, current_price: float) -> Tuple[ScoutState, str]:
        """Confirm a scout (ready for handoff)"""
        if symbol not in self.active_scouts:
            return ScoutState.FAILED, "NOT_FOUND"

        scout = self.active_scouts[symbol]
        scout.state = ScoutState.CONFIRMED
        scout.current_price = current_price

        # Update stats
        self.stats["scout_confirmed"] += 1

        logger.info(
            f"SCOUT_ENTRY_CONFIRMED: {symbol} @ ${current_price:.2f} "
            f"gain={((current_price - scout.entry_price) / scout.entry_price) * 100:.2f}% "
            f"bars={scout.bars_held}"
        )

        return ScoutState.CONFIRMED, "CONFIRM"

    def get_confirmed_scouts(self) -> List[ScoutEntry]:
        """Get scouts that are confirmed and ready for handoff"""
        return [s for s in self.active_scouts.values() if s.state == ScoutState.CONFIRMED]

    def mark_handed_off(self, symbol: str):
        """Mark a scout as handed off to ATS/Scalper"""
        if symbol in self.active_scouts:
            scout = self.active_scouts[symbol]
            scout.handed_off = True
            scout.state = ScoutState.CONFIRMED

            # Move to history
            self.scout_history.append(scout)
            del self.active_scouts[symbol]

            # Update stats
            self.stats["scout_to_trade"] += 1

            logger.info(f"SCOUT_HANDED_OFF: {symbol} to ATS/Scalper")

    def close_scout(self, symbol: str, exit_price: float, reason: str = "MANUAL"):
        """Manually close a scout"""
        if symbol in self.active_scouts:
            scout = self.active_scouts[symbol]
            scout.state = ScoutState.STOPPED
            scout.exit_price = exit_price
            scout.exit_time = datetime.now()
            scout.exit_reason = reason
            scout.pnl = (exit_price - scout.entry_price) * scout.size_shares

            self.scout_history.append(scout)
            del self.active_scouts[symbol]

            self.stats["total_pnl"] += scout.pnl

            logger.info(f"SCOUT_CLOSED: {symbol} @ ${exit_price:.2f} reason={reason}")

    def reset_session(self):
        """Reset session counters (call at start of day)"""
        self.session_counts.clear()
        self.failed_symbols.clear()
        self.cooldowns.clear()
        self.hourly_counts.clear()
        logger.info("SCOUT_SESSION_RESET")

    def get_status(self) -> Dict:
        """Get scout mode status"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        recent_hourly = len([t for t in self.hourly_counts if t > hour_ago])

        return {
            "enabled": self.config.enabled,
            "active_scouts": len(self.active_scouts),
            "active_symbols": list(self.active_scouts.keys()),
            "hourly_count": recent_hourly,
            "hourly_limit": self.config.max_per_hour,
            "session_counts": self.session_counts,
            "cooldowns": {
                s: cd.isoformat() for s, cd in self.cooldowns.items()
                if cd > now
            },
            "failed_symbols": list(self.failed_symbols),
            "stats": self.stats,
            "config": self.config.to_dict(),
            "history_count": len(self.scout_history),
            "timestamp": now.isoformat()
        }

    def get_stats(self) -> Dict:
        """Get scout statistics"""
        attempts = self.stats["scout_attempts"]
        confirmed = self.stats["scout_confirmed"]
        to_trade = self.stats["scout_to_trade"]

        return {
            "scout_attempts": attempts,
            "scout_confirmed": confirmed,
            "scout_stopped": self.stats["scout_stopped"],
            "scout_failed": self.stats["scout_failed"],
            "scout_to_trade": to_trade,
            "confirmation_rate": (confirmed / attempts * 100) if attempts > 0 else 0,
            "conversion_rate": (to_trade / confirmed * 100) if confirmed > 0 else 0,
            "total_pnl": self.stats["total_pnl"],
            "avg_pnl": self.stats["total_pnl"] / (confirmed + self.stats["scout_stopped"]) if (confirmed + self.stats["scout_stopped"]) > 0 else 0
        }

    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get scout history"""
        return [s.to_dict() for s in self.scout_history[-limit:]]

    def update_config(self, updates: Dict) -> Dict:
        """Update configuration"""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self._save_config()
        return self.config.to_dict()


# Singleton instance
_scout: Optional[MomentumScout] = None


def get_momentum_scout() -> MomentumScout:
    """Get the singleton momentum scout"""
    global _scout
    if _scout is None:
        _scout = MomentumScout()
    return _scout
