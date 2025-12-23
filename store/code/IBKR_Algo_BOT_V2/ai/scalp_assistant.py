"""
HFT Scalp Assistant - AI Exit Manager
=====================================
Human enters trades via ThinkOrSwim.
AI monitors and auto-exits based on momentum reversal rules.

The human edge: Entry identification (intuition, pattern recognition)
The AI edge: Exit discipline (no emotion, no FOMO)

Exit Triggers:
1. Stop Loss - Protect capital (default -3%)
2. Trailing Stop - Lock in gains (default 1.5% from high)
3. Profit Target - Start trailing after target hit (default +3%)
4. Momentum Reversal - 3 red candles or velocity death
5. Max Hold Time - Force exit after timeout (default 3 min)

Version: 1.0.0
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class ExitReason(str, Enum):
    """Reason for exit"""
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    PROFIT_TARGET = "profit_target"
    MOMENTUM_REVERSAL = "momentum_reversal"
    VELOCITY_DEATH = "velocity_death"
    MAX_HOLD_TIME = "max_hold_time"
    MANUAL = "manual"
    SPREAD_TOO_WIDE = "spread_too_wide"


@dataclass
class MonitoredPosition:
    """Position being monitored by AI"""
    symbol: str
    entry_price: float
    quantity: int
    entry_time: datetime
    ai_takeover: bool = False

    # Tracking
    high_since_entry: float = 0.0
    low_since_entry: float = 999999.0
    current_price: float = 0.0
    last_update: datetime = None

    # Price history for momentum detection
    price_history: List[float] = field(default_factory=list)
    candle_colors: List[str] = field(default_factory=list)  # 'green', 'red'

    # Status
    trailing_active: bool = False
    trailing_high: float = 0.0
    exit_reason: Optional[ExitReason] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0

    def __post_init__(self):
        if self.high_since_entry == 0.0:
            self.high_since_entry = self.entry_price
        if self.current_price == 0.0:
            self.current_price = self.entry_price
        if self.last_update is None:
            self.last_update = datetime.now()

    def update_price(self, price: float):
        """Update current price and tracking"""
        prev_price = self.current_price
        self.current_price = price
        self.last_update = datetime.now()

        # Track highs/lows
        if price > self.high_since_entry:
            self.high_since_entry = price
        if price < self.low_since_entry:
            self.low_since_entry = price

        # Update trailing high if trailing is active
        if self.trailing_active and price > self.trailing_high:
            self.trailing_high = price

        # Add to price history (keep last 20 ticks)
        self.price_history.append(price)
        if len(self.price_history) > 20:
            self.price_history.pop(0)

        # Track candle colors (simplified - based on tick direction)
        if len(self.price_history) >= 2:
            color = 'green' if price > prev_price else 'red' if price < prev_price else self.candle_colors[-1] if self.candle_colors else 'green'
            self.candle_colors.append(color)
            if len(self.candle_colors) > 10:
                self.candle_colors.pop(0)

        # Calculate P/L
        self.pnl = (price - self.entry_price) * self.quantity
        self.pnl_pct = ((price - self.entry_price) / self.entry_price) * 100

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON"""
        return {
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'ai_takeover': self.ai_takeover,
            'high_since_entry': self.high_since_entry,
            'low_since_entry': self.low_since_entry,
            'current_price': self.current_price,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'trailing_active': self.trailing_active,
            'trailing_high': self.trailing_high,
            'exit_reason': self.exit_reason.value if self.exit_reason else None,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'hold_seconds': (datetime.now() - self.entry_time).total_seconds() if self.entry_time else 0,
            'red_candles': self.candle_colors[-5:].count('red') if self.candle_colors else 0
        }


@dataclass
class ScalpConfig:
    """Scalp Assistant configuration"""
    # Risk Management
    stop_loss_pct: float = 3.0          # Exit if down this %
    profit_target_pct: float = 3.0      # Start trailing after this %
    trailing_stop_pct: float = 1.5      # Trail this % from high
    max_hold_seconds: int = 180         # 3 minutes max hold

    # Momentum Reversal Detection
    reversal_red_candles: int = 3       # Exit after N red candles
    velocity_death_pct: float = 0.5     # Exit if velocity drops below this
    min_gain_for_reversal_exit: float = 0.5  # Only reversal exit if up this %

    # Spread Protection
    max_spread_pct: float = 1.0         # Don't exit into wide spread

    # Behavior
    enabled: bool = True
    paper_mode: bool = True             # Simulate exits, don't actually sell
    check_interval_ms: int = 500        # How often to check positions

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ScalpConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ScalpAssistant:
    """
    HFT Scalp Assistant - Monitors positions and auto-exits

    Usage:
        assistant = ScalpAssistant()
        assistant.enable_ai_takeover("AAPL")  # User enables AI control
        assistant.start()  # Start monitoring
        # ... AI auto-exits when triggers hit ...
        assistant.disable_ai_takeover("AAPL")  # User takes back control
    """

    def __init__(self):
        self.config = ScalpConfig()
        self.positions: Dict[str, MonitoredPosition] = {}
        self.exit_history: List[Dict] = []
        self.running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Stats
        self.stats = {
            'total_exits': 0,
            'stop_loss_exits': 0,
            'trailing_stop_exits': 0,
            'reversal_exits': 0,
            'timeout_exits': 0,
            'manual_exits': 0,
            'total_pnl': 0.0,
            'winning_exits': 0,
            'losing_exits': 0
        }

        # Broker and market data (lazy init)
        self._broker = None
        self._market_data = None

        # Load config
        self._load_config()

        logger.info("ScalpAssistant initialized")

    @property
    def broker(self):
        """Lazy load broker"""
        if self._broker is None:
            try:
                from unified_broker import get_unified_broker
                self._broker = get_unified_broker()
            except Exception as e:
                logger.error(f"Failed to load broker: {e}")
        return self._broker

    @property
    def market_data(self):
        """Lazy load market data"""
        if self._market_data is None:
            try:
                from schwab_market_data import SchwabMarketData, is_schwab_available
                if is_schwab_available():
                    self._market_data = SchwabMarketData()
            except Exception as e:
                logger.warning(f"Failed to load Schwab market data: {e}")
        return self._market_data

    def _config_path(self) -> Path:
        """Path to config file"""
        return Path(__file__).parent / "scalp_assistant_config.json"

    def _history_path(self) -> Path:
        """Path to exit history file"""
        return Path(__file__).parent / "scalp_assistant_history.json"

    def _load_config(self):
        """Load config from file"""
        try:
            if self._config_path().exists():
                with open(self._config_path(), 'r') as f:
                    data = json.load(f)
                    self.config = ScalpConfig.from_dict(data.get('config', {}))
                    self.stats = data.get('stats', self.stats)
                logger.info("ScalpAssistant config loaded")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    def _save_config(self):
        """Save config to file"""
        try:
            with open(self._config_path(), 'w') as f:
                json.dump({
                    'config': self.config.to_dict(),
                    'stats': self.stats,
                    'saved_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def _save_exit(self, position: MonitoredPosition):
        """Save exit to history"""
        try:
            history = []
            if self._history_path().exists():
                with open(self._history_path(), 'r') as f:
                    history = json.load(f)

            history.append(position.to_dict())

            # Keep last 500 exits
            history = history[-500:]

            with open(self._history_path(), 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save exit history: {e}")

    def sync_positions_from_broker(self):
        """Sync positions from ALL Schwab accounts"""
        if not self.broker:
            return

        try:
            # Try to get positions from all accounts
            if hasattr(self.broker, '_schwab') and self.broker._schwab:
                broker_positions = self.broker._schwab.get_all_positions()
            else:
                broker_positions = self.broker.get_positions()

            with self._lock:
                # Track which symbols are still in broker
                broker_symbols = set()

                for pos in broker_positions:
                    symbol = pos.get('symbol', '').upper()
                    if not symbol:
                        continue

                    broker_symbols.add(symbol)
                    quantity = int(float(pos.get('qty', pos.get('quantity', 0))))
                    avg_cost = float(pos.get('avg_price', pos.get('avg_cost', pos.get('average_price', 0))))

                    if quantity > 0:  # Only long positions
                        if symbol not in self.positions:
                            # New position detected
                            self.positions[symbol] = MonitoredPosition(
                                symbol=symbol,
                                entry_price=avg_cost,
                                quantity=quantity,
                                entry_time=datetime.now(),
                                ai_takeover=False  # User must manually enable
                            )
                            logger.info(f"New position detected: {symbol} x{quantity} @ ${avg_cost:.2f}")
                        else:
                            # Update existing position quantity if changed
                            if self.positions[symbol].quantity != quantity:
                                self.positions[symbol].quantity = quantity
                            # Fix entry_price if it was 0.0
                            if self.positions[symbol].entry_price == 0.0 and avg_cost > 0:
                                self.positions[symbol].entry_price = avg_cost
                                logger.info(f"Updated {symbol} entry_price to ${avg_cost:.4f}")

                # Remove positions no longer in broker
                for symbol in list(self.positions.keys()):
                    if symbol not in broker_symbols:
                        if self.positions[symbol].exit_reason is None:
                            # Position was closed externally
                            logger.info(f"Position {symbol} closed externally")
                        del self.positions[symbol]

        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")

    def enable_ai_takeover(self, symbol: str) -> bool:
        """Enable AI takeover for a position"""
        symbol = symbol.upper()
        with self._lock:
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.ai_takeover = True
                pos.entry_time = datetime.now()  # Reset timer
                # Clear any previous exit state (for re-enabling after paper exit)
                pos.exit_reason = None
                pos.exit_price = None
                pos.exit_time = None
                pos.trailing_active = False
                pos.trailing_high = 0.0
                pos.high_since_entry = pos.current_price if pos.current_price > 0 else pos.entry_price
                pos.low_since_entry = pos.current_price if pos.current_price > 0 else 999999.0
                pos.price_history = []
                pos.candle_colors = []
                logger.info(f"AI takeover ENABLED for {symbol} (state reset)")
                return True
            else:
                logger.warning(f"Cannot enable AI takeover - no position for {symbol}")
                return False

    def disable_ai_takeover(self, symbol: str) -> bool:
        """Disable AI takeover - user takes back control"""
        symbol = symbol.upper()
        with self._lock:
            if symbol in self.positions:
                self.positions[symbol].ai_takeover = False
                logger.info(f"AI takeover DISABLED for {symbol} - manual control")
                return True
            return False

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get current quote for symbol"""
        if not self.market_data:
            return None

        try:
            return self.market_data.get_quote(symbol)
        except Exception as e:
            logger.debug(f"Quote error for {symbol}: {e}")
            return None

    def check_exit_triggers(self, position: MonitoredPosition) -> Optional[ExitReason]:
        """Check if any exit trigger is hit"""
        if not position.ai_takeover:
            return None

        # Get current quote
        quote = self.get_quote(position.symbol)
        if quote:
            bid = float(quote.get('bid', quote.get('bidPrice', 0)))
            ask = float(quote.get('ask', quote.get('askPrice', 0)))
            last = float(quote.get('last', quote.get('lastPrice', position.current_price)))

            # Use last price or mid
            if last > 0:
                position.update_price(last)
            elif bid > 0 and ask > 0:
                position.update_price((bid + ask) / 2)

            # Check spread - don't exit into wide spread
            if bid > 0 and ask > 0:
                spread_pct = ((ask - bid) / bid) * 100
                if spread_pct > self.config.max_spread_pct:
                    logger.debug(f"{position.symbol} spread too wide: {spread_pct:.2f}%")
                    # Don't exit, but flag it
                    # return ExitReason.SPREAD_TOO_WIDE

        current_return_pct = position.pnl_pct
        hold_seconds = (datetime.now() - position.entry_time).total_seconds()

        # 1. STOP LOSS - Always protect capital
        if current_return_pct <= -self.config.stop_loss_pct:
            logger.info(f"{position.symbol} HIT STOP LOSS: {current_return_pct:.2f}%")
            return ExitReason.STOP_LOSS

        # 2. PROFIT TARGET - Activate trailing
        if current_return_pct >= self.config.profit_target_pct and not position.trailing_active:
            position.trailing_active = True
            position.trailing_high = position.current_price
            logger.info(f"{position.symbol} HIT PROFIT TARGET +{current_return_pct:.2f}% - TRAILING ACTIVATED")

        # 3. TRAILING STOP - Lock in gains
        if position.trailing_active:
            drop_from_high = ((position.trailing_high - position.current_price) / position.trailing_high) * 100
            if drop_from_high >= self.config.trailing_stop_pct:
                logger.info(f"{position.symbol} TRAILING STOP: dropped {drop_from_high:.2f}% from ${position.trailing_high:.2f}")
                return ExitReason.TRAILING_STOP

        # 4. MOMENTUM REVERSAL - Red candles
        if len(position.candle_colors) >= self.config.reversal_red_candles:
            recent = position.candle_colors[-self.config.reversal_red_candles:]
            if all(c == 'red' for c in recent):
                # Only exit on reversal if we're up (don't compound losses)
                if current_return_pct >= self.config.min_gain_for_reversal_exit:
                    logger.info(f"{position.symbol} MOMENTUM REVERSAL: {self.config.reversal_red_candles} red candles, locking +{current_return_pct:.2f}%")
                    return ExitReason.MOMENTUM_REVERSAL

        # 5. VELOCITY DEATH - Price stalling
        if len(position.price_history) >= 5:
            recent_prices = position.price_history[-5:]
            price_range = max(recent_prices) - min(recent_prices)
            range_pct = (price_range / position.entry_price) * 100

            if range_pct < self.config.velocity_death_pct and current_return_pct < 0:
                # Stalling while down - cut it
                logger.info(f"{position.symbol} VELOCITY DEATH: price stalled, down {current_return_pct:.2f}%")
                return ExitReason.VELOCITY_DEATH

        # 6. MAX HOLD TIME - Only exit if flat or down
        if hold_seconds >= self.config.max_hold_seconds:
            if current_return_pct <= 0 or not position.trailing_active:
                logger.info(f"{position.symbol} MAX HOLD TIME: {hold_seconds:.0f}s, at {current_return_pct:.2f}%")
                return ExitReason.MAX_HOLD_TIME
            # If up and trailing active, let it run

        return None

    def execute_exit(self, position: MonitoredPosition, reason: ExitReason) -> bool:
        """Execute exit for position"""
        position.exit_reason = reason
        position.exit_price = position.current_price
        position.exit_time = datetime.now()

        logger.info(f"EXECUTING EXIT: {position.symbol} x{position.quantity} @ ${position.current_price:.2f} | Reason: {reason.value} | P/L: ${position.pnl:.2f} ({position.pnl_pct:+.2f}%)")

        # Update stats
        self.stats['total_exits'] += 1
        self.stats['total_pnl'] += position.pnl

        if position.pnl > 0:
            self.stats['winning_exits'] += 1
        else:
            self.stats['losing_exits'] += 1

        if reason == ExitReason.STOP_LOSS:
            self.stats['stop_loss_exits'] += 1
        elif reason == ExitReason.TRAILING_STOP:
            self.stats['trailing_stop_exits'] += 1
        elif reason in (ExitReason.MOMENTUM_REVERSAL, ExitReason.VELOCITY_DEATH):
            self.stats['reversal_exits'] += 1
        elif reason == ExitReason.MAX_HOLD_TIME:
            self.stats['timeout_exits'] += 1

        # Save to history
        self._save_exit(position)
        self._save_config()

        # Execute actual sell order (if not paper mode)
        if not self.config.paper_mode and self.broker:
            try:
                result = self.broker.close_position(position.symbol)
                if result.success:
                    logger.info(f"EXIT ORDER PLACED: {result.order_id}")
                    return True
                else:
                    logger.error(f"EXIT ORDER FAILED: {result.error}")
                    return False
            except Exception as e:
                logger.error(f"Exit execution error: {e}")
                return False
        else:
            logger.info(f"PAPER MODE - Exit simulated for {position.symbol}")
            return True

    def _monitor_loop(self):
        """Main monitoring loop"""
        logger.info("ScalpAssistant monitor loop started")

        while self.running:
            try:
                # Sync positions from broker
                self.sync_positions_from_broker()

                # Check each position with AI takeover enabled
                with self._lock:
                    for symbol, position in list(self.positions.items()):
                        if position.ai_takeover and position.exit_reason is None:
                            exit_reason = self.check_exit_triggers(position)
                            if exit_reason:
                                self.execute_exit(position, exit_reason)
                                # Remove from active monitoring
                                self.exit_history.append(position.to_dict())

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

            time.sleep(self.config.check_interval_ms / 1000.0)

        logger.info("ScalpAssistant monitor loop stopped")

    def start(self):
        """Start the scalp assistant"""
        if self.running:
            logger.warning("ScalpAssistant already running")
            return

        if not self.config.enabled:
            logger.warning("ScalpAssistant is disabled in config")
            return

        self.running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("ScalpAssistant STARTED")

    def stop(self):
        """Stop the scalp assistant"""
        self.running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        self._save_config()
        logger.info("ScalpAssistant STOPPED")

    def get_status(self) -> Dict:
        """Get current status"""
        with self._lock:
            monitored = [p.to_dict() for p in self.positions.values() if p.ai_takeover]
            all_positions = [p.to_dict() for p in self.positions.values()]

        return {
            'running': self.running,
            'enabled': self.config.enabled,
            'paper_mode': self.config.paper_mode,
            'config': self.config.to_dict(),
            'stats': self.stats,
            'total_positions': len(all_positions),
            'ai_monitored': len(monitored),
            'positions': all_positions,
            'monitored_positions': monitored,
            'recent_exits': self.exit_history[-10:],
            'win_rate': (self.stats['winning_exits'] / self.stats['total_exits'] * 100) if self.stats['total_exits'] > 0 else 0
        }

    def update_config(self, **kwargs) -> Dict:
        """Update configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self._save_config()
        return self.config.to_dict()


# Singleton instance
_scalp_assistant: Optional[ScalpAssistant] = None


def get_scalp_assistant() -> ScalpAssistant:
    """Get singleton scalp assistant instance"""
    global _scalp_assistant
    if _scalp_assistant is None:
        _scalp_assistant = ScalpAssistant()
    return _scalp_assistant


if __name__ == "__main__":
    # Test the scalp assistant
    logging.basicConfig(level=logging.INFO)

    assistant = get_scalp_assistant()

    # Simulate a position
    assistant.positions['TEST'] = MonitoredPosition(
        symbol='TEST',
        entry_price=10.00,
        quantity=100,
        entry_time=datetime.now(),
        ai_takeover=True
    )

    # Simulate price movement
    pos = assistant.positions['TEST']

    print("Simulating price movement...")
    prices = [10.0, 10.1, 10.2, 10.3, 10.35, 10.4, 10.35, 10.30, 10.25]

    for price in prices:
        pos.update_price(price)
        print(f"  Price: ${price:.2f} | P/L: ${pos.pnl:.2f} ({pos.pnl_pct:+.2f}%) | Trailing: {pos.trailing_active} | High: ${pos.trailing_high:.2f}")

        exit_reason = assistant.check_exit_triggers(pos)
        if exit_reason:
            print(f"  EXIT TRIGGER: {exit_reason.value}")
            break

    print("\nStatus:", json.dumps(assistant.get_status(), indent=2, default=str))
