"""
HFT Momentum Scalper
====================
Automated high-frequency scalping bot for momentum plays.

Detects momentum spikes, enters quickly, trails stops,
and exits automatically on reversal signals.

Key features:
- Real-time momentum detection
- Auto-entry on spike signals
- Trailing stop loss
- Target profit exits
- Reversal detection for quick exits
- Risk management per trade and daily
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import time

logger = logging.getLogger(__name__)

# Config file
SCALPER_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "scalper_config.json")
SCALPER_TRADES_FILE = os.path.join(os.path.dirname(__file__), "scalper_trades.json")


class SignalType(Enum):
    MOMENTUM_SPIKE = "momentum_spike"
    VOLUME_SURGE = "volume_surge"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


class TradeStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass
class ScalperConfig:
    """Configuration for the scalper"""
    # Enable/disable
    enabled: bool = False
    paper_mode: bool = True  # Always start in paper mode

    # Account-based risk management
    account_size: float = 658.0  # Total account size
    risk_percent: float = 1.0  # Risk per trade (1% = 100 attempts)
    use_risk_based_sizing: bool = True  # Use risk-based position sizing

    # Position sizing (fallback if not using risk-based)
    max_position_size: float = 200.0  # Max $ per trade
    max_shares: int = 500  # Max shares per trade
    min_shares: int = 1  # Minimum shares to trade

    # Entry criteria
    min_spike_percent: float = 3.0  # Min % move to trigger entry
    min_volume_surge: float = 2.0  # Min relative volume
    max_spread_percent: float = 1.0  # Max bid/ask spread
    min_price: float = 1.0
    max_price: float = 20.0

    # Exit criteria
    profit_target_percent: float = 5.0  # Target profit %
    stop_loss_percent: float = 3.0  # Initial stop loss %
    trailing_stop_percent: float = 2.0  # Trailing stop %
    max_hold_seconds: int = 300  # Max time to hold (5 min)

    # Reversal detection
    reversal_candle_percent: float = 2.0  # Red candle size to trigger exit
    macd_reversal_exit: bool = True

    # Chronos AI filter
    use_chronos_filter: bool = True  # Filter entries with Chronos AI
    chronos_min_prob_up: float = 0.5  # Min probability to enter (0.5 = 50%)

    # Risk management
    max_daily_loss: float = 50.0  # Max daily loss before stopping
    max_daily_trades: int = 20  # Max trades per day
    cooldown_after_loss: int = 60  # Seconds to wait after a loss

    # Symbols
    watchlist: List[str] = field(default_factory=list)
    blacklist: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    def calculate_position_size(self, price: float) -> tuple:
        """
        Calculate position size based on risk management.
        Returns (shares, position_value, risk_amount)
        """
        if self.use_risk_based_sizing:
            # Risk-based: risk_amount = account * risk_percent / 100
            risk_amount = self.account_size * self.risk_percent / 100
            # Max position = risk_amount / stop_loss_percent
            max_position = risk_amount / (self.stop_loss_percent / 100)
            # Calculate whole shares
            shares = int(max_position / price)  # floor to whole shares
        else:
            # Fixed position sizing
            shares = int(self.max_position_size / price)
            risk_amount = self.max_position_size * self.stop_loss_percent / 100

        # Apply limits
        shares = min(shares, self.max_shares)
        shares = max(shares, 0)

        position_value = shares * price
        actual_risk = position_value * self.stop_loss_percent / 100

        return shares, position_value, actual_risk


@dataclass
class ScalpTrade:
    """Record of a scalp trade"""
    trade_id: str
    symbol: str

    # Entry
    entry_time: str
    entry_price: float
    entry_signal: str
    shares: int

    # Exit
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""

    # P/L
    pnl: float = 0.0
    pnl_percent: float = 0.0

    # Stats
    high_price: float = 0.0  # Highest price while holding
    low_price: float = 0.0  # Lowest price while holding
    max_gain_percent: float = 0.0
    max_drawdown_percent: float = 0.0
    hold_time_seconds: int = 0

    # Status
    status: str = "open"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PricePoint:
    """Price data point for momentum tracking"""
    timestamp: datetime
    price: float
    bid: float
    ask: float
    volume: int


class HFTScalper:
    """
    High-frequency momentum scalper.
    Monitors prices, detects opportunities, and executes trades automatically.
    """

    def __init__(self):
        self.config = ScalperConfig()
        self.trades: List[ScalpTrade] = []
        self.open_positions: Dict[str, ScalpTrade] = {}  # symbol -> trade

        # Price history for momentum detection
        self.price_history: Dict[str, List[PricePoint]] = {}  # symbol -> prices
        self.last_prices: Dict[str, float] = {}

        # State
        self.is_running = False
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_loss_time: Optional[datetime] = None

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Callbacks
        self.on_signal: Optional[Callable] = None
        self.on_entry: Optional[Callable] = None
        self.on_exit: Optional[Callable] = None

        self._load_config()
        self._load_trades()

        logger.info("HFTScalper initialized")

    def _load_config(self):
        """Load config from file"""
        try:
            if os.path.exists(SCALPER_CONFIG_FILE):
                with open(SCALPER_CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
        except Exception as e:
            logger.error(f"Error loading scalper config: {e}")

    def _save_config(self):
        """Save config to file"""
        try:
            with open(SCALPER_CONFIG_FILE, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving scalper config: {e}")

    def _load_trades(self):
        """Load trade history"""
        try:
            if os.path.exists(SCALPER_TRADES_FILE):
                with open(SCALPER_TRADES_FILE, 'r') as f:
                    data = json.load(f)
                    self.trades = [ScalpTrade(**t) for t in data.get('trades', [])]

                    # Calculate daily stats
                    today = datetime.now().date().isoformat()
                    today_trades = [t for t in self.trades
                                   if t.entry_time.startswith(today)]
                    self.daily_trades = len(today_trades)
                    self.daily_pnl = sum(t.pnl for t in today_trades if t.status == 'closed')
        except Exception as e:
            logger.error(f"Error loading trades: {e}")

    def _save_trades(self):
        """Save trade history"""
        try:
            with open(SCALPER_TRADES_FILE, 'w') as f:
                json.dump({
                    'trades': [t.to_dict() for t in self.trades[-500:]],  # Keep last 500
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trades: {e}")

    def _check_chronos_signal(self, symbol: str) -> Optional[Dict]:
        """
        Check Chronos AI prediction for a symbol.
        Returns dict with 'signal' and 'prob_up' or None if unavailable.
        """
        try:
            from ai.chronos_predictor import get_chronos_predictor
            chronos = get_chronos_predictor()
            result = chronos.predict(symbol, horizon=1)
            return {
                'signal': result.get('signal', 'NEUTRAL'),
                'prob_up': result.get('probabilities', {}).get('prob_up', 0.5),
                'expected_return': result.get('expected_return_pct', 0)
            }
        except Exception as e:
            logger.warning(f"Chronos check failed for {symbol}: {e}")
            return None

    def update_config(self, **kwargs):
        """Update configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self._save_config()
        logger.info(f"Config updated: {kwargs}")

    def add_to_watchlist(self, symbol: str):
        """Add symbol to watchlist"""
        symbol = symbol.upper()
        if symbol not in self.config.watchlist:
            self.config.watchlist.append(symbol)
            self._save_config()

    def remove_from_watchlist(self, symbol: str):
        """Remove symbol from watchlist"""
        symbol = symbol.upper()
        if symbol in self.config.watchlist:
            self.config.watchlist.remove(symbol)
            self._save_config()

    async def check_entry_signal(self, symbol: str, quote: Dict) -> Optional[Dict]:
        """
        Check if there's an entry signal for a symbol.

        Returns signal dict if entry opportunity detected, None otherwise.
        """
        price = quote.get('price', 0) or quote.get('last', 0)
        bid = quote.get('bid', 0)
        ask = quote.get('ask', 0)
        volume = quote.get('volume', 0)
        change_pct = quote.get('change_percent', 0)

        if not price or price < self.config.min_price or price > self.config.max_price:
            return None

        # Check spread
        if bid and ask and price > 0:
            spread_pct = (ask - bid) / price * 100
            if spread_pct > self.config.max_spread_percent:
                return None

        # Skip if already in position
        if symbol in self.open_positions:
            return None

        # Skip if on cooldown after loss
        if self.last_loss_time:
            cooldown_end = self.last_loss_time + timedelta(seconds=self.config.cooldown_after_loss)
            if datetime.now() < cooldown_end:
                return None

        # Check daily limits
        if self.daily_trades >= self.config.max_daily_trades:
            return None
        if self.daily_pnl <= -self.config.max_daily_loss:
            return None

        # Track price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append(PricePoint(
            timestamp=datetime.now(),
            price=price,
            bid=bid,
            ask=ask,
            volume=volume
        ))

        # Keep only last 60 data points
        self.price_history[symbol] = self.price_history[symbol][-60:]

        # Need at least 5 data points for momentum detection
        if len(self.price_history[symbol]) < 5:
            return None

        # Calculate momentum
        prices = self.price_history[symbol]
        price_5_ago = prices[-5].price if len(prices) >= 5 else prices[0].price
        momentum = ((price - price_5_ago) / price_5_ago * 100) if price_5_ago > 0 else 0

        # Check for momentum spike
        signal = None

        if momentum >= self.config.min_spike_percent:
            signal = {
                "type": SignalType.MOMENTUM_SPIKE.value,
                "symbol": symbol,
                "price": price,
                "momentum": momentum,
                "change_percent": change_pct,
                "volume": volume,
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"MOMENTUM SPIKE: {symbol} +{momentum:.1f}% in 5 ticks @ ${price:.2f}")

        # Volume surge detection
        if signal is None and len(prices) >= 10:
            vol_5_ago = prices[-5].volume if len(prices) >= 5 else 0
            if vol_5_ago > 0 and volume > 0:
                vol_change = volume / vol_5_ago
                if vol_change >= self.config.min_volume_surge and change_pct > 1:
                    signal = {
                        "type": SignalType.VOLUME_SURGE.value,
                        "symbol": symbol,
                        "price": price,
                        "volume_surge": vol_change,
                        "change_percent": change_pct,
                        "timestamp": datetime.now().isoformat()
                    }
                    logger.info(f"VOLUME SURGE: {symbol} {vol_change:.1f}x volume @ ${price:.2f}")

        # Apply Chronos AI filter if enabled
        if signal and self.config.use_chronos_filter:
            chronos_result = self._check_chronos_signal(symbol)
            if chronos_result:
                prob_up = chronos_result.get('prob_up', 0.5)
                chronos_signal = chronos_result.get('signal', 'NEUTRAL')

                # Reject if Chronos says BEARISH or probability too low
                if 'BEARISH' in chronos_signal or prob_up < self.config.chronos_min_prob_up:
                    logger.info(f"CHRONOS REJECT: {symbol} - {chronos_signal} ({prob_up:.0%} prob up)")
                    signal = None
                else:
                    logger.info(f"CHRONOS APPROVE: {symbol} - {chronos_signal} ({prob_up:.0%} prob up)")
                    signal['chronos_signal'] = chronos_signal
                    signal['chronos_prob_up'] = prob_up

        if signal and self.on_signal:
            self.on_signal(signal)

        return signal

    async def check_exit_signal(self, symbol: str, quote: Dict) -> Optional[Dict]:
        """
        Check if there's an exit signal for an open position.

        Returns exit signal dict if should exit, None otherwise.
        """
        if symbol not in self.open_positions:
            return None

        trade = self.open_positions[symbol]
        price = quote.get('price', 0) or quote.get('last', 0)

        if not price:
            return None

        # Update high/low
        if price > trade.high_price:
            trade.high_price = price
        if price < trade.low_price or trade.low_price == 0:
            trade.low_price = price

        # Calculate current P/L
        pnl_pct = ((price - trade.entry_price) / trade.entry_price * 100)
        max_gain = ((trade.high_price - trade.entry_price) / trade.entry_price * 100)

        trade.max_gain_percent = max_gain
        trade.max_drawdown_percent = ((trade.low_price - trade.entry_price) / trade.entry_price * 100)

        # Calculate hold time
        entry_time = datetime.fromisoformat(trade.entry_time)
        hold_seconds = (datetime.now() - entry_time).total_seconds()
        trade.hold_time_seconds = int(hold_seconds)

        exit_signal = None

        # Check stop loss - always protect capital
        if pnl_pct <= -self.config.stop_loss_percent:
            exit_signal = {
                "reason": "STOP_LOSS",
                "pnl_percent": pnl_pct,
                "price": price
            }

        # Check trailing stop - activates once we hit profit target
        # Let winners run but lock in gains
        elif max_gain >= self.config.profit_target_percent:
            # We hit target, now trail from the high
            trailing_trigger = max_gain - self.config.trailing_stop_percent
            if pnl_pct <= trailing_trigger:
                exit_signal = {
                    "reason": "TRAILING_STOP",
                    "pnl_percent": pnl_pct,
                    "max_gain": max_gain,
                    "locked_profit": trailing_trigger,
                    "price": price
                }

        # Check max hold time - but only exit if in profit or small loss
        elif hold_seconds >= self.config.max_hold_seconds:
            # If we're up, let reversal detection handle it
            # If we're down or flat, exit on time
            if pnl_pct <= 1.0:  # Only time-exit if not running
                exit_signal = {
                    "reason": "MAX_HOLD_TIME",
                    "hold_seconds": hold_seconds,
                    "pnl_percent": pnl_pct,
                    "price": price
                }

        # Check for reversal - exit if we were up and now reversing
        if exit_signal is None and len(self.price_history.get(symbol, [])) >= 3:
            prices = self.price_history[symbol]
            recent_high = max(p.price for p in prices[-3:])
            drop_from_high = ((recent_high - price) / recent_high * 100)

            # Exit on reversal if we had gains (lock in profit)
            if drop_from_high >= self.config.reversal_candle_percent and max_gain > 1.0:
                exit_signal = {
                    "reason": "REVERSAL_DETECTED",
                    "drop_percent": drop_from_high,
                    "pnl_percent": pnl_pct,
                    "max_gain": max_gain,
                    "price": price
                }

        return exit_signal

    async def execute_entry(self, symbol: str, signal: Dict) -> Optional[ScalpTrade]:
        """Execute an entry trade"""
        if not self.config.enabled:
            logger.info(f"Scalper disabled - would have entered {symbol}")
            return None

        price = signal.get('price', 0)
        if not price:
            return None

        # Calculate position size using risk-based sizing
        shares, position_value, risk_amount = self.config.calculate_position_size(price)

        if shares < self.config.min_shares:
            logger.debug(f"Position too small for {symbol}: {shares} shares < {self.config.min_shares} min")
            return None

        trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        trade = ScalpTrade(
            trade_id=trade_id,
            symbol=symbol,
            entry_time=datetime.now().isoformat(),
            entry_price=price,
            entry_signal=signal.get('type', 'unknown'),
            shares=shares,
            high_price=price,
            low_price=price,
            status="open"
        )

        # Execute order (paper or real)
        if self.config.paper_mode:
            logger.info(
                f"PAPER ENTRY: {symbol} {shares} shares @ ${price:.2f} | "
                f"Position: ${position_value:.2f} | Risk: ${risk_amount:.2f}"
            )
        else:
            # Real order execution
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "http://localhost:9100/api/order",
                        json={
                            "symbol": symbol,
                            "side": "buy",
                            "qty": shares,
                            "type": "market"
                        },
                        timeout=5.0
                    )
                    if response.status_code != 200:
                        logger.error(f"Order failed: {response.text}")
                        return None
            except Exception as e:
                logger.error(f"Order execution error: {e}")
                return None

        self.open_positions[symbol] = trade
        self.trades.append(trade)
        self.daily_trades += 1
        self._save_trades()

        logger.info(f"ENTRY: {symbol} {shares} @ ${price:.2f} | Signal: {signal.get('type')}")

        if self.on_entry:
            self.on_entry(trade)

        return trade

    async def execute_exit(self, symbol: str, exit_signal: Dict) -> Optional[ScalpTrade]:
        """Execute an exit trade"""
        if symbol not in self.open_positions:
            return None

        trade = self.open_positions[symbol]
        price = exit_signal.get('price', 0)

        if not price:
            return None

        # Calculate P/L
        trade.exit_time = datetime.now().isoformat()
        trade.exit_price = price
        trade.exit_reason = exit_signal.get('reason', 'unknown')
        trade.pnl = (price - trade.entry_price) * trade.shares
        trade.pnl_percent = ((price - trade.entry_price) / trade.entry_price * 100)
        trade.status = "closed"

        # Execute order (paper or real)
        if self.config.paper_mode:
            logger.info(f"PAPER EXIT: {symbol} {trade.shares} shares @ ${price:.2f}")
        else:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "http://localhost:9100/api/order",
                        json={
                            "symbol": symbol,
                            "side": "sell",
                            "qty": trade.shares,
                            "type": "market"
                        },
                        timeout=5.0
                    )
            except Exception as e:
                logger.error(f"Exit order error: {e}")

        # Update daily stats
        self.daily_pnl += trade.pnl
        if trade.pnl < 0:
            self.last_loss_time = datetime.now()

        # Remove from open positions
        del self.open_positions[symbol]
        self._save_trades()

        pnl_emoji = "WIN" if trade.pnl > 0 else "LOSS"
        logger.info(
            f"EXIT [{pnl_emoji}]: {symbol} @ ${price:.2f} | "
            f"P/L: ${trade.pnl:.2f} ({trade.pnl_percent:+.1f}%) | "
            f"Reason: {trade.exit_reason} | "
            f"Hold: {trade.hold_time_seconds}s"
        )

        if self.on_exit:
            self.on_exit(trade)

        return trade

    def start(self, symbols: List[str] = None):
        """Start the scalper"""
        if self.is_running:
            logger.warning("Scalper already running")
            return

        if symbols:
            self.config.watchlist = [s.upper() for s in symbols]
            self._save_config()

        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        logger.info(f"HFT Scalper started - watching {len(self.config.watchlist)} symbols")

    def stop(self):
        """Stop the scalper"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("HFT Scalper stopped")

    def _run_loop(self):
        """Background monitoring loop"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._monitor_loop())
        except Exception as e:
            logger.error(f"Scalper loop error: {e}")
        finally:
            self._loop.close()

    async def _monitor_loop(self):
        """Main monitoring loop"""
        import httpx

        while self.is_running:
            try:
                async with httpx.AsyncClient() as client:
                    # Check each symbol
                    for symbol in self.config.watchlist[:20]:  # Limit to 20
                        if symbol in self.config.blacklist:
                            continue

                        try:
                            response = await client.get(
                                f"http://localhost:9100/api/price/{symbol}",
                                timeout=2.0
                            )
                            if response.status_code == 200:
                                quote = response.json()

                                # Check for exit signals first
                                if symbol in self.open_positions:
                                    exit_signal = await self.check_exit_signal(symbol, quote)
                                    if exit_signal:
                                        await self.execute_exit(symbol, exit_signal)
                                else:
                                    # Check for entry signals
                                    entry_signal = await self.check_entry_signal(symbol, quote)
                                    if entry_signal and self.config.enabled:
                                        await self.execute_entry(symbol, entry_signal)
                        except Exception as e:
                            logger.debug(f"Error checking {symbol}: {e}")

                        await asyncio.sleep(0.1)  # Small delay between symbols

                await asyncio.sleep(0.5)  # 500ms between full scans

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(2)

    def get_status(self) -> Dict:
        """Get scalper status"""
        # Calculate example position sizes
        example_sizes = {}
        for price in [2.0, 5.0, 10.0]:
            shares, pos_val, risk = self.config.calculate_position_size(price)
            example_sizes[f"${price:.0f}_stock"] = {
                "shares": shares,
                "position": round(pos_val, 2),
                "risk": round(risk, 2)
            }

        return {
            "is_running": self.is_running,
            "enabled": self.config.enabled,
            "paper_mode": self.config.paper_mode,
            "watchlist_count": len(self.config.watchlist),
            "open_positions": len(self.open_positions),
            "daily_trades": self.daily_trades,
            "daily_pnl": round(self.daily_pnl, 2),
            "risk_management": {
                "account_size": self.config.account_size,
                "risk_percent": self.config.risk_percent,
                "risk_per_trade": round(self.config.account_size * self.config.risk_percent / 100, 2),
                "max_trades_before_wipeout": int(100 / self.config.risk_percent),
                "use_risk_based_sizing": self.config.use_risk_based_sizing
            },
            "position_examples": example_sizes,
            "config": {
                "profit_target": self.config.profit_target_percent,
                "stop_loss": self.config.stop_loss_percent,
                "trailing_stop": self.config.trailing_stop_percent,
                "max_position": self.config.max_position_size,
                "min_price": self.config.min_price,
                "max_price": self.config.max_price
            }
        }

    def get_open_positions(self) -> List[Dict]:
        """Get open positions"""
        return [t.to_dict() for t in self.open_positions.values()]

    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get trade history"""
        closed = [t for t in self.trades if t.status == 'closed']
        return [t.to_dict() for t in closed[-limit:]]

    def get_stats(self) -> Dict:
        """Get trading statistics"""
        closed = [t for t in self.trades if t.status == 'closed']

        if not closed:
            return {"message": "No completed trades yet"}

        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in closed)
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        avg_hold = sum(t.hold_time_seconds for t in closed) / len(closed)

        return {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_hold_seconds": round(avg_hold, 1),
            "best_trade": max(t.pnl for t in closed) if closed else 0,
            "worst_trade": min(t.pnl for t in closed) if closed else 0,
            "profit_factor": round(abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in losses)), 2) if losses and sum(t.pnl for t in losses) != 0 else 0
        }

    def reset_daily(self) -> Dict:
        """Reset daily stats for a fresh start"""
        old_trades = self.daily_trades
        old_pnl = self.daily_pnl

        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_loss_time = None

        logger.info(f"Daily stats reset - was: {old_trades} trades, ${old_pnl:.2f} P/L")

        return {
            "reset": True,
            "previous_trades": old_trades,
            "previous_pnl": round(old_pnl, 2),
            "current_trades": 0,
            "current_pnl": 0.0
        }


# Singleton
_scalper: Optional[HFTScalper] = None


def get_hft_scalper() -> HFTScalper:
    """Get or create HFT scalper singleton"""
    global _scalper
    if _scalper is None:
        _scalper = HFTScalper()
    return _scalper


def start_scalper(symbols: List[str] = None) -> HFTScalper:
    """Start the HFT scalper"""
    scalper = get_hft_scalper()
    scalper.start(symbols)
    return scalper


def stop_scalper():
    """Stop the HFT scalper"""
    scalper = get_hft_scalper()
    scalper.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test
    scalper = get_hft_scalper()
    scalper.config.watchlist = ['YCBD', 'ADTX', 'AZI']
    scalper.config.enabled = False  # Paper mode

    print(f"Status: {scalper.get_status()}")
    print(f"Stats: {scalper.get_stats()}")
