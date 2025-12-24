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

    # Order Flow filter
    use_order_flow_filter: bool = True  # Filter entries with bid/ask imbalance
    min_buy_pressure: float = 0.55  # Minimum buy pressure to enter (55%)

    # Regime Gating filter
    use_regime_gating: bool = True  # Filter entries by market regime
    valid_regimes: List[str] = field(default_factory=lambda: ['TRENDING_UP', 'RANGING'])

    # Scalp Fade Filter (uses Polygon real-time data)
    use_scalp_fade_filter: bool = True  # Filter fading spikes
    scalp_fade_threshold: float = 0.45  # Below this = FADE
    scalp_continue_threshold: float = 0.55  # Above this = CONTINUE

    # Warrior Pullback Confirmation (prevents chasing pump & dumps)
    use_pullback_confirmation: bool = True  # Wait for pullback before entry
    pullback_min_percent: float = 2.0  # Min pullback from spike high (%)
    pullback_max_percent: float = 10.0  # Max pullback (too deep = failed)
    confirmation_break_percent: float = 0.5  # Break above pullback high by X%
    pullback_timeout_seconds: int = 120  # Max time to wait for confirmation

    # ATR-based dynamic stops
    use_atr_stops: bool = True  # Use ATR-based stops instead of fixed %
    atr_stop_multiplier: float = 1.5  # Stop at 1.5x ATR
    atr_target_multiplier: float = 2.0  # Target at 2.0x ATR
    atr_trail_multiplier: float = 1.0  # Trail at 1.0x ATR

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

    # Secondary triggers for correlation analysis
    secondary_triggers: Optional[Dict] = None

    # ATR-based dynamic stops
    atr: float = 0.0  # ATR value at entry
    atr_stop_price: float = 0.0  # Dynamic stop price
    atr_target_price: float = 0.0  # Dynamic target price
    atr_trail_distance: float = 0.0  # Trailing stop distance
    volatility_regime: str = ""  # LOW, NORMAL, HIGH, EXTREME

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


@dataclass
class PullbackWatch:
    """Track a symbol waiting for pullback confirmation (Warrior method)"""
    symbol: str
    spike_time: datetime
    spike_price: float  # Price when spike detected
    spike_high: float  # Highest price during spike
    pullback_low: float  # Lowest price during pullback
    state: str = "WATCHING"  # WATCHING -> PULLBACK -> CONFIRMING -> READY
    # WATCHING: Just detected spike, watching for pullback
    # PULLBACK: Price pulled back enough, waiting for new high
    # CONFIRMING: Price breaking above pullback high
    # READY: Confirmed, ready to enter
    # FAILED: Pullback too deep or timed out


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

        # Filter stats
        self.fade_filter_rejects = 0
        self.fade_filter_approves = 0

        # Priority queue for news-triggered entries
        self.priority_symbols: List[str] = []  # Symbols to check immediately

        # Pullback confirmation tracking (Warrior method)
        self.pullback_watches: Dict[str, PullbackWatch] = {}  # symbol -> watch state

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

    def _check_scalp_fade_signal(self, symbol: str) -> tuple:
        """
        Check if scalp model predicts fade using real-time Polygon data.

        Returns:
            (should_trade, probability, verdict)
        """
        try:
            from ai.ensemble_predictor import PolygonScalpScorer
            from polygon_streaming import get_polygon_stream

            # Get or create scorer (singleton)
            if not hasattr(self, '_scalp_scorer'):
                self._scalp_scorer = PolygonScalpScorer()

            if not self._scalp_scorer.available:
                return True, 0.5, "MODEL_N/A"

            # Get real-time minute bars from Polygon stream
            stream = get_polygon_stream()
            df = stream.get_minute_bars(symbol, minutes=30)

            if df is None or len(df) < 20:
                return True, 0.5, "INSUFFICIENT_DATA"

            prob, details = self._scalp_scorer.calculate(df)
            verdict = details.get("verdict", "NEUTRAL")

            fade_threshold = getattr(self.config, 'scalp_fade_threshold', 0.45)

            if prob < fade_threshold:
                return False, prob, "LIKELY_FADE"

            return True, prob, verdict

        except Exception as e:
            logger.warning(f"Scalp fade check failed for {symbol}: {e}")
            return True, 0.5, "ERROR"

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

    def _check_pullback_confirmation(self, symbol: str, price: float,
                                      momentum: float, change_pct: float,
                                      volume: int) -> Optional[Dict]:
        """
        Warrior Trading Method: Wait for pullback confirmation before entry.

        Flow:
        1. Spike detected -> Add to watch list (WATCHING)
        2. Price pulls back X% from high -> State = PULLBACK
        3. Price breaks above pullback high -> State = READY, return signal
        4. Timeout or pullback too deep -> State = FAILED, remove from watch
        """
        now = datetime.now()

        # Check if we're already watching this symbol
        if symbol in self.pullback_watches:
            watch = self.pullback_watches[symbol]

            # Check for timeout
            elapsed = (now - watch.spike_time).total_seconds()
            if elapsed > self.config.pullback_timeout_seconds:
                logger.info(f"PULLBACK TIMEOUT: {symbol} - no confirmation in {elapsed:.0f}s")
                del self.pullback_watches[symbol]
                return None

            # Update high/low tracking
            if price > watch.spike_high:
                watch.spike_high = price
            if price < watch.pullback_low:
                watch.pullback_low = price

            # Calculate pullback from high
            pullback_pct = (watch.spike_high - price) / watch.spike_high * 100 if watch.spike_high > 0 else 0
            recovery_pct = (price - watch.pullback_low) / watch.pullback_low * 100 if watch.pullback_low > 0 else 0

            # State machine
            if watch.state == "WATCHING":
                # Waiting for pullback to start
                if pullback_pct >= self.config.pullback_min_percent:
                    watch.state = "PULLBACK"
                    watch.pullback_low = price
                    logger.info(f"PULLBACK DETECTED: {symbol} -{pullback_pct:.1f}% from high ${watch.spike_high:.2f}")
                elif pullback_pct > self.config.pullback_max_percent:
                    # Pullback too deep - failed
                    logger.info(f"PULLBACK FAILED: {symbol} too deep -{pullback_pct:.1f}%")
                    del self.pullback_watches[symbol]
                    return None

            elif watch.state == "PULLBACK":
                # Check if pullback too deep
                total_pullback = (watch.spike_high - watch.pullback_low) / watch.spike_high * 100
                if total_pullback > self.config.pullback_max_percent:
                    logger.info(f"PULLBACK FAILED: {symbol} too deep -{total_pullback:.1f}%")
                    del self.pullback_watches[symbol]
                    return None

                # Check for confirmation - price breaking above pullback high
                break_level = watch.pullback_low * (1 + self.config.confirmation_break_percent / 100)
                if price > break_level and recovery_pct >= self.config.confirmation_break_percent:
                    # CONFIRMED! Ready to enter
                    logger.info(f"PULLBACK CONFIRMED: {symbol} @ ${price:.2f} (broke ${break_level:.2f})")
                    del self.pullback_watches[symbol]

                    return {
                        "type": SignalType.MOMENTUM_SPIKE.value,
                        "symbol": symbol,
                        "price": price,
                        "momentum": momentum,
                        "change_percent": change_pct,
                        "volume": volume,
                        "pullback_confirmed": True,
                        "spike_high": watch.spike_high,
                        "pullback_low": watch.pullback_low,
                        "timestamp": datetime.now().isoformat()
                    }

            return None  # Still watching/waiting

        else:
            # New spike detected - start watching for pullback
            self.pullback_watches[symbol] = PullbackWatch(
                symbol=symbol,
                spike_time=now,
                spike_price=price,
                spike_high=price,
                pullback_low=price,
                state="WATCHING"
            )
            logger.info(f"SPIKE DETECTED: {symbol} +{momentum:.1f}% @ ${price:.2f} - watching for pullback")
            return None  # Don't enter yet - wait for confirmation

    def get_pullback_watches(self) -> Dict[str, Dict]:
        """Get current pullback watch states for dashboard"""
        result = {}
        for symbol, watch in self.pullback_watches.items():
            result[symbol] = {
                "symbol": watch.symbol,
                "state": watch.state,
                "spike_time": watch.spike_time.isoformat(),
                "spike_price": watch.spike_price,
                "spike_high": watch.spike_high,
                "pullback_low": watch.pullback_low,
                "age_seconds": (datetime.now() - watch.spike_time).total_seconds()
            }
        return result

    async def check_entry_signal(self, symbol: str, quote: Dict,
                                  priority: bool = False) -> Optional[Dict]:
        """
        Check if there's an entry signal for a symbol.

        Args:
            symbol: Stock symbol
            quote: Price quote dict
            priority: If True, this is a news-triggered priority entry
                     (skip momentum detection, already passed news filters)

        Returns signal dict if entry opportunity detected, None otherwise.
        """
        price = quote.get('price', 0) or quote.get('last', 0)
        bid = quote.get('bid', 0)
        ask = quote.get('ask', 0)
        volume = quote.get('volume', 0)
        change_pct = quote.get('change_percent', 0)

        if not price or price < self.config.min_price or price > self.config.max_price:
            return None

        # Check spread - slightly relaxed for priority entries (news-triggered)
        max_spread = self.config.max_spread_percent * (1.5 if priority else 1.0)
        if bid and ask and price > 0:
            spread_pct = (ask - bid) / price * 100
            if spread_pct > max_spread:
                if priority:
                    logger.info(f"PRIORITY REJECT: {symbol} spread {spread_pct:.1f}% > {max_spread:.1f}%")
                return None

        # Skip if already in position
        if symbol in self.open_positions:
            return None

        # Skip if on cooldown after loss (shorter cooldown for priority)
        if self.last_loss_time:
            cooldown = self.config.cooldown_after_loss / (2 if priority else 1)
            cooldown_end = self.last_loss_time + timedelta(seconds=cooldown)
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

        # For priority entries (news-triggered), skip momentum detection
        # They already passed news confidence/urgency filters
        if priority:
            signal = {
                "type": "news_triggered",
                "symbol": symbol,
                "price": price,
                "change_percent": change_pct,
                "volume": volume,
                "priority": True,
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"PRIORITY ENTRY: {symbol} @ ${price:.2f} (news-triggered)")
        else:
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
                # WARRIOR METHOD: Pullback Confirmation
                if self.config.use_pullback_confirmation:
                    signal = self._check_pullback_confirmation(symbol, price, momentum, change_pct, volume)
                else:
                    # Original behavior - enter immediately on spike
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

        # Apply Order Flow filter if enabled
        if signal and self.config.use_order_flow_filter:
            try:
                from ai.order_flow_analyzer import get_order_flow_analyzer
                analyzer = get_order_flow_analyzer()
                flow_signal = analyzer.analyze(quote)

                if flow_signal.recommendation == 'SKIP':
                    logger.info(f"ORDER FLOW REJECT: {symbol} - {flow_signal.reason}")
                    signal = None
                elif flow_signal.recommendation == 'ENTER':
                    logger.info(f"ORDER FLOW APPROVE: {symbol} - Buy pressure {flow_signal.buy_pressure*100:.0f}%")
                    signal['order_flow_buy_pressure'] = flow_signal.buy_pressure
                    signal['order_flow_spread'] = flow_signal.spread_percent
                else:
                    # NEUTRAL - allow with lower confidence
                    logger.info(f"ORDER FLOW NEUTRAL: {symbol} - {flow_signal.reason}")
                    signal['order_flow_buy_pressure'] = flow_signal.buy_pressure
                    signal['order_flow_spread'] = flow_signal.spread_percent
            except Exception as e:
                logger.warning(f"Order flow check failed for {symbol}: {e}")

        # Apply Regime Gating filter if enabled
        if signal and getattr(self.config, 'use_regime_gating', False):
            try:
                from ai.chronos_adapter import get_chronos_adapter
                adapter = get_chronos_adapter()
                context = adapter.get_context(symbol)

                valid_regimes = getattr(self.config, 'valid_regimes', ['TRENDING_UP', 'RANGING'])

                if context.market_regime not in valid_regimes:
                    logger.info(f"REGIME REJECT: {symbol} - {context.market_regime} not in {valid_regimes}")
                    signal = None
                else:
                    logger.info(f"REGIME APPROVE: {symbol} - {context.market_regime} ({context.regime_confidence:.0%} conf)")
                    signal['market_regime'] = context.market_regime
                    signal['regime_confidence'] = context.regime_confidence
            except Exception as e:
                logger.warning(f"Regime gating check failed for {symbol}: {e}")

        # Apply Scalp Fade Filter if enabled
        if signal and getattr(self.config, 'use_scalp_fade_filter', False):
            should_trade, prob, verdict = self._check_scalp_fade_signal(symbol)

            if not should_trade:
                logger.info(f"🚫 FADE FILTER: {symbol} rejected - {prob:.0%} continuation ({verdict})")
                self.fade_filter_rejects += 1
                return None

            # Add scalp data to signal
            signal['scalp_prob'] = prob
            signal['scalp_verdict'] = verdict
            self.fade_filter_approves += 1

            if verdict == "LIKELY_CONTINUE":
                logger.info(f"✅ FADE FILTER: {symbol} approved - {prob:.0%} continuation")

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

        # Use ATR-based stops if available, otherwise fall back to fixed %
        if self.config.use_atr_stops and trade.atr_stop_price > 0:
            # ATR-based stop loss
            if price <= trade.atr_stop_price:
                exit_signal = {
                    "reason": "STOP_LOSS",
                    "pnl_percent": pnl_pct,
                    "price": price,
                    "stop_type": "ATR",
                    "atr_stop": trade.atr_stop_price
                }
            # ATR-based trailing stop - activates once we hit ATR target
            elif trade.high_price >= trade.atr_target_price and trade.atr_trail_distance > 0:
                # Calculate trailing stop from high
                trailing_stop = trade.high_price - trade.atr_trail_distance
                if price <= trailing_stop:
                    exit_signal = {
                        "reason": "TRAILING_STOP",
                        "pnl_percent": pnl_pct,
                        "max_gain": max_gain,
                        "price": price,
                        "stop_type": "ATR",
                        "trailing_stop": trailing_stop
                    }
        else:
            # Fallback to fixed percentage stops
            # Check stop loss - always protect capital
            if pnl_pct <= -self.config.stop_loss_percent:
                exit_signal = {
                    "reason": "STOP_LOSS",
                    "pnl_percent": pnl_pct,
                    "price": price,
                    "stop_type": "FIXED"
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
                        "price": price,
                        "stop_type": "FIXED"
                    }

        # Check max hold time - but only exit if in profit or small loss
        if exit_signal is None and hold_seconds >= self.config.max_hold_seconds:
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

    async def execute_entry(self, symbol: str, signal: Dict, quote: Dict = None) -> Optional[ScalpTrade]:
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

        # Capture secondary triggers for correlation analysis
        secondary_triggers = None
        try:
            from ai.trade_signals import get_secondary_triggers
            # Use quote if provided, otherwise build from signal
            entry_quote = quote if quote else {
                'price': price,
                'last': price,
                'change_percent': signal.get('change_percent', 0),
                'volume': signal.get('volume', 0)
            }
            secondary_triggers = await get_secondary_triggers(symbol, entry_quote, signal)
            logger.debug(f"Captured {len(secondary_triggers)} secondary triggers for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to capture secondary triggers: {e}")

        # Calculate ATR-based dynamic stops if enabled
        atr = 0.0
        atr_stop_price = price * (1 - self.config.stop_loss_percent / 100)  # Fallback
        atr_target_price = price * (1 + self.config.profit_target_percent / 100)  # Fallback
        atr_trail_distance = price * (self.config.trailing_stop_percent / 100)  # Fallback
        volatility_regime = ""

        if self.config.use_atr_stops:
            try:
                from ai.atr_stops import get_atr_calculator
                calculator = get_atr_calculator()
                calculator.stop_multiplier = self.config.atr_stop_multiplier
                calculator.target_multiplier = self.config.atr_target_multiplier
                calculator.trailing_multiplier = self.config.atr_trail_multiplier

                stops = calculator.calculate_stops(symbol, price)
                if stops:
                    atr = stops.atr
                    atr_stop_price = stops.stop_price
                    atr_target_price = stops.target_price
                    atr_trail_distance = stops.trailing_stop_distance
                    volatility_regime = stops.volatility_regime
                    logger.info(
                        f"ATR STOPS: {symbol} ATR=${atr:.3f} ({volatility_regime}) | "
                        f"Stop=${atr_stop_price:.2f} | Target=${atr_target_price:.2f}"
                    )
            except Exception as e:
                logger.warning(f"ATR calculation failed for {symbol}, using fixed %: {e}")

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
            status="open",
            secondary_triggers=secondary_triggers,
            atr=atr,
            atr_stop_price=atr_stop_price,
            atr_target_price=atr_target_price,
            atr_trail_distance=atr_trail_distance,
            volatility_regime=volatility_regime
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

        # Subscribe watchlist to Polygon stream for fade filter
        if getattr(self.config, 'use_scalp_fade_filter', False):
            try:
                from polygon_streaming import get_polygon_stream
                stream = get_polygon_stream()
                for symbol in self.config.watchlist:
                    stream.subscribe_trades(symbol)
                    stream.subscribe_luld(symbol)
                stream.start()
                logger.info(f"Polygon stream started for fade filter: {len(self.config.watchlist)} symbols")
            except Exception as e:
                logger.warning(f"Polygon stream failed to start: {e}")

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
                    # Sync with common worklist (data bus pattern)
                    try:
                        wl_resp = await client.get("http://localhost:9100/api/worklist", timeout=3.0)
                        if wl_resp.status_code == 200:
                            wl_data = wl_resp.json()
                            common_symbols = [s.get('symbol') for s in wl_data.get('data', []) if s.get('symbol')]
                            if common_symbols:
                                # Merge with config watchlist (union)
                                all_symbols = list(set(self.config.watchlist + common_symbols))
                                self.config.watchlist = all_symbols
                    except Exception as e:
                        logger.debug(f"Worklist sync: {e}")

                    # Process priority symbols FIRST (news-triggered)
                    while self.priority_symbols and self.config.enabled:
                        symbol = self.priority_symbols.pop(0)

                        if symbol in self.config.blacklist:
                            continue
                        if symbol in self.open_positions:
                            continue  # Already have position

                        try:
                            response = await client.get(
                                f"http://localhost:9100/api/price/{symbol}",
                                timeout=2.0
                            )
                            if response.status_code == 200:
                                quote = response.json()
                                # Priority symbols get fast-tracked entry
                                # They already passed news filters, just check price/spread
                                entry_signal = await self.check_entry_signal(
                                    symbol, quote, priority=True
                                )
                                if entry_signal:
                                    logger.warning(f"NEWS-TRIGGERED ENTRY: {symbol}")
                                    await self.execute_entry(symbol, entry_signal, quote)
                        except Exception as e:
                            logger.debug(f"Error on priority {symbol}: {e}")

                    # Check each symbol on regular watchlist
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
                                        await self.execute_entry(symbol, entry_signal, quote)
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
