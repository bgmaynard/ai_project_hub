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

    # Chronos Smart Exit (prevents stop loss deaths)
    use_chronos_exit: bool = True  # Use Chronos to detect momentum fading
    chronos_exit_min_hold: int = 10  # Min seconds before Chronos can exit

    # Chronos AI filter
    use_chronos_filter: bool = True  # Filter entries with Chronos AI
    chronos_min_prob_up: float = 0.5  # Min probability to enter (0.5 = 50%)

    # Order Flow filter
    use_order_flow_filter: bool = True  # Filter entries with bid/ask imbalance
    min_buy_pressure: float = 0.55  # Minimum buy pressure to enter (55%)

    # Regime Gating filter
    use_regime_gating: bool = True  # Filter entries by market regime
    valid_regimes: List[str] = field(default_factory=lambda: ['TRENDING_UP', 'RANGING'])

    # Technical Signal Filter (EMA/MACD/VWAP confluence)
    use_signal_filter: bool = True  # Filter entries with technical signals
    signal_min_confluence: float = 70.0  # Minimum confluence score (0-100)
    signal_require_ema_bullish: bool = True  # Require EMA 9 > EMA 20
    signal_require_macd_bullish: bool = True  # Require MACD > Signal
    signal_require_above_vwap: bool = False  # Require price > VWAP (optional)

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

    # Failed momentum early exit (cut losers fast)
    failed_momentum_seconds: int = 45  # Check for failed momentum after X seconds
    failed_momentum_threshold: float = 0.5  # Exit if gain < X% after failed_momentum_seconds

    # Momentum velocity filter (ensure still moving up at entry)
    use_velocity_filter: bool = True  # Check if price still rising at entry
    min_entry_velocity: float = 0.1  # Min % gain in last 2 ticks to enter

    # Risk management
    max_daily_loss: float = 50.0  # Max daily loss before stopping
    max_daily_trades: int = 20  # Max trades per day
    cooldown_after_loss: int = 60  # Seconds to wait after a loss

    # Time of day filter (block bad hours)
    blocked_hours: List[int] = field(default_factory=list)  # Legacy - use time range instead
    blocked_time_start: int = 925  # Block from 9:25 AM ET (HHMM format)
    blocked_time_end: int = 959    # Block until 9:59 AM ET (HHMM format)

    # Volume surge entry filter
    volume_surge_max_hod_distance: float = 2.0  # Max % distance from HOD for volume surge entries

    # Warrior Trading filter (Ross Cameron methodology)
    use_warrior_filter: bool = True  # Filter entries with Warrior setup grading
    warrior_min_grade: str = 'B'  # Minimum grade to enter (A, B, or C)
    warrior_require_pattern: bool = False  # Require confirmed pattern (Bull Flag, ABCD, etc.)
    warrior_require_tape_signal: bool = False  # Require tape confirmation (green flow, seller thinning)
    warrior_max_float: float = 20.0  # Max float in millions (low float = more volatility)
    warrior_min_rvol: float = 2.0  # Minimum relative volume

    # Multi-Timeframe Confirmation (1M + 5M alignment)
    use_mtf_filter: bool = True  # Require 1M and 5M timeframe alignment
    mtf_min_confidence: float = 60.0  # Minimum MTF confidence to enter (0-100)
    mtf_require_vwap_aligned: bool = True  # Both timeframes above VWAP
    mtf_require_macd_aligned: bool = True  # Both timeframes MACD bullish

    # VWAP Filter (Ross Cameron - line in the sand)
    use_vwap_filter: bool = True  # Require price above VWAP for entry
    vwap_max_extension_pct: float = 3.0  # Max % above VWAP (avoid chasing)
    use_vwap_trailing_stop: bool = True  # Use VWAP as trailing stop
    vwap_stop_offset_pct: float = 0.3  # Trail 0.3% below VWAP

    # Float Rotation (Ross Cameron - volume vs float tracking)
    use_float_rotation_boost: bool = True  # Boost confidence when float rotating
    min_rotation_for_boost: float = 0.5  # Min rotation ratio for boost (0.5 = 50% of float)
    require_low_float: bool = False  # Only trade low float stocks (<20M)
    max_float_millions: float = 50.0  # Max float in millions to trade

    # Momentum Exhaustion Detection (early exit before big drops)
    use_exhaustion_exit: bool = True  # Exit on exhaustion signals
    exhaustion_exit_threshold: float = 60.0  # Min exhaustion score to exit (0-100)
    exhaustion_exit_on_divergence: bool = True  # Exit on RSI divergence
    exhaustion_exit_on_red_candles: int = 4  # Exit after N consecutive red candles

    # Level 2 Depth Analysis (order book signals)
    use_depth_analysis: bool = True  # Use order book for entry/exit decisions
    depth_require_bullish_imbalance: bool = False  # Require bid > ask imbalance
    depth_min_imbalance_ratio: float = 1.2  # Min bid/ask ratio for boost
    depth_block_on_ask_wall: bool = True  # Block entry if strong ask wall nearby

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

    # Technical signal state at entry (for correlation analysis)
    signal_confluence: float = 0.0  # Confluence score 0-100
    signal_bias: str = ""  # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    signal_ema_bullish: bool = False  # EMA 9 > EMA 20
    signal_macd_bullish: bool = False  # MACD > Signal
    signal_above_vwap: bool = False  # Price > VWAP
    signal_ema_crossover: str = ""  # BULLISH, BEARISH, NONE
    signal_macd_crossover: str = ""  # BULLISH, BEARISH, NONE
    signal_vwap_crossover: str = ""  # BULLISH, BEARISH, NONE
    signal_candle_momentum: str = ""  # BUILDING, FADING, NEUTRAL

    # Warrior Trading state at entry (Ross Cameron methodology)
    warrior_grade: str = ""  # A, B, or C
    warrior_score: float = 0.0  # 0-100 setup quality score
    warrior_patterns: List[str] = field(default_factory=list)  # Detected patterns
    warrior_tape_signals: List[str] = field(default_factory=list)  # Tape reading signals

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

    def _check_technical_signals(self, symbol: str) -> Optional[Dict]:
        """
        Check technical signals (EMA/MACD/VWAP confluence) for a symbol.
        Fetches OHLC data from API and calculates signal state.

        Returns dict with signal state or None if unavailable.
        """
        try:
            import httpx

            # Fetch OHLC data from our API
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"http://localhost:9100/api/charts/signals/{symbol}",
                    params={"timeframe": "5m", "days": 2}
                )

                if resp.status_code != 200:
                    return None

                data = resp.json()

                if not data.get("success"):
                    return None

                signals = data.get("signals", {})

                return {
                    'confluence_score': signals.get('confluence_score', 0),
                    'signal_bias': signals.get('signal_bias', 'NEUTRAL'),
                    'ema_bullish': signals.get('ema_bullish', False),
                    'macd_bullish': signals.get('macd_bullish', False),
                    'price_above_vwap': signals.get('price_above_vwap', False),
                    'ema_crossover': signals.get('ema_crossover', 'NONE'),
                    'macd_crossover': signals.get('macd_crossover', 'NONE'),
                    'vwap_crossover': signals.get('vwap_crossover', 'NONE'),
                    'candle_momentum': signals.get('candle_momentum', 'NEUTRAL'),
                    'ema9': signals.get('ema9', 0),
                    'ema20': signals.get('ema20', 0),
                    'vwap': signals.get('vwap', 0)
                }

        except Exception as e:
            logger.warning(f"Technical signal check failed for {symbol}: {e}")
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

        # TIME OF DAY FILTER - Block market open chaos (9:25-9:59 AM ET)
        # Pre-market 9:00-9:25 is good for news momentum, but open is chaotic
        # Server is CT (1 hour behind ET), so convert: ET time - 1 hour = CT time
        try:
            from zoneinfo import ZoneInfo
            et_now = datetime.now(ZoneInfo('America/New_York'))
            et_time = et_now.hour * 100 + et_now.minute  # HHMM format
        except:
            # Fallback: assume server is CT (1 hour behind ET)
            ct_now = datetime.now()
            et_time = (ct_now.hour + 1) * 100 + ct_now.minute  # Add 1 hour for ET

        # Configurable blocked time range (default 9:25-9:59 AM ET = 925-959)
        blocked_start = getattr(self.config, 'blocked_time_start', 925)  # 9:25 AM ET
        blocked_end = getattr(self.config, 'blocked_time_end', 959)      # 9:59 AM ET

        if blocked_start <= et_time <= blocked_end and not priority:
            logger.debug(f"TIME BLOCK: {symbol} - {et_time} ET in blocked range {blocked_start}-{blocked_end}")
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
                        # ===== NEAR HOD FILTER FOR VOLUME SURGE =====
                        # Only enter on volume surge if stock is near high-of-day
                        # Prevents entering fading stocks that had volume spike
                        max_distance_from_hod = getattr(self.config, 'volume_surge_max_hod_distance', 2.0)  # 2% default

                        # Get high of day from quote if available
                        hod = getattr(quote, 'high', None) or getattr(quote, 'hod', None) or 0
                        if hod > 0:
                            distance_from_hod = ((hod - price) / hod * 100)
                            if distance_from_hod > max_distance_from_hod:
                                logger.info(f"VOLUME SURGE REJECTED: {symbol} - {distance_from_hod:.1f}% off HOD (max {max_distance_from_hod}%)")
                                signal = None
                            else:
                                signal = {
                                    "type": SignalType.VOLUME_SURGE.value,
                                    "symbol": symbol,
                                    "price": price,
                                    "volume_surge": vol_change,
                                    "change_percent": change_pct,
                                    "distance_from_hod": distance_from_hod,
                                    "timestamp": datetime.now().isoformat()
                                }
                                logger.info(f"VOLUME SURGE: {symbol} {vol_change:.1f}x volume @ ${price:.2f} ({distance_from_hod:.1f}% from HOD)")
                        else:
                            # No HOD data, fall back to momentum check
                            if momentum >= 1.0:  # At least 1% recent momentum
                                signal = {
                                    "type": SignalType.VOLUME_SURGE.value,
                                    "symbol": symbol,
                                    "price": price,
                                    "volume_surge": vol_change,
                                    "change_percent": change_pct,
                                    "timestamp": datetime.now().isoformat()
                                }
                                logger.info(f"VOLUME SURGE: {symbol} {vol_change:.1f}x volume @ ${price:.2f} (momentum +{momentum:.1f}%)")
                            else:
                                logger.debug(f"VOLUME SURGE SKIPPED: {symbol} - No HOD data and weak momentum")

        # ===== MOMENTUM VELOCITY FILTER =====
        # Check if price is STILL moving up right now (not just moved up in past)
        # This prevents entering after momentum has already stalled
        if signal and getattr(self.config, 'use_velocity_filter', True):
            if len(prices) >= 3:
                # Compare current price to price 2 ticks ago
                price_2_ago = prices[-3].price
                current_velocity = ((price - price_2_ago) / price_2_ago * 100) if price_2_ago > 0 else 0
                min_velocity = getattr(self.config, 'min_entry_velocity', 0.1)  # 0.1% min upward movement

                if current_velocity < min_velocity:
                    logger.info(f"VELOCITY REJECT: {symbol} - Momentum stalled ({current_velocity:+.2f}% < {min_velocity}%)")
                    signal = None
                else:
                    logger.debug(f"VELOCITY OK: {symbol} - Still moving +{current_velocity:.2f}%")
                    signal['entry_velocity'] = current_velocity

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

        # Apply Technical Signal Filter if enabled (EMA/MACD/VWAP confluence)
        if signal and getattr(self.config, 'use_signal_filter', True):
            signal_result = self._check_technical_signals(symbol)
            if signal_result:
                confluence = signal_result.get('confluence_score', 0)
                min_conf = getattr(self.config, 'signal_min_confluence', 70.0)

                ema_bullish = signal_result.get('ema_bullish', False)
                macd_bullish = signal_result.get('macd_bullish', False)
                above_vwap = signal_result.get('price_above_vwap', False)

                # Check required conditions
                approved = True
                reject_reasons = []

                if confluence < min_conf:
                    approved = False
                    reject_reasons.append(f"confluence {confluence:.0f}% < {min_conf:.0f}%")

                if getattr(self.config, 'signal_require_ema_bullish', True) and not ema_bullish:
                    approved = False
                    reject_reasons.append("EMA bearish")

                if getattr(self.config, 'signal_require_macd_bullish', True) and not macd_bullish:
                    approved = False
                    reject_reasons.append("MACD bearish")

                if getattr(self.config, 'signal_require_above_vwap', False) and not above_vwap:
                    approved = False
                    reject_reasons.append("below VWAP")

                if not approved:
                    logger.info(f"SIGNAL REJECT: {symbol} - {', '.join(reject_reasons)}")
                    signal = None
                else:
                    logger.info(f"SIGNAL APPROVE: {symbol} - Confluence {confluence:.0f}%, EMA={'Bull' if ema_bullish else 'Bear'}, MACD={'Bull' if macd_bullish else 'Bear'}, VWAP={'Above' if above_vwap else 'Below'}")
                    # Store signal state for correlation analysis
                    signal['signal_confluence'] = confluence
                    signal['signal_bias'] = signal_result.get('signal_bias', 'NEUTRAL')
                    signal['signal_ema_bullish'] = ema_bullish
                    signal['signal_macd_bullish'] = macd_bullish
                    signal['signal_above_vwap'] = above_vwap
                    signal['signal_ema_crossover'] = signal_result.get('ema_crossover', 'NONE')
                    signal['signal_macd_crossover'] = signal_result.get('macd_crossover', 'NONE')
                    signal['signal_vwap_crossover'] = signal_result.get('vwap_crossover', 'NONE')
                    signal['signal_candle_momentum'] = signal_result.get('candle_momentum', 'NEUTRAL')
            else:
                logger.debug(f"SIGNAL CHECK: {symbol} - No signal data available (insufficient history)")

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

        # Apply Warrior Trading filter if enabled (Ross Cameron methodology)
        if signal and getattr(self.config, 'use_warrior_filter', True):
            try:
                from ai.warrior_setup_detector import get_warrior_setup_detector
                detector = get_warrior_setup_detector()
                warrior_signal = detector.analyze(symbol, quote)

                if warrior_signal:
                    grade = warrior_signal.grade
                    min_grade = getattr(self.config, 'warrior_min_grade', 'B')
                    grade_order = {'A': 1, 'B': 2, 'C': 3}

                    # Check grade requirement
                    if grade_order.get(grade, 99) > grade_order.get(min_grade, 2):
                        logger.info(f"WARRIOR REJECT: {symbol} - Grade {grade} < minimum {min_grade}")
                        signal = None
                    else:
                        # Check pattern requirement
                        if getattr(self.config, 'warrior_require_pattern', False):
                            if not warrior_signal.patterns:
                                logger.info(f"WARRIOR REJECT: {symbol} - No confirmed pattern")
                                signal = None

                        # Check tape signal requirement
                        if signal and getattr(self.config, 'warrior_require_tape_signal', False):
                            if not warrior_signal.tape_signals:
                                logger.info(f"WARRIOR REJECT: {symbol} - No tape confirmation")
                                signal = None

                        if signal:
                            logger.info(f"WARRIOR APPROVE: {symbol} - Grade {grade}, Patterns: {len(warrior_signal.patterns)}, Tape: {len(warrior_signal.tape_signals)}")
                            signal['warrior_grade'] = grade
                            signal['warrior_score'] = warrior_signal.score
                            signal['warrior_patterns'] = [p.name for p in warrior_signal.patterns] if warrior_signal.patterns else []
                            signal['warrior_tape_signals'] = warrior_signal.tape_signals
                            signal['warrior_entry'] = warrior_signal.entry_price
                            signal['warrior_stop'] = warrior_signal.stop_loss
                            signal['warrior_target'] = warrior_signal.target_price
                else:
                    logger.debug(f"WARRIOR CHECK: {symbol} - No signal (insufficient data)")
            except Exception as e:
                logger.warning(f"Warrior filter check failed for {symbol}: {e}")

        # Apply Multi-Timeframe Confirmation filter if enabled
        if signal and getattr(self.config, 'use_mtf_filter', True):
            try:
                from ai.mtf_confirmation import get_mtf_engine, MTFSignal

                engine = get_mtf_engine()
                mtf_result = engine.analyze(symbol)

                min_conf = getattr(self.config, 'mtf_min_confidence', 60.0)

                # Check MTF confirmation
                if mtf_result.signal == MTFSignal.CONFIRMED_LONG:
                    logger.info(f"MTF CONFIRMED: {symbol} - {mtf_result.confidence:.0f}% confidence, {', '.join(mtf_result.reasons[:2])}")
                    signal['mtf_signal'] = mtf_result.signal.value
                    signal['mtf_confidence'] = mtf_result.confidence
                    signal['mtf_trend_aligned'] = mtf_result.trend_aligned
                    signal['mtf_vwap_aligned'] = mtf_result.vwap_aligned
                elif mtf_result.signal == MTFSignal.WEAK_LONG:
                    # Weak long - allow if confidence is high enough
                    if mtf_result.confidence >= min_conf:
                        logger.info(f"MTF WEAK LONG: {symbol} - {mtf_result.confidence:.0f}% (>= {min_conf:.0f}%), allowing entry")
                        signal['mtf_signal'] = mtf_result.signal.value
                        signal['mtf_confidence'] = mtf_result.confidence
                    else:
                        logger.info(f"MTF REJECT: {symbol} - Weak long {mtf_result.confidence:.0f}% < {min_conf:.0f}%")
                        signal = None
                elif mtf_result.signal in [MTFSignal.NO_CONFIRMATION, MTFSignal.WEAK_SHORT, MTFSignal.CONFIRMED_SHORT]:
                    logger.info(f"MTF REJECT: {symbol} - {mtf_result.signal.value}, {', '.join(mtf_result.reasons[:2])}")
                    signal = None
                else:
                    # Unknown signal, check confidence
                    if mtf_result.confidence < min_conf:
                        logger.info(f"MTF REJECT: {symbol} - Confidence {mtf_result.confidence:.0f}% < {min_conf:.0f}%")
                        signal = None

                # Additional checks if still have signal
                if signal:
                    require_vwap = getattr(self.config, 'mtf_require_vwap_aligned', True)
                    require_macd = getattr(self.config, 'mtf_require_macd_aligned', True)

                    if require_vwap and not mtf_result.vwap_aligned:
                        logger.info(f"MTF REJECT: {symbol} - VWAP not aligned across timeframes")
                        signal = None
                    elif require_macd and not mtf_result.macd_aligned:
                        logger.info(f"MTF REJECT: {symbol} - MACD not aligned across timeframes")
                        signal = None

            except Exception as e:
                logger.warning(f"MTF filter check failed for {symbol}: {e}")

        # Apply VWAP filter if enabled (Ross Cameron - VWAP is the line in the sand)
        if signal and getattr(self.config, 'use_vwap_filter', True):
            try:
                from ai.vwap_manager import get_vwap_manager

                vwap_manager = get_vwap_manager()
                valid, reason = vwap_manager.is_entry_valid(symbol)

                if valid:
                    vwap_data = vwap_manager.get_vwap(symbol)
                    if vwap_data:
                        # Check if too extended above VWAP
                        max_ext = getattr(self.config, 'vwap_max_extension_pct', 3.0)
                        if vwap_data.distance_pct > max_ext:
                            logger.info(f"VWAP REJECT: {symbol} - Extended {vwap_data.distance_pct:.1f}% > {max_ext}%, wait for pullback")
                            signal = None
                        else:
                            logger.info(f"VWAP APPROVE: {symbol} - {reason}, dist={vwap_data.distance_pct:.1f}%")
                            signal['vwap'] = vwap_data.vwap
                            signal['vwap_distance_pct'] = vwap_data.distance_pct
                            signal['vwap_position'] = vwap_data.position.value
                            signal['vwap_stop'] = vwap_data.stop_price
                    else:
                        logger.debug(f"VWAP: {symbol} - No VWAP data yet, allowing entry")
                else:
                    logger.info(f"VWAP REJECT: {symbol} - {reason}")
                    signal = None

            except Exception as e:
                logger.warning(f"VWAP filter check failed for {symbol}: {e}")

        # Apply Float Rotation boost if enabled (Ross Cameron - volume vs float)
        if signal and getattr(self.config, 'use_float_rotation_boost', True):
            try:
                from ai.float_rotation_tracker import get_float_tracker

                tracker = get_float_tracker()
                data = tracker.get_float_data(symbol)

                if data:
                    # Check float size limits
                    max_float = getattr(self.config, 'max_float_millions', 50.0)
                    if data.float_shares > max_float * 1_000_000:
                        logger.debug(f"FLOAT: {symbol} - Float too large ({data.float_shares/1e6:.1f}M > {max_float}M)")
                        # Don't reject, just skip boost

                    # Check if require low float
                    elif getattr(self.config, 'require_low_float', False) and not data.is_low_float:
                        logger.info(f"FLOAT REJECT: {symbol} - Not a low float stock ({data.float_shares/1e6:.1f}M shares)")
                        signal = None

                    else:
                        # Apply boost based on rotation level
                        boost = tracker.get_rotation_boost(symbol)
                        min_rotation = getattr(self.config, 'min_rotation_for_boost', 0.5)

                        if data.rotation_ratio >= min_rotation and boost > 0:
                            logger.info(
                                f"FLOAT ROTATION BOOST: {symbol} - {data.rotation_ratio:.1f}x rotation, "
                                f"+{boost*100:.0f}% confidence boost, "
                                f"{'LOW FLOAT ' if data.is_low_float else ''}{data.float_shares/1e6:.1f}M float"
                            )
                            signal['float_rotation'] = data.rotation_ratio
                            signal['float_rotation_boost'] = boost
                            signal['float_shares'] = data.float_shares
                            signal['is_low_float'] = data.is_low_float
                            signal['rotation_level'] = data.rotation_level.value

            except Exception as e:
                logger.warning(f"Float rotation check failed for {symbol}: {e}")

        # Apply Level 2 Depth Analysis if enabled
        if signal and getattr(self.config, 'use_depth_analysis', True):
            try:
                from ai.level2_depth_analyzer import get_depth_analyzer

                analyzer = get_depth_analyzer()
                analysis = analyzer.get_analysis(symbol)

                if analysis:
                    # Check entry validity from depth
                    if not analysis.entry_valid:
                        if getattr(self.config, 'depth_block_on_ask_wall', True):
                            logger.info(f"DEPTH REJECT: {symbol} - {analysis.entry_reason}")
                            signal = None

                    # Check for bullish imbalance requirement
                    elif getattr(self.config, 'depth_require_bullish_imbalance', False):
                        min_ratio = getattr(self.config, 'depth_min_imbalance_ratio', 1.2)
                        if analysis.imbalance_ratio < min_ratio:
                            logger.info(
                                f"DEPTH REJECT: {symbol} - Imbalance {analysis.imbalance_ratio:.2f} "
                                f"< {min_ratio} required"
                            )
                            signal = None

                    if signal:
                        # Get boost from depth
                        boost = analyzer.get_entry_boost(symbol)
                        signal['depth_signal'] = analysis.signal.value
                        signal['depth_imbalance'] = analysis.imbalance_ratio
                        signal['depth_boost'] = boost

                        if analysis.suggested_stop:
                            signal['depth_stop'] = analysis.suggested_stop
                        if analysis.suggested_target:
                            signal['depth_target'] = analysis.suggested_target

                        # Log depth info
                        if boost != 0:
                            logger.info(
                                f"DEPTH: {symbol} - {analysis.signal.value}, "
                                f"imbalance {analysis.imbalance_ratio:.2f}, boost {boost:+.0%}"
                            )

            except Exception as e:
                logger.warning(f"Depth analysis check failed for {symbol}: {e}")

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

        # ===== CHRONOS SMART EXIT (Check BEFORE stop loss) =====
        # This prevents stop loss deaths by detecting momentum fading early
        if getattr(self.config, 'use_chronos_exit', True):
            try:
                from ai.chronos_exit_manager import get_chronos_exit_manager
                exit_mgr = get_chronos_exit_manager()

                # Register position if not already tracked
                if symbol not in exit_mgr.positions:
                    exit_mgr.register_position(symbol, trade.entry_price)

                # Check for Chronos exit signal
                chronos_signal = exit_mgr.check_exit(symbol, price, trade.entry_price)

                if chronos_signal.should_exit:
                    exit_signal = {
                        "reason": f"CHRONOS_{chronos_signal.reason}",
                        "pnl_percent": pnl_pct,
                        "price": price,
                        "chronos_urgency": chronos_signal.urgency,
                        "chronos_regime": chronos_signal.regime_after or exit_mgr.positions[symbol].current_regime,
                        "chronos_details": chronos_signal.details
                    }
                    logger.info(
                        f"CHRONOS EXIT: {symbol} - {chronos_signal.reason} "
                        f"({chronos_signal.urgency}) @ {pnl_pct:+.1f}%"
                    )
                    return exit_signal

            except Exception as e:
                logger.debug(f"Chronos exit check failed for {symbol}: {e}")

        # ===== VWAP TRAILING STOP (Ross Cameron - VWAP is support) =====
        # If price breaks below VWAP, momentum is lost
        if getattr(self.config, 'use_vwap_trailing_stop', True):
            try:
                from ai.vwap_manager import get_vwap_manager

                vwap_manager = get_vwap_manager()

                # Create trailing stop if not exists
                if symbol not in vwap_manager.trailing_stops:
                    vwap_manager.create_trailing_stop(symbol, trade.entry_price)
                    # Set offset from config
                    offset = getattr(self.config, 'vwap_stop_offset_pct', 0.3)
                    vwap_manager.trailing_stops[symbol].trail_offset_pct = offset

                # Check VWAP trailing stop
                should_exit, reason = vwap_manager.update_trailing_stop(symbol, price)

                if should_exit:
                    exit_signal = {
                        "reason": "VWAP_STOP",
                        "pnl_percent": pnl_pct,
                        "price": price,
                        "vwap_details": reason,
                        "vwap": vwap_manager.trailing_stops[symbol].current_vwap,
                        "vwap_stop": vwap_manager.trailing_stops[symbol].current_stop
                    }
                    logger.info(f"VWAP EXIT: {symbol} - {reason} @ {pnl_pct:+.1f}%")
                    return exit_signal

            except Exception as e:
                logger.debug(f"VWAP trailing stop check failed for {symbol}: {e}")

        # ===== MOMENTUM EXHAUSTION EXIT (Exit before big drops) =====
        if getattr(self.config, 'use_exhaustion_exit', True):
            try:
                from ai.momentum_exhaustion_detector import get_exhaustion_detector

                detector = get_exhaustion_detector()

                # Register position if not already tracked
                if symbol not in detector.symbols or detector.symbols[symbol].entry_price == 0:
                    detector.register_position(symbol, trade.entry_price)

                # Check exhaustion score
                score, reasons = detector.get_exhaustion_score(symbol)
                threshold = getattr(self.config, 'exhaustion_exit_threshold', 60.0)

                if score >= threshold:
                    exit_signal = {
                        "reason": "EXHAUSTION",
                        "pnl_percent": pnl_pct,
                        "price": price,
                        "exhaustion_score": score,
                        "exhaustion_reasons": reasons
                    }
                    logger.info(
                        f"EXHAUSTION EXIT: {symbol} - Score {score:.0f}% "
                        f"({', '.join(reasons[:2])}) @ {pnl_pct:+.1f}%"
                    )
                    return exit_signal

                # Check for specific exhaustion signals
                state = detector.symbols.get(symbol)
                if state:
                    # Exit on RSI divergence
                    if getattr(self.config, 'exhaustion_exit_on_divergence', True):
                        alert = detector.check_exit(symbol, price)
                        if alert and alert.signal.value == "RSI_DIVERGENCE":
                            exit_signal = {
                                "reason": "EXHAUSTION_RSI_DIVERGENCE",
                                "pnl_percent": pnl_pct,
                                "price": price,
                                "exhaustion_details": alert.details
                            }
                            logger.info(f"EXHAUSTION EXIT: {symbol} - RSI Divergence @ {pnl_pct:+.1f}%")
                            return exit_signal

                    # Exit on consecutive red candles
                    red_threshold = getattr(self.config, 'exhaustion_exit_on_red_candles', 4)
                    if state.consecutive_red_candles >= red_threshold:
                        exit_signal = {
                            "reason": "EXHAUSTION_RED_CANDLES",
                            "pnl_percent": pnl_pct,
                            "price": price,
                            "consecutive_red": state.consecutive_red_candles
                        }
                        logger.info(
                            f"EXHAUSTION EXIT: {symbol} - {state.consecutive_red_candles} "
                            f"red candles @ {pnl_pct:+.1f}%"
                        )
                        return exit_signal

            except Exception as e:
                logger.debug(f"Exhaustion exit check failed for {symbol}: {e}")

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

        # ===== FAILED MOMENTUM EARLY EXIT =====
        # If trade hasn't worked after 45 seconds and is flat/down, exit early
        # Don't wait for MAX_HOLD_TIME - cut losers fast
        failed_momentum_seconds = getattr(self.config, 'failed_momentum_seconds', 45)
        failed_momentum_threshold = getattr(self.config, 'failed_momentum_threshold', 0.5)

        if exit_signal is None and hold_seconds >= failed_momentum_seconds:
            # Exit if: (1) currently flat or down, (2) never reached +1% gain
            if pnl_pct < failed_momentum_threshold and max_gain < 1.0:
                exit_signal = {
                    "reason": "FAILED_MOMENTUM",
                    "hold_seconds": hold_seconds,
                    "pnl_percent": pnl_pct,
                    "max_gain": max_gain,
                    "price": price,
                    "details": f"Flat/down after {hold_seconds:.0f}s, never hit +1%"
                }
                logger.info(f"FAILED_MOMENTUM: {symbol} @ {pnl_pct:+.1f}% after {hold_seconds:.0f}s (max was {max_gain:+.1f}%)")

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

        # Adjust position size based on Warrior grade (A=75%, B=50%, C=25%)
        warrior_grade = signal.get('warrior_grade')
        if warrior_grade and getattr(self.config, 'use_warrior_filter', True):
            grade_multipliers = {'A': 0.75, 'B': 0.50, 'C': 0.25}
            multiplier = grade_multipliers.get(warrior_grade, 0.50)
            original_shares = shares
            shares = max(int(shares * multiplier), self.config.min_shares)
            logger.info(f"WARRIOR SIZING: {symbol} Grade {warrior_grade} = {multiplier*100:.0f}% size ({original_shares} -> {shares} shares)")

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

        # Override with Warrior levels if provided (from setup detector analysis)
        warrior_stop = signal.get('warrior_stop')
        warrior_target = signal.get('warrior_target')
        if warrior_stop and warrior_target and warrior_stop > 0 and warrior_target > 0:
            logger.info(f"WARRIOR LEVELS: {symbol} using Warrior stop=${warrior_stop:.2f}, target=${warrior_target:.2f}")
            atr_stop_price = warrior_stop
            atr_target_price = warrior_target
            # Calculate trailing from distance to stop
            atr_trail_distance = (price - warrior_stop) * 0.5  # Trail at 50% of stop distance

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
            volatility_regime=volatility_regime,
            # Technical signal state at entry (for correlation analysis)
            signal_confluence=signal.get('signal_confluence', 0.0),
            signal_bias=signal.get('signal_bias', ''),
            signal_ema_bullish=signal.get('signal_ema_bullish', False),
            signal_macd_bullish=signal.get('signal_macd_bullish', False),
            signal_above_vwap=signal.get('signal_above_vwap', False),
            signal_ema_crossover=signal.get('signal_ema_crossover', ''),
            signal_macd_crossover=signal.get('signal_macd_crossover', ''),
            signal_vwap_crossover=signal.get('signal_vwap_crossover', ''),
            signal_candle_momentum=signal.get('signal_candle_momentum', ''),
            # Warrior Trading state at entry
            warrior_grade=signal.get('warrior_grade', ''),
            warrior_score=signal.get('warrior_score', 0.0),
            warrior_patterns=signal.get('warrior_patterns', []),
            warrior_tape_signals=signal.get('warrior_tape_signals', [])
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

        # Unregister from Chronos exit manager
        try:
            from ai.chronos_exit_manager import get_chronos_exit_manager
            exit_mgr = get_chronos_exit_manager()
            exit_mgr.unregister_position(symbol)
        except Exception:
            pass

        pnl_emoji = "WIN" if trade.pnl > 0 else "LOSS"
        logger.info(
            f"EXIT [{pnl_emoji}]: {symbol} @ ${price:.2f} | "
            f"P/L: ${trade.pnl:.2f} ({trade.pnl_percent:+.1f}%) | "
            f"Reason: {trade.exit_reason} | "
            f"Hold: {trade.hold_time_seconds}s"
        )

        # Register profitable exits for Phase 2 continuation monitoring
        if trade.pnl_percent >= 2.0:  # Only if +2% or more
            try:
                from .phase2_manager import register_phase1_exit
                phase2_result = register_phase1_exit(symbol, price, trade.pnl_percent)
                if phase2_result.get("action") == "WATCHING":
                    logger.info(f"[PHASE2] {symbol} registered for continuation monitoring")
            except Exception as e:
                logger.debug(f"Phase 2 registration skipped: {e}")

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
