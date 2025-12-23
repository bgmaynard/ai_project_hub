"""
Intelligent Watchlist Manager
============================
Complete workflow from news detection to trade execution with whipsaw protection.

Workflow:
1. News detected -> Check price/volume -> Qualify
2. If qualified -> Add to watchlist
3. Watchlist triggers -> Training, Backtesting, Prediction
4. If meets HFT requirements -> Execute trade
5. Monitor all day -> Look for spike re-entry opportunities
6. Whipsaw protection -> Remove stocks on jacknife/bad reversal

This is the brain that connects all our trading modules.
"""

import requests
import logging
import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pytz

logger = logging.getLogger(__name__)


class StockStatus(Enum):
    """Status of a stock in the watchlist"""
    QUALIFYING = "qualifying"      # Just detected, being evaluated
    WATCHLIST = "watchlist"        # Passed qualification, on active watch
    TRAINING = "training"          # ML model being trained
    READY = "ready"                # Trained and ready to trade
    IN_POSITION = "in_position"    # Currently holding
    COOLDOWN = "cooldown"          # Recently exited, avoiding whipsaw
    REMOVED = "removed"            # Jacknife/reversal - don't trade


@dataclass
class WatchlistEntry:
    """A stock on the intelligent watchlist"""
    symbol: str
    status: StockStatus

    # Discovery info
    discovery_time: datetime
    discovery_source: str  # "news", "scanner", "manual"
    catalyst: str = ""
    headline: str = ""

    # Price tracking
    discovery_price: float = 0.0
    high_of_day: float = 0.0
    low_of_day: float = 0.0
    current_price: float = 0.0
    vwap: float = 0.0

    # Volume tracking
    discovery_volume: int = 0
    current_volume: int = 0
    avg_volume: int = 0
    volume_ratio: float = 0.0

    # Spike tracking
    spike_count: int = 0
    last_spike_time: Optional[datetime] = None
    spike_history: List[Dict] = field(default_factory=list)

    # Trade tracking
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    position_size: int = 0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    trade_count: int = 0

    # ML/Prediction
    prediction_score: float = 0.0
    prediction_direction: str = ""  # "bullish", "bearish", "neutral"
    model_trained: bool = False
    backtest_win_rate: float = 0.0

    # Risk tracking
    jacknife_count: int = 0
    reversal_count: int = 0
    whipsaw_risk: float = 0.0

    # Timestamps
    last_update: datetime = None
    cooldown_until: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "status": self.status.value,
            "discovery_time": self.discovery_time.isoformat(),
            "discovery_source": self.discovery_source,
            "catalyst": self.catalyst,
            "headline": self.headline[:50] if self.headline else "",
            "discovery_price": f"${self.discovery_price:.2f}",
            "current_price": f"${self.current_price:.2f}",
            "high_of_day": f"${self.high_of_day:.2f}",
            "low_of_day": f"${self.low_of_day:.2f}",
            "volume_ratio": f"{self.volume_ratio:.1f}x",
            "spike_count": self.spike_count,
            "prediction_score": f"{self.prediction_score:.0%}",
            "prediction_direction": self.prediction_direction,
            "model_trained": self.model_trained,
            "backtest_win_rate": f"{self.backtest_win_rate:.0%}",
            "whipsaw_risk": f"{self.whipsaw_risk:.0%}",
            "in_position": self.status == StockStatus.IN_POSITION,
            "position_size": self.position_size,
            "unrealized_pnl": f"${self.unrealized_pnl:.2f}",
            "trade_count": self.trade_count
        }


@dataclass
class JacknifeAlert:
    """Alert when stock jacknifes (sharp reversal)"""
    symbol: str
    alert_type: str  # "jacknife", "reversal", "breakdown", "volume_dry"
    severity: str    # "warning", "danger", "critical"
    message: str
    price_before: float
    price_after: float
    drop_pct: float
    time_seconds: int
    timestamp: datetime
    action_taken: str = ""


class IntelligentWatchlistManager:
    """
    Complete trading workflow manager.
    Connects news -> qualification -> watchlist -> training -> trading -> monitoring
    """

    def __init__(self, api_url: str = "http://localhost:9100/api/alpaca"):
        self.api_url = api_url
        self.et_tz = pytz.timezone('US/Eastern')

        # Watchlist storage
        self.watchlist: Dict[str, WatchlistEntry] = {}
        self.max_watchlist_size = 50

        # Qualification thresholds
        self.min_price = 1.00   # Warrior method: $1 minimum
        self.max_price = 20.00  # Warrior method: $20 maximum
        self.min_volume_ratio = 1.5  # 1.5x average volume
        self.min_spike_pct = 2.0     # 2% move to qualify

        # Jacknife / Reversal Detection
        self.jacknife_drop_pct = 5.0       # 5% drop = jacknife
        self.jacknife_time_seconds = 60    # Within 60 seconds
        self.reversal_threshold = 3.0      # 3% reversal from high
        self.max_jacknife_count = 2        # Remove after 2 jacknifes

        # Cooldown settings
        self.cooldown_minutes = 15         # 15 min cooldown after whipsaw

        # Spike re-entry
        self.spike_reentry_pct = 2.0       # 2% spike for re-entry consideration
        self.max_trades_per_stock = 5      # Max 5 trades per stock per day

        # Position limits
        self.max_position_value = 1000
        self.max_total_positions = 10

        # Price history for jacknife detection
        self.price_history: Dict[str, List[Dict]] = {}
        self.price_history_max = 120  # Keep 2 minutes @ 1s updates

        # Callbacks
        self.on_stock_qualified: Optional[Callable] = None
        self.on_jacknife_detected: Optional[Callable] = None
        self.on_spike_reentry: Optional[Callable] = None
        self.on_trade_signal: Optional[Callable] = None

        # Alert history
        self.alerts: List[JacknifeAlert] = []
        self.max_alerts = 100

        # Control
        self.is_running = False
        self._monitor_thread = None
        self._monitor_interval = 1.0  # 1 second

        # Load persisted watchlist
        self._load_watchlist()

        logger.info("IntelligentWatchlistManager initialized")

    def _load_watchlist(self):
        """Load watchlist from disk"""
        try:
            path = Path(__file__).parent.parent / "store" / "intelligent_watchlist.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                    # Only load today's data
                    today = datetime.now(self.et_tz).date()
                    for entry_data in data.get("entries", []):
                        disc_time = datetime.fromisoformat(entry_data["discovery_time"])
                        if disc_time.date() == today:
                            entry = WatchlistEntry(
                                symbol=entry_data["symbol"],
                                status=StockStatus(entry_data["status"]),
                                discovery_time=disc_time,
                                discovery_source=entry_data["discovery_source"],
                                catalyst=entry_data.get("catalyst", ""),
                                headline=entry_data.get("headline", ""),
                                discovery_price=entry_data.get("discovery_price", 0),
                                high_of_day=entry_data.get("high_of_day", 0),
                                spike_count=entry_data.get("spike_count", 0),
                                model_trained=entry_data.get("model_trained", False)
                            )
                            self.watchlist[entry.symbol] = entry
                    logger.info(f"Loaded {len(self.watchlist)} stocks from watchlist")
        except Exception as e:
            logger.warning(f"Could not load watchlist: {e}")

    def _save_watchlist(self):
        """Save watchlist to disk"""
        try:
            path = Path(__file__).parent.parent / "store" / "intelligent_watchlist.json"
            path.parent.mkdir(exist_ok=True)

            entries = []
            for sym, entry in self.watchlist.items():
                entries.append({
                    "symbol": entry.symbol,
                    "status": entry.status.value,
                    "discovery_time": entry.discovery_time.isoformat(),
                    "discovery_source": entry.discovery_source,
                    "catalyst": entry.catalyst,
                    "headline": entry.headline,
                    "discovery_price": entry.discovery_price,
                    "high_of_day": entry.high_of_day,
                    "spike_count": entry.spike_count,
                    "model_trained": entry.model_trained
                })

            with open(path, 'w') as f:
                json.dump({"entries": entries, "saved_at": datetime.now().isoformat()}, f)

        except Exception as e:
            logger.error(f"Could not save watchlist: {e}")

    def start(self):
        """Start the watchlist manager"""
        self.is_running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("IntelligentWatchlistManager STARTED")

    def stop(self):
        """Stop the manager"""
        self.is_running = False
        self._save_watchlist()
        logger.info("IntelligentWatchlistManager STOPPED")

    # =========================================================================
    # QUALIFICATION
    # =========================================================================

    def qualify_stock(self, symbol: str, source: str = "manual",
                     catalyst: str = "", headline: str = "") -> bool:
        """
        Qualify a stock for the watchlist.
        Checks price, volume, and momentum.
        """
        try:
            # Get quote
            r = requests.get(f"{self.api_url}/quote/{symbol}", timeout=2)
            if r.status_code != 200:
                return False

            q = r.json()
            price = float(q.get('last', 0)) or float(q.get('ask', 0))
            volume = int(q.get('volume', 0) or 0)

            if price <= 0:
                return False

            # Price range check
            if price < self.min_price or price > self.max_price:
                logger.info(f"{symbol} failed: price ${price:.2f} out of range")
                return False

            # Volume check (if we have avg volume)
            avg_vol = int(q.get('avg_volume', volume) or volume)
            volume_ratio = volume / avg_vol if avg_vol > 0 else 1.0

            if volume_ratio < self.min_volume_ratio:
                logger.info(f"{symbol} failed: volume ratio {volume_ratio:.1f}x < {self.min_volume_ratio}x")
                return False

            # Passed qualification!
            now = datetime.now(self.et_tz)

            entry = WatchlistEntry(
                symbol=symbol,
                status=StockStatus.WATCHLIST,
                discovery_time=now,
                discovery_source=source,
                catalyst=catalyst,
                headline=headline,
                discovery_price=price,
                high_of_day=price,
                low_of_day=price,
                current_price=price,
                discovery_volume=volume,
                current_volume=volume,
                avg_volume=avg_vol,
                volume_ratio=volume_ratio,
                last_update=now
            )

            self.watchlist[symbol] = entry
            self._save_watchlist()

            logger.warning(f"QUALIFIED: {symbol} @ ${price:.2f} | Vol: {volume_ratio:.1f}x | Source: {source}")

            if self.on_stock_qualified:
                self.on_stock_qualified(entry)

            # Trigger training
            self._trigger_training(symbol)

            return True

        except Exception as e:
            logger.error(f"Error qualifying {symbol}: {e}")
            return False

    def add_from_news(self, symbol: str, headline: str, catalyst: str,
                     confidence: float, validation_score: int) -> bool:
        """Add a stock from news detection pipeline"""

        # Skip if already on watchlist
        if symbol in self.watchlist:
            entry = self.watchlist[symbol]
            if entry.status == StockStatus.REMOVED:
                logger.info(f"{symbol} was removed (whipsaw) - skipping")
                return False
            if entry.status == StockStatus.COOLDOWN:
                logger.info(f"{symbol} in cooldown - skipping")
                return False
            # Update existing entry
            entry.spike_count += 1
            entry.last_spike_time = datetime.now(self.et_tz)
            return True

        # Check validation score
        if validation_score < 60:
            logger.info(f"{symbol} failed: validation score {validation_score} < 60")
            return False

        # Check confidence
        if confidence < 0.6:
            logger.info(f"{symbol} failed: confidence {confidence:.0%} < 60%")
            return False

        return self.qualify_stock(symbol, "news", catalyst, headline)

    # =========================================================================
    # TRAINING / BACKTESTING / PREDICTION
    # =========================================================================

    def _trigger_training(self, symbol: str):
        """Trigger ML training for a symbol"""
        try:
            entry = self.watchlist.get(symbol)
            if not entry:
                return

            entry.status = StockStatus.TRAINING

            # Call training API
            logger.info(f"Triggering training for {symbol}...")

            threading.Thread(
                target=self._do_training,
                args=(symbol,),
                daemon=True
            ).start()

        except Exception as e:
            logger.error(f"Error triggering training for {symbol}: {e}")

    def _do_training(self, symbol: str):
        """Actually train the model (runs in background)"""
        try:
            # Call the AI predictor to train
            r = requests.post(
                "http://localhost:9100/api/ai/train",
                json={"symbol": symbol, "quick": True},
                timeout=120
            )

            entry = self.watchlist.get(symbol)
            if not entry:
                return

            if r.status_code == 200:
                result = r.json()
                entry.model_trained = True
                entry.status = StockStatus.READY

                # Try to get backtest results
                try:
                    bt = requests.get(
                        f"http://localhost:9100/api/ai/backtest/{symbol}",
                        timeout=30
                    )
                    if bt.status_code == 200:
                        bt_data = bt.json()
                        entry.backtest_win_rate = bt_data.get("win_rate", 0)
                except:
                    pass

                logger.info(f"Training complete for {symbol} - win rate: {entry.backtest_win_rate:.0%}")

                # Get prediction
                self._get_prediction(symbol)
            else:
                entry.status = StockStatus.WATCHLIST
                logger.warning(f"Training failed for {symbol}")

        except Exception as e:
            logger.error(f"Error training {symbol}: {e}")
            if symbol in self.watchlist:
                self.watchlist[symbol].status = StockStatus.WATCHLIST

    def _get_prediction(self, symbol: str):
        """Get prediction for a symbol"""
        try:
            r = requests.get(
                f"http://localhost:9100/api/ai/predict/{symbol}",
                timeout=10
            )

            entry = self.watchlist.get(symbol)
            if not entry:
                return

            if r.status_code == 200:
                data = r.json()
                entry.prediction_score = data.get("confidence", 0)

                direction = data.get("signal", "HOLD")
                if direction == "BUY":
                    entry.prediction_direction = "bullish"
                elif direction == "SELL":
                    entry.prediction_direction = "bearish"
                else:
                    entry.prediction_direction = "neutral"

                logger.info(
                    f"Prediction for {symbol}: {entry.prediction_direction} "
                    f"({entry.prediction_score:.0%} confidence)"
                )

        except Exception as e:
            logger.error(f"Error getting prediction for {symbol}: {e}")

    # =========================================================================
    # MONITORING - Jacknife / Reversal Detection
    # =========================================================================

    def _monitor_loop(self):
        """Main monitoring loop - runs every second"""
        while self.is_running:
            try:
                self._update_all_prices()
                self._check_for_jacknifes()
                self._check_for_spike_reentry()
                self._check_cooldowns()
            except Exception as e:
                logger.error(f"Monitor error: {e}")

            time.sleep(self._monitor_interval)

    def _update_all_prices(self):
        """Update prices for all watchlist stocks"""
        now = datetime.now(self.et_tz)

        for symbol, entry in list(self.watchlist.items()):
            if entry.status == StockStatus.REMOVED:
                continue

            try:
                r = requests.get(f"{self.api_url}/quote/{symbol}", timeout=1)
                if r.status_code != 200:
                    continue

                q = r.json()
                price = float(q.get('last', 0)) or float(q.get('ask', 0))
                volume = int(q.get('volume', 0) or 0)

                if price <= 0:
                    continue

                # Update entry
                entry.current_price = price
                entry.current_volume = volume
                entry.last_update = now

                # Update high/low of day
                if price > entry.high_of_day:
                    entry.high_of_day = price
                if entry.low_of_day == 0 or price < entry.low_of_day:
                    entry.low_of_day = price

                # Update volume ratio
                if entry.avg_volume > 0:
                    entry.volume_ratio = volume / entry.avg_volume

                # Store price history for jacknife detection
                if symbol not in self.price_history:
                    self.price_history[symbol] = []

                self.price_history[symbol].append({
                    "price": price,
                    "time": now,
                    "volume": volume
                })

                # Trim history
                if len(self.price_history[symbol]) > self.price_history_max:
                    self.price_history[symbol] = self.price_history[symbol][-self.price_history_max:]

                # Update position P/L if in position
                if entry.status == StockStatus.IN_POSITION and entry.entry_price > 0:
                    entry.unrealized_pnl = (price - entry.entry_price) * entry.position_size

            except Exception as e:
                pass  # Ignore individual quote failures

    def _check_for_jacknifes(self):
        """Check for jacknife (sharp reversal) patterns"""
        now = datetime.now(self.et_tz)

        for symbol, entry in list(self.watchlist.items()):
            if entry.status in [StockStatus.REMOVED, StockStatus.COOLDOWN]:
                continue

            history = self.price_history.get(symbol, [])
            if len(history) < 5:
                continue

            current = history[-1]
            current_price = current["price"]

            # Look back for jacknife (big drop in short time)
            for i in range(min(self.jacknife_time_seconds, len(history))):
                past = history[-(i+1)]
                past_price = past["price"]
                past_time = past["time"]

                time_diff = (now - past_time).total_seconds()
                if time_diff > self.jacknife_time_seconds:
                    break

                drop_pct = (past_price - current_price) / past_price * 100

                # Jacknife detected!
                if drop_pct >= self.jacknife_drop_pct:
                    entry.jacknife_count += 1

                    alert = JacknifeAlert(
                        symbol=symbol,
                        alert_type="jacknife",
                        severity="danger" if drop_pct >= 7 else "warning",
                        message=f"{symbol} jacknifed {drop_pct:.1f}% in {time_diff:.0f}s",
                        price_before=past_price,
                        price_after=current_price,
                        drop_pct=drop_pct,
                        time_seconds=int(time_diff),
                        timestamp=now
                    )

                    self._handle_jacknife(entry, alert)
                    break

            # Check for reversal from high
            if entry.high_of_day > 0:
                reversal_pct = (entry.high_of_day - current_price) / entry.high_of_day * 100

                if reversal_pct >= self.reversal_threshold:
                    entry.reversal_count += 1

                    # Only alert on first reversal
                    if entry.reversal_count == 1:
                        alert = JacknifeAlert(
                            symbol=symbol,
                            alert_type="reversal",
                            severity="warning",
                            message=f"{symbol} reversed {reversal_pct:.1f}% from high ${entry.high_of_day:.2f}",
                            price_before=entry.high_of_day,
                            price_after=current_price,
                            drop_pct=reversal_pct,
                            time_seconds=0,
                            timestamp=now
                        )

                        self.alerts.append(alert)
                        if len(self.alerts) > self.max_alerts:
                            self.alerts = self.alerts[-self.max_alerts:]

                        logger.warning(f"REVERSAL: {alert.message}")

            # Calculate whipsaw risk
            entry.whipsaw_risk = min(1.0,
                (entry.jacknife_count * 0.3) +
                (entry.reversal_count * 0.1)
            )

    def _handle_jacknife(self, entry: WatchlistEntry, alert: JacknifeAlert):
        """Handle a jacknife detection"""
        self.alerts.append(alert)
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]

        logger.warning(f"JACKNIFE: {alert.message}")

        if self.on_jacknife_detected:
            self.on_jacknife_detected(alert)

        # Auto-exit position if in one
        if entry.status == StockStatus.IN_POSITION:
            logger.warning(f"Auto-exiting {entry.symbol} due to jacknife!")
            self._emergency_exit(entry, "jacknife")
            alert.action_taken = "emergency_exit"

        # Remove from watchlist if too many jacknifes
        if entry.jacknife_count >= self.max_jacknife_count:
            logger.warning(f"REMOVING {entry.symbol} - too many jacknifes ({entry.jacknife_count})")
            entry.status = StockStatus.REMOVED
            alert.action_taken = "removed"
            self._save_watchlist()

    def _emergency_exit(self, entry: WatchlistEntry, reason: str):
        """Emergency exit a position"""
        try:
            if entry.position_size <= 0:
                return

            order = {
                'symbol': entry.symbol,
                'quantity': entry.position_size,
                'action': 'sell',
                'order_type': 'market',
                'time_in_force': 'day'
            }

            r = requests.post(f"{self.api_url}/place-order", json=order, timeout=5)
            result = r.json()

            if result.get('success'):
                logger.warning(f"EMERGENCY EXIT: Sold {entry.position_size} {entry.symbol} - reason: {reason}")
                entry.status = StockStatus.COOLDOWN
                entry.cooldown_until = datetime.now(self.et_tz) + timedelta(minutes=self.cooldown_minutes)
                entry.position_size = 0
            else:
                logger.error(f"Emergency exit failed for {entry.symbol}: {result}")

        except Exception as e:
            logger.error(f"Emergency exit error for {entry.symbol}: {e}")

    def _check_for_spike_reentry(self):
        """Check for spike re-entry opportunities"""
        now = datetime.now(self.et_tz)

        for symbol, entry in list(self.watchlist.items()):
            # Only check ready stocks not in position
            if entry.status not in [StockStatus.READY, StockStatus.WATCHLIST]:
                continue

            # Check trade limit
            if entry.trade_count >= self.max_trades_per_stock:
                continue

            # Need price history
            history = self.price_history.get(symbol, [])
            if len(history) < 10:
                continue

            # Look for fresh spike
            prices = [h["price"] for h in history[-10:]]
            recent_low = min(prices[:-3])  # Exclude last 3
            current = prices[-1]

            spike_pct = (current - recent_low) / recent_low * 100 if recent_low > 0 else 0

            if spike_pct >= self.spike_reentry_pct:
                entry.spike_count += 1
                entry.last_spike_time = now

                entry.spike_history.append({
                    "time": now.isoformat(),
                    "low": recent_low,
                    "high": current,
                    "spike_pct": spike_pct
                })

                logger.info(f"SPIKE RE-ENTRY opportunity: {symbol} +{spike_pct:.1f}%")

                if self.on_spike_reentry:
                    self.on_spike_reentry(entry, spike_pct)

                # Generate trade signal if conditions met
                if entry.model_trained and entry.prediction_direction == "bullish":
                    if entry.whipsaw_risk < 0.5:
                        self._generate_trade_signal(entry, "spike_reentry")

    def _check_cooldowns(self):
        """Check and clear cooldowns"""
        now = datetime.now(self.et_tz)

        for symbol, entry in self.watchlist.items():
            if entry.status == StockStatus.COOLDOWN:
                if entry.cooldown_until and now >= entry.cooldown_until:
                    entry.status = StockStatus.WATCHLIST
                    entry.cooldown_until = None
                    logger.info(f"{symbol} cooldown ended")

    # =========================================================================
    # TRADE SIGNALS
    # =========================================================================

    def _generate_trade_signal(self, entry: WatchlistEntry, trigger: str):
        """Generate a trade signal"""
        signal = {
            "symbol": entry.symbol,
            "action": "BUY",
            "trigger": trigger,
            "price": entry.current_price,
            "prediction_score": entry.prediction_score,
            "prediction_direction": entry.prediction_direction,
            "whipsaw_risk": entry.whipsaw_risk,
            "spike_count": entry.spike_count,
            "timestamp": datetime.now(self.et_tz).isoformat()
        }

        logger.warning(f"TRADE SIGNAL: {signal}")

        if self.on_trade_signal:
            self.on_trade_signal(signal)

    def execute_trade(self, symbol: str, action: str = "buy") -> bool:
        """Execute a trade"""
        entry = self.watchlist.get(symbol)
        if not entry:
            return False

        try:
            # Calculate position size
            size = max(1, int(self.max_position_value / entry.current_price))

            order = {
                'symbol': symbol,
                'quantity': size,
                'action': action,
                'order_type': 'market',
                'time_in_force': 'day',
                'extended_hours': True
            }

            r = requests.post(f"{self.api_url}/place-order", json=order, timeout=5)
            result = r.json()

            if result.get('success'):
                entry.status = StockStatus.IN_POSITION
                entry.entry_price = entry.current_price
                entry.entry_time = datetime.now(self.et_tz)
                entry.position_size = size
                entry.trade_count += 1

                logger.warning(f"TRADE EXECUTED: {action.upper()} {size} {symbol} @ ${entry.current_price:.2f}")
                return True
            else:
                logger.error(f"Trade failed: {result}")
                return False

        except Exception as e:
            logger.error(f"Trade error: {e}")
            return False

    # =========================================================================
    # API
    # =========================================================================

    def get_watchlist(self, status: str = None) -> List[Dict]:
        """Get watchlist entries"""
        entries = []
        for sym, entry in self.watchlist.items():
            if status and entry.status.value != status:
                continue
            entries.append(entry.to_dict())
        return entries

    def get_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alerts"""
        alerts = self.alerts[-limit:]
        return [{
            "symbol": a.symbol,
            "type": a.alert_type,
            "severity": a.severity,
            "message": a.message,
            "drop_pct": f"{a.drop_pct:.1f}%",
            "action": a.action_taken,
            "time": a.timestamp.isoformat()
        } for a in alerts]

    def get_status(self) -> Dict:
        """Get manager status"""
        status_counts = {}
        for entry in self.watchlist.values():
            s = entry.status.value
            status_counts[s] = status_counts.get(s, 0) + 1

        return {
            "is_running": self.is_running,
            "total_stocks": len(self.watchlist),
            "status_breakdown": status_counts,
            "alert_count": len(self.alerts),
            "max_watchlist_size": self.max_watchlist_size
        }

    def remove_stock(self, symbol: str, reason: str = "manual"):
        """Manually remove a stock"""
        if symbol in self.watchlist:
            self.watchlist[symbol].status = StockStatus.REMOVED
            logger.info(f"Removed {symbol}: {reason}")
            self._save_watchlist()


# Singleton
_manager: Optional[IntelligentWatchlistManager] = None


def get_intelligent_watchlist() -> IntelligentWatchlistManager:
    """Get or create the intelligent watchlist manager"""
    global _manager
    if _manager is None:
        _manager = IntelligentWatchlistManager()
    return _manager


def start_intelligent_watchlist(
    on_qualified: Callable = None,
    on_jacknife: Callable = None,
    on_spike: Callable = None,
    on_signal: Callable = None
) -> IntelligentWatchlistManager:
    """Start the intelligent watchlist"""
    manager = get_intelligent_watchlist()

    if on_qualified:
        manager.on_stock_qualified = on_qualified
    if on_jacknife:
        manager.on_jacknife_detected = on_jacknife
    if on_spike:
        manager.on_spike_reentry = on_spike
    if on_signal:
        manager.on_trade_signal = on_signal

    manager.start()
    return manager


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')

    def on_qualified(entry):
        print(f"\n*** QUALIFIED: {entry.symbol} @ ${entry.current_price:.2f} ***")

    def on_jacknife(alert):
        print(f"\n*** JACKNIFE: {alert.message} ***")

    def on_spike(entry, pct):
        print(f"\n*** SPIKE RE-ENTRY: {entry.symbol} +{pct:.1f}% ***")

    def on_signal(signal):
        print(f"\n*** TRADE SIGNAL: {signal} ***")

    manager = start_intelligent_watchlist(
        on_qualified=on_qualified,
        on_jacknife=on_jacknife,
        on_spike=on_spike,
        on_signal=on_signal
    )

    # Test qualification
    test_symbols = ['SOFI', 'PLTR', 'NIO', 'LCID']
    for sym in test_symbols:
        manager.qualify_stock(sym, "test")

    print(f"\nWatchlist: {len(manager.watchlist)} stocks")
    print("Monitoring for jacknifes and spike re-entries...")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(5)
            status = manager.get_status()
            print(f"\rStatus: {status}", end="")
    except KeyboardInterrupt:
        manager.stop()
        print("\nStopped")
