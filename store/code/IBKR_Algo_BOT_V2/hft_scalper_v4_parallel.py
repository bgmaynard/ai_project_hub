"""
HFT Scalper V4 - Parallel Multi-Threaded Architecture
======================================================
Optimized for 24-core i9-13900K with 64GB RAM

Architecture:
1. Quote Fetcher Pool - 8 threads fetching quotes in parallel
2. Signal Processor Pool - 8 threads analyzing signals
3. Order Executor Pool - 4 threads placing orders
4. Position Monitor - 2 threads tracking positions
5. Velocity Engine - 2 threads updating price predictions

Speed improvements over V3:
- Serial: 6 symbols × 50ms each = 300ms per cycle
- Parallel: 6 symbols / 8 threads = ~50ms per cycle (6x faster)

With 24 cores, we can monitor 50+ symbols simultaneously.
"""

import logging
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from queue import Empty, Queue
from typing import Dict, List, Optional

import pytz
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import (MarketOrderRequest, StopLossRequest,
                                     TakeProfitRequest)
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(threadName)-12s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """Thread-safe quote data"""

    symbol: str
    bid: float
    ask: float
    last: float
    timestamp: float


@dataclass
class Signal:
    """Trading signal from analysis"""

    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    price: float
    momentum: float
    acceleration: float
    velocity: float
    speed_class: str
    confidence: float


# Configuration optimized for 24-core system
CONFIG = {
    # Thread pool sizes (total: 24 threads to match cores)
    "QUOTE_THREADS": 8,  # Parallel quote fetching
    "SIGNAL_THREADS": 8,  # Parallel signal processing
    "ORDER_THREADS": 4,  # Order execution
    "MONITOR_THREADS": 2,  # Position monitoring
    "VELOCITY_THREADS": 2,  # Price prediction
    # EOD Liquidation for Cash Accounts (T+1 settlement)
    # Sell all positions before market close to free up cash for next day
    # NOTE: Account type is AUTO-DETECTED from Alpaca - no manual config needed
    "EOD_LIQUIDATE_TIME": "15:45",  # 3:45 PM EST (15 min before close)
    "EOD_NO_NEW_ENTRIES_TIME": "15:30",  # No new entries after 3:30 PM EST
    "PDT_THRESHOLD": 25000,  # $25k PDT threshold
    # Risk management - 2:1 ratio
    "STOP_LOSS_PCT": 0.015,
    "TAKE_PROFIT_PCT": 0.03,
    "MIN_STOP_CENTS": 0.02,
    "MIN_PROFIT_CENTS": 0.04,
    "LOW_PRICE_THRESHOLD": 1.00,
    "PENNY_THRESHOLD": 0.50,
    # Entry requirements
    "MIN_MOMENTUM": 0.002,
    "MIN_ACCELERATION": 0.001,
    # Position sizing
    "POSITION_SIZE": 500,
    "MAX_POSITIONS": 8,  # More positions with parallel execution
    # Timing
    "QUOTE_INTERVAL": 0.05,  # 50ms between quote batches
    "SIGNAL_INTERVAL": 0.025,  # 25ms signal processing
    # Expanded symbol list - can handle many more with parallel execution
    "SYMBOLS": [
        # Momentum small caps
        "AG",
        "HL",
        "KGC",
        "EDIT",
        "GOLD",
        "SLV",
        # Add more symbols - parallel execution can handle 50+
        "SNDL",
        "PLUG",
        "FCEL",
        "RIOT",
        "MARA",
        "AMC",
        "BB",
        "NOK",
        "SOFI",
        "PLTR",
    ],
}


class ThreadSafeVelocityTracker:
    """Thread-safe velocity tracking with lock-free reads"""

    def __init__(self):
        self._data = {}  # symbol -> {'prices': [], 'times': [], 'velocity': 0}
        self._locks = defaultdict(threading.Lock)
        self.fill_latency = 0.15

    def update(self, symbol: str, price: float):
        """Update price data (thread-safe)"""
        now = time.time()
        with self._locks[symbol]:
            if symbol not in self._data:
                self._data[symbol] = {
                    "prices": [],
                    "times": [],
                    "velocity": 0,
                    "accel": 0,
                }

            data = self._data[symbol]
            data["prices"].append(price)
            data["times"].append(now)

            # Keep last 20 points
            if len(data["prices"]) > 20:
                data["prices"] = data["prices"][-20:]
                data["times"] = data["times"][-20:]

            # Calculate velocity
            if len(data["prices"]) >= 2:
                dt = data["times"][-1] - data["times"][-2]
                if dt > 0:
                    dp = (data["prices"][-1] - data["prices"][-2]) / data["prices"][-2]
                    data["velocity"] = dp / dt

            # Calculate acceleration
            if len(data["prices"]) >= 4:
                mid = len(data["prices"]) // 2
                v1 = self._calc_velocity(data["prices"][:mid], data["times"][:mid])
                v2 = self._calc_velocity(data["prices"][mid:], data["times"][mid:])
                data["accel"] = v2 - v1

    def _calc_velocity(self, prices, times):
        if len(prices) < 2 or times[-1] - times[0] <= 0:
            return 0
        return ((prices[-1] - prices[0]) / prices[0]) / (times[-1] - times[0])

    def get_velocity(self, symbol: str) -> float:
        """Get velocity (lock-free read)"""
        data = self._data.get(symbol, {})
        return data.get("velocity", 0)

    def get_acceleration(self, symbol: str) -> float:
        """Get acceleration (lock-free read)"""
        data = self._data.get(symbol, {})
        return data.get("accel", 0)

    def get_speed_class(self, symbol: str) -> str:
        """Classify speed"""
        v = abs(self.get_velocity(symbol))
        if v < 0.0001:
            return "SLOW"
        elif v < 0.0005:
            return "NORMAL"
        elif v < 0.002:
            return "FAST"
        return "EXTREME"

    def predict_fill(
        self, symbol: str, current_price: float, lookahead_ms: int = 150
    ) -> float:
        """Predict fill price"""
        v = self.get_velocity(symbol)
        a = self.get_acceleration(symbol)
        t = lookahead_ms / 1000.0
        change = v * t + 0.5 * a * t * t
        return current_price * (1 + change)


class ParallelHFTScalper:
    """Multi-threaded HFT Scalper for 24-core systems"""

    def __init__(self):
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        # Alpaca clients (thread-safe)
        self.trading = TradingClient(api_key, secret_key, paper=True)
        self.data = StockHistoricalDataClient(api_key, secret_key)

        # Thread-safe state
        self.velocity = ThreadSafeVelocityTracker()
        self.positions = {}  # Guarded by positions_lock
        self.positions_lock = threading.Lock()
        self.blocked_symbols = {}
        self.blocked_lock = threading.Lock()

        # Queues for pipeline
        self.quote_queue = Queue(maxsize=1000)
        self.signal_queue = Queue(maxsize=1000)
        self.order_queue = Queue(maxsize=100)

        # Price history for signals
        self.price_history = defaultdict(list)
        self.history_lock = threading.Lock()

        # Thread pools
        self.quote_pool = ThreadPoolExecutor(
            max_workers=CONFIG["QUOTE_THREADS"], thread_name_prefix="Quote"
        )
        self.signal_pool = ThreadPoolExecutor(
            max_workers=CONFIG["SIGNAL_THREADS"], thread_name_prefix="Signal"
        )
        self.order_pool = ThreadPoolExecutor(
            max_workers=CONFIG["ORDER_THREADS"], thread_name_prefix="Order"
        )

        # Stats
        self.stats = {
            "quotes_fetched": 0,
            "signals_generated": 0,
            "orders_placed": 0,
            "cycle_time_ms": 0,
        }

        self.running = False
        self.eod_liquidated = False  # Track if we've done EOD liquidation today
        self.est_tz = pytz.timezone("US/Eastern")

        # Auto-detect account type from Alpaca
        self.account_info = self.detect_account_type()

        logger.info("Parallel HFT Scalper V4 initialized")
        logger.info(
            f"Thread pools: Quote={CONFIG['QUOTE_THREADS']}, "
            f"Signal={CONFIG['SIGNAL_THREADS']}, Order={CONFIG['ORDER_THREADS']}"
        )
        logger.info(
            f"Account: {self.account_info['type']} | "
            f"Portfolio: ${self.account_info['portfolio_value']:,.2f} | "
            f"PDT OK: {self.account_info['pdt_ok']}"
        )
        if self.account_info["eod_liquidate"]:
            logger.info(
                f"EOD Liquidation ENABLED: {CONFIG['EOD_LIQUIDATE_TIME']} EST (Cash/IRA account)"
            )
        else:
            logger.info("EOD Liquidation DISABLED (Margin account above PDT threshold)")

    def detect_account_type(self) -> dict:
        """
        Auto-detect account type and PDT status from Alpaca.

        Returns dict with:
        - type: 'CASH', 'MARGIN', or 'IRA'
        - portfolio_value: float
        - pdt_ok: True if above $25k threshold (can day trade freely on margin)
        - eod_liquidate: True if we need EOD liquidation (cash accounts or below PDT)
        - buying_power: float
        - day_trades_remaining: int (if PDT restricted)
        """
        try:
            acct = self.trading.get_account()

            portfolio_value = float(acct.portfolio_value)
            buying_power = float(acct.buying_power)
            cash = float(acct.cash)

            # Detect account type from Alpaca fields
            # Alpaca uses 'pattern_day_trader' and 'account_type' fields
            is_margin = hasattr(acct, "multiplier") and float(acct.multiplier) > 1
            pdt_flagged = getattr(acct, "pattern_day_trader", False)
            daytrade_count = int(getattr(acct, "daytrade_count", 0))

            # Determine account type
            if is_margin:
                account_type = "MARGIN"
            else:
                account_type = "CASH"

            # PDT check: Above $25k on margin = unlimited day trades
            pdt_ok = is_margin and portfolio_value >= CONFIG["PDT_THRESHOLD"]

            # EOD Liquidation needed for:
            # 1. Cash accounts (T+1 settlement)
            # 2. Margin accounts below PDT threshold (limited day trades)
            eod_liquidate = not pdt_ok

            return {
                "type": account_type,
                "portfolio_value": portfolio_value,
                "buying_power": buying_power,
                "cash": cash,
                "pdt_ok": pdt_ok,
                "pdt_flagged": pdt_flagged,
                "daytrade_count": daytrade_count,
                "day_trades_remaining": 3 - daytrade_count if not pdt_ok else 999,
                "eod_liquidate": eod_liquidate,
            }

        except Exception as e:
            logger.error(f"Account detection failed: {e}")
            # Default to safe mode (cash account behavior)
            return {
                "type": "CASH",
                "portfolio_value": 0,
                "buying_power": 0,
                "cash": 0,
                "pdt_ok": False,
                "pdt_flagged": False,
                "daytrade_count": 0,
                "day_trades_remaining": 3,
                "eod_liquidate": True,
            }

    def fetch_quote(self, symbol: str) -> Optional[Quote]:
        """Fetch quote for a single symbol (runs in thread pool)"""
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data.get_stock_latest_quote(req)
            q = quotes[symbol]

            quote = Quote(
                symbol=symbol,
                bid=float(q.bid_price),
                ask=float(q.ask_price),
                last=float(q.ask_price),
                timestamp=time.time(),
            )

            # Update velocity tracker
            self.velocity.update(symbol, quote.last)

            # Update price history
            with self.history_lock:
                self.price_history[symbol].append(quote.last)
                if len(self.price_history[symbol]) > 20:
                    self.price_history[symbol] = self.price_history[symbol][-20:]

            self.stats["quotes_fetched"] += 1
            return quote

        except Exception as e:
            return None

    def fetch_all_quotes_parallel(self) -> List[Quote]:
        """Fetch all quotes in parallel"""
        futures = {
            self.quote_pool.submit(self.fetch_quote, sym): sym
            for sym in CONFIG["SYMBOLS"]
        }

        quotes = []
        for future in as_completed(futures, timeout=2):
            try:
                quote = future.result()
                if quote:
                    quotes.append(quote)
            except:
                pass

        return quotes

    def analyze_signal(self, quote: Quote) -> Optional[Signal]:
        """Analyze quote and generate signal (runs in thread pool)"""
        symbol = quote.symbol

        # Get price history
        with self.history_lock:
            prices = list(self.price_history.get(symbol, []))

        if len(prices) < 5:
            return None

        # Calculate signals
        momentum = (prices[-1] - prices[-2]) / prices[-2]
        prev_momentum = (prices[-2] - prices[-3]) / prices[-3]
        acceleration = momentum - prev_momentum
        trend = (prices[-1] - prices[-5]) / prices[-5]

        # Get velocity metrics
        velocity = self.velocity.get_velocity(symbol)
        speed_class = self.velocity.get_speed_class(symbol)

        # Check if blocked
        with self.blocked_lock:
            if symbol in self.blocked_symbols:
                if time.time() < self.blocked_symbols[symbol]:
                    return Signal(
                        symbol,
                        "HOLD",
                        quote.last,
                        momentum,
                        acceleration,
                        velocity,
                        speed_class,
                        0,
                    )
                del self.blocked_symbols[symbol]

        # Check if already in position
        with self.positions_lock:
            in_position = symbol in self.positions

        # Generate signal
        action = "HOLD"
        confidence = 0

        if not in_position:
            # Entry conditions
            momentum_ok = momentum > CONFIG["MIN_MOMENTUM"]
            accel_ok = acceleration > CONFIG["MIN_ACCELERATION"]
            trend_ok = trend > 0

            # Check position limit
            with self.positions_lock:
                can_open = len(self.positions) < CONFIG["MAX_POSITIONS"]

            if momentum_ok and accel_ok and trend_ok and can_open:
                action = "BUY"
                confidence = min(1.0, (momentum + acceleration) * 100)

        self.stats["signals_generated"] += 1

        return Signal(
            symbol=symbol,
            action=action,
            price=quote.last,
            momentum=momentum,
            acceleration=acceleration,
            velocity=velocity,
            speed_class=speed_class,
            confidence=confidence,
        )

    def analyze_signals_parallel(self, quotes: List[Quote]) -> List[Signal]:
        """Analyze all quotes in parallel"""
        futures = {
            self.signal_pool.submit(self.analyze_signal, q): q.symbol for q in quotes
        }

        signals = []
        for future in as_completed(futures, timeout=1):
            try:
                signal = future.result()
                if signal and signal.action == "BUY":
                    signals.append(signal)
            except:
                pass

        return signals

    def execute_order(self, signal: Signal) -> bool:
        """Execute order based on signal (runs in thread pool)"""
        symbol = signal.symbol
        price = signal.price

        try:
            # Calculate quantity
            qty = max(1, int(CONFIG["POSITION_SIZE"] / price))

            # Predict fill price
            lookahead = 200 if price < CONFIG["LOW_PRICE_THRESHOLD"] else 150
            predicted_fill = self.velocity.predict_fill(symbol, price, lookahead)

            # Use predicted fill for bracket prices
            base_price = (
                predicted_fill if signal.speed_class in ["FAST", "EXTREME"] else price
            )

            # Calculate stops based on price tier
            if base_price < CONFIG["LOW_PRICE_THRESHOLD"]:
                stop_dist = CONFIG["MIN_STOP_CENTS"]
                profit_dist = CONFIG["MIN_PROFIT_CENTS"]
                if base_price < CONFIG["PENNY_THRESHOLD"]:
                    stop_dist *= 1.5
                    profit_dist *= 1.5
                stop_price = round(base_price - stop_dist, 2)
                take_profit = round(base_price + profit_dist, 2)
            else:
                stop_price = round(base_price * (1 - CONFIG["STOP_LOSS_PCT"]), 2)
                take_profit = round(base_price * (1 + CONFIG["TAKE_PROFIT_PCT"]), 2)

            # Widen for extreme speed
            if signal.speed_class == "EXTREME":
                stop_dist = base_price - stop_price
                stop_price = round(base_price - stop_dist * 1.2, 2)

            logger.info(f">>> BRACKET: {symbol} [{signal.speed_class}]")
            logger.info(f"    Price: ${price:.2f} -> Predicted: ${predicted_fill:.2f}")
            logger.info(f"    Stop: ${stop_price:.2f} | TP: ${take_profit:.2f}")

            # Create bracket order
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=take_profit),
                stop_loss=StopLossRequest(stop_price=stop_price),
            )

            order = self.trading.submit_order(order_request)

            # Track position
            with self.positions_lock:
                self.positions[symbol] = {
                    "entry_price": price,
                    "qty": qty,
                    "order_id": order.id,
                    "stop_price": stop_price,
                    "take_profit": take_profit,
                }

            self.stats["orders_placed"] += 1
            logger.info(f"    Order ID: {order.id}")
            return True

        except Exception as e:
            logger.error(f"    Order failed: {e}")
            return False

    def execute_orders_parallel(self, signals: List[Signal]):
        """Execute multiple orders in parallel"""
        futures = {
            self.order_pool.submit(self.execute_order, sig): sig.symbol
            for sig in signals
        }

        for future in as_completed(futures, timeout=5):
            try:
                future.result()
            except:
                pass

    def sync_positions(self):
        """Sync with Alpaca positions"""
        try:
            actual = {p.symbol: p for p in self.trading.get_all_positions()}

            with self.positions_lock:
                for symbol in list(self.positions.keys()):
                    if symbol not in actual:
                        logger.info(f"<<< EXIT: {symbol} (bracket triggered)")
                        del self.positions[symbol]

                        with self.blocked_lock:
                            self.blocked_symbols[symbol] = time.time() + 60
        except:
            pass

    def get_est_time(self):
        """Get current time in EST"""
        return datetime.now(self.est_tz)

    def is_past_time(self, time_str: str) -> bool:
        """Check if current EST time is past the given time (HH:MM format)"""
        now = self.get_est_time()
        target_hour, target_min = map(int, time_str.split(":"))
        target = now.replace(hour=target_hour, minute=target_min, second=0)
        return now >= target

    def should_allow_new_entries(self) -> bool:
        """Check if we should allow new entries (before EOD cutoff)"""
        # Margin accounts above PDT can trade anytime
        if not self.account_info["eod_liquidate"]:
            return True
        # Cash/restricted accounts: no new entries after cutoff time
        return not self.is_past_time(CONFIG["EOD_NO_NEW_ENTRIES_TIME"])

    def check_eod_liquidation(self):
        """
        Check if it's time for EOD liquidation (Cash/IRA accounts).
        Sells all positions before market close to free up cash for T+1 settlement.

        Auto-detected behavior:
        - Cash accounts: Always liquidate at EOD
        - Margin < $25k: Liquidate to preserve day trades
        - Margin >= $25k: No liquidation needed (PDT exempt)
        """
        if not self.account_info["eod_liquidate"]:
            return  # Margin account above PDT - no liquidation needed

        if self.eod_liquidated:
            return  # Already done today

        if not self.is_past_time(CONFIG["EOD_LIQUIDATE_TIME"]):
            return  # Not time yet

        # Time to liquidate all positions
        logger.info("=" * 60)
        logger.info("EOD LIQUIDATION - Selling all positions for T+1 settlement")
        logger.info("=" * 60)

        try:
            positions = self.trading.get_all_positions()

            for pos in positions:
                symbol = pos.symbol
                qty = int(pos.qty)

                if qty <= 0:
                    continue

                try:
                    # Cancel any open orders for this symbol first
                    self.trading.cancel_orders()

                    # Market sell to ensure fill before close
                    order = self.trading.submit_order(
                        MarketOrderRequest(
                            symbol=symbol,
                            qty=qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                        )
                    )

                    entry = float(pos.avg_entry_price)
                    current = float(pos.current_price)
                    pnl = (current - entry) / entry * 100

                    logger.info(
                        f"<<< EOD SELL: {symbol} x{qty} @ ${current:.2f} "
                        f"(P/L: {pnl:+.1f}%)"
                    )

                except Exception as e:
                    logger.error(f"EOD sell failed for {symbol}: {e}")

            # Clear our position tracking
            with self.positions_lock:
                self.positions.clear()

            self.eod_liquidated = True
            logger.info("EOD Liquidation complete - cash will settle tomorrow")

        except Exception as e:
            logger.error(f"EOD Liquidation error: {e}")

    def run_cycle(self):
        """Run one trading cycle"""
        cycle_start = time.time()

        # Check for EOD liquidation (Cash/IRA accounts)
        self.check_eod_liquidation()

        # Phase 1: Fetch quotes in parallel
        quotes = self.fetch_all_quotes_parallel()

        # Phase 2: Analyze signals in parallel (only if entries allowed)
        signals = []
        if self.should_allow_new_entries():
            signals = self.analyze_signals_parallel(quotes)
        else:
            # After cutoff, just monitor existing positions
            pass

        # Phase 3: Execute orders in parallel
        if signals:
            self.execute_orders_parallel(signals)

        # Phase 4: Sync positions
        self.sync_positions()

        cycle_time = (time.time() - cycle_start) * 1000
        self.stats["cycle_time_ms"] = cycle_time

        return len(quotes), len(signals)

    def run(self):
        """Main loop"""
        logger.info("=" * 60)
        logger.info("PARALLEL HFT SCALPER V4")
        logger.info(f"Symbols: {len(CONFIG['SYMBOLS'])}")
        logger.info(
            f"Threads: {CONFIG['QUOTE_THREADS'] + CONFIG['SIGNAL_THREADS'] + CONFIG['ORDER_THREADS']}"
        )
        logger.info("=" * 60)

        self.running = True
        cycle = 0

        while self.running:
            try:
                quotes, signals = self.run_cycle()

                cycle += 1
                if cycle % 20 == 0:  # Status every ~1 second
                    with self.positions_lock:
                        pos_str = ", ".join(self.positions.keys()) or "FLAT"

                    logger.info(
                        f"[{cycle}] Cycle: {self.stats['cycle_time_ms']:.1f}ms | "
                        f"Quotes: {quotes} | Signals: {signals} | "
                        f"Positions: {pos_str}"
                    )

            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")

            time.sleep(CONFIG["QUOTE_INTERVAL"])

        # Cleanup
        self.quote_pool.shutdown(wait=False)
        self.signal_pool.shutdown(wait=False)
        self.order_pool.shutdown(wait=False)
        logger.info("Shutdown complete")


if __name__ == "__main__":
    scalper = ParallelHFTScalper()
    scalper.run()
