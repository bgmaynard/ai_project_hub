"""
HFT Scalper V2 - Improved Risk Management
==========================================
Enhanced scalper with:
1. Trailing stops - lock in profits as price rises
2. Volatility filter - avoid or reduce size on volatile stocks
3. Quick profit taking - take 0.5-1% on volatile, 2% on stable
4. Loss cooldown - don't re-enter after stop loss
5. Dynamic position sizing based on volatility
"""

import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Set

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

API_BASE = "http://localhost:9100/api/alpaca"

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Price range (Warrior Trading)
    "MIN_PRICE": 2.00,
    "MAX_PRICE": 20.00,
    "BASE_POSITION_SIZE": 500,  # Base $ per trade
    # Momentum thresholds
    "MOMENTUM_THRESHOLD": 0.15,
    "ACCELERATION_THRESHOLD": 0.05,
    "REVERSAL_THRESHOLD": -0.08,
    # Volume
    "VOLUME_BUY_RATIO": 1.5,
    # MACD
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    # Patterns
    "TAIL_RATIO": 0.6,
    "CONSECUTIVE_TAILS": 2,
    # S/R
    "SR_HOLD_TICKS": 3,
    "PEAK_TOLERANCE": 0.005,
    "MIN_TOUCHES": 2,
    # ============== NEW V2 SETTINGS ==============
    # Volatility classification
    "HIGH_VOLATILITY_THRESHOLD": 0.03,  # 3% std dev = high volatility
    "LOW_VOLATILITY_THRESHOLD": 0.015,  # 1.5% std dev = low volatility
    # Dynamic risk based on volatility
    "STOP_LOSS_HIGH_VOL": 0.008,  # 0.8% stop for high volatility
    "STOP_LOSS_LOW_VOL": 0.015,  # 1.5% stop for low volatility
    "TAKE_PROFIT_HIGH_VOL": 0.008,  # 0.8% profit for high vol (quick scalp)
    "TAKE_PROFIT_LOW_VOL": 0.02,  # 2% profit for low vol
    # Trailing stop
    "TRAILING_STOP_ACTIVATE": 0.005,  # Activate trailing at 0.5% profit
    "TRAILING_STOP_DISTANCE": 0.004,  # Trail 0.4% behind high
    # Loss cooldown
    "LOSS_COOLDOWN_SECONDS": 300,  # 5 min cooldown after stop loss
    "MAX_LOSSES_PER_SYMBOL": 2,  # Max 2 stop losses per symbol per day
    # Entry requirements
    "MIN_SIGNALS_HIGH_VOL": 3,  # Need 3+ signals for high vol stocks
    "MIN_SIGNALS_LOW_VOL": 2,  # Need 2 signals for stable stocks
    # Position sizing
    "REDUCE_SIZE_HIGH_VOL": 0.5,  # Use 50% size on high vol stocks
    # Max positions
    "MAX_POSITIONS": 3,
    # Polling
    "POLL_INTERVAL": 0.5,
    "TICK_BUFFER_SIZE": 30,  # More ticks for volatility calc
}


class HFTScalperV2:
    """Improved HFT scalper with better risk management."""

    def __init__(self, symbols: list):
        self.symbols = [s.upper() for s in symbols]
        self.running = False

        # Price/momentum tracking
        self.ticks: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=CONFIG["TICK_BUFFER_SIZE"])
        )
        self.last_price: Dict[str, float] = {}
        self.momentum: Dict[str, float] = defaultdict(float)
        self.acceleration: Dict[str, float] = defaultdict(float)

        # Volatility tracking
        self.volatility: Dict[str, float] = defaultdict(lambda: 0.02)  # Default 2%
        self.volatility_class: Dict[str, str] = defaultdict(
            lambda: "MEDIUM"
        )  # HIGH/MEDIUM/LOW

        # Volume
        self.buy_volume: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.sell_volume: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.volume_ratio: Dict[str, float] = defaultdict(lambda: 1.0)

        # MACD
        self.macd_histogram: Dict[str, float] = defaultdict(float)
        self.macd_bullish: Dict[str, bool] = defaultdict(bool)
        self.last_macd_update: Dict[str, datetime] = {}

        # Patterns
        self.bottoming_tails: Dict[str, int] = defaultdict(int)
        self.topping_tails: Dict[str, int] = defaultdict(int)
        self.pattern_signals: Dict[str, List[str]] = defaultdict(list)
        self.last_pattern_scan: Dict[str, datetime] = {}

        # VWAP
        self.vwap: Dict[str, float] = defaultdict(float)
        self.vwap_cumulative_vol: Dict[str, float] = defaultdict(float)
        self.vwap_cumulative_pv: Dict[str, float] = defaultdict(float)
        self.above_vwap: Dict[str, bool] = defaultdict(bool)
        self.vwap_cross_up: Dict[str, bool] = defaultdict(bool)

        # S/R
        self.sr_breakout: Dict[str, bool] = defaultdict(bool)
        self.sr_level_broken: Dict[str, float] = defaultdict(float)
        self.ticks_above_sr: Dict[str, int] = defaultdict(int)
        self.resistance_levels: Dict[str, List[dict]] = defaultdict(list)
        self.hard_resistance_break: Dict[str, bool] = defaultdict(bool)
        self.last_sr_scan: Dict[str, datetime] = {}

        # Position tracking with trailing stop
        self.positions: Dict[str, dict] = (
            {}
        )  # {symbol: {entry, qty, high_water, trailing_stop, stop_loss, take_profit}}
        self.pending_orders: Set[str] = set()

        # Loss tracking
        self.loss_cooldown: Dict[str, datetime] = {}  # When we can trade again
        self.daily_losses: Dict[str, int] = defaultdict(int)  # Stop losses today

        # Stats
        self.trades_executed = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0

        logger.info(f"HFT Scalper V2 initialized for: {self.symbols}")
        logger.info(f"Features: Trailing stops, Volatility filter, Loss cooldown")

    def get_quote(self, symbol: str) -> dict:
        try:
            r = requests.get(f"{API_BASE}/quote/{symbol}", timeout=1)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return {}

    def get_bars(self, symbol: str, limit: int = 30) -> list:
        try:
            r = requests.get(
                f"{API_BASE}/bars/{symbol}?limit={limit}&timeframe=1Min", timeout=2
            )
            if r.status_code == 200:
                return r.json().get("bars", [])
        except:
            pass
        return []

    def place_order(
        self, symbol: str, qty: int, side: str, limit_price: float = None
    ) -> dict:
        try:
            order_data = {
                "symbol": symbol,
                "quantity": qty,
                "action": side,
                "order_type": "limit" if limit_price else "market",
                "time_in_force": "day",
                "extended_hours": True,
            }
            if limit_price:
                order_data["limit_price"] = limit_price
            r = requests.post(f"{API_BASE}/place-order", json=order_data, timeout=2)
            return r.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def calculate_volatility(self, symbol: str):
        """Calculate volatility from recent ticks."""
        ticks = list(self.ticks[symbol])
        if len(ticks) < 10:
            return

        prices = [t["price"] for t in ticks]
        returns = [
            (prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))
        ]

        if returns:
            import statistics

            vol = statistics.stdev(returns) if len(returns) > 1 else 0.02
            self.volatility[symbol] = vol

            # Classify
            if vol > CONFIG["HIGH_VOLATILITY_THRESHOLD"]:
                self.volatility_class[symbol] = "HIGH"
            elif vol < CONFIG["LOW_VOLATILITY_THRESHOLD"]:
                self.volatility_class[symbol] = "LOW"
            else:
                self.volatility_class[symbol] = "MEDIUM"

    def calculate_macd(self, prices: List[float]) -> tuple:
        if len(prices) < CONFIG["MACD_SLOW"]:
            return 0, 0, 0

        def ema(data, period):
            alpha = 2 / (period + 1)
            result = [data[0]]
            for price in data[1:]:
                result.append(alpha * price + (1 - alpha) * result[-1])
            return result

        fast_ema = ema(prices, CONFIG["MACD_FAST"])
        slow_ema = ema(prices, CONFIG["MACD_SLOW"])
        macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
        signal_line = ema(macd_line, CONFIG["MACD_SIGNAL"])
        return macd_line[-1], signal_line[-1], macd_line[-1] - signal_line[-1]

    def update_macd(self, symbol: str):
        now = datetime.now()
        last = self.last_macd_update.get(symbol)
        if last and (now - last).seconds < 30:
            return

        bars = self.get_bars(symbol, 30)
        if len(bars) >= CONFIG["MACD_SLOW"]:
            prices = [float(b.get("c", b.get("close", 0))) for b in bars]
            if all(p > 0 for p in prices):
                macd, signal, histogram = self.calculate_macd(prices)
                prev_hist = self.macd_histogram[symbol]
                self.macd_histogram[symbol] = histogram
                self.macd_bullish[symbol] = (prev_hist < 0 and histogram > 0) or (
                    histogram > 0 and histogram > prev_hist
                )
                self.last_macd_update[symbol] = now

    def update_candle_patterns(self, symbol: str):
        now = datetime.now()
        last = self.last_pattern_scan.get(symbol)
        if last and (now - last).seconds < 30:
            return

        bars = self.get_bars(symbol, 10)
        if len(bars) < 5:
            return

        patterns = []
        bottoming = 0
        topping = 0

        for i, bar in enumerate(bars[-5:]):
            o = float(bar.get("o", bar.get("open", 0)))
            h = float(bar.get("h", bar.get("high", 0)))
            l = float(bar.get("l", bar.get("low", 0)))
            c = float(bar.get("c", bar.get("close", 0)))

            if o == 0 or h == 0:
                continue

            body = abs(c - o)
            total_range = h - l
            if total_range == 0:
                continue

            if c >= o:
                lower_tail = o - l
                upper_tail = h - c
            else:
                lower_tail = c - l
                upper_tail = h - o

            lower_ratio = lower_tail / total_range
            upper_ratio = upper_tail / total_range

            if lower_ratio > CONFIG["TAIL_RATIO"]:
                bottoming += 1
            if upper_ratio > CONFIG["TAIL_RATIO"]:
                topping += 1

            if body / total_range < 0.3 and lower_ratio > 0.6:
                patterns.append("HAMMER")

        # Bullish engulfing
        if len(bars) >= 2:
            prev = bars[-2]
            curr = bars[-1]
            prev_o = float(prev.get("o", prev.get("open", 0)))
            prev_c = float(prev.get("c", prev.get("close", 0)))
            curr_o = float(curr.get("o", curr.get("open", 0)))
            curr_c = float(curr.get("c", curr.get("close", 0)))

            if prev_c < prev_o and curr_c > curr_o:
                if curr_o < prev_c and curr_c > prev_o:
                    patterns.append("BULLISH_ENGULF")

        # Candle over candle
        if len(bars) >= 3:
            closes = [float(b.get("c", b.get("close", 0))) for b in bars[-3:]]
            if closes[2] > closes[1] > closes[0]:
                patterns.append("CANDLE_OVER_CANDLE")

        self.bottoming_tails[symbol] = bottoming
        self.topping_tails[symbol] = topping
        self.pattern_signals[symbol] = patterns
        self.last_pattern_scan[symbol] = now

    def update_vwap(self, symbol: str, price: float, volume: float):
        if volume <= 0:
            return

        self.vwap_cumulative_vol[symbol] += volume
        self.vwap_cumulative_pv[symbol] += price * volume

        if self.vwap_cumulative_vol[symbol] > 0:
            new_vwap = (
                self.vwap_cumulative_pv[symbol] / self.vwap_cumulative_vol[symbol]
            )
            self.vwap[symbol] = new_vwap

            was_above = self.above_vwap[symbol]
            is_above = price > new_vwap * 1.002

            if is_above and not was_above:
                self.vwap_cross_up[symbol] = True
                logger.info(f"VWAP CROSS UP: {symbol} @ ${price:.2f}")
            elif not is_above:
                self.vwap_cross_up[symbol] = False

            self.above_vwap[symbol] = is_above

    def check_sr_breakout(self, symbol: str, price: float):
        whole_below = int(price)
        half_below = int(price) + 0.5 if price % 1 >= 0.5 else int(price) - 0.5

        if half_below > whole_below and half_below < price:
            nearest_level = half_below
        else:
            nearest_level = whole_below

        just_above = price > nearest_level and price < (nearest_level + 0.10)

        if just_above:
            if self.sr_level_broken[symbol] == nearest_level:
                self.ticks_above_sr[symbol] += 1
            else:
                self.sr_level_broken[symbol] = nearest_level
                self.ticks_above_sr[symbol] = 1

            if self.ticks_above_sr[symbol] >= CONFIG["SR_HOLD_TICKS"]:
                if not self.sr_breakout[symbol]:
                    self.sr_breakout[symbol] = True
                    level_type = (
                        "WHOLE" if nearest_level == int(nearest_level) else "HALF"
                    )
                    logger.info(
                        f"S/R BREAKOUT: {symbol} broke ${nearest_level:.2f} ({level_type})"
                    )
        else:
            self.sr_breakout[symbol] = False
            self.ticks_above_sr[symbol] = 0

    def get_signal_strength(self, symbol: str) -> tuple:
        strength = 0
        signals = []

        if self.momentum.get(symbol, 0) > CONFIG["MOMENTUM_THRESHOLD"]:
            strength += 1
            signals.append("MOM")

        if self.acceleration.get(symbol, 0) > CONFIG["ACCELERATION_THRESHOLD"]:
            strength += 1
            signals.append("ACC")

        if self.volume_ratio.get(symbol, 1) > CONFIG["VOLUME_BUY_RATIO"]:
            strength += 1
            signals.append("VOL")

        if self.macd_bullish.get(symbol, False):
            strength += 1
            signals.append("MACD")

        if self.bottoming_tails.get(symbol, 0) >= CONFIG["CONSECUTIVE_TAILS"]:
            strength += 1
            signals.append("TAILS")

        if self.topping_tails.get(symbol, 0) >= CONFIG["CONSECUTIVE_TAILS"]:
            strength -= 1
            signals.append("TOP_TAILS")

        if self.vwap_cross_up.get(symbol, False):
            strength += 1
            signals.append("VWAP")

        if self.sr_breakout.get(symbol, False):
            strength += 1
            signals.append("SR")

        if self.hard_resistance_break.get(symbol, False):
            strength += 2
            signals.append("HARD_R")

        patterns = self.pattern_signals.get(symbol, [])
        if "HAMMER" in patterns or "BULLISH_ENGULF" in patterns:
            strength += 1
            signals.append("PATTERN")
        if "CANDLE_OVER_CANDLE" in patterns:
            strength += 1
            signals.append("COC")

        return strength, signals

    def can_trade(self, symbol: str) -> tuple:
        """Check if we can trade this symbol."""
        # Check cooldown
        cooldown = self.loss_cooldown.get(symbol)
        if cooldown and datetime.now() < cooldown:
            remaining = (cooldown - datetime.now()).seconds
            return False, f"COOLDOWN ({remaining}s remaining)"

        # Check daily losses
        if self.daily_losses[symbol] >= CONFIG["MAX_LOSSES_PER_SYMBOL"]:
            return False, f"MAX LOSSES ({self.daily_losses[symbol]} today)"

        return True, "OK"

    def process_tick(self, symbol: str, quote: dict):
        price = float(quote.get("last", quote.get("ask", quote.get("bid", 0))))
        volume = float(quote.get("volume", quote.get("last_volume", 0)))

        if price <= 0:
            return

        if price < CONFIG["MIN_PRICE"] or price > CONFIG["MAX_PRICE"]:
            return

        # Store tick
        self.ticks[symbol].append(
            {"price": price, "volume": volume, "time": datetime.now()}
        )

        # Update volatility
        self.calculate_volatility(symbol)

        # Volume profile
        last = self.last_price.get(symbol, price)
        if price >= last:
            self.buy_volume[symbol].append(volume)
        else:
            self.sell_volume[symbol].append(volume)
        self.last_price[symbol] = price

        total_buy = sum(self.buy_volume[symbol]) or 1
        total_sell = sum(self.sell_volume[symbol]) or 1
        self.volume_ratio[symbol] = total_buy / total_sell

        # Momentum
        ticks = list(self.ticks[symbol])
        if len(ticks) >= 3:
            vel1 = (ticks[-1]["price"] - ticks[-2]["price"]) / ticks[-2]["price"] * 100
            vel2 = (ticks[-2]["price"] - ticks[-3]["price"]) / ticks[-3]["price"] * 100
            self.momentum[symbol] = vel1
            self.acceleration[symbol] = vel1 - vel2

        # Other indicators
        self.update_vwap(symbol, price, volume)
        self.check_sr_breakout(symbol, price)
        self.update_macd(symbol)
        self.update_candle_patterns(symbol)

    def check_entry(self, symbol: str):
        if symbol in self.positions or len(self.positions) >= CONFIG["MAX_POSITIONS"]:
            return

        # Check if we can trade
        can_trade, reason = self.can_trade(symbol)
        if not can_trade:
            return

        price = self.last_price.get(symbol, 0)
        if price <= 0:
            return

        strength, signals = self.get_signal_strength(symbol)
        vol_class = self.volatility_class[symbol]

        # Require more signals for high volatility
        min_signals = (
            CONFIG["MIN_SIGNALS_HIGH_VOL"]
            if vol_class == "HIGH"
            else CONFIG["MIN_SIGNALS_LOW_VOL"]
        )

        if strength >= min_signals:
            # Adjust position size based on volatility
            base_size = CONFIG["BASE_POSITION_SIZE"]
            if vol_class == "HIGH":
                position_size = base_size * CONFIG["REDUCE_SIZE_HIGH_VOL"]
            else:
                position_size = base_size

            qty = max(1, int(position_size / price))
            limit_price = round(price * 1.001, 2)

            # Set stops based on volatility
            if vol_class == "HIGH":
                stop_loss = price * (1 - CONFIG["STOP_LOSS_HIGH_VOL"])
                take_profit = price * (1 + CONFIG["TAKE_PROFIT_HIGH_VOL"])
            else:
                stop_loss = price * (1 - CONFIG["STOP_LOSS_LOW_VOL"])
                take_profit = price * (1 + CONFIG["TAKE_PROFIT_LOW_VOL"])

            logger.info(
                f">>> ENTRY: {symbol} @ ${price:.2f} | Vol={vol_class} | Str={strength} | {signals}"
            )
            logger.info(
                f"    Stop=${stop_loss:.2f} | Target=${take_profit:.2f} | Qty={qty}"
            )

            result = self.place_order(symbol, qty, "buy", limit_price)
            if result.get("success"):
                self.positions[symbol] = {
                    "entry": price,
                    "qty": qty,
                    "high_water": price,
                    "trailing_stop": None,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "vol_class": vol_class,
                    "signals": signals,
                    "time": datetime.now(),
                }
                self.trades_executed += 1
                logger.info(f">>> BOUGHT {symbol}: {qty} @ ${limit_price:.2f}")
            else:
                logger.warning(f"Order failed: {result.get('error', 'unknown')}")

    def check_exit(self, symbol: str):
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        price = self.last_price.get(symbol, pos["entry"])
        entry = pos["entry"]
        pnl_pct = (price - entry) / entry

        should_exit = False
        reason = ""

        # Update high water mark
        if price > pos["high_water"]:
            pos["high_water"] = price

            # Activate trailing stop once we hit threshold
            if pnl_pct >= CONFIG["TRAILING_STOP_ACTIVATE"]:
                trailing_stop = price * (1 - CONFIG["TRAILING_STOP_DISTANCE"])
                if pos["trailing_stop"] is None or trailing_stop > pos["trailing_stop"]:
                    pos["trailing_stop"] = trailing_stop
                    logger.info(
                        f"    TRAILING STOP: {symbol} set to ${trailing_stop:.2f}"
                    )

        # Check trailing stop
        if pos["trailing_stop"] and price <= pos["trailing_stop"]:
            should_exit = True
            reason = f"TRAIL STOP (+{pnl_pct*100:.2f}%)"

        # Check take profit
        elif price >= pos["take_profit"]:
            should_exit = True
            reason = f"TAKE PROFIT (+{pnl_pct*100:.2f}%)"

        # Check hard stop loss
        elif price <= pos["stop_loss"]:
            should_exit = True
            reason = f"STOP LOSS ({pnl_pct*100:.2f}%)"
            # Set cooldown
            self.loss_cooldown[symbol] = datetime.now() + timedelta(
                seconds=CONFIG["LOSS_COOLDOWN_SECONDS"]
            )
            self.daily_losses[symbol] += 1
            logger.info(
                f"    COOLDOWN: {symbol} blocked for {CONFIG['LOSS_COOLDOWN_SECONDS']}s"
            )

        # Momentum reversal (but only if we're in profit)
        elif (
            pnl_pct > 0 and self.momentum.get(symbol, 0) < CONFIG["REVERSAL_THRESHOLD"]
        ):
            should_exit = True
            reason = f"REVERSAL (+{pnl_pct*100:.2f}%)"

        if should_exit:
            qty = pos["qty"]
            limit_price = round(price * 0.999, 2)

            logger.info(f"<<< EXIT: {symbol} | {reason}")

            result = self.place_order(symbol, qty, "sell", limit_price)
            if result.get("success"):
                pnl_dollars = (price - entry) * qty
                del self.positions[symbol]
                self.total_pnl += pnl_dollars

                if pnl_dollars >= 0:
                    self.wins += 1
                else:
                    self.losses += 1

                logger.info(
                    f"<<< SOLD {symbol}: {qty} @ ${limit_price:.2f} | P/L: ${pnl_dollars:.2f}"
                )

    def run(self):
        logger.info("=" * 60)
        logger.info("HFT SCALPER V2 - IMPROVED RISK MANAGEMENT")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Features: Trailing stops, Volatility filter, Loss cooldown")
        logger.info("=" * 60)

        self.running = True
        cycle = 0

        try:
            while self.running:
                cycle += 1

                for symbol in self.symbols:
                    try:
                        quote = self.get_quote(symbol)
                        if not quote:
                            continue

                        self.process_tick(symbol, quote)
                        self.check_entry(symbol)
                        self.check_exit(symbol)

                    except Exception as e:
                        pass

                # Status update
                if cycle % 20 == 0:
                    pos_list = []
                    for sym, pos in self.positions.items():
                        pnl = (
                            (self.last_price.get(sym, pos["entry"]) - pos["entry"])
                            / pos["entry"]
                            * 100
                        )
                        pos_list.append(f"{sym}({pnl:+.1f}%)")

                    pos_str = ", ".join(pos_list) or "FLAT"
                    win_rate = (self.wins / max(1, self.wins + self.losses)) * 100

                    logger.info(
                        f"[{cycle}] {pos_str} | W/L: {self.wins}/{self.losses} ({win_rate:.0f}%) | P/L: ${self.total_pnl:.2f}"
                    )

                time.sleep(CONFIG["POLL_INTERVAL"])

        except KeyboardInterrupt:
            logger.info("\nShutting down...")

        logger.info("=" * 60)
        logger.info(f"Session Summary:")
        logger.info(f"  Trades: {self.trades_executed}")
        logger.info(f"  Wins: {self.wins} | Losses: {self.losses}")
        logger.info(
            f"  Win Rate: {(self.wins / max(1, self.wins + self.losses)) * 100:.1f}%"
        )
        logger.info(f"  Total P/L: ${self.total_pnl:.2f}")
        logger.info("=" * 60)


def get_hot_symbols():
    return ["AG", "HL", "KGC", "EDIT", "MAMA", "USGO"]  # Removed VOR for now


if __name__ == "__main__":
    # Don't include VOR initially - it's too volatile
    symbols = get_hot_symbols()
    scalper = HFTScalperV2(symbols)
    scalper.run()
