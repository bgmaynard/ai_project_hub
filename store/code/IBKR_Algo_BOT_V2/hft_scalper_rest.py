"""
HFT Scalper - REST API Version
==============================
Fast polling-based momentum detection using local REST API.
Uses the dashboard API at localhost:9100 to avoid WebSocket connection limits.

Features:
- All indicators from hft_scalper.py (momentum, MACD, volume, patterns)
- Fast 500ms polling via REST API
- Signal strength scoring system
- Automatic entry/exit on momentum
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# API endpoint
API_BASE = "http://localhost:9100/api/alpaca"

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Warrior Trading rules
    "MIN_PRICE": 2.00,
    "MAX_PRICE": 20.00,
    "POSITION_SIZE": 500,  # $ per trade
    # Momentum thresholds
    "MOMENTUM_THRESHOLD": 0.15,
    "ACCELERATION_THRESHOLD": 0.05,
    "REVERSAL_THRESHOLD": -0.08,
    # Volume profile
    "VOLUME_BUY_RATIO": 1.5,
    "VOLUME_SURGE_MULT": 2.0,
    # MACD settings
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    # Candlestick patterns
    "TAIL_RATIO": 0.6,
    "CONSECUTIVE_TAILS": 2,
    # S/R levels
    "SR_BUFFER": 0.01,
    "SR_HOLD_TICKS": 3,
    "PEAK_TOLERANCE": 0.005,
    "MIN_TOUCHES": 2,
    # Risk management
    "MAX_POSITIONS": 3,
    "STOP_LOSS_PCT": 0.015,
    "TAKE_PROFIT_PCT": 0.02,
    # Polling
    "POLL_INTERVAL": 0.5,  # 500ms
    "TICK_BUFFER_SIZE": 20,
}


class HFTScalperREST:
    """REST-based HFT scalper using local API."""

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

        # Volume tracking
        self.buy_volume: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.sell_volume: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.volume_ratio: Dict[str, float] = defaultdict(lambda: 1.0)

        # MACD
        self.macd_histogram: Dict[str, float] = defaultdict(float)
        self.macd_bullish: Dict[str, bool] = defaultdict(bool)
        self.last_macd_update: Dict[str, datetime] = {}

        # Candlestick patterns
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

        # S/R levels
        self.sr_breakout: Dict[str, bool] = defaultdict(bool)
        self.sr_level_broken: Dict[str, float] = defaultdict(float)
        self.ticks_above_sr: Dict[str, int] = defaultdict(int)
        self.resistance_levels: Dict[str, List[dict]] = defaultdict(list)
        self.hard_resistance_break: Dict[str, bool] = defaultdict(bool)
        self.last_sr_scan: Dict[str, datetime] = {}

        # Position tracking
        self.positions: Dict[str, dict] = {}
        self.pending_orders: Set[str] = set()

        # Stats
        self.trades_executed = 0
        self.total_pnl = 0.0

        logger.info(f"HFT Scalper (REST) initialized for: {self.symbols}")

    def get_quote(self, symbol: str) -> dict:
        """Get latest quote via REST API."""
        try:
            r = requests.get(f"{API_BASE}/quote/{symbol}", timeout=1)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return {}

    def get_bars(self, symbol: str, limit: int = 30) -> list:
        """Get historical bars via REST API."""
        try:
            r = requests.get(
                f"{API_BASE}/bars/{symbol}?limit={limit}&timeframe=1Min", timeout=2
            )
            if r.status_code == 200:
                data = r.json()
                return data.get("bars", [])
        except:
            pass
        return []

    def place_order(
        self, symbol: str, qty: int, side: str, limit_price: float = None
    ) -> dict:
        """Place order via REST API."""
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

    def calculate_macd(self, prices: List[float]) -> tuple:
        """Calculate MACD from price series."""
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
        """Update MACD from bars."""
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
        """Scan for candlestick patterns."""
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
        higher_highs = 0

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

            # Tail analysis
            if c >= o:  # Green candle
                lower_tail = o - l
                upper_tail = h - c
            else:  # Red candle
                lower_tail = c - l
                upper_tail = h - o

            lower_ratio = lower_tail / total_range
            upper_ratio = upper_tail / total_range

            # Bottoming tail (bullish)
            if lower_ratio > CONFIG["TAIL_RATIO"]:
                bottoming += 1

            # Topping tail (bearish)
            if upper_ratio > CONFIG["TAIL_RATIO"]:
                topping += 1

            # Hammer pattern (small body, long lower wick)
            if body / total_range < 0.3 and lower_ratio > 0.6:
                patterns.append("HAMMER")

            # Hanging man (small body, long lower wick, but at resistance)
            if (
                body / total_range < 0.3
                and lower_ratio > 0.6
                and self.hard_resistance_break.get(symbol)
            ):
                patterns.append("HANGING_MAN")

            # Higher highs
            if i > 0:
                prev_bar = bars[-5:][i - 1]
                prev_h = float(prev_bar.get("h", prev_bar.get("high", 0)))
                if h > prev_h:
                    higher_highs += 1

        # Bullish engulfing
        if len(bars) >= 2:
            prev = bars[-2]
            curr = bars[-1]
            prev_o = float(prev.get("o", prev.get("open", 0)))
            prev_c = float(prev.get("c", prev.get("close", 0)))
            curr_o = float(curr.get("o", curr.get("open", 0)))
            curr_c = float(curr.get("c", curr.get("close", 0)))

            if prev_c < prev_o and curr_c > curr_o:  # Red then green
                if curr_o < prev_c and curr_c > prev_o:  # Engulfing
                    patterns.append("BULLISH_ENGULF")

            if prev_c > prev_o and curr_c < curr_o:  # Green then red
                if curr_o > prev_c and curr_c < prev_o:
                    patterns.append("BEARISH_ENGULF")

        # Candle over candle (3+ consecutive higher closes)
        if len(bars) >= 3:
            closes = [float(b.get("c", b.get("close", 0))) for b in bars[-3:]]
            if closes[2] > closes[1] > closes[0]:
                patterns.append("CANDLE_OVER_CANDLE")

        self.bottoming_tails[symbol] = bottoming
        self.topping_tails[symbol] = topping
        self.pattern_signals[symbol] = patterns
        self.last_pattern_scan[symbol] = now

        if patterns:
            logger.info(f"PATTERNS {symbol}: {patterns}")

    def update_vwap(self, symbol: str, price: float, volume: float):
        """Update VWAP calculation."""
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
            is_above = price > new_vwap * 1.002  # 0.2% buffer

            if is_above and not was_above:
                self.vwap_cross_up[symbol] = True
                logger.info(
                    f"VWAP CROSS UP: {symbol} @ ${price:.2f} > VWAP ${new_vwap:.2f}"
                )
            elif not is_above:
                self.vwap_cross_up[symbol] = False

            self.above_vwap[symbol] = is_above

    def check_sr_breakout(self, symbol: str, price: float):
        """Check for S/R level breakouts."""
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

    def scan_historical_sr(self, symbol: str):
        """Scan for historical S/R levels (double/triple tops)."""
        now = datetime.now()
        last = self.last_sr_scan.get(symbol)
        if last and (now - last).seconds < 120:
            return

        bars = self.get_bars(symbol, 60)
        if len(bars) < 20:
            return

        highs = [float(b.get("h", b.get("high", 0))) for b in bars]
        lows = [float(b.get("l", b.get("low", 0))) for b in bars]

        # Find peaks
        peaks = []
        for i in range(2, len(highs) - 2):
            if (
                highs[i] > highs[i - 1]
                and highs[i] > highs[i - 2]
                and highs[i] > highs[i + 1]
                and highs[i] > highs[i + 2]
            ):
                peaks.append(highs[i])

        # Cluster peaks to find multi-touch levels
        if peaks:
            tolerance = max(peaks) * CONFIG["PEAK_TOLERANCE"]
            clusters = []
            peaks = sorted(peaks)

            current_cluster = [peaks[0]]
            for p in peaks[1:]:
                if p - current_cluster[-1] <= tolerance:
                    current_cluster.append(p)
                else:
                    if len(current_cluster) >= CONFIG["MIN_TOUCHES"]:
                        avg = sum(current_cluster) / len(current_cluster)
                        clusters.append({"price": avg, "touches": len(current_cluster)})
                    current_cluster = [p]

            if len(current_cluster) >= CONFIG["MIN_TOUCHES"]:
                avg = sum(current_cluster) / len(current_cluster)
                clusters.append({"price": avg, "touches": len(current_cluster)})

            self.resistance_levels[symbol] = clusters
            for c in clusters:
                logger.info(
                    f"HARD RESISTANCE: {symbol} @ ${c['price']:.2f} ({c['touches']} touches)"
                )

        self.last_sr_scan[symbol] = now

    def check_hard_resistance_break(self, symbol: str, price: float):
        """Check if price broke through hard resistance level."""
        for level in self.resistance_levels.get(symbol, []):
            if price > level["price"] * 1.005:  # 0.5% above resistance
                if not self.hard_resistance_break[symbol]:
                    self.hard_resistance_break[symbol] = True
                    logger.info(
                        f"DOUBLE TOP BREAK: {symbol} broke ${level['price']:.2f} ({level['touches']} touches) -> MOMENTUM"
                    )
                return
        self.hard_resistance_break[symbol] = False

    def get_signal_strength(self, symbol: str) -> tuple:
        """Calculate overall signal strength from all indicators."""
        strength = 0
        signals = []

        # Momentum (+1)
        if self.momentum.get(symbol, 0) > CONFIG["MOMENTUM_THRESHOLD"]:
            strength += 1
            signals.append("MOMENTUM")

        # Acceleration (+1)
        if self.acceleration.get(symbol, 0) > CONFIG["ACCELERATION_THRESHOLD"]:
            strength += 1
            signals.append("ACCEL")

        # Volume ratio (+1)
        if self.volume_ratio.get(symbol, 1) > CONFIG["VOLUME_BUY_RATIO"]:
            strength += 1
            signals.append("VOL_RATIO")

        # MACD bullish (+1)
        if self.macd_bullish.get(symbol, False):
            strength += 1
            signals.append("MACD")

        # Bottoming tails (+1)
        if self.bottoming_tails.get(symbol, 0) >= CONFIG["CONSECUTIVE_TAILS"]:
            strength += 1
            signals.append("BOT_TAILS")

        # Topping tails (-1)
        if self.topping_tails.get(symbol, 0) >= CONFIG["CONSECUTIVE_TAILS"]:
            strength -= 1
            signals.append("TOP_TAILS")

        # VWAP cross up (+1)
        if self.vwap_cross_up.get(symbol, False):
            strength += 1
            signals.append("VWAP_UP")

        # S/R breakout (+1)
        if self.sr_breakout.get(symbol, False):
            strength += 1
            signals.append("SR_BREAK")

        # Hard resistance break (+2)
        if self.hard_resistance_break.get(symbol, False):
            strength += 2
            signals.append("HARD_R_BREAK")

        # Candlestick patterns
        patterns = self.pattern_signals.get(symbol, [])
        if "HAMMER" in patterns or "BULLISH_ENGULF" in patterns:
            strength += 1
            signals.append("BULL_PATTERN")
        if "HANGING_MAN" in patterns or "BEARISH_ENGULF" in patterns:
            strength -= 1
            signals.append("BEAR_PATTERN")
        if "CANDLE_OVER_CANDLE" in patterns:
            strength += 1
            signals.append("COC")

        return strength, signals

    def process_tick(self, symbol: str, quote: dict):
        """Process a quote tick."""
        price = float(quote.get("last", quote.get("ask", quote.get("bid", 0))))
        volume = float(quote.get("volume", quote.get("last_volume", 0)))

        if price <= 0:
            return

        # Check price range
        if price < CONFIG["MIN_PRICE"] or price > CONFIG["MAX_PRICE"]:
            return

        # Store tick
        self.ticks[symbol].append(
            {"price": price, "volume": volume, "time": datetime.now()}
        )

        # Update volume profile
        last = self.last_price.get(symbol, price)
        if price >= last:
            self.buy_volume[symbol].append(volume)
        else:
            self.sell_volume[symbol].append(volume)
        self.last_price[symbol] = price

        # Calculate volume ratio
        total_buy = sum(self.buy_volume[symbol]) or 1
        total_sell = sum(self.sell_volume[symbol]) or 1
        self.volume_ratio[symbol] = total_buy / total_sell

        # Calculate momentum
        ticks = list(self.ticks[symbol])
        if len(ticks) >= 3:
            vel1 = (ticks[-1]["price"] - ticks[-2]["price"]) / ticks[-2]["price"] * 100
            vel2 = (ticks[-2]["price"] - ticks[-3]["price"]) / ticks[-3]["price"] * 100

            self.momentum[symbol] = vel1
            self.acceleration[symbol] = vel1 - vel2

        # Update indicators
        self.update_vwap(symbol, price, volume)
        self.check_sr_breakout(symbol, price)
        self.check_hard_resistance_break(symbol, price)

        # Periodic updates
        self.update_macd(symbol)
        self.update_candle_patterns(symbol)
        self.scan_historical_sr(symbol)

    def check_entry(self, symbol: str):
        """Check if we should enter a position."""
        if symbol in self.positions or len(self.positions) >= CONFIG["MAX_POSITIONS"]:
            return

        price = self.last_price.get(symbol, 0)
        if price <= 0:
            return

        strength, signals = self.get_signal_strength(symbol)

        if strength >= 2:  # Need at least 2 confirming signals
            qty = max(1, int(CONFIG["POSITION_SIZE"] / price))
            limit_price = round(price * 1.001, 2)  # 0.1% above for pre-market

            logger.info(
                f">>> ENTRY SIGNAL: {symbol} @ ${price:.2f} | Strength={strength} | {signals}"
            )

            result = self.place_order(symbol, qty, "buy", limit_price)
            if result.get("success"):
                self.positions[symbol] = {
                    "entry": price,
                    "qty": qty,
                    "signals": signals,
                    "time": datetime.now(),
                }
                self.trades_executed += 1
                logger.info(f">>> BOUGHT {symbol}: {qty} shares @ ${limit_price:.2f}")
            else:
                logger.warning(f"Order failed: {result.get('error', 'unknown')}")

    def check_exit(self, symbol: str):
        """Check if we should exit a position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        price = self.last_price.get(symbol, pos["entry"])
        entry = pos["entry"]
        pnl_pct = (price - entry) / entry * 100

        should_exit = False
        reason = ""

        # Take profit
        if pnl_pct >= CONFIG["TAKE_PROFIT_PCT"] * 100:
            should_exit = True
            reason = f"TAKE PROFIT (+{pnl_pct:.2f}%)"

        # Stop loss
        elif pnl_pct <= -CONFIG["STOP_LOSS_PCT"] * 100:
            should_exit = True
            reason = f"STOP LOSS ({pnl_pct:.2f}%)"

        # Momentum reversal
        elif self.momentum.get(symbol, 0) < CONFIG["REVERSAL_THRESHOLD"]:
            should_exit = True
            reason = f"REVERSAL ({pnl_pct:.2f}%)"

        if should_exit:
            qty = pos["qty"]
            limit_price = round(price * 0.999, 2)  # 0.1% below for pre-market

            logger.info(f"<<< EXIT: {symbol} | {reason}")

            result = self.place_order(symbol, qty, "sell", limit_price)
            if result.get("success"):
                del self.positions[symbol]
                self.total_pnl += (price - entry) * qty
                logger.info(
                    f"<<< SOLD {symbol}: {qty} shares @ ${limit_price:.2f} | P/L: ${(price-entry)*qty:.2f}"
                )

    def run(self):
        """Main polling loop."""
        logger.info("=" * 60)
        logger.info("HFT SCALPER (REST) - STARTING")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Poll interval: {CONFIG['POLL_INTERVAL']}s")
        logger.info("=" * 60)

        self.running = True
        cycle = 0

        try:
            while self.running:
                cycle += 1

                for symbol in self.symbols:
                    try:
                        # Get quote
                        quote = self.get_quote(symbol)
                        if not quote:
                            continue

                        # Process tick
                        self.process_tick(symbol, quote)

                        # Check entry/exit
                        self.check_entry(symbol)
                        self.check_exit(symbol)

                    except Exception as e:
                        pass

                # Status update every 20 cycles
                if cycle % 20 == 0:
                    pos_str = ", ".join(self.positions.keys()) or "FLAT"
                    logger.info(
                        f"[{cycle}] Positions: {pos_str} | Trades: {self.trades_executed} | P/L: ${self.total_pnl:.2f}"
                    )

                time.sleep(CONFIG["POLL_INTERVAL"])

        except KeyboardInterrupt:
            logger.info("\nShutting down...")

        logger.info("=" * 60)
        logger.info(f"Session Summary:")
        logger.info(f"  Trades: {self.trades_executed}")
        logger.info(f"  Total P/L: ${self.total_pnl:.2f}")
        logger.info("=" * 60)


def get_hot_symbols():
    """Get active symbols from scanner."""
    try:
        # Try scanner API
        r = requests.get("http://localhost:9100/api/scanner/ALPACA/presets", timeout=2)
        if r.status_code == 200:
            pass  # Scanner available
    except:
        pass

    # Default warrior trading targets
    return ["VOR", "MAMA", "USGO", "KGC", "AG", "HL", "EDIT"]


if __name__ == "__main__":
    symbols = get_hot_symbols()
    scalper = HFTScalperREST(symbols)
    scalper.run()
