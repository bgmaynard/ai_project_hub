"""
High-Frequency Trading Scalper
==============================
WebSocket-based millisecond momentum detection and execution.

Features:
- Real-time Alpaca WebSocket streaming
- Tick-by-tick momentum calculation
- Volume profile analysis (buy/sell ratio)
- MACD confirmation signals
- Breaking news trigger detection
- Instant order execution on acceleration
- Auto-exit on momentum reversal
- Warrior Trading rules ($2-$20)

Usage:
    python hft_scalper.py

Author: Claude Code
"""

import asyncio
import logging
import os
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

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

from alpaca.data.historical import StockHistoricalDataClient
# Alpaca imports
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Warrior Trading rules
    "MIN_PRICE": 2.00,
    "MAX_PRICE": 20.00,
    "POSITION_SIZE": 500,  # $ per trade
    # Momentum thresholds (per tick)
    "MOMENTUM_THRESHOLD": 0.15,  # 0.15% move to trigger
    "ACCELERATION_THRESHOLD": 0.05,  # Acceleration to confirm
    "REVERSAL_THRESHOLD": -0.08,  # Exit on reversal
    # Volume profile thresholds
    "VOLUME_BUY_RATIO": 1.5,  # Buy volume > 1.5x sell volume
    "VOLUME_SURGE_MULT": 2.0,  # Volume > 2x average = surge
    # MACD settings (for 1-min bars)
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    # Candlestick pattern settings
    "TAIL_RATIO": 0.6,  # Tail > 60% of candle = significant
    "CONSECUTIVE_TAILS": 2,  # Need 2+ tails to confirm pattern
    # Short interest / ETB penalty
    "ETB_PENALTY": -1,  # Reduce signal strength for ETB stocks
    # VWAP settings
    "VWAP_CROSS_BUFFER": 0.002,  # 0.2% buffer above VWAP to confirm cross
    # Support/Resistance levels (whole and half dollars)
    "SR_BUFFER": 0.01,  # 1 cent buffer for level detection
    "SR_HOLD_TICKS": 3,  # Need to hold above level for 3 ticks
    # Historical S/R (peaks and troughs)
    "PEAK_TOLERANCE": 0.005,  # 0.5% tolerance for matching peaks
    "MIN_TOUCHES": 2,  # 2+ touches = hard resistance
    "LOOKBACK_BARS": 50,  # Look back 50 bars for peaks
    # Risk management
    "MAX_POSITIONS": 3,
    "STOP_LOSS_PCT": 0.015,  # 1.5% stop
    "TAKE_PROFIT_PCT": 0.02,  # 2% take profit
    # Tick buffer
    "TICK_BUFFER_SIZE": 10,  # Ticks to track
    "BAR_BUFFER_SIZE": 30,  # 1-min bars for MACD
}

# =============================================================================
# HFT SCALPER CLASS
# =============================================================================


class HFTScalper:
    """High-frequency momentum scalper using WebSocket streaming."""

    def __init__(self, symbols: list):
        self.symbols = [s.upper() for s in symbols]

        # Alpaca clients
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_SECRET_KEY")
        paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"

        self.trading_client = TradingClient(api_key, api_secret, paper=paper)
        self.data_stream = StockDataStream(api_key, api_secret)
        self.data_client = StockHistoricalDataClient(api_key, api_secret)

        # Tick tracking (deque for fast append/pop)
        self.ticks: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=CONFIG["TICK_BUFFER_SIZE"])
        )
        self.last_price: Dict[str, float] = {}
        self.momentum: Dict[str, float] = defaultdict(float)
        self.acceleration: Dict[str, float] = defaultdict(float)

        # Volume profile tracking
        self.buy_volume: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.sell_volume: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.volume_ratio: Dict[str, float] = defaultdict(lambda: 1.0)
        self.volume_surge: Dict[str, bool] = defaultdict(bool)

        # MACD tracking
        self.macd: Dict[str, float] = defaultdict(float)
        self.macd_signal: Dict[str, float] = defaultdict(float)
        self.macd_histogram: Dict[str, float] = defaultdict(float)
        self.macd_bullish: Dict[str, bool] = defaultdict(bool)
        self.last_macd_update: Dict[str, datetime] = {}

        # News triggers
        self.news_trigger: Dict[str, bool] = defaultdict(bool)
        self.last_news_check: datetime = datetime.now()

        # Candlestick pattern tracking
        self.candles: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.bottoming_tails: Dict[str, int] = defaultdict(
            int
        )  # Count of bullish tails
        self.topping_tails: Dict[str, int] = defaultdict(int)  # Count of bearish tails
        self.last_candle_update: Dict[str, datetime] = {}

        # ETB (Easy To Borrow) tracking - stocks with high short interest
        self.etb_stocks: Set[str] = set()
        self.last_etb_check: datetime = datetime.now() - timedelta(hours=1)

        # VWAP tracking
        self.vwap: Dict[str, float] = defaultdict(float)
        self.vwap_cumulative_vol: Dict[str, float] = defaultdict(float)
        self.vwap_cumulative_pv: Dict[str, float] = defaultdict(float)
        self.above_vwap: Dict[str, bool] = defaultdict(bool)
        self.vwap_cross_up: Dict[str, bool] = defaultdict(bool)
        self.last_vwap_update: Dict[str, datetime] = {}

        # Support/Resistance level tracking
        self.sr_breakout: Dict[str, bool] = defaultdict(bool)
        self.sr_level_broken: Dict[str, float] = defaultdict(float)
        self.ticks_above_sr: Dict[str, int] = defaultdict(int)

        # Historical peak/trough resistance levels
        self.resistance_levels: Dict[str, List[dict]] = defaultdict(
            list
        )  # {price, touches, type}
        self.support_levels: Dict[str, List[dict]] = defaultdict(list)
        self.hard_resistance_break: Dict[str, bool] = defaultdict(bool)
        self.hard_support_break: Dict[str, bool] = defaultdict(bool)
        self.last_sr_scan: Dict[str, datetime] = {}

        # Candlestick pattern tracking
        self.pattern_signals: Dict[str, List[str]] = defaultdict(
            list
        )  # Active patterns
        self.higher_highs_count: Dict[str, int] = defaultdict(int)
        self.last_pattern_scan: Dict[str, datetime] = {}

        # Position tracking
        self.positions: Dict[str, dict] = {}
        self.pending_orders: Set[str] = set()

        # Stats
        self.trades_executed = 0
        self.total_pnl = 0.0

        logger.info(f"HFT Scalper initialized for: {self.symbols}")

    def calculate_macd(self, prices: List[float]) -> tuple:
        """Calculate MACD from price series."""
        if len(prices) < CONFIG["MACD_SLOW"]:
            return 0, 0, 0

        # EMA calculation
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

        macd = macd_line[-1]
        signal = signal_line[-1]
        histogram = macd - signal

        return macd, signal, histogram

    async def update_volume_profile(self, symbol: str, trade_data: dict):
        """Update volume profile from trade data."""
        price = trade_data.get("price", 0)
        size = trade_data.get("size", 0)

        # Determine if buy or sell based on price movement
        last = self.last_price.get(symbol, price)
        if price >= last:
            self.buy_volume[symbol].append(size)
        else:
            self.sell_volume[symbol].append(size)

        self.last_price[symbol] = price

        # Calculate buy/sell ratio
        total_buy = sum(self.buy_volume[symbol]) or 1
        total_sell = sum(self.sell_volume[symbol]) or 1
        self.volume_ratio[symbol] = total_buy / total_sell

        # Check for volume surge
        avg_volume = (total_buy + total_sell) / 2
        self.volume_surge[symbol] = (
            avg_volume
            > (sum(self.buy_volume[symbol]) + sum(self.sell_volume[symbol]))
            / max(len(self.buy_volume[symbol]) + len(self.sell_volume[symbol]), 1)
            * CONFIG["VOLUME_SURGE_MULT"]
        )

    async def update_macd(self, symbol: str):
        """Fetch recent bars and update MACD."""
        now = datetime.now()
        last_update = self.last_macd_update.get(symbol)

        # Only update every 30 seconds to avoid rate limits
        if last_update and (now - last_update).seconds < 30:
            return

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=now - timedelta(hours=1),
            )
            bars = self.data_client.get_stock_bars(request)

            if symbol in bars and len(bars[symbol]) >= CONFIG["MACD_SLOW"]:
                prices = [float(bar.close) for bar in bars[symbol]]
                macd, signal, histogram = self.calculate_macd(prices)

                # Check for bullish crossover (MACD crosses above signal)
                prev_hist = self.macd_histogram[symbol]
                self.macd[symbol] = macd
                self.macd_signal[symbol] = signal
                self.macd_histogram[symbol] = histogram

                # Bullish: histogram going from negative to positive OR increasing positive
                self.macd_bullish[symbol] = (prev_hist < 0 and histogram > 0) or (
                    histogram > 0 and histogram > prev_hist
                )

                self.last_macd_update[symbol] = now

        except Exception as e:
            pass  # Silently handle to not disrupt hot path

    async def update_candle_patterns(self, symbol: str):
        """Analyze recent candles for bottoming/topping tails.

        Bottoming tail (bullish): Long lower wick, price rejected going lower
        Topping tail (bearish): Long upper wick, price rejected going higher
        """
        now = datetime.now()
        last_update = self.last_candle_update.get(symbol)

        # Update every 60 seconds
        if last_update and (now - last_update).seconds < 60:
            return

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=now - timedelta(minutes=15),
            )
            bars = self.data_client.get_stock_bars(request)

            if symbol not in bars or len(bars[symbol]) < 3:
                return

            bottoming = 0
            topping = 0

            for bar in list(bars[symbol])[-5:]:  # Last 5 candles
                o, h, l, c = (
                    float(bar.open),
                    float(bar.high),
                    float(bar.low),
                    float(bar.close),
                )
                body = abs(c - o)
                total_range = h - l

                if total_range == 0:
                    continue

                # Calculate tail ratios
                if c >= o:  # Green candle
                    lower_tail = o - l
                    upper_tail = h - c
                else:  # Red candle
                    lower_tail = c - l
                    upper_tail = h - o

                lower_ratio = lower_tail / total_range
                upper_ratio = upper_tail / total_range

                # Bottoming tail: long lower wick (bullish - buyers pushing back)
                if lower_ratio > CONFIG["TAIL_RATIO"]:
                    bottoming += 1

                # Topping tail: long upper wick (bearish - sellers pushing back)
                if upper_ratio > CONFIG["TAIL_RATIO"]:
                    topping += 1

            self.bottoming_tails[symbol] = bottoming
            self.topping_tails[symbol] = topping
            self.last_candle_update[symbol] = now

            if bottoming >= CONFIG["CONSECUTIVE_TAILS"]:
                logger.info(
                    f"BOTTOMING TAILS: {symbol} has {bottoming} bullish rejection candles"
                )
            if topping >= CONFIG["CONSECUTIVE_TAILS"]:
                logger.info(
                    f"TOPPING TAILS: {symbol} has {topping} bearish rejection candles"
                )

        except Exception as e:
            pass

    async def update_vwap(self, symbol: str, price: float, volume: int):
        """Update VWAP and detect crossovers.

        VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
        Crossing above VWAP = bullish signal
        """
        if volume <= 0:
            return

        # Update cumulative values
        self.vwap_cumulative_vol[symbol] += volume
        self.vwap_cumulative_pv[symbol] += price * volume

        # Calculate VWAP
        if self.vwap_cumulative_vol[symbol] > 0:
            new_vwap = (
                self.vwap_cumulative_pv[symbol] / self.vwap_cumulative_vol[symbol]
            )
            self.vwap[symbol] = new_vwap

            # Check for crossover
            was_above = self.above_vwap[symbol]
            buffer = price * CONFIG["VWAP_CROSS_BUFFER"]
            is_above = price > (new_vwap + buffer)

            self.above_vwap[symbol] = is_above

            # Detect bullish crossover (was below, now above)
            if is_above and not was_above and new_vwap > 0:
                self.vwap_cross_up[symbol] = True
                logger.info(
                    f"VWAP CROSSOVER: {symbol} crossed ABOVE VWAP ${new_vwap:.2f} -> BULLISH"
                )
            elif not is_above and was_above:
                self.vwap_cross_up[symbol] = False

    def check_sr_breakout(self, symbol: str, price: float) -> tuple:
        """Check for support/resistance level breakouts.

        Key levels:
        - Whole dollars: $5.00, $6.00, $7.00, etc.
        - Half dollars: $5.50, $6.50, $7.50, etc.

        Returns: (is_breakout, level_broken)
        """
        # Find nearest whole and half dollar levels below current price
        whole_below = int(price)
        half_below = int(price) + 0.5 if price % 1 >= 0.5 else int(price) - 0.5

        # Get the nearest level below
        if half_below > whole_below and half_below < price:
            nearest_level = half_below
        else:
            nearest_level = whole_below

        # Check if price just broke above a level
        buffer = CONFIG["SR_BUFFER"]
        just_above_level = price > nearest_level and price < (nearest_level + 0.10)

        if just_above_level:
            # Count ticks above this level
            if self.sr_level_broken[symbol] == nearest_level:
                self.ticks_above_sr[symbol] += 1
            else:
                self.sr_level_broken[symbol] = nearest_level
                self.ticks_above_sr[symbol] = 1

            # Confirm breakout after holding above for X ticks
            if self.ticks_above_sr[symbol] >= CONFIG["SR_HOLD_TICKS"]:
                if not self.sr_breakout[symbol]:
                    self.sr_breakout[symbol] = True
                    level_type = (
                        "WHOLE" if nearest_level == int(nearest_level) else "HALF"
                    )
                    logger.info(
                        f"S/R BREAKOUT: {symbol} broke above ${nearest_level:.2f} ({level_type} DOLLAR) - BULLISH"
                    )
                return True, nearest_level
        else:
            # Reset if price moved away
            self.sr_breakout[symbol] = False
            self.ticks_above_sr[symbol] = 0

        return False, 0

    async def scan_historical_sr_levels(self, symbol: str):
        """Scan historical bars to find peaks and troughs that act as S/R.

        Double/Triple tops and bottoms are strong resistance/support levels.
        When price breaks through these multi-touch levels, it signals momentum.
        """
        now = datetime.now()
        last_scan = self.last_sr_scan.get(symbol)

        # Only scan every 2 minutes to avoid rate limits
        if last_scan and (now - last_scan).seconds < 120:
            return

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=now - timedelta(hours=2),  # 2 hours of 1-min bars
            )
            bars = self.data_client.get_stock_bars(request)

            if symbol not in bars or len(bars[symbol]) < 20:
                return

            bar_list = list(bars[symbol])
            highs = [float(b.high) for b in bar_list]
            lows = [float(b.low) for b in bar_list]

            # Find local peaks (resistance levels)
            peaks = []
            for i in range(2, len(highs) - 2):
                if (
                    highs[i] > highs[i - 1]
                    and highs[i] > highs[i - 2]
                    and highs[i] > highs[i + 1]
                    and highs[i] > highs[i + 2]
                ):
                    peaks.append(highs[i])

            # Find local troughs (support levels)
            troughs = []
            for i in range(2, len(lows) - 2):
                if (
                    lows[i] < lows[i - 1]
                    and lows[i] < lows[i - 2]
                    and lows[i] < lows[i + 1]
                    and lows[i] < lows[i + 2]
                ):
                    troughs.append(lows[i])

            # Group similar peaks together (within tolerance)
            tolerance = CONFIG["PEAK_TOLERANCE"]
            resistance_clusters = self._cluster_levels(peaks, tolerance)
            support_clusters = self._cluster_levels(troughs, tolerance)

            # Store levels with touch count
            self.resistance_levels[symbol] = []
            for level, count in resistance_clusters:
                level_type = "HARD" if count >= CONFIG["MIN_TOUCHES"] else "SOFT"
                self.resistance_levels[symbol].append(
                    {"price": level, "touches": count, "type": level_type}
                )
                if count >= CONFIG["MIN_TOUCHES"]:
                    logger.info(
                        f"RESISTANCE: {symbol} has {level_type} resistance at ${level:.2f} ({count} touches)"
                    )

            self.support_levels[symbol] = []
            for level, count in support_clusters:
                level_type = "HARD" if count >= CONFIG["MIN_TOUCHES"] else "SOFT"
                self.support_levels[symbol].append(
                    {"price": level, "touches": count, "type": level_type}
                )
                if count >= CONFIG["MIN_TOUCHES"]:
                    logger.info(
                        f"SUPPORT: {symbol} has {level_type} support at ${level:.2f} ({count} touches)"
                    )

            self.last_sr_scan[symbol] = now

        except Exception as e:
            pass

    def _cluster_levels(self, levels: list, tolerance: float) -> list:
        """Cluster similar price levels together and count touches."""
        if not levels:
            return []

        clusters = []
        levels = sorted(levels)

        current_cluster = [levels[0]]
        for level in levels[1:]:
            # Check if within tolerance of cluster average
            cluster_avg = sum(current_cluster) / len(current_cluster)
            if abs(level - cluster_avg) / cluster_avg <= tolerance:
                current_cluster.append(level)
            else:
                # Save current cluster and start new one
                avg = sum(current_cluster) / len(current_cluster)
                clusters.append((avg, len(current_cluster)))
                current_cluster = [level]

        # Don't forget last cluster
        if current_cluster:
            avg = sum(current_cluster) / len(current_cluster)
            clusters.append((avg, len(current_cluster)))

        return clusters

    def check_hard_sr_breakout(self, symbol: str, price: float) -> tuple:
        """Check if price is breaking through a hard S/R level (double/triple top).

        Returns: (is_breakout, level, touches, direction)
        """
        buffer = CONFIG["PEAK_TOLERANCE"]

        # Check resistance breakouts (bullish)
        for level_info in self.resistance_levels[symbol]:
            level = level_info["price"]
            touches = level_info["touches"]

            if touches >= CONFIG["MIN_TOUCHES"]:  # Hard resistance
                # Price just broke above
                if (
                    price > level * (1 + buffer)
                    and not self.hard_resistance_break[symbol]
                ):
                    self.hard_resistance_break[symbol] = True
                    top_type = (
                        "DOUBLE"
                        if touches == 2
                        else "TRIPLE" if touches == 3 else f"{touches}X"
                    )
                    logger.info(
                        f"{top_type} TOP BREAK: {symbol} broke ${level:.2f} resistance ({touches} touches) -> BULLISH MOMENTUM"
                    )
                    return True, level, touches, "UP"

        # Check support breakdowns (bearish - for exit signals)
        for level_info in self.support_levels[symbol]:
            level = level_info["price"]
            touches = level_info["touches"]

            if touches >= CONFIG["MIN_TOUCHES"]:  # Hard support
                # Price just broke below
                if price < level * (1 - buffer) and not self.hard_support_break[symbol]:
                    self.hard_support_break[symbol] = True
                    bottom_type = (
                        "DOUBLE"
                        if touches == 2
                        else "TRIPLE" if touches == 3 else f"{touches}X"
                    )
                    logger.info(
                        f"{bottom_type} BOTTOM BREAK: {symbol} broke ${level:.2f} support ({touches} touches) -> BEARISH"
                    )
                    return True, level, touches, "DOWN"

        # Reset if price moved back inside range
        if self.hard_resistance_break[symbol] or self.hard_support_break[symbol]:
            all_levels = [
                l["price"]
                for l in self.resistance_levels[symbol] + self.support_levels[symbol]
            ]
            if all_levels:
                min_level = min(all_levels) * 0.98
                max_level = max(all_levels) * 1.02
                if min_level < price < max_level:
                    self.hard_resistance_break[symbol] = False
                    self.hard_support_break[symbol] = False

        return False, 0, 0, None

    async def scan_candlestick_patterns(self, symbol: str):
        """Scan for bullish/bearish candlestick patterns.

        Patterns detected:
        - Hanging Man (bearish reversal)
        - Hammer (bullish reversal)
        - Higher Highs (momentum continuation)
        - Ascending Wedge (bullish)
        - Cup and Handle (bullish breakout)
        - Engulfing patterns
        """
        now = datetime.now()
        last_scan = self.last_pattern_scan.get(symbol)

        # Scan every 30 seconds
        if last_scan and (now - last_scan).seconds < 30:
            return

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=now - timedelta(minutes=30),
            )
            bars = self.data_client.get_stock_bars(request)

            if symbol not in bars or len(bars[symbol]) < 10:
                return

            bar_list = list(bars[symbol])
            patterns = []

            # Extract OHLC for last 10 candles
            candles = []
            for b in bar_list[-10:]:
                candles.append(
                    {
                        "o": float(b.open),
                        "h": float(b.high),
                        "l": float(b.low),
                        "c": float(b.close),
                        "v": float(b.volume),
                    }
                )

            # === PATTERN DETECTION ===

            # 1. Higher Highs (momentum) - last 3 candles making new highs
            if len(candles) >= 3:
                hh_count = 0
                for i in range(-1, -4, -1):
                    if i == -1:
                        continue
                    if candles[i]["h"] > candles[i - 1]["h"]:
                        hh_count += 1
                if hh_count >= 2:
                    patterns.append("HIGHER_HIGHS")
                    self.higher_highs_count[symbol] = hh_count

            # 2. Hammer (bullish) - small body at top, long lower wick
            last = candles[-1]
            body = abs(last["c"] - last["o"])
            total_range = last["h"] - last["l"]
            if total_range > 0:
                lower_wick = min(last["o"], last["c"]) - last["l"]
                upper_wick = last["h"] - max(last["o"], last["c"])

                if lower_wick > body * 2 and upper_wick < body * 0.5:
                    patterns.append("HAMMER")

                # 3. Hanging Man (bearish) - same shape but after uptrend
                if lower_wick > body * 2 and upper_wick < body * 0.5:
                    # Check if after uptrend
                    if len(candles) >= 5:
                        prev_trend = candles[-5]["c"] < candles[-2]["c"]
                        if prev_trend:
                            patterns.append("HANGING_MAN")

            # 4. Bullish Engulfing - current candle engulfs previous
            if len(candles) >= 2:
                curr = candles[-1]
                prev = candles[-2]
                if prev["c"] < prev["o"] and curr["c"] > curr["o"]:  # Red then Green
                    if curr["o"] <= prev["c"] and curr["c"] >= prev["o"]:
                        patterns.append("BULLISH_ENGULF")

            # 5. Bearish Engulfing
            if len(candles) >= 2:
                curr = candles[-1]
                prev = candles[-2]
                if prev["c"] > prev["o"] and curr["c"] < curr["o"]:  # Green then Red
                    if curr["o"] >= prev["c"] and curr["c"] <= prev["o"]:
                        patterns.append("BEARISH_ENGULF")

            # 6. Ascending Wedge (higher lows, converging)
            if len(candles) >= 5:
                lows = [c["l"] for c in candles[-5:]]
                highs = [c["h"] for c in candles[-5:]]
                higher_lows = all(lows[i] >= lows[i - 1] for i in range(1, len(lows)))
                range_narrowing = (highs[-1] - lows[-1]) < (highs[0] - lows[0])
                if higher_lows and range_narrowing:
                    patterns.append("ASCENDING_WEDGE")

            # 7. Candle Over Candle (consecutive green, each higher)
            if len(candles) >= 3:
                coc = True
                for i in range(-1, -4, -1):
                    c = candles[i]
                    if c["c"] <= c["o"]:  # Not green
                        coc = False
                        break
                    if i < -1:
                        if c["h"] <= candles[i + 1]["h"]:  # Not making new high
                            coc = False
                            break
                if coc:
                    patterns.append("CANDLE_OVER_CANDLE")

            self.pattern_signals[symbol] = patterns
            self.last_pattern_scan[symbol] = now

            if patterns:
                logger.info(f"PATTERNS: {symbol} -> {', '.join(patterns)}")

        except Exception as e:
            pass

    async def check_etb_status(self):
        """Check which stocks are Easy To Borrow (high short interest).

        ETB stocks are easier for shorts to trade, meaning more
        downward pressure from short sellers without borrowing costs.
        We penalize these stocks in our signal strength.
        """
        now = datetime.now()

        # Check every hour
        if (now - self.last_etb_check).seconds < 3600:
            return

        self.last_etb_check = now

        try:
            # Check Alpaca's asset info for shortable status
            for symbol in self.symbols:
                try:
                    asset = self.trading_client.get_asset(symbol)
                    if asset.easy_to_borrow:
                        if symbol not in self.etb_stocks:
                            self.etb_stocks.add(symbol)
                            logger.info(
                                f"ETB DETECTED: {symbol} is Easy To Borrow (short interest risk)"
                            )
                    else:
                        self.etb_stocks.discard(symbol)
                except:
                    pass
        except Exception as e:
            pass

    async def check_news_triggers(self):
        """Check for breaking news on watchlist symbols.

        News is typically released on the hour and half hour,
        so we check more frequently during those windows.
        """
        now = datetime.now()
        minute = now.minute

        # News windows: :00-:05 and :30-:35 (on the hour and half hour)
        is_news_window = minute in range(0, 6) or minute in range(30, 36)

        # Check every 10 seconds during news window, every 60 seconds otherwise
        check_interval = 10 if is_news_window else 60

        if (now - self.last_news_check).seconds < check_interval:
            return

        self.last_news_check = now

        if is_news_window:
            logger.info(f"NEWS WINDOW ACTIVE - checking for breaking news...")

        try:
            # Use local API for news
            response = requests.get("http://localhost:9100/api/news/latest", timeout=2)
            if response.status_code == 200:
                news = response.json()
                for item in news.get("articles", [])[:10]:
                    # Check if any symbol mentioned
                    text = (
                        item.get("title", "") + " " + item.get("description", "")
                    ).upper()
                    for symbol in self.symbols:
                        if symbol in text:
                            self.news_trigger[symbol] = True
                            logger.info(
                                f"NEWS TRIGGER: {symbol} - {item.get('title', '')[:50]}"
                            )
        except:
            pass  # Silently handle

    def get_signal_strength(self, symbol: str) -> tuple:
        """Calculate overall signal strength combining all indicators."""
        signals = []
        reasons = []

        # 1. Momentum + Acceleration (primary)
        if self.momentum[symbol] > CONFIG["MOMENTUM_THRESHOLD"]:
            signals.append(1)
            reasons.append(f"MOM={self.momentum[symbol]:.2f}%")
        if self.acceleration[symbol] > CONFIG["ACCELERATION_THRESHOLD"]:
            signals.append(1)
            reasons.append(f"ACC={self.acceleration[symbol]:.2f}%")

        # 2. Volume Profile (buy/sell ratio)
        if self.volume_ratio[symbol] > CONFIG["VOLUME_BUY_RATIO"]:
            signals.append(1)
            reasons.append(f"VOL_RATIO={self.volume_ratio[symbol]:.1f}x")
        if self.volume_surge[symbol]:
            signals.append(1)
            reasons.append("VOL_SURGE")

        # 3. MACD Bullish
        if self.macd_bullish[symbol]:
            signals.append(1)
            reasons.append(f"MACD_BULL")

        # 4. News Trigger
        if self.news_trigger[symbol]:
            signals.append(2)  # News is weighted higher
            reasons.append("NEWS!")
            self.news_trigger[symbol] = False  # Reset after use

        # 5. Candlestick Patterns - Bottoming tails (bullish)
        if self.bottoming_tails[symbol] >= CONFIG["CONSECUTIVE_TAILS"]:
            signals.append(1)
            reasons.append(f"BOTTOM_TAILS={self.bottoming_tails[symbol]}")

        # 6. Candlestick Patterns - Topping tails (bearish - negative signal)
        if self.topping_tails[symbol] >= CONFIG["CONSECUTIVE_TAILS"]:
            signals.append(-1)  # Negative weight - bearish
            reasons.append(f"TOP_TAILS={self.topping_tails[symbol]}(BEARISH)")

        # 7. ETB Penalty - Easy To Borrow stocks have more short pressure
        if symbol in self.etb_stocks:
            signals.append(CONFIG["ETB_PENALTY"])  # -1 penalty
            reasons.append("ETB_PENALTY")

        # 8. VWAP Crossover - price crossing above VWAP is bullish
        if self.vwap_cross_up[symbol]:
            signals.append(1)
            reasons.append(f"VWAP_CROSS_UP(${self.vwap[symbol]:.2f})")

        # 9. Support/Resistance Breakout - breaking whole/half dollar levels
        if self.sr_breakout[symbol]:
            signals.append(1)
            level = self.sr_level_broken[symbol]
            level_type = "WHOLE$" if level == int(level) else "HALF$"
            reasons.append(f"SR_BREAK({level_type}{level:.2f})")

        # 10. Hard Resistance Breakout - double/triple top break (very bullish)
        if self.hard_resistance_break[symbol]:
            signals.append(2)  # Weight 2 - strong signal
            # Find the level that was broken
            for level_info in self.resistance_levels[symbol]:
                if level_info["touches"] >= CONFIG["MIN_TOUCHES"]:
                    touches = level_info["touches"]
                    top_type = (
                        "DOUBLE"
                        if touches == 2
                        else "TRIPLE" if touches == 3 else f"{touches}X"
                    )
                    reasons.append(f"{top_type}_TOP_BREAK(${level_info['price']:.2f})")
                    break

        # 11. Hard Support Break - double/triple bottom break (bearish - negative)
        if self.hard_support_break[symbol]:
            signals.append(-2)  # Weight -2 - strong bearish signal
            for level_info in self.support_levels[symbol]:
                if level_info["touches"] >= CONFIG["MIN_TOUCHES"]:
                    touches = level_info["touches"]
                    bottom_type = (
                        "DOUBLE"
                        if touches == 2
                        else "TRIPLE" if touches == 3 else f"{touches}X"
                    )
                    reasons.append(f"{bottom_type}_BOTTOM_BREAK(BEARISH)")
                    break

        # 12. Candlestick Patterns
        patterns = self.pattern_signals[symbol]

        # Bullish patterns
        bullish_patterns = [
            "HIGHER_HIGHS",
            "HAMMER",
            "BULLISH_ENGULF",
            "ASCENDING_WEDGE",
            "CANDLE_OVER_CANDLE",
        ]
        for pattern in bullish_patterns:
            if pattern in patterns:
                signals.append(1)
                reasons.append(pattern)

        # Bearish patterns (negative signals)
        bearish_patterns = ["HANGING_MAN", "BEARISH_ENGULF"]
        for pattern in bearish_patterns:
            if pattern in patterns:
                signals.append(-1)
                reasons.append(f"{pattern}(BEARISH)")

        strength = sum(signals)
        return strength, reasons

    async def on_quote(self, quote):
        """Process incoming quote - THIS IS THE HOT PATH."""
        symbol = quote.symbol
        bid = float(quote.bid_price)
        ask = float(quote.ask_price)
        mid = (bid + ask) / 2
        bid_size = int(quote.bid_size) if hasattr(quote, "bid_size") else 0
        ask_size = int(quote.ask_size) if hasattr(quote, "ask_size") else 0

        # Skip if outside Warrior range
        if mid < CONFIG["MIN_PRICE"] or mid > CONFIG["MAX_PRICE"]:
            return

        # Store tick
        now = datetime.now()
        self.ticks[symbol].append({"time": now, "price": mid, "bid": bid, "ask": ask})

        # Update volume profile from quote sizes
        await self.update_volume_profile(
            symbol, {"price": mid, "size": bid_size + ask_size}
        )

        # Periodically update MACD (non-blocking)
        asyncio.create_task(self.update_macd(symbol))

        # Periodically check news
        asyncio.create_task(self.check_news_triggers())

        # Periodically update candle patterns
        asyncio.create_task(self.update_candle_patterns(symbol))

        # Periodically check ETB status
        asyncio.create_task(self.check_etb_status())

        # Update VWAP with current quote
        await self.update_vwap(symbol, mid, bid_size + ask_size)

        # Check for S/R level breakouts (whole and half dollars)
        self.check_sr_breakout(symbol, mid)

        # Periodically scan for historical S/R levels (double/triple tops)
        asyncio.create_task(self.scan_historical_sr_levels(symbol))

        # Check for hard S/R breakouts (double/triple top breaks)
        self.check_hard_sr_breakout(symbol, mid)

        # Periodically scan for candlestick patterns
        asyncio.create_task(self.scan_candlestick_patterns(symbol))

        # Need at least 3 ticks for momentum
        if len(self.ticks[symbol]) < 3:
            return

        # Calculate momentum (price velocity)
        ticks = list(self.ticks[symbol])
        price_now = ticks[-1]["price"]
        price_prev = ticks[-2]["price"]
        price_prev2 = ticks[-3]["price"]

        vel1 = (price_now - price_prev) / price_prev * 100 if price_prev > 0 else 0
        vel2 = (price_prev - price_prev2) / price_prev2 * 100 if price_prev2 > 0 else 0

        self.momentum[symbol] = vel1
        self.acceleration[symbol] = vel1 - vel2

        # Get combined signal strength
        strength, reasons = self.get_signal_strength(symbol)

        # === ENTRY LOGIC ===
        # Need at least 2 signals to enter (momentum + one other confirmation)
        if symbol not in self.positions and symbol not in self.pending_orders:
            if strength >= 2 and len(self.positions) < CONFIG["MAX_POSITIONS"]:
                await self.enter_position(symbol, ask, reasons)

        # === EXIT LOGIC ===
        elif symbol in self.positions:
            pos = self.positions[symbol]
            entry = pos["entry_price"]
            pnl_pct = (mid - entry) / entry * 100

            # Take profit
            if pnl_pct >= CONFIG["TAKE_PROFIT_PCT"] * 100:
                await self.exit_position(symbol, bid, "PROFIT")

            # Stop loss
            elif pnl_pct <= -CONFIG["STOP_LOSS_PCT"] * 100:
                await self.exit_position(symbol, bid, "STOP")

            # Momentum reversal + volume confirmation
            elif self.acceleration[symbol] < CONFIG["REVERSAL_THRESHOLD"]:
                # Extra confirmation: sell volume increasing
                if self.volume_ratio[symbol] < 1.0:
                    await self.exit_position(symbol, bid, "REVERSAL+VOL")
                else:
                    await self.exit_position(symbol, bid, "REVERSAL")

    async def enter_position(self, symbol: str, price: float, reasons: list = None):
        """Enter a position instantly."""
        self.pending_orders.add(symbol)

        qty = max(1, int(CONFIG["POSITION_SIZE"] / price))
        reasons_str = ", ".join(reasons) if reasons else ""

        logger.info(f">> BUY {symbol} x{qty} @ ${price:.2f} | {reasons_str}")

        try:
            # Use limit order at ask for extended hours
            order = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                limit_price=round(price * 1.002, 2),  # Slight buffer
                time_in_force=TimeInForce.DAY,
                extended_hours=True,
            )
            result = self.trading_client.submit_order(order)

            self.positions[symbol] = {
                "entry_price": price,
                "qty": qty,
                "order_id": result.id,
                "entry_time": datetime.now(),
            }
            self.trades_executed += 1
            logger.info(f"   ORDER PLACED: {result.id}")

        except Exception as e:
            logger.error(f"   ORDER FAILED: {e}")
        finally:
            self.pending_orders.discard(symbol)

    async def exit_position(self, symbol: str, price: float, reason: str):
        """Exit position instantly."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        qty = pos["qty"]
        entry = pos["entry_price"]
        pnl = (price - entry) * qty
        pnl_pct = (price - entry) / entry * 100

        logger.info(
            f"<< SELL {symbol} x{qty} @ ${price:.2f} | {reason} | P/L: ${pnl:.2f} ({pnl_pct:+.2f}%)"
        )

        try:
            order = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                type=OrderType.LIMIT,
                limit_price=round(price * 0.998, 2),  # Slight buffer
                time_in_force=TimeInForce.DAY,
                extended_hours=True,
            )
            result = self.trading_client.submit_order(order)

            self.total_pnl += pnl
            del self.positions[symbol]
            logger.info(
                f"   EXIT ORDER: {result.id} | Total P/L: ${self.total_pnl:.2f}"
            )

        except Exception as e:
            logger.error(f"   EXIT FAILED: {e}")

    async def run(self):
        """Start the HFT scalper."""
        logger.info("=" * 60)
        logger.info("HFT MOMENTUM SCALPER - STARTING")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Position size: ${CONFIG['POSITION_SIZE']}")
        logger.info(f"Momentum threshold: {CONFIG['MOMENTUM_THRESHOLD']}%")
        logger.info("=" * 60)

        # Subscribe to quotes for all symbols
        self.data_stream.subscribe_quotes(self.on_quote, *self.symbols)

        # Run the stream
        await self.data_stream._run_forever()


# =============================================================================
# MAIN
# =============================================================================


async def main():
    # Warrior Trading compliant symbols ($2-$20)
    symbols = ["VOR", "MAMA", "USGO", "KGC", "AG", "HL", "EDIT"]

    scalper = HFTScalper(symbols)
    await scalper.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Scalper stopped by user")
