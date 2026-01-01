"""
Trading Halt Detector with LULD Band Calculations
===================================================
Detects and tracks trading halts (LULD, news, circuit breakers).

LULD (Limit Up Limit Down) Rules (Ross Cameron):
- Price must stay at band for 15 SECONDS to trigger halt
- 10-12 seconds = "false halt" (stock pops back)
- 5-minute typical halt duration, can extend

Band Calculations by Price Tier:
- Tier 1 (S&P 500, Russell 1000): +/- 5%
- Tier 2 (Other NMS): +/- 10%
- Under $3.00: +/- 20%
- Under $0.75: Lesser of $0.15 or 75%

This module monitors for:
- LULD band proximity (warn when approaching)
- Halt trigger countdown (15 sec timer)
- False halt detection (10-12 sec bounce)
- Resume direction prediction
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LULDBands:
    """LULD band calculations for a stock"""

    symbol: str
    reference_price: float  # Opening price or last consolidated
    tier: int  # 1 = S&P/Russell, 2 = Other NMS

    # Calculated bands
    upper_band: float = 0.0
    lower_band: float = 0.0
    band_percent: float = 0.0

    # Current status
    current_price: float = 0.0
    distance_to_upper_pct: float = 0.0
    distance_to_lower_pct: float = 0.0
    is_near_band: bool = False  # Within 2% of band
    at_band_since: Optional[datetime] = None

    def calculate_bands(self):
        """Calculate LULD bands based on price tier"""
        ref = self.reference_price

        if ref < 0.75:
            # Under $0.75: Lesser of $0.15 or 75%
            pct_band = ref * 0.75
            fixed_band = 0.15
            self.band_percent = min(pct_band, fixed_band) / ref if ref > 0 else 0
        elif ref < 3.00:
            # Under $3.00: +/- 20%
            self.band_percent = 0.20
        elif self.tier == 1:
            # Tier 1 (S&P 500, Russell 1000): +/- 5%
            self.band_percent = 0.05
        else:
            # Tier 2 (Other NMS): +/- 10%
            self.band_percent = 0.10

        self.upper_band = ref * (1 + self.band_percent)
        self.lower_band = ref * (1 - self.band_percent)

    def update_price(self, price: float):
        """Update current price and calculate distances"""
        self.current_price = price

        if self.upper_band > 0:
            self.distance_to_upper_pct = (self.upper_band - price) / price * 100
        if self.lower_band > 0:
            self.distance_to_lower_pct = (price - self.lower_band) / price * 100

        # Check if near band (within 2%)
        self.is_near_band = (
            self.distance_to_upper_pct < 2.0 or self.distance_to_lower_pct < 2.0
        )

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "reference_price": self.reference_price,
            "tier": self.tier,
            "upper_band": round(self.upper_band, 4),
            "lower_band": round(self.lower_band, 4),
            "band_percent": round(self.band_percent * 100, 1),
            "current_price": self.current_price,
            "distance_to_upper_pct": round(self.distance_to_upper_pct, 2),
            "distance_to_lower_pct": round(self.distance_to_lower_pct, 2),
            "is_near_band": self.is_near_band,
        }


@dataclass
class HaltCountdown:
    """Tracks 15-second halt trigger countdown"""

    symbol: str
    band_hit_time: datetime
    band_type: str  # 'upper' or 'lower'
    band_price: float

    # Countdown status
    seconds_at_band: float = 0.0
    is_false_halt: bool = False  # Bounced before 15s
    halt_triggered: bool = False  # Hit 15s

    def update(self, price: float, band: LULDBands) -> str:
        """Update countdown, return status"""
        now = datetime.now()
        self.seconds_at_band = (now - self.band_hit_time).total_seconds()

        # Check if still at band
        still_at_band = False
        if self.band_type == "upper" and price >= band.upper_band * 0.998:
            still_at_band = True
        elif self.band_type == "lower" and price <= band.lower_band * 1.002:
            still_at_band = True

        if not still_at_band:
            # Price moved away from band
            if 10 <= self.seconds_at_band < 15:
                self.is_false_halt = True
                return "FALSE_HALT"
            else:
                return "CLEARED"

        if self.seconds_at_band >= 15:
            self.halt_triggered = True
            return "HALT_TRIGGERED"

        return f"COUNTDOWN_{int(15 - self.seconds_at_band)}"


@dataclass
class HaltInfo:
    """Information about a trading halt"""

    symbol: str
    halt_price: float
    halt_time: datetime
    halt_type: str  # LULD_UP, LULD_DOWN, NEWS, CIRCUIT_BREAKER, UNKNOWN
    bid_at_halt: float
    ask_at_halt: float
    spread_percent: float

    # LULD specific
    halt_direction: str = ""  # UP (hit upper band) or DOWN (hit lower band)
    consecutive_halts: int = 1
    pre_halt_change_pct: float = 0.0

    # Resume info
    resume_time: Optional[datetime] = None
    resume_price: Optional[float] = None
    estimated_resume: Optional[datetime] = None
    is_resumed: bool = False

    # Continuation analysis (Ross Cameron rules)
    resume_direction: str = ""  # BULLISH, BEARISH, NEUTRAL
    resume_gap_percent: float = 0.0
    continuation_action: str = ""  # BUY, SELL, HOLD, WATCH

    # Stats
    duration_seconds: int = 0

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "halt_price": self.halt_price,
            "halt_time": self.halt_time.isoformat(),
            "halt_type": self.halt_type,
            "bid_at_halt": self.bid_at_halt,
            "ask_at_halt": self.ask_at_halt,
            "spread_percent": self.spread_percent,
            "resume_time": self.resume_time.isoformat() if self.resume_time else None,
            "resume_price": self.resume_price,
            "estimated_resume": (
                self.estimated_resume.isoformat() if self.estimated_resume else None
            ),
            "is_resumed": self.is_resumed,
            "resume_direction": self.resume_direction,
            "resume_gap_percent": self.resume_gap_percent,
            "continuation_action": self.continuation_action,
            "duration_seconds": self.duration_seconds,
        }


class HaltDetector:
    """
    Monitors stocks for trading halts with LULD band tracking.
    Uses bid/ask spread, LULD bands, and countdown timer.

    Ross Cameron Rules:
    - 15 seconds at band = halt trigger
    - 10-12 seconds = false halt (pops back)
    - Down halt opens flat = LONG
    - Up halt opens flat after 2-3 halts = SHORT bias
    """

    # Thresholds for halt detection
    SPREAD_THRESHOLD = 5.0  # 5% spread suggests halt
    FREEZE_TIME_SECONDS = 10  # No movement for 10 seconds
    LULD_RESUME_MINUTES = 5  # Typical LULD halt duration
    HALT_TRIGGER_SECONDS = 15  # Seconds at band to trigger halt
    FALSE_HALT_MIN_SECONDS = 10  # Min seconds for false halt

    def __init__(self):
        self.halted_stocks: Dict[str, HaltInfo] = {}
        self.halt_history: List[HaltInfo] = []
        self.price_history: Dict[str, List[tuple]] = {}  # symbol -> [(time, price)]

        # LULD tracking
        self.luld_bands: Dict[str, LULDBands] = {}
        self.halt_countdowns: Dict[str, HaltCountdown] = {}
        self.false_halts: List[Dict] = []  # Track false halts for pattern analysis

        # Consecutive halt tracking
        self.halt_counts: Dict[str, int] = {}  # symbol -> count today

        # Monitoring state
        self.monitored_symbols: List[str] = []
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Callbacks
        self.on_halt: Optional[callable] = None
        self.on_resume: Optional[callable] = None
        self.on_false_halt: Optional[callable] = None
        self.on_approaching_band: Optional[callable] = None

        logger.info("HaltDetector initialized with LULD band tracking")

    def calculate_luld_bands(
        self, symbol: str, reference_price: float, tier: int = 2
    ) -> LULDBands:
        """
        Calculate LULD bands for a symbol.

        Args:
            symbol: Stock symbol
            reference_price: Opening price or last consolidated price
            tier: 1 = S&P/Russell 1000, 2 = Other NMS stocks

        Returns:
            LULDBands with calculated upper/lower limits
        """
        bands = LULDBands(symbol=symbol, reference_price=reference_price, tier=tier)
        bands.calculate_bands()
        self.luld_bands[symbol] = bands

        logger.debug(
            f"LULD bands for {symbol}: "
            f"${bands.lower_band:.2f} - ${bands.upper_band:.2f} "
            f"({bands.band_percent*100:.0f}% from ${reference_price:.2f})"
        )

        return bands

    def check_band_proximity(self, symbol: str, price: float) -> Dict:
        """
        Check if price is approaching LULD bands.

        Returns warning if within 2% of band.
        """
        if symbol not in self.luld_bands:
            return {"warning": False}

        bands = self.luld_bands[symbol]
        bands.update_price(price)

        result = {
            "symbol": symbol,
            "warning": bands.is_near_band,
            "bands": bands.to_dict(),
        }

        if bands.is_near_band:
            if bands.distance_to_upper_pct < 2.0:
                result["approaching"] = "upper"
                result["distance_pct"] = bands.distance_to_upper_pct
                result["band_price"] = bands.upper_band
            else:
                result["approaching"] = "lower"
                result["distance_pct"] = bands.distance_to_lower_pct
                result["band_price"] = bands.lower_band

            logger.warning(
                f"LULD WARNING: {symbol} ${price:.2f} "
                f"approaching {result['approaching']} band "
                f"${result['band_price']:.2f} ({result['distance_pct']:.1f}% away)"
            )

            if self.on_approaching_band:
                try:
                    self.on_approaching_band(result)
                except Exception as e:
                    logger.error(f"Band approach callback error: {e}")

        return result

    def start_halt_countdown(self, symbol: str, band_type: str, price: float):
        """Start 15-second countdown when price hits band"""
        if symbol not in self.luld_bands:
            return

        bands = self.luld_bands[symbol]

        countdown = HaltCountdown(
            symbol=symbol,
            band_hit_time=datetime.now(),
            band_type=band_type,
            band_price=bands.upper_band if band_type == "upper" else bands.lower_band,
        )
        self.halt_countdowns[symbol] = countdown

        logger.warning(
            f"HALT COUNTDOWN STARTED: {symbol} hit {band_type} band "
            f"${countdown.band_price:.2f} - 15 seconds to halt"
        )

    def update_halt_countdown(self, symbol: str, price: float) -> Optional[str]:
        """
        Update halt countdown and check status.

        Returns: Status string or None
        - "FALSE_HALT" - Price bounced before 15s
        - "HALT_TRIGGERED" - 15 seconds reached, halt triggered
        - "COUNTDOWN_X" - X seconds remaining
        - "CLEARED" - Price moved away
        """
        if symbol not in self.halt_countdowns:
            return None

        if symbol not in self.luld_bands:
            return None

        countdown = self.halt_countdowns[symbol]
        bands = self.luld_bands[symbol]

        status = countdown.update(price, bands)

        if status == "FALSE_HALT":
            logger.info(
                f"FALSE HALT: {symbol} bounced after {countdown.seconds_at_band:.1f}s "
                f"at {countdown.band_type} band"
            )
            self.false_halts.append(
                {
                    "symbol": symbol,
                    "time": datetime.now(),
                    "band_type": countdown.band_type,
                    "seconds": countdown.seconds_at_band,
                }
            )
            del self.halt_countdowns[symbol]

            if self.on_false_halt:
                try:
                    self.on_false_halt(
                        {
                            "symbol": symbol,
                            "band_type": countdown.band_type,
                            "seconds": countdown.seconds_at_band,
                        }
                    )
                except Exception as e:
                    logger.error(f"False halt callback error: {e}")

        elif status == "CLEARED":
            del self.halt_countdowns[symbol]

        elif status == "HALT_TRIGGERED":
            del self.halt_countdowns[symbol]
            # Halt will be detected by spread analysis

        return status

    def predict_resume_direction(self, halt_info: HaltInfo) -> Dict:
        """
        Predict resume direction based on Ross Cameron's rules.

        Rules:
        - Down halt opens flat = LONG opportunity
        - Up halt opens flat after 2-3 halts = SHORT bias
        - First halt usually continues in direction
        - Multiple halts = exhaustion likely
        """
        prediction = {
            "symbol": halt_info.symbol,
            "halt_direction": halt_info.halt_direction,
            "consecutive_halts": halt_info.consecutive_halts,
            "predicted_action": "WAIT",
            "confidence": 0.5,
            "reason": "",
        }

        if halt_info.halt_direction == "UP":
            if halt_info.consecutive_halts == 1:
                # First up halt - usually continues up
                prediction["predicted_action"] = "BUY_BREAKOUT"
                prediction["confidence"] = 0.6
                prediction["reason"] = "First up halt often continues momentum"
            elif halt_info.consecutive_halts >= 3:
                # 3+ up halts - exhaustion
                prediction["predicted_action"] = "FADE"
                prediction["confidence"] = 0.7
                prediction["reason"] = "Multiple halts = buyer exhaustion, fade if flat"
            else:
                prediction["predicted_action"] = "WATCH"
                prediction["confidence"] = 0.5
                prediction["reason"] = "Second halt - wait for direction"

        elif halt_info.halt_direction == "DOWN":
            if halt_info.consecutive_halts == 1:
                # First down halt
                prediction["predicted_action"] = "WATCH_BOUNCE"
                prediction["confidence"] = 0.55
                prediction["reason"] = "Down halt opening flat = potential long"
            else:
                prediction["predicted_action"] = "AVOID"
                prediction["confidence"] = 0.7
                prediction["reason"] = "Multiple down halts = dangerous, avoid"

        return prediction

    def get_luld_status(self, symbol: str) -> Optional[Dict]:
        """Get LULD band status for a symbol"""
        if symbol not in self.luld_bands:
            return None

        bands = self.luld_bands[symbol]
        result = bands.to_dict()

        # Add countdown info if active
        if symbol in self.halt_countdowns:
            countdown = self.halt_countdowns[symbol]
            result["countdown"] = {
                "active": True,
                "seconds_at_band": countdown.seconds_at_band,
                "seconds_remaining": max(0, 15 - countdown.seconds_at_band),
                "band_type": countdown.band_type,
            }

        return result

    async def check_for_halt(self, symbol: str, quote: Dict) -> Optional[HaltInfo]:
        """Check if a stock appears to be halted based on quote data"""
        price = quote.get("price", 0) or quote.get("last", 0)
        bid = quote.get("bid", 0)
        ask = quote.get("ask", 0)

        if not price or not bid or not ask:
            return None

        # Calculate spread percentage
        mid = (bid + ask) / 2 if bid and ask else price
        spread = ask - bid
        spread_percent = (spread / mid * 100) if mid > 0 else 0

        # Check if already tracking this halt
        if symbol in self.halted_stocks:
            halt_info = self.halted_stocks[symbol]

            # Check if resumed (spread normalized)
            if spread_percent < self.SPREAD_THRESHOLD * 0.5:
                halt_info.is_resumed = True
                halt_info.resume_time = datetime.now()
                halt_info.resume_price = price
                halt_info.duration_seconds = int(
                    (halt_info.resume_time - halt_info.halt_time).total_seconds()
                )

                # Calculate continuation direction
                gap = price - halt_info.halt_price
                halt_info.resume_gap_percent = (
                    (gap / halt_info.halt_price * 100)
                    if halt_info.halt_price > 0
                    else 0
                )

                # Determine direction and action
                if halt_info.resume_gap_percent > 2:
                    halt_info.resume_direction = "BULLISH"
                    halt_info.continuation_action = "BUY"
                elif halt_info.resume_gap_percent < -2:
                    halt_info.resume_direction = "BEARISH"
                    halt_info.continuation_action = "SELL"
                elif halt_info.resume_gap_percent > 0.5:
                    halt_info.resume_direction = "BULLISH"
                    halt_info.continuation_action = "WATCH LONG"
                elif halt_info.resume_gap_percent < -0.5:
                    halt_info.resume_direction = "BEARISH"
                    halt_info.continuation_action = "WATCH SHORT"
                else:
                    halt_info.resume_direction = "NEUTRAL"
                    halt_info.continuation_action = "HOLD"

                action_emoji = {
                    "BUY": "BUY",
                    "SELL": "SELL",
                    "WATCH LONG": "WATCH+",
                    "WATCH SHORT": "WATCH-",
                    "HOLD": "HOLD",
                }

                logger.warning(
                    f"\nHALT RESUMED: {symbol}\n"
                    f"  Duration: {halt_info.duration_seconds}s\n"
                    f"  Halt: ${halt_info.halt_price:.2f} -> Resume: ${price:.2f}\n"
                    f"  Gap: {halt_info.resume_gap_percent:+.1f}%\n"
                    f"  Direction: {halt_info.resume_direction}\n"
                    f"  ACTION: {action_emoji.get(halt_info.continuation_action, halt_info.continuation_action)}"
                )

                # Record resume to analytics
                try:
                    from ai.halt_analytics import get_halt_analytics

                    analytics = get_halt_analytics()
                    halt_id = (
                        f"{symbol}_{halt_info.halt_time.strftime('%Y%m%d_%H%M%S')}"
                    )
                    analytics.record_resume(
                        halt_id=halt_id,
                        resume_price=price,
                        resume_time=halt_info.resume_time,
                    )
                except Exception as e:
                    logger.debug(f"Analytics resume record error: {e}")

                # Move to history
                self.halt_history.append(halt_info)
                del self.halted_stocks[symbol]

                if self.on_resume:
                    try:
                        self.on_resume(halt_info)
                    except Exception as e:
                        logger.error(f"Resume callback error: {e}")

            return halt_info

        # Detect new halt
        if spread_percent >= self.SPREAD_THRESHOLD:
            # Determine halt type
            if spread_percent > 20:
                halt_type = "NEWS"  # Major news halts have huge spreads
            elif spread_percent > 10:
                halt_type = "LULD"  # Limit Up Limit Down
            else:
                halt_type = "VOLATILITY"

            halt_info = HaltInfo(
                symbol=symbol,
                halt_price=price,
                halt_time=datetime.now(),
                halt_type=halt_type,
                bid_at_halt=bid,
                ask_at_halt=ask,
                spread_percent=spread_percent,
                estimated_resume=datetime.now()
                + timedelta(minutes=self.LULD_RESUME_MINUTES),
            )

            self.halted_stocks[symbol] = halt_info

            logger.warning(
                f"HALT DETECTED: {symbol} - "
                f"Type: {halt_type} - "
                f"Price: ${price:.2f} - "
                f"Spread: {spread_percent:.1f}% (${bid:.2f} / ${ask:.2f}) - "
                f"Est Resume: {halt_info.estimated_resume.strftime('%H:%M:%S')}"
            )

            # Record to analytics for strategy building
            try:
                from ai.halt_analytics import get_halt_analytics

                analytics = get_halt_analytics()
                analytics.record_halt(
                    symbol=symbol,
                    halt_price=price,
                    halt_time=halt_info.halt_time,
                    halt_type=halt_type,
                    pre_halt_price_5min=price * 0.95,  # Estimate, would need historical
                    pre_halt_volume=quote.get("volume", 0),
                )
            except Exception as e:
                logger.debug(f"Analytics record error: {e}")

            if self.on_halt:
                try:
                    self.on_halt(halt_info)
                except Exception as e:
                    logger.error(f"Halt callback error: {e}")

            return halt_info

        return None

    def get_halt_status(self, symbol: str) -> Optional[Dict]:
        """Get halt status for a symbol"""
        if symbol in self.halted_stocks:
            halt_info = self.halted_stocks[symbol]
            halt_info.duration_seconds = int(
                (datetime.now() - halt_info.halt_time).total_seconds()
            )
            return halt_info.to_dict()
        return None

    def get_all_halts(self) -> List[Dict]:
        """Get all currently halted stocks"""
        result = []
        now = datetime.now()

        for symbol, halt_info in self.halted_stocks.items():
            halt_info.duration_seconds = int(
                (now - halt_info.halt_time).total_seconds()
            )
            result.append(halt_info.to_dict())

        return result

    def get_halt_history(self, limit: int = 20) -> List[Dict]:
        """Get recent halt history"""
        return [h.to_dict() for h in self.halt_history[-limit:]]

    def start_monitoring(self, symbols: List[str]):
        """Start monitoring symbols for halts"""
        if self.is_running:
            # Just update the list
            self.monitored_symbols = symbols
            return

        self.monitored_symbols = symbols
        self.is_running = True

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        logger.info(f"HaltDetector started - monitoring {len(symbols)} symbols")

    def stop_monitoring(self):
        """Stop halt monitoring"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("HaltDetector stopped")

    def _run_loop(self):
        """Background monitoring loop"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._monitor_loop())
        except Exception as e:
            logger.error(f"HaltDetector loop error: {e}")
        finally:
            self._loop.close()

    async def _monitor_loop(self):
        """Monitor symbols for halts"""
        while self.is_running:
            try:
                for symbol in self.monitored_symbols[:20]:  # Limit to prevent overload
                    quote = await self._get_quote(symbol)
                    if quote:
                        await self.check_for_halt(symbol, quote)

                await asyncio.sleep(2)  # Check every 2 seconds

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(5)

    async def _get_quote(self, symbol: str) -> Optional[Dict]:
        """Get quote for a symbol"""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:9100/api/price/{symbol}", timeout=3.0
                )
                if response.status_code == 200:
                    return response.json()
        except:
            pass
        return None


# Singleton
_halt_detector: Optional[HaltDetector] = None


def get_halt_detector() -> HaltDetector:
    """Get or create halt detector singleton"""
    global _halt_detector
    if _halt_detector is None:
        _halt_detector = HaltDetector()
    return _halt_detector


def start_halt_detector(symbols: List[str]) -> HaltDetector:
    """Start halt detector with symbols"""
    detector = get_halt_detector()
    detector.start_monitoring(symbols)
    return detector


def stop_halt_detector():
    """Stop halt detector"""
    detector = get_halt_detector()
    detector.stop_monitoring()


# Quick check function for API use
async def check_halt(symbol: str) -> Optional[Dict]:
    """Quick check if a symbol is halted"""
    detector = get_halt_detector()

    # Check if already tracked
    status = detector.get_halt_status(symbol)
    if status:
        return status

    # Get fresh quote and check
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://localhost:9100/api/price/{symbol}", timeout=3.0
            )
            if response.status_code == 200:
                quote = response.json()
                halt_info = await detector.check_for_halt(symbol, quote)
                if halt_info:
                    return halt_info.to_dict()
    except:
        pass

    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test
    detector = start_halt_detector(["AMCI", "ASNS", "BDRX"])

    print("Monitoring for halts... Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(5)
            halts = detector.get_all_halts()
            if halts:
                print(f"Current halts: {halts}")
    except KeyboardInterrupt:
        stop_halt_detector()
        print("Stopped")
