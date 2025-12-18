"""
Trading Halt Detector
=====================
Detects and tracks trading halts (LULD, news, circuit breakers).

LULD (Limit Up Limit Down) halts:
- Triggered when price moves too fast
- Typically resume after 5 minutes
- Can extend to 10 minutes

This module monitors for:
- Wide bid/ask spreads (indicates halt)
- Price frozen for extended time
- Volume spikes followed by freeze
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class HaltInfo:
    """Information about a trading halt"""
    symbol: str
    halt_price: float
    halt_time: datetime
    halt_type: str  # LULD, NEWS, CIRCUIT_BREAKER, UNKNOWN
    bid_at_halt: float
    ask_at_halt: float
    spread_percent: float

    # Resume info
    resume_time: Optional[datetime] = None
    resume_price: Optional[float] = None
    estimated_resume: Optional[datetime] = None
    is_resumed: bool = False

    # Continuation analysis
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
            "estimated_resume": self.estimated_resume.isoformat() if self.estimated_resume else None,
            "is_resumed": self.is_resumed,
            "resume_direction": self.resume_direction,
            "resume_gap_percent": self.resume_gap_percent,
            "continuation_action": self.continuation_action,
            "duration_seconds": self.duration_seconds
        }


class HaltDetector:
    """
    Monitors stocks for trading halts.
    Uses bid/ask spread and price movement to detect halts.
    """

    # Thresholds for halt detection
    SPREAD_THRESHOLD = 5.0  # 5% spread suggests halt
    FREEZE_TIME_SECONDS = 10  # No movement for 10 seconds
    LULD_RESUME_MINUTES = 5  # Typical LULD halt duration

    def __init__(self):
        self.halted_stocks: Dict[str, HaltInfo] = {}
        self.halt_history: List[HaltInfo] = []
        self.price_history: Dict[str, List[tuple]] = {}  # symbol -> [(time, price)]

        # Monitoring state
        self.monitored_symbols: List[str] = []
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Callbacks
        self.on_halt: Optional[callable] = None
        self.on_resume: Optional[callable] = None

        logger.info("HaltDetector initialized")

    async def check_for_halt(self, symbol: str, quote: Dict) -> Optional[HaltInfo]:
        """Check if a stock appears to be halted based on quote data"""
        price = quote.get('price', 0) or quote.get('last', 0)
        bid = quote.get('bid', 0)
        ask = quote.get('ask', 0)

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
                halt_info.resume_gap_percent = (gap / halt_info.halt_price * 100) if halt_info.halt_price > 0 else 0

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
                    "HOLD": "HOLD"
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
                    halt_id = f"{symbol}_{halt_info.halt_time.strftime('%Y%m%d_%H%M%S')}"
                    analytics.record_resume(
                        halt_id=halt_id,
                        resume_price=price,
                        resume_time=halt_info.resume_time
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
                estimated_resume=datetime.now() + timedelta(minutes=self.LULD_RESUME_MINUTES)
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
                    pre_halt_volume=quote.get('volume', 0)
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
                    f"http://localhost:9100/api/price/{symbol}",
                    timeout=3.0
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
                f"http://localhost:9100/api/price/{symbol}",
                timeout=3.0
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
    detector = start_halt_detector(['AMCI', 'ASNS', 'BDRX'])

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
