"""
Warrior News Stream - REAL-TIME WebSocket News Feed
====================================================
INSTANT news detection using Alpaca's dedicated news WebSocket stream.

NO POLLING - News arrives the instant it's published!

WebSocket: wss://stream.data.alpaca.markets/v1beta1/news

This is a DEDICATED stream for news only, separate from market data.
When breaking news hits, we get it IMMEDIATELY and can act within seconds.

WARRIOR TRADING RULES:
- News = Catalyst = MOMENTUM
- First 60 SECONDS after news = MAX opportunity
- Get in FAST, scalp the pop, GET OUT
"""

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import pytz
import websockets

logger = logging.getLogger(__name__)


@dataclass
class RealTimeNewsAlert:
    """Real-time news alert from WebSocket stream"""

    id: str
    headline: str
    summary: str
    symbols: List[str]
    source: str
    author: str
    published_at: datetime
    received_at: datetime  # When WE received it
    url: Optional[str]

    # Analysis
    urgency: str  # "critical", "high", "medium", "low"
    sentiment: str  # "bullish", "bearish", "neutral"
    sentiment_score: float

    # Scalp signal
    scalp_signal: str  # "long_now", "short_now", "monitor", "avoid"
    scalp_reason: str

    # Timing - CRITICAL for scalping
    latency_ms: float  # How fast we got the news
    time_to_act_seconds: float  # Countdown to act

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "headline": self.headline,
            "summary": self.summary,
            "symbols": self.symbols,
            "source": self.source,
            "author": self.author,
            "published_at": (
                self.published_at.isoformat() if self.published_at else None
            ),
            "received_at": self.received_at.isoformat(),
            "url": self.url,
            "urgency": self.urgency,
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "scalp_signal": self.scalp_signal,
            "scalp_reason": self.scalp_reason,
            "latency_ms": self.latency_ms,
            "time_to_act_seconds": self.time_to_act_seconds,
        }


# Critical catalyst keywords
CRITICAL_KEYWORDS = [
    "fda approval",
    "fda approves",
    "fda grants",
    "breakthrough therapy",
    "acquisition",
    "merger",
    "buyout",
    "takeover",
    "acquire",
    "earnings beat",
    "earnings surprise",
    "eps beat",
    "revenue beat",
    "upgrade",
    "price target raised",
    "raised guidance",
    "raises guidance",
    "contract win",
    "major deal",
    "partnership",
]

HIGH_KEYWORDS = [
    "analyst upgrade",
    "buy rating",
    "outperform",
    "strong buy",
    "guidance raise",
    "outlook positive",
    "beat estimates",
    "expansion",
    "launch",
    "new product",
    "clinical trial success",
]

BEARISH_KEYWORDS = [
    "fda reject",
    "trial fail",
    "earnings miss",
    "revenue miss",
    "downgrade",
    "price target cut",
    "lowered guidance",
    "lowers guidance",
    "lawsuit",
    "investigation",
    "sec",
    "fraud",
    "subpoena",
    "layoff",
    "restructuring",
    "bankruptcy",
    "default",
]


class WarriorNewsStream:
    """
    REAL-TIME WebSocket news stream from Alpaca.

    This is INSTANT - no polling delay!
    News arrives the moment it's published.
    """

    # Alpaca News WebSocket URL
    NEWS_WS_URL = "wss://stream.data.alpaca.markets/v1beta1/news"

    def __init__(self):
        self.et_tz = pytz.timezone("US/Eastern")

        # Alpaca credentials
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET_KEY")

        # Connection state
        self.is_connected = False
        self.is_running = False
        self._ws = None
        self._stream_thread = None
        self._loop = None

        # Alert tracking
        self.active_alerts: Dict[str, RealTimeNewsAlert] = {}
        self.alert_history: List[RealTimeNewsAlert] = []
        self.max_history = 200

        # Symbol filter (empty = ALL news)
        self.watched_symbols: List[str] = []

        # Callbacks
        self.on_news_callback: Optional[Callable] = None
        self.on_scalp_signal_callback: Optional[Callable] = None

        # Scalp timing
        self.scalp_window_seconds = 120  # 2 minutes to act on real-time news
        self.max_chase_seconds = 180  # 3 minutes max chase

        # Stats
        self.news_received_count = 0
        self.alerts_generated_count = 0
        self.avg_latency_ms = 0

        logger.info("WarriorNewsStream initialized - REAL-TIME WebSocket news feed")

    def start(self, symbols: List[str] = None):
        """Start the real-time news stream"""
        if self.is_running:
            logger.warning("News stream already running")
            return

        self.watched_symbols = symbols or []
        self.is_running = True

        # Start WebSocket in background thread
        self._stream_thread = threading.Thread(target=self._run_stream, daemon=True)
        self._stream_thread.start()

        logger.info(f"Warrior News Stream STARTED - REAL-TIME WebSocket")
        if self.watched_symbols:
            logger.info(f"Filtering for symbols: {self.watched_symbols}")
        else:
            logger.info("Watching ALL news (no symbol filter)")

    def stop(self):
        """Stop the news stream"""
        self.is_running = False
        self.is_connected = False

        if self._ws:
            try:
                asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
            except:
                pass

        if self._stream_thread:
            self._stream_thread.join(timeout=3)

        logger.info("Warrior News Stream STOPPED")

    def _run_stream(self):
        """Run the WebSocket stream in a thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        while self.is_running:
            try:
                self._loop.run_until_complete(self._connect_and_stream())
            except Exception as e:
                logger.error(f"Stream error: {e}")
                self.is_connected = False

            if self.is_running:
                logger.info("Reconnecting in 5 seconds...")
                import time

                time.sleep(5)

        self._loop.close()

    async def _connect_and_stream(self):
        """Connect to WebSocket and stream news"""
        logger.info(f"Connecting to Alpaca News WebSocket: {self.NEWS_WS_URL}")

        try:
            async with websockets.connect(self.NEWS_WS_URL) as ws:
                self._ws = ws

                # First message is the "connected" confirmation
                response = await ws.recv()
                connected_response = json.loads(response)
                logger.info(f"Connected response: {connected_response}")

                # Now send authentication
                auth_msg = {
                    "action": "auth",
                    "key": self.api_key,
                    "secret": self.api_secret,
                }
                await ws.send(json.dumps(auth_msg))

                # Wait for auth response (second message)
                response = await ws.recv()
                auth_response = json.loads(response)
                logger.info(f"Auth response: {auth_response}")

                # Check if authenticated
                if isinstance(auth_response, list):
                    for msg in auth_response:
                        if (
                            msg.get("T") == "success"
                            and msg.get("msg") == "authenticated"
                        ):
                            self.is_connected = True
                            logger.info("AUTHENTICATED to Alpaca News Stream!")
                            break

                if not self.is_connected:
                    logger.error(f"Authentication failed: {auth_response}")
                    return

                # Subscribe to news
                if self.watched_symbols:
                    # Subscribe to specific symbols
                    subscribe_msg = {
                        "action": "subscribe",
                        "news": self.watched_symbols,
                    }
                else:
                    # Subscribe to ALL news
                    subscribe_msg = {"action": "subscribe", "news": ["*"]}

                await ws.send(json.dumps(subscribe_msg))
                logger.info(f"Subscribed to news: {subscribe_msg}")

                # Listen for news
                while self.is_running:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=30)
                        await self._handle_message(message)
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed")
                        break

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.is_connected = False

    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message"""
        received_at = datetime.now(self.et_tz)

        try:
            data = json.loads(message)

            # Alpaca sends messages as arrays
            if isinstance(data, list):
                for item in data:
                    await self._process_news_item(item, received_at)
            else:
                await self._process_news_item(data, received_at)

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def _process_news_item(self, item: Dict, received_at: datetime):
        """Process a single news item"""
        msg_type = item.get("T")

        # Skip non-news messages
        if msg_type != "n":  # "n" = news
            return

        self.news_received_count += 1

        # Extract news data
        news_id = str(item.get("id", ""))
        headline = item.get("headline", "")
        summary = item.get("summary", "")
        symbols = item.get("symbols", [])
        source = item.get("source", "")
        author = item.get("author", "")
        url = item.get("url")

        # Parse published time
        created_at = item.get("created_at", "")
        published_at = self._parse_datetime(created_at) if created_at else received_at

        # Calculate latency
        latency_ms = (received_at - published_at).total_seconds() * 1000

        # Update average latency
        self.avg_latency_ms = (self.avg_latency_ms * 0.9) + (latency_ms * 0.1)

        # Analyze the news
        full_text = f"{headline} {summary}"
        urgency = self._assess_urgency(full_text)
        sentiment, sentiment_score = self._analyze_sentiment(full_text)

        # Skip low urgency unless it's for watched symbols
        if urgency == "low" and self.watched_symbols:
            if not any(s in symbols for s in self.watched_symbols):
                return

        # Generate scalp signal
        scalp_signal, scalp_reason = self._generate_scalp_signal(
            urgency, sentiment, sentiment_score, latency_ms
        )

        # Create alert
        alert = RealTimeNewsAlert(
            id=news_id,
            headline=headline,
            summary=summary,
            symbols=symbols,
            source=source,
            author=author,
            published_at=published_at,
            received_at=received_at,
            url=url,
            urgency=urgency,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            scalp_signal=scalp_signal,
            scalp_reason=scalp_reason,
            latency_ms=latency_ms,
            time_to_act_seconds=self.scalp_window_seconds,
        )

        # Store alert
        self.active_alerts[news_id] = alert
        self.alert_history.insert(0, alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[: self.max_history]

        # Only log/alert for medium+ urgency
        if urgency in ["critical", "high", "medium"]:
            self.alerts_generated_count += 1

            urgency_emoji = {"critical": "🚨🚨🚨", "high": "🚨", "medium": "📰"}

            logger.warning(
                f"\n{urgency_emoji.get(urgency, '')} REAL-TIME NEWS [{urgency.upper()}] "
                f"[{latency_ms:.0f}ms latency]\n"
                f"  Headline: {headline[:80]}...\n"
                f"  Symbols: {symbols}\n"
                f"  Sentiment: {sentiment} ({sentiment_score:.2f})\n"
                f"  Signal: {scalp_signal}\n"
                f"  Reason: {scalp_reason}"
            )

            # Trigger callbacks
            if self.on_news_callback:
                try:
                    self.on_news_callback(alert)
                except Exception as e:
                    logger.error(f"News callback error: {e}")

            if (
                scalp_signal in ["long_now", "short_now"]
                and self.on_scalp_signal_callback
            ):
                try:
                    self.on_scalp_signal_callback(alert)
                except Exception as e:
                    logger.error(f"Scalp signal callback error: {e}")

            # Add to spike detector
            await self._add_to_spike_detector(symbols)

    def _parse_datetime(self, dt_str: str) -> datetime:
        """Parse datetime string"""
        try:
            if dt_str.endswith("Z"):
                dt_str = dt_str[:-1] + "+00:00"
            dt = datetime.fromisoformat(dt_str)
            if dt.tzinfo is None:
                dt = self.et_tz.localize(dt)
            return dt.astimezone(self.et_tz)
        except:
            return datetime.now(self.et_tz)

    def _assess_urgency(self, text: str) -> str:
        """Assess news urgency"""
        text_lower = text.lower()

        for keyword in CRITICAL_KEYWORDS:
            if keyword in text_lower:
                return "critical"

        for keyword in HIGH_KEYWORDS:
            if keyword in text_lower:
                return "high"

        for keyword in BEARISH_KEYWORDS:
            if keyword in text_lower:
                return "high"  # Bearish news is also high urgency

        return "medium"

    def _analyze_sentiment(self, text: str) -> tuple:
        """Analyze sentiment"""
        text_lower = text.lower()

        bullish_words = [
            "beat",
            "exceed",
            "surge",
            "soar",
            "rally",
            "upgrade",
            "breakthrough",
            "approval",
            "approved",
            "growth",
            "record",
            "strong",
            "positive",
            "outperform",
            "raise",
            "higher",
            "partnership",
            "contract",
            "deal",
        ]

        bearish_words = [
            "miss",
            "decline",
            "plunge",
            "crash",
            "downgrade",
            "rejection",
            "layoff",
            "weak",
            "negative",
            "lower",
            "cut",
            "warning",
            "lawsuit",
            "investigation",
            "fail",
            "fraud",
            "bankruptcy",
        ]

        bullish_count = sum(1 for w in bullish_words if w in text_lower)
        bearish_count = sum(1 for w in bearish_words if w in text_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return "neutral", 0.0

        score = (bullish_count - bearish_count) / total

        if score >= 0.3:
            return "bullish", score
        elif score <= -0.3:
            return "bearish", score
        return "neutral", score

    def _generate_scalp_signal(
        self, urgency: str, sentiment: str, sentiment_score: float, latency_ms: float
    ) -> tuple:
        """Generate scalp signal"""

        # Too much latency = stale news
        if latency_ms > 30000:  # 30 seconds
            return "avoid", f"News is {latency_ms/1000:.0f}s old - too stale"

        # Critical + Bullish = LONG NOW
        if urgency == "critical" and sentiment == "bullish":
            return (
                "long_now",
                f"CRITICAL bullish catalyst - SCALP NOW! ({latency_ms:.0f}ms)",
            )

        # Critical + Bearish = SHORT NOW
        if urgency == "critical" and sentiment == "bearish":
            return (
                "short_now",
                f"CRITICAL bearish catalyst - SHORT NOW! ({latency_ms:.0f}ms)",
            )

        # High + Bullish
        if urgency == "high" and sentiment == "bullish":
            return "long_now", f"HIGH urgency bullish - act fast! ({latency_ms:.0f}ms)"

        # High + Bearish
        if urgency == "high" and sentiment == "bearish":
            return "short_now", f"HIGH urgency bearish - short it! ({latency_ms:.0f}ms)"

        # Medium urgency
        if urgency == "medium":
            return "monitor", f"Medium urgency - monitor for momentum"

        return "avoid", "Not a scalp setup"

    async def _add_to_spike_detector(self, symbols: List[str]):
        """Add symbols to spike detector"""
        try:
            from .momentum_spike_detector import get_spike_detector

            detector = get_spike_detector()

            for symbol in symbols:
                if symbol and symbol not in detector.watchlist:
                    detector.watchlist.append(symbol)
                    logger.info(f"Added {symbol} to spike watchlist (news catalyst)")
        except Exception as e:
            logger.debug(f"Could not add to spike detector: {e}")

    def get_active_alerts(self, min_urgency: str = None) -> List[RealTimeNewsAlert]:
        """Get active alerts within scalp window"""
        now = datetime.now(self.et_tz)
        active = []

        urgency_order = ["low", "medium", "high", "critical"]
        min_idx = urgency_order.index(min_urgency) if min_urgency else 0

        for alert in self.active_alerts.values():
            age = (now - alert.received_at).total_seconds()

            if age < self.max_chase_seconds:
                alert.time_to_act_seconds = max(0, self.scalp_window_seconds - age)

                if urgency_order.index(alert.urgency) >= min_idx:
                    active.append(alert)

        # Sort by urgency then recency
        active.sort(
            key=lambda x: (
                -urgency_order.index(x.urgency),
                (now - x.received_at).total_seconds(),
            )
        )

        return active

    def get_scalp_signals(self) -> List[RealTimeNewsAlert]:
        """Get actionable scalp signals"""
        active = self.get_active_alerts(min_urgency="high")
        return [a for a in active if a.scalp_signal in ["long_now", "short_now"]]

    def get_status(self) -> Dict:
        """Get stream status"""
        return {
            "is_connected": self.is_connected,
            "is_running": self.is_running,
            "stream_type": "REAL-TIME WebSocket",
            "latency": "INSTANT (no polling)",
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "watched_symbols": self.watched_symbols or "ALL",
            "news_received": self.news_received_count,
            "alerts_generated": self.alerts_generated_count,
            "active_alerts": len(self.get_active_alerts()),
            "scalp_signals": len(self.get_scalp_signals()),
            "scalp_window_seconds": self.scalp_window_seconds,
        }


# Singleton
_news_stream: Optional[WarriorNewsStream] = None


def get_news_stream() -> WarriorNewsStream:
    """Get or create the news stream singleton"""
    global _news_stream
    if _news_stream is None:
        _news_stream = WarriorNewsStream()
    return _news_stream


def start_news_stream(symbols: List[str] = None):
    """Start the real-time news stream"""
    stream = get_news_stream()
    stream.start(symbols)
    return stream


def stop_news_stream():
    """Stop the news stream"""
    stream = get_news_stream()
    stream.stop()
