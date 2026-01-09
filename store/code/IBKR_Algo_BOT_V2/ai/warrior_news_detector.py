"""
Warrior News Alert Detector
===========================
WARRIOR TRADING STYLE - Hyper-fast news detection for momentum scalping.

When breaking news drops:
1. IMMEDIATELY detect the news
2. Analyze the stock for momentum entry
3. Add to spike detector watchlist
4. Generate scalp entry signal

WARRIOR RULES:
- News = Catalyst = MOMENTUM
- First 5 minutes after news = MAX opportunity
- Get in FAST, scalp the pop, GET OUT
- Don't chase after 10 minutes - you missed it

This module integrates with:
- news_feed_monitor.py: News source
- momentum_spike_detector.py: Real-time spike tracking
- market_regime.py: Trading phase awareness
"""

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional

import aiohttp
import pytz

logger = logging.getLogger(__name__)


class NewsUrgency(Enum):
    """News urgency level for scalping"""

    CRITICAL = "critical"  # FDA, M&A, Earnings surprise - TRADE IMMEDIATELY
    HIGH = "high"  # Analyst upgrade, guidance - Act within 2 min
    MEDIUM = "medium"  # Product launch, partnership - Monitor
    LOW = "low"  # Routine news - Skip


class ScalpSignal(Enum):
    """Scalp signal type"""

    LONG_NOW = "long_now"  # BUY IMMEDIATELY
    LONG_PULLBACK = "long_pullback"  # Wait for small dip, then buy
    SHORT_NOW = "short_now"  # SHORT IMMEDIATELY
    AVOID = "avoid"  # Skip this one
    MONITOR = "monitor"  # Watch for setup


@dataclass
class NewsAlert:
    """Breaking news alert with scalp analysis"""

    id: str
    headline: str
    summary: str
    symbols: List[str]
    source: str
    published_at: datetime
    detected_at: datetime

    # Warrior Analysis
    urgency: NewsUrgency
    sentiment: str  # "bullish", "bearish", "neutral"
    sentiment_score: float  # -1 to 1

    # Scalp Signal
    scalp_signal: ScalpSignal
    scalp_reason: str
    target_entry_price: Optional[float] = None
    target_profit_pct: float = 3.0  # Warrior 3% target
    stop_loss_pct: float = 1.0  # Warrior 1% stop

    # Timing
    time_since_news: float = 0  # seconds
    window_remaining: float = 300  # 5 min scalp window

    # Tracking
    added_to_spike_watchlist: bool = False
    momentum_detected: bool = False
    trade_taken: bool = False

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "headline": self.headline,
            "summary": self.summary,
            "symbols": self.symbols,
            "source": self.source,
            "published_at": self.published_at.isoformat(),
            "detected_at": self.detected_at.isoformat(),
            "urgency": self.urgency.value,
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "scalp_signal": self.scalp_signal.value,
            "scalp_reason": self.scalp_reason,
            "target_entry_price": self.target_entry_price,
            "target_profit_pct": self.target_profit_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "time_since_news": self.time_since_news,
            "window_remaining": self.window_remaining,
            "added_to_spike_watchlist": self.added_to_spike_watchlist,
            "momentum_detected": self.momentum_detected,
            "trade_taken": self.trade_taken,
        }


# Critical catalyst keywords (TRADE IMMEDIATELY)
CRITICAL_KEYWORDS = [
    "FDA approval",
    "FDA approves",
    "FDA grants",
    "breakthrough",
    "acquisition",
    "merger",
    "buyout",
    "takeover",
    "earnings beat",
    "earnings surprise",
    "EPS beat",
    "revenue beat",
    "upgrade",
    "price target raised",
    "raised guidance",
    "partnership",
    "contract win",
    "major deal",
]

# High urgency keywords (Act within 2 min)
HIGH_KEYWORDS = [
    "analyst upgrade",
    "buy rating",
    "outperform",
    "guidance raise",
    "outlook positive",
    "beat estimates",
    "expansion",
    "launch",
    "new product",
    "breakthrough",
]

# Bearish catalyst keywords
BEARISH_KEYWORDS = [
    "FDA reject",
    "trial fail",
    "earnings miss",
    "revenue miss",
    "downgrade",
    "price target cut",
    "lowered guidance",
    "lawsuit",
    "investigation",
    "SEC",
    "fraud",
    "layoff",
    "restructuring",
    "bankruptcy",
]


class WarriorNewsDetector:
    """
    WARRIOR TRADING - Hyper-fast news detection for momentum scalps.

    Polls for news every 10-15 seconds during market hours.
    When breaking news detected:
    1. Analyze for scalp opportunity
    2. Add symbols to spike detector
    3. Generate entry signal
    """

    def __init__(self):
        self.et_tz = pytz.timezone("US/Eastern")

        # Alpaca news API
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET_KEY")
        self.news_url = "https://data.alpaca.markets/v1beta1/news"

        # WARRIOR SPEED - HYPER-FAST polling
        self.poll_interval = 5  # 5 seconds during prime time (7-10 AM)
        self.poll_interval_normal = 15  # 15 seconds outside prime time
        self.is_monitoring = False
        self._monitor_thread = None

        # Prime news time (7 AM - 10 AM ET) - This is when catalysts hit!
        self.prime_start_hour = 7
        self.prime_end_hour = 10

        # Alert tracking
        self.active_alerts: Dict[str, NewsAlert] = {}  # by news_id
        self.alert_history: List[NewsAlert] = []
        self.max_history = 100

        # Seen news (to avoid duplicates)
        self._seen_news_ids: set = set()

        # Callbacks for integration
        self.on_news_callback: Optional[Callable] = None
        self.on_scalp_signal_callback: Optional[Callable] = None

        # Scalp window settings - TIGHTER for speed
        self.scalp_window_seconds = 180  # 3 minutes to act on news (was 5)
        self.max_chase_seconds = 300  # Don't enter after 5 min (was 10)

        # Watched symbols (optional filter)
        self.watched_symbols: List[str] = []

        logger.info(
            "WarriorNewsDetector initialized - 5 second polling during 7-10 AM prime time!"
        )

    def start_monitoring(self, symbols: List[str] = None):
        """Start hyper-fast news monitoring"""
        if self.is_monitoring:
            logger.warning("News monitoring already running")
            return

        self.watched_symbols = symbols or []
        self.is_monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        logger.info(
            f"Warrior News Detector STARTED - Polling every {self.poll_interval}s"
        )
        if self.watched_symbols:
            logger.info(f"Watching symbols: {self.watched_symbols}")

    def stop_monitoring(self):
        """Stop news monitoring"""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        logger.info("Warrior News Detector STOPPED")

    def _monitor_loop(self):
        """Main monitoring loop with adaptive speed"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self.is_monitoring:
            try:
                # Only poll during extended/market hours
                if self._is_trading_time():
                    loop.run_until_complete(self._poll_news())
                else:
                    logger.debug("Market closed - skipping news poll")

                # ADAPTIVE POLLING SPEED
                import time

                current_interval = self._get_poll_interval()
                time.sleep(current_interval)

            except Exception as e:
                logger.error(f"News monitor error: {e}")
                import time

                time.sleep(5)

        loop.close()

    def _get_poll_interval(self) -> int:
        """Get polling interval based on time - FASTER during 7-10 AM prime time"""
        now_et = datetime.now(self.et_tz)
        hour = now_et.hour

        # Prime time: 7 AM - 10 AM ET - This is when news catalysts hit!
        if self.prime_start_hour <= hour < self.prime_end_hour:
            return self.poll_interval  # 5 seconds
        else:
            return self.poll_interval_normal  # 15 seconds

    def _is_prime_time(self) -> bool:
        """Check if we're in prime news time (7-10 AM ET)"""
        now_et = datetime.now(self.et_tz)
        return self.prime_start_hour <= now_et.hour < self.prime_end_hour

    def _is_trading_time(self) -> bool:
        """Check if we should be monitoring (4 AM - 8 PM ET)"""
        now_et = datetime.now(self.et_tz)
        hour = now_et.hour
        weekday = now_et.weekday()

        # Skip weekends
        if weekday >= 5:
            return False

        # 4 AM to 8 PM ET
        return 4 <= hour < 20

    async def _poll_news(self):
        """Poll for breaking news"""
        if not self.api_key or not self.api_secret:
            logger.warning("Alpaca credentials not configured")
            return

        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

        params = {"limit": 20}  # Get last 20 news items
        if self.watched_symbols:
            params["symbols"] = ",".join(self.watched_symbols)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.news_url, headers=headers, params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        news_items = data.get("news", [])

                        # Process new news
                        for item in news_items:
                            news_id = str(item.get("id"))

                            # Skip if already seen
                            if news_id in self._seen_news_ids:
                                continue

                            self._seen_news_ids.add(news_id)

                            # Check if this is fresh news (within last 5 minutes)
                            published_at = self._parse_datetime(item.get("created_at"))
                            age_seconds = (
                                datetime.now(self.et_tz) - published_at
                            ).total_seconds()

                            if age_seconds < 300:  # Within 5 minutes
                                await self._process_breaking_news(item, age_seconds)
                    else:
                        error = await response.text()
                        logger.error(f"News API error: {response.status} - {error}")

        except Exception as e:
            logger.error(f"Error polling news: {e}")

    def _parse_datetime(self, dt_str: str) -> datetime:
        """Parse datetime string to timezone-aware datetime"""
        try:
            if dt_str.endswith("Z"):
                dt_str = dt_str[:-1] + "+00:00"
            dt = datetime.fromisoformat(dt_str)
            if dt.tzinfo is None:
                dt = self.et_tz.localize(dt)
            return dt.astimezone(self.et_tz)
        except:
            return datetime.now(self.et_tz)

    async def _process_breaking_news(self, news_item: Dict, age_seconds: float):
        """Process breaking news for scalp opportunity"""
        headline = news_item.get("headline", "")
        summary = news_item.get("summary", "")
        symbols = news_item.get("symbols", [])
        source = news_item.get("source", "Unknown")
        news_id = str(news_item.get("id"))

        full_text = f"{headline} {summary}"

        # Analyze for urgency and sentiment
        urgency = self._assess_urgency(full_text)
        sentiment, sentiment_score = self._analyze_sentiment(full_text)

        # Skip low urgency news
        if urgency == NewsUrgency.LOW:
            logger.debug(f"Skipping low urgency news: {headline[:50]}...")
            return

        # Generate scalp signal
        scalp_signal, scalp_reason = self._generate_scalp_signal(
            urgency, sentiment, sentiment_score, age_seconds
        )

        # Create alert
        alert = NewsAlert(
            id=news_id,
            headline=headline,
            summary=summary,
            symbols=symbols,
            source=source,
            published_at=self._parse_datetime(news_item.get("created_at")),
            detected_at=datetime.now(self.et_tz),
            urgency=urgency,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            scalp_signal=scalp_signal,
            scalp_reason=scalp_reason,
            time_since_news=age_seconds,
            window_remaining=max(0, self.scalp_window_seconds - age_seconds),
        )

        # Store alert
        self.active_alerts[news_id] = alert
        self.alert_history.insert(0, alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[: self.max_history]

        # Log the alert
        urgency_emoji = {
            NewsUrgency.CRITICAL: "🚨",
            NewsUrgency.HIGH: "⚠️",
            NewsUrgency.MEDIUM: "📰",
        }

        logger.warning(
            f"\n{urgency_emoji.get(urgency, '📰')} BREAKING NEWS [{urgency.value.upper()}]\n"
            f"  Headline: {headline[:80]}...\n"
            f"  Symbols: {symbols}\n"
            f"  Sentiment: {sentiment} ({sentiment_score:.2f})\n"
            f"  Signal: {scalp_signal.value}\n"
            f"  Reason: {scalp_reason}\n"
            f"  Time since news: {age_seconds:.0f}s\n"
            f"  Window remaining: {alert.window_remaining:.0f}s"
        )

        # Trigger callbacks
        if self.on_news_callback:
            try:
                self.on_news_callback(alert)
            except Exception as e:
                logger.error(f"News callback error: {e}")

        if (
            scalp_signal in [ScalpSignal.LONG_NOW, ScalpSignal.SHORT_NOW]
            and self.on_scalp_signal_callback
        ):
            try:
                self.on_scalp_signal_callback(alert)
            except Exception as e:
                logger.error(f"Scalp signal callback error: {e}")

        # Add symbols to spike detector watchlist
        await self._add_to_spike_watchlist(symbols, alert)

    def _assess_urgency(self, text: str) -> NewsUrgency:
        """Assess news urgency level"""
        text_lower = text.lower()

        # Check for critical keywords
        for keyword in CRITICAL_KEYWORDS:
            if keyword.lower() in text_lower:
                return NewsUrgency.CRITICAL

        # Check for high keywords
        for keyword in HIGH_KEYWORDS:
            if keyword.lower() in text_lower:
                return NewsUrgency.HIGH

        # Check for bearish keywords (also high urgency)
        for keyword in BEARISH_KEYWORDS:
            if keyword.lower() in text_lower:
                return NewsUrgency.HIGH

        return NewsUrgency.MEDIUM

    def _analyze_sentiment(self, text: str) -> tuple:
        """Analyze sentiment from text"""
        text_lower = text.lower()

        bullish_keywords = [
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
            "bullish",
            "outperform",
            "positive",
            "upside",
            "raise",
            "higher",
            "partnership",
            "contract",
            "deal",
            "expansion",
            "launch",
        ]

        bearish_keywords = [
            "miss",
            "decline",
            "plunge",
            "crash",
            "downgrade",
            "rejection",
            "reject",
            "layoff",
            "weak",
            "bearish",
            "underperform",
            "negative",
            "downside",
            "lower",
            "cut",
            "warning",
            "concern",
            "risk",
            "lawsuit",
            "investigation",
            "fail",
            "fraud",
            "bankruptcy",
        ]

        bullish_count = sum(1 for kw in bullish_keywords if kw in text_lower)
        bearish_count = sum(1 for kw in bearish_keywords if kw in text_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return "neutral", 0.0

        score = (bullish_count - bearish_count) / total

        if score >= 0.3:
            sentiment = "bullish"
        elif score <= -0.3:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        return sentiment, score

    def _generate_scalp_signal(
        self,
        urgency: NewsUrgency,
        sentiment: str,
        sentiment_score: float,
        age_seconds: float,
    ) -> tuple:
        """Generate scalp trading signal"""

        # Too late to trade
        if age_seconds > self.max_chase_seconds:
            return (
                ScalpSignal.AVOID,
                f"News is {age_seconds/60:.1f} min old - missed the window",
            )

        # Critical + Bullish = LONG NOW
        if urgency == NewsUrgency.CRITICAL and sentiment == "bullish":
            return (
                ScalpSignal.LONG_NOW,
                f"CRITICAL bullish catalyst - scalp the momentum!",
            )

        # Critical + Bearish = SHORT NOW
        if urgency == NewsUrgency.CRITICAL and sentiment == "bearish":
            return ScalpSignal.SHORT_NOW, f"CRITICAL bearish catalyst - scalp the drop!"

        # High + Bullish
        if urgency == NewsUrgency.HIGH and sentiment == "bullish":
            if age_seconds < 120:  # Within 2 minutes
                return ScalpSignal.LONG_NOW, f"HIGH urgency bullish - act fast!"
            else:
                return (
                    ScalpSignal.LONG_PULLBACK,
                    f"HIGH bullish but {age_seconds/60:.1f} min old - wait for pullback",
                )

        # High + Bearish
        if urgency == NewsUrgency.HIGH and sentiment == "bearish":
            if age_seconds < 120:
                return ScalpSignal.SHORT_NOW, f"HIGH urgency bearish - act fast!"
            else:
                return ScalpSignal.AVOID, f"Bearish but too late to short"

        # Medium urgency
        if urgency == NewsUrgency.MEDIUM:
            if sentiment == "bullish" and sentiment_score > 0.5:
                return (
                    ScalpSignal.MONITOR,
                    f"Medium urgency bullish - monitor for momentum",
                )
            else:
                return ScalpSignal.AVOID, f"Not strong enough catalyst for scalp"

        return ScalpSignal.AVOID, "No clear scalp setup"

    async def _add_to_spike_watchlist(self, symbols: List[str], alert: NewsAlert):
        """Add symbols to momentum spike detector watchlist"""
        try:
            from .momentum_spike_detector import get_spike_detector

            detector = get_spike_detector()

            for symbol in symbols:
                if symbol not in detector.watchlist:
                    detector.watchlist.append(symbol)
                    logger.info(f"Added {symbol} to spike watchlist (news catalyst)")

            alert.added_to_spike_watchlist = True

        except Exception as e:
            logger.error(f"Failed to add to spike watchlist: {e}")

    def get_active_alerts(self, min_urgency: NewsUrgency = None) -> List[NewsAlert]:
        """Get active alerts (within scalp window)"""
        now = datetime.now(self.et_tz)
        active = []

        for alert in self.active_alerts.values():
            age = (now - alert.published_at).total_seconds()

            # Still within window
            if age < self.max_chase_seconds:
                alert.time_since_news = age
                alert.window_remaining = max(0, self.scalp_window_seconds - age)

                # Filter by urgency
                if min_urgency:
                    urgency_order = [
                        NewsUrgency.LOW,
                        NewsUrgency.MEDIUM,
                        NewsUrgency.HIGH,
                        NewsUrgency.CRITICAL,
                    ]
                    if urgency_order.index(alert.urgency) >= urgency_order.index(
                        min_urgency
                    ):
                        active.append(alert)
                else:
                    active.append(alert)

        # Sort by urgency (critical first) then by recency
        active.sort(
            key=lambda x: (
                -[
                    NewsUrgency.LOW,
                    NewsUrgency.MEDIUM,
                    NewsUrgency.HIGH,
                    NewsUrgency.CRITICAL,
                ].index(x.urgency),
                x.time_since_news,
            )
        )

        return active

    def get_scalp_signals(self) -> List[NewsAlert]:
        """Get alerts with actionable scalp signals"""
        active = self.get_active_alerts(min_urgency=NewsUrgency.HIGH)

        actionable_signals = [
            ScalpSignal.LONG_NOW,
            ScalpSignal.SHORT_NOW,
            ScalpSignal.LONG_PULLBACK,
        ]

        return [a for a in active if a.scalp_signal in actionable_signals]

    def get_status(self) -> Dict:
        """Get detector status"""
        is_prime = self._is_prime_time()
        current_interval = self._get_poll_interval()

        return {
            "is_monitoring": self.is_monitoring,
            "poll_interval": current_interval,
            "poll_mode": "PRIME_TIME (5s)" if is_prime else "NORMAL (15s)",
            "is_prime_time": is_prime,
            "prime_window": f"{self.prime_start_hour}:00 - {self.prime_end_hour}:00 ET",
            "watched_symbols": self.watched_symbols,
            "active_alerts": len(self.get_active_alerts()),
            "scalp_signals": len(self.get_scalp_signals()),
            "history_count": len(self.alert_history),
            "seen_news_count": len(self._seen_news_ids),
            "scalp_window_seconds": self.scalp_window_seconds,
            "max_chase_seconds": self.max_chase_seconds,
        }

    def force_poll(self) -> Dict:
        """Force immediate news poll (sync wrapper)"""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._poll_news())
            return {
                "status": "polled",
                "active_alerts": len(self.get_active_alerts()),
                "scalp_signals": len(self.get_scalp_signals()),
            }
        finally:
            loop.close()


# Singleton instance
_news_detector: Optional[WarriorNewsDetector] = None


def get_news_detector() -> WarriorNewsDetector:
    """Get or create the news detector singleton"""
    global _news_detector
    if _news_detector is None:
        _news_detector = WarriorNewsDetector()
    return _news_detector


def start_news_detector(symbols: List[str] = None):
    """Start the news detector"""
    detector = get_news_detector()
    detector.start_monitoring(symbols)
    return detector


def stop_news_detector():
    """Stop the news detector"""
    detector = get_news_detector()
    detector.stop_monitoring()


# Integration function - connect to spike detector
def setup_news_to_spike_integration():
    """Setup integration between news detector and spike detector"""
    try:
        from .momentum_spike_detector import get_spike_detector

        news_detector = get_news_detector()
        spike_detector = get_spike_detector()

        def on_scalp_signal(alert: NewsAlert):
            """When scalp signal detected, ensure spike detector is watching"""
            for symbol in alert.symbols:
                if symbol not in spike_detector.watchlist:
                    spike_detector.watchlist.append(symbol)
                logger.info(
                    f"News scalp signal: {alert.scalp_signal.value} for {symbol}"
                )

        news_detector.on_scalp_signal_callback = on_scalp_signal

        logger.info("News -> Spike detector integration configured")
        return True

    except Exception as e:
        logger.error(f"Failed to setup news integration: {e}")
        return False
