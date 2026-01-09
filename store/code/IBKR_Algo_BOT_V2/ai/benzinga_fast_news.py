"""
Benzinga Fast News Trigger
==========================
Ultra-fast breaking news detection using Benzinga Pro + multiple sources.

SPEED IS EVERYTHING:
- Poll every 1-2 seconds
- Parse instantly
- Trigger trade within 100ms of detection

Sources (in order of speed):
1. Benzinga Pro WebSocket (if available)
2. Benzinga RSS Feed (free, ~10-30s delay)
3. Alpaca News WebSocket (built-in)
4. Yahoo Finance RSS (backup)
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Set

import aiohttp
import feedparser
import pytz

logger = logging.getLogger(__name__)


@dataclass
class BreakingNews:
    """Breaking news alert"""

    id: str
    headline: str
    symbols: List[str]
    source: str
    published_at: datetime
    detected_at: datetime
    latency_ms: float

    # Analysis
    sentiment: str  # bullish, bearish, neutral
    urgency: str  # critical, high, medium, low
    catalyst_type: str  # fda, earnings, merger, upgrade, etc.

    # Trading signal
    action: str  # buy, sell, watch, avoid
    confidence: float

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "headline": self.headline,
            "symbols": self.symbols,
            "source": self.source,
            "published_at": (
                self.published_at.isoformat() if self.published_at else None
            ),
            "detected_at": self.detected_at.isoformat(),
            "latency_ms": self.latency_ms,
            "sentiment": self.sentiment,
            "urgency": self.urgency,
            "catalyst_type": self.catalyst_type,
            "action": self.action,
            "confidence": self.confidence,
        }


# Catalyst keywords for instant detection
CRITICAL_CATALYSTS = {
    "fda": [
        "fda approval",
        "fda approves",
        "fda grants",
        "breakthrough therapy",
        "fda clears",
        "fda accepts",
        "pdufa",
        "nda approved",
    ],
    "merger": [
        "acquisition",
        "merger",
        "buyout",
        "takeover bid",
        "acquire",
        "all-cash deal",
        "tender offer",
    ],
    "earnings_beat": [
        "beats estimates",
        "eps beat",
        "revenue beat",
        "earnings surprise",
        "profit surge",
        "record earnings",
        "guidance raise",
    ],
    "upgrade": [
        "upgrade",
        "price target raised",
        "buy rating",
        "outperform",
        "strong buy",
        "initiated coverage buy",
    ],
}

BEARISH_CATALYSTS = {
    "fda_reject": [
        "fda rejects",
        "fda denies",
        "crl issued",
        "trial fails",
        "clinical hold",
        "safety concerns",
    ],
    "earnings_miss": [
        "misses estimates",
        "eps miss",
        "revenue miss",
        "earnings warning",
        "profit warning",
        "guidance cut",
        "lowers outlook",
    ],
    "downgrade": [
        "downgrade",
        "price target cut",
        "sell rating",
        "underperform",
        "initiated coverage sell",
    ],
    "legal": [
        "sec investigation",
        "lawsuit",
        "fraud",
        "subpoena",
        "indictment",
        "accounting irregularities",
    ],
}


class BenzingaFastNews:
    """
    Ultra-fast news detection from multiple sources.
    Designed for sub-second reaction time.
    """

    # Benzinga RSS feeds
    BENZINGA_RSS = "https://www.benzinga.com/feed"
    BENZINGA_MOVERS = "https://www.benzinga.com/stock/movers/rss"

    # Yahoo Finance RSS
    YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"

    def __init__(self):
        self.et_tz = pytz.timezone("US/Eastern")

        # Track seen news to avoid duplicates
        self.seen_ids: Set[str] = set()
        self.max_seen = 5000

        # Recent alerts
        self.alerts: deque = deque(maxlen=100)

        # Callbacks for trading triggers
        self.on_breaking_news: Optional[Callable] = None
        self.on_buy_signal: Optional[Callable] = None
        self.on_sell_signal: Optional[Callable] = None

        # Control
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Benzinga API key - USE REAL API FOR SPEED
        self.benzinga_api_key = os.getenv(
            "BENZINGA_API_KEY", "bz.MUTADSLMPPPHDWEGOYUMSFHUGH5TS7TD"
        )

        # Benzinga API endpoints (much faster than RSS!)
        self.BENZINGA_NEWS_API = "https://api.benzinga.com/api/v2/news"
        self.BENZINGA_SIGNALS_API = "https://api.benzinga.com/api/v2.1/signals"

        # Use API if key available
        self.use_api = bool(self.benzinga_api_key)

        # Polling intervals (in seconds)
        # API mode = faster polling possible (1 sec), RSS = slower (2 sec)
        self.benzinga_interval = 1.0 if self.use_api else 2.0
        self.yahoo_interval = 5.0  # Backup, less frequent

        # Watchlist - symbols to monitor for news
        self.watchlist: List[str] = []

        # Small Cap Filter - only show news for penny/small cap stocks
        self.small_cap_only: bool = True  # Filter to small caps only
        self.min_price: float = 0.50  # Min price for alerts
        self.max_price: float = 20.0  # Max price for alerts (excludes large caps)
        self.price_cache: Dict[str, tuple] = {}  # symbol -> (price, timestamp)
        self.price_cache_ttl: int = 60  # Cache prices for 60 seconds

        # Stats
        self.news_detected = 0
        self.signals_generated = 0
        self.avg_latency_ms = 0
        self.filtered_count = 0  # How many filtered out by small cap filter

        logger.info(
            f"BenzingaFastNews initialized - {'REAL API' if self.use_api else 'RSS'} mode, small_cap_only={self.small_cap_only}"
        )

    def start(self, watchlist: List[str] = None):
        """Start the news scanner"""
        if self.is_running:
            logger.warning("Already running")
            return

        self.watchlist = watchlist or []
        self.is_running = True

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        logger.info(
            f"BenzingaFastNews STARTED - watching {len(self.watchlist)} symbols"
        )

    def stop(self):
        """Stop the scanner"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("BenzingaFastNews STOPPED")

    def _run_loop(self):
        """Run the async event loop in thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._scan_loop())
        except Exception as e:
            logger.error(f"Loop error: {e}")
        finally:
            self._loop.close()

    async def _scan_loop(self):
        """Main scanning loop - polls all sources"""
        last_benzinga = 0
        last_yahoo = 0

        while self.is_running:
            now = time.time()

            try:
                # Benzinga - poll frequently
                if now - last_benzinga >= self.benzinga_interval:
                    await self._scan_benzinga()
                    last_benzinga = now

                # Yahoo - poll less frequently as backup
                if now - last_yahoo >= self.yahoo_interval and self.watchlist:
                    await self._scan_yahoo()
                    last_yahoo = now

            except Exception as e:
                logger.error(f"Scan error: {e}")

            # Small sleep to prevent CPU spinning
            await asyncio.sleep(0.1)

    async def _scan_benzinga(self):
        """Scan Benzinga - prefer API if available, fallback to RSS"""
        if self.use_api and self.benzinga_api_key:
            await self._scan_benzinga_api()
        else:
            await self._scan_benzinga_rss()

    async def _scan_benzinga_api(self):
        """Scan Benzinga API for breaking news - MUCH FASTER than RSS"""
        try:
            headers = {"accept": "application/json"}

            # Build API URL with parameters - token goes in query params
            params = {
                "token": self.benzinga_api_key,
                "pageSize": 20,
                "displayOutput": "full",
                "sort": "created:desc",
            }

            # Add tickers filter if watchlist exists
            if self.watchlist:
                params["tickers"] = ",".join(self.watchlist[:10])

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.BENZINGA_NEWS_API,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    if response.status != 200:
                        # Fallback to RSS on API error
                        logger.debug(
                            f"Benzinga API returned {response.status}, falling back to RSS"
                        )
                        await self._scan_benzinga_rss()
                        return

                    data = await response.json()
                    await self._parse_benzinga_api(data)

        except Exception as e:
            logger.debug(f"Benzinga API error: {e}, falling back to RSS")
            await self._scan_benzinga_rss()

    async def _parse_benzinga_api(self, data: dict):
        """Parse Benzinga API response"""
        detected_at = datetime.now(self.et_tz)

        # API returns list directly or in 'data' key
        articles = data if isinstance(data, list) else data.get("data", [])

        for article in articles[:20]:
            try:
                # Benzinga API fields
                news_id = str(article.get("id", ""))
                if not news_id:
                    continue

                # Skip if already seen
                if news_id in self.seen_ids:
                    continue

                self.seen_ids.add(news_id)
                if len(self.seen_ids) > self.max_seen:
                    self.seen_ids = set(list(self.seen_ids)[-self.max_seen :])

                headline = article.get("title", article.get("headline", ""))
                teaser = article.get("teaser", "")
                full_text = f"{headline} {teaser}".lower()

                # Extract symbols from API response (Benzinga provides them!)
                symbols = []
                stocks = article.get("stocks", [])
                if stocks:
                    symbols = [s.get("name", "") for s in stocks if s.get("name")]

                # Fallback to extraction from headline
                if not symbols:
                    symbols = self._extract_symbols(headline)

                # Parse time - Benzinga uses ISO format
                created = article.get("created", article.get("updated", ""))
                published_at = self._parse_time(created)

                # Calculate latency - API is much faster!
                latency_ms = (detected_at - published_at).total_seconds() * 1000

                # Analyze for catalyst
                sentiment, urgency, catalyst_type = self._analyze_catalyst(full_text)

                # Skip low urgency
                if urgency == "low":
                    continue

                # Generate trading signal
                action, confidence = self._generate_signal(
                    sentiment, urgency, catalyst_type, latency_ms
                )

                # Create alert
                alert = BreakingNews(
                    id=news_id,
                    headline=headline,
                    symbols=symbols,
                    source="benzinga_api",
                    published_at=published_at,
                    detected_at=detected_at,
                    latency_ms=latency_ms,
                    sentiment=sentiment,
                    urgency=urgency,
                    catalyst_type=catalyst_type,
                    action=action,
                    confidence=confidence,
                )

                self.alerts.append(alert)
                self.news_detected += 1
                self.avg_latency_ms = (self.avg_latency_ms * 0.9) + (latency_ms * 0.1)

                # Log and trigger callbacks
                self._handle_alert(alert)

            except Exception as e:
                logger.debug(f"Error parsing article: {e}")

    async def _scan_benzinga_rss(self):
        """Scan Benzinga RSS for breaking news (fallback)"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.BENZINGA_RSS, timeout=5) as response:
                    if response.status != 200:
                        return

                    content = await response.text()
                    await self._parse_rss(content, "benzinga")

        except Exception as e:
            logger.debug(f"Benzinga RSS scan error: {e}")

    async def _scan_yahoo(self):
        """Scan Yahoo Finance RSS for watchlist symbols"""
        if not self.watchlist:
            return

        async with aiohttp.ClientSession() as session:
            for symbol in self.watchlist[:10]:  # Limit to avoid rate limits
                try:
                    url = self.YAHOO_RSS.format(symbol=symbol)
                    async with session.get(url, timeout=3) as response:
                        if response.status == 200:
                            content = await response.text()
                            await self._parse_rss(content, "yahoo", [symbol])
                except:
                    continue

    async def _parse_rss(
        self, content: str, source: str, default_symbols: List[str] = None
    ):
        """Parse RSS feed and detect breaking news"""
        detected_at = datetime.now(self.et_tz)

        try:
            feed = feedparser.parse(content)

            for entry in feed.entries[:20]:  # Check latest 20 items
                # Generate unique ID
                news_id = hashlib.md5(
                    f"{entry.get('title', '')}{entry.get('published', '')}".encode()
                ).hexdigest()[:16]

                # Skip if already seen
                if news_id in self.seen_ids:
                    continue

                self.seen_ids.add(news_id)
                if len(self.seen_ids) > self.max_seen:
                    # Clear oldest
                    self.seen_ids = set(list(self.seen_ids)[-self.max_seen :])

                headline = entry.get("title", "")
                summary = entry.get("summary", "")
                full_text = f"{headline} {summary}".lower()

                # Extract symbols from headline
                symbols = self._extract_symbols(headline) or default_symbols or []

                # Parse published time
                published_at = self._parse_time(entry.get("published"))

                # Calculate latency
                latency_ms = (detected_at - published_at).total_seconds() * 1000

                # Analyze for catalyst
                sentiment, urgency, catalyst_type = self._analyze_catalyst(full_text)

                # Skip low urgency
                if urgency == "low":
                    continue

                # Generate trading signal
                action, confidence = self._generate_signal(
                    sentiment, urgency, catalyst_type, latency_ms
                )

                # Create alert
                alert = BreakingNews(
                    id=news_id,
                    headline=headline,
                    symbols=symbols,
                    source=source,
                    published_at=published_at,
                    detected_at=detected_at,
                    latency_ms=latency_ms,
                    sentiment=sentiment,
                    urgency=urgency,
                    catalyst_type=catalyst_type,
                    action=action,
                    confidence=confidence,
                )

                self.alerts.append(alert)
                self.news_detected += 1
                self.avg_latency_ms = (self.avg_latency_ms * 0.9) + (latency_ms * 0.1)

                # Log and trigger callbacks
                self._handle_alert(alert)

        except Exception as e:
            logger.debug(f"RSS parse error: {e}")

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        symbols = []

        # Look for $SYMBOL pattern
        dollar_tickers = re.findall(r"\$([A-Z]{1,5})\b", text)
        symbols.extend(dollar_tickers)

        # Look for (SYMBOL) pattern
        paren_tickers = re.findall(r"\(([A-Z]{1,5})\)", text)
        symbols.extend(paren_tickers)

        # Look for SYMBOL: pattern
        colon_tickers = re.findall(r"\b([A-Z]{2,5}):", text)
        symbols.extend(colon_tickers)

        return list(set(symbols))

    def _parse_time(self, time_str: str) -> datetime:
        """Parse various time formats"""
        if not time_str:
            return datetime.now(self.et_tz)

        try:
            # Try common RSS formats
            for fmt in [
                "%a, %d %b %Y %H:%M:%S %z",
                "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%SZ",
            ]:
                try:
                    dt = datetime.strptime(time_str, fmt)
                    if dt.tzinfo is None:
                        dt = self.et_tz.localize(dt)
                    return dt.astimezone(self.et_tz)
                except:
                    continue

            # Fallback to feedparser's parsed time
            import email.utils

            parsed = email.utils.parsedate_to_datetime(time_str)
            return parsed.astimezone(self.et_tz)

        except:
            return datetime.now(self.et_tz)

    def _analyze_catalyst(self, text: str) -> tuple:
        """Analyze text for catalyst type and sentiment"""
        text_lower = text.lower()

        # Check bullish catalysts
        for catalyst_type, keywords in CRITICAL_CATALYSTS.items():
            for kw in keywords:
                if kw in text_lower:
                    urgency = (
                        "critical" if catalyst_type in ["fda", "merger"] else "high"
                    )
                    return "bullish", urgency, catalyst_type

        # Check bearish catalysts
        for catalyst_type, keywords in BEARISH_CATALYSTS.items():
            for kw in keywords:
                if kw in text_lower:
                    urgency = (
                        "critical"
                        if catalyst_type in ["fda_reject", "legal"]
                        else "high"
                    )
                    return "bearish", urgency, catalyst_type

        # General sentiment analysis
        bullish_words = [
            "surge",
            "soar",
            "rally",
            "jump",
            "gain",
            "up",
            "rise",
            "beat",
            "strong",
        ]
        bearish_words = [
            "crash",
            "plunge",
            "drop",
            "fall",
            "down",
            "miss",
            "weak",
            "decline",
        ]

        bull_count = sum(1 for w in bullish_words if w in text_lower)
        bear_count = sum(1 for w in bearish_words if w in text_lower)

        if bull_count > bear_count + 1:
            return "bullish", "medium", "momentum"
        elif bear_count > bull_count + 1:
            return "bearish", "medium", "momentum"

        return "neutral", "low", "none"

    def _generate_signal(
        self, sentiment: str, urgency: str, catalyst_type: str, latency_ms: float
    ) -> tuple:
        """Generate trading signal from analysis"""

        # Too stale - no signal
        if latency_ms > 60000:  # > 60 seconds
            return "watch", 0.3

        # Critical bullish = BUY NOW
        if urgency == "critical" and sentiment == "bullish":
            conf = 0.9 if latency_ms < 5000 else 0.7
            return "buy", conf

        # Critical bearish = SELL/SHORT NOW
        if urgency == "critical" and sentiment == "bearish":
            conf = 0.9 if latency_ms < 5000 else 0.7
            return "sell", conf

        # High urgency
        if urgency == "high":
            conf = 0.7 if latency_ms < 10000 else 0.5
            return "buy" if sentiment == "bullish" else "sell", conf

        # Medium urgency - watch
        if urgency == "medium":
            return "watch", 0.4

        return "avoid", 0.0

    def _handle_alert(self, alert: BreakingNews):
        """Handle a breaking news alert"""

        urgency_emoji = {"critical": "🚨🚨🚨", "high": "🚨", "medium": "📰", "low": ""}

        action_emoji = {
            "buy": "🟢 BUY",
            "sell": "🔴 SELL",
            "watch": "👀 WATCH",
            "avoid": "⚪ SKIP",
        }

        # Log it
        logger.warning(
            f"\n{urgency_emoji.get(alert.urgency, '')} BREAKING [{alert.source.upper()}] "
            f"[{alert.latency_ms:.0f}ms]\n"
            f"  {alert.headline[:100]}...\n"
            f"  Symbols: {alert.symbols}\n"
            f"  Catalyst: {alert.catalyst_type} | Sentiment: {alert.sentiment}\n"
            f"  Signal: {action_emoji.get(alert.action)} ({alert.confidence:.0%} confidence)"
        )

        # Trigger callbacks
        if self.on_breaking_news:
            try:
                self.on_breaking_news(alert)
            except Exception as e:
                logger.error(f"Breaking news callback error: {e}")

        if alert.action == "buy" and alert.confidence >= 0.7 and self.on_buy_signal:
            self.signals_generated += 1
            try:
                self.on_buy_signal(alert)
            except Exception as e:
                logger.error(f"Buy signal callback error: {e}")

        if alert.action == "sell" and alert.confidence >= 0.7 and self.on_sell_signal:
            self.signals_generated += 1
            try:
                self.on_sell_signal(alert)
            except Exception as e:
                logger.error(f"Sell signal callback error: {e}")

    def get_recent_alerts(self, limit: int = 20, min_urgency: str = None) -> List[Dict]:
        """Get recent alerts"""
        alerts = list(self.alerts)

        if min_urgency:
            urgency_order = ["low", "medium", "high", "critical"]
            min_idx = urgency_order.index(min_urgency)
            alerts = [a for a in alerts if urgency_order.index(a.urgency) >= min_idx]

        return [a.to_dict() for a in alerts[-limit:]]

    def get_actionable_signals(self) -> List[Dict]:
        """Get signals that are actionable (buy/sell with high confidence)"""
        now = datetime.now(self.et_tz)
        cutoff = now - timedelta(minutes=2)  # Only last 2 minutes

        actionable = []
        for alert in self.alerts:
            if alert.detected_at >= cutoff:
                if alert.action in ["buy", "sell"] and alert.confidence >= 0.7:
                    actionable.append(alert.to_dict())

        return actionable

    def get_status(self) -> Dict:
        """Get scanner status"""
        sources = []
        if self.use_api:
            sources.append("benzinga_api")
        else:
            sources.append("benzinga_rss")
        sources.append("yahoo_rss")

        return {
            "is_running": self.is_running,
            "mode": "API" if self.use_api else "RSS",
            "api_key_configured": bool(self.benzinga_api_key),
            "sources": sources,
            "polling_interval_sec": self.benzinga_interval,
            "watchlist_size": len(self.watchlist),
            "news_detected": self.news_detected,
            "signals_generated": self.signals_generated,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "recent_alerts": len(self.alerts),
        }


# Singleton
_fast_news: Optional[BenzingaFastNews] = None


def get_fast_news() -> BenzingaFastNews:
    """Get or create fast news scanner singleton"""
    global _fast_news
    if _fast_news is None:
        _fast_news = BenzingaFastNews()
    return _fast_news


def start_fast_news(
    watchlist: List[str] = None, on_buy: Callable = None, on_sell: Callable = None
):
    """Start fast news scanner with optional callbacks"""
    scanner = get_fast_news()

    if on_buy:
        scanner.on_buy_signal = on_buy
    if on_sell:
        scanner.on_sell_signal = on_sell

    scanner.start(watchlist)
    return scanner


def stop_fast_news():
    """Stop the fast news scanner"""
    scanner = get_fast_news()
    scanner.stop()


if __name__ == "__main__":
    # Test the scanner
    logging.basicConfig(level=logging.INFO)

    def on_buy(alert):
        print(f"\n*** BUY SIGNAL: {alert.symbols} - {alert.headline[:50]} ***\n")

    def on_sell(alert):
        print(f"\n*** SELL SIGNAL: {alert.symbols} - {alert.headline[:50]} ***\n")

    scanner = start_fast_news(
        watchlist=["AAPL", "TSLA", "NVDA", "SPY"], on_buy=on_buy, on_sell=on_sell
    )

    print("Fast news scanner running... Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(10)
            status = scanner.get_status()
            print(
                f"Status: {status['news_detected']} news, {status['signals_generated']} signals"
            )
    except KeyboardInterrupt:
        stop_fast_news()
        print("Stopped")
