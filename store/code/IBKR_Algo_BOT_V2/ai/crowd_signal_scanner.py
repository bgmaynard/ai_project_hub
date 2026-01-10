"""
Crowd Signal Scanner
====================
SECONDARY TRIGGER for the Hot Symbol Engine.

Detects human reaction velocity across social media:
- StockTwits trending symbols
- Reddit ticker frequency (WSB, stocks)
- Mention spike detection

This often beats free news APIs because people post
the moment they see a halt, squeeze, or violent candle.

Author: AI Trading Bot Team
Version: 1.0
Created: 2026-01-10
"""

import logging
import asyncio
import aiohttp
import re
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import deque, Counter

logger = logging.getLogger(__name__)


# Known stock ticker patterns
TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b')

# Common words to exclude (not tickers)
EXCLUDED_WORDS = {
    'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER',
    'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW',
    'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'WAY', 'WHO', 'BOY', 'DID',
    'BUY', 'SELL', 'PUT', 'CALL', 'ATH', 'EOD', 'AMA', 'IMO', 'FYI', 'TBH',
    'ETF', 'IPO', 'CEO', 'CFO', 'WSB', 'DD', 'YOLO', 'HODL', 'FOMO', 'FUD',
    'LOW', 'HIGH', 'UP', 'DOWN', 'LONG', 'SHORT', 'MOON', 'BEAR', 'BULL',
    'RED', 'GREEN', 'GAIN', 'LOSS', 'EPS', 'PE', 'IV', 'OTM', 'ITM', 'ATM',
    'USA', 'NYSE', 'NASDAQ', 'SEC', 'FDA', 'CEO', 'PM', 'AM', 'EST', 'PST'
}


@dataclass
class CrowdSignal:
    """A crowd signal detection event"""
    symbol: str
    source: str  # stocktwits, reddit, etc
    mention_count: int
    baseline_count: int
    ratio: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    sample_posts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "source": self.source,
            "mention_count": self.mention_count,
            "baseline_count": self.baseline_count,
            "ratio": round(self.ratio, 2),
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp.isoformat(),
            "sample_posts": self.sample_posts[:3]
        }


class MentionTracker:
    """Tracks mention frequency for a single symbol"""

    def __init__(self, symbol: str, window_minutes: int = 30):
        self.symbol = symbol
        self.window_minutes = window_minutes
        self.mentions: deque = deque()
        self._lock = threading.Lock()

    def add_mention(self, timestamp: Optional[datetime] = None):
        """Record a mention"""
        ts = timestamp or datetime.now()
        with self._lock:
            self.mentions.append(ts)
            self._cleanup()

    def _cleanup(self):
        """Remove mentions outside the window"""
        cutoff = datetime.now() - timedelta(minutes=self.window_minutes)
        while self.mentions and self.mentions[0] < cutoff:
            self.mentions.popleft()

    def get_count(self, minutes: int = 5) -> int:
        """Get mention count in last N minutes"""
        with self._lock:
            self._cleanup()
            cutoff = datetime.now() - timedelta(minutes=minutes)
            return sum(1 for ts in self.mentions if ts >= cutoff)

    def get_baseline(self, minutes: int = 30) -> float:
        """Get average mentions per 5 minutes over the baseline period"""
        with self._lock:
            self._cleanup()
            total = len(self.mentions)
            intervals = max(1, minutes / 5)
            return total / intervals


class CrowdSignalScanner:
    """
    Scans social media for ticker mention spikes.

    Supported sources:
    - StockTwits trending
    - Reddit (scraping r/wallstreetbets, r/stocks)
    """

    # Detection thresholds
    MIN_MENTION_RATIO = 3.0      # >= 3x baseline to trigger
    MIN_MENTIONS = 5            # Minimum mentions to consider
    MIN_CONFIDENCE = 0.4        # Minimum confidence to emit signal

    def __init__(self):
        self._trackers: Dict[str, MentionTracker] = {}
        self._lock = threading.Lock()
        self._running = False
        self._scan_interval = 60  # seconds

        # Callbacks
        self._on_signal_callbacks: List[callable] = []

        # Stats
        self._total_scans = 0
        self._total_signals = 0
        self._signals_by_source: Dict[str, int] = {}

        # Known valid tickers (loaded from watchlist or hardcoded)
        self._valid_tickers: Set[str] = set()

        logger.info("CrowdSignalScanner initialized")

    def set_valid_tickers(self, tickers: List[str]):
        """Set the list of valid tickers to track"""
        self._valid_tickers = {t.upper() for t in tickers}
        logger.info(f"Set {len(self._valid_tickers)} valid tickers for crowd scanning")

    def add_signal_callback(self, callback: callable):
        """Register a callback for when signals are detected"""
        self._on_signal_callbacks.append(callback)

    def _get_tracker(self, symbol: str) -> MentionTracker:
        """Get or create tracker for a symbol"""
        symbol = symbol.upper()
        with self._lock:
            if symbol not in self._trackers:
                self._trackers[symbol] = MentionTracker(symbol)
            return self._trackers[symbol]

    def record_mention(self, symbol: str, source: str = "unknown"):
        """Record a single mention of a ticker"""
        if not self._is_valid_ticker(symbol):
            return

        tracker = self._get_tracker(symbol)
        tracker.add_mention()

    def _is_valid_ticker(self, symbol: str) -> bool:
        """Check if this looks like a valid ticker"""
        symbol = symbol.upper()

        if symbol in EXCLUDED_WORDS:
            return False

        if len(symbol) < 1 or len(symbol) > 5:
            return False

        # If we have a whitelist, use it
        if self._valid_tickers:
            return symbol in self._valid_tickers

        return True

    def extract_tickers(self, text: str) -> List[str]:
        """Extract ticker symbols from text"""
        tickers = []

        # Find $TICKER patterns
        matches = TICKER_PATTERN.findall(text.upper())
        for match in matches:
            ticker = match[0] or match[1]
            if ticker and self._is_valid_ticker(ticker):
                tickers.append(ticker)

        return list(set(tickers))

    async def scan_stocktwits_trending(self) -> List[CrowdSignal]:
        """Scan StockTwits for trending tickers"""
        signals = []

        try:
            async with aiohttp.ClientSession() as session:
                # StockTwits trending API
                url = "https://api.stocktwits.com/api/2/trending/symbols.json"

                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        symbols = data.get("symbols", [])

                        for sym_data in symbols[:20]:  # Top 20 trending
                            symbol = sym_data.get("symbol", "")
                            if self._is_valid_ticker(symbol):
                                # Record and check for spike
                                tracker = self._get_tracker(symbol)
                                tracker.add_mention()

                                # Since it's trending, give it higher weight
                                signal = self._check_for_signal(
                                    symbol, "stocktwits",
                                    boost=0.2  # Trending boost
                                )
                                if signal:
                                    signals.append(signal)
                    else:
                        logger.debug(f"StockTwits API returned {resp.status}")

        except asyncio.TimeoutError:
            logger.debug("StockTwits scan timed out")
        except Exception as e:
            logger.debug(f"StockTwits scan error: {e}")

        return signals

    async def scan_reddit_mentions(self, subreddit: str = "wallstreetbets") -> List[CrowdSignal]:
        """Scan Reddit for ticker mentions"""
        signals = []

        try:
            async with aiohttp.ClientSession() as session:
                # Reddit JSON API (no auth needed for public posts)
                url = f"https://www.reddit.com/r/{subreddit}/new.json?limit=50"
                headers = {"User-Agent": "MorpheusTradingBot/1.0"}

                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        posts = data.get("data", {}).get("children", [])

                        mention_counts: Counter = Counter()

                        for post in posts:
                            post_data = post.get("data", {})
                            title = post_data.get("title", "")
                            selftext = post_data.get("selftext", "")

                            text = f"{title} {selftext}"
                            tickers = self.extract_tickers(text)

                            for ticker in tickers:
                                mention_counts[ticker] += 1
                                self.record_mention(ticker, f"reddit/{subreddit}")

                        # Check for signals on high-mention tickers
                        for symbol, count in mention_counts.most_common(10):
                            if count >= 3:  # At least 3 mentions in recent posts
                                signal = self._check_for_signal(
                                    symbol, f"reddit/{subreddit}"
                                )
                                if signal:
                                    signals.append(signal)
                    else:
                        logger.debug(f"Reddit API returned {resp.status}")

        except asyncio.TimeoutError:
            logger.debug("Reddit scan timed out")
        except Exception as e:
            logger.debug(f"Reddit scan error: {e}")

        return signals

    def _check_for_signal(
        self,
        symbol: str,
        source: str,
        boost: float = 0.0
    ) -> Optional[CrowdSignal]:
        """Check if a symbol has triggered a crowd signal"""
        tracker = self._get_tracker(symbol)

        recent_count = tracker.get_count(minutes=5)
        baseline = tracker.get_baseline(minutes=30)

        if recent_count < self.MIN_MENTIONS:
            return None

        if baseline == 0:
            baseline = 1  # Avoid division by zero

        ratio = recent_count / baseline

        if ratio < self.MIN_MENTION_RATIO:
            return None

        # Calculate confidence
        confidence = min(1.0, 0.4 + (ratio - 3.0) * 0.1 + boost)

        if confidence < self.MIN_CONFIDENCE:
            return None

        signal = CrowdSignal(
            symbol=symbol,
            source=source,
            mention_count=recent_count,
            baseline_count=int(baseline),
            ratio=ratio,
            confidence=confidence
        )

        # Update stats
        self._total_signals += 1
        self._signals_by_source[source] = self._signals_by_source.get(source, 0) + 1

        logger.info(
            f"CROWD SIGNAL: {symbol} | source={source} | "
            f"mentions={recent_count} (baseline={baseline:.1f}) | "
            f"ratio={ratio:.1f}x | confidence={confidence:.2f}"
        )

        # Fire callbacks
        for callback in self._on_signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

        return signal

    async def scan_all_sources(self) -> List[CrowdSignal]:
        """Scan all crowd sources for signals"""
        self._total_scans += 1
        all_signals = []

        # Run scans in parallel
        tasks = [
            self.scan_stocktwits_trending(),
            self.scan_reddit_mentions("wallstreetbets"),
            self.scan_reddit_mentions("stocks"),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_signals.extend(result)
            elif isinstance(result, Exception):
                logger.debug(f"Scan task failed: {result}")

        # Deduplicate by symbol
        seen = set()
        unique_signals = []
        for signal in all_signals:
            if signal.symbol not in seen:
                seen.add(signal.symbol)
                unique_signals.append(signal)

        return unique_signals

    def get_status(self) -> Dict:
        """Get scanner status"""
        return {
            "running": self._running,
            "tracked_symbols": len(self._trackers),
            "valid_tickers_count": len(self._valid_tickers),
            "total_scans": self._total_scans,
            "total_signals": self._total_signals,
            "signals_by_source": self._signals_by_source,
            "thresholds": {
                "min_mention_ratio": self.MIN_MENTION_RATIO,
                "min_mentions": self.MIN_MENTIONS,
                "min_confidence": self.MIN_CONFIDENCE
            }
        }

    def get_top_mentioned(self, limit: int = 20) -> List[Dict]:
        """Get top mentioned symbols"""
        results = []
        with self._lock:
            for symbol, tracker in self._trackers.items():
                count = tracker.get_count(minutes=30)
                if count > 0:
                    results.append({
                        "symbol": symbol,
                        "mentions_30m": count,
                        "mentions_5m": tracker.get_count(minutes=5)
                    })

        results.sort(key=lambda x: x["mentions_5m"], reverse=True)
        return results[:limit]


# Singleton instance
_crowd_scanner: Optional[CrowdSignalScanner] = None


def get_crowd_scanner() -> CrowdSignalScanner:
    """Get or create the crowd scanner singleton"""
    global _crowd_scanner
    if _crowd_scanner is None:
        _crowd_scanner = CrowdSignalScanner()
    return _crowd_scanner


def wire_crowd_scanner_to_hot_queue():
    """
    Wire the crowd scanner to inject signals into HotSymbolQueue.
    Call this during app startup.
    """
    from ai.hot_symbol_queue import get_hot_symbol_queue

    scanner = get_crowd_scanner()
    queue = get_hot_symbol_queue()

    def on_signal(signal: CrowdSignal):
        """Callback when crowd signal detected"""
        queue.add(
            symbol=signal.symbol,
            reason="CROWD_SURGE",
            confidence=signal.confidence,
            metrics=signal.to_dict()
        )

    scanner.add_signal_callback(on_signal)
    logger.info("Crowd scanner wired to HotSymbolQueue")


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def test():
        scanner = get_crowd_scanner()

        # Set some valid tickers
        scanner.set_valid_tickers(["AMD", "NVDA", "TSLA", "GME", "AMC", "AAPL", "SPY"])

        print("\n=== Testing Crowd Signal Scanner ===\n")

        # Scan StockTwits
        print("Scanning StockTwits...")
        st_signals = await scanner.scan_stocktwits_trending()
        print(f"Found {len(st_signals)} signals from StockTwits")

        # Scan Reddit
        print("\nScanning Reddit...")
        reddit_signals = await scanner.scan_reddit_mentions("wallstreetbets")
        print(f"Found {len(reddit_signals)} signals from Reddit")

        print(f"\nTop mentioned: {scanner.get_top_mentioned(10)}")
        print(f"\nStatus: {scanner.get_status()}")

    asyncio.run(test())
