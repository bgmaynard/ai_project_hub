"""
Pre-Market Scanner (4 AM Daily)
================================
Builds fresh watchlist each morning:
1. Scans for pre-market movers (gap ups/downs)
2. Checks after-hours continuations from previous day
3. Monitors breaking news and logs with timestamps
4. Auto-populates common watchlist

Run at 4:00 AM ET via scheduled task.
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import httpx

logger = logging.getLogger(__name__)

# File paths
DATA_DIR = Path(__file__).parent.parent / "store" / "scanner"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NEWS_LOG_FILE = DATA_DIR / "news_log.json"
PREMARKET_WATCHLIST_FILE = DATA_DIR / "premarket_watchlist.json"
CONTINUATION_FILE = DATA_DIR / "continuations.json"

API_BASE = "http://localhost:9100"


class NewsLogger:
    """Logs all breaking news with timestamps for review"""

    def __init__(self):
        self.news_log: List[Dict] = []
        self._load_log()

    def _load_log(self):
        """Load existing news log"""
        try:
            if NEWS_LOG_FILE.exists():
                with open(NEWS_LOG_FILE, 'r') as f:
                    data = json.load(f)
                    self.news_log = data.get('news', [])
        except Exception as e:
            logger.error(f"Error loading news log: {e}")
            self.news_log = []

    def _save_log(self):
        """Save news log to file"""
        try:
            # Keep last 7 days of news
            cutoff = (datetime.now() - timedelta(days=7)).isoformat()
            self.news_log = [n for n in self.news_log if n.get('timestamp', '') > cutoff]

            with open(NEWS_LOG_FILE, 'w') as f:
                json.dump({
                    'news': self.news_log,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving news log: {e}")

    def log_news(self, symbol: str, headline: str, source: str = "benzinga"):
        """Log a news item with timestamp"""
        now = datetime.now()
        entry = {
            'timestamp': now.isoformat(),
            'time_str': now.strftime('%I:%M %p'),
            'date': now.strftime('%Y-%m-%d'),
            'symbol': symbol.upper(),
            'headline': headline,
            'source': source
        }

        # Avoid duplicates (same symbol + headline in last 5 mins)
        recent_cutoff = (now - timedelta(minutes=5)).isoformat()
        for existing in self.news_log:
            if (existing.get('symbol') == symbol.upper() and
                existing.get('headline') == headline and
                existing.get('timestamp', '') > recent_cutoff):
                return  # Duplicate

        self.news_log.append(entry)
        self._save_log()

        logger.info(f"NEWS: ${symbol} - {headline[:60]}... {entry['time_str']}")
        return entry

    def get_today_news(self) -> List[Dict]:
        """Get all news from today"""
        today = datetime.now().strftime('%Y-%m-%d')
        return [n for n in self.news_log if n.get('date') == today]

    def get_news_for_symbol(self, symbol: str) -> List[Dict]:
        """Get all news for a specific symbol"""
        return [n for n in self.news_log if n.get('symbol') == symbol.upper()]

    def get_formatted_log(self, limit: int = 50) -> str:
        """Get formatted news log like the screenshot"""
        lines = []
        for news in sorted(self.news_log, key=lambda x: x.get('timestamp', ''), reverse=True)[:limit]:
            sym = news.get('symbol', '???')
            headline = news.get('headline', '')[:80]
            time_str = news.get('time_str', '')
            lines.append(f"${sym} - {headline} {time_str}")
        return '\n'.join(lines)


class PreMarketScanner:
    """
    Pre-market scanner that runs at 4 AM to build daily watchlist.
    """

    def __init__(self):
        self.news_logger = NewsLogger()
        self.watchlist: List[str] = []
        self.continuations: List[Dict] = []
        self.premarket_movers: List[Dict] = []
        self.is_running = False
        self._load_state()

    def _load_state(self):
        """Load previous state"""
        try:
            if PREMARKET_WATCHLIST_FILE.exists():
                with open(PREMARKET_WATCHLIST_FILE, 'r') as f:
                    data = json.load(f)
                    self.watchlist = data.get('symbols', [])
                    self.premarket_movers = data.get('movers', [])

            if CONTINUATION_FILE.exists():
                with open(CONTINUATION_FILE, 'r') as f:
                    data = json.load(f)
                    self.continuations = data.get('continuations', [])
        except Exception as e:
            logger.error(f"Error loading state: {e}")

    def _save_state(self):
        """Save current state"""
        try:
            with open(PREMARKET_WATCHLIST_FILE, 'w') as f:
                json.dump({
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'symbols': self.watchlist,
                    'movers': self.premarket_movers,
                    'updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    async def scan_premarket_movers(self) -> List[Dict]:
        """
        Scan for pre-market movers using Schwab/Polygon data.
        Returns list of stocks with significant pre-market movement.
        """
        movers = []

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Get gap scanner results
                try:
                    resp = await client.get(f"{API_BASE}/api/scanner/gaps")
                    if resp.status_code == 200:
                        data = resp.json()
                        for stock in data.get('gaps', []):
                            if abs(stock.get('gap_percent', 0)) >= 5:
                                movers.append({
                                    'symbol': stock.get('symbol'),
                                    'price': stock.get('price'),
                                    'gap_percent': stock.get('gap_percent'),
                                    'volume': stock.get('volume'),
                                    'source': 'gap_scanner'
                                })
                except Exception as e:
                    logger.debug(f"Gap scanner: {e}")

                # Get momentum scanner results
                try:
                    resp = await client.get(f"{API_BASE}/api/scanner/momentum")
                    if resp.status_code == 200:
                        data = resp.json()
                        for stock in data.get('movers', data.get('results', [])):
                            symbol = stock.get('symbol')
                            if symbol and symbol not in [m['symbol'] for m in movers]:
                                change = stock.get('change_percent', stock.get('change', 0))
                                if abs(change) >= 5:
                                    movers.append({
                                        'symbol': symbol,
                                        'price': stock.get('price'),
                                        'change_percent': change,
                                        'volume': stock.get('volume'),
                                        'source': 'momentum_scanner'
                                    })
                except Exception as e:
                    logger.debug(f"Momentum scanner: {e}")

                # Check Benzinga news for pre-market catalysts
                try:
                    resp = await client.get(f"{API_BASE}/api/news/fetch?limit=50")
                    if resp.status_code == 200:
                        data = resp.json()
                        for news in data.get('news', []):
                            symbols = news.get('symbols', [])
                            headline = news.get('headline', '') or news.get('title', '')

                            for sym in symbols:
                                if sym and len(sym) <= 5:  # Valid ticker
                                    # Log the news
                                    self.news_logger.log_news(sym, headline, 'benzinga')

                                    # Check if it's a catalyst
                                    catalyst_words = ['FDA', 'approval', 'earnings', 'contract',
                                                     'acquisition', 'merger', 'upgrade', 'partnership',
                                                     'trial', 'Phase', 'revenue', 'guidance']
                                    is_catalyst = any(w.lower() in headline.lower() for w in catalyst_words)

                                    if is_catalyst and sym not in [m['symbol'] for m in movers]:
                                        movers.append({
                                            'symbol': sym,
                                            'headline': headline[:100],
                                            'source': 'news_catalyst'
                                        })
                except Exception as e:
                    logger.debug(f"News fetch: {e}")

        except Exception as e:
            logger.error(f"Pre-market scan error: {e}")

        self.premarket_movers = movers
        return movers

    async def get_afterhours_continuations(self) -> List[Dict]:
        """
        Get previous day's movers that are showing after-hours continuation.
        """
        continuations = []

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Get yesterday's top movers from our log
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

                # Check if we have continuation data
                if CONTINUATION_FILE.exists():
                    with open(CONTINUATION_FILE, 'r') as f:
                        data = json.load(f)
                        prev_movers = data.get('previous_day_movers', [])

                        for sym in prev_movers:
                            try:
                                resp = await client.get(f"{API_BASE}/api/price/{sym}", timeout=5)
                                if resp.status_code == 200:
                                    quote = resp.json()
                                    # Check for after-hours continuation (still moving)
                                    if quote.get('change_percent', 0) > 2:
                                        continuations.append({
                                            'symbol': sym,
                                            'price': quote.get('price'),
                                            'change_percent': quote.get('change_percent'),
                                            'reason': 'afterhours_continuation'
                                        })
                            except:
                                pass

        except Exception as e:
            logger.error(f"Continuation check error: {e}")

        self.continuations = continuations
        return continuations

    async def build_daily_watchlist(self) -> List[str]:
        """
        Build fresh watchlist for the day:
        1. Clear old watchlist
        2. Add pre-market movers (>5% gap/move)
        3. Add after-hours continuations
        4. Add news catalyst stocks
        5. Filter by price range ($1-$20) and volume
        """
        logger.info("=" * 50)
        logger.info("BUILDING DAILY WATCHLIST - 4 AM SCAN")
        logger.info("=" * 50)

        new_watchlist = []

        # 1. Scan pre-market movers
        movers = await self.scan_premarket_movers()
        logger.info(f"Found {len(movers)} pre-market movers")

        for m in movers:
            sym = m.get('symbol')
            if sym and sym not in new_watchlist:
                new_watchlist.append(sym)

        # 2. Get after-hours continuations
        continuations = await self.get_afterhours_continuations()
        logger.info(f"Found {len(continuations)} continuation plays")

        for c in continuations:
            sym = c.get('symbol')
            if sym and sym not in new_watchlist:
                new_watchlist.append(sym)

        # 3. Filter and validate
        validated = []
        async with httpx.AsyncClient(timeout=30) as client:
            for sym in new_watchlist[:30]:  # Limit to 30 candidates
                try:
                    resp = await client.get(f"{API_BASE}/api/price/{sym}", timeout=3)
                    if resp.status_code == 200:
                        quote = resp.json()
                        price = quote.get('price', 0)

                        # Filter: $1-$20 price range
                        if 1.0 <= price <= 20.0:
                            validated.append(sym)
                except:
                    pass

        self.watchlist = validated
        self._save_state()

        # 4. Push to common watchlist
        await self._sync_to_common_watchlist(validated)

        logger.info(f"Daily watchlist built: {len(validated)} symbols")
        logger.info(f"Symbols: {validated}")

        return validated

    async def _sync_to_common_watchlist(self, symbols: List[str]):
        """Sync new watchlist to the common data bus"""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Clear old watchlist
                await client.delete(f"{API_BASE}/api/worklist/clear")

                # Add new symbols
                for sym in symbols:
                    await client.post(
                        f"{API_BASE}/api/worklist/add",
                        json={"symbol": sym}
                    )

                logger.info(f"Synced {len(symbols)} symbols to common watchlist")
        except Exception as e:
            logger.error(f"Sync error: {e}")

    async def run_continuous_news_monitor(self):
        """
        Continuous news monitoring loop.
        Checks for breaking news every 30 seconds and logs it.
        """
        self.is_running = True
        logger.info("Starting continuous news monitor...")

        seen_headlines = set()

        while self.is_running:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(f"{API_BASE}/api/news/fetch?limit=20")
                    if resp.status_code == 200:
                        data = resp.json()

                        for news in data.get('news', []):
                            headline = news.get('headline', '') or news.get('title', '')
                            symbols = news.get('symbols', [])

                            # Skip if already seen
                            headline_key = f"{headline[:50]}"
                            if headline_key in seen_headlines:
                                continue
                            seen_headlines.add(headline_key)

                            # Log news for each symbol
                            for sym in symbols:
                                if sym and len(sym) <= 5:
                                    self.news_logger.log_news(sym, headline, 'benzinga')

                # Keep seen_headlines from growing too large
                if len(seen_headlines) > 1000:
                    seen_headlines = set(list(seen_headlines)[-500:])

            except Exception as e:
                logger.debug(f"News monitor error: {e}")

            await asyncio.sleep(30)  # Check every 30 seconds

    def stop(self):
        """Stop the scanner"""
        self.is_running = False

    def get_status(self) -> Dict:
        """Get scanner status"""
        return {
            'is_running': self.is_running,
            'watchlist_count': len(self.watchlist),
            'watchlist': self.watchlist,
            'premarket_movers': len(self.premarket_movers),
            'continuations': len(self.continuations),
            'news_today': len(self.news_logger.get_today_news()),
            'last_updated': datetime.now().isoformat()
        }

    def get_news_log(self, limit: int = 50) -> List[Dict]:
        """Get recent news log"""
        return sorted(
            self.news_logger.news_log,
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )[:limit]

    def get_formatted_news(self, limit: int = 50) -> str:
        """Get formatted news like the screenshot"""
        return self.news_logger.get_formatted_log(limit)


# Singleton instance
_scanner: Optional[PreMarketScanner] = None


def get_premarket_scanner() -> PreMarketScanner:
    """Get or create pre-market scanner singleton"""
    global _scanner
    if _scanner is None:
        _scanner = PreMarketScanner()
    return _scanner


async def run_4am_scan():
    """
    Main entry point for 4 AM scheduled task.
    """
    scanner = get_premarket_scanner()

    print("=" * 60)
    print(f"  PRE-MARKET SCAN - {datetime.now().strftime('%Y-%m-%d %I:%M %p')}")
    print("=" * 60)

    # Build daily watchlist
    watchlist = await scanner.build_daily_watchlist()

    print(f"\nDaily Watchlist ({len(watchlist)} symbols):")
    for sym in watchlist:
        print(f"  - {sym}")

    print("\nToday's News:")
    print(scanner.get_formatted_news(20))

    return watchlist


# CLI entry point
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(run_4am_scan())
