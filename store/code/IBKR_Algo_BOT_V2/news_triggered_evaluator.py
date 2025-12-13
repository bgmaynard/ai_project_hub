"""
NEWS-TRIGGERED MARKET CONSENSUS SYSTEM
======================================
Monitors breaking news, evaluates symbols against strategy criteria,
adds to watchlist and runs AI evaluation, tracks sector sentiment.

This builds a real-time consensus of market activity:
1. Monitor Benzinga news feed for breaking news
2. Evaluate symbols against Warrior Trading criteria
3. Add qualifying symbols to watchlist + run AI evaluation
4. Track sector sentiment to identify market themes
5. Generate market consensus report
"""

import sys
import time
import json
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('news_triggered_evaluator.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    # News API
    'BENZINGA_KEY': 'bz.MUTADSLMPPPHDWEGOYUMSFHUGH5TS7TD',
    'BENZINGA_URL': 'https://api.benzinga.com/api/v2/news',

    # API endpoints
    'API_BASE': 'http://localhost:9100',
    'ALPACA_API': 'http://localhost:9100/api/alpaca',

    # Warrior Trading criteria
    'MIN_PRICE': 0.50,
    'MAX_PRICE': 20.00,
    'MIN_GAP_PCT': 4.0,
    'MAX_FLOAT': 20_000_000,  # 20M shares
    'MAX_SPREAD_PCT': 3.0,

    # Timing
    'NEWS_POLL_INTERVAL': 10,  # seconds
    'EVAL_DELAY': 2,  # seconds between evaluations
}

# Sector mapping
SECTOR_MAP = {
    'biotech': ['FDA', 'drug', 'trial', 'pharma', 'clinical', 'therapy', 'cancer', 'treatment'],
    'tech': ['AI', 'software', 'cloud', 'cyber', 'data', 'digital', 'tech', 'semiconductor'],
    'energy': ['oil', 'gas', 'solar', 'wind', 'energy', 'EV', 'battery', 'renewable'],
    'finance': ['bank', 'loan', 'credit', 'financial', 'insurance', 'mortgage', 'fintech'],
    'crypto': ['bitcoin', 'crypto', 'blockchain', 'mining', 'token', 'defi'],
    'retail': ['store', 'retail', 'consumer', 'shop', 'e-commerce', 'sales'],
    'healthcare': ['hospital', 'health', 'medical', 'diagnostic', 'device'],
    'meme': ['reddit', 'squeeze', 'short interest', 'WSB', 'moon', 'ape'],
}

# Bullish catalysts
BULLISH_CATALYSTS = [
    'fda approv', 'breakthrough', 'beat', 'upgrade', 'buy rating',
    'contract', 'partnership', 'acquisition', 'spike', 'surge',
    'soar', 'rally', 'breakout', 'positive', 'strong', 'exceed',
    'profit', 'win', 'green light', 'cleared', 'agree', 'awarded'
]

# Bearish catalysts (to avoid)
BEARISH_CATALYSTS = [
    'downgrade', 'sell', 'cut', 'miss', 'decline', 'lawsuit',
    'dilution', 'offering', 'secondary', 'fraud', 'investigation',
    'bankruptcy', 'default', 'layoff', 'warning'
]

# Storage files
CONSENSUS_FILE = Path('store/market_consensus.json')
SECTOR_SENTIMENT_FILE = Path('store/sector_sentiment.json')


class NewsTriggeredEvaluator:
    """Monitors news, evaluates stocks, builds market consensus"""

    def __init__(self):
        self.seen_headlines: Set[str] = set()
        self.qualified_symbols: Dict[str, Dict] = {}
        self.sector_sentiment: Dict[str, Dict] = defaultdict(lambda: {
            'bullish': 0, 'bearish': 0, 'neutral': 0,
            'symbols': [], 'last_updated': None
        })
        self.market_consensus: Dict = {
            'overall_sentiment': 'NEUTRAL',
            'hot_sectors': [],
            'top_movers': [],
            'news_velocity': 0,
            'qualified_count': 0,
            'last_updated': None
        }
        self.stats = {
            'news_processed': 0,
            'symbols_checked': 0,
            'symbols_qualified': 0,
            'evaluations_run': 0
        }

        # Load existing data
        self._load_data()

        # Initialize watchlist evaluator
        self.evaluator = None
        try:
            from watchlist_evaluator import WatchlistEvaluator
            self.evaluator = WatchlistEvaluator()
            logger.info("Watchlist evaluator initialized")
        except Exception as e:
            logger.warning(f"Could not load watchlist evaluator: {e}")

    def _load_data(self):
        """Load existing consensus and sentiment data"""
        try:
            if CONSENSUS_FILE.exists():
                self.market_consensus = json.loads(CONSENSUS_FILE.read_text())
            if SECTOR_SENTIMENT_FILE.exists():
                self.sector_sentiment = defaultdict(
                    lambda: {'bullish': 0, 'bearish': 0, 'neutral': 0, 'symbols': [], 'last_updated': None},
                    json.loads(SECTOR_SENTIMENT_FILE.read_text())
                )
        except Exception as e:
            logger.error(f"Error loading data: {e}")

    def _save_data(self):
        """Save consensus and sentiment data"""
        try:
            CONSENSUS_FILE.parent.mkdir(parents=True, exist_ok=True)
            CONSENSUS_FILE.write_text(json.dumps(self.market_consensus, indent=2, default=str))
            SECTOR_SENTIMENT_FILE.write_text(json.dumps(dict(self.sector_sentiment), indent=2, default=str))
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def fetch_news(self) -> List[Dict]:
        """Fetch latest news from Benzinga"""
        try:
            params = {
                'token': CONFIG['BENZINGA_KEY'],
                'pageSize': 20,
                'displayOutput': 'full',
                'sort': 'created:desc'
            }
            r = requests.get(CONFIG['BENZINGA_URL'], params=params, timeout=10)

            if r.status_code != 200:
                logger.warning(f"Benzinga API returned {r.status_code}")
                return []

            root = ET.fromstring(r.content)
            news_items = []

            for item in root.findall('.//item'):
                title_el = item.find('title')
                stocks_el = item.find('stocks')
                created_el = item.find('created')

                if title_el is None:
                    continue

                headline = title_el.text or ''
                created = created_el.text if created_el is not None else ''

                # Get associated symbols
                symbols = []
                if stocks_el is not None:
                    for stock in stocks_el.findall('.//stock'):
                        name = stock.get('name', '')
                        if name:
                            symbols.append(name.upper())

                news_items.append({
                    'headline': headline,
                    'symbols': symbols,
                    'created': created,
                    'hash': hash(headline)
                })

            return news_items

        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []

    def analyze_headline(self, headline: str) -> Tuple[str, Optional[str], List[str]]:
        """
        Analyze headline for sentiment and sector
        Returns: (sentiment, catalyst, sectors)
        """
        headline_lower = headline.lower()

        # Check for bearish first
        for kw in BEARISH_CATALYSTS:
            if kw in headline_lower:
                return 'BEARISH', kw, self._detect_sectors(headline_lower)

        # Check for bullish
        for kw in BULLISH_CATALYSTS:
            if kw in headline_lower:
                return 'BULLISH', kw, self._detect_sectors(headline_lower)

        return 'NEUTRAL', None, self._detect_sectors(headline_lower)

    def _detect_sectors(self, text: str) -> List[str]:
        """Detect sectors mentioned in text"""
        sectors = []
        for sector, keywords in SECTOR_MAP.items():
            if any(kw.lower() in text.lower() for kw in keywords):
                sectors.append(sector)
        return sectors

    def check_stock_criteria(self, symbol: str) -> Tuple[bool, str, Dict]:
        """
        Check if stock meets Warrior Trading criteria
        Returns: (passes, reason, data)
        """
        try:
            # Get quote
            r = requests.get(f"{CONFIG['ALPACA_API']}/quote/{symbol}", timeout=5)
            if r.status_code != 200:
                return False, "No quote data", {}

            quote = r.json()
            price = float(quote.get('last', quote.get('bid', 0)))
            bid = float(quote.get('bid', 0))
            ask = float(quote.get('ask', 0))
            volume = int(quote.get('volume', 0))
            change_pct = float(quote.get('change_percent', 0))

            if price <= 0:
                price = ask if ask > 0 else bid

            data = {
                'price': price,
                'bid': bid,
                'ask': ask,
                'volume': volume,
                'change_pct': change_pct
            }

            fails = []

            # Price range
            if price < CONFIG['MIN_PRICE']:
                fails.append(f"price ${price:.2f} < ${CONFIG['MIN_PRICE']}")
            if price > CONFIG['MAX_PRICE']:
                fails.append(f"price ${price:.2f} > ${CONFIG['MAX_PRICE']}")

            # Spread check
            if bid > 0 and ask > 0:
                spread_pct = (ask - bid) / bid * 100
                data['spread_pct'] = spread_pct
                if spread_pct > CONFIG['MAX_SPREAD_PCT']:
                    fails.append(f"spread {spread_pct:.1f}% > {CONFIG['MAX_SPREAD_PCT']}%")

            # Try to get float (optional - don't fail if unavailable)
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                float_shares = ticker.info.get('floatShares', 0)
                if float_shares:
                    data['float'] = float_shares
                    if float_shares > CONFIG['MAX_FLOAT']:
                        fails.append(f"float {float_shares/1e6:.1f}M > {CONFIG['MAX_FLOAT']/1e6:.0f}M")
            except:
                pass

            if fails:
                return False, " | ".join(fails), data

            return True, "QUALIFIED", data

        except Exception as e:
            return False, f"Error: {e}", {}

    def add_to_watchlist(self, symbol: str, catalyst: str = None, sectors: List[str] = None) -> bool:
        """Add symbol to watchlist"""
        try:
            r = requests.post(
                f"{CONFIG['API_BASE']}/api/worklist/add",
                json={'symbol': symbol},
                timeout=5
            )
            return r.json().get('success', False)
        except:
            return False

    def run_evaluation(self, symbol: str) -> Optional[Dict]:
        """Run full AI evaluation on symbol"""
        try:
            if self.evaluator:
                result = self.evaluator.evaluate_symbol(symbol)
                self.stats['evaluations_run'] += 1
                return result
        except Exception as e:
            logger.error(f"Evaluation error for {symbol}: {e}")
        return None

    def update_sector_sentiment(self, sectors: List[str], sentiment: str, symbol: str):
        """Update sector sentiment tracking"""
        for sector in sectors:
            self.sector_sentiment[sector][sentiment.lower()] += 1
            if symbol not in self.sector_sentiment[sector]['symbols']:
                self.sector_sentiment[sector]['symbols'].append(symbol)
            self.sector_sentiment[sector]['last_updated'] = datetime.now().isoformat()

    def calculate_market_consensus(self):
        """Calculate overall market consensus from sector data"""
        total_bullish = sum(s['bullish'] for s in self.sector_sentiment.values())
        total_bearish = sum(s['bearish'] for s in self.sector_sentiment.values())
        total = total_bullish + total_bearish

        if total == 0:
            sentiment = 'NEUTRAL'
        elif total_bullish > total_bearish * 1.5:
            sentiment = 'STRONGLY_BULLISH'
        elif total_bullish > total_bearish:
            sentiment = 'BULLISH'
        elif total_bearish > total_bullish * 1.5:
            sentiment = 'STRONGLY_BEARISH'
        elif total_bearish > total_bullish:
            sentiment = 'BEARISH'
        else:
            sentiment = 'MIXED'

        # Find hot sectors (most activity)
        sector_activity = []
        for sector, data in self.sector_sentiment.items():
            activity = data['bullish'] + data['bearish'] + data['neutral']
            if activity > 0:
                net_sentiment = (data['bullish'] - data['bearish']) / activity
                sector_activity.append({
                    'sector': sector,
                    'activity': activity,
                    'net_sentiment': net_sentiment,
                    'symbols': data['symbols'][-5:]  # Last 5 symbols
                })

        sector_activity.sort(key=lambda x: x['activity'], reverse=True)

        self.market_consensus = {
            'overall_sentiment': sentiment,
            'bullish_count': total_bullish,
            'bearish_count': total_bearish,
            'hot_sectors': sector_activity[:5],
            'qualified_symbols': list(self.qualified_symbols.keys()),
            'qualified_count': len(self.qualified_symbols),
            'stats': self.stats,
            'last_updated': datetime.now().isoformat()
        }

        self._save_data()
        return self.market_consensus

    def process_news(self):
        """Process latest news and evaluate stocks"""
        news_items = self.fetch_news()
        new_qualifiers = []

        for item in news_items:
            headline = item['headline']
            symbols = item['symbols']

            # Skip seen headlines
            if item['hash'] in self.seen_headlines:
                continue

            self.seen_headlines.add(item['hash'])
            self.stats['news_processed'] += 1

            # Analyze headline
            sentiment, catalyst, sectors = self.analyze_headline(headline)

            # Skip bearish news (avoid these stocks)
            if sentiment == 'BEARISH':
                for sym in symbols:
                    self.update_sector_sentiment(sectors, sentiment, sym)
                continue

            # Process bullish/neutral news with catalyst
            if catalyst and symbols:
                logger.info(f"\n[{sentiment}] {headline[:70]}...")
                logger.info(f"  Catalyst: {catalyst} | Symbols: {symbols} | Sectors: {sectors}")

                for symbol in symbols[:3]:  # Max 3 symbols per news
                    self.stats['symbols_checked'] += 1

                    # Check criteria
                    passes, reason, data = self.check_stock_criteria(symbol)

                    if passes:
                        self.stats['symbols_qualified'] += 1
                        logger.info(f"  *** {symbol}: QUALIFIED @ ${data.get('price', 0):.2f} ***")

                        # Store qualified symbol
                        self.qualified_symbols[symbol] = {
                            'catalyst': catalyst,
                            'sectors': sectors,
                            'data': data,
                            'qualified_at': datetime.now().isoformat()
                        }

                        # Add to watchlist
                        if self.add_to_watchlist(symbol, catalyst, sectors):
                            logger.info(f"    -> Added to watchlist")
                            new_qualifiers.append(symbol)

                        # Update sector sentiment
                        self.update_sector_sentiment(sectors, 'BULLISH', symbol)
                    else:
                        logger.info(f"  {symbol}: REJECTED - {reason}")
                        self.update_sector_sentiment(sectors, 'NEUTRAL', symbol)

        return new_qualifiers

    def print_status(self):
        """Print current status"""
        consensus = self.calculate_market_consensus()

        logger.info("\n" + "=" * 60)
        logger.info("MARKET CONSENSUS REPORT")
        logger.info("=" * 60)
        logger.info(f"Overall Sentiment: {consensus['overall_sentiment']}")
        logger.info(f"Bullish: {consensus['bullish_count']} | Bearish: {consensus['bearish_count']}")
        logger.info(f"Qualified Symbols: {consensus['qualified_count']}")

        if consensus['hot_sectors']:
            logger.info("\nHot Sectors:")
            for sector in consensus['hot_sectors'][:3]:
                sentiment_str = "BULLISH" if sector['net_sentiment'] > 0 else "BEARISH" if sector['net_sentiment'] < 0 else "MIXED"
                logger.info(f"  {sector['sector'].upper()}: {sector['activity']} mentions ({sentiment_str})")
                if sector['symbols']:
                    logger.info(f"    Symbols: {', '.join(sector['symbols'])}")

        logger.info(f"\nStats: {self.stats['news_processed']} news | {self.stats['symbols_checked']} checked | {self.stats['symbols_qualified']} qualified")
        logger.info("=" * 60 + "\n")

    def run(self):
        """Main monitoring loop"""
        logger.info("=" * 60)
        logger.info("NEWS-TRIGGERED MARKET CONSENSUS SYSTEM")
        logger.info("=" * 60)
        logger.info(f"Criteria: ${CONFIG['MIN_PRICE']:.2f}-${CONFIG['MAX_PRICE']:.2f} | Float <{CONFIG['MAX_FLOAT']/1e6:.0f}M | Spread <{CONFIG['MAX_SPREAD_PCT']}%")
        logger.info("Monitoring Benzinga news feed...")
        logger.info("=" * 60 + "\n")

        cycle = 0
        while True:
            try:
                # Process news
                new_qualifiers = self.process_news()

                # Run evaluation on new qualifiers
                for symbol in new_qualifiers:
                    logger.info(f"\nRunning AI evaluation for {symbol}...")
                    time.sleep(CONFIG['EVAL_DELAY'])
                    result = self.run_evaluation(symbol)
                    if result:
                        score = result.get('score', 0)
                        rec = result.get('recommendation', 'HOLD')
                        logger.info(f"  {symbol}: Score {score}/100 -> {rec}")

                cycle += 1

                # Status update every minute
                if cycle % 6 == 0:
                    self.print_status()

                # Keep seen_headlines from growing too large
                if len(self.seen_headlines) > 1000:
                    self.seen_headlines = set(list(self.seen_headlines)[-500:])

            except Exception as e:
                logger.error(f"Error in main loop: {e}")

            time.sleep(CONFIG['NEWS_POLL_INTERVAL'])


if __name__ == '__main__':
    evaluator = NewsTriggeredEvaluator()
    try:
        evaluator.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        evaluator._save_data()
