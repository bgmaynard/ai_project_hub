"""
Alpaca Stock Scanner
=====================
Real-time stock scanner using Alpaca market data

Features:
- Pre-market Gappers (stocks gapping up/down)
- High Volume Movers (unusual volume)
- Momentum Stocks (strong price action)
- Breakout Candidates (near resistance levels)
- Technical Screeners (RSI, MACD signals)

Author: AI Trading Bot Team
Version: 1.0
"""

import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from alpaca_market_data import get_alpaca_market_data
from config.broker_config import get_broker_config

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Scanner result for a single stock"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    avg_volume: int
    relative_volume: float
    gap_percent: float = 0.0
    signal: str = "NEUTRAL"
    score: float = 50.0
    timestamp: str = ""
    # News/Catalyst fields (Warrior Trading Pillar)
    has_catalyst: bool = False
    catalyst_type: str = ""  # 'breaking_news', 'earnings', 'fda', 'contract', etc.
    catalyst_headline: str = ""
    catalyst_sentiment: float = 0.0  # -1.0 to 1.0
    catalyst_severity: float = 0.0  # 0.0 to 1.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScannerConfig:
    """Scanner configuration"""
    # Gap scanner settings
    min_gap_percent: float = 3.0
    max_gap_percent: float = 20.0

    # Volume settings
    min_volume: int = 500000
    min_relative_volume: float = 1.5

    # Price settings
    min_price: float = 5.0
    max_price: float = 500.0

    # Momentum settings
    min_momentum_score: float = 60.0

    # General settings
    max_results: int = 20


class AlpacaScanner:
    """
    Comprehensive stock scanner using Alpaca market data
    All data is sourced EXCLUSIVELY from Alpaca API (no IBKR).

    Integrates with WarriorSentimentAnalyzer for breaking news detection
    (Ross Cameron's 5th Pillar: News Catalyst)
    """

    # Predefined watchlists for different strategies
    UNIVERSE = {
        'large_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B',
                      'JPM', 'V', 'UNH', 'XOM', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'CVX',
                      'BAC', 'ABBV', 'KO', 'PEP', 'MRK', 'COST', 'AVGO', 'TMO', 'LLY'],

        'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD',
                 'INTC', 'ORCL', 'CSCO', 'ADBE', 'CRM', 'NFLX', 'AVGO', 'QCOM',
                 'TXN', 'AMAT', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MCHP'],

        'meme': ['GME', 'AMC', 'PLTR', 'SOFI', 'NIO', 'LCID', 'RIVN', 'TLRY',
                 'BBBY', 'BB', 'NOK', 'WISH', 'CLOV', 'SNDL'],

        'high_volume': ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'AMZN',
                        'META', 'GOOGL', 'NFLX', 'DIS', 'BABA', 'NIO', 'PLTR'],

        'etf': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG',
                'BND', 'LQD', 'TLT', 'GLD', 'SLV', 'USO', 'XLE', 'XLF', 'XLK'],

        'popular': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL', 'AMZN',
                    'META', 'AMD', 'NFLX', 'DIS', 'BABA', 'NIO', 'PLTR', 'SOFI',
                    'F', 'INTC', 'BAC', 'JPM'],

        # Most Popular Momentum - High volume retail favorites with strong momentum
        # These are stocks frequently traded by retail with high volume and volatility
        'popular_momentum': ['TSLA', 'NVDA', 'AMD', 'META', 'PLTR', 'SOFI', 'COIN',
                             'HOOD', 'MARA', 'RIOT', 'NIO', 'RIVN', 'LCID', 'GME',
                             'AMC', 'MSTR', 'SMCI', 'ARM', 'RKLB', 'IONQ', 'RGTI',
                             'QUBT', 'MU', 'AVGO', 'AMAT', 'CRWD', 'SNOW', 'DDOG',
                             'NET', 'ZS', 'PANW', 'SQ', 'SHOP', 'ROKU', 'UPST'],

        # Warrior Trading - Low float momentum candidates
        # These are stocks that frequently meet Ross Cameron's 5 Pillars criteria
        'warrior': ['MARA', 'RIOT', 'SNDL', 'TLRY', 'AMC', 'GME', 'BBBY', 'MULN',
                    'FFIE', 'GOEV', 'NKLA', 'WKHS', 'RIDE', 'HYMC', 'ATER', 'RDBX',
                    'BBIG', 'PROG', 'CLOV', 'WISH', 'SOFI', 'LCID', 'RIVN', 'DNA',
                    'OPEN', 'UPST', 'AFRM', 'HOOD', 'COIN', 'MSTR', 'CVNA', 'W',
                    'BYND', 'SNAP', 'PINS', 'RBLX', 'U', 'DKNG', 'PENN', 'CHPT']
    }

    def __init__(self, config: Optional[ScannerConfig] = None):
        """
        Initialize scanner

        Args:
            config: Scanner configuration (uses defaults if None)
        """
        self.config = config or ScannerConfig()
        self.market_data = get_alpaca_market_data()

        # Initialize sentiment analyzer for breaking news detection
        self.sentiment_analyzer = None
        try:
            from ai.warrior_sentiment_analyzer import get_sentiment_analyzer
            self.sentiment_analyzer = get_sentiment_analyzer()
            logger.info("AlpacaScanner initialized with sentiment analyzer")
        except ImportError:
            logger.warning("Sentiment analyzer not available - news catalyst detection disabled")
        except Exception as e:
            logger.warning(f"Could not initialize sentiment analyzer: {e}")

        logger.info("AlpacaScanner initialized")

    def check_breaking_news(self, symbol: str) -> Optional[Dict]:
        """
        Check for breaking news catalyst on a symbol

        Args:
            symbol: Stock symbol to check

        Returns:
            Dict with catalyst info or None if no breaking news
        """
        if not self.sentiment_analyzer:
            return None

        try:
            # Check for breaking news alerts
            import asyncio

            # Run async method in sync context
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Get breaking news for symbol
            if hasattr(self.sentiment_analyzer, 'detect_breaking_news'):
                alert = loop.run_until_complete(
                    self.sentiment_analyzer.detect_breaking_news(symbol)
                )

                if alert:
                    return {
                        'has_catalyst': True,
                        'catalyst_type': 'breaking_news',
                        'catalyst_headline': alert.headline[:100] if alert.headline else '',
                        'catalyst_sentiment': alert.sentiment_score,
                        'catalyst_severity': alert.severity,
                        'sources_count': alert.sources_count
                    }

            # Fallback: Get aggregated sentiment
            if hasattr(self.sentiment_analyzer, 'get_aggregated_sentiment'):
                sentiment = loop.run_until_complete(
                    self.sentiment_analyzer.get_aggregated_sentiment(symbol, hours=4)
                )

                if sentiment and sentiment.is_trending:
                    return {
                        'has_catalyst': True,
                        'catalyst_type': 'trending',
                        'catalyst_headline': f"Trending with {sentiment.signal_count} signals",
                        'catalyst_sentiment': sentiment.overall_score,
                        'catalyst_severity': sentiment.confidence,
                        'sources_count': sentiment.signal_count
                    }

        except Exception as e:
            logger.debug(f"Error checking breaking news for {symbol}: {e}")

        return None

    def scan_breaking_news(self) -> List[ScanResult]:
        """
        Scan for stocks with breaking news catalysts

        This is a key component of Ross Cameron's Warrior Trading strategy.
        Breaking news often triggers significant price movement.

        Returns:
            List of ScanResult objects for stocks with active news catalysts
        """
        logger.info("Scanning for breaking news catalysts...")

        if not self.sentiment_analyzer:
            logger.warning("Sentiment analyzer not available - returning empty results")
            return []

        results = []

        try:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Get all breaking news alerts
            if hasattr(self.sentiment_analyzer, 'get_all_breaking_news'):
                alerts = loop.run_until_complete(
                    self.sentiment_analyzer.get_all_breaking_news()
                )

                for alert in alerts:
                    # Get price data for the symbol
                    quote = self.market_data.get_latest_quote(alert.symbol)
                    if not quote:
                        continue

                    current_price = quote.get('last', 0)

                    result = ScanResult(
                        symbol=alert.symbol,
                        price=round(current_price, 2),
                        change=0.0,
                        change_percent=0.0,
                        volume=0,
                        avg_volume=0,
                        relative_volume=0.0,
                        gap_percent=0.0,
                        signal=f"NEWS - {alert.alert_type.upper()}",
                        score=alert.severity * 100,
                        has_catalyst=True,
                        catalyst_type='breaking_news',
                        catalyst_headline=alert.headline[:100] if alert.headline else '',
                        catalyst_sentiment=alert.sentiment_score,
                        catalyst_severity=alert.severity
                    )
                    results.append(result)

                logger.info(f"Found {len(results)} stocks with breaking news")

        except Exception as e:
            logger.error(f"Error scanning for breaking news: {e}")

        return results[:self.config.max_results]

    def scan_gappers(self, direction: str = "up") -> List[ScanResult]:
        """
        Scan for stocks with significant gaps from previous close

        Args:
            direction: "up" for gap ups, "down" for gap downs, "both" for all

        Returns:
            List of ScanResult objects sorted by gap %
        """
        logger.info(f"Scanning for gap {direction} stocks...")

        # Use popular stocks + tech stocks for gap scanning
        symbols = list(set(self.UNIVERSE['popular'] + self.UNIVERSE['tech']))

        results = []
        for symbol in symbols:
            try:
                result = self._analyze_gap(symbol)
                if result:
                    # Filter by direction
                    if direction == "up" and result.gap_percent >= self.config.min_gap_percent:
                        results.append(result)
                    elif direction == "down" and result.gap_percent <= -self.config.min_gap_percent:
                        results.append(result)
                    elif direction == "both" and abs(result.gap_percent) >= self.config.min_gap_percent:
                        results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {symbol} for gaps: {e}")
                continue

        # Sort by gap % (descending for up, ascending for down)
        if direction == "down":
            results.sort(key=lambda x: x.gap_percent)
        else:
            results.sort(key=lambda x: x.gap_percent, reverse=True)

        return results[:self.config.max_results]

    def scan_volume_leaders(self) -> List[ScanResult]:
        """
        Scan for stocks with unusually high volume (relative to average)

        Returns:
            List of ScanResult objects sorted by relative volume
        """
        logger.info("Scanning for volume leaders...")

        symbols = self.UNIVERSE['high_volume']

        results = []
        for symbol in symbols:
            try:
                result = self._analyze_volume(symbol)
                if result and result.relative_volume >= self.config.min_relative_volume:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {symbol} for volume: {e}")
                continue

        # Sort by relative volume (descending)
        results.sort(key=lambda x: x.relative_volume, reverse=True)

        return results[:self.config.max_results]

    def scan_momentum(self) -> List[ScanResult]:
        """
        Scan for stocks with strong momentum

        Returns:
            List of ScanResult objects sorted by momentum score
        """
        logger.info("Scanning for momentum stocks...")

        symbols = list(set(self.UNIVERSE['popular'] + self.UNIVERSE['tech']))

        results = []
        for symbol in symbols:
            try:
                result = self._analyze_momentum(symbol)
                if result and result.score >= self.config.min_momentum_score:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {symbol} for momentum: {e}")
                continue

        # Sort by momentum score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:self.config.max_results]

    def scan_popular_momentum(self) -> List[ScanResult]:
        """
        Most Popular Momentum Stocks Scanner

        Scans the most actively traded retail favorites for momentum signals.
        Includes: TSLA, NVDA, AMD, META, crypto-related, AI stocks, meme stocks

        Filters:
        - Strong intraday momentum (price action)
        - Above average volume
        - Positive 5-day trend

        Returns:
            List of ScanResult objects sorted by momentum score
        """
        logger.info("=" * 60)
        logger.info("MOST POPULAR MOMENTUM SCANNER")
        logger.info("=" * 60)
        logger.info("Scanning retail favorites for momentum signals...")

        symbols = self.UNIVERSE['popular_momentum']
        logger.info(f"Scanning {len(symbols)} popular momentum stocks...")

        results = []
        for symbol in symbols:
            try:
                result = self._analyze_momentum(symbol)
                if result:
                    # Lower threshold for popular momentum - we want to see all with any momentum
                    if result.score >= 40:  # Show stocks with at least neutral+ momentum
                        results.append(result)
            except Exception as e:
                logger.debug(f"Error analyzing {symbol}: {e}")
                continue

        # Sort by momentum score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Found {len(results)} stocks with momentum signals")
        logger.info("=" * 60)

        return results[:self.config.max_results]

    def scan_breakouts(self) -> List[ScanResult]:
        """
        Scan for stocks breaking out to new highs

        Returns:
            List of ScanResult objects sorted by breakout strength
        """
        logger.info("Scanning for breakout stocks...")

        symbols = list(set(self.UNIVERSE['popular'] + self.UNIVERSE['tech']))

        results = []
        for symbol in symbols:
            try:
                result = self._analyze_breakout(symbol)
                if result and result.signal in ["STRONG_BULLISH", "BULLISH"]:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {symbol} for breakouts: {e}")
                continue

        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:self.config.max_results]

    def scan_warrior(self) -> List[ScanResult]:
        """
        Warrior Trading Scanner - Ross Cameron's 5 Pillars of Stock Selection

        ALL 5 PILLARS MUST BE MET (no exceptions):

        PILLAR 1: Float - Low float stocks (under 20M shares) - We use known low-float universe
        PILLAR 2: Price - Between $2 and $20 STRICTLY
        PILLAR 3: Relative Volume - At least 2x average daily volume
        PILLAR 4: Gap/Catalyst - Gapping at least 4% OR has news catalyst
        PILLAR 5: Trading Volume - At least 500K shares traded today

        Returns:
            List of ScanResult objects meeting ALL 5 Warrior Trading criteria
        """
        logger.info("=" * 60)
        logger.info("WARRIOR TRADING SCANNER - Ross Cameron's 5 Pillars")
        logger.info("=" * 60)
        logger.info("Requirements: ALL 5 must be met STRICTLY")
        logger.info("  1. Low Float (using known low-float universe)")
        logger.info("  2. Price: $2.00 - $20.00 (STRICT)")
        logger.info("  3. Relative Volume: 2.0x+ average (STRICT)")
        logger.info("  4. Gap: 4.0%+ up or down (STRICT)")
        logger.info("  5. Volume: 500,000+ shares today (STRICT)")
        logger.info("-" * 60)

        # Use warrior universe - these are known low-float stocks (PILLAR 1 pre-filtered)
        symbols = list(set(self.UNIVERSE['warrior'] + self.UNIVERSE['meme']))
        logger.info(f"Scanning {len(symbols)} low-float stocks...")

        results = []
        passed = 0
        failed_price = 0
        failed_rvol = 0
        failed_gap = 0
        failed_volume = 0
        failed_data = 0

        for symbol in symbols:
            try:
                # Get snapshot for real-time volume data (more accurate than historical bars)
                snapshot = self.market_data.get_snapshot(symbol)

                # Also get historical data for gap and avg volume calculations
                end = datetime.now()
                start = end - timedelta(days=30)
                df = self.market_data.get_historical_bars(symbol, "1Day", start, end, limit=30)

                if df is None or len(df) < 2:
                    failed_data += 1
                    logger.debug(f"✗ {symbol}: No historical data available")
                    continue

                # Get current price - prefer snapshot, fallback to historical
                if snapshot and snapshot.get('close'):
                    current_price = snapshot['close']
                    current_volume = snapshot.get('volume', 0)
                elif snapshot and snapshot.get('last_price'):
                    current_price = snapshot['last_price']
                    current_volume = snapshot.get('volume', int(df['Volume'].iloc[-1]))
                else:
                    current_price = float(df['Close'].iloc[-1])
                    current_volume = int(df['Volume'].iloc[-1])

                # PILLAR 2: Price $2-$20 (STRICT HARD REQUIREMENT)
                if current_price < 2.0:
                    failed_price += 1
                    logger.debug(f"✗ {symbol}: Price ${current_price:.2f} < $2.00 minimum")
                    continue
                if current_price > 20.0:
                    failed_price += 1
                    logger.debug(f"✗ {symbol}: Price ${current_price:.2f} > $20.00 maximum")
                    continue

                # Calculate average volume (excluding today)
                avg_volume = int(df['Volume'].iloc[:-1].mean()) if len(df) > 1 else 1
                if avg_volume < 1:
                    avg_volume = 1  # Prevent division by zero

                # Calculate relative volume
                relative_volume = current_volume / avg_volume

                # PILLAR 3: Relative Volume 2x+ (STRICT HARD REQUIREMENT)
                if relative_volume < 2.0:
                    failed_rvol += 1
                    logger.debug(f"✗ {symbol}: RVOL {relative_volume:.2f}x < 2.0x minimum")
                    continue

                # Calculate gap from previous close
                previous_close = float(df['Close'].iloc[-2])
                if previous_close <= 0:
                    failed_data += 1
                    continue
                gap_percent = ((current_price - previous_close) / previous_close) * 100

                # PILLAR 4: Gap 4%+ up or down (STRICT HARD REQUIREMENT)
                if abs(gap_percent) < 4.0:
                    failed_gap += 1
                    logger.debug(f"✗ {symbol}: Gap {gap_percent:+.2f}% < 4.0% minimum")
                    continue

                # PILLAR 5: Volume 500K+ (STRICT HARD REQUIREMENT)
                if current_volume < 500000:
                    failed_volume += 1
                    logger.debug(f"✗ {symbol}: Volume {current_volume:,} < 500,000 minimum")
                    continue

                # ============================================================
                # ALL 5 PILLARS MET - This stock qualifies for Warrior Trading
                # ============================================================

                # Calculate score based on how well it exceeds requirements
                score = 100  # Base score for meeting all pillars

                # Bonus points for exceeding thresholds
                # Sweet spot price range $5-$15
                if 5.0 <= current_price <= 15.0:
                    score += 10

                # Very high relative volume
                if relative_volume >= 5.0:
                    score += 15
                elif relative_volume >= 3.0:
                    score += 5

                # Large gap
                if abs(gap_percent) >= 10.0:
                    score += 15
                elif abs(gap_percent) >= 7.0:
                    score += 5

                # Very high volume
                if current_volume >= 2000000:
                    score += 10
                elif current_volume >= 1000000:
                    score += 5

                # Check for NEWS CATALYST (Ross Cameron's key pillar)
                has_catalyst = False
                catalyst_type = ""
                catalyst_headline = ""
                catalyst_sentiment = 0.0
                catalyst_severity = 0.0

                news_info = self.check_breaking_news(symbol)
                if news_info:
                    has_catalyst = True
                    catalyst_type = news_info.get('catalyst_type', 'news')
                    catalyst_headline = news_info.get('catalyst_headline', '')
                    catalyst_sentiment = news_info.get('catalyst_sentiment', 0.0)
                    catalyst_severity = news_info.get('catalyst_severity', 0.0)

                    # Big bonus for having news catalyst!
                    score += 25
                    if catalyst_severity >= 0.7:
                        score += 15  # High severity news
                    logger.info(f"  📰 NEWS CATALYST: {catalyst_headline[:50]}...")

                # Determine signal based on gap direction
                # Scanner shows direction, bot decides whether to trade based on long_only setting
                if gap_percent >= 0:
                    signal = "GAP UP"
                else:
                    signal = "GAP DOWN"

                # Add catalyst indicator to signal if present
                if has_catalyst:
                    signal = f"{signal} + NEWS"

                change = current_price - previous_close

                result = ScanResult(
                    symbol=symbol,
                    price=round(current_price, 2),
                    change=round(change, 2),
                    change_percent=round(gap_percent, 2),
                    volume=current_volume,
                    avg_volume=avg_volume,
                    relative_volume=round(relative_volume, 2),
                    gap_percent=round(gap_percent, 2),
                    signal=f"5/5 PILLARS - {signal}",
                    score=score,
                    has_catalyst=has_catalyst,
                    catalyst_type=catalyst_type,
                    catalyst_headline=catalyst_headline,
                    catalyst_sentiment=catalyst_sentiment,
                    catalyst_severity=catalyst_severity
                )

                results.append(result)
                passed += 1
                catalyst_str = " 📰 NEWS" if has_catalyst else ""
                logger.info(f"✓ {symbol}: ${current_price:.2f} | Gap: {gap_percent:+.1f}% | RVOL: {relative_volume:.1f}x | Vol: {current_volume:,}{catalyst_str}")

            except Exception as e:
                logger.debug(f"Error analyzing {symbol}: {e}")
                failed_data += 1
                continue

        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        logger.info("-" * 60)
        logger.info(f"SCAN COMPLETE:")
        logger.info(f"  ✓ Passed all 5 pillars: {passed}")
        logger.info(f"  ✗ Failed Price ($2-$20): {failed_price}")
        logger.info(f"  ✗ Failed RVOL (2x+): {failed_rvol}")
        logger.info(f"  ✗ Failed Gap (4%+): {failed_gap}")
        logger.info(f"  ✗ Failed Volume (500K+): {failed_volume}")
        logger.info(f"  ✗ No data available: {failed_data}")
        logger.info("=" * 60)

        return results[:self.config.max_results]

    def _analyze_warrior_criteria(self, symbol: str) -> Optional[ScanResult]:
        """
        Analyze a stock against Warrior Trading 5 Pillars

        Scoring:
        - 20 points per pillar met (100 max)
        - Bonus points for exceeding thresholds
        """
        try:
            # Get latest quote first
            quote = self.market_data.get_latest_quote(symbol)
            if not quote:
                return None

            current_price = quote['last']

            # Get historical data
            end = datetime.now()
            start = end - timedelta(days=20)

            df = self.market_data.get_historical_bars(
                symbol,
                timeframe="1Day",
                start=start,
                end=end,
                limit=20
            )

            # If no historical data, create minimal result based on quote
            if df is None or len(df) < 2:
                # Can still score on price range
                score = 0
                pillars_met = []

                if 2.0 <= current_price <= 20.0:
                    score += 20
                    pillars_met.append("PRICE")
                    if 5.0 <= current_price <= 15.0:
                        score += 5

                return ScanResult(
                    symbol=symbol,
                    price=round(current_price, 2),
                    change=0,
                    change_percent=0,
                    volume=0,
                    avg_volume=0,
                    relative_volume=0,
                    gap_percent=0,
                    signal=f"QUOTE_ONLY ({','.join(pillars_met)})" if pillars_met else "QUOTE_ONLY",
                    score=score
                )

            # Initialize scoring
            score = 0
            pillars_met = []

            # PILLAR 1: Price Range ($2-$20 sweet spot)
            if 2.0 <= current_price <= 20.0:
                score += 20
                pillars_met.append("PRICE")
                # Bonus for ideal range $5-$15
                if 5.0 <= current_price <= 15.0:
                    score += 5
            elif current_price < 2.0 or current_price > 50.0:
                score -= 10  # Penalty for outside tradeable range

            # PILLAR 2: Relative Volume (2x+ average)
            avg_volume = df['Volume'].iloc[:-1].mean() if len(df) > 1 else df['Volume'].mean()
            current_volume = df['Volume'].iloc[-1]
            relative_volume = current_volume / avg_volume if avg_volume > 0 else 0

            if relative_volume >= 2.0:
                score += 20
                pillars_met.append("RVOL")
                # Bonus for very high relative volume
                if relative_volume >= 5.0:
                    score += 10
                elif relative_volume >= 3.0:
                    score += 5

            # PILLAR 3: Gap % (4%+ gap up)
            if len(df) >= 2:
                previous_close = df['Close'].iloc[-2]
                gap_percent = ((current_price - previous_close) / previous_close) * 100
            else:
                gap_percent = 0

            if gap_percent >= 4.0:
                score += 20
                pillars_met.append("GAP")
                # Bonus for larger gaps
                if gap_percent >= 10.0:
                    score += 10
                elif gap_percent >= 7.0:
                    score += 5

            # PILLAR 4: Momentum (price action strength)
            if len(df) >= 5:
                five_day_change = ((current_price - df['Close'].iloc[-5]) / df['Close'].iloc[-5]) * 100
                if five_day_change >= 10.0:
                    score += 20
                    pillars_met.append("MOMENTUM")
                    if five_day_change >= 25.0:
                        score += 10

            # PILLAR 5: Volume threshold (min 500K for liquidity)
            if current_volume >= 500000:
                score += 20
                pillars_met.append("VOLUME")
                if current_volume >= 2000000:
                    score += 5

            # Determine signal based on score
            if score >= 80:
                signal = "STRONG_BULLISH"
            elif score >= 60:
                signal = "BULLISH"
            elif score >= 40:
                signal = "NEUTRAL"
            else:
                signal = "WEAK"

            # Calculate change
            if len(df) >= 2:
                change = current_price - df['Close'].iloc[-2]
                change_percent = (change / df['Close'].iloc[-2]) * 100
            else:
                change = 0
                change_percent = 0

            return ScanResult(
                symbol=symbol,
                price=round(current_price, 2),
                change=round(change, 2),
                change_percent=round(change_percent, 2),
                volume=int(current_volume),
                avg_volume=int(avg_volume),
                relative_volume=round(relative_volume, 2),
                gap_percent=round(gap_percent, 2),
                signal=f"{signal} ({','.join(pillars_met)})" if pillars_met else signal,
                score=score
            )

        except Exception as e:
            logger.debug(f"Warrior analysis error for {symbol}: {e}")
            return None

    def scan_preset(self, preset: str) -> List[ScanResult]:
        """
        Run a preset scanner

        Args:
            preset: Preset name (gainers, losers, volume, momentum, popular_momentum, tech, popular, warrior, breaking_news)

        Returns:
            List of ScanResult objects matching the preset criteria
        """
        logger.info(f"Running preset scanner: {preset}")

        if preset == "gainers":
            return self.scan_gappers(direction="up")
        elif preset == "losers":
            return self.scan_gappers(direction="down")
        elif preset == "volume":
            return self.scan_volume_leaders()
        elif preset == "momentum":
            return self.scan_momentum()
        elif preset == "popular_momentum":
            # Most Popular Momentum - retail favorites with momentum analysis
            return self.scan_popular_momentum()
        elif preset == "breakouts":
            return self.scan_breakouts()
        elif preset == "warrior":
            # Warrior Trading - returns all stocks meeting 5 pillars (gap up OR down)
            return self.scan_warrior()
        elif preset == "breaking_news" or preset == "news":
            # Breaking news scanner - stocks with active news catalysts
            return self.scan_breaking_news()
        elif preset in self.UNIVERSE:
            # Return symbols from predefined universe
            return self._get_universe_quotes(preset)
        else:
            # Default to popular stocks
            return self._get_universe_quotes('popular')

    def _analyze_gap(self, symbol: str) -> Optional[ScanResult]:
        """Analyze a stock for gap from previous close"""
        try:
            # Get historical data (last 5 days)
            end = datetime.now()
            start = end - timedelta(days=5)

            df = self.market_data.get_historical_bars(
                symbol,
                timeframe="1Day",
                start=start,
                end=end,
                limit=5
            )

            if df is None or len(df) < 2:
                return None

            # Get latest quote
            quote = self.market_data.get_latest_quote(symbol)
            if not quote:
                return None

            # Calculate gap
            previous_close = df['Close'].iloc[-2]
            current_price = quote['last']
            gap_percent = ((current_price - previous_close) / previous_close) * 100

            # Calculate volume metrics
            avg_volume = df['Volume'].mean()
            current_volume = df['Volume'].iloc[-1]
            relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Calculate change from previous close
            change = current_price - previous_close
            change_percent = gap_percent

            # Calculate score based on gap size and volume
            score = min(100, abs(gap_percent) * 5 + (relative_volume - 1) * 20)

            return ScanResult(
                symbol=symbol,
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=int(current_volume),
                avg_volume=int(avg_volume),
                relative_volume=relative_volume,
                gap_percent=gap_percent,
                signal="BULLISH" if gap_percent > 0 else "BEARISH",
                score=score
            )

        except Exception as e:
            logger.error(f"Error in _analyze_gap for {symbol}: {e}")
            return None

    def _analyze_volume(self, symbol: str) -> Optional[ScanResult]:
        """Analyze a stock for unusual volume"""
        try:
            # Get historical data
            end = datetime.now()
            start = end - timedelta(days=30)

            df = self.market_data.get_historical_bars(
                symbol,
                timeframe="1Day",
                start=start,
                end=end,
                limit=30
            )

            if df is None or len(df) < 5:
                return None

            # Get latest quote
            quote = self.market_data.get_latest_quote(symbol)
            if not quote:
                return None

            # Calculate volume metrics
            avg_volume = df['Volume'].iloc[:-1].mean()  # Exclude today
            current_volume = df['Volume'].iloc[-1]
            relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Calculate price change
            previous_close = df['Close'].iloc[-2]
            current_price = quote['last']
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100

            # Score based on relative volume and price movement
            score = min(100, (relative_volume - 1) * 30 + abs(change_percent) * 5)

            # Signal based on volume and price direction
            if relative_volume >= 2.0 and change_percent > 2:
                signal = "STRONG_BULLISH"
            elif relative_volume >= 1.5 and change_percent > 0:
                signal = "BULLISH"
            elif relative_volume >= 2.0 and change_percent < -2:
                signal = "STRONG_BEARISH"
            elif relative_volume >= 1.5 and change_percent < 0:
                signal = "BEARISH"
            else:
                signal = "NEUTRAL"

            return ScanResult(
                symbol=symbol,
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=int(current_volume),
                avg_volume=int(avg_volume),
                relative_volume=relative_volume,
                signal=signal,
                score=score
            )

        except Exception as e:
            logger.error(f"Error in _analyze_volume for {symbol}: {e}")
            return None

    def _analyze_momentum(self, symbol: str) -> Optional[ScanResult]:
        """Analyze a stock for momentum"""
        try:
            # Get historical data
            end = datetime.now()
            start = end - timedelta(days=60)

            df = self.market_data.get_historical_bars(
                symbol,
                timeframe="1Day",
                start=start,
                end=end,
                limit=60
            )

            if df is None or len(df) < 20:
                return None

            # Get latest quote
            quote = self.market_data.get_latest_quote(symbol)
            if not quote:
                return None

            current_price = quote['last']

            # Calculate momentum indicators
            # 1. Rate of Change (ROC) over different periods
            roc_5 = ((df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6]) * 100
            roc_10 = ((df['Close'].iloc[-1] - df['Close'].iloc[-11]) / df['Close'].iloc[-11]) * 100
            roc_20 = ((df['Close'].iloc[-1] - df['Close'].iloc[-21]) / df['Close'].iloc[-21]) * 100

            # 2. Volume trend
            recent_vol = df['Volume'].iloc[-5:].mean()
            older_vol = df['Volume'].iloc[-20:-5].mean()
            volume_trend = (recent_vol / older_vol) if older_vol > 0 else 1.0

            # 3. Price trend (above moving averages)
            ma_20 = df['Close'].iloc[-20:].mean()
            price_vs_ma20 = ((current_price - ma_20) / ma_20) * 100

            # Calculate momentum score (0-100)
            score = 50  # Baseline
            score += min(20, roc_5)  # Short-term momentum
            score += min(15, roc_10 / 2)  # Medium-term momentum
            score += min(10, roc_20 / 3)  # Long-term momentum
            score += min(10, (volume_trend - 1) * 10)  # Volume support
            score += min(5, price_vs_ma20)  # Trend strength
            score = max(0, min(100, score))

            # Determine signal
            if score >= 80:
                signal = "STRONG_BULLISH"
            elif score >= 65:
                signal = "BULLISH"
            elif score >= 35:
                signal = "NEUTRAL"
            elif score >= 20:
                signal = "BEARISH"
            else:
                signal = "STRONG_BEARISH"

            # Calculate metrics
            previous_close = df['Close'].iloc[-2]
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100

            return ScanResult(
                symbol=symbol,
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=int(df['Volume'].iloc[-1]),
                avg_volume=int(df['Volume'].mean()),
                relative_volume=volume_trend,
                signal=signal,
                score=score
            )

        except Exception as e:
            logger.error(f"Error in _analyze_momentum for {symbol}: {e}")
            return None

    def _analyze_breakout(self, symbol: str) -> Optional[ScanResult]:
        """Analyze a stock for breakout potential"""
        try:
            # Get historical data
            end = datetime.now()
            start = end - timedelta(days=90)

            df = self.market_data.get_historical_bars(
                symbol,
                timeframe="1Day",
                start=start,
                end=end,
                limit=90
            )

            if df is None or len(df) < 30:
                return None

            # Get latest quote
            quote = self.market_data.get_latest_quote(symbol)
            if not quote:
                return None

            current_price = quote['last']

            # Calculate resistance levels
            # 1. 52-week high
            high_52w = df['High'].iloc[-60:].max()
            distance_to_high = ((high_52w - current_price) / current_price) * 100

            # 2. Recent high (20 days)
            recent_high = df['High'].iloc[-20:].max()
            distance_to_recent = ((recent_high - current_price) / current_price) * 100

            # 3. Volume on breakout
            avg_volume = df['Volume'].iloc[-20:-1].mean()
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Calculate breakout score
            score = 50

            # Near 52-week high
            if distance_to_high < 5:
                score += 25
            elif distance_to_high < 10:
                score += 15

            # Breaking recent high
            if current_price > recent_high:
                score += 20
            elif distance_to_recent < 2:
                score += 10

            # Volume confirmation
            if volume_ratio > 2.0:
                score += 15
            elif volume_ratio > 1.5:
                score += 10

            score = max(0, min(100, score))

            # Determine signal
            if score >= 80 and volume_ratio > 1.5:
                signal = "STRONG_BULLISH"
            elif score >= 65:
                signal = "BULLISH"
            else:
                signal = "NEUTRAL"

            # Calculate metrics
            previous_close = df['Close'].iloc[-2]
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100

            return ScanResult(
                symbol=symbol,
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=int(current_volume),
                avg_volume=int(avg_volume),
                relative_volume=volume_ratio,
                signal=signal,
                score=score
            )

        except Exception as e:
            logger.error(f"Error in _analyze_breakout for {symbol}: {e}")
            return None

    def _get_universe_quotes(self, universe: str) -> List[ScanResult]:
        """Get quotes for a predefined universe of stocks"""
        symbols = self.UNIVERSE.get(universe, self.UNIVERSE['popular'])

        results = []
        for symbol in symbols[:self.config.max_results]:
            try:
                quote = self.market_data.get_latest_quote(symbol)
                if quote:
                    results.append(ScanResult(
                        symbol=symbol,
                        price=quote['last'],
                        change=0.0,
                        change_percent=0.0,
                        volume=0,
                        avg_volume=0,
                        relative_volume=1.0,
                        signal="NEUTRAL",
                        score=50.0
                    ))
            except Exception as e:
                logger.error(f"Error getting quote for {symbol}: {e}")
                continue

        return results


# Global scanner instance
_scanner_instance: Optional[AlpacaScanner] = None


def get_alpaca_scanner() -> AlpacaScanner:
    """
    Get or create the global Alpaca scanner instance

    Returns:
        AlpacaScanner instance
    """
    global _scanner_instance

    if _scanner_instance is None:
        _scanner_instance = AlpacaScanner()

    return _scanner_instance
