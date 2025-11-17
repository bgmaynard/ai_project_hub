"""
Warrior Trading Sentiment Analyzer
Phase 5: Sentiment & Alternative Data Integration

Analyzes market sentiment from multiple sources:
- News articles (Financial news APIs)
- Twitter/X (Social media sentiment)
- Reddit (WallStreetBets, stocks, daytrading subreddits)
- StockTwits (Trading community sentiment)

Uses FinBERT for financial sentiment analysis
Aggregates signals for trading decisions
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import asyncio
import aiohttp

# NLP & Sentiment
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Data sources
try:
    import tweepy
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False
    logging.warning("tweepy not installed - Twitter integration disabled")

try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False
    logging.warning("praw not installed - Reddit integration disabled")

# Caching
from functools import lru_cache
import time


logger = logging.getLogger(__name__)


@dataclass
class SentimentSignal:
    """Individual sentiment signal from a source"""
    source: str  # 'news', 'twitter', 'reddit', 'stocktwits'
    symbol: str
    timestamp: datetime
    score: float  # -1.0 to 1.0 (negative to positive)
    confidence: float  # 0.0 to 1.0
    text: str  # Original text
    url: Optional[str] = None
    author: Optional[str] = None
    metrics: Optional[Dict] = None  # likes, retweets, upvotes, etc.


@dataclass
class BreakingNewsAlert:
    """Breaking news alert - trigger for warrior trading system"""
    symbol: str
    timestamp: datetime
    alert_type: str  # 'positive', 'negative', 'neutral'
    severity: float  # 0.0 to 1.0 (how significant is this news)
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    sources_count: int  # Number of sources reporting
    headline: str  # Main headline or summary
    signals: List[SentimentSignal]  # The breaking news signals
    trigger_time: datetime  # When this alert was triggered

    def to_dict(self) -> Dict:
        """Convert to dictionary for API response"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type,
            'severity': self.severity,
            'sentiment_score': self.sentiment_score,
            'confidence': self.confidence,
            'sources_count': self.sources_count,
            'headline': self.headline,
            'trigger_time': self.trigger_time.isoformat(),
            'signals': [
                {
                    'source': s.source,
                    'text': s.text,
                    'score': s.score,
                    'url': s.url,
                    'timestamp': s.timestamp.isoformat()
                }
                for s in self.signals[:3]  # Top 3 signals
            ]
        }


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment across all sources"""
    symbol: str
    timestamp: datetime
    overall_score: float  # -1.0 to 1.0
    overall_confidence: float  # 0.0 to 1.0
    signals_count: int
    source_scores: Dict[str, float]  # Score by source
    source_counts: Dict[str, int]  # Count by source
    trending: bool  # Is this symbol trending?
    momentum: float  # Sentiment momentum (rate of change)
    top_signals: List[SentimentSignal]  # Top 5 signals
    breaking_news: Optional['BreakingNewsAlert'] = None  # Breaking news alert if detected


class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial text
    Specialized for financial news and trading commentary
    """

    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading FinBERT model on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("✓ FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            logger.info("Falling back to rule-based sentiment")
            self.model = None
            self.tokenizer = None

    def analyze(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of text

        Returns:
            (score, confidence) where:
            - score: -1.0 (negative) to 1.0 (positive)
            - confidence: 0.0 to 1.0
        """
        if not text or not text.strip():
            return 0.0, 0.0

        if self.model is None:
            return self._rule_based_sentiment(text)

        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # FinBERT outputs: [negative, neutral, positive]
            probs = predictions[0].cpu().numpy()
            negative, neutral, positive = probs

            # Calculate score: -1 to 1
            score = positive - negative

            # Confidence: how certain we are (low neutral = high confidence)
            confidence = 1.0 - neutral

            return float(score), float(confidence)

        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return self._rule_based_sentiment(text)

    def _rule_based_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Simple rule-based sentiment as fallback
        """
        text_lower = text.lower()

        # Bullish keywords
        bullish = ['bullish', 'buy', 'calls', 'moon', 'rocket', 'breakout',
                   'surge', 'rally', 'strong', 'gap up', 'squeeze', 'momentum']

        # Bearish keywords
        bearish = ['bearish', 'sell', 'puts', 'crash', 'dump', 'breakdown',
                   'plunge', 'fall', 'weak', 'gap down', 'resistance']

        bull_count = sum(1 for word in bullish if word in text_lower)
        bear_count = sum(1 for word in bearish if word in text_lower)

        total = bull_count + bear_count
        if total == 0:
            return 0.0, 0.0

        score = (bull_count - bear_count) / total
        confidence = min(total / 5.0, 1.0)  # More keywords = more confidence

        return score, confidence


class NewsAPIClient:
    """
    News API integration for financial news
    Supports NewsAPI.org and other news aggregators
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEWS_API_KEY")
        self.base_url = "https://newsapi.org/v2"
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def fetch_news(self, symbol: str, hours: int = 24) -> List[Dict]:
        """
        Fetch recent news for a symbol

        Args:
            symbol: Stock symbol
            hours: Look back period in hours

        Returns:
            List of news articles
        """
        if not self.api_key:
            logger.warning("NEWS_API_KEY not set - news integration disabled")
            return []

        cache_key = f"{symbol}_{hours}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data

        await self._ensure_session()

        try:
            from_time = (datetime.now() - timedelta(hours=hours)).isoformat()

            params = {
                "q": symbol,
                "from": from_time,
                "sortBy": "publishedAt",
                "apiKey": self.api_key,
                "language": "en"
            }

            async with self.session.get(
                f"{self.base_url}/everything",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get("articles", [])
                    self.cache[cache_key] = (time.time(), articles)
                    return articles
                else:
                    logger.error(f"NewsAPI error: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Failed to fetch news for {symbol}: {e}")
            return []

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()


class TwitterClient:
    """
    Twitter/X integration for social sentiment
    Uses Twitter API v2 via tweepy
    """

    def __init__(self):
        if not TWITTER_AVAILABLE:
            logger.warning("Twitter integration not available")
            self.client = None
            return

        bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        if not bearer_token:
            logger.warning("TWITTER_BEARER_TOKEN not set - Twitter disabled")
            self.client = None
            return

        try:
            self.client = tweepy.Client(bearer_token=bearer_token)
            logger.info("✓ Twitter client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Twitter client: {e}")
            self.client = None

    async def fetch_tweets(self, symbol: str, max_results: int = 100) -> List[Dict]:
        """
        Fetch recent tweets about a symbol

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            max_results: Maximum tweets to fetch (10-100)

        Returns:
            List of tweet data
        """
        if not self.client:
            return []

        try:
            # Search query
            query = f"${symbol} -is:retweet lang:en"

            # Fetch tweets
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                tweet_fields=['created_at', 'public_metrics', 'author_id']
            )

            if not tweets.data:
                return []

            # Format results
            results = []
            for tweet in tweets.data:
                results.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'metrics': tweet.public_metrics,
                    'author_id': tweet.author_id
                })

            return results

        except Exception as e:
            logger.error(f"Failed to fetch tweets for ${symbol}: {e}")
            return []


class RedditClient:
    """
    Reddit integration for community sentiment
    Monitors r/wallstreetbets, r/stocks, r/daytrading, etc.
    """

    def __init__(self):
        if not REDDIT_AVAILABLE:
            logger.warning("Reddit integration not available")
            self.reddit = None
            return

        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT", "WarriorTradingBot/1.0")

        if not client_id or not client_secret:
            logger.warning("Reddit credentials not set - Reddit disabled")
            self.reddit = None
            return

        try:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            logger.info("✓ Reddit client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            self.reddit = None

    async def fetch_mentions(
        self,
        symbol: str,
        subreddits: List[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch Reddit mentions of a symbol

        Args:
            symbol: Stock symbol
            subreddits: List of subreddits to search (default: WSB, stocks, daytrading)
            limit: Max results per subreddit

        Returns:
            List of Reddit posts/comments
        """
        if not self.reddit:
            return []

        if subreddits is None:
            subreddits = ['wallstreetbets', 'stocks', 'daytrading', 'options']

        results = []

        try:
            for subreddit_name in subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)

                # Search posts
                for post in subreddit.search(symbol, limit=limit, time_filter='day'):
                    results.append({
                        'type': 'post',
                        'id': post.id,
                        'title': post.title,
                        'text': post.selftext,
                        'created_at': datetime.fromtimestamp(post.created_utc),
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'url': post.url,
                        'subreddit': subreddit_name,
                        'author': str(post.author) if post.author else '[deleted]'
                    })

            return results

        except Exception as e:
            logger.error(f"Failed to fetch Reddit mentions for {symbol}: {e}")
            return []


class WarriorSentimentAnalyzer:
    """
    Main sentiment analyzer orchestrating all data sources
    Aggregates sentiment for trading decisions
    """

    def __init__(self):
        logger.info("Initializing Warrior Sentiment Analyzer...")

        # Core sentiment engine
        self.sentiment_engine = FinBERTSentimentAnalyzer()

        # Data sources
        self.news_client = NewsAPIClient()
        self.twitter_client = TwitterClient()
        self.reddit_client = RedditClient()

        # Cache
        self.sentiment_cache: Dict[str, AggregatedSentiment] = {}
        self.cache_ttl = 300  # 5 minutes

        logger.info("✓ Warrior Sentiment Analyzer initialized")

    async def analyze_symbol(
        self,
        symbol: str,
        hours: int = 24,
        sources: List[str] = None
    ) -> AggregatedSentiment:
        """
        Analyze sentiment for a symbol across all sources

        Args:
            symbol: Stock symbol
            hours: Lookback period in hours
            sources: List of sources to use (default: all available)

        Returns:
            AggregatedSentiment object
        """
        # Check cache
        cache_key = f"{symbol}_{hours}"
        if cache_key in self.sentiment_cache:
            cached = self.sentiment_cache[cache_key]
            age = (datetime.now() - cached.timestamp).total_seconds()
            if age < self.cache_ttl:
                logger.debug(f"Using cached sentiment for {symbol}")
                return cached

        if sources is None:
            sources = ['news', 'twitter', 'reddit']

        # Collect signals from all sources
        all_signals = []

        # News
        if 'news' in sources:
            news_signals = await self._analyze_news(symbol, hours)
            all_signals.extend(news_signals)

        # Twitter
        if 'twitter' in sources:
            twitter_signals = await self._analyze_twitter(symbol)
            all_signals.extend(twitter_signals)

        # Reddit
        if 'reddit' in sources:
            reddit_signals = await self._analyze_reddit(symbol)
            all_signals.extend(reddit_signals)

        # Aggregate signals
        aggregated = self._aggregate_signals(symbol, all_signals)

        # Cache result
        self.sentiment_cache[cache_key] = aggregated

        return aggregated

    async def _analyze_news(self, symbol: str, hours: int) -> List[SentimentSignal]:
        """Analyze news articles"""
        signals = []

        articles = await self.news_client.fetch_news(symbol, hours)

        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            if not text.strip():
                continue

            score, confidence = self.sentiment_engine.analyze(text)

            signals.append(SentimentSignal(
                source='news',
                symbol=symbol,
                timestamp=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                score=score,
                confidence=confidence,
                text=text[:200],
                url=article.get('url'),
                author=article.get('source', {}).get('name')
            ))

        logger.info(f"Analyzed {len(signals)} news articles for {symbol}")
        return signals

    async def _analyze_twitter(self, symbol: str) -> List[SentimentSignal]:
        """Analyze tweets"""
        signals = []

        tweets = await self.twitter_client.fetch_tweets(symbol)

        for tweet in tweets:
            score, confidence = self.sentiment_engine.analyze(tweet['text'])

            # Weight by engagement
            metrics = tweet.get('metrics', {})
            engagement = (
                metrics.get('like_count', 0) +
                metrics.get('retweet_count', 0) * 2
            )

            # Boost confidence for high-engagement tweets
            if engagement > 100:
                confidence = min(confidence * 1.2, 1.0)

            signals.append(SentimentSignal(
                source='twitter',
                symbol=symbol,
                timestamp=tweet['created_at'],
                score=score,
                confidence=confidence,
                text=tweet['text'][:200],
                metrics=metrics
            ))

        logger.info(f"Analyzed {len(signals)} tweets for ${symbol}")
        return signals

    async def _analyze_reddit(self, symbol: str) -> List[SentimentSignal]:
        """Analyze Reddit posts"""
        signals = []

        posts = await self.reddit_client.fetch_mentions(symbol)

        for post in posts:
            text = f"{post.get('title', '')} {post.get('text', '')}"
            score, confidence = self.sentiment_engine.analyze(text)

            # Weight by upvotes
            upvotes = post.get('score', 0)
            if upvotes > 100:
                confidence = min(confidence * 1.3, 1.0)

            signals.append(SentimentSignal(
                source='reddit',
                symbol=symbol,
                timestamp=post['created_at'],
                score=score,
                confidence=confidence,
                text=text[:200],
                url=post.get('url'),
                author=post.get('author'),
                metrics={'upvotes': upvotes, 'comments': post.get('num_comments', 0)}
            ))

        logger.info(f"Analyzed {len(signals)} Reddit posts for {symbol}")
        return signals

    def _aggregate_signals(
        self,
        symbol: str,
        signals: List[SentimentSignal]
    ) -> AggregatedSentiment:
        """
        Aggregate signals into overall sentiment
        """
        if not signals:
            return AggregatedSentiment(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_score=0.0,
                overall_confidence=0.0,
                signals_count=0,
                source_scores={},
                source_counts={},
                trending=False,
                momentum=0.0,
                top_signals=[]
            )

        # Calculate weighted average score
        total_weight = 0.0
        weighted_sum = 0.0

        source_scores = defaultdict(list)
        source_counts = defaultdict(int)

        for signal in signals:
            weight = signal.confidence
            weighted_sum += signal.score * weight
            total_weight += weight

            source_scores[signal.source].append(signal.score)
            source_counts[signal.source] += 1

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        overall_confidence = min(len(signals) / 50.0, 1.0)  # More signals = more confident

        # Calculate per-source averages
        avg_source_scores = {
            source: np.mean(scores)
            for source, scores in source_scores.items()
        }

        # Sort signals by confidence * abs(score)
        sorted_signals = sorted(
            signals,
            key=lambda s: s.confidence * abs(s.score),
            reverse=True
        )
        top_signals = sorted_signals[:5]

        # Detect if trending (high volume of signals)
        trending = len(signals) > 20

        # Calculate momentum (recent vs older signals)
        recent_cutoff = datetime.now() - timedelta(hours=6)
        recent_signals = [s for s in signals if s.timestamp > recent_cutoff]
        older_signals = [s for s in signals if s.timestamp <= recent_cutoff]

        recent_score = np.mean([s.score for s in recent_signals]) if recent_signals else 0.0
        older_score = np.mean([s.score for s in older_signals]) if older_signals else 0.0
        momentum = recent_score - older_score

        # Detect breaking news
        breaking_news = self._detect_breaking_news(symbol, signals)

        return AggregatedSentiment(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_score=overall_score,
            overall_confidence=overall_confidence,
            signals_count=len(signals),
            source_scores=avg_source_scores,
            source_counts=dict(source_counts),
            trending=trending,
            momentum=momentum,
            top_signals=top_signals,
            breaking_news=breaking_news
        )

    def _detect_breaking_news(
        self,
        symbol: str,
        signals: List[SentimentSignal]
    ) -> Optional[BreakingNewsAlert]:
        """
        Detect breaking news based on recent, high-impact signals

        Criteria for breaking news:
        1. Very recent (within last 2 hours)
        2. High absolute sentiment score (>0.6)
        3. High confidence (>0.7)
        4. Multiple sources OR high engagement
        """
        if not signals:
            return None

        # Filter for very recent signals (last 2 hours)
        recent_cutoff = datetime.now() - timedelta(hours=2)
        recent_signals = [s for s in signals if s.timestamp > recent_cutoff]

        if not recent_signals:
            return None

        # Filter for high-impact signals (high score & confidence)
        high_impact = [
            s for s in recent_signals
            if abs(s.score) > 0.6 and s.confidence > 0.7
        ]

        if not high_impact:
            return None

        # Check if we have multiple sources or high engagement
        sources = set(s.source for s in high_impact)
        has_multiple_sources = len(sources) >= 2

        # Check for high engagement on social signals
        high_engagement = False
        for signal in high_impact:
            if signal.metrics:
                if signal.source == 'twitter':
                    likes = signal.metrics.get('like_count', 0)
                    retweets = signal.metrics.get('retweet_count', 0)
                    if likes + retweets * 2 > 500:  # High engagement threshold
                        high_engagement = True
                        break
                elif signal.source == 'reddit':
                    upvotes = signal.metrics.get('upvotes', 0)
                    if upvotes > 200:  # High upvotes threshold
                        high_engagement = True
                        break

        # Trigger breaking news if multiple sources OR high engagement
        if not (has_multiple_sources or high_engagement):
            return None

        # Calculate breaking news metrics
        avg_score = np.mean([s.score for s in high_impact])
        avg_confidence = np.mean([s.confidence for s in high_impact])

        # Determine alert type
        if avg_score > 0.3:
            alert_type = 'positive'
        elif avg_score < -0.3:
            alert_type = 'negative'
        else:
            alert_type = 'neutral'

        # Calculate severity (0.0 to 1.0)
        # Based on: absolute score, confidence, number of sources, recency
        score_factor = min(abs(avg_score), 1.0)
        confidence_factor = avg_confidence
        source_factor = min(len(sources) / 3.0, 1.0)  # Cap at 3 sources
        recency_factor = min(len(high_impact) / 10.0, 1.0)  # More signals = more severe

        severity = (
            score_factor * 0.4 +
            confidence_factor * 0.3 +
            source_factor * 0.2 +
            recency_factor * 0.1
        )

        # Sort by confidence * abs(score) to get best signals
        sorted_signals = sorted(
            high_impact,
            key=lambda s: s.confidence * abs(s.score),
            reverse=True
        )

        # Generate headline from top signal
        top_signal = sorted_signals[0]
        headline = top_signal.text[:150] + "..." if len(top_signal.text) > 150 else top_signal.text

        return BreakingNewsAlert(
            symbol=symbol,
            timestamp=top_signal.timestamp,
            alert_type=alert_type,
            severity=severity,
            sentiment_score=avg_score,
            confidence=avg_confidence,
            sources_count=len(sources),
            headline=headline,
            signals=sorted_signals[:5],  # Top 5 signals
            trigger_time=datetime.now()
        )

    async def close(self):
        """Clean up resources"""
        await self.news_client.close()


# Global instance
_sentiment_analyzer: Optional[WarriorSentimentAnalyzer] = None


def get_sentiment_analyzer() -> WarriorSentimentAnalyzer:
    """Get or create global sentiment analyzer instance"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = WarriorSentimentAnalyzer()
    return _sentiment_analyzer


# Example usage
if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def test_sentiment():
        analyzer = get_sentiment_analyzer()

        # Test symbols
        symbols = ['AAPL', 'TSLA', 'SPY']

        for symbol in symbols:
            print(f"\n{'='*60}")
            print(f"Analyzing sentiment for {symbol}")
            print('='*60)

            sentiment = await analyzer.analyze_symbol(symbol, hours=24)

            print(f"\nOverall Score: {sentiment.overall_score:+.3f}")
            print(f"Confidence: {sentiment.overall_confidence:.2%}")
            print(f"Signals: {sentiment.signals_count}")
            print(f"Trending: {sentiment.trending}")
            print(f"Momentum: {sentiment.momentum:+.3f}")
            print(f"\nBy Source:")
            for source, score in sentiment.source_scores.items():
                count = sentiment.source_counts[source]
                print(f"  {source:10s}: {score:+.3f} ({count} signals)")

            print(f"\nTop Signals:")
            for i, signal in enumerate(sentiment.top_signals[:3], 1):
                print(f"  {i}. [{signal.source}] {signal.score:+.2f} - {signal.text[:100]}...")

        await analyzer.close()

    asyncio.run(test_sentiment())
