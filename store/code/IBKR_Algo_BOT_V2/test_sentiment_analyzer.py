"""
Tests for Warrior Trading Sentiment Analyzer
Phase 5: Sentiment Analysis Testing

Tests all components of the sentiment analysis system
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
# Import sentiment analyzer components
from ai.warrior_sentiment_analyzer import (AggregatedSentiment,
                                           FinBERTSentimentAnalyzer,
                                           NewsAPIClient, RedditClient,
                                           SentimentSignal, TwitterClient,
                                           WarriorSentimentAnalyzer,
                                           get_sentiment_analyzer)


class TestFinBERTSentimentAnalyzer:
    """Test FinBERT sentiment analysis"""

    def test_init(self):
        """Test FinBERT initialization"""
        analyzer = FinBERTSentimentAnalyzer()
        assert analyzer is not None

    def test_analyze_positive_text(self):
        """Test positive sentiment detection"""
        analyzer = FinBERTSentimentAnalyzer()

        text = "Stock is breaking out with strong bullish momentum, buy calls now!"
        score, confidence = analyzer.analyze(text)

        assert score > 0, "Should detect positive sentiment"
        assert 0 <= confidence <= 1.0, "Confidence should be in valid range"

    def test_analyze_negative_text(self):
        """Test negative sentiment detection"""
        analyzer = FinBERTSentimentAnalyzer()

        text = "Stock is crashing with massive bearish breakdown, sell everything!"
        score, confidence = analyzer.analyze(text)

        assert score < 0, "Should detect negative sentiment"
        assert 0 <= confidence <= 1.0, "Confidence should be in valid range"

    def test_analyze_neutral_text(self):
        """Test neutral sentiment"""
        analyzer = FinBERTSentimentAnalyzer()

        text = "The company announced quarterly earnings today."
        score, confidence = analyzer.analyze(text)

        assert -1.0 <= score <= 1.0, "Score should be in valid range"
        assert 0 <= confidence <= 1.0, "Confidence should be in valid range"

    def test_analyze_empty_text(self):
        """Test empty text handling"""
        analyzer = FinBERTSentimentAnalyzer()

        score, confidence = analyzer.analyze("")

        assert score == 0.0, "Empty text should return neutral"
        assert confidence == 0.0, "Empty text should have no confidence"

    def test_rule_based_fallback(self):
        """Test rule-based sentiment as fallback"""
        analyzer = FinBERTSentimentAnalyzer()

        # Force rule-based mode
        original_model = analyzer.model
        analyzer.model = None

        text = "bullish breakout momentum rally strong"
        score, confidence = analyzer._rule_based_sentiment(text)

        assert score > 0, "Should detect positive keywords"
        assert confidence > 0, "Should have some confidence"

        # Restore
        analyzer.model = original_model


class TestNewsAPIClient:
    """Test News API integration"""

    def test_init_with_api_key(self):
        """Test initialization with API key"""
        client = NewsAPIClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.cache == {}

    def test_init_without_api_key(self):
        """Test initialization without API key"""
        with patch.dict("os.environ", {}, clear=True):
            client = NewsAPIClient()
            assert client.api_key is None

    @pytest.mark.asyncio
    async def test_fetch_news_without_api_key(self):
        """Test fetch returns empty list without API key"""
        client = NewsAPIClient()
        client.api_key = None

        articles = await client.fetch_news("AAPL")
        assert articles == []

    @pytest.mark.asyncio
    async def test_fetch_news_caching(self):
        """Test news caching mechanism"""
        client = NewsAPIClient(api_key="test_key")

        # Mock the session response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "articles": [
                    {"title": "Test News", "publishedAt": "2025-01-01T00:00:00Z"}
                ]
            }
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            # First fetch
            articles1 = await client.fetch_news("AAPL", hours=24)

            # Second fetch (should use cache)
            articles2 = await client.fetch_news("AAPL", hours=24)

            assert articles1 == articles2
            # Should only make one actual API call due to caching
            assert mock_session.get.call_count == 1


class TestTwitterClient:
    """Test Twitter integration"""

    def test_init_without_credentials(self):
        """Test initialization without credentials"""
        with patch.dict("os.environ", {}, clear=True):
            client = TwitterClient()
            assert client.client is None

    @pytest.mark.asyncio
    async def test_fetch_tweets_without_client(self):
        """Test fetch returns empty list without client"""
        client = TwitterClient()
        client.client = None

        tweets = await client.fetch_tweets("AAPL")
        assert tweets == []


class TestRedditClient:
    """Test Reddit integration"""

    def test_init_without_credentials(self):
        """Test initialization without credentials"""
        with patch.dict("os.environ", {}, clear=True):
            client = RedditClient()
            assert client.reddit is None

    @pytest.mark.asyncio
    async def test_fetch_mentions_without_client(self):
        """Test fetch returns empty list without client"""
        client = RedditClient()
        client.reddit = None

        mentions = await client.fetch_mentions("AAPL")
        assert mentions == []


class TestWarriorSentimentAnalyzer:
    """Test main sentiment analyzer"""

    def test_init(self):
        """Test analyzer initialization"""
        analyzer = WarriorSentimentAnalyzer()

        assert analyzer.sentiment_engine is not None
        assert analyzer.news_client is not None
        assert analyzer.twitter_client is not None
        assert analyzer.reddit_client is not None
        assert analyzer.sentiment_cache == {}

    @pytest.mark.asyncio
    async def test_analyze_symbol_no_data(self):
        """Test analysis with no data available"""
        analyzer = WarriorSentimentAnalyzer()

        # Mock all data sources to return empty
        analyzer.news_client.fetch_news = AsyncMock(return_value=[])
        analyzer.twitter_client.fetch_tweets = AsyncMock(return_value=[])
        analyzer.reddit_client.fetch_mentions = AsyncMock(return_value=[])

        sentiment = await analyzer.analyze_symbol("TEST", hours=24)

        assert sentiment.symbol == "TEST"
        assert sentiment.signals_count == 0
        assert sentiment.overall_score == 0.0
        assert sentiment.overall_confidence == 0.0
        assert sentiment.trending == False

    @pytest.mark.asyncio
    async def test_analyze_symbol_with_news(self):
        """Test analysis with news data"""
        analyzer = WarriorSentimentAnalyzer()

        # Mock news data
        mock_articles = [
            {
                "title": "AAPL stock surges on strong earnings",
                "description": "Apple shows bullish momentum",
                "publishedAt": datetime.now().isoformat() + "Z",
                "url": "http://example.com",
                "source": {"name": "TestNews"},
            }
        ]

        analyzer.news_client.fetch_news = AsyncMock(return_value=mock_articles)
        analyzer.twitter_client.fetch_tweets = AsyncMock(return_value=[])
        analyzer.reddit_client.fetch_mentions = AsyncMock(return_value=[])

        sentiment = await analyzer.analyze_symbol("AAPL", hours=24, sources=["news"])

        assert sentiment.symbol == "AAPL"
        assert sentiment.signals_count > 0
        assert "news" in sentiment.source_scores
        assert sentiment.source_counts["news"] > 0

    @pytest.mark.asyncio
    async def test_caching(self):
        """Test sentiment caching"""
        analyzer = WarriorSentimentAnalyzer()

        # Mock news data
        mock_articles = [
            {
                "title": "Test",
                "description": "Test",
                "publishedAt": datetime.now().isoformat() + "Z",
                "source": {"name": "Test"},
            }
        ]

        analyzer.news_client.fetch_news = AsyncMock(return_value=mock_articles)
        analyzer.twitter_client.fetch_tweets = AsyncMock(return_value=[])
        analyzer.reddit_client.fetch_mentions = AsyncMock(return_value=[])

        # First call
        sentiment1 = await analyzer.analyze_symbol("AAPL", hours=24, sources=["news"])

        # Second call (should use cache)
        sentiment2 = await analyzer.analyze_symbol("AAPL", hours=24, sources=["news"])

        # Should get same timestamp (cached)
        assert sentiment1.timestamp == sentiment2.timestamp

        # News API should only be called once
        assert analyzer.news_client.fetch_news.call_count == 1

    def test_aggregate_signals_trending_detection(self):
        """Test trending detection logic"""
        analyzer = WarriorSentimentAnalyzer()

        # Create many signals to trigger trending
        signals = []
        for i in range(25):
            signals.append(
                SentimentSignal(
                    source="twitter",
                    symbol="MEME",
                    timestamp=datetime.now(),
                    score=0.5,
                    confidence=0.7,
                    text=f"Test signal {i}",
                )
            )

        aggregated = analyzer._aggregate_signals("MEME", signals)

        assert aggregated.trending == True, "Should detect trending with 25+ signals"
        assert aggregated.signals_count == 25

    def test_aggregate_signals_momentum(self):
        """Test momentum calculation"""
        analyzer = WarriorSentimentAnalyzer()

        # Create signals with changing sentiment
        signals = []

        # Old signals (negative)
        for i in range(5):
            signals.append(
                SentimentSignal(
                    source="twitter",
                    symbol="TEST",
                    timestamp=datetime.now() - timedelta(hours=12),
                    score=-0.5,
                    confidence=0.7,
                    text=f"Bearish signal {i}",
                )
            )

        # Recent signals (positive)
        for i in range(5):
            signals.append(
                SentimentSignal(
                    source="twitter",
                    symbol="TEST",
                    timestamp=datetime.now() - timedelta(hours=1),
                    score=0.5,
                    confidence=0.7,
                    text=f"Bullish signal {i}",
                )
            )

        aggregated = analyzer._aggregate_signals("TEST", signals)

        # Momentum should be positive (recent more positive than old)
        assert aggregated.momentum > 0, "Should detect positive momentum shift"


class TestSentimentSignal:
    """Test SentimentSignal dataclass"""

    def test_create_signal(self):
        """Test creating a sentiment signal"""
        signal = SentimentSignal(
            source="news",
            symbol="AAPL",
            timestamp=datetime.now(),
            score=0.5,
            confidence=0.8,
            text="Test news article",
        )

        assert signal.source == "news"
        assert signal.symbol == "AAPL"
        assert -1.0 <= signal.score <= 1.0
        assert 0.0 <= signal.confidence <= 1.0


class TestAggregatedSentiment:
    """Test AggregatedSentiment dataclass"""

    def test_create_aggregated(self):
        """Test creating aggregated sentiment"""
        signal = SentimentSignal(
            source="news",
            symbol="TEST",
            timestamp=datetime.now(),
            score=0.5,
            confidence=0.8,
            text="Test",
        )

        aggregated = AggregatedSentiment(
            symbol="TEST",
            timestamp=datetime.now(),
            overall_score=0.5,
            overall_confidence=0.8,
            signals_count=1,
            source_scores={"news": 0.5},
            source_counts={"news": 1},
            trending=False,
            momentum=0.0,
            top_signals=[signal],
        )

        assert aggregated.symbol == "TEST"
        assert aggregated.signals_count == 1
        assert len(aggregated.top_signals) == 1


class TestGlobalInstance:
    """Test global sentiment analyzer instance"""

    def test_get_sentiment_analyzer(self):
        """Test getting global instance"""
        analyzer1 = get_sentiment_analyzer()
        analyzer2 = get_sentiment_analyzer()

        # Should return same instance
        assert analyzer1 is analyzer2


# Integration tests (require real API keys)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_sentiment_analysis():
    """
    Full integration test with real data

    Requirements:
    - ANTHROPIC_API_KEY (for FinBERT fallback if needed)
    - NEWS_API_KEY (optional)
    - TWITTER_BEARER_TOKEN (optional)
    - REDDIT credentials (optional)
    """
    analyzer = get_sentiment_analyzer()

    # Analyze a popular symbol
    sentiment = await analyzer.analyze_symbol("SPY", hours=24)

    print(f"\nSentiment Analysis for SPY:")
    print(f"  Overall Score: {sentiment.overall_score:+.3f}")
    print(f"  Confidence: {sentiment.overall_confidence:.2%}")
    print(f"  Signals: {sentiment.signals_count}")
    print(f"  Trending: {sentiment.trending}")
    print(f"  Momentum: {sentiment.momentum:+.3f}")

    if sentiment.top_signals:
        print(f"\n  Top Signal:")
        top = sentiment.top_signals[0]
        print(f"    Source: {top.source}")
        print(f"    Score: {top.score:+.2f}")
        print(f"    Text: {top.text[:100]}")

    await analyzer.close()

    # Basic validations
    assert sentiment.symbol == "SPY"
    assert -1.0 <= sentiment.overall_score <= 1.0
    assert 0.0 <= sentiment.overall_confidence <= 1.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

    # Run integration test separately
    print("\n" + "=" * 60)
    print("Running integration test...")
    print("=" * 60)
    asyncio.run(test_full_sentiment_analysis())
