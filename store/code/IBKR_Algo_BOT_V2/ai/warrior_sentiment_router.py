"""
Warrior Trading Sentiment API Router
Phase 5: Sentiment Analysis Endpoints

Provides REST API endpoints for sentiment analysis data
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ai.warrior_sentiment_analyzer import (
    get_sentiment_analyzer,
    AggregatedSentiment,
    SentimentSignal
)


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/sentiment", tags=["Sentiment Analysis"])


# Response models
class SentimentSignalResponse(BaseModel):
    """Individual sentiment signal"""
    source: str
    symbol: str
    timestamp: datetime
    score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score (-1 to 1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence (0 to 1)")
    text: str
    url: Optional[str] = None
    author: Optional[str] = None
    metrics: Optional[dict] = None


class AggregatedSentimentResponse(BaseModel):
    """Aggregated sentiment across all sources"""
    symbol: str
    timestamp: datetime
    overall_score: float = Field(..., ge=-1.0, le=1.0)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    signals_count: int
    source_scores: dict
    source_counts: dict
    trending: bool
    momentum: float
    top_signals: List[SentimentSignalResponse]


class SentimentHealthResponse(BaseModel):
    """Health status of sentiment analyzer"""
    status: str
    finbert_loaded: bool
    news_api_available: bool
    twitter_available: bool
    reddit_available: bool
    cache_size: int


# Endpoints

@router.get("/health", response_model=SentimentHealthResponse)
async def get_sentiment_health():
    """
    Get health status of sentiment analyzer

    Returns information about:
    - FinBERT model status
    - News API availability
    - Twitter API availability
    - Reddit API availability
    - Cache statistics
    """
    try:
        analyzer = get_sentiment_analyzer()

        # Check FinBERT
        finbert_loaded = analyzer.sentiment_engine.model is not None

        # Check API clients
        news_available = analyzer.news_client.api_key is not None
        twitter_available = analyzer.twitter_client.client is not None
        reddit_available = analyzer.reddit_client.reddit is not None

        # Cache stats
        cache_size = len(analyzer.sentiment_cache)

        return SentimentHealthResponse(
            status="healthy" if finbert_loaded else "degraded",
            finbert_loaded=finbert_loaded,
            news_api_available=news_available,
            twitter_available=twitter_available,
            reddit_available=reddit_available,
            cache_size=cache_size
        )

    except Exception as e:
        logger.error(f"Sentiment health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}", response_model=AggregatedSentimentResponse)
async def get_symbol_sentiment(
    symbol: str,
    hours: int = Query(24, ge=1, le=168, description="Lookback period in hours"),
    sources: Optional[str] = Query(
        None,
        description="Comma-separated sources (news,twitter,reddit)"
    )
):
    """
    Get aggregated sentiment for a symbol

    Args:
        symbol: Stock symbol (e.g., AAPL)
        hours: Lookback period in hours (1-168)
        sources: Comma-separated list of sources to include

    Returns:
        Aggregated sentiment with scores, confidence, and top signals

    Example:
        GET /api/sentiment/AAPL?hours=24&sources=news,twitter
    """
    try:
        analyzer = get_sentiment_analyzer()

        # Parse sources
        source_list = None
        if sources:
            source_list = [s.strip() for s in sources.split(',')]

        # Get sentiment
        sentiment = await analyzer.analyze_symbol(
            symbol=symbol.upper(),
            hours=hours,
            sources=source_list
        )

        # Convert to response model
        return AggregatedSentimentResponse(
            symbol=sentiment.symbol,
            timestamp=sentiment.timestamp,
            overall_score=sentiment.overall_score,
            overall_confidence=sentiment.overall_confidence,
            signals_count=sentiment.signals_count,
            source_scores=sentiment.source_scores,
            source_counts=sentiment.source_counts,
            trending=sentiment.trending,
            momentum=sentiment.momentum,
            top_signals=[
                SentimentSignalResponse(
                    source=sig.source,
                    symbol=sig.symbol,
                    timestamp=sig.timestamp,
                    score=sig.score,
                    confidence=sig.confidence,
                    text=sig.text,
                    url=sig.url,
                    author=sig.author,
                    metrics=sig.metrics
                )
                for sig in sentiment.top_signals
            ]
        )

    except Exception as e:
        logger.error(f"Failed to get sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/news", response_model=List[SentimentSignalResponse])
async def get_news_sentiment(
    symbol: str,
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(10, ge=1, le=100)
):
    """
    Get news sentiment signals for a symbol

    Args:
        symbol: Stock symbol
        hours: Lookback period
        limit: Maximum signals to return

    Returns:
        List of news sentiment signals
    """
    try:
        analyzer = get_sentiment_analyzer()

        sentiment = await analyzer.analyze_symbol(
            symbol=symbol.upper(),
            hours=hours,
            sources=['news']
        )

        # Filter and sort news signals
        news_signals = [
            sig for sig in sentiment.top_signals
            if sig.source == 'news'
        ][:limit]

        return [
            SentimentSignalResponse(
                source=sig.source,
                symbol=sig.symbol,
                timestamp=sig.timestamp,
                score=sig.score,
                confidence=sig.confidence,
                text=sig.text,
                url=sig.url,
                author=sig.author,
                metrics=sig.metrics
            )
            for sig in news_signals
        ]

    except Exception as e:
        logger.error(f"Failed to get news sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/social", response_model=List[SentimentSignalResponse])
async def get_social_sentiment(
    symbol: str,
    platform: Optional[str] = Query(None, regex="^(twitter|reddit|all)$"),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get social media sentiment signals

    Args:
        symbol: Stock symbol
        platform: Social platform (twitter, reddit, or all)
        limit: Maximum signals to return

    Returns:
        List of social sentiment signals
    """
    try:
        analyzer = get_sentiment_analyzer()

        # Determine sources
        if platform == 'twitter':
            sources = ['twitter']
        elif platform == 'reddit':
            sources = ['reddit']
        else:
            sources = ['twitter', 'reddit']

        sentiment = await analyzer.analyze_symbol(
            symbol=symbol.upper(),
            hours=24,
            sources=sources
        )

        # Get social signals
        social_signals = [
            sig for sig in sentiment.top_signals
            if sig.source in ['twitter', 'reddit']
        ][:limit]

        return [
            SentimentSignalResponse(
                source=sig.source,
                symbol=sig.symbol,
                timestamp=sig.timestamp,
                score=sig.score,
                confidence=sig.confidence,
                text=sig.text,
                url=sig.url,
                author=sig.author,
                metrics=sig.metrics
            )
            for sig in social_signals
        ]

    except Exception as e:
        logger.error(f"Failed to get social sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{symbol}/analyze")
async def trigger_analysis(
    symbol: str,
    hours: int = Query(24, ge=1, le=168)
):
    """
    Trigger immediate sentiment analysis (bypasses cache)

    Args:
        symbol: Stock symbol
        hours: Lookback period

    Returns:
        Analysis triggered confirmation
    """
    try:
        analyzer = get_sentiment_analyzer()

        # Clear cache for this symbol
        cache_key = f"{symbol.upper()}_{hours}"
        if cache_key in analyzer.sentiment_cache:
            del analyzer.sentiment_cache[cache_key]

        # Trigger new analysis
        sentiment = await analyzer.analyze_symbol(
            symbol=symbol.upper(),
            hours=hours
        )

        return {
            "status": "completed",
            "symbol": symbol.upper(),
            "signals_collected": sentiment.signals_count,
            "timestamp": sentiment.timestamp
        }

    except Exception as e:
        logger.error(f"Failed to trigger analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch/analyze")
async def batch_sentiment_analysis(
    symbols: str = Query(..., description="Comma-separated symbols (max 10)"),
    hours: int = Query(24, ge=1, le=168)
):
    """
    Analyze sentiment for multiple symbols

    Args:
        symbols: Comma-separated list of symbols (max 10)
        hours: Lookback period

    Returns:
        Dict of symbol -> sentiment score
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]

        if len(symbol_list) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 symbols allowed"
            )

        analyzer = get_sentiment_analyzer()

        results = {}

        for symbol in symbol_list:
            try:
                sentiment = await analyzer.analyze_symbol(symbol, hours)
                results[symbol] = {
                    "score": sentiment.overall_score,
                    "confidence": sentiment.overall_confidence,
                    "signals": sentiment.signals_count,
                    "trending": sentiment.trending
                }
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                results[symbol] = {
                    "error": str(e)
                }

        return results

    except Exception as e:
        logger.error(f"Batch sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trending/symbols")
async def get_trending_symbols(
    min_signals: int = Query(20, ge=5, description="Minimum signal count to be trending"),
    limit: int = Query(10, ge=1, le=50)
):
    """
    Get currently trending symbols based on social/news volume

    Args:
        min_signals: Minimum signals to be considered trending
        limit: Maximum results

    Returns:
        List of trending symbols with sentiment scores
    """
    try:
        # This would require tracking all analyzed symbols
        # For now, return cached results
        analyzer = get_sentiment_analyzer()

        trending = []

        for cache_key, sentiment in analyzer.sentiment_cache.items():
            if sentiment.trending and sentiment.signals_count >= min_signals:
                trending.append({
                    "symbol": sentiment.symbol,
                    "score": sentiment.overall_score,
                    "confidence": sentiment.overall_confidence,
                    "signals": sentiment.signals_count,
                    "momentum": sentiment.momentum,
                    "last_updated": sentiment.timestamp
                })

        # Sort by signal count (volume)
        trending.sort(key=lambda x: x['signals'], reverse=True)

        return trending[:limit]

    except Exception as e:
        logger.error(f"Failed to get trending symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Example usage documentation
"""
SENTIMENT API ENDPOINTS

1. Health Check
   GET /api/sentiment/health

   Returns status of sentiment analyzer components

2. Get Symbol Sentiment
   GET /api/sentiment/AAPL?hours=24&sources=news,twitter

   Get aggregated sentiment for a symbol

3. News Sentiment
   GET /api/sentiment/AAPL/news?hours=24&limit=10

   Get recent news sentiment

4. Social Sentiment
   GET /api/sentiment/TSLA/social?platform=twitter&limit=20

   Get social media sentiment

5. Trigger Analysis
   POST /api/sentiment/SPY/analyze?hours=24

   Force immediate analysis (bypass cache)

6. Batch Analysis
   GET /api/sentiment/batch/analyze?symbols=AAPL,TSLA,SPY&hours=24

   Analyze multiple symbols

7. Trending Symbols
   GET /api/sentiment/trending/symbols?min_signals=20&limit=10

   Get trending symbols by social/news volume
"""
