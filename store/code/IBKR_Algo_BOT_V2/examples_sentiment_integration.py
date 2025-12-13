"""
Sentiment Analysis Integration Examples
How to use sentiment analysis with Warrior Trading strategies

Examples:
1. Scanner with sentiment filter
2. Trade validation with sentiment
3. Position sizing based on sentiment confidence
4. Exit timing with sentiment momentum
5. Finding trending stocks pre-market
"""

import asyncio
from typing import List, Dict, Optional
from datetime import datetime

# Import sentiment analyzer
from ai.warrior_sentiment_analyzer import get_sentiment_analyzer, AggregatedSentiment


# ═══════════════════════════════════════════════════════════════════════
#                     EXAMPLE 1: SENTIMENT-FILTERED SCANNER
# ═══════════════════════════════════════════════════════════════════════

async def gap_and_go_with_sentiment(
    min_gap_percent: float = 5.0,
    min_sentiment_score: float = 0.3,
    min_signals: int = 10
) -> List[Dict]:
    """
    Enhanced Gap & Go scanner with sentiment filter

    Only returns stocks with:
    - Gap up >= min_gap_percent
    - Positive sentiment >= min_sentiment_score
    - Sufficient sentiment signals >= min_signals

    Args:
        min_gap_percent: Minimum gap percentage (default 5%)
        min_sentiment_score: Minimum sentiment score -1 to 1 (default 0.3)
        min_signals: Minimum sentiment signals required (default 10)

    Returns:
        List of stocks with gap and sentiment data
    """
    analyzer = get_sentiment_analyzer()

    # Example gap stocks (in real system, get from scanner)
    gap_stocks = [
        {"symbol": "AAPL", "gap_percent": 6.5, "price": 185.50},
        {"symbol": "TSLA", "gap_percent": 8.2, "price": 242.30},
        {"symbol": "NVDA", "gap_percent": 5.5, "price": 495.80},
    ]

    results = []

    for stock in gap_stocks:
        symbol = stock["symbol"]

        # Get sentiment
        sentiment = await analyzer.analyze_symbol(symbol, hours=12)

        # Filter criteria
        if (sentiment.overall_score >= min_sentiment_score and
            sentiment.signals_count >= min_signals):

            results.append({
                "symbol": symbol,
                "gap_percent": stock["gap_percent"],
                "price": stock["price"],
                "sentiment_score": sentiment.overall_score,
                "sentiment_confidence": sentiment.overall_confidence,
                "signals": sentiment.signals_count,
                "trending": sentiment.trending,
                "momentum": sentiment.momentum,
                "quality": "HIGH" if sentiment.overall_confidence > 0.7 else "MEDIUM"
            })

    # Sort by sentiment score (most bullish first)
    results.sort(key=lambda x: x["sentiment_score"], reverse=True)

    return results


# ═══════════════════════════════════════════════════════════════════════
#                     EXAMPLE 2: TRADE VALIDATION
# ═══════════════════════════════════════════════════════════════════════

async def validate_trade_with_sentiment(
    symbol: str,
    direction: str,  # 'long' or 'short'
    pattern: str  # e.g., 'bull_flag', 'breakout'
) -> Dict:
    """
    Validate a potential trade with sentiment analysis

    Checks if sentiment supports the trade direction

    Args:
        symbol: Stock symbol
        direction: 'long' or 'short'
        pattern: Chart pattern detected

    Returns:
        Validation result with recommendation
    """
    analyzer = get_sentiment_analyzer()
    sentiment = await analyzer.analyze_symbol(symbol, hours=24)

    # Sentiment thresholds
    STRONG_BULLISH = 0.5
    STRONG_BEARISH = -0.5
    MIN_CONFIDENCE = 0.6

    # Validation logic
    if direction == 'long':
        sentiment_aligned = sentiment.overall_score > 0
        strong_alignment = sentiment.overall_score > STRONG_BULLISH

        if strong_alignment and sentiment.overall_confidence > MIN_CONFIDENCE:
            recommendation = "STRONG BUY"
            reason = f"Bullish pattern + strong positive sentiment ({sentiment.overall_score:+.2f})"
        elif sentiment_aligned:
            recommendation = "BUY"
            reason = f"Bullish pattern + positive sentiment ({sentiment.overall_score:+.2f})"
        elif sentiment.overall_score < STRONG_BEARISH:
            recommendation = "AVOID"
            reason = f"Bullish pattern but strong negative sentiment ({sentiment.overall_score:+.2f})"
        else:
            recommendation = "CAUTION"
            reason = "Bullish pattern but sentiment not supportive"

    else:  # short
        sentiment_aligned = sentiment.overall_score < 0
        strong_alignment = sentiment.overall_score < STRONG_BEARISH

        if strong_alignment and sentiment.overall_confidence > MIN_CONFIDENCE:
            recommendation = "STRONG SELL"
            reason = f"Bearish pattern + strong negative sentiment ({sentiment.overall_score:+.2f})"
        elif sentiment_aligned:
            recommendation = "SELL"
            reason = f"Bearish pattern + negative sentiment ({sentiment.overall_score:+.2f})"
        elif sentiment.overall_score > STRONG_BULLISH:
            recommendation = "AVOID"
            reason = f"Bearish pattern but strong positive sentiment ({sentiment.overall_score:+.2f})"
        else:
            recommendation = "CAUTION"
            reason = "Bearish pattern but sentiment not supportive"

    return {
        "symbol": symbol,
        "pattern": pattern,
        "direction": direction,
        "recommendation": recommendation,
        "reason": reason,
        "sentiment_score": sentiment.overall_score,
        "sentiment_confidence": sentiment.overall_confidence,
        "signals": sentiment.signals_count,
        "trending": sentiment.trending
    }


# ═══════════════════════════════════════════════════════════════════════
#                     EXAMPLE 3: DYNAMIC POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════

async def calculate_position_size_with_sentiment(
    symbol: str,
    account_size: float,
    base_risk_percent: float = 3.0
) -> Dict:
    """
    Calculate position size adjusted for sentiment confidence

    Higher sentiment confidence = larger position size
    Lower sentiment confidence = smaller position size

    Args:
        symbol: Stock symbol
        account_size: Total account value
        base_risk_percent: Base risk percentage (default 3%)

    Returns:
        Position sizing recommendations
    """
    analyzer = get_sentiment_analyzer()
    sentiment = await analyzer.analyze_symbol(symbol, hours=24)

    # Sentiment confidence multipliers
    if sentiment.overall_confidence >= 0.8:
        multiplier = 1.2  # +20% size
        confidence_level = "VERY HIGH"
    elif sentiment.overall_confidence >= 0.6:
        multiplier = 1.0  # Normal size
        confidence_level = "HIGH"
    elif sentiment.overall_confidence >= 0.4:
        multiplier = 0.8  # -20% size
        confidence_level = "MEDIUM"
    else:
        multiplier = 0.5  # -50% size
        confidence_level = "LOW"

    # Calculate adjusted risk
    adjusted_risk_percent = base_risk_percent * multiplier
    risk_amount = account_size * (adjusted_risk_percent / 100)

    return {
        "symbol": symbol,
        "account_size": account_size,
        "base_risk_percent": base_risk_percent,
        "sentiment_confidence": sentiment.overall_confidence,
        "confidence_level": confidence_level,
        "multiplier": multiplier,
        "adjusted_risk_percent": adjusted_risk_percent,
        "risk_amount": risk_amount,
        "recommendation": f"Risk {adjusted_risk_percent:.1f}% (${risk_amount:,.2f}) based on {confidence_level} sentiment confidence"
    }


# ═══════════════════════════════════════════════════════════════════════
#                     EXAMPLE 4: EXIT TIMING WITH MOMENTUM
# ═══════════════════════════════════════════════════════════════════════

async def check_exit_signal(
    symbol: str,
    entry_price: float,
    current_price: float,
    position_direction: str  # 'long' or 'short'
) -> Dict:
    """
    Check if sentiment momentum suggests taking profits or holding

    Positive momentum = hold
    Negative momentum = consider exit

    Args:
        symbol: Stock symbol
        entry_price: Entry price
        current_price: Current price
        position_direction: 'long' or 'short'

    Returns:
        Exit recommendation
    """
    analyzer = get_sentiment_analyzer()
    sentiment = await analyzer.analyze_symbol(symbol, hours=24)

    # Calculate P&L
    if position_direction == 'long':
        pnl_percent = ((current_price - entry_price) / entry_price) * 100
    else:
        pnl_percent = ((entry_price - current_price) / entry_price) * 100

    # Momentum thresholds
    STRONG_POSITIVE_MOMENTUM = 0.2
    STRONG_NEGATIVE_MOMENTUM = -0.2

    # Exit logic
    if position_direction == 'long':
        if sentiment.momentum < STRONG_NEGATIVE_MOMENTUM:
            recommendation = "EXIT NOW"
            reason = "Sentiment turning negative - take profits"
        elif sentiment.momentum < 0 and pnl_percent > 5:
            recommendation = "CONSIDER PARTIAL EXIT"
            reason = "Weakening sentiment - secure some profits"
        elif sentiment.momentum > STRONG_POSITIVE_MOMENTUM:
            recommendation = "HOLD"
            reason = "Sentiment strengthening - let it run"
        else:
            recommendation = "HOLD WITH TRAILING STOP"
            reason = "Neutral momentum - protect gains"

    else:  # short
        if sentiment.momentum > STRONG_POSITIVE_MOMENTUM:
            recommendation = "EXIT NOW"
            reason = "Sentiment turning positive - cover short"
        elif sentiment.momentum > 0 and pnl_percent > 5:
            recommendation = "CONSIDER PARTIAL EXIT"
            reason = "Strengthening sentiment - secure some profits"
        elif sentiment.momentum < STRONG_NEGATIVE_MOMENTUM:
            recommendation = "HOLD"
            reason = "Sentiment weakening - let it run"
        else:
            recommendation = "HOLD WITH TRAILING STOP"
            reason = "Neutral momentum - protect gains"

    return {
        "symbol": symbol,
        "entry_price": entry_price,
        "current_price": current_price,
        "pnl_percent": pnl_percent,
        "sentiment_momentum": sentiment.momentum,
        "sentiment_score": sentiment.overall_score,
        "recommendation": recommendation,
        "reason": reason
    }


# ═══════════════════════════════════════════════════════════════════════
#                     EXAMPLE 5: FIND TRENDING STOCKS
# ═══════════════════════════════════════════════════════════════════════

async def find_trending_stocks_premarket(
    watchlist: List[str],
    min_momentum: float = 0.1,
    top_n: int = 5
) -> List[Dict]:
    """
    Find stocks with strong positive sentiment momentum
    Perfect for pre-market scanning

    Args:
        watchlist: List of symbols to check
        min_momentum: Minimum momentum threshold (default 0.1)
        top_n: Number of top stocks to return

    Returns:
        Top trending stocks by sentiment
    """
    analyzer = get_sentiment_analyzer()

    trending = []

    for symbol in watchlist:
        sentiment = await analyzer.analyze_symbol(symbol, hours=12)

        if sentiment.trending and sentiment.momentum >= min_momentum:
            trending.append({
                "symbol": symbol,
                "sentiment_score": sentiment.overall_score,
                "momentum": sentiment.momentum,
                "signals": sentiment.signals_count,
                "confidence": sentiment.overall_confidence,
                "reason": f"Trending with {sentiment.signals_count} signals and +{sentiment.momentum:.2f} momentum"
            })

    # Sort by momentum (fastest rising sentiment)
    trending.sort(key=lambda x: x["momentum"], reverse=True)

    return trending[:top_n]


# ═══════════════════════════════════════════════════════════════════════
#                     EXAMPLE 6: BATCH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

async def analyze_portfolio_sentiment(
    positions: List[Dict]
) -> Dict:
    """
    Analyze sentiment for entire portfolio

    Helps identify which positions might need attention

    Args:
        positions: List of position dicts with 'symbol' and 'direction'

    Returns:
        Portfolio sentiment analysis
    """
    analyzer = get_sentiment_analyzer()

    results = []
    total_bullish = 0
    total_bearish = 0

    for position in positions:
        symbol = position["symbol"]
        direction = position["direction"]

        sentiment = await analyzer.analyze_symbol(symbol, hours=24)

        # Check alignment
        if direction == "long":
            aligned = sentiment.overall_score > 0
            alignment_strength = sentiment.overall_score
        else:
            aligned = sentiment.overall_score < 0
            alignment_strength = -sentiment.overall_score

        if aligned:
            if direction == "long":
                total_bullish += 1
            else:
                total_bearish += 1

        results.append({
            "symbol": symbol,
            "direction": direction,
            "sentiment": sentiment.overall_score,
            "aligned": aligned,
            "strength": alignment_strength,
            "momentum": sentiment.momentum,
            "status": "GOOD" if aligned else "WATCH"
        })

    return {
        "total_positions": len(positions),
        "aligned_positions": total_bullish + total_bearish,
        "alignment_rate": (total_bullish + total_bearish) / len(positions) if positions else 0,
        "positions": results,
        "summary": f"{total_bullish + total_bearish}/{len(positions)} positions sentiment-aligned"
    }


# ═══════════════════════════════════════════════════════════════════════
#                     MAIN DEMO
# ═══════════════════════════════════════════════════════════════════════

async def main():
    """Run all examples"""

    print("=" * 60)
    print("SENTIMENT ANALYSIS INTEGRATION EXAMPLES")
    print("=" * 60)

    # Example 1: Scanner with sentiment
    print("\n1. GAP & GO SCANNER WITH SENTIMENT FILTER")
    print("-" * 60)
    results = await gap_and_go_with_sentiment(min_gap_percent=5.0, min_sentiment_score=0.3)
    for stock in results:
        print(f"{stock['symbol']:6s} | Gap: {stock['gap_percent']:5.1f}% | "
              f"Sentiment: {stock['sentiment_score']:+.2f} ({stock['quality']}) | "
              f"Signals: {stock['signals']}")

    # Example 2: Trade validation
    print("\n2. TRADE VALIDATION WITH SENTIMENT")
    print("-" * 60)
    validation = await validate_trade_with_sentiment("AAPL", "long", "bull_flag")
    print(f"Symbol: {validation['symbol']}")
    print(f"Pattern: {validation['pattern']}")
    print(f"Direction: {validation['direction']}")
    print(f"Recommendation: {validation['recommendation']}")
    print(f"Reason: {validation['reason']}")

    # Example 3: Position sizing
    print("\n3. DYNAMIC POSITION SIZING")
    print("-" * 60)
    sizing = await calculate_position_size_with_sentiment("TSLA", account_size=50000)
    print(f"Symbol: {sizing['symbol']}")
    print(f"Account: ${sizing['account_size']:,.2f}")
    print(f"Sentiment Confidence: {sizing['sentiment_confidence']:.1%} ({sizing['confidence_level']})")
    print(f"Recommendation: {sizing['recommendation']}")

    # Example 4: Exit timing
    print("\n4. EXIT TIMING WITH MOMENTUM")
    print("-" * 60)
    exit_check = await check_exit_signal("NVDA", entry_price=480.0, current_price=495.8, position_direction="long")
    print(f"Symbol: {exit_check['symbol']}")
    print(f"P&L: {exit_check['pnl_percent']:+.1f}%")
    print(f"Sentiment Momentum: {exit_check['sentiment_momentum']:+.2f}")
    print(f"Recommendation: {exit_check['recommendation']}")
    print(f"Reason: {exit_check['reason']}")

    # Example 5: Find trending
    print("\n5. FIND TRENDING STOCKS PRE-MARKET")
    print("-" * 60)
    watchlist = ["AAPL", "TSLA", "NVDA", "AMD", "SPY"]
    trending = await find_trending_stocks_premarket(watchlist, min_momentum=0.1, top_n=3)
    for i, stock in enumerate(trending, 1):
        print(f"{i}. {stock['symbol']:6s} | Score: {stock['sentiment_score']:+.2f} | "
              f"Momentum: +{stock['momentum']:.2f} | Signals: {stock['signals']}")

    # Example 6: Portfolio analysis
    print("\n6. PORTFOLIO SENTIMENT ANALYSIS")
    print("-" * 60)
    portfolio = [
        {"symbol": "AAPL", "direction": "long"},
        {"symbol": "TSLA", "direction": "long"},
        {"symbol": "SPY", "direction": "short"}
    ]
    analysis = await analyze_portfolio_sentiment(portfolio)
    print(f"Total Positions: {analysis['total_positions']}")
    print(f"Alignment Rate: {analysis['alignment_rate']:.1%}")
    print(f"Summary: {analysis['summary']}")
    print("\nPosition Details:")
    for pos in analysis['positions']:
        print(f"  {pos['symbol']:6s} {pos['direction']:5s} | "
              f"Sentiment: {pos['sentiment']:+.2f} | "
              f"Status: {pos['status']}")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
