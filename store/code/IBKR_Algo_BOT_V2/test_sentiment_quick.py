"""
Quick Sentiment Analysis Test
Tests basic functionality without requiring API keys
"""

import asyncio
import sys


def test_finbert_import():
    """Test if FinBERT dependencies are available"""
    print("Testing FinBERT dependencies...")
    try:
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)

        print("  [OK] transformers and torch imported successfully")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import failed: {e}")
        return False


def test_sentiment_analyzer_import():
    """Test if sentiment analyzer can be imported"""
    print("\nTesting sentiment analyzer import...")
    try:
        from ai.warrior_sentiment_analyzer import (FinBERTSentimentAnalyzer,
                                                   WarriorSentimentAnalyzer,
                                                   get_sentiment_analyzer)

        print("  [OK] Sentiment analyzer imported successfully")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import failed: {e}")
        return False


def test_finbert_analysis():
    """Test FinBERT sentiment analysis"""
    print("\nTesting FinBERT sentiment analysis...")
    try:
        from ai.warrior_sentiment_analyzer import FinBERTSentimentAnalyzer

        analyzer = FinBERTSentimentAnalyzer()
        print("  [OK] FinBERT analyzer created")

        # Test positive sentiment
        text = (
            "Stock is surging with strong bullish momentum, great buying opportunity!"
        )
        score, confidence = analyzer.analyze(text)
        print(f"  [OK] Positive text: score={score:+.2f}, confidence={confidence:.2f}")

        # Test negative sentiment
        text = "Stock is crashing with bearish breakdown, sell everything now!"
        score, confidence = analyzer.analyze(text)
        print(f"  [OK] Negative text: score={score:+.2f}, confidence={confidence:.2f}")

        return True
    except Exception as e:
        print(f"  [FAIL] Analysis failed: {e}")
        return False


async def test_sentiment_analyzer():
    """Test main sentiment analyzer"""
    print("\nTesting Warrior Sentiment Analyzer...")
    try:
        from ai.warrior_sentiment_analyzer import get_sentiment_analyzer

        analyzer = get_sentiment_analyzer()
        print("  [OK] Analyzer instance created")

        # Note: This will return empty results without API keys
        # but it should still run without errors
        sentiment = await analyzer.analyze_symbol("AAPL", hours=24, sources=[])
        print(f"  [OK] Analysis completed (no data sources configured)")
        print(f"    Symbol: {sentiment.symbol}")
        print(f"    Signals: {sentiment.signals_count}")

        return True
    except Exception as e:
        print(f"  [FAIL] Analyzer test failed: {e}")
        return False


def test_api_router_import():
    """Test if API router can be imported"""
    print("\nTesting API router import...")
    try:
        from ai.warrior_sentiment_router import router

        print("  [OK] API router imported successfully")
        print(f"  [OK] Router has {len(router.routes)} routes")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("=" * 60)
    print("SENTIMENT ANALYSIS QUICK TEST")
    print("=" * 60)

    results = []

    # Test 1: Imports
    results.append(("FinBERT Dependencies", test_finbert_import()))
    results.append(("Sentiment Analyzer Import", test_sentiment_analyzer_import()))

    # Test 2: FinBERT Analysis
    results.append(("FinBERT Analysis", test_finbert_analysis()))

    # Test 3: Main Analyzer
    results.append(("Warrior Sentiment Analyzer", await test_sentiment_analyzer()))

    # Test 4: API Router
    results.append(("API Router Import", test_api_router_import()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status:8s} | {name}")

    print("=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("[OK] All tests passed! Sentiment analysis is ready to use.")
        print("\nNOTE: For full functionality, set these environment variables:")
        print("  - NEWS_API_KEY (newsapi.org)")
        print("  - TWITTER_BEARER_TOKEN (developer.twitter.com)")
        print("  - REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET (reddit.com/prefs/apps)")
        return 0
    else:
        print(f"[FAIL] {total - passed} test(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
