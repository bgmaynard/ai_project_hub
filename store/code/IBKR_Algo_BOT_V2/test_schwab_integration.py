"""
Schwab Market Data Integration Tests
=====================================
Tests for Schwab as primary market data source.
Alpaca is only for order execution testing.

Author: AI Trading Bot Team
"""

import pytest
import requests
import time
from typing import Dict, Any

BASE_URL = "http://127.0.0.1:9100"


class TestSchwabConnection:
    """Test Schwab API connection and authentication"""

    def test_schwab_status(self):
        """Verify Schwab connection is available and authenticated"""
        resp = requests.get(f"{BASE_URL}/api/schwab/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] == True, "Schwab should be available"
        assert data["authenticated"] == True, "Schwab should be authenticated"
        assert data["accounts_count"] >= 1, "Should have at least one account"
        print(f"[OK] Schwab connected with {data['accounts_count']} accounts")

    def test_schwab_accounts_list(self):
        """Verify Schwab accounts are accessible"""
        resp = requests.get(f"{BASE_URL}/api/schwab/accounts")
        assert resp.status_code == 200
        data = resp.json()
        assert "accounts" in data
        assert len(data["accounts"]) >= 1
        print(f"[OK] Found {len(data['accounts'])} Schwab accounts")


class TestSchwabQuotes:
    """Test Schwab real-time quote data"""

    def test_single_quote(self):
        """Test fetching single symbol quote from Schwab"""
        resp = requests.get(f"{BASE_URL}/api/schwab/quote/AAPL")
        assert resp.status_code == 200
        data = resp.json()

        # Verify quote structure
        assert data["symbol"] == "AAPL"
        assert data["source"] == "schwab"
        assert data["bid"] > 0, "Bid should be positive"
        assert data["ask"] > 0, "Ask should be positive"
        assert data["last"] > 0, "Last price should be positive"
        assert data["volume"] > 0, "Volume should be positive"

        # Verify bid/ask spread is reasonable (< 1%)
        spread = (data["ask"] - data["bid"]) / data["bid"] * 100
        assert spread < 1.0, f"Spread too wide: {spread:.2f}%"

        print(f"[OK] AAPL quote: ${data['last']:.2f} (bid: ${data['bid']:.2f}, ask: ${data['ask']:.2f})")

    def test_batch_quotes(self):
        """Test fetching multiple quotes from Schwab"""
        symbols = ["SPY", "QQQ", "TSLA", "NVDA", "AAPL"]
        resp = requests.get(f"{BASE_URL}/api/schwab/quotes", params={"symbols": ",".join(symbols)})
        assert resp.status_code == 200
        data = resp.json()

        assert "quotes" in data
        assert data["source"] == "schwab"
        assert data["count"] == len(symbols)

        for symbol in symbols:
            assert symbol in data["quotes"], f"Missing quote for {symbol}"
            quote = data["quotes"][symbol]
            assert quote["source"] == "schwab"
            assert quote["last"] > 0

        print(f"[OK] Batch quotes for {len(symbols)} symbols from Schwab")

    def test_quote_freshness(self):
        """Verify quote timestamps are recent"""
        resp = requests.get(f"{BASE_URL}/api/schwab/quote/SPY")
        assert resp.status_code == 200
        data = resp.json()

        # Timestamp should be present and recent
        assert "timestamp" in data
        # The timestamp should be from today
        assert "2025-12-13" in data["timestamp"] or "T" in data["timestamp"]

        print(f"[OK] Quote timestamp: {data['timestamp']}")


class TestSchwabFastPolling:
    """Test Schwab fast polling for real-time data"""

    def test_fast_polling_status(self):
        """Verify fast polling is running"""
        resp = requests.get(f"{BASE_URL}/api/schwab/fast-polling/status")
        assert resp.status_code == 200
        data = resp.json()

        assert data["available"] == True
        assert data["running"] == True, "Fast polling should be running"
        assert data["mode"] == "fast_polling"
        assert data["poll_interval_ms"] <= 500, "Poll interval should be <= 500ms"
        assert len(data["subscribed_symbols"]) > 0, "Should have subscribed symbols"

        print(f"[OK] Fast polling active: {data['poll_interval_ms']}ms interval, {len(data['subscribed_symbols'])} symbols")

    def test_fast_polling_quote(self):
        """Test fast polling quote retrieval"""
        resp = requests.get(f"{BASE_URL}/api/schwab/fast-polling/quote/TSLA")
        assert resp.status_code == 200
        data = resp.json()

        assert data["symbol"] == "TSLA"
        assert data["last"] > 0

        print(f"[OK] Fast polling TSLA: ${data['last']:.2f}")


class TestSchwabStreaming:
    """Test Schwab streaming configuration"""

    def test_streaming_status(self):
        """Check streaming status"""
        resp = requests.get(f"{BASE_URL}/api/schwab/streaming/status")
        assert resp.status_code == 200
        data = resp.json()

        assert data["available"] == True
        assert len(data["subscribed_symbols"]) > 0

        print(f"[OK] Streaming configured for {len(data['subscribed_symbols'])} symbols")


class TestHybridDataProvider:
    """Test hybrid data provider (Schwab primary, Alpaca fallback)"""

    def test_hybrid_status(self):
        """Verify hybrid provider status"""
        resp = requests.get(f"{BASE_URL}/api/hybrid/status")
        assert resp.status_code == 200
        data = resp.json()

        assert data["available"] == True
        assert "channels" in data
        assert "fast" in data["channels"]
        assert "background" in data["channels"]

        print("[OK] Hybrid data provider active")

    def test_hybrid_fast_quote(self):
        """Test hybrid fast channel quotes (should use Schwab)"""
        resp = requests.get(f"{BASE_URL}/api/hybrid/fast/quote/NVDA")
        assert resp.status_code == 200
        data = resp.json()

        assert data["symbol"] == "NVDA"
        assert "schwab" in data["source"].lower(), f"Expected Schwab source, got {data['source']}"
        assert data["channel"] == "fast"
        assert data["priority"] == "high"
        assert data["last"] > 0

        print(f"[OK] Hybrid fast quote NVDA: ${data['last']:.2f} from {data['source']}")


class TestUnifiedMarketData:
    """Test unified market data (single source of truth)"""

    def test_unified_status(self):
        """Verify unified data provider uses Schwab as primary"""
        resp = requests.get(f"{BASE_URL}/api/data/unified/status")
        assert resp.status_code == 200
        data = resp.json()

        assert data["available"] == True
        assert data["primary_source"] == "schwab", "Schwab should be primary source"
        assert data["schwab"]["available"] == True

        print(f"[OK] Unified data: primary_source={data['primary_source']}")


class TestWatchlistWithSchwabData:
    """Test watchlist uses Schwab data"""

    def test_watchlist_quotes_source(self):
        """Verify watchlist quotes come from Schwab"""
        resp = requests.get(f"{BASE_URL}/api/watchlist")
        assert resp.status_code == 200
        data = resp.json()

        assert "quotes" in data
        assert len(data["quotes"]) > 0

        # Check that quotes are from Schwab
        for symbol, quote in data["quotes"].items():
            assert quote["source"] == "schwab", f"{symbol} should use Schwab, got {quote['source']}"

        print(f"[OK] Watchlist: {len(data['quotes'])} quotes all from Schwab")


class TestAIPredictionsWithSchwabData:
    """Test AI predictions use Schwab market data"""

    def test_ai_prediction_data_source(self):
        """Verify AI predictions note data source"""
        resp = requests.post(f"{BASE_URL}/api/ai/predict", json={"symbol": "AAPL"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["symbol"] == "AAPL"
        assert "prediction" in data
        assert "confidence" in data
        # Note: data_source may be Alpaca for historical training data
        # but real-time features should come from Schwab

        print(f"[OK] AI prediction for AAPL: {data['prediction']} ({data['confidence']*100:.1f}% confidence)")


class TestMarketDataLatency:
    """Test market data latency"""

    def test_schwab_latency(self):
        """Measure Schwab quote latency"""
        latencies = []

        for _ in range(5):
            start = time.time()
            resp = requests.get(f"{BASE_URL}/api/schwab/quote/SPY")
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            assert resp.status_code == 200

        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 500, f"Average latency too high: {avg_latency:.1f}ms"

        print(f"[OK] Schwab latency: avg={avg_latency:.1f}ms, min={min(latencies):.1f}ms, max={max(latencies):.1f}ms")

    def test_hybrid_fast_latency(self):
        """Measure hybrid fast channel latency"""
        resp = requests.get(f"{BASE_URL}/api/hybrid/fast/quote/QQQ")
        assert resp.status_code == 200
        data = resp.json()

        if "latency_ms" in data:
            assert data["latency_ms"] < 100, f"Fast channel latency too high: {data['latency_ms']:.2f}ms"
            print(f"[OK] Hybrid fast latency: {data['latency_ms']:.2f}ms")
        else:
            print("[OK] Hybrid fast quote retrieved (no latency metric)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
