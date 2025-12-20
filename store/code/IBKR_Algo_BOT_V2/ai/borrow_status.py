"""
Borrow Status Tracker
=====================
Tracks ETB (Easy to Borrow) vs HTB (Hard to Borrow) status for stocks.

Uses multiple sources:
1. Finviz short float data
2. yfinance short interest
3. Cache recent lookups

Usage:
    from ai.borrow_status import get_borrow_status
    status = await get_borrow_status("AAPL")
    print(f"Borrow status: {status['status']}")  # ETB, HTB, or UNKNOWN
"""

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class BorrowInfo:
    """Borrow status information"""
    symbol: str
    status: str  # ETB, HTB, UNKNOWN
    short_percent: float  # Short interest as % of float
    short_ratio: float  # Days to cover
    float_shares: float  # Float in millions
    short_shares: float  # Short shares in millions
    borrow_fee: float  # Estimated annual borrow fee %
    confidence: float  # 0-1 confidence in the data
    source: str
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'status': self.status,
            'short_percent': round(self.short_percent, 2),
            'short_ratio': round(self.short_ratio, 2),
            'float_shares': round(self.float_shares, 2),
            'short_shares': round(self.short_shares, 2),
            'borrow_fee': round(self.borrow_fee, 2),
            'confidence': round(self.confidence, 2),
            'source': self.source,
            'timestamp': self.timestamp
        }


class BorrowStatusTracker:
    """
    Tracks borrow status using short interest data.

    Classification:
    - ETB (Easy to Borrow): Short % < 10%, Days to Cover < 3
    - HTB (Hard to Borrow): Short % > 20% OR Days to Cover > 5
    - CAUTION: In between

    High short interest stocks:
    - More volatile (short squeeze potential)
    - May have borrow fees
    - Can be hard to exit positions if halted
    """

    def __init__(self):
        # Thresholds for classification
        self.etb_short_percent = 10.0  # Below = ETB
        self.htb_short_percent = 20.0  # Above = HTB
        self.etb_days_to_cover = 3.0   # Below = ETB
        self.htb_days_to_cover = 5.0   # Above = HTB

        # Cache
        self._cache: Dict[str, BorrowInfo] = {}
        self._cache_ttl = 3600  # 1 hour cache

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid"""
        if symbol not in self._cache:
            return False
        cached = self._cache[symbol]
        cached_time = datetime.fromisoformat(cached.timestamp)
        return (datetime.now() - cached_time).total_seconds() < self._cache_ttl

    def _classify_status(self, short_pct: float, days_to_cover: float) -> str:
        """Classify borrow status based on short interest"""
        if short_pct <= 0:
            return "UNKNOWN"

        if short_pct >= self.htb_short_percent or days_to_cover >= self.htb_days_to_cover:
            return "HTB"
        elif short_pct <= self.etb_short_percent and days_to_cover <= self.etb_days_to_cover:
            return "ETB"
        else:
            return "CAUTION"

    def _estimate_borrow_fee(self, short_pct: float, status: str) -> float:
        """Estimate annual borrow fee based on short interest"""
        if status == "ETB":
            return 0.5  # Low fee for ETB stocks
        elif status == "CAUTION":
            return 2.0 + (short_pct / 10)  # 2-5% for moderate short
        else:  # HTB
            return 10.0 + (short_pct / 5)  # 10%+ for HTB stocks

    async def get_status(self, symbol: str) -> BorrowInfo:
        """
        Get borrow status for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            BorrowInfo with status and metrics
        """
        symbol = symbol.upper()

        # Check cache
        if self._is_cache_valid(symbol):
            return self._cache[symbol]

        # Try to get data from yfinance
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get short interest data
            shares_short = info.get('sharesShort', 0) or 0
            float_shares = info.get('floatShares', 0) or 0
            short_ratio = info.get('shortRatio', 0) or 0  # Days to cover
            shares_outstanding = info.get('sharesOutstanding', 0) or 0

            # Calculate short percent
            if float_shares > 0:
                short_percent = (shares_short / float_shares) * 100
            elif shares_outstanding > 0:
                short_percent = (shares_short / shares_outstanding) * 100
            else:
                short_percent = 0

            # Classify status
            status = self._classify_status(short_percent, short_ratio)
            borrow_fee = self._estimate_borrow_fee(short_percent, status)

            result = BorrowInfo(
                symbol=symbol,
                status=status,
                short_percent=short_percent,
                short_ratio=short_ratio,
                float_shares=float_shares / 1_000_000 if float_shares else 0,
                short_shares=shares_short / 1_000_000 if shares_short else 0,
                borrow_fee=borrow_fee,
                confidence=0.8 if shares_short > 0 else 0.3,
                source="yfinance",
                timestamp=datetime.now().isoformat()
            )

            # Cache result
            self._cache[symbol] = result
            return result

        except Exception as e:
            logger.warning(f"Failed to get borrow status for {symbol}: {e}")
            return BorrowInfo(
                symbol=symbol,
                status="UNKNOWN",
                short_percent=0,
                short_ratio=0,
                float_shares=0,
                short_shares=0,
                borrow_fee=0,
                confidence=0,
                source="error",
                timestamp=datetime.now().isoformat()
            )

    async def get_batch_status(self, symbols: list) -> Dict[str, BorrowInfo]:
        """Get borrow status for multiple symbols"""
        results = {}
        for symbol in symbols:
            results[symbol] = await self.get_status(symbol)
        return results

    def get_cached_symbols(self) -> list:
        """Get list of cached symbols"""
        return list(self._cache.keys())

    def get_htb_symbols(self) -> list:
        """Get list of HTB symbols from cache"""
        return [s for s, info in self._cache.items() if info.status == "HTB"]


# Singleton instance
_tracker = None


def get_borrow_tracker() -> BorrowStatusTracker:
    """Get singleton borrow status tracker"""
    global _tracker
    if _tracker is None:
        _tracker = BorrowStatusTracker()
    return _tracker


async def get_borrow_status(symbol: str) -> Dict:
    """
    Get borrow status for a symbol.

    Returns dict with:
    - status: ETB, HTB, CAUTION, or UNKNOWN
    - short_percent: Short interest as % of float
    - short_ratio: Days to cover
    """
    tracker = get_borrow_tracker()
    info = await tracker.get_status(symbol)
    return info.to_dict()


if __name__ == "__main__":
    import asyncio

    async def test():
        test_symbols = ["AAPL", "GME", "AMC", "TSLA", "SPY", "SOUN"]

        print("\nBorrow Status Test")
        print("=" * 70)

        for symbol in test_symbols:
            status = await get_borrow_status(symbol)
            print(f"\n{symbol}:")
            print(f"  Status: {status['status']}")
            print(f"  Short %: {status['short_percent']:.1f}%")
            print(f"  Days to Cover: {status['short_ratio']:.1f}")
            print(f"  Float: {status['float_shares']:.1f}M shares")
            print(f"  Short Shares: {status['short_shares']:.2f}M")
            print(f"  Est. Borrow Fee: {status['borrow_fee']:.1f}%")

    asyncio.run(test())
