"""
Top Gappers Scanner
===================
Finds the leading percentage gainers based on Ross Cameron's criteria.
"I typically focus on the top 2-3 leading percentage gainers each day,
as these will be the most obvious." - Ross Cameron

This scanner runs on market data to find stocks meeting Ross's 5 criteria:
1. 5x Relative Volume (vs 30-day average)
2. Already up 10% on the day
3. News Event catalyst
4. Price $1.00 - $20.00
5. Float < 10 million shares

Sources:
- Schwab movers API
- FinViz screener
- Yahoo Finance gainers
"""

import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import httpx

logger = logging.getLogger(__name__)


@dataclass
class GapperStock:
    """A stock from the gapper scan"""
    symbol: str
    price: float
    change_pct: float
    volume: int
    avg_volume: int = 0
    relative_volume: float = 0.0
    float_shares: float = 0.0
    gap_pct: float = 0.0
    has_news: bool = False
    news_headline: str = ""

    # Ross Cameron grading
    grade: str = "C"
    criteria_met: int = 0
    criteria_details: Dict = field(default_factory=dict)

    # Timestamps
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "change_pct": self.change_pct,
            "volume": self.volume,
            "avg_volume": self.avg_volume,
            "relative_volume": self.relative_volume,
            "float_shares": self.float_shares,
            "gap_pct": self.gap_pct,
            "has_news": self.has_news,
            "news_headline": self.news_headline,
            "grade": self.grade,
            "criteria_met": self.criteria_met,
            "criteria_details": self.criteria_details,
            "detected_at": self.detected_at.isoformat()
        }


class TopGappersScanner:
    """
    Scanner for top percentage gainers.
    Combines multiple data sources and grades by Ross Cameron's criteria.
    """

    def __init__(self):
        self.gappers: List[GapperStock] = []
        self.last_scan: Optional[datetime] = None
        self.is_running: bool = False

        # Ross Cameron's criteria thresholds
        self.min_price: float = 1.0
        self.max_price: float = 20.0
        self.min_change_pct: float = 10.0
        self.min_relative_volume: float = 5.0
        self.max_float: float = 10_000_000

        # Scan settings
        self.max_results: int = 20
        self.scan_interval: int = 60  # seconds

        logger.info("Top Gappers Scanner initialized")

    def _grade_stock(self, stock: GapperStock) -> Tuple[str, int, Dict]:
        """
        Grade stock based on Ross Cameron's 5 criteria.

        Returns (grade, criteria_met, details)
        """
        criteria_met = 0
        details = {}

        # 1. Relative Volume >= 5x
        if stock.relative_volume >= 5.0:
            criteria_met += 1
            details['rvol'] = f"✓ RVol {stock.relative_volume:.1f}x"
        else:
            details['rvol'] = f"✗ RVol {stock.relative_volume:.1f}x < 5x"

        # 2. Already up 10%
        if stock.change_pct >= 10.0:
            criteria_met += 1
            details['change'] = f"✓ Up {stock.change_pct:.1f}%"
        else:
            details['change'] = f"✗ Up {stock.change_pct:.1f}% < 10%"

        # 3. News catalyst
        if stock.has_news:
            criteria_met += 1
            details['news'] = "✓ Has news"
        else:
            details['news'] = "✗ No news"

        # 4. Price $1-$20
        if self.min_price <= stock.price <= self.max_price:
            criteria_met += 1
            details['price'] = f"✓ ${stock.price:.2f}"
        else:
            details['price'] = f"✗ ${stock.price:.2f}"

        # 5. Float < 10M
        if stock.float_shares > 0 and stock.float_shares < self.max_float:
            criteria_met += 1
            float_m = stock.float_shares / 1_000_000
            details['float'] = f"✓ {float_m:.1f}M"
        elif stock.float_shares == 0:
            details['float'] = "? Unknown"
        else:
            float_m = stock.float_shares / 1_000_000
            details['float'] = f"✗ {float_m:.1f}M >= 10M"

        # Grade
        if criteria_met >= 5:
            grade = "A"
        elif criteria_met == 4:
            grade = "B"
        else:
            grade = "C"

        return grade, criteria_met, details

    async def scan_schwab_movers(self) -> List[GapperStock]:
        """Scan Schwab market movers API"""
        stocks = []
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "http://localhost:9100/api/market/movers/scalp",
                    timeout=10.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get('movers', []):
                        stock = GapperStock(
                            symbol=item.get('symbol', ''),
                            price=float(item.get('price', 0)),
                            change_pct=float(item.get('change_pct', 0)),
                            volume=int(item.get('volume', 0)),
                            avg_volume=int(item.get('avg_volume', 0)),
                            gap_pct=float(item.get('gap_pct', 0))
                        )
                        if stock.avg_volume > 0:
                            stock.relative_volume = stock.volume / stock.avg_volume
                        stocks.append(stock)
                    logger.debug(f"Schwab movers: {len(stocks)} found")
        except Exception as e:
            logger.error(f"Schwab movers scan error: {e}")
        return stocks

    async def scan_finviz(self) -> List[GapperStock]:
        """Scan FinViz top gainers"""
        stocks = []
        try:
            from .finviz_momentum_scanner import get_finviz_scanner
            scanner = get_finviz_scanner()
            movers = scanner.scan_movers(limit=20)

            for m in movers:
                stock = GapperStock(
                    symbol=m.get('symbol', ''),
                    price=float(m.get('price', 0)),
                    change_pct=float(m.get('change', 0)),
                    volume=int(m.get('volume', 0)),
                    avg_volume=int(m.get('avg_volume', 0)),
                    float_shares=float(m.get('float', 0))
                )
                if stock.avg_volume > 0:
                    stock.relative_volume = stock.volume / stock.avg_volume
                stocks.append(stock)
            logger.debug(f"FinViz: {len(stocks)} found")
        except Exception as e:
            logger.debug(f"FinViz scan skipped: {e}")
        return stocks

    async def scan_yahoo_gainers(self) -> List[GapperStock]:
        """Scan Yahoo Finance gainers (works 24/7)"""
        stocks = []
        try:
            import yfinance as yf

            # Get day gainers
            gainers = yf.Screener().get_screeners(['day_gainers'], count=25)

            for item in gainers.get('day_gainers', {}).get('quotes', []):
                symbol = item.get('symbol', '')
                if not symbol or '.' in symbol:  # Skip non-US symbols
                    continue

                price = float(item.get('regularMarketPrice', 0))
                change_pct = float(item.get('regularMarketChangePercent', 0))
                volume = int(item.get('regularMarketVolume', 0))
                avg_volume = int(item.get('averageDailyVolume10Day', 0))

                stock = GapperStock(
                    symbol=symbol,
                    price=price,
                    change_pct=change_pct,
                    volume=volume,
                    avg_volume=avg_volume
                )

                if avg_volume > 0:
                    stock.relative_volume = volume / avg_volume

                stocks.append(stock)

            logger.debug(f"Yahoo gainers: {len(stocks)} found")
        except Exception as e:
            logger.error(f"Yahoo gainers scan error: {e}")
        return stocks

    async def enrich_with_float(self, stocks: List[GapperStock]) -> List[GapperStock]:
        """Add float data to stocks"""
        async with httpx.AsyncClient() as client:
            for stock in stocks:
                if stock.float_shares == 0:
                    try:
                        resp = await client.get(
                            f"http://localhost:9100/api/stock/float/{stock.symbol}",
                            timeout=3.0
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            stock.float_shares = float(data.get('float', 0))
                    except:
                        pass
        return stocks

    async def enrich_with_news(self, stocks: List[GapperStock]) -> List[GapperStock]:
        """Check for news catalyst"""
        async with httpx.AsyncClient() as client:
            for stock in stocks:
                try:
                    resp = await client.get(
                        f"http://localhost:9100/api/stock/news-check/{stock.symbol}",
                        timeout=3.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        stock.has_news = data.get('has_news', False)
                        stock.news_headline = data.get('headline', '')
                except:
                    pass
        return stocks

    async def scan(self) -> List[GapperStock]:
        """
        Run full scan combining all sources.
        Returns graded stocks sorted by change %.
        """
        logger.info("Running Top Gappers scan...")

        # Collect from all sources
        all_stocks: Dict[str, GapperStock] = {}

        # Schwab movers
        schwab = await self.scan_schwab_movers()
        for s in schwab:
            if s.symbol:
                all_stocks[s.symbol] = s

        # FinViz
        finviz = await self.scan_finviz()
        for s in finviz:
            if s.symbol and s.symbol not in all_stocks:
                all_stocks[s.symbol] = s

        # Yahoo gainers (fallback, works 24/7)
        yahoo = await self.scan_yahoo_gainers()
        for s in yahoo:
            if s.symbol and s.symbol not in all_stocks:
                all_stocks[s.symbol] = s

        stocks = list(all_stocks.values())
        logger.info(f"Combined {len(stocks)} unique stocks from all sources")

        # Enrich with float and news
        stocks = await self.enrich_with_float(stocks)
        stocks = await self.enrich_with_news(stocks)

        # Grade each stock
        for stock in stocks:
            grade, criteria_met, details = self._grade_stock(stock)
            stock.grade = grade
            stock.criteria_met = criteria_met
            stock.criteria_details = details

        # Sort by change % descending
        stocks.sort(key=lambda x: x.change_pct, reverse=True)

        # Apply minimum filters
        filtered = [
            s for s in stocks
            if s.price >= 0.50 and s.change_pct >= 5.0
        ]

        self.gappers = filtered[:self.max_results]
        self.last_scan = datetime.now()

        # Log results
        a_count = sum(1 for s in self.gappers if s.grade == "A")
        b_count = sum(1 for s in self.gappers if s.grade == "B")
        logger.info(f"Top Gappers: {len(self.gappers)} stocks, {a_count} A-grade, {b_count} B-grade")

        return self.gappers

    def get_a_grade(self) -> List[GapperStock]:
        """Get A-grade stocks (5/5 criteria) - FULL POSITION"""
        return [s for s in self.gappers if s.grade == "A"]

    def get_b_grade(self) -> List[GapperStock]:
        """Get B-grade stocks (4/5 criteria) - HALF POSITION"""
        return [s for s in self.gappers if s.grade == "B"]

    def get_top_3(self) -> List[GapperStock]:
        """Get top 3 leading percentage gainers"""
        return self.gappers[:3]

    def get_status(self) -> dict:
        """Get scanner status"""
        return {
            "is_running": self.is_running,
            "last_scan": self.last_scan.isoformat() if self.last_scan else None,
            "gappers_count": len(self.gappers),
            "a_grade_count": len(self.get_a_grade()),
            "b_grade_count": len(self.get_b_grade()),
            "criteria": {
                "min_price": self.min_price,
                "max_price": self.max_price,
                "min_change_pct": self.min_change_pct,
                "min_relative_volume": self.min_relative_volume,
                "max_float": self.max_float
            }
        }

    def to_dict(self) -> dict:
        """Export gappers to dict"""
        return {
            "gappers": [g.to_dict() for g in self.gappers],
            "a_grade": [g.to_dict() for g in self.get_a_grade()],
            "b_grade": [g.to_dict() for g in self.get_b_grade()],
            "top_3": [g.to_dict() for g in self.get_top_3()],
            "scan_time": self.last_scan.isoformat() if self.last_scan else None
        }


# Singleton instance
_gappers_scanner: Optional[TopGappersScanner] = None


def get_gappers_scanner() -> TopGappersScanner:
    """Get or create Top Gappers scanner instance"""
    global _gappers_scanner
    if _gappers_scanner is None:
        _gappers_scanner = TopGappersScanner()
    return _gappers_scanner


async def run_gappers_scan() -> List[GapperStock]:
    """Run gappers scan (convenience function)"""
    scanner = get_gappers_scanner()
    return await scanner.scan()
