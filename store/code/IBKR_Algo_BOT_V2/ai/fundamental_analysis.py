"""
Fundamental Analysis Module
===========================
Provides fundamental data analysis for stocks.
Uses free data sources (Yahoo Finance, etc.)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum

logger = logging.getLogger(__name__)


class HealthRating(Enum):
    """Financial health rating"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class FundamentalMetrics:
    """Fundamental metrics for a stock"""
    symbol: str
    last_updated: datetime = field(default_factory=datetime.now)

    # Valuation
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None

    # Profitability
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None

    # Growth
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None

    # Financial Health
    current_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    free_cash_flow: Optional[float] = None

    # Dividend
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None

    # Other
    beta: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    avg_volume: Optional[float] = None
    shares_outstanding: Optional[float] = None

    # Ratings
    analyst_rating: Optional[str] = None
    price_target: Optional[float] = None

    # Sector info
    sector: Optional[str] = None
    industry: Optional[str] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['last_updated'] = self.last_updated.isoformat()
        return data


@dataclass
class EarningsEvent:
    """Upcoming earnings event"""
    symbol: str
    date: datetime
    estimated_eps: Optional[float] = None
    actual_eps: Optional[float] = None
    estimated_revenue: Optional[float] = None
    actual_revenue: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "date": self.date.isoformat(),
            "estimated_eps": self.estimated_eps,
            "actual_eps": self.actual_eps,
            "estimated_revenue": self.estimated_revenue,
            "actual_revenue": self.actual_revenue
        }


@dataclass
class EconomicEvent:
    """Economic calendar event"""
    name: str
    date: datetime
    importance: str  # high, medium, low
    previous: Optional[str] = None
    forecast: Optional[str] = None
    actual: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "date": self.date.isoformat(),
            "importance": self.importance,
            "previous": self.previous,
            "forecast": self.forecast,
            "actual": self.actual
        }


class FundamentalAnalyzer:
    """
    Analyze fundamental data for stocks.
    Uses free data sources like Yahoo Finance.
    """

    def __init__(self):
        self.cache: Dict[str, FundamentalMetrics] = {}
        self.cache_duration = timedelta(hours=4)  # Cache for 4 hours

        # Try to import yfinance for real data
        try:
            import yfinance
            self.yf = yfinance
            self.has_yfinance = True
            logger.info("FundamentalAnalyzer using yfinance for real data")
        except ImportError:
            self.yf = None
            self.has_yfinance = False
            logger.warning("yfinance not installed - using mock fundamental data")

    async def get_fundamentals(self, symbol: str, force_refresh: bool = False) -> FundamentalMetrics:
        """Get fundamental metrics for a symbol"""
        symbol = symbol.upper()

        # Check cache
        if not force_refresh and symbol in self.cache:
            cached = self.cache[symbol]
            if datetime.now() - cached.last_updated < self.cache_duration:
                return cached

        # Fetch new data
        if self.has_yfinance:
            metrics = await self._fetch_yfinance(symbol)
        else:
            metrics = self._get_mock_fundamentals(symbol)

        self.cache[symbol] = metrics
        return metrics

    async def _fetch_yfinance(self, symbol: str) -> FundamentalMetrics:
        """Fetch fundamentals from Yahoo Finance"""
        try:
            ticker = self.yf.Ticker(symbol)
            info = ticker.info

            return FundamentalMetrics(
                symbol=symbol,
                last_updated=datetime.now(),

                # Valuation
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                forward_pe=info.get('forwardPE'),
                peg_ratio=info.get('pegRatio'),
                price_to_book=info.get('priceToBook'),
                price_to_sales=info.get('priceToSalesTrailing12Months'),

                # Profitability
                profit_margin=info.get('profitMargins'),
                operating_margin=info.get('operatingMargins'),
                return_on_equity=info.get('returnOnEquity'),
                return_on_assets=info.get('returnOnAssets'),

                # Growth
                revenue_growth=info.get('revenueGrowth'),
                earnings_growth=info.get('earningsGrowth'),

                # Financial Health
                current_ratio=info.get('currentRatio'),
                debt_to_equity=info.get('debtToEquity'),
                free_cash_flow=info.get('freeCashflow'),

                # Dividend
                dividend_yield=info.get('dividendYield'),
                payout_ratio=info.get('payoutRatio'),

                # Other
                beta=info.get('beta'),
                fifty_two_week_high=info.get('fiftyTwoWeekHigh'),
                fifty_two_week_low=info.get('fiftyTwoWeekLow'),
                avg_volume=info.get('averageVolume'),
                shares_outstanding=info.get('sharesOutstanding'),

                # Ratings
                analyst_rating=info.get('recommendationKey'),
                price_target=info.get('targetMeanPrice'),

                # Sector
                sector=info.get('sector'),
                industry=info.get('industry')
            )
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return self._get_mock_fundamentals(symbol)

    def _get_mock_fundamentals(self, symbol: str) -> FundamentalMetrics:
        """Generate mock fundamentals for testing"""
        return FundamentalMetrics(
            symbol=symbol,
            last_updated=datetime.now(),
            market_cap=100_000_000_000,
            pe_ratio=25.5,
            forward_pe=22.3,
            peg_ratio=1.5,
            price_to_book=4.2,
            profit_margin=0.15,
            operating_margin=0.20,
            return_on_equity=0.25,
            revenue_growth=0.12,
            earnings_growth=0.15,
            current_ratio=1.8,
            debt_to_equity=0.5,
            dividend_yield=0.015,
            beta=1.1,
            sector="Technology",
            industry="Software",
            analyst_rating="buy",
            price_target=180.0
        )

    async def get_ai_analysis(self, symbol: str) -> Dict:
        """Get AI-powered fundamental analysis"""
        metrics = await self.get_fundamentals(symbol)

        # Calculate health score
        score = 50  # Base score

        if metrics.pe_ratio:
            if metrics.pe_ratio < 15:
                score += 10
            elif metrics.pe_ratio > 40:
                score -= 10

        if metrics.profit_margin:
            if metrics.profit_margin > 0.2:
                score += 15
            elif metrics.profit_margin < 0:
                score -= 20

        if metrics.return_on_equity:
            if metrics.return_on_equity > 0.2:
                score += 10
            elif metrics.return_on_equity < 0.05:
                score -= 10

        if metrics.debt_to_equity:
            if metrics.debt_to_equity < 0.5:
                score += 10
            elif metrics.debt_to_equity > 2:
                score -= 15

        if metrics.revenue_growth:
            if metrics.revenue_growth > 0.15:
                score += 10
            elif metrics.revenue_growth < 0:
                score -= 10

        # Determine rating
        if score >= 75:
            rating = "STRONG BUY"
            health = HealthRating.EXCELLENT
        elif score >= 60:
            rating = "BUY"
            health = HealthRating.GOOD
        elif score >= 40:
            rating = "HOLD"
            health = HealthRating.FAIR
        elif score >= 25:
            rating = "SELL"
            health = HealthRating.POOR
        else:
            rating = "STRONG SELL"
            health = HealthRating.CRITICAL

        return {
            "symbol": symbol,
            "overall_score": min(100, max(0, score)),
            "rating": rating,
            "health": health.value,
            "key_metrics": {
                "pe_ratio": metrics.pe_ratio,
                "profit_margin": metrics.profit_margin,
                "roe": metrics.return_on_equity,
                "debt_to_equity": metrics.debt_to_equity,
                "revenue_growth": metrics.revenue_growth
            },
            "summary": f"{symbol} scores {score}/100 based on fundamental analysis. "
                       f"Rating: {rating}. Sector: {metrics.sector or 'Unknown'}."
        }

    def get_sector_comparison(self, symbol: str) -> Dict:
        """Compare stock to sector averages"""
        if symbol not in self.cache:
            return {"error": "Symbol not loaded. Call get_fundamentals first."}

        metrics = self.cache[symbol]

        # Mock sector averages (in real impl, would fetch sector data)
        sector_avg = {
            "pe_ratio": 22.0,
            "profit_margin": 0.12,
            "roe": 0.18,
            "debt_to_equity": 0.8,
            "revenue_growth": 0.08
        }

        comparison = {}
        if metrics.pe_ratio:
            diff = ((metrics.pe_ratio - sector_avg["pe_ratio"]) / sector_avg["pe_ratio"]) * 100
            comparison["pe_ratio"] = {
                "value": metrics.pe_ratio,
                "sector_avg": sector_avg["pe_ratio"],
                "diff_pct": round(diff, 1),
                "better": diff < 0  # Lower PE is better
            }

        if metrics.profit_margin:
            diff = ((metrics.profit_margin - sector_avg["profit_margin"]) / sector_avg["profit_margin"]) * 100
            comparison["profit_margin"] = {
                "value": metrics.profit_margin,
                "sector_avg": sector_avg["profit_margin"],
                "diff_pct": round(diff, 1),
                "better": diff > 0
            }

        return {
            "symbol": symbol,
            "sector": metrics.sector or "Unknown",
            "comparison": comparison
        }

    async def get_earnings_calendar(self, symbols: List[str] = None, days_ahead: int = 14) -> List[EarningsEvent]:
        """Get upcoming earnings for symbols"""
        # Mock earnings calendar
        events = []

        if symbols:
            for i, symbol in enumerate(symbols[:10]):
                event_date = datetime.now() + timedelta(days=(i * 2) + 1)
                events.append(EarningsEvent(
                    symbol=symbol.upper(),
                    date=event_date,
                    estimated_eps=2.50 + (i * 0.1)
                ))
        else:
            # Default major stocks
            major = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            for i, symbol in enumerate(major):
                event_date = datetime.now() + timedelta(days=(i * 3) + 1)
                events.append(EarningsEvent(
                    symbol=symbol,
                    date=event_date,
                    estimated_eps=3.00 + (i * 0.5)
                ))

        return events

    def get_economic_calendar(self, days_ahead: int = 7) -> List[EconomicEvent]:
        """Get upcoming economic events"""
        now = datetime.now()

        events = [
            EconomicEvent(
                name="FOMC Meeting Minutes",
                date=now + timedelta(days=2),
                importance="high"
            ),
            EconomicEvent(
                name="CPI (Monthly)",
                date=now + timedelta(days=4),
                importance="high",
                forecast="0.2%",
                previous="0.3%"
            ),
            EconomicEvent(
                name="Unemployment Rate",
                date=now + timedelta(days=5),
                importance="high",
                forecast="3.8%",
                previous="3.7%"
            ),
            EconomicEvent(
                name="Retail Sales",
                date=now + timedelta(days=6),
                importance="medium",
                forecast="0.3%"
            )
        ]

        return [e for e in events if (e.date - now).days <= days_ahead]


# Global instance
_fundamental_analyzer: Optional[FundamentalAnalyzer] = None


def get_fundamental_analyzer() -> FundamentalAnalyzer:
    """Get or create the fundamental analyzer instance"""
    global _fundamental_analyzer
    if _fundamental_analyzer is None:
        _fundamental_analyzer = FundamentalAnalyzer()
    return _fundamental_analyzer


logger.info("Fundamental Analysis module loaded")
