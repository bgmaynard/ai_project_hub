"""
FinViz Elite Momentum Scanner
=============================
Discovers small cap momentum plays using FinViz Elite API.
Integrates with news detection and scalper watchlist.

Pipeline:
1. Scan FinViz Elite for movers/gappers
2. Fetch news for each mover
3. Score by momentum + news catalyst
4. Auto-add top picks to scalper watchlist
"""

import os
import asyncio
import logging
import csv
import io
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import aiohttp

logger = logging.getLogger(__name__)

# FinViz Elite token
FINVIZ_ELITE_TOKEN = os.getenv("FINVIZ_ELITE_TOKEN", "")


@dataclass
class MomentumCandidate:
    """Stock candidate with momentum and news data"""
    symbol: str
    company: str
    price: float
    change_pct: float
    gap_pct: float
    volume: int
    float_shares: float
    short_float: float
    sector: str
    industry: str
    market_cap: float

    # News/catalyst data
    has_news: bool = False
    news_count: int = 0
    news_headlines: List[str] = None
    catalyst_type: str = ""  # FDA, earnings, contract, etc.

    # Scoring
    momentum_score: float = 0.0
    news_score: float = 0.0
    combined_score: float = 0.0

    def __post_init__(self):
        if self.news_headlines is None:
            self.news_headlines = []

    def to_dict(self) -> Dict:
        return asdict(self)


class FinVizMomentumScanner:
    """
    Scans FinViz Elite for momentum plays with news catalysts.
    """

    def __init__(self):
        self.token = FINVIZ_ELITE_TOKEN
        self.base_url = "https://elite.finviz.com"
        self.candidates: List[MomentumCandidate] = []
        self.last_scan: Optional[datetime] = None
        self.scan_interval_seconds = 300  # 5 minutes

        # Scoring weights
        self.weights = {
            "change_pct": 2.0,      # Change % contribution
            "gap_pct": 1.5,         # Gap contribution
            "volume_surge": 1.0,    # Above avg volume
            "low_float": 1.5,       # Float < 20M bonus
            "has_news": 3.0,        # News catalyst bonus
            "news_keywords": 2.0,   # FDA/earnings/etc bonus
        }

        # Catalyst keywords for scoring
        self.catalyst_keywords = {
            "fda": 5.0,
            "approval": 4.0,
            "earnings": 3.0,
            "beat": 3.0,
            "contract": 3.0,
            "partnership": 2.5,
            "acquisition": 3.0,
            "merger": 3.0,
            "upgrade": 2.5,
            "patent": 2.0,
            "trial": 2.0,
            "phase 3": 3.0,
            "phase 2": 2.0,
            "breakthrough": 3.0,
            "positive": 1.5,
            "revenue": 2.0,
        }

    async def scan_movers(self, min_change: float = 5.0, max_price: float = 20.0) -> List[MomentumCandidate]:
        """
        Scan for small cap movers using FinViz Elite.

        Args:
            min_change: Minimum % change (default 5%)
            max_price: Maximum price (default $20)

        Returns:
            List of MomentumCandidate objects
        """
        if not self.token:
            logger.warning("FinViz Elite token not configured")
            return []

        try:
            # Build filter for penny stock movers
            # cap_small = Small Cap, ta_change_u5 = Up > 5%
            filters = f"cap_small,sh_avgvol_o100,sh_price_u{int(max_price)},ta_change_u{int(min_change)}"
            url = f"{self.base_url}/export.ashx?v=152&f={filters}&auth={self.token}"

            candidates = []
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        text = await response.text()
                        reader = csv.DictReader(io.StringIO(text))

                        for row in reader:
                            try:
                                candidate = self._parse_row(row)
                                if candidate:
                                    candidates.append(candidate)
                            except Exception as e:
                                logger.warning(f"Error parsing row: {e}")
                                continue
                    else:
                        logger.error(f"FinViz API error: {response.status}")

            self.candidates = candidates
            self.last_scan = datetime.now()
            logger.info(f"Found {len(candidates)} momentum candidates from FinViz")

            return candidates

        except Exception as e:
            logger.error(f"FinViz scan error: {e}")
            return []

    async def scan_gappers(self, min_gap: float = 5.0, max_price: float = 20.0) -> List[MomentumCandidate]:
        """
        Scan for pre-market gap-up stocks.
        """
        if not self.token:
            return []

        try:
            filters = f"sh_price_u{int(max_price)},sh_price_o1,ta_gap_u{int(min_gap)},sh_curvol_o100"
            url = f"{self.base_url}/export.ashx?v=152&f={filters}&auth={self.token}"

            candidates = []
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        text = await response.text()
                        reader = csv.DictReader(io.StringIO(text))

                        for row in reader:
                            try:
                                candidate = self._parse_row(row, is_gapper=True)
                                if candidate:
                                    candidates.append(candidate)
                            except Exception as e:
                                continue

            logger.info(f"Found {len(candidates)} gappers from FinViz")
            return candidates

        except Exception as e:
            logger.error(f"FinViz gapper scan error: {e}")
            return []

    def _parse_row(self, row: Dict, is_gapper: bool = False) -> Optional[MomentumCandidate]:
        """Parse CSV row into MomentumCandidate"""
        try:
            symbol = row.get("Ticker", "").strip()
            if not symbol:
                return None

            # Parse numeric fields
            def parse_num(val, default=0.0):
                if not val:
                    return default
                val = str(val).replace(",", "").replace("%", "").replace("$", "")
                try:
                    return float(val)
                except:
                    return default

            def parse_volume(val):
                if not val:
                    return 0
                val = str(val).replace(",", "")
                try:
                    if "M" in val:
                        return int(float(val.replace("M", "")) * 1_000_000)
                    elif "K" in val:
                        return int(float(val.replace("K", "")) * 1_000)
                    return int(float(val))
                except:
                    return 0

            def parse_market_cap(val):
                if not val:
                    return 0.0
                val = str(val).replace(",", "")
                try:
                    if "B" in val:
                        return float(val.replace("B", "")) * 1_000
                    elif "M" in val:
                        return float(val.replace("M", ""))
                    return float(val)
                except:
                    return 0.0

            candidate = MomentumCandidate(
                symbol=symbol,
                company=row.get("Company", ""),
                price=parse_num(row.get("Price")),
                change_pct=parse_num(row.get("Change")),
                gap_pct=parse_num(row.get("Gap")) if is_gapper else 0.0,
                volume=parse_volume(row.get("Volume")),
                float_shares=parse_num(row.get("Float Short", row.get("Float", 0))),
                short_float=parse_num(row.get("Short Float")),
                sector=row.get("Sector", ""),
                industry=row.get("Industry", ""),
                market_cap=parse_market_cap(row.get("Market Cap"))
            )

            # Calculate initial momentum score
            candidate.momentum_score = self._calculate_momentum_score(candidate)

            return candidate

        except Exception as e:
            logger.warning(f"Error parsing row: {e}")
            return None

    def _calculate_momentum_score(self, candidate: MomentumCandidate) -> float:
        """Calculate momentum score based on technicals"""
        score = 0.0

        # Change % contribution (capped at 50%)
        change_contrib = min(candidate.change_pct, 50) * self.weights["change_pct"]
        score += change_contrib

        # Gap contribution
        if candidate.gap_pct > 0:
            gap_contrib = min(candidate.gap_pct, 30) * self.weights["gap_pct"]
            score += gap_contrib

        # Low float bonus (< 20M shares)
        if 0 < candidate.float_shares < 20:
            score += self.weights["low_float"] * 10

        # High short float bonus (potential squeeze)
        if candidate.short_float > 15:
            score += candidate.short_float * 0.5

        return round(score, 2)

    async def enrich_with_news(self, candidates: List[MomentumCandidate]) -> List[MomentumCandidate]:
        """
        Fetch news for each candidate and update scores.
        """
        try:
            from finvizfinance.quote import finvizfinance
        except ImportError:
            logger.warning("finvizfinance not installed - skipping news enrichment")
            return candidates

        for candidate in candidates:
            try:
                stock = finvizfinance(candidate.symbol)
                news_df = stock.ticker_news()

                if news_df is not None and not news_df.empty:
                    # Get recent news (last 24 hours conceptually - we check headlines)
                    headlines = news_df.head(10)["Title"].tolist()
                    candidate.has_news = True
                    candidate.news_count = len(headlines)
                    candidate.news_headlines = headlines[:5]

                    # Check for catalyst keywords
                    combined_text = " ".join(headlines).lower()
                    candidate.catalyst_type = self._detect_catalyst(combined_text)
                    candidate.news_score = self._calculate_news_score(combined_text)

            except Exception as e:
                logger.debug(f"No news for {candidate.symbol}: {e}")
                continue

        # Calculate combined scores
        for candidate in candidates:
            candidate.combined_score = candidate.momentum_score + candidate.news_score

        # Sort by combined score
        candidates.sort(key=lambda x: x.combined_score, reverse=True)

        return candidates

    def _detect_catalyst(self, text: str) -> str:
        """Detect catalyst type from news text"""
        text_lower = text.lower()

        if "fda" in text_lower or "approval" in text_lower:
            return "FDA"
        elif "earnings" in text_lower or "beat" in text_lower or "revenue" in text_lower:
            return "EARNINGS"
        elif "contract" in text_lower or "deal" in text_lower:
            return "CONTRACT"
        elif "merger" in text_lower or "acquisition" in text_lower:
            return "M&A"
        elif "upgrade" in text_lower or "rating" in text_lower:
            return "UPGRADE"
        elif "trial" in text_lower or "phase" in text_lower:
            return "CLINICAL"
        elif "partnership" in text_lower:
            return "PARTNERSHIP"

        return "NEWS"

    def _calculate_news_score(self, text: str) -> float:
        """Calculate news catalyst score"""
        score = 0.0
        text_lower = text.lower()

        # Add base score for having news
        score += self.weights["has_news"]

        # Check for catalyst keywords
        for keyword, weight in self.catalyst_keywords.items():
            if keyword in text_lower:
                score += weight

        return round(min(score, 30), 2)  # Cap at 30

    async def get_top_plays(self, limit: int = 10) -> List[MomentumCandidate]:
        """
        Get top momentum plays with news catalysts.
        Full pipeline: scan -> enrich -> rank -> return top.
        """
        # Scan for movers
        movers = await self.scan_movers()

        # Enrich with news
        enriched = await self.enrich_with_news(movers)

        # Return top candidates
        return enriched[:limit]

    async def sync_to_scalper_watchlist(self, min_score: float = 20.0, max_add: int = 5):
        """
        Auto-add top candidates to scalper watchlist.
        Only adds stocks with combined_score >= min_score.
        """
        try:
            # Get top plays
            top_plays = await self.get_top_plays(limit=max_add * 2)

            # Filter by score
            qualified = [c for c in top_plays if c.combined_score >= min_score]

            # Load current scalper config
            config_path = "ai/scalper_config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)

            current_watchlist = set(config.get("watchlist", []))
            blacklist = set(config.get("blacklist", []))
            added = []

            for candidate in qualified[:max_add]:
                symbol = candidate.symbol
                if symbol not in current_watchlist and symbol not in blacklist:
                    current_watchlist.add(symbol)
                    added.append({
                        "symbol": symbol,
                        "score": candidate.combined_score,
                        "catalyst": candidate.catalyst_type,
                        "change": candidate.change_pct
                    })

            if added:
                # Save updated config
                config["watchlist"] = list(current_watchlist)
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)

                logger.info(f"Added {len(added)} stocks to scalper watchlist: {[a['symbol'] for a in added]}")

            return {
                "added": added,
                "watchlist_size": len(current_watchlist)
            }

        except Exception as e:
            logger.error(f"Error syncing to watchlist: {e}")
            return {"error": str(e)}


# Singleton instance
_scanner_instance: Optional[FinVizMomentumScanner] = None


def get_finviz_scanner() -> FinVizMomentumScanner:
    """Get or create scanner instance"""
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = FinVizMomentumScanner()
    return _scanner_instance
