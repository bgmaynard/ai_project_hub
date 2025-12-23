"""
Momentum Scorer
================
Ranks watchlist stocks by real-time momentum strength.

Scoring Components (0-100 total):
1. Spike Strength (25 pts) - Recent % gain
2. Volume Surge (25 pts) - Volume vs average
3. HOD Proximity (20 pts) - Distance from high of day
4. VWAP Position (15 pts) - Above VWAP = bullish
5. Float Rotation (15 pts) - Volume / float traded

Usage:
    from ai.momentum_scorer import get_momentum_scorer
    scorer = get_momentum_scorer()
    rankings = await scorer.rank_watchlist(symbols)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import httpx

logger = logging.getLogger(__name__)


@dataclass
class MomentumScore:
    """Momentum score for a single stock"""
    symbol: str
    total_score: float  # 0-100

    # Component scores
    spike_score: float      # 0-25
    volume_score: float     # 0-25
    hod_score: float        # 0-20
    vwap_score: float       # 0-15
    rotation_score: float   # 0-15

    # Raw values
    change_percent: float
    rel_volume: float
    percent_from_hod: float
    vwap_extension: float
    float_rotation: float

    # Metadata
    price: float
    volume: int
    float_shares: Optional[int]
    rank: int = 0
    momentum_grade: str = "F"  # A, B, C, D, F

    def to_dict(self) -> Dict:
        return asdict(self)


class MomentumScorer:
    """
    Scores and ranks stocks by momentum strength.
    Higher score = stronger momentum = better scalp candidate.
    """

    def __init__(self):
        self.api_url = "http://localhost:9100"
        self.cache: Dict[str, MomentumScore] = {}
        self.cache_ttl = 10  # seconds
        self.last_update = None

        # Scoring thresholds (calibrated for penny/momentum stocks)
        self.thresholds = {
            # Spike strength thresholds
            "spike_excellent": 10.0,   # +10% = max points
            "spike_good": 5.0,         # +5%
            "spike_ok": 2.0,           # +2%

            # Volume surge thresholds
            "volume_excellent": 5.0,   # 5x avg = max points
            "volume_good": 3.0,        # 3x avg
            "volume_ok": 1.5,          # 1.5x avg

            # HOD proximity thresholds
            "hod_excellent": 1.0,      # Within 1% of HOD
            "hod_good": 3.0,           # Within 3%
            "hod_ok": 5.0,             # Within 5%

            # VWAP thresholds
            "vwap_excellent": 3.0,     # 3%+ above VWAP
            "vwap_good": 1.0,          # 1%+ above
            "vwap_ok": 0.0,            # At VWAP

            # Float rotation thresholds
            "rotation_excellent": 1.0,  # 100% float traded
            "rotation_good": 0.5,       # 50% float
            "rotation_ok": 0.25,        # 25% float
        }

        logger.info("MomentumScorer initialized")

    def _score_spike(self, change_pct: float) -> float:
        """Score based on % change (0-25 points)"""
        if change_pct <= 0:
            return 0.0

        if change_pct >= self.thresholds["spike_excellent"]:
            return 25.0
        elif change_pct >= self.thresholds["spike_good"]:
            # Scale 5-10% to 15-25 points
            return 15.0 + (change_pct - 5.0) / 5.0 * 10.0
        elif change_pct >= self.thresholds["spike_ok"]:
            # Scale 2-5% to 8-15 points
            return 8.0 + (change_pct - 2.0) / 3.0 * 7.0
        else:
            # Scale 0-2% to 0-8 points
            return change_pct / 2.0 * 8.0

    def _score_volume(self, rel_vol: float) -> float:
        """Score based on relative volume (0-25 points)"""
        if rel_vol <= 0.5:
            return 0.0

        if rel_vol >= self.thresholds["volume_excellent"]:
            return 25.0
        elif rel_vol >= self.thresholds["volume_good"]:
            # Scale 3-5x to 15-25 points
            return 15.0 + (rel_vol - 3.0) / 2.0 * 10.0
        elif rel_vol >= self.thresholds["volume_ok"]:
            # Scale 1.5-3x to 8-15 points
            return 8.0 + (rel_vol - 1.5) / 1.5 * 7.0
        else:
            # Scale 0.5-1.5x to 0-8 points
            return (rel_vol - 0.5) / 1.0 * 8.0

    def _score_hod(self, pct_from_hod: float) -> float:
        """Score based on proximity to high of day (0-20 points)"""
        # Lower is better (closer to HOD)
        if pct_from_hod <= self.thresholds["hod_excellent"]:
            return 20.0
        elif pct_from_hod <= self.thresholds["hod_good"]:
            # Scale 1-3% to 12-20 points
            return 20.0 - (pct_from_hod - 1.0) / 2.0 * 8.0
        elif pct_from_hod <= self.thresholds["hod_ok"]:
            # Scale 3-5% to 5-12 points
            return 12.0 - (pct_from_hod - 3.0) / 2.0 * 7.0
        elif pct_from_hod <= 10.0:
            # Scale 5-10% to 0-5 points
            return 5.0 - (pct_from_hod - 5.0) / 5.0 * 5.0
        else:
            return 0.0

    def _score_vwap(self, vwap_ext: float) -> float:
        """Score based on VWAP extension (0-15 points)"""
        if vwap_ext >= self.thresholds["vwap_excellent"]:
            return 15.0
        elif vwap_ext >= self.thresholds["vwap_good"]:
            # Scale 1-3% to 10-15 points
            return 10.0 + (vwap_ext - 1.0) / 2.0 * 5.0
        elif vwap_ext >= self.thresholds["vwap_ok"]:
            # Scale 0-1% to 5-10 points
            return 5.0 + vwap_ext / 1.0 * 5.0
        elif vwap_ext >= -2.0:
            # Scale -2% to 0% to 0-5 points
            return (vwap_ext + 2.0) / 2.0 * 5.0
        else:
            return 0.0

    def _score_rotation(self, float_rotation: float) -> float:
        """Score based on float rotation (0-15 points)"""
        if float_rotation <= 0:
            return 0.0

        if float_rotation >= self.thresholds["rotation_excellent"]:
            return 15.0
        elif float_rotation >= self.thresholds["rotation_good"]:
            # Scale 50-100% to 10-15 points
            return 10.0 + (float_rotation - 0.5) / 0.5 * 5.0
        elif float_rotation >= self.thresholds["rotation_ok"]:
            # Scale 25-50% to 5-10 points
            return 5.0 + (float_rotation - 0.25) / 0.25 * 5.0
        else:
            # Scale 0-25% to 0-5 points
            return float_rotation / 0.25 * 5.0

    def _get_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 80:
            return "A"
        elif score >= 65:
            return "B"
        elif score >= 50:
            return "C"
        elif score >= 35:
            return "D"
        else:
            return "F"

    async def score_symbol(self, symbol: str, quote_data: Dict = None) -> Optional[MomentumScore]:
        """Calculate momentum score for a single symbol"""
        try:
            # Get quote data if not provided
            if not quote_data:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{self.api_url}/api/worklist", timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        for stock in data.get("data", []):
                            if stock.get("symbol") == symbol:
                                quote_data = stock
                                break

            if not quote_data:
                return None

            # Extract values
            change_pct = float(quote_data.get("change_percent", 0) or 0)
            rel_vol = float(quote_data.get("rel_volume", 0) or 0)
            pct_from_hod = float(quote_data.get("percent_from_hod", 100) or 100)
            vwap_ext = float(quote_data.get("vwap_extension", 0) or 0)
            price = float(quote_data.get("price", 0) or 0)
            volume = int(quote_data.get("volume", 0) or 0)

            # Get float for rotation calculation
            float_shares = quote_data.get("float")
            if float_shares and isinstance(float_shares, str):
                # Parse "42.5M" format
                if "M" in float_shares:
                    float_shares = int(float(float_shares.replace("M", "")) * 1_000_000)
                elif "K" in float_shares:
                    float_shares = int(float(float_shares.replace("K", "")) * 1_000)
                else:
                    try:
                        float_shares = int(float_shares)
                    except:
                        float_shares = None

            # Calculate float rotation
            float_rotation = 0.0
            if float_shares and float_shares > 0:
                float_rotation = volume / float_shares

            # Calculate component scores
            spike_score = self._score_spike(change_pct)
            volume_score = self._score_volume(rel_vol)
            hod_score = self._score_hod(pct_from_hod)
            vwap_score = self._score_vwap(vwap_ext)
            rotation_score = self._score_rotation(float_rotation)

            # Total score
            total = spike_score + volume_score + hod_score + vwap_score + rotation_score

            return MomentumScore(
                symbol=symbol,
                total_score=round(total, 1),
                spike_score=round(spike_score, 1),
                volume_score=round(volume_score, 1),
                hod_score=round(hod_score, 1),
                vwap_score=round(vwap_score, 1),
                rotation_score=round(rotation_score, 1),
                change_percent=round(change_pct, 2),
                rel_volume=round(rel_vol, 2),
                percent_from_hod=round(pct_from_hod, 2),
                vwap_extension=round(vwap_ext, 2),
                float_rotation=round(float_rotation * 100, 1),  # As percentage
                price=price,
                volume=volume,
                float_shares=float_shares,
                momentum_grade=self._get_grade(total)
            )

        except Exception as e:
            logger.error(f"Error scoring {symbol}: {e}")
            return None

    async def rank_watchlist(self, symbols: List[str] = None) -> List[MomentumScore]:
        """Rank all watchlist stocks by momentum"""
        try:
            # Get full watchlist data
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.api_url}/api/worklist", timeout=10)
                if resp.status_code != 200:
                    return []

                data = resp.json()
                stocks = data.get("data", [])

            # Filter by symbols if provided
            if symbols:
                stocks = [s for s in stocks if s.get("symbol") in symbols]

            # Score each stock
            scores = []
            for stock in stocks:
                score = await self.score_symbol(stock.get("symbol"), stock)
                if score:
                    scores.append(score)

            # Sort by total score (highest first)
            scores.sort(key=lambda x: x.total_score, reverse=True)

            # Assign ranks
            for i, score in enumerate(scores):
                score.rank = i + 1

            # Cache results
            self.cache = {s.symbol: s for s in scores}
            self.last_update = datetime.now()

            return scores

        except Exception as e:
            logger.error(f"Error ranking watchlist: {e}")
            return []

    def get_cached_rankings(self) -> List[MomentumScore]:
        """Get cached rankings without re-fetching"""
        return sorted(self.cache.values(), key=lambda x: x.rank)

    def get_top_movers(self, limit: int = 5) -> List[MomentumScore]:
        """Get top N momentum stocks"""
        rankings = self.get_cached_rankings()
        return rankings[:limit]

    def get_grade_summary(self) -> Dict:
        """Get count of stocks by grade"""
        rankings = self.get_cached_rankings()
        summary = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        for score in rankings:
            summary[score.momentum_grade] += 1
        return summary


# Singleton instance
_momentum_scorer: Optional[MomentumScorer] = None


def get_momentum_scorer() -> MomentumScorer:
    """Get or create momentum scorer singleton"""
    global _momentum_scorer
    if _momentum_scorer is None:
        _momentum_scorer = MomentumScorer()
    return _momentum_scorer


# Test
if __name__ == "__main__":
    import asyncio

    async def test():
        scorer = get_momentum_scorer()
        rankings = await scorer.rank_watchlist()

        print("\n=== MOMENTUM RANKINGS ===\n")
        print(f"{'Rank':<5} {'Symbol':<8} {'Score':<7} {'Grade':<6} {'Chg%':<8} {'RVol':<6} {'HOD%':<7} {'VWAP%':<7}")
        print("-" * 65)

        for score in rankings[:20]:
            print(f"{score.rank:<5} {score.symbol:<8} {score.total_score:<7} {score.momentum_grade:<6} "
                  f"{score.change_percent:>+6.1f}% {score.rel_volume:<6.1f} {score.percent_from_hod:<7.1f} {score.vwap_extension:>+6.1f}%")

        print(f"\nGrade Summary: {scorer.get_grade_summary()}")

    asyncio.run(test())
