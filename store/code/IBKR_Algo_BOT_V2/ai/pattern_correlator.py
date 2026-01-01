"""
Pattern Correlation Tracker
===========================
Records data on moving stocks to find correlations and patterns
that predict momentum continuation vs fade.

Tracks:
- Price/volume metrics at time of move
- Technical indicators
- News catalysts
- Time factors
- Outcome (continuation vs fade)

Over time, builds a database to identify which factor combinations
predict successful momentum plays.
"""

import json
import logging
import os
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Data file
PATTERN_DATA_FILE = os.path.join(os.path.dirname(__file__), "pattern_data.json")


@dataclass
class MoverRecord:
    """Record of a stock move event"""

    # Identification
    record_id: str  # unique id
    symbol: str
    timestamp: str  # ISO format

    # Price data at detection
    price: float
    change_percent: float
    volume: int

    # Context metrics
    float_shares: Optional[int] = None
    market_cap: Optional[float] = None
    avg_volume: Optional[int] = None
    relative_volume: float = 1.0
    float_rotation: float = 0.0  # volume / float

    # Technical indicators
    vwap: Optional[float] = None
    vwap_extension: float = 0.0  # % above/below VWAP
    macd_signal: str = "UNKNOWN"  # BULL, BEAR, CROSS_UP, CROSS_DOWN
    rsi: Optional[float] = None
    above_200_ema: bool = False

    # Price levels
    high_of_day: Optional[float] = None
    low_of_day: Optional[float] = None
    percent_from_hod: float = 0.0
    daily_range_percent: float = 0.0

    # Spread/liquidity
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread_percent: float = 0.0

    # Catalyst info
    has_news: bool = False
    news_type: str = "NONE"  # FDA, EARNINGS, MERGER, CONTRACT, ANALYST, NONE
    news_heat: str = "COLD"  # HOT, WARM, COLD

    # Split info
    days_since_split: Optional[int] = None
    split_type: str = "NONE"  # REVERSE, FORWARD, NONE
    smi_score: int = 0

    # Time factors
    hour_of_day: int = 0
    is_premarket: bool = False
    is_first_30_min: bool = False
    day_of_week: int = 0  # 0=Monday

    # Market context
    spy_change: float = 0.0
    qqq_change: float = 0.0
    market_trend: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL

    # OUTCOME (filled in later)
    outcome_15min: Optional[float] = None  # % change 15 min later
    outcome_30min: Optional[float] = None
    outcome_1hr: Optional[float] = None
    outcome_eod: Optional[float] = None  # end of day
    outcome_label: str = "PENDING"  # STRONG_CONT, CONT, FADE, STRONG_FADE, PENDING
    max_gain_after: Optional[float] = None  # max % gain after detection
    max_loss_after: Optional[float] = None  # max % loss after detection

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CorrelationResult:
    """Result of correlation analysis"""

    factor: str
    value: str
    sample_size: int
    avg_outcome_15min: float
    avg_outcome_30min: float
    avg_outcome_1hr: float
    continuation_rate: float  # % that continued vs faded
    avg_max_gain: float
    avg_max_loss: float

    def to_dict(self) -> Dict:
        return asdict(self)


class PatternCorrelator:
    """
    Records and analyzes patterns in stock moves.
    Builds correlation data over time.
    """

    def __init__(self):
        self.records: List[MoverRecord] = []
        self.correlations: Dict[str, List[CorrelationResult]] = {}
        self._load_data()
        logger.info(f"PatternCorrelator initialized with {len(self.records)} records")

    def _load_data(self):
        """Load pattern data from file"""
        try:
            if os.path.exists(PATTERN_DATA_FILE):
                with open(PATTERN_DATA_FILE, "r") as f:
                    data = json.load(f)
                    self.records = [MoverRecord(**r) for r in data.get("records", [])]
                    logger.info(f"Loaded {len(self.records)} pattern records")
        except Exception as e:
            logger.error(f"Error loading pattern data: {e}")

    def _save_data(self):
        """Save pattern data to file"""
        try:
            data = {
                "records": [r.to_dict() for r in self.records],
                "last_updated": datetime.now().isoformat(),
                "total_records": len(self.records),
            }
            with open(PATTERN_DATA_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving pattern data: {e}")

    async def record_mover(
        self, symbol: str, quote: Dict, context: Dict = None
    ) -> MoverRecord:
        """
        Record a moving stock with all relevant data points.

        Args:
            symbol: Stock symbol
            quote: Quote data dict
            context: Additional context (news, technicals, etc)

        Returns:
            The created MoverRecord
        """
        context = context or {}
        now = datetime.now()

        # Generate unique ID
        record_id = f"{symbol}_{now.strftime('%Y%m%d_%H%M%S')}"

        # Extract quote data
        price = quote.get("price", 0) or quote.get("last", 0) or 0
        change_pct = quote.get("change_percent", 0) or 0
        volume = quote.get("volume", 0) or 0
        bid = quote.get("bid", 0)
        ask = quote.get("ask", 0)
        high = quote.get("high", price)
        low = quote.get("low", price)

        # Calculate derived metrics
        spread_pct = ((ask - bid) / price * 100) if price > 0 and ask and bid else 0
        daily_range = ((high - low) / low * 100) if low > 0 else 0
        pct_from_hod = ((high - price) / high * 100) if high > 0 else 0

        # Time factors
        hour = now.hour
        is_premarket = hour < 9 or (hour == 9 and now.minute < 30)
        is_first_30 = hour == 9 and now.minute >= 30 or hour == 10 and now.minute < 0

        # Create record
        record = MoverRecord(
            record_id=record_id,
            symbol=symbol,
            timestamp=now.isoformat(),
            price=price,
            change_percent=change_pct,
            volume=volume,
            bid=bid,
            ask=ask,
            spread_percent=round(spread_pct, 2),
            high_of_day=high,
            low_of_day=low,
            percent_from_hod=round(pct_from_hod, 1),
            daily_range_percent=round(daily_range, 1),
            hour_of_day=hour,
            is_premarket=is_premarket,
            is_first_30_min=is_first_30,
            day_of_week=now.weekday(),
            # From context
            float_shares=context.get("float"),
            relative_volume=context.get("rel_volume", 1.0),
            float_rotation=context.get("float_rotation", 0),
            vwap=context.get("vwap"),
            vwap_extension=context.get("vwap_extension", 0),
            macd_signal=context.get("macd_signal", "UNKNOWN"),
            has_news=context.get("has_news", False),
            news_type=context.get("news_type", "NONE"),
            news_heat=context.get("news_heat", "COLD"),
            days_since_split=context.get("days_since_split"),
            split_type=context.get("split_type", "NONE"),
            smi_score=context.get("smi_score", 0),
            spy_change=context.get("spy_change", 0),
            qqq_change=context.get("qqq_change", 0),
            market_trend=context.get("market_trend", "NEUTRAL"),
        )

        self.records.append(record)
        self._save_data()

        logger.info(f"Recorded mover: {symbol} {change_pct:+.1f}% @ ${price:.2f}")

        return record

    async def update_outcome(self, record_id: str, outcome_data: Dict):
        """
        Update a record with outcome data (called later to see what happened).

        Args:
            record_id: The record to update
            outcome_data: Dict with outcome_15min, outcome_30min, etc.
        """
        for record in self.records:
            if record.record_id == record_id:
                record.outcome_15min = outcome_data.get("outcome_15min")
                record.outcome_30min = outcome_data.get("outcome_30min")
                record.outcome_1hr = outcome_data.get("outcome_1hr")
                record.outcome_eod = outcome_data.get("outcome_eod")
                record.max_gain_after = outcome_data.get("max_gain")
                record.max_loss_after = outcome_data.get("max_loss")

                # Label the outcome
                if record.outcome_30min is not None:
                    if record.outcome_30min >= 5:
                        record.outcome_label = "STRONG_CONT"
                    elif record.outcome_30min >= 1:
                        record.outcome_label = "CONT"
                    elif record.outcome_30min <= -5:
                        record.outcome_label = "STRONG_FADE"
                    elif record.outcome_30min <= -1:
                        record.outcome_label = "FADE"
                    else:
                        record.outcome_label = "FLAT"

                self._save_data()
                logger.info(f"Updated outcome for {record_id}: {record.outcome_label}")
                return

        logger.warning(f"Record not found: {record_id}")

    def analyze_correlations(self) -> Dict:
        """
        Analyze correlations between factors and outcomes.
        Returns insights on what predicts continuation vs fade.
        """
        # Filter to records with outcomes
        completed = [r for r in self.records if r.outcome_label != "PENDING"]

        if len(completed) < 10:
            return {
                "status": "insufficient_data",
                "records_with_outcome": len(completed),
                "message": "Need at least 10 completed records for analysis",
            }

        results = {}

        # Analyze by change % buckets
        results["by_change_percent"] = self._analyze_factor(
            completed, lambda r: self._bucket_change(r.change_percent), "change_percent"
        )

        # Analyze by relative volume
        results["by_rel_volume"] = self._analyze_factor(
            completed, lambda r: self._bucket_rvol(r.relative_volume), "relative_volume"
        )

        # Analyze by float rotation
        results["by_float_rotation"] = self._analyze_factor(
            completed,
            lambda r: self._bucket_float_rotation(r.float_rotation),
            "float_rotation",
        )

        # Analyze by VWAP extension
        results["by_vwap_extension"] = self._analyze_factor(
            completed, lambda r: self._bucket_vwap(r.vwap_extension), "vwap_extension"
        )

        # Analyze by HOD proximity
        results["by_hod_proximity"] = self._analyze_factor(
            completed,
            lambda r: self._bucket_hod(r.percent_from_hod),
            "percent_from_hod",
        )

        # Analyze by news heat
        results["by_news_heat"] = self._analyze_factor(
            completed, lambda r: r.news_heat, "news_heat"
        )

        # Analyze by time
        results["by_premarket"] = self._analyze_factor(
            completed, lambda r: "PREMARKET" if r.is_premarket else "REGULAR", "session"
        )

        # Analyze by MACD
        results["by_macd"] = self._analyze_factor(
            completed, lambda r: r.macd_signal, "macd_signal"
        )

        # Analyze by spread
        results["by_spread"] = self._analyze_factor(
            completed, lambda r: self._bucket_spread(r.spread_percent), "spread"
        )

        # Analyze by split status
        results["by_split_status"] = self._analyze_factor(
            completed,
            lambda r: (
                r.split_type
                if r.days_since_split and r.days_since_split < 30
                else "NO_RECENT"
            ),
            "split_status",
        )

        # Analyze by market trend
        results["by_market_trend"] = self._analyze_factor(
            completed, lambda r: r.market_trend, "market_trend"
        )

        # Overall stats
        cont_rate = sum(
            1 for r in completed if r.outcome_label in ["STRONG_CONT", "CONT"]
        ) / len(completed)

        results["summary"] = {
            "total_records": len(completed),
            "continuation_rate": round(cont_rate * 100, 1),
            "avg_outcome_30min": round(
                statistics.mean(
                    [r.outcome_30min for r in completed if r.outcome_30min]
                ),
                2,
            ),
            "best_factor": self._find_best_factor(results),
            "worst_factor": self._find_worst_factor(results),
        }

        return results

    def _analyze_factor(
        self, records: List[MoverRecord], bucket_fn, factor_name: str
    ) -> List[Dict]:
        """Analyze outcomes by a specific factor"""
        buckets = {}

        for r in records:
            bucket = bucket_fn(r)
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(r)

        results = []
        for bucket, recs in buckets.items():
            if len(recs) < 3:  # Need minimum sample
                continue

            outcomes_30 = [r.outcome_30min for r in recs if r.outcome_30min is not None]
            outcomes_15 = [r.outcome_15min for r in recs if r.outcome_15min is not None]
            outcomes_1h = [r.outcome_1hr for r in recs if r.outcome_1hr is not None]
            max_gains = [r.max_gain_after for r in recs if r.max_gain_after is not None]
            max_losses = [
                r.max_loss_after for r in recs if r.max_loss_after is not None
            ]

            cont_count = sum(
                1 for r in recs if r.outcome_label in ["STRONG_CONT", "CONT"]
            )

            results.append(
                {
                    "value": bucket,
                    "sample_size": len(recs),
                    "avg_outcome_15min": (
                        round(statistics.mean(outcomes_15), 2) if outcomes_15 else 0
                    ),
                    "avg_outcome_30min": (
                        round(statistics.mean(outcomes_30), 2) if outcomes_30 else 0
                    ),
                    "avg_outcome_1hr": (
                        round(statistics.mean(outcomes_1h), 2) if outcomes_1h else 0
                    ),
                    "continuation_rate": round(cont_count / len(recs) * 100, 1),
                    "avg_max_gain": (
                        round(statistics.mean(max_gains), 2) if max_gains else 0
                    ),
                    "avg_max_loss": (
                        round(statistics.mean(max_losses), 2) if max_losses else 0
                    ),
                }
            )

        # Sort by continuation rate
        results.sort(key=lambda x: x["continuation_rate"], reverse=True)

        return results

    def _bucket_change(self, pct: float) -> str:
        if pct >= 50:
            return "50%+"
        if pct >= 30:
            return "30-50%"
        if pct >= 20:
            return "20-30%"
        if pct >= 10:
            return "10-20%"
        return "<10%"

    def _bucket_rvol(self, rvol: float) -> str:
        if rvol >= 5:
            return "5x+"
        if rvol >= 3:
            return "3-5x"
        if rvol >= 2:
            return "2-3x"
        if rvol >= 1:
            return "1-2x"
        return "<1x"

    def _bucket_float_rotation(self, rot: float) -> str:
        if rot >= 2:
            return "2x+ float"
        if rot >= 1:
            return "1-2x float"
        if rot >= 0.5:
            return "0.5-1x float"
        return "<0.5x float"

    def _bucket_vwap(self, ext: float) -> str:
        if ext >= 15:
            return "15%+ above"
        if ext >= 5:
            return "5-15% above"
        if ext >= 0:
            return "0-5% above"
        if ext >= -5:
            return "0-5% below"
        return "5%+ below"

    def _bucket_hod(self, pct: float) -> str:
        if pct <= 2:
            return "Near HOD (0-2%)"
        if pct <= 5:
            return "Close to HOD (2-5%)"
        if pct <= 10:
            return "Mid range (5-10%)"
        return "Far from HOD (10%+)"

    def _bucket_spread(self, spread: float) -> str:
        if spread <= 0.5:
            return "Tight (<0.5%)"
        if spread <= 1:
            return "Normal (0.5-1%)"
        if spread <= 2:
            return "Wide (1-2%)"
        return "Very wide (2%+)"

    def _find_best_factor(self, results: Dict) -> Dict:
        """Find the factor with highest continuation rate"""
        best = {"factor": None, "value": None, "continuation_rate": 0}

        for factor, data in results.items():
            if factor == "summary" or not isinstance(data, list):
                continue
            for item in data:
                if (
                    item.get("sample_size", 0) >= 5
                    and item.get("continuation_rate", 0) > best["continuation_rate"]
                ):
                    best = {
                        "factor": factor,
                        "value": item["value"],
                        "continuation_rate": item["continuation_rate"],
                        "sample_size": item["sample_size"],
                    }

        return best

    def _find_worst_factor(self, results: Dict) -> Dict:
        """Find the factor with lowest continuation rate"""
        worst = {"factor": None, "value": None, "continuation_rate": 100}

        for factor, data in results.items():
            if factor == "summary" or not isinstance(data, list):
                continue
            for item in data:
                if (
                    item.get("sample_size", 0) >= 5
                    and item.get("continuation_rate", 0) < worst["continuation_rate"]
                ):
                    worst = {
                        "factor": factor,
                        "value": item["value"],
                        "continuation_rate": item["continuation_rate"],
                        "sample_size": item["sample_size"],
                    }

        return worst

    def get_prediction_score(
        self, symbol: str, quote: Dict, context: Dict = None
    ) -> Dict:
        """
        Use historical correlations to score a potential trade.
        Returns a score and factors.
        """
        context = context or {}

        # If not enough data, return neutral
        completed = [r for r in self.records if r.outcome_label != "PENDING"]
        if len(completed) < 20:
            return {
                "symbol": symbol,
                "score": 50,
                "confidence": "LOW",
                "message": "Insufficient historical data",
                "factors": [],
            }

        # Run correlation analysis
        correlations = self.analyze_correlations()

        # Score based on current metrics matching high-continuation patterns
        score = 50  # Start neutral
        factors = []

        change_pct = quote.get("change_percent", 0)
        rel_vol = context.get("rel_volume", 1.0)
        vwap_ext = context.get("vwap_extension", 0)
        news_heat = context.get("news_heat", "COLD")
        macd = context.get("macd_signal", "UNKNOWN")

        # Check each factor against correlations
        # Change %
        change_bucket = self._bucket_change(change_pct)
        for item in correlations.get("by_change_percent", []):
            if item["value"] == change_bucket and item["sample_size"] >= 5:
                adj = (item["continuation_rate"] - 50) * 0.3
                score += adj
                factors.append(
                    f"Change {change_bucket}: {item['continuation_rate']:.0f}% cont rate"
                )

        # Rel volume
        rvol_bucket = self._bucket_rvol(rel_vol)
        for item in correlations.get("by_rel_volume", []):
            if item["value"] == rvol_bucket and item["sample_size"] >= 5:
                adj = (item["continuation_rate"] - 50) * 0.25
                score += adj
                factors.append(
                    f"RVOL {rvol_bucket}: {item['continuation_rate']:.0f}% cont rate"
                )

        # VWAP
        vwap_bucket = self._bucket_vwap(vwap_ext)
        for item in correlations.get("by_vwap_extension", []):
            if item["value"] == vwap_bucket and item["sample_size"] >= 5:
                adj = (item["continuation_rate"] - 50) * 0.2
                score += adj
                factors.append(
                    f"VWAP {vwap_bucket}: {item['continuation_rate']:.0f}% cont rate"
                )

        # News heat
        for item in correlations.get("by_news_heat", []):
            if item["value"] == news_heat and item["sample_size"] >= 5:
                adj = (item["continuation_rate"] - 50) * 0.15
                score += adj
                factors.append(
                    f"News {news_heat}: {item['continuation_rate']:.0f}% cont rate"
                )

        # MACD
        for item in correlations.get("by_macd", []):
            if item["value"] == macd and item["sample_size"] >= 5:
                adj = (item["continuation_rate"] - 50) * 0.1
                score += adj
                factors.append(
                    f"MACD {macd}: {item['continuation_rate']:.0f}% cont rate"
                )

        # Clamp score
        score = max(0, min(100, score))

        # Confidence based on data
        if len(completed) >= 100:
            confidence = "HIGH"
        elif len(completed) >= 50:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Signal
        if score >= 70:
            signal = "STRONG_BUY"
            color = "#22c55e"
        elif score >= 55:
            signal = "LEAN_BUY"
            color = "#86efac"
        elif score <= 30:
            signal = "AVOID"
            color = "#ef4444"
        elif score <= 45:
            signal = "CAUTION"
            color = "#fca5a5"
        else:
            signal = "NEUTRAL"
            color = "#fbbf24"

        return {
            "symbol": symbol,
            "score": round(score),
            "signal": signal,
            "color": color,
            "confidence": confidence,
            "data_points": len(completed),
            "factors": factors,
        }

    def get_stats(self) -> Dict:
        """Get tracker statistics"""
        completed = [r for r in self.records if r.outcome_label != "PENDING"]
        pending = [r for r in self.records if r.outcome_label == "PENDING"]

        return {
            "total_records": len(self.records),
            "completed": len(completed),
            "pending": len(pending),
            "symbols_tracked": len(set(r.symbol for r in self.records)),
            "date_range": {
                "first": self.records[0].timestamp if self.records else None,
                "last": self.records[-1].timestamp if self.records else None,
            },
        }

    def get_recent_records(self, limit: int = 20) -> List[Dict]:
        """Get recent records"""
        return [r.to_dict() for r in self.records[-limit:]]


# Singleton
_correlator: Optional[PatternCorrelator] = None


def get_pattern_correlator() -> PatternCorrelator:
    """Get or create correlator singleton"""
    global _correlator
    if _correlator is None:
        _correlator = PatternCorrelator()
    return _correlator


async def record_mover(symbol: str, quote: Dict, context: Dict = None) -> MoverRecord:
    """Quick function to record a mover"""
    correlator = get_pattern_correlator()
    return await correlator.record_mover(symbol, quote, context)


def get_prediction(symbol: str, quote: Dict, context: Dict = None) -> Dict:
    """Get prediction score for a symbol"""
    correlator = get_pattern_correlator()
    return correlator.get_prediction_score(symbol, quote, context)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test
    correlator = get_pattern_correlator()
    print(f"Stats: {correlator.get_stats()}")
    print(f"Correlations: {correlator.analyze_correlations()}")
