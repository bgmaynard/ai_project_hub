"""
Stock Split Tracker & Momentum Analyzer
=======================================
Tracks stock splits (forward and reverse) and analyzes
post-split momentum patterns to build predictive signals.

Key metrics:
- Days since split
- Split ratio
- Post-split momentum decay
- Split momentum index (SMI)
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import yfinance as yf

logger = logging.getLogger(__name__)

# Data file for split history
SPLIT_DATA_FILE = os.path.join(os.path.dirname(__file__), "split_history.json")


@dataclass
class SplitEvent:
    """Record of a stock split"""

    symbol: str
    split_date: str  # YYYY-MM-DD
    split_ratio: str  # e.g., "1:50" for reverse, "2:1" for forward
    split_type: str  # REVERSE or FORWARD
    pre_split_price: float
    post_split_price: float
    pre_split_float: Optional[float] = None
    post_split_float: Optional[float] = None

    # Momentum tracking
    momentum_day_1: Optional[float] = None  # % change day 1
    momentum_day_5: Optional[float] = None  # % change day 5
    momentum_day_10: Optional[float] = None  # % change day 10
    momentum_day_20: Optional[float] = None  # % change day 20
    momentum_day_30: Optional[float] = None  # % change day 30

    def days_since_split(self) -> int:
        """Calculate days since split"""
        try:
            split_dt = datetime.strptime(self.split_date, "%Y-%m-%d")
            return (datetime.now() - split_dt).days
        except:
            return -1

    def to_dict(self) -> Dict:
        return {**asdict(self), "days_since_split": self.days_since_split()}


@dataclass
class SplitMomentumStats:
    """Aggregate statistics for split momentum patterns"""

    total_splits_tracked: int = 0
    avg_day_1_momentum: float = 0.0
    avg_day_5_momentum: float = 0.0
    avg_day_10_momentum: float = 0.0
    avg_day_20_momentum: float = 0.0
    avg_day_30_momentum: float = 0.0

    # By split type
    reverse_split_avg_momentum: float = 0.0
    forward_split_avg_momentum: float = 0.0

    # Patterns
    pct_positive_day_1: float = 0.0
    pct_positive_day_5: float = 0.0
    pct_fade_after_day_5: float = 0.0  # % that fade after initial pop


class SplitTracker:
    """
    Tracks stock splits and analyzes momentum patterns.
    Uses this data to generate predictive signals.
    """

    def __init__(self):
        self.splits: Dict[str, List[SplitEvent]] = {}  # symbol -> list of splits
        self.stats = SplitMomentumStats()
        self._load_data()
        logger.info("SplitTracker initialized")

    def _load_data(self):
        """Load split history from file"""
        try:
            if os.path.exists(SPLIT_DATA_FILE):
                with open(SPLIT_DATA_FILE, "r") as f:
                    data = json.load(f)
                    for symbol, splits in data.get("splits", {}).items():
                        self.splits[symbol] = [SplitEvent(**s) for s in splits]
                    stats_data = data.get("stats", {})
                    if stats_data:
                        self.stats = SplitMomentumStats(**stats_data)
                logger.info(
                    f"Loaded {sum(len(s) for s in self.splits.values())} splits"
                )
        except Exception as e:
            logger.error(f"Error loading split data: {e}")

    def _save_data(self):
        """Save split history to file"""
        try:
            data = {
                "splits": {
                    symbol: [s.to_dict() for s in splits]
                    for symbol, splits in self.splits.items()
                },
                "stats": asdict(self.stats),
                "last_updated": datetime.now().isoformat(),
            }
            with open(SPLIT_DATA_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving split data: {e}")

    def fetch_split_history(self, symbol: str) -> List[SplitEvent]:
        """Fetch split history for a symbol from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            splits = ticker.splits

            if splits is None or len(splits) == 0:
                return []

            events = []
            for date, ratio in splits.items():
                split_date = date.strftime("%Y-%m-%d")

                # Determine split type
                if ratio < 1:
                    split_type = "REVERSE"
                    # Convert to readable format (e.g., 0.02 -> "1:50")
                    reverse_ratio = int(1 / ratio)
                    split_ratio = f"1:{reverse_ratio}"
                else:
                    split_type = "FORWARD"
                    split_ratio = f"{int(ratio)}:1"

                # Get price around split date
                try:
                    hist = ticker.history(
                        start=date - timedelta(days=5), end=date + timedelta(days=35)
                    )
                    if len(hist) > 0:
                        pre_price = hist["Close"].iloc[0] if len(hist) > 0 else 0
                        post_price = hist["Close"].iloc[-1] if len(hist) > 1 else 0

                        # Calculate momentum at different intervals
                        momentum_1 = None
                        momentum_5 = None
                        momentum_10 = None
                        momentum_20 = None
                        momentum_30 = None

                        if len(hist) > 1:
                            day_1_price = (
                                hist["Close"].iloc[1] if len(hist) > 1 else None
                            )
                            if day_1_price and pre_price:
                                momentum_1 = (
                                    (day_1_price - pre_price) / pre_price
                                ) * 100

                        if len(hist) > 5:
                            day_5_price = (
                                hist["Close"].iloc[5] if len(hist) > 5 else None
                            )
                            if day_5_price and pre_price:
                                momentum_5 = (
                                    (day_5_price - pre_price) / pre_price
                                ) * 100

                        if len(hist) > 10:
                            day_10_price = (
                                hist["Close"].iloc[10] if len(hist) > 10 else None
                            )
                            if day_10_price and pre_price:
                                momentum_10 = (
                                    (day_10_price - pre_price) / pre_price
                                ) * 100

                        if len(hist) > 20:
                            day_20_price = (
                                hist["Close"].iloc[20] if len(hist) > 20 else None
                            )
                            if day_20_price and pre_price:
                                momentum_20 = (
                                    (day_20_price - pre_price) / pre_price
                                ) * 100

                        if len(hist) > 30:
                            day_30_price = (
                                hist["Close"].iloc[30] if len(hist) > 30 else None
                            )
                            if day_30_price and pre_price:
                                momentum_30 = (
                                    (day_30_price - pre_price) / pre_price
                                ) * 100
                    else:
                        pre_price = 0
                        post_price = 0
                        momentum_1 = None
                        momentum_5 = None
                        momentum_10 = None
                        momentum_20 = None
                        momentum_30 = None
                except:
                    pre_price = 0
                    post_price = 0
                    momentum_1 = None
                    momentum_5 = None
                    momentum_10 = None
                    momentum_20 = None
                    momentum_30 = None

                event = SplitEvent(
                    symbol=symbol,
                    split_date=split_date,
                    split_ratio=split_ratio,
                    split_type=split_type,
                    pre_split_price=pre_price,
                    post_split_price=post_price,
                    momentum_day_1=momentum_1,
                    momentum_day_5=momentum_5,
                    momentum_day_10=momentum_10,
                    momentum_day_20=momentum_20,
                    momentum_day_30=momentum_30,
                )
                events.append(event)

            # Store and save
            self.splits[symbol] = events
            self._save_data()

            return events

        except Exception as e:
            logger.error(f"Error fetching splits for {symbol}: {e}")
            return []

    def get_recent_split(self, symbol: str) -> Optional[SplitEvent]:
        """Get most recent split for a symbol"""
        if symbol not in self.splits:
            self.fetch_split_history(symbol)

        splits = self.splits.get(symbol, [])
        if not splits:
            return None

        # Sort by date and get most recent
        sorted_splits = sorted(splits, key=lambda x: x.split_date, reverse=True)
        return sorted_splits[0]

    def get_split_momentum_index(self, symbol: str) -> Dict:
        """
        Calculate Split Momentum Index (SMI) for a symbol.

        SMI considers:
        - Days since split (recency)
        - Split type (reverse vs forward)
        - Historical momentum patterns

        Returns a score from -100 to +100:
        - Negative = bearish signal (recent reverse split, fading)
        - Positive = bullish signal (momentum holding)
        - Zero = neutral or no recent split
        """
        recent_split = self.get_recent_split(symbol)

        if not recent_split:
            return {
                "symbol": symbol,
                "has_split": False,
                "smi_score": 0,
                "smi_signal": "NEUTRAL",
                "message": "No split history found",
            }

        days = recent_split.days_since_split()

        # Base score starts at 0
        smi_score = 0
        factors = []

        # Factor 1: Recency (more recent = more impact)
        if days <= 5:
            recency_factor = 50  # High impact
            factors.append(f"Very recent split ({days} days)")
        elif days <= 15:
            recency_factor = 30
            factors.append(f"Recent split ({days} days)")
        elif days <= 30:
            recency_factor = 15
            factors.append(f"Moderate split ({days} days)")
        else:
            recency_factor = 5
            factors.append(f"Old split ({days} days)")

        # Factor 2: Split type
        if recent_split.split_type == "REVERSE":
            # Reverse splits are generally bearish
            type_factor = -20
            factors.append("Reverse split (bearish)")
        else:
            # Forward splits are generally bullish
            type_factor = 10
            factors.append("Forward split (bullish)")

        # Factor 3: Post-split momentum
        momentum_factor = 0
        if recent_split.momentum_day_5 is not None:
            if recent_split.momentum_day_5 > 20:
                momentum_factor = 20
                factors.append(
                    f"Strong 5-day momentum ({recent_split.momentum_day_5:.1f}%)"
                )
            elif recent_split.momentum_day_5 > 0:
                momentum_factor = 10
                factors.append(
                    f"Positive 5-day momentum ({recent_split.momentum_day_5:.1f}%)"
                )
            elif recent_split.momentum_day_5 > -20:
                momentum_factor = -10
                factors.append(
                    f"Weak 5-day momentum ({recent_split.momentum_day_5:.1f}%)"
                )
            else:
                momentum_factor = -20
                factors.append(
                    f"Negative 5-day momentum ({recent_split.momentum_day_5:.1f}%)"
                )

        # Factor 4: Momentum decay pattern
        decay_factor = 0
        if recent_split.momentum_day_1 and recent_split.momentum_day_5:
            if recent_split.momentum_day_5 < recent_split.momentum_day_1:
                decay_factor = -10
                factors.append("Momentum fading")
            else:
                decay_factor = 10
                factors.append("Momentum building")

        # Calculate final SMI
        smi_score = type_factor + momentum_factor + decay_factor

        # Apply recency weight
        smi_score = int(smi_score * (recency_factor / 50))

        # Clamp to -100 to +100
        smi_score = max(-100, min(100, smi_score))

        # Determine signal
        if smi_score >= 30:
            signal = "BULLISH"
            color = "#22c55e"
        elif smi_score >= 10:
            signal = "LEAN_BULLISH"
            color = "#86efac"
        elif smi_score <= -30:
            signal = "BEARISH"
            color = "#ef4444"
        elif smi_score <= -10:
            signal = "LEAN_BEARISH"
            color = "#fca5a5"
        else:
            signal = "NEUTRAL"
            color = "#fbbf24"

        return {
            "symbol": symbol,
            "has_split": True,
            "split_date": recent_split.split_date,
            "split_ratio": recent_split.split_ratio,
            "split_type": recent_split.split_type,
            "days_since_split": days,
            "smi_score": smi_score,
            "smi_signal": signal,
            "smi_color": color,
            "factors": factors,
            "momentum_day_1": recent_split.momentum_day_1,
            "momentum_day_5": recent_split.momentum_day_5,
            "momentum_day_10": recent_split.momentum_day_10,
            "momentum_day_20": recent_split.momentum_day_20,
            "momentum_day_30": recent_split.momentum_day_30,
        }

    def update_momentum_stats(self):
        """Update aggregate momentum statistics"""
        all_splits = []
        for splits in self.splits.values():
            all_splits.extend(splits)

        if not all_splits:
            return

        # Calculate averages
        day_1_values = [
            s.momentum_day_1 for s in all_splits if s.momentum_day_1 is not None
        ]
        day_5_values = [
            s.momentum_day_5 for s in all_splits if s.momentum_day_5 is not None
        ]
        day_10_values = [
            s.momentum_day_10 for s in all_splits if s.momentum_day_10 is not None
        ]
        day_20_values = [
            s.momentum_day_20 for s in all_splits if s.momentum_day_20 is not None
        ]
        day_30_values = [
            s.momentum_day_30 for s in all_splits if s.momentum_day_30 is not None
        ]

        self.stats.total_splits_tracked = len(all_splits)
        self.stats.avg_day_1_momentum = (
            sum(day_1_values) / len(day_1_values) if day_1_values else 0
        )
        self.stats.avg_day_5_momentum = (
            sum(day_5_values) / len(day_5_values) if day_5_values else 0
        )
        self.stats.avg_day_10_momentum = (
            sum(day_10_values) / len(day_10_values) if day_10_values else 0
        )
        self.stats.avg_day_20_momentum = (
            sum(day_20_values) / len(day_20_values) if day_20_values else 0
        )
        self.stats.avg_day_30_momentum = (
            sum(day_30_values) / len(day_30_values) if day_30_values else 0
        )

        # Positive percentages
        self.stats.pct_positive_day_1 = (
            (sum(1 for v in day_1_values if v > 0) / len(day_1_values) * 100)
            if day_1_values
            else 0
        )
        self.stats.pct_positive_day_5 = (
            (sum(1 for v in day_5_values if v > 0) / len(day_5_values) * 100)
            if day_5_values
            else 0
        )

        # Fade percentage
        fades = sum(
            1
            for s in all_splits
            if s.momentum_day_1
            and s.momentum_day_5
            and s.momentum_day_5 < s.momentum_day_1
        )
        total_with_both = sum(
            1 for s in all_splits if s.momentum_day_1 and s.momentum_day_5
        )
        self.stats.pct_fade_after_day_5 = (
            (fades / total_with_both * 100) if total_with_both else 0
        )

        # By split type
        reverse_splits = [s for s in all_splits if s.split_type == "REVERSE"]
        forward_splits = [s for s in all_splits if s.split_type == "FORWARD"]

        reverse_momentum = [
            s.momentum_day_5 for s in reverse_splits if s.momentum_day_5 is not None
        ]
        forward_momentum = [
            s.momentum_day_5 for s in forward_splits if s.momentum_day_5 is not None
        ]

        self.stats.reverse_split_avg_momentum = (
            sum(reverse_momentum) / len(reverse_momentum) if reverse_momentum else 0
        )
        self.stats.forward_split_avg_momentum = (
            sum(forward_momentum) / len(forward_momentum) if forward_momentum else 0
        )

        self._save_data()

    def get_stats(self) -> Dict:
        """Get aggregate statistics"""
        return asdict(self.stats)


# Singleton
_split_tracker: Optional[SplitTracker] = None


def get_split_tracker() -> SplitTracker:
    """Get or create split tracker singleton"""
    global _split_tracker
    if _split_tracker is None:
        _split_tracker = SplitTracker()
    return _split_tracker


# Quick lookup function
def check_split_momentum(symbol: str) -> Dict:
    """Quick check of split momentum index for a symbol"""
    tracker = get_split_tracker()
    return tracker.get_split_momentum_index(symbol)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test
    tracker = get_split_tracker()

    # Check AZI (recent reverse split)
    print("\nChecking AZI...")
    result = tracker.get_split_momentum_index("AZI")
    print(json.dumps(result, indent=2))

    # Check a few more
    for sym in ["YCBD", "ADTX", "AKAN"]:
        print(f"\nChecking {sym}...")
        result = tracker.get_split_momentum_index(sym)
        print(f"  SMI Score: {result['smi_score']}")
        print(f"  Signal: {result['smi_signal']}")
