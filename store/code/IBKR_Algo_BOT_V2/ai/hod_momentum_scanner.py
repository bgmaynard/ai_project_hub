"""
High of Day (HOD) Momentum Scanner
==================================
Real-time detection of stocks making new intraday highs.
Replicates Day Trade Dash "Small Cap - High of Day Momentum" scanner.

Alerts when:
1. Stock makes new HOD
2. Breaks key price level ($1, $2, $5, $10, etc.)
3. Multiple HOD breaks in short time = strong momentum
4. Volume surge accompanies the break

Strategy Classifications (like DTD):
- "Squeeze Alert - Up 5% in 5min"
- "Squeeze Alert - Up 10% in 10min"
- "Medium Float - High Rel Vol - Price under $20"
- "Former Momo Stock"
- "Running Up Alerts"
"""

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class HODAlert:
    """Single HOD alert event"""

    symbol: str
    timestamp: datetime
    price: float
    previous_hod: float
    change_pct: float
    volume: int
    float_shares: float
    relative_volume: float
    gap_pct: float
    short_interest: float
    strategy_name: str
    alert_count_5min: int = 0  # How many alerts in last 5 min
    grade: str = "C"  # A, B, or C based on Ross's 5 criteria
    criteria_met: int = 0  # How many of Ross's 5 criteria met
    has_news: bool = False  # News catalyst present


@dataclass
class SymbolTracker:
    """Tracks a single symbol's intraday data"""

    symbol: str
    open_price: float = 0.0
    high_of_day: float = 0.0
    low_of_day: float = 999999.0
    current_price: float = 0.0
    previous_close: float = 0.0
    volume: int = 0
    avg_volume: int = 0  # 10-day average
    float_shares: float = 0.0
    short_interest: float = 0.0

    # HOD tracking
    hod_breaks: List[datetime] = field(default_factory=list)
    last_hod_alert: Optional[datetime] = None

    # Price history for velocity calculation
    price_history: deque = field(
        default_factory=lambda: deque(maxlen=60)
    )  # Last 60 ticks

    def update_price(self, price: float, volume: int, timestamp: datetime = None):
        """Update with new price tick"""
        if timestamp is None:
            timestamp = datetime.now()

        self.current_price = price
        self.volume = volume

        # Track price history
        self.price_history.append({"price": price, "time": timestamp, "volume": volume})

        # Update low
        if price < self.low_of_day:
            self.low_of_day = price

        # Check for new HOD
        made_new_hod = False
        if price > self.high_of_day:
            self.high_of_day = price
            self.hod_breaks.append(timestamp)
            made_new_hod = True

        return made_new_hod

    @property
    def change_pct(self) -> float:
        """Calculate change from previous close"""
        if self.previous_close > 0:
            return (
                (self.current_price - self.previous_close) / self.previous_close
            ) * 100
        return 0.0

    @property
    def gap_pct(self) -> float:
        """Calculate gap from previous close to open"""
        if self.previous_close > 0 and self.open_price > 0:
            return ((self.open_price - self.previous_close) / self.previous_close) * 100
        return 0.0

    @property
    def relative_volume(self) -> float:
        """Calculate relative volume vs average"""
        if self.avg_volume > 0:
            return self.volume / self.avg_volume
        return 0.0

    def hod_count_in_window(self, minutes: int = 5) -> int:
        """Count HOD breaks in last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return sum(1 for t in self.hod_breaks if t > cutoff)

    def price_change_in_window(self, minutes: int = 5) -> float:
        """Calculate price change in last N minutes"""
        if len(self.price_history) < 2:
            return 0.0

        cutoff = datetime.now() - timedelta(minutes=minutes)

        # Find oldest price in window
        oldest_price = None
        for entry in self.price_history:
            if entry["time"] >= cutoff:
                oldest_price = entry["price"]
                break

        if oldest_price and oldest_price > 0:
            return ((self.current_price - oldest_price) / oldest_price) * 100
        return 0.0


class HODMomentumScanner:
    """
    Real-time HOD momentum scanner.
    Tracks watchlist symbols and alerts on new HOD breaks.
    """

    def __init__(self):
        self.trackers: Dict[str, SymbolTracker] = {}
        self.alerts: List[HODAlert] = []
        self.alert_callbacks: List[callable] = []
        self.is_running: bool = False
        self.scan_interval: float = 1.0  # Check every 1 second

        # Alert cooldown to prevent spam
        self.alert_cooldown_seconds: int = 30

        # Ross Cameron's 5 Criteria (from Warrior Trading PDF)
        # 1. 5x Relative Volume (vs 30-day average)
        # 2. Already up 10% on the day
        # 3. News Event catalyst (checked separately)
        # 4. Price $1.00 - $20.00
        # 5. Float < 10 million shares

        self.min_price: float = 1.0
        self.max_price: float = 20.0
        self.min_change_pct: float = 10.0  # Ross: "at least 10% up on day"
        self.min_relative_volume: float = 5.0  # Ross: "5x Relative Volume"
        self.max_float: float = 10_000_000  # Ross: "less than 10 million shares"

        # Strategy thresholds
        self.squeeze_5min_threshold: float = 5.0  # Up 5% in 5 min
        self.squeeze_10min_threshold: float = 10.0  # Up 10% in 10 min
        self.medium_float_max: float = 20_000_000  # Adjusted for Ross's criteria
        self.low_float_max: float = 10_000_000  # Ross's sweet spot

        logger.info("HOD Momentum Scanner initialized")

    def add_symbol(self, symbol: str, data: dict = None):
        """Add symbol to track"""
        if symbol not in self.trackers:
            tracker = SymbolTracker(symbol=symbol)

            # Initialize with provided data
            if data:
                tracker.previous_close = data.get(
                    "previous_close", data.get("close", 0)
                )
                tracker.open_price = data.get("open", 0)
                tracker.high_of_day = data.get("high", 0)
                tracker.low_of_day = data.get("low", 999999)
                tracker.current_price = data.get("price", 0)
                tracker.volume = data.get("volume", 0)
                tracker.avg_volume = data.get("avg_volume", 0)
                tracker.float_shares = data.get("float_shares", 0)
                tracker.short_interest = data.get("short_interest", 0)

            self.trackers[symbol] = tracker
            logger.debug(f"Added {symbol} to HOD tracker")

    def remove_symbol(self, symbol: str):
        """Remove symbol from tracking"""
        if symbol in self.trackers:
            del self.trackers[symbol]
            logger.debug(f"Removed {symbol} from HOD tracker")

    def update_price(
        self, symbol: str, price: float, volume: int = 0, data: dict = None
    ):
        """
        Update price for a symbol and check for HOD break.
        Returns HODAlert if new HOD detected, None otherwise.
        """
        if symbol not in self.trackers:
            self.add_symbol(symbol, data)

        tracker = self.trackers[symbol]

        # Update additional data if provided
        if data:
            if "avg_volume" in data:
                tracker.avg_volume = data["avg_volume"]
            if "float_shares" in data:
                tracker.float_shares = data["float_shares"]
            if "short_interest" in data:
                tracker.short_interest = data["short_interest"]
            if "close" in data and tracker.previous_close == 0:
                tracker.previous_close = data["close"]
            if "open" in data and tracker.open_price == 0:
                tracker.open_price = data["open"]

        previous_hod = tracker.high_of_day
        made_new_hod = tracker.update_price(price, volume)

        if made_new_hod:
            return self._evaluate_hod_alert(tracker, previous_hod)

        return None

    def _grade_stock(self, tracker: SymbolTracker, has_news: bool = False) -> tuple:
        """
        Grade stock based on Ross Cameron's 5 Criteria.
        Returns (grade, criteria_met, criteria_details)

        Ross's 5 Pillars:
        1. 5x Relative Volume (vs 30-day average)
        2. Already up 10% on the day
        3. News Event catalyst
        4. Price $1.00 - $20.00
        5. Float < 10 million shares

        Grading:
        - A = 5/5 criteria → Full position
        - B = 4/5 criteria → Half position
        - C = 3/5 or less → Quick scalp only
        """
        criteria_met = 0
        details = {}

        # 1. Relative Volume >= 5x
        if tracker.relative_volume >= 5.0:
            criteria_met += 1
            details["rvol"] = f"✓ RVol {tracker.relative_volume:.1f}x >= 5x"
        else:
            details["rvol"] = f"✗ RVol {tracker.relative_volume:.1f}x < 5x"

        # 2. Already up 10%
        if tracker.change_pct >= 10.0:
            criteria_met += 1
            details["change"] = f"✓ Up {tracker.change_pct:.1f}% >= 10%"
        else:
            details["change"] = f"✗ Up {tracker.change_pct:.1f}% < 10%"

        # 3. News catalyst
        if has_news:
            criteria_met += 1
            details["news"] = "✓ Has news catalyst"
        else:
            details["news"] = "✗ No news catalyst"

        # 4. Price $1-$20
        if 1.0 <= tracker.current_price <= 20.0:
            criteria_met += 1
            details["price"] = f"✓ Price ${tracker.current_price:.2f} in range"
        else:
            details["price"] = f"✗ Price ${tracker.current_price:.2f} out of range"

        # 5. Float < 10M
        if tracker.float_shares > 0 and tracker.float_shares < 10_000_000:
            criteria_met += 1
            details["float"] = f"✓ Float {tracker.float_shares/1_000_000:.1f}M < 10M"
        elif tracker.float_shares == 0:
            details["float"] = "? Float unknown"
        else:
            details["float"] = f"✗ Float {tracker.float_shares/1_000_000:.1f}M >= 10M"

        # Determine grade
        if criteria_met >= 5:
            grade = "A"
        elif criteria_met == 4:
            grade = "B"
        else:
            grade = "C"

        return grade, criteria_met, details

    def _evaluate_hod_alert(
        self, tracker: SymbolTracker, previous_hod: float, has_news: bool = False
    ) -> Optional[HODAlert]:
        """Evaluate if HOD break should trigger an alert"""

        # Check cooldown
        if tracker.last_hod_alert:
            elapsed = (datetime.now() - tracker.last_hod_alert).total_seconds()
            if elapsed < self.alert_cooldown_seconds:
                return None

        # Grade the stock first (we'll alert even if not all criteria met, but with grade)
        grade, criteria_met, criteria_details = self._grade_stock(tracker, has_news)

        # For HOD alerts, we use relaxed filters but show the grade
        # Minimum filters to avoid complete noise
        if tracker.current_price < 0.50:  # At least 50 cents
            return None
        if tracker.current_price > 50.0:  # Under $50
            return None
        if tracker.change_pct < 5.0:  # At least 5% move to be interesting
            return None

        # Determine strategy name
        strategy = self._classify_strategy(tracker)

        # Create alert with grade
        alert = HODAlert(
            symbol=tracker.symbol,
            timestamp=datetime.now(),
            price=tracker.current_price,
            previous_hod=previous_hod,
            change_pct=tracker.change_pct,
            volume=tracker.volume,
            float_shares=tracker.float_shares,
            relative_volume=tracker.relative_volume,
            gap_pct=tracker.gap_pct,
            short_interest=tracker.short_interest,
            strategy_name=strategy,
            alert_count_5min=tracker.hod_count_in_window(5),
            grade=grade,
            criteria_met=criteria_met,
            has_news=has_news,
        )

        # Update last alert time
        tracker.last_hod_alert = datetime.now()

        # Store alert
        self.alerts.append(alert)

        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

        # Fire callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        logger.info(
            f"🚀 HOD ALERT [{alert.grade}]: {alert.symbol} ${alert.price:.2f} ({alert.change_pct:+.1f}%) - {strategy} ({criteria_met}/5 criteria)"
        )

        return alert

    def _classify_strategy(self, tracker: SymbolTracker) -> str:
        """Classify the alert into strategy type like DTD"""

        # Check 5-min squeeze
        change_5min = tracker.price_change_in_window(5)
        if change_5min >= self.squeeze_5min_threshold:
            return "Squeeze Alert - Up 5% in 5min"

        # Check 10-min squeeze
        change_10min = tracker.price_change_in_window(10)
        if change_10min >= self.squeeze_10min_threshold:
            return "Squeeze Alert - Up 10% in 10min"

        # Check float and volume conditions
        is_low_float = (
            tracker.float_shares > 0 and tracker.float_shares < self.low_float_max
        )
        is_medium_float = (
            tracker.float_shares > 0 and tracker.float_shares < self.medium_float_max
        )
        is_high_rvol = tracker.relative_volume >= 2.0
        is_under_20 = tracker.current_price < 20

        if is_medium_float and is_high_rvol and is_under_20:
            return "Medium Float - High Rel Vol - Price under $20"

        if is_low_float and is_high_rvol:
            return "Low Float - High Rel Vol"

        # Multiple HOD breaks = running
        if tracker.hod_count_in_window(5) >= 3:
            return "Running Up Alerts"

        return "HOD Break"

    def check_key_level_break(self, symbol: str, price: float) -> Optional[str]:
        """Check if price broke a key psychological level"""
        key_levels = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0]

        if symbol not in self.trackers:
            return None

        tracker = self.trackers[symbol]

        for level in key_levels:
            # Check if we crossed above the level
            if tracker.price_history:
                prev_price = (
                    tracker.price_history[-1]["price"]
                    if len(tracker.price_history) > 1
                    else 0
                )
                if prev_price < level <= price:
                    return f"Broke ${level:.0f} level"

        return None

    def get_recent_alerts(self, minutes: int = 30) -> List[HODAlert]:
        """Get alerts from last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [a for a in self.alerts if a.timestamp > cutoff]

    def get_running_stocks(self) -> List[str]:
        """Get symbols that are actively running (multiple HOD breaks)"""
        running = []
        for symbol, tracker in self.trackers.items():
            if tracker.hod_count_in_window(5) >= 2:
                running.append(symbol)
        return running

    def get_status(self) -> dict:
        """Get scanner status"""
        return {
            "is_running": self.is_running,
            "tracking_count": len(self.trackers),
            "symbols": list(self.trackers.keys()),
            "recent_alerts": len(self.get_recent_alerts(30)),
            "running_stocks": self.get_running_stocks(),
            "filters": {
                "min_price": self.min_price,
                "max_price": self.max_price,
                "min_change_pct": self.min_change_pct,
                "min_relative_volume": self.min_relative_volume,
                "max_float": self.max_float,
            },
        }

    def on_alert(self, callback: callable):
        """Register callback for alerts"""
        self.alert_callbacks.append(callback)

    def to_dict(self) -> dict:
        """Export alerts to dict"""
        return {
            "alerts": [
                {
                    "symbol": a.symbol,
                    "timestamp": a.timestamp.isoformat(),
                    "price": a.price,
                    "change_pct": a.change_pct,
                    "volume": a.volume,
                    "float_shares": a.float_shares,
                    "relative_volume": a.relative_volume,
                    "gap_pct": a.gap_pct,
                    "strategy": a.strategy_name,
                    "alert_count_5min": a.alert_count_5min,
                    "grade": a.grade,
                    "criteria_met": a.criteria_met,
                    "has_news": a.has_news,
                }
                for a in self.alerts[-50:]
            ],
            "running_stocks": self.get_running_stocks(),
            "a_grade_stocks": self.get_a_grade_stocks(),
            "b_grade_stocks": self.get_b_grade_stocks(),
        }

    def get_a_grade_stocks(self) -> List[str]:
        """Get A-grade stocks from recent alerts"""
        seen = set()
        a_stocks = []
        for alert in reversed(self.alerts):
            if alert.grade == "A" and alert.symbol not in seen:
                a_stocks.append(alert.symbol)
                seen.add(alert.symbol)
        return a_stocks

    def get_b_grade_stocks(self) -> List[str]:
        """Get B-grade stocks from recent alerts"""
        seen = set()
        b_stocks = []
        for alert in reversed(self.alerts):
            if alert.grade == "B" and alert.symbol not in seen:
                b_stocks.append(alert.symbol)
                seen.add(alert.symbol)
        return b_stocks

    def grade_symbol(self, symbol: str, has_news: bool = False) -> dict:
        """Grade a symbol based on Ross Cameron's 5 criteria"""
        if symbol not in self.trackers:
            return {"error": f"{symbol} not being tracked"}

        tracker = self.trackers[symbol]
        grade, criteria_met, details = self._grade_stock(tracker, has_news)

        return {
            "symbol": symbol,
            "grade": grade,
            "criteria_met": criteria_met,
            "criteria_details": details,
            "recommendation": (
                "FULL_POSITION"
                if grade == "A"
                else "HALF_POSITION" if grade == "B" else "SCALP_ONLY"
            ),
            "current_price": tracker.current_price,
            "change_pct": tracker.change_pct,
            "relative_volume": tracker.relative_volume,
            "float_shares": tracker.float_shares,
        }


# Singleton instance
_hod_scanner: Optional[HODMomentumScanner] = None


def get_hod_scanner() -> HODMomentumScanner:
    """Get or create HOD scanner instance"""
    global _hod_scanner
    if _hod_scanner is None:
        _hod_scanner = HODMomentumScanner()
    return _hod_scanner
