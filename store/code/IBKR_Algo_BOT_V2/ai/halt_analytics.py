"""
Halt Analytics & Strategy Builder
=================================
Tracks, analyzes, and builds strategies for trading halt resumptions.

Key metrics tracked:
- Pre-halt momentum (% move before halt)
- Halt duration
- Resume gap (halt price vs resume price)
- Post-resume action (continuation or fade)
- Volume patterns

Strategy development:
- Historical win rate by halt type
- Optimal entry timing post-resume
- Position sizing based on confidence
"""

import json
import logging
import os
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Data file for persistent storage
DATA_DIR = Path(__file__).parent.parent / "data"
HALT_DATA_FILE = DATA_DIR / "halt_history.json"


@dataclass
class HaltRecord:
    """Complete record of a halt event"""

    # Identification
    symbol: str
    halt_id: str
    date: str

    # Halt info
    halt_time: str
    halt_price: float
    halt_type: str  # LULD_UP, LULD_DOWN, NEWS, CIRCUIT_BREAKER

    # Pre-halt context
    pre_halt_price_5min: float  # Price 5 min before halt
    pre_halt_momentum: float  # % change leading into halt
    pre_halt_volume: int
    was_running_up: bool

    # Resume info
    resume_time: str
    resume_price: float
    halt_duration_seconds: int
    resume_gap_percent: float  # (resume - halt) / halt * 100
    resume_direction: str  # BULLISH, BEARISH, NEUTRAL

    # Post-resume action (1 min after resume)
    post_1min_price: float = 0.0
    post_1min_change: float = 0.0

    # Post-resume action (5 min after resume)
    post_5min_price: float = 0.0
    post_5min_change: float = 0.0

    # Outcome
    continuation: bool = False  # Did it continue in resume direction?
    fade: bool = False  # Did it reverse?
    outcome_notes: str = ""

    # Trade result (if traded)
    traded: bool = False
    entry_price: float = 0.0
    exit_price: float = 0.0
    trade_pnl: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class HaltStrategy:
    """Strategy rules for trading halt resumptions"""

    name: str
    description: str

    # Entry rules
    min_resume_gap: float = 2.0  # Minimum gap % to trade
    max_resume_gap: float = 20.0  # Max gap (too extended)
    required_direction: str = "BULLISH"  # BULLISH, BEARISH, ANY
    min_pre_halt_momentum: float = 5.0  # Stock was running before halt

    # Timing
    entry_delay_seconds: int = 5  # Wait X seconds after resume
    max_entry_window_seconds: int = 30  # Must enter within X seconds

    # Risk management
    stop_loss_percent: float = 3.0
    take_profit_percent: float = 6.0
    max_position_size: float = 1000.0  # Dollar amount

    # Confidence
    min_confidence: float = 0.6
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class HaltAnalytics:
    """
    Analyzes halt data to build and refine trading strategies.
    """

    def __init__(self):
        self.halt_history: List[HaltRecord] = []
        self.strategies: Dict[str, HaltStrategy] = {}
        self._load_data()
        self._init_default_strategies()

        logger.info(
            f"HaltAnalytics initialized with {len(self.halt_history)} historical records"
        )

    def _load_data(self):
        """Load historical halt data"""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)

            if HALT_DATA_FILE.exists():
                with open(HALT_DATA_FILE, "r") as f:
                    data = json.load(f)
                    self.halt_history = [
                        HaltRecord(**record) for record in data.get("halts", [])
                    ]
                    logger.info(f"Loaded {len(self.halt_history)} halt records")
        except Exception as e:
            logger.error(f"Error loading halt data: {e}")
            self.halt_history = []

    def _save_data(self):
        """Save halt data to file"""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)

            data = {
                "halts": [h.to_dict() for h in self.halt_history],
                "last_updated": datetime.now().isoformat(),
            }

            with open(HALT_DATA_FILE, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.halt_history)} halt records")
        except Exception as e:
            logger.error(f"Error saving halt data: {e}")

    def _init_default_strategies(self):
        """Initialize default halt strategies"""
        self.strategies = {
            "bullish_continuation": HaltStrategy(
                name="Bullish Continuation",
                description="Buy on bullish halt resume, expecting continuation",
                required_direction="BULLISH",
                min_resume_gap=2.0,
                max_resume_gap=15.0,
                min_pre_halt_momentum=5.0,
                entry_delay_seconds=3,
                stop_loss_percent=3.0,
                take_profit_percent=6.0,
            ),
            "bearish_fade": HaltStrategy(
                name="Bearish Fade",
                description="Short on bearish halt resume",
                required_direction="BEARISH",
                min_resume_gap=-15.0,
                max_resume_gap=-2.0,
                min_pre_halt_momentum=-5.0,
                entry_delay_seconds=5,
                stop_loss_percent=3.0,
                take_profit_percent=6.0,
            ),
            "gap_reversal": HaltStrategy(
                name="Gap Reversal",
                description="Fade extreme gaps expecting mean reversion",
                required_direction="ANY",
                min_resume_gap=10.0,  # Large gap
                max_resume_gap=30.0,
                entry_delay_seconds=10,  # Wait for reversal signal
                stop_loss_percent=5.0,
                take_profit_percent=8.0,
            ),
        }

    def record_halt(
        self,
        symbol: str,
        halt_price: float,
        halt_time: datetime,
        halt_type: str,
        pre_halt_price_5min: float,
        pre_halt_volume: int,
    ) -> HaltRecord:
        """Record a new halt event"""

        # Calculate pre-halt momentum
        pre_halt_momentum = (
            (halt_price - pre_halt_price_5min) / pre_halt_price_5min * 100
            if pre_halt_price_5min > 0
            else 0
        )

        record = HaltRecord(
            symbol=symbol,
            halt_id=f"{symbol}_{halt_time.strftime('%Y%m%d_%H%M%S')}",
            date=halt_time.strftime("%Y-%m-%d"),
            halt_time=halt_time.isoformat(),
            halt_price=halt_price,
            halt_type=halt_type,
            pre_halt_price_5min=pre_halt_price_5min,
            pre_halt_momentum=round(pre_halt_momentum, 2),
            pre_halt_volume=pre_halt_volume,
            was_running_up=pre_halt_momentum > 0,
            resume_time="",
            resume_price=0.0,
            halt_duration_seconds=0,
            resume_gap_percent=0.0,
            resume_direction="",
        )

        self.halt_history.append(record)
        self._save_data()

        logger.info(
            f"Recorded halt: {symbol} at ${halt_price:.2f}, momentum: {pre_halt_momentum:+.1f}%"
        )

        return record

    def record_resume(
        self, halt_id: str, resume_price: float, resume_time: datetime
    ) -> Optional[HaltRecord]:
        """Record halt resume and calculate metrics"""

        # Find the halt record
        record = None
        for h in self.halt_history:
            if h.halt_id == halt_id:
                record = h
                break

        if not record:
            logger.warning(f"Halt record not found: {halt_id}")
            return None

        # Calculate resume metrics
        record.resume_time = resume_time.isoformat()
        record.resume_price = resume_price

        halt_time = datetime.fromisoformat(record.halt_time)
        record.halt_duration_seconds = int((resume_time - halt_time).total_seconds())

        record.resume_gap_percent = round(
            (
                (resume_price - record.halt_price) / record.halt_price * 100
                if record.halt_price > 0
                else 0
            ),
            2,
        )

        # Determine direction
        if record.resume_gap_percent > 2:
            record.resume_direction = "BULLISH"
        elif record.resume_gap_percent < -2:
            record.resume_direction = "BEARISH"
        else:
            record.resume_direction = "NEUTRAL"

        self._save_data()

        logger.info(
            f"Recorded resume: {record.symbol} - "
            f"Gap: {record.resume_gap_percent:+.1f}% - "
            f"Direction: {record.resume_direction} - "
            f"Duration: {record.halt_duration_seconds}s"
        )

        return record

    def record_post_resume(self, halt_id: str, price_1min: float, price_5min: float):
        """Record post-resume price action"""

        record = None
        for h in self.halt_history:
            if h.halt_id == halt_id:
                record = h
                break

        if not record:
            return

        record.post_1min_price = price_1min
        record.post_1min_change = round(
            (
                (price_1min - record.resume_price) / record.resume_price * 100
                if record.resume_price > 0
                else 0
            ),
            2,
        )

        record.post_5min_price = price_5min
        record.post_5min_change = round(
            (
                (price_5min - record.resume_price) / record.resume_price * 100
                if record.resume_price > 0
                else 0
            ),
            2,
        )

        # Determine outcome
        if record.resume_direction == "BULLISH":
            record.continuation = record.post_5min_change > 0
            record.fade = record.post_5min_change < -2
        elif record.resume_direction == "BEARISH":
            record.continuation = record.post_5min_change < 0
            record.fade = record.post_5min_change > 2
        else:
            record.continuation = False
            record.fade = False

        self._save_data()

    def analyze_patterns(self) -> Dict:
        """Analyze halt patterns from historical data"""

        if len(self.halt_history) < 5:
            return {
                "status": "insufficient_data",
                "message": f"Need more data. Currently have {len(self.halt_history)} records.",
                "min_required": 5,
            }

        # Filter completed records (have resume data)
        completed = [h for h in self.halt_history if h.resume_time]

        if len(completed) < 3:
            return {
                "status": "insufficient_completed",
                "message": f"Need more completed halts. Have {len(completed)} of {len(self.halt_history)}.",
            }

        # Basic stats
        bullish_resumes = [h for h in completed if h.resume_direction == "BULLISH"]
        bearish_resumes = [h for h in completed if h.resume_direction == "BEARISH"]

        # Continuation rates
        bullish_cont = [h for h in bullish_resumes if h.continuation]
        bearish_cont = [h for h in bearish_resumes if h.continuation]

        # Gap analysis
        gaps = [h.resume_gap_percent for h in completed]
        durations = [h.halt_duration_seconds for h in completed]

        analysis = {
            "status": "success",
            "total_halts": len(self.halt_history),
            "completed_halts": len(completed),
            "direction_breakdown": {
                "bullish": len(bullish_resumes),
                "bearish": len(bearish_resumes),
                "neutral": len(completed) - len(bullish_resumes) - len(bearish_resumes),
            },
            "continuation_rates": {
                "bullish": (
                    round(len(bullish_cont) / len(bullish_resumes) * 100, 1)
                    if bullish_resumes
                    else 0
                ),
                "bearish": (
                    round(len(bearish_cont) / len(bearish_resumes) * 100, 1)
                    if bearish_resumes
                    else 0
                ),
            },
            "gap_stats": {
                "avg_gap": round(statistics.mean(gaps), 2) if gaps else 0,
                "max_gap": round(max(gaps), 2) if gaps else 0,
                "min_gap": round(min(gaps), 2) if gaps else 0,
            },
            "duration_stats": {
                "avg_duration_sec": (
                    round(statistics.mean(durations), 0) if durations else 0
                ),
                "max_duration_sec": max(durations) if durations else 0,
                "min_duration_sec": min(durations) if durations else 0,
            },
            "recommendations": self._generate_recommendations(completed),
        }

        return analysis

    def _generate_recommendations(self, completed: List[HaltRecord]) -> List[str]:
        """Generate strategy recommendations based on data"""
        recommendations = []

        if not completed:
            return ["Collect more halt data to generate recommendations"]

        bullish = [h for h in completed if h.resume_direction == "BULLISH"]
        bullish_cont = [h for h in bullish if h.continuation]

        if bullish:
            cont_rate = len(bullish_cont) / len(bullish) * 100
            if cont_rate > 60:
                recommendations.append(
                    f"BULLISH halts show {cont_rate:.0f}% continuation - favor buying resumptions"
                )
            elif cont_rate < 40:
                recommendations.append(
                    f"BULLISH halts only {cont_rate:.0f}% continuation - consider fading"
                )

        # Gap analysis
        high_gap = [h for h in completed if h.resume_gap_percent > 10]
        if high_gap:
            high_gap_fade = [h for h in high_gap if h.fade]
            fade_rate = len(high_gap_fade) / len(high_gap) * 100 if high_gap else 0
            if fade_rate > 50:
                recommendations.append(
                    f"Large gaps (>10%) fade {fade_rate:.0f}% of time - consider mean reversion"
                )

        if not recommendations:
            recommendations.append("Continue collecting data for pattern refinement")

        return recommendations

    def get_strategy_signal(self, halt_record: HaltRecord) -> Dict:
        """Get trading signal based on strategies"""

        if not halt_record.resume_time:
            return {"action": "WAIT", "reason": "Halt not yet resumed"}

        signals = []

        for name, strategy in self.strategies.items():
            confidence = self._calculate_confidence(halt_record, strategy)

            if confidence >= strategy.min_confidence:
                signals.append(
                    {
                        "strategy": name,
                        "action": (
                            "BUY"
                            if strategy.required_direction == "BULLISH"
                            else "SELL"
                        ),
                        "confidence": round(confidence, 2),
                        "entry_delay": strategy.entry_delay_seconds,
                        "stop_loss": strategy.stop_loss_percent,
                        "take_profit": strategy.take_profit_percent,
                    }
                )

        if not signals:
            return {
                "action": "SKIP",
                "reason": "No strategy meets confidence threshold",
            }

        # Return highest confidence signal
        best = max(signals, key=lambda x: x["confidence"])
        return best

    def _calculate_confidence(
        self, record: HaltRecord, strategy: HaltStrategy
    ) -> float:
        """Calculate confidence for a strategy"""
        confidence = 0.5  # Base

        # Direction match
        if strategy.required_direction == "ANY":
            confidence += 0.1
        elif strategy.required_direction == record.resume_direction:
            confidence += 0.2
        else:
            return 0.0  # Wrong direction

        # Gap in range
        gap = record.resume_gap_percent
        if strategy.required_direction == "BULLISH":
            if strategy.min_resume_gap <= gap <= strategy.max_resume_gap:
                confidence += 0.2
        elif strategy.required_direction == "BEARISH":
            if strategy.max_resume_gap <= gap <= strategy.min_resume_gap:
                confidence += 0.2

        # Pre-halt momentum
        if record.was_running_up and strategy.required_direction == "BULLISH":
            confidence += 0.1
        elif not record.was_running_up and strategy.required_direction == "BEARISH":
            confidence += 0.1

        # Historical win rate boost
        if strategy.win_rate > 0.5:
            confidence += (strategy.win_rate - 0.5) * 0.2

        return min(confidence, 1.0)

    def get_history(self, limit: int = 50, symbol: str = None) -> List[Dict]:
        """Get halt history"""
        history = self.halt_history

        if symbol:
            history = [h for h in history if h.symbol == symbol]

        return [h.to_dict() for h in history[-limit:]]

    def get_stats(self) -> Dict:
        """Get overall halt statistics"""
        return {
            "total_records": len(self.halt_history),
            "symbols_tracked": len(set(h.symbol for h in self.halt_history)),
            "strategies_available": list(self.strategies.keys()),
            "data_file": str(HALT_DATA_FILE),
        }


# Singleton
_halt_analytics: Optional[HaltAnalytics] = None


def get_halt_analytics() -> HaltAnalytics:
    """Get or create halt analytics singleton"""
    global _halt_analytics
    if _halt_analytics is None:
        _halt_analytics = HaltAnalytics()
    return _halt_analytics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    analytics = get_halt_analytics()

    # Example: Record a halt
    record = analytics.record_halt(
        symbol="TEST",
        halt_price=10.50,
        halt_time=datetime.now(),
        halt_type="LULD_UP",
        pre_halt_price_5min=9.80,
        pre_halt_volume=500000,
    )

    print(f"Recorded: {record.halt_id}")

    # Example: Record resume
    analytics.record_resume(
        halt_id=record.halt_id,
        resume_price=11.20,
        resume_time=datetime.now() + timedelta(minutes=5),
    )

    # Get analysis
    analysis = analytics.analyze_patterns()
    print(f"\nAnalysis: {json.dumps(analysis, indent=2)}")
