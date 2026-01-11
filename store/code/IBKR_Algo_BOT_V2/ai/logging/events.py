"""
Trading Event Logger
====================
Structured event logging for strategy attempts and trade results.

All events are logged to both:
1. Python logger (for console/file output)
2. JSON file (for analysis and audit)
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


@dataclass
class SniperAttemptEvent:
    """
    Event structure for ATS + 9 EMA Sniper attempts.

    From build doc:
    {
      "event": "ATS_9EMA_SNIPER_ATTEMPT",
      "symbol": "XYZ",
      "ats_score": 0.82,
      "hrdc_mode": "MOMENTUM_ALLOWED",
      "pullback_depth_pct": 18.3,
      "entry_reason": "RECLAIM_PULLBACK_HIGH",
      "result": "WIN | LOSS | NO_TRADE"
    }
    """
    event: str = "ATS_9EMA_SNIPER_ATTEMPT"
    symbol: str = ""
    timestamp: str = ""

    # ATS qualification
    ats_score: float = 0.0
    ats_qualified: bool = False

    # HRDC context
    hrdc_mode: str = ""

    # Pullback data
    pullback_depth_pct: float = 0.0
    volume_decreasing: bool = False
    ema9_distance_pct: float = 0.0

    # Entry data
    entry_price: float = 0.0
    entry_reason: str = ""  # RECLAIM_PULLBACK_HIGH, BULLISH_CANDLE_OFF_EMA, etc.
    stop_loss: float = 0.0
    target: float = 0.0

    # Exit data (if traded)
    exit_price: float = 0.0
    exit_reason: str = ""

    # Result
    result: str = ""  # WIN, LOSS, NO_TRADE
    pnl: float = 0.0

    # Context
    market_session: str = ""
    vwap_status: str = ""  # ABOVE, BELOW, AT
    rvol: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class TradingEventLogger:
    """
    Logs trading events to JSON file for analysis.
    """

    def __init__(self, log_dir: str = "logs/trading_events"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Daily log file
        self._current_date: Optional[str] = None
        self._log_file: Optional[Path] = None
        self._events: List[Dict] = []

    def _get_log_file(self) -> Path:
        """Get current day's log file"""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._current_date:
            self._current_date = today
            self._log_file = self.log_dir / f"sniper_events_{today}.json"
            self._events = []

            # Load existing events if file exists
            if self._log_file.exists():
                try:
                    with open(self._log_file, 'r') as f:
                        self._events = json.load(f)
                except Exception:
                    self._events = []

        return self._log_file

    def log_event(self, event: Dict[str, Any]):
        """Log a trading event"""
        # Add timestamp if not present
        if "timestamp" not in event or not event["timestamp"]:
            event["timestamp"] = datetime.now().isoformat()

        # Add to in-memory list
        self._events.append(event)

        # Write to file
        log_file = self._get_log_file()
        try:
            with open(log_file, 'w') as f:
                json.dump(self._events, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write event log: {e}")

        # Also log to standard logger
        event_type = event.get("event", "UNKNOWN")
        symbol = event.get("symbol", "")
        result = event.get("result", "")

        logger.info(
            f"[TRADE_EVENT] {event_type} | {symbol} | {result} | "
            f"ATS: {event.get('ats_score', 0):.2f} | "
            f"Pullback: {event.get('pullback_depth_pct', 0):.1f}%"
        )

    def log_sniper_attempt(self,
                           symbol: str,
                           ats_score: float,
                           hrdc_mode: str,
                           result: str,
                           pullback_depth_pct: float = 0.0,
                           entry_reason: str = "",
                           entry_price: float = 0.0,
                           exit_price: float = 0.0,
                           pnl: float = 0.0,
                           **kwargs) -> SniperAttemptEvent:
        """
        Log a sniper strategy attempt.

        Args:
            symbol: Stock symbol
            ats_score: ATS qualification score
            hrdc_mode: HRDC mode (MOMENTUM_ALLOWED, FLAT_ONLY, DEFENSIVE)
            result: WIN, LOSS, or NO_TRADE
            pullback_depth_pct: Depth of pullback as %
            entry_reason: Why entry was triggered
            entry_price: Entry price (0 if NO_TRADE)
            exit_price: Exit price (0 if NO_TRADE)
            pnl: Profit/loss
            **kwargs: Additional event data

        Returns:
            The logged event
        """
        event = SniperAttemptEvent(
            symbol=symbol.upper(),
            timestamp=datetime.now().isoformat(),
            ats_score=ats_score,
            ats_qualified=ats_score >= kwargs.get("ats_min_confidence", 0.6),
            hrdc_mode=hrdc_mode,
            pullback_depth_pct=pullback_depth_pct,
            entry_reason=entry_reason,
            entry_price=entry_price,
            exit_price=exit_price,
            result=result,
            pnl=pnl,
            market_session=kwargs.get("market_session", ""),
            vwap_status=kwargs.get("vwap_status", ""),
            rvol=kwargs.get("rvol", 0.0),
            volume_decreasing=kwargs.get("volume_decreasing", False),
            ema9_distance_pct=kwargs.get("ema9_distance_pct", 0.0),
            stop_loss=kwargs.get("stop_loss", 0.0),
            target=kwargs.get("target", 0.0),
            exit_reason=kwargs.get("exit_reason", "")
        )

        self.log_event(event.to_dict())
        return event

    def get_today_events(self) -> List[Dict]:
        """Get all events from today"""
        self._get_log_file()  # Ensure loaded
        return self._events.copy()

    def get_today_stats(self) -> Dict:
        """Get statistics for today's events"""
        events = self.get_today_events()

        wins = [e for e in events if e.get("result") == "WIN"]
        losses = [e for e in events if e.get("result") == "LOSS"]
        no_trades = [e for e in events if e.get("result") == "NO_TRADE"]

        total_pnl = sum(e.get("pnl", 0) for e in events)

        return {
            "total_attempts": len(events),
            "trades": len(wins) + len(losses),
            "wins": len(wins),
            "losses": len(losses),
            "no_trades": len(no_trades),
            "win_rate": len(wins) / (len(wins) + len(losses)) * 100 if (wins or losses) else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(events), 2) if events else 0
        }

    def get_events_by_symbol(self, symbol: str) -> List[Dict]:
        """Get events for a specific symbol"""
        symbol = symbol.upper()
        return [e for e in self.get_today_events() if e.get("symbol") == symbol]


# Singleton instance
_event_logger: Optional[TradingEventLogger] = None


def get_event_logger() -> TradingEventLogger:
    """Get singleton event logger instance"""
    global _event_logger
    if _event_logger is None:
        _event_logger = TradingEventLogger()
    return _event_logger


def log_sniper_attempt(**kwargs) -> SniperAttemptEvent:
    """Convenience function to log sniper attempt"""
    return get_event_logger().log_sniper_attempt(**kwargs)
