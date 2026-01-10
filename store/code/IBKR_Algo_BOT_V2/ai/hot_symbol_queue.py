"""
Hot Symbol Queue Module
=======================
Acts as a temporary promotion layer for symbols experiencing abnormal market behavior.
Symbols enter this queue when price/volume shocks or crowd signals are detected,
allowing the system to react to market movements BEFORE news headlines arrive.

Key Features:
- TTL-based expiration (default 180s)
- Confidence scoring with decay
- Reason tracking (PRICE_SPIKE, VOLUME_SHOCK, CROWD_SURGE, etc.)
- Auto-eviction of stale symbols
- Priority boost on repeated triggers

Author: AI Trading Bot Team
Version: 1.0
Created: 2026-01-10
"""

import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class HotSymbolReason(Enum):
    """Reasons a symbol can become "hot" """
    PRICE_SPIKE = "PRICE_SPIKE"           # >= 3% move in 2-5 minutes
    VOLUME_SHOCK = "VOLUME_SHOCK"         # >= 3x rolling average volume
    RANGE_EXPANSION = "RANGE_EXPANSION"   # Candle range >= 2x average
    MOMENTUM_CHAIN = "MOMENTUM_CHAIN"     # 3+ consecutive green candles
    CROWD_SURGE = "CROWD_SURGE"           # Social media mention spike
    HALT_RESUME = "HALT_RESUME"           # Trading halt detected
    HOD_BREAK = "HOD_BREAK"               # Breaking high of day
    VWAP_RECLAIM = "VWAP_RECLAIM"         # Reclaimed VWAP with strength


@dataclass
class HotSymbol:
    """Represents a symbol in the hot queue"""
    symbol: str
    first_seen: datetime
    last_update: datetime
    reasons: List[str] = field(default_factory=list)
    confidence: float = 0.5
    ttl_seconds: int = 180
    metrics: Dict[str, Any] = field(default_factory=dict)
    trigger_count: int = 1

    @property
    def is_expired(self) -> bool:
        """Check if this hot symbol has expired"""
        expiry_time = self.last_update + timedelta(seconds=self.ttl_seconds)
        return datetime.now() > expiry_time

    @property
    def time_remaining(self) -> int:
        """Seconds until expiration"""
        expiry_time = self.last_update + timedelta(seconds=self.ttl_seconds)
        remaining = (expiry_time - datetime.now()).total_seconds()
        return max(0, int(remaining))

    @property
    def age_seconds(self) -> int:
        """Seconds since first detection"""
        return int((datetime.now() - self.first_seen).total_seconds())

    def to_dict(self) -> Dict:
        """Convert to dictionary for API/logging"""
        return {
            "symbol": self.symbol,
            "first_seen": self.first_seen.isoformat(),
            "last_update": self.last_update.isoformat(),
            "reasons": self.reasons,
            "confidence": round(self.confidence, 3),
            "ttl_seconds": self.ttl_seconds,
            "time_remaining": self.time_remaining,
            "age_seconds": self.age_seconds,
            "trigger_count": self.trigger_count,
            "metrics": self.metrics
        }


class HotSymbolQueue:
    """
    Manages the hot symbol queue with TTL-based expiration.

    This is the central hub for all "market shock" detections.
    Symbols are added when abnormal behavior is detected and
    automatically expire after their TTL.
    """

    # Confidence adjustments
    CONFIDENCE_BOOST_PER_TRIGGER = 0.15
    CONFIDENCE_DECAY_PER_SECOND = 0.001
    MAX_CONFIDENCE = 1.0
    MIN_CONFIDENCE = 0.1

    # Default TTL
    DEFAULT_TTL = 180  # 3 minutes

    def __init__(self, default_ttl: int = 180):
        """Initialize the hot symbol queue"""
        self._queue: Dict[str, HotSymbol] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl
        self._event_history: List[Dict] = []
        self._max_history = 500

        logger.info(f"HotSymbolQueue initialized with TTL={default_ttl}s")

    def add(
        self,
        symbol: str,
        reason: str,
        confidence: float = 0.5,
        ttl: Optional[int] = None,
        metrics: Optional[Dict] = None
    ) -> HotSymbol:
        """
        Add or update a symbol in the hot queue.

        If symbol already exists:
        - Refresh TTL
        - Boost confidence
        - Add new reason
        - Increment trigger count

        Args:
            symbol: Stock ticker symbol
            reason: Why this symbol is hot (HotSymbolReason value)
            confidence: Initial confidence score (0.0-1.0)
            ttl: Custom TTL in seconds (uses default if not specified)
            metrics: Additional metrics about the trigger

        Returns:
            The HotSymbol object (new or updated)
        """
        symbol = symbol.upper().strip()
        now = datetime.now()
        ttl = ttl or self._default_ttl
        metrics = metrics or {}

        with self._lock:
            if symbol in self._queue:
                # Update existing
                hot = self._queue[symbol]
                hot.last_update = now
                hot.trigger_count += 1
                hot.ttl_seconds = max(hot.ttl_seconds, ttl)  # Use longer TTL

                # Add reason if new
                if reason not in hot.reasons:
                    hot.reasons.append(reason)

                # Boost confidence
                hot.confidence = min(
                    self.MAX_CONFIDENCE,
                    hot.confidence + self.CONFIDENCE_BOOST_PER_TRIGGER
                )

                # Merge metrics
                hot.metrics.update(metrics)

                logger.info(
                    f"HOT SYMBOL UPDATED: {symbol} | "
                    f"reasons={hot.reasons} | confidence={hot.confidence:.2f} | "
                    f"triggers={hot.trigger_count}"
                )
            else:
                # Create new
                hot = HotSymbol(
                    symbol=symbol,
                    first_seen=now,
                    last_update=now,
                    reasons=[reason],
                    confidence=min(self.MAX_CONFIDENCE, max(self.MIN_CONFIDENCE, confidence)),
                    ttl_seconds=ttl,
                    metrics=metrics,
                    trigger_count=1
                )
                self._queue[symbol] = hot

                logger.info(
                    f"HOT SYMBOL CREATED: {symbol} | "
                    f"reason={reason} | confidence={confidence:.2f}"
                )

            # Record event
            self._record_event("HOT_SYMBOL_ADDED", symbol, reason, hot)

            return hot

    def get(self, symbol: str) -> Optional[HotSymbol]:
        """Get a hot symbol if it exists and is not expired"""
        symbol = symbol.upper().strip()

        with self._lock:
            if symbol in self._queue:
                hot = self._queue[symbol]
                if not hot.is_expired:
                    return hot
                else:
                    # Clean up expired
                    del self._queue[symbol]
                    self._record_event("HOT_SYMBOL_EXPIRED", symbol, "TTL_EXPIRED", hot)
            return None

    def get_all(self, include_expired: bool = False) -> List[HotSymbol]:
        """Get all hot symbols, optionally including expired ones"""
        with self._lock:
            if include_expired:
                return list(self._queue.values())

            # Filter and clean expired
            active = []
            expired = []

            for symbol, hot in self._queue.items():
                if hot.is_expired:
                    expired.append(symbol)
                else:
                    active.append(hot)

            # Clean up expired
            for symbol in expired:
                hot = self._queue.pop(symbol)
                self._record_event("HOT_SYMBOL_EXPIRED", symbol, "TTL_EXPIRED", hot)

            return active

    def get_symbols(self) -> List[str]:
        """Get list of active hot symbol tickers"""
        return [h.symbol for h in self.get_all()]

    def get_by_reason(self, reason: str) -> List[HotSymbol]:
        """Get hot symbols with a specific reason"""
        return [h for h in self.get_all() if reason in h.reasons]

    def get_high_confidence(self, min_confidence: float = 0.7) -> List[HotSymbol]:
        """Get hot symbols above a confidence threshold"""
        return [h for h in self.get_all() if h.confidence >= min_confidence]

    def remove(self, symbol: str) -> bool:
        """Manually remove a symbol from the queue"""
        symbol = symbol.upper().strip()

        with self._lock:
            if symbol in self._queue:
                hot = self._queue.pop(symbol)
                self._record_event("HOT_SYMBOL_REMOVED", symbol, "MANUAL_REMOVE", hot)
                logger.info(f"HOT SYMBOL REMOVED: {symbol}")
                return True
            return False

    def clear(self) -> int:
        """Clear all hot symbols"""
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            logger.info(f"HOT SYMBOL QUEUE CLEARED: {count} symbols removed")
            return count

    def cleanup_expired(self) -> int:
        """Remove all expired symbols"""
        with self._lock:
            expired = []
            for symbol, hot in self._queue.items():
                if hot.is_expired:
                    expired.append(symbol)

            for symbol in expired:
                hot = self._queue.pop(symbol)
                self._record_event("HOT_SYMBOL_EXPIRED", symbol, "TTL_EXPIRED", hot)

            if expired:
                logger.debug(f"Cleaned up {len(expired)} expired hot symbols")

            return len(expired)

    def is_hot(self, symbol: str) -> bool:
        """Check if a symbol is currently hot"""
        return self.get(symbol) is not None

    def get_confidence(self, symbol: str) -> float:
        """Get confidence for a symbol (0 if not hot)"""
        hot = self.get(symbol)
        return hot.confidence if hot else 0.0

    def _record_event(self, event_type: str, symbol: str, reason: str, hot: HotSymbol):
        """Record an event for history/debugging"""
        event = {
            "event": event_type,
            "symbol": symbol,
            "reason": reason,
            "confidence": hot.confidence,
            "trigger_count": hot.trigger_count,
            "ts": datetime.now().isoformat()
        }
        self._event_history.append(event)

        # Trim history
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get recent event history"""
        return self._event_history[-limit:]

    def get_status(self) -> Dict:
        """Get queue status for API/dashboard"""
        active = self.get_all()
        return {
            "active_count": len(active),
            "default_ttl": self._default_ttl,
            "symbols": [h.to_dict() for h in active],
            "by_reason": {
                reason.value: len([h for h in active if reason.value in h.reasons])
                for reason in HotSymbolReason
            },
            "avg_confidence": (
                sum(h.confidence for h in active) / len(active)
                if active else 0.0
            ),
            "recent_events": self.get_history(10)
        }

    def get_priority_symbols(self, limit: int = 10) -> List[str]:
        """
        Get symbols ordered by priority for FSM/Scalper.
        Priority = confidence * (1 / age_minutes)
        """
        active = self.get_all()

        def priority_score(h: HotSymbol) -> float:
            age_minutes = max(1, h.age_seconds / 60)
            return h.confidence * (1 / age_minutes) * len(h.reasons)

        sorted_symbols = sorted(active, key=priority_score, reverse=True)
        return [h.symbol for h in sorted_symbols[:limit]]


# Singleton instance
_hot_symbol_queue: Optional[HotSymbolQueue] = None


def get_hot_symbol_queue() -> HotSymbolQueue:
    """Get or create the hot symbol queue singleton"""
    global _hot_symbol_queue
    if _hot_symbol_queue is None:
        _hot_symbol_queue = HotSymbolQueue()
    return _hot_symbol_queue


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    queue = get_hot_symbol_queue()

    # Test adding symbols
    queue.add("AMD", "PRICE_SPIKE", 0.7, metrics={"pct_change": 4.2})
    queue.add("NVDA", "VOLUME_SHOCK", 0.6, metrics={"volume_ratio": 3.5})
    queue.add("AMD", "MOMENTUM_CHAIN", 0.8)  # Should update existing

    print("\n=== Hot Symbol Queue Status ===")
    status = queue.get_status()
    print(f"Active: {status['active_count']}")
    for sym in status['symbols']:
        print(f"  {sym['symbol']}: reasons={sym['reasons']} confidence={sym['confidence']:.2f}")

    print(f"\nPriority symbols: {queue.get_priority_symbols()}")
    print(f"Is AMD hot? {queue.is_hot('AMD')}")
    print(f"AMD confidence: {queue.get_confidence('AMD'):.2f}")
