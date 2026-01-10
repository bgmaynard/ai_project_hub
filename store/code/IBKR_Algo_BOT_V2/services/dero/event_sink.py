"""
DERO Event Sink

Append-only JSONL writer for bot events.
Non-invasive, fire-and-forget event capture for DERO observability.

IMPORTANT:
- This module is READ-ONLY from the bot's perspective
- Events are written asynchronously, never blocking trading
- All failures are gracefully logged, never propagated
- No modifications to trading logic

Output: logs/events/bot_events_YYYY-MM-DD.jsonl
"""

import json
import logging
import threading
import queue
from datetime import datetime, date
from typing import Dict, Any, Optional
from pathlib import Path
from enum import Enum
import pytz

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types emitted by the bot"""
    SCANNER_CANDIDATE = "SCANNER_CANDIDATE"
    WATCHLIST_UPDATE = "WATCHLIST_UPDATE"
    FSM_TRANSITION = "FSM_TRANSITION"
    GATE_DECISION = "GATE_DECISION"
    TRADE_EVENT = "TRADE_EVENT"
    SYSTEM_EVENT = "SYSTEM_EVENT"


class EventSink:
    """
    Fire-and-forget event sink for DERO observability.

    Events are queued and written asynchronously to prevent
    any impact on trading performance.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path("logs/events")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._queue: queue.Queue = queue.Queue(maxsize=10000)
        self._running = False
        self._writer_thread: Optional[threading.Thread] = None
        self._current_file: Optional[Path] = None
        self._current_date: Optional[date] = None
        self._lock = threading.Lock()

        # Stats for monitoring
        self._events_written = 0
        self._events_dropped = 0
        self._errors = 0

        # Start the writer thread
        self._start_writer()

    def _start_writer(self):
        """Start the background writer thread"""
        if self._running:
            return

        self._running = True
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="DERO-EventSink-Writer"
        )
        self._writer_thread.start()
        logger.info("DERO EventSink writer thread started")

    def _writer_loop(self):
        """Background thread that writes events to file"""
        while self._running:
            try:
                # Block with timeout to allow clean shutdown
                event = self._queue.get(timeout=1.0)
                self._write_event(event)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self._errors += 1
                logger.warning(f"EventSink writer error (non-fatal): {e}")

    def _get_file_for_date(self, event_date: date) -> Path:
        """Get or create the event file for a specific date"""
        if self._current_date != event_date:
            self._current_date = event_date
            self._current_file = self.base_dir / f"bot_events_{event_date.isoformat()}.jsonl"
        return self._current_file

    def _write_event(self, event: Dict[str, Any]):
        """Write a single event to the JSONL file"""
        try:
            # Parse timestamp to get the date
            ts = event.get("timestamp", datetime.now(pytz.UTC).isoformat())
            if isinstance(ts, str):
                event_date = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
            else:
                event_date = ts.date()

            filepath = self._get_file_for_date(event_date)

            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")

            self._events_written += 1

        except Exception as e:
            self._errors += 1
            logger.warning(f"Failed to write event (non-fatal): {e}")

    def emit_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        ts: Optional[datetime] = None
    ):
        """
        Emit an event to the sink.

        This is fire-and-forget - it will never block or raise exceptions.
        If the queue is full, the event is dropped with a warning.

        Args:
            event_type: Type of event (from EventType enum or string)
            payload: Event data dictionary
            ts: Optional timestamp (defaults to now UTC)
        """
        try:
            if ts is None:
                ts = datetime.now(pytz.UTC)

            event = {
                "type": event_type if isinstance(event_type, str) else event_type.value,
                "timestamp": ts.isoformat(),
                "payload": payload
            }

            # Non-blocking put
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self._events_dropped += 1
                if self._events_dropped % 100 == 1:
                    logger.warning(f"EventSink queue full, dropped {self._events_dropped} events")

        except Exception as e:
            # Never propagate exceptions - this is fire-and-forget
            self._errors += 1
            logger.debug(f"EventSink emit error (suppressed): {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get event sink statistics"""
        return {
            "running": self._running,
            "events_written": self._events_written,
            "events_dropped": self._events_dropped,
            "errors": self._errors,
            "queue_size": self._queue.qsize(),
            "current_file": str(self._current_file) if self._current_file else None,
        }

    def stop(self):
        """Stop the writer thread gracefully"""
        self._running = False
        if self._writer_thread:
            self._writer_thread.join(timeout=5.0)
        logger.info(f"EventSink stopped. Written: {self._events_written}, Dropped: {self._events_dropped}")


# Singleton instance
_event_sink: Optional[EventSink] = None
_sink_lock = threading.Lock()


def get_event_sink() -> EventSink:
    """Get or create the singleton EventSink instance"""
    global _event_sink
    if _event_sink is None:
        with _sink_lock:
            if _event_sink is None:
                _event_sink = EventSink()
    return _event_sink


def emit_event(
    event_type: str,
    payload: Dict[str, Any],
    ts: Optional[datetime] = None
):
    """
    Convenience function to emit an event.

    This is the primary interface for bot components to emit events.
    It is fire-and-forget and will never block or raise exceptions.

    Args:
        event_type: Type of event (e.g., "SCANNER_CANDIDATE", "FSM_TRANSITION")
        payload: Event data dictionary
        ts: Optional timestamp (defaults to now UTC)

    Example:
        emit_event("SCANNER_CANDIDATE", {
            "symbol": "AAPL",
            "scanner": "hod_momentum",
            "score": 85.5
        })
    """
    try:
        sink = get_event_sink()
        sink.emit_event(event_type, payload, ts)
    except Exception:
        # Absolute safety - never let event emission affect trading
        pass


# Event type constants for easy import
SCANNER_CANDIDATE = EventType.SCANNER_CANDIDATE.value
WATCHLIST_UPDATE = EventType.WATCHLIST_UPDATE.value
FSM_TRANSITION = EventType.FSM_TRANSITION.value
GATE_DECISION = EventType.GATE_DECISION.value
TRADE_EVENT = EventType.TRADE_EVENT.value
SYSTEM_EVENT = EventType.SYSTEM_EVENT.value
