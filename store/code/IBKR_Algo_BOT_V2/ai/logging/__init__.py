"""
Trading Event Logging Module
============================
Structured logging for trading events and strategy attempts.
"""

from ai.logging.events import (
    TradingEventLogger,
    get_event_logger,
    log_sniper_attempt
)

__all__ = [
    'TradingEventLogger',
    'get_event_logger',
    'log_sniper_attempt'
]
