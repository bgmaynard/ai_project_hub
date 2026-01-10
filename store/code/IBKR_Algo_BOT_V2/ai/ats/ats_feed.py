"""
ATS Feed

Main integration layer connecting bar data to ATS detector.
Wires into Polygon streaming or other data sources.
"""

from datetime import datetime
from typing import Optional, List, Callable, Dict, Any
import logging
from .types import Bar, MarketContext, AtsTrigger, AtsState
from .ats_detector import AtsDetector, get_ats_detector
from .ats_registry import AtsRegistry, get_ats_registry
from .time_utils import should_allow_ats_trigger, get_session_time_context


logger = logging.getLogger(__name__)


# Callback types
TriggerCallback = Callable[[AtsTrigger], None]
StateChangeCallback = Callable[[str, AtsState, AtsState], None]


class AtsFeed:
    """
    ATS Feed Integration Layer.

    Responsibilities:
    - Receive bar data from streaming sources
    - Build MarketContext from available data
    - Route to ATS detector
    - Emit triggers to registered callbacks
    - Maintain registry state
    """

    def __init__(self):
        self._detector = get_ats_detector()
        self._registry = get_ats_registry()

        # Callbacks
        self._trigger_callbacks: List[TriggerCallback] = []
        self._state_callbacks: List[StateChangeCallback] = []

        # Context providers (functions to get context data)
        self._context_providers: Dict[str, Callable[[str], Any]] = {}

        # Statistics
        self._bars_processed = 0
        self._triggers_emitted = 0
        self._last_trigger: Optional[AtsTrigger] = None
        self._start_time = datetime.now()

    def on_bar(self, symbol: str, bar: Bar):
        """
        Process incoming bar data.

        Args:
            symbol: Stock symbol
            bar: OHLCV bar
        """
        self._bars_processed += 1

        # Build market context
        context = self._build_context(symbol, bar)

        # Get previous state for change detection
        prev_state_obj = self._registry.get_state(symbol)
        prev_state = prev_state_obj.state if prev_state_obj else AtsState.IDLE

        # Process through detector
        new_state, trigger = self._detector.process_bar(symbol, bar, context)

        # Get score from state
        state_obj = self._detector.get_state(symbol)
        score = state_obj.score if state_obj else 0.0

        # Update registry
        self._registry.update_state(symbol, new_state, score, trigger)

        # Emit state change callbacks
        if new_state != prev_state:
            self._emit_state_change(symbol, prev_state, new_state)

        # Emit trigger callbacks
        if trigger and trigger.is_valid:
            self._emit_trigger(trigger)

    def on_trade(self, symbol: str, price: float, volume: float, timestamp: datetime):
        """
        Process trade tick (builds bars internally).

        For use with Polygon streaming where we receive trade ticks.
        This method accumulates trades into 1-minute bars.
        """
        # TODO: Implement trade-to-bar aggregation
        # For now, create a simple bar from trade
        bar = Bar(
            timestamp=timestamp,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=volume,
        )
        self.on_bar(symbol, bar)

    def _build_context(self, symbol: str, bar: Bar) -> MarketContext:
        """Build market context from available sources"""
        context = MarketContext(
            symbol=symbol,
            current_price=bar.close,
            vwap=bar.vwap,
            timestamp=bar.timestamp,
        )

        # Try to get additional context from providers
        for name, provider in self._context_providers.items():
            try:
                data = provider(symbol)
                if data:
                    if name == "vwap" and "vwap" in data:
                        context.vwap = data["vwap"]
                    if name == "ema" and "ema9" in data:
                        context.ema9 = data["ema9"]
                        context.ema20 = data.get("ema20")
                    if name == "rel_volume" and "rel_volume" in data:
                        context.rel_volume = data["rel_volume"]
                    if name == "float" and "float_shares" in data:
                        context.float_shares = data["float_shares"]
                    if name == "regime" and "regime" in data:
                        context.regime = data["regime"]
            except Exception as e:
                logger.debug(f"Context provider {name} failed for {symbol}: {e}")

        return context

    def register_trigger_callback(self, callback: TriggerCallback):
        """Register callback for trigger events"""
        self._trigger_callbacks.append(callback)

    def register_state_callback(self, callback: StateChangeCallback):
        """Register callback for state change events"""
        self._state_callbacks.append(callback)

    def register_context_provider(self, name: str, provider: Callable[[str], Any]):
        """
        Register a context data provider.

        Args:
            name: Provider name (e.g., "vwap", "ema", "rel_volume")
            provider: Function that takes symbol and returns context data dict
        """
        self._context_providers[name] = provider

    def _emit_trigger(self, trigger: AtsTrigger):
        """Emit trigger to all callbacks"""
        self._triggers_emitted += 1
        self._last_trigger = trigger

        logger.info(
            f"ATS TRIGGER: {trigger.symbol} @ {trigger.entry_price:.2f} "
            f"(score: {trigger.score:.0f}, R:R {trigger.risk_reward:.1f})"
        )

        for callback in self._trigger_callbacks:
            try:
                callback(trigger)
            except Exception as e:
                logger.error(f"Trigger callback failed: {e}")

    def _emit_state_change(self, symbol: str, old_state: AtsState, new_state: AtsState):
        """Emit state change to all callbacks"""
        logger.debug(f"ATS State: {symbol} {old_state.value} -> {new_state.value}")

        for callback in self._state_callbacks:
            try:
                callback(symbol, old_state, new_state)
            except Exception as e:
                logger.error(f"State callback failed: {e}")

    def get_active_symbols(self) -> List[str]:
        """Get symbols in ACTIVE state"""
        return self._registry.get_active_symbols()

    def get_forming_symbols(self) -> List[str]:
        """Get symbols in FORMING state"""
        return self._registry.get_forming_symbols()

    def get_recent_triggers(self, minutes: int = 30) -> List[AtsTrigger]:
        """Get recent triggers"""
        return self._registry.get_recent_triggers(minutes)

    def is_symbol_tradeable(self, symbol: str) -> tuple[bool, str]:
        """
        Check if symbol is tradeable.

        Returns:
            (tradeable, reason) tuple
        """
        # Check time window
        allowed, reason = should_allow_ats_trigger()
        if not allowed:
            return False, reason

        # Check cooldown
        if self._registry.is_on_cooldown(symbol):
            remaining = self._registry.get_cooldown_remaining(symbol)
            return False, f"Cooldown active ({remaining:.0f}s remaining)"

        # Check state
        state = self._registry.get_state(symbol)
        if not state:
            return False, "No ATS state (not tracked)"

        if state.state == AtsState.EXHAUSTION:
            return False, "Momentum exhausted"

        if state.state == AtsState.INVALIDATED:
            return False, "Pattern invalidated"

        if state.state not in [AtsState.ACTIVE, AtsState.FORMING]:
            return False, f"State not tradeable ({state.state.value})"

        return True, "Tradeable"

    def reset(self):
        """Reset all state"""
        self._detector.reset_all()
        self._registry.reset()
        self._bars_processed = 0
        self._triggers_emitted = 0

    def get_status(self) -> dict:
        """Get feed status"""
        time_context = get_session_time_context()

        return {
            "running": True,
            "bars_processed": self._bars_processed,
            "triggers_emitted": self._triggers_emitted,
            "active_symbols": self.get_active_symbols(),
            "forming_symbols": self.get_forming_symbols(),
            "context_providers": list(self._context_providers.keys()),
            "callback_count": {
                "trigger": len(self._trigger_callbacks),
                "state": len(self._state_callbacks),
            },
            "last_trigger": {
                "symbol": self._last_trigger.symbol,
                "score": self._last_trigger.score,
                "timestamp": self._last_trigger.timestamp.isoformat(),
            } if self._last_trigger else None,
            "time_context": time_context,
            "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
            "detector": self._detector.get_status(),
            "registry": self._registry.get_status(),
        }


# Singleton instance
_feed: Optional[AtsFeed] = None


def get_ats_feed() -> AtsFeed:
    """Get singleton ATS feed"""
    global _feed
    if _feed is None:
        _feed = AtsFeed()
        # Register pipeline injection callback
        _register_pipeline_callback(_feed)
    return _feed


def _register_pipeline_callback(feed: AtsFeed):
    """
    Register callback to inject ATS triggers into the task queue pipeline.

    This enables the ATS system to work independently of batch discovery.
    When ATS detects a tradeable setup, it injects the symbol into the
    pipeline, allowing Qlib/Chronos to analyze even if R1 failed.
    """

    def _on_ats_trigger(trigger: AtsTrigger):
        """Inject triggered symbol into pipeline with provenance"""
        try:
            from ..task_queue_manager import get_task_queue_manager

            manager = get_task_queue_manager()
            symbol = trigger.symbol

            logger.info(f"ATS trigger -> injecting {symbol} into pipeline (score: {trigger.score})")

            # Inject with full provenance (Task 3)
            success, result = manager.inject_symbols(
                symbols=[symbol],
                source="ATS_STREAM",
                trigger_reason=trigger.trigger_type or "SMARTZONE_EXPANSION",
                ats_score=trigger.score,
                metadata={
                    "entry_price": trigger.entry_price,
                    "stop_loss": trigger.stop_loss,
                    "target_1": trigger.target_1,
                    "risk_reward": trigger.risk_reward,
                    "permission": trigger.permission
                }
            )

            if success:
                logger.info(f"ATS injection accepted: {result.get('accepted', [])}")
            else:
                logger.warning(f"ATS injection throttled: {result.get('reason', 'unknown')}")

            # If pipeline is WAITING, trigger resume via API
            if manager.is_waiting() and success:
                logger.info(f"Pipeline WAITING, scheduling resume for {symbol}")
                # Note: Can't await here from sync callback
                # The continuous discovery service or API will handle resume

        except ImportError:
            logger.debug("task_queue_manager not available, skipping pipeline injection")
        except Exception as e:
            logger.error(f"Failed to inject ATS trigger to pipeline: {e}")

    # Register the callback
    feed.register_trigger_callback(_on_ats_trigger)
    logger.info("ATS pipeline injection callback registered")
