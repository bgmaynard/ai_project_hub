"""
Candidate Funnel Metrics
========================
Tracks where candidates die in the trading pipeline.

Answers the question: "Where did it die?"

Pipeline stages:
1. Scanner Discovery → found_by_scanners
2. Symbol Injection → injected_symbols
3. Rate Limiting → deferred_by_rate_limiter
4. Quality Gate → rejected_by_quality_gate
5. Chronos Analysis → chronos_signals_emitted
6. Gating → gating_attempts, approvals, vetoes

Exposes metrics via /api/ops/funnel/status
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path
import pytz

logger = logging.getLogger(__name__)


@dataclass
class FunnelSnapshot:
    """Point-in-time snapshot of funnel metrics"""
    timestamp: str
    found_by_scanners: int
    injected_symbols: int
    deferred_by_rate_limiter: int
    rejected_by_quality_gate: int
    chronos_signals_emitted: int
    gating_attempts: int
    gating_approvals: int
    gating_vetoes: int
    veto_reasons: Dict[str, int]
    trade_executions: int
    symbols_in_pipeline: List[str]


class FunnelMetrics:
    """
    Tracks candidate flow through the trading pipeline.

    Records metrics at each stage and exposes aggregated stats.
    """

    def __init__(self):
        self.et_tz = pytz.timezone('US/Eastern')

        # Cumulative counters (reset daily)
        self.found_by_scanners = 0
        self.injected_symbols = 0
        self.deferred_by_rate_limiter = 0
        self.rejected_by_quality_gate = 0
        self.chronos_signals_emitted = 0
        self.gating_attempts = 0
        self.gating_approvals = 0
        self.gating_vetoes = 0
        self.trade_executions = 0

        # Scout metrics (Task P-T)
        self.scout_attempts = 0
        self.scout_confirmed = 0
        self.scout_failed = 0
        self.scout_to_trade = 0

        # Detailed tracking
        self.veto_reasons: Dict[str, int] = defaultdict(int)
        self.quality_reject_reasons: Dict[str, int] = defaultdict(int)
        self.scanner_sources: Dict[str, int] = defaultdict(int)
        self.scout_block_reasons: Dict[str, int] = defaultdict(int)

        # Symbol tracking
        self.symbols_found: List[str] = []
        self.symbols_injected: List[str] = []
        self.symbols_approved: List[str] = []
        self.symbols_vetoed: List[str] = []
        self.symbols_traded: List[str] = []

        # Snapshots (every 5 minutes)
        self.snapshots: List[FunnelSnapshot] = []
        self.max_snapshots = 288  # 24 hours at 5-min intervals

        # Timing
        self.last_reset = datetime.now(self.et_tz)
        self.last_snapshot = datetime.now(self.et_tz)
        self.snapshot_interval = 300  # 5 minutes

        # Background task
        self._running = False
        self._task: Optional[asyncio.Task] = None

        logger.info("FunnelMetrics initialized")

    # =========================================================================
    # RECORDING METHODS (called from pipeline stages)
    # =========================================================================

    def record_scanner_find(self, symbol: str, source: str = "unknown"):
        """Record a symbol found by scanner"""
        self.found_by_scanners += 1
        self.scanner_sources[source] += 1
        if symbol not in self.symbols_found:
            self.symbols_found.append(symbol)
        logger.debug(f"Funnel: Scanner found {symbol} (source: {source})")

    def record_symbol_injection(self, symbol: str):
        """Record a symbol injected into pipeline"""
        self.injected_symbols += 1
        if symbol not in self.symbols_injected:
            self.symbols_injected.append(symbol)
        logger.debug(f"Funnel: Injected {symbol}")

    def record_rate_limit_defer(self, symbol: str, reason: str = "rate_limit"):
        """Record a symbol deferred by rate limiter"""
        self.deferred_by_rate_limiter += 1
        logger.debug(f"Funnel: Rate limited {symbol} ({reason})")

    def record_quality_reject(self, symbol: str, reason: str):
        """Record a symbol rejected by quality gate"""
        self.rejected_by_quality_gate += 1
        self.quality_reject_reasons[reason] += 1
        logger.debug(f"Funnel: Quality rejected {symbol} ({reason})")

    def record_chronos_signal(self, symbol: str, signal: str):
        """Record a Chronos signal emission"""
        self.chronos_signals_emitted += 1
        logger.debug(f"Funnel: Chronos signal for {symbol} ({signal})")

    def record_gating_attempt(self, symbol: str):
        """Record a gating attempt"""
        self.gating_attempts += 1
        logger.debug(f"Funnel: Gating attempt for {symbol}")

    def record_gating_approval(self, symbol: str):
        """Record a gating approval"""
        self.gating_approvals += 1
        if symbol not in self.symbols_approved:
            self.symbols_approved.append(symbol)
        logger.debug(f"Funnel: APPROVED {symbol}")

    def record_gating_veto(self, symbol: str, reason: str):
        """Record a gating veto with reason"""
        self.gating_vetoes += 1
        self.veto_reasons[reason] += 1
        if symbol not in self.symbols_vetoed:
            self.symbols_vetoed.append(symbol)
        logger.debug(f"Funnel: VETOED {symbol} ({reason})")

    def record_trade_execution(self, symbol: str):
        """Record a trade execution"""
        self.trade_executions += 1
        if symbol not in self.symbols_traded:
            self.symbols_traded.append(symbol)
        logger.info(f"Funnel: EXECUTED trade for {symbol}")

    # =========================================================================
    # SCOUT RECORDING METHODS (Task P-T)
    # =========================================================================

    def record_scout_attempt(self, symbol: str, trigger: str = "unknown"):
        """Record a scout entry attempt"""
        self.scout_attempts += 1
        logger.debug(f"Funnel: Scout attempt {symbol} (trigger: {trigger})")

    def record_scout_confirmed(self, symbol: str, gain_pct: float = 0.0):
        """Record a scout that confirmed (held bars + gained)"""
        self.scout_confirmed += 1
        logger.info(f"Funnel: Scout CONFIRMED {symbol} (+{gain_pct:.2f}%)")

    def record_scout_failed(self, symbol: str, reason: str = "stopped_out"):
        """Record a scout that failed"""
        self.scout_failed += 1
        self.scout_block_reasons[reason] += 1
        logger.debug(f"Funnel: Scout FAILED {symbol} ({reason})")

    def record_scout_to_trade(self, symbol: str, target_strategy: str):
        """Record a scout that escalated to a full trade"""
        self.scout_to_trade += 1
        logger.info(f"Funnel: Scout→Trade HANDOFF {symbol} → {target_strategy}")

    def record_scout_blocked(self, symbol: str, reason: str):
        """Record a scout that was blocked before entry"""
        self.scout_block_reasons[reason] += 1
        logger.debug(f"Funnel: Scout BLOCKED {symbol} ({reason})")

    # =========================================================================
    # SNAPSHOT & REPORTING
    # =========================================================================

    def take_snapshot(self) -> FunnelSnapshot:
        """Take a point-in-time snapshot of metrics"""
        snapshot = FunnelSnapshot(
            timestamp=datetime.now(self.et_tz).isoformat(),
            found_by_scanners=self.found_by_scanners,
            injected_symbols=self.injected_symbols,
            deferred_by_rate_limiter=self.deferred_by_rate_limiter,
            rejected_by_quality_gate=self.rejected_by_quality_gate,
            chronos_signals_emitted=self.chronos_signals_emitted,
            gating_attempts=self.gating_attempts,
            gating_approvals=self.gating_approvals,
            gating_vetoes=self.gating_vetoes,
            veto_reasons=dict(self.veto_reasons),
            trade_executions=self.trade_executions,
            symbols_in_pipeline=list(set(self.symbols_injected) - set(self.symbols_traded))
        )

        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]

        self.last_snapshot = datetime.now(self.et_tz)
        return snapshot

    def get_funnel_status(self) -> Dict[str, Any]:
        """Get current funnel status (for /api/ops/funnel/status)"""
        now = datetime.now(self.et_tz)

        # Calculate conversion rates
        scanner_to_inject = (self.injected_symbols / self.found_by_scanners * 100) if self.found_by_scanners > 0 else 0
        inject_to_gate = (self.gating_attempts / self.injected_symbols * 100) if self.injected_symbols > 0 else 0
        gate_to_approve = (self.gating_approvals / self.gating_attempts * 100) if self.gating_attempts > 0 else 0
        approve_to_trade = (self.trade_executions / self.gating_approvals * 100) if self.gating_approvals > 0 else 0
        overall = (self.trade_executions / self.found_by_scanners * 100) if self.found_by_scanners > 0 else 0

        return {
            "timestamp": now.isoformat(),
            "session_start": self.last_reset.isoformat(),
            "session_duration_minutes": (now - self.last_reset).total_seconds() / 60,

            # Stage counts
            "stages": {
                "1_found_by_scanners": self.found_by_scanners,
                "2_injected_symbols": self.injected_symbols,
                "3_deferred_by_rate_limiter": self.deferred_by_rate_limiter,
                "4_rejected_by_quality_gate": self.rejected_by_quality_gate,
                "5_chronos_signals_emitted": self.chronos_signals_emitted,
                "6_gating_attempts": self.gating_attempts,
                "7_gating_approvals": self.gating_approvals,
                "8_gating_vetoes": self.gating_vetoes,
                "9_trade_executions": self.trade_executions
            },

            # Conversion rates
            "conversion_rates": {
                "scanner_to_inject_pct": round(scanner_to_inject, 1),
                "inject_to_gate_pct": round(inject_to_gate, 1),
                "gate_to_approve_pct": round(gate_to_approve, 1),
                "approve_to_trade_pct": round(approve_to_trade, 1),
                "overall_funnel_pct": round(overall, 2)
            },

            # Detailed breakdowns
            "veto_reasons": dict(self.veto_reasons),
            "quality_reject_reasons": dict(self.quality_reject_reasons),
            "scanner_sources": dict(self.scanner_sources),

            # Symbol lists (truncated)
            "symbols": {
                "found": self.symbols_found[-20:],
                "injected": self.symbols_injected[-20:],
                "approved": self.symbols_approved[-10:],
                "vetoed": self.symbols_vetoed[-10:],
                "traded": self.symbols_traded[-10:]
            },

            # Diagnostic
            "diagnostic": {
                "bottleneck": self._identify_bottleneck(),
                "health": self._assess_health()
            },

            # Scout metrics (Task P-T)
            "scout_metrics": {
                "attempts": self.scout_attempts,
                "confirmed": self.scout_confirmed,
                "failed": self.scout_failed,
                "to_trade": self.scout_to_trade,
                "confirmation_rate": round(
                    (self.scout_confirmed / self.scout_attempts * 100) if self.scout_attempts > 0 else 0, 1
                ),
                "escalation_rate": round(
                    (self.scout_to_trade / self.scout_confirmed * 100) if self.scout_confirmed > 0 else 0, 1
                ),
                "block_reasons": dict(self.scout_block_reasons)
            },

            # Recent snapshots
            "recent_snapshots": len(self.snapshots)
        }

    def _identify_bottleneck(self) -> str:
        """Identify where the funnel is losing most candidates"""
        if self.found_by_scanners == 0:
            return "NO_SCANNER_FINDS - Discovery not finding symbols"

        if self.injected_symbols == 0:
            return "NO_INJECTIONS - Symbols found but not injected"

        if self.gating_attempts == 0:
            return "NO_GATING_ATTEMPTS - Symbols not reaching gating"

        if self.gating_approvals == 0 and self.gating_vetoes > 0:
            top_veto = max(self.veto_reasons.items(), key=lambda x: x[1]) if self.veto_reasons else ("unknown", 0)
            return f"ALL_VETOED - Top reason: {top_veto[0]} ({top_veto[1]} times)"

        if self.trade_executions == 0 and self.gating_approvals > 0:
            return "APPROVED_BUT_NOT_TRADED - Execution not happening"

        return "HEALTHY - Pipeline flowing"

    def _assess_health(self) -> str:
        """Assess overall pipeline health"""
        if self.trade_executions > 0:
            return "GREEN - Trades executing"
        if self.gating_approvals > 0:
            return "YELLOW - Approvals but no trades"
        if self.gating_attempts > 0:
            return "ORANGE - Attempts but no approvals"
        if self.found_by_scanners > 0:
            return "RED - Finds but not reaching gating"
        return "GRAY - No activity"

    def reset_daily(self):
        """Reset counters for new trading day"""
        logger.info("Funnel metrics reset for new day")

        # Save yesterday's final snapshot
        if self.found_by_scanners > 0:
            self.take_snapshot()

        # Reset counters
        self.found_by_scanners = 0
        self.injected_symbols = 0
        self.deferred_by_rate_limiter = 0
        self.rejected_by_quality_gate = 0
        self.chronos_signals_emitted = 0
        self.gating_attempts = 0
        self.gating_approvals = 0
        self.gating_vetoes = 0
        self.trade_executions = 0

        # Reset scout metrics (Task P-T)
        self.scout_attempts = 0
        self.scout_confirmed = 0
        self.scout_failed = 0
        self.scout_to_trade = 0

        # Reset detailed tracking
        self.veto_reasons.clear()
        self.quality_reject_reasons.clear()
        self.scanner_sources.clear()
        self.scout_block_reasons.clear()

        # Reset symbol lists
        self.symbols_found.clear()
        self.symbols_injected.clear()
        self.symbols_approved.clear()
        self.symbols_vetoed.clear()
        self.symbols_traded.clear()

        self.last_reset = datetime.now(self.et_tz)

    # =========================================================================
    # BACKGROUND SNAPSHOT LOOP
    # =========================================================================

    async def _snapshot_loop(self):
        """Background loop to take snapshots every 5 minutes"""
        while self._running:
            try:
                await asyncio.sleep(self.snapshot_interval)
                self.take_snapshot()
                logger.debug(f"Funnel snapshot taken (total: {len(self.snapshots)})")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Snapshot loop error: {e}")

    async def start(self):
        """Start background snapshot loop"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._snapshot_loop())
        logger.info("Funnel metrics snapshot loop started")

    async def stop(self):
        """Stop background snapshot loop"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Funnel metrics snapshot loop stopped")


# Singleton instance
_funnel_metrics: Optional[FunnelMetrics] = None


def get_funnel_metrics() -> FunnelMetrics:
    """Get or create the funnel metrics singleton"""
    global _funnel_metrics
    if _funnel_metrics is None:
        _funnel_metrics = FunnelMetrics()
    return _funnel_metrics


# Convenience functions for recording from other modules
def record_scanner_find(symbol: str, source: str = "unknown"):
    get_funnel_metrics().record_scanner_find(symbol, source)

def record_symbol_injection(symbol: str):
    get_funnel_metrics().record_symbol_injection(symbol)

def record_rate_limit_defer(symbol: str, reason: str = "rate_limit"):
    get_funnel_metrics().record_rate_limit_defer(symbol, reason)

def record_quality_reject(symbol: str, reason: str):
    get_funnel_metrics().record_quality_reject(symbol, reason)

def record_chronos_signal(symbol: str, signal: str):
    get_funnel_metrics().record_chronos_signal(symbol, signal)

def record_gating_attempt(symbol: str):
    get_funnel_metrics().record_gating_attempt(symbol)

def record_gating_approval(symbol: str):
    get_funnel_metrics().record_gating_approval(symbol)

def record_gating_veto(symbol: str, reason: str):
    get_funnel_metrics().record_gating_veto(symbol, reason)

def record_trade_execution(symbol: str):
    get_funnel_metrics().record_trade_execution(symbol)


# Scout convenience functions (Task P-T)
def record_scout_attempt(symbol: str, trigger: str = "unknown"):
    get_funnel_metrics().record_scout_attempt(symbol, trigger)

def record_scout_confirmed(symbol: str, gain_pct: float = 0.0):
    get_funnel_metrics().record_scout_confirmed(symbol, gain_pct)

def record_scout_failed(symbol: str, reason: str = "stopped_out"):
    get_funnel_metrics().record_scout_failed(symbol, reason)

def record_scout_to_trade(symbol: str, target_strategy: str):
    get_funnel_metrics().record_scout_to_trade(symbol, target_strategy)

def record_scout_blocked(symbol: str, reason: str):
    get_funnel_metrics().record_scout_blocked(symbol, reason)
