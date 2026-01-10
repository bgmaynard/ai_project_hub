"""
Task Queue Manager for Small-Cap Top Gappers / HOD Momentum Bot
================================================================

Orchestrates sequential task execution with:
- Named artifact outputs (report_R1 through report_R15)
- Fail condition handling (halt pipeline on upstream failure)
- Gating approval requirement for trade execution

Task Groups:
1. MARKET DISCOVERY (R1-R4)
2. QLIB RESEARCH (R5-R7)
3. CHRONOS CONTEXT (R8-R9)
4. SIGNAL GATING (R10-R11)
5. POST-TRADE (R12-R15)
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# INJECTION RATE LIMITING (Task 1 - Post-Resilience Hardening)
# =============================================================================

# Default rate limit configuration
DEFAULT_INJECTION_LIMITS = {
    "window_seconds": 600,  # 10 minute rolling window
    "max_total": 5,         # Max symbols across all sources
    "per_source": {
        "ATS": 3,
        "ATS_TRIGGER": 3,
        "ATS_STREAM": 3,
        "CONTINUOUS_DISCOVERY": 3,
        "MANUAL": 5,
        "NEWS": 2,
    }
}

# =============================================================================
# TASK 6: Deferred Queue TTL + Session Boundary Safety
# =============================================================================

DEFAULT_TTL_CONFIG = {
    "deferred_symbol_ttl_seconds": 1200,  # 20 minutes default
    "clear_deferred_on_new_session": True,
    "session_start_hour_et": 4,  # 4 AM ET = new trading day
}

# =============================================================================
# CONDITIONAL SYNTHETIC BYPASS (Task 2 - Post-Resilience Hardening)
# =============================================================================

# Quality gate configuration for R1-R4 bypass
DEFAULT_BYPASS_CONFIG = {
    "min_chronos_confidence": 0.7,      # Minimum Chronos pattern confidence
    "ats_active_bypasses": True,         # ATS ACTIVE state allows bypass
    "require_quality_check": True,       # Enable/disable quality gate
    "sources_always_bypass": ["ATS_STREAM", "ATS_TRIGGER", "ATS"],  # High-priority sources
    "defer_on_quality_fail": True,       # Defer (not discard) if quality check fails
}


@dataclass
class InjectionEvent:
    """Record of a symbol injection event"""
    symbol: str
    source: str
    timestamp: float  # Unix timestamp
    accepted: bool
    reason: Optional[str] = None


class InjectionRateLimiter:
    """
    Rate limiter for symbol injection to prevent overwhelming Chronos.

    Implements:
    - Global limit per rolling window
    - Per-source limits
    - Soft rejection (logs and queues, doesn't hard fail)
    - TASK 6: TTL for deferred symbols
    - TASK 6: Session boundary auto-clear
    """

    def __init__(self, config: Optional[Dict] = None, ttl_config: Optional[Dict] = None):
        self.config = config or DEFAULT_INJECTION_LIMITS
        self.ttl_config = ttl_config or DEFAULT_TTL_CONFIG
        self._events: deque = deque(maxlen=1000)  # Ring buffer of events
        self._deferred_queue: List[Dict] = []  # Symbols deferred for later
        self._last_session_date: Optional[str] = None  # Track session for auto-clear
        self._expired_symbols: List[Dict] = []  # Track expired symbols for reporting

    def _get_window_events(self) -> List[InjectionEvent]:
        """Get events within the current rolling window"""
        cutoff = time.time() - self.config["window_seconds"]
        return [e for e in self._events if e.timestamp >= cutoff and e.accepted]

    def _count_by_source(self, source: str) -> int:
        """Count accepted injections from a source in current window"""
        source_upper = source.upper()
        return sum(1 for e in self._get_window_events()
                   if e.source.upper() == source_upper)

    def _count_total(self) -> int:
        """Count total accepted injections in current window"""
        return len(self._get_window_events())

    def check_rate_limit(self, symbols: List[str], source: str) -> Tuple[List[str], List[str], str]:
        """
        Check if symbols can be injected under rate limits.

        Returns:
            (accepted_symbols, deferred_symbols, reason)
        """
        source_upper = source.upper()

        # Get current counts
        total_count = self._count_total()
        source_count = self._count_by_source(source_upper)

        # Get limits
        max_total = self.config["max_total"]
        source_limits = self.config.get("per_source", {})
        max_source = source_limits.get(source_upper, max_total)

        # Calculate available slots
        total_available = max(0, max_total - total_count)
        source_available = max(0, max_source - source_count)
        available = min(total_available, source_available)

        if available == 0:
            # All symbols deferred
            reason = f"SYMBOL_INJECTION_THROTTLED: {source_upper} at limit ({source_count}/{max_source}) or total at limit ({total_count}/{max_total})"
            logger.warning(reason)
            return [], symbols, reason

        if available >= len(symbols):
            # All symbols accepted
            return symbols, [], "OK"

        # Partial acceptance
        accepted = symbols[:available]
        deferred = symbols[available:]
        reason = f"PARTIAL_THROTTLE: Accepted {len(accepted)}, deferred {len(deferred)} from {source_upper}"
        logger.info(reason)

        return accepted, deferred, reason

    def record_injection(self, symbol: str, source: str, accepted: bool, reason: Optional[str] = None):
        """Record an injection event"""
        event = InjectionEvent(
            symbol=symbol,
            source=source.upper(),
            timestamp=time.time(),
            accepted=accepted,
            reason=reason
        )
        self._events.append(event)

        if not accepted and reason:
            logger.warning(f"Injection rejected: {symbol} from {source} - {reason}")

    def defer_symbols(self, symbols: List[str], source: str):
        """Add symbols to deferred queue for next discovery cycle"""
        for sym in symbols:
            self._deferred_queue.append({
                "symbol": sym,
                "source": source,
                "deferred_at": datetime.now().isoformat(),
                "deferred_timestamp": time.time()  # TASK 6: Unix timestamp for TTL
            })
            self.record_injection(sym, source, accepted=False, reason="DEFERRED_TO_NEXT_CYCLE")

        logger.info(f"Deferred {len(symbols)} symbols to next discovery cycle")

    def _check_session_boundary(self) -> bool:
        """
        TASK 6: Check if we've crossed a session boundary (new trading day).

        Returns:
            True if session changed and deferred queue was cleared
        """
        if not self.ttl_config.get("clear_deferred_on_new_session", True):
            return False

        try:
            import pytz
            et_tz = pytz.timezone('US/Eastern')
            now_et = datetime.now(et_tz)
            session_hour = self.ttl_config.get("session_start_hour_et", 4)

            # Session date is the trading date (changes at 4 AM ET)
            if now_et.hour < session_hour:
                # Before 4 AM, we're still in previous day's session
                session_date = (now_et - timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                session_date = now_et.strftime("%Y-%m-%d")

            if self._last_session_date is None:
                self._last_session_date = session_date
                return False

            if session_date != self._last_session_date:
                # Session boundary crossed - clear deferred queue
                old_count = len(self._deferred_queue)
                old_symbols = [d["symbol"] for d in self._deferred_queue]

                if old_count > 0:
                    logger.warning(
                        f"SESSION_BOUNDARY_CLEAR: Clearing {old_count} deferred symbols from "
                        f"previous session {self._last_session_date}: {old_symbols}"
                    )
                    self._deferred_queue.clear()

                self._last_session_date = session_date
                return True

        except Exception as e:
            logger.error(f"Session boundary check failed: {e}")

        return False

    def _expire_old_deferred(self) -> List[Dict]:
        """
        TASK 6: Expire deferred symbols that have exceeded TTL.

        Returns:
            List of expired symbol dicts
        """
        ttl_seconds = self.ttl_config.get("deferred_symbol_ttl_seconds", 1200)
        cutoff = time.time() - ttl_seconds

        expired = []
        remaining = []

        for d in self._deferred_queue:
            deferred_ts = d.get("deferred_timestamp", 0)
            if deferred_ts < cutoff:
                # Symbol has expired
                expired.append(d)
                logger.warning(
                    f"SYMBOL_DEFERRED_EXPIRED: {d['symbol']} from {d['source']} "
                    f"(deferred at {d['deferred_at']}, TTL={ttl_seconds}s exceeded)"
                )
            else:
                remaining.append(d)

        if expired:
            self._expired_symbols.extend(expired)
            self._deferred_queue = remaining
            logger.info(f"Expired {len(expired)} deferred symbols, {len(remaining)} remaining")

        return expired

    def get_deferred(self, check_ttl: bool = True) -> List[Dict]:
        """
        Get and clear deferred symbols.

        Args:
            check_ttl: If True, expire old symbols first (TASK 6)
        """
        # TASK 6: Check session boundary first
        self._check_session_boundary()

        # TASK 6: Expire old symbols
        if check_ttl:
            self._expire_old_deferred()

        deferred = self._deferred_queue.copy()
        self._deferred_queue.clear()
        return deferred

    def get_expired_symbols(self) -> List[Dict]:
        """TASK 6: Get list of symbols that expired (for reporting)"""
        return self._expired_symbols.copy()

    def clear_expired_log(self):
        """TASK 6: Clear the expired symbols log"""
        self._expired_symbols.clear()

    def get_stats(self) -> Dict:
        """Get rate limiter statistics with TASK 6 TTL info"""
        window_events = self._get_window_events()

        by_source = {}
        for e in window_events:
            by_source[e.source] = by_source.get(e.source, 0) + 1

        # TASK 6: Calculate TTL info for deferred symbols
        ttl_seconds = self.ttl_config.get("deferred_symbol_ttl_seconds", 1200)
        now = time.time()
        deferred_with_ttl = []
        for d in self._deferred_queue:
            age = now - d.get("deferred_timestamp", now)
            remaining = max(0, ttl_seconds - age)
            deferred_with_ttl.append({
                "symbol": d["symbol"],
                "source": d["source"],
                "deferred_at": d["deferred_at"],
                "age_seconds": int(age),
                "ttl_remaining_seconds": int(remaining)
            })

        return {
            "window_seconds": self.config["window_seconds"],
            "max_total": self.config["max_total"],
            "per_source_limits": self.config.get("per_source", {}),
            "current_window": {
                "total_accepted": len(window_events),
                "by_source": by_source,
                "deferred_count": len(self._deferred_queue),
            },
            # TASK 6: TTL and session info
            "ttl_config": {
                "deferred_symbol_ttl_seconds": ttl_seconds,
                "clear_deferred_on_new_session": self.ttl_config.get("clear_deferred_on_new_session", True),
                "session_start_hour_et": self.ttl_config.get("session_start_hour_et", 4)
            },
            "session": {
                "current_session_date": self._last_session_date,
                "deferred_symbols": deferred_with_ttl,
                "expired_count": len(self._expired_symbols)
            },
            "recent_events": [
                {
                    "symbol": e.symbol,
                    "source": e.source,
                    "accepted": e.accepted,
                    "timestamp": datetime.fromtimestamp(e.timestamp).isoformat()
                }
                for e in list(self._events)[-10:]  # Last 10 events
            ]
        }

# Report output directory - saves to project root/reports for easy ChatGPT upload
REPORTS_DIR = Path("C:/ai_project_hub/store/code/IBKR_Algo_BOT_V2/reports")
REPORTS_DIR.mkdir(exist_ok=True)


class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"


class PipelineStatus(Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    HALTED = "HALTED"
    FAILED = "FAILED"
    WAITING = "WAITING"  # Waiting for symbols (empty discovery, not permanent halt)


@dataclass
class TaskResult:
    """Result from a task execution"""
    task_id: str
    status: TaskStatus
    output_file: Optional[str] = None
    data: Optional[Dict] = None
    error: Optional[str] = None
    duration_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Task:
    """Task definition"""
    id: str
    name: str
    group: str
    inputs: List[str]  # Required input report files
    process: Callable
    output_file: str
    fail_conditions: List[str]
    next_task: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[TaskResult] = None


class TaskQueueManager:
    """
    Manages sequential task execution for the trading pipeline.

    Rules:
    - Tasks execute sequentially
    - No downstream task runs if upstream fails
    - Every task produces a named artifact
    - No trade execution without gating approval
    """

    def __init__(self, injection_limits: Optional[Dict] = None):
        self.tasks: Dict[str, Task] = {}
        self.task_order: List[str] = []
        self.results: Dict[str, TaskResult] = {}
        self.pipeline_status = PipelineStatus.IDLE
        self.current_task: Optional[str] = None
        self.halted_reason: Optional[str] = None
        self.run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Injection rate limiter (Task 1 - Post-Resilience Hardening)
        self.rate_limiter = InjectionRateLimiter(injection_limits)

        # Ensure reports directory exists
        self.reports_dir = REPORTS_DIR / self.run_id
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"TaskQueueManager initialized, run_id={self.run_id}")

    def register_task(self, task: Task):
        """Register a task in the queue"""
        self.tasks[task.id] = task
        if task.id not in self.task_order:
            self.task_order.append(task.id)
        logger.info(f"Registered task: {task.id} -> {task.output_file}")

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)

    def _check_inputs_available(self, task: Task) -> tuple[bool, List[str]]:
        """Check if all required input files exist"""
        missing = []
        for input_file in task.inputs:
            # Check in current run directory first, then global
            path = self.reports_dir / input_file
            if not path.exists():
                path = REPORTS_DIR / input_file
                if not path.exists():
                    missing.append(input_file)
        return len(missing) == 0, missing

    def _load_input(self, filename: str) -> Optional[Dict]:
        """Load input report file"""
        path = self.reports_dir / filename
        if not path.exists():
            path = REPORTS_DIR / filename

        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def _save_output(self, filename: str, data: Dict):
        """Save output report file"""
        path = self.reports_dir / filename
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved report: {path}")
        return str(path)

    def _check_fail_conditions(self, task: Task, result_data: Dict) -> tuple[bool, Optional[str], str]:
        """
        Check if any fail conditions are met.
        
        Returns:
            (failed, reason, action) where action is 'HALT', 'WAIT', or 'CONTINUE'
            - HALT: Permanent failure, stop pipeline
            - WAIT: Temporary condition, pipeline can resume when symbols arrive
            - CONTINUE: No failure
        """
        for condition in task.fail_conditions:
            # Parse condition - format: "field:operator:value" or custom
            if condition == "empty_universe":
                if not result_data.get("symbols") or len(result_data.get("symbols", [])) == 0:
                    # WAIT not HALT - symbols may arrive later via ATS/news feed
                    return True, "Empty universe - waiting for symbols", "WAIT"

            elif condition == "no_symbols":
                if result_data.get("symbol_count", 0) == 0:
                    # WAIT not HALT - discovery can be re-run when symbols arrive
                    return True, "No symbols meet criteria - waiting for opportunities", "WAIT"

            elif condition == "low_rel_vol":
                symbols = result_data.get("symbols", [])
                # TASK 2: Exclude degraded symbols from rel_vol check
                # Only check symbols with known rel_vol data (data_quality != "DEGRADED")
                non_degraded = [s for s in symbols if s.get("data_quality") != "DEGRADED"]

                if not non_degraded:
                    # All symbols are degraded - proceed with caution, don't halt
                    logger.warning("All symbols have DEGRADED data quality - proceeding with warnings")
                elif all((s.get("rel_vol_5m") or 0) < 300 for s in non_degraded):
                    return True, "Rel Vol (5m) < 300 for all non-degraded symbols", "WAIT"

            elif condition == "low_win_rate":
                if result_data.get("win_rate", 0) < result_data.get("baseline_win_rate", 0.5):
                    return True, "Win rate below baseline", "HALT"

            elif condition == "kill_regime":
                regime = result_data.get("regime", "")
                kill_regimes = result_data.get("kill_regimes", ["CHOP", "CRASH"])
                if regime in kill_regimes:
                    return True, f"Regime {regime} is in KILL_REGIMES", "HALT"

            elif condition == "gate_failed":
                if not result_data.get("approved", False):
                    return True, f"Gating failed: {result_data.get('veto_reason', 'Unknown')}", "HALT"

        return False, None, "CONTINUE"

    async def execute_task(self, task_id: str) -> TaskResult:
        """Execute a single task"""
        task = self.tasks.get(task_id)
        if not task:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=f"Task {task_id} not found"
            )

        # Check if pipeline is halted
        if self.pipeline_status == PipelineStatus.HALTED:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.BLOCKED,
                error=f"Pipeline halted: {self.halted_reason}"
            )

        # Check inputs available
        inputs_ok, missing = self._check_inputs_available(task)
        if not inputs_ok:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.BLOCKED,
                error=f"Missing inputs: {missing}"
            )

        # Load inputs
        inputs = {}
        for input_file in task.inputs:
            data = self._load_input(input_file)
            if data:
                inputs[input_file] = data

        # Execute task
        task.status = TaskStatus.RUNNING
        self.current_task = task_id
        start_time = datetime.now()

        try:
            logger.info(f"Executing task: {task_id}")
            result_data = await task.process(inputs)
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Check fail conditions
            failed, fail_reason, fail_action = self._check_fail_conditions(task, result_data)

            if failed:
                task.status = TaskStatus.FAILED
                
                # WAIT vs HALT: WAIT allows pipeline to resume when conditions change
                if fail_action == "WAIT":
                    self.pipeline_status = PipelineStatus.WAITING
                    self.halted_reason = fail_reason
                    logger.info(f"Pipeline WAITING: {fail_reason} (can resume when symbols arrive)")
                else:
                    self.pipeline_status = PipelineStatus.HALTED
                    self.halted_reason = fail_reason
                    logger.warning(f"Pipeline HALTED: {fail_reason}")

                # Save the report (status reflects action type)
                output_path = self._save_output(task.output_file, {
                    "status": "WAITING" if fail_action == "WAIT" else "FAILED",
                    "reason": fail_reason,
                    "action": fail_action,
                    "can_resume": fail_action == "WAIT",
                    "data": result_data,
                    "timestamp": datetime.now().isoformat()
                })

                result = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    output_file=output_path,
                    data=result_data,
                    error=fail_reason,
                    duration_ms=duration_ms
                )
            else:
                task.status = TaskStatus.COMPLETED

                # Save successful output
                result_data["_meta"] = {
                    "task_id": task_id,
                    "status": "COMPLETED",
                    "timestamp": datetime.now().isoformat(),
                    "duration_ms": duration_ms
                }
                output_path = self._save_output(task.output_file, result_data)

                result = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    output_file=output_path,
                    data=result_data,
                    duration_ms=duration_ms
                )

            task.result = result
            self.results[task_id] = result
            return result

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Task {task_id} failed with exception: {e}")

            task.status = TaskStatus.FAILED
            self.pipeline_status = PipelineStatus.HALTED
            self.halted_reason = str(e)

            result = TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                duration_ms=duration_ms
            )
            task.result = result
            self.results[task_id] = result
            return result
        finally:
            self.current_task = None

    async def run_pipeline(self, start_from: Optional[str] = None) -> Dict[str, TaskResult]:
        """Run the full pipeline sequentially"""
        self.pipeline_status = PipelineStatus.RUNNING
        logger.info(f"Starting pipeline run: {self.run_id}")

        started = start_from is None

        for task_id in self.task_order:
            if not started:
                if task_id == start_from:
                    started = True
                else:
                    continue

            if self.pipeline_status == PipelineStatus.HALTED:
                logger.warning(f"Pipeline halted, skipping {task_id}")
                self.tasks[task_id].status = TaskStatus.SKIPPED
                continue
                
            # WAITING status: pause but don't skip - can resume later
            if self.pipeline_status == PipelineStatus.WAITING:
                logger.info(f"Pipeline waiting, pausing at {task_id} (can resume with inject_symbols)")
                break

            result = await self.execute_task(task_id)

            if result.status == TaskStatus.FAILED:
                # Check if this is a WAIT or HALT
                if self.pipeline_status == PipelineStatus.WAITING:
                    logger.info(f"Task {task_id} waiting for symbols, pipeline paused")
                else:
                    logger.error(f"Task {task_id} failed, halting pipeline")
                break

        # Only mark complete if not halted or waiting
        if self.pipeline_status == PipelineStatus.RUNNING:
            self.pipeline_status = PipelineStatus.COMPLETED

        # Generate summary report
        self._generate_summary_report()

        return self.results

    def _generate_summary_report(self):
        """Generate pipeline summary report"""
        summary = {
            "run_id": self.run_id,
            "pipeline_status": self.pipeline_status.value,
            "halted_reason": self.halted_reason,
            "tasks": {},
            "timestamp": datetime.now().isoformat()
        }

        for task_id, task in self.tasks.items():
            summary["tasks"][task_id] = {
                "status": task.status.value,
                "output_file": task.result.output_file if task.result else None,
                "error": task.result.error if task.result else None,
                "duration_ms": task.result.duration_ms if task.result else 0
            }

        self._save_output("report_PIPELINE_SUMMARY.json", summary)

    def get_status(self) -> Dict:
        """Get current pipeline status"""
        return {
            "run_id": self.run_id,
            "pipeline_status": self.pipeline_status.value,
            "current_task": self.current_task,
            "halted_reason": self.halted_reason,
            "tasks": {
                task_id: {
                    "status": task.status.value,
                    "group": task.group
                }
                for task_id, task in self.tasks.items()
            },
            "reports_dir": str(self.reports_dir)
        }

    def can_execute_trade(self) -> tuple[bool, str]:
        """Check if trade execution is allowed (gating must be approved)"""
        gating_result = self.results.get("GATING_SIGNAL_EVAL")

        if not gating_result:
            return False, "Gating evaluation not completed"

        if gating_result.status != TaskStatus.COMPLETED:
            return False, f"Gating status: {gating_result.status.value}"

        if not gating_result.data:
            return False, "No gating data available"

        if not gating_result.data.get("approved", False):
            return False, f"Gating rejected: {gating_result.data.get('veto_reason', 'Unknown')}"

        return True, "Trade execution approved"

    def is_waiting(self) -> bool:
        """Check if pipeline is in WAITING state (can resume)"""
        return self.pipeline_status == PipelineStatus.WAITING

    def inject_symbols(
        self,
        symbols: List[str],
        source: str = "external",
        trigger_reason: str = "UNKNOWN",
        ats_score: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, Dict]:
        """
        Inject symbols into the pipeline from external source (ATS, news, manual).

        Implements rate limiting and provenance tagging per Post-Resilience Hardening spec.

        Args:
            symbols: List of symbols to inject
            source: Source of symbols (ATS_STREAM, CONTINUOUS_DISCOVERY, MANUAL, NEWS)
            trigger_reason: Why injection was triggered (SMARTZONE_EXPANSION, GAP_SCAN, etc.)
            ats_score: Optional ATS score if from ATS trigger
            metadata: Optional additional metadata

        Returns:
            (success, result_dict) where result_dict contains accepted/deferred info
        """
        if not symbols:
            return False, {"error": "No symbols provided"}

        logger.info(f"Injection request: {len(symbols)} symbols from {source}: {symbols}")

        # TASK 1: Rate limiting check
        accepted, deferred, reason = self.rate_limiter.check_rate_limit(symbols, source)

        # Record all injection attempts
        for sym in accepted:
            self.rate_limiter.record_injection(sym, source, accepted=True)

        # Defer throttled symbols (soft rejection)
        if deferred:
            self.rate_limiter.defer_symbols(deferred, source)

        if not accepted:
            logger.warning(f"All symbols throttled: {reason}")
            return False, {
                "accepted": [],
                "deferred": deferred,
                "reason": reason,
                "rate_limit_stats": self.rate_limiter.get_stats()
            }

        # TASK 3: Build provenance metadata for accepted symbols
        timestamp = datetime.now().isoformat()
        provenance = {
            "injection_source": source.upper(),
            "trigger_reason": trigger_reason,
            "ats_score": ats_score,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }

        # Store injected symbols with provenance
        inject_file = self.reports_dir / "injected_symbols.json"
        existing_data = {"symbols": [], "provenance": {}}
        if inject_file.exists():
            with open(inject_file, 'r') as f:
                existing_data = json.load(f)

        existing_symbols = existing_data.get("symbols", [])
        existing_provenance = existing_data.get("provenance", {})

        # Merge with existing, dedupe
        all_symbols = list(set(existing_symbols + accepted))

        # Add provenance for each new symbol
        for sym in accepted:
            existing_provenance[sym] = provenance

        inject_data = {
            "symbols": all_symbols,
            "symbol_count": len(all_symbols),
            "source": source,
            "provenance": existing_provenance,
            "latest_injection": {
                "symbols": accepted,
                "source": source,
                "trigger_reason": trigger_reason,
                "ats_score": ats_score,
                "timestamp": timestamp
            },
            "deferred": [d["symbol"] for d in self.rate_limiter.get_deferred()] if deferred else [],
            "timestamp": timestamp
        }

        with open(inject_file, 'w') as f:
            json.dump(inject_data, f, indent=2)

        logger.info(f"Injected {len(accepted)} symbols (deferred {len(deferred)}): {accepted}")

        return True, {
            "accepted": accepted,
            "deferred": deferred,
            "total_in_pipeline": len(all_symbols),
            "provenance": provenance,
            "rate_limit_stats": self.rate_limiter.get_stats()
        }

    def get_injection_stats(self) -> Dict:
        """Get injection rate limiter statistics"""
        return self.rate_limiter.get_stats()

    def _check_bypass_quality(
        self,
        symbols: List[str],
        source: str,
        provenance: Optional[Dict] = None
    ) -> Tuple[List[str], List[str], Dict]:
        """
        TASK 2: Check if symbols pass quality gate for R1-R4 bypass.

        Quality Rules (from Post-Resilience Hardening spec):
        - IF injected_source == ATS AND ATS_ACTIVE → bypass R1–R4
        - ELSE IF chronos_pattern_confidence >= 0.7 → bypass R1–R4
        - ELSE → defer to discovery cycle (don't discard)

        Args:
            symbols: List of symbols to check
            source: Injection source (ATS_STREAM, MANUAL, etc.)
            provenance: Provenance metadata from injection

        Returns:
            (approved_for_bypass, deferred_to_discovery, quality_details)
        """
        config = DEFAULT_BYPASS_CONFIG
        approved = []
        deferred = []
        details = {
            "quality_check_enabled": config["require_quality_check"],
            "checks": {}
        }

        # If quality check is disabled, approve all
        if not config["require_quality_check"]:
            logger.info("Quality check disabled, approving all symbols for bypass")
            return symbols, [], {"quality_check_enabled": False, "all_approved": True}

        # Check if source is in always-bypass list
        source_upper = source.upper()
        if source_upper in config["sources_always_bypass"]:
            logger.info(f"Source {source_upper} in always-bypass list, approving all symbols")
            details["source_bypass"] = True
            return symbols, [], details

        # Check each symbol individually
        for sym in symbols:
            sym_check = {"symbol": sym, "passed": False, "reason": None}

            # 1. Check ATS state (if ATS active, allow bypass)
            ats_state = self._get_ats_state(sym)
            sym_check["ats_state"] = ats_state

            if config["ats_active_bypasses"] and ats_state == "ACTIVE":
                sym_check["passed"] = True
                sym_check["reason"] = "ATS_ACTIVE"
                approved.append(sym)
                details["checks"][sym] = sym_check
                continue

            # 2. Check Chronos confidence
            chronos_confidence = self._get_chronos_confidence(sym)
            sym_check["chronos_confidence"] = chronos_confidence

            if chronos_confidence is not None and chronos_confidence >= config["min_chronos_confidence"]:
                sym_check["passed"] = True
                sym_check["reason"] = f"CHRONOS_CONFIDENCE_{chronos_confidence:.2f}"
                approved.append(sym)
                details["checks"][sym] = sym_check
                continue

            # 3. Check if ATS score from provenance meets threshold
            if provenance and provenance.get("ats_score"):
                ats_score = provenance.get("ats_score", 0)
                sym_check["ats_score"] = ats_score
                # ATS score > 70 indicates high-quality trigger
                if ats_score >= 70:
                    sym_check["passed"] = True
                    sym_check["reason"] = f"ATS_SCORE_{ats_score}"
                    approved.append(sym)
                    details["checks"][sym] = sym_check
                    continue

            # Failed all quality checks - defer to discovery
            sym_check["passed"] = False
            sym_check["reason"] = "QUALITY_GATE_FAILED"
            deferred.append(sym)
            details["checks"][sym] = sym_check

        # Log results
        if approved:
            logger.info(f"Quality gate: {len(approved)} approved for bypass: {approved}")
        if deferred:
            logger.info(f"Quality gate: {len(deferred)} deferred to discovery cycle: {deferred}")

        return approved, deferred, details

    def _get_ats_state(self, symbol: str) -> str:
        """Get ATS state for a symbol"""
        try:
            from .ats.ats_registry import get_ats_registry
            registry = get_ats_registry()
            state = registry.get_state(symbol)
            if state:
                return state.state.value  # IDLE, FORMING, ACTIVE, EXHAUSTION, INVALIDATED
            return "UNKNOWN"
        except ImportError:
            logger.debug("ATS registry not available")
            return "UNAVAILABLE"
        except Exception as e:
            logger.debug(f"Error getting ATS state for {symbol}: {e}")
            return "ERROR"

    def _get_chronos_confidence(self, symbol: str) -> Optional[float]:
        """Get Chronos pattern confidence for a symbol"""
        try:
            from .chronos_adapter import get_chronos_adapter
            adapter = get_chronos_adapter()
            context = adapter.get_context(symbol)
            if context:
                return context.regime_confidence
            return None
        except ImportError:
            logger.debug("Chronos adapter not available")
            return None
        except Exception as e:
            logger.debug(f"Error getting Chronos confidence for {symbol}: {e}")
            return None

    def _create_synthetic_discovery_reports(self, symbols: List[str], source: str = "injected") -> bool:
        """
        Create synthetic R1-R4 reports from injected symbols.

        This allows downstream Qlib/Chronos tasks to run even when
        batch discovery finds nothing. The injected symbols bypass
        discovery and go straight to analysis.

        Args:
            symbols: List of symbols to analyze
            source: Source of symbols

        Returns:
            True if reports were created successfully
        """
        timestamp = datetime.now().isoformat()

        # Build minimal symbol objects
        symbol_objects = [
            {
                "symbol": s,
                "source": source,
                "priority": "HIGH",  # Injected symbols are high priority
                "gap_pct": 0,  # Will be fetched by downstream tasks
                "rel_vol_5m": 0,
                "price": 0,
                "float_shares": None,
                "at_hod": False,
                "near_hod": False,
                "hod_status": "UNKNOWN",
                "injected": True
            }
            for s in symbols
        ]

        # R1: Discovery Gappers (synthetic)
        r1_data = {
            "task_id": "DISCOVERY_GAPPERS",
            "status": "COMPLETED",
            "source": "injected_symbols",
            "symbols": symbol_objects,
            "symbol_count": len(symbols),
            "injected": True,
            "note": f"Synthetic report from {len(symbols)} injected symbols via {source}",
            "timestamp": timestamp
        }
        self._save_output("report_R1_daily_top_gappers.json", r1_data)

        # R2: Low Float Filter (synthetic - pass all through)
        r2_data = {
            "task_id": "DISCOVERY_FLOAT_FILTER",
            "status": "COMPLETED",
            "source": "injected_symbols",
            "symbols": symbol_objects,
            "symbol_count": len(symbols),
            "passed_filter": len(symbols),
            "injected": True,
            "note": "Injected symbols bypass float filter - will be checked by downstream",
            "timestamp": timestamp
        }
        self._save_output("report_R2_low_float_universe.json", r2_data)

        # R3: Rel Volume (synthetic)
        r3_data = {
            "task_id": "DISCOVERY_REL_VOLUME",
            "status": "COMPLETED",
            "source": "injected_symbols",
            "symbols": symbol_objects,
            "symbol_count": len(symbols),
            "injected": True,
            "note": "Injected symbols bypass volume filter - will be checked by downstream",
            "timestamp": timestamp
        }
        self._save_output("report_R3_rel_volume.json", r3_data)

        # R4: HOD Behavior (synthetic)
        r4_data = {
            "task_id": "DISCOVERY_HOD_BEHAVIOR",
            "status": "COMPLETED",
            "source": "injected_symbols",
            "symbols": symbol_objects,
            "symbol_count": len(symbols),
            "injected": True,
            "note": "Injected symbols - HOD status will be checked by Chronos/ATS",
            "timestamp": timestamp
        }
        self._save_output("report_R4_hod_behavior.json", r4_data)

        logger.info(f"Created synthetic R1-R4 reports for {len(symbols)} injected symbols")
        return True

    async def resume_from_waiting(self) -> Dict[str, TaskResult]:
        """
        Resume pipeline from WAITING state.

        TASK 2: Now includes quality gate check before creating synthetic reports.
        Only symbols that pass quality check get synthetic R1-R4 bypass.
        Symbols that fail are deferred to next discovery cycle.
        """
        if self.pipeline_status != PipelineStatus.WAITING:
            logger.warning(f"Cannot resume: pipeline is {self.pipeline_status.value}, not WAITING")
            return self.results

        # Check for injected symbols
        inject_file = self.reports_dir / "injected_symbols.json"
        if not inject_file.exists():
            logger.warning("No injected symbols available, cannot resume")
            return self.results

        with open(inject_file, 'r') as f:
            inject_data = json.load(f)

        symbols = inject_data.get("symbols", [])
        if not symbols:
            logger.warning("Injected symbols list is empty")
            return self.results

        source = inject_data.get("source", "external")
        provenance = inject_data.get("provenance", {})

        logger.info(f"Resume request with {len(symbols)} injected symbols from {source}")

        # TASK 2: Quality gate check - only approved symbols bypass R1-R4
        # Get provenance for first symbol (if per-symbol provenance exists)
        first_sym_provenance = provenance.get(symbols[0], {}) if isinstance(provenance, dict) else provenance

        approved, deferred, quality_details = self._check_bypass_quality(
            symbols=symbols,
            source=source,
            provenance=first_sym_provenance
        )

        # Save quality gate report
        quality_report = {
            "task_id": "QUALITY_GATE_CHECK",
            "timestamp": datetime.now().isoformat(),
            "input_symbols": symbols,
            "approved_for_bypass": approved,
            "deferred_to_discovery": deferred,
            "source": source,
            "quality_details": quality_details
        }
        self._save_output("report_QUALITY_GATE.json", quality_report)

        # If none passed quality gate, keep waiting
        if not approved:
            logger.warning(f"No symbols passed quality gate, deferring {len(deferred)} to discovery")

            # Keep deferred symbols in queue for next discovery cycle
            if deferred and DEFAULT_BYPASS_CONFIG.get("defer_on_quality_fail", True):
                for sym in deferred:
                    self.rate_limiter.defer_symbols([sym], source)

            # Stay in WAITING state
            self.halted_reason = f"Quality gate: {len(deferred)} symbols deferred to discovery"
            return self.results

        logger.info(f"Quality gate passed: {len(approved)} symbols approved for bypass")

        # Create synthetic discovery reports ONLY for approved symbols
        self._create_synthetic_discovery_reports(approved, source)

        # Mark discovery tasks as completed (synthetic)
        discovery_tasks = [
            "DISCOVERY_GAPPERS", "DISCOVERY_FLOAT_FILTER",
            "DISCOVERY_REL_VOLUME", "DISCOVERY_HOD_BEHAVIOR"
        ]
        for task_id in discovery_tasks:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = TaskStatus.COMPLETED
                task.result = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    output_file=f"report_R{discovery_tasks.index(task_id)+1}_*.json",
                    data={
                        "synthetic": True,
                        "symbols": approved,
                        "quality_gate_passed": len(approved),
                        "quality_gate_deferred": len(deferred)
                    }
                )
                self.results[task_id] = task.result

        # Handle deferred symbols - keep for next discovery cycle
        if deferred:
            logger.info(f"Deferring {len(deferred)} symbols to next discovery: {deferred}")
            # Update injected_symbols.json to only contain deferred symbols
            deferred_data = {
                "symbols": deferred,
                "symbol_count": len(deferred),
                "source": "DEFERRED_QUALITY_GATE",
                "original_source": source,
                "deferred_at": datetime.now().isoformat(),
                "note": "Symbols deferred from quality gate, awaiting next discovery cycle"
            }
            deferred_file = self.reports_dir / "deferred_symbols.json"
            with open(deferred_file, 'w') as f:
                json.dump(deferred_data, f, indent=2)

        # Reset to RUNNING and continue from Qlib tasks
        self.pipeline_status = PipelineStatus.RUNNING
        self.halted_reason = None

        # Start from first Qlib task (R5)
        return await self.run_pipeline(start_from="QLIB_HOD_PROBABILITY")


# Singleton instance
_task_queue_manager: Optional[TaskQueueManager] = None


def get_task_queue_manager() -> TaskQueueManager:
    """Get or create the task queue manager"""
    global _task_queue_manager
    if _task_queue_manager is None:
        _task_queue_manager = TaskQueueManager()
    return _task_queue_manager


def reset_task_queue_manager():
    """Reset the task queue manager for a new run"""
    global _task_queue_manager
    _task_queue_manager = TaskQueueManager()
    return _task_queue_manager
