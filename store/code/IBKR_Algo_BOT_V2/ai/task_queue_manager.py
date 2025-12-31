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
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

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

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_order: List[str] = []
        self.results: Dict[str, TaskResult] = {}
        self.pipeline_status = PipelineStatus.IDLE
        self.current_task: Optional[str] = None
        self.halted_reason: Optional[str] = None
        self.run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    def _check_fail_conditions(self, task: Task, result_data: Dict) -> tuple[bool, Optional[str]]:
        """Check if any fail conditions are met"""
        for condition in task.fail_conditions:
            # Parse condition - format: "field:operator:value" or custom
            if condition == "empty_universe":
                if not result_data.get("symbols") or len(result_data.get("symbols", [])) == 0:
                    return True, "Empty universe - no symbols meet criteria"

            elif condition == "no_symbols":
                if result_data.get("symbol_count", 0) == 0:
                    return True, "No symbols meet criteria"

            elif condition == "low_rel_vol":
                symbols = result_data.get("symbols", [])
                # Handle None values properly - .get() returns None if key exists with None value
                if all((s.get("rel_vol_5m") or 0) < 300 for s in symbols):
                    return True, "Rel Vol (5m) < 300 for all symbols"

            elif condition == "low_win_rate":
                if result_data.get("win_rate", 0) < result_data.get("baseline_win_rate", 0.5):
                    return True, "Win rate below baseline"

            elif condition == "kill_regime":
                regime = result_data.get("regime", "")
                kill_regimes = result_data.get("kill_regimes", ["CHOP", "CRASH"])
                if regime in kill_regimes:
                    return True, f"Regime {regime} is in KILL_REGIMES"

            elif condition == "gate_failed":
                if not result_data.get("approved", False):
                    return True, f"Gating failed: {result_data.get('veto_reason', 'Unknown')}"

        return False, None

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
            failed, fail_reason = self._check_fail_conditions(task, result_data)

            if failed:
                task.status = TaskStatus.FAILED
                self.pipeline_status = PipelineStatus.HALTED
                self.halted_reason = fail_reason

                # Still save the report for debugging
                output_path = self._save_output(task.output_file, {
                    "status": "FAILED",
                    "reason": fail_reason,
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

            result = await self.execute_task(task_id)

            if result.status == TaskStatus.FAILED:
                logger.error(f"Task {task_id} failed, halting pipeline")
                break

        if self.pipeline_status != PipelineStatus.HALTED:
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
