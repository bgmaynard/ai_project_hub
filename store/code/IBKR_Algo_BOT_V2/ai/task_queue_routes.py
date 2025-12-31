"""
Task Queue API Routes
======================

REST API endpoints for the Task Queue system.

Endpoints:
- GET  /api/task-queue/status           - Pipeline status
- POST /api/task-queue/run              - Run full pipeline
- POST /api/task-queue/run/{task_id}    - Run single task
- GET  /api/task-queue/reports          - List all reports
- GET  /api/task-queue/report/{name}    - Get specific report
- GET  /api/task-queue/can-trade        - Check if trading allowed
- POST /api/task-queue/reset            - Reset pipeline for new run
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .task_queue_manager import (
    get_task_queue_manager,
    reset_task_queue_manager,
    TaskStatus,
    PipelineStatus,
    REPORTS_DIR
)
from .task_group_1_discovery import register_discovery_tasks
from .task_group_2_qlib import register_qlib_tasks
from .task_group_3_chronos import register_chronos_tasks
from .task_group_4_gating import register_gating_tasks
from .task_group_5_post_trade import register_post_trade_tasks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/task-queue", tags=["Task Queue"])

# Track if tasks are registered
_tasks_registered = False


def ensure_tasks_registered():
    """Ensure all task groups are registered"""
    global _tasks_registered
    if not _tasks_registered:
        register_discovery_tasks()
        register_qlib_tasks()
        register_chronos_tasks()
        register_gating_tasks()
        register_post_trade_tasks()
        _tasks_registered = True
        logger.info("All task groups registered")


# =============================================================================
# Response Models
# =============================================================================

class PipelineStatusResponse(BaseModel):
    run_id: str
    pipeline_status: str
    current_task: Optional[str]
    halted_reason: Optional[str]
    tasks: dict
    reports_dir: str


class TradeAllowedResponse(BaseModel):
    can_trade: bool
    reason: str
    r10_approved: bool
    r11_ready: bool


class RunPipelineResponse(BaseModel):
    success: bool
    run_id: str
    pipeline_status: str
    tasks_completed: int
    tasks_failed: int
    reports_generated: list


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/status", response_model=PipelineStatusResponse)
async def get_pipeline_status():
    """Get current pipeline status"""
    ensure_tasks_registered()
    manager = get_task_queue_manager()
    return manager.get_status()


@router.post("/run")
async def run_full_pipeline(start_from: Optional[str] = Query(None)):
    """
    Run the full task queue pipeline

    Args:
        start_from: Optional task ID to start from (skips earlier tasks)
    """
    ensure_tasks_registered()
    manager = get_task_queue_manager()

    # Check if already running
    if manager.pipeline_status == PipelineStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Pipeline already running")

    logger.info(f"Starting pipeline run, start_from={start_from}")

    try:
        results = await manager.run_pipeline(start_from=start_from)

        # Count results
        completed = sum(1 for r in results.values() if r.status == TaskStatus.COMPLETED)
        failed = sum(1 for r in results.values() if r.status == TaskStatus.FAILED)
        reports = [r.output_file for r in results.values() if r.output_file]

        return {
            "success": manager.pipeline_status != PipelineStatus.HALTED,
            "run_id": manager.run_id,
            "pipeline_status": manager.pipeline_status.value,
            "tasks_completed": completed,
            "tasks_failed": failed,
            "halted_reason": manager.halted_reason,
            "reports_generated": reports
        }

    except Exception as e:
        logger.error(f"Pipeline run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/{task_id}")
async def run_single_task(task_id: str):
    """Run a single task by ID"""
    ensure_tasks_registered()
    manager = get_task_queue_manager()

    task = manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    try:
        result = await manager.execute_task(task_id)

        return {
            "task_id": task_id,
            "status": result.status.value,
            "output_file": result.output_file,
            "error": result.error,
            "duration_ms": result.duration_ms
        }

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports")
async def list_reports():
    """List all generated reports"""
    ensure_tasks_registered()
    manager = get_task_queue_manager()

    reports = []
    reports_dir = manager.reports_dir

    if reports_dir.exists():
        for f in sorted(reports_dir.glob("*.json")):
            stat = f.stat()
            reports.append({
                "name": f.name,
                "path": str(f),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

    return {
        "run_id": manager.run_id,
        "reports_dir": str(reports_dir),
        "report_count": len(reports),
        "reports": reports
    }


@router.get("/report/{report_name}")
async def get_report(report_name: str):
    """Get a specific report by name"""
    ensure_tasks_registered()
    manager = get_task_queue_manager()

    # Try current run directory first
    report_path = manager.reports_dir / report_name
    if not report_path.exists():
        # Try global reports directory
        report_path = REPORTS_DIR / report_name

    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report {report_name} not found")

    try:
        with open(report_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading report: {e}")


@router.get("/can-trade", response_model=TradeAllowedResponse)
async def check_can_trade():
    """
    Check if trade execution is allowed

    CRITICAL: Returns False unless R10 is APPROVED
    """
    ensure_tasks_registered()
    manager = get_task_queue_manager()

    can_trade, reason = manager.can_execute_trade()

    # Check R10 status
    r10_result = manager.results.get("GATING_SIGNAL_EVAL")
    r10_approved = False
    if r10_result and r10_result.status == TaskStatus.COMPLETED:
        if r10_result.data:
            r10_approved = r10_result.data.get("approved", False)

    # Check R11 status
    r11_result = manager.results.get("EXECUTION_QUEUE")
    r11_ready = False
    if r11_result and r11_result.status == TaskStatus.COMPLETED:
        if r11_result.data:
            r11_ready = r11_result.data.get("ready_for_execution", False)

    return {
        "can_trade": can_trade,
        "reason": reason,
        "r10_approved": r10_approved,
        "r11_ready": r11_ready
    }


@router.post("/reset")
async def reset_pipeline():
    """Reset the pipeline for a new run"""
    global _tasks_registered
    _tasks_registered = False

    manager = reset_task_queue_manager()
    ensure_tasks_registered()

    return {
        "success": True,
        "new_run_id": manager.run_id,
        "message": "Pipeline reset for new run"
    }


@router.get("/task/{task_id}")
async def get_task_details(task_id: str):
    """Get details for a specific task"""
    ensure_tasks_registered()
    manager = get_task_queue_manager()

    task = manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    result = manager.results.get(task_id)

    return {
        "id": task.id,
        "name": task.name,
        "group": task.group,
        "inputs": task.inputs,
        "output_file": task.output_file,
        "fail_conditions": task.fail_conditions,
        "next_task": task.next_task,
        "status": task.status.value,
        "result": {
            "status": result.status.value if result else None,
            "output_file": result.output_file if result else None,
            "error": result.error if result else None,
            "duration_ms": result.duration_ms if result else None
        } if result else None
    }


@router.get("/execution-contract")
async def get_execution_contract():
    """Get the execution contract rules"""
    return {
        "rules": [
            "Tasks execute sequentially",
            "No downstream task runs if upstream fails",
            "Every task produces a named artifact",
            "All decisions must be explainable via reports",
            "NO TRADE MAY EXECUTE WITHOUT report_R10 = APPROVED"
        ],
        "task_groups": [
            {
                "group": "MARKET_DISCOVERY",
                "tasks": ["DISCOVERY_GAPPERS", "DISCOVERY_FLOAT_FILTER", "DISCOVERY_REL_VOLUME", "DISCOVERY_HOD_BEHAVIOR"],
                "outputs": ["R1", "R2", "R3", "R4"]
            },
            {
                "group": "QLIB_RESEARCH",
                "tasks": ["QLIB_HOD_PROBABILITY", "QLIB_REGIME_CHECK", "QLIB_FEATURE_RANK"],
                "outputs": ["R5", "R6", "R7"]
            },
            {
                "group": "CHRONOS_CONTEXT",
                "tasks": ["CHRONOS_PERSISTENCE", "CHRONOS_PULLBACK_DEPTH"],
                "outputs": ["R8", "R9"]
            },
            {
                "group": "SIGNAL_GATING",
                "tasks": ["GATING_SIGNAL_EVAL", "EXECUTION_QUEUE"],
                "outputs": ["R10", "R11"],
                "critical": True
            },
            {
                "group": "POST_TRADE",
                "tasks": ["POST_TRADE_OUTCOME", "POST_STRATEGY_HEALTH", "POST_STRATEGY_TOGGLE"],
                "outputs": ["R12", "R13", "R15"]
            }
        ],
        "gating_requirements": {
            "R10_approved_required": True,
            "all_upstream_must_pass": True,
            "fail_conditions_halt_pipeline": True
        }
    }
