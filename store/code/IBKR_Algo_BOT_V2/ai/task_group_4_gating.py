"""
Task Group 4 - Signal Gating (CRITICAL)
=======================================

Tasks:
- GATING_SIGNAL_EVAL (R10): Signal evaluation and approval
- EXECUTION_QUEUE (R11): Trade queue management

*** NO TRADE MAY EXECUTE WITHOUT R10 = APPROVED ***
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import json
from pathlib import Path

from .task_queue_manager import Task, get_task_queue_manager

logger = logging.getLogger(__name__)

# Report paths
REPORTS_DIR = Path("ai/task_queue_reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Gating thresholds
MIN_ENTRY_SCORE = 0.50
MIN_CONTINUATION_PROB = 0.55
MIN_PERSISTENCE_SCORE = 0.50
MIN_REL_VOL = 200
BLOCKED_REGIMES = ["CHOP", "CRASH", "HALT_RISK"]


# =============================================================================
# TASK 4.1 - GATING_SIGNAL_EVAL (R10) - CRITICAL
# =============================================================================

async def task_gating_signal_eval(inputs: Dict) -> Dict:
    """
    Signal Evaluation and Approval

    INPUTS: All upstream reports (R1-R9)
    PROCESS:
        - Evaluate entry quality from R4
        - Check Qlib probability from R5
        - Check Chronos persistence from R8
        - Validate regime from R6
        - Check HOD status, VWAP position
    OUTPUT: report_R10_signal_decision.json
    FAIL: Any check fails -> VETO with reason
    """
    logger.info("=== TASK 4.1: GATING_SIGNAL_EVAL (R10) ===")

    try:
        # Load upstream reports
        r4_path = REPORTS_DIR / "report_R4_hod_behavior.json"
        r5_path = REPORTS_DIR / "report_R5_hod_probability.json"
        r6_path = REPORTS_DIR / "report_R6_regime_validity.json"
        r8_path = REPORTS_DIR / "report_R8_momentum_persistence.json"

        # Initialize decision structure
        decisions = []
        vetoes = []
        approvals = []

        # Get candidates from inputs or R4
        candidates = inputs.get('candidates', [])
        if not candidates and r4_path.exists():
            with open(r4_path, 'r') as f:
                r4_data = json.load(f)
                candidates = r4_data.get('candidates', [])

        # Load probability data
        prob_data = {}
        if r5_path.exists():
            with open(r5_path, 'r') as f:
                r5_data = json.load(f)
                prob_data = {p['symbol']: p for p in r5_data.get('predictions', [])}

        # Load regime data
        regime_valid = True
        current_regime = "UNKNOWN"
        if r6_path.exists():
            with open(r6_path, 'r') as f:
                r6_data = json.load(f)
                regime_valid = r6_data.get('regime_valid', True)
                current_regime = r6_data.get('current_regime', 'UNKNOWN')

        # Load persistence data
        persistence_data = {}
        if r8_path.exists():
            with open(r8_path, 'r') as f:
                r8_data = json.load(f)
                persistence_data = {p['symbol']: p for p in r8_data.get('predictions', [])}

        # Evaluate each candidate
        for candidate in candidates:
            symbol = candidate.get('symbol', 'UNKNOWN')
            entry_score = candidate.get('entry_score', 0)
            hod_status = candidate.get('hod_status', 'UNKNOWN')
            vwap_position = candidate.get('vwap_position', 'UNKNOWN')
            rel_vol = candidate.get('rel_vol', 0)

            # Get additional data
            qlib_prob = prob_data.get(symbol, {}).get('continuation_prob', 0)
            persistence = persistence_data.get(symbol, {}).get('persistence_score', 0)

            # Run gating checks
            checks = {
                'entry_quality': entry_score >= MIN_ENTRY_SCORE,
                'qlib_prob': qlib_prob >= MIN_CONTINUATION_PROB,
                'chronos_persistence': persistence >= MIN_PERSISTENCE_SCORE,
                'hod_status': hod_status in ['AT_HOD', 'NEAR_HOD'],
                'vwap_position': vwap_position in ['ABOVE', 'AT'],
                'regime_check': current_regime not in BLOCKED_REGIMES,
                'rel_vol_check': rel_vol >= MIN_REL_VOL or rel_vol == 0  # 0 means no data
            }

            all_passed = all(checks.values())
            failed_checks = [k for k, v in checks.items() if not v]

            decision = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'approved': all_passed,
                'checks': checks,
                'failed_checks': failed_checks,
                'entry_score': entry_score,
                'qlib_prob': qlib_prob,
                'persistence': persistence,
                'hod_status': hod_status,
                'vwap_position': vwap_position,
                'current_regime': current_regime
            }

            decisions.append(decision)

            if all_passed:
                approvals.append(symbol)
                logger.info(f"  APPROVED: {symbol} - All checks passed")
            else:
                vetoes.append({'symbol': symbol, 'reasons': failed_checks})
                logger.info(f"  VETOED: {symbol} - Failed: {failed_checks}")

        # Build report
        report = {
            'task': 'GATING_SIGNAL_EVAL',
            'report_id': 'R10',
            'timestamp': datetime.now().isoformat(),
            'total_evaluated': len(candidates),
            'approved_count': len(approvals),
            'vetoed_count': len(vetoes),
            'approved_symbols': approvals,
            'vetoes': vetoes,
            'decisions': decisions,
            'thresholds': {
                'min_entry_score': MIN_ENTRY_SCORE,
                'min_continuation_prob': MIN_CONTINUATION_PROB,
                'min_persistence_score': MIN_PERSISTENCE_SCORE,
                'min_rel_vol': MIN_REL_VOL,
                'blocked_regimes': BLOCKED_REGIMES
            }
        }

        # Save report
        report_path = REPORTS_DIR / "report_R10_signal_decision.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"  Report saved: {report_path}")
        logger.info(f"  Approved: {len(approvals)}, Vetoed: {len(vetoes)}")

        return {
            'success': True,
            'report_path': str(report_path),
            'approved': approvals,
            'vetoed': vetoes
        }

    except Exception as e:
        logger.error(f"GATING_SIGNAL_EVAL failed: {e}")
        return {'success': False, 'error': str(e)}


# =============================================================================
# TASK 4.2 - EXECUTION_QUEUE (R11)
# =============================================================================

async def task_execution_queue(inputs: Dict) -> Dict:
    """
    Trade Queue Management

    INPUTS: report_R10_signal_decision.json
    PROCESS:
        - Build execution queue from approved signals
        - Prioritize by score and timing
        - Set position sizes based on confidence
    OUTPUT: report_R11_trade_queue.json
    FAIL: Queue build fails
    """
    logger.info("=== TASK 4.2: EXECUTION_QUEUE (R11) ===")

    try:
        # Load R10 decisions
        r10_path = REPORTS_DIR / "report_R10_signal_decision.json"
        if not r10_path.exists():
            logger.warning("R10 report not found - no approved trades")
            return {'success': True, 'queue': []}

        with open(r10_path, 'r') as f:
            r10_data = json.load(f)

        approved_symbols = r10_data.get('approved_symbols', [])
        decisions = {d['symbol']: d for d in r10_data.get('decisions', [])}

        # Build execution queue
        queue = []
        for symbol in approved_symbols:
            decision = decisions.get(symbol, {})

            # Calculate position size based on confidence
            entry_score = decision.get('entry_score', 0.5)
            qlib_prob = decision.get('qlib_prob', 0.5)
            avg_confidence = (entry_score + qlib_prob) / 2

            # Size: 25% at 0.5 confidence, 100% at 1.0 confidence
            position_pct = min(100, max(25, int(avg_confidence * 100)))

            queue_entry = {
                'symbol': symbol,
                'position_size_pct': position_pct,
                'entry_score': entry_score,
                'qlib_prob': qlib_prob,
                'priority': int(avg_confidence * 100),
                'status': 'PENDING',
                'queued_at': datetime.now().isoformat()
            }
            queue.append(queue_entry)

        # Sort by priority (highest first)
        queue.sort(key=lambda x: x['priority'], reverse=True)

        # Build report
        report = {
            'task': 'EXECUTION_QUEUE',
            'report_id': 'R11',
            'timestamp': datetime.now().isoformat(),
            'queue_size': len(queue),
            'queue': queue
        }

        # Save report
        report_path = REPORTS_DIR / "report_R11_trade_queue.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"  Report saved: {report_path}")
        logger.info(f"  Queue size: {len(queue)}")

        return {
            'success': True,
            'report_path': str(report_path),
            'queue_size': len(queue),
            'queue': queue
        }

    except Exception as e:
        logger.error(f"EXECUTION_QUEUE failed: {e}")
        return {'success': False, 'error': str(e)}


# =============================================================================
# REGISTRATION
# =============================================================================

def register_gating_tasks():
    """Register all gating tasks with the task queue manager"""
    manager = get_task_queue_manager()

    # R10 - Signal Evaluation (CRITICAL)
    manager.register_task(Task(
        id="GATING_SIGNAL_EVAL",
        name="Signal Evaluation (R10)",
        group="4_GATING",
        inputs=["report_R8_momentum_persistence.json", "report_R9_pullback_depth.json"],
        process=task_gating_signal_eval,
        output_file="report_R10_signal_decision.json",
        fail_conditions=["All candidates vetoed"],
        next_task="EXECUTION_QUEUE"
    ))

    # R11 - Execution Queue
    manager.register_task(Task(
        id="EXECUTION_QUEUE",
        name="Execution Queue (R11)",
        group="4_GATING",
        inputs=["report_R10_signal_decision.json"],
        process=task_execution_queue,
        output_file="report_R11_trade_queue.json",
        fail_conditions=["Queue build failed"],
        next_task=None
    ))

    logger.info("Registered Task Group 4 (Gating) tasks")
