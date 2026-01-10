"""
Funnel Metrics API Routes
=========================
REST endpoints for pipeline funnel monitoring.
"""

import logging
from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ops/funnel", tags=["Ops - Funnel Metrics"])

try:
    from ai.funnel_metrics import get_funnel_metrics
    HAS_FUNNEL = True
except ImportError as e:
    logger.warning(f"Funnel metrics not available: {e}")
    HAS_FUNNEL = False


@router.get("/status")
async def get_funnel_status():
    """
    Get current funnel status.

    Shows where candidates are dying in the pipeline:
    - found_by_scanners
    - injected_symbols
    - deferred_by_rate_limiter
    - rejected_by_quality_gate
    - chronos_signals_emitted
    - gating_attempts
    - gating_approvals / vetoes
    - trade_executions

    Also provides conversion rates and bottleneck diagnosis.
    """
    if not HAS_FUNNEL:
        return {"error": "Funnel metrics not available", "available": False}

    metrics = get_funnel_metrics()
    return metrics.get_funnel_status()


@router.get("/snapshot")
async def get_latest_snapshot():
    """Get the most recent funnel snapshot"""
    if not HAS_FUNNEL:
        return {"error": "Funnel metrics not available"}

    metrics = get_funnel_metrics()
    snapshot = metrics.take_snapshot()

    return {
        "timestamp": snapshot.timestamp,
        "found_by_scanners": snapshot.found_by_scanners,
        "injected_symbols": snapshot.injected_symbols,
        "deferred_by_rate_limiter": snapshot.deferred_by_rate_limiter,
        "rejected_by_quality_gate": snapshot.rejected_by_quality_gate,
        "chronos_signals_emitted": snapshot.chronos_signals_emitted,
        "gating_attempts": snapshot.gating_attempts,
        "gating_approvals": snapshot.gating_approvals,
        "gating_vetoes": snapshot.gating_vetoes,
        "veto_reasons": snapshot.veto_reasons,
        "trade_executions": snapshot.trade_executions,
        "symbols_in_pipeline": snapshot.symbols_in_pipeline
    }


@router.get("/snapshots")
async def get_snapshots(limit: int = Query(12, description="Number of snapshots")):
    """Get historical snapshots (default last hour at 5-min intervals)"""
    if not HAS_FUNNEL:
        return {"error": "Funnel metrics not available", "snapshots": []}

    metrics = get_funnel_metrics()
    snapshots = metrics.snapshots[-limit:]

    return {
        "count": len(snapshots),
        "snapshots": [
            {
                "timestamp": s.timestamp,
                "found": s.found_by_scanners,
                "injected": s.injected_symbols,
                "gating_attempts": s.gating_attempts,
                "approvals": s.gating_approvals,
                "vetoes": s.gating_vetoes,
                "trades": s.trade_executions
            }
            for s in snapshots
        ]
    }


@router.get("/vetoes")
async def get_veto_breakdown():
    """Get detailed breakdown of veto reasons"""
    if not HAS_FUNNEL:
        return {"error": "Funnel metrics not available"}

    metrics = get_funnel_metrics()
    veto_reasons = dict(metrics.veto_reasons)

    # Sort by count descending
    sorted_reasons = sorted(veto_reasons.items(), key=lambda x: x[1], reverse=True)

    total_vetoes = sum(veto_reasons.values())

    return {
        "total_vetoes": total_vetoes,
        "breakdown": [
            {
                "reason": reason,
                "count": count,
                "percentage": round(count / total_vetoes * 100, 1) if total_vetoes > 0 else 0
            }
            for reason, count in sorted_reasons
        ],
        "symbols_vetoed": metrics.symbols_vetoed[-20:]
    }


@router.get("/quality-rejects")
async def get_quality_reject_breakdown():
    """Get detailed breakdown of quality gate rejections"""
    if not HAS_FUNNEL:
        return {"error": "Funnel metrics not available"}

    metrics = get_funnel_metrics()
    reject_reasons = dict(metrics.quality_reject_reasons)

    sorted_reasons = sorted(reject_reasons.items(), key=lambda x: x[1], reverse=True)
    total_rejects = sum(reject_reasons.values())

    return {
        "total_rejects": total_rejects,
        "breakdown": [
            {
                "reason": reason,
                "count": count,
                "percentage": round(count / total_rejects * 100, 1) if total_rejects > 0 else 0
            }
            for reason, count in sorted_reasons
        ]
    }


@router.get("/scanners")
async def get_scanner_sources():
    """Get breakdown of symbols by scanner source"""
    if not HAS_FUNNEL:
        return {"error": "Funnel metrics not available"}

    metrics = get_funnel_metrics()
    sources = dict(metrics.scanner_sources)

    sorted_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)
    total = sum(sources.values())

    return {
        "total_found": total,
        "by_source": [
            {
                "source": source,
                "count": count,
                "percentage": round(count / total * 100, 1) if total > 0 else 0
            }
            for source, count in sorted_sources
        ]
    }


@router.get("/diagnostic")
async def get_diagnostic():
    """Get pipeline diagnostic - where is the bottleneck?"""
    if not HAS_FUNNEL:
        return {"error": "Funnel metrics not available"}

    metrics = get_funnel_metrics()
    status = metrics.get_funnel_status()

    return {
        "bottleneck": status["diagnostic"]["bottleneck"],
        "health": status["diagnostic"]["health"],
        "conversion_rates": status["conversion_rates"],
        "recommendation": _get_recommendation(status)
    }


def _get_recommendation(status: dict) -> str:
    """Generate recommendation based on funnel status"""
    stages = status["stages"]
    diagnostic = status["diagnostic"]

    if "NO_SCANNER_FINDS" in diagnostic["bottleneck"]:
        return "Check Finviz/Schwab scanners. Run /api/scanner/finviz/scan-all manually."

    if "NO_INJECTIONS" in diagnostic["bottleneck"]:
        return "Symbols found but not injected. Check quality filters (price, volume thresholds)."

    if "NO_GATING_ATTEMPTS" in diagnostic["bottleneck"]:
        return "Symbols not reaching gating. Check momentum detection and state machine."

    if "ALL_VETOED" in diagnostic["bottleneck"]:
        return "All symbols vetoed. Check gating config - may need to relax regime requirements or enable micro-override."

    if "APPROVED_BUT_NOT_TRADED" in diagnostic["bottleneck"]:
        return "Approvals but no trades. Check scalper enabled status and order execution."

    return "Pipeline healthy. Monitor for changes."


@router.post("/reset")
async def reset_funnel():
    """Reset funnel metrics (typically called at start of trading day)"""
    if not HAS_FUNNEL:
        return {"error": "Funnel metrics not available", "success": False}

    metrics = get_funnel_metrics()
    metrics.reset_daily()

    return {
        "success": True,
        "message": "Funnel metrics reset",
        "timestamp": metrics.last_reset.isoformat()
    }


@router.post("/start")
async def start_funnel_snapshots():
    """Start automatic snapshot collection (every 5 minutes)"""
    if not HAS_FUNNEL:
        return {"error": "Funnel metrics not available", "success": False}

    metrics = get_funnel_metrics()
    await metrics.start()

    return {
        "success": True,
        "message": "Funnel snapshot collection started",
        "interval_seconds": metrics.snapshot_interval
    }


@router.post("/stop")
async def stop_funnel_snapshots():
    """Stop automatic snapshot collection"""
    if not HAS_FUNNEL:
        return {"error": "Funnel metrics not available", "success": False}

    metrics = get_funnel_metrics()
    await metrics.stop()

    return {
        "success": True,
        "message": "Funnel snapshot collection stopped"
    }
