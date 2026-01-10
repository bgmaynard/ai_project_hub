"""
TASK 7: Postmortem "Why Not Traded" Report
==========================================

Generates daily report explaining why top movers were or weren't traded.

For each top mover, tracks:
- Were they discovered?
- Were they injected?
- Did they pass quality gate?
- Did Chronos emit patterns?
- Did R10 gate approve?
- If veto: exact veto reason(s)

Output: reports/postmortem_YYYY-MM-DD.json
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pytz

logger = logging.getLogger(__name__)

ET_TZ = pytz.timezone('US/Eastern')
REPORTS_DIR = Path("C:/ai_project_hub/store/code/IBKR_Algo_BOT_V2/reports")


class PostmortemReportGenerator:
    """
    Generates daily "Why Not Traded" postmortem reports.

    Collects data from:
    - Market movers (Schwab/Finviz)
    - Discovery logs
    - Injection logs
    - Quality gate reports
    - Chronos patterns
    - R10 gating decisions
    """

    def __init__(self):
        self.report_date = datetime.now(ET_TZ).strftime("%Y-%m-%d")
        self.reports_dir = REPORTS_DIR

    async def get_top_movers(self, limit: int = 20) -> List[Dict]:
        """Fetch top movers from available sources"""
        movers = []

        # Try Schwab movers
        try:
            from .connection_manager import get_connection_manager
            conn = get_connection_manager()
            data, _ = await conn.fetch_market_movers("up")
            if isinstance(data, list):
                for m in data[:limit]:
                    movers.append({
                        "symbol": m.get("symbol", ""),
                        "price": m.get("price") or m.get("lastPrice", 0),
                        "change_pct": m.get("changePct") or m.get("change_pct", 0),
                        "volume": m.get("volume") or m.get("totalVolume", 0),
                        "source": "schwab"
                    })
        except Exception as e:
            logger.debug(f"Schwab movers failed: {e}")

        # Try Finviz if we need more
        if len(movers) < limit:
            try:
                from .finviz_scanner import get_finviz_scanner
                scanner = get_finviz_scanner()
                gainers = await scanner.scan_top_gainers(limit=limit)
                if gainers:
                    existing_symbols = {m["symbol"] for m in movers}
                    for g in gainers:
                        if g.get("symbol") not in existing_symbols:
                            movers.append({
                                "symbol": g.get("symbol", ""),
                                "price": g.get("price", 0),
                                "change_pct": g.get("change_pct", 0),
                                "volume": g.get("volume", 0),
                                "source": "finviz"
                            })
            except Exception as e:
                logger.debug(f"Finviz movers failed: {e}")

        return movers[:limit]

    def _load_json_file(self, path: Path) -> Optional[Dict]:
        """Load JSON file safely"""
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.debug(f"Failed to load {path}: {e}")
        return None

    def _find_latest_run_dir(self) -> Optional[Path]:
        """Find the latest task queue run directory"""
        try:
            run_dirs = [d for d in self.reports_dir.iterdir() if d.is_dir() and d.name.startswith("20")]
            if run_dirs:
                return sorted(run_dirs, reverse=True)[0]
        except Exception:
            pass
        return None

    def get_discovery_status(self, symbol: str) -> Dict:
        """Check if symbol was discovered"""
        status = {
            "discovered": False,
            "discovery_source": None,
            "discovery_time": None,
            "filter_excluded": False,
            "exclusion_reason": None
        }

        # Check continuous discovery
        try:
            from .continuous_discovery import _discovered_symbols, _last_filter_stats

            if symbol in _discovered_symbols:
                status["discovered"] = True
                status["discovery_source"] = "continuous_discovery"

            # Check if symbol was excluded by filters
            if _last_filter_stats:
                excluded = _last_filter_stats.get("excluded_symbols", {})
                for reason, symbols in excluded.items():
                    if symbol in symbols:
                        status["filter_excluded"] = True
                        status["exclusion_reason"] = reason
                        break
        except Exception as e:
            logger.debug(f"Discovery check failed for {symbol}: {e}")

        return status

    def get_injection_status(self, symbol: str) -> Dict:
        """Check if symbol was injected"""
        status = {
            "injected": False,
            "injection_source": None,
            "injection_time": None,
            "rate_limited": False,
            "deferred": False,
            "expired": False,
            "provenance": None
        }

        # Check latest run directory for injection data
        run_dir = self._find_latest_run_dir()
        if run_dir:
            inject_file = run_dir / "injected_symbols.json"
            inject_data = self._load_json_file(inject_file)

            if inject_data:
                symbols = inject_data.get("symbols", [])
                provenance = inject_data.get("provenance", {})

                if symbol in symbols:
                    status["injected"] = True
                    if symbol in provenance:
                        prov = provenance[symbol]
                        status["injection_source"] = prov.get("injection_source")
                        status["injection_time"] = prov.get("timestamp")
                        status["provenance"] = prov

            # Check deferred symbols
            deferred_file = run_dir / "deferred_symbols.json"
            deferred_data = self._load_json_file(deferred_file)
            if deferred_data:
                deferred_symbols = deferred_data.get("symbols", [])
                if symbol in deferred_symbols:
                    status["deferred"] = True

        # Check rate limiter stats
        try:
            from .task_queue_manager import get_task_queue_manager
            manager = get_task_queue_manager()
            stats = manager.get_injection_stats()

            # Check expired symbols
            expired = stats.get("session", {}).get("expired_count", 0)
            if expired > 0:
                expired_syms = manager.rate_limiter.get_expired_symbols()
                for exp in expired_syms:
                    if exp.get("symbol") == symbol:
                        status["expired"] = True
                        break
        except Exception as e:
            logger.debug(f"Rate limiter check failed for {symbol}: {e}")

        return status

    def get_quality_gate_status(self, symbol: str) -> Dict:
        """Check quality gate status"""
        status = {
            "evaluated": False,
            "approved_for_bypass": False,
            "deferred_to_discovery": False,
            "check_results": None
        }

        run_dir = self._find_latest_run_dir()
        if run_dir:
            quality_file = run_dir / "report_QUALITY_GATE.json"
            quality_data = self._load_json_file(quality_file)

            if quality_data:
                approved = quality_data.get("approved_for_bypass", [])
                deferred = quality_data.get("deferred_to_discovery", [])
                checks = quality_data.get("quality_details", {}).get("checks", {})

                if symbol in approved:
                    status["evaluated"] = True
                    status["approved_for_bypass"] = True

                if symbol in deferred:
                    status["evaluated"] = True
                    status["deferred_to_discovery"] = True

                if symbol in checks:
                    status["check_results"] = checks[symbol]

        return status

    def get_chronos_status(self, symbol: str) -> Dict:
        """Check Chronos pattern status"""
        status = {
            "analyzed": False,
            "regime": None,
            "confidence": None,
            "pattern_detected": False,
            "pattern_type": None
        }

        try:
            from .chronos_adapter import get_chronos_adapter
            adapter = get_chronos_adapter()
            context = adapter.get_context(symbol)

            if context:
                status["analyzed"] = True
                status["regime"] = context.market_regime
                status["confidence"] = context.regime_confidence
                status["pattern_detected"] = context.regime_confidence >= 0.7
        except Exception as e:
            logger.debug(f"Chronos check failed for {symbol}: {e}")

        return status

    def get_gating_status(self, symbol: str) -> Dict:
        """Check R10 gating decision"""
        status = {
            "evaluated": False,
            "approved": False,
            "vetoed": False,
            "veto_reasons": [],
            "checks": None,
            "ats_state": None
        }

        run_dir = self._find_latest_run_dir()
        if run_dir:
            # Check main reports dir and task queue reports
            for reports_path in [run_dir, Path("ai/task_queue_reports")]:
                r10_file = reports_path / "report_R10_signal_decision.json"
                r10_data = self._load_json_file(r10_file)

                if r10_data:
                    decisions = r10_data.get("decisions", [])
                    for decision in decisions:
                        if decision.get("symbol") == symbol:
                            status["evaluated"] = True
                            status["approved"] = decision.get("approved", False)
                            status["vetoed"] = not decision.get("approved", False)
                            status["veto_reasons"] = decision.get("failed_checks", [])
                            status["checks"] = decision.get("checks")
                            status["ats_state"] = decision.get("ats_state")
                            break

        return status

    async def generate_report(self, save: bool = True) -> Dict:
        """
        Generate the full postmortem report.

        Args:
            save: If True, save report to file

        Returns:
            Complete postmortem report dict
        """
        logger.info("Generating postmortem report...")

        # Get top movers
        top_movers = await self.get_top_movers(limit=20)

        # Analyze each mover
        mover_analysis = []
        opportunity_misses = []

        for mover in top_movers:
            symbol = mover.get("symbol", "")
            if not symbol:
                continue

            analysis = {
                "symbol": symbol,
                "price": mover.get("price"),
                "change_pct": mover.get("change_pct"),
                "volume": mover.get("volume"),
                "mover_source": mover.get("source"),
                "discovery": self.get_discovery_status(symbol),
                "injection": self.get_injection_status(symbol),
                "quality_gate": self.get_quality_gate_status(symbol),
                "chronos": self.get_chronos_status(symbol),
                "gating": self.get_gating_status(symbol)
            }

            # Determine outcome
            if analysis["gating"]["approved"]:
                analysis["outcome"] = "TRADED"
            elif analysis["gating"]["vetoed"]:
                analysis["outcome"] = "VETOED"
                analysis["veto_summary"] = analysis["gating"]["veto_reasons"]
            elif not analysis["discovery"]["discovered"]:
                analysis["outcome"] = "NOT_DISCOVERED"
                if analysis["discovery"]["filter_excluded"]:
                    analysis["miss_reason"] = f"Excluded by filter: {analysis['discovery']['exclusion_reason']}"
            elif not analysis["injection"]["injected"]:
                analysis["outcome"] = "NOT_INJECTED"
                if analysis["injection"]["rate_limited"]:
                    analysis["miss_reason"] = "Rate limited"
                elif analysis["injection"]["expired"]:
                    analysis["miss_reason"] = "Expired (TTL)"
            elif analysis["quality_gate"]["deferred_to_discovery"]:
                analysis["outcome"] = "QUALITY_GATE_DEFERRED"
                analysis["miss_reason"] = "Failed quality check, deferred to discovery"
            else:
                analysis["outcome"] = "UNKNOWN"

            mover_analysis.append(analysis)

            # Track opportunity misses
            if analysis["outcome"] not in ["TRADED", "VETOED"]:
                opportunity_misses.append({
                    "symbol": symbol,
                    "change_pct": mover.get("change_pct"),
                    "outcome": analysis["outcome"],
                    "reason": analysis.get("miss_reason", "Unknown")
                })

        # Build summary statistics
        outcomes = {}
        for a in mover_analysis:
            outcome = a.get("outcome", "UNKNOWN")
            outcomes[outcome] = outcomes.get(outcome, 0) + 1

        # Compile full report
        report = {
            "report_type": "postmortem",
            "report_date": self.report_date,
            "generated_at": datetime.now(ET_TZ).isoformat(),
            "summary": {
                "top_movers_count": len(top_movers),
                "outcomes": outcomes,
                "opportunity_misses_count": len(opportunity_misses)
            },
            "sections": {
                "top_movers_snapshot": [
                    {
                        "symbol": m["symbol"],
                        "price": m.get("price"),
                        "change_pct": m.get("change_pct"),
                        "volume": m.get("volume"),
                        "source": m.get("source")
                    }
                    for m in top_movers
                ],
                "discovery_status": {
                    a["symbol"]: a["discovery"]
                    for a in mover_analysis
                },
                "injection_provenance": {
                    a["symbol"]: a["injection"]
                    for a in mover_analysis
                },
                "quality_gate_results": {
                    a["symbol"]: a["quality_gate"]
                    for a in mover_analysis
                },
                "chronos_pattern_summary": {
                    a["symbol"]: a["chronos"]
                    for a in mover_analysis
                },
                "gating_veto_reasons": {
                    a["symbol"]: {
                        "approved": a["gating"]["approved"],
                        "vetoed": a["gating"]["vetoed"],
                        "reasons": a["gating"]["veto_reasons"],
                        "ats_state": a["gating"]["ats_state"]
                    }
                    for a in mover_analysis
                },
                "opportunity_misses": opportunity_misses
            },
            "full_analysis": mover_analysis
        }

        # Save report
        if save:
            report_path = self.reports_dir / f"postmortem_{self.report_date}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Postmortem report saved: {report_path}")
            report["report_path"] = str(report_path)

        return report


# Singleton instance
_report_generator: Optional[PostmortemReportGenerator] = None


def get_postmortem_generator() -> PostmortemReportGenerator:
    """Get or create the postmortem report generator"""
    global _report_generator
    if _report_generator is None:
        _report_generator = PostmortemReportGenerator()
    return _report_generator


async def generate_daily_postmortem(save: bool = True) -> Dict:
    """Generate daily postmortem report"""
    generator = get_postmortem_generator()
    return await generator.generate_report(save=save)
