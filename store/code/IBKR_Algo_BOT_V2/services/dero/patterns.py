"""
DERO Pattern Analyzer

Detects recurring patterns across trading sessions for investigation.
This is READ-ONLY and NON-INVASIVE - it only flags patterns with suggested actions.

Pattern Severity Levels:
- NONE: No pattern detected
- POSSIBLE: Pattern seen 1-2 times
- CONFIRMED: Pattern repeats >= 3 times in last 10 sessions with GREEN data health

Confirmation Rule:
CONFIRMED only if pattern repeats >= 3 sessions inside the last 10 trading sessions,
AND data health is GREEN on those sessions.

NO AUTO-CHANGES. Only flags + suggested actions.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class PatternSeverity(Enum):
    """Pattern severity levels"""
    NONE = "NONE"
    POSSIBLE = "POSSIBLE"
    CONFIRMED = "CONFIRMED"


class PatternCategory(Enum):
    """Pattern categories"""
    GATING = "GATING"               # Gating-related patterns
    TIMING = "TIMING"               # Time-of-day patterns
    REGIME = "REGIME"               # Market regime patterns
    ENTRY = "ENTRY"                 # Entry quality patterns
    EXIT = "EXIT"                   # Exit quality patterns
    INFRASTRUCTURE = "INFRASTRUCTURE"  # System health patterns


# Pattern definitions
PATTERN_DEFINITIONS = {
    "GATE_SPREAD_DOMINANT": {
        "category": PatternCategory.GATING,
        "description": "Spread-based vetoes dominating gating decisions",
        "suggested_action": "INVESTIGATE spread threshold settings",
        "detection": "spread_veto_rate > 40%",
    },
    "GATE_REGIME_DOMINANT": {
        "category": PatternCategory.GATING,
        "description": "Regime-based vetoes dominating gating decisions",
        "suggested_action": "REVIEW regime detection accuracy",
        "detection": "regime_veto_rate > 50%",
    },
    "LATE_ENTRIES": {
        "category": PatternCategory.TIMING,
        "description": "Entry window consistently delayed vs optimal",
        "suggested_action": "INVESTIGATE pipeline latency",
        "detection": "avg_entry_delay > 30s",
    },
    "OPEN_DRIVE_WEAKNESS": {
        "category": PatternCategory.TIMING,
        "description": "Underperformance during market open (9:30-10:30)",
        "suggested_action": "CONSIDER reducing size or avoiding open drive",
        "detection": "open_drive_win_rate < 30%",
    },
    "MIDDAY_WEAKNESS": {
        "category": PatternCategory.TIMING,
        "description": "Underperformance during midday (10:30-14:00)",
        "suggested_action": "CONSIDER reducing midday activity",
        "detection": "midday_win_rate < 25%",
    },
    "FALSE_POSITIVES_HIGH": {
        "category": PatternCategory.ENTRY,
        "description": "Too many allowed trades failing quickly",
        "suggested_action": "TIGHTEN entry criteria or gating thresholds",
        "detection": "allowed_fail_rate > 60%",
    },
    "FALSE_NEGATIVES_HIGH": {
        "category": PatternCategory.ENTRY,
        "description": "Blocked trades that would have been profitable",
        "suggested_action": "REVIEW gating criteria for over-filtering",
        "detection": "blocked_success_rate > 40%",
    },
    "CHOP_REGIME_LOSSES": {
        "category": PatternCategory.REGIME,
        "description": "Significant losses during CHOP regime",
        "suggested_action": "REDUCE position size or pause during CHOP",
        "detection": "chop_regime_pnl < -$50",
    },
    "TREND_REGIME_MISSED": {
        "category": PatternCategory.REGIME,
        "description": "Missing profitable moves during TREND regime",
        "suggested_action": "INVESTIGATE why entries not triggered in trends",
        "detection": "trend_missed_moves > 5",
    },
    "PREMATURE_EXITS": {
        "category": PatternCategory.EXIT,
        "description": "Exits occurring before optimal profit taken",
        "suggested_action": "REVIEW exit criteria - may be too tight",
        "detection": "avg_mfe_left > 50%",
    },
    "LATE_EXITS": {
        "category": PatternCategory.EXIT,
        "description": "Giving back too much profit before exit",
        "suggested_action": "REVIEW trailing stop tightness",
        "detection": "avg_giveback > 40%",
    },
    "FEED_INSTABILITY": {
        "category": PatternCategory.INFRASTRUCTURE,
        "description": "Frequent feed disconnections or errors",
        "suggested_action": "INVESTIGATE connection stability",
        "detection": "reconnects > 3 per session",
    },
    "DATA_GAPS": {
        "category": PatternCategory.INFRASTRUCTURE,
        "description": "Missing data periods affecting decisions",
        "suggested_action": "REVIEW data source reliability",
        "detection": "data_gaps > 2 per session",
    },
}


class PatternAnalyzer:
    """
    Analyzes trading patterns across sessions.

    All operations are read-only. Patterns are flagged for human investigation,
    never auto-applied.
    """

    def __init__(self, lookback_sessions: int = 10):
        self.lookback_sessions = lookback_sessions
        self._session_history: List[Dict[str, Any]] = []
        self._pattern_counts: Dict[str, int] = {}
        self._history_file = Path("services/dero/pattern_history.json")

    def load_history(self) -> List[Dict[str, Any]]:
        """Load session history from file"""
        if self._history_file.exists():
            try:
                with open(self._history_file, "r") as f:
                    self._session_history = json.load(f)
            except:
                self._session_history = []
        return self._session_history

    def save_history(self):
        """Save session history to file"""
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._history_file, "w") as f:
                json.dump(self._session_history[-self.lookback_sessions:], f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save pattern history: {e}")

    def add_session(self, session_metrics: Dict[str, Any]):
        """Add a session's metrics to history for pattern analysis"""
        self.load_history()

        session_data = {
            "date": session_metrics.get("date"),
            "data_health": session_metrics.get("data_health", "UNKNOWN"),
            "gating": session_metrics.get("gating", {}),
            "outcomes": session_metrics.get("outcomes", {}),
            "infrastructure": session_metrics.get("infrastructure", {}),
            "market": session_metrics.get("market", {}),
            "time_window_outcomes": session_metrics.get("time_window_outcomes", {}),
        }

        self._session_history.append(session_data)

        # Keep only last N sessions
        if len(self._session_history) > self.lookback_sessions:
            self._session_history = self._session_history[-self.lookback_sessions:]

        self.save_history()

    def _get_green_sessions(self) -> List[Dict[str, Any]]:
        """Get sessions with GREEN data health"""
        return [s for s in self._session_history if s.get("data_health") == "GREEN"]

    def _check_pattern(
        self,
        pattern_id: str,
        check_fn,
        min_occurrences: int = 3
    ) -> Tuple[PatternSeverity, int, List[str]]:
        """
        Check if a pattern is present across sessions.

        Args:
            pattern_id: Pattern identifier
            check_fn: Function(session) -> bool to detect pattern
            min_occurrences: Minimum occurrences for CONFIRMED

        Returns:
            (severity, occurrence_count, session_dates)
        """
        green_sessions = self._get_green_sessions()
        occurrences = 0
        dates = []

        for session in green_sessions:
            try:
                if check_fn(session):
                    occurrences += 1
                    dates.append(session.get("date", "unknown"))
            except:
                continue

        if occurrences >= min_occurrences:
            return PatternSeverity.CONFIRMED, occurrences, dates
        elif occurrences >= 1:
            return PatternSeverity.POSSIBLE, occurrences, dates
        else:
            return PatternSeverity.NONE, 0, []

    def analyze_gating_patterns(self) -> List[Dict[str, Any]]:
        """Analyze gating-related patterns"""
        patterns = []

        # GATE_SPREAD_DOMINANT
        def check_spread_dominant(session):
            gating = session.get("gating", {})
            reasons = gating.get("top_block_reasons", [])
            total = gating.get("blocked", 0)
            spread_blocks = sum(r.get("count", 0) for r in reasons if "spread" in r.get("reason", "").lower())
            return total > 0 and (spread_blocks / total) > 0.4

        severity, count, dates = self._check_pattern("GATE_SPREAD_DOMINANT", check_spread_dominant)
        if severity != PatternSeverity.NONE:
            patterns.append(self._build_pattern_flag("GATE_SPREAD_DOMINANT", severity, count, dates))

        # GATE_REGIME_DOMINANT
        def check_regime_dominant(session):
            gating = session.get("gating", {})
            reasons = gating.get("top_block_reasons", [])
            total = gating.get("blocked", 0)
            regime_blocks = sum(r.get("count", 0) for r in reasons if "regime" in r.get("reason", "").lower())
            return total > 0 and (regime_blocks / total) > 0.5

        severity, count, dates = self._check_pattern("GATE_REGIME_DOMINANT", check_regime_dominant)
        if severity != PatternSeverity.NONE:
            patterns.append(self._build_pattern_flag("GATE_REGIME_DOMINANT", severity, count, dates))

        return patterns

    def analyze_timing_patterns(self) -> List[Dict[str, Any]]:
        """Analyze time-of-day patterns"""
        patterns = []

        # OPEN_DRIVE_WEAKNESS
        def check_open_drive_weakness(session):
            time_outcomes = session.get("time_window_outcomes", {})
            open_drive = time_outcomes.get("OPEN_DRIVE", {})
            win_rate = open_drive.get("win_rate", 50)
            trades = open_drive.get("trades", 0)
            return trades >= 3 and win_rate < 30

        severity, count, dates = self._check_pattern("OPEN_DRIVE_WEAKNESS", check_open_drive_weakness)
        if severity != PatternSeverity.NONE:
            patterns.append(self._build_pattern_flag("OPEN_DRIVE_WEAKNESS", severity, count, dates))

        return patterns

    def analyze_entry_patterns(self) -> List[Dict[str, Any]]:
        """Analyze entry quality patterns"""
        patterns = []

        # FALSE_POSITIVES_HIGH
        def check_false_positives(session):
            outcomes = session.get("outcomes", {})
            gating = session.get("gating", {})
            allowed = gating.get("allowed", 0)
            losses = outcomes.get("losses", 0)
            return allowed > 5 and losses > 0 and (losses / allowed) > 0.6

        severity, count, dates = self._check_pattern("FALSE_POSITIVES_HIGH", check_false_positives)
        if severity != PatternSeverity.NONE:
            patterns.append(self._build_pattern_flag("FALSE_POSITIVES_HIGH", severity, count, dates))

        return patterns

    def analyze_exit_patterns(self) -> List[Dict[str, Any]]:
        """Analyze exit quality patterns"""
        patterns = []

        # PREMATURE_EXITS
        def check_premature_exits(session):
            outcomes = session.get("outcomes", {})
            avg_mfe = outcomes.get("avg_mfe", 0)
            avg_r = outcomes.get("avg_r", 0)
            # If MFE was much higher than realized R, exits were premature
            return avg_mfe > 0 and avg_r > 0 and (avg_r / avg_mfe) < 0.5

        severity, count, dates = self._check_pattern("PREMATURE_EXITS", check_premature_exits)
        if severity != PatternSeverity.NONE:
            patterns.append(self._build_pattern_flag("PREMATURE_EXITS", severity, count, dates))

        return patterns

    def analyze_infrastructure_patterns(self) -> List[Dict[str, Any]]:
        """Analyze infrastructure patterns"""
        patterns = []

        # FEED_INSTABILITY
        def check_feed_instability(session):
            infra = session.get("infrastructure", {})
            reconnects = infra.get("reconnect_count", 0)
            return reconnects > 3

        severity, count, dates = self._check_pattern("FEED_INSTABILITY", check_feed_instability)
        if severity != PatternSeverity.NONE:
            patterns.append(self._build_pattern_flag("FEED_INSTABILITY", severity, count, dates))

        return patterns

    def analyze_regime_patterns(self) -> List[Dict[str, Any]]:
        """Analyze market regime patterns"""
        patterns = []

        # CHOP_REGIME_LOSSES
        def check_chop_losses(session):
            market = session.get("market", {})
            outcomes = session.get("outcomes", {})
            regime = market.get("regime", "")
            pnl = outcomes.get("total_pnl", 0)
            return regime == "CHOP" and pnl < -50

        severity, count, dates = self._check_pattern("CHOP_REGIME_LOSSES", check_chop_losses)
        if severity != PatternSeverity.NONE:
            patterns.append(self._build_pattern_flag("CHOP_REGIME_LOSSES", severity, count, dates))

        return patterns

    def _build_pattern_flag(
        self,
        pattern_id: str,
        severity: PatternSeverity,
        count: int,
        dates: List[str]
    ) -> Dict[str, Any]:
        """Build a pattern flag dictionary"""
        definition = PATTERN_DEFINITIONS.get(pattern_id, {})

        return {
            "id": pattern_id,
            "category": definition.get("category", PatternCategory.GATING).value,
            "severity": severity.value,
            "count_lookback": self.lookback_sessions,
            "count_hits": count,
            "dates": dates,
            "description": definition.get("description", "Unknown pattern"),
            "suggested_action": definition.get("suggested_action", "INVESTIGATE"),
            "detection_rule": definition.get("detection", ""),
            "notes": f"Detected in {count}/{len(self._get_green_sessions())} GREEN sessions",
        }

    def analyze_all(self, current_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run all pattern analyses.

        Args:
            current_metrics: Optional current session metrics to add to history

        Returns:
            Pattern analysis results
        """
        # Add current session if provided
        if current_metrics:
            self.add_session(current_metrics)
        else:
            self.load_history()

        # Collect all patterns
        all_patterns = []
        all_patterns.extend(self.analyze_gating_patterns())
        all_patterns.extend(self.analyze_timing_patterns())
        all_patterns.extend(self.analyze_entry_patterns())
        all_patterns.extend(self.analyze_exit_patterns())
        all_patterns.extend(self.analyze_infrastructure_patterns())
        all_patterns.extend(self.analyze_regime_patterns())

        # Sort by severity (CONFIRMED first)
        severity_order = {PatternSeverity.CONFIRMED.value: 0, PatternSeverity.POSSIBLE.value: 1, PatternSeverity.NONE.value: 2}
        all_patterns.sort(key=lambda p: severity_order.get(p["severity"], 2))

        confirmed_count = sum(1 for p in all_patterns if p["severity"] == "CONFIRMED")
        possible_count = sum(1 for p in all_patterns if p["severity"] == "POSSIBLE")

        return {
            "pattern_flags": all_patterns,
            "summary": {
                "total_patterns": len(all_patterns),
                "confirmed": confirmed_count,
                "possible": possible_count,
                "sessions_analyzed": len(self._session_history),
                "green_sessions": len(self._get_green_sessions()),
            },
            "analysis_timestamp": datetime.now().isoformat(),
        }


# Singleton instance
_pattern_analyzer: Optional[PatternAnalyzer] = None


def get_pattern_analyzer() -> PatternAnalyzer:
    """Get or create the singleton PatternAnalyzer instance"""
    global _pattern_analyzer
    if _pattern_analyzer is None:
        _pattern_analyzer = PatternAnalyzer()
    return _pattern_analyzer
