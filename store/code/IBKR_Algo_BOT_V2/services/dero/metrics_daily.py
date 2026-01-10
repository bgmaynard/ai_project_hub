"""
DERO Daily Metrics Aggregator

Aggregates daily metrics from trading bot event logs and trade history.
This is READ-ONLY and does not affect trading execution.

Metrics Categories:
1. Infrastructure Health (feed uptime, errors, reconnects)
2. Discovery (scanner candidates, unique symbols)
3. Pipeline Flow (FSM state progression)
4. Gating (allowed vs blocked decisions)
5. Outcomes (trades, wins/losses, R-multiples)
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class DataHealthStatus(Enum):
    """Data health color coding"""
    GREEN = "GREEN"   # All systems operational
    YELLOW = "YELLOW" # Some issues but operational
    RED = "RED"       # Critical issues


class DailyMetricsAggregator:
    """
    Aggregates daily metrics from bot data sources.

    All operations are read-only. Sources:
    - Event logs (JSONL)
    - Trade history
    - Gating decisions
    - FSM state transitions
    """

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(".")
        self._metrics: Dict[str, Any] = {}

    def _load_jsonl(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load events from a JSONL file"""
        events = []
        if not filepath.exists():
            return events

        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.warning(f"Error loading JSONL {filepath}: {e}")

        return events

    def _load_json(self, filepath: Path) -> Any:
        """Load data from a JSON file"""
        if not filepath.exists():
            return None

        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading JSON {filepath}: {e}")
            return None

    def aggregate_infrastructure_health(
        self,
        events: List[Dict[str, Any]],
        target_date: date
    ) -> Dict[str, Any]:
        """
        Aggregate infrastructure health metrics.

        Looks for: errors, reconnects, feed status changes
        """
        error_count = 0
        reconnect_count = 0
        feed_drops = 0
        uptime_minutes = 0
        total_minutes = 0

        # Count event types
        feed_active = False
        last_event_time = None

        for event in events:
            event_type = event.get("type", event.get("event", ""))
            ts = event.get("timestamp")

            if ts:
                try:
                    if isinstance(ts, str):
                        event_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    else:
                        event_time = ts

                    # Only process events from target date
                    if event_time.date() != target_date:
                        continue

                    # Track uptime
                    if last_event_time:
                        delta = (event_time - last_event_time).total_seconds() / 60
                        total_minutes += delta
                        if feed_active:
                            uptime_minutes += delta

                    last_event_time = event_time
                except:
                    pass

            # Categorize events
            event_lower = str(event_type).lower()

            if "error" in event_lower or "exception" in event_lower:
                error_count += 1
            elif "reconnect" in event_lower or "reconnection" in event_lower:
                reconnect_count += 1
            elif "disconnect" in event_lower or "dropped" in event_lower:
                feed_drops += 1
                feed_active = False
            elif "connect" in event_lower or "started" in event_lower:
                feed_active = True

        # Calculate uptime percentage
        uptime_pct = 100.0
        if total_minutes > 0:
            uptime_pct = (uptime_minutes / total_minutes) * 100

        # Determine health color
        health = DataHealthStatus.GREEN
        if error_count > 10 or reconnect_count > 5:
            health = DataHealthStatus.RED
        elif error_count > 3 or reconnect_count > 2:
            health = DataHealthStatus.YELLOW

        return {
            "data_health": health.value,
            "feed_uptime_pct": round(uptime_pct, 1),
            "error_count": error_count,
            "reconnect_count": reconnect_count,
            "feed_drops": feed_drops,
            "total_events": len(events),
        }

    def aggregate_discovery(
        self,
        events: List[Dict[str, Any]],
        target_date: date
    ) -> Dict[str, Any]:
        """
        Aggregate discovery/scanner metrics.

        Tracks: candidates per scanner, unique symbols, churn
        """
        scanner_candidates: Dict[str, int] = {}
        unique_symbols = set()
        added_symbols = set()
        removed_symbols = set()

        for event in events:
            event_type = event.get("type", event.get("event", ""))
            symbol = event.get("symbol")
            scanner = event.get("scanner", "unknown")

            # DERO Event Sink format: SCANNER_CANDIDATE
            if event_type == "SCANNER_CANDIDATE":
                if scanner not in scanner_candidates:
                    scanner_candidates[scanner] = 0
                scanner_candidates[scanner] += 1
                if symbol:
                    unique_symbols.add(symbol)

            # Legacy format: scanner discovery events
            elif "scanner" in str(event_type).lower() or "discovery" in str(event_type).lower():
                if scanner not in scanner_candidates:
                    scanner_candidates[scanner] = 0
                scanner_candidates[scanner] += 1

                if symbol:
                    unique_symbols.add(symbol)

            # DERO Event Sink format: WATCHLIST_UPDATE
            if event_type == "WATCHLIST_UPDATE":
                action = event.get("action", "")
                if action == "ADD" and symbol:
                    added_symbols.add(symbol)
                elif action == "REMOVE" and symbol:
                    removed_symbols.add(symbol)
            # Legacy format: Watchlist add/remove
            elif "add" in str(event_type).lower() and symbol:
                added_symbols.add(symbol)
            elif "remove" in str(event_type).lower() and symbol:
                removed_symbols.add(symbol)

            # Any symbol mention
            if symbol:
                unique_symbols.add(symbol)

        return {
            "candidates_by_scanner": scanner_candidates,
            "total_candidates": sum(scanner_candidates.values()),
            "unique_symbols": len(unique_symbols),
            "symbols_added": len(added_symbols),
            "symbols_removed": len(removed_symbols),
            "churn": len(added_symbols) + len(removed_symbols),
            "symbol_list": sorted(list(unique_symbols))[:50],  # Top 50
        }

    def aggregate_pipeline_flow(
        self,
        events: List[Dict[str, Any]],
        target_date: date
    ) -> Dict[str, Any]:
        """
        Aggregate FSM pipeline flow metrics.

        Tracks symbols reaching each state: EXPANSION, FIRST_PULLBACK, ENTRY_WINDOW
        """
        state_counts: Dict[str, int] = {
            "EXPANSION": 0,
            "FIRST_PULLBACK": 0,
            "ENTRY_WINDOW": 0,
            "CONFIRMED": 0,
            "IGNITION": 0,
            "GATED": 0,
            "DEAD": 0,
        }
        symbols_by_state: Dict[str, set] = {k: set() for k in state_counts}
        transitions = []

        for event in events:
            event_type = event.get("type", event.get("event", ""))
            symbol = event.get("symbol")
            new_state = event.get("new_state", event.get("to_state", event.get("state")))
            old_state = event.get("old_state", event.get("from_state"))

            # DERO Event Sink format: FSM_TRANSITION
            if event_type == "FSM_TRANSITION":
                if new_state and new_state.upper() in state_counts:
                    state_key = new_state.upper()
                    state_counts[state_key] += 1
                    if symbol:
                        symbols_by_state[state_key].add(symbol)

                if old_state and new_state:
                    transitions.append({
                        "symbol": symbol,
                        "from": old_state,
                        "to": new_state
                    })

            # Legacy format: FSM state transitions
            elif "state" in str(event_type).lower() or "transition" in str(event_type).lower():
                if new_state and new_state.upper() in state_counts:
                    state_key = new_state.upper()
                    state_counts[state_key] += 1
                    if symbol:
                        symbols_by_state[state_key].add(symbol)

                if old_state and new_state:
                    transitions.append({
                        "symbol": symbol,
                        "from": old_state,
                        "to": new_state
                    })

            # Momentum events
            if "momentum" in str(event_type).lower():
                if "ignit" in str(event_type).lower():
                    state_counts["IGNITION"] += 1
                    if symbol:
                        symbols_by_state["IGNITION"].add(symbol)
                elif "confirm" in str(event_type).lower():
                    state_counts["CONFIRMED"] += 1
                    if symbol:
                        symbols_by_state["CONFIRMED"].add(symbol)

        return {
            "state_counts": state_counts,
            "symbols_by_state": {k: list(v) for k, v in symbols_by_state.items()},
            "total_transitions": len(transitions),
            "funnel": {
                "discovery_to_ignition": state_counts.get("IGNITION", 0),
                "ignition_to_confirmed": state_counts.get("CONFIRMED", 0),
                "confirmed_to_gated": state_counts.get("GATED", 0),
            }
        }

    def aggregate_gating(
        self,
        events: List[Dict[str, Any]],
        target_date: date
    ) -> Dict[str, Any]:
        """
        Aggregate gating decision metrics.

        Tracks: allowed vs blocked, top block reasons
        """
        allowed_count = 0
        blocked_count = 0
        block_reasons: Dict[str, int] = {}
        gated_symbols = set()
        blocked_symbols = set()

        for event in events:
            event_type = event.get("type", event.get("event", ""))
            symbol = event.get("symbol")
            decision = event.get("decision", event.get("result"))
            reason = event.get("reason", event.get("block_reason", "unknown"))

            # DERO Event Sink format: GATE_DECISION
            if event_type == "GATE_DECISION":
                if decision == "APPROVED":
                    allowed_count += 1
                    if symbol:
                        gated_symbols.add(symbol)
                elif decision == "VETOED":
                    blocked_count += 1
                    if symbol:
                        blocked_symbols.add(symbol)
                    if reason:
                        block_reasons[reason] = block_reasons.get(reason, 0) + 1

            # Legacy format: Gating events
            elif "gat" in str(event_type).lower() or "veto" in str(event_type).lower():
                if decision:
                    decision_lower = str(decision).lower()
                    if "allow" in decision_lower or "approve" in decision_lower or "pass" in decision_lower:
                        allowed_count += 1
                        if symbol:
                            gated_symbols.add(symbol)
                    elif "block" in decision_lower or "veto" in decision_lower or "reject" in decision_lower:
                        blocked_count += 1
                        if symbol:
                            blocked_symbols.add(symbol)
                        if reason:
                            block_reasons[reason] = block_reasons.get(reason, 0) + 1

        # Top 5 block reasons
        top_reasons = sorted(block_reasons.items(), key=lambda x: x[1], reverse=True)[:5]

        total = allowed_count + blocked_count
        approval_rate = (allowed_count / total * 100) if total > 0 else 0

        return {
            "total_decisions": total,
            "allowed": allowed_count,
            "blocked": blocked_count,
            "approval_rate_pct": round(approval_rate, 1),
            "top_block_reasons": [{"reason": r, "count": c} for r, c in top_reasons],
            "gated_symbols": sorted(list(gated_symbols)),
            "blocked_symbols": sorted(list(blocked_symbols)),
        }

    def aggregate_outcomes(
        self,
        trades: List[Dict[str, Any]],
        target_date: date
    ) -> Dict[str, Any]:
        """
        Aggregate trade outcome metrics.

        Tracks: trade count, wins/losses, R-multiples, MAE/MFE, slippage
        """
        wins = 0
        losses = 0
        breakeven = 0
        total_pnl = 0.0
        r_multiples = []
        mae_values = []
        mfe_values = []
        slippage_values = []
        trade_details = []

        for trade in trades:
            # Check if trade is from target date
            entry_time = trade.get("entry_time")
            if entry_time:
                try:
                    if isinstance(entry_time, str):
                        trade_date = datetime.fromisoformat(entry_time.replace("Z", "+00:00")).date()
                    else:
                        trade_date = entry_time.date()

                    if trade_date != target_date:
                        continue
                except:
                    pass

            # Extract metrics
            pnl = trade.get("pnl", 0)
            r_multiple = trade.get("r_multiple", trade.get("pnl_percent", 0) / 100)
            mae = trade.get("max_drawdown_percent", trade.get("mae", 0))
            mfe = trade.get("max_gain_percent", trade.get("mfe", 0))
            slippage = trade.get("slippage", 0)

            # Categorize outcome
            if pnl > 0.01:
                wins += 1
            elif pnl < -0.01:
                losses += 1
            else:
                breakeven += 1

            total_pnl += pnl

            if r_multiple:
                r_multiples.append(r_multiple)
            if mae:
                mae_values.append(mae)
            if mfe:
                mfe_values.append(mfe)
            if slippage:
                slippage_values.append(slippage)

            # Store trade detail
            trade_details.append({
                "symbol": trade.get("symbol"),
                "pnl": pnl,
                "r_multiple": r_multiple,
                "entry_time": str(entry_time) if entry_time else None,
            })

        total_trades = wins + losses + breakeven
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "win_rate_pct": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_r": round(sum(r_multiples) / len(r_multiples), 3) if r_multiples else 0,
            "avg_mae": round(sum(mae_values) / len(mae_values), 2) if mae_values else 0,
            "avg_mfe": round(sum(mfe_values) / len(mfe_values), 2) if mfe_values else 0,
            "avg_slippage": round(sum(slippage_values) / len(slippage_values), 4) if slippage_values else 0,
            "trade_details": trade_details[:20],  # Last 20 trades
        }

    def aggregate_all(
        self,
        target_date: date,
        events: Optional[List[Dict[str, Any]]] = None,
        trades: Optional[List[Dict[str, Any]]] = None,
        mode: str = "PAPER"
    ) -> Dict[str, Any]:
        """
        Aggregate all daily metrics.

        Args:
            target_date: Date to aggregate metrics for
            events: List of bot events (or will try to load from files)
            trades: List of trades (or will try to load from files)
            mode: Trading mode ("PAPER" or "LIVE")

        Returns:
            Complete daily metrics dictionary
        """
        # Load data if not provided
        if events is None:
            events = self._load_events_for_date(target_date)

        if trades is None:
            trades = self._load_trades_for_date(target_date)

        # Aggregate each category
        infrastructure = self.aggregate_infrastructure_health(events, target_date)
        discovery = self.aggregate_discovery(events, target_date)
        pipeline = self.aggregate_pipeline_flow(events, target_date)
        gating = self.aggregate_gating(events, target_date)
        outcomes = self.aggregate_outcomes(trades, target_date)

        return {
            "date": target_date.isoformat(),
            "mode": mode,
            "data_health": infrastructure["data_health"],
            "infrastructure": infrastructure,
            "discovery": discovery,
            "pipeline": pipeline,
            "gating": gating,
            "outcomes": outcomes,
            "artifacts": {
                "raw_events_count": len(events),
                "raw_trades_count": len(trades),
                "report_version": "1.0",
                "generated_at": datetime.now().isoformat(),
            }
        }

    def _load_events_for_date(self, target_date: date) -> List[Dict[str, Any]]:
        """Load events for a specific date from various sources"""
        events = []

        # Primary source: DERO Event Sink (bot_events_YYYY-MM-DD.jsonl)
        event_sink_log = self.base_path / "logs" / "events" / f"bot_events_{target_date.isoformat()}.jsonl"
        if event_sink_log.exists():
            raw_events = self._load_jsonl(event_sink_log)
            for event in raw_events:
                # Flatten payload into event for easier processing
                payload = event.get("payload", {})
                flattened = {
                    "type": event.get("type"),
                    "timestamp": event.get("timestamp"),
                    **payload
                }
                events.append(flattened)
            logger.info(f"Loaded {len(raw_events)} events from event sink for {target_date}")

        # Fallback: Legacy event log format
        event_log = self.base_path / "logs" / f"events_{target_date.isoformat()}.jsonl"
        if event_log.exists():
            events.extend(self._load_jsonl(event_log))

        # Fallback: Combined dero events log
        dero_log = self.base_path / "logs" / "dero_events.jsonl"
        if dero_log.exists():
            all_events = self._load_jsonl(dero_log)
            for event in all_events:
                ts = event.get("timestamp")
                if ts:
                    try:
                        if isinstance(ts, str):
                            event_date = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
                        else:
                            event_date = ts.date()
                        if event_date == target_date:
                            events.append(event)
                    except:
                        pass

        return events

    def _load_trades_for_date(self, target_date: date) -> List[Dict[str, Any]]:
        """Load trades for a specific date"""
        trades = []

        # Try loading from scalper trades
        trades_file = self.base_path / "ai" / "scalper_trades.json"
        if trades_file.exists():
            all_trades = self._load_json(trades_file)
            if isinstance(all_trades, list):
                for trade in all_trades:
                    entry_time = trade.get("entry_time")
                    if entry_time:
                        try:
                            if isinstance(entry_time, str):
                                trade_date = datetime.fromisoformat(entry_time.replace("Z", "+00:00")).date()
                            else:
                                trade_date = entry_time.date()
                            if trade_date == target_date:
                                trades.append(trade)
                        except:
                            pass

        return trades


# Singleton instance
_daily_metrics_aggregator: Optional[DailyMetricsAggregator] = None


def get_daily_metrics_aggregator(base_path: Optional[Path] = None) -> DailyMetricsAggregator:
    """Get or create the singleton DailyMetricsAggregator instance"""
    global _daily_metrics_aggregator
    if _daily_metrics_aggregator is None:
        _daily_metrics_aggregator = DailyMetricsAggregator(base_path)
    return _daily_metrics_aggregator
