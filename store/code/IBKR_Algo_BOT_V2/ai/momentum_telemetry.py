"""
Momentum Telemetry Module (ChatGPT Spec)
========================================
Centralized logging and metrics for the Momentum FSM pipeline.

Features:
- All state transitions logged with full context
- MFE/MAE (Maximum Favorable/Adverse Excursion) tracking
- Trade outcome correlation with entry conditions
- API endpoints for dashboard and analysis
- Rolling performance metrics
"""

import logging
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)

# Telemetry storage path
TELEMETRY_PATH = os.path.join(os.path.dirname(__file__), "momentum_telemetry.json")


@dataclass
class TradeRecord:
    """Complete record of a trade from entry to exit"""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime] = None

    # Entry conditions
    entry_price: float = 0
    entry_score: int = 0
    entry_grade: str = ""
    entry_veto_reasons: List[str] = field(default_factory=list)
    entry_r_5s: float = 0
    entry_r_15s: float = 0
    entry_r_30s: float = 0
    entry_accel: float = 0
    entry_buy_pressure: float = 0
    entry_vwap_position: str = ""
    entry_regime: str = ""
    gating_result: str = ""

    # Position data
    shares: int = 0
    stop_price: float = 0
    target_price: float = 0

    # Excursions
    high_price: float = 0         # Highest price during trade
    low_price: float = 0          # Lowest price during trade
    mfe: float = 0                # Maximum Favorable Excursion %
    mae: float = 0                # Maximum Adverse Excursion %

    # Exit conditions
    exit_price: float = 0
    exit_signal: str = ""
    exit_reason: str = ""
    pnl_pct: float = 0
    pnl_dollars: float = 0
    hold_time_seconds: float = 0

    # State tracking
    states_visited: List[str] = field(default_factory=list)
    transition_count: int = 0

    # Outcome
    is_winner: bool = False
    is_complete: bool = False

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['entry_time'] = self.entry_time.isoformat() if self.entry_time else None
        d['exit_time'] = self.exit_time.isoformat() if self.exit_time else None
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeRecord':
        if data.get('entry_time'):
            data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        if data.get('exit_time'):
            data['exit_time'] = datetime.fromisoformat(data['exit_time'])
        return cls(**data)


@dataclass
class TransitionLog:
    """Log entry for state transition"""
    timestamp: datetime
    symbol: str
    from_state: str
    to_state: str
    reason: str
    owner: str
    score: int = 0
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'from_state': self.from_state,
            'to_state': self.to_state,
            'reason': self.reason,
            'owner': self.owner,
            'score': self.score,
            'details': self.details
        }


@dataclass
class PerformanceMetrics:
    """Rolling performance metrics"""
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0
    total_pnl: float = 0
    avg_pnl: float = 0
    avg_winner: float = 0
    avg_loser: float = 0
    profit_factor: float = 0
    avg_mfe: float = 0
    avg_mae: float = 0
    avg_hold_time: float = 0

    # By exit reason
    exits_by_reason: Dict[str, int] = field(default_factory=dict)
    pnl_by_reason: Dict[str, float] = field(default_factory=dict)

    # By entry score
    wins_by_score_bucket: Dict[str, int] = field(default_factory=dict)
    losses_by_score_bucket: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class MomentumTelemetry:
    """
    Centralized telemetry for the Momentum FSM pipeline.

    Responsibilities:
    - Log all state transitions
    - Track MFE/MAE for each trade
    - Correlate entry conditions with outcomes
    - Provide API endpoints for analysis
    """

    def __init__(self):
        self._trades: Dict[str, TradeRecord] = {}  # trade_id -> TradeRecord
        self._active_trades: Dict[str, str] = {}   # symbol -> trade_id
        self._transition_log: List[TransitionLog] = []
        self._metrics = PerformanceMetrics()
        self._lock = threading.Lock()

        # Load persisted data
        self._load_data()

    def _load_data(self):
        """Load persisted telemetry data"""
        if os.path.exists(TELEMETRY_PATH):
            try:
                with open(TELEMETRY_PATH, 'r') as f:
                    data = json.load(f)

                # Load trades
                for trade_data in data.get('trades', []):
                    try:
                        trade = TradeRecord.from_dict(trade_data)
                        self._trades[trade.trade_id] = trade
                    except Exception as e:
                        logger.warning(f"Failed to load trade: {e}")

                # Load active trades
                self._active_trades = data.get('active_trades', {})

                # Recalculate metrics
                self._recalculate_metrics()

                logger.info(f"Loaded {len(self._trades)} trades from telemetry")

            except Exception as e:
                logger.warning(f"Failed to load telemetry: {e}")

    def _save_data(self):
        """Persist telemetry data"""
        try:
            data = {
                'trades': [t.to_dict() for t in self._trades.values()],
                'active_trades': self._active_trades,
                'last_updated': datetime.now().isoformat()
            }

            with open(TELEMETRY_PATH, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save telemetry: {e}")

    def log_transition(self, symbol: str, from_state: str, to_state: str,
                       reason: str, owner: str, score: int = 0,
                       details: Dict = None):
        """Log a state transition"""
        with self._lock:
            log_entry = TransitionLog(
                timestamp=datetime.now(),
                symbol=symbol,
                from_state=from_state,
                to_state=to_state,
                reason=reason,
                owner=owner,
                score=score,
                details=details or {}
            )
            self._transition_log.append(log_entry)

            # Keep last 1000 transitions
            if len(self._transition_log) > 1000:
                self._transition_log = self._transition_log[-1000:]

            # Update active trade states
            if symbol in self._active_trades:
                trade_id = self._active_trades[symbol]
                if trade_id in self._trades:
                    self._trades[trade_id].states_visited.append(to_state)
                    self._trades[trade_id].transition_count += 1

            logger.debug(f"TELEMETRY: {symbol} {from_state}->{to_state} ({reason})")

    def start_trade(self, symbol: str, entry_price: float, shares: int,
                    momentum_result: Any = None, gating_result: str = "") -> str:
        """
        Start tracking a new trade.

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            shares: Number of shares
            momentum_result: MomentumResult from scorer
            gating_result: Result of gating check

        Returns:
            trade_id: Unique trade identifier
        """
        with self._lock:
            trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            trade = TradeRecord(
                trade_id=trade_id,
                symbol=symbol,
                entry_time=datetime.now(),
                entry_price=entry_price,
                shares=shares,
                high_price=entry_price,
                low_price=entry_price,
                gating_result=gating_result,
                states_visited=['IN_POSITION']
            )

            # Extract momentum data if available
            if momentum_result:
                trade.entry_score = getattr(momentum_result, 'score', 0)
                trade.entry_grade = getattr(momentum_result, 'grade', '')
                if hasattr(momentum_result.grade, 'value'):
                    trade.entry_grade = momentum_result.grade.value
                trade.entry_veto_reasons = [v.value for v in getattr(momentum_result, 'veto_reasons', [])]

                if hasattr(momentum_result, 'price_urgency'):
                    pu = momentum_result.price_urgency
                    trade.entry_r_5s = getattr(pu, 'r_5s', 0)
                    trade.entry_r_15s = getattr(pu, 'r_15s', 0)
                    trade.entry_r_30s = getattr(pu, 'r_30s', 0)
                    trade.entry_accel = getattr(pu, 'accel', 0)
                    trade.entry_vwap_position = getattr(pu, 'vwap_position', '')

                if hasattr(momentum_result, 'liquidity'):
                    liq = momentum_result.liquidity
                    trade.entry_buy_pressure = getattr(liq, 'buy_pressure', 0)

            self._trades[trade_id] = trade
            self._active_trades[symbol] = trade_id

            logger.info(f"TELEMETRY: Started trade {trade_id} @ ${entry_price:.2f}")
            self._save_data()

            return trade_id

    def update_price(self, symbol: str, current_price: float):
        """
        Update price and calculate MFE/MAE.

        Call this on every price update for active trades.
        """
        with self._lock:
            if symbol not in self._active_trades:
                return

            trade_id = self._active_trades[symbol]
            if trade_id not in self._trades:
                return

            trade = self._trades[trade_id]

            # Update high/low
            if current_price > trade.high_price:
                trade.high_price = current_price
            if current_price < trade.low_price or trade.low_price == 0:
                trade.low_price = current_price

            # Calculate MFE/MAE
            if trade.entry_price > 0:
                trade.mfe = ((trade.high_price - trade.entry_price) / trade.entry_price) * 100
                trade.mae = ((trade.entry_price - trade.low_price) / trade.entry_price) * 100

    def complete_trade(self, symbol: str, exit_price: float,
                       exit_signal: str, exit_reason: str):
        """
        Complete a trade and calculate final metrics.

        Args:
            symbol: Stock symbol
            exit_price: Exit price
            exit_signal: Signal that triggered exit
            exit_reason: Detailed reason
        """
        with self._lock:
            if symbol not in self._active_trades:
                logger.warning(f"No active trade for {symbol}")
                return

            trade_id = self._active_trades[symbol]
            if trade_id not in self._trades:
                return

            trade = self._trades[trade_id]
            trade.exit_time = datetime.now()
            trade.exit_price = exit_price
            trade.exit_signal = exit_signal
            trade.exit_reason = exit_reason
            trade.is_complete = True

            # Calculate P&L
            if trade.entry_price > 0:
                trade.pnl_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
                trade.pnl_dollars = (exit_price - trade.entry_price) * trade.shares

            trade.is_winner = trade.pnl_pct > 0

            # Calculate hold time
            trade.hold_time_seconds = (trade.exit_time - trade.entry_time).total_seconds()

            # Remove from active
            del self._active_trades[symbol]

            logger.info(
                f"TELEMETRY: Completed {trade_id} | "
                f"P&L: {trade.pnl_pct:+.2f}% (${trade.pnl_dollars:+.2f}) | "
                f"MFE: {trade.mfe:.2f}% | MAE: {trade.mae:.2f}% | "
                f"Exit: {exit_signal}"
            )

            # Update metrics
            self._recalculate_metrics()
            self._save_data()

    def _recalculate_metrics(self):
        """Recalculate performance metrics from completed trades"""
        completed = [t for t in self._trades.values() if t.is_complete]

        if not completed:
            return

        self._metrics.total_trades = len(completed)
        self._metrics.winners = sum(1 for t in completed if t.is_winner)
        self._metrics.losers = self._metrics.total_trades - self._metrics.winners

        if self._metrics.total_trades > 0:
            self._metrics.win_rate = (self._metrics.winners / self._metrics.total_trades) * 100

        # P&L metrics
        self._metrics.total_pnl = sum(t.pnl_dollars for t in completed)
        self._metrics.avg_pnl = self._metrics.total_pnl / len(completed)

        winners = [t for t in completed if t.is_winner]
        losers = [t for t in completed if not t.is_winner]

        if winners:
            self._metrics.avg_winner = sum(t.pnl_dollars for t in winners) / len(winners)
        if losers:
            self._metrics.avg_loser = sum(t.pnl_dollars for t in losers) / len(losers)

        # Profit factor
        gross_profit = sum(t.pnl_dollars for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl_dollars for t in losers)) if losers else 0
        if gross_loss > 0:
            self._metrics.profit_factor = gross_profit / gross_loss

        # MFE/MAE
        self._metrics.avg_mfe = sum(t.mfe for t in completed) / len(completed)
        self._metrics.avg_mae = sum(t.mae for t in completed) / len(completed)

        # Hold time
        self._metrics.avg_hold_time = sum(t.hold_time_seconds for t in completed) / len(completed)

        # By exit reason
        self._metrics.exits_by_reason = defaultdict(int)
        self._metrics.pnl_by_reason = defaultdict(float)
        for t in completed:
            reason = t.exit_signal or 'UNKNOWN'
            self._metrics.exits_by_reason[reason] += 1
            self._metrics.pnl_by_reason[reason] += t.pnl_dollars

        # By score bucket
        self._metrics.wins_by_score_bucket = defaultdict(int)
        self._metrics.losses_by_score_bucket = defaultdict(int)
        for t in completed:
            bucket = self._get_score_bucket(t.entry_score)
            if t.is_winner:
                self._metrics.wins_by_score_bucket[bucket] += 1
            else:
                self._metrics.losses_by_score_bucket[bucket] += 1

    def _get_score_bucket(self, score: int) -> str:
        """Get score bucket for analysis"""
        if score >= 90:
            return "90-100"
        elif score >= 80:
            return "80-89"
        elif score >= 70:
            return "70-79"
        elif score >= 60:
            return "60-69"
        elif score >= 50:
            return "50-59"
        else:
            return "0-49"

    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        return self._metrics.to_dict()

    def get_active_trades(self) -> List[Dict]:
        """Get all active trades"""
        with self._lock:
            result = []
            for symbol, trade_id in self._active_trades.items():
                if trade_id in self._trades:
                    result.append(self._trades[trade_id].to_dict())
            return result

    def get_completed_trades(self, limit: int = 50) -> List[Dict]:
        """Get recent completed trades"""
        with self._lock:
            completed = [t for t in self._trades.values() if t.is_complete]
            # Sort by exit time, most recent first
            completed.sort(key=lambda t: t.exit_time or datetime.min, reverse=True)
            return [t.to_dict() for t in completed[:limit]]

    def get_transitions(self, limit: int = 100, symbol: str = None) -> List[Dict]:
        """Get recent transitions, optionally filtered by symbol"""
        with self._lock:
            logs = self._transition_log
            if symbol:
                logs = [l for l in logs if l.symbol == symbol]
            return [l.to_dict() for l in logs[-limit:]]

    def get_mfe_mae_analysis(self) -> Dict:
        """
        Get MFE/MAE analysis for optimizing exits.

        Returns analysis of how MFE relates to final outcomes.
        """
        with self._lock:
            completed = [t for t in self._trades.values() if t.is_complete]

            if not completed:
                return {'error': 'No completed trades'}

            winners = [t for t in completed if t.is_winner]
            losers = [t for t in completed if not t.is_winner]

            return {
                'total_trades': len(completed),
                'winners': {
                    'count': len(winners),
                    'avg_mfe': sum(t.mfe for t in winners) / len(winners) if winners else 0,
                    'avg_mae': sum(t.mae for t in winners) / len(winners) if winners else 0,
                    'avg_exit_vs_mfe': sum(t.pnl_pct / t.mfe * 100 for t in winners if t.mfe > 0) / len(winners) if winners else 0,
                },
                'losers': {
                    'count': len(losers),
                    'avg_mfe': sum(t.mfe for t in losers) / len(losers) if losers else 0,
                    'avg_mae': sum(t.mae for t in losers) / len(losers) if losers else 0,
                    'mfe_before_loss': sum(1 for t in losers if t.mfe > 0.5),  # Trades that went green first
                },
                'optimal_trail_suggestion': self._suggest_optimal_trail(completed),
            }

    def _suggest_optimal_trail(self, trades: List[TradeRecord]) -> Dict:
        """Suggest optimal trailing stop based on MFE analysis"""
        if not trades:
            return {}

        # For winners, see what % of MFE was captured
        winners = [t for t in trades if t.is_winner and t.mfe > 0]
        if not winners:
            return {'message': 'Not enough winners to analyze'}

        # Average capture rate
        capture_rates = [(t.pnl_pct / t.mfe) for t in winners if t.mfe > 0]
        avg_capture = sum(capture_rates) / len(capture_rates) if capture_rates else 0

        # For losers, see how far they went before reversing
        losers = [t for t in trades if not t.is_winner]
        losers_with_mfe = [t for t in losers if t.mfe > 0.5]

        return {
            'avg_mfe_capture': round(avg_capture * 100, 1),
            'losers_that_went_green': len(losers_with_mfe),
            'suggestion': f"Consider trailing at {round((1 - avg_capture) * 100, 1)}% below peak" if avg_capture > 0 else "Need more data"
        }

    def get_score_analysis(self) -> Dict:
        """Analyze win rate by entry score"""
        with self._lock:
            return {
                'wins_by_score': dict(self._metrics.wins_by_score_bucket),
                'losses_by_score': dict(self._metrics.losses_by_score_bucket),
                'win_rate_by_score': self._calc_win_rate_by_score()
            }

    def _calc_win_rate_by_score(self) -> Dict[str, float]:
        """Calculate win rate for each score bucket"""
        result = {}
        for bucket in self._metrics.wins_by_score_bucket.keys():
            wins = self._metrics.wins_by_score_bucket.get(bucket, 0)
            losses = self._metrics.losses_by_score_bucket.get(bucket, 0)
            total = wins + losses
            if total > 0:
                result[bucket] = round((wins / total) * 100, 1)
        return result

    def clear(self):
        """Clear all telemetry data (for testing)"""
        with self._lock:
            self._trades.clear()
            self._active_trades.clear()
            self._transition_log.clear()
            self._metrics = PerformanceMetrics()
            if os.path.exists(TELEMETRY_PATH):
                os.remove(TELEMETRY_PATH)


# Singleton instance
_telemetry: Optional[MomentumTelemetry] = None


def get_telemetry() -> MomentumTelemetry:
    """Get singleton telemetry instance"""
    global _telemetry
    if _telemetry is None:
        _telemetry = MomentumTelemetry()
    return _telemetry


# FastAPI routes (can be imported by main API)
def create_telemetry_routes(app):
    """
    Create FastAPI routes for telemetry endpoints.

    Call from main API: create_telemetry_routes(app)
    """
    from fastapi import APIRouter
    router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])

    @router.get("/metrics")
    async def get_metrics():
        """Get performance metrics"""
        return get_telemetry().get_metrics()

    @router.get("/active")
    async def get_active():
        """Get active trades"""
        return get_telemetry().get_active_trades()

    @router.get("/completed")
    async def get_completed(limit: int = 50):
        """Get completed trades"""
        return get_telemetry().get_completed_trades(limit)

    @router.get("/transitions")
    async def get_transitions(limit: int = 100, symbol: str = None):
        """Get state transitions"""
        return get_telemetry().get_transitions(limit, symbol)

    @router.get("/mfe-mae")
    async def get_mfe_mae():
        """Get MFE/MAE analysis"""
        return get_telemetry().get_mfe_mae_analysis()

    @router.get("/score-analysis")
    async def get_score_analysis():
        """Get win rate by score analysis"""
        return get_telemetry().get_score_analysis()

    app.include_router(router)


if __name__ == "__main__":
    # Test the telemetry module
    print("=" * 60)
    print("MOMENTUM TELEMETRY TEST")
    print("=" * 60)

    telemetry = MomentumTelemetry()

    # Simulate a winning trade
    print("\n--- Simulating Winning Trade ---")
    trade_id = telemetry.start_trade(
        symbol="WIN_TEST",
        entry_price=10.00,
        shares=100,
        gating_result="APPROVED"
    )
    print(f"Started trade: {trade_id}")

    # Simulate price movement
    for price in [10.10, 10.25, 10.50, 10.75, 10.60, 10.45]:
        telemetry.update_price("WIN_TEST", price)

    telemetry.complete_trade("WIN_TEST", 10.45, "TRAILING_STOP", "Trail triggered")

    # Simulate a losing trade
    print("\n--- Simulating Losing Trade ---")
    trade_id = telemetry.start_trade(
        symbol="LOSS_TEST",
        entry_price=10.00,
        shares=100,
        gating_result="APPROVED"
    )

    for price in [10.05, 9.90, 9.75, 9.50]:
        telemetry.update_price("LOSS_TEST", price)

    telemetry.complete_trade("LOSS_TEST", 9.50, "STOP_LOSS", "Stop hit")

    # Print metrics
    print("\n--- Performance Metrics ---")
    metrics = telemetry.get_metrics()
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Total P&L: ${metrics['total_pnl']:.2f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Avg MFE: {metrics['avg_mfe']:.2f}%")
    print(f"Avg MAE: {metrics['avg_mae']:.2f}%")

    # MFE/MAE analysis
    print("\n--- MFE/MAE Analysis ---")
    mfe_mae = telemetry.get_mfe_mae_analysis()
    print(json.dumps(mfe_mae, indent=2))

    print("\n" + "=" * 60)
    print("TELEMETRY TEST COMPLETE")
    print("=" * 60)
