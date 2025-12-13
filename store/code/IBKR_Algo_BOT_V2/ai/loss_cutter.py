"""
Loss Cutter Module - WARRIOR TRADING Style Capital Preservation

Core Philosophy: "The only safe position is FLAT"
- Cut losers in 30 SECONDS - NO HESITATION
- Don't hold hoping for recovery - EVER
- Preserve capital for next opportunity
- Quick scalps: Take 3-5% profit, 1% max loss
- If it's not working immediately, GET OUT

WARRIOR RULES:
1. Max loss per trade: -1% (scalp style)
2. Cut losers within 30 seconds if wrong
3. Take profits at 3-5% - don't get greedy
4. Trail winners tight after 2%
5. Done trading by 11 AM unless news catalyst

This module provides:
1. Real-time position monitoring
2. AGGRESSIVE loss cutting based on Warrior thresholds
3. Time-based position evaluation (quick scalp style)
4. Capital preservation metrics
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pytz

logger = logging.getLogger(__name__)


@dataclass
class PositionEvaluation:
    """Evaluation result for a position"""
    symbol: str
    action: str  # "HOLD", "CUT", "TAKE_PROFIT", "TRAIL_STOP"
    reason: str
    urgency: int  # 1-5, 5 = immediate action
    current_pnl: float
    current_pnl_pct: float
    time_held: float  # hours
    recommendation: str


class LossCutter:
    """
    WARRIOR TRADING Style - Aggressive loss cutting for capital preservation.

    WARRIOR RULES:
    1. Max loss per position: -1% (cut IMMEDIATELY - no exceptions!)
    2. Time decay: If not profitable after 5 min, GET OUT
    3. Quick scalps: 15 min max hold time
    4. Take profits: 3% partial, 5% full - DON'T GET GREEDY
    5. Done by 11 AM unless news catalyst
    6. Cut losers in 30 SECONDS - NO HESITATION

    The only safe position is FLAT.
    Get in, get out, get paid!
    """

    def __init__(self):
        self.et_tz = pytz.timezone('US/Eastern')

        # MOMENTUM ONLY - ULTRA TIGHT STOPS (NO HOLD AND HOPE)
        self.max_loss_pct = -1.0           # Cut at -1% loss (HARD LIMIT)
        self.alert_loss_pct = -0.5         # Alert at -0.5% loss
        self.quick_cut_pct = -0.25         # Quick cut for scalps - GET OUT FAST
        self.time_threshold_minutes = 5    # If not profitable after 5 min = MOMENTUM FAILED

        # QUICK PROFIT TAKING (Don't get greedy - take the money!)
        self.min_profit_to_trail = 1.5     # Start trailing at +1.5% (tightened from 2%)
        self.take_partial_at = 2.5         # Take partial at +2.5% (tightened from 3%)
        self.take_full_at = 4.0            # Take full profit at +4% (tightened from 5%)

        # Position entry times tracking
        self._entry_times: Dict[str, datetime] = {}

        # Recent cuts for learning
        self._cut_history: List[Dict] = []

        # MOMENTUM TIMING - GET IN GET OUT
        self.max_hold_minutes = 15         # Max 15 min hold for momentum plays
        self.stale_minutes = 5             # 5 min not profitable = STALE = EXIT
        self.morning_cutoff = 11           # Done trading by 11 AM (chop zone after)

        logger.info("LossCutter initialized - MOMENTUM ONLY: 1% max loss, 5min stale rule, 15min max hold")

    def evaluate_position(self, position: Dict, market_data: Dict = None) -> PositionEvaluation:
        """
        Evaluate a single position for potential cutting.

        Args:
            position: Dict with keys: symbol, quantity, avg_price, current_price, unrealized_pl, unrealized_plpc
            market_data: Optional market context

        Returns:
            PositionEvaluation with recommended action
        """
        symbol = position.get("symbol", "UNKNOWN")
        pnl = float(position.get("unrealized_pl", 0))
        pnl_pct = float(position.get("unrealized_plpc", 0)) * 100  # Convert to percentage
        qty = float(position.get("quantity", 0))

        # Track entry time if not already tracked
        if symbol not in self._entry_times:
            self._entry_times[symbol] = datetime.now(self.et_tz)

        entry_time = self._entry_times.get(symbol, datetime.now(self.et_tz))
        time_held_hours = (datetime.now(self.et_tz) - entry_time).total_seconds() / 3600

        # Default evaluation
        action = "HOLD"
        reason = ""
        urgency = 1
        recommendation = ""

        # === LOSS CUTTING RULES ===

        # Rule 1: Hard stop - CUT at max loss
        if pnl_pct <= self.max_loss_pct:
            action = "CUT"
            reason = f"MAX LOSS TRIGGERED: {pnl_pct:.1f}% loss exceeds {self.max_loss_pct}% threshold"
            urgency = 5  # IMMEDIATE
            recommendation = f"CLOSE IMMEDIATELY - Cut loss at {pnl_pct:.1f}% to preserve capital"

        # Rule 2: Alert level - getting close to max loss
        elif pnl_pct <= self.alert_loss_pct:
            action = "CUT"
            reason = f"ALERT LOSS: {pnl_pct:.1f}% - approaching max loss threshold"
            urgency = 4
            recommendation = f"STRONGLY consider closing - Don't let this hit {self.max_loss_pct}%"

        # Rule 3: Time decay - not profitable after threshold
        elif pnl_pct < 0 and time_held_hours > (self.time_threshold_minutes / 60):
            action = "CUT"
            reason = f"TIME DECAY: Held {time_held_hours:.1f}h with {pnl_pct:.1f}% loss - thesis not working"
            urgency = 3
            recommendation = "Setup not working as expected - cut and move on"

        # Rule 4: Quick scalp cut
        elif pnl_pct <= self.quick_cut_pct and time_held_hours < 0.25:  # Within 15 min
            action = "CUT"
            reason = f"QUICK CUT: Scalp down {pnl_pct:.1f}% - momentum not confirmed"
            urgency = 3
            recommendation = "Quick cut on failed scalp entry - preserve capital"

        # Rule 5: End of day - no overnight losers
        elif self._is_near_close() and pnl_pct < 0:
            action = "CUT"
            reason = f"END OF DAY: Don't hold {pnl_pct:.1f}% loser overnight"
            urgency = 4
            recommendation = "Close losing position before market close - flat is safe"

        # === PROFIT MANAGEMENT RULES ===

        elif pnl_pct >= self.take_partial_at:
            action = "TAKE_PROFIT"
            reason = f"PROFIT TARGET: +{pnl_pct:.1f}% - consider taking partial"
            urgency = 2
            recommendation = f"Take partial profit at +{pnl_pct:.1f}% - let rest run with trail"

        elif pnl_pct >= self.min_profit_to_trail:
            action = "TRAIL_STOP"
            reason = f"PROFITABLE: +{pnl_pct:.1f}% - protect gains with trail"
            urgency = 2
            recommendation = f"Set trailing stop to protect +{pnl_pct:.1f}% gain"

        else:
            action = "HOLD"
            reason = f"Position at {pnl_pct:.1f}% - within tolerance"
            urgency = 1
            recommendation = "Monitor - no immediate action needed"

        return PositionEvaluation(
            symbol=symbol,
            action=action,
            reason=reason,
            urgency=urgency,
            current_pnl=pnl,
            current_pnl_pct=pnl_pct,
            time_held=time_held_hours,
            recommendation=recommendation
        )

    def evaluate_all_positions(self, positions: List[Dict]) -> Dict:
        """
        Evaluate all positions and generate action list.

        Returns prioritized list of actions needed.
        """
        evaluations = []
        immediate_cuts = []
        warning_cuts = []
        profit_takes = []

        for position in positions:
            eval_result = self.evaluate_position(position)
            evaluations.append(eval_result)

            if eval_result.action == "CUT":
                if eval_result.urgency >= 4:
                    immediate_cuts.append(eval_result)
                else:
                    warning_cuts.append(eval_result)
            elif eval_result.action in ["TAKE_PROFIT", "TRAIL_STOP"]:
                profit_takes.append(eval_result)

        # Sort by urgency (highest first)
        immediate_cuts.sort(key=lambda x: x.urgency, reverse=True)
        warning_cuts.sort(key=lambda x: x.urgency, reverse=True)

        # Calculate total exposure
        total_loss = sum(e.current_pnl for e in evaluations if e.current_pnl < 0)
        total_profit = sum(e.current_pnl for e in evaluations if e.current_pnl > 0)

        return {
            "summary": {
                "total_positions": len(positions),
                "positions_to_cut_immediately": len(immediate_cuts),
                "positions_warning": len(warning_cuts),
                "positions_profitable": len(profit_takes),
                "total_unrealized_loss": total_loss,
                "total_unrealized_profit": total_profit,
                "net_pnl": total_profit + total_loss
            },
            "immediate_actions": [self._eval_to_dict(e) for e in immediate_cuts],
            "warnings": [self._eval_to_dict(e) for e in warning_cuts],
            "profit_management": [self._eval_to_dict(e) for e in profit_takes],
            "all_evaluations": [self._eval_to_dict(e) for e in evaluations]
        }

    def get_positions_to_cut(self, positions: List[Dict]) -> List[str]:
        """Get list of symbols that should be cut immediately"""
        cuts = []
        for position in positions:
            eval_result = self.evaluate_position(position)
            if eval_result.action == "CUT" and eval_result.urgency >= 3:
                cuts.append(eval_result.symbol)
        return cuts

    def _is_near_close(self) -> bool:
        """Check if we're within 30 minutes of market close"""
        now_et = datetime.now(self.et_tz)
        close_time = now_et.replace(hour=16, minute=0, second=0)

        if now_et.weekday() >= 5:
            return False

        time_to_close = (close_time - now_et).total_seconds() / 60
        return 0 < time_to_close < 30

    def _eval_to_dict(self, eval_result: PositionEvaluation) -> Dict:
        """Convert PositionEvaluation to dict"""
        return {
            "symbol": eval_result.symbol,
            "action": eval_result.action,
            "reason": eval_result.reason,
            "urgency": eval_result.urgency,
            "current_pnl": eval_result.current_pnl,
            "current_pnl_pct": eval_result.current_pnl_pct,
            "time_held_hours": eval_result.time_held,
            "recommendation": eval_result.recommendation
        }

    def record_cut(self, symbol: str, pnl: float, reason: str):
        """Record a cut for learning purposes"""
        self._cut_history.append({
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "pnl": pnl,
            "reason": reason
        })

        # Clear entry time
        if symbol in self._entry_times:
            del self._entry_times[symbol]

        # Keep last 100 cuts
        if len(self._cut_history) > 100:
            self._cut_history = self._cut_history[-100:]

    def get_capital_preservation_status(self, positions: List[Dict], account: Dict) -> Dict:
        """Get overall capital preservation status"""
        evaluation = self.evaluate_all_positions(positions)

        equity = float(account.get("equity", 100000))
        total_loss = evaluation["summary"]["total_unrealized_loss"]
        loss_pct_of_equity = (total_loss / equity) * 100 if equity > 0 else 0

        risk_level = "LOW"
        if loss_pct_of_equity < -2:
            risk_level = "CRITICAL"
        elif loss_pct_of_equity < -1:
            risk_level = "HIGH"
        elif loss_pct_of_equity < -0.5:
            risk_level = "MEDIUM"

        return {
            "risk_level": risk_level,
            "total_unrealized_loss": total_loss,
            "loss_pct_of_equity": loss_pct_of_equity,
            "positions_requiring_action": evaluation["summary"]["positions_to_cut_immediately"],
            "immediate_cuts_needed": evaluation["immediate_actions"],
            "message": self._get_preservation_message(risk_level, evaluation)
        }

    def _get_preservation_message(self, risk_level: str, evaluation: Dict) -> str:
        """Generate capital preservation message"""
        cuts_needed = evaluation["summary"]["positions_to_cut_immediately"]

        if risk_level == "CRITICAL":
            return f"CRITICAL: {cuts_needed} positions need IMMEDIATE cuts. Capital at risk!"
        elif risk_level == "HIGH":
            return f"HIGH RISK: {cuts_needed} positions approaching max loss. Action recommended."
        elif risk_level == "MEDIUM":
            return f"ELEVATED: Monitor positions closely. {cuts_needed} positions need attention."
        else:
            return "Capital preservation status: GOOD. Positions within tolerance."


# Singleton instance
_loss_cutter: Optional[LossCutter] = None


def get_loss_cutter() -> LossCutter:
    """Get or create the loss cutter singleton"""
    global _loss_cutter
    if _loss_cutter is None:
        _loss_cutter = LossCutter()
    return _loss_cutter
