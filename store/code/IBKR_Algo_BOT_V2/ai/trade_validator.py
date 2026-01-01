"""
Claude AI Trade Validator
Validates trading signals before execution to prevent emotional/impulsive trades
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class TradeAction(Enum):
    """Trading actions"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class ValidationResult(Enum):
    """Trade validation results"""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REVIEW = "REVIEW"  # Requires manual review


class TradeValidator:
    """
    Validates trades using AI analysis before execution
    Helps prevent emotional trading and costly mistakes
    """

    def __init__(self, risk_settings: Optional[Dict] = None):
        """
        Initialize trade validator

        Args:
            risk_settings: Optional risk parameters
        """
        self.risk_settings = risk_settings or self._default_risk_settings()
        self.validation_history = []

    def _default_risk_settings(self) -> Dict:
        """Default risk settings for conservative trading"""
        return {
            "max_position_size_percent": 10,  # % of portfolio
            "max_daily_trades": 5,
            "max_single_trade_risk_percent": 2,  # % of portfolio at risk
            "require_stop_loss": True,
            "min_risk_reward_ratio": 1.5,
            "blacklist_symbols": [],
            "trading_hours_only": True,
        }

    async def validate_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reason: str = "",
        portfolio_value: float = 100000,
        current_positions: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Validate a proposed trade

        Args:
            symbol: Stock ticker
            action: BUY, SELL, or HOLD
            quantity: Number of shares
            entry_price: Proposed entry price
            stop_loss: Stop loss price (optional)
            take_profit: Take profit target (optional)
            reason: Reason for the trade
            portfolio_value: Current portfolio value
            current_positions: List of current positions

        Returns:
            Validation result with detailed feedback
        """

        validation_result = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "entry_price": entry_price,
            "validation_status": ValidationResult.REVIEW.value,
            "checks": {},
            "warnings": [],
            "suggestions": [],
            "ai_commentary": "",
            "should_execute": False,
        }

        # Perform validation checks
        checks = []

        # Check 1: Position Size
        position_size_check = self._check_position_size(
            entry_price, quantity, portfolio_value
        )
        checks.append(position_size_check)
        validation_result["checks"]["position_size"] = position_size_check

        # Check 2: Risk Management
        risk_check = self._check_risk_management(
            entry_price, quantity, stop_loss, portfolio_value
        )
        checks.append(risk_check)
        validation_result["checks"]["risk_management"] = risk_check

        # Check 3: Symbol Restrictions
        symbol_check = self._check_symbol_restrictions(symbol)
        checks.append(symbol_check)
        validation_result["checks"]["symbol_restrictions"] = symbol_check

        # Check 4: Daily Limits
        daily_limit_check = self._check_daily_limits()
        checks.append(daily_limit_check)
        validation_result["checks"]["daily_limits"] = daily_limit_check

        # Check 5: Risk/Reward Ratio
        if stop_loss and take_profit:
            risk_reward_check = self._check_risk_reward_ratio(
                entry_price, stop_loss, take_profit, action
            )
            checks.append(risk_reward_check)
            validation_result["checks"]["risk_reward"] = risk_reward_check

        # Check 6: Portfolio Concentration
        if current_positions:
            concentration_check = self._check_portfolio_concentration(
                symbol, entry_price, quantity, current_positions, portfolio_value
            )
            checks.append(concentration_check)
            validation_result["checks"]["concentration"] = concentration_check

        # Determine overall validation status
        all_passed = all(check["passed"] for check in checks)
        any_critical_failure = any(
            check.get("severity") == "CRITICAL" and not check["passed"]
            for check in checks
        )

        if any_critical_failure:
            validation_result["validation_status"] = ValidationResult.REJECTED.value
            validation_result["should_execute"] = False
        elif all_passed:
            validation_result["validation_status"] = ValidationResult.APPROVED.value
            validation_result["should_execute"] = True
        else:
            validation_result["validation_status"] = ValidationResult.REVIEW.value
            validation_result["should_execute"] = False

        # Generate AI commentary
        validation_result["ai_commentary"] = self._generate_commentary(
            validation_result, reason
        )

        # Store validation
        self.validation_history.append(validation_result)

        return validation_result

    def _check_position_size(
        self, entry_price: float, quantity: int, portfolio_value: float
    ) -> Dict:
        """Check if position size is within limits"""

        position_value = entry_price * quantity
        position_percent = (position_value / portfolio_value) * 100
        max_allowed = self.risk_settings["max_position_size_percent"]

        passed = position_percent <= max_allowed

        return {
            "name": "Position Size",
            "passed": passed,
            "severity": "HIGH" if not passed else "NORMAL",
            "details": {
                "position_value": position_value,
                "portfolio_value": portfolio_value,
                "position_percent": round(position_percent, 2),
                "max_allowed_percent": max_allowed,
            },
            "message": f"Position size is {position_percent:.1f}% of portfolio (max: {max_allowed}%)",
        }

    def _check_risk_management(
        self,
        entry_price: float,
        quantity: int,
        stop_loss: Optional[float],
        portfolio_value: float,
    ) -> Dict:
        """Check if proper risk management is in place"""

        # Check if stop loss is required and present
        if self.risk_settings["require_stop_loss"] and not stop_loss:
            return {
                "name": "Risk Management",
                "passed": False,
                "severity": "CRITICAL",
                "details": {"stop_loss": None},
                "message": "Stop loss is required but not provided",
            }

        if stop_loss:
            # Calculate risk
            risk_per_share = abs(entry_price - stop_loss)
            total_risk = risk_per_share * quantity
            risk_percent = (total_risk / portfolio_value) * 100
            max_risk = self.risk_settings["max_single_trade_risk_percent"]

            passed = risk_percent <= max_risk

            return {
                "name": "Risk Management",
                "passed": passed,
                "severity": "HIGH" if not passed else "NORMAL",
                "details": {
                    "total_risk": total_risk,
                    "risk_percent": round(risk_percent, 2),
                    "max_risk_percent": max_risk,
                    "stop_loss": stop_loss,
                },
                "message": f"Risk is {risk_percent:.2f}% of portfolio (max: {max_risk}%)",
            }

        return {
            "name": "Risk Management",
            "passed": True,
            "severity": "NORMAL",
            "details": {},
            "message": "Stop loss not required",
        }

    def _check_symbol_restrictions(self, symbol: str) -> Dict:
        """Check if symbol is allowed to trade"""

        blacklist = self.risk_settings.get("blacklist_symbols", [])
        passed = symbol not in blacklist

        return {
            "name": "Symbol Restrictions",
            "passed": passed,
            "severity": "CRITICAL" if not passed else "NORMAL",
            "details": {"symbol": symbol, "blacklist": blacklist},
            "message": f"Symbol {'is' if not passed else 'is not'} on blacklist",
        }

    def _check_daily_limits(self) -> Dict:
        """Check if daily trading limits are respected"""

        today = datetime.now().date()
        today_trades = [
            v
            for v in self.validation_history
            if datetime.fromisoformat(v["timestamp"]).date() == today
            and v["validation_status"] == ValidationResult.APPROVED.value
        ]

        trade_count = len(today_trades)
        max_trades = self.risk_settings["max_daily_trades"]
        passed = trade_count < max_trades

        return {
            "name": "Daily Trading Limit",
            "passed": passed,
            "severity": "MEDIUM" if not passed else "NORMAL",
            "details": {"trades_today": trade_count, "max_trades": max_trades},
            "message": f"{trade_count} of {max_trades} daily trades used",
        }

    def _check_risk_reward_ratio(
        self, entry_price: float, stop_loss: float, take_profit: float, action: str
    ) -> Dict:
        """Check if risk/reward ratio meets minimum requirements"""

        if action.upper() == "BUY":
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:  # SELL
            risk = stop_loss - entry_price
            reward = entry_price - take_profit

        if risk <= 0:
            return {
                "name": "Risk/Reward Ratio",
                "passed": False,
                "severity": "HIGH",
                "details": {},
                "message": "Invalid stop loss placement",
            }

        risk_reward_ratio = reward / risk
        min_ratio = self.risk_settings["min_risk_reward_ratio"]
        passed = risk_reward_ratio >= min_ratio

        return {
            "name": "Risk/Reward Ratio",
            "passed": passed,
            "severity": "MEDIUM" if not passed else "NORMAL",
            "details": {
                "risk": round(risk, 2),
                "reward": round(reward, 2),
                "ratio": round(risk_reward_ratio, 2),
                "min_required": min_ratio,
            },
            "message": f"R/R ratio is {risk_reward_ratio:.2f} (min: {min_ratio})",
        }

    def _check_portfolio_concentration(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
        current_positions: List[Dict],
        portfolio_value: float,
    ) -> Dict:
        """Check portfolio concentration limits"""

        # Calculate total exposure to this symbol
        existing_quantity = sum(
            pos.get("quantity", 0)
            for pos in current_positions
            if pos.get("symbol") == symbol
        )

        total_quantity = existing_quantity + quantity
        total_value = total_quantity * entry_price
        concentration_percent = (total_value / portfolio_value) * 100

        # Warning if concentration exceeds 20%
        passed = concentration_percent <= 20

        return {
            "name": "Portfolio Concentration",
            "passed": passed,
            "severity": "MEDIUM" if not passed else "NORMAL",
            "details": {
                "existing_quantity": existing_quantity,
                "new_quantity": quantity,
                "total_quantity": total_quantity,
                "concentration_percent": round(concentration_percent, 2),
            },
            "message": f"Total {symbol} exposure: {concentration_percent:.1f}% of portfolio",
        }

    def _generate_commentary(self, validation_result: Dict, reason: str) -> str:
        """Generate AI commentary on the trade"""

        status = validation_result["validation_status"]
        symbol = validation_result["symbol"]
        action = validation_result["action"]

        if status == ValidationResult.APPROVED.value:
            commentary = f"✅ Trade APPROVED: {action} {symbol}\n\n"
            commentary += f"Your reasoning: {reason}\n\n"
            commentary += "All risk checks passed. This trade meets your risk management criteria.\n"

        elif status == ValidationResult.REJECTED.value:
            commentary = f"❌ Trade REJECTED: {action} {symbol}\n\n"
            commentary += "Critical risk management issues detected:\n"
            for check_name, check_data in validation_result["checks"].items():
                if (
                    not check_data["passed"]
                    and check_data.get("severity") == "CRITICAL"
                ):
                    commentary += f"  • {check_data['message']}\n"
            commentary += (
                "\nRecommendation: Revise your trade parameters before execution.\n"
            )

        else:  # REVIEW
            commentary = f"⚠️  Trade NEEDS REVIEW: {action} {symbol}\n\n"
            commentary += "Some concerns were identified:\n"
            for check_name, check_data in validation_result["checks"].items():
                if not check_data["passed"]:
                    commentary += f"  • {check_data['message']}\n"
            commentary += (
                "\nRecommendation: Review the warnings carefully before proceeding.\n"
            )

        return commentary

    def get_validation_summary(self, days: int = 1) -> Dict:
        """Get summary of recent validations"""

        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)

        recent_validations = [
            v
            for v in self.validation_history
            if datetime.fromisoformat(v["timestamp"]) >= cutoff
        ]

        approved = sum(
            1
            for v in recent_validations
            if v["validation_status"] == ValidationResult.APPROVED.value
        )
        rejected = sum(
            1
            for v in recent_validations
            if v["validation_status"] == ValidationResult.REJECTED.value
        )
        review = sum(
            1
            for v in recent_validations
            if v["validation_status"] == ValidationResult.REVIEW.value
        )

        return {
            "period_days": days,
            "total_validations": len(recent_validations),
            "approved": approved,
            "rejected": rejected,
            "review_required": review,
            "approval_rate": (
                round(approved / len(recent_validations) * 100, 1)
                if recent_validations
                else 0
            ),
        }


# Simple usage example
async def quick_trade_check(
    symbol: str, action: str, shares: int, price: float, stop_loss: float = None
) -> str:
    """
    Quick trade validation for investors
    Returns simple yes/no with explanation

    Usage:
        result = await quick_trade_check('AAPL', 'BUY', 100, 150.00, 145.00)
        print(result)
    """

    validator = TradeValidator()
    result = await validator.validate_trade(
        symbol=symbol,
        action=action,
        quantity=shares,
        entry_price=price,
        stop_loss=stop_loss,
        reason="Manual trade entry",
    )

    return result["ai_commentary"]


if __name__ == "__main__":
    # Test the validator
    import asyncio

    async def test():
        result = await quick_trade_check("AAPL", "BUY", 100, 150.00, 145.00)
        print(result)

    asyncio.run(test())
