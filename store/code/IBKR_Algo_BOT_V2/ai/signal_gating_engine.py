"""
Signal Gating Engine
====================
Enforces proper sequencing between Qlib signals, Chronos context, and execution.

ARCHITECTURAL PRINCIPLES:
1. EVERY trade attempt MUST go through the gating engine
2. Gating engine has VETO power - it can reject any signal
3. ALL veto decisions are logged for audit
4. Execution NEVER invents signals - only acts on approved contracts

Flow:
1. Qlib (offline) → SignalContract
2. Live trigger → find matching SignalContract
3. Chronos → ChronosContext (regime + confidence)
4. GatingEngine.gate() → GateResult (approved/vetoed)
5. If approved → RiskEngine → Execution

The gating engine ensures:
- Signal has a valid contract (from Qlib)
- Current regime is compatible
- Chronos confidence meets threshold
- Risk limits are not exceeded
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from ai.signal_contract import (
    SignalContract,
    ChronosContext,
    RiskState,
    GateResult,
    VetoReason,
    get_contract_repository
)

# Micro-momentum override import
try:
    from ai.micro_momentum_override import check_micro_override, get_micro_override
    HAS_MICRO_OVERRIDE = True
except ImportError:
    HAS_MICRO_OVERRIDE = False
    check_micro_override = None

logger = logging.getLogger(__name__)


class VetoLogger:
    """
    Dedicated logger for veto decisions.
    Provides full audit trail of all gating decisions.
    """

    def __init__(self, log_file: str = "ai/gate_vetoes.log"):
        self.log_file = log_file
        self.veto_history: List[GateResult] = []
        self.max_history = 1000

        # Setup dedicated file handler
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setFormatter(
            logging.Formatter('%(asctime)s | %(message)s')
        )
        self.file_logger = logging.getLogger("gating.vetoes")
        self.file_logger.addHandler(self.file_handler)
        self.file_logger.setLevel(logging.INFO)

    def log_decision(self, result: GateResult):
        """Log a gating decision."""
        self.file_logger.info(result.log_entry())
        self.veto_history.append(result)

        if len(self.veto_history) > self.max_history:
            self.veto_history.pop(0)

    def get_recent_vetoes(self, count: int = 50) -> List[GateResult]:
        """Get recent veto decisions."""
        vetoes = [r for r in self.veto_history if not r.approved]
        return vetoes[-count:]

    def get_veto_stats(self) -> Dict[str, Any]:
        """Get statistics on veto reasons."""
        if not self.veto_history:
            return {"total": 0, "approved": 0, "vetoed": 0}

        approved = sum(1 for r in self.veto_history if r.approved)
        vetoed = len(self.veto_history) - approved

        # Count by reason
        reasons = {}
        for r in self.veto_history:
            if not r.approved and r.veto_reason:
                reasons[r.veto_reason] = reasons.get(r.veto_reason, 0) + 1

        return {
            "total": len(self.veto_history),
            "approved": approved,
            "vetoed": vetoed,
            "approval_rate": approved / len(self.veto_history) if self.veto_history else 0,
            "veto_reasons": reasons
        }


class SignalGatingEngine:
    """
    Core gating engine that validates signals before execution.

    CRITICAL: This is the ONLY path to execution. No trade can bypass gating.
    """

    def __init__(self):
        self.veto_logger = VetoLogger()
        self.contract_repo = get_contract_repository()

        # Configuration
        self.min_chronos_confidence = 0.50  # Base minimum confidence
        self.require_regime_match = True     # Enforce regime matching
        self.log_all_decisions = True        # Log approvals too (not just vetoes)
        self.enable_micro_override = True    # Allow micro-momentum to bypass macro veto

        logger.info("SignalGatingEngine initialized")

    def gate_signal(
        self,
        contract: SignalContract,
        chronos_context: ChronosContext,
        risk_state: RiskState
    ) -> GateResult:
        """
        Gate a signal against its contract and current context.

        This is the CORE function of the gating engine.
        Every trade attempt MUST call this function.

        Args:
            contract: The SignalContract defining the valid trade
            chronos_context: Current market context from Chronos
            risk_state: Current risk state

        Returns:
            GateResult with approval/veto decision
        """
        result = GateResult(
            signal_id=contract.signal_id,
            symbol=contract.symbol,
            approved=True,  # Assume approved, check for vetoes
            current_regime=chronos_context.market_regime,
            chronos_confidence=chronos_context.regime_confidence,
            required_confidence=contract.confidence_required,
            required_regimes=contract.valid_regimes
        )

        # Check 1: Contract not expired
        if contract.is_expired():
            result.approved = False
            result.veto_reason = VetoReason.CONTRACT_EXPIRED.value
            result.veto_details = f"Contract expired at {contract.expires_at}"
            self._log_result(result)
            return result

        # =====================================================================
        # TASK 3: Regime reconciliation with MACRO/MICRO split
        # =====================================================================
        # Use effective regime (considers both macro and micro with override logic)
        effective_regime = chronos_context.get_effective_regime()
        regime_explanation = chronos_context.get_regime_decision_explanation()

        logger.info(
            f"Regime reconciliation for {contract.symbol}: MACRO={chronos_context.macro_regime} "
            f"({chronos_context.macro_confidence:.0%}), MICRO={chronos_context.micro_regime} "
            f"({chronos_context.micro_confidence:.0%}) -> EFFECTIVE={effective_regime}. "
            f"Reason: {regime_explanation}"
        )

        # Check 2: Regime is valid for this signal
        if self.require_regime_match:
            if not contract.is_regime_valid(effective_regime):
                # ================================================================
                # TASK D: Micro-Momentum Override Check
                # Before vetoing for REGIME_MISMATCH, check if strong micro
                # momentum allows bypassing the bad macro regime.
                # ================================================================
                micro_override_applied = False
                override_size_mult = 1.0

                if self.enable_micro_override and HAS_MICRO_OVERRIDE and check_micro_override:
                    # Map micro regime to ATS state for override check
                    # Strong bullish micro = "ACTIVE", moderate = "CONFIRMED"
                    ats_state_map = {
                        "TRENDING_UP": "ACTIVE",
                        "BULLISH": "ACTIVE",
                        "RANGING": "CONFIRMED",
                        "NEUTRAL": "IDLE"
                    }
                    proxy_ats = ats_state_map.get(
                        chronos_context.micro_regime.upper(),
                        "IDLE"
                    )

                    # Check if micro-momentum override allows bypassing
                    override_allowed, override_reason, override_size_mult = check_micro_override(
                        symbol=contract.symbol,
                        macro_regime=chronos_context.macro_regime,
                        micro_regime=chronos_context.micro_regime,
                        micro_confidence=chronos_context.micro_confidence,
                        ats_state=proxy_ats
                    )

                    if override_allowed:
                        micro_override_applied = True
                        logger.warning(
                            f"GATING_MICRO_OVERRIDE_APPLIED: {contract.symbol} | "
                            f"macro={chronos_context.macro_regime}, "
                            f"micro={chronos_context.micro_regime} ({chronos_context.micro_confidence:.0%}), "
                            f"size_mult={override_size_mult}. Reason: {override_reason}"
                        )
                        # Add override info to result
                        result.override_applied = True
                        result.override_reason = override_reason
                        result.override_size_multiplier = override_size_mult
                    else:
                        logger.debug(
                            f"Micro-override denied for {contract.symbol}: {override_reason}"
                        )

                # If no override was applied, apply the original veto
                if not micro_override_applied:
                    result.approved = False
                    result.veto_reason = VetoReason.REGIME_MISMATCH.value
                    result.veto_details = (
                        f"Effective regime '{effective_regime}' not in valid regimes "
                        f"{contract.valid_regimes}. {regime_explanation}"
                    )
                    self._log_result(result)
                    return result
                # If override was applied, continue to next checks (don't veto)

        # Check 3: Regime is explicitly invalid
        if effective_regime in contract.invalid_regimes:
            result.approved = False
            result.veto_reason = VetoReason.INVALID_REGIME.value
            result.veto_details = (
                f"Effective regime '{effective_regime}' explicitly invalid for this signal. "
                f"{regime_explanation}"
            )
            self._log_result(result)
            return result

        # Check 4: Chronos confidence meets contract threshold
        effective_conf_required = max(
            contract.confidence_required,
            self.min_chronos_confidence
        )

        if chronos_context.regime_confidence < effective_conf_required:
            result.approved = False
            result.veto_reason = VetoReason.CONFIDENCE_LOW.value
            result.veto_details = (
                f"Chronos confidence {chronos_context.regime_confidence:.2%} < "
                f"required {effective_conf_required:.2%}"
            )
            self._log_result(result)
            return result

        # Check 5: Symbol not on cooldown
        if risk_state.is_symbol_on_cooldown(contract.symbol):
            result.approved = False
            result.veto_reason = VetoReason.COOLDOWN_ACTIVE.value
            result.veto_details = (
                f"Symbol {contract.symbol} on cooldown until "
                f"{risk_state.symbol_cooldowns.get(contract.symbol, 'unknown')}"
            )
            self._log_result(result)
            return result

        # Check 6: Risk limits
        violates, reason = risk_state.violates_limits(contract)
        if violates:
            result.approved = False
            result.veto_reason = VetoReason.RISK_LIMIT_EXCEEDED.value
            result.veto_details = reason
            self._log_result(result)
            return result

        # All checks passed - APPROVED
        result.veto_details = "All checks passed"
        self._log_result(result)
        return result

    def gate_trigger(
        self,
        symbol: str,
        trigger_type: str,
        chronos_context: ChronosContext,
        risk_state: RiskState
    ) -> Optional[tuple[SignalContract, GateResult]]:
        """
        Gate a trigger by finding matching contract and validating.

        This is for when a live trigger fires and we need to find
        if there's an approved SignalContract for it.

        Args:
            symbol: Triggered symbol
            trigger_type: Type of trigger (e.g., "spike", "news", "breakout")
            chronos_context: Current market context
            risk_state: Current risk state

        Returns:
            (contract, result) if approved, None if no valid contract or vetoed
        """
        # Find contracts for this symbol
        contracts = self.contract_repo.get_for_symbol(symbol)

        if not contracts:
            logger.debug(f"No contracts found for {symbol}")
            return None

        # Try each contract until one passes
        for contract in contracts:
            result = self.gate_signal(contract, chronos_context, risk_state)
            if result.approved:
                return (contract, result)

        # All contracts vetoed
        logger.debug(f"All {len(contracts)} contracts vetoed for {symbol}")
        return None

    def _log_result(self, result: GateResult):
        """Log a gating result."""
        if self.log_all_decisions or not result.approved:
            self.veto_logger.log_decision(result)

        # Also log to main logger
        if result.approved:
            logger.info(f"GATE APPROVED: {result.symbol} | {result.veto_details}")
        else:
            logger.warning(
                f"GATE VETOED: {result.symbol} | "
                f"{result.veto_reason}: {result.veto_details}"
            )

        # Record to funnel metrics
        try:
            from ai.funnel_metrics import (
                record_gating_attempt,
                record_gating_approval,
                record_gating_veto
            )
            record_gating_attempt(result.symbol)
            if result.approved:
                record_gating_approval(result.symbol)
            else:
                record_gating_veto(result.symbol, result.veto_reason or "UNKNOWN")
        except ImportError:
            pass  # Funnel metrics not available

    def get_stats(self) -> Dict[str, Any]:
        """Get gating statistics."""
        return self.veto_logger.get_veto_stats()

    def get_recent_vetoes(self, count: int = 20) -> List[Dict]:
        """Get recent veto decisions."""
        vetoes = self.veto_logger.get_recent_vetoes(count)
        return [v.to_dict() for v in vetoes]


@dataclass
class ExecutionRequest:
    """
    Request to execute a trade.

    IMPORTANT: This can ONLY be created after gating approval.
    Execution engine MUST verify gate_result.approved == True.
    """
    contract: SignalContract
    gate_result: GateResult
    chronos_context: ChronosContext

    # Execution parameters (from contract + context)
    symbol: str = ""
    direction: str = "LONG"
    position_size_pct: float = 0.01  # % of portfolio

    # Stop/target from contract
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.03

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if not self.symbol:
            self.symbol = self.contract.symbol
        if not self.direction:
            self.direction = self.contract.direction

    def is_valid(self) -> bool:
        """Verify this execution request is valid."""
        # CRITICAL: Gate must have approved
        if not self.gate_result.approved:
            logger.error("INVALID ExecutionRequest: gate_result not approved!")
            return False
        return True


def create_execution_request(
    contract: SignalContract,
    gate_result: GateResult,
    chronos_context: ChronosContext,
    position_size_pct: float = 0.01
) -> Optional[ExecutionRequest]:
    """
    Create an execution request from approved gate result.

    Returns None if gate_result is not approved.
    """
    if not gate_result.approved:
        logger.error(f"Cannot create ExecutionRequest: signal {contract.signal_id} was vetoed")
        return None

    return ExecutionRequest(
        contract=contract,
        gate_result=gate_result,
        chronos_context=chronos_context,
        symbol=contract.symbol,
        direction=contract.direction,
        position_size_pct=min(position_size_pct, contract.max_position_pct),
        stop_loss_pct=contract.max_drawdown_allowed,
        take_profit_pct=contract.expected_return
    )


# Singleton gating engine
_gating_engine: Optional[SignalGatingEngine] = None


def get_gating_engine() -> SignalGatingEngine:
    """Get the global gating engine instance."""
    global _gating_engine
    if _gating_engine is None:
        _gating_engine = SignalGatingEngine()
    return _gating_engine


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )

    print("=" * 60)
    print("SIGNAL GATING ENGINE TEST")
    print("=" * 60)

    # Create test contract
    contract = SignalContract(
        symbol="AAPL",
        direction="LONG",
        horizon="SCALP",
        confidence_required=0.55,
        valid_regimes=["TRENDING_UP", "RANGING"],
        invalid_regimes=["VOLATILE"],
        expected_return=0.02
    )

    # Create contexts
    good_context = ChronosContext(
        market_regime="TRENDING_UP",
        regime_confidence=0.7,
        prob_up=0.65
    )

    bad_regime_context = ChronosContext(
        market_regime="VOLATILE",
        regime_confidence=0.8,
        prob_up=0.4
    )

    low_conf_context = ChronosContext(
        market_regime="TRENDING_UP",
        regime_confidence=0.3,  # Below threshold
        prob_up=0.55
    )

    risk_state = RiskState(
        current_drawdown=0.01,
        daily_pnl=-50.0,
        open_positions=1,
        max_positions=3
    )

    # Test gating
    engine = get_gating_engine()

    print("\n1. Testing GOOD context (should approve):")
    result = engine.gate_signal(contract, good_context, risk_state)
    print(f"   Result: {'APPROVED' if result.approved else 'VETOED'}")
    print(f"   Details: {result.veto_details}")

    print("\n2. Testing BAD regime (should veto):")
    result = engine.gate_signal(contract, bad_regime_context, risk_state)
    print(f"   Result: {'APPROVED' if result.approved else 'VETOED'}")
    print(f"   Reason: {result.veto_reason}")
    print(f"   Details: {result.veto_details}")

    print("\n3. Testing LOW confidence (should veto):")
    result = engine.gate_signal(contract, low_conf_context, risk_state)
    print(f"   Result: {'APPROVED' if result.approved else 'VETOED'}")
    print(f"   Reason: {result.veto_reason}")
    print(f"   Details: {result.veto_details}")

    print("\n4. Gating stats:")
    stats = engine.get_stats()
    print(f"   Total decisions: {stats['total']}")
    print(f"   Approved: {stats['approved']}")
    print(f"   Vetoed: {stats['vetoed']}")
    print(f"   Veto reasons: {stats.get('veto_reasons', {})}")
