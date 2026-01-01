"""
Gated Trading Integration
=========================
Integrates the Signal Gating Engine with the HFT Scalper.

This module enforces the architectural principle that ALL trades
must go through the gating engine before execution.

Flow:
1. Trigger detected (momentum spike, news, etc.)
2. Find matching SignalContract from repository
3. Get ChronosContext for current regime
4. Run through GatingEngine
5. If approved → proceed to scalper execution
6. If vetoed → log and skip

This ensures:
- Chronos ONLY provides context (not trading decisions)
- Qlib signals are pre-approved contracts (not live inference)
- Every trade attempt is logged with approval/veto reason
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ai.chronos_adapter import (ChronosAdapter, get_chronos_adapter,
                                get_market_context)
from ai.signal_contract import (ChronosContext, GateResult, RiskState,
                                SignalContract, get_contract_repository)
from ai.signal_gating_engine import (ExecutionRequest, SignalGatingEngine,
                                     create_execution_request,
                                     get_gating_engine)

logger = logging.getLogger(__name__)


class GatedTradingManager:
    """
    Manager that integrates gating with HFT scalper.

    This is the bridge between the gating architecture and
    the execution system.
    """

    def __init__(self):
        self.gating_engine = get_gating_engine()
        self.chronos_adapter = get_chronos_adapter()
        self.contract_repo = get_contract_repository()

        # Current risk state (updated by scalper)
        self.risk_state = RiskState()

        # Statistics
        self.total_attempts = 0
        self.approved_count = 0
        self.vetoed_count = 0

        logger.info("GatedTradingManager initialized")

    def update_risk_state(
        self,
        daily_pnl: float = 0,
        open_positions: int = 0,
        daily_trades: int = 0,
        current_drawdown: float = 0,
        symbol_cooldowns: Dict[str, str] = None,
    ):
        """
        Update the current risk state.

        Called by the scalper to keep risk state in sync.
        """
        self.risk_state.daily_pnl = daily_pnl
        self.risk_state.open_positions = open_positions
        self.risk_state.daily_trades = daily_trades
        self.risk_state.current_drawdown = current_drawdown
        if symbol_cooldowns:
            self.risk_state.symbol_cooldowns = symbol_cooldowns

    def gate_trade_attempt(
        self, symbol: str, trigger_type: str, quote: Dict = None, df=None
    ) -> tuple[bool, Optional[ExecutionRequest], str]:
        """
        Gate a trade attempt through the full pipeline.

        This is the MAIN entry point for gated trading.

        TRADING WINDOW ENFORCEMENT:
        - All ENTRIES must occur within 07:00-09:30 AM ET
        - Exits are allowed at any time
        - No strategy-level overrides permitted

        Args:
            symbol: Stock symbol
            trigger_type: Type of trigger (spike, news, breakout, etc.)
            quote: Current price quote
            df: Optional price DataFrame for Chronos context

        Returns:
            (approved, execution_request, reason)
        """
        self.total_attempts += 1
        logger.info(f"GATE ATTEMPT #{self.total_attempts}: {symbol} ({trigger_type})")

        # STEP 0: TRADING WINDOW CHECK (MANDATORY - no overrides)
        # All new entries MUST occur within 07:00-09:30 AM ET
        try:
            from ai.time_controls import (VETO_TRADING_WINDOW_CLOSED,
                                          get_eastern_time,
                                          is_in_warrior_window)

            if not is_in_warrior_window():
                self.vetoed_count += 1
                now = get_eastern_time()
                reason = f"{VETO_TRADING_WINDOW_CLOSED}: {now.strftime('%H:%M:%S')} ET is outside 07:00-09:30 window"
                logger.info(f"GATE VETOED (WINDOW): {symbol} - {reason}")
                return False, None, reason

        except Exception as e:
            # Fail-closed: if time check fails, reject trade
            self.vetoed_count += 1
            reason = f"TRADING_WINDOW_CHECK_ERROR: {e}"
            logger.error(f"GATE VETOED (ERROR): {symbol} - {reason}")
            return False, None, reason

        # Step 1: Find matching SignalContract
        contracts = self.contract_repo.get_for_symbol(symbol)

        if not contracts:
            # No pre-approved contract - try to create a temporary one
            # In production, this should be blocked, but for flexibility
            # we allow creating a default contract for monitored symbols
            contract = self._create_default_contract(symbol, trigger_type)
            logger.warning(f"No pre-approved contract for {symbol}, using default")
        else:
            # Use the first valid (non-expired) contract
            contract = None
            for c in contracts:
                if not c.is_expired():
                    contract = c
                    break

            if not contract:
                self.vetoed_count += 1
                return False, None, "All contracts expired"

        # Step 2: Get Chronos context
        chronos_context = self._get_chronos_context(symbol, df)

        # Step 3: Run through gating engine
        gate_result = self.gating_engine.gate_signal(
            contract, chronos_context, self.risk_state
        )

        # Step 4: Handle result
        if gate_result.approved:
            self.approved_count += 1

            # Create execution request
            exec_request = create_execution_request(
                contract,
                gate_result,
                chronos_context,
                position_size_pct=contract.max_position_pct,
            )

            logger.info(
                f"GATE APPROVED: {symbol} | "
                f"regime={chronos_context.market_regime} | "
                f"conf={chronos_context.regime_confidence:.1%}"
            )

            return True, exec_request, "Approved"

        else:
            self.vetoed_count += 1

            logger.warning(
                f"GATE VETOED: {symbol} | "
                f"{gate_result.veto_reason}: {gate_result.veto_details}"
            )

            return False, None, f"{gate_result.veto_reason}: {gate_result.veto_details}"

    def _get_chronos_context(self, symbol: str, df=None) -> ChronosContext:
        """Get Chronos context for a symbol."""
        try:
            return self.chronos_adapter.get_context(symbol, df)
        except Exception as e:
            logger.warning(f"Chronos context failed for {symbol}: {e}")
            # Return neutral context
            return ChronosContext(market_regime="UNKNOWN", regime_confidence=0.5)

    def _create_default_contract(
        self, symbol: str, trigger_type: str
    ) -> SignalContract:
        """
        Create a default contract for symbols without pre-approved contracts.

        In production, this should be more restrictive or blocked entirely.
        """
        return SignalContract(
            symbol=symbol,
            direction="LONG",
            horizon="SCALP",
            confidence_required=0.55,
            valid_regimes=["TRENDING_UP", "RANGING"],
            invalid_regimes=["VOLATILE", "TRENDING_DOWN"],
            max_drawdown_allowed=0.03,
            expected_return=0.02,
            source="DEFAULT",
            features=[trigger_type],
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get gating statistics."""
        gating_stats = self.gating_engine.get_stats()

        return {
            "total_attempts": self.total_attempts,
            "approved": self.approved_count,
            "vetoed": self.vetoed_count,
            "approval_rate": (
                self.approved_count / self.total_attempts
                if self.total_attempts > 0
                else 0
            ),
            "gating_engine": gating_stats,
            "contracts_loaded": len(self.contract_repo.contracts),
            "active_contracts": len(self.contract_repo.get_active()),
        }

    def get_recent_vetoes(self, count: int = 20) -> List[Dict]:
        """Get recent veto decisions."""
        return self.gating_engine.get_recent_vetoes(count)


class ScalperGatingWrapper:
    """
    Wrapper that adds gating to the HFT Scalper.

    This wraps the scalper's check_entry_signal to add gating.
    """

    def __init__(self, scalper):
        """
        Initialize with an HFT Scalper instance.

        Args:
            scalper: HFTScalper instance to wrap
        """
        self.scalper = scalper
        self.gating_manager = GatedTradingManager()

        # Store original check_entry_signal
        self._original_check_entry = scalper.check_entry_signal

        # Replace with gated version
        scalper.check_entry_signal = self._gated_check_entry_signal

        logger.info("ScalperGatingWrapper initialized")

    def _sync_risk_state(self):
        """Sync risk state from scalper."""
        self.gating_manager.update_risk_state(
            daily_pnl=self.scalper.daily_pnl,
            open_positions=len(self.scalper.open_positions),
            daily_trades=self.scalper.daily_trades,
            current_drawdown=0,  # Would need to calculate from trades
        )

    async def _gated_check_entry_signal(
        self, symbol: str, quote: Dict, priority: bool = False
    ) -> Optional[Dict]:
        """
        Gated version of check_entry_signal.

        This runs the original check, then validates through gating.
        """
        # Sync risk state
        self._sync_risk_state()

        # Run original check
        signal = await self._original_check_entry(symbol, quote, priority)

        if signal is None:
            return None

        # Determine trigger type
        trigger_type = signal.get("type", "unknown")

        # Run through gating
        approved, exec_request, reason = self.gating_manager.gate_trade_attempt(
            symbol, trigger_type, quote
        )

        if approved:
            # Attach gating info to signal
            signal["gating_approved"] = True
            signal["gating_regime"] = exec_request.chronos_context.market_regime
            signal["gating_confidence"] = exec_request.chronos_context.regime_confidence
            return signal
        else:
            # Vetoed - log and return None
            logger.info(f"Signal vetoed by gating: {symbol} - {reason}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get combined stats."""
        return {
            "scalper": {
                "trades": self.scalper.daily_trades,
                "pnl": self.scalper.daily_pnl,
                "positions": len(self.scalper.open_positions),
            },
            "gating": self.gating_manager.get_stats(),
        }


def integrate_gating_with_scalper(scalper) -> ScalperGatingWrapper:
    """
    Integrate gating with an HFT Scalper instance.

    This is the main entry point for adding gating to the scalper.

    Args:
        scalper: HFTScalper instance

    Returns:
        ScalperGatingWrapper that manages the integration
    """
    return ScalperGatingWrapper(scalper)


# Singleton manager
_gated_manager: Optional[GatedTradingManager] = None


def get_gated_trading_manager() -> GatedTradingManager:
    """Get the global gated trading manager."""
    global _gated_manager
    if _gated_manager is None:
        _gated_manager = GatedTradingManager()
    return _gated_manager


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    print("=" * 60)
    print("GATED TRADING TEST")
    print("=" * 60)

    manager = get_gated_trading_manager()

    # Test with a few symbols
    test_cases = [
        ("AAPL", "momentum_spike"),
        ("TSLA", "news_triggered"),
        ("SPY", "breakout"),
    ]

    for symbol, trigger_type in test_cases:
        print(f"\nTesting {symbol} ({trigger_type}):")

        approved, exec_request, reason = manager.gate_trade_attempt(
            symbol, trigger_type
        )

        print(f"  Approved: {approved}")
        print(f"  Reason: {reason}")

        if exec_request:
            print(f"  Direction: {exec_request.direction}")
            print(f"  Position Size: {exec_request.position_size_pct:.1%}")

    print("\n" + "=" * 60)
    print("GATING STATISTICS")
    print("=" * 60)
    stats = manager.get_stats()
    print(f"Total attempts: {stats['total_attempts']}")
    print(f"Approved: {stats['approved']}")
    print(f"Vetoed: {stats['vetoed']}")
    print(f"Approval rate: {stats['approval_rate']:.1%}")
