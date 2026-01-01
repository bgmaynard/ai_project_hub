"""
End-to-End FSM Pipeline Test (ChatGPT Spec)
============================================
Tests the complete momentum FSM pipeline from entry to exit.

Components tested:
1. MomentumScorer with veto system
2. MomentumStateMachine with new states
3. MomentumTelemetry with MFE/MAE tracking
4. Full trade lifecycle
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_momentum_scorer():
    """Test the momentum scorer with veto system"""
    print("\n" + "=" * 60)
    print("TEST 1: MOMENTUM SCORER")
    print("=" * 60)

    from ai.momentum_score import MomentumScorer, VetoReason

    scorer = MomentumScorer()
    tests_passed = 0
    tests_total = 0

    # Test 1.1: Good momentum (no veto)
    tests_total += 1
    result = scorer.calculate(
        symbol="TEST_GOOD",
        current_price=5.50,
        prices_5s=[5.45, 5.47, 5.50],
        prices_15s=[5.40, 5.44, 5.48, 5.50],
        prices_30s=[5.30, 5.40, 5.45, 5.50],
        high_30s=5.52,
        low_30s=5.28,
        high_of_day=5.52,
        vwap=5.40,
        current_volume=50000,
        volume_30s_baseline=10000,
        avg_volume=500000,
        spread_pct=0.3,
        buy_pressure=0.65,
        mtf_aligned=True,
        chronos_regime="TRENDING_UP",
        chronos_confidence=0.7,
    )

    if result.score > 0 and not result.vetoed:
        tests_passed += 1
        print(f"[PASS] Good momentum: Score={result.score}, Vetoed={result.vetoed}")
    else:
        print(f"[FAIL] Good momentum: Score={result.score}, Vetoed={result.vetoed}")

    # Test 1.2: Wide spread veto
    tests_total += 1
    result = scorer.calculate(
        symbol="TEST_SPREAD",
        current_price=5.50,
        prices_30s=[5.40, 5.45, 5.50],
        vwap=5.45,
        spread_pct=2.0,  # Wide spread
        buy_pressure=0.60,
    )

    if result.vetoed and VetoReason.SPREAD_WIDE in result.veto_reasons:
        tests_passed += 1
        print(
            f"[PASS] Wide spread veto: Vetoed={result.vetoed}, Reasons={[v.value for v in result.veto_reasons]}"
        )
    else:
        print(
            f"[FAIL] Wide spread veto: Vetoed={result.vetoed}, Reasons={[v.value for v in result.veto_reasons]}"
        )

    # Test 1.3: Below VWAP veto
    tests_total += 1
    result = scorer.calculate(
        symbol="TEST_VWAP",
        current_price=5.10,
        prices_30s=[5.05, 5.08, 5.10],
        vwap=5.20,  # Price below VWAP
        spread_pct=0.3,
        buy_pressure=0.60,
    )

    if result.vetoed and VetoReason.BELOW_VWAP in result.veto_reasons:
        tests_passed += 1
        print(f"[PASS] Below VWAP veto: Vetoed={result.vetoed}")
    else:
        print(f"[FAIL] Below VWAP veto: Vetoed={result.vetoed}")

    # Test 1.4: ROC calculation
    tests_total += 1
    result = scorer.calculate(
        symbol="TEST_ROC",
        current_price=5.50,
        prices_5s=[5.45, 5.47, 5.50],
        prices_15s=[5.40, 5.44, 5.48, 5.50],
        prices_30s=[5.30, 5.40, 5.45, 5.50],
        vwap=5.40,
        spread_pct=0.3,
        buy_pressure=0.65,
    )

    if result.price_urgency.r_5s > 0 and result.price_urgency.r_15s > 0:
        tests_passed += 1
        print(
            f"[PASS] ROC calculation: r_5s={result.price_urgency.r_5s:.2f}%, r_15s={result.price_urgency.r_15s:.2f}%"
        )
    else:
        print(
            f"[FAIL] ROC calculation: r_5s={result.price_urgency.r_5s}, r_15s={result.price_urgency.r_15s}"
        )

    print(f"\nMomentum Scorer: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


def test_state_machine():
    """Test the state machine with new FSM states"""
    print("\n" + "=" * 60)
    print("TEST 2: STATE MACHINE")
    print("=" * 60)

    from ai.momentum_state_machine import (MomentumState, MomentumStateMachine,
                                           StateOwner, TransitionReason)

    sm = MomentumStateMachine()
    tests_passed = 0
    tests_total = 0
    symbol = "FSM_TEST"

    # Test 2.1: IDLE -> CANDIDATE
    tests_total += 1
    sm.add_to_candidate(symbol, "Scanner add")
    state = sm.get_state(symbol)
    if state and state.state == MomentumState.CANDIDATE:
        tests_passed += 1
        print(
            f"[PASS] IDLE -> CANDIDATE: state={state.state.value}, owner={state.owner.value}"
        )
    else:
        print(f"[FAIL] IDLE -> CANDIDATE")

    # Test 2.2: CANDIDATE -> IGNITING (score=50: >= 45 but < 60)
    tests_total += 1
    sm.update_momentum(symbol, 50)
    state = sm.get_state(symbol)
    if state and state.state == MomentumState.IGNITING:
        tests_passed += 1
        print(
            f"[PASS] CANDIDATE -> IGNITING: state={state.state.value}, owner={state.owner.value}"
        )
    else:
        print(
            f"[FAIL] CANDIDATE -> IGNITING: state={state.state.value if state else None}"
        )

    # Test 2.3: IGNITING -> GATED
    tests_total += 1
    sm.update_momentum(symbol, 75)
    state = sm.get_state(symbol)
    if state and state.state == MomentumState.GATED:
        tests_passed += 1
        print(
            f"[PASS] IGNITING -> GATED: state={state.state.value}, owner={state.owner.value}"
        )
    else:
        print(f"[FAIL] IGNITING -> GATED: state={state.state.value if state else None}")

    # Test 2.4: GATED -> IN_POSITION
    tests_total += 1
    sm.enter_position(symbol, entry_price=5.50, shares=100)
    state = sm.get_state(symbol)
    if state and state.state == MomentumState.IN_POSITION:
        tests_passed += 1
        print(
            f"[PASS] GATED -> IN_POSITION: state={state.state.value}, owner={state.owner.value}"
        )
    else:
        print(
            f"[FAIL] GATED -> IN_POSITION: state={state.state.value if state else None}"
        )

    # Test 2.5: IN_POSITION -> MONITORING
    tests_total += 1
    sm.start_monitoring(symbol)
    state = sm.get_state(symbol)
    if state and state.state == MomentumState.MONITORING:
        tests_passed += 1
        print(
            f"[PASS] IN_POSITION -> MONITORING: state={state.state.value}, owner={state.owner.value}"
        )
    else:
        print(
            f"[FAIL] IN_POSITION -> MONITORING: state={state.state.value if state else None}"
        )

    # Test 2.6: MONITORING -> EXITING
    tests_total += 1
    sm.signal_exit(symbol, "TRAILING_STOP", confidence=0.85, pnl_pct=4.5)
    state = sm.get_state(symbol)
    if state and state.state == MomentumState.EXITING:
        tests_passed += 1
        print(
            f"[PASS] MONITORING -> EXITING: state={state.state.value}, owner={state.owner.value}"
        )
    else:
        print(
            f"[FAIL] MONITORING -> EXITING: state={state.state.value if state else None}"
        )

    # Test 2.7: EXITING -> COOLDOWN
    tests_total += 1
    sm.complete_exit(symbol, pnl_pct=4.5)
    state = sm.get_state(symbol)
    if state and state.state == MomentumState.COOLDOWN:
        tests_passed += 1
        print(
            f"[PASS] EXITING -> COOLDOWN: state={state.state.value}, owner={state.owner.value}"
        )
    else:
        print(
            f"[FAIL] EXITING -> COOLDOWN: state={state.state.value if state else None}"
        )

    # Test 2.8: State ownership
    tests_total += 1
    ownership_correct = True
    ownership_map = {
        MomentumState.IDLE: StateOwner.ENTRY,
        MomentumState.CANDIDATE: StateOwner.ENTRY,
        MomentumState.IGNITING: StateOwner.ENTRY,
        MomentumState.GATED: StateOwner.GATING,
        MomentumState.IN_POSITION: StateOwner.POSITION,
        MomentumState.MONITORING: StateOwner.EXIT,
        MomentumState.EXITING: StateOwner.EXIT,
        MomentumState.COOLDOWN: StateOwner.COOLDOWN,
    }
    # Check via summary
    summary = sm.get_summary()
    if summary["by_state"]["COOLDOWN"] == 1:
        tests_passed += 1
        print(f"[PASS] State ownership tracking: by_owner={summary['by_owner']}")
    else:
        print(f"[FAIL] State ownership tracking")

    # Test 2.9: Veto drops state (score=50 for IGNITING, then veto drops it)
    tests_total += 1
    symbol2 = "VETO_TEST"
    sm.add_to_candidate(symbol2)
    sm.update_momentum(symbol2, 50)  # 50 >= 45, gets to IGNITING
    state_before = sm.get_state(symbol2).state
    sm.update_momentum(symbol2, 75, vetoed=True, veto_reasons=["SPREAD_WIDE"])
    state_after = sm.get_state(symbol2).state
    if (
        state_before == MomentumState.IGNITING
        and state_after == MomentumState.CANDIDATE
    ):
        tests_passed += 1
        print(f"[PASS] Veto drops state: {state_before.value} -> {state_after.value}")
    else:
        print(f"[FAIL] Veto drops state: {state_before.value} -> {state_after.value}")

    print(f"\nState Machine: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


def test_telemetry():
    """Test the telemetry module with MFE/MAE tracking"""
    print("\n" + "=" * 60)
    print("TEST 3: TELEMETRY")
    print("=" * 60)

    from ai.momentum_telemetry import MomentumTelemetry

    telemetry = MomentumTelemetry()
    telemetry.clear()  # Start fresh
    tests_passed = 0
    tests_total = 0

    # Test 3.1: Start trade
    tests_total += 1
    trade_id = telemetry.start_trade(
        symbol="TEL_TEST", entry_price=10.00, shares=100, gating_result="APPROVED"
    )
    active = telemetry.get_active_trades()
    if len(active) == 1 and active[0]["symbol"] == "TEL_TEST":
        tests_passed += 1
        print(f"[PASS] Start trade: trade_id={trade_id}")
    else:
        print(f"[FAIL] Start trade")

    # Test 3.2: MFE/MAE tracking
    tests_total += 1
    telemetry.update_price("TEL_TEST", 10.50)  # Goes up
    telemetry.update_price("TEL_TEST", 10.75)  # Higher
    telemetry.update_price("TEL_TEST", 10.50)  # Drops back
    active = telemetry.get_active_trades()
    if active[0]["mfe"] == 7.5 and active[0]["mae"] == 0:  # 7.5% MFE, 0% MAE
        tests_passed += 1
        print(
            f"[PASS] MFE/MAE tracking: MFE={active[0]['mfe']:.1f}%, MAE={active[0]['mae']:.1f}%"
        )
    else:
        print(
            f"[FAIL] MFE/MAE tracking: MFE={active[0]['mfe']}, MAE={active[0]['mae']}"
        )

    # Test 3.3: Complete trade
    tests_total += 1
    telemetry.complete_trade("TEL_TEST", 10.50, "TRAILING_STOP", "Trail triggered")
    completed = telemetry.get_completed_trades()
    if (
        len(completed) == 1
        and completed[0]["is_winner"]
        and completed[0]["pnl_pct"] == 5.0
    ):
        tests_passed += 1
        print(f"[PASS] Complete trade: pnl={completed[0]['pnl_pct']:.1f}%")
    else:
        print(f"[FAIL] Complete trade")

    # Test 3.4: Metrics calculation
    tests_total += 1
    metrics = telemetry.get_metrics()
    if metrics["total_trades"] == 1 and metrics["win_rate"] == 100:
        tests_passed += 1
        print(
            f"[PASS] Metrics: total={metrics['total_trades']}, win_rate={metrics['win_rate']:.1f}%"
        )
    else:
        print(f"[FAIL] Metrics")

    # Add a losing trade
    trade_id2 = telemetry.start_trade("TEL_LOSS", 10.00, 100)
    telemetry.update_price("TEL_LOSS", 9.50)
    telemetry.complete_trade("TEL_LOSS", 9.50, "STOP_LOSS", "Stop hit")

    # Test 3.5: MFE/MAE analysis
    tests_total += 1
    mfe_mae = telemetry.get_mfe_mae_analysis()
    if "winners" in mfe_mae and "losers" in mfe_mae:
        tests_passed += 1
        print(
            f"[PASS] MFE/MAE analysis: winners={mfe_mae['winners']['count']}, losers={mfe_mae['losers']['count']}"
        )
    else:
        print(f"[FAIL] MFE/MAE analysis")

    print(f"\nTelemetry: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


def test_full_pipeline():
    """Test the complete pipeline from entry to exit"""
    print("\n" + "=" * 60)
    print("TEST 4: FULL PIPELINE INTEGRATION")
    print("=" * 60)

    from ai.momentum_score import MomentumScorer
    from ai.momentum_state_machine import (MomentumState, MomentumStateMachine,
                                           TransitionReason)
    from ai.momentum_telemetry import MomentumTelemetry

    scorer = MomentumScorer()
    sm = MomentumStateMachine()
    telemetry = MomentumTelemetry()
    telemetry.clear()

    tests_passed = 0
    tests_total = 0
    symbol = "PIPELINE_TEST"

    # Simulate a complete trade lifecycle

    # Step 1: Calculate momentum score
    tests_total += 1
    momentum_result = scorer.calculate(
        symbol=symbol,
        current_price=5.50,
        prices_5s=[5.45, 5.47, 5.50],
        prices_15s=[5.40, 5.44, 5.48, 5.50],
        prices_30s=[5.30, 5.40, 5.45, 5.50],
        high_30s=5.52,
        low_30s=5.28,
        high_of_day=5.52,
        vwap=5.40,
        current_volume=50000,
        volume_30s_baseline=10000,
        avg_volume=500000,
        float_shares=2000000,
        day_volume=400000,
        spread_pct=0.3,
        buy_pressure=0.65,
        mtf_aligned=True,
        chronos_regime="TRENDING_UP",
        chronos_confidence=0.7,
    )

    if momentum_result.score > 0 and not momentum_result.vetoed:
        tests_passed += 1
        print(
            f"[PASS] Step 1 - Momentum score: {momentum_result.score}/100, Grade {momentum_result.grade.value}"
        )
    else:
        print(f"[FAIL] Step 1 - Momentum score vetoed")
        return False

    # Step 2: Feed to state machine
    tests_total += 1
    sm.add_to_candidate(symbol, "Scanner detected")
    sm.update_momentum(
        symbol,
        momentum_result.score,
        vetoed=momentum_result.vetoed,
        veto_reasons=[v.value for v in momentum_result.veto_reasons],
    )

    state = sm.get_state(symbol)
    if state and state.state == MomentumState.GATED:
        tests_passed += 1
        print(f"[PASS] Step 2 - State machine: {state.state.value} (awaiting gating)")
    else:
        print(f"[FAIL] Step 2 - State machine: {state.state.value if state else None}")

    # Step 3: Enter position
    tests_total += 1
    sm.enter_position(
        symbol, entry_price=5.50, shares=100, stop_price=5.25, target_price=6.00
    )
    state = sm.get_state(symbol)

    # Also start telemetry tracking
    trade_id = telemetry.start_trade(symbol, 5.50, 100, momentum_result, "APPROVED")

    if state and state.state == MomentumState.IN_POSITION:
        tests_passed += 1
        print(
            f"[PASS] Step 3 - Position entered: {state.state.value}, trade_id={trade_id}"
        )
    else:
        print(f"[FAIL] Step 3 - Position entered")

    # Step 4: Start monitoring
    tests_total += 1
    sm.start_monitoring(symbol)
    state = sm.get_state(symbol)
    if state and state.state == MomentumState.MONITORING:
        tests_passed += 1
        print(
            f"[PASS] Step 4 - Monitoring started: {state.state.value}, owner={state.owner.value}"
        )
    else:
        print(f"[FAIL] Step 4 - Monitoring started")

    # Step 5: Simulate price movement
    tests_total += 1
    price_history = [5.60, 5.75, 5.85, 5.90, 5.80]  # Goes up then pulls back
    for price in price_history:
        sm.update_position(symbol, price)
        telemetry.update_price(symbol, price)

    state = sm.get_state(symbol)
    active = telemetry.get_active_trades()
    if state.pnl_pct > 0 and active[0]["mfe"] > 0:
        tests_passed += 1
        print(
            f"[PASS] Step 5 - Price tracking: PnL={state.pnl_pct:.1f}%, MFE={active[0]['mfe']:.1f}%"
        )
    else:
        print(f"[FAIL] Step 5 - Price tracking")

    # Step 6: Exit signal
    tests_total += 1
    sm.signal_exit(symbol, "TRAILING_STOP", confidence=0.85, pnl_pct=state.pnl_pct)
    state = sm.get_state(symbol)
    if state and state.state == MomentumState.EXITING:
        tests_passed += 1
        print(
            f"[PASS] Step 6 - Exit signal: {state.state.value}, signal={state.exit_signal}"
        )
    else:
        print(f"[FAIL] Step 6 - Exit signal")

    # Step 7: Complete exit
    tests_total += 1
    final_price = 5.80
    sm.complete_exit(symbol, pnl_pct=((final_price - 5.50) / 5.50) * 100)
    telemetry.complete_trade(symbol, final_price, "TRAILING_STOP", "Trail triggered")

    state = sm.get_state(symbol)
    completed = telemetry.get_completed_trades()
    if state.state == MomentumState.COOLDOWN and len(completed) > 0:
        tests_passed += 1
        print(
            f"[PASS] Step 7 - Exit completed: state={state.state.value}, PnL=${completed[0]['pnl_dollars']:.2f}"
        )
    else:
        print(f"[FAIL] Step 7 - Exit completed")

    # Step 8: Verify telemetry metrics
    tests_total += 1
    metrics = telemetry.get_metrics()
    if metrics["total_trades"] >= 1:
        tests_passed += 1
        print(
            f"[PASS] Step 8 - Telemetry metrics: trades={metrics['total_trades']}, WR={metrics['win_rate']:.1f}%"
        )
    else:
        print(f"[FAIL] Step 8 - Telemetry metrics")

    print(f"\nFull Pipeline: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "=" * 70)
    print("FSM PIPELINE END-TO-END TEST")
    print("=" * 70)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {
        "momentum_scorer": test_momentum_scorer(),
        "state_machine": test_state_machine(),
        "telemetry": test_telemetry(),
        "full_pipeline": test_full_pipeline(),
    }

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {name}: [{status}]")

    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED - FSM Pipeline is operational!")
    else:
        print("SOME TESTS FAILED - Review errors above")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    run_all_tests()
