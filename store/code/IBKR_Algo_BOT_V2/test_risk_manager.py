"""
Test script for Warrior Trading Risk Manager

Tests all risk management features
"""

import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai.warrior_risk_manager import WarriorRiskManager, ValidationResult
from ai.warrior_pattern_detector import TradingSetup, SetupType
from config.config_loader import get_config


def test_risk_manager():
    """Test the risk manager"""

    print("=" * 80)
    print("WARRIOR TRADING RISK MANAGER - TEST MODE")
    print("=" * 80)

    try:
        # Test 1: Initialize
        print("\n[1/7] Initializing risk manager...")
        risk_mgr = WarriorRiskManager()
        print("[OK] Risk manager initialized")

        config = get_config()
        print(f"\nConfiguration:")
        print(f"  Daily profit goal: ${config.risk.daily_profit_goal}")
        print(f"  Max loss per trade: ${config.risk.max_loss_per_trade}")
        print(f"  Max loss per day: ${config.risk.max_loss_per_day}")
        print(f"  Default risk per trade: ${config.risk.default_risk_per_trade}")
        print(f"  Min R:R ratio: {config.risk.min_reward_to_risk}:1")
        print(f"  Max consecutive losses: {config.risk.max_consecutive_losses}")

        # Test 2: Position Sizing
        print("\n[2/7] Testing position sizing...")
        test_position_sizing(risk_mgr)

        # Test 3: Trade Validation
        print("\n[3/7] Testing trade validation...")
        test_trade_validation(risk_mgr)

        # Test 4: Trade Recording
        print("\n[4/7] Testing trade recording...")
        test_trade_recording(risk_mgr)

        # Test 5: Loss Limits
        print("\n[5/7] Testing loss limits...")
        test_loss_limits()

        # Test 6: Daily Statistics
        print("\n[6/7] Testing daily statistics...")
        test_daily_stats(risk_mgr)

        # Test 7: Integration
        print("\n[7/7] Testing full integration...")
        test_full_integration()

        print("\n" + "=" * 80)
        print("[OK] ALL RISK MANAGER TESTS PASSED")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_position_sizing(risk_mgr):
    """Test position sizing calculations"""

    print("\nPosition Sizing Tests:")
    print("-" * 60)

    test_cases = [
        # (entry, stop, risk, expected_shares)
        (5.00, 4.90, 100.0, 1000),   # $0.10 stop
        (10.00, 9.50, 100.0, 200),    # $0.50 stop
        (2.50, 2.40, 50.0, 500),      # $0.10 stop, lower risk
        (100.00, 99.00, 100.0, 100),  # $1.00 stop
    ]

    for entry, stop, risk, expected in test_cases:
        shares = risk_mgr.calculate_position_size(entry, stop, risk)
        actual_risk = shares * (entry - stop)

        print(f"\n  Entry ${entry:.2f}, Stop ${stop:.2f}, Risk ${risk:.2f}")
        print(f"    → {shares} shares (expected ~{expected})")
        print(f"    → Actual risk: ${actual_risk:.2f}")

        # Allow small deviation due to rounding
        if abs(shares - expected) / expected > 0.1:
            print(f"    [WARN]  Warning: Shares deviate from expected by >10%")

    print("\n  [OK] Position sizing tests complete")


def test_trade_validation(risk_mgr):
    """Test trade validation logic"""

    print("\nTrade Validation Tests:")
    print("-" * 60)

    # Test Case 1: Valid trade
    setup1 = create_test_setup(
        entry=5.00,
        stop=4.90,
        target=5.20,
        rr=2.0
    )

    validation1 = risk_mgr.validate_trade(setup1)
    print(f"\n  Test 1: Valid Trade")
    print(f"    Entry ${setup1.entry_price:.2f}, Stop ${setup1.stop_price:.2f}, "
          f"Target ${setup1.target_2r:.2f}")
    print(f"    Result: {validation1.result.value}")
    print(f"    Position size: {validation1.position_size} shares")
    print(f"    Risk: ${validation1.risk_dollars:.2f}")
    print(f"    Reward: ${validation1.reward_dollars:.2f}")

    assert validation1.result in [ValidationResult.APPROVED, ValidationResult.WARNING], \
        "Valid trade should be approved"

    # Test Case 2: Low R:R (should reject)
    setup2 = create_test_setup(
        entry=5.00,
        stop=4.90,
        target=5.05,  # Only 1:0.5 R:R
        rr=0.5
    )

    validation2 = risk_mgr.validate_trade(setup2)
    print(f"\n  Test 2: Low R:R Trade")
    print(f"    R:R: {setup2.risk_reward_ratio:.1f}:1")
    print(f"    Result: {validation2.result.value}")
    print(f"    Reason: {validation2.reason}")

    assert validation2.result == ValidationResult.REJECTED, \
        "Low R:R trade should be rejected"

    # Test Case 3: Wide stop (high risk per trade)
    setup3 = create_test_setup(
        entry=10.00,
        stop=8.00,  # $2 stop on 50 shares = $100 risk
        target=14.00,
        rr=2.0
    )

    validation3 = risk_mgr.validate_trade(setup3)
    print(f"\n  Test 3: Wide Stop Trade")
    print(f"    Entry ${setup3.entry_price:.2f}, Stop ${setup3.stop_price:.2f}")
    print(f"    Result: {validation3.result.value}")
    print(f"    Position size: {validation3.position_size} shares")
    print(f"    Risk: ${validation3.risk_dollars:.2f}")

    # Should work but with reduced size
    assert validation3.result in [ValidationResult.APPROVED, ValidationResult.WARNING], \
        "Wide stop should still be approved with reduced size"

    print("\n  [OK] Trade validation tests complete")


def test_trade_recording(risk_mgr):
    """Test trade entry and exit recording"""

    print("\nTrade Recording Tests:")
    print("-" * 60)

    # Enter a trade
    trade_id = risk_mgr.record_trade_entry(
        symbol="TEST",
        setup_type=SetupType.BULL_FLAG,
        entry_price=5.00,
        shares=1000,
        stop_price=4.90,
        target_price=5.20
    )

    print(f"\n  Trade entered: {trade_id}")
    print(f"    Open positions: {len(risk_mgr.open_positions)}")

    assert trade_id in risk_mgr.open_positions, "Trade should be in open positions"

    # Exit the trade (winner)
    completed_trade = risk_mgr.record_trade_exit(
        trade_id=trade_id,
        exit_price=5.15,
        exit_reason="TARGET_HIT"
    )

    print(f"\n  Trade exited: {trade_id}")
    print(f"    Exit price: ${completed_trade.exit_price:.2f}")
    print(f"    P&L: ${completed_trade.pnl:+.2f}")
    print(f"    R multiple: {completed_trade.r_multiple:+.2f}R")
    print(f"    Current P&L: ${risk_mgr.current_pnl:+.2f}")

    assert completed_trade.pnl > 0, "Trade should be profitable"
    assert risk_mgr.current_pnl > 0, "Daily P&L should be positive"

    # Enter another trade and exit as loser
    trade_id2 = risk_mgr.record_trade_entry(
        symbol="TEST2",
        setup_type=SetupType.HOD_BREAKOUT,
        entry_price=10.00,
        shares=500,
        stop_price=9.80,
        target_price=10.40
    )

    completed_trade2 = risk_mgr.record_trade_exit(
        trade_id=trade_id2,
        exit_price=9.80,
        exit_reason="STOP_HIT"
    )

    print(f"\n  Trade 2 exited (loser):")
    print(f"    P&L: ${completed_trade2.pnl:+.2f}")
    print(f"    R multiple: {completed_trade2.r_multiple:+.2f}R")
    print(f"    Current P&L: ${risk_mgr.current_pnl:+.2f}")
    print(f"    Consecutive losses: {risk_mgr.consecutive_losses}")

    assert completed_trade2.pnl < 0, "Trade should be a loss"
    assert risk_mgr.consecutive_losses == 1, "Should have 1 consecutive loss"

    print("\n  [OK] Trade recording tests complete")


def test_loss_limits():
    """Test daily loss limits and halting"""

    print("\nLoss Limit Tests:")
    print("-" * 60)

    # Create new risk manager
    risk_mgr = WarriorRiskManager()

    # Simulate multiple losing trades to hit daily limit
    max_daily_loss = risk_mgr.risk_config.max_loss_per_day

    print(f"\n  Max daily loss: ${max_daily_loss:.2f}")
    print(f"  Simulating losses...")

    # Enter and exit trades until we hit limit
    loss_per_trade = 50.0
    trades_needed = int(max_daily_loss / loss_per_trade) + 1

    for i in range(trades_needed):
        trade_id = risk_mgr.record_trade_entry(
            symbol=f"TEST{i}",
            setup_type=SetupType.BULL_FLAG,
            entry_price=5.00,
            shares=500,
            stop_price=4.90,
            target_price=5.20
        )

        # Exit at stop
        risk_mgr.record_trade_exit(
            trade_id=trade_id,
            exit_price=4.90,
            exit_reason="STOP_HIT"
        )

        print(f"    Trade {i+1}: P&L ${risk_mgr.current_pnl:+.2f}")

        if risk_mgr.is_trading_halted:
            print(f"    🛑 Trading halted after {i+1} trades")
            print(f"    Reason: {risk_mgr.halt_reason}")
            break

    assert risk_mgr.is_trading_halted, "Trading should be halted after max daily loss"

    # Try to validate a new trade (should reject)
    setup = create_test_setup()
    validation = risk_mgr.validate_trade(setup)

    print(f"\n  Attempting trade after halt:")
    print(f"    Result: {validation.result.value}")
    print(f"    Reason: {validation.reason}")

    assert validation.result == ValidationResult.REJECTED, \
        "Trades should be rejected when halted"

    print("\n  [OK] Loss limit tests complete")


def test_daily_stats(risk_mgr):
    """Test daily statistics calculation"""

    print("\nDaily Statistics:")
    print("-" * 60)

    stats = risk_mgr.get_daily_stats()

    print(f"\n  Date: {stats['date']}")
    print(f"  Total trades: {stats['total_trades']}")
    print(f"  Winning trades: {stats['winning_trades']}")
    print(f"  Losing trades: {stats['losing_trades']}")
    print(f"  Win rate: {stats['win_rate']:.1f}%")
    print(f"  Current P&L: ${stats['current_pnl']:+.2f}")
    print(f"  Avg win: ${stats['avg_win']:+.2f}")
    print(f"  Avg loss: ${stats['avg_loss']:+.2f}")
    print(f"  Avg R multiple: {stats['avg_r_multiple']:+.2f}R")
    print(f"  Distance to goal: ${stats['distance_to_goal']:.2f}")
    print(f"  Open positions: {stats['open_positions']}")
    print(f"  Consecutive wins: {stats['consecutive_wins']}")
    print(f"  Consecutive losses: {stats['consecutive_losses']}")

    if stats['best_trade']:
        print(f"\n  Best trade:")
        print(f"    Symbol: {stats['best_trade']['symbol']}")
        print(f"    P&L: ${stats['best_trade']['pnl']:+.2f}")
        print(f"    R: {stats['best_trade']['r_multiple']:+.2f}R")

    if stats['worst_trade']:
        print(f"\n  Worst trade:")
        print(f"    Symbol: {stats['worst_trade']['symbol']}")
        print(f"    P&L: ${stats['worst_trade']['pnl']:+.2f}")
        print(f"    R: {stats['worst_trade']['r_multiple']:+.2f}R")

    print("\n  [OK] Statistics tests complete")


def test_full_integration():
    """Test full workflow integration"""

    print("\nFull Integration Test:")
    print("-" * 60)

    from ai.warrior_scanner import WarriorScanner
    from ai.warrior_pattern_detector import WarriorPatternDetector

    print("\n  Complete Warrior Trading workflow:")
    print("  1. Scanner → 2. Pattern Detector → 3. Risk Manager → 4. Execution")

    # Initialize all components
    scanner = WarriorScanner()
    detector = WarriorPatternDetector()
    risk_mgr = WarriorRiskManager()

    print("\n  [OK] All components initialized")
    print(f"    Scanner ready: {scanner is not None}")
    print(f"    Detector ready: {detector is not None}")
    print(f"    Risk Manager ready: {risk_mgr is not None}")

    # Create mock setup
    setup = create_test_setup()

    # Validate with risk manager
    validation = risk_mgr.validate_trade(setup)

    print(f"\n  Mock setup validated:")
    print(f"    Result: {validation.result.value}")
    print(f"    Position size: {validation.position_size} shares")
    print(f"    Risk: ${validation.risk_dollars:.2f}")
    print(f"    Reward: ${validation.reward_dollars:.2f}")

    if validation.result == ValidationResult.APPROVED:
        print(f"\n  [OK] Trade would be approved for execution")
    else:
        print(f"\n  [WARN]  Trade rejected: {validation.reason}")

    print("\n  [OK] Integration test complete")


def create_test_setup(entry=5.00, stop=4.90, target=5.20, rr=2.0):
    """Create a test trading setup"""

    risk_per_share = abs(entry - stop)
    reward_per_share = abs(target - entry)

    return TradingSetup(
        setup_type=SetupType.BULL_FLAG,
        symbol="TEST",
        timeframe="5min",
        entry_price=entry,
        entry_condition="Test entry condition",
        stop_price=stop,
        stop_reason="Test stop reason",
        target_1r=entry + risk_per_share,
        target_2r=target,
        target_3r=entry + (risk_per_share * 3),
        risk_per_share=risk_per_share,
        reward_per_share=reward_per_share,
        risk_reward_ratio=rr,
        confidence=75.0,
        strength_factors=["Test strength"],
        risk_factors=[],
        current_price=entry - 0.05
    )


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    success = test_risk_manager()

    sys.exit(0 if success else 1)
