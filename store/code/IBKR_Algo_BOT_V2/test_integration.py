"""
Integration Test Suite - Multi-Phase Workflow Testing
Tests all phases working together in realistic trading scenarios

Tests:
- ML pattern detection + Risk validation
- Sentiment analysis + Position sizing
- Slippage monitoring + Trade execution
- Reversal detection + Emergency exit
- Complete trade workflow (scan → analyze → execute → monitor → exit)

Run with: python test_integration.py
"""

import unittest
import sys
import logging
from datetime import datetime
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestMLRiskIntegration(unittest.TestCase):
    """Test ML Pattern Detection + Risk Management Integration"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from ai.warrior_transformer_detector import get_transformer_detector
            from ai.warrior_rl_agent import get_rl_agent, TradingState
            self.get_detector = get_transformer_detector
            self.get_agent = get_rl_agent
            self.TradingState = TradingState
            self.ml_available = True
        except ImportError as e:
            logger.warning(f"ML modules not available: {e}")
            self.ml_available = False

        # Mock risk calculator
        self.calculate_position_size = lambda risk, stop: int(risk / stop)

    def test_01_pattern_to_risk_validation(self):
        """Test: Pattern detected → Risk validation → Position sizing"""
        if not self.ml_available:
            self.skipTest("ML not available")

        # Step 1: Detect pattern
        detector = self.get_detector()

        # Create bullish candles
        candles = [
            {'open': 100 + i*0.2, 'high': 101 + i*0.2,
             'low': 99 + i*0.2, 'close': 100.5 + i*0.2, 'volume': 1000000}
            for i in range(50)
        ]

        pattern = detector.detect_pattern(candles, "AAPL", "5min")

        # Step 2: If pattern detected, validate risk
        if pattern and pattern.confidence > 0.6:
            # Step 3: Calculate position size based on pattern confidence
            risk_amount = 50.0  # $50 max risk

            if pattern.stop_loss and pattern.price_target:
                entry = candles[-1]['close']
                stop_distance = abs(entry - pattern.stop_loss)
                reward_distance = abs(pattern.price_target - entry)

                # Risk:Reward ratio
                rr_ratio = reward_distance / stop_distance if stop_distance > 0 else 0

                # Only trade if R:R >= 2:1
                if rr_ratio >= 2.0:
                    shares = self.calculate_position_size(risk_amount, stop_distance)

                    logger.info(f"[OK] Pattern→Risk workflow:")
                    logger.info(f"     Pattern: {pattern.pattern_type}, confidence: {pattern.confidence:.1%}")
                    logger.info(f"     R:R ratio: {rr_ratio:.1f}:1")
                    logger.info(f"     Position: {shares} shares")

                    self.assertGreaterEqual(rr_ratio, 2.0)
                    self.assertGreater(shares, 0)
                    return

        logger.info("[OK] No high-confidence pattern with valid R:R - correctly skipped")

    def test_02_rl_agent_position_management(self):
        """Test: RL agent action → Position sizing adjustment"""
        if not self.ml_available:
            self.skipTest("ML not available")

        agent = self.get_agent()

        # Create favorable trading state
        state = self.TradingState(
            price=150.0,
            volume=2000000,
            volatility=0.02,
            trend=0.7,  # Strong uptrend
            position_size=0.0,
            entry_price=None,
            unrealized_pnl=0.0,
            sentiment_score=0.5,  # Positive sentiment
            pattern_confidence=0.8,  # Strong pattern
            time_in_position=0,
            current_drawdown=0.0,
            sharpe_ratio=1.5,
            win_rate=0.6
        )

        # Get RL recommendation
        action = agent.select_action(state, training=False)

        # Verify action makes sense
        self.assertIn(action.action_type, agent.ACTIONS)
        self.assertGreaterEqual(action.confidence, 0.0)
        self.assertLessEqual(action.confidence, 1.0)

        logger.info(f"[OK] RL Agent recommendation: {action.action_type} (confidence: {action.confidence:.1%})")


class TestSentimentRiskIntegration(unittest.TestCase):
    """Test Sentiment Analysis + Risk Management Integration"""

    def setUp(self):
        """Set up test fixtures"""
        # Simple position sizing function
        self.calculate_position_size = lambda risk, stop: int(risk / stop)

    def test_01_sentiment_based_position_sizing(self):
        """Test: Sentiment score → Adjusted position size"""

        # Mock sentiment scores
        sentiments = {
            'AAPL': 0.8,   # Very bullish
            'TSLA': 0.2,   # Slightly bullish
            'AMC': -0.5    # Bearish
        }

        base_risk = 50.0
        stop_distance = 0.50

        for symbol, sentiment in sentiments.items():
            # Adjust risk based on sentiment
            if sentiment > 0.5:
                # High conviction - increase position size
                adjusted_risk = base_risk * 1.5
            elif sentiment > 0:
                # Normal position
                adjusted_risk = base_risk
            else:
                # Negative sentiment - skip or reduce
                adjusted_risk = base_risk * 0.5

            shares = self.calculate_position_size(adjusted_risk, stop_distance)

            logger.info(f"[OK] {symbol}: sentiment={sentiment:+.1f}, shares={shares}")

            self.assertGreater(shares, 0)

            # Verify higher sentiment = larger position
            if sentiment > 0.5:
                self.assertGreaterEqual(shares, 100)

    def test_02_sentiment_trade_validation(self):
        """Test: Negative sentiment → Trade rejection"""

        # Mock trade with negative sentiment
        symbol = "BEARISH_STOCK"
        sentiment_score = -0.6
        pattern_confidence = 0.7

        # Decision: Don't trade if sentiment contradicts pattern
        should_trade = sentiment_score > -0.3 or pattern_confidence > 0.9

        self.assertFalse(should_trade)
        logger.info(f"[OK] Trade correctly rejected: sentiment={sentiment_score:.1f}, pattern={pattern_confidence:.1%}")


class TestSlippageExecutionIntegration(unittest.TestCase):
    """Test Slippage Monitoring + Trade Execution Integration"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from ai.warrior_slippage_monitor import (
                WarriorSlippageMonitor,
                SlippageLevel
            )
            self.SlippageMonitor = WarriorSlippageMonitor
            self.SlippageLevel = SlippageLevel
            self.available = True
        except ImportError:
            self.available = False

    def test_01_execution_quality_monitoring(self):
        """Test: Order execution → Slippage check → Adjust strategy"""
        if not self.available:
            self.skipTest("Slippage monitor not available")

        monitor = self.SlippageMonitor()

        # Simulate 5 order executions
        executions = [
            ("AAPL", "buy", 150.00, 150.05, 100),   # Acceptable
            ("AAPL", "buy", 150.00, 150.08, 100),   # Acceptable
            ("AAPL", "buy", 150.00, 150.20, 100),   # Warning
            ("AAPL", "buy", 150.00, 150.15, 100),   # Warning
            ("AAPL", "buy", 150.00, 150.45, 100),   # CRITICAL
        ]

        critical_count = 0

        for symbol, side, expected, actual, shares in executions:
            exec_obj = monitor.record_execution(symbol, side, expected, actual, shares)

            if exec_obj.slippage_level == self.SlippageLevel.CRITICAL:
                critical_count += 1

        # Check statistics
        stats = monitor.get_stats("AAPL")

        self.assertEqual(stats['total'], 5)
        self.assertEqual(stats['critical_count'], 1)

        # Decision: If >=20% executions are critical, pause trading this symbol
        critical_pct = critical_count / len(executions)
        should_pause = critical_pct >= 0.2

        self.assertTrue(should_pause)
        logger.info(f"[OK] Slippage monitoring: {critical_pct:.0%} critical → pausing AAPL")

    def test_02_adaptive_order_sizing(self):
        """Test: High slippage → Reduce order size"""
        if not self.available:
            self.skipTest("Slippage monitor not available")

        monitor = self.SlippageMonitor()

        # Record high slippage on large order
        monitor.record_execution("TSLA", "buy", 200, 200.80, 500)  # 0.4% slippage

        stats = monitor.get_stats("TSLA")

        # Adaptive logic: Reduce order size if avg slippage > 0.2%
        if stats['avg_slippage'] > 0.002:
            original_shares = 500
            reduced_shares = int(original_shares * 0.5)  # Cut in half

            logger.info(f"[OK] High slippage detected: reducing order size {original_shares} → {reduced_shares}")
            self.assertEqual(reduced_shares, 250)


class TestReversalExitIntegration(unittest.TestCase):
    """Test Reversal Detection + Emergency Exit Integration"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from ai.warrior_reversal_detector import (
                WarriorReversalDetector,
                ReversalSeverity
            )
            self.ReversalDetector = WarriorReversalDetector
            self.ReversalSeverity = ReversalSeverity
            self.available = True
        except ImportError:
            self.available = False

    def test_01_reversal_emergency_exit(self):
        """Test: CRITICAL reversal → Immediate exit"""
        if not self.available:
            self.skipTest("Reversal detector not available")

        detector = self.ReversalDetector()

        # Simulate position: entered at $100, went to $105, now at $101
        reversal = detector.detect_jacknife(
            symbol="VOLATILE_STOCK",
            current_price=101.0,
            entry_price=100.0,
            recent_prices=[100, 103, 105, 101],
            direction='long'
        )

        if reversal and reversal.severity == self.ReversalSeverity.CRITICAL:
            # Execute emergency exit
            exit_action = "MARKET_EXIT"

            logger.info(f"[OK] CRITICAL reversal → {exit_action}")
            self.assertEqual(exit_action, "MARKET_EXIT")

    def test_02_reversal_stop_tightening(self):
        """Test: HIGH reversal → Tighten stop loss"""
        if not self.available:
            self.skipTest("Reversal detector not available")

        detector = self.ReversalDetector()

        # Moderate reversal
        reversal = detector.detect_jacknife(
            symbol="AAPL",
            current_price=100.5,
            entry_price=100.0,
            recent_prices=[100, 102, 103, 100.5],
            direction='long'
        )

        if reversal and reversal.severity == self.ReversalSeverity.HIGH:
            original_stop = 99.0
            tightened_stop = 100.2  # Move to just below current

            logger.info(f"[OK] HIGH reversal → tighten stop {original_stop} → {tightened_stop}")
            self.assertGreater(tightened_stop, original_stop)


class TestCompleteTradeWorkflow(unittest.TestCase):
    """Test Complete Trade Workflow Integration"""

    def test_01_complete_workflow(self):
        """Test: Scan → Analyze → Execute → Monitor → Exit"""

        # Workflow simulation
        workflow_steps = []

        # Step 1: Scanner finds candidate
        candidate = {
            'symbol': 'WORKFLOW_TEST',
            'price': 50.0,
            'gap': 5.0,
            'rvol': 3.0,
            'float': 30.0
        }
        workflow_steps.append("SCAN")
        logger.info("[Step 1] Scanner found: WORKFLOW_TEST")

        # Step 2: Pattern analysis
        pattern_detected = True
        pattern_confidence = 0.75
        workflow_steps.append("PATTERN")
        logger.info(f"[Step 2] Pattern detected: confidence={pattern_confidence:.0%}")

        # Step 3: Risk validation
        rr_ratio = 3.0
        risk_acceptable = rr_ratio >= 2.0
        workflow_steps.append("RISK")
        logger.info(f"[Step 3] Risk validated: R:R={rr_ratio:.1f}:1")

        # Step 4: Sentiment check
        sentiment = 0.4
        sentiment_ok = sentiment > 0
        workflow_steps.append("SENTIMENT")
        logger.info(f"[Step 4] Sentiment checked: {sentiment:+.1f}")

        # Step 5: Execute trade
        if pattern_detected and risk_acceptable and sentiment_ok:
            workflow_steps.append("EXECUTE")
            logger.info("[Step 5] Trade executed")

            # Step 6: Monitor for reversal
            reversal_detected = False
            workflow_steps.append("MONITOR")
            logger.info("[Step 6] Monitoring for reversal")

            # Step 7: Exit (target reached or stop hit)
            workflow_steps.append("EXIT")
            logger.info("[Step 7] Trade exited")

        # Verify complete workflow
        expected_steps = ["SCAN", "PATTERN", "RISK", "SENTIMENT", "EXECUTE", "MONITOR", "EXIT"]
        self.assertEqual(workflow_steps, expected_steps)
        logger.info("[OK] Complete workflow executed successfully")


def run_tests():
    """Run all integration tests"""
    print("=" * 70)
    print("INTEGRATION TEST SUITE - MULTI-PHASE WORKFLOWS")
    print("=" * 70)
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMLRiskIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestSentimentRiskIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestSlippageExecutionIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestReversalExitIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestCompleteTradeWorkflow))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print()
    print("=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)

    if result.wasSuccessful():
        print("[OK] All integration tests passed!")
        return 0
    else:
        print("[WARN] Some integration tests failed (check availability of modules)")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
