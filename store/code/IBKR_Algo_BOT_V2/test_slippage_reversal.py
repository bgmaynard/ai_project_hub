"""
Comprehensive Test Suite for Slippage Monitor & Reversal Detector
Phase 4+: High-speed trading protection

Run with: python test_slippage_reversal_fixed.py
"""

import unittest
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestSlippageMonitor(unittest.TestCase):
    """Test Slippage Monitor"""

    def setUp(self):
        try:
            from ai.warrior_slippage_monitor import (
                WarriorSlippageMonitor,
                get_slippage_monitor,
                SlippageLevel
            )
            self.SlippageMonitor = WarriorSlippageMonitor
            self.get_monitor = get_slippage_monitor
            self.SlippageLevel = SlippageLevel
            self.available = True
        except ImportError as e:
            logger.warning(f"Slippage monitor not available: {e}")
            self.available = False

    def test_01_import(self):
        self.assertTrue(self.available)
        logger.info("[OK] Slippage monitor imports")

    def test_02_initialization(self):
        if not self.available:
            self.skipTest("Not available")
        monitor = self.get_monitor()
        self.assertIsNotNone(monitor)
        logger.info("[OK] Monitor initialized")

    def test_03_buy_slippage(self):
        if not self.available:
            self.skipTest("Not available")
        monitor = self.SlippageMonitor()
        exec = monitor.record_execution("AAPL", "buy", 150.00, 150.15, 100)
        self.assertAlmostEqual(exec.slippage_pct, 0.001, places=4)
        logger.info("[OK] Buy slippage: 0.1%")

    def test_04_sell_slippage(self):
        if not self.available:
            self.skipTest("Not available")
        monitor = self.SlippageMonitor()
        exec = monitor.record_execution("AAPL", "sell", 150.00, 149.85, 100)
        self.assertAlmostEqual(exec.slippage_pct, 0.001, places=4)
        logger.info("[OK] Sell slippage: 0.1%")

    def test_05_severity_levels(self):
        if not self.available:
            self.skipTest("Not available")
        monitor = self.SlippageMonitor()
        
        e1 = monitor.record_execution("T", "buy", 100, 100.08, 100)
        self.assertEqual(e1.slippage_level, self.SlippageLevel.ACCEPTABLE)
        
        e2 = monitor.record_execution("T", "buy", 100, 100.15, 100)
        self.assertEqual(e2.slippage_level, self.SlippageLevel.WARNING)
        
        e3 = monitor.record_execution("T", "buy", 100, 100.30, 100)
        self.assertEqual(e3.slippage_level, self.SlippageLevel.CRITICAL)
        
        logger.info("[OK] Severity levels correct")

    def test_06_statistics(self):
        if not self.available:
            self.skipTest("Not available")
        monitor = self.SlippageMonitor()
        
        monitor.record_execution("AAPL", "buy", 150, 150.05, 100)
        monitor.record_execution("AAPL", "buy", 150, 150.20, 100)
        monitor.record_execution("AAPL", "sell", 150, 149.50, 100)
        
        stats = monitor.get_stats("AAPL")
        self.assertEqual(stats['total'], 3)
        self.assertEqual(stats['critical_count'], 1)
        logger.info(f"[OK] Stats: {stats['total']} executions")


class TestReversalDetector(unittest.TestCase):
    """Test Reversal Detector"""

    def setUp(self):
        try:
            from ai.warrior_reversal_detector import (
                WarriorReversalDetector,
                get_reversal_detector,
                ReversalType,
                ReversalSeverity
            )
            self.ReversalDetector = WarriorReversalDetector
            self.get_detector = get_reversal_detector
            self.ReversalType = ReversalType
            self.ReversalSeverity = ReversalSeverity
            self.available = True
        except ImportError as e:
            logger.warning(f"Reversal detector not available: {e}")
            self.available = False

    def test_01_import(self):
        self.assertTrue(self.available)
        logger.info("[OK] Reversal detector imports")

    def test_02_initialization(self):
        if not self.available:
            self.skipTest("Not available")
        detector = self.get_detector()
        self.assertIsNotNone(detector)
        logger.info("[OK] Detector initialized")

    def test_03_no_reversal(self):
        if not self.available:
            self.skipTest("Not available")
        detector = self.ReversalDetector()
        rev = detector.detect_jacknife("AAPL", 102, 100, [100, 101, 102], 'long')
        self.assertIsNone(rev)
        logger.info("[OK] No reversal on uptrend")

    def test_04_jacknife_high(self):
        if not self.available:
            self.skipTest("Not available")
        detector = self.ReversalDetector()
        # Up 3%, down 2.5% = HIGH severity
        rev = detector.detect_jacknife("AAPL", 100.5, 100, [100, 101, 103, 100.5], 'long')
        self.assertIsNotNone(rev)
        self.assertEqual(rev.severity, self.ReversalSeverity.HIGH)
        self.assertEqual(rev.recommendation, 'tighten_stop')
        logger.info(f"[OK] HIGH jacknife detected")

    def test_05_jacknife_critical(self):
        if not self.available:
            self.skipTest("Not available")
        detector = self.ReversalDetector()
        # Up 5%, down 4.4% = CRITICAL severity
        rev = detector.detect_jacknife("TSLA", 241, 240, [240, 245, 252, 241], 'long')
        self.assertIsNotNone(rev)
        self.assertEqual(rev.severity, self.ReversalSeverity.CRITICAL)
        self.assertEqual(rev.recommendation, 'exit')
        logger.info(f"[OK] CRITICAL jacknife detected")

    def test_06_fast_exit(self):
        if not self.available:
            self.skipTest("Not available")
        rev = detector.detect_jacknife("TEST", 240, 240, [240, 250, 240], 'long')
        # CRITICAL reversal
        rev = detector.detect_jacknife("TEST", 241, 240, [240, 248, 241], 'long')
        self.assertTrue(detector.should_exit_fast(rev))
        logger.info("[OK] Fast exit for CRITICAL")

    def test_07_insufficient_data(self):
        if not self.available:
            self.skipTest("Not available")
        detector = self.ReversalDetector()
        rev = detector.detect_jacknife("TEST", 102, 100, [100, 102], 'long')
        self.assertIsNone(rev)
        logger.info("[OK] No reversal with <3 prices")


def run_tests():
    print("=" * 70)
    print("SLIPPAGE & REVERSAL DETECTION TEST SUITE")
    print("=" * 70)
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestSlippageMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestReversalDetector))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    if result.wasSuccessful():
        print("[OK] All tests passed!")
        return 0
    else:
        print("[FAIL] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
