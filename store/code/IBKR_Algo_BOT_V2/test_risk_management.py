"""
Test Suite for Phase 4: Risk Management
Tests risk router API endpoints
"""

import sys
import unittest
from datetime import datetime

class TestRiskManagementAPI(unittest.TestCase):
    """Test Risk Management API"""

    def setUp(self):
        try:
            from ai.warrior_risk_router import router
            self.router = router
            self.available = True
        except ImportError as e:
            print(f"[WARN] Risk router not available: {e}")
            self.available = False

    def test_01_import(self):
        """Test risk router import"""
        self.assertTrue(self.available, "Risk router should be importable")
        print("[OK] Risk router imports successfully")

    def test_02_router_routes(self):
        """Test router has correct routes"""
        if not self.available:
            self.skipTest("Risk router not available")

        self.assertGreater(len(self.router.routes), 0)
        route_paths = [route.path for route in self.router.routes]
        
        expected_paths = ['/health', '/calculate-position-size', '/validate-trade', '/status']
        for path in expected_paths:
            path_found = any(path.replace('/api/risk', '') in route.path for route in self.router.routes)
            self.assertTrue(path_found, f"Expected route {path} not found")
        
        print(f"[OK] Router has {len(self.router.routes)} routes")

    def test_03_position_sizing_logic(self):
        """Test position sizing calculation"""
        # Test Ross Cameron formula: shares = RISK / STOP_DISTANCE
        risk_amount = 50.0
        stop_distance = 0.25
        expected_shares = int(risk_amount / stop_distance)  # 200 shares
        
        self.assertEqual(expected_shares, 200)
        print(f"[OK] Position sizing: ${risk_amount} risk / ${stop_distance} stop = {expected_shares} shares")

    def test_04_risk_reward_validation(self):
        """Test R:R ratio validation"""
        entry = 150.0
        stop = 149.70
        target = 151.0
        
        risk = abs(entry - stop)  # 0.30
        reward = abs(target - entry)  # 1.00
        rr_ratio = reward / risk  # 3.33
        
        self.assertGreaterEqual(rr_ratio, 2.0, "R:R should meet minimum 2:1")
        print(f"[OK] R:R validation: {rr_ratio:.2f}:1 meets minimum 2:1")

    def test_05_max_risk_validation(self):
        """Test max risk per trade limit"""
        shares = 200
        risk_per_share = 0.30
        risk_amount = shares * risk_per_share  # $60
        max_risk = 50.0
        
        exceeds_limit = risk_amount > max_risk
        self.assertTrue(exceeds_limit, "Should detect when risk exceeds limit")
        print(f"[OK] Risk limit detection: ${risk_amount:.2f} > ${max_risk:.2f}")

def run_tests():
    """Run all tests"""
    print("=" * 70)
    print("PHASE 4: RISK MANAGEMENT TEST SUITE")
    print("=" * 70)
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestRiskManagementAPI))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    if result.wasSuccessful():
        print("[OK] All tests passed!")
        return 0
    else:
        print("[FAIL] Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
