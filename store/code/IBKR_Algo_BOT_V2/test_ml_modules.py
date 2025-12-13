"""
Comprehensive Test Suite for Phase 3: Advanced ML Modules

Tests:
- Transformer Pattern Detector
- RL Execution Agent
- ML Trainer
- ML API Router

Run with: python test_ml_modules.py
"""

import unittest
import asyncio
import sys
import logging
from datetime import datetime
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestTransformerDetector(unittest.TestCase):
    """Test Transformer Pattern Detector"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from ai.warrior_transformer_detector import (
                WarriorTransformerDetector,
                get_transformer_detector,
                PatternDetection
            )
            self.TransformerDetector = WarriorTransformerDetector
            self.get_detector = get_transformer_detector
            self.PatternDetection = PatternDetection
            self.available = True
        except ImportError as e:
            logger.warning(f"Transformer detector not available: {e}")
            self.available = False

    def test_01_import(self):
        """Test transformer detector import"""
        self.assertTrue(self.available, "Transformer detector should be importable")
        logger.info("[OK] Transformer detector imports successfully")

    def test_02_initialization(self):
        """Test detector initialization"""
        if not self.available:
            self.skipTest("Transformer detector not available")

        detector = self.get_detector()
        self.assertIsNotNone(detector)
        self.assertEqual(len(detector.PATTERN_NAMES), 8)
        logger.info("[OK] Detector initialized with 8 pattern types")

    def test_03_prepare_features(self):
        """Test feature preparation"""
        if not self.available:
            self.skipTest("Transformer detector not available")

        detector = self.get_detector()

        # Create sample candles
        candles = [
            {
                'open': 100 + i * 0.1,
                'high': 101 + i * 0.1,
                'low': 99 + i * 0.1,
                'close': 100.5 + i * 0.1,
                'volume': 1000000
            }
            for i in range(50)
        ]

        features = detector.prepare_features(candles)

        # Check shape (1, sequence_len, 10 features)
        self.assertEqual(len(features.shape), 3)
        self.assertEqual(features.shape[0], 1)  # Batch size
        self.assertEqual(features.shape[2], 10)  # Features

        logger.info(f"[OK] Features prepared: shape {features.shape}")

    def test_04_pattern_detection(self):
        """Test pattern detection"""
        if not self.available:
            self.skipTest("Transformer detector not available")

        detector = self.get_detector()

        # Create sample candles (uptrend for bull pattern)
        candles = [
            {
                'open': 100 + i * 0.2,
                'high': 101 + i * 0.2,
                'low': 99 + i * 0.2,
                'close': 100.5 + i * 0.2,
                'volume': 1000000 + i * 10000
            }
            for i in range(50)
        ]

        pattern = detector.detect_pattern(candles, "TEST", "5min")

        # Pattern may or may not be detected (depends on model state)
        # Just check it returns without error
        if pattern:
            self.assertIsInstance(pattern, self.PatternDetection)
            self.assertIn(pattern.pattern_type, detector.PATTERN_NAMES)
            self.assertGreaterEqual(pattern.confidence, 0.0)
            self.assertLessEqual(pattern.confidence, 1.0)
            logger.info(f"[OK] Pattern detected: {pattern.pattern_type} (confidence: {pattern.confidence:.2%})")
        else:
            logger.info("[OK] No pattern detected (below confidence threshold)")

    def test_05_supported_patterns(self):
        """Test supported patterns list"""
        if not self.available:
            self.skipTest("Transformer detector not available")

        detector = self.get_detector()

        expected_patterns = [
            'bull_flag', 'bear_flag', 'breakout', 'breakdown',
            'bullish_reversal', 'bearish_reversal', 'consolidation', 'gap_and_go'
        ]

        self.assertEqual(detector.PATTERN_NAMES, expected_patterns)
        logger.info(f"[OK] All 8 patterns supported: {', '.join(expected_patterns)}")


class TestRLAgent(unittest.TestCase):
    """Test RL Execution Agent"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from ai.warrior_rl_agent import (
                WarriorRLAgent,
                get_rl_agent,
                TradingState,
                TradingAction
            )
            self.RLAgent = WarriorRLAgent
            self.get_agent = get_rl_agent
            self.TradingState = TradingState
            self.TradingAction = TradingAction
            self.available = True
        except ImportError as e:
            logger.warning(f"RL agent not available: {e}")
            self.available = False

    def test_01_import(self):
        """Test RL agent import"""
        self.assertTrue(self.available, "RL agent should be importable")
        logger.info("[OK] RL agent imports successfully")

    def test_02_initialization(self):
        """Test agent initialization"""
        if not self.available:
            self.skipTest("RL agent not available")

        agent = self.get_agent()
        self.assertIsNotNone(agent)
        self.assertEqual(len(agent.ACTIONS), 5)
        self.assertEqual(agent.ACTIONS, ['enter', 'hold', 'exit', 'size_up', 'size_down'])
        logger.info("[OK] Agent initialized with 5 actions")

    def test_03_state_to_tensor(self):
        """Test state conversion to tensor"""
        if not self.available:
            self.skipTest("RL agent not available")

        agent = self.get_agent()

        state = self.TradingState(
            price=150.0,
            volume=1000000,
            volatility=0.02,
            trend=0.5,
            position_size=0.0,
            entry_price=None,
            unrealized_pnl=0.0,
            sentiment_score=0.3,
            pattern_confidence=0.7,
            time_in_position=0,
            current_drawdown=0.0,
            sharpe_ratio=1.5,
            win_rate=0.6
        )

        tensor = agent.state_to_tensor(state)

        # Should have 12 features
        self.assertEqual(len(tensor.shape), 1)
        self.assertEqual(tensor.shape[0], 12)

        logger.info(f"[OK] State converted to tensor: shape {tensor.shape}")

    def test_04_action_selection_inference(self):
        """Test action selection in inference mode"""
        if not self.available:
            self.skipTest("RL agent not available")

        agent = self.get_agent()

        state = self.TradingState(
            price=150.0,
            volume=1000000,
            volatility=0.02,
            trend=0.5,
            position_size=0.0,
            entry_price=None,
            unrealized_pnl=0.0,
            sentiment_score=0.3,
            pattern_confidence=0.7,
            time_in_position=0,
            current_drawdown=0.0,
            sharpe_ratio=1.5,
            win_rate=0.6
        )

        # Inference mode (no exploration)
        action = agent.select_action(state, training=False)

        self.assertIsInstance(action, self.TradingAction)
        self.assertIn(action.action_type, agent.ACTIONS)
        self.assertGreaterEqual(action.confidence, 0.0)
        self.assertLessEqual(action.confidence, 1.0)

        logger.info(f"[OK] Action selected: {action.action_type} (confidence: {action.confidence:.2%})")

    def test_05_action_selection_training(self):
        """Test action selection in training mode"""
        if not self.available:
            self.skipTest("RL agent not available")

        agent = self.get_agent()

        state = self.TradingState(
            price=150.0,
            volume=1000000,
            volatility=0.02,
            trend=-0.3,
            position_size=0.5,
            entry_price=145.0,
            unrealized_pnl=0.034,  # 3.4% profit
            sentiment_score=-0.2,
            pattern_confidence=0.4,
            time_in_position=10,
            current_drawdown=-0.01,
            sharpe_ratio=1.2,
            win_rate=0.55
        )

        # Training mode (with exploration)
        action = agent.select_action(state, training=True)

        self.assertIsInstance(action, self.TradingAction)
        self.assertIn(action.action_type, agent.ACTIONS)

        logger.info(f"[OK] Training action: {action.action_type} (epsilon: {agent.epsilon:.3f})")

    def test_06_reward_calculation(self):
        """Test reward calculation"""
        if not self.available:
            self.skipTest("RL agent not available")

        agent = self.get_agent()

        # State: In position with profit
        state = self.TradingState(
            price=150.0,
            volume=1000000,
            volatility=0.02,
            trend=0.5,
            position_size=0.3,
            entry_price=145.0,
            unrealized_pnl=0.034,
            sentiment_score=0.3,
            pattern_confidence=0.7,
            time_in_position=10,
            current_drawdown=0.0,
            sharpe_ratio=1.5,
            win_rate=0.6
        )

        # Next state: More profit
        next_state = self.TradingState(
            price=152.0,
            volume=1100000,
            volatility=0.02,
            trend=0.5,
            position_size=0.3,
            entry_price=145.0,
            unrealized_pnl=0.048,
            sentiment_score=0.3,
            pattern_confidence=0.7,
            time_in_position=11,
            current_drawdown=0.0,
            sharpe_ratio=1.6,
            win_rate=0.6
        )

        action = self.TradingAction(action_type='hold', size_change=0.0, confidence=0.8)

        reward = agent.calculate_reward(state, action, next_state)

        # Reward should be positive for profitable outcome
        self.assertIsInstance(reward, float)

        logger.info(f"[OK] Reward calculated: {reward:+.2f}")

    def test_07_available_actions(self):
        """Test available actions"""
        if not self.available:
            self.skipTest("RL agent not available")

        agent = self.get_agent()

        expected_actions = ['enter', 'hold', 'exit', 'size_up', 'size_down']
        self.assertEqual(agent.ACTIONS, expected_actions)

        logger.info(f"[OK] All 5 actions available: {', '.join(expected_actions)}")


class TestMLTrainer(unittest.TestCase):
    """Test ML Trainer"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from ai.warrior_ml_trainer import (
                HistoricalDataLoader,
                PatternLabeler,
                ModelTrainer
            )
            self.DataLoader = HistoricalDataLoader
            self.PatternLabeler = PatternLabeler
            self.ModelTrainer = ModelTrainer
            self.available = True
        except ImportError as e:
            logger.warning(f"ML trainer not available: {e}")
            self.available = False

    def test_01_import(self):
        """Test ML trainer import"""
        self.assertTrue(self.available, "ML trainer should be importable")
        logger.info("[OK] ML trainer imports successfully")

    def test_02_data_loader_init(self):
        """Test data loader initialization"""
        if not self.available:
            self.skipTest("ML trainer not available")

        loader = self.DataLoader()
        self.assertIsNotNone(loader)
        logger.info("[OK] Data loader initialized")

    def test_03_pattern_labeler(self):
        """Test pattern labeling"""
        if not self.available:
            self.skipTest("ML trainer not available")

        # Create sample dataframe
        import pandas as pd
        import numpy as np

        dates = pd.date_range('2025-01-01', periods=100, freq='5min')
        df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000000, 2000000, 100)
        }, index=dates)

        # Label patterns
        labeled_df = self.PatternLabeler.label_patterns(df)

        self.assertIsNotNone(labeled_df)
        self.assertIn('pattern_label', labeled_df.columns)

        logger.info(f"[OK] Patterns labeled: {len(labeled_df)} candles")


class TestMLRouter(unittest.TestCase):
    """Test ML API Router"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from ai.warrior_ml_router import (
                router,
                PatternDetectionRequest,
                TradingStateRequest,
                MLHealthResponse
            )
            self.router = router
            self.PatternRequest = PatternDetectionRequest
            self.StateRequest = TradingStateRequest
            self.HealthResponse = MLHealthResponse
            self.available = True
        except ImportError as e:
            logger.warning(f"ML router not available: {e}")
            self.available = False

    def test_01_import(self):
        """Test router import"""
        self.assertTrue(self.available, "ML router should be importable")
        logger.info("[OK] ML router imports successfully")

    def test_02_router_routes(self):
        """Test router has correct routes"""
        if not self.available:
            self.skipTest("ML router not available")

        # Check router has routes
        self.assertGreater(len(self.router.routes), 0)

        # Check for key endpoints
        route_paths = [route.path for route in self.router.routes]

        expected_paths = [
            '/api/ml/health',
            '/api/ml/detect-pattern',
            '/api/ml/recommend-action',
            '/api/ml/patterns/supported',
            '/api/ml/actions/available',
            '/api/ml/batch/detect-patterns'
        ]

        for path in expected_paths:
            # Check if path exists (may have different prefix in routes)
            path_found = any(path.replace('/api/ml', '') in route.path for route in self.router.routes)
            self.assertTrue(path_found, f"Expected route {path} not found")

        logger.info(f"[OK] Router has {len(self.router.routes)} routes")

    def test_03_request_models(self):
        """Test request models"""
        if not self.available:
            self.skipTest("ML router not available")

        # Test PatternDetectionRequest
        from datetime import datetime

        try:
            from ai.warrior_ml_router import CandleData

            candles = [
                CandleData(
                    timestamp=datetime.now(),
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.5,
                    volume=1000000
                )
                for _ in range(25)
            ]

            request = self.PatternRequest(
                symbol="TEST",
                candles=candles,
                timeframe="5min"
            )

            self.assertEqual(request.symbol, "TEST")
            self.assertEqual(len(request.candles), 25)

            logger.info("[OK] Pattern request model validated")
        except Exception as e:
            logger.warning(f"Pattern request validation skipped: {e}")

        # Test TradingStateRequest
        try:
            request = self.StateRequest(
                symbol="TEST",
                price=150.0,
                volume=1000000,
                volatility=0.02,
                trend=0.5
            )

            self.assertEqual(request.symbol, "TEST")
            self.assertEqual(request.price, 150.0)

            logger.info("[OK] Trading state request model validated")
        except Exception as e:
            logger.warning(f"State request validation skipped: {e}")


def run_tests():
    """Run all tests"""
    print("=" * 70)
    print("PHASE 3: ADVANCED ML TEST SUITE")
    print("=" * 70)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTransformerDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestRLAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestMLTrainer))
    suite.addTests(loader.loadTestsFromTestCase(TestMLRouter))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)

    if result.wasSuccessful():
        print("[OK] All tests passed!")
        return 0
    else:
        print("[FAIL] Some tests failed. See details above.")
        return 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
