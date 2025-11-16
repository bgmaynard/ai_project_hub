"""
Test script for Warrior Trading Pattern Detector

Tests all 5 pattern types with real market data
"""

import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai.warrior_pattern_detector import WarriorPatternDetector, SetupType
from config.config_loader import get_config

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


def test_pattern_detector():
    """Test the pattern detector with real data"""

    print("=" * 80)
    print("WARRIOR TRADING PATTERN DETECTOR - TEST MODE")
    print("=" * 80)

    if not YFINANCE_AVAILABLE:
        print("\n[FAIL] yfinance not installed")
        print("Install with: pip install yfinance")
        return False

    try:
        # Test 1: Load configuration
        print("\n[1/4] Loading configuration...")
        config = get_config()
        print(f"[OK] Configuration loaded")
        print(f"    - Enabled patterns: {config.patterns.enabled_patterns}")

        # Test 2: Initialize detector
        print("\n[2/4] Initializing pattern detector...")
        detector = WarriorPatternDetector()
        print("[OK] Pattern detector initialized")

        # Test 3: Fetch market data
        print("\n[3/4] Fetching market data...")

        # Test with multiple symbols
        test_symbols = ["TSLA", "AMD", "NVDA", "AAPL"]

        for symbol in test_symbols:
            print(f"\n{'=' * 60}")
            print(f"Testing: {symbol}")
            print('=' * 60)

            try:
                ticker = yf.Ticker(symbol)
                hist_5m = ticker.history(period="1d", interval="5m")
                hist_1m = ticker.history(period="1d", interval="1m")

                if hist_5m.empty or hist_1m.empty:
                    print(f"  [WARN]  No data available for {symbol}")
                    continue

                # Calculate indicators
                vwap = (hist_5m['Close'] * hist_5m['Volume']).sum() / hist_5m['Volume'].sum()
                ema9_5m = hist_5m['Close'].ewm(span=9, adjust=False).mean().iloc[-1]
                ema20_5m = hist_5m['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
                ema9_1m = hist_1m['Close'].ewm(span=9, adjust=False).mean().iloc[-1]
                hod = hist_1m['High'].max()
                current_price = hist_5m['Close'].iloc[-1]

                print(f"\n  Market Data:")
                print(f"    Current Price: ${current_price:.2f}")
                print(f"    VWAP: ${vwap:.2f}")
                print(f"    9 EMA (5m): ${ema9_5m:.2f}")
                print(f"    20 EMA (5m): ${ema20_5m:.2f}")
                print(f"    High of Day: ${hod:.2f}")

                # Test 4: Detect patterns
                print(f"\n  Detecting patterns...")
                setups = detector.detect_all_patterns(
                    symbol=symbol,
                    candles_1m=hist_1m,
                    candles_5m=hist_5m,
                    vwap=vwap,
                    ema9_1m=ema9_1m,
                    ema9_5m=ema9_5m,
                    ema20_5m=ema20_5m,
                    high_of_day=hod
                )

                if not setups:
                    print(f"    No patterns detected")
                else:
                    print(f"\n  [OK] Found {len(setups)} pattern(s):\n")

                    for i, setup in enumerate(setups, 1):
                        print(f"  [{i}] {setup.setup_type.value}")
                        print(f"      Timeframe: {setup.timeframe}")
                        print(f"      Entry: ${setup.entry_price:.2f}")
                        print(f"      Stop: ${setup.stop_price:.2f}")
                        print(f"      Target 2R: ${setup.target_2r:.2f}")
                        print(f"      Risk/Share: ${setup.risk_per_share:.2f}")
                        print(f"      Reward/Share: ${setup.reward_per_share:.2f}")
                        print(f"      R:R Ratio: {setup.risk_reward_ratio:.1f}:1")
                        print(f"      Confidence: {setup.confidence:.0f}%")
                        print(f"      Entry Condition: {setup.entry_condition}")

                        if setup.strength_factors:
                            print(f"      [OK] Strengths:")
                            for factor in setup.strength_factors:
                                print(f"        • {factor}")

                        if setup.risk_factors:
                            print(f"      ⚠ Risks:")
                            for factor in setup.risk_factors:
                                print(f"        • {factor}")

                        print()

            except Exception as e:
                print(f"  [FAIL] Error testing {symbol}: {e}")
                continue

        print("\n" + "=" * 80)
        print("[OK] PATTERN DETECTOR TESTS COMPLETE")
        print("=" * 80)

        # Test individual pattern methods
        print("\n[4/4] Testing individual pattern methods...")
        test_individual_patterns()

        return True

    except ImportError as e:
        print(f"\n[FAIL] Import Error: {e}")
        print("\nMissing dependencies. Install with:")
        print("  pip install yfinance pandas numpy")
        return False

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_patterns():
    """Test individual pattern detection methods"""

    print("\nTesting individual pattern methods...")

    try:
        import yfinance as yf

        detector = WarriorPatternDetector()
        ticker = yf.Ticker("TSLA")
        hist_5m = ticker.history(period="1d", interval="5m")

        if hist_5m.empty:
            print("  [WARN]  No data available")
            return

        vwap = (hist_5m['Close'] * hist_5m['Volume']).sum() / hist_5m['Volume'].sum()
        ema9 = hist_5m['Close'].ewm(span=9, adjust=False).mean().iloc[-1]
        ema20 = hist_5m['Close'].ewm(span=20, adjust=False).mean().iloc[-1]

        # Test bull flag
        print("\n  Testing Bull Flag detector...")
        bull_flag = detector.detect_bull_flag("TSLA", hist_5m, vwap, ema9, ema20)
        if bull_flag:
            print(f"    [OK] Bull Flag detected (confidence: {bull_flag.confidence:.0f}%)")
        else:
            print(f"    ℹ️  No Bull Flag pattern found")

        # Test HOD breakout
        print("\n  Testing HOD Breakout detector...")
        hist_1m = ticker.history(period="1d", interval="1m")
        if not hist_1m.empty:
            hod = hist_1m['High'].max()
            ema9_1m = hist_1m['Close'].ewm(span=9, adjust=False).mean().iloc[-1]
            hod_setup = detector.detect_hod_breakout("TSLA", hist_1m, hod, vwap, ema9_1m)
            if hod_setup:
                print(f"    [OK] HOD Breakout detected (confidence: {hod_setup.confidence:.0f}%)")
            else:
                print(f"    ℹ️  No HOD Breakout pattern found")

        # Test Whole Dollar
        print("\n  Testing Whole Dollar Breakout detector...")
        if not hist_1m.empty:
            whole_dollar = detector.detect_whole_dollar_breakout("TSLA", hist_1m, vwap, ema9_1m)
            if whole_dollar:
                print(f"    [OK] Whole Dollar detected (confidence: {whole_dollar.confidence:.0f}%)")
            else:
                print(f"    ℹ️  No Whole Dollar pattern found")

        # Test Micro Pullback
        print("\n  Testing Micro Pullback detector...")
        pullback = detector.detect_micro_pullback("TSLA", hist_5m, vwap, ema9, ema20)
        if pullback:
            print(f"    [OK] Micro Pullback detected (confidence: {pullback.confidence:.0f}%)")
        else:
            print(f"    ℹ️  No Micro Pullback pattern found")

        print("\n  [OK] Individual pattern tests complete")

    except Exception as e:
        print(f"  [FAIL] Error testing individual patterns: {e}")


def print_pattern_config():
    """Print current pattern configuration"""

    print("\nCurrent Pattern Configuration:")
    print("-" * 60)

    config = get_config()

    print(f"\nEnabled Patterns:")
    for pattern in config.patterns.enabled_patterns:
        print(f"  [OK] {pattern}")

    print(f"\nBull Flag Settings:")
    for key, value in config.patterns.bull_flag.items():
        print(f"  {key}: {value}")

    print(f"\nHOD Breakout Settings:")
    for key, value in config.patterns.hod_breakout.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Print configuration
    print_pattern_config()

    # Run tests
    success = test_pattern_detector()

    sys.exit(0 if success else 1)
