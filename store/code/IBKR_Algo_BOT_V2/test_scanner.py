"""
Quick test script for Warrior Trading Scanner

Run this to verify the scanner is working correctly.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai.warrior_scanner import WarriorScanner
from config.config_loader import get_config


def main():
    """Test the Warrior Trading Scanner"""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("=" * 80)
    print("WARRIOR TRADING SCANNER - TEST MODE")
    print("=" * 80)

    try:
        # Test 1: Load configuration
        print("\n[1/3] Loading configuration...")
        config = get_config()
        print(f"[OK] Configuration loaded successfully")
        print(f"    - Scanner enabled: {config.scanner.enabled}")
        print(f"    - Min gap: {config.scanner.min_gap_percent}%")
        print(f"    - Min RVOL: {config.scanner.min_rvol}")
        print(f"    - Max float: {config.scanner.max_float_millions}M")

        # Test 2: Initialize scanner
        print("\n[2/3] Initializing scanner...")
        scanner = WarriorScanner()
        print("[OK] Scanner initialized")

        # Test 3: Run scan (with lower requirements for testing)
        print("\n[3/3] Running pre-market scan...")
        print("    (Using relaxed criteria for testing)")

        candidates = scanner.scan_premarket(
            min_gap_percent=3.0,  # Lower for testing
            min_rvol=1.5,  # Lower for testing
            max_float=100.0,  # Higher for testing
        )

        if not candidates:
            print("[WARN]  No candidates found (this is normal if markets are closed)")
            print("    Try running during market hours (9:30 AM - 4:00 PM ET)")
        else:
            print(f"\n[OK] Found {len(candidates)} candidates!\n")

            # Display results
            print(
                f"{'Symbol':8} {'Price':>8} {'Gap %':>8} {'RVOL':>6} "
                f"{'Float':>8} {'Score':>6}"
            )
            print("-" * 60)

            for c in candidates[:10]:  # Show top 10
                print(
                    f"{c.symbol:8} ${c.price:7.2f} {c.gap_percent:+7.1f}% "
                    f"{c.relative_volume:6.1f} {c.float_shares:7.1f}M "
                    f"{c.confidence_score:6.0f}"
                )

            # Show details for #1
            if candidates:
                print(f"\nTOP CANDIDATE DETAILS:")
                top = candidates[0]
                print(f"  Symbol: {top.symbol}")
                print(f"  Price: ${top.price:.2f}")
                print(f"  Gap: {top.gap_percent:+.1f}%")
                print(f"  RVOL: {top.relative_volume:.1f}x")
                print(f"  Float: {top.float_shares:.1f}M shares")
                print(f"  Daily: {top.daily_chart_signal}")
                print(f"  Catalyst: {top.catalyst}")
                print(f"  Score: {top.confidence_score:.0f}/100")

        print("\n" + "=" * 80)
        print("[OK] ALL TESTS PASSED")
        print("=" * 80)

        return True

    except ImportError as e:
        print(f"\n[FAIL] Import Error: {e}")
        print("\nMissing dependencies. Install with:")
        print("  pip install -r requirements_warrior.txt")
        return False

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
