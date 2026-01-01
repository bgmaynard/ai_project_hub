"""
Test script for Warrior Trading API endpoints

Tests all REST endpoints and WebSocket functionality
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import httpx

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# API base URL
BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/warrior"


class WarriorAPITester:
    """Test harness for Warrior Trading API"""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.tests_passed = 0
        self.tests_failed = 0

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    def log_result(self, test_name: str, passed: bool, message: str = ""):
        """Log test result"""
        if passed:
            self.tests_passed += 1
            logger.info(f"[OK] {test_name}: PASSED {message}")
        else:
            self.tests_failed += 1
            logger.error(f"[FAIL] {test_name}: FAILED {message}")

    async def test_health_check(self):
        """Test 1: Health check endpoint"""
        print("\n" + "=" * 80)
        print("TEST 1: Health Check")
        print("=" * 80)

        try:
            response = await self.client.get(f"{self.base_url}{API_PREFIX}/health")

            if response.status_code == 200:
                data = response.json()
                print(f"Status: {data.get('status')}")
                print(f"Timestamp: {data.get('timestamp')}")
                self.log_result("Health Check", True)
                return True
            else:
                self.log_result("Health Check", False, f"Status {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Health Check", False, str(e))
            return False

    async def test_system_status(self):
        """Test 2: System status endpoint"""
        print("\n" + "=" * 80)
        print("TEST 2: System Status")
        print("=" * 80)

        try:
            response = await self.client.get(f"{self.base_url}{API_PREFIX}/status")

            if response.status_code == 200:
                data = response.json()
                print(f"Scanner initialized: {data.get('scanner_initialized')}")
                print(f"Detector initialized: {data.get('detector_initialized')}")
                print(
                    f"Risk Manager initialized: {data.get('risk_manager_initialized')}"
                )
                print(f"Watchlist size: {data.get('watchlist_size')}")
                print(f"Trading halted: {data.get('is_trading_halted')}")
                self.log_result("System Status", True)
                return True
            else:
                self.log_result(
                    "System Status", False, f"Status {response.status_code}"
                )
                return False
        except Exception as e:
            self.log_result("System Status", False, str(e))
            return False

    async def test_configuration(self):
        """Test 3: Configuration endpoint"""
        print("\n" + "=" * 80)
        print("TEST 3: Configuration")
        print("=" * 80)

        try:
            response = await self.client.get(f"{self.base_url}{API_PREFIX}/config")

            if response.status_code == 200:
                data = response.json()
                print(
                    f"Daily profit goal: ${data.get('risk_management', {}).get('daily_profit_goal')}"
                )
                print(
                    f"Max loss per trade: ${data.get('risk_management', {}).get('max_loss_per_trade')}"
                )
                print(
                    f"Min R:R ratio: {data.get('risk_management', {}).get('min_reward_to_risk')}:1"
                )
                print(
                    f"Enabled patterns: {len(data.get('pattern_detection', {}).get('enabled_patterns', []))}"
                )
                self.log_result("Configuration", True)
                return True
            else:
                self.log_result(
                    "Configuration", False, f"Status {response.status_code}"
                )
                return False
        except Exception as e:
            self.log_result("Configuration", False, str(e))
            return False

    async def test_premarket_scan(self):
        """Test 4: Pre-market scan endpoint"""
        print("\n" + "=" * 80)
        print("TEST 4: Pre-Market Scan")
        print("=" * 80)

        try:
            # Run scan with default parameters
            response = await self.client.post(
                f"{self.base_url}{API_PREFIX}/scan/premarket", json={}
            )

            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                print(f"Scan completed: {data.get('success')}")
                print(f"Candidates found: {len(candidates)}")
                print(f"Scan time: {data.get('scan_time_seconds'):.2f}s")

                if candidates:
                    print(f"\nTop candidate: {candidates[0].get('symbol')}")
                    print(f"  Gap: {candidates[0].get('gap_percent'):.1f}%")
                    print(f"  RVOL: {candidates[0].get('relative_volume'):.1f}x")
                    print(f"  Confidence: {candidates[0].get('confidence_score'):.0f}%")

                self.log_result("Pre-Market Scan", True, f"({len(candidates)} found)")
                return candidates
            else:
                self.log_result(
                    "Pre-Market Scan", False, f"Status {response.status_code}"
                )
                return []
        except Exception as e:
            self.log_result("Pre-Market Scan", False, str(e))
            return []

    async def test_watchlist(self):
        """Test 5: Watchlist endpoint"""
        print("\n" + "=" * 80)
        print("TEST 5: Watchlist Retrieval")
        print("=" * 80)

        try:
            response = await self.client.get(f"{self.base_url}{API_PREFIX}/watchlist")

            if response.status_code == 200:
                data = response.json()
                watchlist = data.get("watchlist", [])
                print(f"Watchlist size: {len(watchlist)}")
                print(f"Last scan: {data.get('last_scan_time', 'Never')}")

                if watchlist:
                    print(f"\nTop 3 symbols:")
                    for i, candidate in enumerate(watchlist[:3], 1):
                        print(
                            f"  {i}. {candidate.get('symbol')} - "
                            f"Gap: {candidate.get('gap_percent'):.1f}%, "
                            f"RVOL: {candidate.get('relative_volume'):.1f}x"
                        )

                self.log_result("Watchlist Retrieval", True)
                return watchlist
            else:
                self.log_result(
                    "Watchlist Retrieval", False, f"Status {response.status_code}"
                )
                return []
        except Exception as e:
            self.log_result("Watchlist Retrieval", False, str(e))
            return []

    async def test_risk_status(self):
        """Test 6: Risk manager status endpoint"""
        print("\n" + "=" * 80)
        print("TEST 6: Risk Manager Status")
        print("=" * 80)

        try:
            response = await self.client.get(f"{self.base_url}{API_PREFIX}/risk/status")

            if response.status_code == 200:
                data = response.json()
                print(f"Trading halted: {data.get('is_halted')}")
                print(f"Open positions: {data.get('open_positions')}")
                print(f"Total trades: {data.get('total_trades')}")
                print(f"Win rate: {data.get('win_rate'):.1f}%")
                print(f"Current P&L: ${data.get('current_pnl'):+.2f}")
                print(f"Distance to goal: ${data.get('distance_to_goal'):.2f}")
                print(f"Consecutive losses: {data.get('consecutive_losses')}")

                self.log_result("Risk Manager Status", True)
                return True
            else:
                self.log_result(
                    "Risk Manager Status", False, f"Status {response.status_code}"
                )
                return False
        except Exception as e:
            self.log_result("Risk Manager Status", False, str(e))
            return False

    async def test_trade_history(self):
        """Test 7: Trade history endpoint"""
        print("\n" + "=" * 80)
        print("TEST 7: Trade History")
        print("=" * 80)

        try:
            response = await self.client.get(
                f"{self.base_url}{API_PREFIX}/trades/history"
            )

            if response.status_code == 200:
                data = response.json()
                trades = data.get("trades", [])
                print(f"Total trades: {len(trades)}")

                if trades:
                    print(f"\nMost recent trade:")
                    recent = trades[-1]
                    print(f"  Symbol: {recent.get('symbol')}")
                    print(f"  Setup: {recent.get('setup_type')}")
                    print(f"  Entry: ${recent.get('entry_price'):.2f}")
                    print(f"  Exit: ${recent.get('exit_price', 0):.2f}")
                    print(f"  P&L: ${recent.get('pnl', 0):+.2f}")
                    print(f"  R: {recent.get('r_multiple', 0):+.2f}R")

                self.log_result("Trade History", True, f"({len(trades)} trades)")
                return True
            else:
                self.log_result(
                    "Trade History", False, f"Status {response.status_code}"
                )
                return False
        except Exception as e:
            self.log_result("Trade History", False, str(e))
            return False

    async def test_mock_trade_lifecycle(self):
        """Test 8: Mock trade entry and exit"""
        print("\n" + "=" * 80)
        print("TEST 8: Trade Entry & Exit (Mock)")
        print("=" * 80)

        try:
            # Enter a mock trade
            entry_request = {
                "symbol": "TEST",
                "setup_type": "BULL_FLAG",
                "entry_price": 10.00,
                "shares": 100,
                "stop_price": 9.80,
                "target_price": 10.40,
            }

            print("\nEntering mock trade...")
            entry_response = await self.client.post(
                f"{self.base_url}{API_PREFIX}/trades/enter", json=entry_request
            )

            if entry_response.status_code == 200:
                entry_data = entry_response.json()
                trade_id = entry_data.get("trade_id")
                print(f"Trade entered: {trade_id}")
                print(f"  Symbol: {entry_data.get('symbol')}")
                print(f"  Shares: {entry_data.get('shares')}")
                print(f"  Entry: ${entry_data.get('entry_price'):.2f}")
                print(f"  Stop: ${entry_data.get('stop_price'):.2f}")
                print(f"  Target: ${entry_data.get('target_price'):.2f}")

                # Exit the trade (winner)
                print("\nExiting mock trade (winner)...")
                exit_request = {
                    "trade_id": trade_id,
                    "exit_price": 10.35,
                    "exit_reason": "TARGET_HIT",
                }

                exit_response = await self.client.post(
                    f"{self.base_url}{API_PREFIX}/trades/exit", json=exit_request
                )

                if exit_response.status_code == 200:
                    exit_data = exit_response.json()
                    print(f"Trade exited: {exit_data.get('trade_id')}")
                    print(f"  Exit price: ${exit_data.get('exit_price'):.2f}")
                    print(f"  P&L: ${exit_data.get('pnl'):+.2f}")
                    print(f"  R multiple: {exit_data.get('r_multiple'):+.2f}R")

                    self.log_result("Trade Entry & Exit", True)
                    return True
                else:
                    self.log_result("Trade Entry & Exit", False, "Exit failed")
                    return False
            else:
                self.log_result("Trade Entry & Exit", False, "Entry failed")
                return False
        except Exception as e:
            self.log_result("Trade Entry & Exit", False, str(e))
            return False

    async def test_reset_daily_stats(self):
        """Test 9: Reset daily statistics"""
        print("\n" + "=" * 80)
        print("TEST 9: Reset Daily Stats")
        print("=" * 80)

        try:
            response = await self.client.post(
                f"{self.base_url}{API_PREFIX}/risk/reset-daily"
            )

            if response.status_code == 200:
                data = response.json()
                print(f"Reset successful: {data.get('success')}")
                print(f"Message: {data.get('message')}")

                self.log_result("Reset Daily Stats", True)
                return True
            else:
                self.log_result(
                    "Reset Daily Stats", False, f"Status {response.status_code}"
                )
                return False
        except Exception as e:
            self.log_result("Reset Daily Stats", False, str(e))
            return False

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        total_tests = self.tests_passed + self.tests_failed
        pass_rate = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0

        print(f"Total tests: {total_tests}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        print(f"Pass rate: {pass_rate:.1f}%")

        if self.tests_failed == 0:
            print("\n[OK] ALL TESTS PASSED!")
        else:
            print(f"\n[WARN]  {self.tests_failed} test(s) failed")


async def run_all_tests():
    """Run all Warrior Trading API tests"""
    print("=" * 80)
    print("WARRIOR TRADING API - TEST SUITE")
    print("=" * 80)
    print(f"Testing API at: {BASE_URL}{API_PREFIX}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tester = WarriorAPITester()

    try:
        # Run tests in sequence
        await tester.test_health_check()
        await tester.test_system_status()
        await tester.test_configuration()
        await tester.test_premarket_scan()
        await tester.test_watchlist()
        await tester.test_risk_status()
        await tester.test_trade_history()
        await tester.test_mock_trade_lifecycle()
        await tester.test_reset_daily_stats()

        # Print summary
        tester.print_summary()

        return tester.tests_failed == 0

    finally:
        await tester.close()


async def test_specific_endpoint(endpoint: str):
    """Test a specific endpoint"""
    tester = WarriorAPITester()

    try:
        if endpoint == "health":
            await tester.test_health_check()
        elif endpoint == "status":
            await tester.test_system_status()
        elif endpoint == "config":
            await tester.test_configuration()
        elif endpoint == "scan":
            await tester.test_premarket_scan()
        elif endpoint == "watchlist":
            await tester.test_watchlist()
        elif endpoint == "risk":
            await tester.test_risk_status()
        elif endpoint == "trades":
            await tester.test_trade_history()
        elif endpoint == "lifecycle":
            await tester.test_mock_trade_lifecycle()
        else:
            print(f"Unknown endpoint: {endpoint}")

        tester.print_summary()

    finally:
        await tester.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Warrior Trading API endpoints")
    parser.add_argument(
        "--endpoint",
        type=str,
        help="Test specific endpoint (health, status, config, scan, watchlist, risk, trades, lifecycle)",
    )
    parser.add_argument(
        "--url", type=str, default=BASE_URL, help=f"API base URL (default: {BASE_URL})"
    )

    args = parser.parse_args()

    if args.url != BASE_URL:
        BASE_URL = args.url

    try:
        if args.endpoint:
            asyncio.run(test_specific_endpoint(args.endpoint))
        else:
            success = asyncio.run(run_all_tests())
            sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
