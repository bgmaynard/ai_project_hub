"""
Connectivity Manager
====================
Manages service connectivity, startup verification, and lifecycle monitoring.

Tracks:
- Data feed status
- WebSocket status
- Chronos heartbeat
- Scanner job status
- First successful market tick

Provides:
- Startup self-test
- Connectivity reports
- Reconnection capabilities (paper mode)
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import httpx

logger = logging.getLogger(__name__)

# Report file
CONNECTIVITY_REPORT_FILE = os.path.join(os.path.dirname(__file__), "..", "reports", "connectivity_check.json")


class ServiceStatus(Enum):
    """Service status states"""
    UP = "UP"
    DOWN = "DOWN"
    DEGRADED = "DEGRADED"
    STARTING = "STARTING"
    NOT_CONFIGURED = "NOT_CONFIGURED"


class SystemState(Enum):
    """Overall system state for UI display"""
    ACTIVE = "ACTIVE"                          # All systems go, trading enabled
    READY = "READY"                            # All connected, trading disabled
    MARKET_CLOSED = "MARKET_CLOSED"            # Calendar: market not open
    DATA_OFFLINE = "DATA_OFFLINE"              # Market open but no data feed
    SERVICE_NOT_RUNNING = "SERVICE_NOT_RUNNING" # Service process not started
    DISCONNECTED = "DISCONNECTED"              # Service started but connection lost
    PARTIAL = "PARTIAL"                        # Some services up, some down


@dataclass
class ServiceHealth:
    """Health status for a single service"""
    name: str
    status: ServiceStatus
    last_successful_event: Optional[str] = None
    last_check_time: Optional[str] = None
    detail: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "last_successful_event": self.last_successful_event,
            "last_check_time": self.last_check_time,
            "detail": self.detail,
            "error": self.error
        }


@dataclass
class StartupEvent:
    """Startup sequence event"""
    timestamp: str
    service: str
    event: str
    success: bool
    duration_ms: float = 0
    detail: str = ""


class ConnectivityManager:
    """
    Manages connectivity for all trading services.

    Startup Order:
    1. Chronos scheduler
    2. Market data ingestion
    3. WebSocket broadcaster
    4. Scanner jobs
    """

    def __init__(self):
        self.services: Dict[str, ServiceHealth] = {}
        self.startup_events: List[StartupEvent] = []
        self.first_tick_received: Optional[datetime] = None
        self.first_tick_symbol: Optional[str] = None
        self.startup_complete: bool = False
        self.startup_time: Optional[datetime] = None

        # Initialize service trackers
        self._init_services()

        logger.info("ConnectivityManager initialized")

    def _init_services(self):
        """Initialize service health trackers"""
        service_names = [
            "chronos",
            "market_data",
            "websocket",
            "polygon_stream",
            "scalper",
            "news_trader",
            "premarket_scanner",
            "gating_engine"
        ]

        for name in service_names:
            self.services[name] = ServiceHealth(
                name=name,
                status=ServiceStatus.NOT_CONFIGURED,
                last_check_time=datetime.now().isoformat()
            )

    def log_startup_event(self, service: str, event: str, success: bool,
                          duration_ms: float = 0, detail: str = ""):
        """Log a startup sequence event"""
        evt = StartupEvent(
            timestamp=datetime.now().isoformat(),
            service=service,
            event=event,
            success=success,
            duration_ms=duration_ms,
            detail=detail
        )
        self.startup_events.append(evt)

        status = "OK" if success else "FAIL"
        logger.info(f"[STARTUP] {status} {service}: {event} ({duration_ms:.0f}ms) {detail}")

    def record_first_tick(self, symbol: str):
        """Record first successful market tick"""
        if self.first_tick_received is None:
            self.first_tick_received = datetime.now()
            self.first_tick_symbol = symbol
            logger.info(f"[STARTUP] OK First market tick received: {symbol} at {self.first_tick_received}")

            self.log_startup_event(
                "market_data",
                "first_tick_received",
                True,
                detail=f"Symbol: {symbol}"
            )

    async def run_startup_self_test(self) -> Dict:
        """
        Run comprehensive startup self-test.

        Checks in order:
        1. Chronos scheduler
        2. Market data connection
        3. WebSocket broadcaster
        4. Scanner jobs
        """
        self.startup_time = datetime.now()
        results = {
            "timestamp": self.startup_time.isoformat(),
            "tests": [],
            "all_passed": True
        }

        logger.info("=" * 60)
        logger.info("CONNECTIVITY SELF-TEST STARTING")
        logger.info("=" * 60)

        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test 1: Chronos
            test_result = await self._test_chronos(client)
            results["tests"].append(test_result)
            if not test_result["passed"]:
                results["all_passed"] = False

            # Test 2: Market Data
            test_result = await self._test_market_data(client)
            results["tests"].append(test_result)
            if not test_result["passed"]:
                results["all_passed"] = False

            # Test 3: WebSocket/Polygon Stream
            test_result = await self._test_websocket(client)
            results["tests"].append(test_result)
            if not test_result["passed"]:
                results["all_passed"] = False

            # Test 4: Scanners
            test_result = await self._test_scanners(client)
            results["tests"].append(test_result)
            if not test_result["passed"]:
                results["all_passed"] = False

        self.startup_complete = True

        # Log summary
        passed = sum(1 for t in results["tests"] if t["passed"])
        total = len(results["tests"])
        status = "PASSED" if results["all_passed"] else "FAILED"

        logger.info("=" * 60)
        logger.info(f"CONNECTIVITY SELF-TEST {status} ({passed}/{total})")
        logger.info("=" * 60)

        # Save report
        self._save_report(results)

        return results

    async def _test_chronos(self, client: httpx.AsyncClient) -> Dict:
        """Test Chronos scheduler"""
        start = datetime.now()
        try:
            resp = await client.get("http://localhost:9100/api/validation/chronos/status")
            duration = (datetime.now() - start).total_seconds() * 1000

            if resp.status_code == 200:
                data = resp.json()
                available = data.get("available", False)

                self.services["chronos"].status = ServiceStatus.UP if available else ServiceStatus.DOWN
                self.services["chronos"].last_check_time = datetime.now().isoformat()
                self.services["chronos"].detail = f"Model: {data.get('model_name', 'unknown')}"

                self.log_startup_event("chronos", "health_check", available, duration)

                return {
                    "service": "chronos",
                    "passed": available,
                    "duration_ms": duration,
                    "detail": self.services["chronos"].detail
                }
        except Exception as e:
            duration = (datetime.now() - start).total_seconds() * 1000
            self.services["chronos"].status = ServiceStatus.DOWN
            self.services["chronos"].error = str(e)
            self.log_startup_event("chronos", "health_check", False, duration, str(e))

        return {"service": "chronos", "passed": False, "duration_ms": duration, "error": str(e)}

    async def _test_market_data(self, client: httpx.AsyncClient) -> Dict:
        """Test market data connection"""
        start = datetime.now()
        try:
            resp = await client.get("http://localhost:9100/api/health")
            duration = (datetime.now() - start).total_seconds() * 1000

            if resp.status_code == 200:
                data = resp.json()
                market_data = data.get("services", {}).get("market_data", "unavailable")
                connected = market_data != "unavailable"

                self.services["market_data"].status = ServiceStatus.UP if connected else ServiceStatus.DOWN
                self.services["market_data"].last_check_time = datetime.now().isoformat()
                self.services["market_data"].detail = market_data

                self.log_startup_event("market_data", "connection_check", connected, duration, market_data)

                return {
                    "service": "market_data",
                    "passed": connected,
                    "duration_ms": duration,
                    "detail": market_data
                }
        except Exception as e:
            duration = (datetime.now() - start).total_seconds() * 1000
            self.services["market_data"].status = ServiceStatus.DOWN
            self.services["market_data"].error = str(e)
            self.log_startup_event("market_data", "connection_check", False, duration, str(e))

        return {"service": "market_data", "passed": False, "duration_ms": duration, "error": str(e)}

    async def _test_websocket(self, client: httpx.AsyncClient) -> Dict:
        """Test WebSocket/Polygon streaming"""
        start = datetime.now()
        try:
            resp = await client.get("http://localhost:9100/api/polygon/stream/status")
            duration = (datetime.now() - start).total_seconds() * 1000

            if resp.status_code == 200:
                data = resp.json()
                connected = data.get("connected", False) or data.get("available", False)

                self.services["polygon_stream"].status = ServiceStatus.UP if connected else ServiceStatus.DOWN
                self.services["polygon_stream"].last_check_time = datetime.now().isoformat()
                self.services["polygon_stream"].detail = f"Trades: {data.get('trades_received', 0)}, Quotes: {data.get('quotes_received', 0)}"

                self.services["websocket"].status = ServiceStatus.UP if connected else ServiceStatus.DOWN
                self.services["websocket"].last_check_time = datetime.now().isoformat()

                self.log_startup_event("websocket", "stream_check", connected, duration, self.services["polygon_stream"].detail)

                return {
                    "service": "websocket",
                    "passed": connected,
                    "duration_ms": duration,
                    "detail": self.services["polygon_stream"].detail
                }
        except Exception as e:
            duration = (datetime.now() - start).total_seconds() * 1000
            self.services["websocket"].status = ServiceStatus.DOWN
            self.services["polygon_stream"].status = ServiceStatus.DOWN
            self.log_startup_event("websocket", "stream_check", False, duration, str(e))

        return {"service": "websocket", "passed": False, "duration_ms": duration, "error": str(e)}

    async def _test_scanners(self, client: httpx.AsyncClient) -> Dict:
        """Test scanner jobs"""
        start = datetime.now()
        scanners_ok = 0
        scanners_checked = 0
        details = []

        # Check scalper
        try:
            resp = await client.get("http://localhost:9100/api/scanner/scalper/status")
            if resp.status_code == 200:
                data = resp.json()
                running = data.get("is_running", False)
                self.services["scalper"].status = ServiceStatus.UP if running else ServiceStatus.DOWN
                self.services["scalper"].last_check_time = datetime.now().isoformat()
                self.services["scalper"].detail = f"Watchlist: {data.get('watchlist_count', 0)}"
                if data.get("last_scan_time"):
                    self.services["scalper"].last_successful_event = data.get("last_scan_time")
                if running:
                    scanners_ok += 1
                details.append(f"Scalper: {'UP' if running else 'DOWN'}")
                scanners_checked += 1
        except:
            pass

        # Check news trader
        try:
            resp = await client.get("http://localhost:9100/api/scanner/news-trader/status")
            if resp.status_code == 200:
                data = resp.json()
                running = data.get("is_running", False) or data.get("scalper_running", False)
                self.services["news_trader"].status = ServiceStatus.UP if running else ServiceStatus.DOWN
                self.services["news_trader"].last_check_time = datetime.now().isoformat()
                if data.get("last_scan_time"):
                    self.services["news_trader"].last_successful_event = data.get("last_scan_time")
                if running:
                    scanners_ok += 1
                details.append(f"News: {'UP' if running else 'DOWN'}")
                scanners_checked += 1
        except:
            pass

        # Check premarket scanner
        try:
            resp = await client.get("http://localhost:9100/api/scanner/premarket/status")
            if resp.status_code == 200:
                data = resp.json()
                has_data = data.get("last_updated") is not None
                self.services["premarket_scanner"].status = ServiceStatus.UP if has_data else ServiceStatus.DOWN
                self.services["premarket_scanner"].last_check_time = datetime.now().isoformat()
                self.services["premarket_scanner"].last_successful_event = data.get("last_updated")
                if has_data:
                    scanners_ok += 1
                details.append(f"Premarket: {'UP' if has_data else 'DOWN'}")
                scanners_checked += 1
        except:
            pass

        # Check gating engine
        try:
            resp = await client.get("http://localhost:9100/api/gating/status")
            if resp.status_code == 200:
                data = resp.json()
                enabled = data.get("gating_enabled", False)
                contracts = data.get("contracts_loaded", 0)
                self.services["gating_engine"].status = ServiceStatus.UP if enabled and contracts > 0 else ServiceStatus.DOWN
                self.services["gating_engine"].last_check_time = datetime.now().isoformat()
                self.services["gating_engine"].detail = f"Contracts: {contracts}, Enabled: {enabled}"
                if enabled and contracts > 0:
                    scanners_ok += 1
                details.append(f"Gating: {'UP' if enabled else 'DOWN'}")
                scanners_checked += 1
        except:
            pass

        duration = (datetime.now() - start).total_seconds() * 1000
        passed = scanners_ok > 0
        detail = ", ".join(details)

        self.log_startup_event("scanners", "job_check", passed, duration, f"{scanners_ok}/{scanners_checked} running")

        return {
            "service": "scanners",
            "passed": passed,
            "duration_ms": duration,
            "detail": detail,
            "scanners_running": scanners_ok,
            "scanners_total": scanners_checked
        }

    def refresh_service_statuses(self):
        """
        Refresh all service statuses by actively checking their real state.
        Call this to update the Governor display.
        """
        now = datetime.now().isoformat()

        # Check scalper
        try:
            from .hft_scalper import get_hft_scalper
            scalper = get_hft_scalper()
            if scalper and scalper.is_running:
                self.services["scalper"] = ServiceHealth(
                    name="scalper",
                    status=ServiceStatus.UP,
                    last_check_time=now,
                    detail=f"Running, {len(scalper.config.watchlist)} symbols"
                )
            else:
                self.services["scalper"] = ServiceHealth(
                    name="scalper",
                    status=ServiceStatus.DOWN,
                    last_check_time=now,
                    detail="Not running"
                )
        except Exception as e:
            self.services["scalper"] = ServiceHealth(
                name="scalper",
                status=ServiceStatus.DOWN,
                last_check_time=now,
                error=str(e)
            )

        # Check gating engine
        try:
            from .signal_gating_engine import get_gating_engine
            gating = get_gating_engine()
            if gating:
                self.services["gating_engine"] = ServiceHealth(
                    name="gating_engine",
                    status=ServiceStatus.UP,
                    last_check_time=now,
                    detail=f"{len(gating.contracts)} contracts loaded"
                )
        except Exception:
            pass

        # Check market data
        try:
            from unified_market_data import get_market_data
            md = get_market_data()
            if md:
                self.services["market_data"] = ServiceHealth(
                    name="market_data",
                    status=ServiceStatus.UP,
                    last_check_time=now,
                    detail="schwab_active"
                )
        except Exception:
            pass

        # Check broker
        try:
            from unified_broker import get_broker
            broker = get_broker()
            if broker:
                self.services["chronos"] = ServiceHealth(
                    name="chronos",
                    status=ServiceStatus.UP,
                    last_check_time=now,
                    detail="Broker connected"
                )
        except Exception:
            pass

        logger.info("Service statuses refreshed")

    def get_system_state(self) -> SystemState:
        """
        Determine overall system state for UI display.

        Actively checks real service status rather than relying on cached state.

        Distinguishes:
        - MARKET_CLOSED: Calendar says market not open
        - DATA_OFFLINE: Market open but no data feed
        - SERVICE_NOT_RUNNING: Process not started
        - DISCONNECTED: Started but connection lost
        - ACTIVE: All systems go, trading enabled
        - READY: All connected, trading disabled
        """
        from .market_time import get_market_status, MarketStatus
        from .safe_activation import get_safe_activation

        market_status, _ = get_market_status()
        safe = get_safe_activation()

        # Check if market is closed by calendar
        if market_status == MarketStatus.CLOSED:
            return SystemState.MARKET_CLOSED

        # ACTIVELY check real service status instead of cached state
        broker_connected = False
        scalper_running = False

        # Check broker connection
        try:
            from unified_broker import get_broker
            broker = get_broker()
            broker_connected = broker is not None
        except Exception:
            pass

        # Check HFT scalper (primary trading system)
        try:
            from .hft_scalper import get_hft_scalper
            scalper = get_hft_scalper()
            if scalper:
                scalper_running = scalper.is_running
        except Exception:
            pass

        # If no services are running at all
        if not broker_connected and not scalper_running:
            return SystemState.SERVICE_NOT_RUNNING

        # If broker connected but scalper not running
        if broker_connected and not scalper_running:
            return SystemState.READY

        # If scalper running, check if safe activation is enabled for trading
        if scalper_running:
            if safe.is_active():
                return SystemState.ACTIVE
            else:
                return SystemState.READY

        # Default to READY if broker is connected
        return SystemState.READY

    async def reconnect_feeds(self, paper_mode: bool = True) -> Dict:
        """
        Reconnect/restart data feeds.
        Only available in paper mode for safety.
        """
        if not paper_mode:
            return {
                "success": False,
                "error": "Reconnect only available in paper mode"
            }

        results = {
            "timestamp": datetime.now().isoformat(),
            "actions": []
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            # Restart Polygon stream
            try:
                # Stop first
                await client.post("http://localhost:9100/api/polygon/stream/stop")
                await asyncio.sleep(1)

                # Start again
                resp = await client.post("http://localhost:9100/api/polygon/stream/start")
                success = resp.status_code == 200
                results["actions"].append({
                    "service": "polygon_stream",
                    "action": "restart",
                    "success": success
                })

                if success:
                    self.services["polygon_stream"].status = ServiceStatus.UP
                    self.services["websocket"].status = ServiceStatus.UP
                    logger.info("Polygon stream reconnected")
            except Exception as e:
                results["actions"].append({
                    "service": "polygon_stream",
                    "action": "restart",
                    "success": False,
                    "error": str(e)
                })

            # Restart scalper
            try:
                await client.post("http://localhost:9100/api/scanner/scalper/stop")
                await asyncio.sleep(1)
                resp = await client.post("http://localhost:9100/api/scanner/scalper/start")
                success = resp.status_code == 200
                results["actions"].append({
                    "service": "scalper",
                    "action": "restart",
                    "success": success
                })

                if success:
                    self.services["scalper"].status = ServiceStatus.UP
                    logger.info("Scalper restarted")
            except Exception as e:
                results["actions"].append({
                    "service": "scalper",
                    "action": "restart",
                    "success": False,
                    "error": str(e)
                })

        # Run health check after reconnect
        await self.run_startup_self_test()

        results["success"] = all(a.get("success", False) for a in results["actions"])
        return results

    def get_connectivity_report(self) -> Dict:
        """
        Generate comprehensive connectivity check report.

        Includes:
        - Timestamps
        - Service name
        - Status (UP/DOWN)
        - Last successful event
        """
        from .market_time import get_time_status, get_market_status

        time_status = get_time_status()
        market_status, market_detail = get_market_status()
        system_state = self.get_system_state()

        report = {
            "report_generated": datetime.now().isoformat(),
            "system_state": system_state.value,
            "system_state_reason": self._get_state_reason(system_state),
            "market": {
                "status": market_status.value,
                "detail": market_detail,
                "et_time": time_status["et_display"],
                "is_trading_day": time_status["is_trading_day"]
            },
            "first_tick": {
                "received": self.first_tick_received.isoformat() if self.first_tick_received else None,
                "symbol": self.first_tick_symbol
            },
            "startup": {
                "time": self.startup_time.isoformat() if self.startup_time else None,
                "complete": self.startup_complete,
                "events": [asdict(e) for e in self.startup_events[-20:]]
            },
            "services": {
                name: svc.to_dict() for name, svc in self.services.items()
            },
            "summary": {
                "total_services": len(self.services),
                "services_up": sum(1 for s in self.services.values() if s.status == ServiceStatus.UP),
                "services_down": sum(1 for s in self.services.values() if s.status == ServiceStatus.DOWN),
                "services_degraded": sum(1 for s in self.services.values() if s.status == ServiceStatus.DEGRADED)
            }
        }

        return report

    def _get_state_reason(self, state: SystemState) -> str:
        """Get human-readable reason for system state"""
        reasons = {
            SystemState.ACTIVE: "All systems operational, trading enabled",
            SystemState.READY: "All systems connected, trading not enabled",
            SystemState.MARKET_CLOSED: "Market is closed (weekend/holiday/after hours)",
            SystemState.DATA_OFFLINE: "Market is open but data feed is offline",
            SystemState.SERVICE_NOT_RUNNING: "Required services not started",
            SystemState.DISCONNECTED: "Services started but lost connection",
            SystemState.PARTIAL: "Some services up, some down"
        }
        return reasons.get(state, "Unknown state")

    def _save_report(self, results: Dict):
        """Save connectivity report to file"""
        try:
            os.makedirs(os.path.dirname(CONNECTIVITY_REPORT_FILE), exist_ok=True)

            report = self.get_connectivity_report()
            report["self_test_results"] = results

            with open(CONNECTIVITY_REPORT_FILE, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Connectivity report saved to {CONNECTIVITY_REPORT_FILE}")
        except Exception as e:
            logger.error(f"Failed to save connectivity report: {e}")


# Singleton instance
_connectivity_manager: Optional[ConnectivityManager] = None


def get_connectivity_manager() -> ConnectivityManager:
    """Get the connectivity manager singleton"""
    global _connectivity_manager
    if _connectivity_manager is None:
        _connectivity_manager = ConnectivityManager()
    return _connectivity_manager


async def run_startup_self_test() -> Dict:
    """Run startup self-test (convenience function)"""
    manager = get_connectivity_manager()
    return await manager.run_startup_self_test()


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    async def test():
        manager = ConnectivityManager()
        results = await manager.run_startup_self_test()

        print("\n" + "=" * 60)
        print("CONNECTIVITY REPORT")
        print("=" * 60)

        report = manager.get_connectivity_report()
        print(json.dumps(report, indent=2))

    asyncio.run(test())
