"""
Connection Health Monitor
=========================
Centralized monitoring of all data connections with auto-recovery.

Monitors:
- Schwab broker connection
- Schwab market data
- Polygon WebSocket (if enabled)
- API server health

Features:
- Unified health status
- Auto-reconnect on failure
- Health history for trend analysis
- Alert callbacks for failures
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"
    DEGRADED = "DEGRADED"  # Partial functionality
    UNKNOWN = "UNKNOWN"


@dataclass
class ConnectionStatus:
    name: str
    state: ConnectionState
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    latency_ms: Optional[float] = None
    error_message: Optional[str] = None
    extra: Dict = field(default_factory=dict)


class ConnectionHealthMonitor:
    """
    Centralized connection health monitoring with auto-recovery.
    """

    def __init__(self):
        self._lock = threading.RLock()  # Reentrant lock to allow nested calls
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Connection statuses
        self._connections: Dict[str, ConnectionStatus] = {
            "schwab_broker": ConnectionStatus("Schwab Broker", ConnectionState.UNKNOWN),
            "schwab_market_data": ConnectionStatus(
                "Schwab Market Data", ConnectionState.UNKNOWN
            ),
            "api_server": ConnectionStatus(
                "API Server", ConnectionState.CONNECTED
            ),  # We're running
        }

        # Optional connections (not monitored by default)
        self._optional_connections: Dict[str, ConnectionStatus] = {
            "polygon_stream": ConnectionStatus(
                "Polygon Stream", ConnectionState.DISCONNECTED
            ),
        }

        # Polygon disabled - using Schwab only (saves $199/mo)
        self._polygon_enabled = False

        # Health history (last 60 samples, one per minute)
        self._health_history: List[Dict] = []
        self._max_history = 60

        # Alert callbacks
        self._alert_callbacks: List[Callable] = []

        # Recovery functions
        self._recovery_functions: Dict[str, Callable] = {}

        # Configuration
        self._check_interval = 10  # seconds
        self._max_consecutive_failures = 3  # before triggering recovery

    def register_recovery(self, connection_name: str, recovery_fn: Callable):
        """Register a recovery function for a connection"""
        self._recovery_functions[connection_name] = recovery_fn

    def on_alert(self, callback: Callable):
        """Register callback for connection alerts"""
        self._alert_callbacks.append(callback)

    def start(self):
        """Start the health monitoring loop"""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            logger.info("Connection health monitor started")

    def stop(self):
        """Stop the health monitoring loop"""
        with self._lock:
            self._running = False
            if self._thread:
                self._thread.join(timeout=5)
            logger.info("Connection health monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                self._check_all_connections()
                self._record_health_snapshot()
                time.sleep(self._check_interval)
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                time.sleep(5)

    def _check_all_connections(self):
        """Check all registered connections"""
        self._check_schwab_broker()
        self._check_schwab_market_data()
        # Polygon disabled - Schwab only mode (saves $199/mo)
        # Uncomment to re-enable: self._check_polygon_stream()

    def _check_schwab_broker(self):
        """Check Schwab broker connection"""
        conn = self._connections["schwab_broker"]
        try:
            from unified_broker import get_unified_broker

            broker = get_unified_broker()
            if broker and broker.is_connected:
                self._mark_connected(conn, extra={"broker": broker.broker_name})
            else:
                self._mark_disconnected(conn, "Broker not connected")
        except Exception as e:
            self._mark_disconnected(conn, str(e))

    def _check_schwab_market_data(self):
        """Check Schwab market data connection"""
        conn = self._connections["schwab_market_data"]
        try:
            from schwab_market_data import (get_token_status,
                                            is_schwab_available)

            if is_schwab_available():
                token_status = get_token_status()
                if token_status.get("valid"):
                    self._mark_connected(
                        conn, extra={"token_status": token_status.get("status")}
                    )
                else:
                    self._mark_degraded(conn, f"Token: {token_status.get('message')}")
            else:
                self._mark_disconnected(conn, "Schwab not available")
        except Exception as e:
            self._mark_disconnected(conn, str(e))

    def _check_polygon_stream(self):
        """Check Polygon WebSocket stream"""
        conn = self._connections["polygon_stream"]
        try:
            from polygon_streaming import (get_polygon_stream,
                                           is_polygon_streaming_available)

            if not is_polygon_streaming_available():
                self._mark_disconnected(conn, "Polygon API key not configured")
                return

            stream = get_polygon_stream()
            if stream and stream.is_healthy():
                stats = stream.get_status()
                self._mark_connected(
                    conn,
                    extra={
                        "trades_received": stats.get("trades_received", 0),
                        "quotes_received": stats.get("quotes_received", 0),
                        "reconnects": stats.get("reconnects", 0),
                    },
                )
            elif stream and stream.running:
                # Running but not healthy - degraded
                self._mark_degraded(conn, "Stream running but not connected")
            else:
                self._mark_disconnected(conn, "Stream not running")
        except Exception as e:
            self._mark_disconnected(conn, str(e))

    def _mark_connected(self, conn: ConnectionStatus, extra: Dict = None):
        """Mark connection as connected"""
        with self._lock:
            was_disconnected = conn.state in [
                ConnectionState.DISCONNECTED,
                ConnectionState.RECONNECTING,
            ]
            conn.state = ConnectionState.CONNECTED
            conn.last_success = datetime.now()
            conn.consecutive_failures = 0
            conn.error_message = None
            if extra:
                conn.extra.update(extra)

            if was_disconnected:
                logger.info(f"[HealthMonitor] {conn.name} RECOVERED")
                self._notify_alert(conn.name, "info", f"{conn.name} connected")

    def _mark_disconnected(self, conn: ConnectionStatus, error: str):
        """Mark connection as disconnected"""
        with self._lock:
            was_connected = conn.state == ConnectionState.CONNECTED
            conn.state = ConnectionState.DISCONNECTED
            conn.last_failure = datetime.now()
            conn.consecutive_failures += 1
            conn.error_message = error

            if was_connected:
                logger.warning(f"[HealthMonitor] {conn.name} DISCONNECTED: {error}")
                self._notify_alert(
                    conn.name, "error", f"{conn.name} disconnected: {error}"
                )

            # Trigger recovery if too many failures
            if conn.consecutive_failures >= self._max_consecutive_failures:
                self._trigger_recovery(conn.name)

    def _mark_degraded(self, conn: ConnectionStatus, error: str):
        """Mark connection as degraded (partial functionality)"""
        with self._lock:
            was_connected = conn.state == ConnectionState.CONNECTED
            conn.state = ConnectionState.DEGRADED
            conn.error_message = error

            if was_connected:
                logger.warning(f"[HealthMonitor] {conn.name} DEGRADED: {error}")
                self._notify_alert(
                    conn.name, "warning", f"{conn.name} degraded: {error}"
                )

    def _trigger_recovery(self, connection_name: str):
        """Trigger recovery for a connection"""
        if connection_name not in self._recovery_functions:
            logger.warning(
                f"[HealthMonitor] No recovery function for {connection_name}"
            )
            return

        conn = self._connections[connection_name]
        conn.state = ConnectionState.RECONNECTING
        logger.info(f"[HealthMonitor] Triggering recovery for {connection_name}")

        try:
            recovery_fn = self._recovery_functions[connection_name]
            threading.Thread(target=recovery_fn, daemon=True).start()
        except Exception as e:
            logger.error(f"[HealthMonitor] Recovery failed for {connection_name}: {e}")

    def _notify_alert(self, source: str, level: str, message: str):
        """Notify all alert callbacks"""
        for callback in self._alert_callbacks:
            try:
                callback(source, level, message)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def _record_health_snapshot(self):
        """Record current health state for history"""
        with self._lock:
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "overall_healthy": self.is_healthy(),
                "connections": {
                    name: {
                        "state": conn.state.value,
                        "failures": conn.consecutive_failures,
                    }
                    for name, conn in self._connections.items()
                },
            }
            self._health_history.append(snapshot)
            if len(self._health_history) > self._max_history:
                self._health_history.pop(0)

    def is_healthy(self) -> bool:
        """Check if all critical connections are healthy"""
        with self._lock:
            # Schwab broker and market data are critical
            critical = ["schwab_broker", "schwab_market_data"]
            for name in critical:
                conn = self._connections.get(name)
                if conn and conn.state not in [
                    ConnectionState.CONNECTED,
                    ConnectionState.DEGRADED,
                ]:
                    return False
            return True

    def get_status(self) -> Dict:
        """Get full health status"""
        with self._lock:
            # Build connections dict
            connections = {
                name: {
                    "state": conn.state.value,
                    "last_success": (
                        conn.last_success.isoformat() if conn.last_success else None
                    ),
                    "last_failure": (
                        conn.last_failure.isoformat() if conn.last_failure else None
                    ),
                    "consecutive_failures": conn.consecutive_failures,
                    "latency_ms": conn.latency_ms,
                    "error": conn.error_message,
                    **conn.extra,
                }
                for name, conn in self._connections.items()
            }

            # Add optional connections with disabled status
            for name, conn in self._optional_connections.items():
                connections[name] = {
                    "state": "DISABLED",
                    "enabled": False,
                    "reason": "Schwab-only mode (Polygon disabled to save $199/mo)",
                    "last_success": None,
                    "last_failure": None,
                    "consecutive_failures": 0,
                    "latency_ms": None,
                    "error": None,
                }

            return {
                "healthy": self.is_healthy(),
                "timestamp": datetime.now().isoformat(),
                "mode": "schwab_only",
                "connections": connections,
            }

    def get_connection_state(self, name: str) -> Optional[ConnectionState]:
        """Get state of a specific connection"""
        with self._lock:
            conn = self._connections.get(name)
            return conn.state if conn else None

    def get_health_history(self, minutes: int = 30) -> List[Dict]:
        """Get health history for the last N minutes"""
        with self._lock:
            cutoff = datetime.now() - timedelta(minutes=minutes)
            return [
                h
                for h in self._health_history
                if datetime.fromisoformat(h["timestamp"]) > cutoff
            ]


# Singleton instance
_health_monitor: Optional[ConnectionHealthMonitor] = None


def get_health_monitor() -> ConnectionHealthMonitor:
    """Get or create the health monitor instance"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = ConnectionHealthMonitor()

        # Register recovery functions
        def recover_schwab_market_data():
            """Attempt to recover Schwab market data"""
            try:
                from schwab_market_data import _refresh_token

                _refresh_token()
            except Exception as e:
                logger.error(f"Schwab market data recovery failed: {e}")

        def recover_polygon_stream():
            """Attempt to recover Polygon stream"""
            try:
                from polygon_streaming import get_polygon_stream

                stream = get_polygon_stream()
                if stream:
                    stream.restart()
            except Exception as e:
                logger.error(f"Polygon stream recovery failed: {e}")

        _health_monitor.register_recovery(
            "schwab_market_data", recover_schwab_market_data
        )
        _health_monitor.register_recovery("polygon_stream", recover_polygon_stream)

    return _health_monitor


def start_health_monitor():
    """Start the health monitor"""
    monitor = get_health_monitor()
    monitor.start()
    return monitor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    monitor = start_health_monitor()

    try:
        while True:
            print("\n--- Health Status ---")
            status = monitor.get_status()
            print(f"Overall Healthy: {status['healthy']}")
            for name, conn in status["connections"].items():
                print(
                    f"  {name}: {conn['state']} (failures: {conn['consecutive_failures']})"
                )
            time.sleep(10)
    except KeyboardInterrupt:
        monitor.stop()
