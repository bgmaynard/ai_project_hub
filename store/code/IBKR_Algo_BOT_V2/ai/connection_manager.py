"""
Connection Manager - Resilient API Client with Circuit Breaker
==============================================================

Handles all external API calls (Schwab, Polygon, etc.) with:
1. Retry policy with exponential backoff + jitter
2. Circuit breaker to prevent cascade failures
3. Data freshness enforcement (staleness detection)
4. Last good snapshot persistence

Usage:
    from ai.connection_manager import get_connection_manager

    conn = get_connection_manager()
    data, fresh = await conn.fetch_quote("AAPL")

    if conn.is_data_stale:
        # Halt trading - data too old
        pass
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Blocking requests
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered


class DataFreshnessState(Enum):
    """Data freshness states"""

    FRESH = "FRESH"  # Data within threshold
    STALE = "STALE"  # Data exceeds staleness threshold
    UNKNOWN = "UNKNOWN"  # No data received yet


@dataclass
class RetryPolicy:
    """Configuration for retry behavior"""

    max_retries: int = 3
    base_delay_ms: int = 500
    max_delay_ms: int = 10000
    jitter_pct: float = 0.2
    retry_on_status: Tuple[int, ...] = (429, 500, 502, 503, 504)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff + jitter"""
        delay_ms = min(self.base_delay_ms * (2**attempt), self.max_delay_ms)
        jitter = delay_ms * self.jitter_pct * (2 * random.random() - 1)
        return (delay_ms + jitter) / 1000  # Return seconds


@dataclass
class CircuitBreaker:
    """
    Circuit breaker to prevent cascade failures.
    Opens after consecutive failures, closes after recovery.
    """

    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30
    half_open_max_calls: int = 3

    # State
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None
    half_open_calls: int = 0

    def record_success(self):
        """Record a successful call"""
        self.success_count += 1
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self._close()
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def record_failure(self):
        """Record a failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)

        if self.state == CircuitState.HALF_OPEN:
            self._open()
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._open()

    def can_proceed(self) -> bool:
        """Check if calls should be allowed"""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_state_change:
                elapsed = (
                    datetime.now(timezone.utc) - self.last_state_change
                ).total_seconds()
                if elapsed >= self.recovery_timeout_seconds:
                    self._half_open()
                    return True
            return False

        # HALF_OPEN - allow limited calls
        return True

    def _open(self):
        """Open the circuit"""
        self.state = CircuitState.OPEN
        self.last_state_change = datetime.now(timezone.utc)
        logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")

    def _close(self):
        """Close the circuit (normal operation)"""
        self.state = CircuitState.CLOSED
        self.last_state_change = datetime.now(timezone.utc)
        self.failure_count = 0
        self.half_open_calls = 0
        logger.info("Circuit breaker CLOSED - service recovered")

    def _half_open(self):
        """Enter half-open state for testing"""
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        logger.info("Circuit breaker HALF_OPEN - testing recovery")

    def to_dict(self) -> Dict:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "last_state_change": (
                self.last_state_change.isoformat() if self.last_state_change else None
            ),
        }


@dataclass
class DataFreshness:
    """
    Tracks data freshness for trading decisions.
    Trading MUST halt if data is stale.
    """

    staleness_threshold_seconds: float = 10.0  # Max age before stale
    last_good_data_time: Optional[datetime] = None
    last_update_attempt_time: Optional[datetime] = None
    state: DataFreshnessState = DataFreshnessState.UNKNOWN
    stale_reason: Optional[str] = None

    def record_good_data(self):
        """Record successful data fetch"""
        self.last_good_data_time = datetime.now(timezone.utc)
        self.state = DataFreshnessState.FRESH
        self.stale_reason = None

    def record_failed_fetch(self, reason: str):
        """Record failed data fetch"""
        self.last_update_attempt_time = datetime.now(timezone.utc)
        self._check_staleness(reason)

    def check_freshness(self) -> Tuple[bool, Optional[float]]:
        """
        Check if data is fresh enough for trading.
        Returns (is_fresh, stale_seconds)
        """
        if self.last_good_data_time is None:
            self.state = DataFreshnessState.UNKNOWN
            return False, None

        elapsed = (
            datetime.now(timezone.utc) - self.last_good_data_time
        ).total_seconds()
        if elapsed > self.staleness_threshold_seconds:
            self.state = DataFreshnessState.STALE
            self.stale_reason = f"Last good data {elapsed:.1f}s ago"
            return False, elapsed

        self.state = DataFreshnessState.FRESH
        return True, elapsed

    def _check_staleness(self, reason: str):
        """Update staleness state after failed fetch"""
        if self.last_good_data_time:
            elapsed = (
                datetime.now(timezone.utc) - self.last_good_data_time
            ).total_seconds()
            if elapsed > self.staleness_threshold_seconds:
                self.state = DataFreshnessState.STALE
                self.stale_reason = reason

    def to_dict(self) -> Dict:
        is_fresh, stale_seconds = self.check_freshness()
        return {
            "state": self.state.value,
            "is_fresh": is_fresh,
            "stale_seconds": stale_seconds,
            "staleness_threshold_seconds": self.staleness_threshold_seconds,
            "last_good_data_time": (
                self.last_good_data_time.isoformat()
                if self.last_good_data_time
                else None
            ),
            "stale_reason": self.stale_reason,
        }


class ConnectionManager:
    """
    Central manager for all external API connections.
    Provides resilience, retry logic, and staleness detection.
    """

    def __init__(self, base_url: str = "http://localhost:9100"):
        self.base_url = base_url
        self.retry_policy = RetryPolicy()
        self.circuit_breaker = CircuitBreaker()
        self.data_freshness = DataFreshness()

        # Last known good snapshots
        self._last_good_snapshots: Dict[str, Dict] = {}
        self._snapshot_cache_path = Path("ai/last_good_snapshots.json")

        # Load cached snapshots
        self._load_cached_snapshots()

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _reset_http_client(self):
        """
        Close and recreate the HTTP client.

        Called on repeated timeouts or connection errors to recover
        from stale connection pool or network state issues.
        """
        logger.warning("Resetting HTTP client due to connection issues")
        if self._client and not self._client.is_closed:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.warning(f"Error closing HTTP client: {e}")
        self._client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _load_cached_snapshots(self):
        """Load last good snapshots from disk"""
        try:
            if self._snapshot_cache_path.exists():
                with open(self._snapshot_cache_path, "r") as f:
                    self._last_good_snapshots = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cached snapshots: {e}")

    def _save_cached_snapshots(self):
        """Persist last good snapshots to disk"""
        try:
            self._snapshot_cache_path.parent.mkdir(exist_ok=True)
            with open(self._snapshot_cache_path, "w") as f:
                json.dump(self._last_good_snapshots, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save cached snapshots: {e}")

    @property
    def is_data_stale(self) -> bool:
        """Check if market data is stale - TRADING MUST HALT if True"""
        is_fresh, _ = self.data_freshness.check_freshness()
        return not is_fresh

    @property
    def circuit_open(self) -> bool:
        """Check if circuit breaker is open"""
        return self.circuit_breaker.state == CircuitState.OPEN

    async def fetch_with_retry(
        self, endpoint: str, method: str = "GET", **kwargs
    ) -> Tuple[Optional[Dict], bool]:
        """
        Fetch data with retry policy and circuit breaker.

        Returns:
            Tuple of (data, is_fresh)
            - data: Response data or None on failure
            - is_fresh: True if data is fresh, False if stale/cached
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_proceed():
            logger.warning(f"Circuit breaker OPEN - blocking request to {endpoint}")
            return None, False

        url = (
            f"{self.base_url}{endpoint}"
            if not endpoint.startswith("http")
            else endpoint
        )
        client = await self._get_client()

        last_error = None
        for attempt in range(self.retry_policy.max_retries + 1):
            try:
                if method.upper() == "GET":
                    response = await client.get(url, **kwargs)
                elif method.upper() == "POST":
                    response = await client.post(url, **kwargs)
                else:
                    response = await client.request(method, url, **kwargs)

                # Handle different status codes
                if response.status_code == 200:
                    self.circuit_breaker.record_success()
                    self.data_freshness.record_good_data()
                    data = response.json()
                    return data, True

                elif response.status_code == 401 or response.status_code == 403:
                    # Auth error - would need token refresh
                    logger.warning(f"Auth error {response.status_code} on {endpoint}")
                    # Retry once after potential token refresh
                    if attempt == 0:
                        await asyncio.sleep(1)
                        continue
                    break

                elif response.status_code == 429:
                    # Rate limited - respect Retry-After if present
                    retry_after = response.headers.get("Retry-After", "5")
                    try:
                        wait_seconds = int(retry_after)
                    except ValueError:
                        wait_seconds = 5
                    logger.warning(
                        f"Rate limited on {endpoint}, waiting {wait_seconds}s"
                    )
                    await asyncio.sleep(wait_seconds)
                    continue

                elif response.status_code in self.retry_policy.retry_on_status:
                    # Retryable error
                    last_error = f"HTTP {response.status_code}"
                    delay = self.retry_policy.get_delay(attempt)
                    logger.warning(
                        f"Retryable error {response.status_code} on {endpoint}, attempt {attempt+1}, waiting {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                    continue

                else:
                    # Non-retryable error
                    logger.error(
                        f"Non-retryable error {response.status_code} on {endpoint}"
                    )
                    self.circuit_breaker.record_failure()
                    break

            except httpx.TimeoutException:
                last_error = "Timeout"
                delay = self.retry_policy.get_delay(attempt)
                logger.warning(
                    f"Timeout on {endpoint}, attempt {attempt+1}, waiting {delay:.2f}s"
                )
                # Reset HTTP client on final retry
                if attempt >= self.retry_policy.max_retries:
                    await self._reset_http_client()
                await asyncio.sleep(delay)

            except httpx.ConnectError as e:
                last_error = f"Connection error: {e}"
                delay = self.retry_policy.get_delay(attempt)
                logger.warning(
                    f"Connection error on {endpoint}: {e}, attempt {attempt+1}"
                )
                # Reset HTTP client on final retry
                if attempt >= self.retry_policy.max_retries:
                    await self._reset_http_client()
                await asyncio.sleep(delay)

            except Exception as e:
                last_error = str(e)
                logger.error(f"Unexpected error on {endpoint}: {e}")
                self.circuit_breaker.record_failure()
                break

        # All retries exhausted
        self.circuit_breaker.record_failure()
        self.data_freshness.record_failed_fetch(f"Failed after retries: {last_error}")
        return None, False

    async def fetch_quote(self, symbol: str) -> Tuple[Optional[Dict], bool]:
        """
        Fetch quote for a symbol with caching.

        Returns:
            Tuple of (quote_data, is_fresh)
        """
        data, is_fresh = await self.fetch_with_retry(f"/api/price/{symbol}")

        if data and is_fresh:
            # Cache the good data
            self._last_good_snapshots[symbol] = {
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._save_cached_snapshots()
            return data, True

        # Return cached data if available
        if symbol in self._last_good_snapshots:
            cached = self._last_good_snapshots[symbol]
            logger.warning(
                f"Using cached quote for {symbol} from {cached.get('timestamp')}"
            )
            return cached.get("data"), False

        return None, False

    async def fetch_float(self, symbol: str) -> Tuple[Optional[Dict], bool]:
        """Fetch float data for a symbol"""
        return await self.fetch_with_retry(f"/api/stock/float/{symbol}")

    async def fetch_worklist(self) -> Tuple[Optional[Dict], bool]:
        """Fetch worklist with all symbols"""
        return await self.fetch_with_retry("/api/worklist")

    async def fetch_market_movers(
        self, direction: str = "up"
    ) -> Tuple[Optional[Dict], bool]:
        """Fetch market movers"""
        return await self.fetch_with_retry(f"/api/market/movers?direction={direction}")

    def get_status(self) -> Dict:
        """Get connection manager status"""
        is_fresh, stale_seconds = self.data_freshness.check_freshness()
        return {
            "circuit_breaker": self.circuit_breaker.to_dict(),
            "data_freshness": self.data_freshness.to_dict(),
            "is_data_stale": self.is_data_stale,
            "circuit_open": self.circuit_open,
            "cached_symbols": list(self._last_good_snapshots.keys()),
            "trading_allowed": is_fresh and not self.circuit_open,
        }

    def get_stale_veto_reason(self) -> Optional[str]:
        """Get veto reason if data is stale"""
        if self.is_data_stale:
            return f"DATA_STALE: {self.data_freshness.stale_reason}"
        if self.circuit_open:
            return f"CIRCUIT_OPEN: API unavailable after {self.circuit_breaker.failure_count} failures"
        return None


# Global singleton
_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get or create the global connection manager"""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


def reset_connection_manager():
    """Reset the connection manager (for testing)"""
    global _connection_manager
    _connection_manager = None
