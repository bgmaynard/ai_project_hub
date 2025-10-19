# src/bridge/ib_adapter.py
import asyncio, random, time
from dataclasses import dataclass
from typing import Callable, Awaitable, Optional
from ib_insync import IB

# ---- Config dataclass --------------------------------------------------------
@dataclass
class IBConfig:
    host: str = "127.0.0.1"
    port: int = 7497             # 7497 paper, 7496 live
    client_id: int = 1101
    heartbeat_sec: float = 2.5   # poll interval
    backoff_base: float = 1.0    # seconds
    backoff_factor: float = 2.0
    backoff_max: float = 30.0
    jitter_ratio: float = 0.15   # ±15% jitter
    max_fail_streak: int = 10    # mark FAILED after this many attempts

# ---- Simple backoff helper ---------------------------------------------------
class Backoff:
    def __init__(self, base=1.0, factor=2.0, max_sleep=30.0, jitter_ratio=0.15):
        self.base = base
        self.factor = factor
        self.max_sleep = max_sleep
        self.jitter_ratio = jitter_ratio
        self.n = 0

    def next(self) -> float:
        raw = min(self.base * (self.factor ** self.n), self.max_sleep)
        self.n += 1
        jitter = raw * self.jitter_ratio
        return max(0.25, raw + random.uniform(-jitter, jitter))

    def reset(self):
        self.n = 0

# ---- Adapter ----------------------------------------------------------------
class IBAdapter:
    """
    Single-owner async adapter. Start the watchdog once at app startup.
    """
    def __init__(self, cfg: IBConfig):
        self.cfg = cfg
        self.ib = IB()
        self.state = "DISCONNECTED"
        self.last_change = time.time()
        self._watchdog_task: Optional[asyncio.Task] = None
        self._backoff = Backoff(
            base=cfg.backoff_base, factor=cfg.backoff_factor,
            max_sleep=cfg.backoff_max, jitter_ratio=cfg.jitter_ratio
        )
        self._fail_streak = 0
        self._on_resubscribe: Optional[Callable[[], Awaitable[None]]] = None

    # Public surface -----------------------------------------------------------
    def set_resubscribe_hook(self, fn: Callable[[], Awaitable[None]]):
        self._on_resubscribe = fn

    def is_connected(self) -> bool:
        try:
            return bool(self.ib.isConnected())
        except Exception:
            return False

    def get_status(self) -> dict:
        return {
            "state": self.state,
            "connected": self.is_connected(),
            "host": self.cfg.host,
            "port": self.cfg.port,
            "clientId": self.cfg.client_id,
            "failStreak": self._fail_streak,
            "backoffLevel": self._backoff.n,
            "lastChangeTs": self.last_change,
        }

    async def start(self):
        # initial connect
        await self._connect_once()
        # start watchdog
        if not self._watchdog_task:
            self._watchdog_task = asyncio.create_task(self._watchdog_loop(), name="ib_watchdog")

    async def stop(self):
        if self._watchdog_task:
            self._watchdog_task.cancel()
            self._watchdog_task = None
        if self.is_connected():
            try:
                self.ib.disconnect()
            except Exception:
                pass
        self._set_state("DISCONNECTED")

    async def ensure_connected(self):
        """Use this before any IB call."""
        if self.is_connected():
            return
        self._set_state("RECONNECTING")
        await self._reconnect_with_backoff()

    # Internal -----------------------------------------------------------------
    def _set_state(self, s: str):
        if s != self.state:
            self.state = s
            self.last_change = time.time()

    async def _connect_once(self):
        try:
            await self.ib.connectAsync(self.cfg.host, self.cfg.port, clientId=self.cfg.client_id)
            self._set_state("CONNECTED")
            self._backoff.reset()
            self._fail_streak = 0
            # Optional: light ping to verify roundtrip
            try:
                await self.ib.reqCurrentTimeAsync(timeout=5)
            except Exception:
                self._set_state("DEGRADED")
            # resubscribe handlers
            if self._on_resubscribe:
                try:
                    await self._on_resubscribe()
                except Exception:
                    # failing resubscribe shouldn't kill connection
                    pass
        except Exception:
            self._fail_streak += 1
            self._set_state("RECONNECTING")
            raise

    async def _reconnect_with_backoff(self):
        # Attempt until success or max fail streak indicates FAILED
        while True:
            try:
                await self._connect_once()
                return
            except Exception:
                delay = self._backoff.next()
                if self._fail_streak >= self.cfg.max_fail_streak:
                    self._set_state("FAILED")
                await asyncio.sleep(delay)

    async def _watchdog_loop(self):
        """Periodic heartbeat to keep/restore connectivity."""
        while True:
            await asyncio.sleep(self.cfg.heartbeat_sec)
            try:
                if not self.is_connected():
                    self._set_state("RECONNECTING")
                    await self._reconnect_with_backoff()
                    continue
                # Light health check
                try:
                    await self.ib.reqCurrentTimeAsync(timeout=3)
                    if self.state != "CONNECTED":
                        self._set_state("CONNECTED")
                except Exception:
                    # Could be transient; mark degraded and force reconnect next tick
                    self._set_state("DEGRADED")
            except asyncio.CancelledError:
                break
            except Exception:
                # Any unexpected error — try reconnect on next tick
                self._set_state("RECONNECTING")
