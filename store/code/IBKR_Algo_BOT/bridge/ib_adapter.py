import os
import asyncio
import random
import time
from dataclasses import dataclass
from typing import Callable, Awaitable, Optional
from ib_insync import IB


def _get_env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if (v is not None and v != "") else default


@dataclass
class IBConfig:
    # Defaults come from environment so callers can just use IBConfig()
    host: str = _get_env("TWS_HOST", "127.0.0.1")
    port: int = int(_get_env("TWS_PORT", "7497"))                # 7497 paper, 7496 live
    client_id: int = int(_get_env("TWS_CLIENT_ID", "1101"))
    heartbeat_sec: float = float(_get_env("IB_HEARTBEAT_SEC", "2.5"))
    backoff_base: float = 1.0
    backoff_factor: float = 2.0
    backoff_max: float = 30.0
    jitter_ratio: float = 0.15
    max_fail_streak: int = 10
    connect_timeout_sec: float = float(_get_env("IB_CONNECT_TIMEOUT_SEC", "15.0"))


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
            base=cfg.backoff_base,
            factor=cfg.backoff_factor,
            max_sleep=cfg.backoff_max,
            jitter_ratio=cfg.jitter_ratio,
        )
        self._fail_streak = 0
        self._on_resubscribe: Optional[Callable[[], Awaitable[None]]] = None

    # -------- Public surface --------------------------------------------------
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
        if not self._watchdog_task or self._watchdog_task.done():
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

    # -------- Internal --------------------------------------------------------
    def _set_state(self, s: str):
        if s != self.state:
            self.state = s
            self.last_change = time.time()

    async def _connect_once(self):
        """
        Try to connect, automatically rotating clientId if the base one is in use.
        Attempts up to 50 consecutive IDs: base .. base+49.
        """
        base_id = int(self.cfg.client_id)
        last_exc: Optional[Exception] = None
        saw_326 = False

        for offset in range(50):
            cid = base_id + offset
            try:
                print(f"IBAdapter: trying clientId {cid}")
                await self.ib.connectAsync(
                    self.cfg.host, self.cfg.port, clientId=cid, timeout=self.cfg.connect_timeout_sec
                )
                # success
                self.cfg.client_id = cid  # persist the working id
                self._set_state("CONNECTED")
                self._backoff.reset()
                self._fail_streak = 0
                # Light health check
                try:
                    await self.ib.reqCurrentTimeAsync(timeout=5)
                except Exception:
                    self._set_state("DEGRADED")
                # Resubscribe any streams if provided
                if self._on_resubscribe:
                    try:
                        await self._on_resubscribe()
                    except Exception:
                        pass
                return
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if "client id is already in use" in msg or "error 326" in msg:
                    saw_326 = True
                    continue  # try next client id
                if saw_326 and ("timeout" in msg or "timed out" in msg):
                    continue
                break  # non-326 persistent error → bail to backoff

        self._fail_streak += 1
        self._set_state("RECONNECTING")
        if last_exc:
            raise last_exc
        raise RuntimeError("Failed to connect to TWS/Gateway with rotating clientId")

    async def _reconnect_with_backoff(self):
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
                # Light health probe
                try:
                    await self.ib.reqCurrentTimeAsync(timeout=3)
                    if self.state != "CONNECTED":
                        self._set_state("CONNECTED")
                except Exception:
                    self._set_state("DEGRADED")
            except asyncio.CancelledError:
                break
            except Exception:
                self._set_state("RECONNECTING")
