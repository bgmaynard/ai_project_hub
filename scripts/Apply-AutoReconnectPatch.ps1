Param(
  [string]$Root = (Get-Location).Path,
  [switch]$OverwriteStatusHtml
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$p) { if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p | Out-Null } }

Write-Host "== Applying Auto-Reconnect patch to $Root =="

# Paths
$apiFile = Join-Path $Root "store\code\IBKR_Algo_BOT\dashboard_api.py"
$adapterDir = Join-Path $Root "store\code\IBKR_Algo_BOT\bridge"
$adapterFile = Join-Path $adapterDir "ib_adapter.py"
$initA = Join-Path $Root "store\__init__.py"
$initB = Join-Path $Root "store\code\__init__.py"
$initC = Join-Path $Root "store\code\IBKR_Algo_BOT\__init__.py"
$initD = Join-Path $Root "store\code\IBKR_Algo_BOT\bridge\__init__.py"
$bootstrap = Join-Path $Root "scripts\bootstrap.ps1"
$statusHtml = Join-Path $Root "store\code\IBKR_Algo_BOT\ui\status.html"

# Sanity checks
if (-not (Test-Path $apiFile)) {
  throw "Could not find $apiFile. Run this from your repo root."
}

# Create dirs & __init__.py
Ensure-Dir $adapterDir
foreach ($f in @($initA,$initB,$initC,$initD)) { if (-not (Test-Path $f)) { New-Item -ItemType File -Path $f | Out-Null } }

# Write ib_adapter.py
$adapterCode = @"
import asyncio
import random
import time
from dataclasses import dataclass
from typing import Callable, Awaitable, Optional
from ib_insync import IB

@dataclass
class IBConfig:
    host: str = "127.0.0.1"
    port: int = 7497              # 7497 paper, 7496 live
    client_id: int = 1101
    heartbeat_sec: float = 2.5    # status poll interval
    backoff_base: float = 1.0     # seconds
    backoff_factor: float = 2.0
    backoff_max: float = 30.0
    jitter_ratio: float = 0.15    # ±15% jitter on delays
    max_fail_streak: int = 10     # mark FAILED after this many attempts

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
        try:
            await self.ib.connectAsync(self.cfg.host, self.cfg.port, clientId=self.cfg.client_id)
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
                    # resubscribe failure shouldn't break connection
                    pass
        except Exception:
            self._fail_streak += 1
            self._set_state("RECONNECTING")
            raise

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
                    # Transient issues show as DEGRADED; next tick will attempt reconnect if it worsens
                    self._set_state("DEGRADED")
            except asyncio.CancelledError:
                break
            except Exception:
                self._set_state("RECONNECTING")
"@

Set-Content -Path $adapterFile -Value $adapterCode -Encoding UTF8
Write-Host "Wrote $adapterFile"

# Backup and edit dashboard_api.py import
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backup = "$apiFile.$stamp.bak"
Copy-Item $apiFile $backup -Force
Write-Host "Backed up $apiFile -> $backup"

$content = Get-Content $apiFile -Raw
# Replace any previous adapter import with canonical import
$content = $content -replace 'from\s+src\.bridge\.ib_adapter\s+import\s+IBAdapter,\s*IBConfig', 'from store.code.IBKR_Algo_BOT.bridge.ib_adapter import IBAdapter, IBConfig'
$content = $content -replace 'from\s+store\.code\.IBKR_Algo_BOT\.ib_adapter\s+import\s+IBAdapter,\s*IBConfig', 'from store.code.IBKR_Algo_BOT.bridge.ib_adapter import IBAdapter, IBConfig'
$content = $content -replace 'from\s+\.ib_adapter\s+import\s+IBAdapter,\s*IBConfig', 'from store.code.IBKR_Algo_BOT.bridge.ib_adapter import IBAdapter, IBConfig'
if ($content -notmatch 'from\s+store\.code\.IBKR_Algo_BOT\.bridge\.ib_adapter\s+import\s+IBAdapter,\s*IBConfig') {
  # Insert after first block of imports
  $content = $content -replace '(^from .*\n|^import .*\n)+', "$&from store.code.IBKR_Algo_BOT.bridge.ib_adapter import IBAdapter, IBConfig`n"
}
Set-Content -Path $apiFile -Value $content -Encoding UTF8
Write-Host "Updated import in $apiFile"

# Write fixed bootstrap.ps1
$bootstrapCode = @"
Param(
    [string]$PythonExe = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Move to repo root (assumes this script lives in scripts\)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
Set-Location $root

Write-Host "== AI_Project_Hub bootstrap (PowerShell) =="

function Resolve-Python {
    param([string]$Prefer="")
    if ($Prefer -and (Test-Path $Prefer)) { return $Prefer }
    foreach ($c in @("py","python","python3")) {
        try { & $c -V *> $null; if ($LASTEXITCODE -eq 0) { return $c } } catch {}
    }
    throw "Python not found. Install Python 3.11+ and re-run."
}
$py = Resolve-Python -Prefer $PythonExe
Write-Host "Using Python executable: $py"

if (!(Test-Path ".venv")) {
    Write-Host "Creating virtual environment at .venv ..."
    & $py -m venv .venv
}

$venvActivate = Join-Path $root ".venv\Scripts\Activate.ps1"
. $venvActivate

python -m ensurepip --upgrade | Out-Null
python -m pip install --upgrade pip wheel setuptools | Out-Null

if (Test-Path "requirements.txt") {
    Write-Host "Installing from requirements.txt ..."
    python -m pip install -r requirements.txt
} else {
    Write-Host "Installing core dependencies ..."
    python -m pip install fastapi==0.110.* "uvicorn[standard]==0.27.*" ib-insync==0.9.* python-dotenv==1.*
}

function Set-EnvVar {
    param([string]$Name, [string]$Value)
    if ([string]::IsNullOrWhiteSpace($Name)) { return }
    Set-Item -Path ("Env:{0}" -f $Name) -Value $Value
}

$envPath = ".env"
if (Test-Path $envPath) {
    Get-Content $envPath | ForEach-Object {
        $line = $_.Trim()
        if ($line -match '^\s*#' -or $line -eq "") { return }
        $kv = $line -split '=',2
        if ($kv.Count -eq 2) {
            $k = $kv[0].Trim()
            $v = $kv[1].Trim().Trim('"').Trim("'")
            Set-EnvVar -Name $k -Value $v
        }
    }
} else {
    if (-not $env:LOCAL_API_KEY) { Set-EnvVar -Name "LOCAL_API_KEY" -Value "dev_local_key_change_me" }
    if (-not $env:API_HOST) { Set-EnvVar -Name "API_HOST" -Value "127.0.0.1" }
    if (-not $env:API_PORT) { Set-EnvVar -Name "API_PORT" -Value "9101" }
    if (-not $env:TWS_HOST) { Set-EnvVar -Name "TWS_HOST" -Value "127.0.0.1" }
    if (-not $env:TWS_PORT) { Set-EnvVar -Name "TWS_PORT" -Value "7497" }
    if (-not $env:TWS_CLIENT_ID) { Set-EnvVar -Name "TWS_CLIENT_ID" -Value "1101" }
}

$module = "store.code.IBKR_Algo_BOT.dashboard_api:app"
$apiHost = $env:API_HOST; if (-not $apiHost) { $apiHost = "127.0.0.1" }
$apiPort = $env:API_PORT; if (-not $apiPort) { $apiPort = "9101" }

Write-Host ("Starting Uvicorn: {0} on http://{1}:{2} ..." -f $module, $apiHost, $apiPort)
python -m uvicorn $module --host $apiHost --port $apiPort --reload
"@
Set-Content -Path $bootstrap -Value $bootstrapCode -Encoding UTF8
Write-Host "Wrote $bootstrap"

# (Optional) Status UI with state pill
if ($OverwriteStatusHtml -or -not (Test-Path $statusHtml)) {
  $uiDir = Split-Path $statusHtml -Parent
  Ensure-Dir $uiDir
  $statusHtmlCode = @"
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>AI Project Hub — Status</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: system-ui, Segoe UI, Roboto, sans-serif; margin: 24px; }
    .row { display:flex; align-items:center; gap:10px; }
    .pill { padding:4px 10px; border-radius:999px; color:#fff; font-weight:600; }
    .ok { background:#16a34a; } .warn { background:#f59e0b; } .bad { background:#dc2626; }
    .muted { color:#6b7280; } .mono { font-family: ui-monospace, Menlo, Consolas, monospace; }
    .grid { display:grid; grid-template-columns: 160px 1fr; gap:6px 16px; margin-top:12px; }
  </style>
</head>
<body>
  <h1>AI Project Hub — Status</h1>
  <div class="row">
    <div>IBKR Connection:</div>
    <div id="pill" class="pill warn">Checking…</div>
    <div id="state" class="mono muted"></div>
  </div>
  <div class="grid mono">
    <div>Connected</div><div id="connected">—</div>
    <div>Host:Port</div><div id="hp">—</div>
    <div>ClientId</div><div id="cid">—</div>
    <div>Fail Streak</div><div id="fs">—</div>
    <div>Backoff Level</div><div id="bo">—</div>
    <div>Last Change</div><div id="lc">—</div>
  </div>
  <script>
    async function refresh() {
      try {
        const r = await fetch('/api/status', { cache: 'no-store' });
        const j = await r.json();
        const s = j?.ibkr ?? j?.status?.ibkr ?? {};
        const state = s.state || 'UNKNOWN';
        const connected = !!s.connected;
        const pill = document.getElementById('pill');
        pill.classList.remove('ok','warn','bad');
        if (state === 'CONNECTED' && connected) pill.classList.add('ok');
        else if (state === 'RECONNECTING' || state === 'DEGRADED') pill.classList.add('warn');
        else pill.classList.add('bad');
        pill.textContent = state;
        document.getElementById('state').textContent = connected ? 'connected' : 'not connected';
        document.getElementById('connected').textContent = String(connected);
        document.getElementById('hp').textContent = (s.host||'?') + ':' + (s.port||'?');
        document.getElementById('cid').textContent = s.clientId ?? '—';
        document.getElementById('fs').textContent = s.failStreak ?? '—';
        document.getElementById('bo').textContent = s.backoffLevel ?? '—';
        const ts = s.lastChangeTs ? new Date(s.lastChangeTs * 1000).toLocaleString() : '—';
        document.getElementById('lc').textContent = ts;
      } catch (e) {
        const pill = document.getElementById('pill'); pill.classList.remove('ok','warn'); pill.classList.add('bad');
        pill.textContent = 'ERROR';
        document.getElementById('state').textContent = 'status fetch failed';
      }
    }
    refresh(); setInterval(refresh, 2500);
  </script>
</body>
</html>
"@
  Set-Content -Path $statusHtml -Value $statusHtmlCode -Encoding UTF8
  Write-Host "Wrote $statusHtml"
} else {
  Write-Host "Skipped UI: $statusHtml exists (use -OverwriteStatusHtml to replace)"
}

Write-Host ""
Write-Host "Patch applied."
Write-Host "Next:"
Write-Host "  1) powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1"
Write-Host "  2) start http://127.0.0.1:9101/ui/status.html"
Write-Host "  3) irm http://127.0.0.1:9101/api/status"
