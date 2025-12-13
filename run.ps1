# C:\ai_project_hub\run.ps1
$ErrorActionPreference = "Stop"

# --- Project root ---
Set-Location "C:\ai_project_hub"

# --- Inline heartbeat (no external script needed) ---
$hbPath = "C:\ai_project_hub\store\ai_shared\mesh_heartbeat.json"
$hbDir  = Split-Path $hbPath
if (-not (Test-Path $hbDir)) { New-Item -ItemType Directory -Force -Path $hbDir | Out-Null }
$now = [DateTime]::Now.ToString("o")
"{`"updated`":`"$now`"}" | Set-Content $hbPath -Encoding UTF8
Write-Host "Heartbeat updated at $hbPath"

# --- Defaults ---
$apiHost = "127.0.0.1"
$apiPort = "9101"

# --- Read .env overrides if present ---
if (Test-Path ".env") {
  foreach ($line in Get-Content .env) {
    if ($line -match '^\s*API_HOST\s*=\s*(.+)\s*$') { $apiHost = $Matches[1].Trim() }
    if ($line -match '^\s*API_PORT\s*=\s*(.+)\s*$') { $apiPort = $Matches[1].Trim() }
  }
} else {
  Write-Warning ("'.env' not found. Using defaults: {0}:{1}" -f $apiHost, $apiPort)
}

# --- Where your FastAPI app lives ---
$AppDir = "C:\ai_project_hub\store\code\IBKR_Algo_BOT"
$AppMod = "dashboard_api:app"

if (-not (Test-Path (Join-Path $AppDir "dashboard_api.py"))) {
  Write-Error "Could not find dashboard_api.py in $AppDir"
  exit 1
}

# --- Start API ---
Write-Host ("==> Starting API on {0}:{1} ..." -f $apiHost, $apiPort)
& .\.venv\Scripts\python -m uvicorn $AppMod `
  --host $apiHost `
  --port $apiPort `
  --app-dir $AppDir
