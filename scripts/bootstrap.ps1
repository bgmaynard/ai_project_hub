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

# -- pick python ---------------------------------------------------------------
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

# -- venv ----------------------------------------------------------------------
if (!(Test-Path ".venv")) {
    Write-Host "Creating virtual environment at .venv ..."
    & $py -m venv .venv
}

# Activate
$venvActivate = Join-Path $root ".venv\Scripts\Activate.ps1"
. $venvActivate

# Ensure pip
python -m ensurepip --upgrade | Out-Null
python -m pip install --upgrade pip wheel setuptools | Out-Null

# -- deps ----------------------------------------------------------------------
if (Test-Path "requirements.txt") {
    Write-Host "Installing from requirements.txt ..."
    python -m pip install -r requirements.txt
} else {
    Write-Host "Installing core dependencies ..."
    python -m pip install fastapi==0.110.* "uvicorn[standard]==0.27.*" ib-insync==0.9.* python-dotenv==1.*
}

# -- .env loader (Process scope only) ------------------------------------------
function Set-EnvVar {
    param([string]$Name, [string]$Value)
    if ([string]::IsNullOrWhiteSpace($Name)) { return }
    # Process scope
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

# -- launch uvicorn ------------------------------------------------------------
$module = "store.code.IBKR_Algo_BOT.dashboard_api:app"
$apiHost = $env:API_HOST; if (-not $apiHost) { $apiHost = "127.0.0.1" }
$apiPort = $env:API_PORT; if (-not $apiPort) { $apiPort = "9101" }

Write-Host ("Starting Uvicorn: {0} on http://{1}:{2} ..." -f $module, $apiHost, $apiPort)
python -m uvicorn $module --host $apiHost --port $apiPort --reload
