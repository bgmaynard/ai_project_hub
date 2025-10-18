# scripts/bootstrap.ps1
param(
  [string]$Ref = "refs/tags/working-2025-10-18"  # can be branch: refs/heads/chore/sync-working-2025-10-18
)

Write-Host "==> Syncing repo to $Ref"
git fetch --all --prune
git checkout --detach $Ref
git submodule update --init --recursive 2>$null

Write-Host "==> Creating py311 venv and installing requirements"
if (Test-Path .\.venv) { Remove-Item -Recurse -Force .\.venv }
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip wheel
.\.venv\Scripts\python -m pip install -r requirements.txt

Write-Host "==> Creating .env (if missing)"
if (-not (Test-Path .env)) {
@"
API_HOST=127.0.0.1
API_PORT=9101
OFFLINE_MODE=0
TWS_HOST=127.0.0.1
TWS_PORT=7497
TWS_CLIENT_ID=777
"@ | Set-Content .env -Encoding UTF8
}

Write-Host "==> Launching API"
powershell -ExecutionPolicy Bypass -File .\run.ps1
