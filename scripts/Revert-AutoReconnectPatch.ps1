Param(
  [string]$Root = (Get-Location).Path
)
$apiFile = Join-Path $Root "store\code\IBKR_Algo_BOT\dashboard_api.py"
$backups = Get-ChildItem "$apiFile.*.bak" -ErrorAction SilentlyContinue | Sort-Object -Property LastWriteTime -Descending
if (-not $backups -or $backups.Count -eq 0) { Write-Error "No backups found (*.bak)."; exit 1 }
$latest = $backups[0].FullName
Copy-Item $latest $apiFile -Force
Write-Host "Restored $apiFile from $latest"
