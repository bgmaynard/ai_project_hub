# RESTART_BOT.ps1
# Cleanly restart the trading bot

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  RESTARTING TRADING BOT" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Kill all Python processes
Write-Host "[1] Stopping all Python processes..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "    Killing PID: $($_.Id)" -ForegroundColor White
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2

# Verify stopped
$remaining = Get-Process python -ErrorAction SilentlyContinue
if ($remaining) {
    Write-Host "    [WARNING] Some Python processes still running" -ForegroundColor Red
} else {
    Write-Host "    [OK] All Python processes stopped" -ForegroundColor Green
}

# Start the bot
Write-Host "`n[2] Starting IBKR Trading Bot..." -ForegroundColor Yellow
Set-Location "C:\ai_project_hub\store\code\IBKR_Algo_BOT"

Write-Host "    Server will start on: http://127.0.0.1:9101" -ForegroundColor White
Write-Host "    Dashboard: http://127.0.0.1:9101/ui/complete_platform.html" -ForegroundColor White
Write-Host "`n    Press Ctrl+C to stop the bot`n" -ForegroundColor Gray

python dashboard_api.py
