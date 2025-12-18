# ============================================================
# SETUP 4AM PRE-MARKET SCHEDULED TASK
# Run this script as Administrator to create the scheduled task
# ============================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  MORPHEUS TRADING BOT - 4AM SCHEDULE SETUP" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script requires Administrator privileges!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host ""
    pause
    exit 1
}

$botPath = "C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2"
$taskName = "Morpheus_4AM_PreMarket"

# Your timezone - CST (Central)
# 4AM EST = 3AM CST
$runTime = "03:00"

Write-Host "Creating scheduled task: $taskName" -ForegroundColor Green
Write-Host "Run time: $runTime (3AM CST = 4AM EST)" -ForegroundColor Green
Write-Host ""

# Remove existing task if it exists
schtasks /delete /tn $taskName /f 2>$null

# Create the scheduled task
$action = New-ScheduledTaskAction -Execute "$botPath\START_4AM_PREMARKET.bat" -WorkingDirectory $botPath
$trigger = New-ScheduledTaskTrigger -Daily -At $runTime
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -WakeToRun

# Register the task
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description "Morpheus Trading Bot - Start 4AM EST Pre-Market Auto-Trading" -Force

Write-Host ""
Write-Host "SUCCESS! Scheduled task created." -ForegroundColor Green
Write-Host ""
Write-Host "Task Details:" -ForegroundColor Cyan
Write-Host "  Name: $taskName"
Write-Host "  Time: 3:00 AM CST (4:00 AM EST)"
Write-Host "  Runs: Daily (weekdays only have pre-market)"
Write-Host ""
Write-Host "IMPORTANT: Make sure the trading server is running!" -ForegroundColor Yellow
Write-Host "  Run: START_TRADING_PLATFORM.bat before bed" -ForegroundColor Yellow
Write-Host ""
Write-Host "When you wake up at 6AM CST, run:" -ForegroundColor Cyan
Write-Host "  CHECK_PREMARKET_REPORT.bat" -ForegroundColor White
Write-Host ""

# Verify task was created
Write-Host "Verifying task..." -ForegroundColor Gray
schtasks /query /tn $taskName /fo LIST

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
pause
