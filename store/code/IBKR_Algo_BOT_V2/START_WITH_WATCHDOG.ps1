# START_WITH_WATCHDOG.ps1
# Starts the IBKR bot with automatic crash recovery and restart

param(
    [int]$MaxRestarts = 100,  # Maximum number of restarts
    [int]$RestartDelay = 5    # Seconds to wait before restart
)

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  IBKR BOT WATCHDOG - AUTO RECOVERY" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "[CONFIG] Max restarts: $MaxRestarts" -ForegroundColor White
Write-Host "[CONFIG] Restart delay: $RestartDelay seconds" -ForegroundColor White
Write-Host ""

$RestartCount = 0
$BotPath = "C:\ai_project_hub\store\code\IBKR_Algo_BOT"
$ScriptPath = Join-Path $BotPath "dashboard_api.py"

# Kill any existing Python processes
Write-Host "[CLEANUP] Stopping existing Python processes..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

while ($RestartCount -lt $MaxRestarts) {
    $RestartCount++

    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "  STARTING BOT (Attempt #$RestartCount)" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor White
    Write-Host ""

    # Start the bot process
    try {
        $Process = Start-Process -FilePath "python" `
                                 -ArgumentList $ScriptPath `
                                 -WorkingDirectory $BotPath `
                                 -PassThru `
                                 -NoNewWindow `
                                 -RedirectStandardOutput "$BotPath\bot_output.log" `
                                 -RedirectStandardError "$BotPath\bot_error.log"

        Write-Host "[OK] Bot started with PID: $($Process.Id)" -ForegroundColor Green
        Write-Host "[OK] Dashboard: http://127.0.0.1:9101/ui/complete_platform.html" -ForegroundColor Green
        Write-Host "[MONITORING] Watching for crashes..." -ForegroundColor Cyan
        Write-Host "[INFO] Press Ctrl+C to stop watchdog" -ForegroundColor Gray
        Write-Host ""

        # Wait for process to exit
        $Process.WaitForExit()

        $ExitCode = $Process.ExitCode
        $ExitTime = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

        Write-Host "`n[WARNING] Bot stopped at $ExitTime" -ForegroundColor Yellow
        Write-Host "[INFO] Exit code: $ExitCode" -ForegroundColor Yellow

        # Check if it was a clean exit
        if ($ExitCode -eq 0) {
            Write-Host "[INFO] Bot exited cleanly - stopping watchdog" -ForegroundColor Green
            break
        }

        # Check uptime - if crashed immediately, increase delay
        $Uptime = (Get-Date) - $Process.StartTime
        if ($Uptime.TotalSeconds -lt 30) {
            Write-Host "[WARNING] Bot crashed within 30 seconds - possible startup issue" -ForegroundColor Red
            $DelayTime = $RestartDelay * 3  # Triple the delay for immediate crashes
        } else {
            $DelayTime = $RestartDelay
        }

        Write-Host "[AUTO-RECOVERY] Restarting in $DelayTime seconds..." -ForegroundColor Yellow
        Start-Sleep -Seconds $DelayTime

    } catch {
        Write-Host "[ERROR] Failed to start bot: $_" -ForegroundColor Red
        Start-Sleep -Seconds $RestartDelay
    }
}

if ($RestartCount -ge $MaxRestarts) {
    Write-Host "`n[CRITICAL] Maximum restart limit ($MaxRestarts) reached!" -ForegroundColor Red
    Write-Host "[INFO] Check logs: bot_output.log and bot_error.log" -ForegroundColor Yellow
    Write-Host "[ACTION] Fix the underlying issue before restarting" -ForegroundColor Yellow
}

Write-Host "`nWatchdog stopped." -ForegroundColor Gray
