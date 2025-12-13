#!/usr/bin/env pwsh
# Watch AI Auto-Trading Scanner in Real-Time

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AI AUTO-TRADING SCANNER MONITOR" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Watching scanner activity... (Press Ctrl+C to stop)" -ForegroundColor Yellow
Write-Host ""

# Get the log file path
$logPath = "C:\ai_project_hub\store\code\IBKR_Algo_BOT\logs\auto_trader.log"

# Check if log file exists
if (-not (Test-Path $logPath)) {
    Write-Host "[WARNING] Log file not found at: $logPath" -ForegroundColor Yellow
    Write-Host "[INFO] Scanner output will appear in console..." -ForegroundColor Cyan
    Write-Host ""

    # Watch Python process output instead
    Write-Host "Checking for Python bot process..." -ForegroundColor Cyan
    $pythonProc = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
        $_.Path -like "*dashboard_api.py*" -or $_.MainWindowTitle -like "*dashboard*"
    }

    if ($pythonProc) {
        Write-Host "[OK] Bot is running (PID: $($pythonProc.Id))" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] Bot process not found" -ForegroundColor Yellow
    }
    Write-Host ""
}

# Function to display scanner status
function Show-ScannerStatus {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  CURRENT STATUS - $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    # Check bot process
    $botRunning = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
        $_.Path -like "*python*"
    }

    if ($botRunning) {
        Write-Host "[✓] Bot Status: RUNNING" -ForegroundColor Green
    } else {
        Write-Host "[✗] Bot Status: STOPPED" -ForegroundColor Red
        return
    }

    # Try to get worklist via API
    try {
        $worklist = Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/worklist" -Method GET -TimeoutSec 2
        Write-Host "[✓] Worklist: $($worklist.count) symbols" -ForegroundColor Green

        foreach ($stock in $worklist.symbols) {
            $symbol = $stock.symbol
            $price = $stock.current_price
            $change = $stock.percent_change

            $color = if ($change -ge 0) { "Green" } else { "Red" }
            $arrow = if ($change -ge 0) { "↑" } else { "↓" }

            Write-Host "    $symbol`: `$$price $arrow $change%" -ForegroundColor $color
        }
    } catch {
        Write-Host "[!] Could not fetch worklist (API may be starting...)" -ForegroundColor Yellow
    }

    # Try to get AI predictions
    Write-Host "`n[AI PREDICTIONS]" -ForegroundColor Cyan
    try {
        $predictions = Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/ai/predictions" -Method GET -TimeoutSec 2 -ErrorAction SilentlyContinue

        foreach ($pred in $predictions) {
            $symbol = $pred.symbol
            $signal = $pred.prediction
            $confidence = [math]::Round($pred.confidence * 100, 2)

            $color = switch ($signal) {
                "BULLISH" { "Green" }
                "BEARISH" { "Red" }
                default { "Yellow" }
            }

            Write-Host "    $symbol`: $signal (Confidence: $confidence%)" -ForegroundColor $color
        }
    } catch {
        Write-Host "    Waiting for predictions..." -ForegroundColor Yellow
    }

    Write-Host ""
}

# Show initial status
Show-ScannerStatus

# Watch for scanner activity
Write-Host "[LIVE SCANNER OUTPUT]" -ForegroundColor Cyan
Write-Host "Monitoring scanner activity... (updates every 60 seconds)" -ForegroundColor Yellow
Write-Host ""

# Tail the Python output (if we can find it)
# For now, just poll the API every 10 seconds
$counter = 0
while ($true) {
    Start-Sleep -Seconds 10
    $counter++

    # Show full status every 60 seconds (6 cycles of 10 seconds)
    if ($counter -ge 6) {
        Show-ScannerStatus
        $counter = 0
    } else {
        # Quick update - just show timestamp
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Scanner running... (next update in $((6-$counter)*10)s)" -ForegroundColor DarkGray
    }
}
