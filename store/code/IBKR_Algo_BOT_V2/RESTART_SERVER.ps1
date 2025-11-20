# Restart IBKR Dashboard Server
# Run this script in PowerShell to properly restart the server

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "RESTARTING IBKR DASHBOARD SERVER" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Kill any existing server on port 9101
Write-Host "[1/3] Stopping existing server on port 9101..." -ForegroundColor Yellow
try {
    $connection = Get-NetTCPConnection -LocalPort 9101 -ErrorAction SilentlyContinue
    if ($connection) {
        $pid = $connection.OwningProcess
        Write-Host "      Found process $pid, stopping..." -ForegroundColor Gray
        Stop-Process -Id $pid -Force
        Start-Sleep -Seconds 2
        Write-Host "      Server stopped" -ForegroundColor Green
    } else {
        Write-Host "      No server running on port 9101" -ForegroundColor Green
    }
} catch {
    Write-Host "      Could not check port 9101" -ForegroundColor Red
}

# Step 2: Verify port is free
Write-Host ""
Write-Host "[2/3] Verifying port 9101 is free..." -ForegroundColor Yellow
Start-Sleep -Seconds 1
$test = Get-NetTCPConnection -LocalPort 9101 -ErrorAction SilentlyContinue
if ($test) {
    Write-Host "      Port 9101 still in use!" -ForegroundColor Red
    Write-Host "      Try manually killing process:" -ForegroundColor Red
    Write-Host "      taskkill /F /PID $($test.OwningProcess)" -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "      Port 9101 is free" -ForegroundColor Green
}

# Step 3: Start server
Write-Host ""
Write-Host "[3/3] Starting server..." -ForegroundColor Yellow
Write-Host ""
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
python dashboard_api.py
