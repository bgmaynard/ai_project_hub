# FORCE_RESTART_BOT.ps1
# Aggressively kill all processes using port 9101 and restart the bot

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  FORCE RESTART - Freeing Port 9101" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Function to kill processes using port 9101
function Kill-PortProcesses {
    param([int]$Port)

    Write-Host "[1] Finding processes using port $Port..." -ForegroundColor Yellow

    # Get processes using the port
    $connections = netstat -ano | Select-String ":$Port" | Select-String "LISTENING"

    if ($connections) {
        $pids = @()
        foreach ($line in $connections) {
            # Convert MatchInfo to string and extract PID (last column after LISTENING)
            $lineText = $line.ToString()
            # netstat output format: TCP  0.0.0.0:9101  0.0.0.0:0  LISTENING  12345
            if ($lineText -match 'LISTENING\s+(\d+)') {
                $pid = [int]$matches[1]
                if ($pid -gt 0 -and $pids -notcontains $pid) {
                    $pids += $pid
                    Write-Host "    DEBUG: Found PID $pid from netstat" -ForegroundColor Gray
                }
            }
        }

        if ($pids.Count -gt 0) {
            Write-Host "    Found $($pids.Count) process(es) using port $Port" -ForegroundColor White
            foreach ($pid in $pids) {
                try {
                    $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
                    if ($process) {
                        Write-Host "    Killing PID $pid ($($process.ProcessName))" -ForegroundColor White
                        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
                    }
                } catch {
                    Write-Host "    Failed to kill PID $pid" -ForegroundColor Red
                }
            }
        } else {
            Write-Host "    No processes found using port $Port" -ForegroundColor Green
        }
    } else {
        Write-Host "    Port $Port is free" -ForegroundColor Green
    }
}

# Kill processes using port 9101
Kill-PortProcesses -Port 9101

# Also kill all Python processes as backup
Write-Host "`n[2] Killing all Python processes..." -ForegroundColor Yellow
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    $pythonProcesses | ForEach-Object {
        Write-Host "    Killing Python PID: $($_.Id)" -ForegroundColor White
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "    No Python processes found" -ForegroundColor Green
}

# Wait for processes to fully terminate
Write-Host "`n[3] Waiting for processes to terminate..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Verify port is free
Write-Host "`n[4] Verifying port 9101 is free..." -ForegroundColor Yellow
$stillUsed = netstat -ano | Select-String ":9101" | Select-String "LISTENING"
if ($stillUsed) {
    Write-Host "    [ERROR] Port 9101 is still in use!" -ForegroundColor Red
    Write-Host "    Trying one more time..." -ForegroundColor Yellow
    Kill-PortProcesses -Port 9101
    Start-Sleep -Seconds 2

    $stillUsed = netstat -ano | Select-String ":9101" | Select-String "LISTENING"
    if ($stillUsed) {
        Write-Host "    [CRITICAL] Cannot free port 9101" -ForegroundColor Red
        Write-Host "    You may need to restart your computer" -ForegroundColor Red
        exit 1
    }
}
Write-Host "    [OK] Port 9101 is free!" -ForegroundColor Green

# Verify no Python processes remain
Write-Host "`n[5] Verifying no Python processes remain..." -ForegroundColor Yellow
$remaining = Get-Process python -ErrorAction SilentlyContinue
if ($remaining) {
    Write-Host "    [WARNING] Some Python processes still running:" -ForegroundColor Red
    $remaining | ForEach-Object {
        Write-Host "      PID: $($_.Id)" -ForegroundColor Red
    }
} else {
    Write-Host "    [OK] All Python processes stopped" -ForegroundColor Green
}

# Start the bot
Write-Host "`n[6] Starting IBKR Trading Bot..." -ForegroundColor Yellow
Set-Location "C:\ai_project_hub\store\code\IBKR_Algo_BOT"

Write-Host "    Server will start on: http://127.0.0.1:9101" -ForegroundColor White
Write-Host "    Dashboard: http://127.0.0.1:9101/ui/complete_platform.html" -ForegroundColor White
Write-Host "`n    Press Ctrl+C to stop the bot`n" -ForegroundColor Gray

python dashboard_api.py
