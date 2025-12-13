# IBKR Connection Diagnostic Script
# Tests connection to TWS/IB Gateway and provides troubleshooting steps

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  IBKR CONNECTION DIAGNOSTIC TOOL" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if TWS/Gateway ports are listening
Write-Host "[1] Checking for TWS/IB Gateway processes..." -ForegroundColor Yellow
$twsProcesses = Get-Process -Name "tws*","ibgateway*","java*" -ErrorAction SilentlyContinue | Where-Object {$_.MainWindowTitle -like "*TWS*" -or $_.MainWindowTitle -like "*IB Gateway*"}

if ($twsProcesses) {
    Write-Host "    [OK] Found TWS/Gateway process(es):" -ForegroundColor Green
    $twsProcesses | ForEach-Object {
        Write-Host "        - PID: $($_.Id) | Name: $($_.ProcessName) | Title: $($_.MainWindowTitle)" -ForegroundColor White
    }
} else {
    Write-Host "    [WARNING] No TWS/IB Gateway process detected!" -ForegroundColor Red
    Write-Host "    ACTION: Start TWS or IB Gateway before running the bot" -ForegroundColor Yellow
}

# Check if ports 7496 or 7497 are listening
Write-Host "`n[2] Checking API ports..." -ForegroundColor Yellow
$ports = @(7496, 7497)
foreach ($port in $ports) {
    $listening = netstat -an | Select-String ":$port " | Select-String "LISTENING"
    if ($listening) {
        Write-Host "    [OK] Port $port is LISTENING" -ForegroundColor Green
    } else {
        Write-Host "    [X] Port $port is NOT listening" -ForegroundColor Red
    }
}

# Test socket connection
Write-Host "`n[3] Testing socket connections..." -ForegroundColor Yellow
foreach ($port in $ports) {
    try {
        $socket = New-Object System.Net.Sockets.TcpClient
        $socket.Connect("127.0.0.1", $port)
        if ($socket.Connected) {
            Write-Host "    [OK] Successfully connected to 127.0.0.1:$port" -ForegroundColor Green
            $socket.Close()
        }
    } catch {
        Write-Host "    [X] Cannot connect to 127.0.0.1:$port - $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Check if Python bot is running
Write-Host "`n[4] Checking for running bot servers..." -ForegroundColor Yellow
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    Write-Host "    [INFO] Found Python process(es):" -ForegroundColor Cyan
    $pythonProcesses | ForEach-Object {
        Write-Host "        - PID: $($_.Id) | Started: $($_.StartTime)" -ForegroundColor White
    }
} else {
    Write-Host "    [INFO] No Python bot processes running" -ForegroundColor White
}

# Summary and recommendations
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  RECOMMENDATIONS" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "To fix IBKR connection issues:" -ForegroundColor Yellow
Write-Host "1. Ensure TWS or IB Gateway is running" -ForegroundColor White
Write-Host "2. In TWS/Gateway: Go to Edit -> Global Configuration -> API -> Settings" -ForegroundColor White
Write-Host "   - Check 'Enable ActiveX and Socket Clients'" -ForegroundColor White
Write-Host "   - Socket port: 7497 (TWS) or 7496 (Gateway)" -ForegroundColor White
Write-Host "   - Uncheck 'Read-Only API' if you want to place orders" -ForegroundColor White
Write-Host "   - Add '127.0.0.1' to Trusted IP Addresses" -ForegroundColor White
Write-Host "3. Restart TWS/Gateway after changing settings" -ForegroundColor White
Write-Host "4. Run this bot: cd C:\ai_project_hub\store\code\IBKR_Algo_BOT && python dashboard_api.py" -ForegroundColor White

Write-Host "`nPress any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
