# Enhanced Bootstrap Script for IBKR Trading Bot
param([switch]$SkipDiagnostics = $false)

Write-Host "🚀 IBKR Trading Bot - Enhanced Bootstrap" -ForegroundColor Green

# Check if we're in the right directory
if (-not (Test-Path "store\code\IBKR_Algo_BOT")) {
    Write-Host "❌ Error: Run this from the ai_project_hub root directory" -ForegroundColor Red
    exit 1
}

# Test TWS connection
Write-Host "🔍 Testing TWS connection..." -ForegroundColor Cyan
try {
    $tcpClient = New-Object System.Net.Sockets.TcpClient
    $tcpClient.Connect("127.0.0.1", 7497)
    $tcpClient.Close()
    Write-Host "✅ TWS is listening on 127.0.0.1:7497" -ForegroundColor Green
} catch {
    Write-Host "❌ Cannot connect to TWS on 127.0.0.1:7497" -ForegroundColor Red
    Write-Host "💡 Start TWS/IB Gateway and enable API" -ForegroundColor Yellow
    Write-Host "💡 Global Config > API > Settings > Enable ActiveX and Socket Clients" -ForegroundColor Yellow
}

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Cyan
pip install --upgrade python-dotenv ib-insync fastapi uvicorn[standard]

# Set environment
$env:PYTHONPATH = (Get-Location).Path

# Start the API
Write-Host "🚀 Starting IBKR Dashboard API..." -ForegroundColor Green
try {
    Push-Location "store\code\IBKR_Algo_BOT"
    python dashboard_api.py
} catch {
    Write-Host "❌ Error starting server: $_" -ForegroundColor Red
} finally {
    Pop-Location
}
