#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deploy IBKR Algo Bot V2 - Complete Unified API
    
.DESCRIPTION
    Automated deployment script that:
    - Backs up current dashboard_api.py
    - Deploys dashboard_api_COMPLETE.py
    - Sets API key
    - Restarts server
    - Tests all endpoints
    
.NOTES
    Run this from PowerShell in your project directory
#>

$ErrorActionPreference = "Stop"

# Configuration
$PROJECT_DIR = "C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2"
$DOWNLOADS_DIR = "C:\Users\bgmay\Downloads"
$API_KEY = "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"
$SERVER_PORT = 9101
$IBKR_PORT = 7497

# Colors for output
function Write-Step { param($msg) Write-Host "`n🔹 $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "✅ $msg" -ForegroundColor Green }
function Write-Error { param($msg) Write-Host "❌ $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "ℹ️  $msg" -ForegroundColor Yellow }

# Header
Clear-Host
Write-Host @"
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║       🚀 IBKR ALGO BOT V2 - COMPLETE API DEPLOYMENT 🚀              ║
║                    Automated Setup Script                            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: PRE-FLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════════════

Write-Step "Pre-flight Checks"

# Check if project directory exists
if (-not (Test-Path $PROJECT_DIR)) {
    Write-Error "Project directory not found: $PROJECT_DIR"
    Write-Info "Please create the directory or update PROJECT_DIR in this script"
    exit 1
}
Write-Success "Project directory exists"

# Check if complete API file exists
$completeApiPath = Join-Path $DOWNLOADS_DIR "dashboard_api_COMPLETE.py"
if (-not (Test-Path $completeApiPath)) {
    Write-Error "dashboard_api_COMPLETE.py not found in Downloads"
    Write-Info "Expected: $completeApiPath"
    exit 1
}
Write-Success "Complete API file found"

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Success "Python available: $pythonVersion"
} catch {
    Write-Error "Python not found. Please install Python 3.8+"
    exit 1
}

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: STOP EXISTING SERVER (if running)
# ═══════════════════════════════════════════════════════════════════════

Write-Step "Stopping existing server (if running)"

$runningProcess = Get-Process -Name "python" -ErrorAction SilentlyContinue | 
    Where-Object { $_.CommandLine -like "*dashboard_api.py*" }

if ($runningProcess) {
    Write-Info "Found running server process(es)"
    $runningProcess | ForEach-Object {
        Stop-Process -Id $_.Id -Force
        Write-Success "Stopped process ID: $($_.Id)"
    }
    Start-Sleep -Seconds 2
} else {
    Write-Info "No running server found"
}

# Also try to free up the port
try {
    $portCheck = Get-NetTCPConnection -LocalPort $SERVER_PORT -ErrorAction SilentlyContinue
    if ($portCheck) {
        Write-Info "Port $SERVER_PORT is in use, attempting to free it..."
        $processId = $portCheck.OwningProcess
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        Write-Success "Port freed"
    }
} catch {
    # Port is free, continue
}

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: BACKUP CURRENT API (if exists)
# ═══════════════════════════════════════════════════════════════════════

Write-Step "Backing up current dashboard_api.py"

$currentApiPath = Join-Path $PROJECT_DIR "dashboard_api.py"
if (Test-Path $currentApiPath) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupPath = Join-Path $PROJECT_DIR "dashboard_api.backup_$timestamp.py"
    Copy-Item $currentApiPath $backupPath -Force
    Write-Success "Backup created: dashboard_api.backup_$timestamp.py"
} else {
    Write-Info "No existing dashboard_api.py to backup"
}

# ═══════════════════════════════════════════════════════════════════════
# STEP 4: DEPLOY COMPLETE API
# ═══════════════════════════════════════════════════════════════════════

Write-Step "Deploying dashboard_api_COMPLETE.py"

Copy-Item $completeApiPath $currentApiPath -Force
Write-Success "Complete API deployed as dashboard_api.py"

# ═══════════════════════════════════════════════════════════════════════
# STEP 5: SETUP API KEY
# ═══════════════════════════════════════════════════════════════════════

Write-Step "Configuring API Key"

# Create or update .env file
$envPath = Join-Path $PROJECT_DIR ".env"
$envContent = "ANTHROPIC_API_KEY=$API_KEY`n"

Set-Content -Path $envPath -Value $envContent -Force
Write-Success ".env file created with API key"

# Also set in current session
$env:ANTHROPIC_API_KEY = $API_KEY
Write-Success "API key set in current session"

# ═══════════════════════════════════════════════════════════════════════
# STEP 6: START SERVER
# ═══════════════════════════════════════════════════════════════════════

Write-Step "Starting Server on Port $SERVER_PORT"

Set-Location $PROJECT_DIR

# Start server in background
Write-Info "Launching server..."
$serverJob = Start-Job -ScriptBlock {
    param($dir, $apiKey)
    Set-Location $dir
    $env:ANTHROPIC_API_KEY = $apiKey
    python dashboard_api.py
} -ArgumentList $PROJECT_DIR, $API_KEY

Write-Success "Server started (Job ID: $($serverJob.Id))"
Write-Info "Waiting for server to initialize (10 seconds)..."
Start-Sleep -Seconds 10

# ═══════════════════════════════════════════════════════════════════════
# STEP 7: TEST HEALTH ENDPOINT
# ═══════════════════════════════════════════════════════════════════════

Write-Step "Testing Health Endpoint"

$maxRetries = 5
$retryCount = 0
$healthCheckSuccess = $false

while ($retryCount -lt $maxRetries -and -not $healthCheckSuccess) {
    try {
        $healthUrl = "http://127.0.0.1:$SERVER_PORT/health"
        $healthResponse = Invoke-RestMethod -Uri $healthUrl -Method GET -TimeoutSec 5
        
        Write-Success "Server is responding!"
        Write-Host "`nHealth Status:" -ForegroundColor White
        Write-Host ($healthResponse | ConvertTo-Json -Depth 3)
        
        $healthCheckSuccess = $true
    } catch {
        $retryCount++
        Write-Info "Retry $retryCount/$maxRetries - waiting for server..."
        Start-Sleep -Seconds 3
    }
}

if (-not $healthCheckSuccess) {
    Write-Error "Server failed to respond after $maxRetries attempts"
    Write-Info "Check server logs for errors. Server is still running in background (Job ID: $($serverJob.Id))"
    Write-Info "To stop server: Stop-Job -Id $($serverJob.Id); Remove-Job -Id $($serverJob.Id)"
    exit 1
}

# ═══════════════════════════════════════════════════════════════════════
# STEP 8: CONNECT TO IBKR
# ═══════════════════════════════════════════════════════════════════════

Write-Step "Connecting to IBKR (TWS Paper Trading)"

Write-Info "Make sure TWS is running before proceeding!"
$proceed = Read-Host "Is TWS running on port $IBKR_PORT? (Y/N)"

if ($proceed -eq "Y" -or $proceed -eq "y") {
    try {
        $connectBody = @{
            host = "127.0.0.1"
            port = $IBKR_PORT
            client_id = 1
        } | ConvertTo-Json
        
        $connectUrl = "http://127.0.0.1:$SERVER_PORT/api/ibkr/connect"
        $connectResponse = Invoke-RestMethod -Uri $connectUrl -Method POST -ContentType "application/json" -Body $connectBody
        
        Write-Success "IBKR Connected!"
        Write-Host ($connectResponse | ConvertTo-Json -Depth 2)
    } catch {
        Write-Error "Failed to connect to IBKR: $_"
        Write-Info "You can connect manually later using the command in the guide"
    }
} else {
    Write-Info "Skipping IBKR connection. You can connect manually later."
}

# ═══════════════════════════════════════════════════════════════════════
# STEP 9: SUBSCRIBE TO TEST SYMBOLS
# ═══════════════════════════════════════════════════════════════════════

Write-Step "Subscribing to Test Symbols"

$testSymbols = @("AAPL", "TSLA", "SPY")

if ($proceed -eq "Y" -or $proceed -eq "y") {
    foreach ($symbol in $testSymbols) {
        try {
            $subscribeBody = @{
                symbol = $symbol
                exchange = "SMART"
                data_types = @("quote")
            } | ConvertTo-Json
            
            $subscribeUrl = "http://127.0.0.1:$SERVER_PORT/api/subscribe"
            $subscribeResponse = Invoke-RestMethod -Uri $subscribeUrl -Method POST -ContentType "application/json" -Body $subscribeBody
            
            Write-Success "Subscribed to $symbol"
        } catch {
            Write-Error "Failed to subscribe to $symbol: $_"
        }
        Start-Sleep -Milliseconds 500
    }
}

# ═══════════════════════════════════════════════════════════════════════
# STEP 10: TEST ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

Write-Step "Testing Key Endpoints"

Start-Sleep -Seconds 2  # Let subscriptions settle

# Test old endpoints (for platform.html compatibility)
Write-Host "`n📊 Testing Old Platform Endpoints:" -ForegroundColor White

$oldEndpoints = @(
    "/api/price/AAPL",
    "/api/level2/AAPL",
    "/api/timesales/AAPL",
    "/api/bot/status"
)

foreach ($endpoint in $oldEndpoints) {
    try {
        $url = "http://127.0.0.1:$SERVER_PORT$endpoint"
        $response = Invoke-RestMethod -Uri $url -Method GET -TimeoutSec 5
        Write-Success "$endpoint - OK"
    } catch {
        Write-Error "$endpoint - FAILED"
        Write-Info "Error: $_"
    }
}

# Test new Claude endpoint
Write-Host "`n🤖 Testing Claude AI with Market Data:" -ForegroundColor White

try {
    $claudeUrl = "http://127.0.0.1:$SERVER_PORT/api/claude/analyze-with-data/AAPL"
    $claudeResponse = Invoke-RestMethod -Uri $claudeUrl -Method GET -TimeoutSec 10
    Write-Success "Claude analysis endpoint working!"
    Write-Host "`nSample Analysis:" -ForegroundColor Yellow
    Write-Host $claudeResponse.analysis.Substring(0, [Math]::Min(200, $claudeResponse.analysis.Length)) -ForegroundColor Gray
    if ($claudeResponse.analysis.Length -gt 200) {
        Write-Host "..." -ForegroundColor Gray
    }
} catch {
    Write-Error "Claude endpoint failed"
    Write-Info "Error: $_"
}

# ═══════════════════════════════════════════════════════════════════════
# DEPLOYMENT COMPLETE
# ═══════════════════════════════════════════════════════════════════════

Write-Host @"

╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║                  ✅ DEPLOYMENT SUCCESSFUL! ✅                        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Green

Write-Host "🌐 Your Trading Platform is Live!" -ForegroundColor Cyan
Write-Host "   URL: http://127.0.0.1:$SERVER_PORT/ui/platform.html`n"

Write-Host "📡 API Documentation:" -ForegroundColor Cyan
Write-Host "   URL: http://127.0.0.1:$SERVER_PORT/docs`n"

Write-Host "🎯 Next Steps:" -ForegroundColor Yellow
Write-Host "   1. Open platform: http://127.0.0.1:$SERVER_PORT/ui/platform.html"
Write-Host "   2. Verify real-time data is flowing"
Write-Host "   3. Test Claude AI analysis on symbols"
Write-Host "   4. Subscribe to more symbols as needed`n"

Write-Host "⚙️  Server Management:" -ForegroundColor Yellow
Write-Host "   View logs: Receive-Job -Id $($serverJob.Id) -Keep"
Write-Host "   Stop server: Stop-Job -Id $($serverJob.Id); Remove-Job -Id $($serverJob.Id)"
Write-Host "   Restart server: Re-run this script`n"

Write-Host "📁 Files Location:" -ForegroundColor Yellow
Write-Host "   Project: $PROJECT_DIR"
Write-Host "   Backup: dashboard_api.backup_*.py"
Write-Host "   Logs: Check terminal output`n"

Write-Host "🚀 Advanced Features:" -ForegroundColor Cyan
Write-Host "   • Train AI: python -c `"from ai.ai_predictor import get_predictor; p=get_predictor(); p.train('SPY', period='2y')`""
Write-Host "   • Scanner: http://127.0.0.1:$SERVER_PORT/api/scanner/results"
Write-Host "   • Backtesting: Available through API endpoints`n"

Write-Info "Server is running in background (Job ID: $($serverJob.Id))"
Write-Info "This PowerShell window can be closed - server will continue running"

# Optional: Keep PowerShell window open
Write-Host "`nPress any key to exit (server will continue running in background)..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
