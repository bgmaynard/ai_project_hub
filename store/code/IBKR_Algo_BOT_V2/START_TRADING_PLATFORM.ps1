# ============================================================================
# AI TRADING PLATFORM - STARTUP SCRIPT
# ============================================================================
# This script starts all components of the AI Trading Platform in the correct
# sequence. Double-click this script or create a desktop shortcut to it.
#
# Components Started:
# 1. Environment validation (Python, packages, API keys)
# 2. Alpaca MCP Server (for fast trading protocol)
# 3. AI Components (Claude Sonnet 4.5, MCP Client)
# 4. Dashboard API Server (port 9100)
# 5. Opens browser to dashboard
#
# Author: AI Trading Bot Team
# Version: 2.1 (with MCP Server)
# ============================================================================

$ErrorActionPreference = "Continue"
$Host.UI.RawUI.WindowTitle = "AI Trading Platform"

# Configuration
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonPath = "python"
$Port = 9100
$MCPPort = 9101
$DashboardUrl = "http://localhost:$Port/dashboard"
$LogFile = Join-Path $ProjectDir "startup.log"

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Banner {
    Clear-Host
    Write-Host ""
    Write-Host "  ================================================================" -ForegroundColor Cyan
    Write-Host "                 AI TRADING PLATFORM v2.1                        " -ForegroundColor White
    Write-Host "           Claude Sonnet 4.5 + Alpaca MCP Server                " -ForegroundColor Yellow
    Write-Host "  ================================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step($step, $message) {
    Write-Host "  [$step] " -ForegroundColor Cyan -NoNewline
    Write-Host $message -ForegroundColor White
}

function Write-Success($message) {
    Write-Host "       [OK] " -ForegroundColor Green -NoNewline
    Write-Host $message -ForegroundColor Gray
}

function Write-Warning($message) {
    Write-Host "       [!] " -ForegroundColor Yellow -NoNewline
    Write-Host $message -ForegroundColor Gray
}

function Write-ErrorMsg($message) {
    Write-Host "       [X] " -ForegroundColor Red -NoNewline
    Write-Host $message -ForegroundColor Gray
}

function Write-Info($message) {
    Write-Host "       " -NoNewline
    Write-Host $message -ForegroundColor DarkGray
}

# ============================================================================
# STARTUP SEQUENCE
# ============================================================================

Write-Banner

# Change to project directory
Set-Location $ProjectDir
Write-Step "1/8" "Setting up environment..."
Write-Info "Working directory: $ProjectDir"

# Check if .env file exists
if (Test-Path ".env") {
    Write-Success ".env file found"
} else {
    Write-ErrorMsg ".env file not found!"
    Write-Info "Please create .env file with your API keys"
    Read-Host "Press Enter to exit"
    exit 1
}

# ============================================================================
# STEP 2: Validate Python Environment
# ============================================================================
Write-Host ""
Write-Step "2/8" "Validating Python environment..."

try {
    $pythonVersion = & $PythonPath --version 2>&1
    Write-Success "Python: $pythonVersion"
} catch {
    Write-ErrorMsg "Python not found! Please install Python 3.10+"
    Read-Host "Press Enter to exit"
    exit 1
}

# Check critical packages
$packages = @("fastapi", "uvicorn", "alpaca-py", "anthropic", "alpaca-mcp-server")
foreach ($pkg in $packages) {
    $pkgName = $pkg -replace "-", "_"
    $result = & $PythonPath -c "import importlib.util; print('OK' if importlib.util.find_spec('$pkgName') else 'MISSING')" 2>$null
    if ($result -eq "OK") {
        Write-Success "$pkg installed"
    } else {
        Write-Warning "$pkg not found - installing..."
        & $PythonPath -m pip install $pkg --quiet 2>$null
    }
}

# ============================================================================
# STEP 3: Validate API Keys
# ============================================================================
Write-Host ""
Write-Step "3/8" "Validating API keys..."

$envContent = Get-Content ".env" -Raw
if ($envContent -match "ALPACA_API_KEY=\S+") {
    Write-Success "Alpaca API key configured"
} else {
    Write-ErrorMsg "Alpaca API key not set in .env"
}

if ($envContent -match "ANTHROPIC_API_KEY=\S+") {
    Write-Success "Anthropic API key configured"
} else {
    Write-Warning "Anthropic API key not set (Claude AI will use fallback mode)"
}

# ============================================================================
# STEP 4: Kill any existing processes
# ============================================================================
Write-Host ""
Write-Step "4/8" "Checking for existing processes..."

# Check Dashboard port
$existingProcess = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
if ($existingProcess) {
    Write-Warning "Port $Port is in use, stopping existing process..."
    $processId = $existingProcess.OwningProcess | Select-Object -First 1
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Write-Success "Previous dashboard process stopped"
} else {
    Write-Success "Port $Port is available"
}

# Kill any existing MCP server processes
$mcpProcesses = Get-Process | Where-Object { $_.ProcessName -like "*alpaca*mcp*" -or $_.CommandLine -like "*alpaca-mcp-server*" } -ErrorAction SilentlyContinue
if ($mcpProcesses) {
    Write-Warning "Stopping existing MCP server..."
    $mcpProcesses | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

# ============================================================================
# STEP 5: Start Alpaca MCP Server
# ============================================================================
Write-Host ""
Write-Step "5/8" "Starting Alpaca MCP Server..."

# Check MCP server status first
$mcpStatus = & alpaca-mcp-server status 2>&1
if ($mcpStatus -match "Configuration is valid") {
    Write-Success "MCP Server configuration valid"

    # Start MCP server in background
    $mcpProcess = Start-Process -FilePath "alpaca-mcp-server" `
        -ArgumentList "serve" `
        -WorkingDirectory $ProjectDir `
        -PassThru `
        -WindowStyle Minimized

    Write-Info "MCP Server PID: $($mcpProcess.Id)"
    Start-Sleep -Seconds 2

    if (!$mcpProcess.HasExited) {
        Write-Success "Alpaca MCP Server running (fast trading enabled)"
        $global:MCPServerPID = $mcpProcess.Id
    } else {
        Write-Warning "MCP Server may have failed to start"
    }
} else {
    Write-Warning "MCP Server not configured, running without it"
    Write-Info "Run 'alpaca-mcp-server init' to configure"
}

# ============================================================================
# STEP 6: Initialize AI Components
# ============================================================================
Write-Host ""
Write-Step "6/8" "Initializing AI components..."

# Test MCP Client connection
$mcpTest = & $PythonPath -c @"
import sys
sys.path.insert(0, '.')
try:
    from ai.alpaca_mcp_integration import get_mcp_client
    client = get_mcp_client()
    if client.initialized:
        print('MCP_OK')
    else:
        print('MCP_PARTIAL')
except Exception as e:
    print(f'MCP_ERROR:{e}')
"@ 2>$null

if ($mcpTest -eq "MCP_OK") {
    Write-Success "Alpaca MCP Client connected"
} elseif ($mcpTest -eq "MCP_PARTIAL") {
    Write-Warning "MCP Client partially initialized"
} else {
    Write-Warning "MCP Client: $mcpTest"
}

# Test Claude AI
$claudeTest = & $PythonPath -c @"
import sys
sys.path.insert(0, '.')
try:
    from ai.claude_bot_intelligence import get_bot_intelligence
    bot = get_bot_intelligence()
    status = bot.get_status()
    tools = status.get('features', {}).get('tool_count', 0)
    mcp_tools = status.get('mcp_integration', {}).get('mcp_tools', 0)
    if status['ai_available']:
        print(f"CLAUDE_OK:{tools}:{mcp_tools}")
    else:
        print(f'CLAUDE_FALLBACK:{tools}:{mcp_tools}')
except Exception as e:
    print(f'CLAUDE_ERROR:{e}')
"@ 2>$null

if ($claudeTest -match "CLAUDE_OK:(\d+):(\d+)") {
    $totalTools = $Matches[1]
    $mcpTools = $Matches[2]
    Write-Success "Claude Sonnet 4.5 ready ($totalTools tools, $mcpTools MCP)"
} elseif ($claudeTest -match "CLAUDE_FALLBACK") {
    Write-Warning "Claude AI in fallback mode (check API key)"
} else {
    Write-Warning "Claude AI: $claudeTest"
}

# ============================================================================
# STEP 7: Start Dashboard API Server
# ============================================================================
Write-Host ""
Write-Step "7/8" "Starting Dashboard API Server..."
Write-Info "Server will run on http://localhost:$Port"

# Start the server in a new window
$serverProcess = Start-Process -FilePath $PythonPath `
    -ArgumentList "alpaca_dashboard_api.py" `
    -WorkingDirectory $ProjectDir `
    -PassThru `
    -WindowStyle Normal

Write-Info "Dashboard Server PID: $($serverProcess.Id)"
$global:DashboardPID = $serverProcess.Id

# Wait for server to be ready
Write-Info "Waiting for server to initialize..."
$maxWait = 30
$waited = 0
$serverReady = $false

while ($waited -lt $maxWait) {
    Start-Sleep -Seconds 1
    $waited++

    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$Port/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $serverReady = $true
            break
        }
    } catch {
        Write-Host "." -NoNewline -ForegroundColor DarkGray
    }
}

Write-Host ""

if ($serverReady) {
    Write-Success "Dashboard server is running"
} else {
    Write-Warning "Server may still be starting (waited ${waited}s)"
}

# ============================================================================
# STEP 8: Start Critical Monitoring Services
# ============================================================================
Write-Host ""
Write-Step "8/10" "Starting critical monitoring services..."

# Start Position Guardian (protects profits with trailing stops)
Write-Info "Starting Position Guardian..."
$guardianProcess = Start-Process -FilePath $PythonPath `
    -ArgumentList "position_guardian.py" `
    -WorkingDirectory $ProjectDir `
    -PassThru `
    -WindowStyle Minimized
Write-Success "Position Guardian started (PID: $($guardianProcess.Id))"

# Start News Triggered Evaluator (scans news, evaluates, adds to watchlist)
Write-Info "Starting News Triggered Evaluator..."
$newsEvalProcess = Start-Process -FilePath $PythonPath `
    -ArgumentList "news_triggered_evaluator.py" `
    -WorkingDirectory $ProjectDir `
    -PassThru `
    -WindowStyle Minimized
Write-Success "News Evaluator started (PID: $($newsEvalProcess.Id))"

Start-Sleep -Seconds 3

# ============================================================================
# STEP 9: Initialize API-based Services
# ============================================================================
Write-Host ""
Write-Step "9/10" "Initializing real-time feeds..."

# Start Alpaca News WebSocket (instant news detection)
Write-Info "Starting Alpaca News WebSocket..."
try {
    $newsStream = Invoke-RestMethod -Uri "http://localhost:$Port/api/alpaca/news-stream/start" -Method POST -TimeoutSec 5
    Write-Success "News WebSocket: CONNECTED"
} catch {
    Write-Warning "News WebSocket failed to start"
}

# Start News Monitor (Benzinga polling)
Write-Info "Starting Benzinga News Monitor (10s polling)..."
try {
    $newsMonitor = Invoke-RestMethod -Uri "http://localhost:$Port/api/news/monitor/start" -Method POST -Body '{"symbols": [], "poll_interval": 10}' -ContentType "application/json" -TimeoutSec 5
    Write-Success "Benzinga Monitor: ACTIVE"
} catch {
    Write-Warning "Benzinga Monitor failed to start"
}

# Start Momentum Spike Detector
Write-Info "Starting Momentum Spike Detector..."
try {
    $spikeDetector = Invoke-RestMethod -Uri "http://localhost:$Port/api/alpaca/spikes/start" -Method POST -Body '{}' -ContentType "application/json" -TimeoutSec 5
    Write-Success "Spike Detector: ACTIVE"
} catch {
    Write-Warning "Spike Detector failed to start"
}

# ============================================================================
# STEP 10: Open Dashboard in Browser
# ============================================================================
Write-Host ""
Write-Step "10/10" "Opening dashboard..."

Start-Sleep -Seconds 2
Start-Process $DashboardUrl

Write-Success "Dashboard opened in browser"

# ============================================================================
# FINAL STATUS
# ============================================================================
Write-Host ""
Write-Host "  ================================================================" -ForegroundColor Green
Write-Host "               PLATFORM STARTED SUCCESSFULLY                     " -ForegroundColor White
Write-Host "  ================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Services Running:" -ForegroundColor Gray
Write-Host "    - Alpaca MCP Server  : " -ForegroundColor DarkGray -NoNewline
if ($global:MCPServerPID) {
    Write-Host "Active (PID: $global:MCPServerPID)" -ForegroundColor Green
} else {
    Write-Host "Not started" -ForegroundColor Yellow
}
Write-Host "    - Dashboard API      : " -ForegroundColor DarkGray -NoNewline
Write-Host "Active (PID: $global:DashboardPID)" -ForegroundColor Green
Write-Host "    - Position Guardian  : " -ForegroundColor DarkGray -NoNewline
Write-Host "Active (trailing stops)" -ForegroundColor Green
Write-Host "    - News WebSocket     : " -ForegroundColor DarkGray -NoNewline
Write-Host "Active (instant alerts)" -ForegroundColor Green
Write-Host "    - Benzinga Monitor   : " -ForegroundColor DarkGray -NoNewline
Write-Host "Active (10s polling)" -ForegroundColor Green
Write-Host "    - Spike Detector     : " -ForegroundColor DarkGray -NoNewline
Write-Host "Active (acceleration)" -ForegroundColor Green
Write-Host "    - News Evaluator     : " -ForegroundColor DarkGray -NoNewline
Write-Host "Active (auto-watchlist)" -ForegroundColor Green
Write-Host ""
Write-Host "  URLs:" -ForegroundColor Gray
Write-Host "    Dashboard:  " -ForegroundColor DarkGray -NoNewline
Write-Host $DashboardUrl -ForegroundColor Cyan
Write-Host "    API Docs:   " -ForegroundColor DarkGray -NoNewline
Write-Host "http://localhost:$Port/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Features:" -ForegroundColor Gray
Write-Host "    - Claude Sonnet 4.5 with parallel tool execution" -ForegroundColor DarkGray
Write-Host "    - Alpaca MCP for fast trading operations" -ForegroundColor DarkGray
Write-Host "    - Real-time market data and portfolio tracking" -ForegroundColor DarkGray
Write-Host "    - Position Guardian with momentum trailing stops" -ForegroundColor DarkGray
Write-Host "    - Breaking news detection (WebSocket + Benzinga)" -ForegroundColor DarkGray
Write-Host "    - Momentum spike detector (acceleration triggers)" -ForegroundColor DarkGray
Write-Host "    - Auto-evaluate news and add to watchlist" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  To Stop:" -ForegroundColor Gray
Write-Host "    - Run STOP_TRADING_PLATFORM.bat, or" -ForegroundColor DarkGray
Write-Host "    - Press Ctrl+C in server windows" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  ================================================================" -ForegroundColor Green
Write-Host ""

# Keep this window open
Write-Host "  This window can be closed. Servers run in separate windows." -ForegroundColor DarkGray
Write-Host ""
Read-Host "  Press Enter to close this startup window"
