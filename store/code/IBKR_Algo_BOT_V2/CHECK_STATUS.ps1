#!/usr/bin/env pwsh
<#
.SYNOPSIS
    IBKR Algo Bot V2 - System Status Checker
    
.DESCRIPTION
    Comprehensive status check for all system components:
    - Server health
    - IBKR connectivity
    - Claude AI availability
    - Market data subscriptions
    - Endpoint functionality
    
.NOTES
    Run this anytime to verify system status
#>

$ErrorActionPreference = "Continue"
$SERVER_URL = "http://127.0.0.1:9101"

# Colors
function Write-Check { param($msg) Write-Host "🔍 $msg" -ForegroundColor Cyan }
function Write-Pass { param($msg) Write-Host "✅ $msg" -ForegroundColor Green }
function Write-Fail { param($msg) Write-Host "❌ $msg" -ForegroundColor Red }
function Write-Warn { param($msg) Write-Host "⚠️  $msg" -ForegroundColor Yellow }
function Write-Info { param($msg) Write-Host "ℹ️  $msg" -ForegroundColor Gray }

# Header
Clear-Host
Write-Host @"
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║          🔍 IBKR ALGO BOT V2 - SYSTEM STATUS CHECK 🔍              ║
║                   Comprehensive Diagnostics                          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

$script:passCount = 0
$script:failCount = 0
$script:warnCount = 0

function Test-Component {
    param(
        [string]$Name,
        [scriptblock]$Test,
        [string]$FailMessage = "Failed",
        [string]$SuccessMessage = "OK"
    )
    
    Write-Check "Testing: $Name"
    try {
        $result = & $Test
        if ($result) {
            Write-Pass "$Name - $SuccessMessage"
            $script:passCount++
            return $true
        } else {
            Write-Fail "$Name - $FailMessage"
            $script:failCount++
            return $false
        }
    } catch {
        Write-Fail "$Name - Error: $_"
        $script:failCount++
        return $false
    }
}

# ═══════════════════════════════════════════════════════════════════════
# 1. SERVER CONNECTIVITY
# ═══════════════════════════════════════════════════════════════════════

Write-Host "`n═══════════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host "                    SERVER CONNECTIVITY" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════════════════`n" -ForegroundColor White

$serverRunning = Test-Component -Name "Server Reachable" -Test {
    try {
        $response = Invoke-RestMethod -Uri "$SERVER_URL/health" -TimeoutSec 5
        return $true
    } catch {
        return $false
    }
} -SuccessMessage "Server responding on port 9101"

if (-not $serverRunning) {
    Write-Warn "Server is not running or not reachable"
    Write-Info "Start server with: python dashboard_api.py"
    Write-Host "`n═══════════════════════════════════════════════════════════════`n" -ForegroundColor White
    exit 1
}

# ═══════════════════════════════════════════════════════════════════════
# 2. HEALTH STATUS
# ═══════════════════════════════════════════════════════════════════════

Write-Host "`n═══════════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host "                      HEALTH STATUS" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════════════════`n" -ForegroundColor White

try {
    $health = Invoke-RestMethod -Uri "$SERVER_URL/health"
    
    # Overall status
    if ($health.status -eq "healthy") {
        Write-Pass "Overall Status: HEALTHY"
        $script:passCount++
    } else {
        Write-Fail "Overall Status: $($health.status)"
        $script:failCount++
    }
    
    # IBKR Connection
    if ($health.ibkr_connected) {
        Write-Pass "IBKR Connection: CONNECTED"
        $script:passCount++
    } else {
        Write-Warn "IBKR Connection: DISCONNECTED"
        Write-Info "Connect with: /api/ibkr/connect"
        $script:warnCount++
    }
    
    # IBKR Available
    if ($health.ibkr_available) {
        Write-Pass "IBKR Library: AVAILABLE"
        $script:passCount++
    } else {
        Write-Fail "IBKR Library: NOT AVAILABLE"
        Write-Info "Install with: pip install ib-insync"
        $script:failCount++
    }
    
    # Claude AI
    if ($health.claude_available) {
        Write-Pass "Claude AI: AVAILABLE"
        $script:passCount++
    } else {
        Write-Fail "Claude AI: NOT AVAILABLE"
        Write-Info "Check API key in environment or .env file"
        $script:failCount++
    }
    
    # AI Predictor
    if ($health.ai_predictor_loaded) {
        Write-Pass "AI Predictor: TRAINED"
        $script:passCount++
    } else {
        Write-Warn "AI Predictor: NOT TRAINED"
        Write-Info "Train with: python -c 'from ai.ai_predictor import get_predictor; p=get_predictor(); p.train(\"SPY\")'"
        $script:warnCount++
    }
    
    # Active Subscriptions
    if ($health.active_subscriptions -gt 0) {
        Write-Pass "Active Subscriptions: $($health.active_subscriptions)"
        Write-Info "Tracking: $($health.symbols_tracked -join ', ')"
        $script:passCount++
    } else {
        Write-Warn "Active Subscriptions: NONE"
        Write-Info "Subscribe to symbols with: /api/subscribe"
        $script:warnCount++
    }
    
} catch {
    Write-Fail "Failed to get health status: $_"
    $script:failCount++
}

# ═══════════════════════════════════════════════════════════════════════
# 3. CORE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

Write-Host "`n═══════════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host "                     CORE ENDPOINTS" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════════════════`n" -ForegroundColor White

$coreEndpoints = @(
    @{Name="IBKR Status"; Url="/api/ibkr/status"},
    @{Name="Bot Status"; Url="/api/bot/status"},
    @{Name="Scanner Results"; Url="/api/scanner/results"}
)

foreach ($endpoint in $coreEndpoints) {
    Test-Component -Name $endpoint.Name -Test {
        try {
            $response = Invoke-RestMethod -Uri "$SERVER_URL$($endpoint.Url)" -TimeoutSec 5
            return $true
        } catch {
            return $false
        }
    } | Out-Null
}

# ═══════════════════════════════════════════════════════════════════════
# 4. MARKET DATA ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

Write-Host "`n═══════════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host "                  MARKET DATA ENDPOINTS" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════════════════`n" -ForegroundColor White

# Check if we have any subscriptions first
$testSymbol = $null
if ($health.symbols_tracked -and $health.symbols_tracked.Count -gt 0) {
    $testSymbol = $health.symbols_tracked[0]
    Write-Info "Testing with symbol: $testSymbol"
} else {
    Write-Warn "No symbols subscribed - skipping market data tests"
    Write-Info "Subscribe to a symbol first: /api/subscribe"
}

if ($testSymbol) {
    $marketEndpoints = @(
        @{Name="Price Data"; Url="/api/price/$testSymbol"},
        @{Name="Level 2 Data"; Url="/api/level2/$testSymbol"},
        @{Name="Time & Sales"; Url="/api/timesales/$testSymbol"}
    )
    
    foreach ($endpoint in $marketEndpoints) {
        Test-Component -Name $endpoint.Name -Test {
            try {
                $response = Invoke-RestMethod -Uri "$SERVER_URL$($endpoint.Url)" -TimeoutSec 5
                return $true
            } catch {
                return $false
            }
        } | Out-Null
    }
}

# ═══════════════════════════════════════════════════════════════════════
# 5. CLAUDE AI ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

Write-Host "`n═══════════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host "                   CLAUDE AI ENDPOINTS" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════════════════`n" -ForegroundColor White

if ($testSymbol -and $health.claude_available) {
    Write-Check "Testing Claude AI with real market data..."
    try {
        $claudeUrl = "$SERVER_URL/api/claude/analyze-with-data/$testSymbol"
        $claudeResponse = Invoke-RestMethod -Uri $claudeUrl -TimeoutSec 15
        
        if ($claudeResponse.analysis) {
            Write-Pass "Claude Analysis: WORKING"
            Write-Info "Data Source: $($claudeResponse.data_source)"
            Write-Host "`nSample Analysis:" -ForegroundColor Yellow
            $preview = $claudeResponse.analysis.Substring(0, [Math]::Min(150, $claudeResponse.analysis.Length))
            Write-Host $preview -ForegroundColor Gray
            if ($claudeResponse.analysis.Length -gt 150) {
                Write-Host "..." -ForegroundColor Gray
            }
            $script:passCount++
        } else {
            Write-Fail "Claude Analysis: NO RESPONSE"
            $script:failCount++
        }
    } catch {
        Write-Fail "Claude Analysis: ERROR - $_"
        $script:failCount++
    }
} else {
    Write-Warn "Claude AI test skipped (no symbols subscribed or Claude unavailable)"
    $script:warnCount++
}

# ═══════════════════════════════════════════════════════════════════════
# 6. FRONTEND ACCESS
# ═══════════════════════════════════════════════════════════════════════

Write-Host "`n═══════════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host "                    FRONTEND ACCESS" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════════════════`n" -ForegroundColor White

$frontendPaths = @(
    @{Name="Platform HTML"; Url="/ui/platform.html"},
    @{Name="API Documentation"; Url="/docs"}
)

foreach ($path in $frontendPaths) {
    Test-Component -Name $path.Name -Test {
        try {
            $response = Invoke-WebRequest -Uri "$SERVER_URL$($path.Url)" -TimeoutSec 5 -UseBasicParsing
            return $response.StatusCode -eq 200
        } catch {
            return $false
        }
    } | Out-Null
}

# ═══════════════════════════════════════════════════════════════════════
# 7. SYSTEM RESOURCES
# ═══════════════════════════════════════════════════════════════════════

Write-Host "`n═══════════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host "                    SYSTEM RESOURCES" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════════════════`n" -ForegroundColor White

# Check Python process
$pythonProcess = Get-Process -Name "python" -ErrorAction SilentlyContinue | 
    Where-Object { $_.CommandLine -like "*dashboard_api*" }

if ($pythonProcess) {
    Write-Pass "Server Process: RUNNING (PID: $($pythonProcess.Id))"
    Write-Info "Memory Usage: $([math]::Round($pythonProcess.WorkingSet64 / 1MB, 2)) MB"
    Write-Info "CPU Time: $($pythonProcess.CPU) seconds"
    $script:passCount++
} else {
    Write-Warn "Server Process: NOT FOUND"
    Write-Info "Server may be running in a different process"
    $script:warnCount++
}

# Check port
try {
    $portCheck = Get-NetTCPConnection -LocalPort 9101 -ErrorAction SilentlyContinue
    if ($portCheck) {
        Write-Pass "Port 9101: IN USE"
        Write-Info "State: $($portCheck.State)"
        $script:passCount++
    } else {
        Write-Fail "Port 9101: NOT IN USE"
        $script:failCount++
    }
} catch {
    Write-Warn "Port Check: UNABLE TO VERIFY"
    $script:warnCount++
}

# Check TWS (if IBKR is supposed to be connected)
if ($health.ibkr_connected -or $health.ibkr_available) {
    $twsProcess = Get-Process -Name "*tws*", "*ibgateway*" -ErrorAction SilentlyContinue
    if ($twsProcess) {
        Write-Pass "TWS/IB Gateway: RUNNING"
        $script:passCount++
    } else {
        Write-Warn "TWS/IB Gateway: NOT DETECTED"
        Write-Info "Make sure TWS or IB Gateway is running for IBKR connectivity"
        $script:warnCount++
    }
}

# ═══════════════════════════════════════════════════════════════════════
# 8. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

Write-Host "`n═══════════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host "                     CONFIGURATION" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════════════════`n" -ForegroundColor White

# Check .env file
$envPath = "C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2\.env"
if (Test-Path $envPath) {
    Write-Pass "Configuration File: FOUND"
    $envContent = Get-Content $envPath -Raw
    if ($envContent -match "ANTHROPIC_API_KEY") {
        Write-Pass "API Key in .env: CONFIGURED"
        $script:passCount += 2
    } else {
        Write-Warn "API Key in .env: NOT FOUND"
        $script:warnCount++
    }
} else {
    Write-Warn "Configuration File: NOT FOUND"
    Write-Info "Create .env file with ANTHROPIC_API_KEY"
    $script:warnCount++
}

# Check environment variable
if ($env:ANTHROPIC_API_KEY) {
    Write-Pass "API Key in Environment: SET"
    $script:passCount++
} else {
    Write-Warn "API Key in Environment: NOT SET"
    Write-Info "Set with: `$env:ANTHROPIC_API_KEY = 'your-key'"
    $script:warnCount++
}

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════

Write-Host "`n╔══════════════════════════════════════════════════════════════════════╗" -ForegroundColor White
Write-Host "║                         STATUS SUMMARY                               ║" -ForegroundColor White
Write-Host "╚══════════════════════════════════════════════════════════════════════╝`n" -ForegroundColor White

Write-Host "✅ Passed:  " -NoNewline -ForegroundColor Green
Write-Host $script:passCount

Write-Host "⚠️  Warnings: " -NoNewline -ForegroundColor Yellow
Write-Host $script:warnCount

Write-Host "❌ Failed:  " -NoNewline -ForegroundColor Red
Write-Host $script:failCount

$totalChecks = $script:passCount + $script:warnCount + $script:failCount
$successRate = [math]::Round(($script:passCount / $totalChecks) * 100, 1)

Write-Host "`nSuccess Rate: " -NoNewline
if ($successRate -ge 80) {
    Write-Host "$successRate% 🎯" -ForegroundColor Green
} elseif ($successRate -ge 60) {
    Write-Host "$successRate% 📊" -ForegroundColor Yellow
} else {
    Write-Host "$successRate% 🔧" -ForegroundColor Red
}

# Overall recommendation
Write-Host "`n─────────────────────────────────────────────────────────────────────────`n"

if ($script:failCount -eq 0 -and $script:warnCount -le 2) {
    Write-Host "🎉 " -NoNewline -ForegroundColor Green
    Write-Host "SYSTEM STATUS: EXCELLENT" -ForegroundColor Green
    Write-Host "Your trading bot is fully operational!`n"
    Write-Info "Platform URL: http://127.0.0.1:9101/ui/platform.html"
} elseif ($script:failCount -le 2) {
    Write-Host "✅ " -NoNewline -ForegroundColor Yellow
    Write-Host "SYSTEM STATUS: GOOD" -ForegroundColor Yellow
    Write-Host "Most components are working. Address warnings to improve.`n"
} else {
    Write-Host "⚠️  " -NoNewline -ForegroundColor Red
    Write-Host "SYSTEM STATUS: NEEDS ATTENTION" -ForegroundColor Red
    Write-Host "Several critical issues detected. Review failed checks above.`n"
}

Write-Host "─────────────────────────────────────────────────────────────────────────`n"

# Quick fixes
if ($script:failCount -gt 0 -or $script:warnCount -gt 2) {
    Write-Host "🔧 Quick Fixes:" -ForegroundColor Cyan
    
    if (-not $serverRunning) {
        Write-Host "  • Start server: python dashboard_api.py"
    }
    
    if ($health -and -not $health.ibkr_connected) {
        Write-Host "  • Connect IBKR: Use /api/ibkr/connect endpoint"
    }
    
    if ($health -and -not $health.claude_available) {
        Write-Host "  • Set API key: Check .env file or environment variable"
    }
    
    if ($health -and $health.active_subscriptions -eq 0) {
        Write-Host "  • Subscribe symbols: Use /api/subscribe endpoint"
    }
    
    Write-Host ""
}

# Useful links
Write-Host "📚 Quick Links:" -ForegroundColor Cyan
Write-Host "  • Platform: http://127.0.0.1:9101/ui/platform.html"
Write-Host "  • API Docs: http://127.0.0.1:9101/docs"
Write-Host "  • Health: http://127.0.0.1:9101/health"
Write-Host ""

Write-Host "Run this script anytime to check system status!" -ForegroundColor Gray
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
