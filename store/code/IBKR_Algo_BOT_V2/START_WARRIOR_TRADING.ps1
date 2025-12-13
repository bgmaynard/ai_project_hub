# Warrior Trading System Startup Script
# Launches the complete trading system with all components

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   WARRIOR TRADING AI SYSTEM - STARTUP" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Change to project directory
$ProjectRoot = "C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2"
Set-Location $ProjectRoot

# Check Python environment
Write-Host "[1/8] Checking Python environment..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Python OK" -ForegroundColor Green
Write-Host ""

# Check if virtual environment exists
Write-Host "[2/8] Checking virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv\Scripts\activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    & "venv\Scripts\activate.ps1"
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "! No virtual environment found - using system Python" -ForegroundColor Yellow
}
Write-Host ""

# Check required packages
Write-Host "[3/8] Checking required packages..." -ForegroundColor Yellow
$RequiredPackages = @(
    "fastapi",
    "uvicorn",
    "anthropic",
    "pandas",
    "yfinance"
)

foreach ($package in $RequiredPackages) {
    python -c "import $package" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ $package" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $package - MISSING!" -ForegroundColor Red
        Write-Host "    Installing $package..." -ForegroundColor Yellow
        pip install $package -q
    }
}
Write-Host ""

# Check IBKR TWS connection
Write-Host "[4/8] Checking IBKR TWS..." -ForegroundColor Yellow
Write-Host "Please ensure IBKR TWS/Gateway is running on:" -ForegroundColor Cyan
Write-Host "  - Paper Trading: localhost:7497" -ForegroundColor Cyan
Write-Host "  - Live Trading:  localhost:7496" -ForegroundColor Cyan

# Wait for user confirmation
Write-Host ""
$response = Read-Host "Is IBKR TWS/Gateway running? (Y/N)"
if ($response -ne "Y" -and $response -ne "y") {
    Write-Host "Please start IBKR TWS/Gateway first, then run this script again." -ForegroundColor Yellow
    exit 0
}
Write-Host "✓ IBKR TWS confirmed running" -ForegroundColor Green
Write-Host ""

# Initialize database
Write-Host "[5/8] Initializing Warrior Trading database..." -ForegroundColor Yellow
if (Test-Path "init_warrior_database.py") {
    python init_warrior_database.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Database initialized" -ForegroundColor Green
    } else {
        Write-Host "! Database initialization had issues (may already exist)" -ForegroundColor Yellow
    }
} else {
    Write-Host "! Database initialization script not found - skipping" -ForegroundColor Yellow
}
Write-Host ""

# Check Claude AI API key
Write-Host "[6/8] Checking Claude AI configuration..." -ForegroundColor Yellow
if ($env:ANTHROPIC_API_KEY) {
    Write-Host "✓ ANTHROPIC_API_KEY is set" -ForegroundColor Green
} else {
    Write-Host "! ANTHROPIC_API_KEY not set - AI features will be disabled" -ForegroundColor Yellow
    Write-Host "  To enable AI features, set: `$env:ANTHROPIC_API_KEY='your-key'" -ForegroundColor Cyan
}
Write-Host ""

# Start backend API server
Write-Host "[7/8] Starting backend API server..." -ForegroundColor Yellow
Write-Host "Starting FastAPI server on http://localhost:8000" -ForegroundColor Cyan

# Create a flag file to signal the server is starting
New-Item -Path "data" -ItemType Directory -Force | Out-Null
"running" | Out-File -FilePath "data\.server_status" -Encoding UTF8

# Start the server in a new window
$ServerPath = Join-Path $ProjectRoot "dashboard_api.py"
if (Test-Path $ServerPath) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot'; python dashboard_api.py"
    Write-Host "✓ Backend server starting in new window..." -ForegroundColor Green
} else {
    Write-Host "ERROR: dashboard_api.py not found!" -ForegroundColor Red
    exit 1
}

# Wait for server to start
Write-Host "Waiting for server to initialize (10 seconds)..." -ForegroundColor Cyan
Start-Sleep -Seconds 10
Write-Host ""

# Start React frontend
Write-Host "[8/8] Starting React frontend..." -ForegroundColor Yellow
$FrontendPath = Join-Path $ProjectRoot "ui\ai-control-center"

if (Test-Path $FrontendPath) {
    Write-Host "Starting React development server..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$FrontendPath'; npm start"
    Write-Host "✓ Frontend starting in new window..." -ForegroundColor Green
} else {
    Write-Host "! Frontend not found at expected location" -ForegroundColor Yellow
    Write-Host "  You can start it manually from: $FrontendPath" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   WARRIOR TRADING SYSTEM - READY!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend API:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Docs:     http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Frontend UI:  http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Quick Test URLs:" -ForegroundColor Yellow
Write-Host "  - System Status:    http://localhost:8000/api/warrior/status" -ForegroundColor White
Write-Host "  - Pre-Market Scan:  http://localhost:8000/api/warrior/scan/premarket" -ForegroundColor White
Write-Host "  - Risk Status:      http://localhost:8000/api/warrior/risk/status" -ForegroundColor White
Write-Host "  - AI Health:        http://localhost:8000/api/warrior/ai/health/status" -ForegroundColor White
Write-Host ""
Write-Host "Dashboard:            http://localhost:3000/warrior" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop this script (servers will keep running)" -ForegroundColor Gray
Write-Host ""

# Keep script running to show logs
Write-Host "Monitoring system..." -ForegroundColor Cyan
Write-Host "Check the other windows for backend and frontend logs" -ForegroundColor Gray
Write-Host ""

# Wait for user to press Ctrl+C
try {
    while ($true) {
        Start-Sleep -Seconds 5
        # Could add health checks here
    }
} finally {
    Write-Host "`nScript terminated. Servers are still running in other windows." -ForegroundColor Yellow
}
