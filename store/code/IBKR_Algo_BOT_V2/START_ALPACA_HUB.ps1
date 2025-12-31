# ============================================================================
# AI TRADING HUB - ALPACA EDITION LAUNCHER
# ============================================================================

param(
    [switch]$InstallDependencies,
    [switch]$TrainModel,
    [string]$TrainSymbol = "SPY",
    [switch]$AutoTrader,
    [switch]$SkipChecks
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "  AI TRADING HUB - ALPACA EDITION" -ForegroundColor Yellow
Write-Host "  Unified AI Trading Platform with Alpaca Integration" -ForegroundColor White
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# FUNCTIONS
# ============================================================================

function Write-Status {
    param($Message, $Type = "Info")

    $timestamp = Get-Date -Format "HH:mm:ss"

    switch ($Type) {
        "Success" { Write-Host "[$timestamp] ✅ $Message" -ForegroundColor Green }
        "Error"   { Write-Host "[$timestamp] ❌ $Message" -ForegroundColor Red }
        "Warning" { Write-Host "[$timestamp] ⚠️  $Message" -ForegroundColor Yellow }
        "Info"    { Write-Host "[$timestamp] ℹ️  $Message" -ForegroundColor Cyan }
    }
}

function Test-PythonInstallation {
    Write-Status "Checking Python installation..."

    try {
        $pythonVersion = python --version 2>&1
        Write-Status "Found: $pythonVersion" "Success"
        return $true
    }
    catch {
        Write-Status "Python not found! Please install Python 3.8+" "Error"
        return $false
    }
}

function Test-EnvironmentFile {
    Write-Status "Checking .env configuration..."

    if (-not (Test-Path ".env")) {
        Write-Status ".env file not found!" "Error"
        return $false
    }

    $envContent = Get-Content ".env" -Raw

    # Check for required Alpaca keys
    if ($envContent -notmatch "ALPACA_API_KEY=.+") {
        Write-Status "ALPACA_API_KEY not configured in .env" "Error"
        return $false
    }

    if ($envContent -notmatch "ALPACA_SECRET_KEY=.+") {
        Write-Status "ALPACA_SECRET_KEY not configured in .env" "Error"
        return $false
    }

    Write-Status "Environment configuration OK" "Success"
    return $true
}

function Install-Dependencies {
    Write-Status "Installing Python dependencies..."

    try {
        python -m pip install --upgrade pip
        python -m pip install -r requirements.txt
        Write-Status "Dependencies installed successfully" "Success"
        return $true
    }
    catch {
        Write-Status "Failed to install dependencies: $_" "Error"
        return $false
    }
}

function Test-AlpacaConnection {
    Write-Status "Testing Alpaca API connection..."

    try {
        # Create a temporary test file in the current directory
        $tempFile = "._alpaca_test_temp.py"

        $testScript = @'
import os
import sys
from dotenv import load_dotenv

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# Load environment
load_dotenv('.env')

from alpaca_integration import get_alpaca_connector
connector = get_alpaca_connector()
if connector.is_connected():
    account = connector.get_account()
    print(f"Connected to Alpaca - Account: {account['account_id']}")
    print(f"Buying Power: ${account['buying_power']:,.2f}")
else:
    print("Connection failed!")
    exit(1)
'@

        # Write to temp file in current directory and execute
        Set-Content -Path $tempFile -Value $testScript
        python $tempFile
        $exitCode = $LASTEXITCODE

        # Clean up
        Remove-Item -Path $tempFile -ErrorAction SilentlyContinue

        if ($exitCode -eq 0) {
            Write-Status "Alpaca connection successful" "Success"
            return $true
        }
        else {
            Write-Status "Alpaca connection failed" "Error"
            return $false
        }
    }
    catch {
        Write-Status "Error testing Alpaca connection: $_" "Error"
        return $false
    }
}

function Start-AlpacaDashboard {
    Write-Status "Starting Alpaca Dashboard API Server..."
    Write-Host ""
    Write-Host "  Dashboard will be available at:" -ForegroundColor Yellow
    Write-Host "  📊 http://localhost:9100/trading-new" -ForegroundColor Cyan
    Write-Host "  📚 http://localhost:9100/docs (API Documentation)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Press Ctrl+C to stop the server" -ForegroundColor Gray
    Write-Host ""

    python morpheus_trading_api.py
}

function Start-AutoTrader {
    Write-Status "Starting Alpaca Auto-Trader..."
    Write-Host ""
    Write-Host "  Auto-trader will scan and execute trades automatically" -ForegroundColor Yellow
    Write-Host "  Press Ctrl+C to stop" -ForegroundColor Gray
    Write-Host ""

    python alpaca_ai_trader.py
}

function Train-AIModel {
    param($Symbol)

    Write-Status "Training AI model on $Symbol..."

    try {
        # Create a temporary training script
        $tempFile = [System.IO.Path]::GetTempFileName() + ".py"

        $trainScript = @'
from ai.alpaca_ai_predictor import AlpacaAIPredictor
import logging

logging.basicConfig(level=logging.INFO)

predictor = AlpacaAIPredictor()
print(f"\n[TRAINING] Starting model training on {SYMBOL}...")
result = predictor.train(symbol='{SYMBOL}')
print(f"\n[OK] Training complete!")
print(f"    Accuracy: {result['metrics']['accuracy']:.4f}")
print(f"    Samples: {result['samples']}")
print(f"    Model saved to: {result['model_path']}")
'@

        # Replace symbol and write to file
        $trainScript -replace '\{SYMBOL\}', $Symbol | Set-Content -Path $tempFile
        python $tempFile
        $exitCode = $LASTEXITCODE

        # Clean up
        Remove-Item -Path $tempFile -ErrorAction SilentlyContinue

        if ($exitCode -eq 0) {
            Write-Status "Model training successful" "Success"
            return $true
        }
        else {
            Write-Status "Model training failed" "Error"
            return $false
        }
    }
    catch {
        Write-Status "Error during model training: $_" "Error"
        return $false
    }
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Change to script directory
Set-Location $PSScriptRoot

# Handle install dependencies flag
if ($InstallDependencies) {
    Write-Status "Installing dependencies..." "Info"
    if (-not (Install-Dependencies)) {
        Write-Status "Dependency installation failed. Exiting." "Error"
        exit 1
    }
    Write-Host ""
    Write-Host "✅ Dependencies installed successfully!" -ForegroundColor Green
    Write-Host ""
    exit 0
}

# Run pre-flight checks (unless skipped)
if (-not $SkipChecks) {
    Write-Status "Running pre-flight checks..." "Info"
    Write-Host ""

    # Check Python
    if (-not (Test-PythonInstallation)) {
        Write-Status "Pre-flight checks failed" "Error"
        exit 1
    }

    # Check .env
    if (-not (Test-EnvironmentFile)) {
        Write-Status "Please configure your .env file with Alpaca API keys" "Error"
        exit 1
    }

    # Test Alpaca connection
    if (-not (Test-AlpacaConnection)) {
        Write-Status "Pre-flight checks failed" "Error"
        exit 1
    }

    Write-Host ""
    Write-Status "All pre-flight checks passed!" "Success"
    Write-Host ""
}

# Handle model training
if ($TrainModel) {
    Train-AIModel -Symbol $TrainSymbol
    Write-Host ""
    Write-Host "Model training complete! You can now start the dashboard." -ForegroundColor Green
    Write-Host ""
    exit 0
}

# Handle auto-trader
if ($AutoTrader) {
    Start-AutoTrader
    exit 0
}

# Default: Start dashboard
Start-AlpacaDashboard
