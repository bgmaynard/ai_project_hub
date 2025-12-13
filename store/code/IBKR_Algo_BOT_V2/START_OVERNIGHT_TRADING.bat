@echo off
title AI Trading Bot - Overnight Mode
color 0A

echo.
echo ================================================================
echo     AI TRADING BOT - OVERNIGHT MODE
echo     Pre-Market Scanning + 4 AM Trading Start
echo ================================================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

:: Set working directory
cd /d "C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2"

echo [1/3] Starting Dashboard API Server (Port 9100)...
start /B python alpaca_dashboard_api.py
timeout /t 5 /nobreak >nul

echo [2/3] Starting Pre-Market Preparation Service...
start /B python premarket_prep.py
timeout /t 3 /nobreak >nul

echo [3/3] Starting AI Auto-Trader (Extended Hours)...
echo.
echo ================================================================
echo     OVERNIGHT TRADING MODE ACTIVE
echo ================================================================
echo.
echo   Dashboard:   http://localhost:9100/ui/complete_platform.html
echo   API:         http://localhost:9100/health
echo.
echo   Schedule (EST):
echo   - 12:00 AM - 3:30 AM : Scanning + AI Preparation
echo   - 4:00 AM  - 9:30 AM : Pre-Market Trading (LIMIT orders)
echo   - 9:30 AM  - 4:00 PM : Regular Trading (LIMIT orders)
echo   - 4:00 PM  - 8:00 PM : After-Hours Trading (LIMIT orders)
echo.
echo   Warrior Trading Criteria:
echo   - Gap Up >= 4%%
echo   - Relative Volume >= 2x
echo   - Float < 20M shares
echo   - Price $2 - $20
echo.
echo   Press Ctrl+C to stop all services
echo ================================================================
echo.

:: Start the AI trader in foreground
python alpaca_ai_trader.py

pause
