@echo off
:: ============================================================================
:: AI TRADING PLATFORM - FULL STARTUP
:: ============================================================================
:: Double-click this file to start the complete trading platform
:: ============================================================================

title AI Trading Platform - Starting...
setlocal enabledelayedexpansion

:: Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Colors via echo
echo.
echo ================================================================
echo            AI TRADING PLATFORM v2.1 - STARTUP
echo         Schwab Data + Alpaca Trading + Claude AI
echo ================================================================
echo.

:: ============================================================================
:: STEP 1: Check Environment
:: ============================================================================
echo [1/8] Checking environment...

if not exist ".env" (
    echo       [X] .env file not found!
    echo       Please create .env file with your API keys
    pause
    exit /b 1
)
echo       [OK] .env file found

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo       [X] Python not found!
    pause
    exit /b 1
)
echo       [OK] Python found

:: ============================================================================
:: STEP 2: Kill existing processes on port 9100
:: ============================================================================
echo.
echo [2/8] Checking for existing processes...

for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":9100" ^| findstr "LISTENING"') do (
    echo       Stopping existing process on port 9100 (PID: %%a)
    taskkill /F /PID %%a >nul 2>&1
    timeout /t 2 /nobreak >nul
)
echo       [OK] Port 9100 available

:: ============================================================================
:: STEP 3: Start Dashboard API Server
:: ============================================================================
echo.
echo [3/8] Starting Dashboard API Server...
echo       Server will run on http://localhost:9100

start "Alpaca Dashboard API" /min cmd /c "cd /d %SCRIPT_DIR% && python alpaca_dashboard_api.py"

echo       Waiting for server to start...
timeout /t 5 /nobreak >nul

:: Check if server is running
curl -s http://localhost:9100/health >nul 2>&1
if errorlevel 1 (
    echo       [!] Server may still be starting...
    timeout /t 5 /nobreak >nul
)
echo       [OK] Dashboard API started

:: ============================================================================
:: STEP 4: Start Position Guardian
:: ============================================================================
echo.
echo [4/8] Starting Position Guardian...

start "Position Guardian" /min cmd /c "cd /d %SCRIPT_DIR% && python position_guardian.py"
echo       [OK] Position Guardian started (trailing stops protection)

:: ============================================================================
:: STEP 5: Start News Triggered Evaluator
:: ============================================================================
echo.
echo [5/8] Starting News Triggered Evaluator...

start "News Evaluator" /min cmd /c "cd /d %SCRIPT_DIR% && python news_triggered_evaluator.py"
echo       [OK] News Evaluator started (auto-watchlist)

:: Wait for services to initialize
timeout /t 3 /nobreak >nul

:: ============================================================================
:: STEP 6: Start News WebSocket via API
:: ============================================================================
echo.
echo [6/8] Starting News WebSocket...

curl -s -X POST "http://localhost:9100/api/alpaca/news-stream/start" >nul 2>&1
if errorlevel 1 (
    echo       [!] News WebSocket may have failed
) else (
    echo       [OK] News WebSocket connected
)

:: ============================================================================
:: STEP 7: Start News Monitor (Benzinga)
:: ============================================================================
echo.
echo [7/8] Starting Benzinga News Monitor...

curl -s -X POST "http://localhost:9100/api/alpaca/news/start" >nul 2>&1
if errorlevel 1 (
    echo       [!] Benzinga Monitor may have failed
) else (
    echo       [OK] Benzinga Monitor active
)

:: ============================================================================
:: STEP 8: Open Dashboard in Browser
:: ============================================================================
echo.
echo [8/8] Opening dashboard...

timeout /t 2 /nobreak >nul
start http://127.0.0.1:9100/dashboard

echo       [OK] Dashboard opened

:: ============================================================================
:: FINAL STATUS
:: ============================================================================
echo.
echo ================================================================
echo              PLATFORM STARTED SUCCESSFULLY
echo ================================================================
echo.
echo   Services Running:
echo     - Dashboard API      : http://127.0.0.1:9100/dashboard
echo     - Position Guardian  : Active (trailing stops)
echo     - News Evaluator     : Active (auto-watchlist)
echo     - News WebSocket     : Active (instant alerts)
echo     - Benzinga Monitor   : Active (news polling)
echo.
echo   Data Flow:
echo     - Market Data: Schwab
echo     - Trade Execution: Alpaca (Paper)
echo     - AI Analysis: Claude
echo.
echo   To Stop: Run STOP_TRADING_PLATFORM.bat
echo            or close the server windows
echo.
echo ================================================================
echo.
echo   Press any key to close this window.
echo   (Servers will continue running in background)
echo.
pause >nul
