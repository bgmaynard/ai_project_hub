@echo off
REM ============================================
REM  MORPHEUS TRADING BOT - PRE-MARKET STARTUP
REM  One-click startup for pre-market trading
REM ============================================

echo.
echo ============================================
echo   MORPHEUS PRE-MARKET STARTUP
echo ============================================
echo.

cd /d C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2

REM Kill any existing Python processes
echo [1/7] Stopping existing processes...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

REM Start the server
echo [2/7] Starting Morpheus server...
start /min "Morpheus Server" python morpheus_trading_api.py

REM Wait for server to be ready
echo [3/7] Waiting for server startup...
:WAIT_LOOP
timeout /t 2 /nobreak >nul
curl -s http://localhost:9100/api/status >nul 2>&1
if errorlevel 1 goto WAIT_LOOP
echo        Server is ready!

REM Run connectivity self-test
echo [4/7] Running connectivity self-test...
curl -s -X POST http://localhost:9100/api/validation/connectivity/self-test >nul

REM Start all services
echo [5/7] Starting trading services...
curl -s -X POST "http://localhost:9100/api/scanner/scalper/start" >nul
curl -s -X POST "http://localhost:9100/api/scanner/news-trader/start?paper_mode=true" >nul 2>&1

REM Run pre-market scans
echo [6/7] Running pre-market scans...
curl -s -X POST http://localhost:9100/api/scanner/premarket/scan >nul
curl -s -X POST http://localhost:9100/api/scanner/hod/scan-finviz >nul
curl -s -X POST http://localhost:9100/api/scanner/hod/enrich >nul 2>&1

REM Open dashboards
echo [7/7] Opening dashboards...
start http://localhost:9100/trading-new
timeout /t 1 /nobreak >nul
start http://localhost:9100/ai-control-center

echo.
echo ============================================
echo   STARTUP COMPLETE!
echo ============================================
echo.
echo   Trading Dashboard: http://localhost:9100/trading-new
echo   AI Control Center: http://localhost:9100/ai-control-center
echo.
echo   Press any key to view system status...
pause >nul

REM Show final status
curl -s http://localhost:9100/api/validation/safe/posture
echo.
curl -s http://localhost:9100/api/scanner/scalper/status | findstr "is_running paper_mode watchlist_count"
echo.
pause
