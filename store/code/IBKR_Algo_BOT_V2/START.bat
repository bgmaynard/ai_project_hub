@echo off
REM ============================================================
REM MORPHEUS TRADING BOT - MAIN STARTUP SCRIPT
REM
REM This is the ONE batch file to start everything.
REM Can be scheduled for 3AM local (4AM ET) or run manually.
REM
REM What this does:
REM   1. Ensures server is running
REM   2. Purges previous day watchlist
REM   3. Scans for movers (gaps, volume, news)
REM   4. Builds fresh daily watchlist
REM   5. Starts HFT scalper (paper mode)
REM   6. Starts Scalp Assistant
REM   7. Starts news monitors
REM   8. Opens trading dashboard
REM ============================================================

title Morpheus Trading Bot

echo.
echo ============================================================
echo   MORPHEUS TRADING BOT
echo   %date% %time%
echo ============================================================
echo.

cd /d C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2

REM Create logs directory if not exists
if not exist logs mkdir logs

REM Log start time
echo [%date% %time%] Starting Morpheus Trading Bot >> logs\startup.log

REM Check if server is running
echo [1/8] Checking server status...
curl -s http://localhost:9100/api/status >nul 2>&1
if %errorlevel% neq 0 (
    echo       Server not running - starting now...
    start /min cmd /c "python morpheus_trading_api.py"
    echo       Waiting 20 seconds for server startup...
    timeout /t 20 /nobreak > nul
) else (
    echo       Server already running - OK
)

REM Verify server is up
curl -s http://localhost:9100/api/status >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Server failed to start!
    echo [%date% %time%] ERROR: Server failed to start >> logs\startup.log
    pause
    exit /b 1
)
echo       Server running - OK

echo.
echo [2/8] Purging previous day watchlist...
curl -s -X POST "http://localhost:9100/api/watchlist/purge" >nul
curl -s -X POST "http://localhost:9100/api/scanner/scalper/watchlist/purge" >nul
curl -s -X POST "http://localhost:9100/api/scanner/scalper/reset" >nul
echo       Main watchlist + scalper watchlist + daily stats cleared

echo.
echo [3/8] Running pre-market scanner...
python -c "import asyncio; from ai.premarket_scanner import run_4am_scan; asyncio.run(run_4am_scan())" 2>nul
echo       Scanner complete

echo.
echo [4/8] Running FinViz scanner...
python -c "import asyncio; from ai.finviz_momentum_scanner import get_finviz_scanner; scanner = get_finviz_scanner(); asyncio.run(scanner.sync_to_scalper_watchlist(min_score=15.0, max_add=5))" 2>nul
echo       FinViz scan complete

echo.
echo [5/8] Starting FinViz auto-scanner (every 60s)...
curl -s -X POST "http://localhost:9100/api/scanner/finviz/start?interval=60&min_change=10&auto_add=true" >nul
echo       Auto-scanner started

echo.
echo [6/8] Starting HFT Scalper (paper mode)...
curl -s -X POST "http://localhost:9100/api/scanner/scalper/start?paper_mode=true" >nul
echo       Scalper started

echo.
echo [7/8] Starting Scalp Assistant...
curl -s -X POST "http://localhost:9100/api/scalp/start" >nul
echo       Scalp Assistant started

echo.
echo [8/8] Starting news monitors...
curl -s -X POST "http://localhost:9100/api/scanner/premarket/news-monitor/start" >nul
curl -s -X POST "http://localhost:9100/api/scanner/news-trader/start?paper_mode=true" >nul
echo       News monitors started

echo.
echo ============================================================
echo   STARTUP COMPLETE
echo ============================================================
echo.

REM Show status
echo --- SYSTEM STATUS ---
curl -s http://localhost:9100/api/validation/safe/trading-window
echo.
echo.

echo --- SCALPER STATUS ---
curl -s http://localhost:9100/api/scanner/scalper/status | findstr /C:"is_running" /C:"paper_mode" /C:"watchlist_count"
echo.

REM Log completion
echo [%date% %time%] Startup complete >> logs\startup.log

echo.
echo   Dashboard: http://localhost:9100/trading-new
echo.
echo Press any key to open dashboard...
pause > nul
start http://localhost:9100/trading-new
