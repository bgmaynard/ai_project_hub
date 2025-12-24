@echo off
REM ============================================================
REM START 4AM PRE-MARKET AUTO-TRADING (Paper Mode)
REM Runs automatically at 4AM EST via scheduled task
REM
REM What this does:
REM   1. Ensures server is running
REM   2. Clears previous day watchlist
REM   3. Scans for pre-market movers (gaps, volume, news)
REM   4. Checks after-hours continuations
REM   5. Logs all breaking news with timestamps
REM   6. Builds fresh daily watchlist
REM   7. Starts HFT scalper in PAPER mode
REM   8. Starts news monitor for live updates
REM ============================================================

title Morpheus 4AM Pre-Market Scanner

echo.
echo ============================================================
echo   MORPHEUS TRADING BOT - 4AM PRE-MARKET SCANNER
echo   %date% %time%
echo ============================================================
echo.

cd /d C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2

REM Create logs directory if not exists
if not exist logs mkdir logs
if not exist store\scanner mkdir store\scanner

REM Log start time
echo [%date% %time%] Starting 4AM pre-market scan >> logs\premarket_schedule.log

REM Check if server is running
echo [1/6] Checking server status...
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
    echo [%date% %time%] ERROR: Server failed to start >> logs\premarket_schedule.log
    pause
    exit /b 1
)

echo.
echo [2/6] Running pre-market scanner...
echo       Building fresh daily watchlist...
echo.

REM Run the new pre-market scanner
python -c "import asyncio; from ai.premarket_scanner import run_4am_scan; asyncio.run(run_4am_scan())"

echo.
echo [3/6] Starting news monitor...
curl -s -X POST "http://localhost:9100/api/scanner/premarket/news-monitor/start"

echo.
echo [4/6] Starting scalper (paper mode)...
curl -s -X POST "http://localhost:9100/api/scanner/scalper/start"

echo.
echo [5/6] Starting news pipeline...
curl -s -X POST "http://localhost:9100/api/news-pipeline/start"
curl -s -X POST "http://localhost:9100/api/scanner/news-trader/start?paper_mode=true"

echo.
echo [6/6] Verifying all systems...
echo.
echo --- SCALPER STATUS ---
curl -s http://localhost:9100/api/scanner/scalper/status
echo.
echo.
echo --- PRE-MARKET STATUS ---
curl -s http://localhost:9100/api/scanner/premarket/status
echo.

REM Log completion
echo [%date% %time%] Pre-market scan complete >> logs\premarket_schedule.log

echo.
echo ============================================================
echo   4AM PRE-MARKET SCAN COMPLETE
echo ============================================================
echo.
echo   Dashboard:     http://localhost:9100/dashboard
echo   News Log:      curl http://localhost:9100/api/news-log
echo   Watchlist:     curl http://localhost:9100/api/scanner/premarket/watchlist
echo   Scalper:       curl http://localhost:9100/api/scanner/scalper/status
echo.
echo   All breaking news is being logged with timestamps.
echo   Refresh dashboard to see today's movers.
echo.
echo ============================================================
echo.

REM Keep window open to see output
echo Press any key to open dashboard...
pause > nul
start http://localhost:9100/dashboard
