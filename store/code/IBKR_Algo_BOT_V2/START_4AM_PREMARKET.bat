@echo off
REM ============================================================
REM START 4AM PRE-MARKET AUTO-TRADING (Paper Mode)
REM Runs automatically at 4AM EST via scheduled task
REM
REM What this does:
REM   1. Ensures server is running
REM   2. Scans Yahoo/Schwab for momentum stocks
REM   3. Filters for scalp criteria ($1-$20, 5%+ gap, 500K+ vol)
REM   4. Adds top 5 picks to scalper watchlist
REM   5. Starts HFT scalper in PAPER mode
REM ============================================================

title Morpheus 4AM Pre-Market Scanner

echo.
echo ============================================================
echo   MORPHEUS TRADING BOT - 4AM PRE-MARKET AUTO-TRADING
echo   %date% %time%
echo ============================================================
echo.

cd /d C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2

REM Create logs directory if not exists
if not exist logs mkdir logs

REM Log start time
echo [%date% %time%] Starting pre-market auto-trading >> logs\premarket_schedule.log

REM Check if server is running
echo [1/4] Checking server status...
curl -s http://localhost:9100/api/status >nul 2>&1
if %errorlevel% neq 0 (
    echo       Server not running - starting now...
    start /min cmd /c "python morpheus_trading_api.py"
    echo       Waiting 15 seconds for server startup...
    timeout /t 15 /nobreak > nul
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
echo [2/4] Running pre-market scanner...
echo       Scanning Yahoo Finance + Schwab for movers...
echo.

REM Run the auto scan and trade script
python auto_scan_trade.py

echo.
echo [3/4] Verifying scalper status...
curl -s http://localhost:9100/api/scanner/scalper/status

echo.
echo [4/4] Pre-market auto-trading is now ACTIVE!
echo.

REM Log completion
echo [%date% %time%] Pre-market trading started successfully >> logs\premarket_schedule.log

echo ============================================================
echo   TRADING ACTIVE (PAPER MODE)
echo ============================================================
echo.
echo   Monitor:    python monitor.py
echo   Dashboard:  http://localhost:9100/dashboard
echo   Status:     curl http://localhost:9100/api/scanner/scalper/status
echo   Trades:     curl http://localhost:9100/api/scanner/scalper/trades
echo.
echo   Stop:       curl -X POST http://localhost:9100/api/scanner/scalper/stop
echo.
echo ============================================================
echo.

REM Keep window open to see output
echo Press any key to open dashboard...
pause > nul
start http://localhost:9100/dashboard
