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
echo [2/8] Starting Polygon WebSocket stream...
curl -s -X POST "http://localhost:9100/api/polygon/stream/start"
echo       Waiting 3 seconds for connection...
timeout /t 3 /nobreak > nul

echo.
echo [3/8] Subscribing core symbols to Polygon...
REM Subscribe core symbols for data feed - SPY first to establish connection
curl -s -X POST "http://localhost:9100/api/polygon/stream/subscribe" -H "Content-Type: application/json" -d "{\"symbol\":\"SPY\",\"data_type\":\"trades\"}"
curl -s -X POST "http://localhost:9100/api/polygon/stream/subscribe" -H "Content-Type: application/json" -d "{\"symbol\":\"QQQ\",\"data_type\":\"trades\"}"
curl -s -X POST "http://localhost:9100/api/polygon/stream/subscribe" -H "Content-Type: application/json" -d "{\"symbol\":\"AAPL\",\"data_type\":\"trades\"}"
curl -s -X POST "http://localhost:9100/api/polygon/stream/subscribe" -H "Content-Type: application/json" -d "{\"symbol\":\"TSLA\",\"data_type\":\"trades\"}"
curl -s -X POST "http://localhost:9100/api/polygon/stream/subscribe" -H "Content-Type: application/json" -d "{\"symbol\":\"NVDA\",\"data_type\":\"trades\"}"
echo       Core symbols subscribed - OK

echo.
echo [4/9] Running pre-market scanner...
echo       Building fresh daily watchlist...
echo.

REM Run the new pre-market scanner
python -c "import asyncio; from ai.premarket_scanner import run_4am_scan; asyncio.run(run_4am_scan())"

echo.
echo [5/9] Running FinViz Elite scanner...
REM Scan FinViz for additional momentum plays and add to watchlist
python -c "import asyncio; from ai.finviz_momentum_scanner import get_finviz_scanner; scanner = get_finviz_scanner(); asyncio.run(scanner.sync_to_scalper_watchlist(min_score=15.0, max_add=5))"
echo       FinViz scan complete

echo.
echo [6/9] Subscribing scanned symbols to Polygon...
REM Use Python to reliably subscribe all worklist symbols to Polygon
python -c "import requests; import json; worklist = requests.get('http://localhost:9100/api/worklist').json(); symbols = [item.get('symbol') for item in worklist if item.get('symbol')]; [requests.post('http://localhost:9100/api/polygon/stream/subscribe', json={'symbol': s, 'data_type': 'trades'}) for s in symbols[:20]]; print(f'       Subscribed {min(len(symbols), 20)} symbols to Polygon')"

echo.
echo [7/9] Starting news monitor with symbols...
REM Start news monitor - will sync symbols from worklist
curl -s -X POST "http://localhost:9100/api/scanner/premarket/news-monitor/start"
REM Also start general news monitor with worklist symbols
python -c "import requests; worklist = requests.get('http://localhost:9100/api/worklist').json(); symbols = [item.get('symbol') for item in worklist if item.get('symbol')][:15]; requests.post('http://localhost:9100/api/news/monitor/start', json={'symbols': symbols}) if symbols else None; print(f'       News monitor started with {len(symbols)} symbols')"

echo.
echo [8/10] Starting FinViz auto-scanner (scans every 60s, auto-adds to worklist)...
curl -s -X POST "http://localhost:9100/api/scanner/finviz/start?interval=60&min_change=10&auto_add=true"
echo.

echo.
echo [9/10] Starting scalper and news pipeline...
curl -s -X POST "http://localhost:9100/api/scanner/scalper/start"
curl -s -X POST "http://localhost:9100/api/news-pipeline/start"
curl -s -X POST "http://localhost:9100/api/scanner/news-trader/start?paper_mode=true"

echo.
echo [10/10] Verifying all systems...
echo.
echo --- POLYGON STREAM STATUS ---
curl -s http://localhost:9100/api/polygon/stream/status
echo.
echo.
echo --- SCALPER STATUS ---
curl -s http://localhost:9100/api/scanner/scalper/status
echo.
echo.
echo --- NEWS MONITOR STATUS ---
curl -s http://localhost:9100/api/news/info
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
echo   Trading UI:    http://localhost:9100/trading-new
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
echo Press any key to open trading dashboard...
pause > nul
start http://localhost:9100/trading-new
