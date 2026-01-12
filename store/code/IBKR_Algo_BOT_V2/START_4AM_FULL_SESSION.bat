@echo off
REM ============================================
REM MORPHEUS TRADING BOT - 4AM FULL SESSION
REM ============================================
REM Schedule: 4:00 AM ET start
REM
REM TRADING WINDOWS:
REM   4:00-7:00 AM  - Market analysis, build worklist
REM   7:00-9:30 AM  - Pre-market trading (HFT Scalper)
REM   9:30-9:40 AM  - WAIT (open chaos)
REM   9:40-11:00 AM - ATS + 9 EMA Sniper Strategy
REM   11:00+ AM     - Generate reports
REM ============================================

cd /d C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2

echo ============================================
echo MORPHEUS TRADING BOT - 4AM FULL SESSION
echo Started: %date% %time%
echo ============================================

REM Kill any existing Python processes
taskkill /F /IM python.exe 2>nul

REM Wait 2 seconds
timeout /t 2 /nobreak >nul

REM Start the main server
echo Starting Morpheus Trading API...
start /B python morpheus_trading_api.py

REM Wait for server to initialize
echo Waiting for server to initialize...
timeout /t 15 /nobreak >nul

REM Check server status
curl -s http://localhost:9100/api/status
echo.

REM Start pre-market scanner and build worklist
echo.
echo Starting pre-market scanner...
curl -s -X POST "http://localhost:9100/api/scanner/premarket/scan"
echo.

REM Start news monitor
echo Starting news monitor...
curl -s -X POST "http://localhost:9100/api/scanner/premarket/news-monitor/start"
echo.

REM Start HFT Scalper in paper mode for pre-market (7:00-9:30)
echo Starting HFT Scalper (paper mode)...
curl -s -X POST "http://localhost:9100/api/scanner/scalper/start"
echo.

REM Enable the ATS + 9 EMA Sniper Strategy (will auto-activate at 9:40)
echo Enabling ATS + 9 EMA Sniper Strategy...
curl -s -X POST "http://localhost:9100/api/strategy/sniper/enable"
echo.

REM Start continuous discovery
echo Starting continuous discovery...
curl -s -X POST "http://localhost:9100/api/task-queue/discovery/start?poll_interval_seconds=300"
echo.

echo.
echo ============================================
echo TRADING SCHEDULE ACTIVE
echo ============================================
echo   4:00-7:00 AM  - Building worklist
echo   7:00-9:30 AM  - Pre-market scalping (HFT)
echo   9:30-9:40 AM  - WAITING (open chaos)
echo   9:40-11:00 AM - ATS + 9 EMA Sniper
echo ============================================
echo.
echo Server running. Check dashboards:
echo   http://localhost:9100/trading-new
echo   http://localhost:9100/orchestrator
echo.

REM Keep window open
pause
