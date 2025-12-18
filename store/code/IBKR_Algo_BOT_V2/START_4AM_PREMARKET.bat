@echo off
REM ============================================================
REM START 4AM PRE-MARKET AUTO-TRADING (Paper Mode)
REM Runs automatically at 4AM EST via scheduled task
REM ============================================================

echo.
echo ============================================================
echo   MORPHEUS TRADING BOT - PRE-MARKET AUTO-TRADING
echo   Starting at 4:00 AM EST
echo ============================================================
echo.

REM Log start time
echo [%date% %time%] Starting pre-market auto-trading >> "%~dp0logs\premarket_schedule.log"

REM Wait for server to be ready (in case it just started)
timeout /t 10 /nobreak > nul

REM Start pre-market paper trading via API
echo Starting pre-market paper trading...
curl -X POST "http://localhost:9100/api/premarket/start?paper_mode=true" -H "Content-Type: application/json"

echo.
echo [%date% %time%] Pre-market trading started >> "%~dp0logs\premarket_schedule.log"
echo.
echo Pre-market auto-trading is now running!
echo.
echo Check status:  curl http://localhost:9100/api/premarket/status
echo View trades:   curl http://localhost:9100/api/premarket/trades
echo Full report:   curl http://localhost:9100/api/premarket/report
echo.
