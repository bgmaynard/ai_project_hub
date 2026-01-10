@echo off
REM ============================================================
REM MORPHEUS TRADING BOT - STOP ALL
REM ============================================================

title Stopping Morpheus Trading Bot

echo.
echo ============================================================
echo   STOPPING MORPHEUS TRADING BOT
echo   %date% %time%
echo ============================================================
echo.

cd /d C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2

echo [1/4] Stopping Scalper...
curl -s -X POST "http://localhost:9100/api/scanner/scalper/stop" >nul 2>&1
echo       Done

echo [2/4] Stopping Scalp Assistant...
curl -s -X POST "http://localhost:9100/api/scalp/stop" >nul 2>&1
echo       Done

echo [3/4] Stopping News Trader...
curl -s -X POST "http://localhost:9100/api/scanner/news-trader/stop" >nul 2>&1
echo       Done

echo [4/4] Killing Python processes...
taskkill /F /IM python.exe >nul 2>&1
echo       Done

echo.
echo ============================================================
echo   ALL SYSTEMS STOPPED
echo ============================================================
echo.
echo [%date% %time%] All systems stopped >> logs\startup.log
pause
