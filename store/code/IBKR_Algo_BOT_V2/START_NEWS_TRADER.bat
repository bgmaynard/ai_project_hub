@echo off
title News Triggered Trader - Benzinga Fast
color 0E

echo ============================================================
echo   NEWS TRIGGERED TRADER
echo   Benzinga RSS + Alpaca WebSocket
echo ============================================================
echo.

cd /d C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2

echo [1/2] Checking API Server...
curl -s http://localhost:9100/api/health >nul 2>&1
if %errorlevel% neq 0 (
    echo     API Server not running. Starting it...
    start "API Server" cmd /k "python morpheus_trading_api.py"
    timeout /t 3 /nobreak > nul
)

echo [2/2] Starting News Triggered Trader...
start "News Trader" cmd /k "python news_triggered_trader.py"

echo.
echo ============================================================
echo   NEWS TRADER STARTED
echo ============================================================
echo.
echo   Features:
echo     - Benzinga RSS polling (every 2 sec)
echo     - Auto-buys on bullish catalysts
echo     - 2%% stop loss, 5%% take profit
echo     - Max 5 min hold time
echo.
echo   Press any key to open dashboard...
pause > nul

start http://localhost:9100/ui/complete_platform.html
