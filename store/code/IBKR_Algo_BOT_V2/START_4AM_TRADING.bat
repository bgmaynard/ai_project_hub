@echo off
title Alpaca Trading Platform - 4AM Pre-Market Session
color 0A

echo ============================================
echo   ALPACA TRADING PLATFORM - 4AM STARTUP
echo   Pre-Market Trading Session
echo ============================================
echo.

REM Check if server is already running
curl -s http://localhost:9100/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Server already running on port 9100
    goto :check_brain
)

echo [STARTING] Trading Platform Server...
cd /d C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
start /B python alpaca_dashboard_api.py
timeout /t 5 /nobreak >nul

:check_brain
echo.
echo [CHECKING] System Status...

REM Health Check
curl -s http://localhost:9100/health
echo.

REM Account Status
echo.
echo [ACCOUNT] Checking Alpaca Connection...
curl -s http://localhost:9100/api/alpaca/account | findstr buying_power portfolio_value
echo.

REM Background Brain Status
echo.
echo [BRAIN] Checking Background Brain...
curl -s http://localhost:9100/api/alpaca/ai/brain/status | findstr running
echo.

REM Circuit Breaker
echo.
echo [SAFETY] Circuit Breaker Status...
curl -s http://localhost:9100/api/alpaca/ai/circuit-breaker/status | findstr level can_trade
echo.

REM Positions
echo.
echo [POSITIONS] Current Holdings...
curl -s http://localhost:9100/api/alpaca/positions
echo.

echo.
echo ============================================
echo   READY FOR 4AM PRE-MARKET TRADING
echo ============================================
echo.
echo Dashboard: http://localhost:9100
echo.
echo Quick Actions:
echo   - Run Warrior Scanner: Scanner Tab ^> Warrior Trading Preset
echo   - Run Gapper Scanner: Scanner Tab ^> Pre-Market Gappers
echo   - Check AI Brain: AI Tab ^> Brain Status
echo.
echo Press any key to open dashboard...
pause >nul
start http://localhost:9100
