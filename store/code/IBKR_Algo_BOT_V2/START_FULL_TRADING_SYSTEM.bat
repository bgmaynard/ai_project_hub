@echo off
title Alpaca Full Trading System - 4AM Pre-Market
color 0A

echo ============================================
echo   ALPACA FULL TRADING SYSTEM
echo   Pre-Market Session Startup
echo ============================================
echo.
echo   Components:
echo     1. API Server (morpheus_trading_api.py)
echo     2. Position Guardian (position_guardian.py)
echo     3. Pre-Market Scanner
echo     4. Dashboard UI
echo.
echo ============================================
echo.

cd /d C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2

REM Check if server is already running
curl -s http://localhost:9100/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] API Server already running
) else (
    echo [1/4] Starting API Server...
    start "API Server" cmd /k "python morpheus_trading_api.py"
    timeout /t 8 /nobreak >nul
)

REM Start Position Guardian
echo [2/4] Starting Position Guardian...
start "Position Guardian" cmd /k "python position_guardian.py"
timeout /t 2 /nobreak >nul

REM Check connections
echo.
echo [3/4] Checking System Status...
echo.

echo --- Account Status ---
curl -s http://localhost:9100/api/alpaca/account 2>nul | findstr "portfolio_value buying_power"
echo.

echo --- Current Positions ---
curl -s http://localhost:9100/api/alpaca/positions 2>nul
echo.

echo --- Circuit Breaker ---
curl -s http://localhost:9100/api/alpaca/ai/circuit-breaker/status 2>nul | findstr "level can_trade"
echo.

echo [4/4] Opening Dashboard...
timeout /t 2 /nobreak >nul
start http://localhost:9100

echo.
echo ============================================
echo   TRADING SYSTEM READY
echo ============================================
echo.
echo   Dashboard: http://localhost:9100
echo   API Health: http://localhost:9100/health
echo.
echo   Quick Actions:
echo     - Scanner Tab: Run Warrior/Gapper presets
echo     - AI Tab: Check predictions
echo     - Positions: Monitored by Position Guardian
echo.
echo   Protection Active:
echo     - 1.5%% trailing stop from high
echo     - 1.5%% hard stop from entry
echo     - 3%% zombie killer
echo     - $0.03 spread limit
echo.
echo ============================================
echo.
echo Press any key to exit (services keep running)...
pause >nul
