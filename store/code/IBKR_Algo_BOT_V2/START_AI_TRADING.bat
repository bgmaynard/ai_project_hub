@echo off
title AI Trading Bot - Startup
color 0A

echo.
echo ================================================================
echo     AI TRADING BOT - FULL SYSTEM STARTUP
echo ================================================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

:: Set working directory
cd /d "C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2"

echo [1/3] Starting Dashboard API Server (Port 9100)...
start /B python alpaca_dashboard_api.py
timeout /t 5 /nobreak >nul

echo [2/3] Waiting for server to initialize...
timeout /t 3 /nobreak >nul

echo [3/3] Starting AI Auto-Trader...
echo.
echo ================================================================
echo     AI TRADING BOT IS NOW RUNNING
echo ================================================================
echo.
echo   Dashboard: http://localhost:9100/ui/complete_platform.html
echo   API:       http://localhost:9100/health
echo.
echo   Press Ctrl+C to stop the bot
echo ================================================================
echo.

:: Start the AI trader in foreground so we can see output
python alpaca_ai_trader.py

pause
