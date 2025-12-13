@echo off
title HFT Trading Platform - Parallel V4
color 0A

echo ============================================================
echo   HFT TRADING PLATFORM - PARALLEL V4
echo   24-Core Optimized | 2:1 Profit/Loss Ratio
echo ============================================================
echo.

cd /d C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2

echo [1/4] Starting API Server...
start "API Server" cmd /k "python alpaca_dashboard_api.py"
timeout /t 3 /nobreak > nul

echo [2/4] Starting Position Guardian...
start "Position Guardian" cmd /k "python position_guardian.py"
timeout /t 2 /nobreak > nul

echo [3/4] Starting HFT Scalper V4 (Parallel)...
start "HFT Scalper V4" cmd /k "python hft_scalper_v4_parallel.py"
timeout /t 2 /nobreak > nul

echo [4/4] Running Status Check...
python check_status.py

echo.
echo ============================================================
echo   ALL SYSTEMS RUNNING
echo ============================================================
echo.
echo   API Server:        http://localhost:9100
echo   Dashboard:         http://localhost:9100/ui/complete_platform.html
echo.
echo   Windows open:
echo     - API Server
echo     - Position Guardian (trailing stops)
echo     - HFT Scalper V4 (parallel execution)
echo.
echo   Press any key to open dashboard in browser...
pause > nul

start http://localhost:9100/ui/complete_platform.html
