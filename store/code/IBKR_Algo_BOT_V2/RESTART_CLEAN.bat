@echo off
echo ==========================================
echo CLEAN RESTART - Killing All Python Processes
echo ==========================================
echo.
echo You may see "Access Denied" for elevated processes.
echo In that case, run this as Administrator.
echo.
taskkill /F /IM python.exe /T 2>nul
echo.
echo Waiting 3 seconds for processes to terminate...
timeout /t 3 /nobreak >nul
echo.
echo Starting API Server with fresh scanner routes...
cd /d C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
start /B python alpaca_dashboard_api.py
echo.
echo Waiting for API to start...
timeout /t 5 /nobreak >nul
echo.
echo Starting Position Guardian...
start /B python position_guardian.py
echo.
echo ==========================================
echo RESTART COMPLETE
echo ==========================================
echo.
echo Test scanner: curl http://localhost:9100/api/scanner/ALPACA/status
echo.
pause
