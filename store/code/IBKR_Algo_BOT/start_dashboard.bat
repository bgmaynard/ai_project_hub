@echo off
echo Starting Trading Dashboard...
echo.
echo Starting Backend...
start cmd /k "python dashboard_api.py"
timeout /t 3 /nobreak >nul
echo.
echo Starting Frontend...
start cmd /k "cd frontend && npm run dev"
echo.
echo Dashboard will open at http://localhost:3000
echo Backend API at http://localhost:5000
echo.
echo Press any key to stop all services...
pause >nul
taskkill /F /FI "WINDOWTITLE eq Trading Dashboard*"
