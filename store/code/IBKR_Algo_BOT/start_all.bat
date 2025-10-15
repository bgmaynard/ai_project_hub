@echo off
echo Starting Trading Dashboard...
echo.
start cmd /k "title Backend API && python dashboard_api.py"
timeout /t 3
start cmd /k "title Frontend UI && cd frontend && npm run dev"
echo.
echo Dashboard starting...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
