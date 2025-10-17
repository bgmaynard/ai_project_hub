@echo off
setlocal
set REPO=C:\ai_project_hub
set PYTHONPATH=%REPO%\store\code
echo Launching Dashboard API on http://127.0.0.1:9101 ...
python %REPO%\store\code\IBKR_Algo_BOT\dashboard_api.py
pause