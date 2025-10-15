@echo off
echo Starting AI Trading Bot...
call C:\IBKR_Algo_BOT\venv\Scripts\activate.bat
python ibkr_trading_backend.py
pause
