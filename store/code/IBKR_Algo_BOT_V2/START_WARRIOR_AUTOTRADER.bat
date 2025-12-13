@echo off
title Warrior Auto-Trader - LIVE TRADING
color 0E

echo ============================================
echo   WARRIOR TRADING AUTO-TRADER
echo   4AM Pre-Market Momentum Strategy
echo ============================================
echo.
echo WARNING: This will execute REAL TRADES!
echo Paper trading account is configured.
echo.
echo Risk Controls:
echo   - Max Position: $2,000
echo   - Max Daily Loss: $500
echo   - Stop Loss: 3%
echo   - Take Profit: 6%
echo.
echo Press any key to START or Ctrl+C to cancel...
pause >nul

cd /d C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
python warrior_autotrader.py 2>&1 | tee warrior_autotrader_output.log
