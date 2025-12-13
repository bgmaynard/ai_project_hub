@echo off
:: ============================================================================
:: STOP AI TRADING PLATFORM
:: ============================================================================
:: Stops all running instances of the trading platform
:: ============================================================================

echo.
echo ================================================================
echo            STOPPING AI TRADING PLATFORM
echo ================================================================
echo.

:: Kill any process running on port 9100 (Dashboard API)
echo [1/4] Stopping Dashboard API Server...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :9100 ^| findstr LISTENING') do (
    echo       Stopping process %%a on port 9100...
    taskkill /F /PID %%a >nul 2>&1
)
echo       [OK] Dashboard API stopped

:: Kill Position Guardian
echo.
echo [2/4] Stopping Position Guardian...
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%position_guardian%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo       [OK] Position Guardian stopped

:: Kill News Evaluator
echo.
echo [3/4] Stopping News Evaluator...
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%news_triggered_evaluator%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo       [OK] News Evaluator stopped

:: Kill any remaining Python processes related to our app
echo.
echo [4/4] Cleaning up remaining processes...
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%morpheus_trading_api%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%warrior_autotrader%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo       [OK] Cleanup complete

echo.
echo ================================================================
echo            PLATFORM STOPPED SUCCESSFULLY
echo ================================================================
echo.
echo All services have been stopped.
echo Run START_TRADING_PLATFORM.bat to restart.
echo.
pause
