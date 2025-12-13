@echo off
:: ============================================================================
:: RESTART AI TRADING PLATFORM
:: ============================================================================
:: Gracefully stops and restarts all platform components
:: Can be scheduled via Windows Task Scheduler for nightly restarts
::
:: Usage:
::   Manual:    Double-click this file
::   Scheduled: Task Scheduler -> Run RESTART_TRADING_PLATFORM.bat
:: ============================================================================

echo.
echo ================================================================
echo         RESTARTING AI TRADING PLATFORM
echo         %date% %time%
echo ================================================================
echo.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Log file for restart history
set "LOG_FILE=%SCRIPT_DIR%logs\restart_log.txt"
if not exist "%SCRIPT_DIR%logs" mkdir "%SCRIPT_DIR%logs"

echo [%date% %time%] Restart initiated >> "%LOG_FILE%"

:: ============================================================================
:: STEP 1: Stop existing processes
:: ============================================================================
echo [1/4] Stopping existing services...

:: Kill Dashboard API (port 9100)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :9100 ^| findstr LISTENING 2^>nul') do (
    echo       Stopping Dashboard API (PID: %%a)...
    taskkill /F /PID %%a >nul 2>&1
)

:: Kill MCP Server
taskkill /F /IM alpaca-mcp-server.exe >nul 2>&1
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%alpaca-mcp%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /F /PID %%a >nul 2>&1
)

:: Kill any Python processes running our API
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%morpheus_trading_api%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo       [OK] Services stopped
echo [%date% %time%] Services stopped >> "%LOG_FILE%"

:: Wait for processes to fully terminate
echo.
echo [2/4] Waiting for cleanup...
timeout /t 5 /nobreak >nul
echo       [OK] Cleanup complete

:: ============================================================================
:: STEP 3: Clear any stale data (optional)
:: ============================================================================
echo.
echo [3/4] Clearing session cache...

:: Clear Python cache if needed
if exist "%SCRIPT_DIR%__pycache__" (
    rd /s /q "%SCRIPT_DIR%__pycache__" 2>nul
)
if exist "%SCRIPT_DIR%ai\__pycache__" (
    rd /s /q "%SCRIPT_DIR%ai\__pycache__" 2>nul
)

echo       [OK] Cache cleared

:: ============================================================================
:: STEP 4: Restart services
:: ============================================================================
echo.
echo [4/4] Starting services...

:: Start the platform
start "" "%SCRIPT_DIR%START_TRADING_PLATFORM.bat"

echo       [OK] Platform restart initiated
echo [%date% %time%] Restart complete >> "%LOG_FILE%"

echo.
echo ================================================================
echo         RESTART COMPLETE
echo ================================================================
echo.
echo Platform is restarting. Check the new window for status.
echo.
echo Log file: %LOG_FILE%
echo.

:: If running interactively, pause
if "%1"=="" (
    pause
)
