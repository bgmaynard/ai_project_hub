@echo off
:: ============================================================================
:: SCHEDULE 4AM PRE-MARKET AUTO-TRADING
:: ============================================================================
:: Creates a Windows Task Scheduler task to automatically:
::   1. Scan for momentum stocks at 4:00 AM
::   2. Add top picks to watchlist
::   3. Start paper trading
::
:: Run this ONCE as Administrator to set up the scheduled task
:: ============================================================================

echo.
echo ================================================================
echo     SCHEDULE 4AM PRE-MARKET SCANNER
echo ================================================================
echo.
echo This will create a scheduled task to run the pre-market
echo scanner every weekday at 4:00 AM Eastern.
echo.
echo NOTE: Run this as Administrator!
echo.

set "SCRIPT_DIR=%~dp0"
set "TASK_NAME=Morpheus_4AM_PreMarket"
set "PREMARKET_SCRIPT=%SCRIPT_DIR%START_4AM_PREMARKET.bat"

:: Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] This script requires Administrator privileges!
    echo.
    echo Right-click this file and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo Creating scheduled task: %TASK_NAME%
echo Script: %PREMARKET_SCRIPT%
echo Schedule: Weekdays at 4:00 AM
echo.

:: Delete existing task if present
schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1

:: Create the scheduled task
:: /sc WEEKLY = run weekly
:: /d MON,TUE,WED,THU,FRI = weekdays only
:: /st 04:00 = start time 4:00 AM
schtasks /create /tn "%TASK_NAME%" /tr "\"%PREMARKET_SCRIPT%\"" /sc WEEKLY /d MON,TUE,WED,THU,FRI /st 04:00 /f

if %errorLevel% equ 0 (
    echo.
    echo ================================================================
    echo     SUCCESS! Pre-market scanner scheduled
    echo ================================================================
    echo.
    echo Task Name:  %TASK_NAME%
    echo Schedule:   Weekdays at 4:00 AM
    echo Action:     Scan movers + Start paper trading
    echo.
    echo What happens at 4AM:
    echo   1. Server starts (if not running)
    echo   2. Scans Yahoo/Schwab for momentum stocks
    echo   3. Filters: Price $1-$20, Gap 5%+, Vol 500K+
    echo   4. Adds top 5 picks to scalper watchlist
    echo   5. Starts HFT Scalper in PAPER mode
    echo.
    echo To test now:
    echo   schtasks /run /tn "%TASK_NAME%"
    echo.
    echo To view/modify:
    echo   Open Task Scheduler (taskschd.msc)
    echo.
) else (
    echo.
    echo [ERROR] Failed to create scheduled task
    echo.
)

pause
