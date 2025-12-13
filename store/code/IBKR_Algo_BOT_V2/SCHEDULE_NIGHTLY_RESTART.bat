@echo off
:: ============================================================================
:: SCHEDULE NIGHTLY RESTART FOR AI TRADING PLATFORM
:: ============================================================================
:: This script creates a Windows Task Scheduler task to automatically
:: restart the trading platform every night at 4:00 AM (during market close)
::
:: Run this ONCE as Administrator to set up the scheduled task
:: ============================================================================

echo.
echo ================================================================
echo     SCHEDULE NIGHTLY RESTART - AI TRADING PLATFORM
echo ================================================================
echo.
echo This will create a scheduled task to restart the platform
echo every day at 4:00 AM Eastern (market is closed 8pm-4am)
echo.
echo NOTE: Run this as Administrator!
echo.

set "SCRIPT_DIR=%~dp0"
set "TASK_NAME=AI_Trading_Platform_Nightly_Restart"
set "RESTART_SCRIPT=%SCRIPT_DIR%RESTART_TRADING_PLATFORM.bat"

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
echo Restart script: %RESTART_SCRIPT%
echo Schedule: Daily at 4:00 AM
echo.

:: Delete existing task if present
schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1

:: Create the scheduled task
:: /sc DAILY = run every day
:: /st 04:00 = start time 4:00 AM
:: /ru SYSTEM = run as SYSTEM (works even if not logged in)
schtasks /create /tn "%TASK_NAME%" /tr "\"%RESTART_SCRIPT%\" scheduled" /sc DAILY /st 04:00 /ru SYSTEM /rl HIGHEST /f

if %errorLevel% equ 0 (
    echo.
    echo ================================================================
    echo     SUCCESS! Nightly restart scheduled
    echo ================================================================
    echo.
    echo Task Name: %TASK_NAME%
    echo Schedule:  Daily at 4:00 AM
    echo Action:    Restart trading platform
    echo.
    echo To modify or remove this task:
    echo   1. Open Task Scheduler (taskschd.msc)
    echo   2. Find "%TASK_NAME%"
    echo   3. Right-click to modify/delete
    echo.
    echo To manually trigger a restart now:
    echo   schtasks /run /tn "%TASK_NAME%"
    echo.
) else (
    echo.
    echo [ERROR] Failed to create scheduled task
    echo.
    echo Try running as Administrator or manually create the task:
    echo   1. Open Task Scheduler
    echo   2. Create Basic Task
    echo   3. Name: %TASK_NAME%
    echo   4. Trigger: Daily at 4:00 AM
    echo   5. Action: Start a program
    echo   6. Program: %RESTART_SCRIPT%
    echo.
)

pause
