@echo off
REM ============================================================
REM SETUP 4AM SCHEDULED TASK (Run as Administrator)
REM ============================================================

echo.
echo ============================================================
echo   MORPHEUS TRADING BOT - 4AM SCHEDULE SETUP
echo ============================================================
echo.

REM Check for admin privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Please run this script as Administrator!
    echo Right-click and select "Run as administrator"
    pause
    exit /b 1
)

echo Creating scheduled task for 4AM pre-market trading...
echo.

schtasks /create /tn "Morpheus_4AM_PreMarket" /tr "C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2\START_4AM_PREMARKET.bat" /sc daily /st 04:00 /f

if %errorlevel% equ 0 (
    echo.
    echo ============================================================
    echo   SUCCESS! Scheduled task created.
    echo ============================================================
    echo.
    echo   Task Name:  Morpheus_4AM_PreMarket
    echo   Schedule:   Daily at 4:00 AM
    echo   Action:     Starts server, scans pre-market, runs scalper
    echo.
    echo   To verify:  schtasks /query /tn "Morpheus_4AM_PreMarket"
    echo   To delete:  schtasks /delete /tn "Morpheus_4AM_PreMarket" /f
    echo.
) else (
    echo.
    echo ERROR: Failed to create scheduled task.
    echo.
)

pause
