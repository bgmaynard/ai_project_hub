@echo off
REM ============================================
REM Schedule 4AM Trading Session Task
REM Run as Administrator!
REM ============================================

echo Creating Windows Task Scheduler job for 4AM trading...
echo.

REM Delete existing task if present
schtasks /delete /tn "Morpheus_4AM_Trading" /f 2>nul

REM Create new scheduled task
REM Runs Monday-Friday at 4:00 AM
schtasks /create ^
    /tn "Morpheus_4AM_Trading" ^
    /tr "C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2\START_4AM_FULL_SESSION.bat" ^
    /sc weekly ^
    /d MON,TUE,WED,THU,FRI ^
    /st 04:00 ^
    /rl HIGHEST

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================
    echo SUCCESS! Task scheduled:
    echo   Name: Morpheus_4AM_Trading
    echo   Schedule: Monday-Friday at 4:00 AM
    echo   Script: START_4AM_FULL_SESSION.bat
    echo ============================================
    echo.
    echo To view: Task Scheduler ^> Task Scheduler Library ^> Morpheus_4AM_Trading
    echo To delete: schtasks /delete /tn "Morpheus_4AM_Trading" /f
    echo.
) else (
    echo.
    echo ERROR: Failed to create scheduled task.
    echo Make sure you run this as Administrator!
    echo.
)

pause
