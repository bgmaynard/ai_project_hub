@echo off
title Stop All Trading
color 0C

echo ============================================================
echo   STOPPING ALL TRADING PROCESSES
echo ============================================================
echo.

echo Killing all Python processes...
taskkill /F /IM python.exe 2>nul

echo.
echo All trading processes stopped.
echo.
pause
