@echo off
REM ============================================================
REM CHECK PRE-MARKET REPORT
REM Run this when you wake up to see what happened overnight
REM ============================================================

echo.
echo ============================================================
echo   MORPHEUS TRADING BOT - PRE-MARKET REPORT
echo   Review your overnight paper trading results
echo ============================================================
echo.

echo === TRADING WINDOW ===
curl -s http://localhost:9100/api/coach/window 2>nul | python -m json.tool 2>nul || curl -s http://localhost:9100/api/coach/window

echo.
echo === PRE-MARKET STATUS ===
curl -s http://localhost:9100/api/premarket/status 2>nul | python -m json.tool 2>nul || curl -s http://localhost:9100/api/premarket/status

echo.
echo === TRADES EXECUTED ===
curl -s http://localhost:9100/api/premarket/trades 2>nul | python -m json.tool 2>nul || curl -s http://localhost:9100/api/premarket/trades

echo.
echo === FULL REPORT ===
curl -s http://localhost:9100/api/premarket/report 2>nul | python -m json.tool 2>nul || curl -s http://localhost:9100/api/premarket/report

echo.
echo === MORNING BRIEFING ===
curl -s http://localhost:9100/api/coach/briefing 2>nul | python -m json.tool 2>nul || curl -s http://localhost:9100/api/coach/briefing

echo.
pause
