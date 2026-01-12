@echo off
REM ============================================
REM Generate Session Report
REM Run after trading sessions complete
REM ============================================

cd /d C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2

echo ============================================
echo GENERATING SESSION REPORT
echo %date% %time%
echo ============================================
echo.

REM Generate and save report via API
curl -s -X POST http://localhost:9100/api/reports/session/save

echo.
echo.

REM Also run the Python script for console output
python -c "from ai.session_report import generate_report, print_report_summary, save_report; r = generate_report(); print_report_summary(r); save_report(r)"

echo.
echo ============================================
echo Report saved to reports/ directory
echo ============================================

pause
