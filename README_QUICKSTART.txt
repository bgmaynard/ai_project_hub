AI Dashboard Replacement (No‑Code Steps)
=======================================

1) Unzip this into C:\ai_project_hub\
   After unzipping, you will have:
   - C:\ai_project_hub\store\code\IBKR_Algo_BOT\dashboard_api.py
   - C:\ai_project_hub\store\code\IBKR_Algo_BOT\ui\index.html, app.js, styles.css
   - C:\ai_project_hub\run_api.bat  (double-click to run)
   - C:\ai_project_hub\Start-API.ps1 (PowerShell runner)
   - C:\ai_project_hub\requirements.txt

2) (First time) Install requirements
   Open PowerShell and run:
     cd C:\ai_project_hub
     python -m pip install -r requirements.txt

3) Start the dashboard API
   - Double-click C:\ai_project_hub\run_api.bat
     OR
   - Right-click C:\ai_project_hub\Start-API.ps1 > Run with PowerShell

4) Open the Dashboard
   http://127.0.0.1:9101/ui/

5) What works now
   - Buttons call these endpoints and show JSON:
       /api/status
       /api/bots/mtf/start     { "symbol": "SPY" }
       /api/bots/mtf/stop
       /api/bots/warrior/start { "symbol": "AAPL" }
       /api/bots/warrior/stop
       /api/signals/tail       (shows a rolling log file)

6) When you're ready for live logic
   - Replace the placeholder state in dashboard_api.py with your TWS/IBKR bridge checks.
   - Insert your actual MTF/Warrior loops in the start endpoints.
   - Append real events to store/code/IBKR_Algo_BOT/data/signals_tail.log