$ErrorActionPreference = "Stop"
$repo = "C:\ai_project_hub"
$env:PYTHONPATH = "$repo\store\code"
Write-Host "Launching Dashboard API on http://127.0.0.1:9101 ..." -ForegroundColor Green
python "$repo\store\code\IBKR_Algo_BOT\dashboard_api.py"