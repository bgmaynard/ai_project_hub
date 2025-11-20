# Start IBKR Algorithmic Trading Bot V2
Write-Host "Starting IBKR Algo Bot V2..." -ForegroundColor Cyan

# Start server
Write-Host "Starting FastAPI server..." -ForegroundColor Green
Start-Process python -ArgumentList "dashboard_api.py" -NoNewWindow

Start-Sleep -Seconds 3

# Open browser
Write-Host "Opening platform..." -ForegroundColor Green
Start-Process "http://127.0.0.1:9101/ui/platform.html"

Write-Host ""
Write-Host "Server running on http://127.0.0.1:9101" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
