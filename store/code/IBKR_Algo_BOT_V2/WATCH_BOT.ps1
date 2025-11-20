# Simple Bot Monitoring Script
# Run this to watch your bot in real-time

Write-Host @"
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║           🤖 AUTONOMOUS BOT MONITOR                        ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

Write-Host "`nStarting monitor... Press Ctrl+C to stop`n" -ForegroundColor Yellow

$count = 0

while ($true) {
    $count++

    try {
        # Get bot status
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/bot/status" -Method Get -ErrorAction Stop

        # Clear screen every 10 iterations
        if ($count % 10 -eq 0) {
            Clear-Host
            Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
            Write-Host "║                 🤖 AUTONOMOUS BOT MONITOR                  ║" -ForegroundColor Cyan
            Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
            Write-Host ""
        }

        $timestamp = Get-Date -Format "HH:mm:ss"
        Write-Host "[$timestamp] " -NoNewline -ForegroundColor Gray

        # Status indicators
        if ($response.running) {
            Write-Host "🟢 RUNNING " -NoNewline -ForegroundColor Green
        } else {
            Write-Host "🔴 STOPPED " -NoNewline -ForegroundColor Red
        }

        if ($response.enabled) {
            Write-Host "🟢 TRADING " -NoNewline -ForegroundColor Green
        } else {
            Write-Host "🟡 PAUSED " -NoNewline -ForegroundColor Yellow
        }

        # Performance metrics
        $pnl = [math]::Round($response.trading_engine.daily_pnl, 2)
        $pnlColor = if ($pnl -gt 0) { "Green" } elseif ($pnl -lt 0) { "Red" } else { "Gray" }

        Write-Host "| Signals: " -NoNewline
        Write-Host $response.total_signals_generated -NoNewline -ForegroundColor Cyan

        Write-Host " | Trades: " -NoNewline
        Write-Host $response.total_trades_executed -NoNewline -ForegroundColor Cyan

        Write-Host " | Rejected: " -NoNewline
        Write-Host $response.total_trades_rejected -NoNewline -ForegroundColor Yellow

        Write-Host " | Positions: " -NoNewline
        Write-Host $response.trading_engine.open_positions -NoNewline -ForegroundColor Cyan

        Write-Host " | P&L: " -NoNewline
        Write-Host "`$$pnl" -ForegroundColor $pnlColor

        # Show positions if any
        if ($response.trading_engine.open_positions -gt 0) {
            Write-Host "    📊 Open: " -NoNewline -ForegroundColor Cyan
            foreach ($symbol in $response.trading_engine.positions.Keys) {
                $pos = $response.trading_engine.positions[$symbol]
                Write-Host "$symbol " -NoNewline -ForegroundColor White
            }
            Write-Host ""
        }

        # Show pending predictions
        if ($response.alpha_fusion.pending_predictions -gt 0) {
            Write-Host "    🔮 Analyzing: " -NoNewline -ForegroundColor Magenta
            Write-Host "$($response.alpha_fusion.pending_predictions) predictions" -ForegroundColor Gray
        }

    } catch {
        Write-Host "[$timestamp] " -NoNewline -ForegroundColor Gray
        Write-Host "⚠️  ERROR: Cannot connect to bot" -ForegroundColor Red
        Write-Host "    Make sure server is running on port 9101" -ForegroundColor Yellow
    }

    Start-Sleep -Seconds 5
}
