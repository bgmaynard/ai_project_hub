#!/usr/bin/env pwsh
# Quick AI Auto-Trading Status Check

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  AI AUTO-TRADING STATUS" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# 1. Check if bot is running
$botProc = Get-Process python -ErrorAction SilentlyContinue
if ($botProc) {
    Write-Host "[✓] Bot Running: YES (PID: $($botProc[0].Id))" -ForegroundColor Green
} else {
    Write-Host "[✗] Bot Running: NO" -ForegroundColor Red
    exit
}

# 2. Check auto-trading config
$envFile = "C:\ai_project_hub\store\code\IBKR_Algo_BOT\.env"
$enabled = (Select-String -Path $envFile -Pattern "AUTO_TRADE_ENABLED=(\w+)" | ForEach-Object { $_.Matches.Groups[1].Value })
$minConf = (Select-String -Path $envFile -Pattern "AUTO_TRADE_MIN_CONFIDENCE=([\d.]+)" | ForEach-Object { $_.Matches.Groups[1].Value })
$minProb = (Select-String -Path $envFile -Pattern "AUTO_TRADE_MIN_PROB=([\d.]+)" | ForEach-Object { $_.Matches.Groups[1].Value })

Write-Host "[✓] Auto-Trading: $enabled" -ForegroundColor $(if($enabled -eq "true"){"Green"}else{"Yellow"})
Write-Host "[✓] Min Confidence: $($minConf * 100)%" -ForegroundColor Cyan
Write-Host "[✓] Min Probability: $($minProb * 100)%" -ForegroundColor Cyan

# 3. Get worklist
try {
    $worklist = Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/worklist" -TimeoutSec 2
    Write-Host "`n[WATCHING $($worklist.count) STOCKS]" -ForegroundColor Cyan

    foreach ($stock in $worklist.symbols) {
        $sym = $stock.symbol
        $price = if($stock.current_price) { "`$$($stock.current_price)" } else { "N/A" }
        $change = if($stock.percent_change) { "$($stock.percent_change)%" } else { "N/A" }

        if ($stock.percent_change -ge 0) {
            Write-Host "  $sym : $price  ↑ $change" -ForegroundColor Green
        } else {
            Write-Host "  $sym : $price  ↓ $change" -ForegroundColor Red
        }
    }
} catch {
    Write-Host "`n[!] Could not fetch worklist" -ForegroundColor Yellow
}

# 4. Try to call AI evaluate endpoint for each symbol
Write-Host "`n[AI PREDICTIONS]" -ForegroundColor Cyan
try {
    $symbols = $worklist.symbols | ForEach-Object { $_.symbol }

    foreach ($symbol in $symbols) {
        try {
            $eval = Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/ai/evaluate/$symbol" -TimeoutSec 3 -ErrorAction SilentlyContinue

            $signal = $eval.prediction
            $conf = [math]::Round($eval.confidence * 100, 1)
            $prob = [math]::Round($eval.probability * 100, 1)
            $should = $eval.should_trade

            $color = switch ($signal) {
                "UP" { "Green" }
                "DOWN" { "Red" }
                default { "Yellow" }
            }

            $tradeStatus = if($should) { "✓ TRADE" } else { "✗ NO TRADE" }
            $tradeColor = if($should) { "Green" } else { "Red" }

            Write-Host "  $symbol : " -NoNewline
            Write-Host "$signal " -ForegroundColor $color -NoNewline
            Write-Host "| Conf: $conf% Prob: $prob% | " -NoNewline
            Write-Host "$tradeStatus" -ForegroundColor $tradeColor

        } catch {
            Write-Host "  $symbol : Analyzing..." -ForegroundColor DarkGray
        }
    }
} catch {
    Write-Host "  AI predictions not available yet" -ForegroundColor Yellow
}

Write-Host "`n[DASHBOARD]" -ForegroundColor Cyan
Write-Host "  http://127.0.0.1:9101/ui/complete_platform.html" -ForegroundColor Blue

Write-Host "`n[SCANNER LOGS]" -ForegroundColor Cyan
Write-Host "  To see live scanner output, run:" -ForegroundColor Yellow
Write-Host "  Get-Content C:\ai_project_hub\store\code\IBKR_Algo_BOT\logs\dashboard.log -Tail 20 -Wait`n" -ForegroundColor White
