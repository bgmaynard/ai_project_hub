#!/usr/bin/env pwsh
# Quick Add Stock to AI Trading Watchlist

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Symbol
)

$Symbol = $Symbol.ToUpper()

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  ADDING $Symbol TO WATCHLIST" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Add to worklist via API
try {
    Write-Host "[1/4] Adding $Symbol to watchlist..." -ForegroundColor Yellow

    $response = Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/worklist/add/$Symbol" `
        -Method POST `
        -TimeoutSec 5

    Write-Host "      ✓ Added successfully" -ForegroundColor Green

    # Wait a moment for subscription
    Start-Sleep -Seconds 2

    # Get current price
    Write-Host "`n[2/4] Fetching current price..." -ForegroundColor Yellow
    $price = Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/price/$Symbol" -TimeoutSec 3

    if ($price.price) {
        Write-Host "      $Symbol : `$$($price.price)" -ForegroundColor Cyan
        Write-Host "      Change: $($price.change_percent)%" -ForegroundColor $(if($price.change_percent -ge 0){"Green"}else{"Red"})
    }

    # Trigger immediate AI evaluation
    Write-Host "`n[3/4] Running AI evaluation..." -ForegroundColor Yellow
    $eval = Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/ai/evaluate/$Symbol" -TimeoutSec 10

    $signal = $eval.prediction
    $conf = [math]::Round($eval.confidence * 100, 1)
    $prob = [math]::Round($eval.probability * 100, 1)
    $should = $eval.should_trade
    $reason = $eval.reason

    Write-Host "      Prediction: " -NoNewline
    Write-Host "$signal" -ForegroundColor $(if($signal -eq "UP"){"Green"}else{"Red"})
    Write-Host "      Confidence: $conf%" -ForegroundColor Cyan
    Write-Host "      Probability: $prob%" -ForegroundColor Cyan

    # Check if it will trade
    Write-Host "`n[4/4] Trade Decision..." -ForegroundColor Yellow
    if ($should) {
        Write-Host "      ✓ WILL TRADE!" -ForegroundColor Green
        Write-Host "      Bot will attempt to place order on next scan cycle (within 60s)" -ForegroundColor Green
    } else {
        Write-Host "      ✗ NO TRADE" -ForegroundColor Red
        Write-Host "      Reason: $reason" -ForegroundColor Yellow
    }

    # Show current watchlist
    Write-Host "`n[CURRENT WATCHLIST]" -ForegroundColor Cyan
    $worklist = Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/worklist" -TimeoutSec 3

    foreach ($stock in $worklist.symbols) {
        $sym = $stock.symbol
        $p = if($stock.current_price) { "`$$($stock.current_price)" } else { "N/A" }
        $chg = if($stock.percent_change) { "$($stock.percent_change)%" } else { "N/A" }

        $marker = if($sym -eq $Symbol) { "← NEW" } else { "" }

        if ($stock.percent_change -ge 0) {
            Write-Host "  $sym : $p  ↑ $chg $marker" -ForegroundColor $(if($sym -eq $Symbol){"Yellow"}else{"Green"})
        } else {
            Write-Host "  $sym : $p  ↓ $chg $marker" -ForegroundColor $(if($sym -eq $Symbol){"Yellow"}else{"Red"})
        }
    }

    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "✓ $Symbol is now being monitored!" -ForegroundColor Green
    Write-Host "========================================`n" -ForegroundColor Cyan

} catch {
    Write-Host "`n✗ ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "`nTroubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Check if bot is running: Get-Process python" -ForegroundColor White
    Write-Host "  2. Check API status: curl http://127.0.0.1:9101/health" -ForegroundColor White
    Write-Host "  3. Verify symbol is valid (use valid ticker symbols)`n" -ForegroundColor White
}
