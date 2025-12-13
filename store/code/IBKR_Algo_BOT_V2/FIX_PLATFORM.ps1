#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Fix platform.html - Add missing calculateOrderCost function
#>

$platformPath = "C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2\ui\platform.html"

Write-Host "`n🔧 Fixing platform.html..." -ForegroundColor Cyan

# Backup first
Copy-Item $platformPath "${platformPath}.backup3" -Force
Write-Host "✅ Backup created" -ForegroundColor Green

# Read the file
$content = Get-Content $platformPath -Raw

# Find where to insert the function (before selectFromWatchlist)
$insertMarker = "function selectFromWatchlist"

# The missing calculateOrderCost function
$missingFunction = @"

        // Calculate order cost
        function calculateOrderCost() {
            const shares = parseInt(document.getElementById('orderShares').value) || 0;
            const price = parseFloat(document.getElementById('orderPrice').value) || 0;
            const orderType = document.getElementById('orderType').value;
            
            let cost = shares * price;
            
            // Add estimated commission (adjust based on your broker)
            const commission = 0.005 * shares; // $0.005 per share
            cost += commission;
            
            // Update display
            const costElement = document.getElementById('orderCost');
            if (costElement) {
                costElement.textContent = `$`${cost.toFixed(2)}`;
            }
            
            return cost;
        }

"@

# Check if function already exists
if ($content -match "function calculateOrderCost") {
    Write-Host "⚠️  calculateOrderCost function already exists" -ForegroundColor Yellow
} else {
    # Insert the function before selectFromWatchlist
    $content = $content -replace "(\s+function selectFromWatchlist)", "$missingFunction`$1"
    
    # Save the file
    $content | Set-Content $platformPath -NoNewline
    
    Write-Host "✅ Added calculateOrderCost function!" -ForegroundColor Green
}

Write-Host "`n✅ Platform.html fixed!" -ForegroundColor Green
Write-Host "Refresh your browser (Ctrl+Shift+R)" -ForegroundColor Yellow
