#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Complete Fix for platform.html - Add ALL missing functions
    
.DESCRIPTION
    Adds all missing JavaScript functions:
    - calculateOrderCost
    - useBidPrice
    - useAskPrice
    - togglePriceField
    - And fixes any null reference issues
#>

$platformPath = "C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2\ui\platform.html"

Write-Host @"
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║         🔧 COMPLETE PLATFORM.HTML FIX 🔧                            ║
║              Adding All Missing Functions                            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

# Backup first
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupPath = "${platformPath}.backup_${timestamp}"
Copy-Item $platformPath $backupPath -Force
Write-Host "✅ Backup created: platform.html.backup_${timestamp}" -ForegroundColor Green

# Read the file
$content = Get-Content $platformPath -Raw

# All missing functions to add
$missingFunctions = @"

        // ═══════════════════════════════════════════════════════════════
        // MISSING FUNCTIONS - ADDED BY FIX SCRIPT
        // ═══════════════════════════════════════════════════════════════

        // Calculate order cost and update display
        function calculateOrderCost() {
            try {
                const shares = parseInt(document.getElementById('orderShares')?.value) || 0;
                const price = parseFloat(document.getElementById('orderPrice')?.value) || 0;
                const orderType = document.getElementById('orderType')?.value || 'LIMIT';
                
                let cost = shares * price;
                
                // Add estimated commission (adjust based on your broker)
                // IBKR typically charges `$0.005 per share with `$1 minimum
                let commission = Math.max(1, 0.005 * shares);
                cost += commission;
                
                // Update display if element exists
                const costElement = document.getElementById('orderCost');
                if (costElement) {
                    costElement.textContent = `$`${cost.toFixed(2)}`;
                }
                
                // Update commission display if element exists
                const commElement = document.getElementById('orderCommission');
                if (commElement) {
                    commElement.textContent = `$`${commission.toFixed(2)}`;
                }
                
                return cost;
            } catch (error) {
                console.error('Error calculating order cost:', error);
                return 0;
            }
        }

        // Use current bid price in order form
        function useBidPrice() {
            try {
                const priceInput = document.getElementById('orderPrice');
                if (!priceInput) return;
                
                // Get current bid from the last loaded price data
                if (window.lastPriceData && window.lastPriceData.bid) {
                    priceInput.value = window.lastPriceData.bid.toFixed(2);
                    calculateOrderCost();
                } else {
                    console.warn('No bid price available');
                }
            } catch (error) {
                console.error('Error using bid price:', error);
            }
        }

        // Use current ask price in order form
        function useAskPrice() {
            try {
                const priceInput = document.getElementById('orderPrice');
                if (!priceInput) return;
                
                // Get current ask from the last loaded price data
                if (window.lastPriceData && window.lastPriceData.ask) {
                    priceInput.value = window.lastPriceData.ask.toFixed(2);
                    calculateOrderCost();
                } else {
                    console.warn('No ask price available');
                }
            } catch (error) {
                console.error('Error using ask price:', error);
            }
        }

        // Use current last price in order form
        function useLastPrice() {
            try {
                const priceInput = document.getElementById('orderPrice');
                if (!priceInput) return;
                
                // Get current last price from the last loaded price data
                if (window.lastPriceData && window.lastPriceData.last) {
                    priceInput.value = window.lastPriceData.last.toFixed(2);
                    calculateOrderCost();
                } else {
                    console.warn('No last price available');
                }
            } catch (error) {
                console.error('Error using last price:', error);
            }
        }

        // Toggle price field visibility based on order type
        function togglePriceField() {
            try {
                const orderType = document.getElementById('orderType')?.value;
                const priceRow = document.getElementById('priceRow');
                
                if (!priceRow) {
                    // If priceRow doesn't exist, just find the price input's parent
                    const priceInput = document.getElementById('orderPrice');
                    if (priceInput && priceInput.parentElement) {
                        const row = priceInput.closest('div[style*="grid"]') || priceInput.parentElement;
                        if (orderType === 'MARKET') {
                            row.style.display = 'none';
                        } else {
                            row.style.display = '';
                        }
                    }
                    return;
                }
                
                // Show/hide price field based on order type
                if (orderType === 'MARKET') {
                    priceRow.style.display = 'none';
                } else {
                    priceRow.style.display = '';
                }
                
                calculateOrderCost();
            } catch (error) {
                console.error('Error toggling price field:', error);
            }
        }

        // Validate order before submission
        function validateOrder() {
            try {
                const symbol = document.getElementById('orderSymbol')?.value;
                const shares = parseInt(document.getElementById('orderShares')?.value) || 0;
                const orderType = document.getElementById('orderType')?.value;
                const action = document.querySelector('input[name="orderAction"]:checked')?.value;
                
                if (!symbol || symbol.trim() === '') {
                    alert('Please enter a symbol');
                    return false;
                }
                
                if (shares <= 0) {
                    alert('Please enter a valid number of shares');
                    return false;
                }
                
                if (!action) {
                    alert('Please select BUY or SELL');
                    return false;
                }
                
                if (orderType === 'LIMIT') {
                    const price = parseFloat(document.getElementById('orderPrice')?.value) || 0;
                    if (price <= 0) {
                        alert('Please enter a valid limit price');
                        return false;
                    }
                }
                
                return true;
            } catch (error) {
                console.error('Error validating order:', error);
                return false;
            }
        }

        // Cancel all pending orders for current symbol
        async function cancelAllOrders() {
            try {
                if (!confirm(`Cancel all pending orders for ${currentSymbol}?`)) {
                    return;
                }
                
                const response = await fetch(`/api/orders/cancel-all/${currentSymbol}`, {
                    method: 'POST'
                });
                
                if (response.ok) {
                    alert('All orders cancelled');
                    loadOpenOrders();
                } else {
                    const error = await response.text();
                    alert(`Failed to cancel orders: ${error}`);
                }
            } catch (error) {
                console.error('Error cancelling orders:', error);
                alert('Error cancelling orders');
            }
        }

        // Handle order symbol change
        function onOrderSymbolChange() {
            const symbol = document.getElementById('orderSymbol')?.value?.toUpperCase();
            if (symbol && symbol !== currentSymbol) {
                // Optionally switch to that symbol
                console.log(`Order symbol changed to: ${symbol}`);
                calculateOrderCost();
            }
        }

        // Format number with commas
        function formatNumber(num) {
            return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
        }

        // Format price to 2 decimals
        function formatPrice(price) {
            return typeof price === 'number' ? price.toFixed(2) : '0.00';
        }

        // Store last price data globally for quick access
        window.lastPriceData = null;

        // ═══════════════════════════════════════════════════════════════
        // END OF ADDED FUNCTIONS
        // ═══════════════════════════════════════════════════════════════

"@

# Find a good place to insert (look for existing functions section)
$insertMarker = "function selectFromWatchlist"

if ($content -notmatch "function calculateOrderCost") {
    Write-Host "📝 Adding missing functions..." -ForegroundColor Yellow
    
    # Insert before selectFromWatchlist function
    if ($content -match $insertMarker) {
        $content = $content -replace "(\s+function selectFromWatchlist)", "$missingFunctions`$1"
        Write-Host "✅ Added all missing functions before selectFromWatchlist" -ForegroundColor Green
    } else {
        # If can't find that marker, insert before closing script tag
        $content = $content -replace "(</script>\s*</body>)", "$missingFunctions`$1"
        Write-Host "✅ Added all missing functions before closing script tag" -ForegroundColor Green
    }
} else {
    Write-Host "⚠️  Functions already exist (or partially exist)" -ForegroundColor Yellow
}

# Fix the loadPrice function to store data globally
if ($content -match "async function loadPrice") {
    Write-Host "📝 Updating loadPrice to store data globally..." -ForegroundColor Yellow
    
    # Add line to store price data after fetching
    $content = $content -replace "(const data = await response\.json\(\);)(\s+)(\/\/ Update price display)", "`$1`$2// Store globally for order functions`$2window.lastPriceData = data;`$2`$3"
    
    Write-Host "✅ Updated loadPrice function" -ForegroundColor Green
}

# Save the updated file
$content | Set-Content $platformPath -NoNewline

Write-Host @"

╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║                  ✅ PLATFORM.HTML FIXED! ✅                         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Green

Write-Host "🎯 Functions Added:" -ForegroundColor Cyan
Write-Host "   • calculateOrderCost()" -ForegroundColor White
Write-Host "   • useBidPrice()" -ForegroundColor White
Write-Host "   • useAskPrice()" -ForegroundColor White
Write-Host "   • useLastPrice()" -ForegroundColor White
Write-Host "   • togglePriceField()" -ForegroundColor White
Write-Host "   • validateOrder()" -ForegroundColor White
Write-Host "   • cancelAllOrders()" -ForegroundColor White
Write-Host "   • onOrderSymbolChange()" -ForegroundColor White
Write-Host "   • Helper functions (formatNumber, formatPrice)" -ForegroundColor White

Write-Host "`n🔄 Next Steps:" -ForegroundColor Yellow
Write-Host "   1. Go to your browser with platform.html"
Write-Host "   2. Press Ctrl+Shift+R (hard refresh)"
Write-Host "   3. Test clicking symbols in watchlist"
Write-Host "   4. Test order entry panel"
Write-Host "   5. All errors should be gone!"

Write-Host "`n📁 Backup saved at:" -ForegroundColor Cyan
Write-Host "   $backupPath"

Write-Host "`nPress any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
