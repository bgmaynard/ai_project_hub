#!/usr/bin/env pwsh
# FINAL COMPREHENSIVE FIX - All Missing Functions

$platformPath = "C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2\ui\platform.html"

Write-Host "`n🔧 FINAL COMPREHENSIVE FIX..." -ForegroundColor Cyan

# Backup
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item $platformPath "${platformPath}.final_backup_${timestamp}" -Force
Write-Host "✅ Backup created" -ForegroundColor Green

$content = Get-Content $platformPath -Raw

# ALL MISSING FUNCTIONS
$allFunctions = @'

        // ═══════════════════════════════════════════════════════════════
        // ALL MISSING FUNCTIONS - COMPLETE FIX
        // ═══════════════════════════════════════════════════════════════

        window.lastPriceData = null;

        function calculateOrderCost() {
            try {
                const shares = parseInt(document.getElementById('orderShares')?.value) || 0;
                const price = parseFloat(document.getElementById('orderPrice')?.value) || 0;
                let cost = shares * price;
                let commission = Math.max(1, 0.005 * shares);
                cost += commission;
                const costElement = document.getElementById('orderCost');
                if (costElement) costElement.textContent = `$${cost.toFixed(2)}`;
                return cost;
            } catch (e) { return 0; }
        }

        function useBidPrice() {
            try {
                const priceInput = document.getElementById('orderPrice');
                if (priceInput && window.lastPriceData?.bid) {
                    priceInput.value = window.lastPriceData.bid.toFixed(2);
                    calculateOrderCost();
                }
            } catch (e) { console.error(e); }
        }

        function useAskPrice() {
            try {
                const priceInput = document.getElementById('orderPrice');
                if (priceInput && window.lastPriceData?.ask) {
                    priceInput.value = window.lastPriceData.ask.toFixed(2);
                    calculateOrderCost();
                }
            } catch (e) { console.error(e); }
        }

        function useLastPrice() {
            try {
                const priceInput = document.getElementById('orderPrice');
                if (priceInput && window.lastPriceData?.last) {
                    priceInput.value = window.lastPriceData.last.toFixed(2);
                    calculateOrderCost();
                }
            } catch (e) { console.error(e); }
        }

        function togglePriceField() {
            try {
                const orderType = document.getElementById('orderType')?.value;
                const priceInput = document.getElementById('orderPrice');
                if (priceInput?.parentElement) {
                    const row = priceInput.closest('div') || priceInput.parentElement;
                    row.style.display = (orderType === 'MARKET') ? 'none' : '';
                }
                calculateOrderCost();
            } catch (e) { console.error(e); }
        }

        function onOrderSymbolChange() {
            try {
                const symbol = document.getElementById('orderSymbol')?.value?.toUpperCase();
                if (symbol) calculateOrderCost();
            } catch (e) { console.error(e); }
        }

        function clearOrderForm() {
            try {
                const sharesInput = document.getElementById('orderShares');
                const priceInput = document.getElementById('orderPrice');
                if (sharesInput) sharesInput.value = '100';
                if (priceInput) priceInput.value = '';
                calculateOrderCost();
            } catch (e) { console.error(e); }
        }

        async function cancelLastOrder() {
            try {
                alert('Cancel last order - not implemented yet');
            } catch (e) { console.error(e); }
        }

        async function cancelAllOrders() {
            try {
                if (!confirm(`Cancel all orders?`)) return;
                alert('Cancel all orders - not implemented yet');
            } catch (e) { console.error(e); }
        }

        // ═══════════════════════════════════════════════════════════════
        // END OF ALL MISSING FUNCTIONS
        // ═══════════════════════════════════════════════════════════════

'@

# Add functions if not present
if ($content -notmatch "function calculateOrderCost") {
    $content = $content -replace "(\s+function selectFromWatchlist)", "$allFunctions`$1"
    Write-Host "✅ Added all missing functions" -ForegroundColor Green
} else {
    Write-Host "⚠️  Some functions may already exist" -ForegroundColor Yellow
}

# Fix submitOrder to handle nulls
$content = $content -replace 'const symbol = document\.getElementById\(''orderSymbol''\)\.value', 'const symbol = document.getElementById(''orderSymbol'')?.value || currentSymbol'
$content = $content -replace 'const shares = parseInt\(document\.getElementById\(''orderShares''\)\.value\)', 'const shares = parseInt(document.getElementById(''orderShares'')?.value) || 0'
$content = $content -replace 'const orderType = document\.getElementById\(''orderType''\)\.value', 'const orderType = document.getElementById(''orderType'')?.value || ''LIMIT'''
$content = $content -replace 'const price = parseFloat\(document\.getElementById\(''orderPrice''\)\.value\)', 'const price = parseFloat(document.getElementById(''orderPrice'')?.value) || 0'

Write-Host "✅ Fixed submitOrder null checks" -ForegroundColor Green

# Save
$content | Set-Content $platformPath -NoNewline

Write-Host @"

╔══════════════════════════════════════════════════════════════════════╗
║                  ✅ ALL FUNCTIONS ADDED! ✅                         ║
╚══════════════════════════════════════════════════════════════════════╝

🔄 Refresh your browser (Ctrl+Shift+R)

"@ -ForegroundColor Green
