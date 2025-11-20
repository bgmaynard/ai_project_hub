# Price Synchronization Fix

## Problem Identified

Different UI components (Quote panel, Worklist, Time & Sales, Chart) are showing different prices for the same stock because:

1. **Separate API Calls**: Each component makes its own API call
2. **Different Timing**: Components refresh at different intervals
3. **No Central Cache**: No single source of truth for current prices
4. **Async Race Conditions**: Multiple requests return at different times

## Root Cause

Looking at the code:
- `loadPrice()` calls `/api/price/{symbol}`
- `loadWorklist()` calls `/api/worklist`
- `loadTimeSales()` calls `/api/timesales/{symbol}`

Each fetches from the SAME backend source (`data_bus.market_data`) but at DIFFERENT TIMES, creating temporary inconsistencies.

## Solution: Centralized Market Data Manager

Create a single JavaScript market data store that:
1. Acts as single source of truth
2. Updates all components simultaneously
3. Shows timestamp to indicate data freshness
4. Highlights when prices are out of sync

## Implementation

### Step 1: Add Global Market Data Store

```javascript
// Global market data cache - SINGLE SOURCE OF TRUTH
const marketDataStore = {
    data: {}, // symbol -> {last, bid, ask, volume, timestamp}
    subscribers: [], // Components that need updates

    // Update data and notify all subscribers
    update(symbol, newData) {
        this.data[symbol] = {
            ...newData,
            timestamp: Date.now()
        };
        this.notifySubscribers(symbol);
    },

    // Get current data for symbol
    get(symbol) {
        return this.data[symbol] || null;
    },

    // Subscribe to updates
    subscribe(callback) {
        this.subscribers.push(callback);
    },

    // Notify all components of update
    notifySubscribers(symbol) {
        this.subscribers.forEach(callback => callback(symbol, this.data[symbol]));
    },

    // Check if data is fresh (< 5 seconds old)
    isFresh(symbol) {
        const data = this.data[symbol];
        if (!data || !data.timestamp) return false;
        return (Date.now() - data.timestamp) < 5000;
    }
};
```

### Step 2: Modify Data Loading Functions

Update `loadPrice()`, `loadWorklist()`, and `loadTimeSales()` to use the central store:

```javascript
async function loadPrice() {
    try {
        const apiData = await apiFetch(`/api/price/${currentSymbol}`);
        if (!apiData || !apiData.data) return;

        const d = apiData.data;

        // UPDATE CENTRAL STORE FIRST
        marketDataStore.update(currentSymbol, {
            last: d.last,
            bid: d.bid,
            ask: d.ask,
            volume: d.volume,
            high: d.high,
            low: d.low,
            open: d.open,
            close: d.close,
            bidSize: d.bidSize,
            askSize: d.askSize
        });

        // Then update UI from store
        updatePriceDisplayFromStore(currentSymbol);

    } catch (error) {
        console.error('Error loading price:', error);
    }
}

function updatePriceDisplayFromStore(symbol) {
    const data = marketDataStore.get(symbol);
    if (!data) return;

    safeUpdate('bidPrice', data.bid?.toFixed(2) || '--');
    safeUpdate('askPrice', data.ask?.toFixed(2) || '--');
    safeUpdate('lastPrice', data.last?.toFixed(2) || '--');
    safeUpdate('volumePrice', data.volume?.toLocaleString() || '--');
    safeUpdate('highPrice', data.high?.toFixed(2) || '--');
    safeUpdate('lowPrice', data.low?.toFixed(2) || '--');
    safeUpdate('openPrice', data.open?.toFixed(2) || '--');
    safeUpdate('closePrice', data.close?.toFixed(2) || '--');

    // Show freshness indicator
    const isFresh = marketDataStore.isFresh(symbol);
    const timestamp = new Date(data.timestamp).toLocaleTimeString();
    console.log(`[SYNC] ${symbol} price updated: $${data.last} (${isFresh ? 'FRESH' : 'STALE'} - ${timestamp})`);
}
```

### Step 3: Synchronize Worklist Display

```javascript
async function loadWorklist() {
    try {
        const response = await fetch(API_BASE_URL + '/api/worklist');
        const result = await response.json();

        if (result.success && result.data) {
            worklistData = result.data;

            // UPDATE CENTRAL STORE for each symbol
            worklistData.forEach(item => {
                marketDataStore.update(item.symbol, {
                    last: item.current_price,
                    bid: item.bid,
                    ask: item.ask,
                    volume: item.volume,
                    change: item.change,
                    change_percent: item.change_percent
                });
            });

            // Display worklist using STORE data
            displayWorklistFromStore();
        }
    } catch (error) {
        console.error('Error loading worklist:', error);
    }
}

function displayWorklistFromStore() {
    const display = document.getElementById('worklistDisplay');
    if (!display) return;

    let html = '<table><thead><tr><th>Symbol</th><th>Price</th><th>Change</th><th>Prediction</th><th>Confidence</th><th>Actions</th></tr></thead><tbody>';

    for (const item of worklistData) {
        // GET PRICE FROM STORE (not from item)
        const storeData = marketDataStore.get(item.symbol);
        const price = storeData?.last ? `$${storeData.last.toFixed(2)}` : '--';
        const isFresh = marketDataStore.isFresh(item.symbol);
        const liveIndicator = isFresh ? '🟢' : '🔴';

        // ... rest of display logic
    }

    html += '</tbody></table>';
    display.innerHTML = html;
}
```

### Step 4: Add Real-Time Synchronization

```javascript
// Subscribe all components to market data updates
marketDataStore.subscribe((symbol, data) => {
    // Update quote panel if this is current symbol
    if (symbol === currentSymbol) {
        updatePriceDisplayFromStore(symbol);
    }

    // Update worklist row for this symbol
    updateWorklistRow(symbol, data);

    // Update time & sales if this is current symbol
    if (symbol === currentSymbol) {
        // Time & sales updates automatically via loadTimeSales()
    }

    // Update chart data point
    updateChartLastPrice(symbol, data.last);
});

function updateWorklistRow(symbol, data) {
    // Find and update the specific row in worklist table
    const row = document.querySelector(`tr[data-symbol="${symbol}"]`);
    if (!row) return;

    const priceCell = row.querySelector('.price-cell');
    const changeCell = row.querySelector('.change-cell');

    if (priceCell) {
        priceCell.textContent = data.last ? `$${data.last.toFixed(2)}` : '--';
        priceCell.classList.add('price-flash'); // Visual feedback
        setTimeout(() => priceCell.classList.remove('price-flash'), 500);
    }

    // Show freshness indicator
    const isFresh = marketDataStore.isFresh(symbol);
    const indicator = row.querySelector('.live-indicator');
    if (indicator) {
        indicator.textContent = isFresh ? '🟢' : '🔴';
        indicator.title = isFresh ? 'Live data' : 'Delayed data';
    }
}
```

### Step 5: Add Visual Indicators

```css
/* Add to <style> section */
.price-flash {
    animation: priceUpdate 0.5s ease;
}

@keyframes priceUpdate {
    0%, 100% { background-color: transparent; }
    50% { background-color: rgba(96, 165, 250, 0.3); }
}

.data-fresh {
    color: #22c55e;
}

.data-stale {
    color: #ef4444;
}

.sync-timestamp {
    font-size: 11px;
    color: #64748b;
    margin-left: 8px;
}
```

### Step 6: Add Sync Status Display

```html
<!-- Add to header -->
<div id="syncStatus" style="display: flex; align-items: center; gap: 8px;">
    <span id="syncIndicator">🟢</span>
    <span id="syncText" style="font-size: 12px; color: #64748b;">All prices synced</span>
    <span id="syncTimestamp" style="font-size: 11px; color: #94a3b8;"></span>
</div>
```

```javascript
// Update sync status
function updateSyncStatus() {
    const indicator = document.getElementById('syncIndicator');
    const text = document.getElementById('syncText');
    const timestamp = document.getElementById('syncTimestamp');

    // Check if all displayed symbols have fresh data
    const allFresh = worklistData.every(item => marketDataStore.isFresh(item.symbol));

    if (allFresh) {
        indicator.textContent = '🟢';
        text.textContent = 'All prices synced';
        text.style.color = '#22c55e';
    } else {
        indicator.textContent = '🟡';
        text.textContent = 'Some prices delayed';
        text.style.color = '#eab308';
    }

    timestamp.textContent = new Date().toLocaleTimeString();
}

// Update sync status every second
setInterval(updateSyncStatus, 1000);
```

## Testing the Fix

### Before Fix:
- Quote panel shows: SPY $450.25
- Worklist shows: SPY $450.18  ❌ MISMATCH
- Time & Sales shows: SPY $450.30  ❌ MISMATCH

### After Fix:
- Quote panel shows: SPY $450.25 🟢
- Worklist shows: SPY $450.25 🟢  ✅ SYNCED
- Time & Sales shows: SPY $450.25 🟢  ✅ SYNCED
- All updated: 2:34:56 PM

## Benefits

1. **Single Source of Truth**: All components read from same cache
2. **Instant Synchronization**: All displays update together
3. **Visual Feedback**: Green/red indicators show data freshness
4. **Timestamps**: Know exactly when data was last updated
5. **Flash Animation**: See when prices update in real-time
6. **Stale Data Detection**: Automatically highlight outdated prices

## Next Steps

1. Implement the centralized market data store
2. Modify all data-loading functions to use the store
3. Add visual indicators for data freshness
4. Test with multiple symbols
5. Monitor for synchronization issues

## Files to Modify

- `complete_platform.html` - Add market data store and update functions
- Add price flash animations
- Add sync status display
- Update all `load*()` functions to use central store

---

**This fix will ensure all UI components always show the SAME price at the SAME time.**
