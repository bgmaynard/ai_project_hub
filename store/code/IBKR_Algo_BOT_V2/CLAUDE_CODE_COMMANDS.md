# CLAUDE CODE COMMANDS - MARKET DATA INTEGRATION

## 🎯 MAIN TASK

Implement real IBKR market data integration to replace mock data in the trading platform UI.

## 📝 SPECIFICATION FILE

Read and follow: `CLAUDE_CODE_IMPLEMENTATION_SPEC.md`

## 🚀 SUGGESTED COMMANDS FOR CLAUDE CODE

### Command 1: Start with Level 2 Market Depth
```
Read CLAUDE_CODE_IMPLEMENTATION_SPEC.md and implement Task 1 (Level 2 Market Depth Integration). Update ibkr_connector.py to add market depth subscription methods, update dashboard_api.py to enhance the /api/level2/ endpoint with real data, and update ui/customizable_platform.html to display live bid/ask ladder. Test with curl http://127.0.0.1:9101/api/level2/AAPL when done.
```

### Command 2: Add Historical Chart Data
```
Read CLAUDE_CODE_IMPLEMENTATION_SPEC.md and implement Task 2 (Historical Chart Data Integration). Add get_historical_data method to ibkr_connector.py, create /api/historical/ endpoint in dashboard_api.py, and update the chart initialization in customizable_platform.html to load real candlestick data from IBKR. Test with different timeframes (1m, 5m).
```

### Command 3: Integrate Account Data
```
Read CLAUDE_CODE_IMPLEMENTATION_SPEC.md and implement Task 3 (Account Data Integration). Add get_account_summary and get_positions methods to ibkr_connector.py, create /api/account endpoint in dashboard_api.py, and update the account window in customizable_platform.html to display real balances and positions.
```

### Command 4: Add Time & Sales Tape
```
Read CLAUDE_CODE_IMPLEMENTATION_SPEC.md and implement Task 4 (Time & Sales Integration). Add tick-by-tick subscription to ibkr_connector.py, create /api/timesales/ endpoint in dashboard_api.py, and update the Time & Sales window to show real trade data with color-coded upticks/downticks.
```

### Command 5: Full Integration (All at Once)
```
Read CLAUDE_CODE_IMPLEMENTATION_SPEC.md and implement all 4 tasks: Level 2 Market Depth, Historical Charts, Account Data, and Time & Sales. Follow the implementation notes and priority order. Test each component after implementation. Ensure no mock data remains in the UI.
```

---

## 🧪 TESTING COMMANDS

After each implementation, test with:

```bash
# Test health
curl http://127.0.0.1:9101/health

# Test Level 2
curl http://127.0.0.1:9101/api/level2/AAPL

# Test historical data
curl "http://127.0.0.1:9101/api/historical/AAPL?timeframe=1m&duration=1%20D"

# Test account
curl http://127.0.0.1:9101/api/account

# Test time & sales
curl http://127.0.0.1:9101/api/timesales/AAPL
```

---

## 📋 FILES TO MODIFY

1. **ibkr_connector.py** - Add all data fetching methods
2. **dashboard_api.py** - Add/update API endpoints
3. **ui/customizable_platform.html** - Update UI to use real data

---

## ⚠️ IMPORTANT NOTES

- Server must be running on port 9101
- IBKR TWS must be connected
- Test after each major change
- Don't break existing functionality
- Add error handling to all methods
- Log errors to console

---

## ✅ SUCCESS CRITERIA

When complete:
- Level 2 shows real bid/ask prices
- Charts display actual market data
- Account shows true balances
- Time & Sales shows live trades
- No console errors
- UI updates automatically

---

## 🎯 RECOMMENDED APPROACH

**Start with Command 1 (Level 2)** - Most important for trading  
**Then Command 2 (Charts)** - Visual confirmation  
**Then Command 3 (Account)** - Position tracking  
**Finally Command 4 (Time & Sales)** - Polish

Or use **Command 5 for full integration** if you want everything at once!
