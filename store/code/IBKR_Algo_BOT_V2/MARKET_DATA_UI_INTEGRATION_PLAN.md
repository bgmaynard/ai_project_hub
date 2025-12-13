# 🚀 MARKET DATA INTEGRATION & UI COMPLETION PLAN

**Date:** November 14, 2025  
**Project:** IBKR Algo Bot V2  
**Phase:** Market Data & UI Integration  

---

## 🎯 OBJECTIVES

1. **Complete Real IBKR Market Data Integration**
   - Real-time Level 2 market depth
   - Time & Sales tape with actual trades
   - Historical bar data for charts
   - Account data from IBKR API

2. **Enhance UI Functionality**
   - Connect all windows to real data
   - Implement proper data refresh
   - Add error handling and status indicators
   - Optimize performance

3. **Test & Validate**
   - Verify all data feeds work
   - Test under various market conditions
   - Ensure UI responsiveness

---

## 📋 PHASE 1: MARKET DATA INTEGRATION

### Task 1.1: Level 2 Market Depth ⏳

**Current State:** Mock data in UI  
**Target:** Real IBKR market depth data

**Implementation Steps:**

1. **Update `ibkr_connector.py`**
   - Enable market depth subscription
   - Handle depth update callbacks
   - Store depth data efficiently

2. **Update `dashboard_api.py`**
   - Enhance `/api/level2/{symbol}` endpoint
   - Return real bid/ask ladder
   - Include market maker info if available

3. **Update UI (`customizable_platform.html`)**
   - Connect to real Level 2 endpoint
   - Display dynamic bid/ask updates
   - Color code price levels
   - Show size aggregation

**Code Snippet for ibkr_connector.py:**
```python
def request_market_depth(self, symbol, exchange="SMART", depth=10):
    """Request Level 2 market depth data"""
    contract = Stock(symbol, exchange, "USD")
    self.ib.reqMktDepth(contract, numRows=depth)
    
def on_depth_update(self, ticker):
    """Handle market depth updates"""
    symbol = ticker.contract.symbol
    self.market_depth[symbol] = {
        'bids': [(bid.price, bid.size) for bid in ticker.domBids],
        'asks': [(ask.price, ask.size) for ask in ticker.domAsks],
        'timestamp': datetime.now().isoformat()
    }
```

---

### Task 1.2: Time & Sales Tape ⏳

**Current State:** Mock trade data  
**Target:** Real trade tape from IBKR

**Implementation Steps:**

1. **Update `ibkr_connector.py`**
   - Subscribe to tick-by-tick trades
   - Store recent trades in circular buffer
   - Include trade conditions

2. **Update `dashboard_api.py`**
   - Enhance `/api/timesales/{symbol}` endpoint
   - Return last N trades
   - Include timestamp, price, size, conditions

3. **Update UI**
   - Real-time trade display
   - Color code upticks/downticks
   - Show trade size and conditions
   - Auto-scroll newest trades

**Code Snippet:**
```python
def request_tick_by_tick(self, symbol, exchange="SMART"):
    """Request tick-by-tick trade data"""
    contract = Stock(symbol, exchange, "USD")
    self.ib.reqTickByTickData(contract, "Last")
    
def on_tick_by_tick(self, ticker, tickType, time):
    """Handle individual trade ticks"""
    if tickType == "Last":
        symbol = ticker.contract.symbol
        trade = {
            'time': time.isoformat(),
            'price': ticker.last,
            'size': ticker.lastSize,
            'exchange': ticker.lastExchange
        }
        self.add_trade_to_tape(symbol, trade)
```

---

### Task 1.3: Historical Chart Data ⏳

**Current State:** Mock candlestick data  
**Target:** Real historical bars from IBKR

**Implementation Steps:**

1. **Update `ibkr_connector.py`**
   - Add historical data request method
   - Support multiple timeframes (1m, 5m, 15m, 1h, 1d)
   - Handle bar updates for real-time charts

2. **Update `dashboard_api.py`**
   - Create `/api/historical/{symbol}` endpoint
   - Parameters: timeframe, duration, bar_size
   - Return OHLCV data

3. **Update UI**
   - Fetch real historical data
   - Update charts with live bars
   - Support multiple timeframes
   - Add volume bars

**Code Snippet:**
```python
def get_historical_data(self, symbol, duration="1 D", bar_size="1 min"):
    """Get historical bar data"""
    contract = Stock(symbol, "SMART", "USD")
    bars = self.ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=True
    )
    return [{
        'time': bar.date.isoformat(),
        'open': bar.open,
        'high': bar.high,
        'low': bar.low,
        'close': bar.close,
        'volume': bar.volume
    } for bar in bars]
```

---

### Task 1.4: Account Data Integration ⏳

**Current State:** Mock account balances  
**Target:** Real IBKR account data

**Implementation Steps:**

1. **Update `ibkr_connector.py`**
   - Request account summary
   - Get portfolio positions
   - Track P&L updates

2. **Update `dashboard_api.py`**
   - Create `/api/account` endpoint
   - Return buying power, cash, equity
   - Include positions with P&L

3. **Update UI**
   - Display real account data
   - Show position details
   - Real-time P&L updates

**Code Snippet:**
```python
def get_account_summary(self):
    """Get account summary data"""
    summary = self.ib.accountSummary()
    return {
        'total_cash': self.get_summary_value(summary, 'TotalCashValue'),
        'net_liquidation': self.get_summary_value(summary, 'NetLiquidation'),
        'buying_power': self.get_summary_value(summary, 'BuyingPower'),
        'equity': self.get_summary_value(summary, 'EquityWithLoanValue')
    }

def get_positions(self):
    """Get current portfolio positions"""
    positions = self.ib.positions()
    return [{
        'symbol': pos.contract.symbol,
        'quantity': pos.position,
        'avg_cost': pos.avgCost,
        'market_value': pos.marketValue,
        'unrealized_pnl': pos.unrealizedPNL
    } for pos in positions]
```

---

## 📋 PHASE 2: UI ENHANCEMENTS

### Task 2.1: Real-Time Data Refresh ⏳

**Implement:**
- WebSocket connection for live updates (or SSE)
- Efficient data polling with rate limiting
- Visual indicators for stale data
- Reconnection logic

### Task 2.2: Error Handling ⏳

**Implement:**
- Connection status indicators
- Error messages for failed data fetches
- Retry logic with exponential backoff
- Fallback to cached data

### Task 2.3: Performance Optimization ⏳

**Implement:**
- Throttle rapid updates
- Virtual scrolling for long lists
- Lazy loading for charts
- Memory management

### Task 2.4: UI Polish ⏳

**Implement:**
- Loading spinners
- Smooth animations
- Tooltip enhancements
- Keyboard shortcuts

---

## 📋 PHASE 3: TESTING & VALIDATION

### Task 3.1: Data Accuracy Testing
- [ ] Compare Level 2 data with TWS
- [ ] Verify Time & Sales matches tape
- [ ] Validate chart data accuracy
- [ ] Check account data consistency

### Task 3.2: Performance Testing
- [ ] Test with 10+ symbols subscribed
- [ ] Measure UI responsiveness
- [ ] Check memory usage over time
- [ ] Test reconnection scenarios

### Task 3.3: Edge Case Testing
- [ ] Test with halted stocks
- [ ] Test pre-market/after-hours
- [ ] Test with low liquidity stocks
- [ ] Test during market close

---

## 🛠️ IMPLEMENTATION PRIORITY

### HIGH PRIORITY (Do First)
1. ✅ Level 2 Market Depth - Critical for trading
2. ✅ Historical Chart Data - Essential for analysis
3. ✅ Account Data - Required for position management

### MEDIUM PRIORITY (Do Next)
4. ⏳ Time & Sales Tape - Useful for order flow
5. ⏳ Real-time Data Refresh - Improves UX
6. ⏳ Error Handling - Prevents crashes

### LOW PRIORITY (Polish)
7. ⏳ Performance Optimization - Nice to have
8. ⏳ UI Polish - Aesthetic improvements

---

## 📝 TECHNICAL NOTES

### IBKR API Considerations

**Market Depth:**
- Requires market data subscription
- Not all exchanges provide Level 2
- May have additional costs

**Tick Data:**
- High frequency, needs throttling
- Store in circular buffer (last 100 trades)
- Clean up old data regularly

**Historical Data:**
- Rate limited by IBKR (60 requests/10 min)
- Cache aggressively
- Use appropriate bar sizes for timeframe

**Account Data:**
- Updates on position changes
- Subscribe to account updates
- Handle multiple accounts if needed

---

## 🚀 GETTING STARTED

### Step 1: Check Current State
```powershell
# Run the state checker
# (Use the check_current_state.ps1 script)
```

### Step 2: Choose Starting Point
Based on your priorities:
- **For Trading:** Start with Level 2 + Account Data
- **For Analysis:** Start with Historical Charts
- **For Monitoring:** Start with Time & Sales

### Step 3: Implement Incrementally
- Do one task at a time
- Test after each implementation
- Commit working code frequently

---

## 📞 NEXT STEPS

**What would you like to start with?**

1. **Level 2 Market Depth** - Show real bid/ask ladder
2. **Historical Charts** - Get real candlestick data
3. **Account Data** - Display actual account info
4. **Time & Sales** - Real trade tape
5. **All at once** - Comprehensive integration

Let me know and I'll create the specific code for that component!

---

*Plan created: November 14, 2025*  
*Ready to implement: Choose your starting point*
