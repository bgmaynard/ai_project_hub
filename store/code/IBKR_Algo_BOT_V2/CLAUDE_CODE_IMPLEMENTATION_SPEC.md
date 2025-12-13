# 🛠️ MARKET DATA INTEGRATION - IMPLEMENTATION SPEC FOR CLAUDE CODE

**Project:** IBKR Algo Bot V2  
**Date:** November 14, 2025  
**Purpose:** Detailed technical specifications for Claude Code to implement

---

## 🎯 OVERVIEW

Integrate real IBKR market data into the trading platform UI. Replace all mock data with live feeds from Interactive Brokers API.

---

## 📦 TASK 1: LEVEL 2 MARKET DEPTH INTEGRATION

### File: `ibkr_connector.py`

**Add these methods to the IBKRConnector class:**

```python
def request_market_depth(self, symbol, exchange="SMART", depth=10):
    """
    Request Level 2 market depth data for a symbol.
    
    Args:
        symbol: Stock ticker (e.g., "AAPL")
        exchange: Trading venue (default "SMART")
        depth: Number of price levels (default 10)
    
    Returns:
        bool: True if subscription successful
    """
    try:
        contract = Stock(symbol, exchange, "USD")
        self.ib.reqMktDepth(contract, numRows=depth)
        
        # Initialize storage if not exists
        if symbol not in self.market_depth:
            self.market_depth[symbol] = {
                'bids': [],
                'asks': [],
                'timestamp': None
            }
        
        return True
    except Exception as e:
        print(f"Error requesting market depth for {symbol}: {e}")
        return False

def get_market_depth(self, symbol):
    """
    Get current market depth data for a symbol.
    
    Args:
        symbol: Stock ticker
    
    Returns:
        dict: Market depth with bids and asks
    """
    if symbol not in self.market_depth:
        return {'bids': [], 'asks': [], 'timestamp': None}
    
    return self.market_depth[symbol]

def on_depth_update(self, ticker):
    """
    Callback for market depth updates.
    Updates the internal market_depth dictionary.
    
    Args:
        ticker: ib_insync Ticker object with depth data
    """
    try:
        symbol = ticker.contract.symbol
        
        # Extract bid data
        bids = []
        for bid in ticker.domBids:
            if bid.price and bid.size:
                bids.append({
                    'price': float(bid.price),
                    'size': int(bid.size),
                    'market_maker': getattr(bid, 'marketMaker', '')
                })
        
        # Extract ask data
        asks = []
        for ask in ticker.domAsks:
            if ask.price and ask.size:
                asks.append({
                    'price': float(ask.price),
                    'size': int(ask.size),
                    'market_maker': getattr(ask, 'marketMaker', '')
                })
        
        # Update storage
        self.market_depth[symbol] = {
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error processing depth update for {symbol}: {e}")
```

**Add to `__init__` method:**

```python
# In __init__, add:
self.market_depth = {}

# In the connection setup, add depth callback:
self.ib.pendingTickersEvent += self.on_depth_update
```

---

### File: `dashboard_api.py`

**Update the `/api/level2/{symbol}` endpoint:**

```python
@app.get("/api/level2/{symbol}")
async def get_level2_data(symbol: str):
    """
    Get Level 2 market depth data for a symbol.
    Returns real data from IBKR.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        dict: Market depth with bids and asks
    """
    try:
        if not ibkr_connector or not ibkr_connector.is_connected():
            raise HTTPException(status_code=503, detail="IBKR not connected")
        
        # Get depth data
        depth = ibkr_connector.get_market_depth(symbol)
        
        # If no data yet, request it
        if not depth['bids'] and not depth['asks']:
            ibkr_connector.request_market_depth(symbol)
            # Return empty structure
            return {
                'symbol': symbol,
                'bids': [],
                'asks': [],
                'timestamp': datetime.now().isoformat(),
                'status': 'requesting'
            }
        
        return {
            'symbol': symbol,
            'bids': depth['bids'][:10],  # Top 10 levels
            'asks': depth['asks'][:10],  # Top 10 levels
            'timestamp': depth['timestamp'],
            'status': 'active'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

### File: `ui/customizable_platform.html`

**Update the Level 2 display functions:**

```javascript
// Update the updateLevel2Bids function
function updateLevel2Bids(symbol) {
    fetch(`/api/level2/${symbol}`)
        .then(response => response.json())
        .then(data => {
            const bidsContainer = document.getElementById('level2-bids-data');
            if (!bidsContainer) return;
            
            if (!data.bids || data.bids.length === 0) {
                bidsContainer.innerHTML = '<div style="padding: 10px; color: #888;">Requesting data...</div>';
                return;
            }
            
            let html = '';
            data.bids.forEach(bid => {
                html += `
                    <div class="level2-row">
                        <span class="level2-price" style="color: #4ec9b0;">${bid.price.toFixed(2)}</span>
                        <span class="level2-size">${bid.size}</span>
                        ${bid.market_maker ? `<span class="level2-mm">${bid.market_maker}</span>` : ''}
                    </div>
                `;
            });
            
            bidsContainer.innerHTML = html;
        })
        .catch(error => {
            console.error('Error fetching Level 2 bids:', error);
        });
}

// Update the updateLevel2Asks function
function updateLevel2Asks(symbol) {
    fetch(`/api/level2/${symbol}`)
        .then(response => response.json())
        .then(data => {
            const asksContainer = document.getElementById('level2-asks-data');
            if (!asksContainer) return;
            
            if (!data.asks || data.asks.length === 0) {
                asksContainer.innerHTML = '<div style="padding: 10px; color: #888;">Requesting data...</div>';
                return;
            }
            
            let html = '';
            data.asks.forEach(ask => {
                html += `
                    <div class="level2-row">
                        <span class="level2-price" style="color: #f48771;">${ask.price.toFixed(2)}</span>
                        <span class="level2-size">${ask.size}</span>
                        ${ask.market_maker ? `<span class="level2-mm">${ask.market_maker}</span>` : ''}
                    </div>
                `;
            });
            
            asksContainer.innerHTML = html;
        })
        .catch(error => {
            console.error('Error fetching Level 2 asks:', error);
        });
}

// Add CSS for Level 2 display
const level2Styles = `
.level2-row {
    display: flex;
    justify-content: space-between;
    padding: 2px 5px;
    font-family: 'Courier New', monospace;
    font-size: 10px;
}
.level2-price {
    width: 60px;
    text-align: right;
    font-weight: bold;
}
.level2-size {
    width: 50px;
    text-align: right;
    color: #d4d4d4;
}
.level2-mm {
    width: 40px;
    text-align: left;
    color: #888;
    font-size: 9px;
}
`;
```

---

## 📦 TASK 2: HISTORICAL CHART DATA INTEGRATION

### File: `ibkr_connector.py`

**Add historical data method:**

```python
def get_historical_data(self, symbol, duration="1 D", bar_size="1 min", exchange="SMART"):
    """
    Get historical bar data from IBKR.
    
    Args:
        symbol: Stock ticker
        duration: Time period (e.g., "1 D", "5 D", "1 W")
        bar_size: Bar size (e.g., "1 min", "5 mins", "1 hour")
        exchange: Trading venue
    
    Returns:
        list: Historical bars with OHLCV data
    """
    try:
        contract = Stock(symbol, exchange, "USD")
        
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        # Convert to list of dicts
        result = []
        for bar in bars:
            result.append({
                'time': bar.date.isoformat() if hasattr(bar.date, 'isoformat') else str(bar.date),
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume)
            })
        
        return result
        
    except Exception as e:
        print(f"Error getting historical data for {symbol}: {e}")
        return []
```

---

### File: `dashboard_api.py`

**Add historical data endpoint:**

```python
@app.get("/api/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    timeframe: str = "1m",
    duration: str = "1 D"
):
    """
    Get historical bar data for charting.
    
    Args:
        symbol: Stock ticker
        timeframe: Bar size (1m, 5m, 15m, 1h, 1d)
        duration: How far back (1 D, 5 D, 1 W, 1 M)
    
    Returns:
        dict: Historical OHLCV data
    """
    try:
        if not ibkr_connector or not ibkr_connector.is_connected():
            raise HTTPException(status_code=503, detail="IBKR not connected")
        
        # Map timeframe to IBKR bar size
        bar_size_map = {
            "1m": "1 min",
            "5m": "5 mins",
            "15m": "15 mins",
            "1h": "1 hour",
            "1d": "1 day"
        }
        
        bar_size = bar_size_map.get(timeframe, "1 min")
        
        # Get historical data
        bars = ibkr_connector.get_historical_data(
            symbol=symbol,
            duration=duration,
            bar_size=bar_size
        )
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'bars': bars,
            'count': len(bars)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

### File: `ui/customizable_platform.html`

**Update chart initialization:**

```javascript
// Update the initChart function to use real data
function initChart(containerId, timeframe) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Create chart
    const chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight - 30,
        layout: {
            background: { color: '#1e1e1e' },
            textColor: '#d4d4d4',
        },
        grid: {
            vertLines: { color: '#2d2d30' },
            horzLines: { color: '#2d2d30' },
        },
        timeScale: {
            timeVisible: true,
            secondsVisible: false,
        }
    });
    
    const candlestickSeries = chart.addCandlestickSeries({
        upColor: '#4ec9b0',
        downColor: '#f48771',
        borderVisible: false,
        wickUpColor: '#4ec9b0',
        wickDownColor: '#f48771',
    });
    
    // Load real data
    loadChartData(candlestickSeries, timeframe);
    
    // Update every 60 seconds
    setInterval(() => loadChartData(candlestickSeries, timeframe), 60000);
    
    return chart;
}

// Function to load chart data from API
function loadChartData(series, timeframe) {
    const symbol = getCurrentSymbol(); // Get currently selected symbol
    
    fetch(`/api/historical/${symbol}?timeframe=${timeframe}&duration=1 D`)
        .then(response => response.json())
        .then(data => {
            if (data.bars && data.bars.length > 0) {
                const chartData = data.bars.map(bar => ({
                    time: bar.time,
                    open: bar.open,
                    high: bar.high,
                    low: bar.low,
                    close: bar.close
                }));
                
                series.setData(chartData);
            }
        })
        .catch(error => {
            console.error('Error loading chart data:', error);
        });
}

// Helper to get current symbol from quote panel
function getCurrentSymbol() {
    const symbolElement = document.getElementById('quote-symbol');
    return symbolElement ? symbolElement.textContent : 'AAPL';
}
```

---

## 📦 TASK 3: ACCOUNT DATA INTEGRATION

### File: `ibkr_connector.py`

**Add account data methods:**

```python
def get_account_summary(self):
    """
    Get account summary with cash, equity, buying power.
    
    Returns:
        dict: Account summary data
    """
    try:
        # Get account values
        account_values = self.ib.accountValues()
        
        result = {
            'total_cash': 0.0,
            'net_liquidation': 0.0,
            'buying_power': 0.0,
            'equity_with_loan': 0.0
        }
        
        for av in account_values:
            if av.tag == 'TotalCashValue' and av.currency == 'USD':
                result['total_cash'] = float(av.value)
            elif av.tag == 'NetLiquidation' and av.currency == 'USD':
                result['net_liquidation'] = float(av.value)
            elif av.tag == 'BuyingPower' and av.currency == 'USD':
                result['buying_power'] = float(av.value)
            elif av.tag == 'EquityWithLoanValue' and av.currency == 'USD':
                result['equity_with_loan'] = float(av.value)
        
        return result
        
    except Exception as e:
        print(f"Error getting account summary: {e}")
        return {}

def get_positions(self):
    """
    Get current portfolio positions.
    
    Returns:
        list: List of position dicts
    """
    try:
        positions = self.ib.positions()
        
        result = []
        for pos in positions:
            result.append({
                'symbol': pos.contract.symbol,
                'quantity': float(pos.position),
                'avg_cost': float(pos.avgCost),
                'market_value': float(pos.marketValue) if pos.marketValue else 0.0,
                'unrealized_pnl': float(pos.unrealizedPNL) if pos.unrealizedPNL else 0.0,
                'realized_pnl': float(pos.realizedPNL) if pos.realizedPNL else 0.0
            })
        
        return result
        
    except Exception as e:
        print(f"Error getting positions: {e}")
        return []
```

---

### File: `dashboard_api.py`

**Add account endpoints:**

```python
@app.get("/api/account")
async def get_account_data():
    """
    Get account summary and positions.
    
    Returns:
        dict: Account data including cash, positions, P&L
    """
    try:
        if not ibkr_connector or not ibkr_connector.is_connected():
            raise HTTPException(status_code=503, detail="IBKR not connected")
        
        summary = ibkr_connector.get_account_summary()
        positions = ibkr_connector.get_positions()
        
        # Calculate total P&L
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in positions)
        total_realized_pnl = sum(pos['realized_pnl'] for pos in positions)
        
        return {
            'summary': summary,
            'positions': positions,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 📦 TASK 4: TIME & SALES INTEGRATION

### File: `ibkr_connector.py`

**Add Time & Sales methods:**

```python
def __init__(self):
    # ... existing init code ...
    self.trade_tape = {}  # Store recent trades by symbol
    self.max_tape_size = 100  # Keep last 100 trades

def request_tick_by_tick(self, symbol, exchange="SMART"):
    """
    Subscribe to tick-by-tick trade data.
    
    Args:
        symbol: Stock ticker
        exchange: Trading venue
    
    Returns:
        bool: Success status
    """
    try:
        contract = Stock(symbol, exchange, "USD")
        self.ib.reqTickByTickData(contract, "Last", 0, True)
        
        # Initialize tape storage
        if symbol not in self.trade_tape:
            self.trade_tape[symbol] = []
        
        return True
        
    except Exception as e:
        print(f"Error requesting tick data for {symbol}: {e}")
        return False

def on_tick_data(self, ticker, ticks):
    """
    Callback for tick-by-tick data.
    
    Args:
        ticker: Ticker object
        ticks: List of tick objects
    """
    try:
        symbol = ticker.contract.symbol
        
        for tick in ticks:
            if hasattr(tick, 'price') and hasattr(tick, 'size'):
                trade = {
                    'time': tick.time.isoformat() if hasattr(tick.time, 'isoformat') else str(tick.time),
                    'price': float(tick.price),
                    'size': int(tick.size),
                    'exchange': getattr(tick, 'exchange', '')
                }
                
                # Add to tape
                if symbol not in self.trade_tape:
                    self.trade_tape[symbol] = []
                
                self.trade_tape[symbol].insert(0, trade)
                
                # Keep only last N trades
                if len(self.trade_tape[symbol]) > self.max_tape_size:
                    self.trade_tape[symbol] = self.trade_tape[symbol][:self.max_tape_size]
        
    except Exception as e:
        print(f"Error processing tick data: {e}")

def get_time_sales(self, symbol, limit=50):
    """
    Get recent trades for Time & Sales display.
    
    Args:
        symbol: Stock ticker
        limit: Max trades to return
    
    Returns:
        list: Recent trades
    """
    if symbol not in self.trade_tape:
        return []
    
    return self.trade_tape[symbol][:limit]
```

---

## 🎯 IMPLEMENTATION NOTES

### Priority Order:
1. **Level 2 Market Depth** - Most critical for trading
2. **Historical Charts** - Essential for analysis
3. **Account Data** - Required for position management  
4. **Time & Sales** - Nice to have

### Testing After Each Task:
```python
# Test Level 2:
curl http://127.0.0.1:9101/api/level2/AAPL

# Test Historical:
curl http://127.0.0.1:9101/api/historical/AAPL?timeframe=1m

# Test Account:
curl http://127.0.0.1:9101/api/account
```

### Error Handling:
- All methods should have try/except blocks
- Log errors to console
- Return empty/default data on errors
- Don't crash the server

### Performance:
- Cache historical data for 60 seconds
- Throttle UI updates to every 1-2 seconds
- Limit Level 2 to 10 price levels
- Keep tape to last 100 trades

---

## ✅ COMPLETION CHECKLIST

After implementation, verify:
- [ ] Level 2 shows real bid/ask data
- [ ] Charts display real historical bars
- [ ] Account data shows real balances
- [ ] Time & Sales shows real trades
- [ ] No console errors
- [ ] UI updates smoothly
- [ ] Server handles errors gracefully
- [ ] Data refreshes automatically

---

**Ready for Claude Code to implement!**

Run: `claude "Implement the market data integration as specified in this file"`
