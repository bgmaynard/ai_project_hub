# 🚀 IBKR Integration Quick Start Guide

**Connect Your MTF Models to Live Trading**

---

## 📋 What You Have

✅ **Trained MTF Models** (V2) with excellent results:
- AAPL: +25% return, 62.6% accuracy, 61.4% win rate
- TSLA: +69% return, 57.6% accuracy, 58.2% win rate

✅ **3 New Integration Files** Created:
1. `ibkr_live_trading_connector.py` - Main trading bot
2. `validate_ibkr_connection.py` - Connection tester
3. This guide

---

## 🎯 Step-by-Step Integration

### Step 1: Install IBKR API (5 minutes)

```bash
# Install the IBKR Python API
pip install ibapi
```

**Verify installation:**
```bash
python -c "from ibapi.client import EClient; print('✓ IBKR API installed')"
```

---

### Step 2: Configure TWS/Gateway (10 minutes)

#### Option A: TWS (Trader Workstation)

1. **Download & Install TWS**
   - Go to: https://www.interactivebrokers.com/en/trading/tws.php
   - Download for your OS
   - Install and login with paper trading account

2. **Enable API Access**
   - Go to: **File → Global Configuration → API → Settings**
   - ✅ Check "Enable ActiveX and Socket Clients"
   - ✅ Check "Read-Only API" (safer for testing)
   - Port: **7497** (paper) or **7496** (live)
   - Click **Apply** and **OK**

3. **Restart TWS**

#### Option B: IB Gateway (Lighter Weight)

1. Download IB Gateway instead of TWS
2. Same configuration steps
3. Ports: **4002** (paper) or **4001** (live)

---

### Step 3: Validate Connection (5 minutes)

**Run the validator:**

```bash
python validate_ibkr_connection.py
```

**Expected output:**

```
======================================================================
IBKR CONNECTION & FUNCTIONALITY VALIDATOR
======================================================================

TEST 1: Connection to TWS/Gateway
----------------------------------------------------------------------
  Trying TWS Paper Trading (port 7497)... ✓ CONNECTED
✓ Connected! Next order ID: 1

TEST 2: Real-Time Market Data
----------------------------------------------------------------------
  Requesting market data for AAPL...
  ✓ Market data received:
    BID: 245.50
    ASK: 245.52
    LAST: 245.51
    VOLUME: 45,234,567

TEST 3: Historical Data
----------------------------------------------------------------------
  Requesting 2 days of 1-hour bars for AAPL...
  ✓ Historical data complete: 13 bars
  ✓ Historical data received: 13 bars
    First bar: 20251010  09:30:00
    Last bar: 20251011  15:30:00

TEST 4: Account Data
----------------------------------------------------------------------
  Requesting account updates...
  ✓ Account data received:
    NetLiquidation: 1000000.00 USD
    TotalCashValue: 1000000.00 USD
    AvailableFunds: 1000000.00 USD

TEST 5: Positions
----------------------------------------------------------------------
  Requesting positions...
  ℹ️  No open positions

======================================================================
VALIDATION SUMMARY
======================================================================

✓ Tests Passed: 5
  ✓ Connection
  ✓ Market Data
  ✓ Historical Data
  ✓ Account Data
  ✓ Positions

======================================================================
✅ ALL TESTS PASSED - READY FOR LIVE TRADING!
======================================================================
```

**If tests fail**, see Troubleshooting section below.

---

### Step 4: Test Order Placement (Optional, 5 minutes)

**⚠️ PAPER TRADING ONLY!**

```bash
python validate_ibkr_connection.py --test-orders
```

This places a test limit order at $100 (way below market) so it won't fill, then cancels it.

---

### Step 5: Run Live Trading Bot (GO TIME!)

**Paper Trading First (Recommended):**

```bash
python ibkr_live_trading_connector.py \
    --mode paper \
    --symbols AAPL TSLA \
    --port 7497 \
    --interval 300 \
    --confidence 0.52
```

**What happens:**

```
======================================================================
IBKR LIVE TRADING CONNECTOR - MTF MODELS
======================================================================
Mode: PAPER
Symbols: AAPL, TSLA
Port: 7497
Check interval: 300s
Confidence threshold: 0.52
======================================================================

Connecting to IBKR at 127.0.0.1:7497...
✓ Connected to IBKR
✓ Account data complete for DU123456
✓ Received 0 positions

Loading model for AAPL...
✓ Model loaded for AAPL
Loading model for TSLA...
✓ Model loaded for TSLA

======================================================================
STARTING LIVE TRADING LOOP
======================================================================
Symbols: AAPL, TSLA
Check interval: 300s (5.0 min)
Confidence threshold: 0.52
======================================================================

✓ Subscribed to AAPL market data (reqId=1000)
✓ Subscribed to TSLA market data (reqId=1001)

--- Iteration 1 @ 14:35:22 ---
Requesting 2 D of 1 hour data for AAPL...
✓ Historical data complete for AAPL: 13 bars
AAPL BUY signal: confidence=0.685
✓ Order placed: BUY 100 AAPL @ Market (ID=1)
Order 1: PreSubmitted (filled 0, remaining 100)
Order 1: Submitted (filled 0, remaining 100)
Order 1: Filled (filled 100, remaining 0)
  ✓ Executed: BOT 100 AAPL @ $245.51

Requesting 2 D of 1 hour data for TSLA...
✓ Historical data complete for TSLA: 13 bars

Active positions: 1
Daily P&L: $0.00
Sleeping 300s...

--- Iteration 2 @ 14:40:22 ---
...
```

---

## 🎛️ Command-Line Options

### Basic Usage

```bash
python ibkr_live_trading_connector.py [OPTIONS]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | paper | Trading mode: `paper` or `live` |
| `--symbols` | AAPL TSLA | Symbols to trade (space-separated) |
| `--port` | 7497 | TWS port (7497=paper, 7496=live) |
| `--interval` | 300 | Check interval in seconds (300 = 5 min) |
| `--confidence` | 0.52 | Confidence threshold (0-1) |

### Examples

**1. Trade only AAPL, check every 1 minute:**
```bash
python ibkr_live_trading_connector.py --symbols AAPL --interval 60
```

**2. Trade 4 symbols with higher confidence:**
```bash
python ibkr_live_trading_connector.py \
    --symbols AAPL TSLA NVDA AMD \
    --confidence 0.58
```

**3. Live trading (REAL MONEY!):**
```bash
python ibkr_live_trading_connector.py \
    --mode live \
    --port 7496 \
    --symbols AAPL \
    --confidence 0.65
```

---

## 📊 Understanding the Output

### Signal Generation

```
AAPL BUY signal: confidence=0.685
```
- Model predicts price will go UP
- Confidence is 68.5% (above 52% threshold)
- Bot will place BUY order

### Order Execution

```
✓ Order placed: BUY 100 AAPL @ Market (ID=1)
Order 1: PreSubmitted (filled 0, remaining 100)
Order 1: Submitted (filled 0, remaining 100)
Order 1: Filled (filled 100, remaining 0)
  ✓ Executed: BOT 100 AAPL @ $245.51
```

**Order Status Progression:**
1. **PreSubmitted** - Order sent to IBKR
2. **Submitted** - Order sent to exchange
3. **Filled** - Order executed!

### Exit Signal

```
AAPL SELL signal: confidence=0.652
✓ Order placed: SELL 100 AAPL @ Market (ID=2)
Order 2: Filled (filled 100, remaining 0)
AAPL P&L: $+234.00
```

Bot closes position and calculates profit/loss.

---

## ⚙️ Configuration

### Risk Parameters

Edit these in `ibkr_live_trading_connector.py`:

```python
# Position sizing
self.max_position_size = 10000  # Max $10k per position

# Daily limits
self.daily_loss_limit = 2000    # Stop trading at -$2k loss

# Holding periods
self.min_hold_bars = 2          # Min 2 check intervals (10 min if interval=300s)
self.max_hold_bars = 24         # Max 24 intervals (2 hours)

# Signal threshold
self.confidence_threshold = 0.52  # Or pass via --confidence
```

### Position Size Calculation

Currently fixed at 100 shares. To make dynamic:

```python
def calculate_position_size(self, symbol, confidence):
    """Dynamic position sizing based on confidence"""
    
    # Get current price
    price = self.ibkr.market_data[symbol].get('last', 0)
    
    # Base size on confidence and price
    # Higher confidence = larger position
    confidence_factor = (confidence - 0.5) * 2  # Scale to 0-1
    max_shares = int(self.max_position_size / price)
    
    shares = int(max_shares * confidence_factor * 0.5)  # Conservative
    shares = max(10, min(shares, max_shares))  # Between 10 and max
    
    return shares
```

---

## 🔧 Troubleshooting

### Connection Issues

**Problem:** `Failed to connect to IBKR`

**Solutions:**
1. Check TWS/Gateway is running
2. Verify port number:
   - Paper TWS: 7497
   - Live TWS: 7496
   - Paper Gateway: 4002
   - Live Gateway: 4001
3. Check API settings (see Step 2)
4. Restart TWS/Gateway
5. Check firewall isn't blocking connection

---

### No Market Data

**Problem:** `No market data received`

**Solutions:**
1. **Market Hours:** Trading hours are 9:30 AM - 4:00 PM ET
2. **Subscription:** Check Account → Market Data Subscriptions
3. **Symbol:** Verify symbol exists and is tradable

---

### Model Loading Errors

**Problem:** `Failed to load model for AAPL`

**Solutions:**
1. Check model files exist:
   ```bash
   ls models/lstm_mtf_v2/AAPL_mtf_v2.keras
   ls models/lstm_mtf_v2/AAPL_mtf_v2_scaler.pkl
   ```

2. If missing, retrain:
   ```bash
   python EASY_MTF_TRAINER_V2.py
   ```

3. Check model directory path in script

---

### Orders Not Placing

**Problem:** Orders placed but not filling

**Solutions:**
1. **Market Orders:** Should fill instantly during market hours
2. **Paper Trading:** Check paper account has funds
3. **Check Order Status:** Look for rejection messages
4. **Position Limits:** May have hit daily trade limit

---

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'EASY_MTF_TRAINER_V2'`

**Solution:** The script tries to import feature functions. Either:

**Option A:** Copy feature functions into the connector file

**Option B:** Ensure EASY_MTF_TRAINER_V2.py is in same directory

**Option C:** Create a separate features module

---

## 🎯 Best Practices

### 1. Start Small
- **Paper trade for 1-2 weeks** before going live
- Start with **1 symbol** (AAPL or TSLA)
- Use **small position sizes** (10-50 shares)

### 2. Monitor Closely
- **First day:** Watch constantly
- **First week:** Check every hour
- **After 2 weeks:** Can check less frequently

### 3. Track Performance
- Keep a trading log
- Compare to backtest expectations
- Expected: 40-70% of backtest returns

### 4. Risk Management
- Never risk more than 1-2% per trade
- Set daily loss limits
- Use stop losses
- Diversify across symbols

### 5. Market Conditions
- **Best times:** 9:30-11:00 AM, 2:00-4:00 PM ET (high volume)
- **Avoid:** First/last 5 minutes (volatile)
- **Earnings:** Skip days with earnings announcements

---

## 📈 Expected Performance

### Realistic Expectations

Your backtests showed:
- AAPL: +25% return
- TSLA: +69% return

**In live trading, expect 40-70% of backtest:**
- AAPL: +10-18% realistic
- TSLA: +28-48% realistic

### Why Lower?

1. **Slippage:** Real orders have worse fills
2. **Market Impact:** Your orders affect price
3. **Latency:** Delays in execution
4. **Market Conditions:** Market may be different

### Success Metrics

**After 2 weeks paper trading:**

| Metric | Target | Status |
|--------|--------|--------|
| Win Rate | >55% | ✅ Good |
| Avg Return/Trade | >0.3% | ✅ Good |
| Total Return | >5% | ✅ Good |
| Max Drawdown | <10% | ✅ Good |
| Sharpe Ratio | >1.5 | ✅ Good |

**If you hit these targets → Ready for live with small size!**

---

## 🚦 Go-Live Checklist

Before trading real money:

- [ ] Paper traded for 2+ weeks
- [ ] Win rate > 55%
- [ ] Total return > 5%
- [ ] Understand all code
- [ ] Tested emergency stop (Ctrl+C)
- [ ] Have monitoring alerts
- [ ] Risk < 2% per trade
- [ ] Started with ONE symbol
- [ ] Position size < $1000
- [ ] Documented strategy
- [ ] Comfortable with max loss

---

## 🆘 Emergency Procedures

### Stop Trading Immediately

**Press `Ctrl+C` in terminal**

The bot will:
1. Stop checking for signals
2. Close all open positions
3. Disconnect from IBKR

### Manual Override

If bot isn't responding:

1. **Log into TWS/Gateway**
2. **Portfolio → Positions**
3. **Right-click position → Close Position**

### Kill Switch

Create this script: `emergency_stop.py`

```python
from ibkr_live_trading_connector import MTFModelTrader

trader = MTFModelTrader()
trader.connect()

# Close all positions
for symbol in trader.active_positions:
    pos = trader.active_positions[symbol]
    trader.place_order(symbol, 'SELL', pos['quantity'])
    print(f"Closed {symbol}")

trader.disconnect()
```

Run with: `python emergency_stop.py`

---

## 📞 Support Resources

### IBKR Support
- **Phone:** 877-442-2757 (US)
- **Chat:** https://www.interactivebrokers.com/en/support/chat.php
- **API Docs:** https://interactivebrokers.github.io/tws-api/

### Your Models
- **Training:** `python EASY_MTF_TRAINER_V2.py`
- **Testing:** `python validate_ibkr_connection.py`
- **Live Trading:** `python ibkr_live_trading_connector.py`

---

## 🎉 You're Ready!

You now have:

✅ Profitable MTF models (+25-69% backtest)
✅ IBKR integration code
✅ Connection validator
✅ Complete documentation

### Next Steps:

1. **Run validator** - Make sure everything connects
2. **Paper trade 2 weeks** - Verify performance
3. **Start small live** - 1 symbol, tiny size
4. **Scale gradually** - Add symbols, increase size

**Good luck and trade safely!** 📈💰

---

**Questions? Issues? Need help?**

Check the troubleshooting section or review the code comments in the Python files.
