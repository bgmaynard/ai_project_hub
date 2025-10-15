# 🚀 Training & Backtesting Setup Guide

## What's New in This Update

✅ **Training Integration** - Train models directly from watchlist
✅ **Progress Monitoring** - Real-time training progress with logs
✅ **Automatic Backtesting** - Tests run after training completes
✅ **Trade Execution** - One-click trading from backtest results
✅ **Enhanced UI** - New Training and Backtesting tabs

---

## 📁 Files to Update

### 1. Backend API (`dashboard_api.py`)

Replace your existing `dashboard_api.py` with the new version that includes:
- `TrainingManager` class
- `BacktestManager` class
- Training endpoints (`/api/train/start`, `/api/train/status`)
- Backtest endpoints (`/api/backtest/status`, `/api/backtest/results`)
- Trade execution endpoint (`/api/trade/execute`)

**Location:** `C:\IBKR_Algo_BOT\dashboard_api.py`

### 2. Frontend UI (`frontend/src/App.jsx`)

Replace your existing `App.jsx` with the enhanced version that includes:
- Training tab with progress bars
- Backtesting tab with results table
- Symbol selection checkboxes in Watchlist
- Training configuration panel
- Execute Trade buttons

**Location:** `C:\IBKR_Algo_BOT\frontend\src\App.jsx`

---

## 🎯 Quick Start

### Step 1: Update Backend

```powershell
cd C:\IBKR_Algo_BOT

# Backup your existing file
Copy-Item dashboard_api.py dashboard_api.py.backup

# Replace with new version (copy from artifact above)
# Then start the backend
python dashboard_api.py
```

### Step 2: Update Frontend

```powershell
cd C:\IBKR_Algo_BOT\frontend

# Backup existing file
Copy-Item src/App.jsx src/App.jsx.backup

# Replace with new version (copy from artifact above)
# Then start frontend
npm run dev
```

### Step 3: Open Dashboard

Navigate to: **http://localhost:3000**

---

## 🎮 How to Use New Features

### Training Models from Watchlist

1. **Go to Watchlist Tab**
   - You'll see checkboxes next to each symbol

2. **Select Symbols to Train**
   - Check the symbols you want to train
   - Or check the header checkbox to select all

3. **Configure Training (Optional)**
   - Period: 1y, 2y, or 5y
   - Interval: 1h, 1d, or 1wk
   - Epochs: 50 (default)
   - Batch Size: 32 (default)

4. **Click "Train Selected"**
   - Training starts in background
   - Switch to Training tab to watch progress

5. **Monitor Progress**
   - Go to Training tab
   - See real-time progress bars
   - View training logs
   - Current symbol being trained

### Viewing Backtest Results

1. **After Training Completes**
   - Backtesting starts automatically
   - Go to Backtesting tab

2. **View Results Table**
   - See all symbols with:
     - Total Return %
     - Win Rate %
     - Sharpe Ratio
     - Max Drawdown %
     - Total Trades

3. **Sort by Performance**
   - Click column headers to sort (future feature)
   - Identify best performers

### Executing Trades

1. **From Backtesting Tab**
   - Find symbol with good results
   - Click "Execute Trade" button

2. **Confirm Trade**
   - Enter quantity (shares)
   - Confirm order

3. **Monitor Execution**
   - Check Positions tab
   - View in Activity Log

---

## 🔄 Complete Workflow

```
Scanner → Watchlist → Training → Backtesting → Trade Execution → Monitoring
```

### Step-by-Step Example

1. **Find Stocks**
   ```
   Scanner Tab → Run Scanner → Add to Watchlist
   ```

2. **Select and Train**
   ```
   Watchlist Tab → Check symbols → Configure → Train Selected
   ```

3. **Monitor Training**
   ```
   Training Tab → Watch progress bars → View logs
   ```

4. **Review Results**
   ```
   Backtesting Tab → Check performance metrics
   ```

5. **Execute Trade**
   ```
   Click "Execute Trade" → Enter quantity → Confirm
   ```

6. **Track Position**
   ```
   Positions Tab → See active trades → Monitor P&L
   ```

---

## 📊 Dashboard Layout

### New Tabs

| Tab | Purpose |
|-----|---------|
| **Overview** | System stats, active trainings, recent backtests |
| **Training** | Monitor active and completed trainings |
| **Backtesting** | View test results, execute trades |
| **Watchlist** | Select symbols, configure training |

### Overview Tab Features

- **4 Stat Cards**: Positions, Watchlist, Training, Backtests
- **Active Trainings Section**: Real-time progress
- **Recent Backtest Results**: Top 5 with quick trade buttons

### Training Tab Features

- **Active Trainings**: Live progress bars with logs
- **Completed Trainings**: History table with duration
- **Status Indicators**: Visual status (running, completed, failed)

### Backtesting Tab Features

- **Results Table**: Comprehensive performance metrics
- **Execute Trade Buttons**: One-click trading
- **Performance Indicators**: Color-coded returns

---

## 🔧 Integration with Existing Components

### Connects to Your Existing Code

The new features integrate with:

1. **EASY_MTF_TRAINER_V2.py**
   - Called when training starts
   - Runs `LSTMTrainingPipeline`

2. **ibkr_live_trading_connector.py**
   - Used for trade execution
   - Places orders via IBKR API

3. **Watchlist JSON Files**
   - Loads symbols from `dashboard_data/watchlists/`
   - Saves selected symbols for training

4. **LSTM Models**
   - Saved to `models/lstm_mtf_v2/`
   - Loaded for predictions

---

## 🎨 UI Improvements

### Visual Enhancements

✅ **Color-Coded Status**
- Green: Profitable, Running, Success
- Red: Loss, Stopped, Failed
- Purple: Training Active
- Orange: Backtesting

✅ **Progress Bars**
- Smooth animations
- Real-time updates via WebSocket
- Shows current symbol

✅ **Responsive Design**
- Works on multiple monitors
- Grid layout adapts to screen size

---

## 🚨 Important Notes

### Training Process

⚠️ **Training takes time**
- Depends on: number of symbols, epochs, data period
- Runs in background (non-blocking)
- Can start multiple trainings

### Backtesting

⚠️ **Auto-starts after training**
- Results appear in ~2-5 seconds per symbol
- Uses same data as training validation

### Trade Execution

⚠️ **Requires IBKR connection**
- Connect first via status card
- Market orders execute immediately
- Check Positions tab to confirm

---

## 📈 Example Training Session

```
1. Select Symbols
   ☑ AAPL
   ☑ TSLA
   ☑ NVDA

2. Configure Training
   Period: 2 years
   Interval: 1 hour
   Epochs: 50
   Batch Size: 32

3. Click "Train Selected (3)"
   → Training ID: train_1728756234
   → Progress: 0% → 33% → 66% → 100%
   → Duration: ~5-10 minutes

4. Automatic Backtesting
   → Backtest ID: backtest_1728756890
   → Results appear in Backtesting tab

5. Review Results
   AAPL: +15.5% return, 62.5% win rate
   TSLA: +8.2% return, 58.1% win rate
   NVDA: +22.3% return, 65.7% win rate

6. Execute Best Performer (NVDA)
   → Buy 100 shares
   → Order placed successfully
```

---

## 🐛 Troubleshooting

### Training Doesn't Start

**Check:**
- Backend is running (`python dashboard_api.py`)
- EASY_MTF_TRAINER_V2.py is in same directory
- Selected at least one symbol
- No Python errors in console

### No Backtest Results

**Check:**
- Training completed successfully
- Check Training tab for "completed" status
- Look in Backtesting tab (may take a few seconds)
- Check Activity Log for errors

### Execute Trade Button Disabled

**Check:**
- IBKR is connected (blue status card)
- TWS is running on port 7497
- API is enabled in TWS settings

### Progress Bar Not Updating

**Check:**
- Frontend is running (`npm run dev`)
- WebSocket connection active
- Refresh browser (Ctrl+Shift+R)

---

## 🎯 Next Enhancements (Future)

After this update, consider adding:

1. **Real-time Market Data**
   - Live prices in Watchlist
   - Real P&L updates

2. **Advanced Backtesting**
   - Custom date ranges
   - Strategy comparison
   - Equity curves

3. **Risk Management**
   - Position sizing calculator
   - Stop-loss automation
   - Portfolio limits

4. **Alerts & Notifications**
   - Training complete alerts
   - Trade execution confirmations
   - Email/SMS notifications

---

## 📞 Support

If you encounter issues:

1. Check browser console (F12) for errors
2. Check backend terminal for Python errors
3. Verify all files are updated
4. Try restarting both servers
5. Clear browser cache

---

## ✅ Success Checklist

Before you start:

- [ ] Backend updated with TrainingManager and BacktestManager
- [ ] Frontend updated with new Training and Backtesting tabs
- [ ] Both servers running (backend port 5000, frontend port 3000)
- [ ] IBKR TWS running on port 7497
- [ ] Dashboard loads at http://localhost:3000
- [ ] Can see all 8 tabs in navigation
- [ ] Watchlist shows checkboxes for symbol selection
- [ ] "Train Selected" button appears when symbols checked

You're ready to train models! 🚀

---

**Last Updated:** October 12, 2025
**Version:** 2.0 - Training & Backtesting Integration
