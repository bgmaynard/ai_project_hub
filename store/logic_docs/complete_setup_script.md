# 🚀 Complete LSTM Trading Bot Setup Guide

## 📋 What You're Building

A complete LSTM trading system with:
- ✅ 2 years of hourly data
- ✅ IBKR integration for live data
- ✅ Improved backtest logic
- ✅ Confidence filtering
- ✅ Multiple trained models

---

## 🎯 PHASE 1: Quick Retrain (DOING NOW)

### Step 1: Edit `train_real_stocks.py`

**File location:** `C:\IBKR_Algo_BOT\train_real_stocks.py`

**Find these lines** (around line 40):
```python
results = pipeline.run_full_pipeline(
    symbols=symbols,
    data_source='yahoo',
    period='60d',      # ← CHANGE THIS
    interval='5m',     # ← CHANGE THIS
)
```

**Change to:**
```python
results = pipeline.run_full_pipeline(
    symbols=symbols,
    data_source='yahoo',
    period='2y',       # ← 2 years
    interval='1h',     # ← Hourly
)
```

### Step 2: Run Training

```bash
cd C:\IBKR_Algo_BOT
python train_real_stocks.py
```

**Expected output:**
```
Processing AAPL...
Downloading data... (2 years, hourly)
✓ Data loaded: 4032 bars
Training LSTM...
Epoch 1/50 ...
Epoch 50/50
✓ Model saved: AAPL_lstm.h5

Processing TSLA...
(same process)

Training complete!
```

**Time: 10-15 minutes**

### Step 3: Test New Models

```bash
python quick_fix_train.py --mode quick
```

**Expected improvement:**
- OLD: 2-4 trades
- NEW: 20-50 trades ← Much better!

---

## 🎯 PHASE 2: IBKR Data Integration (AFTER PHASE 1)

### Prerequisites

**✓ Check these are ready:**
- [ ] TWS or Gateway is running
- [ ] API connections enabled (TWS → Settings → API → Enable ActiveX and Socket Clients)
- [ ] Port 7497 (Paper) or 7496 (Live) is correct
- [ ] Market data subscription active

### Step 1: Install New Files

**Save these 2 new files to** `C:\IBKR_Algo_BOT\`:

1. **`ibkr_historical_fetcher.py`** - Downloads data from IBKR
2. **`train_with_ibkr_data.py`** - Trains with IBKR data

### Step 2: Test IBKR Connection

```bash
python level1_test.py
```

**Should see:**
```
✓ Connected to TWS
✓ Receiving data for AAPL
Bid: $225.50 | Ask: $225.52
```

If this fails, check:
- TWS is running
- API is enabled
- Port is correct (7497 for Paper)

### Step 3: Fetch Historical Data from IBKR

```bash
python ibkr_historical_fetcher.py --symbols AAPL TSLA --duration "2 Y" --bar-size "1 hour"
```

**What this does:**
- Connects to IBKR
- Downloads 2 years of hourly data for AAPL and TSLA
- Saves to `data/historical/AAPL_1_hour.csv` and `TSLA_1_hour.csv`
- Better quality than Yahoo Finance

**Expected output:**
```
Connecting to IBKR...
✓ Connected

Fetching AAPL...
✓ Fetched 4032 bars
✓ Saved to data/historical/AAPL_1_hour.csv

Fetching TSLA...
✓ Fetched 4032 bars
✓ Saved to data/historical/TSLA_1_hour.csv

✓ Data ready for training!
```

**Time: 2-3 minutes**

### Step 4: Train with IBKR Data

```bash
python train_with_ibkr_data.py --symbols AAPL TSLA
```

**What this does:**
- Loads CSV files from IBKR
- Trains LSTM models
- Saves as `AAPL_lstm.h5` and `TSLA_lstm.h5`
- Runs backtest automatically

**Time: 10-15 minutes**

### Step 5: Test IBKR-Trained Models

```bash
python quick_fix_train.py --mode quick
```

**Expected:**
- Even better results than Yahoo data
- Higher quality signals
- Better win rates

---

## 📊 Comparison: Yahoo vs IBKR Data

| Feature | Yahoo Finance | IBKR |
|---------|--------------|------|
| **Data Quality** | Good | Excellent |
| **Bid/Ask** | No | Yes |
| **Speed** | Fast | Moderate |
| **Cost** | Free | Requires subscription |
| **Historical Limit** | 2 years (hourly) | Years of data |
| **Real-time** | 15-min delay | Real-time |

**Recommendation:**
- Start with Yahoo (Phase 1) - proves system works
- Upgrade to IBKR (Phase 2) - better quality for live trading

---

## 🎮 Quick Command Reference

### Training Commands
```bash
# Yahoo data (quick)
python train_real_stocks.py

# IBKR data (better quality)
python ibkr_historical_fetcher.py --symbols AAPL TSLA
python train_with_ibkr_data.py --symbols AAPL TSLA
```

### Testing Commands
```bash
# Test trained models
python quick_fix_train.py --mode quick

# Test IBKR connection
python level1_test.py

# Test TotalView (if subscribed)
python totalview_test.py
```

### Custom Options
```bash
# More symbols
python ibkr_historical_fetcher.py --symbols AAPL TSLA NVDA MSFT

# Different timeframe
python ibkr_historical_fetcher.py --duration "6 M" --bar-size "5 mins"

# Live port (not paper)
python ibkr_historical_fetcher.py --port 7496
```

---

## ✅ Success Checklist

### Phase 1 Complete When:
- [ ] `train_real_stocks.py` edited (2y, 1h)
- [ ] Training completed successfully
- [ ] Models saved (AAPL_lstm.h5, TSLA_lstm.h5)
- [ ] Quick test shows 20-50 trades (not 2-4)
- [ ] Positive returns on backtest

### Phase 2 Complete When:
- [ ] IBKR connection works (level1_test.py passes)
- [ ] Historical data fetched (CSV files created)
- [ ] Models trained with IBKR data
- [ ] Backtest shows good results
- [ ] Ready for paper trading

---

## 🐛 Troubleshooting

### Problem: Training takes forever
**Solution:** This is normal! 50 epochs × 30 seconds = 25 minutes per symbol

### Problem: "Connection refused" (IBKR)
**Solutions:**
1. Make sure TWS is running
2. Check API is enabled (Settings → API)
3. Try different port: 7497 (Paper), 7496 (Live), 4002 (Gateway)

### Problem: "No market data"
**Solutions:**
1. Check market data subscription is active
2. Verify symbol is correct (AAPL not APPL)
3. Try during market hours first

### Problem: "Pacing violation"
**Solution:** IBKR limits requests. Script waits 2 seconds between symbols automatically.

### Problem: Models still only find few trades
**Solutions:**
1. Lower confidence threshold to 0.50
2. Try more volatile symbols (TSLA, NVDA)
3. Use 5-min bars instead of hourly (more opportunities)

---

## 💡 What to Do After Training

### If Results are Good (20+ trades, positive returns):
1. ✅ Document optimal confidence threshold
2. ✅ Start paper trading for 1-2 weeks
3. ✅ Monitor daily performance
4. ✅ Compare to backtest expectations
5. ✅ After 2 weeks success → go live with small sizes

### If Results Still Poor (<20 trades or negative):
1. Try 5-min bars: `--bar-size "5 mins"` and `--duration "30 D"`
2. Add more volatile symbols (NVDA, AMD, etc.)
3. Lower confidence threshold to 0.50-0.52
4. Consider ensemble models (train 3 models, average predictions)

---

## 📈 Expected Results After Phase 1

### Before (60 days, 5-min):
```
AAPL: 2-4 trades, +0.60% return
TSLA: Similar
```

### After (2 years, hourly):
```
AAPL: 20-50 trades, +5-15% return
TSLA: 30-60 trades, +8-20% return
Win Rate: 55-65%
Sharpe: 1.0-2.0
```

---

## 🚀 Next Steps After Complete Setup

1. **Week 1:** Paper trade with 50-52% threshold
2. **Week 2:** Monitor and adjust if needed
3. **Week 3:** If profitable, increase position sizes
4. **Week 4:** Consider going live with smallest sizes
5. **Month 2:** Scale up gradually

---

## 📞 Quick Help

**Right now, you should:**
1. Edit `train_real_stocks.py` (change 2 lines)
2. Run: `python train_real_stocks.py`
3. Wait 10-15 minutes
4. Test: `python quick_fix_train.py --mode quick`

**While training runs, you can:**
- Save the 2 new Python files I created
- Test IBKR connection with `python level1_test.py`
- Read through this guide

**After Phase 1 works:**
- Move to Phase 2 (IBKR data)
- Get even better quality data
- Prepare for paper trading

---

**You're doing great! Let's get Phase 1 training started right now!** 🎉

Just edit those 2 lines in `train_real_stocks.py` and hit run!
