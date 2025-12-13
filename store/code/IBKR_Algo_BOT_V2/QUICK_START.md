# Quick Start Guide - Alpaca Trading Hub

## ✅ System Status: READY

Your Alpaca trading system is fully configured and tested!

```
[OK] Alpaca Connected - Account: PA3S5IDP6VH1
[OK] Buying Power: $284,828.15
[OK] Portfolio Value: $142,897.46
[OK] Market Data Working
[OK] AI Model Loaded (54.95% accuracy)
```

## 🚀 Start Trading Dashboard

```powershell
.\START_ALPACA_HUB.ps1
```

Then open in your browser:
- **Dashboard:** http://localhost:9100/dashboard
- **API Docs:** http://localhost:9100/docs

## 🧪 Quick Test

Run a quick connectivity test:

```powershell
.\quick_test.ps1
```

Or the full test suite:

```powershell
python test_alpaca_setup.py
```

## 🎯 Common Commands

### Start the Dashboard
```powershell
.\START_ALPACA_HUB.ps1
```

### Train AI Model on a Symbol
```powershell
.\START_ALPACA_HUB.ps1 -TrainModel -TrainSymbol "AAPL"
```

### Run Auto-Trader Bot
```powershell
.\START_ALPACA_HUB.ps1 -AutoTrader
```

### Skip Pre-flight Checks (faster startup)
```powershell
.\START_ALPACA_HUB.ps1 -SkipChecks
```

### Install/Update Dependencies
```powershell
.\START_ALPACA_HUB.ps1 -InstallDependencies
```

## 📊 What You Can Do

### 1. View Live Market Data
- Real-time quotes from Alpaca
- TradingView charts
- Multiple timeframes

### 2. Get AI Predictions
- Train models on any symbol
- Get buy/sell signals
- See confidence levels
- Batch predictions on watchlists

### 3. Place Trades
- Market orders
- Limit orders
- View positions
- Track order history

### 4. Auto-Trade
- Automated signal scanning
- Position management
- Risk controls
- Daily trade limits

## 🔍 Dashboard Features

Once the dashboard is running at http://localhost:9100/dashboard:

1. **Market Data Panel** - Live quotes and charts
2. **AI Control Panel** - Train models and view predictions
3. **Trading Panel** - Place orders and manage positions
4. **Bot Control** - Configure and run auto-trader
5. **Worklist** - Track symbols of interest
6. **Scanner** - Find trading opportunities

## 📚 API Endpoints

Interactive docs available at: http://localhost:9100/docs

Key endpoints:
- `GET /api/market/quote/{symbol}` - Get quote
- `POST /api/ai/predict` - Get AI prediction
- `POST /api/alpaca/place-order` - Place order
- `GET /api/alpaca/positions` - View positions
- `GET /api/alpaca/account` - Account info

## 🎓 Example API Calls

### Get Quote
```bash
curl http://localhost:9100/api/market/quote/AAPL
```

### Get AI Prediction
```bash
curl -X POST http://localhost:9100/api/ai/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
```

### Place Order
```bash
curl -X POST http://localhost:9100/api/alpaca/place-order \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "action": "BUY",
    "quantity": 1,
    "order_type": "MKT"
  }'
```

## ⚙️ Configuration

Your system is configured in `.env`:

```env
BROKER_TYPE=alpaca
ALPACA_API_KEY=PKNXPP647QUNKNPGPGM7ZXFALX
ALPACA_SECRET_KEY=[configured]
ALPACA_PAPER=true
```

## 🔧 Troubleshooting

### Script Won't Run
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Connection Issues
- Check your internet connection
- Verify API keys in `.env`
- Ensure you're using paper trading keys (start with `PK`)

### Port Already in Use
- Close other instances of the server
- Or change the port in `alpaca_dashboard_api.py` (line 347)

### Module Import Errors
```powershell
.\START_ALPACA_HUB.ps1 -InstallDependencies
```

## 📖 Documentation

- **Complete Guide:** `README_ALPACA.md`
- **Migration Details:** `ALPACA_MIGRATION_SUMMARY.md`
- **This Guide:** `QUICK_START.md`

## 🎯 Next Steps

1. ✅ **Test the Dashboard**
   - Run: `.\START_ALPACA_HUB.ps1`
   - Open: http://localhost:9100/dashboard
   - Check live market data

2. ✅ **Get AI Predictions**
   - Train a model if needed
   - View predictions on watchlist
   - Test different symbols

3. ✅ **Try Paper Trading**
   - Place a test order
   - Monitor positions
   - Track performance

4. ✅ **Explore Auto-Trading**
   - Review settings in `alpaca_ai_trader.py`
   - Start with conservative thresholds
   - Monitor activity closely

## 📞 Need Help?

- API Documentation: http://localhost:9100/docs (when running)
- Test Connection: `.\quick_test.ps1`
- Full Test Suite: `python test_alpaca_setup.py`

---

**System Ready!** 🚀 Happy Trading!
