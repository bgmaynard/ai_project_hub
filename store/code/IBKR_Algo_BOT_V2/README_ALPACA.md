# AI Trading Hub - Alpaca Edition

Complete AI-powered trading platform integrated with Alpaca Markets API.

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
.\START_ALPACA_HUB.ps1 -InstallDependencies
```

### 2. Configure Alpaca API Keys

Edit the `.env` file and add your Alpaca API credentials:

```env
BROKER_TYPE=alpaca
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_PAPER=true
```

### 3. Train AI Model (Optional but Recommended)

```powershell
.\START_ALPACA_HUB.ps1 -TrainModel -TrainSymbol "SPY"
```

### 4. Start Dashboard

```powershell
.\START_ALPACA_HUB.ps1
```

Then open: http://localhost:9100/dashboard

## 📊 Features

### Market Data
- ✅ Real-time quotes from Alpaca
- ✅ Historical OHLCV bars
- ✅ Multi-symbol snapshot data
- ✅ Market snapshots with daily stats

### AI Predictions
- ✅ LightGBM-based prediction model
- ✅ 40+ technical indicators
- ✅ Train on any symbol
- ✅ Batch predictions for watchlists
- ✅ Real-time signal generation

### Trading
- ✅ Market orders
- ✅ Limit orders
- ✅ Position management
- ✅ Order tracking
- ✅ Account information

### Auto-Trading
- ✅ Automated signal scanning
- ✅ Position management
- ✅ Risk controls
- ✅ Daily trade limits

## 🎯 Available Endpoints

### Dashboard & UI
- `http://localhost:9100/` - API root
- `http://localhost:9100/dashboard` - Trading dashboard
- `http://localhost:9100/docs` - API documentation

### Health & Config
- `GET /api/health` - System health check
- `GET /api/config` - System configuration

### Market Data
- `GET /api/market/quote/{symbol}` - Latest quote
- `GET /api/market/snapshot/{symbol}` - Market snapshot
- `POST /api/market/bars` - Historical bars
- `GET /api/market/multi-quote?symbols=AAPL,TSLA` - Multiple quotes

### AI Predictions
- `POST /api/ai/train` - Train model
- `POST /api/ai/predict` - Get prediction
- `GET /api/ai/model-info` - Model information
- `GET /api/ai/batch-predict?symbols=AAPL,TSLA` - Batch predictions

### Trading (Alpaca)
- `GET /api/alpaca/status` - Connection status
- `GET /api/alpaca/account` - Account info
- `GET /api/alpaca/positions` - Open positions
- `GET /api/alpaca/orders` - Orders
- `POST /api/alpaca/place-order` - Place order
- `DELETE /api/alpaca/positions/{symbol}` - Close position

### Watchlist
- `GET /api/watchlist` - Default watchlist with quotes

## 🤖 Auto-Trading Bot

Start the automated trading bot:

```powershell
.\START_ALPACA_HUB.ps1 -AutoTrader
```

The auto-trader will:
1. Scan your watchlist for AI signals
2. Execute trades when confidence threshold is met
3. Manage positions automatically
4. Respect daily trade limits

Configuration in `alpaca_ai_trader.py`:
- `confidence_threshold`: Minimum confidence (default: 0.15 = 15%)
- `max_positions`: Maximum concurrent positions (default: 3)
- `max_daily_trades`: Daily trade limit (default: 10)
- `position_size`: Shares per trade (default: 1)

## 📁 Project Structure

```
IBKR_Algo_BOT_V2/
├── config/
│   └── broker_config.py          # Unified broker configuration
├── ai/
│   └── alpaca_ai_predictor.py    # AI predictor using Alpaca data
├── ui/
│   └── complete_platform.html    # Trading dashboard UI
├── alpaca_integration.py         # Alpaca connector
├── alpaca_market_data.py         # Market data provider
├── alpaca_api_routes.py          # Alpaca API routes
├── alpaca_dashboard_api.py       # Main API server
├── alpaca_ai_trader.py           # Auto-trading bot
├── START_ALPACA_HUB.ps1         # Startup script
├── requirements.txt              # Python dependencies
└── .env                          # Configuration
```

## 🔧 Advanced Usage

### Train Model on Multiple Symbols

```python
from ai.alpaca_ai_predictor import AlpacaAIPredictor

predictor = AlpacaAIPredictor()

# Train on SPY
predictor.train("SPY")

# Get predictions
pred = predictor.predict("AAPL")
print(f"Signal: {pred['signal']}, Confidence: {pred['confidence']}")
```

### Custom Market Data Queries

```python
from alpaca_market_data import get_alpaca_market_data
from datetime import datetime, timedelta

market_data = get_alpaca_market_data()

# Get 1 year of daily data
start = datetime.now() - timedelta(days=365)
bars = market_data.get_historical_bars(
    symbol="AAPL",
    timeframe="1Day",
    start=start
)

print(bars.head())
```

### Manual Trading via API

```python
from alpaca_integration import get_alpaca_connector

connector = get_alpaca_connector()

# Place market order
order = connector.place_market_order(
    symbol="AAPL",
    quantity=10,
    side="BUY"
)

print(f"Order placed: {order['order_id']}")
```

## 🛠️ Troubleshooting

### "Alpaca connection failed"
- Check your API keys in `.env`
- Ensure you're using paper trading keys (start with `PK`)
- Verify your internet connection

### "Model not trained"
- Run: `.\START_ALPACA_HUB.ps1 -TrainModel`
- Or use the dashboard AI Control Panel to train

### "No data from Alpaca"
- Check if symbol is valid
- Ensure market hours (or use data from recent trading day)
- Check Alpaca service status

### Port already in use
- Change port in `alpaca_dashboard_api.py`:
  ```python
  uvicorn.run(app, host="0.0.0.0", port=9100)
  ```

## 📝 API Examples

### Get Quote
```bash
curl http://localhost:9100/api/market/quote/AAPL
```

### Train Model
```bash
curl -X POST http://localhost:9100/api/ai/train \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "test_size": 0.2}'
```

### Get Prediction
```bash
curl -X POST http://localhost:9100/api/ai/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "timeframe": "1Day"}'
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

## 🔐 Security Notes

- **Never commit your `.env` file to version control**
- Use paper trading keys for testing
- Keep your API keys secure
- Review all trades before going live

## 🎓 Learning Resources

- [Alpaca API Documentation](https://alpaca.markets/docs/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review API documentation at http://localhost:9100/docs
3. Check Alpaca service status

## 🚦 Next Steps

1. ✅ Train your AI model on key symbols
2. ✅ Test predictions in the dashboard
3. ✅ Paper trade to verify system
4. ✅ Monitor performance
5. ✅ Refine and optimize

---

**Happy Trading! 🚀📈**
