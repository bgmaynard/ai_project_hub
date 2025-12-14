# 🚀 AI Router Module Installation Guide

## 📦 What's Included

- `server/ai_router.py` - Complete AI endpoints with:
  - `/api/ai/predict` - Real-time predictions
  - `/api/ai/predict/last` - Last prediction from CSV
  - `/api/ai/predict/history` - Prediction history with filtering
  - `/api/ai/train` - Model training endpoint
  - `/api/ai/backtest` - Backtesting endpoint
  - `/api/ai/status` - AI module status

- `server/__init__.py` - Python package configuration

## 🔧 Installation Steps

### Step 1: Copy Files to Your Repository

```powershell
# Navigate to your project directory
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT

# Create server directory if it doesn't exist
New-Item -ItemType Directory -Force -Path "server"

# Copy the files
# (Copy server/ai_router.py and server/__init__.py to this location)
```

### Step 2: Update dashboard_api.py

Add these lines after line 60 (where `app = FastAPI(...)` is):

```python
# === AI Router Integration ===
# Mount AI router BEFORE the if __name__ == "__main__" block
try:
    from server.ai_router import router as ai_router
    app.include_router(ai_router)
    print("✅ AI Router mounted at /api/ai/*")
except ImportError as e:
    print(f"⚠️ Could not mount AI router: {e}")
```

**CRITICAL:** This must be added **BEFORE** line 520 (the `if __name__ == "__main__":` block)

### Step 3: Verify Installation

```powershell
# Start the server
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT
python dashboard_api.py
```

You should see:
```
✅ AI Router mounted at /api/ai/*
🚀 Starting IBKR Trading Bot Dashboard API on http://127.0.0.1:9101
```

### Step 4: Test the Endpoints

```powershell
# Test status
Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/ai/status" | ConvertTo-Json

# Test prediction
$body = @{symbol="SPY"} | ConvertToJson
Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/ai/predict" -Method POST -Body $body -ContentType "application/json" | ConvertTo-Json

# Get last prediction
Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/ai/predict/last" | ConvertTo-Json

# Get history
Invoke-RestMethod -Uri "http://127.0.0.1:9101/api/ai/predict/history?symbol=SPY&limit=10" | ConvertTo-Json
```

## 📊 Expected Response Examples

### /api/ai/status
```json
{
  "status": "operational",
  "ai_available": true,
  "version": "1.0.0",
  "endpoints": {
    "predict": "/api/ai/predict",
    "last_prediction": "/api/ai/predict/last",
    "history": "/api/ai/predict/history",
    "train": "/api/ai/train",
    "backtest": "/api/ai/backtest"
  }
}
```

### /api/ai/predict
```json
{
  "symbol": "SPY",
  "signal": "bullish",
  "prob_up": 0.6234,
  "confidence": 0.75,
  "timestamp": "2025-11-03T02:45:00Z",
  "features_used": ["price_to_vwap", "momentum", "volatility"]
}
```

### /api/ai/predict/history
```json
{
  "predictions": [
    {
      "symbol": "SPY",
      "signal": "bullish",
      "prob_up": 0.6234,
      "confidence": 0.75,
      "timestamp": "2025-11-03T02:45:00Z"
    },
    {
      "symbol": "AAPL",
      "signal": "neutral",
      "prob_up": 0.5012,
      "confidence": 0.68,
      "timestamp": "2025-11-03T02:40:00Z"
    }
  ],
  "total": 2,
  "symbol_filter": null
}
```

## 🐛 Troubleshooting

### "Module 'server' not found"
- Ensure `server/__init__.py` exists
- Check that you're running from `IBKR_Algo_BOT` directory
- Verify PYTHONPATH includes current directory

### "AI Router mounted" not showing
- Check mount code is BEFORE `if __name__ == "__main__":`
- Restart the server after changes
- Check for syntax errors in ai_router.py

### Predictions CSV not creating
- Ensure `logs/` directory exists
- Check write permissions
- First prediction will create the file automatically

## ✅ Commit to GitHub

Once everything works:

```powershell
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT

# Check status
git status

# Stage files
git add server/ai_router.py
git add server/__init__.py
git add dashboard_api.py

# Commit
git commit -m "feat: add AI router with history endpoint

- Created server/ai_router.py with 5 endpoints
- Added prediction history with CSV logging
- Mounted router in dashboard_api.py
- All endpoints tested and operational"

# Push
git push origin feat/unified-claude-chatgpt-2025-10-31
```

## 🎯 Next Steps

After installation:
1. ✅ Test all endpoints
2. ✅ Verify predictions are logging to CSV
3. ✅ Update UI to display prediction history
4. ✅ Wire train endpoint to EnhancedAIPredictor
5. ✅ Wire backtest endpoint to Backtester

## 📝 Integration with Existing AI Modules

The router is designed to work with your existing AI modules:

- **ai/ai_predictor.py** - Automatically loaded for `/predict` and `/train`
- **ai/backtester.py** - Automatically loaded for `/backtest`
- Falls back to mock responses if modules unavailable

## 🔒 Security Note

The AI endpoints are currently open. For production:
- Add API key protection similar to other endpoints
- Use `Depends(require_api_key)` on sensitive endpoints
- Consider rate limiting for prediction requests
