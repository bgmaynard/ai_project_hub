"""
IBKR Algo Bot - Complete Dashboard API Server
Provides market data, AI predictions, and bot control
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
from datetime import datetime
import random

# Import the AI router if available
try:
    from ai_router import router as ai_router
    AI_ROUTER_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: ai_router.py not found. AI endpoints will not be available.")
    AI_ROUTER_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="IBKR Algo Bot Dashboard",
    description="Real-time trading bot control and AI prediction interface",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the AI router if available
if AI_ROUTER_AVAILABLE:
    app.include_router(ai_router, prefix="/api/ai", tags=["AI Predictions"])
    print("✅ AI Router mounted at /api/ai")

# Get the project root directory
BASE_DIR = Path(__file__).parent.parent
UI_DIR = BASE_DIR / "ui"
DATA_DIR = BASE_DIR / "data"

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)

# Serve static files from ui directory
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
    print(f"✅ UI directory mounted: {UI_DIR}")

# ==================== ROOT & HEALTH ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "message": "IBKR Algo Bot Dashboard API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "ui": "/ui/prediction_history.html",
            "api_docs": "/docs",
            "health": "/health",
            "market_data": {
                "price": "/api/price/{symbol}",
                "level2": "/api/level2/{symbol}",
                "timesales": "/api/timesales/{symbol}"
            },
            "ai": {
                "predict": "/api/ai/predict",
                "status": "/api/ai/status",
                "history": "/api/ai/predict/history"
            },
            "bot": {
                "status": "/api/bot/status",
                "start": "/api/bot/start",
                "stop": "/api/bot/stop"
            }
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_router": AI_ROUTER_AVAILABLE,
        "ui_available": UI_DIR.exists()
    }

# ==================== MARKET DATA ENDPOINTS ====================

@app.get("/api/price/{symbol}")
async def get_price(symbol: str):
    """
    Get current price data for a symbol
    This is a placeholder - integrate with your actual IBKR data feed
    """
    # TODO: Replace with actual IBKR TWS/Gateway data feed
    # For now, returning mock data
    base_price = 100.0 + random.uniform(-10, 10)
    
    return {
        "symbol": symbol.upper(),
        "timestamp": datetime.now().isoformat(),
        "data": {
            "last": round(base_price, 2),
            "bid": round(base_price - 0.05, 2),
            "ask": round(base_price + 0.05, 2),
            "volume": random.randint(100000, 1000000),
            "change": round(random.uniform(-2, 2), 2),
            "changePercent": round(random.uniform(-2, 2), 2)
        },
        "source": "mock_data",
        "note": "Connect to IBKR TWS/Gateway for real data"
    }

@app.get("/api/level2/{symbol}")
async def get_level2(symbol: str):
    """
    Get Level 2 market depth data
    This is a placeholder - integrate with your actual IBKR data feed
    """
    # TODO: Replace with actual IBKR Level 2 data
    base_price = 100.0
    
    bids = []
    asks = []
    
    # Generate mock bid data
    for i in range(5):
        bids.append({
            "price": round(base_price - (i * 0.05), 2),
            "size": random.randint(100, 1000),
            "orders": random.randint(1, 5)
        })
    
    # Generate mock ask data
    for i in range(5):
        asks.append({
            "price": round(base_price + (i * 0.05), 2),
            "size": random.randint(100, 1000),
            "orders": random.randint(1, 5)
        })
    
    return {
        "symbol": symbol.upper(),
        "timestamp": datetime.now().isoformat(),
        "bids": bids,
        "asks": asks,
        "source": "mock_data",
        "note": "Connect to IBKR TWS/Gateway for real Level 2 data"
    }

@app.get("/api/timesales/{symbol}")
async def get_timesales(
    symbol: str,
    limit: int = Query(default=50, ge=1, le=500)
):
    """
    Get time & sales data (tick-by-tick trades)
    This is a placeholder - integrate with your actual IBKR data feed
    """
    # TODO: Replace with actual IBKR time & sales data
    trades = []
    base_price = 100.0
    
    for i in range(limit):
        timestamp = datetime.now().timestamp() - (i * 10)
        price = base_price + random.uniform(-1, 1)
        
        trades.append({
            "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
            "price": round(price, 2),
            "size": random.randint(100, 1000),
            "condition": random.choice(["@", "I", "T"])
        })
    
    return {
        "symbol": symbol.upper(),
        "timestamp": datetime.now().isoformat(),
        "trades": trades,
        "count": len(trades),
        "source": "mock_data",
        "note": "Connect to IBKR TWS/Gateway for real time & sales data"
    }

# ==================== BOT CONTROL ENDPOINTS ====================

@app.get("/api/bot/status")
async def get_bot_status():
    """Get current bot status"""
    # TODO: Integrate with your actual bot controller
    return {
        "status": "idle",
        "connected": False,
        "ibkr_connection": "disconnected",
        "account": None,
        "positions": [],
        "message": "Connect to IBKR TWS/Gateway to activate bot",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/bot/start")
async def start_bot():
    """Start the trading bot"""
    # TODO: Implement bot start logic
    return {
        "status": "starting",
        "message": "Bot start sequence initiated",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/bot/stop")
async def stop_bot():
    """Stop the trading bot"""
    # TODO: Implement bot stop logic
    return {
        "status": "stopping",
        "message": "Bot stop sequence initiated",
        "timestamp": datetime.now().isoformat()
    }

# ==================== UI SERVING ====================

@app.get("/ui/prediction_history.html")
async def serve_prediction_history():
    """Serve the prediction history UI"""
    ui_file = UI_DIR / "prediction_history.html"
    if ui_file.exists():
        return FileResponse(ui_file)
    else:
        raise HTTPException(
            status_code=404, 
            detail="Prediction history UI not found. Create ui/prediction_history.html"
        )

# ==================== MAIN ====================


@app.get("/api/orders")
async def get_orders():
    """Get current orders"""
    return {
        "orders": [],
        "count": 0,
        "message": "Connect to IBKR TWS/Gateway for real orders",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/positions")
async def get_positions():
    """Get current positions"""
    return {
        "positions": [],
        "count": 0,
        "message": "Connect to IBKR TWS/Gateway for real positions",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 IBKR Algo Bot Dashboard Server")
    print("=" * 70)
    print(f"📂 Base Directory: {BASE_DIR}")
    print(f"📂 UI Directory: {UI_DIR}")
    print(f"📂 Data Directory: {DATA_DIR}")
    print(f"🤖 AI Router: {'✅ Available' if AI_ROUTER_AVAILABLE else '❌ Not Available'}")
    print("=" * 70)
    print("🌐 Server URLs:")
    print("   - API Root:          http://127.0.0.1:9101/")
    print("   - API Docs:          http://127.0.0.1:9101/docs")
    print("   - Health Check:      http://127.0.0.1:9101/health")
    print("   - Prediction UI:     http://127.0.0.1:9101/ui/prediction_history.html")
    print("=" * 70)
    print("📊 Available Endpoints:")
    print("   Market Data:")
    print("   - GET  /api/price/{symbol}")
    print("   - GET  /api/level2/{symbol}")
    print("   - GET  /api/timesales/{symbol}")
    print("   Bot Control:")
    print("   - GET  /api/bot/status")
    print("   - POST /api/bot/start")
    print("   - POST /api/bot/stop")
    if AI_ROUTER_AVAILABLE:
        print("   AI Predictions:")
        print("   - POST /api/ai/predict")
        print("   - GET  /api/ai/predict/last")
        print("   - GET  /api/ai/predict/history")
        print("   - GET  /api/ai/status")
    print("=" * 70)
    print("⚠️  Note: Market data endpoints return mock data.")
    print("    Connect to IBKR TWS/Gateway for real-time data.")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=9101,
        log_level="info"
    )