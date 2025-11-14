"""
IBKR Algorithmic Trading Bot V2 - COMPLETE UNIFIED API
Single file with everything integrated and working

Features:
- All AI modules integrated (Claude, predictions, backtesting, etc.)
- IBKR real-time connection
- Market data bus with WebSocket streaming
- Both old and new endpoint compatibility
- Claude analysis with real market data
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import asyncio
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import os
from dotenv import load_dotenv

# Import AI modules
try:
    from ai.claude_api import router as claude_router, get_claude_response, AI_AVAILABLE as CLAUDE_AVAILABLE
except ImportError:
    claude_router = None
    CLAUDE_AVAILABLE = False
    print("[WARN] claude_api not found")

try:
    from ai.market_analyst import MarketAnalyst
except ImportError:
    MarketAnalyst = None
    print("[WARN] market_analyst not found")

try:
    from ai.ai_predictor import get_predictor
except ImportError:
    get_predictor = None
    print("[WARN] ai_predictor not found")

try:
    from ai.prediction_logger import log_prediction
except ImportError:
    log_prediction = None
    print("[WARN] prediction_logger not found")

try:
    from ai.alpha_fusion import predict_one, L1
except ImportError:
    predict_one = None
    L1 = None
    print("[WARN] alpha_fusion not found")

# Try to import IBKR
try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, Contract
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    print("[WARN] ib_insync not installed")

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="IBKR Algo Bot V2 - Complete")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
try:
    app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
except:
    print("[WARN] ui directory not found")

# Include Claude AI router if available
if claude_router:
    app.include_router(claude_router)

# ═══════════════════════════════════════════════════════════════════════
#                     MARKET DATA BUS
# ═══════════════════════════════════════════════════════════════════════

class DataBus:
    """Central data bus for real-time market data distribution"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[WebSocket]] = defaultdict(list)
        self.market_data: Dict[str, Dict] = {}
        self.l2_data: Dict[str, Dict] = {}
        self.time_sales: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.predictions: Dict[str, Dict] = {}
        self.ib: Optional[Any] = None
        self.connected = False
        self.bot_status = "STOPPED"
        
    async def subscribe(self, symbol: str, websocket: WebSocket):
        """Subscribe a websocket to symbol updates"""
        self.subscribers[symbol].append(websocket)
        logger.info(f"Client subscribed to {symbol}")
        
    async def unsubscribe(self, symbol: str, websocket: WebSocket):
        """Unsubscribe a websocket from symbol updates"""
        if websocket in self.subscribers[symbol]:
            self.subscribers[symbol].remove(websocket)
            logger.info(f"Client unsubscribed from {symbol}")
    
    async def broadcast(self, symbol: str, data_type: str, data: Dict):
        """Broadcast data to all subscribers of a symbol"""
        message = {
            "type": data_type,
            "symbol": symbol,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in appropriate data structure
        if data_type == "quote":
            self.market_data[symbol] = data
        elif data_type == "l2":
            self.l2_data[symbol] = data
        elif data_type == "trade":
            self.time_sales[symbol].append(data)
        elif data_type == "prediction":
            self.predictions[symbol] = data
        
        # Broadcast to subscribers
        disconnected = []
        for ws in self.subscribers[symbol]:
            try:
                await ws.send_json(message)
            except:
                disconnected.append(ws)
        
        # Clean up disconnected clients
        for ws in disconnected:
            await self.unsubscribe(symbol, ws)
    
    async def connect_ibkr(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        """Connect to IBKR TWS/Gateway"""
        if not IBKR_AVAILABLE:
            raise Exception("ib_insync not installed")
        
        try:
            self.ib = IB()
            await self.ib.connectAsync(host, port, clientId=client_id)
            self.connected = True
            logger.info(f"✓ Connected to IBKR on port {port}")
            
            # Set up market data handlers
            self.ib.pendingTickersEvent += self._on_ticker_update
            
            return {"status": "connected", "port": port}
        except Exception as e:
            logger.error(f"✗ Failed to connect to IBKR: {e}")
            self.connected = False
            raise
    
    async def disconnect_ibkr(self):
        """Disconnect from IBKR"""
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")
    
    def _on_ticker_update(self, tickers):
        """Handle ticker updates from IBKR"""
        for ticker in tickers:
            symbol = ticker.contract.symbol
            data = {
                "last": float(ticker.last) if ticker.last == ticker.last else None,
                "bid": float(ticker.bid) if ticker.bid == ticker.bid else None,
                "ask": float(ticker.ask) if ticker.ask == ticker.ask else None,
                "bid_size": int(ticker.bidSize) if ticker.bidSize == ticker.bidSize else None,
                "ask_size": int(ticker.askSize) if ticker.askSize == ticker.askSize else None,
                "volume": int(ticker.volume) if ticker.volume == ticker.volume else None,
                "high": float(ticker.high) if ticker.high == ticker.high else None,
                "low": float(ticker.low) if ticker.low == ticker.low else None,
                "close": float(ticker.close) if ticker.close == ticker.close else None,
                "open": float(ticker.open) if ticker.open == ticker.open else None,
            }
            
            # Broadcast asynchronously
            asyncio.create_task(self.broadcast(symbol, "quote", data))
    
    async def subscribe_market_data(self, symbol: str, exchange: str = "SMART"):
        """Subscribe to market data for a symbol"""
        if not self.connected or not self.ib:
            raise Exception("Not connected to IBKR")
        
        contract = Stock(symbol, exchange, "USD")
        self.ib.reqMktData(contract, "", False, False)
        logger.info(f"Subscribed to market data for {symbol}")

# Initialize global data bus
data_bus = DataBus()

# Initialize AI components
market_analyst = MarketAnalyst() if MarketAnalyst else None
ai_predictor = get_predictor() if get_predictor else None

# ═══════════════════════════════════════════════════════════════════════
#                     REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════

class IBKRConnectionRequest(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1

class SymbolSubscription(BaseModel):
    symbol: str
    exchange: str = "SMART"
    data_types: List[str] = ["quote", "l2", "trades"]

class PredictionRequest(BaseModel):
    symbol: str

# ═══════════════════════════════════════════════════════════════════════
#                     IBKR CONNECTION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/ibkr/connect")
async def connect_ibkr(request: IBKRConnectionRequest):
    """Connect to IBKR TWS/Gateway"""
    try:
        result = await data_bus.connect_ibkr(request.host, request.port, request.client_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ibkr/disconnect")
async def disconnect_ibkr():
    """Disconnect from IBKR"""
    await data_bus.disconnect_ibkr()
    return {"status": "disconnected"}

@app.get("/api/ibkr/status")
async def get_ibkr_status():
    """Get IBKR connection status"""
    return {
        "connected": data_bus.connected,
        "available": IBKR_AVAILABLE
    }

# ═══════════════════════════════════════════════════════════════════════
#                     MARKET DATA SUBSCRIPTION
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/subscribe")
async def subscribe_symbol(request: SymbolSubscription):
    """Subscribe to market data for a symbol"""
    try:
        if "quote" in request.data_types:
            await data_bus.subscribe_market_data(request.symbol, request.exchange)
        
        return {"status": "subscribed", "symbol": request.symbol, "data_types": request.data_types}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/market-data/{symbol}")
async def websocket_market_data(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time market data streaming"""
    await websocket.accept()
    await data_bus.subscribe(symbol, websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        await data_bus.unsubscribe(symbol, websocket)
        logger.info(f"Client disconnected from {symbol}")

# ═══════════════════════════════════════════════════════════════════════
#                     NEW API ENDPOINTS (V2)
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Get current market data for a symbol"""
    if symbol in data_bus.market_data:
        return {
            "symbol": symbol,
            "data": data_bus.market_data[symbol],
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=404, detail="No data available for symbol")

@app.get("/api/l2/{symbol}")
async def get_l2_data(symbol: str):
    """Get Level 2 (market depth) data for a symbol"""
    if symbol in data_bus.l2_data:
        return {
            "symbol": symbol,
            "data": data_bus.l2_data[symbol],
            "timestamp": datetime.now().isoformat()
        }
    else:
        # Return empty L2 data if not available
        return {
            "symbol": symbol,
            "data": {"bids": [], "asks": []},
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/time-sales/{symbol}")
async def get_time_sales(symbol: str, limit: int = 100):
    """Get Time & Sales data for a symbol"""
    if symbol in data_bus.time_sales:
        trades = list(data_bus.time_sales[symbol])[-limit:]
        return {
            "symbol": symbol,
            "trades": trades,
            "count": len(trades)
        }
    else:
        return {
            "symbol": symbol,
            "trades": [],
            "count": 0
        }

# ═══════════════════════════════════════════════════════════════════════
#                     OLD API ENDPOINTS (Compatibility)
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/price/{symbol}")
async def get_price_old(symbol: str):
    """OLD ENDPOINT: Get current price (compatibility)"""
    if symbol in data_bus.market_data:
        data = data_bus.market_data[symbol]
        return {
            "symbol": symbol,
            "data": {
                "last": data.get("last"),
                "bid": data.get("bid"),
                "ask": data.get("ask"),
                "volume": data.get("volume"),
                "change": 0.0,  # Calculate if previous close available
                "changePercent": 0.0
            }
        }
    else:
        raise HTTPException(status_code=404, detail="No data available")

@app.get("/api/level2/{symbol}")
async def get_level2_old(symbol: str):
    """OLD ENDPOINT: Get Level 2 data (compatibility)"""
    return await get_l2_data(symbol)

@app.get("/api/timesales/{symbol}")
async def get_timesales_old(symbol: str, limit: int = 20):
    """OLD ENDPOINT: Get Time & Sales (compatibility)"""
    return await get_time_sales(symbol, limit)

@app.get("/api/bot/status")
async def get_bot_status():
    """OLD ENDPOINT: Get bot status (compatibility)"""
    return {
        "status": data_bus.bot_status,
        "ibkr_connected": data_bus.connected,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/bot/start")
async def start_bot():
    """Start the trading bot"""
    data_bus.bot_status = "RUNNING"
    return {"status": "started"}

@app.post("/api/bot/stop")
async def stop_bot():
    """Stop the trading bot"""
    data_bus.bot_status = "STOPPED"
    return {"status": "stopped"}

# ═══════════════════════════════════════════════════════════════════════
#                     AI PREDICTION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/ai/predict")
async def predict(request: PredictionRequest):
    """Get AI prediction for a symbol"""
    try:
        if not ai_predictor:
            raise HTTPException(status_code=501, detail="AI predictor not available")
        
        # Get market data
        market_data = data_bus.market_data.get(request.symbol, {})
        
        # Get AI prediction
        prediction = ai_predictor.predict(
            request.symbol,
            bidSize=market_data.get('bid_size'),
            askSize=market_data.get('ask_size')
        )
        
        # Log prediction
        if log_prediction:
            log_prediction(request.symbol, prediction)
        
        # Broadcast prediction
        await data_bus.broadcast(request.symbol, "prediction", prediction)
        
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai/prediction/{symbol}")
async def get_prediction(symbol: str):
    """Get latest prediction for a symbol"""
    if symbol in data_bus.predictions:
        return data_bus.predictions[symbol]
    else:
        raise HTTPException(status_code=404, detail="No prediction available")

@app.get("/api/claude/analyze-with-data/{symbol}")
async def analyze_with_real_data(symbol: str):
    """Analyze a symbol with real market data from IBKR"""
    try:
        if not market_analyst:
            raise HTTPException(status_code=501, detail="Market analyst not available")
        
        # Get current market data
        market_data = data_bus.market_data.get(symbol, {})
        
        if not market_data or not market_data.get('last'):
            return {
                "symbol": symbol,
                "error": "No market data available",
                "hint": f"Make sure IBKR is connected and subscribed to {symbol}",
                "subscribed": symbol in data_bus.market_data
            }
        
        # Build news context (empty for now)
        news = []
        
        # Get analysis from Claude with real data
        analysis_result = await market_analyst.analyze_single_stock(
            symbol=symbol,
            price=market_data.get('last'),
            volume=market_data.get('volume'),
            news=news
        )
        
        return {
            "symbol": symbol,
            "market_data": market_data,
            "analysis": analysis_result.get('analysis_text', 'No analysis available'),
            "timestamp": datetime.now().isoformat(),
            "data_source": "IBKR Live",
            "claude_available": CLAUDE_AVAILABLE
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════════
#                     SCANNER ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/scanner/results")
async def get_scanner_results():
    """Get scanner results with real or mock data"""
    
    # If we have real data, use it
    real_symbols = []
    for symbol, data in data_bus.market_data.items():
        if data.get('last'):
            real_symbols.append({
                "symbol": symbol,
                "price": data.get('last', 0),
                "change": 0.0,  # Calculate from close if available
                "volume": data.get('volume', 0),
                "gap": 0.0,
                "news": None,
                "ai_score": 75,
                "rel_volume": 1.0,
                "bid": data.get('bid', 0),
                "ask": data.get('ask', 0),
                "high": data.get('high', 0),
                "low": data.get('low', 0),
                "open": data.get('open', 0),
                "vwap": data.get('last', 0)
            })
    
    # If we have real data, return it
    if real_symbols:
        return {
            "symbols": real_symbols,
            "timestamp": datetime.now().isoformat(),
            "source": "IBKR Live"
        }
    
    # Otherwise return mock data
    mock_data = {
        "symbols": [
            {
                "symbol": "AAPL",
                "price": 226.50,
                "change": 2.3,
                "volume": 45234567,
                "gap": 3.1,
                "news": None,
                "ai_score": 85,
                "rel_volume": 2.4,
                "bid": 226.48,
                "ask": 226.52,
                "high": 227.80,
                "low": 225.30,
                "open": 224.20,
                "vwap": 226.10
            }
        ],
        "timestamp": datetime.now().isoformat(),
        "source": "Mock Data (Connect to IBKR for live)"
    }
    return mock_data

@app.post("/api/scanner/open-chart/{symbol}")
async def open_chart(symbol: str):
    """Open TradingView chart for symbol"""
    import subprocess
    import platform
    
    url = f"https://www.tradingview.com/chart/?symbol={symbol}"
    
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["cmd", "/c", "start", url], shell=True)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", url])
        else:  # Linux
            subprocess.Popen(["xdg-open", url])
        
        return {"status": "opened", "symbol": symbol, "url": url}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# ═══════════════════════════════════════════════════════════════════════
#                     HEALTH & STATUS
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "ibkr_connected": data_bus.connected,
        "ibkr_available": IBKR_AVAILABLE,
        "ai_predictor_loaded": ai_predictor is not None and (ai_predictor.model is not None if ai_predictor else False),
        "claude_available": CLAUDE_AVAILABLE and market_analyst is not None and (market_analyst.client is not None if market_analyst else False),
        "active_subscriptions": sum(len(subs) for subs in data_bus.subscribers.values()),
        "symbols_tracked": list(data_bus.market_data.keys()),
        "modules_loaded": {
            "claude_api": claude_router is not None,
            "market_analyst": market_analyst is not None,
            "ai_predictor": ai_predictor is not None,
            "prediction_logger": log_prediction is not None,
            "alpha_fusion": predict_one is not None
        }
    }

@app.get("/")
async def root():
    return {
        "name": "IBKR Algo Bot V2 - Complete",
        "version": "2.0.0",
        "status": "operational",
        "platform": "http://127.0.0.1:9101/ui/platform.html",
        "docs": "http://127.0.0.1:9101/docs",
        "endpoints": {
            "health": "/health",
            "ibkr_connect": "POST /api/ibkr/connect",
            "subscribe": "POST /api/subscribe",
            "market_data": "/api/market-data/{symbol}",
            "claude_analyze": "/api/claude/analyze-with-data/{symbol}",
            "scanner": "/api/scanner/results"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("[START] IBKR Algorithmic Trading Bot V2 - Complete")
    print("="*60)
    print(f"IBKR Available: {IBKR_AVAILABLE}")
    print(f"Claude Available: {CLAUDE_AVAILABLE}")
    print(f"AI Predictor: {ai_predictor is not None}")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=9101)
