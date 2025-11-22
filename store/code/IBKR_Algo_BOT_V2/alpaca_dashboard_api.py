"""
Unified Trading Dashboard API with Alpaca Integration
Main API server for AI Project Hub
"""
from dotenv import load_dotenv
load_dotenv()

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Broker configuration
from config.broker_config import get_broker_config, BrokerType

# Alpaca integration
from alpaca_integration import get_alpaca_connector
from alpaca_market_data import get_alpaca_market_data
from alpaca_api_routes import router as alpaca_router

# Compatibility routes for legacy UI
from compatibility_routes import router as compat_router

# AI Predictor
from ai.alpaca_ai_predictor import get_alpaca_predictor

# Setup logging
logging.basicConfig(
    handlers=[RotatingFileHandler('bot.log', maxBytes=10*1024*1024, backupCount=3)],
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Trading Hub - Alpaca Edition",
    description="AI-powered trading platform with Alpaca integration",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Alpaca router
app.include_router(alpaca_router, tags=["Alpaca Trading"])

# Include compatibility router for legacy UI endpoints
app.include_router(compat_router)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class TrainRequest(BaseModel):
    symbol: str
    test_size: float = 0.2


class PredictRequest(BaseModel):
    symbol: str
    timeframe: str = "1Day"


class MarketDataRequest(BaseModel):
    symbol: str
    timeframe: str = "1Day"
    limit: Optional[int] = 100


# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "AI Trading Hub",
        "version": "2.0.0",
        "broker": "Alpaca",
        "status": "operational"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    config = get_broker_config()
    connector = get_alpaca_connector()

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "broker": {
            "type": config.broker_type,
            "connected": connector.is_connected(),
            "paper_trading": config.alpaca.paper if config.is_alpaca() else False
        },
        "services": {
            "api": "operational",
            "ai_predictor": "operational",
            "market_data": "operational"
        }
    }


@app.get("/api/config")
async def get_config():
    """Get system configuration"""
    config = get_broker_config()
    return config.get_config_dict()


# ============================================================================
# MARKET DATA ENDPOINTS
# ============================================================================

@app.get("/api/market/quote/{symbol}")
async def get_quote(symbol: str):
    """Get latest quote for a symbol"""
    try:
        market_data = get_alpaca_market_data()
        quote = market_data.get_latest_quote(symbol.upper())

        if quote is None:
            raise HTTPException(status_code=404, detail=f"No quote data for {symbol}")

        return quote

    except Exception as e:
        logger.error(f"Error fetching quote: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/snapshot/{symbol}")
async def get_snapshot(symbol: str):
    """Get market snapshot for a symbol"""
    try:
        market_data = get_alpaca_market_data()
        snapshot = market_data.get_snapshot(symbol.upper())

        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"No snapshot data for {symbol}")

        return snapshot

    except Exception as e:
        logger.error(f"Error fetching snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/market/bars")
async def get_historical_bars(request: MarketDataRequest):
    """Get historical bars for a symbol"""
    try:
        market_data = get_alpaca_market_data()
        bars = market_data.get_historical_bars(
            symbol=request.symbol.upper(),
            timeframe=request.timeframe,
            limit=request.limit
        )

        if bars.empty:
            raise HTTPException(status_code=404, detail=f"No data for {request.symbol}")

        # Convert DataFrame to dict for JSON response
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "bars": bars.reset_index().to_dict(orient='records')
        }

    except Exception as e:
        logger.error(f"Error fetching bars: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/multi-quote")
async def get_multi_quote(symbols: str = Query(..., description="Comma-separated symbols")):
    """Get quotes for multiple symbols"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        market_data = get_alpaca_market_data()
        quotes = market_data.get_multiple_quotes(symbol_list)

        return quotes

    except Exception as e:
        logger.error(f"Error fetching multiple quotes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AI PREDICTION ENDPOINTS
# ============================================================================

@app.post("/api/ai/train")
async def train_model(request: TrainRequest):
    """Train AI prediction model"""
    try:
        predictor = get_alpaca_predictor()
        result = predictor.train(
            symbol=request.symbol.upper(),
            test_size=request.test_size
        )

        return result

    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/predict")
async def predict(request: PredictRequest):
    """Get AI prediction for a symbol"""
    try:
        predictor = get_alpaca_predictor()
        prediction = predictor.predict(
            symbol=request.symbol.upper(),
            timeframe=request.timeframe
        )

        return prediction

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai/model-info")
async def get_model_info():
    """Get information about the trained model"""
    try:
        predictor = get_alpaca_predictor()

        if predictor.model is None:
            return {
                "trained": False,
                "message": "No model trained yet"
            }

        # Get top features
        sorted_features = sorted(
            predictor.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "trained": True,
            "accuracy": predictor.accuracy,
            "training_date": predictor.training_date,
            "num_features": len(predictor.feature_names),
            "top_features": [
                {"name": name, "importance": float(importance)}
                for name, importance in sorted_features
            ],
            "data_source": "Alpaca"
        }

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai/batch-predict")
async def batch_predict(symbols: str = Query(..., description="Comma-separated symbols")):
    """Get AI predictions for multiple symbols"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        predictor = get_alpaca_predictor()

        results = []
        for symbol in symbol_list:
            try:
                prediction = predictor.predict(symbol)
                results.append(prediction)
            except Exception as e:
                logger.warning(f"Failed to predict {symbol}: {e}")
                results.append({
                    "symbol": symbol,
                    "error": str(e)
                })

        return results

    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WATCHLIST ENDPOINTS
# ============================================================================

@app.get("/api/watchlist")
async def get_watchlist():
    """Get default watchlist"""
    watchlist = [
        "SPY", "QQQ", "AAPL", "MSFT", "GOOGL",
        "AMZN", "TSLA", "NVDA", "META", "AMD"
    ]

    market_data = get_alpaca_market_data()
    quotes = market_data.get_multiple_quotes(watchlist)

    return {
        "name": "Default Watchlist",
        "symbols": watchlist,
        "quotes": quotes
    }


# ============================================================================
# STATIC FILES
# ============================================================================

# Mount UI directory if it exists
ui_path = Path("ui")
if ui_path.exists():
    app.mount("/ui", StaticFiles(directory="ui"), name="ui")

    @app.get("/dashboard")
    async def dashboard():
        """Serve main dashboard"""
        return FileResponse("ui/complete_platform.html")


# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("="*80)
    logger.info("AI TRADING HUB - ALPACA EDITION")
    logger.info("="*80)

    # Check broker configuration
    config = get_broker_config()
    logger.info(f"Broker Type: {config.broker_type}")

    # Test Alpaca connection
    try:
        connector = get_alpaca_connector()
        if connector.is_connected():
            account = connector.get_account()
            logger.info("[OK] Alpaca Connected")
            logger.info(f"   Account: {account['account_id']}")
            logger.info(f"   Buying Power: ${account['buying_power']:,.2f}")
            logger.info(f"   Paper Trading: {config.alpaca.paper}")
        else:
            logger.error("[FAIL] Alpaca connection failed")
    except Exception as e:
        logger.error(f"[FAIL] Alpaca initialization error: {e}")

    # Load AI model if available
    try:
        predictor = get_alpaca_predictor()
        if predictor.model:
            logger.info(f"[OK] AI Model Loaded (Accuracy: {predictor.accuracy:.4f})")
        else:
            logger.info("[INFO] AI Model not trained - use /api/ai/train to train")
    except Exception as e:
        logger.info(f"[INFO] AI Model status: {e}")

    logger.info("="*80)
    logger.info("Server ready on http://localhost:9100")
    logger.info("Dashboard: http://localhost:9100/dashboard")
    logger.info("API Docs: http://localhost:9100/docs")
    logger.info("="*80)


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("Shutting down AI Trading Hub...")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("AI TRADING HUB - ALPACA EDITION")
    print("="*80)
    print("Starting server on http://localhost:9100")
    print("Dashboard: http://localhost:9100/dashboard")
    print("API Docs: http://localhost:9100/docs")
    print("="*80 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9100,
        log_level="info"
    )
