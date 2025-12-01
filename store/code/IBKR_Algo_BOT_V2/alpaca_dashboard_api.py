"""
Unified Trading Dashboard API with Alpaca Integration
Main API server for AI Project Hub
"""
from dotenv import load_dotenv
load_dotenv()

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
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

# Watchlist management
from watchlist_routes import router as watchlist_router

# TradingView integration
from tradingview_integration import router as tradingview_router

# Analytics routes
from analytics_routes import router as analytics_router

# Advanced Pipeline (Task Queue Mesh Mode, XGBoost, Optuna, Regime/Drift Detection)
try:
    from src.pipeline.api_routes import router as pipeline_router
    HAS_ADVANCED_PIPELINE = True
except ImportError:
    HAS_ADVANCED_PIPELINE = False
    pipeline_router = None

# AI Predictor
from ai.alpaca_ai_predictor import get_alpaca_predictor

# Multi-Channel Data Provider for parallel market data access
try:
    from multi_channel_data import (
        get_multi_channel_provider,
        DataChannel,
        get_quote_fast,
        get_quotes_fast,
        get_snapshot_fast,
        get_bars_for_ai,
        get_bars_for_chart
    )
    HAS_MULTI_CHANNEL = True
except ImportError as e:
    logger.warning(f"Multi-channel data provider not available: {e}")
    HAS_MULTI_CHANNEL = False

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

# Include watchlist router
app.include_router(watchlist_router)

# Include TradingView integration router
app.include_router(tradingview_router)

# Include analytics router
app.include_router(analytics_router)

# Include compatibility router for legacy UI endpoints
app.include_router(compat_router)

# Include advanced pipeline router (if available)
if HAS_ADVANCED_PIPELINE and pipeline_router:
    app.include_router(pipeline_router, tags=["Advanced Pipeline"])


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class TrainRequest(BaseModel):
    symbol: str
    test_size: float = 0.2


class TrainMultiRequest(BaseModel):
    symbols: List[str]
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

    # Check multi-channel status
    multi_channel_status = "not_available"
    if HAS_MULTI_CHANNEL:
        try:
            provider = get_multi_channel_provider()
            multi_channel_status = f"active ({len(provider.channels)} channels)"
        except Exception:
            multi_channel_status = "error"

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
            "market_data": "operational",
            "multi_channel_data": multi_channel_status
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
# MULTI-CHANNEL DATA ENDPOINTS (Parallel Market Data Access)
# ============================================================================

@app.get("/api/data/channels/status")
async def get_channel_status():
    """
    Get status of all data channels.

    Multi-channel architecture provides:
    - ORDERS channel: Dedicated for order execution
    - CHARTS channel: Dedicated for charting data
    - AI channel: Dedicated for AI predictions/analysis
    - SCANNER channel: Dedicated for market scanning
    - REALTIME channel: Dedicated for live quotes
    """
    if not HAS_MULTI_CHANNEL:
        return {
            "available": False,
            "message": "Multi-channel data provider not available"
        }

    try:
        provider = get_multi_channel_provider()
        return {
            "available": True,
            "status": provider.get_all_status(),
            "aggregate": provider.get_aggregate_stats()
        }
    except Exception as e:
        logger.error(f"Error getting channel status: {e}")
        return {"available": False, "error": str(e)}


@app.get("/api/data/channels/quote/{symbol}")
async def get_fast_quote(symbol: str, channel: str = "realtime"):
    """
    Get a quote using a specific data channel.

    Channels:
    - realtime: Live quotes (default)
    - orders: Order execution context
    - charts: Charting context
    - ai: AI analysis context
    - scanner: Market scanning context
    """
    if not HAS_MULTI_CHANNEL:
        # Fallback to standard market data
        market_data = get_alpaca_market_data()
        return market_data.get_latest_quote(symbol.upper())

    try:
        channel_enum = DataChannel(channel.lower())
        quote = get_quote_fast(symbol.upper(), channel_enum)

        if quote is None:
            raise HTTPException(status_code=404, detail=f"No quote data for {symbol}")

        return quote
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid channel: {channel}")
    except Exception as e:
        logger.error(f"Error fetching fast quote: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/channels/multi-quote")
async def get_parallel_quotes(symbols: str = Query(..., description="Comma-separated symbols")):
    """
    Get quotes for multiple symbols using parallel channels.

    This splits the work across multiple data channels for maximum speed.
    Much faster than sequential fetching for large watchlists.
    """
    if not HAS_MULTI_CHANNEL:
        # Fallback to standard market data
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        market_data = get_alpaca_market_data()
        return market_data.get_multiple_quotes(symbol_list)

    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        import time

        start = time.time()
        quotes = get_quotes_fast(symbol_list)
        elapsed_ms = (time.time() - start) * 1000

        return {
            "quotes": quotes,
            "count": len(quotes),
            "elapsed_ms": round(elapsed_ms, 2),
            "parallel": True
        }
    except Exception as e:
        logger.error(f"Error fetching parallel quotes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/channels/snapshots")
async def get_parallel_snapshots(symbols: str = Query(..., description="Comma-separated symbols")):
    """
    Get full market snapshots for multiple symbols using parallel channels.

    Includes: quote, last trade, daily bar, previous close, change %.
    """
    if not HAS_MULTI_CHANNEL:
        raise HTTPException(status_code=503, detail="Multi-channel provider not available")

    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        import time

        provider = get_multi_channel_provider()
        start = time.time()
        snapshots = provider.get_snapshots_parallel(symbol_list)
        elapsed_ms = (time.time() - start) * 1000

        return {
            "snapshots": snapshots,
            "count": len(snapshots),
            "elapsed_ms": round(elapsed_ms, 2),
            "parallel": True
        }
    except Exception as e:
        logger.error(f"Error fetching parallel snapshots: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/channels/bars/{symbol}")
async def get_channel_bars(
    symbol: str,
    channel: str = "charts",
    timeframe: str = "1Day",
    days: int = 30
):
    """
    Get historical bars using a specific channel.

    Channels optimized for different use cases:
    - charts: For TradingView/charting (5m, 15m bars)
    - ai: For AI training data (daily bars, longer history)
    - scanner: For market scanning
    """
    if not HAS_MULTI_CHANNEL:
        # Fallback to standard market data
        market_data = get_alpaca_market_data()
        bars = market_data.get_historical_bars(symbol.upper(), timeframe, days * 10)
        return {"symbol": symbol, "bars": bars.to_dict(orient='records') if hasattr(bars, 'to_dict') else []}

    try:
        provider = get_multi_channel_provider()

        if channel.lower() == "ai":
            bars = get_bars_for_ai(symbol.upper(), days)
        elif channel.lower() == "charts":
            bars = get_bars_for_chart(symbol.upper(), timeframe, days)
        else:
            channel_client = provider.get_channel(DataChannel(channel.lower()))
            bars = channel_client.get_bars(symbol.upper(), timeframe, days)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "days": days,
            "channel": channel,
            "bars": bars,
            "count": len(bars)
        }
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid channel: {channel}")
    except Exception as e:
        logger.error(f"Error fetching channel bars: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/channels/multi-bars")
async def get_multi_symbol_bars(
    symbols: str = Query(..., description="Comma-separated symbols"),
    timeframe: str = "1Day",
    days: int = 30
):
    """
    Get historical bars for multiple symbols in parallel.

    Ideal for:
    - AI batch training data collection
    - Multi-symbol backtesting
    - Watchlist historical analysis
    """
    if not HAS_MULTI_CHANNEL:
        raise HTTPException(status_code=503, detail="Multi-channel provider not available")

    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        import time

        provider = get_multi_channel_provider()
        start = time.time()
        bars = provider.get_multi_symbol_bars(symbol_list, timeframe, days)
        elapsed_ms = (time.time() - start) * 1000

        return {
            "symbols": list(bars.keys()),
            "timeframe": timeframe,
            "days": days,
            "bars": bars,
            "count": sum(len(b) for b in bars.values()),
            "elapsed_ms": round(elapsed_ms, 2),
            "parallel": True
        }
    except Exception as e:
        logger.error(f"Error fetching multi-symbol bars: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AI PREDICTION ENDPOINTS
# ============================================================================

@app.post("/api/ai/train")
async def train_model(request: TrainRequest):
    """Train AI prediction model on a single symbol"""
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


@app.post("/api/ai/train-multi")
async def train_model_multi(request: TrainMultiRequest):
    """Train AI prediction model on multiple symbols for better generalization"""
    try:
        predictor = get_alpaca_predictor()
        symbols = [s.upper() for s in request.symbols]
        result = predictor.train_multi(
            symbols=symbols,
            test_size=request.test_size
        )

        return result

    except Exception as e:
        logger.error(f"Error training multi-symbol model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/models/train")
async def train_model_compat(data: dict):
    """Compatibility route for UI - Train AI prediction model"""
    import time
    try:
        predictor = get_alpaca_predictor()
        symbol = data.get('symbol', 'AAPL').upper()
        test_size = data.get('test_size', 0.2)

        result = predictor.train(symbol=symbol, test_size=test_size)

        return {
            "success": True,
            "data": {
                "training_id": f"train_{symbol}_{int(time.time())}",
                "symbol": symbol,
                "status": "completed",
                "metrics": result
            }
        }
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return {"success": False, "message": str(e)}


@app.post("/api/ai/train-walkforward")
async def train_walkforward(data: dict):
    """
    Walk-Forward Training - The PROPER way to train financial ML models

    This ensures NO DATA LEAKAGE:
    - Training data: Historical data BEFORE cutoff date
    - Test data: Data AFTER cutoff (truly out-of-sample)

    Parameters:
        symbols: List of stock symbols (default: major tech + momentum stocks)
        train_months: Months of training data (default: 12)
        test_months: Months of test data (default: 3)

    Returns:
        Out-of-sample accuracy, precision, recall, F1 score
    """
    try:
        predictor = get_alpaca_predictor()

        # Default to a diverse set of stocks
        default_symbols = ["TSLA", "NVDA", "AMD", "META", "AAPL", "GOOGL", "MSFT",
                          "AMZN", "PLTR", "COIN", "MARA", "SOFI"]

        symbols = data.get('symbols', default_symbols)
        if isinstance(symbols, str):
            symbols = [s.strip().upper() for s in symbols.split(',')]

        train_months = int(data.get('train_months', 12))
        test_months = int(data.get('test_months', 3))

        result = predictor.train_walkforward(
            symbols=symbols,
            train_months=train_months,
            test_months=test_months
        )

        return result

    except Exception as e:
        logger.error(f"Error in walk-forward training: {e}")
        return {"success": False, "message": str(e)}


@app.post("/api/backtest/run")
async def run_backtest_endpoint(data: dict):
    """Run a backtest simulation using AI predictions on historical data"""
    try:
        from backtester import get_backtester
        from dataclasses import asdict

        backtester = get_backtester()

        # Extract parameters
        symbols = data.get('symbols', ['SPY'])
        if isinstance(symbols, str):
            symbols = [s.strip().upper() for s in symbols.split(',')]

        start_date = data.get('start_date', (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'))
        end_date = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        initial_capital = float(data.get('initial_capital', 100000))
        position_size_pct = float(data.get('position_size_pct', 0.1))
        confidence_threshold = float(data.get('confidence_threshold', 0.15))
        max_positions = int(data.get('max_positions', 5))
        holding_period = int(data.get('holding_period', 5))

        # Run backtest
        result = backtester.run_backtest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            confidence_threshold=confidence_threshold,
            max_positions=max_positions,
            holding_period=holding_period
        )

        return {
            "success": True,
            "data": asdict(result)
        }

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return {
            "success": False,
            "message": str(e)
        }


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
# CLAUDE AI CHAT ENDPOINTS
# ============================================================================

@app.post("/api/claude/query")
async def claude_query(request: Dict[str, Any]):
    """
    Send a query to Claude AI with NLP trading capabilities.

    Supports natural language commands like:
    - "Buy 10 shares of AAPL"
    - "Sell all my TSLA position"
    - "What's my account balance?"
    - "Analyze MSFT"
    - "Set stop loss to 2%"
    - "Start the auto trading bot"
    - "Show my positions"
    """
    import re

    query = request.get("query", "")
    context = request.get("context", {})
    symbol = request.get("symbol", "SPY")
    conversation_id = context.get("conversation_id")

    try:
        # Import Claude bot intelligence
        from ai.claude_bot_intelligence import get_bot_intelligence

        ai = get_bot_intelligence()
        connector = get_alpaca_connector()

        # Build rich context for Claude
        trading_context = {
            "current_symbol": symbol,
            "connected": connector.is_connected() if connector else False,
            "account": None,
            "positions": [],
            "recent_orders": []
        }

        # Fetch real account data if connected
        if connector and connector.is_connected():
            try:
                trading_context["account"] = connector.get_account()
                trading_context["positions"] = connector.get_positions()
                trading_context["recent_orders"] = connector.get_orders("all")[:10]
            except Exception as e:
                logger.warning(f"Could not fetch trading context: {e}")

        # Merge with any provided context
        if context:
            trading_context.update(context)

        # Check for NLP trading commands
        trade_result = await _parse_and_execute_nlp_trade(query, connector, trading_context)

        if trade_result.get("executed"):
            # A trade was executed via NLP
            response_text = trade_result.get("response", "Command executed successfully.")
            action_taken = trade_result.get("action", "trade")
            action_details = trade_result.get("details", {})
        else:
            # No trade command - use Claude AI for analysis/chat
            result = ai.chat_with_tools(query, use_tools=True)
            response_text = result.get("response", "I couldn't process your request. Please try again.")
            action_taken = "chat"
            action_details = {"mood": result.get("mood", "neutral")}

        return {
            "success": True,
            "data": {
                "response": response_text,
                "conversation_id": conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "action": action_taken,
                "action_details": action_details
            }
        }

    except ImportError as e:
        logger.warning(f"Claude AI module not available: {e}")
        return {
            "success": False,
            "data": {
                "response": "I'm here to help with your trading questions. The AI service may be initializing - please try again in a moment.",
                "conversation_id": conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Claude query error: {e}")
        return {
            "success": False,
            "data": {
                "response": f"Error processing request: {str(e)}",
                "conversation_id": conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat()
            }
        }


async def _parse_and_execute_nlp_trade(query: str, connector, context: Dict) -> Dict:
    """
    Parse natural language trading commands and execute them.

    Supported commands:
    - Buy/sell orders: "buy 10 AAPL", "sell 50 shares of TSLA"
    - Close positions: "close AAPL", "close all positions"
    - Account queries: "show balance", "what's my buying power"
    - Position queries: "show positions", "what am I holding"
    - Bot control: "start bot", "stop bot", "enable auto trading"
    """
    import re

    query_lower = query.lower().strip()

    # Pattern: Buy/Sell orders
    trade_pattern = r"(buy|sell|purchase|short)\s+(\d+)\s*(?:shares?\s+(?:of\s+)?)?([A-Za-z]+)(?:\s+(?:at|@|for)\s+\$?([\d.]+))?"
    trade_match = re.search(trade_pattern, query_lower)

    if trade_match and connector and connector.is_connected():
        action = trade_match.group(1)
        quantity = int(trade_match.group(2))
        symbol = trade_match.group(3).upper()
        limit_price = float(trade_match.group(4)) if trade_match.group(4) else None

        side = "buy" if action in ["buy", "purchase"] else "sell"

        try:
            if limit_price:
                result = connector.place_limit_order(symbol, quantity, side, limit_price)
                return {
                    "executed": True,
                    "action": "limit_order",
                    "response": f"✅ Placed {side.upper()} limit order: {quantity} shares of {symbol} at ${limit_price:.2f}",
                    "details": {"symbol": symbol, "quantity": quantity, "side": side, "limit_price": limit_price, "order": result}
                }
            else:
                result = connector.place_market_order(symbol, quantity, side)
                return {
                    "executed": True,
                    "action": "market_order",
                    "response": f"✅ Placed {side.upper()} market order: {quantity} shares of {symbol}",
                    "details": {"symbol": symbol, "quantity": quantity, "side": side, "order": result}
                }
        except Exception as e:
            return {
                "executed": True,
                "action": "order_failed",
                "response": f"❌ Order failed: {str(e)}",
                "details": {"error": str(e)}
            }

    # Pattern: Close position
    close_pattern = r"(?:close|exit|liquidate|sell all)\s+(?:my\s+)?(?:position\s+(?:in|on)\s+)?([A-Za-z]+)(?:\s+position)?"
    close_match = re.search(close_pattern, query_lower)

    if close_match and connector and connector.is_connected():
        symbol = close_match.group(1).upper()
        try:
            result = connector.close_position(symbol)
            return {
                "executed": True,
                "action": "close_position",
                "response": f"✅ Closed position in {symbol}",
                "details": {"symbol": symbol, "result": result}
            }
        except Exception as e:
            return {
                "executed": True,
                "action": "close_failed",
                "response": f"❌ Could not close {symbol}: {str(e)}",
                "details": {"error": str(e)}
            }

    # Pattern: Close all positions
    if any(phrase in query_lower for phrase in ["close all", "liquidate all", "sell everything", "flatten"]):
        if connector and connector.is_connected():
            try:
                result = connector.close_all_positions()
                return {
                    "executed": True,
                    "action": "close_all",
                    "response": "✅ Closed all positions",
                    "details": {"result": result}
                }
            except Exception as e:
                return {
                    "executed": True,
                    "action": "close_all_failed",
                    "response": f"❌ Could not close all positions: {str(e)}",
                    "details": {"error": str(e)}
                }

    # Pattern: Show positions
    if any(phrase in query_lower for phrase in ["show position", "my position", "what am i holding", "current holdings", "show holdings"]):
        positions = context.get("positions", [])
        if positions:
            pos_text = "\n".join([f"• {p.get('symbol', 'N/A')}: {p.get('qty', 0)} shares @ ${float(p.get('avg_entry_price', 0)):.2f} (P&L: ${float(p.get('unrealized_pl', 0)):.2f})" for p in positions[:10]])
            return {
                "executed": True,
                "action": "show_positions",
                "response": f"📊 **Current Positions:**\n{pos_text}",
                "details": {"positions": positions}
            }
        else:
            return {
                "executed": True,
                "action": "show_positions",
                "response": "📊 No open positions",
                "details": {"positions": []}
            }

    # Pattern: Show account/balance
    if any(phrase in query_lower for phrase in ["show balance", "account balance", "buying power", "how much", "my account", "show account"]):
        account = context.get("account", {})
        if account:
            equity = float(account.get("equity", 0))
            buying_power = float(account.get("buying_power", 0))
            cash = float(account.get("cash", 0))
            return {
                "executed": True,
                "action": "show_account",
                "response": f"💰 **Account Summary:**\n• Equity: ${equity:,.2f}\n• Buying Power: ${buying_power:,.2f}\n• Cash: ${cash:,.2f}",
                "details": {"account": account}
            }
        else:
            return {
                "executed": True,
                "action": "show_account",
                "response": "💰 Account data unavailable. Please check connection.",
                "details": {}
            }

    # Pattern: Start/stop bot
    if any(phrase in query_lower for phrase in ["start bot", "start trading", "enable auto", "turn on bot"]):
        try:
            from bot_manager import get_bot_manager
            bot = get_bot_manager()
            await bot.start()
            return {
                "executed": True,
                "action": "start_bot",
                "response": "🤖 Auto-trading bot STARTED. Monitoring markets...",
                "details": {"status": "running"}
            }
        except Exception as e:
            return {
                "executed": True,
                "action": "start_bot_failed",
                "response": f"❌ Could not start bot: {str(e)}",
                "details": {"error": str(e)}
            }

    if any(phrase in query_lower for phrase in ["stop bot", "stop trading", "disable auto", "turn off bot", "pause bot"]):
        try:
            from bot_manager import get_bot_manager
            bot = get_bot_manager()
            await bot.stop()
            return {
                "executed": True,
                "action": "stop_bot",
                "response": "🛑 Auto-trading bot STOPPED.",
                "details": {"status": "stopped"}
            }
        except Exception as e:
            return {
                "executed": True,
                "action": "stop_bot_failed",
                "response": f"❌ Could not stop bot: {str(e)}",
                "details": {"error": str(e)}
            }

    # No command matched - return not executed
    return {"executed": False}


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

    # Initialize multi-channel data provider
    if HAS_MULTI_CHANNEL:
        try:
            provider = get_multi_channel_provider()
            logger.info(f"[OK] Multi-Channel Data Provider Active ({len(provider.channels)} channels)")
            for channel in provider.channels:
                logger.info(f"     - {channel.value.upper()} channel ready")
        except Exception as e:
            logger.warning(f"[WARN] Multi-channel provider error: {e}")
    else:
        logger.info("[INFO] Multi-channel data provider not available")

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
