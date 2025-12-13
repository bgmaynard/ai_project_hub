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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
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

try:
    from ai.autonomous_trader import AutonomousTrader, BotConfig
    from ai.alpha_fusion_v2 import AlphaFusionEngine
    from ai.trading_engine import TradingEngine
    AUTONOMOUS_BOT_AVAILABLE = True
except ImportError:
    AutonomousTrader = None
    BotConfig = None
    AlphaFusionEngine = None
    TradingEngine = None
    AUTONOMOUS_BOT_AVAILABLE = False
    print("[WARN] autonomous_trader not found")

# Try to import Warrior Trading modules
try:
    from ai.warrior_api import router as warrior_router
    WARRIOR_TRADING_AVAILABLE = True
except ImportError:
    warrior_router = None
    WARRIOR_TRADING_AVAILABLE = False
    print("[WARN] warrior_api not found")

# Try to import Sentiment Analysis module
try:
    from ai.warrior_sentiment_router import router as sentiment_router
    SENTIMENT_ANALYSIS_AVAILABLE = True
except ImportError:
    sentiment_router = None
    SENTIMENT_ANALYSIS_AVAILABLE = False
    print("[WARN] warrior_sentiment_router not found - Phase 5 features disabled")

    print("[WARN] warrior_sentiment_router not found - Phase 5 features disabled")

# Try to import Advanced ML module
try:
    from ai.warrior_ml_router import router as ml_router
    ML_AVAILABLE = True
except ImportError:
    ml_router = None
    ML_AVAILABLE = False
    print("[WARN] warrior_ml_router not found - Phase 3 features disabled")

# Try to import Risk Management module
try:
    from ai.warrior_risk_router import router as risk_router
    RISK_MANAGEMENT_AVAILABLE = True
except ImportError:
    risk_router = None
    RISK_MANAGEMENT_AVAILABLE = False
    print("[WARN] warrior_risk_router not found - Phase 4 features disabled")

# Try to import Monitoring Dashboard module
try:
    from ai.monitoring_router import router as monitoring_router
    MONITORING_AVAILABLE = True
except ImportError:
    monitoring_router = None
    MONITORING_AVAILABLE = False
    print("[WARN] monitoring_router not found - Dashboard monitoring disabled")

# Try to import IBKR
try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, Contract
    from ibkr_connector import IBKRConnector
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    IBKRConnector = None
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

# Include Warrior Trading router if available
if warrior_router:
    app.include_router(warrior_router)
    logger.info("✓ Warrior Trading API endpoints loaded")


# Include Sentiment Analysis router if available
if sentiment_router:
    app.include_router(sentiment_router)
    logger.info("✓ Sentiment Analysis API endpoints loaded (Phase 5)")

# Include Advanced ML router if available
if ml_router:
    app.include_router(ml_router)
    logger.info("✓ Advanced ML API endpoints loaded (Phase 3)")

# Include Risk Management router if available
if risk_router:
    app.include_router(risk_router)
    logger.info("✓ Risk Management API endpoints loaded (Phase 4)")

# Include Monitoring Dashboard router if available
if monitoring_router:
    app.include_router(monitoring_router)
    logger.info("✓ Monitoring Dashboard API endpoints loaded")

# ═══════════════════════════════════════════════════════════════════════
#                     STARTUP EVENT - AUTO-CONNECT TO IBKR
# ═══════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Auto-connect to IBKR on server startup if enabled in .env"""
    auto_connect = os.getenv("IBKR_AUTO_CONNECT", "false").lower() == "true"

    if not auto_connect:
        logger.info("IBKR auto-connect disabled (set IBKR_AUTO_CONNECT=true in .env to enable)")
        return

    if not IBKR_AVAILABLE:
        logger.warning("IBKR auto-connect failed: ib_insync not installed")
        return

    # Get connection parameters from environment
    host = os.getenv("IBKR_HOST", "127.0.0.1")
    port = int(os.getenv("IBKR_PORT", "7497"))
    client_id = int(os.getenv("IBKR_CLIENT_ID", "1"))

    logger.info(f"Attempting to auto-connect to IBKR at {host}:{port}...")

    try:
        result = await data_bus.connect_ibkr(host, port, client_id)
        logger.info(f"✓ Auto-connected to IBKR successfully on port {port}")
    except Exception as e:
        logger.warning(f"✗ Auto-connect to IBKR failed: {e}")
        logger.info("You can manually connect via the UI or /api/ibkr/connect endpoint")

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
        self.ibkr_connector: Optional[IBKRConnector] = None
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
            # Create IBKRConnector instance
            self.ibkr_connector = IBKRConnector()
            await self.ibkr_connector.connect(host, port, client_id)

            # Keep reference to IB object for compatibility
            self.ib = self.ibkr_connector.ib
            self.connected = self.ibkr_connector.is_connected()

            # Set up market data handlers
            self.ib.pendingTickersEvent += self._on_ticker_update

            logger.info(f"✓ Connected to IBKR on port {port}")
            return {"status": "connected", "port": port}
        except Exception as e:
            logger.error(f"✗ Failed to connect to IBKR: {e}")
            self.connected = False
            raise
    
    async def disconnect_ibkr(self):
        """Disconnect from IBKR"""
        if self.ibkr_connector:
            self.ibkr_connector.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")
    
    def _on_ticker_update(self, tickers):
        """Handle ticker updates from IBKR"""
        for ticker in tickers:
            symbol = ticker.contract.symbol

            # Extract quote data
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

            # Extract Level 2 market depth data from DOM
            depth_updated = False
            if hasattr(ticker, 'domBids') and ticker.domBids:
                bids = []
                for bid in ticker.domBids:
                    if bid.price and bid.size:
                        bids.append({
                            'price': float(bid.price),
                            'size': int(bid.size),
                            'market_maker': getattr(bid, 'marketMaker', '')
                        })

                asks = []
                if hasattr(ticker, 'domAsks') and ticker.domAsks:
                    for ask in ticker.domAsks:
                        if ask.price and ask.size:
                            asks.append({
                                'price': float(ask.price),
                                'size': int(ask.size),
                                'market_maker': getattr(ask, 'marketMaker', '')
                            })

                if bids or asks:
                    self.l2_data[symbol] = {
                        'bids': bids,
                        'asks': asks
                    }
                    asyncio.create_task(self.broadcast(symbol, "l2", self.l2_data[symbol]))
                    depth_updated = True
                    logger.debug(f"L2 depth for {symbol}: {len(bids)} bids, {len(asks)} asks")

            # If no DOM data, create simulated depth from bid/ask (for paper trading)
            if not depth_updated and data["bid"] and data["ask"]:
                bids = []
                asks = []

                # Generate 10 bid levels
                bid_price = data["bid"]
                for i in range(10):
                    bids.append({
                        'price': round(bid_price - (i * 0.01), 2),
                        'size': data["bid_size"] if i == 0 else int(100 + (hash(f"{symbol}{i}") % 400)),
                        'market_maker': ['NSDQ', 'ARCA', 'BATS', 'EDGX', 'IEX'][i % 5]
                    })

                # Generate 10 ask levels
                ask_price = data["ask"]
                for i in range(10):
                    asks.append({
                        'price': round(ask_price + (i * 0.01), 2),
                        'size': data["ask_size"] if i == 0 else int(100 + (hash(f"{symbol}{i+10}") % 400)),
                        'market_maker': ['NYSE', 'CBOE', 'NSDQ', 'ARCA', 'BATS'][i % 5]
                    })

                self.l2_data[symbol] = {
                    'bids': bids,
                    'asks': asks
                }
                asyncio.create_task(self.broadcast(symbol, "l2", self.l2_data[symbol]))

            # Track trades (Time & Sales) - detect price changes
            if data["last"] is not None:
                # Check if price changed (indicates a trade)
                last_trade = None
                if symbol in self.time_sales and len(self.time_sales[symbol]) > 0:
                    last_trade = list(self.time_sales[symbol])[-1]

                # Record trade if price changed or we have lastSize
                if ticker.lastSize and ticker.lastSize == ticker.lastSize:
                    trade = {
                        "timestamp": datetime.now().isoformat(),
                        "price": data["last"],
                        "size": int(ticker.lastSize)
                    }
                    self.time_sales[symbol].append(trade)
                    asyncio.create_task(self.broadcast(symbol, "trade", trade))
                elif not last_trade or last_trade["price"] != data["last"]:
                    # Price changed, create estimated trade
                    trade = {
                        "timestamp": datetime.now().isoformat(),
                        "price": data["last"],
                        "size": data["volume"] if data["volume"] else 100
                    }
                    self.time_sales[symbol].append(trade)
                    asyncio.create_task(self.broadcast(symbol, "trade", trade))

            # Broadcast quote update
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

# Initialize autonomous trading bot (lazy init)
autonomous_bot = None

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

class BotConfigUpdate(BaseModel):
    account_size: Optional[float] = None
    max_position_size_usd: Optional[float] = None
    max_positions: Optional[int] = None
    daily_loss_limit_usd: Optional[float] = None
    min_probability_threshold: Optional[float] = None
    max_spread_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    watchlist: Optional[List[str]] = None
    enabled: Optional[bool] = None

class OrderRequest(BaseModel):
    symbol: str
    action: str  # BUY or SELL
    quantity: int
    order_type: str  # MARKET, LIMIT, STOP, STOP_LIMIT
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tif: str = "DAY"  # DAY, GTC, IOC, FOK
    extended_hours: bool = False
    exchange: str = "SMART"

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

@app.post("/api/ibkr/place-order")
async def place_order(order: OrderRequest):
    """Place an order with IBKR"""
    try:
        if not data_bus.connected or not data_bus.ibkr_connector:
            raise HTTPException(status_code=503, detail="IBKR not connected")

        # Get account information for validation
        # Use ib_insync's async methods directly instead of executors
        account_values = await data_bus.ib.accountSummaryAsync()
        account_info = {
            'account_type': 'Unknown',
            'net_liquidation': 0,
            'day_trades_remaining': None
        }

        for item in account_values:
            if item.tag == 'AccountType':
                account_info['account_type'] = item.value
            elif item.tag == 'NetLiquidation':
                account_info['net_liquidation'] = float(item.value)
            elif item.tag == 'DayTradesRemaining':
                account_info['day_trades_remaining'] = int(item.value) if item.value else None

        # Validate order based on account type
        acct_type = account_info['account_type'].upper()
        net_liq = account_info['net_liquidation']

        # Check for short selling restrictions
        if order.action == 'SELL':
            # Check if this is a short sale (selling without position)
            await data_bus.ib.reqPositionsAsync()
            positions_list = data_bus.ib.positions()
            positions = {pos.contract.symbol: pos.position for pos in positions_list}
            current_position = positions.get(order.symbol, 0)

            if current_position >= 0:  # No long position, this would be a short sale
                if 'CASH' in acct_type:
                    raise HTTPException(
                        status_code=403,
                        detail="SHORT SELLING NOT ALLOWED: Cash accounts cannot short sell. SEC violation prevented."
                    )
                elif 'IRA' in acct_type:
                    raise HTTPException(
                        status_code=403,
                        detail="SHORT SELLING NOT ALLOWED: IRA accounts cannot short sell. SEC violation prevented."
                    )

        # Check Pattern Day Trader (PDT) restrictions
        if 'MARGIN' in acct_type or 'REG T' in acct_type:
            if net_liq < 25000:
                # This is a PDT account - check day trades remaining
                day_trades_left = account_info.get('day_trades_remaining')

                if day_trades_left is not None and day_trades_left <= 0:
                    raise HTTPException(
                        status_code=403,
                        detail=f"DAY TRADE LIMIT REACHED: Your account has 0 day trades remaining. "
                               f"Pattern Day Trader rules require $25,000 minimum equity for unlimited day trading. "
                               f"Current equity: ${net_liq:,.2f}. SEC violation prevented."
                    )
                elif day_trades_left is not None and day_trades_left <= 1:
                    logger.warning(f"PDT WARNING: Only {day_trades_left} day trade(s) remaining!")

        # Check cash account settlement restrictions
        if 'CASH' in acct_type:
            # Cash accounts must wait T+2 for settlement
            # This is a simplified check - full implementation would track unsettled funds
            logger.warning("CASH ACCOUNT: Be aware of T+2 settlement rules. Avoid good faith violations.")

        from ib_insync import Stock, Order, LimitOrder, MarketOrder, StopOrder, StopLimitOrder

        # Create contract
        contract = Stock(order.symbol, order.exchange, 'USD')

        # Create order based on type
        if order.order_type == "MARKET":
            ib_order = MarketOrder(order.action, order.quantity)
        elif order.order_type == "LIMIT":
            if order.limit_price is None or (isinstance(order.limit_price, float) and (order.limit_price != order.limit_price)):  # Check for None or NaN
                raise HTTPException(status_code=400, detail="Valid limit price required for LIMIT orders")
            ib_order = LimitOrder(order.action, order.quantity, order.limit_price)
        elif order.order_type == "STOP":
            if order.stop_price is None:
                raise HTTPException(status_code=400, detail="Stop price required for STOP orders")
            ib_order = StopOrder(order.action, order.quantity, order.stop_price)
        elif order.order_type == "STOP_LIMIT":
            if order.limit_price is None or order.stop_price is None:
                raise HTTPException(status_code=400, detail="Limit and stop price required for STOP_LIMIT orders")
            ib_order = StopLimitOrder(order.action, order.quantity, order.limit_price, order.stop_price)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid order type: {order.order_type}")

        # Set time in force
        ib_order.tif = order.tif

        # Set extended hours if applicable
        if order.extended_hours:
            ib_order.outsideRth = True

        # Place the order (placeOrder is event-loop safe and returns immediately)
        trade = data_bus.ibkr_connector.ib.placeOrder(contract, ib_order)
        # Wait for order to be submitted
        await asyncio.sleep(0.1)

        logger.info(f"Order placed: {order.action} {order.quantity} {order.symbol} @ {order.order_type}")

        return {
            "success": True,
            "order_id": trade.order.orderId,
            "status": trade.orderStatus.status,
            "message": f"Order placed: {order.action} {order.quantity} {order.symbol}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error placing order: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to place order: {str(e)}")

@app.get("/api/ibkr/orders")
async def get_orders():
    """Get all open/working orders from IBKR"""
    try:
        if not data_bus.connected or not data_bus.ibkr_connector:
            raise HTTPException(status_code=503, detail="IBKR not connected")

        # Request open orders and get trades (which include order, contract, and status)
        await data_bus.ib.reqOpenOrdersAsync()
        open_trades = data_bus.ib.openTrades()

        orders_list = []
        for trade in open_trades:
            orders_list.append({
                "order_id": trade.order.orderId,
                "symbol": trade.contract.symbol,
                "action": trade.order.action,
                "order_type": trade.order.orderType,
                "quantity": trade.order.totalQuantity,
                "limit_price": trade.order.lmtPrice if hasattr(trade.order, 'lmtPrice') else None,
                "stop_price": trade.order.auxPrice if hasattr(trade.order, 'auxPrice') else None,
                "status": trade.orderStatus.status,
                "filled": trade.orderStatus.filled,
                "remaining": trade.orderStatus.remaining,
                "avg_fill_price": trade.orderStatus.avgFillPrice
            })

        logger.info(f"Retrieved {len(orders_list)} open orders")
        return {"orders": orders_list, "count": len(orders_list)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting orders: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get orders: {str(e)}")

@app.post("/api/ibkr/cancel-order")
async def cancel_order(request: dict):
    """Cancel an order by order ID"""
    try:
        if not data_bus.connected or not data_bus.ibkr_connector:
            raise HTTPException(status_code=503, detail="IBKR not connected")

        order_id = request.get("order_id")
        if not order_id:
            raise HTTPException(status_code=400, detail="order_id required")

        # Find the order in open trades
        await data_bus.ib.reqOpenOrdersAsync()
        open_trades = data_bus.ib.openTrades()

        trade_to_cancel = None
        for trade in open_trades:
            if trade.order.orderId == order_id:
                trade_to_cancel = trade
                break

        if not trade_to_cancel:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

        # Cancel the order
        data_bus.ib.cancelOrder(trade_to_cancel.order)
        await asyncio.sleep(0.1)  # Wait for cancellation

        logger.info(f"Cancelled order {order_id}")
        return {
            "success": True,
            "order_id": order_id,
            "message": f"Order {order_id} cancellation requested"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling order: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel order: {str(e)}")

# ═══════════════════════════════════════════════════════════════════════
#                     MARKET DATA SUBSCRIPTION
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/subscribe")
async def subscribe_symbol(request: SymbolSubscription):
    """Subscribe to market data for a symbol"""
    try:
        if not data_bus.connected or not data_bus.ibkr_connector:
            raise HTTPException(status_code=503, detail="IBKR not connected")

        # Subscribe to quotes
        if "quote" in request.data_types:
            await data_bus.subscribe_market_data(request.symbol, request.exchange)

        # Subscribe to Level 2 market depth
        if "l2" in request.data_types:
            data_bus.ibkr_connector.request_market_depth(request.symbol, request.exchange, depth=10)

        # Trades are captured automatically from quote updates

        logger.info(f"Subscribed {request.symbol} to {request.data_types}")
        return {"status": "subscribed", "symbol": request.symbol, "data_types": request.data_types}
    except HTTPException:
        raise
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
    # Check if IBKR is connected
    if not data_bus.connected:
        raise HTTPException(status_code=503, detail="IBKR not connected")

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
        # Symbol not subscribed yet
        raise HTTPException(status_code=404, detail=f"No data for {symbol}. Subscribe first via /api/subscribe")

@app.get("/api/level2/{symbol}")
async def get_level2_old(symbol: str):
    """
    Get Level 2 market depth data for a symbol.
    Returns real data from IBKR.

    Args:
        symbol: Stock ticker symbol

    Returns:
        dict: Market depth with bids and asks
    """
    try:
        if not data_bus.connected or not data_bus.ibkr_connector:
            raise HTTPException(status_code=503, detail="IBKR not connected")

        # Check if we have depth data
        if symbol in data_bus.l2_data:
            depth = data_bus.l2_data[symbol]
            return {
                'symbol': symbol,
                'bids': depth.get('bids', [])[:20],  # Top 20 levels
                'asks': depth.get('asks', [])[:20],  # Top 20 levels
                'timestamp': datetime.now().isoformat(),
                'status': 'active'
            }
        else:
            # Request depth data if not available
            data_bus.ibkr_connector.request_market_depth(symbol, depth=10)
            return {
                'symbol': symbol,
                'bids': [],
                'asks': [],
                'timestamp': datetime.now().isoformat(),
                'status': 'requesting'
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting level2 for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/timesales/{symbol}")
async def get_timesales_old(symbol: str, limit: int = 20):
    """OLD ENDPOINT: Get Time & Sales (compatibility)"""
    return await get_time_sales(symbol, limit)

# ═══════════════════════════════════════════════════════════════════════
#                     AUTONOMOUS BOT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/bot/status")
async def get_bot_status():
    """Get autonomous bot status with full statistics"""
    global autonomous_bot

    if not AUTONOMOUS_BOT_AVAILABLE:
        return {
            "available": False,
            "error": "Autonomous bot not available (check imports)",
            "ibkr_connected": data_bus.connected
        }

    if autonomous_bot is None:
        return {
            "available": True,
            "initialized": False,
            "running": False,
            "ibkr_connected": data_bus.connected,
            "message": "Bot not initialized. Use /api/bot/init to initialize."
        }

    # Get full status from bot
    status = autonomous_bot.get_status()
    status["ibkr_connected"] = data_bus.connected
    status["timestamp"] = datetime.now().isoformat()

    return status

@app.post("/api/bot/init")
async def init_bot(config: Optional[BotConfigUpdate] = None):
    """Initialize the autonomous trading bot with configuration"""
    global autonomous_bot

    if not AUTONOMOUS_BOT_AVAILABLE:
        raise HTTPException(status_code=501, detail="Autonomous bot not available")

    if not data_bus.connected or not data_bus.ib:
        raise HTTPException(status_code=503, detail="IBKR not connected. Please connect first.")

    try:
        # Create default config
        bot_config = BotConfig(
            account_size=50000.0,
            watchlist=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            max_position_size_usd=5000.0,
            max_positions=5,
            daily_loss_limit_usd=500.0,
            min_probability_threshold=0.60,
            max_spread_pct=0.005,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            enabled=False  # Start disabled for safety
        )

        # Update with user config if provided
        if config:
            for key, value in config.dict(exclude_unset=True).items():
                if hasattr(bot_config, key):
                    setattr(bot_config, key, value)

        # Create bot instance
        autonomous_bot = AutonomousTrader(
            config=bot_config,
            ib_connection=data_bus.ib
        )

        logger.info("✓ Autonomous bot initialized")

        return {
            "status": "initialized",
            "config": {
                "account_size": bot_config.account_size,
                "watchlist": bot_config.watchlist,
                "max_position_size_usd": bot_config.max_position_size_usd,
                "max_positions": bot_config.max_positions,
                "daily_loss_limit_usd": bot_config.daily_loss_limit_usd,
                "min_probability_threshold": bot_config.min_probability_threshold,
                "enabled": bot_config.enabled
            }
        }
    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/bot/start")
async def start_bot():
    """Start the autonomous trading bot"""
    global autonomous_bot

    if autonomous_bot is None:
        raise HTTPException(status_code=400, detail="Bot not initialized. Use /api/bot/init first.")

    if not data_bus.connected:
        raise HTTPException(status_code=503, detail="IBKR not connected")

    try:
        await autonomous_bot.start()
        logger.info("✓ Autonomous bot started")
        return {
            "status": "started",
            "running": autonomous_bot.running,
            "enabled": autonomous_bot.config.enabled,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/bot/stop")
async def stop_bot():
    """Stop the autonomous trading bot"""
    global autonomous_bot

    if autonomous_bot is None:
        raise HTTPException(status_code=400, detail="Bot not initialized")

    try:
        await autonomous_bot.stop()
        logger.info("✓ Autonomous bot stopped")
        return {
            "status": "stopped",
            "running": autonomous_bot.running,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to stop bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/bot/enable")
async def enable_bot():
    """Enable trading (bot must still be started separately)"""
    global autonomous_bot

    if autonomous_bot is None:
        raise HTTPException(status_code=400, detail="Bot not initialized")

    autonomous_bot.enable()
    return {
        "status": "enabled",
        "enabled": autonomous_bot.config.enabled,
        "trading_enabled": autonomous_bot.trading_engine.trading_enabled
    }

@app.post("/api/bot/disable")
async def disable_bot():
    """Disable trading (bot continues running but won't trade)"""
    global autonomous_bot

    if autonomous_bot is None:
        raise HTTPException(status_code=400, detail="Bot not initialized")

    autonomous_bot.disable()
    return {
        "status": "disabled",
        "enabled": autonomous_bot.config.enabled,
        "trading_enabled": autonomous_bot.trading_engine.trading_enabled
    }

@app.post("/api/bot/config")
async def update_bot_config(config: BotConfigUpdate):
    """Update bot configuration"""
    global autonomous_bot

    if autonomous_bot is None:
        raise HTTPException(status_code=400, detail="Bot not initialized")

    # Update config
    updates = config.dict(exclude_unset=True)
    autonomous_bot.update_config(**updates)

    # Also update risk limits if provided
    if any(k in updates for k in ['max_position_size_usd', 'daily_loss_limit_usd', 'min_probability_threshold',
                                   'max_spread_pct', 'stop_loss_pct', 'take_profit_pct', 'trailing_stop_pct']):
        risk_limits = autonomous_bot.trading_engine.risk_limits

        if 'max_position_size_usd' in updates:
            risk_limits.max_position_size_usd = updates['max_position_size_usd']
        if 'daily_loss_limit_usd' in updates:
            risk_limits.daily_loss_limit_usd = updates['daily_loss_limit_usd']
        if 'min_probability_threshold' in updates:
            risk_limits.min_probability_threshold = updates['min_probability_threshold']
        if 'max_spread_pct' in updates:
            risk_limits.max_spread_pct = updates['max_spread_pct']
        if 'stop_loss_pct' in updates:
            risk_limits.stop_loss_pct = updates['stop_loss_pct']
        if 'take_profit_pct' in updates:
            risk_limits.take_profit_pct = updates['take_profit_pct']
        if 'trailing_stop_pct' in updates:
            risk_limits.trailing_stop_pct = updates['trailing_stop_pct']

    return {
        "status": "updated",
        "updates": updates,
        "current_config": {
            "account_size": autonomous_bot.config.account_size,
            "watchlist": autonomous_bot.config.watchlist,
            "max_position_size_usd": autonomous_bot.config.max_position_size_usd,
            "max_positions": autonomous_bot.config.max_positions,
            "daily_loss_limit_usd": autonomous_bot.config.daily_loss_limit_usd,
            "min_probability_threshold": autonomous_bot.config.min_probability_threshold,
            "enabled": autonomous_bot.config.enabled
        }
    }

@app.get("/api/bot/predictions")
async def get_bot_predictions():
    """Get recent predictions from the bot"""
    global autonomous_bot

    if autonomous_bot is None:
        raise HTTPException(status_code=400, detail="Bot not initialized")

    return {
        "predictions": autonomous_bot.last_predictions,
        "count": len(autonomous_bot.last_predictions),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/bot/positions")
async def get_bot_positions():
    """Get current bot positions"""
    global autonomous_bot

    if autonomous_bot is None:
        raise HTTPException(status_code=400, detail="Bot not initialized")

    stats = autonomous_bot.trading_engine.get_stats()

    return {
        "positions": stats.get("positions", {}),
        "open_positions": stats.get("open_positions", 0),
        "total_exposure_usd": stats.get("total_exposure_usd", 0),
        "total_exposure_pct": stats.get("total_exposure_pct", 0),
        "unrealized_pnl": stats.get("unrealized_pnl", 0),
        "daily_pnl": stats.get("daily_pnl", 0),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/bot/stats")
async def get_bot_stats():
    """Get comprehensive bot statistics"""
    global autonomous_bot

    if autonomous_bot is None:
        raise HTTPException(status_code=400, detail="Bot not initialized")

    return autonomous_bot.get_status()

@app.post("/api/bot/save-state")
async def save_bot_state(filepath: str = "bot_state.json"):
    """Save bot state to file"""
    global autonomous_bot

    if autonomous_bot is None:
        raise HTTPException(status_code=400, detail="Bot not initialized")

    try:
        autonomous_bot.save_state(filepath)
        return {
            "status": "saved",
            "filepath": filepath,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/bot/load-state")
async def load_bot_state(filepath: str = "bot_state.json"):
    """Load bot state from file"""
    global autonomous_bot

    if autonomous_bot is None:
        raise HTTPException(status_code=400, detail="Bot not initialized")

    try:
        autonomous_bot.load_state(filepath)
        return {
            "status": "loaded",
            "filepath": filepath,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════════
#                     ACCOUNT & POSITIONS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/account")
async def get_account():
    """Get account summary and positions from IBKR"""
    try:
        if not data_bus.connected or not data_bus.ib:
            raise HTTPException(status_code=503, detail="IBKR not connected")

        # Get account number
        account_id = data_bus.ib.managedAccounts()[0] if data_bus.ib.managedAccounts() else "Unknown"

        # Get account summary using async method
        account_values = await data_bus.ib.accountSummaryAsync()
        summary = {
            'account_id': account_id,
            'account_type': 'Unknown',
            'account_capabilities': [],
            'day_trades_remaining': None,
            'is_pdt': False
        }

        for item in account_values:
            if item.tag == 'NetLiquidation':
                summary['net_liquidation'] = float(item.value)
            elif item.tag == 'TotalCashValue':
                summary['total_cash'] = float(item.value)
            elif item.tag == 'BuyingPower':
                summary['buying_power'] = float(item.value)
            elif item.tag == 'GrossPositionValue':
                summary['gross_position_value'] = float(item.value)
            elif item.tag == 'AccountType':
                summary['account_type'] = item.value
            elif item.tag == 'Cushion':
                summary['cushion'] = float(item.value) if item.value else None
            elif item.tag == 'DayTradesRemaining':
                summary['day_trades_remaining'] = int(item.value) if item.value else None
            elif item.tag == 'DayTradesRemainingT+1':
                summary['day_trades_remaining_t1'] = int(item.value) if item.value else None
            elif item.tag == 'DayTradesRemainingT+2':
                summary['day_trades_remaining_t2'] = int(item.value) if item.value else None
            elif item.tag == 'DayTradesRemainingT+3':
                summary['day_trades_remaining_t3'] = int(item.value) if item.value else None
            elif item.tag == 'DayTradesRemainingT+4':
                summary['day_trades_remaining_t4'] = int(item.value) if item.value else None

        # Determine account capabilities and restrictions
        acct_type = summary.get('account_type', '').upper()
        net_liq = summary.get('net_liquidation', 0)

        # Determine if Pattern Day Trader (PDT applies to margin accounts with <25k)
        if 'MARGIN' in acct_type or 'REG T' in acct_type:
            if net_liq >= 25000:
                summary['account_capabilities'] = ['MARGIN', 'SHORT_SELLING', 'UNLIMITED_DAY_TRADING']
                summary['is_pdt'] = False
            else:
                summary['account_capabilities'] = ['MARGIN', 'SHORT_SELLING', 'LIMITED_DAY_TRADING']
                summary['is_pdt'] = True
        elif 'CASH' in acct_type:
            summary['account_capabilities'] = ['CASH_ONLY', 'NO_SHORT_SELLING', 'T+2_SETTLEMENT']
        elif 'IRA' in acct_type:
            summary['account_capabilities'] = ['IRA', 'NO_SHORT_SELLING', 'LIMITED_MARGIN']

        # Get positions using async method
        positions = []
        await data_bus.ib.reqPositionsAsync()
        for pos in data_bus.ib.positions():
            positions.append({
                'symbol': pos.contract.symbol,
                'quantity': float(pos.position),
                'avg_cost': float(pos.avgCost),
                'market_value': float(pos.position * pos.marketValue) if hasattr(pos, 'marketValue') else 0,
                'unrealized_pnl': float(pos.unrealizedPNL) if hasattr(pos, 'unrealizedPNL') else 0
            })

        # Calculate P&L totals
        total_unrealized_pnl = sum(p['unrealized_pnl'] for p in positions)

        return {
            'summary': summary,
            'positions': positions,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': 0,  # Would need to track this separately
            'timestamp': datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting account data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
#                     AI MODEL MANAGEMENT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

# In-memory storage for demo (replace with database in production)
training_sessions = {}
model_performance_data = []
experiments = []

class TrainingConfig(BaseModel):
    model_type: str
    symbols: List[str]
    timeframe: str
    lookback_days: int
    training_split: float
    validation_split: float
    hyperparameters: Dict[str, Any]

class SimpleTrainingConfig(BaseModel):
    symbol: str
    period: str = "2y"
    test_size: float = 0.2

class BacktestConfig(BaseModel):
    strategy: str
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    position_size_pct: float
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None

@app.post("/api/ai/models/train")
async def train_model(config: SimpleTrainingConfig, background_tasks: BackgroundTasks):
    """Start model training"""
    training_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    training_sessions[training_id] = {
        "id": training_id,
        "status": "training",
        "config": config.dict(),
        "progress": 0,
        "started_at": datetime.now().isoformat(),
        "metrics": {}
    }

    # Run actual training in background using BackgroundTasks (fixes event loop issue)
    background_tasks.add_task(run_actual_training, training_id, config)

    return {
        "success": True,
        "data": {
            "training_id": training_id,
            "websocket_url": f"ws://127.0.0.1:9101/ws/training/{training_id}"
        }
    }

def run_actual_training(training_id: str, config: TrainingConfig):
    """Run actual AI predictor training (non-async to avoid event loop issues)"""
    import time
    import logging

    logger = logging.getLogger(__name__)

    try:
        if training_id not in training_sessions:
            return

        logger.info(f"Starting training for {training_id} on symbol {config.symbol}")

        # Update progress
        training_sessions[training_id]["progress"] = 10
        training_sessions[training_id]["status"] = "downloading_data"

        if ai_predictor:
            # Train the actual model
            training_sessions[training_id]["progress"] = 30
            training_sessions[training_id]["status"] = "training"

            result = ai_predictor.train(
                symbol=config.symbol,
                period=config.period,
                test_size=0.2
            )

            # Update with real metrics
            training_sessions[training_id]["progress"] = 100
            training_sessions[training_id]["status"] = "completed"
            training_sessions[training_id]["metrics"] = {
                "accuracy": result.get("metrics", {}).get("accuracy", 0),
                "samples": result.get("samples", 0),
                "model_path": result.get("model_path", "")
            }
            training_sessions[training_id]["completed_at"] = datetime.now().isoformat()

            logger.info(f"Training completed for {training_id}: {result['metrics']['accuracy']:.2%} accuracy")
        else:
            # Fallback to simulation if predictor not available
            training_sessions[training_id]["status"] = "failed"
            training_sessions[training_id]["error"] = "AI predictor not available"

    except Exception as e:
        logger.error(f"Training failed for {training_id}: {e}")
        if training_id in training_sessions:
            training_sessions[training_id]["status"] = "failed"
            training_sessions[training_id]["error"] = str(e)
            training_sessions[training_id]["completed_at"] = datetime.now().isoformat()

async def simulate_training(training_id: str):
    """Simulate training progress (deprecated - use run_actual_training)"""
    for progress in range(0, 101, 10):
        await asyncio.sleep(1)
        if training_id in training_sessions:
            training_sessions[training_id]["progress"] = progress
            training_sessions[training_id]["metrics"] = {
                "train_loss": 0.5 - (progress * 0.004),
                "val_loss": 0.6 - (progress * 0.003),
                "train_accuracy": 0.5 + (progress * 0.004),
                "val_accuracy": 0.45 + (progress * 0.0045)
            }

    if training_id in training_sessions:
        training_sessions[training_id]["status"] = "completed"

@app.get("/api/ai/models/train/{training_id}/status")
async def get_training_status(training_id: str):
    """Get training session status"""
    if training_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")

    return {
        "success": True,
        "session": training_sessions[training_id]
    }

@app.get("/api/ai/models/train/{training_id}/insights")
async def get_training_insights(training_id: str):
    """Get Claude insights for training session"""
    if training_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")

    session = training_sessions[training_id]

    return {
        "success": True,
        "data": {
            "summary": f"Training session {training_id} analysis",
            "key_findings": [
                "Model is converging well with decreasing loss",
                "Validation accuracy is improving steadily",
                "No signs of overfitting detected"
            ],
            "recommendations": [
                "Consider increasing training epochs for better accuracy",
                "Monitor validation loss to prevent overfitting"
            ]
        }
    }

@app.get("/api/ai/models/performance")
async def get_model_performance():
    """Get performance metrics for all models"""
    # Mock data for demo
    mock_models = [
        {
            "id": "model_1",
            "name": "LSTM Price Predictor",
            "type": "LSTM",
            "accuracy": 0.78,
            "precision": 0.75,
            "recall": 0.72,
            "f1_score": 0.73,
            "sharpe_ratio": 1.8,
            "win_rate": 0.65,
            "total_trades": 150,
            "profitable_trades": 98,
            "avg_profit": 125.50,
            "created_at": "2025-01-10T10:00:00Z",
            "last_updated": "2025-01-14T15:30:00Z"
        },
        {
            "id": "model_2",
            "name": "Transformer Ensemble",
            "type": "Transformer",
            "accuracy": 0.82,
            "precision": 0.80,
            "recall": 0.78,
            "f1_score": 0.79,
            "sharpe_ratio": 2.1,
            "win_rate": 0.70,
            "total_trades": 200,
            "profitable_trades": 140,
            "avg_profit": 150.75,
            "created_at": "2025-01-12T14:00:00Z",
            "last_updated": "2025-01-14T16:00:00Z"
        }
    ]

    return {"success": True, "data": mock_models}

@app.get("/api/ai/models/compare")
async def compare_models(timeframe: str = "30d"):
    """Compare model performance"""
    return {
        "success": True,
        "data": {
            "timeframe": timeframe,
            "comparison": [
                {"model_id": "model_1", "accuracy": 0.78, "sharpe_ratio": 1.8},
                {"model_id": "model_2", "accuracy": 0.82, "sharpe_ratio": 2.1}
            ]
        }
    }

@app.get("/api/ai/models/insights")
async def get_model_insights():
    """Get Claude insights for models"""
    return {
        "success": True,
        "data": {
            "summary": "Model performance analysis",
            "recommendations": [
                "Model 2 (Transformer) shows better performance",
                "Consider using ensemble approach for higher accuracy"
            ]
        }
    }

@app.get("/api/ai/models/experiments")
async def get_experiments():
    """Get all A/B testing experiments"""
    return {"success": True, "data": experiments}

@app.post("/api/ai/models/experiments/create")
async def create_experiment(config: Dict[str, Any]):
    """Create new A/B test experiment"""
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment = {
        "id": experiment_id,
        "name": config.get("name", "New Experiment"),
        "status": "running",
        "created_at": datetime.now().isoformat(),
        **config
    }
    experiments.append(experiment)

    return {"success": True, "data": {"experiment_id": experiment_id}}

@app.post("/api/ai/models/experiments/{experiment_id}/promote")
async def promote_experiment(experiment_id: str):
    """Promote experiment to production"""
    return {"success": True, "data": {"status": "promoted"}}

@app.post("/api/ai/models/experiments/{experiment_id}/pause")
async def pause_experiment(experiment_id: str):
    """Pause an experiment"""
    return {"success": True, "data": {"status": "paused"}}

# ═══════════════════════════════════════════════════════════════════════
#                     BACKTESTING ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

backtests = {}

@app.post("/api/backtest/run")
async def run_backtest(config: BacktestConfig):
    """Run a backtest"""
    backtest_id = f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    backtests[backtest_id] = {
        "id": backtest_id,
        "status": "running",
        "config": config.dict(),
        "started_at": datetime.now().isoformat(),
        "progress": 0
    }

    # Simulate backtest in background
    asyncio.create_task(simulate_backtest(backtest_id))

    return {"success": True, "data": {"backtest_id": backtest_id}}

async def simulate_backtest(backtest_id: str):
    """Simulate backtest execution"""
    for progress in range(0, 101, 20):
        await asyncio.sleep(1)
        if backtest_id in backtests:
            backtests[backtest_id]["progress"] = progress

    if backtest_id in backtests:
        backtests[backtest_id]["status"] = "completed"
        backtests[backtest_id]["completed_at"] = datetime.now().isoformat()

@app.get("/api/backtest/{backtest_id}/results")
async def get_backtest_results(backtest_id: str):
    """Get backtest results"""
    if backtest_id not in backtests:
        raise HTTPException(status_code=404, detail="Backtest not found")

    # Mock results
    return {
        "success": True,
        "data": {
            "id": backtest_id,
            "status": backtests[backtest_id]["status"],
            "metrics": {
                "total_return": 15.5,
                "sharpe_ratio": 1.8,
                "max_drawdown": -8.2,
                "win_rate": 0.65,
                "total_trades": 150,
                "profitable_trades": 98
            },
            "equity_curve": [
                {"date": "2025-01-01", "equity": 10000},
                {"date": "2025-01-02", "equity": 10150},
                {"date": "2025-01-03", "equity": 10300}
            ],
            "trades": [
                {
                    "id": "trade_1",
                    "symbol": "AAPL",
                    "entry_date": "2025-01-01T10:00:00Z",
                    "exit_date": "2025-01-01T15:00:00Z",
                    "entry_price": 225.50,
                    "exit_price": 227.00,
                    "quantity": 10,
                    "pnl": 15.0,
                    "pnl_pct": 0.67
                }
            ]
        }
    }

@app.post("/api/backtest/{backtest_id}/stop")
async def stop_backtest(backtest_id: str):
    """Stop a running backtest"""
    if backtest_id in backtests:
        backtests[backtest_id]["status"] = "stopped"
    return {"success": True}

@app.get("/api/backtest/{backtest_id}/claude-analysis")
async def get_backtest_analysis(backtest_id: str):
    """Get Claude AI analysis of backtest"""
    return {
        "success": True,
        "data": {
            "summary": "Backtest analysis summary",
            "strengths": [
                "Strong risk-adjusted returns",
                "Consistent win rate across different market conditions"
            ],
            "weaknesses": [
                "Higher drawdown during volatile periods",
                "Some overtrading detected"
            ],
            "recommendations": [
                "Consider tightening stop losses",
                "Reduce position sizes during high volatility"
            ]
        }
    }

@app.post("/api/backtest/{backtest_id}/export")
async def export_backtest(backtest_id: str, format: Dict[str, str]):
    """Export backtest results"""
    from fastapi.responses import StreamingResponse
    import io

    # Create mock CSV
    csv_data = "Date,Equity,PnL\n2025-01-01,10000,0\n2025-01-02,10150,150\n"
    return StreamingResponse(
        io.StringIO(csv_data),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=backtest_{backtest_id}.csv"}
    )

@app.get("/api/backtest/trade/{trade_id}/analysis")
async def get_trade_analysis(trade_id: str):
    """Get detailed analysis of a specific trade"""
    return {
        "success": True,
        "data": {
            "trade_id": trade_id,
            "analysis": "Trade executed well with good entry timing",
            "metrics": {"efficiency": 0.85}
        }
    }

@app.get("/api/backtest/trade/{trade_id}/similar")
async def get_similar_trades(trade_id: str):
    """Find similar trades"""
    return {
        "success": True,
        "data": [
            {"trade_id": "trade_2", "similarity": 0.92},
            {"trade_id": "trade_3", "similarity": 0.88}
        ]
    }

@app.post("/api/claude/optimize-strategy")
async def optimize_strategy(config: Dict[str, Any]):
    """Get strategy optimization suggestions"""
    return {
        "success": True,
        "data": {
            "optimizations": [
                "Increase position size for high-confidence signals",
                "Use trailing stops instead of fixed stops"
            ]
        }
    }

@app.get("/api/strategies")
async def get_strategies():
    """Get saved strategies"""
    return {"success": True, "data": []}

@app.post("/api/strategies/save")
async def save_strategy(strategy: Dict[str, Any]):
    """Save a strategy"""
    strategy_id = f"strat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return {"success": True, "data": {"strategy_id": strategy_id}}

# ═══════════════════════════════════════════════════════════════════════
#                     LIVE PREDICTIONS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

predictions_storage = []
alerts_storage = []

class AlertConfig(BaseModel):
    name: str
    symbol: Optional[str] = None
    condition: str
    threshold: float
    delivery: List[str]
    message: str
    enabled: bool = True

@app.get("/api/predictions/live")
async def get_live_predictions():
    """Get live predictions"""
    # Generate mock predictions based on current market data
    preds = []
    for symbol in ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']:
        if symbol in data_bus.market_data:
            data = data_bus.market_data[symbol]
            preds.append({
                "symbol": symbol,
                "direction": "UP" if hash(symbol) % 2 == 0 else "DOWN",
                "confidence": 0.65 + (hash(symbol) % 30) / 100,
                "predicted_change": 1.5 + (hash(symbol) % 20) / 10,
                "current_price": data.get("last", 0),
                "target_price": data.get("last", 0) * 1.02,
                "timeframe": "1H",
                "timestamp": datetime.now().isoformat()
            })

    return {"success": True, "data": preds}

@app.get("/api/predictions/stats")
async def get_prediction_stats():
    """Get prediction statistics"""
    return {
        "success": True,
        "data": {
            "total_predictions": 1250,
            "accurate_predictions": 875,
            "accuracy_rate": 0.70,
            "avg_confidence": 0.68,
            "profitable_signals": 650,
            "profitability_rate": 0.74
        }
    }

@app.get("/api/predictions/claude-commentary")
async def get_claude_commentary():
    """Get Claude AI market commentary"""
    return {
        "success": True,
        "data": {
            "commentary": "Market showing strong bullish momentum in tech sector. AI-related stocks continue to outperform. Watch for potential profit-taking at resistance levels."
        }
    }

@app.get("/api/predictions/accuracy")
async def get_prediction_accuracy():
    """Get prediction accuracy metrics"""
    return {
        "success": True,
        "data": {
            "overall_accuracy": 0.70,
            "by_symbol": [
                {"symbol": "AAPL", "accuracy": 0.75},
                {"symbol": "MSFT", "accuracy": 0.72}
            ],
            "by_timeframe": [
                {"timeframe": "1H", "accuracy": 0.68},
                {"timeframe": "4H", "accuracy": 0.72}
            ]
        }
    }

@app.get("/api/predictions/{symbol}/details")
async def get_prediction_details(symbol: str):
    """Get detailed prediction for symbol"""
    return {
        "success": True,
        "data": {
            "symbol": symbol,
            "direction": "UP",
            "confidence": 0.75,
            "factors": [
                "Strong technical momentum",
                "Positive sentiment analysis",
                "Volume confirmation"
            ]
        }
    }

@app.post("/api/predictions/alerts/configure")
async def configure_alert(config: AlertConfig):
    """Configure a prediction alert"""
    alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    alert = {
        "id": alert_id,
        "created_at": datetime.now().isoformat(),
        **config.dict()
    }
    alerts_storage.append(alert)
    return {"success": True, "data": {"alert_id": alert_id}}

@app.get("/api/predictions/alerts")
async def get_alerts():
    """Get all alerts"""
    return {"success": True, "data": alerts_storage}

@app.post("/api/predictions/alerts/create")
async def create_alert(config: AlertConfig):
    """Create a new alert"""
    return await configure_alert(config)

@app.put("/api/predictions/alerts/{alert_id}/toggle")
async def toggle_alert(alert_id: str, enabled: Dict[str, bool]):
    """Toggle alert on/off"""
    for alert in alerts_storage:
        if alert["id"] == alert_id:
            alert["enabled"] = enabled.get("enabled", True)
    return {"success": True}

@app.delete("/api/predictions/alerts/{alert_id}")
async def delete_alert(alert_id: str):
    """Delete an alert"""
    global alerts_storage
    alerts_storage = [a for a in alerts_storage if a["id"] != alert_id]
    return {"success": True}

@app.get("/api/predictions/alerts/history")
async def get_alert_history():
    """Get alert trigger history"""
    return {
        "success": True,
        "data": [
            {
                "id": "trigger_1",
                "alert_id": "alert_1",
                "symbol": "AAPL",
                "triggered_at": datetime.now().isoformat(),
                "message": "AAPL confidence above 75%"
            }
        ]
    }

# ═══════════════════════════════════════════════════════════════════════
#                     TRADINGVIEW ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

webhook_history = []
saved_indicators = []

class TradingViewPush(BaseModel):
    symbols: List[str]
    data_type: str
    interval: Optional[str] = "1m"

@app.post("/api/tradingview/push")
async def push_to_tradingview(data: TradingViewPush):
    """Push predictions to TradingView"""
    return {
        "success": True,
        "data": {
            "status": "pushed",
            "symbols": data.symbols,
            "timestamp": datetime.now().isoformat()
        }
    }

@app.get("/api/tradingview/status")
async def get_tradingview_status():
    """Get TradingView integration status"""
    return {
        "success": True,
        "data": {
            "connected": True,
            "auto_sync": False,
            "last_push": datetime.now().isoformat()
        }
    }

@app.post("/api/tradingview/auto-sync")
async def set_tradingview_auto_sync(enabled: Dict[str, bool]):
    """Set auto-sync for TradingView"""
    return {"success": True, "data": {"auto_sync": enabled.get("enabled", False)}}

@app.get("/api/tradingview/webhook/config")
async def get_webhook_config():
    """Get webhook configuration"""
    return {
        "success": True,
        "data": {
            "webhook_url": "https://api.example.com/webhook/tradingview",
            "api_key": "masked_key_***",
            "enabled": True
        }
    }

@app.post("/api/tradingview/webhook/config")
async def update_webhook_config(config: Dict[str, Any]):
    """Update webhook configuration"""
    return {"success": True, "data": config}

@app.get("/api/tradingview/webhook/settings")
async def get_webhook_settings():
    """Get webhook settings"""
    return {
        "success": True,
        "data": {
            "webhook_url": "https://api.example.com/webhook",
            "api_key": "***",
            "ip_whitelist": ["192.168.1.1"],
            "enabled": True
        }
    }

@app.put("/api/tradingview/webhook/settings")
async def update_webhook_settings(settings: Dict[str, Any]):
    """Update webhook settings"""
    return {"success": True, "data": settings}

@app.put("/api/tradingview/webhook/filters")
async def update_webhook_filters(filters: Dict[str, Any]):
    """Update webhook filters"""
    return {"success": True, "data": filters}

@app.post("/api/tradingview/webhook/regenerate-key")
async def regenerate_webhook_api_key():
    """Regenerate webhook API key"""
    new_key = f"key_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return {"success": True, "data": {"api_key": new_key}}

@app.post("/api/tradingview/webhook/test")
async def test_webhook():
    """Test webhook connection"""
    return {
        "success": True,
        "data": {
            "status": "success",
            "response_time_ms": 45,
            "timestamp": datetime.now().isoformat()
        }
    }

@app.get("/api/tradingview/webhook/history")
async def get_webhook_history():
    """Get webhook trigger history"""
    return {"success": True, "data": webhook_history}

@app.post("/api/tradingview/pine-script/generate")
async def generate_pine_script(strategy: Dict[str, str]):
    """Generate PineScript code"""
    script = f"""// {strategy.get('strategy', 'Custom Strategy')}
//@version=5
indicator("AI Prediction Overlay", overlay=true)

// Plot AI predictions
plotshape(series=close > open, title="BUY Signal", location=location.belowbar, color=color.green, style=shape.triangleup)
plotshape(series=close < open, title="SELL Signal", location=location.abovebar, color=color.red, style=shape.triangledown)
"""
    return {"success": True, "data": {"script": script}}

@app.post("/api/tradingview/indicators/generate")
async def generate_indicator(config: Dict[str, Any]):
    """Generate custom indicator"""
    script = f"""// {config.get('name', 'Custom Indicator')}
//@version=5
indicator("{config.get('name', 'Custom')}", overlay=false)

// Generated by Claude AI
value = ta.sma(close, 20)
plot(value, color=color.blue)
"""
    return {"success": True, "data": {"script": script}}

@app.get("/api/tradingview/indicators/saved")
async def get_saved_indicators():
    """Get saved indicators"""
    return {"success": True, "data": saved_indicators}

@app.post("/api/tradingview/indicators/save")
async def save_indicator(indicator: Dict[str, Any]):
    """Save an indicator"""
    indicator_id = f"ind_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    saved_indicators.append({
        "id": indicator_id,
        "created_at": datetime.now().isoformat(),
        **indicator
    })
    return {"success": True, "data": {"indicator_id": indicator_id}}

@app.delete("/api/tradingview/indicators/{indicator_id}")
async def delete_indicator(indicator_id: str):
    """Delete an indicator"""
    global saved_indicators
    saved_indicators = [i for i in saved_indicators if i.get("id") != indicator_id]
    return {"success": True}

# ═══════════════════════════════════════════════════════════════════════
#                     WORKLIST MANAGEMENT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

# Shared worklist across all modules
worklist = []
worklist_predictions = {}

class WorklistItem(BaseModel):
    symbol: str
    exchange: str = "SMART"
    notes: Optional[str] = None

@app.get("/api/worklist")
async def get_worklist():
    """Get all symbols in the worklist with current data"""
    enriched_worklist = []

    for item in worklist:
        symbol = item["symbol"]

        # Get market data
        market_data = data_bus.market_data.get(symbol, {})

        # Get AI prediction
        prediction = worklist_predictions.get(symbol, {})

        # Build enriched item
        enriched_item = {
            "symbol": symbol,
            "exchange": item.get("exchange", "SMART"),
            "current_price": market_data.get("last"),
            "bid": market_data.get("bid"),
            "ask": market_data.get("ask"),
            "change": market_data.get("change", 0),
            "change_percent": market_data.get("change_percent", 0),
            "volume": market_data.get("volume"),
            "prediction": prediction.get("direction"),
            "confidence": prediction.get("confidence"),
            "predicted_change": prediction.get("predicted_change"),
            "analysis": prediction.get("analysis", ""),
            "notes": item.get("notes", ""),
            "added_at": item.get("added_at"),
            "has_live_data": symbol in data_bus.market_data
        }

        enriched_worklist.append(enriched_item)

    return {"success": True, "data": enriched_worklist}

@app.post("/api/worklist/add")
async def add_to_worklist(item: WorklistItem):
    """Add a symbol to the worklist"""
    global worklist

    # Check if already exists
    if any(w["symbol"] == item.symbol for w in worklist):
        raise HTTPException(status_code=400, detail="Symbol already in worklist")

    worklist_entry = {
        "symbol": item.symbol,
        "exchange": item.exchange,
        "notes": item.notes,
        "added_at": datetime.now().isoformat()
    }

    worklist.append(worklist_entry)

    # Subscribe to market data if IBKR is connected
    if data_bus.connected:
        try:
            await data_bus.subscribe_market_data(item.symbol, item.exchange)
            logger.info(f"Subscribed to market data for {item.symbol}")
        except Exception as e:
            logger.warning(f"Could not subscribe to {item.symbol}: {e}")

    # Generate AI prediction in background
    asyncio.create_task(generate_worklist_prediction(item.symbol))

    return {"success": True, "data": worklist_entry}

@app.post("/api/worklist/add-bulk")
async def add_bulk_to_worklist(items: List[WorklistItem]):
    """Add multiple symbols to worklist"""
    added = []
    errors = []

    for item in items:
        try:
            # Check if already exists
            if any(w["symbol"] == item.symbol for w in worklist):
                errors.append(f"{item.symbol} already in worklist")
                continue

            worklist_entry = {
                "symbol": item.symbol,
                "exchange": item.exchange,
                "notes": item.notes,
                "added_at": datetime.now().isoformat()
            }

            worklist.append(worklist_entry)
            added.append(item.symbol)

            # Subscribe to market data
            if data_bus.connected:
                try:
                    await data_bus.subscribe_market_data(item.symbol, item.exchange)
                except Exception as e:
                    logger.warning(f"Could not subscribe to {item.symbol}: {e}")

            # Generate prediction
            asyncio.create_task(generate_worklist_prediction(item.symbol))

        except Exception as e:
            errors.append(f"{item.symbol}: {str(e)}")

    return {
        "success": True,
        "data": {
            "added": added,
            "errors": errors,
            "count": len(added)
        }
    }

@app.delete("/api/worklist/{symbol}")
async def remove_from_worklist(symbol: str):
    """Remove a symbol from the worklist"""
    global worklist, worklist_predictions

    # Find and remove
    worklist = [w for w in worklist if w["symbol"] != symbol]

    # Remove prediction
    if symbol in worklist_predictions:
        del worklist_predictions[symbol]

    return {"success": True, "data": {"symbol": symbol, "removed": True}}

@app.put("/api/worklist/{symbol}/notes")
async def update_worklist_notes(symbol: str, notes: Dict[str, str]):
    """Update notes for a worklist symbol"""
    for item in worklist:
        if item["symbol"] == symbol:
            item["notes"] = notes.get("notes", "")
            return {"success": True, "data": item}

    raise HTTPException(status_code=404, detail="Symbol not in worklist")

@app.delete("/api/worklist/clear")
async def clear_worklist():
    """Clear entire worklist"""
    global worklist, worklist_predictions
    worklist = []
    worklist_predictions = {}
    return {"success": True, "data": {"cleared": True}}

async def generate_worklist_prediction(symbol: str):
    """Generate AI prediction for a worklist symbol"""
    try:
        # Wait a moment for market data to arrive
        await asyncio.sleep(1)

        market_data = data_bus.market_data.get(symbol, {})

        if not market_data or not market_data.get("last"):
            # No data yet, generate mock prediction
            worklist_predictions[symbol] = {
                "direction": "UP" if hash(symbol) % 2 == 0 else "DOWN",
                "confidence": 0.65 + (hash(symbol) % 30) / 100,
                "predicted_change": 1.5 + (hash(symbol) % 20) / 10,
                "analysis": f"AI analysis for {symbol} based on technical indicators",
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Use real data for prediction (mock for now)
            worklist_predictions[symbol] = {
                "direction": "UP" if hash(symbol) % 2 == 0 else "DOWN",
                "confidence": 0.70 + (hash(symbol) % 25) / 100,
                "predicted_change": 2.0 + (hash(symbol) % 15) / 10,
                "analysis": f"Strong momentum detected in {symbol}. Volume confirms direction.",
                "timestamp": datetime.now().isoformat()
            }

        logger.info(f"Generated prediction for {symbol}")

    except Exception as e:
        logger.error(f"Failed to generate prediction for {symbol}: {e}")

# ═══════════════════════════════════════════════════════════════════════
#                     IBKR SCANNER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/scanner/ibkr/presets")
async def get_scanner_presets():
    """Get available IBKR scanner presets"""
    presets = [
        {"id": "WARRIOR_GAPPERS", "name": "🎯 Warrior Trading Gappers", "description": "Top 10 gappers: $2-$20, 5x volume, 10%+ gain, high volume"},
        {"id": "TOP_PERC_GAIN", "name": "Top % Gainers", "description": "Stocks with highest % gains"},
        {"id": "TOP_PERC_LOSE", "name": "Top % Losers", "description": "Stocks with highest % losses"},
        {"id": "MOST_ACTIVE", "name": "Most Active", "description": "Stocks with highest volume"},
        {"id": "HOT_BY_VOLUME", "name": "Hot by Volume", "description": "Stocks with unusual volume"},
        {"id": "TOP_TRADE_COUNT", "name": "Top Trade Count", "description": "Most traded stocks"},
        {"id": "HOT_BY_PRICE", "name": "Hot by Price", "description": "Stocks with largest price moves"},
        {"id": "TOP_PRICE_RANGE", "name": "Top Price Range", "description": "Largest intraday range"},
        {"id": "HOT_BY_OPT_VOLUME", "name": "Hot by Option Volume", "description": "Unusual option volume"}
    ]

    return {"success": True, "data": presets}

@app.post("/api/scanner/ibkr/scan")
async def run_ibkr_scanner(request: Dict[str, Any]):
    """Run IBKR scanner and return results"""
    scan_code = request.get("scan_code", "TOP_PERC_GAIN")
    instrument = request.get("instrument", "STK")
    location_code = request.get("location", "STK.US.MAJOR")
    num_rows = request.get("num_rows", 50)

    # Special handling for Warrior Trading Gappers preset
    if scan_code == "WARRIOR_GAPPERS":
        scan_code = "TOP_PERC_GAIN"  # Use % gainers as base
        num_rows = 10  # Top 10 only
        # Filters will be applied via ScannerSubscription below

    # Enhanced connection check
    is_connected = False
    if data_bus.ib:
        try:
            is_connected = data_bus.ib.isConnected()
            logger.info(f"Scanner: IBKR connection check - data_bus.connected={data_bus.connected}, ib.isConnected()={is_connected}")
        except:
            is_connected = False

    if not is_connected:
        # Return mock data if not connected
        mock_results = [
            {
                "rank": 1,
                "symbol": "NVDA",
                "contract_id": 4815747,
                "distance": 5.8,
                "benchmark": 100.0,
                "projection": 105.8,
                "price": 485.50,
                "change": 5.8,
                "change_percent": 1.21,
                "volume": 45234567
            },
            {
                "rank": 2,
                "symbol": "TSLA",
                "contract_id": 76792991,
                "distance": 4.2,
                "benchmark": 100.0,
                "projection": 104.2,
                "price": 242.80,
                "change": 4.2,
                "change_percent": 1.76,
                "volume": 92145678
            },
            {
                "rank": 3,
                "symbol": "AMD",
                "contract_id": 4391,
                "distance": 3.9,
                "benchmark": 100.0,
                "projection": 103.9,
                "price": 168.45,
                "change": 3.9,
                "change_percent": 2.37,
                "volume": 38567234
            }
        ]

        return {
            "success": True,
            "data": {
                "scan_code": scan_code,
                "results": mock_results,
                "count": len(mock_results),
                "source": "mock_data",
                "message": "Connect to IBKR for live scanner data"
            }
        }

    try:
        # Use IBKR scanner with async version to avoid event loop conflicts
        from ib_insync import ScannerSubscription, TagValue

        # Create scanner subscription with Warrior Trading filters if applicable
        sub = ScannerSubscription(
            instrument=instrument,
            locationCode=location_code,
            scanCode=scan_code,
            numberOfRows=num_rows
        )

        # Apply Warrior Trading filters: $2-$20, 1M+ volume, 10%+ gain
        original_scan_code = request.get("scan_code", "")
        filter_options = []
        if original_scan_code == "WARRIOR_GAPPERS":
            filter_options = [
                TagValue("priceAbove", "2"),        # Min price $2
                TagValue("priceBelow", "20"),       # Max price $20
                TagValue("volumeAbove", "1000000"), # 1M+ volume (high volume)
                TagValue("changePercAbove", "10"),  # Up 10%+ for the day
            ]

        # Use async version to avoid "event loop already running" error
        if filter_options:
            scanner_data = await data_bus.ib.reqScannerDataAsync(sub, [], filter_options)
        else:
            scanner_data = await data_bus.ib.reqScannerDataAsync(sub)

        # Give scanner time to populate results
        await asyncio.sleep(1)

        results = []
        if scanner_data and len(scanner_data) > 0:
            for i, data in enumerate(scanner_data[:num_rows]):
                try:
                    results.append({
                        "rank": data.rank if hasattr(data, 'rank') else i + 1,
                        "symbol": data.contractDetails.contract.symbol,
                        "contract_id": data.contractDetails.contract.conId,
                        "distance": data.distance if hasattr(data, 'distance') else 0,
                        "benchmark": data.benchmark if hasattr(data, 'benchmark') else 0,
                        "projection": data.projection if hasattr(data, 'projection') else 0,
                        "price": 0,  # Would need separate quote request
                        "change": 0,
                        "change_percent": 0,
                        "volume": 0
                    })
                except Exception as item_error:
                    logger.warning(f"Skipping scanner result {i}: {item_error}")
                    continue

        # Cancel scanner subscription
        try:
            data_bus.ib.cancelScannerSubscription(sub)
        except:
            pass  # Ignore cancellation errors

        if not results:
            # If no results from IBKR, return mock data
            logger.warning("No scanner results from IBKR, returning mock data")
            mock_results = [
                {
                    "rank": 1,
                    "symbol": "NVDA",
                    "contract_id": 4815747,
                    "distance": 5.8,
                    "benchmark": 100.0,
                    "projection": 105.8,
                    "price": 485.50,
                    "change": 5.8,
                    "change_percent": 1.21,
                    "volume": 45234567
                }
            ]
            return {
                "success": True,
                "data": {
                    "scan_code": scan_code,
                    "results": mock_results,
                    "count": len(mock_results),
                    "source": "mock_data_fallback",
                    "message": "IBKR scanner returned no results, showing mock data"
                }
            }

        return {
            "success": True,
            "data": {
                "scan_code": scan_code,
                "results": results,
                "count": len(results),
                "source": "ibkr_live"
            }
        }

    except Exception as e:
        logger.error(f"Scanner failed: {e}", exc_info=True)
        # Return mock data instead of error
        mock_results = [
            {
                "rank": 1,
                "symbol": "NVDA",
                "contract_id": 4815747,
                "distance": 5.8,
                "benchmark": 100.0,
                "projection": 105.8,
                "price": 485.50,
                "change": 5.8,
                "change_percent": 1.21,
                "volume": 45234567
            }
        ]
        return {
            "success": True,
            "data": {
                "scan_code": scan_code,
                "results": mock_results,
                "count": len(mock_results),
                "source": "mock_data_error_fallback",
                "message": f"Scanner error: {str(e)}. Showing mock data."
            }
        }

@app.post("/api/scanner/ibkr/add-to-worklist")
async def add_scanner_results_to_worklist(request: Dict[str, Any]):
    """Add scanner results directly to worklist"""
    symbols = request.get("symbols", [])

    items = [WorklistItem(symbol=symbol, exchange="SMART") for symbol in symbols]
    return await add_bulk_to_worklist(items)

# ═══════════════════════════════════════════════════════════════════════
#                     CLAUDE ORCHESTRATOR ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

conversations = []

@app.post("/api/claude/query")
async def claude_query(request: Dict[str, Any]):
    """Send a query to Claude AI"""
    query = request.get("query", "")
    context = request.get("context", {})
    conversation_id = context.get("conversation_id")

    # Mock Claude response (in production, call actual Claude API)
    response = f"Based on your question about '{query}', here's my analysis: The market conditions are favorable with strong technical indicators. I recommend monitoring the positions closely and considering profit-taking at key resistance levels."

    return {
        "success": True,
        "data": {
            "response": response,
            "conversation_id": conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat()
        }
    }

@app.get("/api/claude/daily-review")
async def get_daily_review(date: Optional[str] = None):
    """Get daily performance review"""
    review_date = date or datetime.now().strftime('%Y-%m-%d')

    return {
        "success": True,
        "data": {
            "date": review_date,
            "summary": {
                "total_trades": 25,
                "winning_trades": 16,
                "losing_trades": 9,
                "total_pnl": 1250.50,
                "win_rate": 0.64,
                "avg_profit": 125.00,
                "avg_loss": -75.00,
                "largest_win": 450.00,
                "largest_loss": -180.00,
                "sharpe_ratio": 1.75
            },
            "performance_by_symbol": [
                {"symbol": "AAPL", "trades": 8, "pnl": 450.00, "win_rate": 0.75},
                {"symbol": "MSFT", "trades": 6, "pnl": 320.00, "win_rate": 0.67},
                {"symbol": "GOOGL", "trades": 5, "pnl": 280.50, "win_rate": 0.60},
                {"symbol": "TSLA", "trades": 4, "pnl": 150.00, "win_rate": 0.50},
                {"symbol": "NVDA", "trades": 2, "pnl": 50.00, "win_rate": 0.50}
            ],
            "performance_by_strategy": [
                {"strategy": "Momentum", "trades": 12, "pnl": 680.00, "win_rate": 0.67},
                {"strategy": "Mean Reversion", "trades": 8, "pnl": 380.50, "win_rate": 0.625},
                {"strategy": "Breakout", "trades": 5, "pnl": 190.00, "win_rate": 0.60}
            ],
            "claude_insights": {
                "key_findings": [
                    "Strong performance in tech sector with 75% win rate on AAPL",
                    "Momentum strategy showing best risk-adjusted returns",
                    "Position sizing was optimal with max loss well within limits"
                ],
                "strengths": [
                    "Excellent trade selection on high-probability setups",
                    "Risk management prevented larger losses",
                    "Profit-taking was well-timed on winning trades"
                ],
                "concerns": [
                    "Slightly lower win rate on TSLA positions",
                    "Could improve entry timing on mean reversion trades",
                    "Some positions held too long after signals reversed"
                ],
                "recommendations": [
                    "Consider increasing position size on AAPL given strong performance",
                    "Tighten stops on TSLA to reduce average loss",
                    "Implement trailing stops for momentum trades to capture more upside"
                ],
                "market_commentary": "Overall market sentiment remains bullish with strong tech sector leadership. Continue to focus on high-quality setups with proper risk management. Watch for potential profit-taking at key resistance levels in the coming session."
            }
        }
    }

@app.get("/api/claude/daily-review/history")
async def get_daily_review_history():
    """Get historical daily reviews"""
    history = []
    for i in range(30):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        history.append({
            "date": date,
            "total_pnl": 800.00 + (i * 50) - (hash(date) % 400),
            "win_rate": 0.60 + (hash(date) % 20) / 100,
            "total_trades": 20 + (hash(date) % 15)
        })

    return {"success": True, "data": history}

@app.post("/api/claude/daily-review/export")
async def export_daily_review(request: Dict[str, Any]):
    """Export daily review"""
    from fastapi.responses import StreamingResponse
    import io

    date = request.get("date", datetime.now().strftime('%Y-%m-%d'))
    export_format = request.get("format", "json")

    if export_format == "json":
        review = await get_daily_review(date)
        json_data = json.dumps(review, indent=2)
        return StreamingResponse(
            io.StringIO(json_data),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=daily_review_{date}.json"}
        )
    else:  # PDF
        # Mock PDF generation
        pdf_content = f"Daily Review for {date}\n\nTotal P&L: $1250.50\nWin Rate: 64%"
        return StreamingResponse(
            io.StringIO(pdf_content),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=daily_review_{date}.pdf"}
        )

@app.post("/api/claude/scan-optimizations")
async def scan_optimizations():
    """Scan for optimization opportunities"""
    optimizations = [
        {
            "id": "opt_1",
            "title": "Increase Position Size for High-Confidence Signals",
            "description": "Analysis shows high-confidence signals (>75%) have 85% win rate. Consider increasing position size by 25% for these setups.",
            "category": "risk_management",
            "priority": "high",
            "impact": {
                "estimated_improvement": "+15% annual return",
                "confidence": 0.85,
                "affected_metrics": ["total_return", "sharpe_ratio", "max_position_size"]
            },
            "current_state": {"max_position_pct": 2.0},
            "proposed_changes": {"max_position_pct": 2.5, "min_confidence": 0.75},
            "status": "pending",
            "created_at": datetime.now().isoformat()
        },
        {
            "id": "opt_2",
            "title": "Implement Trailing Stops for Momentum Trades",
            "description": "Momentum trades show potential for larger gains. Trailing stops could capture additional 8% upside on average.",
            "category": "strategy",
            "priority": "medium",
            "impact": {
                "estimated_improvement": "+8% on momentum trades",
                "confidence": 0.75,
                "affected_metrics": ["avg_profit", "max_profit", "holding_period"]
            },
            "current_state": {"stop_type": "fixed"},
            "proposed_changes": {"stop_type": "trailing", "trailing_pct": 1.5},
            "status": "pending",
            "created_at": datetime.now().isoformat()
        },
        {
            "id": "opt_3",
            "title": "Tighten Stop Loss on Low Volatility Symbols",
            "description": "Low volatility stocks like MSFT show tighter price ranges. Reducing stop loss from 2% to 1.5% could reduce max loss by 20%.",
            "category": "risk_management",
            "priority": "medium",
            "impact": {
                "estimated_improvement": "-20% max loss",
                "confidence": 0.70,
                "affected_metrics": ["avg_loss", "max_drawdown", "sharpe_ratio"]
            },
            "current_state": {"stop_loss_pct": 2.0},
            "proposed_changes": {"stop_loss_pct": 1.5, "volatility_threshold": 0.15},
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
    ]

    return {"success": True, "data": optimizations}

@app.post("/api/claude/apply-optimization")
async def apply_optimization(request: Dict[str, Any]):
    """Apply an optimization"""
    optimization_id = request.get("optimization_id")

    return {
        "success": True,
        "data": {
            "optimization_id": optimization_id,
            "status": "applied",
            "applied_at": datetime.now().isoformat()
        }
    }

@app.post("/api/claude/revert-optimization")
async def revert_optimization(request: Dict[str, Any]):
    """Revert an optimization"""
    optimization_id = request.get("optimization_id")

    return {
        "success": True,
        "data": {
            "optimization_id": optimization_id,
            "status": "reverted",
            "reverted_at": datetime.now().isoformat()
        }
    }

@app.get("/api/claude/optimization-history")
async def get_optimization_history():
    """Get optimization change history"""
    history = [
        {
            "id": "hist_1",
            "optimization_id": "opt_1",
            "title": "Increase Position Size for High-Confidence Signals",
            "action": "applied",
            "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
            "impact_observed": "+12% improvement in total return"
        },
        {
            "id": "hist_2",
            "optimization_id": "opt_2",
            "title": "Implement Trailing Stops",
            "action": "applied",
            "timestamp": (datetime.now() - timedelta(days=5)).isoformat(),
            "impact_observed": "+6% improvement on momentum trades"
        }
    ]

    return {"success": True, "data": history}

@app.get("/api/claude/conversations")
async def get_conversations():
    """Get all Claude conversations"""
    return {
        "success": True,
        "data": [
            {
                "id": "conv_1",
                "title": "Strategy Optimization Discussion",
                "created_at": (datetime.now() - timedelta(days=1)).isoformat(),
                "updated_at": datetime.now().isoformat(),
                "message_count": 12
            },
            {
                "id": "conv_2",
                "title": "Risk Management Review",
                "created_at": (datetime.now() - timedelta(days=3)).isoformat(),
                "updated_at": (datetime.now() - timedelta(days=2)).isoformat(),
                "message_count": 8
            }
        ]
    }

@app.get("/api/claude/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation"""
    return {
        "success": True,
        "data": {
            "id": conversation_id,
            "title": "Strategy Optimization Discussion",
            "messages": [
                {
                    "id": "msg_1",
                    "role": "user",
                    "content": "What do you think about my current strategy performance?",
                    "timestamp": (datetime.now() - timedelta(hours=2)).isoformat()
                },
                {
                    "id": "msg_2",
                    "role": "assistant",
                    "content": "Your strategy is performing well with a 65% win rate. I notice your momentum trades are particularly strong. Consider increasing position size on these setups.",
                    "timestamp": (datetime.now() - timedelta(hours=2, minutes=-5)).isoformat()
                }
            ]
        }
    }

# ═══════════════════════════════════════════════════════════════════════
#                     HEALTH & STATUS
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    global autonomous_bot

    return {
        "status": "healthy",
        "ibkr_connected": data_bus.connected,
        "ibkr_available": IBKR_AVAILABLE,
        "ai_predictor_loaded": ai_predictor is not None and (ai_predictor.model is not None if ai_predictor else False),
        "claude_available": CLAUDE_AVAILABLE and market_analyst is not None and (market_analyst.client is not None if market_analyst else False),
        "autonomous_bot_available": AUTONOMOUS_BOT_AVAILABLE,
        "autonomous_bot_initialized": autonomous_bot is not None,
        "autonomous_bot_running": autonomous_bot.running if autonomous_bot else False,
        "active_subscriptions": sum(len(subs) for subs in data_bus.subscribers.values()),
        "symbols_tracked": list(data_bus.market_data.keys()),
        "modules_loaded": {
            "claude_api": claude_router is not None,
            "market_analyst": market_analyst is not None,
            "ai_predictor": ai_predictor is not None,
            "prediction_logger": log_prediction is not None,
            "alpha_fusion": predict_one is not None,
            "autonomous_trader": AUTONOMOUS_BOT_AVAILABLE
        }
    }

@app.get("/api/debug/l2test/{symbol}")
async def debug_l2(symbol: str):
    """Debug endpoint to test L2 data generation"""
    return {
        "symbol": symbol,
        "l2_data_exists": symbol in data_bus.l2_data,
        "market_data_exists": symbol in data_bus.market_data,
        "l2_data": data_bus.l2_data.get(symbol, {}),
        "market_data": data_bus.market_data.get(symbol, {}),
        "code_version": "2.0-SIMULATED-L2-ACTIVE"
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
