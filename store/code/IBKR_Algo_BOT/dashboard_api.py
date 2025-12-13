import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
"""IBKR Trading Bot Dashboard API - Complete with Backtesting"""
import os
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

try:
    from ai.ai_predictor import get_predictor
    AI_AVAILABLE = True
    print("? AI Predictor loaded")
except Exception as e:
    AI_AVAILABLE = False
    print(f"?? AI Predictor not available: {e}")

try:
    from ai.auto_trader import create_auto_trader, TradingConfig
    AUTO_TRADER_AVAILABLE = True
    print("? Auto-Trader loaded")
except Exception as e:
    AUTO_TRADER_AVAILABLE = False
    print(f"?? Auto-Trader not available: {e}")

try:
    from ai.analytics_engine import create_analytics_engine
    ANALYTICS_AVAILABLE = True
    print("? Analytics Engine loaded")
except Exception as e:
    ANALYTICS_AVAILABLE = False
    print(f"?? Analytics not available: {e}")

try:
    from ai.backtester import create_backtester
    BACKTEST_AVAILABLE = True
    print("? Backtester loaded")
except Exception as e:
    BACKTEST_AVAILABLE = False
    print(f"?? Backtester not available: {e}")

ADAPTER_OK = False
ib_adapter = None
startup_error: Optional[str] = None

try:
    from bridge.ib_adapter import IBConfig, IBAdapter
    ADAPTER_OK = True
    print("? IB Adapter import OK")
except Exception as e:
    print(f"?? IB Adapter import failed: {e}")

app = FastAPI(title="IBKR Trading Bot API", version="4.0.0")

# === AI Router Integration ===
try:
    from server.ai_router import router as ai_router
    app.include_router(ai_router)
    print('✅ AI Router mounted successfully')
except ImportError as e:
    print(f'⚠️ AI router error: {e}')

@app.on_event("startup")
def _on_startup():
    global ib_adapter, startup_error
    print("?? Starting IBKR Dashboard API...")
    try:
        cfg = IBConfig()
        ib_adapter = IBAdapter(cfg)
        print("?? Connecting to IBKR...")
        ib_adapter.connect()
    except Exception as e:
        startup_error = str(e)
        print(f"?? Startup error: {e}")
    print("? Startup complete")

class OrderIn(BaseModel):
    symbol: str
    side: str
    qty: int
    orderType: Optional[str] = "LMT"
    limitPrice: Optional[float] = None
    stopPrice: Optional[float] = None
    trailAmount: Optional[float] = None
    outsideRth: bool = False  # Extended hours trading flag

def require_api_key(x_api_key: str = Header(None)):
    expected = os.getenv("LOCAL_API_KEY")
    if not expected:
        raise HTTPException(500, "LOCAL_API_KEY not configured")
    if x_api_key != expected:
        raise HTTPException(401, "Invalid or missing X-API-Key")
    return True

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/status")
def status():
    out = {
        "timestamp": datetime.utcnow().isoformat(),
        "ib_connection": False,
        "ai_connection": AI_AVAILABLE,
        "auto_trader": AUTO_TRADER_AVAILABLE,
        "analytics": ANALYTICS_AVAILABLE,
        "backtest": BACKTEST_AVAILABLE,
        "current_client_id": None,
        "state": "UNKNOWN",
        "api_key_configured": bool(os.getenv("LOCAL_API_KEY")),
        "mode": "simulation"
    }
    if ib_adapter:
        try:
            status_data = ib_adapter.get_status()
            out.update({
                "ib_connection": ib_adapter.is_connected(),
                "current_client_id": status_data.get("current_client_id"),
                "state": status_data.get("state"),
                "host": status_data.get("host"),
                "port": status_data.get("port"),
                "mode": "live" if ib_adapter.is_connected() else "simulation"
            })
        except Exception as e:
            out["error"] = str(e)
    else:
        out["state"] = "NOT_INITIALIZED"
        out["error"] = startup_error or "Adapter not created"
    return out

@app.get("/api/positions")
def get_positions():
    if not ib_adapter:
        raise HTTPException(503, "Adapter not available")
    return {"positions": ib_adapter.get_positions(), "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/account")
def get_account():
    if not ib_adapter:
        raise HTTPException(503, "Adapter not available")
    return {"account": ib_adapter.get_account_summary(), "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/account/live")
def get_live_account():
    if not ib_adapter or not ib_adapter.is_connected():
        raise HTTPException(503, "IBKR not connected")
    try:
        account = ib_adapter.get_account_summary()
        positions = ib_adapter.get_positions()
        net_liquidation = 0
        total_cash = 0
        for tag, value in account.items():
            if tag == "NetLiquidation":
                net_liquidation = float(value.get("value", 0))
            if tag == "TotalCashValue":
                total_cash = float(value.get("value", 0))
        position_value = sum(p.get('position', 0) * p.get('avgCost', 0) for p in positions)
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "net_liquidation": net_liquidation,
            "total_cash": total_cash,
            "position_value": position_value,
            "positions": positions,
            "account_details": account
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get account: {e}")

@app.post("/api/order/preview")
def order_preview(body: OrderIn, _=Depends(require_api_key)):
    if not ib_adapter:
        raise HTTPException(503, "Adapter not available")
    return {"status": "preview_ready", "order": body.dict(), "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/order/place")
def order_place(body: OrderIn, _=Depends(require_api_key)):
    if not ib_adapter:
        raise HTTPException(503, "Adapter not available")
    if not ib_adapter.is_connected():
        raise HTTPException(503, "IBKR not connected")
    try:
        action = "BUY" if body.side.upper() == "BUY" else "SELL"
        
        # Route to appropriate order method
        if body.orderType == "MKT":
            result = ib_adapter.place_market_order_sync(body.symbol, body.qty, action)
        elif body.orderType == "LMT":
            if not body.limitPrice:
                raise HTTPException(400, "Limit price required")
            result = ib_adapter.place_limit_order_sync(body.symbol, body.qty, body.limitPrice, action)
        elif body.orderType == "STP":
            if not body.stopPrice:
                raise HTTPException(400, "Stop price required")
            result = ib_adapter.place_stop_order_sync(body.symbol, body.qty, body.stopPrice, action)
        elif body.orderType == "STP LMT":
            if not body.stopPrice or not body.limitPrice:
                raise HTTPException(400, "Stop and limit prices required")
            result = ib_adapter.place_stop_limit_order_sync(body.symbol, body.qty, body.stopPrice, body.limitPrice, action)
        elif body.orderType == "TRAIL":
            if not body.trailAmount:
                raise HTTPException(400, "Trail amount required")
            result = ib_adapter.place_trailing_stop_sync(body.symbol, body.qty, body.trailAmount, action)
        else:
            raise HTTPException(400, f"Unsupported order type: {body.orderType}")
            
        return result
    except Exception as e:
        raise HTTPException(500, f"Order failed: {e}")




@app.post("/api/order/cancel")
def cancel_order(body: dict, _=Depends(require_api_key)):
    """Cancel an order by order ID"""
    if not ib_adapter:
        return {"success": False, "error": "IB adapter not initialized"}
    
    try:
        order_id = body.get("orderId")
        if not order_id:
            return {"success": False, "error": "Missing orderId"}
        
        # Try to cancel the order
        result = ib_adapter.cancel_order(int(order_id))
        
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
@app.get("/api/price/{symbol}")
def get_price(symbol: str):
    from ib_insync import Stock
    from ib_insync import Stock
    """Get current price for a symbol"""
    try:
        symbol = symbol.upper()
        contract = Stock(symbol, 'SMART', 'USD')
        ticker = ib_adapter.ib.reqMktData(contract, '', False, False)
        ib_adapter.ib.sleep(1.5)
        
        # Get price - try multiple sources
        price = ticker.last
        if not price or price <= 0:
            price = ticker.marketPrice()
        if not price or price <= 0:
            price = (ticker.bid + ticker.ask) / 2 if ticker.bid and ticker.ask else None
        
        if price and price > 0:
            return {
                "symbol": symbol,
                "price": round(price, 4),
                "bid": round(ticker.bid, 4) if ticker.bid else round(price - 0.01, 4),
                "ask": round(ticker.ask, 4) if ticker.ask else round(price + 0.01, 4),
                "last": round(price, 4)
            }
        
        return {"error": "No price data available", "symbol": symbol, "price": 0}
    except Exception as e:
        return {"error": str(e), "symbol": symbol, "price": 0}
def get_price(symbol: str):
    """Get current price for a symbol"""
    if not ib_adapter:
        return {"error": "IB adapter not initialized", "price": 0}
    
    try:
        symbol = symbol.upper()
        # Try to get price from IB
        contract = Stock(symbol, 'SMART', 'USD')
        print(f"DEBUG: Requesting contract for {symbol}")
        ticker = ib_adapter.ib.reqMktData(contract, '', False, False)
        print(f"DEBUG: Ticker object: {ticker}")
        print(f"DEBUG: Contract details: {ticker.contract}")
        ib_adapter.ib.sleep(1)  # Wait for data
        
        price = ticker.marketPrice()
        print(f"DEBUG {symbol}: bid={ticker.bid}, ask={ticker.ask}, last={ticker.last}, close={ticker.close}, marketPrice={price}")
        print(f"DEBUG: Full ticker dump: {ticker}")
        print(f"DEBUG {symbol}: bid={ticker.bid}, ask={ticker.ask}, last={ticker.last}, marketPrice={price}")
        if price and price > 0:
            print(f"PRICE API returning for {symbol}: price={price}, bid={ticker.bid}, ask={ticker.ask}")
            return {
                "symbol": symbol,
                "price": price,
                "bid": ticker.bid if ticker.bid else price - 0.01,
                "ask": ticker.ask if ticker.ask else price + 0.01,
                "last": price
            }
        else:
            # Fallback to a simulated price
            import random
            base_price = 100.0
            return {
                "symbol": symbol,
                "price": base_price,
                "bid": base_price - 0.01,
                "ask": base_price + 0.01,
                "last": base_price
            }
    except Exception as e:
        # Return simulated price on error
        import random
        base_price = 100.0 + random.uniform(-5, 5)
        return {
            "symbol": symbol,
            "price": base_price,
            "bid": base_price - 0.01,
            "ask": base_price + 0.01,
            "last": base_price,
            "note": f"Simulated price (error: {str(e)})"
        }
@app.get("/api/orders")
def get_orders():
    if not ib_adapter:
        raise HTTPException(503, "Adapter not available")
    return {"orders": ib_adapter.get_open_orders(), "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/ai/predict")
async def ai_predict(data: dict):
    if not AI_AVAILABLE:
        raise HTTPException(503, "AI predictor not available")
    symbol = data.get("symbol")
    if not symbol:
        raise HTTPException(400, "Symbol required")
    try:
        predictor = get_predictor()
        return predictor.predict(symbol)
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "prob_up": 0.5, "confidence": 0.0}

@app.post("/api/model/train")
def train_model(data: dict, _=Depends(require_api_key)):
    if not AI_AVAILABLE:
        raise HTTPException(503, "AI not available")
    symbol = data.get("symbol")
    period = data.get("period", "2y")
    if not symbol:
        raise HTTPException(400, "Symbol required")
    try:
        predictor = get_predictor()
        accuracy = predictor.train(symbol, period)
        return {"success": True, "symbol": symbol, "period": period, "accuracy": accuracy, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/backtest/run")
def run_backtest(data: dict, _=Depends(require_api_key)):
    if not AI_AVAILABLE or not BACKTEST_AVAILABLE:
        raise HTTPException(503, "AI or Backtester not available")
    symbol = data.get("symbol")
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    initial_capital = data.get("initial_capital", 10000.0)
    if not all([symbol, start_date, end_date]):
        raise HTTPException(400, "symbol, start_date, end_date required")
    try:
        predictor = get_predictor()
        backtester = create_backtester(initial_capital)
        results = backtester.backtest(symbol, predictor, start_date, end_date)
        return results
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/auto-trade/config")
def get_auto_config():
    if not AUTO_TRADER_AVAILABLE:
        raise HTTPException(503, "Auto-trader not available")
    config = TradingConfig()
    return {"enabled": config.enabled, "min_confidence": config.min_confidence, "min_prob_up": config.min_prob_up, "max_position_size": config.max_position_size, "max_daily_trades": config.max_daily_trades, "max_daily_loss": config.max_daily_loss, "stop_loss_pct": config.stop_loss_pct}

@app.post("/api/auto-trade/evaluate")
def evaluate_auto_trade(data: dict, _=Depends(require_api_key)):
    if not AUTO_TRADER_AVAILABLE or not AI_AVAILABLE:
        raise HTTPException(503, "Auto-trader or AI not available")
    symbol = data.get("symbol")
    if not symbol:
        raise HTTPException(400, "Symbol required")
    if not ib_adapter or not ib_adapter.is_connected():
        raise HTTPException(503, "IBKR not connected")
    predictor = get_predictor()
    trader = create_auto_trader(predictor, ib_adapter)
    return trader.should_trade(symbol)

@app.post("/api/auto-trade/execute")
def execute_auto_trade(data: dict, _=Depends(require_api_key)):
    if not AUTO_TRADER_AVAILABLE or not AI_AVAILABLE:
        raise HTTPException(503, "Auto-trader or AI not available")
    symbol = data.get("symbol")
    if not symbol:
        raise HTTPException(400, "Symbol required")
    if not ib_adapter or not ib_adapter.is_connected():
        raise HTTPException(503, "IBKR not connected")
    predictor = get_predictor()
    trader = create_auto_trader(predictor, ib_adapter)
    return trader.execute_trade(symbol)

@app.get("/api/auto-trade/logs")
def get_trade_logs():
    if not AUTO_TRADER_AVAILABLE:
        raise HTTPException(503, "Auto-trader not available")
    from ai.auto_trader import TradeLogger
    logger = TradeLogger()
    return {"trades": logger.get_all_trades(), "today": logger.get_today_trades()}

@app.get("/api/analytics/summary")
def get_analytics_summary():
    if not ANALYTICS_AVAILABLE:
        raise HTTPException(503, "Analytics not available")
    analytics = create_analytics_engine()
    return analytics.get_performance_summary()

@app.get("/api/analytics/daily")
def get_daily_analytics(days: int = 30):
    if not ANALYTICS_AVAILABLE:
        raise HTTPException(503, "Analytics not available")
    analytics = create_analytics_engine()
    return {"daily_pnl": analytics.get_daily_pnl(days)}

@app.get("/api/analytics/symbols")
def get_symbol_analytics():
    if not ANALYTICS_AVAILABLE:
        raise HTTPException(503, "Analytics not available")
    analytics = create_analytics_engine()
    return {"symbols": analytics.get_symbol_performance()}

@app.get("/api/analytics/recent")
def get_recent_analytics(limit: int = 10):
    if not ANALYTICS_AVAILABLE:
        raise HTTPException(503, "Analytics not available")
    analytics = create_analytics_engine()
    return {"recent_trades": analytics.get_recent_activity(limit)}

@app.post("/api/tws/connect")
def tws_connect():
    if not ib_adapter:
        raise HTTPException(503, "Adapter not available")
    try:
        if ib_adapter.is_connected():
            return {"success": True, "message": "Already connected", "status": ib_adapter.get_status()}
        ib_adapter.connect()
        return {"success": ib_adapter.is_connected(), "status": ib_adapter.get_status()}
    except Exception as e:
        raise HTTPException(500, f"Connection failed: {e}")

@app.get("/api/tws/ping")
def tws_ping():
    if not ib_adapter:
        return {"connected": False, "state": "NOT_INITIALIZED"}
    return {"connected": ib_adapter.is_connected(), "state": ib_adapter.connection_state, "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/debug/ibkr")
def debug_ibkr():
    import socket
    if not ib_adapter:
        return {"error": "Adapter not initialized", "adapter_available": False}
    host = ib_adapter.config.host
    port = ib_adapter.config.port
    socket_test = {"success": False, "error": None}
    try:
        with socket.create_connection((host, port), timeout=3):
            socket_test["success"] = True
    except Exception as e:
        socket_test["error"] = str(e)
    return {"timestamp": datetime.utcnow().isoformat(), "socket_test": {"host": host, "port": port, "success": socket_test["success"], "error": socket_test.get("error")}, "adapter_status": ib_adapter.get_status(), "adapter_available": True}

@app.get("/ui/{file_path:path}")
async def serve_ui(file_path: str):
    from pathlib import Path
    import os
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent.parent
    ui_path = project_root / "ui" / file_path
    
    if ui_path.exists() and ui_path.is_file():
        return FileResponse(ui_path)
    raise HTTPException(404, f"File not found: {file_path}")

@app.get("/")
def root():
    return {"message": "IBKR Trading Bot API", "version": "4.0.0", "status": "/api/status"}














@app.get("/api/timesales/{symbol}")
def get_timesales(symbol: str, limit: int = 50, _=Depends(require_api_key)):
    """Get Time & Sales for a symbol"""
    if not ib_adapter or not ib_adapter.is_connected():
        raise HTTPException(503, "Not connected")
    try:
        return ib_adapter.get_recent_trades(symbol, limit)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/level2/{symbol}")
def get_level2(symbol: str, _=Depends(require_api_key)):
    """Get Level 2 market depth data"""
    if not ib_adapter:
        raise HTTPException(503, "Adapter not available")
    if not ib_adapter.is_connected():
        raise HTTPException(503, "IBKR not connected")
    try:
        depth = ib_adapter.get_market_depth_sync(symbol)
        return depth
    except Exception as e:
        raise HTTPException(500, f"Level 2 fetch failed: {e}")

# === AI router mount ===
try:
    from server.ai_router import router as ai_router
    app.include_router(ai_router)
    print("? AI router mounted successfully")
except Exception as e:
    print("? AI router failed:", e)
# === end AI router mount ===

if __name__ == "__main__":
    import uvicorn
    print("?? Starting IBKR Trading Bot Dashboard API on http://127.0.0.1:9101")
    print("?? UI Available at: http://127.0.0.1:9101/ui/trading.html")
    uvicorn.run(app, host="127.0.0.1", port=9101, log_level="info")

















# === AI router mount (added by collab patch) ===
try:
    from server.ai_router import router as ai_router
    app.include_router(ai_router)
except Exception as _e:
    print("? AI router not mounted:", _e)
else:
    print("? AI router mounted successfully")
# === end AI router mount ===





