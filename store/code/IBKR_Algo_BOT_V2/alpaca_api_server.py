"""
Alpaca API Server - Port 9102
Separate from main dashboard
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from alpaca_integration import get_alpaca_connector
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="Alpaca Trading API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OrderRequest(BaseModel):
    symbol: str
    action: str
    quantity: int
    order_type: str = "MKT"
    limit_price: Optional[float] = None

@app.get("/api/alpaca/status")
async def get_status():
    connector = get_alpaca_connector()
    return {
        "connected": connector.is_connected(),
        "broker": "Alpaca",
        "paper_trading": True
    }

@app.get("/api/alpaca/account")
async def get_account():
    connector = get_alpaca_connector()
    return connector.get_account()

@app.get("/api/alpaca/positions")
async def get_positions():
    connector = get_alpaca_connector()
    return connector.get_positions()

@app.get("/api/alpaca/orders")
async def get_orders(status: str = "all"):
    connector = get_alpaca_connector()
    return connector.get_orders(status)

@app.post("/api/alpaca/place-order")
async def place_order(order: OrderRequest):
    connector = get_alpaca_connector()
    
    if order.order_type.upper() == "MKT":
        result = connector.place_market_order(
            symbol=order.symbol,
            quantity=order.quantity,
            side=order.action
        )
    else:
        result = connector.place_limit_order(
            symbol=order.symbol,
            quantity=order.quantity,
            side=order.action,
            limit_price=order.limit_price
        )
    
    return result

@app.get("/api/alpaca/quote/{symbol}")
async def get_quote(symbol: str):
    connector = get_alpaca_connector()
    return connector.get_quote(symbol)

@app.delete("/api/alpaca/positions/{symbol}")
async def close_position(symbol: str):
    connector = get_alpaca_connector()
    success = connector.close_position(symbol)
    return {"success": success}

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ALPACA API SERVER")
    print("Running on: http://localhost:9102")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=9102)
