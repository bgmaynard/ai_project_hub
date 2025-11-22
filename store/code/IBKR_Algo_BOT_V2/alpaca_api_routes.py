"""
Alpaca API Routes for Dashboard
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from alpaca_integration import get_alpaca_connector

router = APIRouter(prefix="/api/alpaca", tags=["alpaca"])

class OrderRequest(BaseModel):
    symbol: str
    action: str  # BUY or SELL
    quantity: int
    order_type: str = "MKT"  # MKT or LMT
    limit_price: Optional[float] = None

@router.get("/status")
async def get_status():
    """Get Alpaca connection status"""
    connector = get_alpaca_connector()
    
    return {
        "connected": connector.is_connected(),
        "broker": "Alpaca",
        "paper_trading": True
    }

@router.get("/account")
async def get_account():
    """Get account information"""
    connector = get_alpaca_connector()
    
    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")
    
    return connector.get_account()

@router.get("/positions")
async def get_positions():
    """Get all positions"""
    connector = get_alpaca_connector()
    
    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")
    
    return connector.get_positions()

@router.get("/orders")
async def get_orders(status: str = "all"):
    """Get orders"""
    connector = get_alpaca_connector()
    
    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")
    
    return connector.get_orders(status)

@router.post("/place-order")
async def place_order(order: OrderRequest):
    """Place an order"""
    connector = get_alpaca_connector()
    
    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")
    
    try:
        if order.order_type.upper() == "MKT":
            result = connector.place_market_order(
                symbol=order.symbol,
                quantity=order.quantity,
                side=order.action
            )
        elif order.order_type.upper() == "LMT":
            if order.limit_price is None:
                raise HTTPException(status_code=400, detail="Limit price required for limit orders")
            
            result = connector.place_limit_order(
                symbol=order.symbol,
                quantity=order.quantity,
                side=order.action,
                limit_price=order.limit_price
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid order type")
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an order"""
    connector = get_alpaca_connector()
    
    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")
    
    success = connector.cancel_order(order_id)
    
    return {"success": success}

@router.get("/quote/{symbol}")
async def get_quote(symbol: str):
    """Get quote for symbol"""
    connector = get_alpaca_connector()
    
    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")
    
    return connector.get_quote(symbol)

@router.delete("/positions/{symbol}")
async def close_position(symbol: str):
    """Close a position"""
    connector = get_alpaca_connector()
    
    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")
    
    success = connector.close_position(symbol)
    
    return {"success": success}

@router.delete("/positions")
async def close_all_positions():
    """Close all positions"""
    connector = get_alpaca_connector()
    
    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to Alpaca")
    
    success = connector.close_all_positions()
    
    return {"success": success}
