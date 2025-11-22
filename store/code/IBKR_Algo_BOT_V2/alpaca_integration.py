"""
Alpaca Integration for AI Trading Dashboard
Drop-in replacement for IBKR connectivity
"""
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

load_dotenv()

class AlpacaConnector:
    """Alpaca connection matching IBKR interface"""
    
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        # Trading client
        self.trading_client = TradingClient(
            self.api_key,
            self.secret_key,
            paper=True
        )
        
        # Data client
        self.data_client = StockHistoricalDataClient(
            self.api_key,
            self.secret_key
        )
        
        self.connected = False
        self._connect()
    
    def _connect(self):
        """Connect to Alpaca"""
        try:
            account = self.trading_client.get_account()
            self.connected = True
            print(f"[OK] Connected to Alpaca: {account.account_number}")
            return True
        except Exception as e:
            print(f"[FAIL] Alpaca connection failed: {e}")
            self.connected = False
            return False
    
    def is_connected(self) -> bool:
        """Check connection status"""
        return self.connected
    
    def get_account(self) -> Dict:
        """Get account information"""
        account = self.trading_client.get_account()
        
        return {
            "account_id": account.account_number,
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "equity": float(account.equity),
            "last_equity": float(account.last_equity),
            "currency": account.currency,
            "status": str(account.status),
            "pattern_day_trader": account.pattern_day_trader
        }
    
    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        positions = self.trading_client.get_all_positions()
        
        result = []
        for pos in positions:
            result.append({
                "symbol": pos.symbol,
                "quantity": float(pos.qty),
                "avg_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "cost_basis": float(pos.cost_basis),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "side": pos.side
            })
        
        return result
    
    def get_orders(self, status: str = "all") -> List[Dict]:
        """Get orders"""
        from alpaca.trading.requests import GetOrdersRequest
        
        request = GetOrdersRequest(status=status)
        orders = self.trading_client.get_orders(filter=request)
        
        result = []
        for order in orders:
            result.append({
                "order_id": str(order.id),
                "symbol": order.symbol,
                "side": str(order.side),
                "quantity": float(order.qty),
                "order_type": str(order.type),
                "status": str(order.status),
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else 0,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "submitted_at": str(order.submitted_at),
                "filled_at": str(order.filled_at) if order.filled_at else None
            })
        
        return result
    
    def place_market_order(self, symbol: str, quantity: int, side: str) -> Dict:
        """Place market order"""
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=order_side,
            time_in_force=TimeInForce.DAY
        )
        
        order = self.trading_client.submit_order(order_request)
        
        return {
            "order_id": str(order.id),
            "symbol": order.symbol,
            "status": str(order.status),
            "side": str(order.side),
            "quantity": float(order.qty)
        }
    
    def place_limit_order(self, symbol: str, quantity: int, side: str, limit_price: float) -> Dict:
        """Place limit order"""
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        
        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price
        )
        
        order = self.trading_client.submit_order(order_request)
        
        return {
            "order_id": str(order.id),
            "symbol": order.symbol,
            "status": str(order.status),
            "side": str(order.side),
            "quantity": float(order.qty),
            "limit_price": float(order.limit_price)
        }
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            print(f"Cancel order failed: {e}")
            return False
    
    def get_quote(self, symbol: str) -> Dict:
        """Get latest quote for symbol"""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)
            
            quote = quotes[symbol]
            
            return {
                "symbol": symbol,
                "bid": float(quote.bid_price),
                "ask": float(quote.ask_price),
                "last": (float(quote.bid_price) + float(quote.ask_price)) / 2,
                "bid_size": int(quote.bid_size),
                "ask_size": int(quote.ask_size)
            }
        except Exception as e:
            print(f"Get quote failed for {symbol}: {e}")
            return {
                "symbol": symbol,
                "bid": 0,
                "ask": 0,
                "last": 0,
                "bid_size": 0,
                "ask_size": 0
            }
    
    def close_position(self, symbol: str) -> bool:
        """Close a position"""
        try:
            self.trading_client.close_position(symbol)
            return True
        except Exception as e:
            print(f"Close position failed: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """Close all positions"""
        try:
            self.trading_client.close_all_positions(cancel_orders=True)
            return True
        except Exception as e:
            print(f"Close all positions failed: {e}")
            return False

# Global instance
alpaca_connector = None

def get_alpaca_connector():
    """Get or create Alpaca connector instance"""
    global alpaca_connector
    
    if alpaca_connector is None:
        alpaca_connector = AlpacaConnector()
    
    return alpaca_connector
