"""IBKR Adapter - With TIF fix"""
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv
import nest_asyncio

nest_asyncio.apply()

project_root = Path(__file__).parent.parent.parent.parent
env_file = project_root / '.env'
load_dotenv(env_file)

from ib_insync import IB, Stock, MarketOrder, LimitOrder

class IBConfig:
    def __init__(self):
        self.host = os.getenv("TWS_HOST", "127.0.0.1")
        self.port = int(os.getenv("TWS_PORT", "7497"))
        self.base_client_id = int(os.getenv("TWS_CLIENT_ID", "1"))
        print(f"IBConfig: {self.host}:{self.port}, clientId: {self.base_client_id}")

class IBConnectionState:
    DISCONNECTED = "DISCONNECTED"
    CONNECTED = "CONNECTED"
    FAILED = "FAILED"

class IBAdapter:
    def __init__(self, config: IBConfig):
        self.config = config
        self.ib = IB()
        self.current_client_id = config.base_client_id
        self.connection_state = IBConnectionState.DISCONNECTED
        self.last_error = None
        
    def connect(self) -> bool:
        print(f"Connecting to TWS on {self.config.port}...")
        try:
            self.ib.connect(self.config.host, self.config.port, clientId=self.current_client_id, timeout=30)
            self.connection_state = IBConnectionState.CONNECTED
            print("SUCCESS! Connected!")
            print(f"Server version: {self.ib.client.serverVersion()}")
            return True
        except Exception as e:
            self.last_error = str(e)
            self.connection_state = IBConnectionState.FAILED
            print(f"Failed: {e}")
            return False
    
    def is_connected(self) -> bool:
        return self.connection_state == IBConnectionState.CONNECTED and self.ib and self.ib.isConnected()
    
    def get_positions(self) -> List[Dict[str, Any]]:
        if not self.is_connected():
            return []
        try:
            positions = self.ib.positions()
            return [{"symbol": p.contract.symbol, "position": p.position, "avgCost": p.avgCost} for p in positions]
        except:
            return []
    
    def get_account_summary(self) -> Dict[str, Any]:
        if not self.is_connected():
            return {}
        try:
            values = self.ib.accountSummary()
            return {av.tag: {"value": av.value, "currency": av.currency} for av in values}
        except:
            return {}
    
        """Place a limit order synchronously"""
        """Place a limit order synchronously"""
        if not self.is_connected():
            raise Exception("Not connected to IBKR")
        
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        order = Order()
        order.action = action
        order.orderType = "LMT"  # CRITICAL: Set order type to LIMIT
        order.totalQuantity = quantity
        order.lmtPrice = limit_price
        order.outsideRth = outside_rth  # Allow extended hours trading
        order.tif = "GTC" if outside_rth else "DAY"  # Good-til-canceled for extended hours
        
        order_id = self.next_order_id
        self.next_order_id += 1
        
        self.logger.info(f"Placing LIMIT order: {action} {quantity} {symbol} @ ${limit_price} (outsideRth={outside_rth})")
        self.ib_client.placeOrder(order_id, contract, order)
        
        time.sleep(2)
        
        return {
            "success": True,
            "orderId": order_id,
            "symbol": symbol,
            "quantity": quantity,
            "action": action,
            "limitPrice": limit_price,
            "orderType": "LMT",
            "outsideRth": outside_rth,
            "status": "Submitted"
        }
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            order = LimitOrder(action, qty, price)
            order.tif = "GTC"
            
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(0.5)
            
            return {
                "success": True,
                "orderId": trade.order.orderId,
                "symbol": symbol,
                "quantity": qty,
                "action": action,
                "limitPrice": price,
                "status": trade.orderStatus.status if trade.orderStatus else "Submitted"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def place_market_order_sync(self, symbol: str, quantity: int, action: str = "BUY", outside_rth: bool = False):
        """Place a market order synchronously"""
        if not self.is_connected():
            raise Exception("Not connected to IBKR")
        
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        order = Order()
        order.action = action
        order.orderType = "MKT"  # CRITICAL: Set order type to MARKET
        order.totalQuantity = quantity
        order.outsideRth = outside_rth
        order.tif = "GTC" if outside_rth else "DAY"
        
        order_id = self.next_order_id
        self.next_order_id += 1
        
        self.logger.info(f"Placing MARKET order: {action} {quantity} {symbol} (outsideRth={outside_rth})")
        self.ib_client.placeOrder(order_id, contract, order)
        
        time.sleep(2)
        
        return {
            "success": True,
            "orderId": order_id,
            "symbol": symbol,
            "quantity": quantity,
            "action": action,
            "orderType": "MKT",
            "outsideRth": outside_rth,
            "status": "Submitted"
        }
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            order = MarketOrder(action, qty)
            order.tif = "DAY"
            
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(0.5)
            
            return {
                "success": True,
                "orderId": trade.order.orderId,
                "symbol": symbol,
                "quantity": qty,
                "action": action,
                "status": trade.orderStatus.status if trade.orderStatus else "Submitted"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def place_limit_order_sync(self, symbol: str, quantity: int, limit_price: float, action: str = "BUY", outside_rth: bool = False):
        from ib_insync import Stock, LimitOrder
        contract = Stock(symbol, 'SMART', 'USD')
        order = LimitOrder(action, quantity, limit_price)
        order.outsideRth = outside_rth
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(0.5)
        return {'success': True, 'orderId': trade.order.orderId, 'status': trade.orderStatus.status, 'trade': trade}

    def place_stop_order_sync(self, symbol: str, quantity: int, stop_price: float, action: str = "BUY", outside_rth: bool = False):
        from ib_insync import Stock, StopOrder
        contract = Stock(symbol, 'SMART', 'USD')
        order = StopOrder(action, quantity, stop_price)
        order.outsideRth = outside_rth
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(0.5)
        return {'success': True, 'orderId': trade.order.orderId, 'status': trade.orderStatus.status, 'trade': trade}
    def get_open_orders(self) -> List[Dict[str, Any]]:
        if not self.is_connected():
            return []
        try:
            trades = self.ib.openTrades()
            return [{
                "orderId": t.order.orderId,
                "symbol": t.contract.symbol,
                "action": t.order.action,
                "quantity": t.order.totalQuantity,
                "status": t.orderStatus.status
            } for t in trades]
        except:
            return []
    
    def cancel_order(self, order_id: int):
        """Cancel an order by ID"""
        if not self.is_connected():
            raise Exception("Not connected to IBKR")
        
        # Get all open trades
        trades = self.ib.trades()
        
        # Find the trade with matching order ID
        for trade in trades:
            if trade.order.orderId == order_id:
                self.ib.cancelOrder(trade.order)
                self.ib.sleep(1)
                return {'success': True, 'orderId': order_id, 'status': 'Cancelled'}
        
        # If not found in trades, try direct cancel
        from ib_insync import Order
        order = Order()
        order.orderId = order_id
        self.ib.cancelOrder(order)
        self.ib.sleep(1)
        return {'success': True, 'orderId': order_id, 'status': 'Cancelled'}
    def get_recent_trades(self, symbol: str, limit: int = 50):
        """Get recent trades (Time & Sales)"""
        if not self.is_connected():
            raise Exception("Not connected")
        
        from ib_insync import Stock
        from datetime import datetime, timedelta
        
        contract = Stock(symbol, 'SMART', 'USD')
        
        # Request tick-by-tick trade data
        ticker = self.ib.reqTickByTickData(contract, 'Last')
        self.ib.sleep(2)
        
        trades = []
        if hasattr(ticker, 'tickByTicks'):
            for tick in ticker.tickByTicks[-limit:]:
                trades.append({
                    'time': tick.time.strftime('%H:%M:%S'),
                    'price': float(tick.price),
                    'size': int(tick.size)
                })
        
        self.ib.cancelTickByTickData(contract, 'Last')
        
        return {'symbol': symbol, 'trades': trades}
    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self.connection_state,
            "host": self.config.host,
            "port": self.config.port,
            "current_client_id": self.current_client_id,
            "last_error": self.last_error,
            "ib_connected": self.ib.isConnected() if self.ib else False,
            "timestamp": datetime.utcnow().isoformat()
        }

    def place_order(self, symbol: str, action: str, quantity: int, order_type: str, 
                    limit_price: float = None, stop_price: float = None, 
                    trail_amount: float = None, outside_rth: bool = False):
        from ib_insync import Stock, MarketOrder, LimitOrder, StopOrder
        contract = Stock(symbol, 'SMART', 'USD')
        if order_type == 'MKT':
            order = MarketOrder(action, quantity)
        elif order_type == 'LMT':
            order = LimitOrder(action, quantity, limit_price)
        elif order_type == 'STP':
            order = StopOrder(action, quantity, stop_price)
        else:
            raise Exception(f"Unsupported: {order_type}")
        order.outsideRth = outside_rth
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(0.5)
        return {'success': True, 'orderId': trade.order.orderId, 'status': trade.orderStatus.status}

    def get_market_depth_sync(self, symbol: str, num_rows: int = 10):
        from ib_insync import Stock
        from datetime import datetime
        contract = Stock(symbol, 'ISLAND', 'USD')
        self.ib.reqMktDepth(contract, numRows=num_rows)
        self.ib.sleep(2)
        ticker = self.ib.ticker(contract)
        bids = []
        asks = []
        if hasattr(ticker, 'domBids') and ticker.domBids:
            for level in ticker.domBids[:num_rows]:
                bids.append({'price': float(level.price or 0), 'size': int(level.size or 0), 'marketMaker': getattr(level, 'marketMaker', '')})
        if hasattr(ticker, 'domAsks') and ticker.domAsks:
            for level in ticker.domAsks[:num_rows]:
                asks.append({'price': float(level.price or 0), 'size': int(level.size or 0), 'marketMaker': getattr(level, 'marketMaker', '')})
        self.ib.cancelMktDepth(contract)
        if not bids and hasattr(ticker, 'bid') and ticker.bid:
            bids = [{'price': float(ticker.bid), 'size': int(ticker.bidSize or 0), 'marketMaker': ''}]
        if not asks and hasattr(ticker, 'ask') and ticker.ask:
            asks = [{'price': float(ticker.ask), 'size': int(ticker.askSize or 0), 'marketMaker': ''}]
        return {'symbol': symbol, 'bids': bids, 'asks': asks, 'timestamp': datetime.now().isoformat()}






