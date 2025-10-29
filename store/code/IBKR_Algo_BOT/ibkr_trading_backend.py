"""
AI Trading Bot - IBKR TWS Integration Backend
Modular architecture with AI-powered trading strategies
"""

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import threading
import time
import numpy as np
from collections import deque

# ============================================================================
# IBKR Connection Module
# ============================================================================

class IBKRClient(EWrapper, EClient):
    """Handles IBKR TWS API connection and data management"""
    
    def __init__(self, account_type='paper'):
        EClient.__init__(self, self)
        self.account_type = account_type
        self.next_order_id = None
        self.positions = {}
        self.account_data = {}
        self.market_data = {}
        self.watchlist = []
        self.data_queue = deque(maxlen=1000)
        
    def nextValidId(self, orderId: int):
        """Callback for next valid order ID"""
        super().nextValidId(orderId)
        self.next_order_id = orderId
        print(f"Connection established. Next Order ID: {orderId}")
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Error handling"""
        if errorCode in [2104, 2106, 2158]:  # Info messages
            print(f"Info: {errorString}")
        else:
            print(f"Error {errorCode}: {errorString}")
            
    def position(self, account, contract, position, avgCost):
        """Track positions"""
        self.positions[contract.symbol] = {
            'symbol': contract.symbol,
            'position': position,
            'avg_cost': avgCost,
            'contract': contract
        }
        
    def accountSummary(self, reqId, account, tag, value, currency):
        """Track account data"""
        self.account_data[tag] = value
        
    def tickPrice(self, reqId, tickType, price, attrib):
        """Real-time price updates"""
        if reqId in self.market_data:
            symbol = self.market_data[reqId]['symbol']
            if tickType == 1:  # Bid
                self.market_data[reqId]['bid'] = price
            elif tickType == 2:  # Ask
                self.market_data[reqId]['ask'] = price
            elif tickType == 4:  # Last
                self.market_data[reqId]['last'] = price
                
    def tickSize(self, reqId, tickType, size):
        """Volume and size updates"""
        if reqId in self.market_data:
            if tickType == 0:  # Bid Size
                self.market_data[reqId]['bid_size'] = size
            elif tickType == 3:  # Ask Size
                self.market_data[reqId]['ask_size'] = size
                
    def historicalData(self, reqId, bar):
        """Historical data callback"""
        data = {
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }
        self.data_queue.append(data)

# ============================================================================
# AI Trading Module - Alpha Fusion Strategy
# ============================================================================

class AlphaFusionStrategy:
    """
    Advanced AI strategy using LLN, momentum, sentiment, and microstructure
    Based on the mathematical model in project documentation
    """
    
    def __init__(self):
        self.beta = np.array([0.1, 0.3, 0.2, 0.25, 0.15, 0.1, 0.05, 0.05])  # Model weights
        self.learning_rate = 0.001
        self.ewma_alpha = 0.3
        self.volatility = {}
        self.drift = {}
        self.calibration_bins = {i: {'sum_p': 0, 'sum_y': 0, 'count': 0} 
                                for i in range(10)}
        
    def compute_features(self, symbol, market_data):
        """Extract features from market data"""
        bid = market_data.get('bid', 0)
        ask = market_data.get('ask', 0)
        bid_size = market_data.get('bid_size', 0)
        ask_size = market_data.get('ask_size', 0)
        last = market_data.get('last', 0)
        vwap = market_data.get('vwap', last)
        
        # Calculate mid price
        mid = (bid + ask) / 2 if bid and ask else last
        
        # 1. Order book imbalance
        epsilon = 1e-10
        imbalance = (bid_size - ask_size) / (bid_size + ask_size + epsilon)
        
        # 2. Short-term momentum (volatility normalized)
        if symbol not in self.volatility:
            self.volatility[symbol] = 0.01
        returns = (mid - market_data.get('prev_mid', mid)) / market_data.get('prev_mid', mid)
        self.volatility[symbol] = self.ewma_alpha * abs(returns) + (1 - self.ewma_alpha) * self.volatility[symbol]
        momentum = returns / (self.volatility[symbol] + epsilon)
        
        # 3. Drift (slow moving average of returns)
        if symbol not in self.drift:
            self.drift[symbol] = 0
        self.drift[symbol] = self.ewma_alpha * returns + (1 - self.ewma_alpha) * self.drift[symbol]
        
        # 4. VWAP distance
        vwap_dist = (mid - vwap) / (0.001 * mid + epsilon)
        
        # 5. Spread (normalized)
        spread = (ask - bid) / (mid + epsilon) if bid and ask else 0.001
        
        # 6. Barrier strength (round/half dollar levels)
        fractional = mid - int(mid)
        p_00 = max(0, 1 - abs(fractional - 0.00) / 0.05)
        p_50 = max(0, 1 - abs(fractional - 0.50) / 0.05)
        barrier = max(p_00, p_50)
        
        # 7. Sentiment (placeholder - integrate NewsAPI/Twitter)
        sentiment = 0.0  # To be populated by external feed
        
        return np.array([1, imbalance, momentum, self.drift[symbol], 
                        sentiment, barrier, vwap_dist, spread])
    
    def predict_probability(self, features):
        """Logistic regression for direction probability"""
        z = np.dot(self.beta, features)
        return 1 / (1 + np.exp(-z))
    
    def update_model(self, features, actual_outcome):
        """Online learning - SGD update"""
        predicted = self.predict_probability(features)
        error = actual_outcome - predicted
        self.beta += self.learning_rate * error * features - 0.01 * self.beta  # L2 reg
        
    def calibrate_probability(self, raw_prob):
        """Reliability-based calibration"""
        bin_idx = min(int(raw_prob * 10), 9)
        bin_data = self.calibration_bins[bin_idx]
        
        if bin_data['count'] > 10:
            avg_outcome = bin_data['sum_y'] / bin_data['count']
            avg_pred = bin_data['sum_p'] / bin_data['count']
            calibrated = raw_prob * (avg_outcome / (avg_pred + 1e-10))
            return np.clip(calibrated, 0, 1)
        return raw_prob
    
    def generate_signal(self, symbol, market_data):
        """Main signal generation"""
        features = self.compute_features(symbol, market_data)
        raw_prob = self.predict_probability(features)
        calibrated_prob = self.calibrate_probability(raw_prob)
        
        # Similarity boost (simplified - would use k-NN in production)
        similarity_boost = 1.0
        
        final_prob = np.clip(calibrated_prob * similarity_boost, 0, 1)
        
        return {
            'probability': final_prob,
            'direction': 'BUY' if final_prob > 0.6 else 'SELL' if final_prob < 0.4 else 'HOLD',
            'confidence': abs(final_prob - 0.5) * 2,
            'features': features
        }

# ============================================================================
# Risk Management Module
# ============================================================================

class RiskManager:
    """Handles position sizing, stop-loss, and compliance"""
    
    def __init__(self, config):
        self.max_position_size = config.get('max_position_size', 10000)
        self.daily_loss_limit = config.get('daily_loss_limit', 2000)
        self.stop_loss_pct = config.get('stop_loss_percent', 1.5) / 100
        self.target_gain_pct = config.get('target_gain_percent', 2.5) / 100
        self.daily_pnl = 0
        self.account_type = config.get('account_type', 'cash')
        
    def check_compliance(self, symbol, quantity, price):
        """Verify FINRA/SEC compliance"""
        # Cash account - no shorting
        if self.account_type == 'cash' and quantity < 0:
            return False, "Cash accounts cannot short"
            
        # PDT rule for margin accounts
        if self.account_type == 'margin':
            # Check day trade count (simplified)
            pass
            
        # Daily loss limit
        if self.daily_pnl <= -self.daily_loss_limit:
            return False, "Daily loss limit reached"
            
        return True, "OK"
    
    def calculate_position_size(self, signal_confidence, available_capital):
        """Kelly criterion-based sizing"""
        edge = (signal_confidence - 0.5) * 2  # Convert to [-1, 1]
        kelly_fraction = abs(edge) * 0.25  # Conservative Kelly
        size = min(available_capital * kelly_fraction, self.max_position_size)
        return int(size)
    
    def get_stop_loss(self, entry_price, direction):
        """Calculate stop loss price"""
        if direction == 'BUY':
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    def get_take_profit(self, entry_price, direction):
        """Calculate take profit price"""
        if direction == 'BUY':
            return entry_price * (1 + self.target_gain_pct)
        else:
            return entry_price * (1 - self.target_gain_pct)

# ============================================================================
# Main Trading Bot
# ============================================================================

class AITradingBot:
    """Main orchestrator - modular architecture"""
    
    def __init__(self, config):
        self.config = config
        self.ibkr = IBKRClient(config.get('account_type', 'paper'))
        self.strategy = AlphaFusionStrategy()
        self.risk_manager = RiskManager(config)
        self.active = False
        self.watchlist = config.get('watchlist', ['AAPL', 'TSLA', 'NVDA'])
        
    def connect(self, host='127.0.0.1', port=7497, client_id=1):
        """Connect to IBKR TWS"""
        self.ibkr.connect(host, port, client_id)
        
        # Start message processing thread
        thread = threading.Thread(target=self.ibkr.run, daemon=True)
        thread.start()
        
        # Wait for connection
        time.sleep(2)
        print(f"Connected to IBKR TWS on {host}:{port}")
        
    def create_stock_contract(self, symbol):
        """Create stock contract for IBKR"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = 'STK'
        contract.exchange = 'SMART'
        contract.currency = 'USD'
        return contract
    
    def subscribe_market_data(self, symbols):
        """Subscribe to real-time market data"""
        for idx, symbol in enumerate(symbols):
            contract = self.create_stock_contract(symbol)
            req_id = idx + 1000
            self.ibkr.market_data[req_id] = {'symbol': symbol}
            self.ibkr.reqMktData(req_id, contract, '', False, False, [])
            print(f"Subscribed to {symbol} market data")
    
    def place_order(self, symbol, action, quantity):
        """Place order with IBKR"""
        # Risk check
        market_data = self.get_symbol_data(symbol)
        price = market_data.get('last', 0)
        
        compliant, msg = self.risk_manager.check_compliance(
            symbol, 
            quantity if action == 'BUY' else -quantity,
            price
        )
        
        if not compliant:
            print(f"Order rejected: {msg}")
            return None
        
        # Create contract
        contract = self.create_stock_contract(symbol)
        
        # Create order
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = 'MKT'  # Market order (use 'LMT' for limit)
        
        # Place order
        order_id = self.ibkr.next_order_id
        self.ibkr.placeOrder(order_id, contract, order)
        self.ibkr.next_order_id += 1
        
        print(f"Order placed: {action} {quantity} {symbol} @ Market")
        return order_id
    
    def place_bracket_order(self, symbol, action, quantity, stop_loss, take_profit):
        """Place bracket order with stop-loss and take-profit"""
        contract = self.create_stock_contract(symbol)
        
        # Parent order
        parent = Order()
        parent.orderId = self.ibkr.next_order_id
        parent.action = action
        parent.orderType = 'MKT'
        parent.totalQuantity = quantity
        parent.transmit = False
        
        # Stop-loss order
        stop_loss_order = Order()
        stop_loss_order.orderId = self.ibkr.next_order_id + 1
        stop_loss_order.action = 'SELL' if action == 'BUY' else 'BUY'
        stop_loss_order.orderType = 'STP'
        stop_loss_order.auxPrice = stop_loss
        stop_loss_order.totalQuantity = quantity
        stop_loss_order.parentId = parent.orderId
        stop_loss_order.transmit = False
        
        # Take-profit order
        take_profit_order = Order()
        take_profit_order.orderId = self.ibkr.next_order_id + 2
        take_profit_order.action = 'SELL' if action == 'BUY' else 'BUY'
        take_profit_order.orderType = 'LMT'
        take_profit_order.lmtPrice = take_profit
        take_profit_order.totalQuantity = quantity
        take_profit_order.parentId = parent.orderId
        take_profit_order.transmit = True
        
        # Place all orders
        self.ibkr.placeOrder(parent.orderId, contract, parent)
        self.ibkr.placeOrder(stop_loss_order.orderId, contract, stop_loss_order)
        self.ibkr.placeOrder(take_profit_order.orderId, contract, take_profit_order)
        
        self.ibkr.next_order_id += 3
        print(f"Bracket order placed: {action} {quantity} {symbol}")
        
    def get_symbol_data(self, symbol):
        """Get market data for symbol"""
        for req_id, data in self.ibkr.market_data.items():
            if data.get('symbol') == symbol:
                return data
        return {}
    
    def trading_loop(self):
        """Main trading loop - AI signal generation and execution"""
        print("AI Trading Bot Started")
        
        while self.active:
            try:
                for symbol in self.watchlist:
                    # Get current market data
                    market_data = self.get_symbol_data(symbol)
                    
                    if not market_data or 'last' not in market_data:
                        continue
                    
                    # Generate AI signal
                    signal = self.strategy.generate_signal(symbol, market_data)
                    
                    print(f"{symbol}: {signal['direction']} - "
                          f"Prob: {signal['probability']:.2f}, "
                          f"Confidence: {signal['confidence']:.2f}")
                    
                    # Execute if confidence threshold met
                    if signal['confidence'] > 0.6:
                        # Get available capital
                        cash = float(self.ibkr.account_data.get('AvailableFunds', 100000))
                        
                        # Calculate position size
                        quantity = self.risk_manager.calculate_position_size(
                            signal['confidence'], 
                            cash
                        )
                        
                        price = market_data.get('last', 0)
                        shares = int(quantity / price) if price > 0 else 0
                        
                        if shares > 0 and signal['direction'] in ['BUY', 'SELL']:
                            # Calculate stop-loss and take-profit
                            stop_loss = self.risk_manager.get_stop_loss(
                                price, signal['direction']
                            )
                            take_profit = self.risk_manager.get_take_profit(
                                price, signal['direction']
                            )
                            
                            # Place bracket order
                            self.place_bracket_order(
                                symbol,
                                signal['direction'],
                                shares,
                                stop_loss,
                                take_profit
                            )
                
                # Sleep before next iteration
                time.sleep(5)
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(5)
    
    def start_trading(self):
        """Start the AI trading system"""
        self.active = True
        
        # Subscribe to market data for watchlist
        self.subscribe_market_data(self.watchlist)
        
        # Request account updates
        self.ibkr.reqAccountSummary(9001, "All", "$LEDGER")
        
        # Start trading loop in separate thread
        trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        trading_thread.start()
        
        print("AI Trading System Activated")
    
    def stop_trading(self):
        """Stop trading"""
        self.active = False
        print("AI Trading System Stopped")
    
    def get_positions(self):
        """Get current positions"""
        self.ibkr.reqPositions()
        time.sleep(1)
        return self.ibkr.positions
    
    def get_account_summary(self):
        """Get account summary"""
        return {
            'cash': self.ibkr.account_data.get('AvailableFunds', 0),
            'equity': self.ibkr.account_data.get('NetLiquidation', 0),
            'buying_power': self.ibkr.account_data.get('BuyingPower', 0),
        }

# ============================================================================
# Configuration & Main Execution
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = {
        'account_type': 'paper',  # 'paper', 'cash', or 'margin'
        'watchlist': ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL'],
        'max_position_size': 10000,
        'daily_loss_limit': 2000,
        'stop_loss_percent': 1.5,
        'target_gain_percent': 2.5,
        'min_volume': 1000000,
        'price_range_min': 5,
        'price_range_max': 500
    }
    
    # Initialize bot
    bot = AITradingBot(config)
    
    # Connect to IBKR TWS
    # Paper trading: port 7497, Live trading: port 7496
    bot.connect(host='127.0.0.1', port=7497, client_id=1)
    
    # Start trading
    bot.start_trading()
    
    # Keep running
    try:
        while True:
            time.sleep(10)
            # Print account summary periodically
            summary = bot.get_account_summary()
            print(f"\nAccount Summary: {summary}")
            positions = bot.get_positions()
            print(f"Positions: {len(positions)}")
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        bot.stop_trading()
        bot.ibkr.disconnect()