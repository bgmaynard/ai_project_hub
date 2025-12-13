"""
IBKR Live Trading Connector with MTF Model Integration
======================================================

Connects your trained MTF models to Interactive Brokers for live trading.

Features:
- Real-time market data streaming
- Order placement with risk management
- Position reconciliation
- Account monitoring
- Paper/Live trading support

Usage:
    python ibkr_live_trading_connector.py --mode paper --symbols AAPL TSLA
"""

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import threading
import time
import pandas as pd
from datetime import datetime
import logging
from tensorflow import keras
import pickle
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IBKRConnector(EWrapper, EClient):
    """
    IBKR TWS API Connector
    
    Handles connection, market data, orders, and positions
    """
    
    def __init__(self):
        EClient.__init__(self, self)
        
        # Connection status
        self.connected = False
        self.next_order_id = None
        
        # Market data
        self.market_data = {}  # {symbol: {bid, ask, last, volume, ...}}
        self.req_id_to_symbol = {}  # Map request IDs to symbols
        self.symbol_to_req_id = {}  # Map symbols to request IDs
        
        # Historical data
        self.historical_data = {}  # {symbol: DataFrame}
        self.historical_complete = {}  # {symbol: bool}
        
        # Positions & Account
        self.positions = {}  # {symbol: {quantity, avg_price, ...}}
        self.account_values = {}  # {key: value}
        self.orders = {}  # {order_id: order_status}
        
        # Feature buffers for LSTM
        self.feature_buffers = {}  # {symbol: deque of recent bars}
        
    # ========================================
    # Connection Callbacks
    # ========================================
    
    def nextValidId(self, orderId: int):
        """Callback when connection established"""
        super().nextValidId(orderId)
        self.next_order_id = orderId
        self.connected = True
        logger.info(f"✓ Connected to IBKR. Next order ID: {orderId}")
    
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Error handler"""
        if errorCode == 202:  # Order cancelled
            logger.info(f"Order {reqId} cancelled")
        elif errorCode in [2104, 2106, 2158]:  # Market data farm connection
            logger.info(f"Market data: {errorString}")
        elif errorCode >= 2000:  # Informational
            logger.info(f"Info ({errorCode}): {errorString}")
        else:
            logger.error(f"Error {errorCode} (reqId {reqId}): {errorString}")
    
    # ========================================
    # Market Data Callbacks
    # ========================================
    
    def tickPrice(self, reqId, tickType, price, attrib):
        """Real-time price updates"""
        if reqId not in self.req_id_to_symbol:
            return
            
        symbol = self.req_id_to_symbol[reqId]
        
        if symbol not in self.market_data:
            self.market_data[symbol] = {}
        
        # Map tick types to fields
        tick_map = {
            1: 'bid',
            2: 'ask',
            4: 'last',
            6: 'high',
            7: 'low',
            9: 'close'
        }
        
        if tickType in tick_map:
            field = tick_map[tickType]
            self.market_data[symbol][field] = price
            self.market_data[symbol]['timestamp'] = datetime.now()
            
            logger.debug(f"{symbol} {field}: ${price:.2f}")
    
    def tickSize(self, reqId, tickType, size):
        """Real-time size updates"""
        if reqId not in self.req_id_to_symbol:
            return
            
        symbol = self.req_id_to_symbol[reqId]
        
        if symbol not in self.market_data:
            self.market_data[symbol] = {}
        
        # Map size types
        size_map = {
            0: 'bid_size',
            3: 'ask_size',
            5: 'last_size',
            8: 'volume'
        }
        
        if tickType in size_map:
            field = size_map[tickType]
            self.market_data[symbol][field] = size
    
    def historicalData(self, reqId, bar):
        """Historical data bars"""
        if reqId not in self.req_id_to_symbol:
            return
            
        symbol = self.req_id_to_symbol[reqId]
        
        if symbol not in self.historical_data:
            self.historical_data[symbol] = []
        
        # Store bar data
        bar_data = {
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }
        
        self.historical_data[symbol].append(bar_data)
    
    def historicalDataEnd(self, reqId, start, end):
        """Historical data complete"""
        if reqId not in self.req_id_to_symbol:
            return
            
        symbol = self.req_id_to_symbol[reqId]
        self.historical_complete[symbol] = True
        
        # Convert to DataFrame
        df = pd.DataFrame(self.historical_data[symbol])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        self.historical_data[symbol] = df
        
        logger.info(f"✓ Historical data complete for {symbol}: {len(df)} bars")
    
    # ========================================
    # Position & Account Callbacks
    # ========================================
    
    def position(self, account, contract, position, avgCost):
        """Position updates"""
        symbol = contract.symbol
        
        self.positions[symbol] = {
            'quantity': position,
            'avg_cost': avgCost,
            'contract': contract,
            'account': account
        }
        
        logger.info(f"Position: {symbol} = {position} @ ${avgCost:.2f}")
    
    def positionEnd(self):
        """All positions received"""
        logger.info(f"✓ Received {len(self.positions)} positions")
    
    def updateAccountValue(self, key, val, currency, accountName):
        """Account value updates"""
        self.account_values[key] = {
            'value': val,
            'currency': currency,
            'account': accountName
        }
    
    def accountDownloadEnd(self, accountName):
        """Account data complete"""
        logger.info(f"✓ Account data complete for {accountName}")
    
    # ========================================
    # Order Callbacks
    # ========================================
    
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, 
                   permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        """Order status updates"""
        self.orders[orderId] = {
            'status': status,
            'filled': filled,
            'remaining': remaining,
            'avg_fill_price': avgFillPrice,
            'timestamp': datetime.now()
        }
        
        logger.info(f"Order {orderId}: {status} (filled {filled}, remaining {remaining})")
    
    def openOrder(self, orderId, contract, order, orderState):
        """Open order updates"""
        logger.info(f"Open order {orderId}: {order.action} {order.totalQuantity} {contract.symbol}")
    
    def execDetails(self, reqId, contract, execution):
        """Execution details"""
        logger.info(f"Execution: {execution.side} {execution.shares} {contract.symbol} @ ${execution.price:.2f}")


class MTFModelTrader:
    """
    Integrates MTF models with IBKR for live trading
    """
    
    def __init__(self, model_dir='models/lstm_mtf_v2'):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.ibkr = IBKRConnector()
        
        # Trading parameters
        self.confidence_threshold = 0.52
        self.min_hold_bars = 2
        self.max_hold_bars = 24
        
        # Position tracking
        self.active_positions = {}  # {symbol: {entry_time, entry_price, ...}}
        
        # Risk management
        self.max_position_size = 10000  # $10k per position
        self.daily_loss_limit = 2000    # $2k daily loss limit
        self.daily_pnl = 0
        
    def connect(self, host='127.0.0.1', port=7497, client_id=1):
        """Connect to IBKR TWS"""
        logger.info(f"Connecting to IBKR at {host}:{port}...")
        
        self.ibkr.connect(host, port, client_id)
        
        # Start API thread
        api_thread = threading.Thread(target=self.ibkr.run, daemon=True)
        api_thread.start()
        
        # Wait for connection
        time.sleep(2)
        
        if not self.ibkr.connected:
            raise ConnectionError("Failed to connect to IBKR")
        
        logger.info("✓ Connected to IBKR")
        
        # Request account data
        self.ibkr.reqAccountUpdates(True, "")
        
        # Request positions
        self.ibkr.reqPositions()
        
        time.sleep(1)
    
    def disconnect(self):
        """Disconnect from IBKR"""
        self.ibkr.disconnect()
        logger.info("Disconnected from IBKR")
    
    def load_model(self, symbol):
        """Load trained MTF model for symbol"""
        try:
            model_path = f"{self.model_dir}/{symbol}_mtf_v2.keras"
            scaler_path = f"{self.model_dir}/{symbol}_mtf_v2_scaler.pkl"
            
            logger.info(f"Loading model for {symbol}...")
            
            # Load model
            self.models[symbol] = keras.models.load_model(model_path)
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scalers[symbol] = pickle.load(f)
            
            logger.info(f"✓ Model loaded for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model for {symbol}: {e}")
            return False
    
    def subscribe_market_data(self, symbol, req_id):
        """Subscribe to real-time market data"""
        # Create contract
        contract = Contract()
        contract.symbol = symbol
        contract.secType = 'STK'
        contract.exchange = 'SMART'
        contract.currency = 'USD'
        
        # Map request ID to symbol
        self.ibkr.req_id_to_symbol[req_id] = symbol
        self.ibkr.symbol_to_req_id[symbol] = req_id
        
        # Request market data
        self.ibkr.reqMktData(req_id, contract, '', False, False, [])
        
        logger.info(f"✓ Subscribed to {symbol} market data (reqId={req_id})")
    
    def request_historical_data(self, symbol, req_id, duration='2 D', bar_size='1 hour'):
        """Request historical data for feature calculation"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = 'STK'
        contract.exchange = 'SMART'
        contract.currency = 'USD'
        
        # Map request ID
        self.ibkr.req_id_to_symbol[req_id] = symbol
        
        # Request historical bars
        end_time = datetime.now().strftime('%Y%m%d %H:%M:%S')
        
        self.ibkr.reqHistoricalData(
            req_id, contract, end_time, duration, bar_size,
            'TRADES', 1, 1, False, []
        )
        
        logger.info(f"Requesting {duration} of {bar_size} data for {symbol}...")
        
        # Wait for data
        timeout = 30
        start_time = time.time()
        while symbol not in self.ibkr.historical_complete:
            if time.time() - start_time > timeout:
                logger.error(f"Timeout waiting for {symbol} historical data")
                return None
            time.sleep(0.5)
        
        return self.ibkr.historical_data[symbol]
    
    def calculate_features(self, df, symbol):
        """Calculate MTF features from DataFrame"""
        # Import feature calculation from your training script
        from EASY_MTF_TRAINER_V2 import add_base_features, add_mtf_features
        
        # Add features
        df = add_base_features(df)
        df = add_mtf_features(df)
        df = df.dropna()
        
        return df
    
    def predict(self, symbol):
        """Generate prediction for symbol"""
        # Get historical data
        df = self.ibkr.historical_data.get(symbol)
        
        if df is None or len(df) < 100:
            logger.warning(f"Insufficient data for {symbol}")
            return None, None
        
        # Calculate features
        df = self.calculate_features(df, symbol)
        
        # Select features (exclude OHLCV)
        exclude = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = [c for c in df.columns if c not in exclude]
        
        # Get latest bar
        X = df[feature_cols].iloc[-1:].values
        
        # Scale
        X_scaled = self.scalers[symbol].transform(X)
        
        # Reshape for LSTM
        X_lstm = X_scaled.reshape((1, 1, X_scaled.shape[1]))
        
        # Predict
        prediction_proba = self.models[symbol].predict(X_lstm, verbose=0)[0][0]
        prediction_class = 1 if prediction_proba > 0.5 else 0
        
        return prediction_class, prediction_proba
    
    def place_order(self, symbol, action, quantity):
        """Place market order"""
        # Create contract
        contract = Contract()
        contract.symbol = symbol
        contract.secType = 'STK'
        contract.exchange = 'SMART'
        contract.currency = 'USD'
        
        # Create order
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = 'MKT'
        
        # Place order
        order_id = self.ibkr.next_order_id
        self.ibkr.placeOrder(order_id, contract, order)
        self.ibkr.next_order_id += 1
        
        logger.info(f"✓ Order placed: {action} {quantity} {symbol} @ Market (ID={order_id})")
        
        return order_id
    
    def check_entry_signal(self, symbol):
        """Check if should enter position"""
        # Skip if already in position
        if symbol in self.active_positions:
            return False
        
        # Skip if daily loss limit hit
        if self.daily_pnl <= -self.daily_loss_limit:
            logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return False
        
        # Get prediction
        pred_class, pred_confidence = self.predict(symbol)
        
        if pred_class is None:
            return False
        
        # Check signal
        if pred_class == 1 and pred_confidence >= self.confidence_threshold:
            logger.info(f"{symbol} BUY signal: confidence={pred_confidence:.3f}")
            return True
        
        return False
    
    def check_exit_signal(self, symbol):
        """Check if should exit position"""
        if symbol not in self.active_positions:
            return False
        
        position = self.active_positions[symbol]
        
        # Check holding period
        bars_held = position.get('bars_held', 0)
        
        if bars_held < self.min_hold_bars:
            return False
        
        # Get prediction
        pred_class, pred_confidence = self.predict(symbol)
        
        if pred_class is None:
            return False
        
        # Exit conditions
        if pred_class == 0:  # Sell signal
            logger.info(f"{symbol} SELL signal: confidence={1-pred_confidence:.3f}")
            return True
        
        if bars_held >= self.max_hold_bars:  # Max hold reached
            logger.info(f"{symbol} max hold period reached ({bars_held} bars)")
            return True
        
        return False
    
    def run_trading_loop(self, symbols, check_interval=300):
        """
        Main trading loop
        
        Args:
            symbols: List of symbols to trade
            check_interval: Seconds between checks (300 = 5 minutes)
        """
        logger.info(f"\n{'='*70}")
        logger.info("STARTING LIVE TRADING LOOP")
        logger.info(f"{'='*70}")
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"Check interval: {check_interval}s ({check_interval/60:.1f} min)")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info(f"{'='*70}\n")
        
        # Load models
        for symbol in symbols:
            if not self.load_model(symbol):
                logger.error(f"Failed to load model for {symbol}, skipping")
                symbols.remove(symbol)
        
        # Subscribe to market data
        for i, symbol in enumerate(symbols):
            req_id = 1000 + i
            self.subscribe_market_data(symbol, req_id)
            time.sleep(1)
        
        # Main loop
        try:
            iteration = 0
            while True:
                iteration += 1
                logger.info(f"\n--- Iteration {iteration} @ {datetime.now().strftime('%H:%M:%S')} ---")
                
                for symbol in symbols:
                    # Request fresh historical data
                    hist_req_id = 2000 + symbols.index(symbol)
                    df = self.request_historical_data(symbol, hist_req_id)
                    
                    if df is None:
                        continue
                    
                    # Check signals
                    if self.check_entry_signal(symbol):
                        # Calculate position size (simple: 100 shares)
                        quantity = 100
                        
                        # Place buy order
                        order_id = self.place_order(symbol, 'BUY', quantity)
                        
                        # Track position
                        self.active_positions[symbol] = {
                            'entry_time': datetime.now(),
                            'entry_price': self.ibkr.market_data[symbol].get('last'),
                            'quantity': quantity,
                            'order_id': order_id,
                            'bars_held': 0
                        }
                    
                    elif self.check_exit_signal(symbol):
                        position = self.active_positions[symbol]
                        quantity = position['quantity']
                        
                        # Place sell order
                        order_id = self.place_order(symbol, 'SELL', quantity)
                        
                        # Calculate P&L
                        entry_price = position['entry_price']
                        exit_price = self.ibkr.market_data[symbol].get('last')
                        pnl = (exit_price - entry_price) * quantity
                        
                        logger.info(f"{symbol} P&L: ${pnl:+.2f}")
                        
                        # Update daily P&L
                        self.daily_pnl += pnl
                        
                        # Remove from active positions
                        del self.active_positions[symbol]
                    
                    else:
                        # Update bars held
                        if symbol in self.active_positions:
                            self.active_positions[symbol]['bars_held'] += 1
                
                # Display status
                logger.info(f"\nActive positions: {len(self.active_positions)}")
                logger.info(f"Daily P&L: ${self.daily_pnl:+.2f}")
                
                # Sleep until next check
                logger.info(f"Sleeping {check_interval}s...")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("\n\nStopped by user")
        
        finally:
            # Close any open positions
            for symbol in list(self.active_positions.keys()):
                position = self.active_positions[symbol]
                self.place_order(symbol, 'SELL', position['quantity'])
                logger.info(f"Closed position in {symbol}")
            
            self.disconnect()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='IBKR Live Trading with MTF Models')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                       help='Trading mode (paper or live)')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'TSLA'],
                       help='Symbols to trade')
    parser.add_argument('--port', type=int, default=7497,
                       help='TWS port (7497=paper, 7496=live)')
    parser.add_argument('--interval', type=int, default=300,
                       help='Check interval in seconds')
    parser.add_argument('--confidence', type=float, default=0.52,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("IBKR LIVE TRADING CONNECTOR - MTF MODELS")
    print("="*70)
    print(f"Mode: {args.mode.upper()}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Port: {args.port}")
    print(f"Check interval: {args.interval}s")
    print(f"Confidence threshold: {args.confidence}")
    print("="*70 + "\n")
    
    # Create trader
    trader = MTFModelTrader()
    trader.confidence_threshold = args.confidence
    
    # Connect
    trader.connect(port=args.port)
    
    # Run trading loop
    trader.run_trading_loop(args.symbols, check_interval=args.interval)


if __name__ == "__main__":
    main()
