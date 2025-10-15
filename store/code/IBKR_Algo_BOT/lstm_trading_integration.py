"""
LSTM Trading Integration Module
Connects LSTM predictions to the trading bot's data bus
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMTradingModule:
    """
    Trading module that uses LSTM predictions for signal generation
    Integrates with the modular data bus architecture
    """
    
    def __init__(self, data_bus, config=None):
        self.data_bus = data_bus
        self.config = config or self.default_config()
        self.lstm_model = None
        self.ensemble_model = None
        self.historical_data = {}  # symbol -> deque of bars
        self.predictions = {}  # symbol -> latest prediction
        self.is_running = False
        self.update_thread = None
        
        # Performance tracking
        self.trade_log = []
        self.performance_metrics = {
            'accuracy': 0.0,
            'total_predictions': 0,
            'correct_predictions': 0
        }
        
    def default_config(self):
        return {
            'sequence_length': 60,
            'prediction_horizon': 5,
            'update_interval': 60,  # seconds
            'min_confidence': 0.6,
            'use_ensemble': True,
            'retrain_interval': 86400,  # 24 hours
            'min_bars_required': 100
        }
    
    def initialize(self):
        """Initialize LSTM models"""
        from lstm_model_complete import LSTMTradingModel, EnsembleLSTM
        
        logger.info("Initializing LSTM Trading Module...")
        
        if self.config['use_ensemble']:
            self.ensemble_model = EnsembleLSTM(
                n_models=3,
                sequence_length=self.config['sequence_length'],
                prediction_horizon=self.config['prediction_horizon']
            )
            logger.info("Ensemble model initialized")
        else:
            self.lstm_model = LSTMTradingModel(
                sequence_length=self.config['sequence_length'],
                prediction_horizon=self.config['prediction_horizon']
            )
            logger.info("Single LSTM model initialized")
        
        # Subscribe to data bus
        self.data_bus.subscribe('market_data', self.on_market_data)
        self.data_bus.subscribe('historical_data', self.on_historical_data)
        
        logger.info("LSTM Module initialized successfully")
    
    def on_market_data(self, data):
        """Handle real-time market data updates"""
        symbol = data.get('symbol')
        
        if symbol not in self.historical_data:
            self.historical_data[symbol] = deque(maxlen=500)  # keep last 500 bars
        
        # Add new bar
        bar = {
            'timestamp': data.get('timestamp', datetime.now()),
            'open': data.get('open'),
            'high': data.get('high'),
            'low': data.get('low'),
            'close': data.get('close'),
            'volume': data.get('volume', 0)
        }
        
        self.historical_data[symbol].append(bar)
        
        # Update prediction if enough data
        if len(self.historical_data[symbol]) >= self.config['min_bars_required']:
            self.update_prediction(symbol)
    
    def on_historical_data(self, data):
        """Handle historical data load for training"""
        symbol = data.get('symbol')
        bars = data.get('bars', [])
        
        if symbol not in self.historical_data:
            self.historical_data[symbol] = deque(maxlen=500)
        
        for bar in bars:
            self.historical_data[symbol].append(bar)
        
        logger.info(f"Loaded {len(bars)} historical bars for {symbol}")
    
    def bars_to_dataframe(self, symbol):
        """Convert stored bars to pandas DataFrame"""
        if symbol not in self.historical_data or not self.historical_data[symbol]:
            return None
        
        bars = list(self.historical_data[symbol])
        df = pd.DataFrame(bars)
        
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def update_prediction(self, symbol):
        """Generate new prediction for symbol"""
        df = self.bars_to_dataframe(symbol)
        
        if df is None or len(df) < self.config['sequence_length']:
            return
        
        try:
            if self.config['use_ensemble'] and self.ensemble_model:
                probability, confidence = self.ensemble_model.predict_with_confidence(df)
            elif self.lstm_model:
                probability = self.lstm_model.predict(df)
                confidence = 0.8  # default confidence for single model
            else:
                logger.warning("No model available for prediction")
                return
            
            # Store prediction
            self.predictions[symbol] = {
                'probability': probability,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'direction': 'BUY' if probability > 0.5 else 'SELL',
                'strength': abs(probability - 0.5) * 2  # 0-1 scale
            }
            
            # Publish to data bus if meets confidence threshold
            if confidence >= self.config['min_confidence']:
                signal = {
                    'module': 'lstm',
                    'symbol': symbol,
                    'signal_type': 'BUY' if probability > 0.5 else 'SELL',
                    'probability': probability,
                    'confidence': confidence,
                    'strength': abs(probability - 0.5) * 2,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.data_bus.publish('ai_signals', signal)
                logger.info(f"LSTM Signal: {symbol} {signal['signal_type']} @ {probability:.3f} (conf: {confidence:.3f})")
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
    
    def train_on_historical_data(self, symbol, epochs=50):
        """Train model on historical data for a symbol"""
        df = self.bars_to_dataframe(symbol)
        
        if df is None or len(df) < 200:
            logger.warning(f"Insufficient data to train on {symbol}")
            return None
        
        logger.info(f"Training LSTM on {symbol} with {len(df)} bars...")
        
        try:
            if self.config['use_ensemble']:
                results = self.ensemble_model.train_ensemble(df, epochs=epochs)
                logger.info(f"Ensemble training complete for {symbol}")
                return results
            else:
                history, results = self.lstm_model.train(df, epochs=epochs)
                self.lstm_model.save(f"lstm_{symbol}")
                logger.info(f"Model training complete for {symbol}")
                return results
                
        except Exception as e:
            logger.error(f"Training error for {symbol}: {e}")
            return None
    
    def start(self):
        """Start the LSTM module"""
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        logger.info("LSTM Trading Module started")
    
    def stop(self):
        """Stop the LSTM module"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logger.info("LSTM Trading Module stopped")
    
    def _update_loop(self):
        """Background loop for periodic updates"""
        last_retrain = datetime.now()
        
        while self.is_running:
            try:
                # Update predictions for all symbols
                for symbol in list(self.historical_data.keys()):
                    self.update_prediction(symbol)
                
                # Periodic retraining
                if (datetime.now() - last_retrain).total_seconds() > self.config['retrain_interval']:
                    logger.info("Starting periodic model retraining...")
                    for symbol in list(self.historical_data.keys()):
                        self.train_on_historical_data(symbol, epochs=20)
                    last_retrain = datetime.now()
                
                time.sleep(self.config['update_interval'])
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(5)
    
    def get_signal_for_symbol(self, symbol):
        """Get latest LSTM signal for a symbol"""
        if symbol in self.predictions:
            pred = self.predictions[symbol]
            
            # Check if prediction is recent (within 5 minutes)
            age = (datetime.now() - pred['timestamp']).total_seconds()
            if age < 300:
                return {
                    'valid': True,
                    'direction': pred['direction'],
                    'probability': pred['probability'],
                    'confidence': pred['confidence'],
                    'strength': pred['strength']
                }
        
        return {'valid': False}
    
    def evaluate_performance(self):
        """Calculate model performance metrics"""
        if not self.trade_log:
            return self.performance_metrics
        
        df = pd.DataFrame(self.trade_log)
        
        if len(df) > 0:
            self.performance_metrics = {
                'accuracy': df['correct'].mean(),
                'total_predictions': len(df),
                'correct_predictions': df['correct'].sum(),
                'avg_profit': df['profit'].mean() if 'profit' in df else 0,
                'sharpe_ratio': self._calculate_sharpe(df['profit']) if 'profit' in df else 0
            }
        
        return self.performance_metrics
    
    def _calculate_sharpe(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        
        excess_returns = returns - (risk_free_rate / 252)  # daily risk-free rate
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() > 0 else 0
    
    def get_status(self):
        """Get module status"""
        return {
            'running': self.is_running,
            'symbols_tracked': len(self.historical_data),
            'active_predictions': len(self.predictions),
            'model_type': 'ensemble' if self.config['use_ensemble'] else 'single',
            'performance': self.performance_metrics,
            'config': self.config
        }


# Integration with main trading bot
class LSTMStrategyExecutor:
    """
    Executes trades based on LSTM predictions
    Integrates with risk management and order execution
    """
    
    def __init__(self, lstm_module, risk_manager, order_executor):
        self.lstm = lstm_module
        self.risk_manager = risk_manager
        self.order_executor = order_executor
        self.active_positions = {}
        
    def execute_strategy(self, symbol, account_data):
        """Main strategy execution logic"""
        signal = self.lstm.get_signal_for_symbol(symbol)
        
        if not signal['valid']:
            return None
        
        # Check confidence threshold
        if signal['confidence'] < 0.65:
            logger.debug(f"Signal confidence too low for {symbol}: {signal['confidence']:.3f}")
            return None
        
        # Determine position size using Kelly Criterion
        edge = abs(signal['probability'] - 0.5) * 2  # 0-1 scale
        kelly_fraction = edge * signal['confidence']
        
        position_size = self.risk_manager.calculate_position_size(
            symbol=symbol,
            signal_strength=kelly_fraction,
            available_capital=account_data['buying_power']
        )
        
        if position_size <= 0:
            return None
        
        # Create order
        order = {
            'symbol': symbol,
            'action': signal['direction'],
            'quantity': position_size,
            'order_type': 'LIMIT',
            'strategy': 'LSTM',
            'signal_probability': signal['probability'],
            'signal_confidence': signal['confidence']
        }
        
        # Validate with risk manager
        if self.risk_manager.validate_order(order, account_data):
            # Execute order
            result = self.order_executor.place_order(order)
            
            if result['success']:
                logger.info(f"LSTM trade executed: {symbol} {signal['direction']} {position_size} shares")
                self.active_positions[symbol] = {
                    'order': order,
                    'entry_time': datetime.now(),
                    'prediction': signal
                }
                return result
        
        return None
    
    def manage_positions(self, market_data):
        """Monitor and manage active positions"""
        for symbol, position in list(self.active_positions.items()):
            # Get latest signal
            current_signal = self.lstm.get_signal_for_symbol(symbol)
            
            if current_signal['valid']:
                # Exit if signal reverses with high confidence
                if current_signal['direction'] != position['prediction']['direction'] and current_signal['confidence'] > 0.7:
                    logger.info(f"Closing {symbol} position - signal reversal detected")
                    self.close_position(symbol, reason='signal_reversal')
                    
    def close_position(self, symbol, reason='manual'):
        """Close an active position"""
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            
            # Create closing order
            close_order = {
                'symbol': symbol,
                'action': 'SELL' if position['order']['action'] == 'BUY' else 'BUY',
                'quantity': position['order']['quantity'],
                'order_type': 'MARKET',
                'reason': reason
            }
            
            result = self.order_executor.place_order(close_order)
            
            if result['success']:
                del self.active_positions[symbol]
                logger.info(f"Position closed: {symbol} - Reason: {reason}")
                
            return result
        
        return None


# Example integration with main bot
if __name__ == "__main__":
    print("LSTM Trading Integration Module")
    print("=" * 60)
    
    # Mock data bus for testing
    class MockDataBus:
        def __init__(self):
            self.channels = {}
            
        def subscribe(self, channel, callback):
            if channel not in self.channels:
                self.channels[channel] = []
            self.channels[channel].append(callback)
            
        def publish(self, channel, data):
            if channel in self.channels:
                for callback in self.channels[channel]:
                    callback(data)
    
    # Initialize
    data_bus = MockDataBus()
    
    config = {
        'sequence_length': 60,
        'prediction_horizon': 5,
        'update_interval': 30,
        'min_confidence': 0.6,
        'use_ensemble': True
    }
    
    lstm_module = LSTMTradingModule(data_bus, config)
    lstm_module.initialize()
    
    # Simulate historical data
    print("\nSimulating historical data load...")
    historical_bars = []
    price = 100
    for i in range(200):
        price += np.random.randn() * 0.5
        bar = {
            'timestamp': datetime.now() - timedelta(minutes=200-i),
            'open': price,
            'high': price + abs(np.random.randn() * 0.3),
            'low': price - abs(np.random.randn() * 0.3),
            'close': price + np.random.randn() * 0.2,
            'volume': np.random.randint(100000, 1000000)
        }
        historical_bars.append(bar)
    
    data_bus.publish('historical_data', {
        'symbol': 'AAPL',
        'bars': historical_bars
    })
    
    # Train model
    print("\nTraining LSTM model on historical data...")
    lstm_module.train_on_historical_data('AAPL', epochs=10)
    
    # Simulate real-time updates
    print("\nSimulating real-time market data...")
    for i in range(5):
        price += np.random.randn() * 0.5
        market_data = {
            'symbol': 'AAPL',
            'timestamp': datetime.now(),
            'open': price,
            'high': price + 0.3,
            'low': price - 0.3,
            'close': price,
            'volume': np.random.randint(100000, 500000)
        }
        
        data_bus.publish('market_data', market_data)
        time.sleep(1)
    
    # Get status
    status = lstm_module.get_status()
    print("\nLSTM Module Status:")
    print(f"  Running: {status['running']}")
    print(f"  Symbols Tracked: {status['symbols_tracked']}")
    print(f"  Active Predictions: {status['active_predictions']}")
    print(f"  Model Type: {status['model_type']}")
    
    # Get signal
    signal = lstm_module.get_signal_for_symbol('AAPL')
    if signal['valid']:
        print(f"\nLatest Signal for AAPL:")
        print(f"  Direction: {signal['direction']}")
        print(f"  Probability: {signal['probability']:.3f}")
        print(f"  Confidence: {signal['confidence']:.3f}")
        print(f"  Strength: {signal['strength']:.3f}")
    
    print("\n" + "=" * 60)
    print("LSTM Integration module ready!")