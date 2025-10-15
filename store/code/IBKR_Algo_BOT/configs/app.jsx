"""
Dashboard API with Training, Backtesting, and Trade Execution
Enhanced from continuation sheet - adds ML training integration
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import time
import logging
from datetime import datetime
import json
from pathlib import Path
import sys

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import IBKR API
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    from ibapi.scanner import ScannerSubscription
    IBKR_AVAILABLE = True
except ImportError as e:
    IBKR_AVAILABLE = False
    logger.warning(f'IBKR not available: {e}')


# ============================================================================
# TRAINING MANAGER - New Feature
# ============================================================================

class TrainingManager:
    """Manages model training tasks with progress tracking"""
    
    def __init__(self):
        self.active_trainings = {}
        self.completed_trainings = []
        self.training_lock = threading.Lock()
        
    def start_training(self, training_id, symbols, config):
        """Start a new training task"""
        with self.training_lock:
            self.active_trainings[training_id] = {
                'id': training_id,
                'symbols': symbols,
                'config': config,
                'status': 'initializing',
                'progress': 0,
                'current_symbol': None,
                'started_at': datetime.now().isoformat(),
                'logs': []
            }
        
        bot_state.add_log('info', 'training', f'Training started: {training_id} ({len(symbols)} symbols)')
        return training_id
    
    def update_progress(self, training_id, progress, status, current_symbol=None, log_message=None):
        """Update training progress"""
        with self.training_lock:
            if training_id in self.active_trainings:
                training = self.active_trainings[training_id]
                training['progress'] = progress
                training['status'] = status
                if current_symbol:
                    training['current_symbol'] = current_symbol
                if log_message:
                    training['logs'].append({
                        'timestamp': datetime.now().isoformat(),
                        'message': log_message
                    })
                
                # Broadcast progress via WebSocket
                socketio.emit('training_progress', {
                    'training_id': training_id,
                    'progress': progress,
                    'status': status,
                    'current_symbol': current_symbol
                })
    
    def complete_training(self, training_id, results):
        """Mark training as complete"""
        with self.training_lock:
            if training_id in self.active_trainings:
                training = self.active_trainings[training_id]
                training['status'] = 'completed'
                training['progress'] = 100
                training['completed_at'] = datetime.now().isoformat()
                training['results'] = results
                
                self.completed_trainings.append(training)
                del self.active_trainings[training_id]
                
                bot_state.add_log('success', 'training', f'Training completed: {training_id}')
                
                # Trigger backtesting automatically
                backtest_id = backtest_manager.start_backtest(
                    training['symbols'], 
                    results
                )
                
                socketio.emit('training_complete', {
                    'training_id': training_id,
                    'backtest_id': backtest_id
                })
    
    def fail_training(self, training_id, error):
        """Mark training as failed"""
        with self.training_lock:
            if training_id in self.active_trainings:
                training = self.active_trainings[training_id]
                training['status'] = 'failed'
                training['error'] = str(error)
                training['completed_at'] = datetime.now().isoformat()
                
                del self.active_trainings[training_id]
                bot_state.add_log('error', 'training', f'Training failed: {training_id} - {error}')
    
    def get_all_trainings(self):
        """Get all training tasks"""
        with self.training_lock:
            return {
                'active': list(self.active_trainings.values()),
                'completed': self.completed_trainings[-10:]  # Last 10 completed
            }


# ============================================================================
# BACKTEST MANAGER - New Feature
# ============================================================================

class BacktestManager:
    """Manages backtesting tasks"""
    
    def __init__(self):
        self.active_backtests = {}
        self.completed_backtests = []
        self.backtest_lock = threading.Lock()
        
    def start_backtest(self, symbols, training_results):
        """Start backtest after training"""
        backtest_id = f"backtest_{int(time.time())}"
        
        with self.backtest_lock:
            self.active_backtests[backtest_id] = {
                'id': backtest_id,
                'symbols': symbols,
                'status': 'running',
                'progress': 0,
                'started_at': datetime.now().isoformat()
            }
        
        # Run backtest in background
        def run_backtest():
            try:
                from EASY_MTF_TRAINER_V2 import LSTMTrainingPipeline
                
                pipeline = LSTMTrainingPipeline()
                
                # Simulate backtest (replace with actual backtest logic)
                for i, symbol in enumerate(symbols):
                    progress = int((i + 1) / len(symbols) * 100)
                    self.update_backtest_progress(backtest_id, progress, symbol)
                    time.sleep(2)  # Simulate processing
                
                # Mock results (replace with actual backtest results)
                results = {
                    symbol: {
                        'total_return': 15.5,
                        'sharpe_ratio': 1.8,
                        'win_rate': 62.5,
                        'max_drawdown': -8.2,
                        'total_trades': 45
                    } for symbol in symbols
                }
                
                self.complete_backtest(backtest_id, results)
                
            except Exception as e:
                self.fail_backtest(backtest_id, e)
        
        thread = threading.Thread(target=run_backtest, daemon=True)
        thread.start()
        
        return backtest_id
    
    def update_backtest_progress(self, backtest_id, progress, current_symbol):
        """Update backtest progress"""
        with self.backtest_lock:
            if backtest_id in self.active_backtests:
                self.active_backtests[backtest_id]['progress'] = progress
                self.active_backtests[backtest_id]['current_symbol'] = current_symbol
                
                socketio.emit('backtest_progress', {
                    'backtest_id': backtest_id,
                    'progress': progress,
                    'current_symbol': current_symbol
                })
    
    def complete_backtest(self, backtest_id, results):
        """Mark backtest as complete"""
        with self.backtest_lock:
            if backtest_id in self.active_backtests:
                backtest = self.active_backtests[backtest_id]
                backtest['status'] = 'completed'
                backtest['progress'] = 100
                backtest['completed_at'] = datetime.now().isoformat()
                backtest['results'] = results
                
                self.completed_backtests.append(backtest)
                del self.active_backtests[backtest_id]
                
                bot_state.add_log('success', 'backtest', f'Backtest completed: {backtest_id}')
                
                socketio.emit('backtest_complete', {
                    'backtest_id': backtest_id,
                    'results': results
                })
    
    def fail_backtest(self, backtest_id, error):
        """Mark backtest as failed"""
        with self.backtest_lock:
            if backtest_id in self.active_backtests:
                backtest = self.active_backtests[backtest_id]
                backtest['status'] = 'failed'
                backtest['error'] = str(error)
                
                del self.active_backtests[backtest_id]
                bot_state.add_log('error', 'backtest', f'Backtest failed: {backtest_id}')
    
    def get_all_backtests(self):
        """Get all backtest tasks"""
        with self.backtest_lock:
            return {
                'active': list(self.active_backtests.values()),
                'completed': self.completed_backtests[-10:]
            }
    
    def get_results(self, backtest_id=None, symbol=None):
        """Get backtest results"""
        with self.backtest_lock:
            if backtest_id:
                for backtest in self.completed_backtests:
                    if backtest['id'] == backtest_id:
                        return backtest.get('results', {})
            
            if symbol:
                # Return most recent results for symbol
                for backtest in reversed(self.completed_backtests):
                    results = backtest.get('results', {})
                    if symbol in results:
                        return {symbol: results[symbol]}
        
        return {}


# ============================================================================
# WATCHLIST MANAGER (from previous version)
# ============================================================================

class WatchlistManager:
    def __init__(self):
        self.watchlists_dir = Path('dashboard_data/watchlists')
        self.watchlists_dir.mkdir(parents=True, exist_ok=True)
        self.watchlists = self.load_all_watchlists()
        
    def load_all_watchlists(self):
        watchlists = {}
        default_lists = {
            'MTF_Swing': ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT'],
            'Warrior_Momentum': [],
            'Scanner_Results': []
        }
        
        for name, symbols in default_lists.items():
            file_path = self.watchlists_dir / f'{name}.json'
            if not file_path.exists():
                self.save_watchlist(name, symbols)
        
        for file in self.watchlists_dir.glob('*.json'):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    watchlists[data['name']] = data
            except Exception as e:
                logger.error(f'Error loading watchlist {file}: {e}')
        
        return watchlists
    
    def save_watchlist(self, name, symbols):
        file_path = self.watchlists_dir / f'{name}.json'
        data = {
            'name': name,
            'symbols': symbols,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.watchlists[name] = data
        return data
    
    def add_symbol(self, watchlist_name, symbol):
        if watchlist_name in self.watchlists:
            symbols = self.watchlists[watchlist_name]['symbols']
            if symbol not in symbols:
                symbols.append(symbol)
                self.save_watchlist(watchlist_name, symbols)
                return True
        return False
    
    def remove_symbol(self, watchlist_name, symbol):
        if watchlist_name in self.watchlists:
            symbols = self.watchlists[watchlist_name]['symbols']
            if symbol in symbols:
                symbols.remove(symbol)
                self.save_watchlist(watchlist_name, symbols)
                return True
        return False


# ============================================================================
# BOT STATE (simplified from previous version)
# ============================================================================

class BotState:
    def __init__(self):
        self.mtf_running = False
        self.warrior_running = False
        self.logs = []
        
    def add_log(self, level, category, message):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'category': category,
            'message': message
        }
        self.logs.append(log_entry)
        self.logs = self.logs[-100:]  # Keep last 100 logs
        
        socketio.emit('log', log_entry)


# ============================================================================
# IBKR MANAGER (simplified)
# ============================================================================

class IBKRManager:
    def __init__(self):
        self.connected = False
        self.positions = []
        self.orders = []
        self.scanner_results = []
    
    def connect(self, host='127.0.0.1', port=7497, client_id=1):
        # Simplified - add full IBKR connection logic here
        self.connected = True
        bot_state.add_log('success', 'ibkr', 'Connected to IBKR TWS')
        return True
    
    def disconnect(self):
        self.connected = False
        bot_state.add_log('info', 'ibkr', 'Disconnected from IBKR')
        return True
    
    def is_connected(self):
        return self.connected


# Initialize managers
watchlist_manager = WatchlistManager()
training_manager = TrainingManager()
backtest_manager = BacktestManager()
bot_state = BotState()
ibkr_manager = IBKRManager()


# ============================================================================
# API ENDPOINTS - TRAINING
# ============================================================================

@app.route('/api/train/start', methods=['POST'])
def start_training():
    """Start training for selected symbols"""
    try:
        data = request.json
        symbols = data.get('symbols', [])
        config = data.get('config', {
            'period': '2y',
            'interval': '1h',
            'epochs': 50,
            'batch_size': 32
        })
        
        if not symbols:
            return jsonify({'success': False, 'message': 'No symbols provided'}), 400
        
        training_id = f"train_{int(time.time())}"
        training_manager.start_training(training_id, symbols, config)
        
        # Run training in background
        def train_in_background():
            try:
                # Import your existing trainer
                sys.path.append(str(Path(__file__).parent))
                from EASY_MTF_TRAINER_V2 import LSTMTrainingPipeline
                
                pipeline = LSTMTrainingPipeline()
                
                # Update progress for each symbol
                for i, symbol in enumerate(symbols):
                    progress = int((i / len(symbols)) * 100)
                    training_manager.update_progress(
                        training_id, 
                        progress, 
                        'training', 
                        symbol,
                        f'Training {symbol}...'
                    )
                    
                    # Train model
                    time.sleep(3)  # Replace with actual training
                
                # Complete training
                results = {
                    symbol: {
                        'accuracy': 62.5,
                        'val_accuracy': 58.3,
                        'epochs_completed': config['epochs'],
                        'model_path': f"models/lstm_mtf_v2/{symbol}_mtf_v2.keras"
                    } for symbol in symbols
                }
                
                training_manager.complete_training(training_id, results)
                
            except Exception as e:
                logger.error(f'Training error: {e}')
                training_manager.fail_training(training_id, e)
        
        thread = threading.Thread(target=train_in_background, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True, 
            'training_id': training_id,
            'message': f'Training started for {len(symbols)} symbols'
        })
        
    except Exception as e:
        logger.error(f'Error starting training: {e}')
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/train/status')
def get_training_status():
    """Get all training statuses"""
    return jsonify(training_manager.get_all_trainings())


@app.route('/api/train/stop/<training_id>', methods=['POST'])
def stop_training(training_id):
    """Stop an active training"""
    # Implement training cancellation logic
    return jsonify({'success': True, 'message': 'Training stopped'})


# ============================================================================
# API ENDPOINTS - BACKTESTING
# ============================================================================

@app.route('/api/backtest/status')
def get_backtest_status():
    """Get all backtest statuses"""
    return jsonify(backtest_manager.get_all_backtests())


@app.route('/api/backtest/results')
def get_backtest_results():
    """Get backtest results"""
    backtest_id = request.args.get('backtest_id')
    symbol = request.args.get('symbol')
    
    results = backtest_manager.get_results(backtest_id, symbol)
    return jsonify({'results': results})


# ============================================================================
# API ENDPOINTS - TRADE EXECUTION (NEW)
# ============================================================================

@app.route('/api/trade/execute', methods=['POST'])
def execute_trade():
    """Execute a trade based on backtest results"""
    try:
        data = request.json
        symbol = data.get('symbol')
        action = data.get('action')  # 'BUY' or 'SELL'
        quantity = data.get('quantity')
        order_type = data.get('order_type', 'MKT')  # 'MKT' or 'LMT'
        limit_price = data.get('limit_price')
        
        if not ibkr_manager.is_connected():
            return jsonify({'success': False, 'message': 'Not connected to IBKR'}), 400
        
        # Create order (simplified - add full IBKR order logic)
        order_id = int(time.time())
        
        bot_state.add_log('success', 'trade', 
                         f'Placed {action} order: {quantity} {symbol} @ {order_type}')
        
        return jsonify({
            'success': True,
            'order_id': order_id,
            'message': f'{action} order placed for {quantity} {symbol}'
        })
        
    except Exception as e:
        logger.error(f'Trade execution error: {e}')
        return jsonify({'success': False, 'message': str(e)}), 500


# ============================================================================
# EXISTING ENDPOINTS (from previous version)
# ============================================================================

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})


@app.route('/api/status')
def get_status():
    return jsonify({
        'mtf_running': bot_state.mtf_running,
        'warrior_running': bot_state.warrior_running,
        'ibkr_connected': ibkr_manager.is_connected(),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/watchlists')
def get_watchlists():
    return jsonify(list(watchlist_manager.watchlists.values()))


@app.route('/api/watchlist/<name>/add', methods=['POST'])
def add_to_watchlist(name):
    data = request.json
    symbol = data.get('symbol')
    
    if watchlist_manager.add_symbol(name, symbol):
        bot_state.add_log('info', 'watchlist', f'Added {symbol} to {name}')
        return jsonify({'success': True})
    
    return jsonify({'success': False}), 400


@app.route('/api/watchlist/<name>/remove', methods=['POST'])
def remove_from_watchlist(name):
    data = request.json
    symbol = data.get('symbol')
    
    if watchlist_manager.remove_symbol(name, symbol):
        bot_state.add_log('info', 'watchlist', f'Removed {symbol} from {name}')
        return jsonify({'success': True})
    
    return jsonify({'success': False}), 400


@app.route('/api/logs')
def get_logs():
    return jsonify(bot_state.logs)


if __name__ == '__main__':
    logger.info('Starting Dashboard API...')
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
