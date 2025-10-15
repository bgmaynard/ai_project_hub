"""
Modular Dashboard System with Plugin Architecture
Supports custom layouts, window chains, and TradingView integration
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import threading
from queue import Queue

# ============================================================================
# Data Bus - Shared Communication Architecture
# ============================================================================

class DataBus:
    """Central data bus for modular communication"""
    
    def __init__(self):
        self.subscribers = {}
        self.data_store = {}
        self.lock = threading.Lock()
        
    def publish(self, channel: str, data: Any):
        """Publish data to a channel"""
        with self.lock:
            self.data_store[channel] = data
            
            if channel in self.subscribers:
                for callback in self.subscribers[channel]:
                    try:
                        callback(data)
                    except Exception as e:
                        print(f"Error in subscriber callback: {e}")
    
    def subscribe(self, channel: str, callback):
        """Subscribe to a channel"""
        with self.lock:
            if channel not in self.subscribers:
                self.subscribers[channel] = []
            self.subscribers[channel].append(callback)
    
    def get(self, channel: str):
        """Get latest data from channel"""
        with self.lock:
            return self.data_store.get(channel)
    
    def unsubscribe(self, channel: str, callback):
        """Unsubscribe from channel"""
        with self.lock:
            if channel in self.subscribers and callback in self.subscribers[channel]:
                self.subscribers[channel].remove(callback)


# ============================================================================
# Module Base Class
# ============================================================================

class TradingModule(ABC):
    """Base class for all trading modules (plugins)"""
    
    def __init__(self, name: str, data_bus: DataBus):
        self.name = name
        self.data_bus = data_bus
        self.config = {}
        self.active = False
    
    @abstractmethod
    def initialize(self):
        """Initialize the module"""
        pass
    
    @abstractmethod
    def start(self):
        """Start the module"""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop the module"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict:
        """Get module status"""
        pass
    
    def publish_data(self, channel: str, data: Any):
        """Publish data to bus"""
        self.data_bus.publish(f"{self.name}.{channel}", data)
    
    def subscribe_data(self, channel: str, callback):
        """Subscribe to bus channel"""
        self.data_bus.subscribe(channel, callback)


# ============================================================================
# Specific Trading Modules
# ============================================================================

class IBKRModule(TradingModule):
    """IBKR connection and order management module"""
    
    def __init__(self, data_bus: DataBus):
        super().__init__("ibkr", data_bus)
        self.connection = None
        self.orders = []
        self.positions = {}
    
    def initialize(self):
        print(f"[{self.name}] Initializing IBKR module...")
        # Initialize IBKR connection
        self.config = {
            'host': '127.0.0.1',
            'port': 7497,
            'client_id': 1
        }
    
    def start(self):
        self.active = True
        print(f"[{self.name}] Started")
        
        # Subscribe to order requests
        self.subscribe_data('order_request', self.handle_order_request)
    
    def stop(self):
        self.active = False
        print(f"[{self.name}] Stopped")
    
    def get_status(self):
        return {
            'name': self.name,
            'active': self.active,
            'connected': self.connection is not None,
            'orders': len(self.orders),
            'positions': len(self.positions)
        }
    
    def handle_order_request(self, order_data):
        """Handle incoming order requests"""
        print(f"[{self.name}] Order received: {order_data}")
        self.orders.append(order_data)
        self.publish_data('order_placed', order_data)


class AIStrategyModule(TradingModule):
    """AI strategy and signal generation module"""
    
    def __init__(self, data_bus: DataBus):
        super().__init__("ai_strategy", data_bus)
        self.models = []
        self.signals = Queue()
    
    def initialize(self):
        print(f"[{self.name}] Initializing AI strategy module...")
        self.config = {
            'strategy_type': 'ensemble',
            'confidence_threshold': 0.6
        }
    
    def start(self):
        self.active = True
        print(f"[{self.name}] Started")
        
        # Subscribe to market data
        self.subscribe_data('ibkr.market_data', self.process_market_data)
    
    def stop(self):
        self.active = False
        print(f"[{self.name}] Stopped")
    
    def get_status(self):
        return {
            'name': self.name,
            'active': self.active,
            'models_loaded': len(self.models),
            'pending_signals': self.signals.qsize()
        }
    
    def process_market_data(self, market_data):
        """Process market data and generate signals"""
        # Generate AI signal
        signal = {
            'symbol': market_data.get('symbol'),
            'direction': 'BUY',
            'confidence': 0.75,
            'strategy': 'LSTM+MACD'
        }
        
        self.publish_data('signal_generated', signal)
        
        # Publish order request if confidence threshold met
        if signal['confidence'] >= self.config['confidence_threshold']:
            self.data_bus.publish('order_request', {
                'symbol': signal['symbol'],
                'action': signal['direction'],
                'quantity': 100,
                'source': 'ai_strategy'
            })


class RiskManagementModule(TradingModule):
    """Risk management and position monitoring module"""
    
    def __init__(self, data_bus: DataBus):
        super().__init__("risk_mgmt", data_bus)
        self.daily_pnl = 0
        self.max_loss = -2000
        self.position_limits = {}
    
    def initialize(self):
        print(f"[{self.name}] Initializing risk management module...")
        self.config = {
            'max_position_size': 10000,
            'daily_loss_limit': 2000,
            'max_positions': 10
        }
    
    def start(self):
        self.active = True
        print(f"[{self.name}] Started")
        
        # Subscribe to order requests for validation
        self.subscribe_data('order_request', self.validate_order)
        self.subscribe_data('ibkr.position_update', self.monitor_positions)
    
    def stop(self):
        self.active = False
        print(f"[{self.name}] Stopped")
    
    def get_status(self):
        return {
            'name': self.name,
            'active': self.active,
            'daily_pnl': self.daily_pnl,
            'risk_level': 'LOW' if self.daily_pnl > -500 else 'MEDIUM' if self.daily_pnl > -1500 else 'HIGH'
        }
    
    def validate_order(self, order_data):
        """Validate order against risk rules"""
        if self.daily_pnl <= self.max_loss:
            print(f"[{self.name}] Order blocked: Daily loss limit reached")
            self.publish_data('order_blocked', order_data)
            return False
        
        print(f"[{self.name}] Order validated: {order_data}")
        self.publish_data('order_validated', order_data)
        return True
    
    def monitor_positions(self, position_data):
        """Monitor open positions"""
        self.publish_data('risk_update', {
            'daily_pnl': self.daily_pnl,
            'positions': position_data
        })


class WarriorTradingModule(TradingModule):
    """Warrior Trading methodology implementation"""
    
    def __init__(self, data_bus: DataBus):
        super().__init__("warrior_trading", data_bus)
        self.momentum_stocks = []
    
    def initialize(self):
        print(f"[{self.name}] Initializing Warrior Trading module...")
        self.config = {
            'min_volume': 1000000,
            'min_price': 2,
            'max_price': 20,
            'gap_percentage': 3.0,
            'strategies': ['bull_flag', 'vwap_hold', 'reversal']
        }
    
    def start(self):
        self.active = True
        print(f"[{self.name}] Started - Scanning for momentum setups")
        
        # Subscribe to market data
        self.subscribe_data('ibkr.market_data', self.scan_momentum)
    
    def stop(self):
        self.active = False
        print(f"[{self.name}] Stopped")
    
    def get_status(self):
        return {
            'name': self.name,
            'active': self.active,
            'momentum_stocks': len(self.momentum_stocks),
            'strategies_active': self.config['strategies']
        }
    
    def scan_momentum(self, market_data):
        """Scan for momentum trading opportunities"""
        symbol = market_data.get('symbol')
        price = market_data.get('last', 0)
        volume = market_data.get('volume', 0)
        
        # Check momentum criteria
        if (self.config['min_price'] <= price <= self.config['max_price'] and 
            volume >= self.config['min_volume']):
            
            # Bull flag pattern detection (simplified)
            signal = {
                'symbol': symbol,
                'pattern': 'bull_flag',
                'entry': price,
                'stop': price * 0.97,
                'target': price * 1.05
            }
            
            self.publish_data('momentum_signal', signal)
            self.momentum_stocks.append(symbol)


class TradingViewModule(TradingModule):
    """TradingView chart and indicator integration"""
    
    def __init__(self, data_bus: DataBus):
        super().__init__("tradingview", data_bus)
        self.active_charts = {}
        self.indicators = []
    
    def initialize(self):
        print(f"[{self.name}] Initializing TradingView module...")
        self.config = {
            'default_indicators': ['MACD', 'RSI', 'VWAP', 'EMA9', 'EMA20'],
            'chart_theme': 'dark',
            'timeframe': '5min'
        }
    
    def start(self):
        self.active = True
        print(f"[{self.name}] Started")
        
        # Subscribe to symbol selection changes
        self.subscribe_data('symbol_selected', self.load_chart)
    
    def stop(self):
        self.active = False
        print(f"[{self.name}] Stopped")
    
    def get_status(self):
        return {
            'name': self.name,
            'active': self.active,
            'active_charts': len(self.active_charts),
            'indicators_loaded': len(self.config['default_indicators'])
        }
    
    def load_chart(self, symbol):
        """Load TradingView chart for symbol"""
        self.active_charts[symbol] = {
            'symbol': symbol,
            'indicators': self.config['default_indicators'],
            'timeframe': self.config['timeframe']
        }
        
        print(f"[{self.name}] Chart loaded for {symbol}")
        self.publish_data('chart_loaded', self.active_charts[symbol])


# ============================================================================
# Dashboard Layout System
# ============================================================================

class DashboardLayout:
    """Manages dashboard layout and window chains"""
    
    def __init__(self):
        self.layouts = {}
        self.window_chains = {}
        self.active_layout = None
    
    def create_layout(self, name: str, config: Dict):
        """Create a new dashboard layout"""
        self.layouts[name] = {
            'name': name,
            'grid': config.get('grid', {'rows': 2, 'cols': 3}),
            'widgets': config.get('widgets', []),
            'window_chain': config.get('window_chain', None)
        }
        
        print(f"Layout '{name}' created")
    
    def create_window_chain(self, chain_name: str, windows: List[str]):
        """Create a window chain that shares symbol selection"""
        self.window_chains[chain_name] = {
            'windows': windows,
            'current_symbol': None
        }
        
        print(f"Window chain '{chain_name}' created with windows: {windows}")
    
    def sync_symbol(self, chain_name: str, symbol: str):
        """Synchronize symbol across window chain"""
        if chain_name in self.window_chains:
            self.window_chains[chain_name]['current_symbol'] = symbol
            print(f"Symbol '{symbol}' synced across chain '{chain_name}'")
    
    def get_layout_config(self, name: str):
        """Get layout configuration"""
        return self.layouts.get(name)
    
    def set_active_layout(self, name: str):
        """Set active layout"""
        if name in self.layouts:
            self.active_layout = name
            print(f"Active layout set to '{name}'")
        else:
            print(f"Layout '{name}' not found")
    
    def export_layout(self, name: str, filepath: str):
        """Export layout to JSON file"""
        if name in self.layouts:
            with open(filepath, 'w') as f:
                json.dump(self.layouts[name], f, indent=2)
            print(f"Layout '{name}' exported to {filepath}")
    
    def import_layout(self, filepath: str):
        """Import layout from JSON file"""
        with open(filepath, 'r') as f:
            layout_config = json.load(f)
            name = layout_config['name']
            self.layouts[name] = layout_config
            print(f"Layout '{name}' imported from {filepath}")


# ============================================================================
# Module Manager
# ============================================================================

class ModuleManager:
    """Manages all trading modules"""
    
    def __init__(self):
        self.data_bus = DataBus()
        self.modules = {}
    
    def register_module(self, module: TradingModule):
        """Register a new module"""
        module.initialize()
        self.modules[module.name] = module
        print(f"Module '{module.name}' registered")
    
    def start_module(self, name: str):
        """Start a specific module"""
        if name in self.modules:
            self.modules[name].start()
        else:
            print(f"Module '{name}' not found")
    
    def stop_module(self, name: str):
        """Stop a specific module"""
        if name in self.modules:
            self.modules[name].stop()
        else:
            print(f"Module '{name}' not found")
    
    def start_all(self):
        """Start all registered modules"""
        for module in self.modules.values():
            module.start()
        print("All modules started")
    
    def stop_all(self):
        """Stop all modules"""
        for module in self.modules.values():
            module.stop()
        print("All modules stopped")
    
    def get_status_all(self):
        """Get status of all modules"""
        return {name: module.get_status() for name, module in self.modules.items()}
    
    def get_data_bus(self):
        """Get the shared data bus"""
        return self.data_bus


# ============================================================================
# Main System Configuration
# ============================================================================

if __name__ == "__main__":
    print("=== AI Trading Bot - Modular System ===\n")
    
    # Initialize module manager
    manager = ModuleManager()
    
    # Register modules
    manager.register_module(IBKRModule(manager.get_data_bus()))
    manager.register_module(AIStrategyModule(manager.get_data_bus()))
    manager.register_module(RiskManagementModule(manager.get_data_bus()))
    manager.register_module(WarriorTradingModule(manager.get_data_bus()))
    manager.register_module(TradingViewModule(manager.get_data_bus()))
    
    # Create dashboard layouts
    dashboard = DashboardLayout()
    
    # Layout 1: Trading Dashboard
    dashboard.create_layout('trading_main', {
        'grid': {'rows': 2, 'cols': 3},
        'widgets': [
            {'type': 'chart', 'position': [0, 0, 2, 2]},
            {'type': 'watchlist', 'position': [0, 2, 1, 1]},
            {'type': 'positions', 'position': [1, 2, 1, 1]},
            {'type': 'orders', 'position': [2, 0, 1, 2]},
            {'type': 'account', 'position': [2, 2, 1, 1]}
        ],
        'window_chain': 'main_chain'
    })
    
    # Layout 2: Multi-Monitor Setup
    dashboard.create_layout('multi_monitor', {
        'grid': {'rows': 3, 'cols': 4},
        'widgets': [
            {'type': 'chart', 'position': [0, 0, 2, 3]},
            {'type': 'level2', 'position': [0, 3, 2, 1]},
            {'type': 'time_sales', 'position': [2, 0, 1, 1]},
            {'type': 'watchlist', 'position': [2, 1, 1, 1]},
            {'type': 'ai_signals', 'position': [2, 2, 1, 2]}
        ],
        'window_chain': 'advanced_chain'
    })
    
    # Create window chains
    dashboard.create_window_chain('main_chain', ['chart', 'level2', 'time_sales'])
    dashboard.create_window_chain('advanced_chain', ['chart', 'level2', 'time_sales', 'options'])
    
    # Set active layout
    dashboard.set_active_layout('trading_main')
    
    # Start all modules
    manager.start_all()
    
    # Simulate symbol selection
    print("\n=== Testing Window Chain ===")
    manager.get_data_bus().publish('symbol_selected', 'AAPL')
    dashboard.sync_symbol('main_chain', 'AAPL')
    
    # Simulate market data
    print("\n=== Simulating Market Data ===")
    manager.get_data_bus().publish('ibkr.market_data', {
        'symbol': 'AAPL',
        'last': 178.50,
        'bid': 178.48,
        'ask': 178.52,
        'volume': 2500000
    })
    
    # Get system status
    print("\n=== System Status ===")
    status = manager.get_status_all()
    for module_name, module_status in status.items():
        print(f"{module_name}: {module_status}")
    
    # Export layout
    dashboard.export_layout('trading_main', 'trading_layout.json')
    
    print("\n=== Modular Trading System Ready ===")
    print("Available modules:", list(manager.modules.keys()))
    print("Available layouts:", list(dashboard.layouts.keys()))
    print("Window chains:", list(dashboard.window_chains.keys()))