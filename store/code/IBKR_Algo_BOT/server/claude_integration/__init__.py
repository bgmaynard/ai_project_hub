"""
Claude AI Integration Module
Provides AI-powered market analysis and trade validation
"""

from .market_analyst import MarketAnalyst, simple_market_check
from .trade_validator import TradeValidator, quick_trade_check

__all__ = [
    'MarketAnalyst',
    'TradeValidator',
    'simple_market_check',
    'quick_trade_check'
]

__version__ = '2.0'
