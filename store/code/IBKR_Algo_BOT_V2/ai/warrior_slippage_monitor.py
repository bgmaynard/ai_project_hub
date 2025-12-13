# Warrior Slippage Monitor
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)

class SlippageLevel(Enum):
    ACCEPTABLE = "acceptable"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class OrderExecution:
    symbol: str
    side: str
    expected_price: float
    actual_price: float
    shares: int
    timestamp: datetime
    slippage_pct: float
    slippage_level: SlippageLevel

class WarriorSlippageMonitor:
    def __init__(self, max_acceptable=0.001, max_warning=0.0025):
        self.max_acceptable = max_acceptable
        self.max_warning = max_warning
        self.executions = deque(maxlen=1000)
        
    def record_execution(self, symbol, side, expected, actual, shares):
        if side.lower() == 'buy':
            slippage = (actual - expected) / expected
        else:
            slippage = (expected - actual) / expected
            
        abs_slip = abs(slippage)
        if abs_slip <= self.max_acceptable:
            level = SlippageLevel.ACCEPTABLE
        elif abs_slip <= self.max_warning:
            level = SlippageLevel.WARNING
        else:
            level = SlippageLevel.CRITICAL
            logger.warning(f"CRITICAL SLIPPAGE: {symbol} {side} {slippage:.2%}")
            
        exec_obj = OrderExecution(symbol, side, expected, actual, shares, datetime.now(), slippage, level)
        self.executions.append(exec_obj)
        return exec_obj
    
    def get_stats(self, symbol=None):
        execs = [e for e in self.executions if not symbol or e.symbol == symbol]
        if not execs:
            return {"total": 0}
        return {
            "total": len(execs),
            "avg_slippage": sum(abs(e.slippage_pct) for e in execs) / len(execs),
            "critical_count": sum(1 for e in execs if e.slippage_level == SlippageLevel.CRITICAL)
        }

_monitor = None
def get_slippage_monitor(**kwargs):
    global _monitor
    if not _monitor:
        _monitor = WarriorSlippageMonitor(**kwargs)
    return _monitor
