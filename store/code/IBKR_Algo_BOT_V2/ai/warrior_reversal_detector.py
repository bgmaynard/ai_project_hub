# Warrior Reversal Detector
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class ReversalType(Enum):
    JACKNIFE = "jacknife"
    FAILED_BREAKOUT = "failed_breakout"

class ReversalSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ReversalSignal:
    symbol: str
    reversal_type: ReversalType
    severity: ReversalSeverity
    timestamp: datetime
    current_price: float
    entry_price: float
    recommendation: str

class WarriorReversalDetector:
    def __init__(self, jacknife_threshold=0.015):
        self.jacknife_threshold = jacknife_threshold
        
    def detect_jacknife(self, symbol, current_price, entry_price, recent_prices, direction='long'):
        if len(recent_prices) < 3:
            return None
            
        recent_high = max(recent_prices[-3:])
        recent_low = min(recent_prices[-3:])
        
        if direction == 'long':
            upside = (recent_high - entry_price) / entry_price
            reversal_pct = (recent_high - current_price) / recent_high
            
            if upside > 0.02 and reversal_pct > self.jacknife_threshold:
                if reversal_pct > 0.03:
                    severity = ReversalSeverity.CRITICAL
                else:
                    severity = ReversalSeverity.HIGH
                    
                logger.warning(f"JACKNIFE: {symbol} reversal={reversal_pct:.2%}")
                
                return ReversalSignal(
                    symbol, ReversalType.JACKNIFE, severity, 
                    datetime.now(), current_price, entry_price,
                    'exit' if severity == ReversalSeverity.CRITICAL else 'tighten_stop'
                )
        return None
    
    def should_exit_fast(self, reversal):
        return reversal and reversal.severity == ReversalSeverity.CRITICAL

_detector = None
def get_reversal_detector(**kwargs):
    global _detector
    if not _detector:
        _detector = WarriorReversalDetector(**kwargs)
    return _detector
