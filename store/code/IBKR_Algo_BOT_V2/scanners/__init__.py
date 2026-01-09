"""
Warrior Trading Scanner Implementation
======================================
Bot-native discovery scanners using Schwab-only market data.

Scanners identify WHAT stocks are in play.
They do NOT determine WHEN to trade or place orders.

Architecture:
- gap_scanner.py   -> Pre-market gappers (04:00-09:15 ET)
- gainer_scanner.py -> Top % gainers (07:00-09:30 ET)
- hod_scanner.py   -> High of Day momentum (09:15+ ET)

All output feeds into MomentumWatchlist for ranking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class ScannerType(Enum):
    GAPPER = "GAPPER"
    GAINER = "GAINER"
    HOD = "HOD"


@dataclass
class ScannerResult:
    """Standard output format for all scanners"""

    symbol: str
    scanner: ScannerType
    price: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Scanner-specific fields
    gap_pct: Optional[float] = None  # GAPPER
    premarket_volume: Optional[int] = None  # GAPPER
    pct_change: Optional[float] = None  # GAINER
    rel_vol: Optional[float] = None  # GAINER, HOD
    volume: Optional[int] = None  # GAINER, HOD
    day_high: Optional[float] = None  # HOD

    def to_dict(self) -> Dict:
        """Convert to dictionary for API output"""
        result = {
            "symbol": self.symbol,
            "scanner": self.scanner.value,
            "price": self.price,
            "timestamp": self.timestamp,
        }
        # Add non-None optional fields
        if self.gap_pct is not None:
            result["gap_pct"] = self.gap_pct
        if self.premarket_volume is not None:
            result["premarket_volume"] = self.premarket_volume
        if self.pct_change is not None:
            result["pct_change"] = self.pct_change
        if self.rel_vol is not None:
            result["rel_vol"] = self.rel_vol
        if self.volume is not None:
            result["volume"] = self.volume
        if self.day_high is not None:
            result["day_high"] = self.day_high
        return result


@dataclass
class ScannerConfig:
    """Common configuration for all scanners"""

    # Price filters
    min_price: float = 2.00
    max_price: float = 20.00
    max_spread_pct: float = 0.015  # 1.5% max spread

    # Gap scanner
    gap_min_pct: float = 0.04  # 4% minimum gap
    gap_min_volume: int = 200_000  # Pre-market volume

    # Gainer scanner
    gainer_min_pct: float = 0.05  # 5% minimum change
    gainer_min_rel_vol: float = 2.5
    gainer_min_volume: int = 750_000

    # HOD scanner
    hod_min_rel_vol: float = 2.0
    hod_min_volume: int = 500_000


# Global config instance
_config: Optional[ScannerConfig] = None


def get_scanner_config() -> ScannerConfig:
    """Get the scanner configuration"""
    global _config
    if _config is None:
        _config = ScannerConfig()
    return _config


def set_scanner_config(config: ScannerConfig):
    """Update scanner configuration"""
    global _config
    _config = config


# Imports moved to scanner_coordinator.py to avoid circular imports
# Use: from scanners.gap_scanner import get_gap_scanner
# Use: from scanners.gainer_scanner import get_gainer_scanner
# Use: from scanners.hod_scanner import get_hod_scanner
# Use: from scanners.scanner_coordinator import get_scanner_coordinator
