"""
HIGH-OF-DAY (HOD) SCANNER
=========================
Warrior Trading: "HOD Momentum"

Purpose: Detect active breakout leaders making or holding new highs.

Active: 09:15 ET onwards

Required Data:
- Intraday high
- Last trade
- Volume
- Relative volume
- Spread

Conditions:
- last_price >= day_high (at or making new HOD)
- rel_vol >= session floor
- volume >= min_volume
- spread <= max_spread
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional

import pytz

logger = logging.getLogger(__name__)

from scanners import ScannerResult, ScannerType, get_scanner_config


@dataclass
class HODCandidate:
    """Internal representation of an HOD candidate"""

    symbol: str
    last_price: float
    day_high: float
    volume: int
    avg_volume: int
    rel_vol: float
    bid: float = 0.0
    ask: float = 0.0
    spread_pct: float = 0.0
    at_hod: bool = False  # True if price == day_high


class HODScanner:
    """
    High-of-Day momentum scanner using Schwab data only.

    Scans for stocks at or near their intraday highs with strong volume.
    """

    def __init__(self):
        self._candidates: Dict[str, HODCandidate] = {}
        self._last_scan: Optional[datetime] = None
        self._avg_volumes: Dict[str, int] = {}  # Cache for avg volumes
        self._et_tz = pytz.timezone("US/Eastern")

    def is_active_window(self) -> bool:
        """Check if scanner should be active (04:00 - 16:00 ET)"""
        now_et = datetime.now(self._et_tz).time()
        return time(4, 0) <= now_et <= time(16, 0)

    def _check_schwab_health(self) -> bool:
        """Verify Schwab feed is healthy - fail closed"""
        try:
            from schwab_market_data import (get_token_status,
                                            is_schwab_available)

            if not is_schwab_available():
                return False
            status = get_token_status()
            return status.get("valid", False)
        except Exception as e:
            logger.error(f"[HODScanner] Health check failed: {e}")
            return False

    def _get_quote_data(self, symbol: str) -> Optional[Dict]:
        """Get current quote from Schwab"""
        try:
            from schwab_market_data import get_schwab_market_data

            schwab = get_schwab_market_data()
            if not schwab:
                return None

            quote = schwab.get_quote(symbol)
            if not quote:
                return None

            return {
                "last": quote.get("last", 0),
                "bid": quote.get("bid", 0),
                "ask": quote.get("ask", 0),
                "high": quote.get("high", 0),
                "volume": quote.get("volume", 0),
            }

        except Exception as e:
            logger.debug(f"[HODScanner] Error getting quote for {symbol}: {e}")
            return None

    def _get_average_volume(self, symbol: str) -> int:
        """Get average daily volume for relative volume calculation."""
        if symbol in self._avg_volumes:
            return self._avg_volumes[symbol]

        try:
            from schwab_market_data import get_schwab_market_data

            schwab = get_schwab_market_data()
            if not schwab:
                return 0

            history = schwab.get_price_history(
                symbol,
                period_type="month",
                period=1,
                frequency_type="daily",
                frequency=1,
            )

            if not history or "candles" not in history:
                return 0

            candles = history["candles"]
            if len(candles) < 5:
                return 0

            volumes = [c.get("volume", 0) for c in candles[:-1]]
            avg_vol = int(sum(volumes) / len(volumes)) if volumes else 0

            self._avg_volumes[symbol] = avg_vol
            return avg_vol

        except Exception as e:
            logger.debug(f"[HODScanner] Error getting avg volume for {symbol}: {e}")
            return 0

    def _calculate_rel_vol(self, current_volume: int, avg_volume: int) -> float:
        """Calculate relative volume adjusted for time of day."""
        if avg_volume <= 0:
            return 0.0

        now_et = datetime.now(self._et_tz)
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

        if now_et < market_open:
            expected = avg_volume * 0.10
        else:
            minutes_since_open = (now_et - market_open).total_seconds() / 60
            total_market_minutes = 390
            fraction = min(minutes_since_open / total_market_minutes, 1.0)
            expected = avg_volume * fraction

        if expected <= 0:
            return 0.0

        return current_volume / expected

    def scan_symbol(self, symbol: str) -> Optional[ScannerResult]:
        """
        Scan a single symbol for HOD criteria.

        Returns ScannerResult if symbol qualifies, None otherwise.
        """
        config = get_scanner_config()

        # Get quote data
        quote = self._get_quote_data(symbol)
        if not quote:
            return None

        last_price = quote["last"]
        day_high = quote["high"]
        volume = quote["volume"]
        bid = quote["bid"]
        ask = quote["ask"]

        # Fail closed - no partial data
        if last_price <= 0 or day_high <= 0 or bid <= 0 or ask <= 0:
            return None

        # Check if at high of day (within 0.5% tolerance)
        hod_tolerance = 0.005  # 0.5%
        at_hod = last_price >= day_high * (1 - hod_tolerance)

        if not at_hod:
            return None

        # Calculate spread
        spread = ask - bid
        spread_pct = spread / last_price if last_price > 0 else 1.0

        # Check spread filter
        if spread_pct > config.max_spread_pct:
            return None

        # Check price range
        if not (config.min_price <= last_price <= config.max_price):
            return None

        # Check volume
        if volume < config.hod_min_volume:
            return None

        # Calculate relative volume
        avg_volume = self._get_average_volume(symbol)
        rel_vol = self._calculate_rel_vol(volume, avg_volume)

        # Check relative volume
        if rel_vol < config.hod_min_rel_vol:
            return None

        # Store candidate
        self._candidates[symbol] = HODCandidate(
            symbol=symbol,
            last_price=last_price,
            day_high=day_high,
            volume=volume,
            avg_volume=avg_volume,
            rel_vol=rel_vol,
            bid=bid,
            ask=ask,
            spread_pct=spread_pct,
            at_hod=(last_price >= day_high),
        )

        # Return result
        return ScannerResult(
            symbol=symbol,
            scanner=ScannerType.HOD,
            price=last_price,
            day_high=day_high,
            rel_vol=round(rel_vol, 2),
            volume=volume,
        )

    def scan_symbols(self, symbols: List[str]) -> List[ScannerResult]:
        """
        Scan multiple symbols for HOD criteria.

        Returns list of qualifying ScannerResults.
        """
        # Fail closed if not healthy
        if not self._check_schwab_health():
            logger.warning("[HODScanner] Schwab unhealthy - returning empty results")
            return []

        # Check time window
        if not self.is_active_window():
            logger.debug("[HODScanner] Outside active window (09:15+ ET)")
            return []

        results = []
        for symbol in symbols:
            try:
                result = self.scan_symbol(symbol)
                if result:
                    results.append(result)
            except Exception as e:
                logger.debug(f"[HODScanner] Error scanning {symbol}: {e}")
                continue

        self._last_scan = datetime.now()
        logger.info(
            f"[HODScanner] Scanned {len(symbols)} symbols, found {len(results)} at HOD"
        )

        return results

    def get_candidates(self) -> List[ScannerResult]:
        """Get all current HOD candidates as ScannerResults"""
        return [
            ScannerResult(
                symbol=c.symbol,
                scanner=ScannerType.HOD,
                price=c.last_price,
                day_high=c.day_high,
                rel_vol=round(c.rel_vol, 2),
                volume=c.volume,
            )
            for c in self._candidates.values()
        ]

    def clear(self):
        """Clear all candidates (call at end of session)"""
        self._candidates.clear()

    def clear_volume_cache(self):
        """Clear average volume cache (call daily)"""
        self._avg_volumes.clear()

    def get_status(self) -> Dict:
        """Get scanner status"""
        return {
            "scanner": "HOD",
            "active": self.is_active_window(),
            "candidates": len(self._candidates),
            "volume_cache_size": len(self._avg_volumes),
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
            "schwab_healthy": self._check_schwab_health(),
        }


# Singleton
_hod_scanner: Optional[HODScanner] = None


def get_hod_scanner() -> HODScanner:
    """Get the HOD scanner instance"""
    global _hod_scanner
    if _hod_scanner is None:
        _hod_scanner = HODScanner()
    return _hod_scanner
