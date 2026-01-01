"""
TOP GAINER SCANNER
==================
Warrior Trading: "Top % Gainers"

Purpose: Identify stocks already moving today and attracting momentum traders.

Active: 07:00 - 09:30 ET

Required Data:
- Last trade price
- Prior close
- Intraday volume
- Relative volume

Conditions:
- pct_change >= 5%
- rel_vol >= 2.5
- volume_today >= 750,000
- 2 <= price <= 20
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional

import pytz

logger = logging.getLogger(__name__)

from scanners import ScannerResult, ScannerType, get_scanner_config


@dataclass
class GainerCandidate:
    """Internal representation of a gainer candidate"""

    symbol: str
    last_price: float
    prior_close: float
    pct_change: float
    volume: int
    avg_volume: int
    rel_vol: float
    bid: float = 0.0
    ask: float = 0.0
    spread_pct: float = 0.0


class GainerScanner:
    """
    Top percentage gainer scanner using Schwab data only.

    Scans for stocks with significant gains and high relative volume.
    """

    def __init__(self):
        self._candidates: Dict[str, GainerCandidate] = {}
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
            logger.error(f"[GainerScanner] Health check failed: {e}")
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
                "volume": quote.get("volume", 0),
                "close": quote.get("close", 0),  # Prior close
                "change_percent": quote.get("change_percent", 0),
            }

        except Exception as e:
            logger.debug(f"[GainerScanner] Error getting quote for {symbol}: {e}")
            return None

    def _get_average_volume(self, symbol: str) -> int:
        """
        Get average daily volume for relative volume calculation.

        Uses cached value if available, otherwise fetches from Schwab.
        """
        if symbol in self._avg_volumes:
            return self._avg_volumes[symbol]

        try:
            from schwab_market_data import get_schwab_market_data

            schwab = get_schwab_market_data()
            if not schwab:
                return 0

            # Get 20-day daily bars for average volume
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

            # Calculate average volume (exclude today)
            volumes = [c.get("volume", 0) for c in candles[:-1]]
            avg_vol = int(sum(volumes) / len(volumes)) if volumes else 0

            # Cache it
            self._avg_volumes[symbol] = avg_vol
            return avg_vol

        except Exception as e:
            logger.debug(f"[GainerScanner] Error getting avg volume for {symbol}: {e}")
            return 0

    def _calculate_rel_vol(self, current_volume: int, avg_volume: int) -> float:
        """
        Calculate relative volume.

        rel_vol = current_volume / avg_volume_for_time_of_day
        """
        if avg_volume <= 0:
            return 0.0

        # Adjust for time of day (simplified)
        now_et = datetime.now(self._et_tz)
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

        if now_et < market_open:
            # Pre-market: compare to fraction of daily volume
            # Assume ~10% of daily volume trades pre-market
            expected = avg_volume * 0.10
        else:
            # During market hours: linear interpolation
            minutes_since_open = (now_et - market_open).total_seconds() / 60
            total_market_minutes = 390  # 6.5 hours
            fraction = min(minutes_since_open / total_market_minutes, 1.0)
            expected = avg_volume * fraction

        if expected <= 0:
            return 0.0

        return current_volume / expected

    def scan_symbol(self, symbol: str) -> Optional[ScannerResult]:
        """
        Scan a single symbol for gainer criteria.

        Returns ScannerResult if symbol qualifies, None otherwise.
        """
        config = get_scanner_config()

        # Get quote data
        quote = self._get_quote_data(symbol)
        if not quote:
            return None

        last_price = quote["last"]
        prior_close = quote["close"]
        volume = quote["volume"]
        bid = quote["bid"]
        ask = quote["ask"]

        # Fail closed - no partial data
        if last_price <= 0 or prior_close <= 0 or bid <= 0 or ask <= 0:
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

        # Calculate percentage change
        pct_change = (last_price - prior_close) / prior_close

        # Check minimum gain
        if pct_change < config.gainer_min_pct:
            return None

        # Check volume
        if volume < config.gainer_min_volume:
            return None

        # Calculate relative volume
        avg_volume = self._get_average_volume(symbol)
        rel_vol = self._calculate_rel_vol(volume, avg_volume)

        # Check relative volume
        if rel_vol < config.gainer_min_rel_vol:
            return None

        # Store candidate
        self._candidates[symbol] = GainerCandidate(
            symbol=symbol,
            last_price=last_price,
            prior_close=prior_close,
            pct_change=pct_change,
            volume=volume,
            avg_volume=avg_volume,
            rel_vol=rel_vol,
            bid=bid,
            ask=ask,
            spread_pct=spread_pct,
        )

        # Return result
        return ScannerResult(
            symbol=symbol,
            scanner=ScannerType.GAINER,
            price=last_price,
            pct_change=round(pct_change * 100, 2),  # As percentage
            rel_vol=round(rel_vol, 2),
            volume=volume,
        )

    def scan_symbols(self, symbols: List[str]) -> List[ScannerResult]:
        """
        Scan multiple symbols for gainer criteria.

        Returns list of qualifying ScannerResults.
        """
        # Fail closed if not healthy
        if not self._check_schwab_health():
            logger.warning("[GainerScanner] Schwab unhealthy - returning empty results")
            return []

        # Check time window
        if not self.is_active_window():
            logger.debug("[GainerScanner] Outside active window (07:00-09:30 ET)")
            return []

        results = []
        for symbol in symbols:
            try:
                result = self.scan_symbol(symbol)
                if result:
                    results.append(result)
            except Exception as e:
                logger.debug(f"[GainerScanner] Error scanning {symbol}: {e}")
                continue

        self._last_scan = datetime.now()
        logger.info(
            f"[GainerScanner] Scanned {len(symbols)} symbols, found {len(results)} gainers"
        )

        return results

    def get_candidates(self) -> List[ScannerResult]:
        """Get all current gainer candidates as ScannerResults"""
        return [
            ScannerResult(
                symbol=c.symbol,
                scanner=ScannerType.GAINER,
                price=c.last_price,
                pct_change=round(c.pct_change * 100, 2),
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
            "scanner": "GAINER",
            "active": self.is_active_window(),
            "candidates": len(self._candidates),
            "volume_cache_size": len(self._avg_volumes),
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
            "schwab_healthy": self._check_schwab_health(),
        }


# Singleton
_gainer_scanner: Optional[GainerScanner] = None


def get_gainer_scanner() -> GainerScanner:
    """Get the gainer scanner instance"""
    global _gainer_scanner
    if _gainer_scanner is None:
        _gainer_scanner = GainerScanner()
    return _gainer_scanner
