"""
TOP GAPPER SCANNER
==================
Warrior Trading: "Top Gappers"

Purpose: Identify stocks gapping up premarket that are likely in play all morning.

Active: 04:00 - 09:15 ET

Required Data:
- Prior day close (daily bar)
- Premarket last price or mid
- Premarket volume

Conditions:
- gap_pct >= 4%
- 2 <= price <= 20
- premarket_volume >= 200,000
"""

import logging
from datetime import datetime, time
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import pytz

logger = logging.getLogger(__name__)

# Import after to avoid circular
from scanners import ScannerResult, ScannerType, get_scanner_config


@dataclass
class GapCandidate:
    """Internal representation of a gap candidate"""
    symbol: str
    prior_close: float
    premarket_price: float
    premarket_volume: int
    gap_pct: float
    bid: float = 0.0
    ask: float = 0.0
    spread_pct: float = 0.0


class GapScanner:
    """
    Pre-market gap scanner using Schwab data only.

    Scans for stocks gapping up in pre-market that meet Warrior criteria.
    """

    def __init__(self):
        self._candidates: Dict[str, GapCandidate] = {}
        self._last_scan: Optional[datetime] = None
        self._schwab_healthy = False
        self._et_tz = pytz.timezone('US/Eastern')

    def is_active_window(self) -> bool:
        """Check if scanner should be active (04:00 - 16:00 ET)"""
        now_et = datetime.now(self._et_tz).time()
        return time(4, 0) <= now_et <= time(16, 0)

    def _check_schwab_health(self) -> bool:
        """Verify Schwab feed is healthy - fail closed"""
        try:
            from schwab_market_data import is_schwab_available, get_token_status
            if not is_schwab_available():
                logger.warning("[GapScanner] Schwab not available - returning empty")
                return False
            status = get_token_status()
            if not status.get("valid"):
                logger.warning("[GapScanner] Schwab token invalid - returning empty")
                return False
            return True
        except Exception as e:
            logger.error(f"[GapScanner] Health check failed: {e}")
            return False

    def _get_prior_close(self, symbol: str) -> Optional[float]:
        """Get prior day close from Schwab daily bars"""
        try:
            from schwab_market_data import get_schwab_market_data
            schwab = get_schwab_market_data()
            if not schwab:
                return None

            # Get daily price history
            history = schwab.get_price_history(
                symbol,
                period_type="day",
                period=2,
                frequency_type="daily",
                frequency=1
            )

            if not history or 'candles' not in history:
                return None

            candles = history['candles']
            if len(candles) < 1:
                return None

            # Prior close is the last complete daily candle
            return float(candles[-1].get('close', 0))

        except Exception as e:
            logger.debug(f"[GapScanner] Error getting prior close for {symbol}: {e}")
            return None

    def _get_premarket_data(self, symbol: str) -> Optional[Dict]:
        """Get current premarket quote from Schwab"""
        try:
            from schwab_market_data import get_schwab_market_data
            schwab = get_schwab_market_data()
            if not schwab:
                return None

            quote = schwab.get_quote(symbol)
            if not quote:
                return None

            return {
                "last": quote.get("last", 0) or quote.get("bid", 0),
                "bid": quote.get("bid", 0),
                "ask": quote.get("ask", 0),
                "volume": quote.get("volume", 0)
            }

        except Exception as e:
            logger.debug(f"[GapScanner] Error getting premarket data for {symbol}: {e}")
            return None

    def scan_symbol(self, symbol: str) -> Optional[ScannerResult]:
        """
        Scan a single symbol for gap criteria.

        Returns ScannerResult if symbol qualifies, None otherwise.
        """
        config = get_scanner_config()

        # Get prior close
        prior_close = self._get_prior_close(symbol)
        if not prior_close or prior_close <= 0:
            return None

        # Get premarket data
        pm_data = self._get_premarket_data(symbol)
        if not pm_data:
            return None

        premarket_price = pm_data["last"]
        premarket_volume = pm_data["volume"]
        bid = pm_data["bid"]
        ask = pm_data["ask"]

        # Fail closed - no partial data
        if premarket_price <= 0 or bid <= 0 or ask <= 0:
            return None

        # Calculate spread
        spread = ask - bid
        spread_pct = spread / premarket_price if premarket_price > 0 else 1.0

        # Check spread filter
        if spread_pct > config.max_spread_pct:
            return None

        # Calculate gap
        gap_pct = (premarket_price - prior_close) / prior_close

        # Check gap criteria
        if gap_pct < config.gap_min_pct:
            return None

        # Check price range
        if not (config.min_price <= premarket_price <= config.max_price):
            return None

        # Check volume
        if premarket_volume < config.gap_min_volume:
            return None

        # Store candidate
        self._candidates[symbol] = GapCandidate(
            symbol=symbol,
            prior_close=prior_close,
            premarket_price=premarket_price,
            premarket_volume=premarket_volume,
            gap_pct=gap_pct,
            bid=bid,
            ask=ask,
            spread_pct=spread_pct
        )

        # Return result
        return ScannerResult(
            symbol=symbol,
            scanner=ScannerType.GAPPER,
            price=premarket_price,
            gap_pct=round(gap_pct * 100, 2),  # As percentage
            premarket_volume=premarket_volume
        )

    def scan_symbols(self, symbols: List[str]) -> List[ScannerResult]:
        """
        Scan multiple symbols for gap criteria.

        Returns list of qualifying ScannerResults.
        """
        # Fail closed if not healthy
        if not self._check_schwab_health():
            logger.warning("[GapScanner] Schwab unhealthy - returning empty results")
            return []

        # Check time window
        if not self.is_active_window():
            logger.debug("[GapScanner] Outside active window (04:00-09:15 ET)")
            return []

        results = []
        for symbol in symbols:
            try:
                result = self.scan_symbol(symbol)
                if result:
                    results.append(result)
            except Exception as e:
                logger.debug(f"[GapScanner] Error scanning {symbol}: {e}")
                continue

        self._last_scan = datetime.now()
        logger.info(f"[GapScanner] Scanned {len(symbols)} symbols, found {len(results)} gappers")

        return results

    def get_candidates(self) -> List[ScannerResult]:
        """Get all current gap candidates as ScannerResults"""
        return [
            ScannerResult(
                symbol=c.symbol,
                scanner=ScannerType.GAPPER,
                price=c.premarket_price,
                gap_pct=round(c.gap_pct * 100, 2),
                premarket_volume=c.premarket_volume
            )
            for c in self._candidates.values()
        ]

    def clear(self):
        """Clear all candidates (call at end of session)"""
        self._candidates.clear()

    def get_status(self) -> Dict:
        """Get scanner status"""
        return {
            "scanner": "GAPPER",
            "active": self.is_active_window(),
            "candidates": len(self._candidates),
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
            "schwab_healthy": self._check_schwab_health()
        }


# Singleton
_gap_scanner: Optional[GapScanner] = None


def get_gap_scanner() -> GapScanner:
    """Get the gap scanner instance"""
    global _gap_scanner
    if _gap_scanner is None:
        _gap_scanner = GapScanner()
    return _gap_scanner
