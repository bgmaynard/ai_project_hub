"""
RelVol Resolver with Fallback Chain
====================================
Resolves average volume for relative volume calculation.

Problem: Discovery can fail/over-exclude when avgVolume missing.

Solution: Fallback chain:
1. Schwab (primary)
2. yfinance (cached 1/day)
3. Mark as RELVOL_UNKNOWN (don't exclude, route as degraded)

Symbols with RELVOL_UNKNOWN are NOT excluded at R1 - they are
routed as "degraded candidates" with provenance metadata.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
import pytz

logger = logging.getLogger(__name__)


@dataclass
class VolumeData:
    """Average volume data with provenance"""
    symbol: str
    avg_volume: Optional[int]
    current_volume: int
    rel_vol: Optional[float]
    source: str  # "schwab", "yfinance", "UNKNOWN"
    is_degraded: bool
    cached_at: Optional[str] = None
    error: Optional[str] = None


class RelVolResolver:
    """
    Resolves relative volume with fallback chain.

    Ensures symbols are not excluded just because avgVolume is missing.
    """

    def __init__(self, cache_file: str = "ai/avg_volume_cache.json"):
        self.cache_file = Path(cache_file)
        self.et_tz = pytz.timezone('US/Eastern')

        # In-memory cache (persisted to file)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl_hours = 24  # Cache avgVolume for 24 hours

        # Load cache from file
        self._load_cache()

        # Stats
        self.resolve_count = 0
        self.schwab_hits = 0
        self.yfinance_hits = 0
        self.cache_hits = 0
        self.unknown_count = 0

        logger.info("RelVolResolver initialized")

    def _load_cache(self):
        """Load avgVolume cache from file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached avgVolume entries")
            except Exception as e:
                logger.warning(f"Failed to load avgVolume cache: {e}")
                self._cache = {}

    def _save_cache(self):
        """Save avgVolume cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning(f"Failed to save avgVolume cache: {e}")

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached avgVolume is still valid"""
        if symbol not in self._cache:
            return False

        cached = self._cache[symbol]
        cached_at = datetime.fromisoformat(cached.get("cached_at", "2020-01-01"))
        now = datetime.now(self.et_tz).replace(tzinfo=None)

        return (now - cached_at) < timedelta(hours=self._cache_ttl_hours)

    def _get_from_schwab(self, symbol: str) -> Optional[int]:
        """Try to get avgVolume from Schwab"""
        try:
            from schwab_market_data import get_schwab_market_data

            market_data = get_schwab_market_data()
            if market_data and market_data.available:
                quote = market_data.get_quote(symbol)
                if quote:
                    avg_vol = quote.get("averageDailyVolume") or quote.get("avgVol10Day")
                    if avg_vol and avg_vol > 0:
                        return int(avg_vol)
        except Exception as e:
            logger.debug(f"Schwab avgVolume failed for {symbol}: {e}")

        return None

    def _get_from_yfinance(self, symbol: str) -> Optional[int]:
        """Try to get avgVolume from yfinance"""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Try different field names
            avg_vol = info.get("averageVolume") or info.get("averageDailyVolume10Day") or info.get("averageVolume10days")

            if avg_vol and avg_vol > 0:
                return int(avg_vol)

        except Exception as e:
            logger.debug(f"yfinance avgVolume failed for {symbol}: {e}")

        return None

    def resolve(self, symbol: str, current_volume: int = 0) -> VolumeData:
        """
        Resolve avgVolume with fallback chain.

        Args:
            symbol: Stock symbol
            current_volume: Today's volume (for RelVol calculation)

        Returns:
            VolumeData with avgVolume, source, and degraded flag
        """
        self.resolve_count += 1
        now = datetime.now(self.et_tz)

        # Check cache first
        if self._is_cache_valid(symbol):
            cached = self._cache[symbol]
            avg_vol = cached.get("avg_volume")
            source = cached.get("source", "cache")
            self.cache_hits += 1

            rel_vol = (current_volume / avg_vol) if avg_vol and avg_vol > 0 else None

            return VolumeData(
                symbol=symbol,
                avg_volume=avg_vol,
                current_volume=current_volume,
                rel_vol=rel_vol,
                source=f"{source} (cached)",
                is_degraded=False,
                cached_at=cached.get("cached_at")
            )

        # Try Schwab first
        avg_vol = self._get_from_schwab(symbol)
        if avg_vol:
            self.schwab_hits += 1
            self._cache[symbol] = {
                "avg_volume": avg_vol,
                "source": "schwab",
                "cached_at": now.isoformat()
            }
            self._save_cache()

            rel_vol = (current_volume / avg_vol) if avg_vol > 0 else None

            return VolumeData(
                symbol=symbol,
                avg_volume=avg_vol,
                current_volume=current_volume,
                rel_vol=rel_vol,
                source="schwab",
                is_degraded=False,
                cached_at=now.isoformat()
            )

        # Fallback to yfinance
        avg_vol = self._get_from_yfinance(symbol)
        if avg_vol:
            self.yfinance_hits += 1
            self._cache[symbol] = {
                "avg_volume": avg_vol,
                "source": "yfinance",
                "cached_at": now.isoformat()
            }
            self._save_cache()

            rel_vol = (current_volume / avg_vol) if avg_vol > 0 else None

            return VolumeData(
                symbol=symbol,
                avg_volume=avg_vol,
                current_volume=current_volume,
                rel_vol=rel_vol,
                source="yfinance",
                is_degraded=False,
                cached_at=now.isoformat()
            )

        # Mark as UNKNOWN - DO NOT EXCLUDE
        self.unknown_count += 1
        logger.warning(f"avgVolume UNKNOWN for {symbol} - routing as degraded candidate")

        return VolumeData(
            symbol=symbol,
            avg_volume=None,
            current_volume=current_volume,
            rel_vol=None,
            source="UNKNOWN",
            is_degraded=True,
            error="avgVolume not available from Schwab or yfinance"
        )

    def batch_resolve(self, symbols: list, volumes: Dict[str, int] = None) -> Dict[str, VolumeData]:
        """Resolve avgVolume for multiple symbols"""
        volumes = volumes or {}
        results = {}

        for symbol in symbols:
            current_vol = volumes.get(symbol, 0)
            results[symbol] = self.resolve(symbol, current_vol)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get resolver statistics"""
        return {
            "resolve_count": self.resolve_count,
            "schwab_hits": self.schwab_hits,
            "yfinance_hits": self.yfinance_hits,
            "cache_hits": self.cache_hits,
            "unknown_count": self.unknown_count,
            "cache_size": len(self._cache),
            "cache_ttl_hours": self._cache_ttl_hours,
            "hit_rates": {
                "schwab": (self.schwab_hits / self.resolve_count * 100) if self.resolve_count > 0 else 0,
                "yfinance": (self.yfinance_hits / self.resolve_count * 100) if self.resolve_count > 0 else 0,
                "cache": (self.cache_hits / self.resolve_count * 100) if self.resolve_count > 0 else 0,
                "unknown": (self.unknown_count / self.resolve_count * 100) if self.resolve_count > 0 else 0
            }
        }

    def clear_cache(self):
        """Clear the avgVolume cache"""
        self._cache.clear()
        self._save_cache()
        logger.info("avgVolume cache cleared")


# Singleton instance
_resolver: Optional[RelVolResolver] = None


def get_relvol_resolver() -> RelVolResolver:
    """Get or create the RelVol resolver singleton"""
    global _resolver
    if _resolver is None:
        _resolver = RelVolResolver()
    return _resolver


def resolve_avg_volume(symbol: str, current_volume: int = 0) -> VolumeData:
    """Convenience function to resolve avgVolume for a symbol"""
    return get_relvol_resolver().resolve(symbol, current_volume)


def is_relvol_known(symbol: str) -> Tuple[bool, Optional[float]]:
    """
    Quick check if RelVol is known for a symbol.

    Returns:
        (is_known, rel_vol) - is_known is True if avgVolume was resolved
    """
    data = get_relvol_resolver().resolve(symbol, 0)
    return (not data.is_degraded, data.rel_vol)
