"""
Finviz Scanner Integration
==========================
Free stock screening using Finviz data.
Provides gap scanners, momentum filters, and pre-market movers.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from finvizfinance.screener.overview import Overview
    from finvizfinance.screener.performance import Performance

    HAS_FINVIZ = True
except ImportError:
    logger.warning("finvizfinance not installed: pip install finvizfinance")
    HAS_FINVIZ = False


@dataclass
class ScanResult:
    symbol: str
    price: float
    change_pct: float
    volume: int
    avg_volume: int = 0
    float_short: float = 0.0
    market_cap: str = ""
    sector: str = ""
    source: str = "finviz"


class FinvizScanner:
    """Scanner using Finviz free data"""

    def __init__(self):
        self._cache: Dict[str, List[ScanResult]] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl = 60  # 1 minute cache

    def _is_cache_valid(self, scan_type: str) -> bool:
        if scan_type not in self._cache_time:
            return False
        elapsed = (datetime.now() - self._cache_time[scan_type]).total_seconds()
        return elapsed < self._cache_ttl

    def get_top_gainers(
        self, min_change: float = 5.0, max_price: float = 20.0, min_volume: int = 500000
    ) -> List[ScanResult]:
        """
        Get top percentage gainers matching scalping criteria

        Args:
            min_change: Minimum % change (default 5%)
            max_price: Maximum price (default $20)
            min_volume: Minimum average volume (default 500K)
        """
        cache_key = f"gainers_{min_change}_{max_price}_{min_volume}"

        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        if not HAS_FINVIZ:
            logger.warning("Finviz not available")
            return []

        try:
            foverview = Overview()

            # Build filter based on change threshold
            if min_change >= 10:
                change_filter = "Up 10%"
            elif min_change >= 5:
                change_filter = "Up 5%"
            else:
                change_filter = "Up"

            # Build price filter (exact Finviz filter values)
            if max_price <= 5:
                price_filter = "Under $5"
            elif max_price <= 10:
                price_filter = "Under $10"
            elif max_price <= 15:
                price_filter = "Under $15"
            elif max_price <= 20:
                price_filter = "Under $20"
            elif max_price <= 30:
                price_filter = "Under $30"
            else:
                price_filter = "Under $50"

            filters = {
                "Change": change_filter,
                "Price": price_filter,
                "Average Volume": "Over 500K" if min_volume >= 500000 else "Over 200K",
            }

            foverview.set_filter(filters_dict=filters)
            df = foverview.screener_view()

            results = []
            for _, row in df.iterrows():
                try:
                    change = row.get("Change", 0)
                    if isinstance(change, str):
                        change = float(change.replace("%", "")) / 100

                    volume = row.get("Volume", 0)
                    if isinstance(volume, str):
                        volume = int(volume.replace(",", ""))

                    price = float(row.get("Price", 0))

                    # Apply our filters
                    if price > max_price:
                        continue
                    if change * 100 < min_change:
                        continue

                    results.append(
                        ScanResult(
                            symbol=row.get("Ticker", ""),
                            price=price,
                            change_pct=change * 100,
                            volume=int(volume),
                            market_cap=str(row.get("Market Cap", "")),
                            sector=str(row.get("Sector", "")),
                        )
                    )
                except Exception as e:
                    logger.debug(f"Error parsing row: {e}")
                    continue

            # Sort by change percentage
            results.sort(key=lambda x: x.change_pct, reverse=True)

            # Cache results
            self._cache[cache_key] = results
            self._cache_time[cache_key] = datetime.now()

            logger.info(f"Finviz scan found {len(results)} gainers")
            return results

        except Exception as e:
            logger.error(f"Finviz scan error: {e}")
            return []

    def get_low_float_movers(self, max_float: float = 20.0) -> List[ScanResult]:
        """Get low float stocks with momentum"""
        cache_key = f"low_float_{max_float}"

        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        if not HAS_FINVIZ:
            return []

        try:
            foverview = Overview()

            filters = {
                "Change": "Up 5%",
                "Price": "Under $20",
                "Float Short": "Low (<10%)",
                "Average Volume": "Over 200K",
            }

            foverview.set_filter(filters_dict=filters)
            df = foverview.screener_view()

            results = []
            for _, row in df.iterrows():
                try:
                    change = row.get("Change", 0)
                    if isinstance(change, str):
                        change = float(change.replace("%", "")) / 100

                    results.append(
                        ScanResult(
                            symbol=row.get("Ticker", ""),
                            price=float(row.get("Price", 0)),
                            change_pct=change * 100,
                            volume=int(str(row.get("Volume", 0)).replace(",", "")),
                            market_cap=str(row.get("Market Cap", "")),
                            sector=str(row.get("Sector", "")),
                        )
                    )
                except:
                    continue

            results.sort(key=lambda x: x.change_pct, reverse=True)
            self._cache[cache_key] = results
            self._cache_time[cache_key] = datetime.now()

            return results

        except Exception as e:
            logger.error(f"Finviz low float scan error: {e}")
            return []

    def get_high_volume_breakouts(self) -> List[ScanResult]:
        """Get stocks with unusual volume (potential breakouts)"""
        cache_key = "high_volume"

        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        if not HAS_FINVIZ:
            return []

        try:
            foverview = Overview()

            filters = {
                "Change": "Up",
                "Price": "Under $20",
                "Relative Volume": "Over 3",  # 3x normal volume
                "Average Volume": "Over 200K",
            }

            foverview.set_filter(filters_dict=filters)
            df = foverview.screener_view()

            results = []
            for _, row in df.iterrows():
                try:
                    change = row.get("Change", 0)
                    if isinstance(change, str):
                        change = float(change.replace("%", "")) / 100

                    results.append(
                        ScanResult(
                            symbol=row.get("Ticker", ""),
                            price=float(row.get("Price", 0)),
                            change_pct=change * 100,
                            volume=int(str(row.get("Volume", 0)).replace(",", "")),
                            market_cap=str(row.get("Market Cap", "")),
                            sector=str(row.get("Sector", "")),
                        )
                    )
                except:
                    continue

            results.sort(key=lambda x: x.change_pct, reverse=True)
            self._cache[cache_key] = results
            self._cache_time[cache_key] = datetime.now()

            return results

        except Exception as e:
            logger.error(f"Finviz breakout scan error: {e}")
            return []

    def get_premarket_movers(self) -> List[ScanResult]:
        """
        Get pre-market movers (Note: Finviz updates delayed in pre-market)
        Use Schwab or Yahoo for real pre-market data.
        """
        # Finviz doesn't have real pre-market, just use gainers
        return self.get_top_gainers(min_change=3.0)

    def scan_all(self) -> Dict[str, List[ScanResult]]:
        """Run all scans and return combined results"""
        return {
            "top_gainers": self.get_top_gainers(),
            "low_float": self.get_low_float_movers(),
            "high_volume": self.get_high_volume_breakouts(),
            "timestamp": datetime.now().isoformat(),
        }


# Singleton
_scanner: Optional[FinvizScanner] = None


def get_finviz_scanner() -> FinvizScanner:
    """Get the Finviz scanner instance"""
    global _scanner
    if _scanner is None:
        _scanner = FinvizScanner()
    return _scanner


def scan_top_gainers(min_change: float = 5.0, max_price: float = 20.0) -> List[Dict]:
    """Quick function to get top gainers as dicts"""
    scanner = get_finviz_scanner()
    results = scanner.get_top_gainers(min_change=min_change, max_price=max_price)
    return [
        {
            "symbol": r.symbol,
            "price": r.price,
            "change_pct": r.change_pct,
            "volume": r.volume,
            "source": "finviz",
        }
        for r in results
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    scanner = get_finviz_scanner()

    print("\n=== TOP GAINERS (5%+, <$20) ===")
    gainers = scanner.get_top_gainers()
    for g in gainers[:10]:
        print(f"  {g.symbol}: ${g.price:.2f} ({g.change_pct:+.1f}%) Vol: {g.volume:,}")

    print("\n=== HIGH VOLUME BREAKOUTS ===")
    breakouts = scanner.get_high_volume_breakouts()
    for b in breakouts[:10]:
        print(f"  {b.symbol}: ${b.price:.2f} ({b.change_pct:+.1f}%) Vol: {b.volume:,}")
