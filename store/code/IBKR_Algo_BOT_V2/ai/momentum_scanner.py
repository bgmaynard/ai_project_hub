"""
Momentum Stock Scanner
======================
Automatically finds high-momentum stocks and adds them to the watchlist.

Scans for:
1. Pre-market gappers (10%+ gaps)
2. Intraday runners (50%+ momentum)
3. Unusual volume (RVOL > 2x)
4. Breaking news catalysts
5. Float rotation candidates

Filters by trading criteria:
- Price: $0.50 - $20
- Volume: >100K
- Spread: <1%
- Float: <10M shares
"""

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

logger = logging.getLogger(__name__)

# =============================================================================
# TRADING CRITERIA (Ross Cameron / Warrior Trading Style)
# =============================================================================
CRITERIA = {
    "min_price": 0.50,
    "max_price": 20.00,
    "min_volume": 100000,  # 100K minimum daily volume
    "max_spread_pct": 1.0,  # 1% max spread
    "max_float": 10_000_000,  # 10M shares max float
    "min_gap_pct": 10.0,  # 10% minimum gap for gapper scan
    "min_momentum_pct": 20.0,  # 20% minimum momentum (was 50%, lowered for more hits)
    "min_rvol": 2.0,  # 2x relative volume minimum
    "min_confidence": 0.60,  # 60% minimum confidence score
}

# Scanner presets for different strategies
SCANNER_PRESETS = {
    "warrior_momentum": {
        "min_price": 0.50,
        "max_price": 20.00,
        "min_gap_pct": 10.0,
        "min_rvol": 2.0,
        "max_float": 10_000_000,
    },
    "penny_rockets": {
        "min_price": 0.10,
        "max_price": 5.00,
        "min_gap_pct": 20.0,
        "min_rvol": 3.0,
        "max_float": 5_000_000,
    },
    "mid_cap_breakout": {
        "min_price": 5.00,
        "max_price": 50.00,
        "min_gap_pct": 5.0,
        "min_rvol": 1.5,
        "max_float": 50_000_000,
    },
}


@dataclass
class MomentumStock:
    """Represents a stock meeting momentum criteria"""

    symbol: str
    price: float
    change_pct: float  # % change from prev close
    gap_pct: float  # Pre-market gap %
    volume: int
    avg_volume: int
    rvol: float  # Relative volume
    float_shares: int
    spread_pct: float
    high_of_day: float
    low_of_day: float
    momentum_score: float  # 0-100 composite score
    catalyst: str  # News/catalyst if any
    timestamp: str
    added_to_watchlist: bool = False


class MomentumScanner:
    """
    Scans for high-momentum stocks matching trading criteria.

    Runs continuously and auto-populates the watchlist with top movers.
    """

    def __init__(self):
        self.api_url = "http://localhost:9100/api/alpaca"
        self.running = False
        self.scan_thread: Optional[threading.Thread] = None

        # Track found stocks
        self.momentum_stocks: Dict[str, MomentumStock] = {}
        self.seen_today: Set[str] = set()
        self.last_scan_time: Optional[str] = None

        # Stock universe to scan
        self.scan_universe: Set[str] = set()
        self._load_scan_universe()

        # Cache for float data (expensive to fetch)
        self.float_cache: Dict[str, int] = {}
        self.float_cache_file = Path("store/float_cache.json")
        self._load_float_cache()

        # Statistics
        self.stats = {
            "scans_completed": 0,
            "stocks_found": 0,
            "stocks_added_to_watchlist": 0,
            "last_scan_duration": 0,
        }

        logger.info("MomentumScanner initialized")

    def _load_scan_universe(self):
        """Load universe of stocks to scan"""
        # Start with common momentum tickers
        base_universe = {
            # Popular momentum/meme stocks
            "TSLA",
            "NVDA",
            "AMD",
            "PLTR",
            "SOFI",
            "NIO",
            "LCID",
            "MARA",
            "RIOT",
            "COIN",
            "HOOD",
            "GME",
            "AMC",
            "BBBY",
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            # Biotech (often gap hard)
            "MRNA",
            "BNTX",
            "SAVA",
            "SRPT",
            "VKTX",
            "DRUG",
            # Small caps (add as we find them)
        }
        self.scan_universe = base_universe

        # Load additional symbols from file if exists
        universe_file = Path("store/scan_universe.json")
        if universe_file.exists():
            try:
                data = json.loads(universe_file.read_text())
                self.scan_universe.update(data.get("symbols", []))
            except:
                pass

        logger.info(f"Scan universe loaded: {len(self.scan_universe)} symbols")

    def _load_float_cache(self):
        """Load cached float data"""
        if self.float_cache_file.exists():
            try:
                self.float_cache = json.loads(self.float_cache_file.read_text())
                logger.info(f"Float cache loaded: {len(self.float_cache)} symbols")
            except:
                pass

    def _save_float_cache(self):
        """Save float cache to file"""
        try:
            self.float_cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.float_cache_file.write_text(json.dumps(self.float_cache, indent=2))
        except Exception as e:
            logger.warning(f"Could not save float cache: {e}")

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get quote data for a symbol"""
        try:
            r = requests.get(f"{self.api_url}/quote/{symbol}", timeout=3)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return None

    def get_bars(
        self, symbol: str, timeframe: str = "1Day", limit: int = 5
    ) -> List[Dict]:
        """Get historical bars for a symbol"""
        try:
            r = requests.get(
                f"{self.api_url}/bars/{symbol}",
                params={"timeframe": timeframe, "limit": limit},
                timeout=5,
            )
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return []

    def get_float(self, symbol: str) -> int:
        """Get float shares for a symbol (cached)"""
        if symbol in self.float_cache:
            return self.float_cache[symbol]

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info
            float_shares = info.get("floatShares", 0) or 0

            # Cache it
            if float_shares > 0:
                self.float_cache[symbol] = float_shares
                self._save_float_cache()

            return float_shares
        except:
            return 0

    def check_criteria(self, symbol: str) -> Optional[MomentumStock]:
        """
        Check if a stock meets all trading criteria.
        Returns MomentumStock if qualified, None otherwise.
        """
        quote = self.get_quote(symbol)
        if not quote:
            return None

        # Extract quote data
        price = float(quote.get("last", 0) or quote.get("bid", 0) or 0)
        bid = float(quote.get("bid", 0) or 0)
        ask = float(quote.get("ask", 0) or 0)
        volume = int(quote.get("volume", 0) or 0)
        prev_close = float(quote.get("prev_close", 0) or price)
        high = float(quote.get("high", price) or price)
        low = float(quote.get("low", price) or price)

        if price <= 0:
            return None

        # Calculate metrics
        spread = ask - bid if ask > 0 and bid > 0 else 0
        spread_pct = (spread / price * 100) if price > 0 else 100
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0

        # Get historical data for average volume
        bars = self.get_bars(symbol, "1Day", 20)
        avg_volume = 0
        if bars:
            volumes = [b.get("volume", 0) for b in bars if b.get("volume", 0) > 0]
            avg_volume = int(sum(volumes) / len(volumes)) if volumes else volume

        rvol = (volume / avg_volume) if avg_volume > 0 else 1.0

        # Get float
        float_shares = self.get_float(symbol)

        # =========================================
        # APPLY TRADING CRITERIA FILTERS
        # =========================================

        # Price filter
        if price < CRITERIA["min_price"] or price > CRITERIA["max_price"]:
            return None

        # Volume filter
        if volume < CRITERIA["min_volume"]:
            return None

        # Spread filter
        if spread_pct > CRITERIA["max_spread_pct"]:
            return None

        # Float filter (if we have data)
        if float_shares > 0 and float_shares > CRITERIA["max_float"]:
            return None

        # Momentum filter - must have significant move
        if abs(change_pct) < CRITERIA["min_momentum_pct"]:
            return None

        # =========================================
        # CALCULATE MOMENTUM SCORE (0-100)
        # =========================================
        score = 0

        # Change % contribution (max 30 points)
        # 50%+ move = 30 points, 20% move = 12 points
        score += min(30, abs(change_pct) * 0.6)

        # RVOL contribution (max 25 points)
        # 5x RVOL = 25 points, 2x = 10 points
        score += min(25, rvol * 5)

        # Float rotation contribution (max 20 points)
        # If volume > float, that's huge momentum
        if float_shares > 0:
            float_rotation = volume / float_shares
            score += min(20, float_rotation * 20)

        # Price range contribution (max 15 points)
        # Prefer $2-10 range for best momentum
        if 2.0 <= price <= 10.0:
            score += 15
        elif 1.0 <= price <= 15.0:
            score += 10
        else:
            score += 5

        # Spread tightness contribution (max 10 points)
        # Tighter spread = better
        if spread_pct < 0.3:
            score += 10
        elif spread_pct < 0.5:
            score += 7
        elif spread_pct < 1.0:
            score += 4

        # Must meet minimum confidence
        if score < CRITERIA["min_confidence"] * 100:
            return None

        # Determine catalyst (placeholder - would integrate with news)
        catalyst = "MOMENTUM" if change_pct > 0 else "REVERSAL"
        if rvol > 5:
            catalyst = "UNUSUAL_VOLUME"
        if float_shares > 0 and volume > float_shares:
            catalyst = "FLOAT_ROTATION"

        return MomentumStock(
            symbol=symbol,
            price=price,
            change_pct=change_pct,
            gap_pct=change_pct,  # For intraday, gap = change
            volume=volume,
            avg_volume=avg_volume,
            rvol=rvol,
            float_shares=float_shares,
            spread_pct=spread_pct,
            high_of_day=high,
            low_of_day=low,
            momentum_score=min(100, score),
            catalyst=catalyst,
            timestamp=datetime.now().isoformat(),
        )

    def add_to_watchlist(self, symbol: str) -> bool:
        """Add a symbol to the trading watchlist"""
        try:
            r = requests.post(
                "http://localhost:9100/api/worklist/add",
                json={"symbol": symbol, "action": "watch"},
                timeout=3,
            )
            return r.json().get("success", False)
        except:
            return False

    def scan_gappers(self) -> List[MomentumStock]:
        """
        Scan for pre-market gappers.
        Returns list of stocks gapping 10%+ that meet criteria.
        """
        gappers = []

        # In production, would use a screener API
        # For now, check our universe
        for symbol in list(self.scan_universe)[:50]:  # Limit to avoid rate limits
            try:
                stock = self.check_criteria(symbol)
                if stock and stock.gap_pct >= CRITERIA["min_gap_pct"]:
                    gappers.append(stock)
            except Exception as e:
                logger.debug(f"Error checking {symbol}: {e}")

        return sorted(gappers, key=lambda x: x.momentum_score, reverse=True)

    def scan_intraday_momentum(self) -> List[MomentumStock]:
        """
        Scan for intraday momentum runners.
        Returns list of stocks with 20%+ moves that meet criteria.
        """
        runners = []

        for symbol in list(self.scan_universe)[:50]:
            try:
                stock = self.check_criteria(symbol)
                if stock and abs(stock.change_pct) >= CRITERIA["min_momentum_pct"]:
                    runners.append(stock)
            except Exception as e:
                logger.debug(f"Error checking {symbol}: {e}")

        return sorted(runners, key=lambda x: x.momentum_score, reverse=True)

    def scan_unusual_volume(self) -> List[MomentumStock]:
        """
        Scan for unusual volume (RVOL > 2x).
        Often precedes big moves.
        """
        unusual = []

        for symbol in list(self.scan_universe)[:50]:
            try:
                stock = self.check_criteria(symbol)
                if stock and stock.rvol >= CRITERIA["min_rvol"]:
                    unusual.append(stock)
            except Exception as e:
                logger.debug(f"Error checking {symbol}: {e}")

        return sorted(unusual, key=lambda x: x.rvol, reverse=True)

    def discover_new_symbols(self) -> List[str]:
        """
        Discover new symbols from external sources.
        Adds them to scan universe.
        """
        new_symbols = []

        # Try to get top gainers from various sources
        sources = [
            "http://localhost:9100/api/scanner/ALPACA/scan?preset=gappers",
            "http://localhost:9100/api/scanner/ALPACA/scan?preset=momentum",
        ]

        for url in sources:
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    for stock in data.get("results", []):
                        sym = stock.get("symbol", "")
                        if sym and sym not in self.scan_universe:
                            new_symbols.append(sym)
                            self.scan_universe.add(sym)
            except:
                pass

        if new_symbols:
            logger.info(
                f"Discovered {len(new_symbols)} new symbols: {new_symbols[:10]}"
            )

        return new_symbols

    def run_full_scan(self) -> Dict:
        """
        Run a complete scan and return results.
        Auto-adds top stocks to watchlist.
        """
        start_time = time.time()

        # Discover new symbols first
        self.discover_new_symbols()

        # Run all scans
        gappers = self.scan_gappers()
        runners = self.scan_intraday_momentum()
        unusual = self.scan_unusual_volume()

        # Combine and dedupe, sort by score
        all_stocks = {}
        for stock in gappers + runners + unusual:
            if stock.symbol not in all_stocks:
                all_stocks[stock.symbol] = stock
            elif stock.momentum_score > all_stocks[stock.symbol].momentum_score:
                all_stocks[stock.symbol] = stock

        # Sort by momentum score
        sorted_stocks = sorted(
            all_stocks.values(), key=lambda x: x.momentum_score, reverse=True
        )

        # Auto-add top 5 to watchlist
        added_count = 0
        for stock in sorted_stocks[:5]:
            if stock.symbol not in self.seen_today:
                if self.add_to_watchlist(stock.symbol):
                    stock.added_to_watchlist = True
                    self.seen_today.add(stock.symbol)
                    added_count += 1
                    logger.info(
                        f"AUTO-ADDED: {stock.symbol} "
                        f"(Score: {stock.momentum_score:.0f}, "
                        f"Change: {stock.change_pct:+.1f}%, "
                        f"RVOL: {stock.rvol:.1f}x)"
                    )

        # Store results
        self.momentum_stocks = all_stocks
        self.last_scan_time = datetime.now().isoformat()

        # Update stats
        elapsed = time.time() - start_time
        self.stats["scans_completed"] += 1
        self.stats["stocks_found"] = len(sorted_stocks)
        self.stats["stocks_added_to_watchlist"] += added_count
        self.stats["last_scan_duration"] = elapsed

        return {
            "success": True,
            "timestamp": self.last_scan_time,
            "duration_seconds": elapsed,
            "stocks_found": len(sorted_stocks),
            "stocks_added": added_count,
            "top_stocks": [asdict(s) for s in sorted_stocks[:10]],
            "by_category": {
                "gappers": len(gappers),
                "momentum_runners": len(runners),
                "unusual_volume": len(unusual),
            },
        }

    def get_top_stocks(self, limit: int = 10) -> List[Dict]:
        """Get current top momentum stocks"""
        sorted_stocks = sorted(
            self.momentum_stocks.values(), key=lambda x: x.momentum_score, reverse=True
        )
        return [asdict(s) for s in sorted_stocks[:limit]]

    def add_symbol_to_universe(self, symbol: str):
        """Manually add a symbol to the scan universe"""
        self.scan_universe.add(symbol.upper())
        logger.info(f"Added {symbol} to scan universe")

    def start_continuous_scan(self, interval_seconds: int = 60):
        """Start continuous scanning in background thread"""
        if self.running:
            return

        self.running = True

        def scan_loop():
            while self.running:
                try:
                    result = self.run_full_scan()
                    logger.info(
                        f"Scan complete: {result['stocks_found']} found, "
                        f"{result['stocks_added']} added to watchlist"
                    )
                except Exception as e:
                    logger.error(f"Scan error: {e}")

                time.sleep(interval_seconds)

        self.scan_thread = threading.Thread(target=scan_loop, daemon=True)
        self.scan_thread.start()
        logger.info(f"Continuous scanning started (every {interval_seconds}s)")

    def stop_continuous_scan(self):
        """Stop continuous scanning"""
        self.running = False
        logger.info("Continuous scanning stopped")

    def get_status(self) -> Dict:
        """Get scanner status"""
        return {
            "running": self.running,
            "last_scan": self.last_scan_time,
            "universe_size": len(self.scan_universe),
            "stocks_tracked": len(self.momentum_stocks),
            "seen_today": len(self.seen_today),
            "stats": self.stats,
            "criteria": CRITERIA,
        }


# Singleton instance
_scanner: Optional[MomentumScanner] = None


def get_momentum_scanner() -> MomentumScanner:
    """Get or create the momentum scanner singleton"""
    global _scanner
    if _scanner is None:
        _scanner = MomentumScanner()
    return _scanner


# =============================================================================
# CLI for testing
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    print("=" * 60)
    print("  MOMENTUM STOCK SCANNER")
    print("=" * 60)
    print(f"Criteria: ${CRITERIA['min_price']:.2f}-${CRITERIA['max_price']:.2f}")
    print(
        f"Volume: >{CRITERIA['min_volume']:,} | Spread: <{CRITERIA['max_spread_pct']}%"
    )
    print(
        f"Float: <{CRITERIA['max_float']/1_000_000:.0f}M | RVOL: >{CRITERIA['min_rvol']}x"
    )
    print(f"Min Momentum: {CRITERIA['min_momentum_pct']}%")
    print("=" * 60)

    scanner = get_momentum_scanner()

    print("\nRunning scan...")
    result = scanner.run_full_scan()

    print(f"\nScan completed in {result['duration_seconds']:.1f}s")
    print(f"Stocks found: {result['stocks_found']}")
    print(f"Added to watchlist: {result['stocks_added']}")

    print("\n" + "=" * 60)
    print("TOP MOMENTUM STOCKS:")
    print("=" * 60)

    for i, stock in enumerate(result["top_stocks"], 1):
        print(
            f"{i}. {stock['symbol']:6} | ${stock['price']:.2f} | "
            f"{stock['change_pct']:+.1f}% | RVOL: {stock['rvol']:.1f}x | "
            f"Score: {stock['momentum_score']:.0f}"
        )

    if not result["top_stocks"]:
        print("No stocks meeting criteria found.")
        print("Try expanding the scan universe or adjusting criteria.")
