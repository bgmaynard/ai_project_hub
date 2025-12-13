"""
Warrior Trading Scanner
Identifies high-probability momentum stocks based on Ross Cameron's criteria

Scans for:
- High relative volume (RVOL ≥ 2.0)
- Low float (< 50M shares)
- Gap up ≥ 5%
- Strong catalysts (news, earnings)
- Favorable daily chart structure
"""

import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime, time
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from finvizfinance.screener.overview import Overview
    FINVIZ_AVAILABLE = True
except ImportError:
    logging.warning("finvizfinance not installed. Scanner will have limited functionality.")
    FINVIZ_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    logging.warning("yfinance not installed. Daily chart analysis will be disabled.")
    YFINANCE_AVAILABLE = False

from config.config_loader import get_config

logger = logging.getLogger(__name__)


@dataclass
class WarriorCandidate:
    """
    Stock meeting Warrior Trading criteria

    Attributes:
        symbol: Stock ticker
        price: Current price
        gap_percent: % gap from previous close
        relative_volume: Current volume / average volume
        float_shares: Shares available for trading (millions)
        pre_market_volume: Pre-market volume
        catalyst: News catalyst or "TECHNICAL" or "NONE"
        daily_chart_signal: UPTREND, BREAKOUT, NEUTRAL, DOWNTREND
        distance_to_resistance: % to nearest resistance
        confidence_score: Overall quality score (0-100)
        market_cap: Market capitalization (millions)
        sector: Stock sector
        timestamp: When scan was performed
    """
    symbol: str
    price: float
    gap_percent: float
    relative_volume: float
    float_shares: float
    pre_market_volume: int
    catalyst: str
    daily_chart_signal: str
    distance_to_resistance: float
    confidence_score: float
    market_cap: float = 0.0
    sector: str = "Unknown"
    timestamp: str = ""

    def __post_init__(self):
        """Set timestamp if not provided"""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class WarriorScanner:
    """
    Scan for Warrior Trading momentum candidates

    Uses FinViz for screening and yfinance for detailed analysis
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize scanner

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = get_config(config_path)
        self.scanner_config = self.config.scanner

        if FINVIZ_AVAILABLE:
            self.finviz = Overview()
        else:
            self.finviz = None
            logger.warning("FinViz not available - scanner functionality limited")

        self.cache: Dict[str, WarriorCandidate] = {}
        self.cache_timestamp: Optional[datetime] = None

        logger.info("WarriorScanner initialized")

    def scan_premarket(
        self,
        min_gap_percent: Optional[float] = None,
        min_rvol: Optional[float] = None,
        max_float: Optional[float] = None,
        min_premarket_vol: Optional[int] = None
    ) -> List[WarriorCandidate]:
        """
        Run pre-market scan for momentum candidates

        Args:
            min_gap_percent: Minimum gap % (default from config)
            min_rvol: Minimum relative volume (default from config)
            max_float: Maximum float in millions (default from config)
            min_premarket_vol: Minimum pre-market volume (default from config)

        Returns:
            List of WarriorCandidate objects sorted by confidence score
        """
        # Use config defaults if not provided
        min_gap_percent = min_gap_percent or self.scanner_config.min_gap_percent
        min_rvol = min_rvol or self.scanner_config.min_rvol
        max_float = max_float or self.scanner_config.max_float_millions
        min_premarket_vol = min_premarket_vol or self.scanner_config.min_premarket_volume

        logger.info(
            f"Starting pre-market scan: gap≥{min_gap_percent}%, "
            f"RVOL≥{min_rvol}, float<{max_float}M"
        )

        # Check if FinViz is available
        if not self.finviz:
            logger.error("FinViz not available - cannot perform scan")
            return []

        try:
            # Set FinViz filters
            # FinViz requires specific integer values for filters (no decimals)
            filters_dict = {
                'Change': f'Up {int(min_gap_percent)}%',
                'Relative Volume': f'Over {int(min_rvol)}',
                'Float': f'Under {int(max_float)}M',
                'Price': f'Over ${int(self.scanner_config.min_price)}'
            }

            logger.debug(f"FinViz filters: {filters_dict}")

            self.finviz.set_filter(filters_dict=filters_dict)
            df = self.finviz.screener_view()

            if df is None or df.empty:
                logger.warning("No stocks found matching criteria")
                return []

            logger.info(f"Found {len(df)} stocks from FinViz")

            # Process each stock
            candidates = []
            for _, row in df.iterrows():
                try:
                    candidate = self._process_stock(row)
                    if candidate:
                        candidates.append(candidate)
                except Exception as e:
                    logger.error(f"Error processing {row.get('Ticker', 'UNKNOWN')}: {e}")
                    continue

            # Sort by confidence score (descending)
            candidates.sort(key=lambda x: x.confidence_score, reverse=True)

            # Limit to max watchlist size
            max_size = self.scanner_config.max_watchlist_size
            candidates = candidates[:max_size]

            logger.info(
                f"Scan complete: {len(candidates)} candidates "
                f"(avg confidence: {np.mean([c.confidence_score for c in candidates]):.1f})"
            )

            # Update cache
            self.cache = {c.symbol: c for c in candidates}
            self.cache_timestamp = datetime.now()

            return candidates

        except Exception as e:
            logger.error(f"Error in pre-market scan: {e}", exc_info=True)
            return []

    def _process_stock(self, row: pd.Series) -> Optional[WarriorCandidate]:
        """
        Process a single stock from FinViz results

        Args:
            row: FinViz data row

        Returns:
            WarriorCandidate or None if invalid
        """
        try:
            symbol = row['Ticker']

            # Extract basic data from FinViz
            price = self._safe_float(row.get('Price', 0))
            gap_percent = self._parse_percent(row.get('Change', '0%'))
            relative_volume = self._safe_float(row.get('Rel Volume', 0))
            volume = int(row.get('Volume', 0))
            float_shares = self._parse_float_string(row.get('Float', '0M'))
            market_cap = self._parse_float_string(row.get('Market Cap', '0M'))
            sector = row.get('Sector', 'Unknown')

            # Filter by price range
            if price < self.scanner_config.min_price or price > self.scanner_config.max_price:
                logger.debug(f"{symbol}: Price ${price} out of range")
                return None

            # Get additional analysis
            catalyst = self._check_catalyst(symbol)
            daily_signal = self._analyze_daily_chart(symbol)
            resistance_distance = self._distance_to_resistance(symbol)

            # Calculate confidence score
            confidence = self._calculate_confidence(
                gap=gap_percent,
                rvol=relative_volume,
                float_shares=float_shares,
                catalyst=catalyst,
                daily_signal=daily_signal
            )

            candidate = WarriorCandidate(
                symbol=symbol,
                price=price,
                gap_percent=gap_percent,
                relative_volume=relative_volume,
                float_shares=float_shares,
                pre_market_volume=volume,
                catalyst=catalyst,
                daily_chart_signal=daily_signal,
                distance_to_resistance=resistance_distance,
                confidence_score=confidence,
                market_cap=market_cap,
                sector=sector
            )

            logger.debug(
                f"{symbol}: ${price:.2f}, Gap {gap_percent:+.1f}%, "
                f"RVOL {relative_volume:.1f}, Score {confidence:.0f}"
            )

            return candidate

        except Exception as e:
            logger.error(f"Error processing stock data: {e}")
            return None

    def _check_catalyst(self, symbol: str) -> str:
        """
        Check for news catalyst

        Args:
            symbol: Stock ticker

        Returns:
            News headline, "TECHNICAL", or "NONE"
        """
        try:
            if not YFINANCE_AVAILABLE:
                return "UNKNOWN"

            ticker = yf.Ticker(symbol)
            news = ticker.news

            if news and len(news) > 0:
                # Get most recent headline
                latest_news = news[0]
                headline = latest_news.get('title', '')

                # Check for key catalyst words
                catalyst_words = [
                    'earnings', 'beat', 'miss', 'guidance', 'upgrade', 'downgrade',
                    'fda', 'approval', 'trial', 'acquisition', 'merger', 'contract',
                    'partnership', 'product', 'launch'
                ]

                headline_lower = headline.lower()
                if any(word in headline_lower for word in catalyst_words):
                    return headline[:100]  # Truncate to 100 chars

            return "TECHNICAL"

        except Exception as e:
            logger.debug(f"Error checking catalyst for {symbol}: {e}")
            return "UNKNOWN"

    def _analyze_daily_chart(self, symbol: str) -> str:
        """
        Analyze daily chart for trend and breakout status

        Args:
            symbol: Stock ticker

        Returns:
            "UPTREND", "BREAKOUT", "NEUTRAL", or "DOWNTREND"
        """
        try:
            if not YFINANCE_AVAILABLE:
                return "NEUTRAL"

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="60d")

            if len(hist) < 20:
                return "NEUTRAL"

            # Calculate moving averages
            hist['SMA20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA50'] = hist['Close'].rolling(window=50).mean()

            current_price = hist['Close'].iloc[-1]
            sma20 = hist['SMA20'].iloc[-1]
            sma50 = hist['SMA50'].iloc[-1] if len(hist) >= 50 else sma20

            # Check for breakout (new 30-day high)
            high_30d = hist['High'].iloc[-30:].max()
            if current_price >= high_30d * 0.99:  # Within 1% of 30-day high
                return "BREAKOUT"

            # Check trend
            if current_price > sma20 and sma20 > sma50:
                return "UPTREND"
            elif current_price < sma20 and sma20 < sma50:
                return "DOWNTREND"
            else:
                return "NEUTRAL"

        except Exception as e:
            logger.debug(f"Error analyzing daily chart for {symbol}: {e}")
            return "NEUTRAL"

    def _distance_to_resistance(self, symbol: str) -> float:
        """
        Calculate % distance to nearest major resistance

        Args:
            symbol: Stock ticker

        Returns:
            Distance in % (999.0 if no resistance found)
        """
        try:
            if not YFINANCE_AVAILABLE:
                return 50.0

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d")

            if len(hist) < 10:
                return 50.0

            current_price = hist['Close'].iloc[-1]

            # Find swing highs (local maxima)
            highs = hist['High'].rolling(window=5, center=True).max()
            swing_highs = hist[hist['High'] == highs]['High'].values

            # Find next resistance above current price
            resistance_levels = [h for h in swing_highs if h > current_price]

            if resistance_levels:
                nearest_resistance = min(resistance_levels)
                distance_pct = ((nearest_resistance - current_price) / current_price) * 100
                return distance_pct

            return 999.0  # No clear resistance = "blue sky breakout"

        except Exception as e:
            logger.debug(f"Error calculating resistance for {symbol}: {e}")
            return 50.0

    def _calculate_confidence(
        self,
        gap: float,
        rvol: float,
        float_shares: float,
        catalyst: str,
        daily_signal: str
    ) -> float:
        """
        Calculate confidence score (0-100)

        Weights:
        - Gap: 20%
        - RVOL: 25%
        - Float: 20%
        - Catalyst: 20%
        - Daily signal: 15%

        Args:
            gap: Gap percent
            rvol: Relative volume
            float_shares: Float in millions
            catalyst: Catalyst string
            daily_signal: Daily chart signal

        Returns:
            Confidence score 0-100
        """
        score = 0.0

        # Gap score (higher is better, cap at 20%)
        gap_score = min(gap / 20.0 * 20, 20)
        score += gap_score

        # RVOL score (higher is better, cap at 25)
        rvol_score = min(rvol / 5.0 * 25, 25)
        score += rvol_score

        # Float score (lower is better)
        if float_shares < 10:
            float_score = 20
        elif float_shares < 20:
            float_score = 15
        elif float_shares < 50:
            float_score = 10
        else:
            float_score = 0
        score += float_score

        # Catalyst score
        if catalyst not in ["NONE", "TECHNICAL", "UNKNOWN"]:
            catalyst_score = 20  # Strong news catalyst
        elif catalyst == "TECHNICAL":
            catalyst_score = 10  # Technical breakout
        else:
            catalyst_score = 5  # No clear catalyst
        score += catalyst_score

        # Daily signal score
        daily_scores = {
            "BREAKOUT": 15,
            "UPTREND": 12,
            "NEUTRAL": 5,
            "DOWNTREND": 0,
            "UNKNOWN": 3
        }
        score += daily_scores.get(daily_signal, 5)

        return min(score, 100.0)

    # Helper methods
    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert value to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _parse_percent(percent_str: str) -> float:
        """Parse percentage string like '+5.2%' to 5.2"""
        try:
            return float(percent_str.strip('%').strip('+'))
        except (ValueError, AttributeError):
            return 0.0

    @staticmethod
    def _parse_float_string(float_str: str) -> float:
        """Parse float string like '15.2M' or '1.5B' to number"""
        try:
            if isinstance(float_str, (int, float)):
                return float(float_str)

            float_str = str(float_str).strip().upper()

            if 'M' in float_str:
                return float(float_str.replace('M', ''))
            elif 'B' in float_str:
                return float(float_str.replace('B', '')) * 1000
            elif 'K' in float_str:
                return float(float_str.replace('K', '')) / 1000
            else:
                return float(float_str)
        except (ValueError, AttributeError):
            return 0.0

    def get_cached_results(self) -> List[WarriorCandidate]:
        """Get cached scan results"""
        if not self.cache:
            logger.warning("No cached results available")
            return []

        candidates = list(self.cache.values())
        candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        return candidates

    def is_cache_valid(self, max_age_minutes: int = 15) -> bool:
        """Check if cache is still valid"""
        if not self.cache_timestamp:
            return False

        age = (datetime.now() - self.cache_timestamp).total_seconds() / 60
        return age < max_age_minutes


# Example usage / testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("WARRIOR TRADING PRE-MARKET SCANNER")
    print("=" * 80)

    try:
        # Initialize scanner
        scanner = WarriorScanner()

        # Run scan
        print("\nRunning pre-market scan...")
        candidates = scanner.scan_premarket()

        if not candidates:
            print("\n❌ No candidates found")
        else:
            print(f"\n✅ Found {len(candidates)} candidates:\n")
            print(f"{'Symbol':8} {'Price':>8} {'Gap %':>8} {'RVOL':>6} {'Float':>8} "
                  f"{'Signal':12} {'Score':>6}")
            print("-" * 80)

            for c in candidates:
                print(f"{c.symbol:8} ${c.price:7.2f} {c.gap_percent:+7.1f}% "
                      f"{c.relative_volume:6.1f} {c.float_shares:7.1f}M "
                      f"{c.daily_chart_signal:12} {c.confidence_score:6.0f}")

            print("\n" + "=" * 80)

            # Show top 3 with details
            print("\nTOP 3 CANDIDATES:")
            for i, c in enumerate(candidates[:3], 1):
                print(f"\n{i}. {c.symbol} - Score: {c.confidence_score:.0f}/100")
                print(f"   Price: ${c.price:.2f}")
                print(f"   Gap: {c.gap_percent:+.1f}%")
                print(f"   RVOL: {c.relative_volume:.1f}x")
                print(f"   Float: {c.float_shares:.1f}M shares")
                print(f"   Daily: {c.daily_chart_signal}")
                print(f"   Catalyst: {c.catalyst}")
                if c.distance_to_resistance < 100:
                    print(f"   Resistance: {c.distance_to_resistance:.1f}% away")
                else:
                    print(f"   Resistance: Blue sky breakout")

    except Exception as e:
        logger.error(f"Error running scanner: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
